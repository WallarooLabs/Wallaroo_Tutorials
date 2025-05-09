import logging
import time
import uuid
import asyncio
import os
import threading
import time
from typing import Any, Awaitable, Set

import numpy as np
from mac.config.inference import CustomInferenceConfig
from mac.inference import AsyncInference
from mac.inference.creation import InferenceBuilder
from mac.types import InferenceData
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs

logger = logging.getLogger(__name__)


class AsyncVLLMInference(AsyncInference):
    """This class runs inference using AsyncLLMEngine from vLLM."""

    @property
    def expected_model_types(self) -> Set[Any]:
        """Returns a set of model instance types that are expected by this inference.

        :return: A set of model instance types that are expected by this inference.
        """
        return {AsyncLLMEngine}

    @AsyncInference.model.setter  # type: ignore
    def model(self, model: int) -> None:
        """Sets the model on which the inference is calculated.

        :param model: A custom model instance on which the inference is calculated.

        :raises TypeError: If the model is not an instance of expected_model_types.
        """
        self._raise_error_if_model_is_wrong_type(model)
        self._model = model

    async def _predict(self, input_data: InferenceData) -> Awaitable[InferenceData]:
        """Calculates the inference on the given input data.
        This is the core function that each subclass needs to implement
        in order to calculate the inference.

        :param input_data: The input data on which the inference is calculated.
        Depending on the number of inputs of the model, the input data can be either a single
        numpy array or a dictionary of numpy arrays.

        :raises InferenceDataValidationError: If the input data is not valid.
        Ideally, every subclass should raise this error if the input data is not valid.

        :return: The output of the model. Depending on the number of outputs of the model,
        the output data can be either a single numpy array or a dictionary of numpy arrays.
        """
        prompt = input_data["prompt"].tolist()[0]
        max_tokens = max(input_data["max_tokens"].tolist()[0], 1)

        request_id = str(uuid.uuid4())

        logger.info(f"Generating text for request_id: {request_id}")
        results_generator = self.model.generate(
            prompt,
            SamplingParams(
                temperature=1.0,
                top_p=1.0,
                max_tokens=max_tokens,
                ignore_eos=True,
            ),
            request_id,
        )
        logger.info("Generator initialized.")

        final_output = None
        ttft = None
        start = time.time()
        async for request_output in results_generator:
            ttft = time.time() - start
            final_output = request_output

        prompt = final_output.prompt
        output = [output.text for output in final_output.outputs][0]
        num_output_tokens = len(final_output.outputs[0].token_ids)
        logger.info(f"Generated text: {output}")
        logger.info(f"Num output tokens: {num_output_tokens}")
    

        return {
            "generated_text": np.array([output]),
            "num_output_tokens": np.array([num_output_tokens])
        }


class AsyncVLLMInferenceBuilder(InferenceBuilder):
    """Inference builder class for AsyncVLLMInference."""

    @property
    def inference(self) -> AsyncVLLMInference:
        """Returns an Inference subclass instance.
        This specifies the Inference instance to be used
        by create() to build additionally needed components."""
        return AsyncVLLMInference()

    def create(self, config: CustomInferenceConfig) -> AsyncVLLMInference:
        """Creates an Inference subclass and assigns a model to it.

        :param config: Inference configuration

        :return: Inference subclass
        """
        inference = self.inference
        inference.model = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(
                model=(config.model_path / "model").as_posix(),
            ),
        )
        return inference
