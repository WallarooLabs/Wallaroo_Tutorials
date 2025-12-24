import os
from typing import Any, Set

import numpy as np
from llama_cpp import Llama
from mac.config.inference import CustomInferenceConfig
from mac.inference import Inference
from mac.inference.creation import InferenceBuilder
from mac.types import InferenceData
    
class LlamacppInference(Inference):
    @property
    def expected_model_types(self) -> Set[Any]:
        return {Llama}
    
    @Inference.model.setter
    def model(self, model) -> None:
        # self._raise_error_if_model_is_wrong_type(model)
        self._model = model

    def _predict(self, input_data: InferenceData):
        generated_texts = []
        prompts = input_data["text"].tolist()

        for prompt in prompts:
            full_prompt = f"""
                <|begin_of_text|><|start_header_id|>system<|end_header_id|>

                You are the helpful assistant. <|eot_id|><|start_header_id|>user<|end_header_id|>

                {prompt} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
            result = self.model(
                full_prompt, max_tokens=256, stop=["<|eot_id|>"], echo=False
            )
            generated_texts.append(result["choices"][0]["text"])

        return {"generated_text": np.array(generated_texts)}

class LlamacppInference2(Inference):
    @property
    def expected_model_types(self) -> Set[Any]:
        return {Llama}
    
    @Inference.model.setter
    def model(self, model) -> None:
        # self._raise_error_if_model_is_wrong_type(model)
        self._model = model

    def _predict(self, input_data: InferenceData):
        generated_texts = []
        prompts = input_data["text"].tolist()

        for prompt in prompts:
            full_prompt = f"""
                <|begin_of_text|><|start_header_id|>system<|end_header_id|>

                You are the helpful assistant. <|eot_id|><|start_header_id|>user<|end_header_id|>

                {prompt} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
            result = self.model(
                full_prompt, max_tokens=256, stop=["<|eot_id|>"], echo=False
            )
            generated_texts.append(result["choices"][0]["text"])

        return {"generated_text": np.array(generated_texts)}
    
class LlamacppInferenceBuilder(InferenceBuilder):
    @property
    def inference(self) -> LlamacppInference:
        return LlamacppInference()

    def create(self, config: CustomInferenceConfig) -> LlamacppInference:
        inference = self.inference
        model = self._load_model(config.model_path)
        inference.model = model

        return inference

    def _load_model(self, model_path):
        llm = Llama(
            model_path=f"{model_path}/artifacts/stories260K.gguf",
        )

        return llm
