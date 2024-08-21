from typing import Any, Set

import base64
import subprocess
import sys
import cv2
import numpy as np

from mac.config.inference import CustomInferenceConfig
from mac.inference import Inference
from mac.inference.creation import InferenceBuilder
from mac.types import InferenceData

pip_command = (
    f'CMAKE_ARGS="-DLLAMA_CUDA=on" {sys.executable} -m pip install llama-cpp-python'
)

subprocess.check_call(pip_command, shell=True)

from llama_cpp import Llama
from llama_cpp.llama_chat_format import NanoLlavaChatHandler


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
        images = input_data["image"].tolist()
        system_prompts = input_data["system_prompt"].tolist()

        for prompt, system_prompt, image in zip(prompts, system_prompts, images):
            _, buffer = cv2.imencode(".png", np.array(image))
            prefix = "data:image/png;base64,"
            b64encoded_image = prefix + base64.b64encode(buffer).decode("utf-8")

            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": b64encoded_image}},
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                },
            ]

            result = self.model.create_chat_completion(
                messages=messages, max_tokens=256, stop=["<|im_end|>"]
            )
            generated_texts.append(result["choices"][0]["message"]["content"])

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
        chat_handler = NanoLlavaChatHandler(
            clip_model_path=f"{model_path}/artifacts/mmproj-model-f16.gguf"
        )
        llm = Llama(
            model_path=f"{model_path}/artifacts/llava-v1.6-34b.Q5_K_M.gguf",
            chat_format="llava-1-6",
            chat_handler=chat_handler,
            n_ctx=4096,
            n_gpu_layers=-1,
            logits_all=True,
        )

        return llm
