import numpy as np
from openai import OpenAI
import json
import os

from mac.inference import Inference
from mac.inference.creation import InferenceBuilder
from mac.types import InferenceData
from mac.config.inference import CustomInferenceConfig

from typing import Any, Set


class GPTInference(Inference):
    @property
    def expected_model_types(self) -> Set[Any]:
        return {OpenAI}

    @Inference.model.setter
    def model(self, model) -> None:
        # self._raise_error_if_model_is_wrong_type(model)
        self._model = model

    def _predict(self, input_data: InferenceData):
        generated_texts = []
        prompts = input_data["text"].tolist()

        for prompt in prompts:
            result = self.model.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=1,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            
            generated_texts.append(result.choices[0].message.content)

        prompt = np.array([str(x) for x in input_data["text"]])
        
        return {"text": prompt, "generated_text": np.array(generated_texts)}


class GPTInferenceBuilder(InferenceBuilder):
    @property
    def inference(self) -> GPTInference:
        return GPTInference()

    def create(self, config: CustomInferenceConfig) -> GPTInference:
        inference = self.inference
        
        with open(os.path.join(config.model_path, 'secret_key.json')) as file:
            auth = json.load(file)
        
        inference.model = OpenAI(api_key=auth['API_SECRET'])

        return inference