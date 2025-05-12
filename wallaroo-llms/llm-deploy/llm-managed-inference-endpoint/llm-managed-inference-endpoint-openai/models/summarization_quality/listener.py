import torch
import numpy as np

from mac.inference import Inference
from mac.inference.creation import InferenceBuilder
from mac.types import InferenceData
from mac.config.inference import CustomInferenceConfig

from typing import Any, Set
from sentence_transformers import SentenceTransformer, util


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SummarisationQualityInference(Inference):
    @property
    def expected_model_types(self) -> Set[Any]:
        return {SentenceTransformer}

    @Inference.model.setter
    def model(self, model) -> None:
        # self._raise_error_if_model_is_wrong_type(model)
        self._model = model

    def _predict(self, input_data: InferenceData):
        prompts = input_data["text"].tolist()
        results = input_data["generated_text"].tolist()

        scores = []

        for prompt, result in zip(prompts, results):
            embedding_1 = self.model.encode(prompt, convert_to_tensor=True)
            embedding_2 = self.model.encode(result, convert_to_tensor=True)

            scores.append(util.pytorch_cos_sim(embedding_1, embedding_2)[0])

        generated_texts = np.array([str(x) for x in input_data["generated_text"]])

        return {"score": np.array(scores).reshape(-1,1), "generated_text": generated_texts}


class SummarisationQualityInferenceBuilder(InferenceBuilder):
    @property
    def inference(self) -> SummarisationQualityInference:
        return SummarisationQualityInference()

    def create(self, config: CustomInferenceConfig) -> SummarisationQualityInference:
        inference = self.inference
        model = self._load_model()
        inference.model = model

        return inference

    def _load_model(self):
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
