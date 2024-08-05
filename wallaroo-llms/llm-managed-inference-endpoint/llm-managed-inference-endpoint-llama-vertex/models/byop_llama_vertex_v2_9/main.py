import numpy as np
import requests
import os
import json

from mac.inference import Inference
from mac.inference.creation import InferenceBuilder
from mac.types import InferenceData
from mac.config.inference import CustomInferenceConfig

from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials

from typing import Any, Set


credentials = Credentials.from_service_account_info(
    json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS"].replace("'", '"')),
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)


class LlamaInference(Inference):
    @property
    def expected_model_types(self) -> Set[Any]:
        return {str}

    @Inference.model.setter
    def model(self, model) -> None:
        # self._raise_error_if_model_is_wrong_type(model)
        self._model = model

    def _predict(self, input_data: InferenceData):
        credentials.refresh(Request())
        token = credentials.token

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        prompts = input_data["text"].tolist()
        instances = [{"prompt": prompt, "max_tokens": 200} for prompt in prompts]

        response = requests.post(
            f"{self.model}",
            json={"instances": instances},
            headers=headers,
        )

        predictions = response.json()

        if isinstance(predictions["predictions"], str):
            generated_text = [
                prediction.split("Output:\n")[-1]
                for prediction in predictions["predictions"]
            ]
        else:
            generated_text = [
                prediction["predictions"][0].split("Output:\n")[-1]
                for prediction in predictions["predictions"]
            ]

        return {"generated_text": np.array(generated_text)}


class LlamaInferenceBuilder(InferenceBuilder):
    @property
    def inference(self) -> LlamaInference:
        return LlamaInference()

    def create(self, config: CustomInferenceConfig) -> LlamaInference:
        PROJECT_ID = "<your_gcp_project_id>"
        ENDPOINT_ID = "<your_vertex_endpoint_id>"
        LOCATION = "<your_cloud_region>"

        inference = self.inference
        model = f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}:predict"
        inference.model = model

        return inference
