from pathlib import Path
from typing import Any, Set

import numpy as np
import onnx
import onnxruntime as ort

from mac.config.inference import CustomInferenceConfig
from mac.inference import Inference
from mac.inference.creation import InferenceBuilder
from mac.types import InferenceData


# Expected input:
# A Dictionary with fields 'outputs',
# which contains a list with an element that is a dictionary
# with the fields 'data' - native array of outputs
#                 'dim' - of data
#                 'v'
class PixIntensityResnet(Inference):
    @property
    def expected_model_types(self) -> Set[Any]:
        return {onnx.onnx_ml_pb2.ModelProto}

    @Inference.model.setter  # type: ignore
    def model(self, model) -> None:
        """Sets the model on which the inference is calculated.

        :param model: A model instance on which the inference is calculated.

        :raises TypeError: If the model is not an instance of expected_model_types
            (i.e. KMeans).
        """
        self._raise_error_if_model_is_wrong_type(
            model
        )  # this will make sure an error will be raised if the model is of wrong type
        self._model = model

    def _predict(self, input_data: InferenceData):
        # Parse inputs
        inputs = input_data["tensor"]

        # Pass to onnx model
        # `dynamic_axes` hasn't been set in torch.onnx.export()
        # that is used in CVDemoUtils.loadPytorchAndConvertToOnnx()
        # therefore we cannot do batch inference
        ort_sess = ort.InferenceSession(self._model.SerializeToString())
        outputs = ort_sess.run(None, {"data": inputs.astype(np.float32)})

        boxes, classes, confidences = outputs

        # Calculate input derivatives
        avg_px_intensity = np.mean(inputs[0])
        
        # Calculate output derivatives
        avg_confidence = np.mean(confidences)

        # batch size isn't specified in the onnx session output
        # but we need to return a batch of outputs
        return {
            "boxes": np.array([boxes]),
            "classes": np.array([classes]),
            "confidences": np.array([confidences]),
            "avg_px_intensity": np.array([[avg_px_intensity]]),
            "avg_confidence": np.array([[avg_confidence]]),
        }


class PixIntensityResnetBuilder(InferenceBuilder):
    @property
    def inference(self) -> PixIntensityResnet:
        return PixIntensityResnet

    def create(self, config: CustomInferenceConfig) -> PixIntensityResnet:
        inference = self.inference()
        inference.model = self._load_model(
            config.model_path / "frcnn-resnet.pt.onnx"
        )
        return inference

    def _load_model(self, file_path: Path) -> onnx.onnx_ml_pb2.ModelProto:
        return onnx.load(file_path)
