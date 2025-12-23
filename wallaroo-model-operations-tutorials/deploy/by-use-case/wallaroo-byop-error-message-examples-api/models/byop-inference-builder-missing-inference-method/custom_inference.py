"""This module features an example implementation of a custom Inference and its
corresponding InferenceBuilder."""

import pathlib
import pickle
from typing import Any, Set

import tensorflow as tf
from mac.config.inference import CustomInferenceConfig
from mac.inference import Inference
from mac.inference.creation import InferenceBuilder
from mac.types import InferenceData
from sklearn.cluster import KMeans


class ImageClustering(Inference):
    """Inference class for image clustering, that uses
    a pre-trained VGG16 model on cifar10 as a feature extractor
    and performs clustering on a trained KMeans model.

    Attributes:
        - feature_extractor: The embedding model we will use
        as a feature extractor (i.e. a trained VGG16).
        - expected_model_types: A set of model instance types that are expected by this inference.
        - model: The model on which the inference is calculated.
    """

    def __init__(self, feature_extractor: tf.keras.Model):
        self.feature_extractor = feature_extractor
        super().__init__()

    @property
    def expected_model_types(self) -> Set[Any]:
        return {KMeans}

    @Inference.model.setter  # type: ignore
    def model(self, model) -> None:
        """Sets the model on which the inference is calculated.

        :param model: A model instance on which the inference is calculated.

        :raises TypeError: If the model is not an instance of expected_model_types
            (i.e. KMeans).
        """
        self._raise_error_if_model_is_wrong_type(model) # this will make sure an error will be raised if the model is of wrong type
        self._model = model

    def _predict(self, input_data: InferenceData) -> InferenceData:
        """Calculates the inference on the given input data.
        This is the core function that each subclass needs to implement
        in order to calculate the inference.

        :param input_data: The input data on which the inference is calculated.
        It is of type InferenceData, meaning it comes as a dictionary of numpy
        arrays.

        :raises InferenceDataValidationError: If the input data is not valid.
        Ideally, every subclass should raise this error if the input data is not valid.

        :return: The output of the model, that is a dictionary of numpy arrays.
        """

        # input_data maps to the input_schema we have defined
        # with PyArrow, coming as a dictionary of numpy arrays
        inputs = input_data["images"]

        # Forward inputs to the models
        embeddings = self.feature_extractor(inputs)
        predictions = self.model.predict(embeddings.numpy())

        # Return predictions as dictionary of numpy arrays
        return {"predictions": predictions}


class ImageClusteringBuilder(InferenceBuilder):
    """InferenceBuilder subclass for ImageClustering, that loads
    a pre-trained VGG16 model on cifar10 as a feature extractor
    and a trained KMeans model, and creates an ImageClustering object."""

    

    def create(self, config: CustomInferenceConfig) -> ImageClustering:
        """Creates an Inference subclass and assigns a model and additionally
        needed attributes to it.

        :param config: Custom inference configuration. In particular, we're
        interested in `config.model_path` that is a pathlib.Path object
        pointing to the folder where the model artifacts are saved.
        Every artifact we need to load from this folder has to be
        relative to `config.model_path`.

        :return: A custom Inference instance.
        """
        feature_extractor = self._load_feature_extractor(
            config.model_path / "feature_extractor.h5"
        )
        inference = self.inference(feature_extractor)
        model = self._load_model(config.model_path / "kmeans.pkl")
        inference.model = model

        return inference

    def _load_feature_extractor(
        self, file_path: pathlib.Path
    ) -> tf.keras.Model:
        return tf.keras.models.load_model(file_path)

    def _load_model(self, file_path: pathlib.Path) -> KMeans:
        with open(file_path.as_posix(), "rb") as fp:
            model = pickle.load(fp)
        return model