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





class ImageClusteringBuilder(InferenceBuilder):
    """InferenceBuilder subclass for ImageClustering, that loads
    a pre-trained VGG16 model on cifar10 as a feature extractor
    and a trained KMeans model, and creates an ImageClustering object."""

    @property
    def inference(self) -> ImageClustering:
        return ImageClustering

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