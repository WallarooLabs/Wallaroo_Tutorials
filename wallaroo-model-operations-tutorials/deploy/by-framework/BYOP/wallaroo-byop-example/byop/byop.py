#byop modules
from pathlib import Path
from typing import Any, Dict, Set
from mac.config.inference import CustomInferenceConfig
from mac.inference import Inference
from mac.inference.creation import InferenceBuilder
from mac.types import InferenceData

import pandas as pd
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(__file__))
from custom_packages.custom_script import complex_algorithm

import logging
import traceback

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

from datetime import datetime, timezone

class BYOPInference(Inference):
    @property
    def expected_model_types(self) -> Set[Any]:
        return {Dict}

    @Inference.model.setter
    def model(self, model) -> None:
        self._model = model

    def _predict(self, input_data: InferenceData) -> InferenceData:

        logger.info("Starting prediction process")
        try:
            logger.info(f"Gathering of input data features: {len(input_data)}")
            results = []

            logger.info("Converting input data to DataFrame")
            df = pd.DataFrame({
                key : value.tolist() for key, value in input_data.items()
                })
            
            try:
                # --- Run model prediction ---
                logger.info("Running model prediction")
                for index, row in df.iterrows():
                    input_number = row['input_number']
                    result = complex_algorithm(input_number)
                    results.append(result)


            except Exception as e:
                logger.error(f"Error during model prediction: {e}")
                logger.error(traceback.format_exc())
                raise e
                        

            logger.info("Predictions completed.")
            
            output_dictionary = { 
                "result": np.array(results, dtype=np.int64),
                "id": np.array(input_data["id"].tolist(), dtype=np.int64)
                }

            return output_dictionary

            
        
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            logger.error(traceback.format_exc())
            raise e


class BYOPInferenceBuilder(InferenceBuilder):
    @property
    def inference(self) -> BYOPInference:
        return BYOPInference

    def create(self, config: CustomInferenceConfig) -> BYOPInference:
        inference = self.inference()

        # when loading a model artifacts
        model = self._load_models(config.model_path)
        inference.model = model

        return inference
    
    def _load_models(self, model_path: Path):

        return {
            'dummy_model': "wallaroo",
        }