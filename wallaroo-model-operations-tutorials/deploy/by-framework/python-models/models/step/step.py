import pandas as pd

import logging

import numpy as np

from mac.types import InferenceData

def process_data(input_data: InferenceData) -> InferenceData:
    # just changing the output data field
    return {
        'output': np.float32(input_data.pop('dense_2'))
    }
