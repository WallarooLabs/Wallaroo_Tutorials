import pandas as pd

import logging

import numpy as np

from mac.types import InferenceData

def process_data(input_data: InferenceData) -> InferenceData:
    # just changing the output data field
    input_data['output'] = input_data.pop('dense_2')
    return input_data
