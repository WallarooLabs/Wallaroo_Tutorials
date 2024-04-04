import pandas as pd

import logging

import numpy as np

from mac.types import InferenceData

logger = logging.getLogger(__name__)

# # take a dataframe output of the house price model, and reformat the `dense_2`
# # column as `output`
# def wallaroo_json(data: pd.DataFrame):
#     print(data)
#     return [{"output": [data["dense_2"].to_list()[0][0]]}]


def process_data(input_data: InferenceData) -> InferenceData:
    # just changing the 
    output_data = {
        'output': input_data['dense_2']
    }
    return output_data

# for generic testing
# def process_data(input_data):
#     # convert to log10
#     output_data = {
#         'output': input_data['dense_2']
#     }
#     return output_data

# def process_data(input_data: InferenceData) -> InferenceData:
#     # convert to log10
#     input_data["variable"] = np.rint(np.power(10, input_data["variable"]))
#     return input_data