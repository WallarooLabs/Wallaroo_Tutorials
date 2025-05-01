import logging

import numpy as np

from mac.types import InferenceData

logger = logging.getLogger(__name__)


def process_data(input_data: InferenceData) -> InferenceData:
    # convert to log10
    input_data["variable"] = np.rint(np.power(10, input_data["variable"]))
    return input_data
