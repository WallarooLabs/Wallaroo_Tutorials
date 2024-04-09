import numpy as np
import pandas as pd
import statistics 
from statistics import mean
import json

from mac.types import InferenceData

def process_data(input_data: InferenceData)-> InferenceData:

        # Calculate output derivatives for confidences and store into avg_confidence
        input_data['avg_confidence'] = np.array([np.mean(input_data["confidences"])])
        return input_data
