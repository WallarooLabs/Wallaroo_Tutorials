import numpy
import pandas as pd
import statistics 
from statistics import mean
import json

def wallaroo_json(data: pd.DataFrame):
    print("-- wallaroo_json -- ")
    # calculate precision
    confidences = data["confidences"].values
    boxes = data["boxes"].values
    classes = data["classes"].values
    avg_conf = data["confidences"][0].mean()
    return pd.DataFrame(
        {
            'boxes' : boxes,
            'classes' : classes,
            'confidences' : confidences,
            'avg_conf': [avg_conf]
        }
    )