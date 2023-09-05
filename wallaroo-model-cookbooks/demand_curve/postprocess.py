import numpy
import pandas

import json


def remove_zero(x):
    if x[0] < 0:
        x[0] = 0
        return x
    else:
        x[0] = x[0]
        return x

# Check for negative predictions, and zero them out
def actual_postprocess(data:pandas.DataFrame):

    # if variables[x] < 0, replace with [0]
    data.variable.apply(remove_zero)

    # rename 'variable' to 'prediction'
    prediction = data.rename(columns={'variable':'prediction'})
    return prediction

def wallaroo_json(data:pandas.DataFrame):
    return actual_postprocess(data).to_dict(orient="records")
