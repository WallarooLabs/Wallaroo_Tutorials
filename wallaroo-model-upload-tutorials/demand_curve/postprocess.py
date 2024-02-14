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


    # rename 'variable' to 'prediction'
    # data = data['variable'].apply(lambda x: [0] if x[0] < 0 else x[0])

    # prediction = data_nozero.rename(columns={'out.variable':'out.prediction'})

    # prediction = 
    # # if variables[x] < 0, replace with [0]
    data = data.variable.apply(remove_zero)
    return data

def wallaroo_json(data:pandas.DataFrame):
    return [{'prediction': actual_postprocess(data)}]
