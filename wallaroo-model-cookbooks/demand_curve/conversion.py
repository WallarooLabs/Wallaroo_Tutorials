import pandas
import numpy

def pandas_to_dict(df):
    input_dict = {
    'colnames': list(df.columns),
    'query': df.to_numpy().tolist()
    }
    return input_dict