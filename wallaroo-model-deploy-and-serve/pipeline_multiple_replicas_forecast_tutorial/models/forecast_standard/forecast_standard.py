import numpy as np
import pandas as pd
import statistics 
from statistics import mean
import json

from mac.types import InferenceData

from statsmodels.tsa.arima.model import ARIMA

def _fit_model(dataframe):
    model = ARIMA(dataframe['count'], 
                    order=(1, 0, 1)
                    ).fit()
    return model

def process_data(input_data: InferenceData)-> InferenceData:
    evaluation_frame = pd.DataFrame({"count": input_data['count']})

    nforecast = 7
    model = _fit_model(evaluation_frame)

    # get a numpy array
    forecast = model.forecast(steps=nforecast).round().to_numpy()
    forecast = forecast.astype(int)

    # get the average across the week
    weekly_average = np.array([forecast.mean()])

    return { 
            "forecast" : forecast,
            "weekly_average": weekly_average
        }


# def wallaroo_json(data: pd.DataFrame):

#     evaluation_frame = pd.DataFrame({"count": data.loc[0, 'count']})

#     nforecast = 7
#     model = _fit_model(evaluation_frame)

#     forecast =  model.forecast(steps=nforecast).round().to_numpy()
#     forecast = forecast.astype(int)

#     # get the average across the week
#     weekly_average = forecast.mean()

#     return [
#         { "forecast" : forecast.tolist(),
#           "weekly_average": [weekly_average] 
#         }
#     ]
