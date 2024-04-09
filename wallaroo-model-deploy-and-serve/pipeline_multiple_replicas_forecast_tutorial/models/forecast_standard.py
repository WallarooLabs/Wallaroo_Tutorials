import json
import pandas as pd

import numpy as np

from statsmodels.tsa.arima.model import ARIMA

def _fit_model(dataframe):
    model = ARIMA(dataframe['count'], 
                    order=(1, 0, 1)
                    ).fit()
    return model

from mac.types import InferenceData

#logger = logging.getLogger(__name__)

def process_data(input_data: InferenceData) -> InferenceData:
    evaluation_frame = pd.DataFrame([{"count": input_data['count'].tolist()}])
    print(evaluation_frame['count'])
    nforecast = 7
    model = _fit_model(evaluation_frame)

    # forecast =  model.forecast(steps=nforecast).round().to_numpy()(dtype=np.int)
    # # forecast = forecast.astype(int)

    # # get the average across the week
    # weekly_average = forecast.mean()

    # return { 
    #     "forecast" : forecast.tolist(),
    #     "weekly_average": [weekly_average] 
    # }
