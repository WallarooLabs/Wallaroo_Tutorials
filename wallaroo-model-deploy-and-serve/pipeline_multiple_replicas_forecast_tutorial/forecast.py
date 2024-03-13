import json
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA

def _fit_model(dataframe):
    model = ARIMA(dataframe['count'], 
                    order=(1, 0, 1)
                    ).fit()
    return model


def wallaroo_json(data):
    obj = json.loads(data)
    evaluation_frame = pd.DataFrame.from_dict(obj)

    nforecast = 7
    model = _fit_model(evaluation_frame)

    forecast =  model.forecast(steps=nforecast).round().to_numpy()
    forecast = forecast.astype(int)

    return {"forecast": forecast.tolist()}
