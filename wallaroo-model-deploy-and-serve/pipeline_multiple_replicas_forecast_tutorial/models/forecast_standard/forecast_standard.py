import numpy as np
import pandas as pd
import logging
from mac.types import InferenceData

from statsmodels.tsa.arima.model import ARIMA

logger = logging.getLogger(__name__)


def _fit_model(dataframe):
    model = ARIMA(dataframe["count"], order=(1, 0, 1)).fit()
    return model


def process_data(input_data: InferenceData) -> InferenceData:
    n_forecast = 7
    forecasts = []
    weekly_averages = []

    for row in input_data["count"]:
        evaluation_frame = pd.DataFrame({"count": row})
        model = _fit_model(evaluation_frame)

        # get a numpy array
        forecast = model.forecast(steps=n_forecast).round().to_numpy()
        forecast = forecast.astype(int)

        # get the average across the week
        weekly_average = forecast.mean()

        forecasts.append(forecast)
        weekly_averages.append(weekly_average)

    return {
        "forecast": np.array(forecasts),
        "weekly_average": np.array(weekly_averages),
    }
