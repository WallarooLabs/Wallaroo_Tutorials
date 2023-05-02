import json
import pandas as pd

def wallaroo_json(data):
    obj = json.loads(data)
    evaluation_frame = pd.DataFrame.from_dict(obj)
    extra_regressors = ["temp", "holiday", "workingday", "windspeed"]
    forecast = model.forecast(steps=7, exog=evaluation_frame.loc[:, extra_regressors])

    return {"forecast": forecast.tolist()}