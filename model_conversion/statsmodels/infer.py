import json
import pandas as pd

# What if we just change data to a DataFrame.
def wallaroo_json(data):
    # obj = json.loads(data)
    # evaluation_frame = pd.DataFrame.from_dict(obj)
    evaluation_frame = data
    extra_regressors = ["temp", "holiday", "workingday", "windspeed"]
    forecast = model.forecast(steps=7, exog=evaluation_frame.loc[:, extra_regressors])

    # return {"forecast": forecast.tolist()}
    return {"forecast": forecast}
