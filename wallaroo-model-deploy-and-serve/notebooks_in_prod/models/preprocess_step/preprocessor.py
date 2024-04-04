import datetime
import logging

import numpy as np
import pandas as pd

from mac.types import InferenceData

logger = logging.getLogger(__name__)

_vars = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
    "house_age",
    "renovated",
    "yrs_since_reno",
]


def process_data(input_data: InferenceData) -> InferenceData:
    input_df = pd.DataFrame(input_data)
    thisyear = datetime.datetime.now().year
    input_df["house_age"] = thisyear - input_df["yr_built"]
    input_df["renovated"] = np.where((input_df["yr_renovated"] > 0), 1, 0)
    input_df["yrs_since_reno"] = np.where(
        input_df["renovated"],
        input_df["yr_renovated"] - input_df["yr_built"],
        0,
    )
    input_df = input_df.loc[:, _vars]

    return {"tensor": input_df.to_numpy(dtype=np.float32)}
