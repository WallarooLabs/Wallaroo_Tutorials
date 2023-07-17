## Statsmodel Forecast with Wallaroo Features: Model Creation

This tutorial series demonstrates how to use Wallaroo to create a Statsmodel forecasting model based on bike rentals.  This tutorial series is broken down into the following:

* Create and Train the Model:  This first notebook shows how the model is trained from existing data.
* Deploy and Sample Inference:  With the model developed, we will deploy it into Wallaroo and perform a sample inference.
* Parallel Infer:  A sample of multiple weeks of data will be retrieved and submitted as an asynchronous parallel inference.  The results will be collected and uploaded to a sample database.
* External Connection:  A sample data connection to Google BigQuery to retrieve input data and store the results in a table.
* ML Workload Orchestration:  Take all of the previous steps and automate the request into a single Wallaroo ML Workload Orchestration.

## Prerequisites

* A Wallaroo instance version 2023.2.1 or greater.

## References

* [Wallaroo SDK Essentials Guide: Model Uploads and Registrations: Python Models](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/wallaroo-sdk-model-upload-python/)
* [Wallaroo SDK Essentials Guide: Pipeline Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline/)
* [Wallaroo SDK Essentials: Inference Guide: Parallel Inferences](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-inferences/#parallel-inferences)

```python
import pandas as pd
import datetime
import os

from statsmodels.tsa.arima.model import ARIMA
from resources import simdb as simdb
```

### Train the Model

The resources to train the model will start with the local file `day.csv`.  This data is load and prepared for use in training the model.

For this example, the simulated database is controled by the resources `simbdb`.

```python
def mk_dt_range_query(*, tablename: str, seed_day: str) -> str:
    assert isinstance(tablename, str)
    assert isinstance(seed_day, str)
    query = f"select cnt from {tablename} where date > DATE(DATE('{seed_day}'), '-1 month') AND date <= DATE('{seed_day}')"
    return query

conn = simdb.get_db_connection()

# create the query
query = mk_dt_range_query(tablename=simdb.tablename, seed_day='2011-03-01')
print(query)

# read in the data
training_frame = pd.read_sql_query(query, conn)
training_frame
```

    select cnt from bikerentals where date > DATE(DATE('2011-03-01'), '-1 month') AND date <= DATE('2011-03-01')

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1526</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1550</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1708</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1005</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1623</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1712</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1530</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1605</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1538</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1746</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1472</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1589</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1913</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1815</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2115</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2475</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2927</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1635</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1812</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1107</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1450</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1917</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1807</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1461</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1969</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2402</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1446</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1851</td>
    </tr>
  </tbody>
</table>

## Test the Forecast

The training frame is then loaded, and tested against our `forecast` model.

```python
# test
import forecast
import json

# create the appropriate json
jsonstr = json.dumps(training_frame.to_dict(orient='list'))
print(jsonstr)

forecast.wallaroo_json(jsonstr)
```

    {"cnt": [1526, 1550, 1708, 1005, 1623, 1712, 1530, 1605, 1538, 1746, 1472, 1589, 1913, 1815, 2115, 2475, 2927, 1635, 1812, 1107, 1450, 1917, 1807, 1461, 1969, 2402, 1446, 1851]}

    {'forecast': [1764, 1749, 1743, 1741, 1740, 1740, 1740]}

### Reload New Model

The `forecast` model is reloaded in preparation of creating the evaluation data.

```python
import importlib
importlib.reload(forecast)
```

    <module 'forecast' from '/home/jovyan/pipeline_multiple_replicas_forecast_tutorial/forecast.py'>

### Prepare evaluation data

For ease of inference, we save off the evaluation data to a separate json file.

```python
# save off the evaluation frame json, too
import json
with open("./data/testdata_dict.json", "w") as f:
    json.dump(training_frame.to_dict(orient='list'), f)

```
