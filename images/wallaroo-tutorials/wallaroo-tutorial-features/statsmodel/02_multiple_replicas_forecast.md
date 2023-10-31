This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-features/pipeline_multiple_replicas_forecast_tutorial).

## Statsmodel Forecast with Wallaroo Features: Parallel Inference

This tutorial series demonstrates how to use Wallaroo to create a Statsmodel forecasting model based on bike rentals.  This tutorial series is broken down into the following:

* Create and Train the Model:  This first notebook shows how the model is trained from existing data.
* Deploy and Sample Inference:  With the model developed, we will deploy it into Wallaroo and perform a sample inference.
* Parallel Infer:  A sample of multiple weeks of data will be retrieved and submitted as an asynchronous parallel inference.  The results will be collected and uploaded to a sample database.
* External Connection:  A sample data connection to Google BigQuery to retrieve input data and store the results in a table.
* ML Workload Orchestration:  Take all of the previous steps and automate the request into a single Wallaroo ML Workload Orchestration.

This step will use the simulated database `simdb` to gather 4 weeks of inference data, then submit the inference request through the asynchronous Pipeline method `parallel_infer`.  This receives a List of inference data, submits it to the Wallaroo pipeline, then receives the results as a separate list with each inference matched to the input submitted.

The results are then compared against the actual data to see if the model was accurate.

## Prerequisites

* A Wallaroo instance version 2023.2.1 or greater.

## References

* [Wallaroo SDK Essentials Guide: Model Uploads and Registrations: Python Models](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/wallaroo-sdk-model-upload-python/)
* [Wallaroo SDK Essentials Guide: Pipeline Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline/)
* [Wallaroo SDK Essentials: Inference Guide: Parallel Inferences](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-inferences/#parallel-inferences)

## Parallel Infer Steps

### Import Libraries

The first step is to import the libraries that we will need.

```python
import json
import os
import datetime

import wallaroo
from wallaroo.object import EntityNotFoundError
from wallaroo.framework import Framework

# used to display dataframe information without truncating
from IPython.display import display
import pandas as pd
import numpy as np

from resources import simdb
from resources import util

pd.set_option('display.max_colwidth', None)
```

```python
display(wallaroo.__version__)
```

    '2023.2.1rc2'

### Initialize connection

Start a connect to the Wallaroo instance and save the connection into the variable `wl`.

```python
# Login through local Wallaroo instance

wl = wallaroo.Client()
```

### Set Configurations

The following will set the workspace, model name, and pipeline that will be used for this example.  If the workspace or pipeline already exist, then they will assigned for use in this example.  If they do not exist, they will be created based on the names listed below.

Workspace names must be unique.  To allow this tutorial to run in the same Wallaroo instance for multiple users, the `suffix` variable is generated from a random set of 4 ASCII characters.  To use the same workspace across the tutorial notebooks, hard code `suffix` and verify the workspace name created is is unique across the Wallaroo instance.

```python
# used for unique connection names

import string
import random

suffix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))

workspace_name = f'multiple-replica-forecast-tutorial-{suffix}'
pipeline_name = 'bikedaypipe'
model_name = 'bikedaymodel'
```

### Set the Workspace and Pipeline

The workspace will be either used or created if it does not exist, along with the pipeline.

```python
def get_workspace(name):
    workspace = None
    for ws in wl.list_workspaces():
        if ws.name() == name:
            workspace= ws
    if(workspace == None):
        workspace = wl.create_workspace(name)
    return workspace

def get_pipeline(name):
    try:
        pipeline = wl.pipelines_by_name(name)[0]
    except EntityNotFoundError:
        pipeline = wl.build_pipeline(name)
    return pipeline

workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)

pipeline = get_pipeline(pipeline_name)
```

```python
model_file_name = 'forecast.py'

bike_day_model = wl.upload_model(model_name, model_file_name, Framework.PYTHON).configure(runtime="python")
```

### Upload Model

The Python model created in "Forecast and Parallel Infer with Statsmodel: Model Creation" will now be uploaded.  Note that the Framework and runtime are set to `python`.

```python
pipeline.add_model_step(bike_day_model)
```

<table><tr><th>name</th> <td>bikedaypipe</td></tr><tr><th>created</th> <td>2023-07-14 15:50:50.014326+00:00</td></tr><tr><th>last_updated</th> <td>2023-07-14 15:50:52.029628+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>7aae4653-9e9f-468c-b266-4433be652313, 48983f9b-7c43-41fe-9688-df72a6aa55e9</td></tr><tr><th>steps</th> <td>bikedaymodel</td></tr></table>

### Deploy the Pipeline

We will now add the uploaded model as a step for the pipeline, then deploy it.  The pipeline configuration will allow for multiple replicas of the pipeline to be deployed and spooled up in the cluster.  Each pipeline replica will use 0.25 cpu and 512 Gi RAM.

```python
# Set the deployment to allow for additional engines to run
deploy_config = (wallaroo.DeploymentConfigBuilder()
                        .replica_count(1)
                        .replica_autoscale_min_max(minimum=2, maximum=5)
                        .cpus(0.25)
                        .memory("512Mi")
                        .build()
                    )

pipeline.deploy(deployment_config = deploy_config)
```

<table><tr><th>name</th> <td>bikedaypipe</td></tr><tr><th>created</th> <td>2023-07-14 15:53:07.284131+00:00</td></tr><tr><th>last_updated</th> <td>2023-07-14 15:56:07.413409+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>9c67dd93-014c-4cc9-9b44-549829e613ad, 258dafaf-c272-4bda-881b-5998a4a9be26</td></tr><tr><th>steps</th> <td>bikedaymodel</td></tr></table>

### Run Inference
For this example, we will forecast bike rentals by looking back one month from "today" which will be set as 2011-02-22.  The data from 2011-01-23 to 2011-01-27 (the 5 days starting from one month back) are used to generate a forecast for what bike sales will be over the next week from "today", which will be 2011-02-23 to 2011-03-01.

```python
# retrieve forecast schedule
first_day, analysis_days = util.get_forecast_days()

print(f'Running analysis on {first_day}')
```

    Running analysis on 2011-02-22

```python
# connect to SQL data base 
conn = simdb.get_db_connection()
print(f'Bike rentals table: {simdb.tablename}')

# create the query and retrieve data
query = util.mk_dt_range_query(tablename=simdb.tablename, forecast_day=first_day)
print(query)
data = pd.read_sql_query(query, conn)
data.head()
```

    Bike rentals table: bikerentals
    select cnt from bikerentals where date > DATE(DATE('2011-02-22'), '-1 month') AND date <= DATE('2011-02-22')

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
      <td>986</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1416</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1985</td>
    </tr>
    <tr>
      <th>3</th>
      <td>506</td>
    </tr>
    <tr>
      <th>4</th>
      <td>431</td>
    </tr>
  </tbody>
</table>

```python
pd.read_sql_query("select date, cnt from bikerentals where date > DATE(DATE('2011-02-22'), '-1 month') AND date <= DATE('2011-02-22') LIMIT 5", conn)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>cnt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-23</td>
      <td>986</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-24</td>
      <td>1416</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-25</td>
      <td>1985</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-26</td>
      <td>506</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-27</td>
      <td>431</td>
    </tr>
  </tbody>
</table>

```python
# send data to model for forecast

results = pipeline.infer(data.to_dict(orient='list'))[0]
results

```

    {'forecast': [1462, 1483, 1497, 1507, 1513, 1518, 1521]}

```python
# annotate with the appropriate dates (the next seven days)
resultframe = pd.DataFrame({
    'date' : util.get_forecast_dates(first_day),
    'forecast' : results['forecast']
})

# write the new data to the db table "bikeforecast"
resultframe.to_sql('bikeforecast', conn, index=False, if_exists='append')

# display the db table
query = "select date, forecast from bikeforecast"
pd.read_sql_query(query, conn)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>forecast</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-02-23</td>
      <td>1462</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-02-24</td>
      <td>1483</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-02-25</td>
      <td>1497</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-02-26</td>
      <td>1507</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-02-27</td>
      <td>1513</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2011-02-28</td>
      <td>1518</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2011-03-01</td>
      <td>1521</td>
    </tr>
  </tbody>
</table>

### Four Weeks of Inference Data

Now we'll go back staring at the "current data" of 2011-03-01, and fetch each week's data across the month.  This will be used to submit 5 inference requests through the Pipeline `parallel_infer` method.

The inference data is saved into the `inference_data` List - each element in the list will be a separate inference request.

```python
# get our list of items to run through

inference_data = []

content_type = "application/json"

days = []

for day in analysis_days:
    print(f"Current date: {day}")
    days.append(day)
    query = util.mk_dt_range_query(tablename=simdb.tablename, forecast_day=day)
    print(query)
    data = pd.read_sql_query(query, conn)
    inference_data.append(data.to_dict(orient='list'))
```

    Current date: 2011-03-01
    select cnt from bikerentals where date > DATE(DATE('2011-03-01'), '-1 month') AND date <= DATE('2011-03-01')
    Current date: 2011-03-08
    select cnt from bikerentals where date > DATE(DATE('2011-03-08'), '-1 month') AND date <= DATE('2011-03-08')
    Current date: 2011-03-15
    select cnt from bikerentals where date > DATE(DATE('2011-03-15'), '-1 month') AND date <= DATE('2011-03-15')
    Current date: 2011-03-22
    select cnt from bikerentals where date > DATE(DATE('2011-03-22'), '-1 month') AND date <= DATE('2011-03-22')
    Current date: 2011-03-29
    select cnt from bikerentals where date > DATE(DATE('2011-03-29'), '-1 month') AND date <= DATE('2011-03-29')

### Parallel Inference Request

The List `inference_data` will be submitted.  Recall that the pipeline deployment can spool up to 5 replicas.

The pipeline `parallel_infer(tensor_list, timeout, num_parallel, retries)` **asynchronous** method performs an inference as defined by the pipeline steps and takes the following arguments:

* **tensor_list** (*REQUIRED List*): The data submitted to the pipeline for inference as a List of the supported data types:
  * [pandas.DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html):  Data submitted as a pandas DataFrame are returned as a pandas DataFrame.  For models that output one column  based on the models outputs.
  * [Apache Arrow](https://arrow.apache.org/) (**Preferred**): Data submitted as an Apache Arrow are returned as an Apache Arrow.
* **timeout** (*OPTIONAL int*): A timeout in seconds before the inference throws an exception.  The default is 15 second per call to accommodate large, complex models.  Note that for a batch inference, this is **per list item** - with 10 inference requests, each would have a default timeout of 15 seconds.
* **num_parallel** (*OPTIONAL int*):  The number of parallel threads used for the submission.  **This should be no more than four times the number of pipeline replicas**.
* **retries** (*OPTIONAL int*):  The number of retries per inference request submitted.

`parallel_infer` is an asynchronous method that returns the Python callback list of tasks. Calling `parallel_infer` should be called with the `await` keyword to retrieve the callback results.

For more details, see the Wallaroo [parallel inferences guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-inferences/#parallel-inferences).

```python
parallel_results = await pipeline.parallel_infer(tensor_list=inference_data, timeout=20, num_parallel=16, retries=2)

display(parallel_results)
```

    [[{'forecast': [1764, 1749, 1743, 1741, 1740, 1740, 1740]}],
     [{'forecast': [1735, 1858, 1755, 1841, 1770, 1829, 1780]}],
     [{'forecast': [1878, 1851, 1858, 1856, 1857, 1856, 1856]}],
     [{'forecast': [2363, 2316, 2277, 2243, 2215, 2192, 2172]}],
     [{'forecast': [2225, 2133, 2113, 2109, 2108, 2108, 2108]}]]

### Upload into DataBase

With our results, we'll merge the results we have into the days we were looking to analyze.  Then we can upload the results into the sample database and display the results.

```python
# merge the days and the results

days_results = list(zip(days, parallel_results))
```

```python
# upload to the database
for day_result in days_results:
    resultframe = pd.DataFrame({
        'date' : util.get_forecast_dates(day_result[0]),
        'forecast' : day_result[1][0]['forecast']
    })
    resultframe.to_sql('bikeforecast', conn, index=False, if_exists='append')
```

On April 1st, we can compare March forecasts to actuals

```python
query = f'''SELECT bikeforecast.date AS date, forecast, cnt AS actual
            FROM bikeforecast LEFT JOIN bikerentals
            ON bikeforecast.date = bikerentals.date
            WHERE bikeforecast.date >= DATE('2011-03-01')
            AND bikeforecast.date <  DATE('2011-04-01')
            ORDER BY 1'''

print(query)

comparison = pd.read_sql_query(query, conn)
comparison
```

    SELECT bikeforecast.date AS date, forecast, cnt AS actual
                FROM bikeforecast LEFT JOIN bikerentals
                ON bikeforecast.date = bikerentals.date
                WHERE bikeforecast.date >= DATE('2011-03-01')
                AND bikeforecast.date <  DATE('2011-04-01')
                ORDER BY 1

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>forecast</th>
      <th>actual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-03-02</td>
      <td>1764</td>
      <td>2134</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-03-03</td>
      <td>1749</td>
      <td>1685</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-03-04</td>
      <td>1743</td>
      <td>1944</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-03-05</td>
      <td>1741</td>
      <td>2077</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-03-06</td>
      <td>1740</td>
      <td>605</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2011-03-07</td>
      <td>1740</td>
      <td>1872</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2011-03-08</td>
      <td>1740</td>
      <td>2133</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2011-03-09</td>
      <td>1735</td>
      <td>1891</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2011-03-10</td>
      <td>1858</td>
      <td>623</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2011-03-11</td>
      <td>1755</td>
      <td>1977</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2011-03-12</td>
      <td>1841</td>
      <td>2132</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2011-03-13</td>
      <td>1770</td>
      <td>2417</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2011-03-14</td>
      <td>1829</td>
      <td>2046</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2011-03-15</td>
      <td>1780</td>
      <td>2056</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2011-03-16</td>
      <td>1878</td>
      <td>2192</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2011-03-17</td>
      <td>1851</td>
      <td>2744</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2011-03-18</td>
      <td>1858</td>
      <td>3239</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2011-03-19</td>
      <td>1856</td>
      <td>3117</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2011-03-20</td>
      <td>1857</td>
      <td>2471</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2011-03-21</td>
      <td>1856</td>
      <td>2077</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2011-03-22</td>
      <td>1856</td>
      <td>2703</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2011-03-23</td>
      <td>2363</td>
      <td>2121</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2011-03-24</td>
      <td>2316</td>
      <td>1865</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2011-03-25</td>
      <td>2277</td>
      <td>2210</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2011-03-26</td>
      <td>2243</td>
      <td>2496</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2011-03-27</td>
      <td>2215</td>
      <td>1693</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2011-03-28</td>
      <td>2192</td>
      <td>2028</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2011-03-29</td>
      <td>2172</td>
      <td>2425</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2011-03-30</td>
      <td>2225</td>
      <td>1536</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2011-03-31</td>
      <td>2133</td>
      <td>1685</td>
    </tr>
  </tbody>
</table>

### Undeploy the Pipeline

Undeploy the pipeline and return the resources back to the Wallaroo instance.

```python
conn.close()
pipeline.undeploy()
```

<table><tr><th>name</th> <td>bikedaypipe</td></tr><tr><th>created</th> <td>2023-07-14 15:53:07.284131+00:00</td></tr><tr><th>last_updated</th> <td>2023-07-14 15:56:07.413409+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>9c67dd93-014c-4cc9-9b44-549829e613ad, 258dafaf-c272-4bda-881b-5998a4a9be26</td></tr><tr><th>steps</th> <td>bikedaymodel</td></tr></table>

