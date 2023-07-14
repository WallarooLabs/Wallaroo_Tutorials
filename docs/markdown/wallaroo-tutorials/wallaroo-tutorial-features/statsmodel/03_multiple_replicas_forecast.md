## Statsmodel Forecast with Wallaroo Features: Data Connection

This tutorial series demonstrates how to use Wallaroo to create a Statsmodel forecasting model based on bike rentals.  This tutorial series is broken down into the following:

* Create and Train the Model:  This first notebook shows how the model is trained from existing data.
* Deploy and Sample Inference:  With the model developed, we will deploy it into Wallaroo and perform a sample inference.
* Parallel Infer:  A sample of multiple weeks of data will be retrieved and submitted as an asynchronous parallel inference.  The results will be collected and uploaded to a sample database.
* External Connection:  A sample data connection to Google BigQuery to retrieve input data and store the results in a table.
* ML Workload Orchestration:  Take all of the previous steps and automate the request into a single Wallaroo ML Workload Orchestration.

For this step, we will use a Google BigQuery dataset to retrieve the inference information, predict the next month of sales, then store those predictions into another table.  This will use the Wallaroo Connection feature to create a Connection, assign it to our workspace, then perform our inferences by using the Connection details to connect to the BigQuery dataset and tables.

## Prerequisites

* A Wallaroo instance version 2023.2.1 or greater.
* Install the libraries from `./resources/requirements.txt` that include the following:
  * google-cloud-bigquery==3.10.0
  * google-auth==2.17.3
  * db-dtypes==1.1.1

## References

* [Wallaroo SDK Essentials Guide: Model Uploads and Registrations: Python Models](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/wallaroo-sdk-model-upload-python/)
* [Wallaroo SDK Essentials Guide: Pipeline Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline/)
* [Wallaroo SDK Essentials: Inference Guide: Parallel Inferences](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-inferences/#parallel-inferences)

## Statsmodel Forecast Connection Steps

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

# for Big Query connections
from google.cloud import bigquery
from google.oauth2 import service_account
import db_dtypes

import time
```

```python
display(wallaroo.__version__)
```

    '2023.3.0+65834aca6'

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

### Upload Model

The Python model created in "Forecast and Parallel Infer with Statsmodel: Model Creation" will now be uploaded.  Note that the Framework and runtime are set to `python`.

```python
model_file_name = 'forecast.py'

bike_day_model = wl.upload_model(model_name, model_file_name, Framework.PYTHON).configure(runtime="python")
```

```python
pipeline.add_model_step(bike_day_model)
```

<table><tr><th>name</th> <td>bikedaypipe</td></tr><tr><th>created</th> <td>2023-06-28 20:11:58.734248+00:00</td></tr><tr><th>last_updated</th> <td>2023-06-29 21:10:19.250680+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>93b113a2-f31a-4e05-883e-66a3d1fa10fb, 7d687c43-a833-4585-b607-7085eff16e9d, 504bb140-d9e2-4964-8f82-27b1d234f7f2, db1a14ad-c40c-41ac-82db-0cdd372172f3, 01d60d1c-7834-4d1f-b9a8-8ad569e114b6, a165cbbb-84d9-42e7-99ec-aa8e244aeb55, 0fefef8b-105e-4a6e-9193-d2e6d61248a1</td></tr><tr><th>steps</th> <td>bikedaymodel</td></tr></table>

### Deploy the Pipeline

We will now add the uploaded model as a step for the pipeline, then deploy it.  The pipeline configuration will allow for multiple replicas of the pipeline to be deployed and spooled up in the cluster.  Each pipeline replica will use 0.25 cpu and 512 Gi RAM.

```python
# Set the deployment to allow for additional engines to run
deploy_config = (wallaroo.DeploymentConfigBuilder()
                        .replica_count(4)
                        .cpus(0.25)
                        .memory("512Mi")
                        .build()
                    )

pipeline.deploy(deployment_config = deploy_config)
```

     ok

<table><tr><th>name</th> <td>bikedaypipe</td></tr><tr><th>created</th> <td>2023-06-28 20:11:58.734248+00:00</td></tr><tr><th>last_updated</th> <td>2023-06-29 21:12:00.676013+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>f5051ddf-1111-49e6-b914-f8d24f1f6a8a, 93b113a2-f31a-4e05-883e-66a3d1fa10fb, 7d687c43-a833-4585-b607-7085eff16e9d, 504bb140-d9e2-4964-8f82-27b1d234f7f2, db1a14ad-c40c-41ac-82db-0cdd372172f3, 01d60d1c-7834-4d1f-b9a8-8ad569e114b6, a165cbbb-84d9-42e7-99ec-aa8e244aeb55, 0fefef8b-105e-4a6e-9193-d2e6d61248a1</td></tr><tr><th>steps</th> <td>bikedaymodel</td></tr></table>

### Create the Connection

We have already demonstrated through the other notebooks in this series that we can use the statsmodel forecast model to perform an inference through a simulated database.  Now we'll create a Wallaroo connection that will store the credentials to a Google BigQuery database containining the information we're looking for.

The details of the connection are stored in the file `./resources/bigquery_service_account_statsmodel.json` that include the  [service account key file(SAK)](https://cloud.google.com/bigquery/docs/authentication/service-account-file) information, as well as the dataset and table used.  The details on how to generate the table and data for the sample `bike_rentals` table are stored in the file `./resources/create_bike_rentals.table`, with the data used stored in `./resources/bike_rentals.csv`.

Wallaroo connections are created through the Wallaroo Client `create_connection(name, type, details)` method.  See the [Wallaroo SDK Essentials Guide: Data Connections Management guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-dataconnections/) for full details.

With the credentials are three other important fields:

* `dataset`: The BigQuery dataset from the project specified in the service account credentials file.
* `input_table`: The table used for inference inputs.
* `output_table`: The table used to store results.

We'll add the helper method `get_connection`.  If the connection already exists, then Wallaroo will return an error.  If the connection with the same name already exists, it will retrieve it.  Verify that the connection does not already exist in the Wallaroo instance for proper functioning of this tutorial.

```python
forecast_connection_input_name = f'statsmodel-bike-rentals-{suffix}'
forecast_connection_input_type = "BIGQUERY"
forecast_connection_input_argument = json.load(open('./resources/bigquery_service_account_statsmodel.json'))

statsmodel_connection = wl.create_connection(forecast_connection_input_name, 
                                             forecast_connection_input_type,
                                             forecast_connection_input_argument)
display(statsmodel_connection)
```

<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Name</td><td>statsmodel-bike-rentals-jch</td>
  </tr>
  <tr>
    <td>Connection Type</td><td>BIGQUERY</td>
  </tr>
  <tr>
    <td>Details</td><td>*****</td>
  </tr>
  <tr>
    <td>Created At</td><td>2023-06-29T19:55:17.866728+00:00</td>
  </tr>
  <tr>
    <td>Linked Workspaces</td><td>['multiple-replica-forecast-tutorial-jch']</td>
  </tr>
</table>

### Add Connection to Workspace

We'll now add the connection to our workspace so it can be retrieved by other workspace users.  The method Workspace `add_connection(connection_name)` adds a Data Connection to a workspace.

```python
workspace.add_connection(forecast_connection_input_name)
```

### Retrieve Connection from Workspace

To simulate a data scientist's procedural flow, we'll now retrieve the connection from the workspace.

The method Workspace `list_connections()` displays a list of connections attached to the workspace. By default the details field is obfuscated.  Specific connections are retrieved by specifying their position in the returned list.

```python
forecast_connection = workspace.list_connections()[0]
display(forecast_connection)
```

<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Name</td><td>statsmodel-bike-rentals-jch</td>
  </tr>
  <tr>
    <td>Connection Type</td><td>BIGQUERY</td>
  </tr>
  <tr>
    <td>Details</td><td>*****</td>
  </tr>
  <tr>
    <td>Created At</td><td>2023-06-29T19:55:17.866728+00:00</td>
  </tr>
  <tr>
    <td>Linked Workspaces</td><td>['multiple-replica-forecast-tutorial-jch']</td>
  </tr>
</table>

### Run Inference from BigQuery Table

We'll now retrieve sample data through the Wallaroo connection, and perform a sample inference.  The connection details are retrieved through the Connection `details()` method.

The process is:

* Create the BigQuery credentials.
* Connect to the BigQuery dataset.
* Retrieve the inference data.

```python
bigquery_statsmodel_credentials = service_account.Credentials.from_service_account_info(
    forecast_connection.details())

bigquery_statsmodel_client = bigquery.Client(
    credentials=bigquery_statsmodel_credentials, 
    project=forecast_connection.details()['project_id']
)
```

```python
inference_inputs = bigquery_statsmodel_client.query(
        f"""
        select dteday as date, cnt FROM {forecast_connection.details()['dataset']}.{forecast_connection.details()['input_table']}
        where dteday > DATE_SUB(DATE('2011-02-22'), 
        INTERVAL 1 month) AND dteday <= DATE('2011-02-22') 
        ORDER BY dteday 
        LIMIT 5
        """
    ).to_dataframe().apply({"date":str, "cnt":int}).to_dict(orient='list')

# the original table sends back the date schema as a date, not text.  We'll convert it here.

# inference_inputs = inference_inputs.apply({"date":str, "cnt":int})

display(inference_inputs)

```

    {'date': ['2011-01-23',
      '2011-01-24',
      '2011-01-25',
      '2011-01-26',
      '2011-01-27'],
     'cnt': [986, 1416, 1985, 506, 431]}

### Perform Inference from BigQuery Connection Data

With the data retrieved, we'll perform an inference through it and display the result.

```python
results = pipeline.infer(inference_inputs)
results
```

    [{'forecast': [1177, 1023, 1082, 1060, 1068, 1065, 1066]}]

### Four Weeks of Inference Data

Now we'll go back staring at the "current data" of the next month in 2011, and fetch the previous month to that date, then use that to predict what sales will be over the next 7 days.

The inference data is saved into the `inference_data` List - each element in the list will be a separate inference request.

```python
# Start by getting the current month - we'll alway assume we're in 2011 to match the data store

month = datetime.datetime.now().month
month=5
start_date = f"{month+1}-1-2011"
display(start_date)
```

    '6-1-2011'

```python
def get_forecast_days(firstdate) :
    days = [i*7 for i in [-1,0,1,2,3,4]]
    deltadays = pd.to_timedelta(pd.Series(days), unit='D') 

    analysis_days = (pd.to_datetime(firstdate) + deltadays).dt.date
    analysis_days = [str(day) for day in analysis_days]
    analysis_days
    seed_day = analysis_days.pop(0)

    return analysis_days
```

```python
forecast_dates = get_forecast_days(start_date)
display(forecast_dates)
```

    ['2011-06-01', '2011-06-08', '2011-06-15', '2011-06-22', '2011-06-29']

```python
# get our list of items to run through

inference_data = []
days = []

# get the days from the start date to the end date
def get_forecast_dates(forecast_day: str, nforecast=7):
    days = [i for i in range(nforecast)]
    deltadays = pd.to_timedelta(pd.Series(days), unit='D')
    
    last_day = pd.to_datetime(forecast_day)
    dates = last_day + deltadays
    datestr = dates.dt.date.astype(str)
    return datestr 

# used to generate our queries
def mk_dt_range_query(*, tablename: str, forecast_day: str) -> str:
    assert isinstance(tablename, str)
    assert isinstance(forecast_day, str)
    query = f"""
            select cnt from {tablename} where 
            dteday >= DATE_SUB(DATE('{forecast_day}'), INTERVAL 1 month) 
            AND dteday < DATE('{forecast_day}') 
            ORDER BY dteday
            """
    return query

for day in forecast_dates:
    print(f"Current date: {day}")
    day_range=get_forecast_dates(day)
    days.append({"date": day_range})
    query = mk_dt_range_query(tablename=f"{forecast_connection.details()['dataset']}.{forecast_connection.details()['input_table']}", forecast_day=day)
    print(query)
    data = bigquery_statsmodel_client.query(query).to_dataframe().apply({"cnt":int}).to_dict(orient='list')
    # add the date into the list
    inference_data.append(data)
```

    Current date: 2011-06-01
    
                select cnt from release_testing_2023_2.bike_rentals where 
                dteday >= DATE_SUB(DATE('2011-06-01'), INTERVAL 1 month) 
                AND dteday < DATE('2011-06-01') 
                ORDER BY dteday
                
    Current date: 2011-06-08
    
                select cnt from release_testing_2023_2.bike_rentals where 
                dteday >= DATE_SUB(DATE('2011-06-08'), INTERVAL 1 month) 
                AND dteday < DATE('2011-06-08') 
                ORDER BY dteday
                
    Current date: 2011-06-15
    
                select cnt from release_testing_2023_2.bike_rentals where 
                dteday >= DATE_SUB(DATE('2011-06-15'), INTERVAL 1 month) 
                AND dteday < DATE('2011-06-15') 
                ORDER BY dteday
                
    Current date: 2011-06-22
    
                select cnt from release_testing_2023_2.bike_rentals where 
                dteday >= DATE_SUB(DATE('2011-06-22'), INTERVAL 1 month) 
                AND dteday < DATE('2011-06-22') 
                ORDER BY dteday
                
    Current date: 2011-06-29
    
                select cnt from release_testing_2023_2.bike_rentals where 
                dteday >= DATE_SUB(DATE('2011-06-29'), INTERVAL 1 month) 
                AND dteday < DATE('2011-06-29') 
                ORDER BY dteday
                

```python
parallel_results = await pipeline.parallel_infer(tensor_list=inference_data, timeout=20, num_parallel=16, retries=2)
```

```python
display(parallel_results)
```

    [[{'forecast': [4373, 4385, 4379, 4382, 4380, 4381, 4380]}],
     [{'forecast': [4666, 4582, 4560, 4555, 4553, 4553, 4552]}],
     [{'forecast': [4683, 4634, 4625, 4623, 4622, 4622, 4622]}],
     [{'forecast': [4732, 4637, 4648, 4646, 4647, 4647, 4647]}],
     [{'forecast': [4692, 4698, 4699, 4699, 4699, 4699, 4699]}]]

```python
days_results = list(zip(days, parallel_results))
```

```python
# merge our parallel results into the predicted date sales

# results_table = pd.DataFrame(list(zip(days, parallel_results)),
#                             columns=["date", "forecast"])
results_table = pd.DataFrame(columns=["date", "forecast"])

# display(days_results)
for date in days_results:
    # display(date)
    new_days = date[0]['date'].tolist()
    new_forecast = date[1][0]['forecast']
    new_results = list(zip(new_days, new_forecast))
    results_table = results_table.append(pd.DataFrame(list(zip(new_days, new_forecast)), columns=['date','forecast']))
```

Based on all of the predictions, here are the results for the next month.

```python
results_table
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
      <td>2011-06-01</td>
      <td>4373</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-06-02</td>
      <td>4385</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-06-03</td>
      <td>4379</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-06-04</td>
      <td>4382</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-06-05</td>
      <td>4380</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2011-06-06</td>
      <td>4381</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2011-06-07</td>
      <td>4380</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2011-06-08</td>
      <td>4666</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-06-09</td>
      <td>4582</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-06-10</td>
      <td>4560</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-06-11</td>
      <td>4555</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-06-12</td>
      <td>4553</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2011-06-13</td>
      <td>4553</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2011-06-14</td>
      <td>4552</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2011-06-15</td>
      <td>4683</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-06-16</td>
      <td>4634</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-06-17</td>
      <td>4625</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-06-18</td>
      <td>4623</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-06-19</td>
      <td>4622</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2011-06-20</td>
      <td>4622</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2011-06-21</td>
      <td>4622</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2011-06-22</td>
      <td>4732</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-06-23</td>
      <td>4637</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-06-24</td>
      <td>4648</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-06-25</td>
      <td>4646</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-06-26</td>
      <td>4647</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2011-06-27</td>
      <td>4647</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2011-06-28</td>
      <td>4647</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2011-06-29</td>
      <td>4692</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-06-30</td>
      <td>4698</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-07-01</td>
      <td>4699</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-07-02</td>
      <td>4699</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-07-03</td>
      <td>4699</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2011-07-04</td>
      <td>4699</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2011-07-05</td>
      <td>4699</td>
    </tr>
  </tbody>
</table>

### Upload into DataBase

With our results, we'll upload the results into the table listed in our connection as the `results_table`.  To save time, we'll just upload the dataframe directly with the Google Query `insert_rows_from_dataframe` method.

```python
output_table = bigquery_statsmodel_client.get_table(f"{forecast_connection.details()['dataset']}.{forecast_connection.details()['results_table']}")

bigquery_statsmodel_client.insert_rows_from_dataframe(
    output_table, 
    dataframe=results_table
)
```

    [[]]

We'll grab the last 5 results from our results table to verify the data was inserted.

```python
# Get the last insert to the output table to verify
# wait 10 seconds for the insert to finish
time.sleep(10)
task_inference_results = bigquery_statsmodel_client.query(
        f"""
        SELECT *
        FROM {forecast_connection.details()['dataset']}.{forecast_connection.details()['results_table']}
        ORDER BY date DESC
        LIMIT 5
        """
    ).to_dataframe()

display(task_inference_results)
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
      <td>2011-07-05</td>
      <td>4699</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-07-05</td>
      <td>4699</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-07-04</td>
      <td>4699</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-07-04</td>
      <td>4699</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-07-03</td>
      <td>4699</td>
    </tr>
  </tbody>
</table>

### Undeploy the Pipeline

Undeploy the pipeline and return the resources back to the Wallaroo instance.

```python
pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s ..................................... ok

<table><tr><th>name</th> <td>bikedaypipe</td></tr><tr><th>created</th> <td>2023-06-28 20:11:58.734248+00:00</td></tr><tr><th>last_updated</th> <td>2023-06-29 21:12:00.676013+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>f5051ddf-1111-49e6-b914-f8d24f1f6a8a, 93b113a2-f31a-4e05-883e-66a3d1fa10fb, 7d687c43-a833-4585-b607-7085eff16e9d, 504bb140-d9e2-4964-8f82-27b1d234f7f2, db1a14ad-c40c-41ac-82db-0cdd372172f3, 01d60d1c-7834-4d1f-b9a8-8ad569e114b6, a165cbbb-84d9-42e7-99ec-aa8e244aeb55, 0fefef8b-105e-4a6e-9193-d2e6d61248a1</td></tr><tr><th>steps</th> <td>bikedaymodel</td></tr></table>

