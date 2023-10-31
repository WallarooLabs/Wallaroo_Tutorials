This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-features/pipeline_multiple_replicas_forecast_tutorial).

## Statsmodel Forecast with Wallaroo Features: ML Workload Orchestration

This tutorial series demonstrates how to use Wallaroo to create a Statsmodel forecasting model based on bike rentals.  This tutorial series is broken down into the following:

* Create and Train the Model:  This first notebook shows how the model is trained from existing data.
* Deploy and Sample Inference:  With the model developed, we will deploy it into Wallaroo and perform a sample inference.
* Parallel Infer:  A sample of multiple weeks of data will be retrieved and submitted as an asynchronous parallel inference.  The results will be collected and uploaded to a sample database.
* External Connection:  A sample data connection to Google BigQuery to retrieve input data and store the results in a table.
* ML Workload Orchestration:  Take all of the previous steps and automate the request into a single Wallaroo ML Workload Orchestration.

This step will expand upon using the Connection and create a ML Workload Orchestration that automates requesting the inference data, submitting it in parallel, and storing the results into a database table.

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

## Orchestrations, Taks, and Tasks Runs

We've details how Wallaroo Connections work.  Now we'll use Orchestrations, Tasks, and Task Runs.

| Item | Description |
|---|---|
| Orchestration | ML Workload orchestration allows data scientists and ML Engineers to automate and scale production ML workflows in Wallaroo to ensure a tight feedback loop and continuous tuning of models from training to production. Wallaroo platform users (data scientists or ML Engineers) have the ability to deploy, automate and scale recurring batch production ML workloads that can ingest data from predefined data sources to run inferences in Wallaroo, chain pipelines, and send inference results to predefined destinations to analyze model insights and assess business outcomes. |
| Task | An implementation of an Orchestration.  Tasks can be either `Run Once`:  They run once and upon completion, stop. `Run Scheduled`: The task runs whenever a specific `cron` like schedule is reached.  Scheduled tasks will run until the `kill` command is issued. |
| Task Run | The execusion of a task.  For `Run Once` tasks, there will be only one `Run Task`.  A `Run Scheduled` tasks will have multiple tasks, one for every time the schedule parameter is met.  Task Runs have their own log files that can be examined to track progress and results. |

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

    '2023.3.0+785595cda'

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
connection_name = f'statsmodel-bike-rentals-{suffix}'
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

### Deploy Pipeline

The pipeline is already set witht the model.  For our demo we'll verify that it's deployed.

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

    Waiting for deployment - this will take up to 45s .................... ok

<table><tr><th>name</th> <td>bikedaypipe</td></tr><tr><th>created</th> <td>2023-06-30 15:42:56.781150+00:00</td></tr><tr><th>last_updated</th> <td>2023-06-30 15:45:23.267621+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>6552b04e-d074-4773-982b-a2885ce6f9bf, b884c20c-c491-46ec-b438-74384a963acc, 4e8d2a88-1a41-482c-831d-f057a48e18c1</td></tr><tr><th>steps</th> <td>bikedaymodel</td></tr></table>

### BigQuery Sample Orchestration

The orchestration that will automate this process is `./resources/forecast-bigquer-orchestration.zip`.  The files used are stored in the directory `forecast-bigquery-orchestration`, created with the command:

`zip -r forecast-bigquery-connection.zip main.py requirements.txt`.

This contains the following:

* `requirements.txt`:  The Python requirements file to specify the following libraries used:

```python
google-cloud-bigquery==3.10.0
google-auth==2.17.3
db-dtypes==1.1.1
```

* `main.py`: The entry file that takes the previous statsmodel BigQuery connection and statsmodel Forecast model and uses it to predict the next month's sales based on the previous month's performance.  The details are listed below.  Since we are using the async `parallel_infer`, we'll use the `asyncio` library to run our sample `main` method.

```python
import json
import os
import datetime
import asyncio

import wallaroo
from wallaroo.object import EntityNotFoundError
from wallaroo.framework import Framework

import pandas as pd
import numpy as np

pd.set_option('display.max_colwidth', None)

# for Big Query connections
from google.cloud import bigquery
from google.oauth2 import service_account
import db_dtypes

import time

async def main():
    
    wl = wallaroo.Client()

    # get the arguments
    arguments = wl.task_args()

    if "workspace_name" in arguments:
        workspace_name = arguments['workspace_name']
    else:
        workspace_name="multiple-replica-forecast-tutorial"

    if "pipeline_name" in arguments:
        pipeline_name = arguments['pipeline_name']
    else:
        pipeline_name="bikedaypipe"

    if "bigquery_connection_input_name" in arguments:
        bigquery_connection_name = arguments['bigquery_connection_input_name']
    else:
        bigquery_connection_name = "statsmodel-bike-rentals"

    print(bigquery_connection_name)
    def get_workspace(name):
        workspace = None
        for ws in wl.list_workspaces():
            if ws.name() == name:
                workspace= ws
        return workspace

    def get_pipeline(name):
        try:
            pipeline = wl.pipelines_by_name(name)[0]
        except EntityNotFoundError:
            print(f"Pipeline not found:{name}")
        return pipeline

    print(f"BigQuery Connection: {bigquery_connection_name}")
    forecast_connection = wl.get_connection(bigquery_connection_name)

    print(f"Workspace: {workspace_name}")
    workspace = get_workspace(workspace_name)

    wl.set_current_workspace(workspace)
    print(workspace)

    # the pipeline is assumed to be deployed
    print(f"Pipeline: {pipeline_name}")
    pipeline = get_pipeline(pipeline_name)
    print(pipeline)

    print("Getting date and input query.")

    bigquery_statsmodel_credentials = service_account.Credentials.from_service_account_info(
        forecast_connection.details())

    bigquery_statsmodel_client = bigquery.Client(
        credentials=bigquery_statsmodel_credentials, 
        project=forecast_connection.details()['project_id']
    )

    print("Get the current month and retrieve next month's forecasts")
    month = datetime.datetime.now().month
    start_date = f"{month+1}-1-2011"
    print(f"Start date: {start_date}")

    def get_forecast_days(firstdate) :
        days = [i*7 for i in [-1,0,1,2,3,4]]
        deltadays = pd.to_timedelta(pd.Series(days), unit='D') 

        analysis_days = (pd.to_datetime(firstdate) + deltadays).dt.date
        analysis_days = [str(day) for day in analysis_days]
        analysis_days
        seed_day = analysis_days.pop(0)

        return analysis_days

    forecast_dates = get_forecast_days(start_date)
    print(f"Forecast dates: {forecast_dates}")

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

    print(inference_data)

    parallel_results = await pipeline.parallel_infer(tensor_list=inference_data, timeout=20, num_parallel=16, retries=2)

    days_results = list(zip(days, parallel_results))
    print(days_results)

    # merge our parallel results into the predicted date sales
    results_table = pd.DataFrame(columns=["date", "forecast"])

    # match the dates to predictions
    # display(days_results)
    for date in days_results:
        # display(date)
        new_days = date[0]['date'].tolist()
        new_forecast = date[1][0]['forecast']
        new_results = list(zip(new_days, new_forecast))
        results_table = results_table.append(pd.DataFrame(list(zip(new_days, new_forecast)), columns=['date','forecast']))

    print("Uploading results to results table.")
    output_table = bigquery_statsmodel_client.get_table(f"{forecast_connection.details()['dataset']}.{forecast_connection.details()['results_table']}")

    bigquery_statsmodel_client.insert_rows_from_dataframe(
        output_table, 
        dataframe=results_table
    )
    
asyncio.run(main())
```

This orchestration allows a user to specify the workspace, pipeline, and data connection.  As long as they all match the previous conditions, then the orchestration will run successfully.

### Upload the Orchestration

Orchestrations are uploaded with the Wallaroo client `upload_orchestration(path)` method with the following parameters.

| Parameter | Type | Description |
| --- | --- | ---|
| **path** | string (Required) | The path to the ZIP file to be uploaded. |

Once uploaded, the deployment will be prepared and any requirements will be downloaded and installed.

For this example, the orchestration `./bigquery_remote_inference/bigquery_remote_inference.zip` will be uploaded and saved to the variable `orchestration`.  Then we will loop until the uploaded orchestration's `status` displays `ready`.

```python
orchestration = wl.upload_orchestration(name="statsmodel-orchestration", path="./resources/forecast-bigquery-orchestration.zip")

while orchestration.status() != 'ready':
    print(orchestration.status())
    time.sleep(5)
```

    pending_packaging
    pending_packaging
    packaging
    packaging
    packaging
    packaging
    packaging
    packaging
    packaging
    packaging
    packaging

```python
wl.list_orchestrations()
```

<table><tr><th>id</th><th>name</th><th>status</th><th>filename</th><th>sha</th><th>created at</th><th>updated at</th></tr><tr><td>8211497d-292a-4145-b28b-f6364e12544e</td><td>statsmodel-orchestration</td><td>packaging</td><td>forecast-bigquery-orchestration.zip</td><td>44f591...1fa8d6</td><td>2023-30-Jun 15:45:48</td><td>2023-30-Jun 15:45:58</td></tr><tr><td>f8f31494-41c4-4336-bfd6-5b3b1607dedc</td><td>statsmodel-orchestration</td><td>ready</td><td>forecast-bigquery-orchestration.zip</td><td>27ad14...306ad1</td><td>2023-30-Jun 15:51:08</td><td>2023-30-Jun 15:51:57</td></tr><tr><td>fd776f89-ea63-45e9-b8d6-a749074fd579</td><td>statsmodel-orchestration</td><td>ready</td><td>forecast-bigquery-orchestration.zip</td><td>bd6a0e...3a6a09</td><td>2023-30-Jun 16:45:50</td><td>2023-30-Jun 16:46:39</td></tr><tr><td>8200995b-3e33-49f4-ac4f-98ea2b1330db</td><td>statsmodel-orchestration</td><td>ready</td><td>forecast-bigquery-orchestration.zip</td><td>8d0c2f...a3c89f</td><td>2023-30-Jun 15:54:14</td><td>2023-30-Jun 15:55:07</td></tr><tr><td>5449a104-abc5-423d-a973-31a3cfdf8b55</td><td>statsmodel-orchestration</td><td>ready</td><td>forecast-bigquery-orchestration.zip</td><td>e00646...45d2a7</td><td>2023-30-Jun 16:12:39</td><td>2023-30-Jun 16:13:29</td></tr><tr><td>9fd1e58c-942d-495b-b3bd-d51f5c03b5ed</td><td>statsmodel-orchestration</td><td>ready</td><td>forecast-bigquery-orchestration.zip</td><td>bd6a0e...3a6a09</td><td>2023-30-Jun 16:48:53</td><td>2023-30-Jun 16:49:44</td></tr><tr><td>73f2e90a-13ab-4182-bde1-0fe55c4446cf</td><td>statsmodel-orchestration</td><td>ready</td><td>forecast-bigquery-orchestration.zip</td><td>f78c26...f494d9</td><td>2023-30-Jun 16:27:37</td><td>2023-30-Jun 16:28:31</td></tr><tr><td>64b085c7-5317-4152-81c3-c0c77b4f683b</td><td>statsmodel-orchestration</td><td>ready</td><td>forecast-bigquery-orchestration.zip</td><td>37257f...4b4547</td><td>2023-30-Jun 16:39:49</td><td>2023-30-Jun 16:40:38</td></tr><tr><td>4a3a73ab-014c-4aa4-9896-44c313d80daa</td><td>statsmodel-orchestration</td><td>ready</td><td>forecast-bigquery-orchestration.zip</td><td>23bf29...17b780</td><td>2023-30-Jun 16:52:45</td><td>2023-30-Jun 16:53:38</td></tr><tr><td>b4ef4449-9afe-4fba-aaa0-b7fd49687443</td><td>statsmodel-orchestration</td><td>ready</td><td>forecast-bigquery-orchestration.zip</td><td>d4f02b...0e6c5d</td><td>2023-30-Jun 16:42:29</td><td>2023-30-Jun 16:43:26</td></tr></table>

### Create the Task

The orchestration is now ready to be implemented as a Wallaroo Task.  We'll just run it once as an example.  This specific Orchestration that creates the Task assumes that the pipeline is deployed, and accepts the arguments:

* workspace_name
* pipeline_name
* bigquery_connection_name

We'll supply the workspaces, pipeline and connection created in previous steps and stored in the initial variables above.  Verify these exist and match the existing workspace, pipeline and connection used in the previous notebooks in this series.

Tasks are generated and run once with the Orchestration `run_once(name, json_args, timeout)` method.  Any arguments for the orchestration are passed in as a `Dict`.  If there are no arguments, then an empty set `{}` is passed.

```python
task = orchestration.run_once(name="statsmodel single run", json_args={"workspace_name":workspace_name, "pipeline_name": pipeline_name, "bigquery_connection_input_name":connection_name})
```

### Monitor Run with Task Status

We'll monitor the run first with it's status.

For this example, the status of the previously created task will be generated, then looped until it has reached status `started`.

```python
while task.status() != "started":
    display(task.status())
    time.sleep(5)
```

    'pending'

    'pending'

```python
display(connection_name)
```

    'statsmodel-bike-rentals-jch'

### List Tasks

We'll use the Wallaroo client `list_tasks` method to view the tasks currently running.

```python
wl.list_tasks()
```

<table><tr><th>id</th><th>name</th><th>last run status</th><th>type</th><th>active</th><th>schedule</th><th>created at</th><th>updated at</th></tr><tr><td>c7279e5e-e162-42f8-90ce-b7c0c0bb30f8</td><td>statsmodel single run</td><td>running</td><td>Temporary Run</td><td>True</td><td>-</td><td>2023-30-Jun 16:53:41</td><td>2023-30-Jun 16:53:47</td></tr><tr><td>a47dbca0-e568-44d3-9715-1fed0f17b9a7</td><td>statsmodel single run</td><td>failure</td><td>Temporary Run</td><td>True</td><td>-</td><td>2023-30-Jun 16:49:44</td><td>2023-30-Jun 16:49:54</td></tr><tr><td>15c80ad0-537f-4e6a-84c6-6c2f35b5f441</td><td>statsmodel single run</td><td>failure</td><td>Temporary Run</td><td>True</td><td>-</td><td>2023-30-Jun 16:46:41</td><td>2023-30-Jun 16:46:51</td></tr><tr><td>d0935da6-480a-420d-a70c-570160b0b6b3</td><td>statsmodel single run</td><td>failure</td><td>Temporary Run</td><td>True</td><td>-</td><td>2023-30-Jun 16:44:50</td><td>2023-30-Jun 16:44:56</td></tr><tr><td>e510e8c5-048b-43b1-9524-974934a9e4f5</td><td>statsmodel single run</td><td>failure</td><td>Temporary Run</td><td>True</td><td>-</td><td>2023-30-Jun 16:43:30</td><td>2023-30-Jun 16:43:35</td></tr><tr><td>0f62befb-c788-4779-bcfb-0595e3ca6f24</td><td>statsmodel single run</td><td>failure</td><td>Temporary Run</td><td>True</td><td>-</td><td>2023-30-Jun 16:40:39</td><td>2023-30-Jun 16:40:50</td></tr><tr><td>f00c6a97-32f9-4124-bf86-34a0068c1314</td><td>statsmodel single run</td><td>failure</td><td>Temporary Run</td><td>True</td><td>-</td><td>2023-30-Jun 16:28:32</td><td>2023-30-Jun 16:28:38</td></tr><tr><td>10c8af33-8ff4-4aae-b08d-89665bcb0481</td><td>statsmodel single run</td><td>failure</td><td>Temporary Run</td><td>True</td><td>-</td><td>2023-30-Jun 16:13:30</td><td>2023-30-Jun 16:13:35</td></tr><tr><td>9ae4e6e6-3849-4039-acfe-6810699edef8</td><td>statsmodel single run</td><td>failure</td><td>Temporary Run</td><td>True</td><td>-</td><td>2023-30-Jun 16:00:05</td><td>2023-30-Jun 16:00:15</td></tr></table>

### Display Task Run Results

The Task Run is the implementation of the task - the actual running of the script and it's results.  Tasks that are Run Once will only have one Task Run, while a Task set to Run Scheduled will have a Task Run for each time the task is executed.  Each Task Run has its own set of logs and results that are monitoried through the Task Run `logs()` method.

We'll wait 30 seconds, then retrieve the task run for our generated task, then start checking the logs for our task run.  It may take longer than 30 seconds to launch the task, so be prepared to run the `.logs()` method again to view the logs.

```python
#wait 30 seconds for the task to finish
time.sleep(30)
statsmodel_task_run = task.last_runs()[0]
```

```python
statsmodel_task_run.logs()
```

<pre><code>2023-30-Jun 16:53:57 statsmodel-bike-rentals-jch
2023-30-Jun 16:53:57 BigQuery Connection: statsmodel-bike-rentals-jch
2023-30-Jun 16:53:57 Workspace: multiple-replica-forecast-tutorial-jch
2023-30-Jun 16:53:57 {'name': 'multiple-replica-forecast-tutorial-jch', 'id': 7, 'archived': False, 'created_by': '34b86cac-021e-4cf0-aa30-40da7db5a77f', 'created_at': '2023-06-30T15:42:56.551195+00:00', 'models': [{'name': 'bikedaymodel', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 6, 30, 15, 42, 56, 979723, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 6, 30, 15, 42, 56, 979723, tzinfo=tzutc())}], 'pipelines': [{'name': 'bikedaypipe', 'create_time': datetime.datetime(2023, 6, 30, 15, 42, 56, 781150, tzinfo=tzutc()), 'definition': '[]'}]}
2023-30-Jun 16:53:57 Pipeline: bikedaypipe
2023-30-Jun 16:53:57 {'name': 'bikedaypipe', 'create_time': datetime.datetime(2023, 6, 30, 15, 42, 56, 781150, tzinfo=tzutc()), 'definition': '[]'}
2023-30-Jun 16:53:57 Getting date and input query.
2023-30-Jun 16:53:57 Get the current month and retrieve next month's forecasts
2023-30-Jun 16:53:57 Start date: 7-1-2011
2023-30-Jun 16:53:57 Forecast dates: ['2011-07-01', '2011-07-08', '2011-07-15', '2011-07-22', '2011-07-29']
2023-30-Jun 16:53:57 Current date: 2011-07-01
2023-30-Jun 16:53:57 
2023-30-Jun 16:53:57                 select cnt from release_testing_2023_2.bike_rentals where 
2023-30-Jun 16:53:57                 dteday >= DATE_SUB(DATE('2011-07-01'), INTERVAL 1 month) 
2023-30-Jun 16:53:57                 AND dteday < DATE('2011-07-01') 
2023-30-Jun 16:53:57                 ORDER BY dteday
2023-30-Jun 16:53:57 Current date: 2011-07-08
2023-30-Jun 16:53:57                 
2023-30-Jun 16:53:57 
2023-30-Jun 16:53:57                 select cnt from release_testing_2023_2.bike_rentals where 
2023-30-Jun 16:53:57                 dteday >= DATE_SUB(DATE('2011-07-08'), INTERVAL 1 month) 
2023-30-Jun 16:53:57                 ORDER BY dteday
2023-30-Jun 16:53:57                 AND dteday < DATE('2011-07-08') 
2023-30-Jun 16:53:57                 
2023-30-Jun 16:53:57 Current date: 2011-07-15
2023-30-Jun 16:53:57 
2023-30-Jun 16:53:57                 select cnt from release_testing_2023_2.bike_rentals where 
2023-30-Jun 16:53:57                 dteday >= DATE_SUB(DATE('2011-07-15'), INTERVAL 1 month) 
2023-30-Jun 16:53:57                 ORDER BY dteday
2023-30-Jun 16:53:57                 AND dteday < DATE('2011-07-15') 
2023-30-Jun 16:53:57                 
2023-30-Jun 16:53:57 Current date: 2011-07-22
2023-30-Jun 16:53:57 
2023-30-Jun 16:53:57                 select cnt from release_testing_2023_2.bike_rentals where 
2023-30-Jun 16:53:57                 dteday >= DATE_SUB(DATE('2011-07-22'), INTERVAL 1 month) 
2023-30-Jun 16:53:57                 AND dteday < DATE('2011-07-22') 
2023-30-Jun 16:53:57                 ORDER BY dteday
2023-30-Jun 16:53:57                 
2023-30-Jun 16:53:57 Current date: 2011-07-29
2023-30-Jun 16:53:57                 select cnt from release_testing_2023_2.bike_rentals where 
2023-30-Jun 16:53:57 
2023-30-Jun 16:53:57                 dteday >= DATE_SUB(DATE('2011-07-29'), INTERVAL 1 month) 
2023-30-Jun 16:53:57                 ORDER BY dteday
2023-30-Jun 16:53:57                 AND dteday < DATE('2011-07-29') 
2023-30-Jun 16:53:57                 
2023-30-Jun 16:53:57 [({'date': 0    2011-07-01
2023-30-Jun 16:53:57 [{'cnt': [3974, 4968, 5312, 5342, 4906, 4548, 4833, 4401, 3915, 4586, 4966, 4460, 5020, 4891, 5180, 3767, 4844, 5119, 4744, 4010, 4835, 4507, 4790, 4991, 5202, 5305, 4708, 4648, 5225, 5515]}, {'cnt': [4401, 3915, 4586, 4966, 4460, 5020, 4891, 5180, 3767, 4844, 5119, 4744, 4010, 4835, 4507, 4790, 4991, 5202, 5305, 4708, 4648, 5225, 5515, 5362, 5119, 4649, 6043, 4665, 4629, 4592]}, {'cnt': [5180, 3767, 4844, 5119, 4744, 4010, 4835, 4507, 4790, 4991, 5202, 5305, 4708, 4648, 5225, 5515, 5362, 5119, 4649, 6043, 4665, 4629, 4592, 4040, 5336, 4881, 4086, 4258, 4342, 5084]}, {'cnt': [4507, 4790, 4991, 5202, 5305, 4708, 4648, 5225, 5515, 5362, 5119, 4649, 6043, 4665, 4629, 4592, 4040, 5336, 4881, 4086, 4258, 4342, 5084, 5538, 5923, 5302, 4458, 4541, 4332, 3784]}, {'cnt': [5225, 5515, 5362, 5119, 4649, 6043, 4665, 4629, 4592, 4040, 5336, 4881, 4086, 4258, 4342, 5084, 5538, 5923, 5302, 4458, 4541, 4332, 3784, 3387, 3285, 3606, 3840, 4590, 4656, 4390]}]
2023-30-Jun 16:53:57 1    2011-07-02
2023-30-Jun 16:53:57 2    2011-07-03
2023-30-Jun 16:53:57 3    2011-07-04
2023-30-Jun 16:53:57 4    2011-07-05
2023-30-Jun 16:53:57 5    2011-07-06
2023-30-Jun 16:53:57 6    2011-07-07
2023-30-Jun 16:53:57 dtype: object}, [{'forecast': [4894, 4767, 4786, 4783, 4783, 4783, 4783]}]), ({'date': 0    2011-07-08
2023-30-Jun 16:53:57 2    2011-07-10
2023-30-Jun 16:53:57 1    2011-07-09
2023-30-Jun 16:53:57 4    2011-07-12
2023-30-Jun 16:53:57 3    2011-07-11
2023-30-Jun 16:53:57 5    2011-07-13
2023-30-Jun 16:53:57 6    2011-07-14
2023-30-Jun 16:53:57 dtype: object}, [{'forecast': [4842, 4839, 4836, 4833, 4831, 4830, 4828]}]), ({'date': 0    2011-07-15
2023-30-Jun 16:53:57 1    2011-07-16
2023-30-Jun 16:53:57 2    2011-07-17
2023-30-Jun 16:53:57 3    2011-07-18
2023-30-Jun 16:53:57 4    2011-07-19
2023-30-Jun 16:53:57 5    2011-07-20
2023-30-Jun 16:53:57 6    2011-07-21
2023-30-Jun 16:53:57 dtype: object}, [{'forecast': [4895, 4759, 4873, 4777, 4858, 4789, 4848]}]), ({'date': 0    2011-07-22
2023-30-Jun 16:53:57 1    2011-07-23
2023-30-Jun 16:53:57 2    2011-07-24
2023-30-Jun 16:53:57 3    2011-07-25
2023-30-Jun 16:53:57 5    2011-07-27
2023-30-Jun 16:53:57 4    2011-07-26
2023-30-Jun 16:53:57 6    2011-07-28
2023-30-Jun 16:53:57 dtype: object}, [{'forecast': [4559, 4953, 4829, 4868, 4856, 4860, 4858]}]), ({'date': 0    2011-07-29
2023-30-Jun 16:53:57 1    2011-07-30
2023-30-Jun 16:53:57 3    2011-08-01
2023-30-Jun 16:53:57 2    2011-07-31
2023-30-Jun 16:53:57 5    2011-08-03
2023-30-Jun 16:53:57 4    2011-08-02
2023-30-Jun 16:53:57 6    2011-08-04
2023-30-Jun 16:53:57 dtype: object}, [{'forecast': [4490, 4549, 4586, 4610, 4624, 4634, 4640]}])]
2023-30-Jun 16:53:57 Uploading results to results table.</code></pre>

### Undeploy the Pipeline

Undeploy the pipeline and return the resources back to the Wallaroo instance.

```python
pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s ..................................... ok

<table><tr><th>name</th> <td>bikedaypipe</td></tr><tr><th>created</th> <td>2023-06-30 15:42:56.781150+00:00</td></tr><tr><th>last_updated</th> <td>2023-06-30 15:45:23.267621+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>6552b04e-d074-4773-982b-a2885ce6f9bf, b884c20c-c491-46ec-b438-74384a963acc, 4e8d2a88-1a41-482c-831d-f057a48e18c1</td></tr><tr><th>steps</th> <td>bikedaymodel</td></tr></table>

```python

```
