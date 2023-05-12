This can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/quickstartguide_samples/blob/20230314_2023.2_updates/pipeline-orchestrators/orchestration_sdk_simple_tutorial).

## Pipeline Orchestrations Simple Tutorial

This tutorial provides a quick set of methods and examples regarding Wallaroo Connections and Wallaroo ML Workload Orchestration.  For full details, see the Wallaroo Documentation site.

Wallaroo provides Data Connections and ML Workload Orchestrations to provide organizations with a method of creating and managing automated tasks that can either be run on demand or a regular schedule.

## Definitions

* **Orchestration**: A set of instructions written as a python script with a requirements library.  Orchestrations are uploaded to the Wallaroo instance as a .zip file.
* **Task**: An implementation of an orchestration.  Tasks are run either once when requested, on a repeating schedule, or as a service.
* **Connection**: Definitions set by MLOps engineers that are used by other Wallaroo users for connection information to a data source.  Usually paired with orchestrations.

## Tutorial Goals

The tutorial will demonstrate the following:

1. Create a Wallaroo connection to retrieving information from an external source.
1. Upload Wallaroo ML Workload Orchestration.
1. Run the orchestration once as a Run Once Task and verify that the information was saved the pipeline logs.
1. Schedule the orchestration as a Scheduled Task and verify that the information was saved to the pipeline logs.

## Prerequisites

* An installed Wallaroo instance.
* The following Python libraries installed.  These are included by default in a Wallaroo instance's JupyterHub service.
  * `os`
  * [`wallaroo`](https://pypi.org/project/wallaroo/): The Wallaroo SDK. Included with the Wallaroo JupyterHub service by default.
  * [`pandas`](https://pypi.org/project/pandas/): Pandas, mainly used for Pandas DataFrame
  * [`pyarrow`](https://pypi.org/project/pyarrow/): PyArrow for Apache Arrow support

## Initial Steps

For this tutorial, we'll create a workspace, upload our sample model and deploy a pipeline.  We'll perform some quick sample inferences to verify that everything it working.

### Load Libraries

Here we'll import the various libraries we'll use for the tutorial.

```python
import wallaroo
from wallaroo.object import EntityNotFoundError, RequiredAttributeMissing

# to display dataframe tables
from IPython.display import display
# used to display dataframe information without truncating
import pandas as pd
pd.set_option('display.max_colwidth', None)
import pyarrow as pa

import os
# Used for the Wallaroo SDK version 2023.1
os.environ["ARROW_ENABLED"]="True"

import time
```

### Connect to the Wallaroo Instance

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  If logging in externally, update the `wallarooPrefix` and `wallarooSuffix` variables with the proper DNS information.  For more information on Wallaroo DNS settings, see the [Wallaroo DNS Integration Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-dns-guide/).

```python
# Login through local Wallaroo instance

wl = wallaroo.Client()
```

```python
# Setting variables for later steps

workspace_name = 'simpleorchestrationworkspace'
pipeline_name = 'simpleorchestrationpipeline'
model_name = 'simpleorchestrationmodel'
model_file_name = './models/rf_model.onnx'

inference_connection_name = "external_inference_connection"
inference_connection_type = "HTTP"
inference_connection_argument = {'host':'https://github.com/WallarooLabs/Wallaroo_Tutorials/blob/20230314_2023.2_updates/wallaroo-testing-tutorials/houseprice-saga/data/xtest-1k.arrow?raw=true'}
```

### Helper Methods

The following helper methods are used to either create or get workspaces, pipelines, and connections.

```python
# helper methods to retrieve workspaces and pipelines

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

def get_connection(name, connection_type, connection_arguments):
    try:
        connection = wl.get_connection(name)
    except RequiredAttributeMissing:
        connection =wl.create_connection(name, 
                  connection_type, 
                  connection_arguments)
    return connection

```

### Create the Workspace and Pipeline

We'll now create our workspace and pipeline for the tutorial.  If this tutorial has been run previously, then this will retrieve the existing ones with the assumption they're for us with this tutorial.

We'll set the retrieved workspace as the current workspace in the SDK, so all commands will default to that workspace.

```python
workspace = get_workspace(workspace_name)
wl.set_current_workspace(workspace)

pipeline = get_pipeline(pipeline_name)
```

### Upload the Model and Deploy Pipeline

We'll upload our model into our sample workspace, then add it as a pipeline step before deploying the pipeline to it's ready to accept inference requests.

```python
# Upload the model

housing_model_control = wl.upload_model(model_name, model_file_name).configure()

# Add the model as a pipeline step

pipeline.add_model_step(housing_model_control)
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>simpleorchestrationpipeline</td></tr><tr><th>created</th> <td>2023-05-08 22:08:15.458098+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-09 20:07:34.164370+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>afb5d36f-338e-4bb2-a7e4-24d1280c1575, 0daa5b47-0852-4b33-8157-ea07c03a98df, bc38bf67-f916-4bb4-87e2-27836eda031f, 6e91a9ae-13d7-40db-97e8-c62af1b81392, 90f7408e-c85f-42aa-b124-43eb9252cc23, 8e68f5d6-94dd-46e7-8897-2f85a6acbbc6, 8a042c5b-6913-41e5-a3b8-f9bde2f6c493, a80664ed-48d6-460b-8289-ff8e1cc31a9e, 7dfcb260-3c2e-4e62-b296-7e9a2f57bde2, 59003cd3-bd4b-43c8-b457-4d0b9cdbea51, 6eaa00b8-0789-4a51-b182-c9bee83ee4a3, cad9521a-fa40-464a-9ec8-0749d1ffbe73, 42170c80-c78a-43fc-b463-356730d40de8, c1c09de1-8da2-431f-ad58-cf469129b12b, 5300e180-9340-481a-a861-99ca20749c18, 4279ec2c-8f83-4c76-a469-c49e654a7168, 6c792f92-9d7a-42d5-b246-4d793edc2180, ee8be66d-3b4d-491d-aa2f-65fcef339aec</td></tr><tr><th>steps</th> <td>simpleorchestrationmodel</td></tr></table>
{{</table>}}

```python
#deploy the pipeline
pipeline.deploy()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>simpleorchestrationpipeline</td></tr><tr><th>created</th> <td>2023-05-08 22:08:15.458098+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-09 20:24:40.407056+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>187cb5e7-c67e-499c-ad56-c7867727ad6d, afb5d36f-338e-4bb2-a7e4-24d1280c1575, 0daa5b47-0852-4b33-8157-ea07c03a98df, bc38bf67-f916-4bb4-87e2-27836eda031f, 6e91a9ae-13d7-40db-97e8-c62af1b81392, 90f7408e-c85f-42aa-b124-43eb9252cc23, 8e68f5d6-94dd-46e7-8897-2f85a6acbbc6, 8a042c5b-6913-41e5-a3b8-f9bde2f6c493, a80664ed-48d6-460b-8289-ff8e1cc31a9e, 7dfcb260-3c2e-4e62-b296-7e9a2f57bde2, 59003cd3-bd4b-43c8-b457-4d0b9cdbea51, 6eaa00b8-0789-4a51-b182-c9bee83ee4a3, cad9521a-fa40-464a-9ec8-0749d1ffbe73, 42170c80-c78a-43fc-b463-356730d40de8, c1c09de1-8da2-431f-ad58-cf469129b12b, 5300e180-9340-481a-a861-99ca20749c18, 4279ec2c-8f83-4c76-a469-c49e654a7168, 6c792f92-9d7a-42d5-b246-4d793edc2180, ee8be66d-3b4d-491d-aa2f-65fcef339aec</td></tr><tr><th>steps</th> <td>simpleorchestrationmodel</td></tr></table>
{{</table>}}

## Create Connections

We will create the data source connection via the Wallaroo client command `create_connection`.

Connections are created with the Wallaroo client command [`create_connection`](https://staging.docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-ml-workload-orchestration/#create-orchestration) with the following parameters.

| Parameter | Type | Description |
| --- | --- | ---|
| **name** | string (Required) | The name of the connection. This must be unique - **if submitting the name of an existing** connection it will return an error. |
| **type** | string (Required) | The user defined type of connection. |
| **details** | Dict (Required) | User defined configuration details for the data connection.  These can be `{'username':'dataperson', 'password':'datapassword', 'port': 3339}`, or `{'token':'abcde123==', 'host':'example.com', 'port:1234'}`, or other user defined combinations.  |

* **IMPORTANT NOTE**:  Data connections names **must** be unique.  Attempting to create a data connection with the same `name` as an existing data connection will result in an error.

We'll also create a data connection named `inference_results_connection` with our helper function `get_connection` that will either create **or** retrieve a connection if it already exists.  From there we'll create out connections:  

* `houseprice_arrow_table`: An Apache Arrow file stored on GitHub that will be used for our inference input.

```python
get_connection(inference_connection_name, inference_connection_type, inference_connection_argument)
```

{{<table "table table-striped table-bordered" >}}
<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Name</td><td>external_inference_connection</td>
  </tr>
  <tr>
    <td>Connection Type</td><td>HTTP</td>
  </tr>
  <tr>
    <td>Details</td><td>*****</td>
  </tr>
  <tr>
    <td>Created At</td><td>2023-05-09T14:56:28.092172+00:00</td>
  </tr>
</table>
{{</table>}}

### Get Connection by Name

The Wallaroo client method `get_connection(name)` retrieves the connection that matches the `name` parameter.  We'll retrieve our connection and store it as `inference_source_connection`.

```python
inference_source_connection = wl.get_connection(name="external_inference_connection")
display(inference_source_connection)
```

{{<table "table table-striped table-bordered" >}}
<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Name</td><td>external_inference_connection</td>
  </tr>
  <tr>
    <td>Connection Type</td><td>HTTP</td>
  </tr>
  <tr>
    <td>Details</td><td>*****</td>
  </tr>
  <tr>
    <td>Created At</td><td>2023-05-09T14:56:28.092172+00:00</td>
  </tr>
</table>
{{</table>}}

### Add Connection to Workspace

The method Workspace [`add_connection(connection_name)`](https://staging.docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-dataconnectors/#add-data-connection-to-workspace) adds a Data Connection to a workspace, and takes the following parameters.

| Parameter | Type | Description |
| --- | --- | ---|
| **name** | string (Required) | The name of the Data Connection |

We'll add both connections to our sample workspace, then list the connections available to the workspace to confirm.

```python
workspace.add_connection("external_inference_connection")
workspace.list_connections()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th><th>connection type</th><th>details</th><th>created at</th></tr><tr><td>inference_results_connection</td><td>LOCALFILE</td><td>*****</td><td>2023-05-09T14:56:38.869916+00:00</td></tr><tr><td>external_inference_connection</td><td>HTTP</td><td>*****</td><td>2023-05-09T14:56:28.092172+00:00</td></tr></table>
{{</table>}}

## Wallaroo ML Workload Orchestration Example

With the pipeline deployed and our connections set, we will now generate our ML Workload Orchestration.  See the [Wallaroo ML Workload Orchestrations guide](https://staging.docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-ml-workload-orchestration/) for full details.

Orchestrations are uploaded to the Wallaroo instance as a ZIP file with the following requirements:

| Parameter | Type | Description |
|---|---|---|
| **User Code** | (*Required*) Python script as `.py` files | If `main.py` exists, then that will be used as the task entrypoint. Otherwise, the **first** `main.py` found in any subdirectory will be used as the entrypoint. |
| Python Library Requirements | (*Optional*) `requirements.txt` file in the [requirements file format](https://pip.pypa.io/en/stable/reference/requirements-file-format/).  A standard Python requirements.txt for any dependencies to be provided in the task environment. The Wallaroo SDK will already be present and **should not be included in the requirements.txt**. Multiple requirements.txt files are not allowed. |
| Other artifacts | &nbsp; | Other artifacts such as files, data, or code to support the orchestration.

For our example, our orchestration will:

1. Use the `inference_results_connection` to open a HTTP Get connection to the inference data file and use it in an inference request in the deployed pipeline.
1. Submit the inference results to the location specified in the `external_inference_connection`.

This sample script is stored in `remote_inference/main.py` with an empty `requirements.txt` file, and packaged into the orchestration as `./remote_inference/remote_inference.zip`.  We'll display the steps in uploading the orchestration to the Wallaroo instance.

Note that the orchestration assumes the pipeline is already deployed.

### Upload the Orchestration

Orchestrations are uploaded with the Wallaroo client `upload_orchestration(path)` method with the following parameters.

| Parameter | Type | Description |
| --- | --- | ---|
| **path** | string (Required) | The path to the ZIP file to be uploaded. |

Once uploaded, the deployment will be prepared and any requirements will be downloaded and installed.

For this example, the orchestration `./remote_inference/remote_inference.zip` will be uploaded and saved to the variable `orchestration`.

```python
orchestration = wl.upload_orchestration(path="./remote_inference/remote_inference.zip")
```

### Orchestration Status

We will loop until the uploaded orchestration's `status` displays `ready`.

```python
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

```python
orchestrations = wl.list_orchestrations()
```

## Task Management Tutorial

Once an Orchestration has the status `ready`, it can be run as a task.  Tasks have three run options.

| Type | SDK Call |  How triggered |
|---|---|:---|
| Once       | `orchestration.run_once(json_args)` | User makes one api call. Task runs once and exits.| Single batch, experimentation. |
| Scheduled  | `orchestration.run_scheduled(name, schedule, timeout, json_args)` | User provides schedule. Task runs exits whenever schedule dictates. | Recurrent batch. |

### Run Task Once

We'll do both a Run Once task and generate our Run Once Task from our orchestration.

Tasks are generated and run once with the Orchestration `run_once(arguments)` method.  Any arguments for the orchestration are passed in as a `Dict`.  If there are no arguments, then an empty set `{}` is passed.

```python
# Example: run once

import datetime
task_start = datetime.datetime.now()

task = orchestration.run_once({})
task
```

{{<table "table table-striped table-bordered" >}}
<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>ID</td><td>989f6579-a617-4600-b326-ae3c4483561f</td>
  </tr>
  <tr>
    <td>Status</td><td>pending</td>
  </tr>
  <tr>
    <td>Type</td><td>Temporary Run</td>
  </tr>
  <tr>
    <td>Created At</td><td>2023-09-May 20:26:00</td>
  </tr>
  <tr>
    <td>Updated At</td><td>2023-09-May 20:26:00</td>
  </tr>
</table>
{{</table>}}

### Task Status

The list of tasks in the Wallaroo instance is retrieves through the Wallaroo Client `list_tasks()` method.  This returns an array list of the following.

| Parameter | Type | Description |
| --- | --- | ---|
| **id** | string | The UUID identifier for the task. |
| **status** | enum | The status of the task.  Values are: <br><ul><li>`pending`: The task has not been started.</li><li>`started`: The task has been scheduled to execute.</li><li>`pending_kill`: The task kill command has been issued and the task is scheduled to be stopped.</li></ul> |
| **type** | string | The type of the task.  Values are: <br><ul><li>`Temporary Run`: The task runs once then stop.</li><li>`Scheduled Run`: The task repeats on a `cron` like schedule.</li></ul> |
| **created at** | DateTime | The date and time the task was started. |
| **updated at** | DateTime | The date and time the task was updated. |

For this example, the status of the previously created task will be generated, then looped until it has reached status `started`.

```python
while task.status() != "started":
    display(task.status())
    time.sleep(5)
```

    'pending'

### Task Results

We can view the inferences from our logs and verify that new entries were added from our task.  In our case, we'll assume the task once started takes about 1 minute to run (deploy the pipeline, run the inference, undeploy the pipeline).  We'll add in a wait of 1 minute, then display the logs during the time period the task was running.

```python
time.sleep(60)

task_end = datetime.datetime.now()
display(task_end)

pipeline.logs(start_datetime = task_start, end_datetime = task_end)
```

    datetime.datetime(2023, 5, 9, 14, 27, 7, 784253)

    Warning: Pipeline log size limit exceeded. Please request logs using export_logs

{{<table "table table-striped table-bordered" >}}
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.variable</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-05-09 20:26:09.822</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.75]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-05-09 20:26:09.822</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[615094.56]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-05-09 20:26:09.822</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
      <td>[448627.72]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-05-09 20:26:09.822</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[758714.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-05-09 20:26:09.822</td>
      <td>[3.0, 1.75, 2200.0, 11520.0, 1.0, 0.0, 0.0, 4.0, 7.0, 2200.0, 0.0, 47.7659, -122.341, 1690.0, 8038.0, 62.0, 0.0, 0.0]</td>
      <td>[513264.7]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>496</th>
      <td>2023-05-09 20:26:09.822</td>
      <td>[4.0, 2.5, 2510.0, 47044.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2510.0, 0.0, 47.7699, -122.085, 2600.0, 42612.0, 27.0, 0.0, 0.0]</td>
      <td>[721143.6]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>497</th>
      <td>2023-05-09 20:26:09.822</td>
      <td>[4.0, 2.5, 4090.0, 11225.0, 2.0, 0.0, 0.0, 3.0, 10.0, 4090.0, 0.0, 47.581, -121.971, 3510.0, 8762.0, 9.0, 0.0, 0.0]</td>
      <td>[1048372.4]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>498</th>
      <td>2023-05-09 20:26:09.822</td>
      <td>[2.0, 1.0, 720.0, 5000.0, 1.0, 0.0, 0.0, 5.0, 6.0, 720.0, 0.0, 47.5195, -122.374, 810.0, 5000.0, 63.0, 0.0, 0.0]</td>
      <td>[244566.38]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>499</th>
      <td>2023-05-09 20:26:09.822</td>
      <td>[4.0, 2.75, 2930.0, 22000.0, 1.0, 0.0, 3.0, 4.0, 9.0, 1580.0, 1350.0, 47.3227, -122.384, 2930.0, 9758.0, 36.0, 0.0, 0.0]</td>
      <td>[518869.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>500</th>
      <td>2023-05-09 20:26:09.822</td>
      <td>[2.0, 1.0, 850.0, 5000.0, 1.0, 0.0, 0.0, 3.0, 6.0, 850.0, 0.0, 47.3817, -122.314, 1160.0, 5000.0, 39.0, 0.0, 0.0]</td>
      <td>[236238.66]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}
<p>501 rows × 4 columns</p>

## Scheduled Run Task Example

The other method of using tasks is as a **scheduled run** through the Orchestration `run_scheduled(name, schedule, timeout, json_args)`.  This sets up a task to run on an regular schedule as defined by the `schedule` parameter in the `cron` service format.  For example:

```python
schedule={'42 * * * *'}
```

Runs on the 42nd minute of every hour.

For our example, we will create a scheduled task to run every 1 minute, display the inference results, then use the Orchestration `kill` task to keep the task from running any further.

```python
scheduled_task_start = datetime.datetime.now()
```

```python

scheduled_task = orchestration.run_scheduled(name="simple_inference_schedule", schedule="*/1 * * * *", timeout=120, json_args={})
```

```python
while scheduled_task.status() != "started":
    display(scheduled_task.status())
    time.sleep(5)
```

    'pending'

```python
#wait 120 seconds to give the scheduled event time to finish
time.sleep(120)
scheduled_task_end = datetime.datetime.now()

pipeline.logs(start_datetime = scheduled_task_start, end_datetime = scheduled_task_end)
```

    Warning: Pipeline log size limit exceeded. Please request logs using export_logs

{{<table "table table-striped table-bordered" >}}
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.variable</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-05-09 20:28:09.254</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.75]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-05-09 20:28:09.254</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[615094.56]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-05-09 20:28:09.254</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
      <td>[448627.72]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-05-09 20:28:09.254</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[758714.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-05-09 20:28:09.254</td>
      <td>[3.0, 1.75, 2200.0, 11520.0, 1.0, 0.0, 0.0, 4.0, 7.0, 2200.0, 0.0, 47.7659, -122.341, 1690.0, 8038.0, 62.0, 0.0, 0.0]</td>
      <td>[513264.7]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>496</th>
      <td>2023-05-09 20:28:09.254</td>
      <td>[4.0, 2.5, 2510.0, 47044.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2510.0, 0.0, 47.7699, -122.085, 2600.0, 42612.0, 27.0, 0.0, 0.0]</td>
      <td>[721143.6]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>497</th>
      <td>2023-05-09 20:28:09.254</td>
      <td>[4.0, 2.5, 4090.0, 11225.0, 2.0, 0.0, 0.0, 3.0, 10.0, 4090.0, 0.0, 47.581, -121.971, 3510.0, 8762.0, 9.0, 0.0, 0.0]</td>
      <td>[1048372.4]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>498</th>
      <td>2023-05-09 20:28:09.254</td>
      <td>[2.0, 1.0, 720.0, 5000.0, 1.0, 0.0, 0.0, 5.0, 6.0, 720.0, 0.0, 47.5195, -122.374, 810.0, 5000.0, 63.0, 0.0, 0.0]</td>
      <td>[244566.38]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>499</th>
      <td>2023-05-09 20:28:09.254</td>
      <td>[4.0, 2.75, 2930.0, 22000.0, 1.0, 0.0, 3.0, 4.0, 9.0, 1580.0, 1350.0, 47.3227, -122.384, 2930.0, 9758.0, 36.0, 0.0, 0.0]</td>
      <td>[518869.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>500</th>
      <td>2023-05-09 20:28:09.254</td>
      <td>[2.0, 1.0, 850.0, 5000.0, 1.0, 0.0, 0.0, 3.0, 6.0, 850.0, 0.0, 47.3817, -122.314, 1160.0, 5000.0, 39.0, 0.0, 0.0]</td>
      <td>[236238.66]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}
<p>501 rows × 4 columns</p>

## Kill Task

With our testing complete, we will kill the scheduled task so it will not run again.  First we'll show all the tasks to verify that our task is there, then issue it the kill command.

```python
wl.list_tasks()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>id</th><th>status</th><th>type</th><th>created at</th><th>updated at</th></tr><tr><td>d2c59ab8-41d6-402a-acc3-c99cd7c7d421</td><td>started</td><td>Temporary Run</td><td>2023-08-May 22:13:43</td><td>2023-08-May 22:13:56</td></tr><tr><td>6a403c5e-bee9-46ea-965c-cc4da3a8bc0b</td><td>started</td><td>Temporary Run</td><td>2023-08-May 22:13:43</td><td>2023-08-May 22:13:51</td></tr><tr><td>95d04724-9752-4719-bb66-971f35d89d0c</td><td>started</td><td>Temporary Run</td><td>2023-09-May 14:29:13</td><td>2023-09-May 14:29:17</td></tr><tr><td>8691db2a-6a2f-4c20-ac20-140f938af8c0</td><td>failed</td><td>Scheduled Run</td><td>2023-08-May 22:15:20</td><td>2023-08-May 22:15:21</td></tr><tr><td>30e634c3-5152-4eb5-9302-26b18ce97e2c</td><td>started</td><td>Temporary Run</td><td>2023-08-May 22:25:44</td><td>2023-08-May 22:25:59</td></tr><tr><td>ac3b1dc2-1e8d-44ae-a7bb-fa3e17b37166</td><td>failed</td><td>Scheduled Run</td><td>2023-08-May 22:16:55</td><td>2023-08-May 22:16:55</td></tr><tr><td>4088fae9-fc0c-4a83-b886-c85d083f52ed</td><td>started</td><td>Temporary Run</td><td>2023-08-May 22:25:44</td><td>2023-08-May 22:25:48</td></tr><tr><td>077f6c9e-6609-4d2b-b115-aa447c163c4f</td><td>started</td><td>Temporary Run</td><td>2023-09-May 14:59:29</td><td>2023-09-May 14:59:33</td></tr><tr><td>a09b526f-6149-42e2-86f3-37a6e7ee5f96</td><td>started</td><td>Temporary Run</td><td>2023-09-May 14:59:29</td><td>2023-09-May 14:59:42</td></tr><tr><td>c4f4b23e-ccc4-4d35-9b87-ea4331e15527</td><td>started</td><td>Temporary Run</td><td>2023-09-May 15:07:44</td><td>2023-09-May 15:07:49</td></tr><tr><td>beb1ee60-528b-41bb-ba7e-e49e866391ea</td><td>started</td><td>Temporary Run</td><td>2023-09-May 15:07:44</td><td>2023-09-May 15:07:53</td></tr><tr><td>daef0167-6bc9-4d10-88e8-0835671d6e23</td><td>started</td><td>Temporary Run</td><td>2023-09-May 15:11:51</td><td>2023-09-May 15:11:55</td></tr><tr><td>bf265485-fd9c-4867-a15d-337a36eae223</td><td>started</td><td>Temporary Run</td><td>2023-09-May 15:17:46</td><td>2023-09-May 15:17:55</td></tr><tr><td>916ce33d-d0d2-43c7-8f31-fc97aa50448c</td><td>started</td><td>Temporary Run</td><td>2023-09-May 15:35:34</td><td>2023-09-May 15:35:44</td></tr><tr><td>8ea4bc88-9ed8-49db-9928-efb0b916d787</td><td>started</td><td>Temporary Run</td><td>2023-09-May 15:36:59</td><td>2023-09-May 15:37:04</td></tr><tr><td>1d473a4c-f0be-4cba-8960-b399fa8a58e7</td><td>started</td><td>Temporary Run</td><td>2023-09-May 15:38:30</td><td>2023-09-May 15:38:34</td></tr><tr><td>e3aa67b1-509b-4d62-8c4d-71b8b84c5c97</td><td>started</td><td>Temporary Run</td><td>2023-09-May 15:41:09</td><td>2023-09-May 15:41:13</td></tr><tr><td>26584b3f-ce6e-4574-ad9c-304fc010a0b4</td><td>started</td><td>Temporary Run</td><td>2023-09-May 15:50:00</td><td>2023-09-May 15:50:10</td></tr><tr><td>86b8f502-3c9a-461b-9510-209f1d6c8b40</td><td>started</td><td>Temporary Run</td><td>2023-09-May 15:53:56</td><td>2023-09-May 15:54:00</td></tr><tr><td>3b61654e-6131-4430-8949-a3175c90a278</td><td>started</td><td>Temporary Run</td><td>2023-09-May 15:51:26</td><td>2023-09-May 15:51:30</td></tr><tr><td>b709a877-67cd-40d3-b569-719175a8391e</td><td>started</td><td>Temporary Run</td><td>2023-09-May 16:00:32</td><td>2023-09-May 16:00:39</td></tr><tr><td>e3acd950-6a6a-4464-9aa9-9865bdd29fa0</td><td>started</td><td>Temporary Run</td><td>2023-09-May 16:00:32</td><td>2023-09-May 16:00:43</td></tr><tr><td>95196607-1b7b-4d83-9177-623ad0626604</td><td>started</td><td>Temporary Run</td><td>2023-09-May 19:40:42</td><td>2023-09-May 19:40:47</td></tr><tr><td>8efddfe2-170c-4ef0-99d3-f821cfb37405</td><td>started</td><td>Temporary Run</td><td>2023-09-May 19:40:42</td><td>2023-09-May 19:40:58</td></tr><tr><td>916900e4-4927-4578-98d1-51d6a24ccdb4</td><td>started</td><td>Temporary Run</td><td>2023-09-May 19:49:35</td><td>2023-09-May 19:49:40</td></tr><tr><td>9106b611-885d-4003-b89a-422c6f4e4810</td><td>started</td><td>Temporary Run</td><td>2023-09-May 20:08:54</td><td>2023-09-May 20:09:00</td></tr><tr><td>989f6579-a617-4600-b326-ae3c4483561f</td><td>started</td><td>Temporary Run</td><td>2023-09-May 20:26:00</td><td>2023-09-May 20:26:06</td></tr><tr><td>0b740afd-2e7d-4c50-b391-39353aeca97a</td><td>started</td><td>Scheduled Run</td><td>2023-09-May 20:27:09</td><td>2023-09-May 20:27:11</td></tr></table>
{{</table>}}

```python
scheduled_task.kill()
```

    <ArbexStatus.PENDING_KILL: 'pending_kill'>

## Cleanup

With the tutorial complete, we can undeploy the pipeline and return the resources back to the Wallaroo instance.

```python
pipeline.undeploy()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>simpleorchestrationpipeline</td></tr><tr><th>created</th> <td>2023-05-08 22:08:15.458098+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-09 20:24:40.407056+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>187cb5e7-c67e-499c-ad56-c7867727ad6d, afb5d36f-338e-4bb2-a7e4-24d1280c1575, 0daa5b47-0852-4b33-8157-ea07c03a98df, bc38bf67-f916-4bb4-87e2-27836eda031f, 6e91a9ae-13d7-40db-97e8-c62af1b81392, 90f7408e-c85f-42aa-b124-43eb9252cc23, 8e68f5d6-94dd-46e7-8897-2f85a6acbbc6, 8a042c5b-6913-41e5-a3b8-f9bde2f6c493, a80664ed-48d6-460b-8289-ff8e1cc31a9e, 7dfcb260-3c2e-4e62-b296-7e9a2f57bde2, 59003cd3-bd4b-43c8-b457-4d0b9cdbea51, 6eaa00b8-0789-4a51-b182-c9bee83ee4a3, cad9521a-fa40-464a-9ec8-0749d1ffbe73, 42170c80-c78a-43fc-b463-356730d40de8, c1c09de1-8da2-431f-ad58-cf469129b12b, 5300e180-9340-481a-a861-99ca20749c18, 4279ec2c-8f83-4c76-a469-c49e654a7168, 6c792f92-9d7a-42d5-b246-4d793edc2180, ee8be66d-3b4d-491d-aa2f-65fcef339aec</td></tr><tr><th>steps</th> <td>simpleorchestrationmodel</td></tr></table>
{{</table>}}

