This can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/workload-orchestrations/orchestration_sdk_simple_tutorial).

## Wallaroo Connection and ML Workload Orchestration Simple Tutorial

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

import time

# Used to create unique workspace and pipeline names
import string
import random

# make a random 4 character suffix
suffix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))
display(suffix)
```

    'dtzw'

### Connect to the Wallaroo Instance

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
# Login through local Wallaroo instance

wl = wallaroo.Client()
```

```python
# Setting variables for later steps

workspace_name = f'simpleorchestrationworkspace{suffix}'
pipeline_name = f'simpleorchestrationpipeline{suffix}'
model_name = f'simpleorchestrationmodel{suffix}'
model_file_name = './models/rf_model.onnx'

inference_connection_name = f'external_inference_connection{suffix}'
inference_connection_type = "HTTP"
inference_connection_argument = {'host':'https://github.com/WallarooLabs/Wallaroo_Tutorials/raw/main/wallaroo-testing-tutorials/houseprice-saga/data/xtest-1k.arrow'}
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

housing_model_control = wl.upload_model(model_name, model_file_name, framework=wallaroo.framework.Framework.ONNX).configure()

# Add the model as a pipeline step

pipeline.add_model_step(housing_model_control)
```

<table><tr><th>name</th> <td>simpleorchestrationpipelinedtzw</td></tr><tr><th>created</th> <td>2023-05-23 15:26:00.268667+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-23 15:26:00.268667+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>9a5afba3-c664-4d57-8c08-fc072d3f549c</td></tr><tr><th>steps</th> <td></td></tr></table>

```python
#deploy the pipeline
pipeline.deploy()
```

    Waiting for deployment - this will take up to 45s ....................... ok

<table><tr><th>name</th> <td>simpleorchestrationpipelinedtzw</td></tr><tr><th>created</th> <td>2023-05-23 15:26:00.268667+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-23 15:26:00.568944+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>24f5d5e9-59fe-4440-9e08-78c0003226df, 9a5afba3-c664-4d57-8c08-fc072d3f549c</td></tr><tr><th>steps</th> <td>simpleorchestrationmodeldtzw</td></tr></table>

## Create Connections

We will create the data source connection via the Wallaroo client command `create_connection`.

Connections are created with the Wallaroo client command [`create_connection`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-ml-workload-orchestration/#create-orchestration) with the following parameters.

| Parameter | Type | Description |
| --- | --- | ---|
| **name** | string (Required) | The name of the connection. This must be unique - **if submitting the name of an existing** connection it will return an error. |
| **type** | string (Required) | The user defined type of connection. |
| **details** | Dict (Required) | User defined configuration details for the data connection.  These can be `{'username':'dataperson', 'password':'datapassword', 'port': 3339}`, or `{'token':'abcde123==', 'host':'example.com', 'port:1234'}`, or other user defined combinations.  |

* **IMPORTANT NOTE**:  Data connections names **must** be unique.  Attempting to create a data connection with the same `name` as an existing data connection will result in an error.

We'll also create a data connection named `inference_results_connection` with our helper function `get_connection` that will either create **or** retrieve a connection if it already exists.  From there we'll create out connections:  

* `houseprice_arrow_table`: An Apache Arrow file stored on GitHub that will be used for our inference input.

```python
wl.create_connection(inference_connection_name, inference_connection_type, inference_connection_argument)
```

<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Name</td><td>external_inference_connectiondtzw</td>
  </tr>
  <tr>
    <td>Connection Type</td><td>HTTP</td>
  </tr>
  <tr>
    <td>Details</td><td>*****</td>
  </tr>
  <tr>
    <td>Created At</td><td>2023-05-23T15:26:24.613152+00:00</td>
  </tr>
  <tr>
    <td>Linked Workspaces</td><td>[]</td>
  </tr>
</table>

### Get Connection by Name

The Wallaroo client method `get_connection(name)` retrieves the connection that matches the `name` parameter.  We'll retrieve our connection and store it as `inference_source_connection`.

```python
inference_source_connection = wl.get_connection(name=inference_connection_name)
display(inference_source_connection)
```

<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Name</td><td>external_inference_connectiondtzw</td>
  </tr>
  <tr>
    <td>Connection Type</td><td>HTTP</td>
  </tr>
  <tr>
    <td>Details</td><td>*****</td>
  </tr>
  <tr>
    <td>Created At</td><td>2023-05-23T15:26:24.613152+00:00</td>
  </tr>
  <tr>
    <td>Linked Workspaces</td><td>[]</td>
  </tr>
</table>

### Add Connection to Workspace

The method Workspace [`add_connection(connection_name)`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-dataconnections/#add-data-connection-to-workspace) adds a Data Connection to a workspace, and takes the following parameters.

| Parameter | Type | Description |
| --- | --- | ---|
| **name** | string (Required) | The name of the Data Connection |

We'll add both connections to our sample workspace, then list the connections available to the workspace to confirm.

```python
workspace.add_connection(inference_connection_name)
workspace.list_connections()
```

<table><tr><th>name</th><th>connection type</th><th>details</th><th>created at</th><th>linked workspaces</th></tr><tr><td>external_inference_connectiondtzw</td><td>HTTP</td><td>*****</td><td>2023-05-23T15:26:24.613152+00:00</td><td>['simpleorchestrationworkspacedtzw']</td></tr></table>

## Wallaroo ML Workload Orchestration Example

With the pipeline deployed and our connections set, we will now generate our ML Workload Orchestration.  See the [Wallaroo ML Workload Orchestrations guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-ml-workload-orchestration/) for full details.

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

### Orchestration Status

We will loop until the uploaded orchestration's `status` displays `ready`.

```python
orchestration = wl.upload_orchestration(path="./remote_inference/remote_inference.zip")

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

```python
wl.list_orchestrations()
```

<table><tr><th>id</th><th>name</th><th>status</th><th>filename</th><th>sha</th><th>created at</th><th>updated at</th></tr><tr><td>701c991a-a716-4bca-aa69-afd957ff189e</td><td>None</td><td>ready</td><td>remote_inference.zip</td><td>b4593d...d5a86f</td><td>2023-23-May 15:26:24</td><td>2023-23-May 15:27:13</td></tr></table>

### Upload Orchestration via File Object

Another method to upload the orchestration is as a file object.  For that, we will open the zip file as a binary, then upload it using the `bytes_buffer` parameter to specify the file object, and the `file_name` to give it a new name.

```python
zipfile = open("./remote_inference/remote_inference.zip", "rb").read()

wl.upload_orchestration(bytes_buffer=zipfile, file_name="inferencetest.zip", name="uploadedbytesdemo")
```

<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>ID</td><td>7683f0f9-13fa-4257-840a-8e1cf8b12089</td>
  </tr>
  <tr>
    <td>Name</td><td>uploadedbytesdemo</td>
  </tr>
  <tr>
    <td>File Name</td><td>inferencetest.zip</td>
  </tr>
  <tr>
    <td>SHA</td><td>b4593d6084e07e9ad1b57367258ca425d7f290540ab4378b8cba168b91d5a86f</td>
  </tr>
  <tr>
    <td>Status</td><td>pending_packaging</td>
  </tr>
  <tr>
    <td>Created At</td><td>2023-23-May 15:27:15</td>
  </tr>
  <tr>
    <td>Updated At</td><td>2023-23-May 15:27:15</td>
  </tr>
</table>

```python
wl.list_orchestrations()
```

<table><tr><th>id</th><th>name</th><th>status</th><th>filename</th><th>sha</th><th>created at</th><th>updated at</th></tr><tr><td>701c991a-a716-4bca-aa69-afd957ff189e</td><td>None</td><td>ready</td><td>remote_inference.zip</td><td>b4593d...d5a86f</td><td>2023-23-May 15:26:24</td><td>2023-23-May 15:27:13</td></tr><tr><td>7683f0f9-13fa-4257-840a-8e1cf8b12089</td><td>uploadedbytesdemo</td><td>pending_packaging</td><td>inferencetest.zip</td><td>b4593d...d5a86f</td><td>2023-23-May 15:27:15</td><td>2023-23-May 15:27:15</td></tr></table>

## Task Management Tutorial

Once an Orchestration has the status `ready`, it can be run as a task.  Tasks have three run options.

| Type | SDK Call |  How triggered |
|---|---|:---|
| Once       | `orchestration.run_once(name, json_args, timeout)` | Task runs once and exits.| Single batch, experimentation. |
| Scheduled  | `orchestration.run_scheduled(name, schedule, timeout, json_args)` | User provides schedule. Task runs exits whenever schedule dictates. | Recurrent batch. |

### Run Task Once

We'll do both a Run Once task and generate our Run Once Task from our orchestration.

Tasks are generated and run once with the Orchestration `run_once(name, json_args, timeout)` method.  Any arguments for the orchestration are passed in as a `Dict`.  If there are no arguments, then an empty set `{}` is passed.

```python
# Example: run once

import datetime
task_start = datetime.datetime.now()
task = orchestration.run_once(name="simpletaskdemo", 
                              json_args={"workspace_name": workspace_name, 
                                         "pipeline_name": pipeline_name,
                                         "connection_name": inference_connection_name
                                            })
```

### Task Status

The list of tasks in the Wallaroo instance is retrieves through the Wallaroo Client `list_tasks()` method.  This returns an array list of the following.

| Parameter | Type | Description |
| --- | --- | ---|
| **id** | string | The UUID identifier for the task. |
| **last run status** | string | The last reported status the task.  Values are: <br><ul><li>`pending`: The task has not been started.</li><li>`started`: The task has been scheduled to execute.</li><li>`pending_kill`: The task kill command has been issued and the task is scheduled to be stopped.</li></ul> |
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

    'pending'

    'pending'

### Task Results

We can view the inferences from our logs and verify that new entries were added from our task.  We can do that with the task `logs()` method.

In our case, we'll assume the task once started takes about 1 minute to run (deploy the pipeline, run the inference, undeploy the pipeline).  We'll add in a wait of 1 minute, then display the logs during the time period the task was running.

```python
time.sleep(60)

task_end = datetime.datetime.now()
display(task_end)

pipeline.logs(start_datetime = task_start, end_datetime = task_end)
```

    datetime.datetime(2023, 5, 23, 15, 28, 30, 718361)

    Warning: Pipeline log size limit exceeded. Please request logs using export_logs

<table border="1" class="dataframe">
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
      <td>2023-05-23 15:27:28.001</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.75]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-05-23 15:27:28.001</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[615094.56]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-05-23 15:27:28.001</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
      <td>[448627.72]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-05-23 15:27:28.001</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[758714.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-05-23 15:27:28.001</td>
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
      <th>487</th>
      <td>2023-05-23 15:27:28.001</td>
      <td>[3.0, 1.5, 1030.0, 8414.0, 1.0, 0.0, 0.0, 4.0, 7.0, 1030.0, 0.0, 47.7654, -122.297, 1750.0, 8414.0, 47.0, 0.0, 0.0]</td>
      <td>[340764.53]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>488</th>
      <td>2023-05-23 15:27:28.001</td>
      <td>[4.0, 2.75, 2450.0, 15002.0, 1.0, 0.0, 0.0, 5.0, 9.0, 2450.0, 0.0, 47.4268, -122.343, 2650.0, 15055.0, 40.0, 0.0, 0.0]</td>
      <td>[508746.75]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>489</th>
      <td>2023-05-23 15:27:28.001</td>
      <td>[2.0, 1.0, 1010.0, 4000.0, 1.0, 0.0, 0.0, 3.0, 6.0, 1010.0, 0.0, 47.5536, -122.267, 1040.0, 4000.0, 103.0, 0.0, 0.0]</td>
      <td>[435628.56]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>490</th>
      <td>2023-05-23 15:27:28.001</td>
      <td>[3.0, 2.5, 1330.0, 1200.0, 3.0, 0.0, 0.0, 3.0, 7.0, 1330.0, 0.0, 47.7034, -122.344, 1330.0, 1206.0, 12.0, 0.0, 0.0]</td>
      <td>[342604.4]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>491</th>
      <td>2023-05-23 15:27:28.001</td>
      <td>[2.0, 1.75, 2770.0, 19700.0, 2.0, 0.0, 0.0, 3.0, 8.0, 1780.0, 990.0, 47.7581, -122.365, 2360.0, 9700.0, 31.0, 0.0, 0.0]</td>
      <td>[536371.25]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>492 rows × 4 columns</p>

## Scheduled Run Task Example

The other method of using tasks is as a **scheduled run** through the Orchestration `run_scheduled(name, schedule, timeout, json_args)`.  This sets up a task to run on an regular schedule as defined by the `schedule` parameter in the `cron` service format.  For example:

```python
schedule={'42 * * * *'}
```

Runs on the 42nd minute of every hour.

The following schedule runs every day at 12 noon from February 1 to February 15 2024 - and then ends.

```python
schedule={'0 0 12 1-15 2 2024'}
```

For our example, we will create a scheduled task to run every 5 minutes, display the inference results, then use the Orchestration `kill` task to keep the task from running any further.

It is recommended that orchestrations that have pipeline deploy or undeploy commands be spaced out no less than 5 minutes to prevent colliding with other tasks that use the same pipeline.

```python
scheduled_task_start = datetime.datetime.now()
```

```python

scheduled_task = orchestration.run_scheduled(name="simple_inference_schedule", 
                                             schedule="*/5 * * * *", 
                                             timeout=120, 
                                             json_args={"workspace_name": workspace_name, 
                                                        "pipeline_name": pipeline_name,
                                                        "connection_name": inference_connection_name
                                            })
```

```python
while scheduled_task.status() != "started":
    display(scheduled_task.status())
    time.sleep(5)
```

    'pending'

```python
#wait 420 seconds to give the scheduled event time to finish
time.sleep(420)
scheduled_task_end = datetime.datetime.now()

pipeline.logs(start_datetime = scheduled_task_start, end_datetime = scheduled_task_end)
```

    Warning: Pipeline log size limit exceeded. Please request logs using export_logs

<table border="1" class="dataframe">
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
      <td>2023-05-23 15:30:08.633</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.75]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-05-23 15:30:08.633</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[615094.56]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-05-23 15:30:08.633</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
      <td>[448627.72]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-05-23 15:30:08.633</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[758714.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-05-23 15:30:08.633</td>
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
      <th>487</th>
      <td>2023-05-23 15:30:08.633</td>
      <td>[3.0, 1.5, 1030.0, 8414.0, 1.0, 0.0, 0.0, 4.0, 7.0, 1030.0, 0.0, 47.7654, -122.297, 1750.0, 8414.0, 47.0, 0.0, 0.0]</td>
      <td>[340764.53]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>488</th>
      <td>2023-05-23 15:30:08.633</td>
      <td>[4.0, 2.75, 2450.0, 15002.0, 1.0, 0.0, 0.0, 5.0, 9.0, 2450.0, 0.0, 47.4268, -122.343, 2650.0, 15055.0, 40.0, 0.0, 0.0]</td>
      <td>[508746.75]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>489</th>
      <td>2023-05-23 15:30:08.633</td>
      <td>[2.0, 1.0, 1010.0, 4000.0, 1.0, 0.0, 0.0, 3.0, 6.0, 1010.0, 0.0, 47.5536, -122.267, 1040.0, 4000.0, 103.0, 0.0, 0.0]</td>
      <td>[435628.56]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>490</th>
      <td>2023-05-23 15:30:08.633</td>
      <td>[3.0, 2.5, 1330.0, 1200.0, 3.0, 0.0, 0.0, 3.0, 7.0, 1330.0, 0.0, 47.7034, -122.344, 1330.0, 1206.0, 12.0, 0.0, 0.0]</td>
      <td>[342604.4]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>491</th>
      <td>2023-05-23 15:30:08.633</td>
      <td>[2.0, 1.75, 2770.0, 19700.0, 2.0, 0.0, 0.0, 3.0, 8.0, 1780.0, 990.0, 47.7581, -122.365, 2360.0, 9700.0, 31.0, 0.0, 0.0]</td>
      <td>[536371.25]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>492 rows × 4 columns</p>

## Kill Task

With our testing complete, we will kill the scheduled task so it will not run again.  First we'll show all the tasks to verify that our task is there, then issue it the kill command.

```python
wl.list_tasks()
```

<table><tr><th>id</th><th>name</th><th>last run status</th><th>type</th><th>active</th><th>schedule</th><th>created at</th><th>updated at</th></tr><tr><td>d0b6a83c-a3a1-41f0-98c9-92422d2544c4</td><td>simple_inference_schedule</td><td>success</td><td>Scheduled Run</td><td>True</td><td>*/5 * * * *</td><td>2023-23-May 15:28:30</td><td>2023-23-May 15:28:31</td></tr><tr><td>74046e01-68ab-42a0-bcd2-3493cbc66576</td><td>simpletaskdemo</td><td>success</td><td>Temporary Run</td><td>True</td><td>-</td><td>2023-23-May 15:27:15</td><td>2023-23-May 15:27:26</td></tr></table>

```python
scheduled_task.kill()
```

    <ArbexStatus.PENDING_KILL: 'pending_kill'>

```python
wl.list_tasks()
```

<table><tr><th>id</th><th>name</th><th>last run status</th><th>type</th><th>active</th><th>schedule</th><th>created at</th><th>updated at</th></tr><tr><td>74046e01-68ab-42a0-bcd2-3493cbc66576</td><td>simpletaskdemo</td><td>success</td><td>Temporary Run</td><td>True</td><td>-</td><td>2023-23-May 15:27:15</td><td>2023-23-May 15:27:26</td></tr></table>

## Cleanup

With the tutorial complete, we can undeploy the pipeline and return the resources back to the Wallaroo instance.

```python
pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s ..................................... ok

<table><tr><th>name</th> <td>simpleorchestrationpipelinedtzw</td></tr><tr><th>created</th> <td>2023-05-23 15:26:00.268667+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-23 15:26:00.568944+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>24f5d5e9-59fe-4440-9e08-78c0003226df, 9a5afba3-c664-4d57-8c08-fc072d3f549c</td></tr><tr><th>steps</th> <td>simpleorchestrationmodeldtzw</td></tr></table>

