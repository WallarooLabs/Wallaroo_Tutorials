This can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/pipeline-orchestrators/orchestration_api_simple_tutorial).

## Pipeline Orchestrations API Tutorial

This tutorial provides a quick set of methods and examples regarding Wallaroo Connections and Wallaroo ML Workload Orchestration.  For full details, see the Wallaroo Documentation site.

Wallaroo provides Data Connections and ML Workload Orchestrations to provide organizations with a method of creating and managing automated tasks that can either be run on demand or a regular schedule.

## Definitions

* **Orchestration**: A set of instructions written as a python script with a requirements library.  Orchestrations are uploaded to the Wallaroo instance as a .zip file.
* **Task**: An implementation of an orchestration.  Tasks are run either once when requested, on a repeating schedule, or as a service.
* **Connection**: Definitions set by MLOps engineers that are used by other Wallaroo users for connection information to a data source.  Usually paired with orchestrations.

## Tutorial Goals

The tutorial will demonstrate the following:

1. Create a workspace and pipeline with a sample model.
1. Upload Wallaroo ML Workload Orchestration through the Wallaroo MLOps API.
1. List available orchestrations through the Wallaroo MLOps API.
1. Run the orchestration once as a Run Once Task through the Wallaroo MLOps API and verify that the information was saved the pipeline logs.

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

import requests

# Used to create unique workspace and pipeline names
import string
import random

# make a random 4 character prefix
suffix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))
display(suffix)
```

    'gsze'

### Connect to the Wallaroo Instance

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  If logging in externally, update the `wallarooPrefix` and `wallarooSuffix` variables with the proper DNS information.  For more information on Wallaroo DNS settings, see the [Wallaroo DNS Integration Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-dns-guide/).

```python
# Login through local Wallaroo instance

wl = wallaroo.Client()
```

### API URL

The variable `APIURL` is used to specify the connection to the Wallaroo instance's MLOps API URL, and is composed of the Wallaroo DNS prefix and suffix.  For full details, see the [Wallaroo API Connection Guide
](https://staging.docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-connection-guide/).

The variables `wallarooPrefix` and `wallarooSuffix` variables will be used to derive the API url.  For example, if the Wallaroo Prefix is `doc-test` and the url is `example.com`, then the MLOps API URL would be `doc-test.api.example.com/v1/api/{request}`.

Set the Wallaroo Prefix and Suffix in the code segment below based on your Wallaroo instance.

```python
# Setting variables for later steps

wallarooPrefix = "YOUR PREFIX"

wallarooSuffix = "YOUR SUFFIX"

APIURL = f"https://{wallarooPrefix}.api.{wallarooSuffix}"

workspace_name = f'apiorchestrationworkspace{suffix}'
pipeline_name = f'apipipeline{suffix}'
model_name = f'apiorchestrationmodel{suffix}'
model_file_name = './models/rf_model.onnx'
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

workspace_id = workspace.id()

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
<table><tr><th>name</th> <td>apipipelinegsze</td></tr><tr><th>created</th> <td>2023-05-22 20:48:30.700499+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-22 20:48:30.700499+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>101b252a-623c-4185-a24d-ec00593dda79</td></tr><tr><th>steps</th> <td></td></tr></table>
{{</table>}}

```python
#deploy the pipeline
pipeline.deploy()
```

    Waiting for deployment - this will take up to 45s ......... ok

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>apipipelinegsze</td></tr><tr><th>created</th> <td>2023-05-22 20:48:30.700499+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-22 20:48:31.357336+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>aac61b5a-e4f4-4ea3-9347-6482c330b5f5, 101b252a-623c-4185-a24d-ec00593dda79</td></tr><tr><th>steps</th> <td>apiorchestrationmodelgsze</td></tr></table>
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

### API Upload the Orchestration

Orchestrations are uploaded via POST as a `application/octet-stream` with MLOps API route:

* **REQUEST**
  * POST `/v1/api/orchestration/upload`
  * Content-Type `multipart/form-data`
* **PARAMETERS**
  * `file`: The file uploaded as Content-Type as `application/octet-stream`.
  * `metadata`: Included as Content-Type as `application/json` with:
    * workspace_id: The numerical id of the workspace to upload the orchestration to.
    * name:  The name of the orchestration.
    * The metadata specifying the workspace id and Content-Type as `application/json`.

Once uploaded, the deployment will be prepared and any requirements will be downloaded and installed.

```python
# retrieve the authorization token
headers = wl.auth.auth_header()

url = f"{APIURL}/v1/api/orchestration/upload"

fp = open("./api_inference_orchestration.zip", "rb")

metadata = f'{{"workspace_id": {workspace_id},"name": "apiorchestrationsample"}}'

response = requests.post(
    url,
    headers=headers,
    files=[
        ("file", 
            ("api_inference_orchestration.zip", fp, "application/octet-stream")
        ),
        ("metadata", 
            ("metadata", metadata, "application/json")
        )
    ],
).json()

display(response)
orchestration_id = response['id']
```

    {'id': 'b951f7b8-0690-4004-86bf-cc9802359313'}

### API List Orchestrations

A list of orchestrations retrieved via POST MLOps API route:

* **REQUEST**
  * POST `/v1/api/orchestration/list`
* **PARAMETERS**
  * **workspace_id**:  The numerical identifier of the workspace associated with the orchestration.

```python
# retrieve the authorization token
headers = wl.auth.auth_header()

url = f"{APIURL}/v1/api/orchestration/list"

data = {
    'workspace_id': workspace_id
}

response=requests.post(url, headers=headers, json=data)
display(response.json())

```

    [{'id': 'b951f7b8-0690-4004-86bf-cc9802359313',
      'workspace_id': 8,
      'sha': 'd3b93c9f280734106376e684792aa8b4285d527092fe87d89c74ec804f8e169e',
      'name': 'apiorchestrationsample',
      'file_name': 'api_inference_orchestration.zip',
      'task_id': 'c9622cf8-cbe5-4e3c-b64c-dabd6c5b7fef',
      'owner_id': 'a6857628-b0aa-451e-a0a0-bbc1d6eea6e0',
      'status': 'pending_packaging',
      'created_at': '2023-05-22T20:48:41.233482+00:00',
      'updated_at': '2023-05-22T20:48:41.233482+00:00'}]

```python

```

### API Get Orchestration

A list of orchestrations retrieved via POST MLOps API route:

* **REQUEST**
  * POST `/v1/api/orchestration/get_by_id`
* **PARAMETERS**
  * **id**:  The UUID of the orchestration being retrieved.
* **RETURNS**
  * id: The ID of the orchestration in UUID format.
  * workspace_id: Numerical value of the workspace the orchestration was uploaded to.
  * sha: The SHA hash value of the orchestration.
  * file_name: The file name the orchestration was uploaded as.
  * task_id: The task id in UUID format for unpacking and preparing the orchestration.
  * owner_id: The Keycloak ID of the user that uploaded the orchestration.
  * status: The status of the orchestration.  Status values are:
    * `packing`: Preparing the orchestration to be used as a task.
    * `ready`:  The orchestration is ready to be deployed as a task.

```python
# retrieve the authorization token
headers = wl.auth.auth_header()

url = f"{APIURL}/v1/api/orchestration/get_by_id"

data = {
    'id': orchestration_id
}

# loop until status is ready
status = None

while status != 'ready':
    response=requests.post(url, headers=headers, json=data).json()
    display(response)
    status = response['status']
    time.sleep(10)

orchestration_sha = response['sha']
```

    {'id': 'b951f7b8-0690-4004-86bf-cc9802359313',
     'workspace_id': 8,
     'sha': 'd3b93c9f280734106376e684792aa8b4285d527092fe87d89c74ec804f8e169e',
     'file_name': 'api_inference_orchestration.zip',
     'task_id': 'c9622cf8-cbe5-4e3c-b64c-dabd6c5b7fef',
     'owner_id': 'a6857628-b0aa-451e-a0a0-bbc1d6eea6e0',
     'status': 'packaging'}

    {'id': 'b951f7b8-0690-4004-86bf-cc9802359313',
     'workspace_id': 8,
     'sha': 'd3b93c9f280734106376e684792aa8b4285d527092fe87d89c74ec804f8e169e',
     'file_name': 'api_inference_orchestration.zip',
     'task_id': 'c9622cf8-cbe5-4e3c-b64c-dabd6c5b7fef',
     'owner_id': 'a6857628-b0aa-451e-a0a0-bbc1d6eea6e0',
     'status': 'packaging'}

    {'id': 'b951f7b8-0690-4004-86bf-cc9802359313',
     'workspace_id': 8,
     'sha': 'd3b93c9f280734106376e684792aa8b4285d527092fe87d89c74ec804f8e169e',
     'file_name': 'api_inference_orchestration.zip',
     'task_id': 'c9622cf8-cbe5-4e3c-b64c-dabd6c5b7fef',
     'owner_id': 'a6857628-b0aa-451e-a0a0-bbc1d6eea6e0',
     'status': 'packaging'}

    {'id': 'b951f7b8-0690-4004-86bf-cc9802359313',
     'workspace_id': 8,
     'sha': 'd3b93c9f280734106376e684792aa8b4285d527092fe87d89c74ec804f8e169e',
     'file_name': 'api_inference_orchestration.zip',
     'task_id': 'c9622cf8-cbe5-4e3c-b64c-dabd6c5b7fef',
     'owner_id': 'a6857628-b0aa-451e-a0a0-bbc1d6eea6e0',
     'status': 'packaging'}

    {'id': 'b951f7b8-0690-4004-86bf-cc9802359313',
     'workspace_id': 8,
     'sha': 'd3b93c9f280734106376e684792aa8b4285d527092fe87d89c74ec804f8e169e',
     'file_name': 'api_inference_orchestration.zip',
     'task_id': 'c9622cf8-cbe5-4e3c-b64c-dabd6c5b7fef',
     'owner_id': 'a6857628-b0aa-451e-a0a0-bbc1d6eea6e0',
     'status': 'packaging'}

    {'id': 'b951f7b8-0690-4004-86bf-cc9802359313',
     'workspace_id': 8,
     'sha': 'd3b93c9f280734106376e684792aa8b4285d527092fe87d89c74ec804f8e169e',
     'file_name': 'api_inference_orchestration.zip',
     'task_id': 'c9622cf8-cbe5-4e3c-b64c-dabd6c5b7fef',
     'owner_id': 'a6857628-b0aa-451e-a0a0-bbc1d6eea6e0',
     'status': 'packaging'}

    {'id': 'b951f7b8-0690-4004-86bf-cc9802359313',
     'workspace_id': 8,
     'sha': 'd3b93c9f280734106376e684792aa8b4285d527092fe87d89c74ec804f8e169e',
     'file_name': 'api_inference_orchestration.zip',
     'task_id': 'c9622cf8-cbe5-4e3c-b64c-dabd6c5b7fef',
     'owner_id': 'a6857628-b0aa-451e-a0a0-bbc1d6eea6e0',
     'status': 'packaging'}

    {'id': 'b951f7b8-0690-4004-86bf-cc9802359313',
     'workspace_id': 8,
     'sha': 'd3b93c9f280734106376e684792aa8b4285d527092fe87d89c74ec804f8e169e',
     'file_name': 'api_inference_orchestration.zip',
     'task_id': 'c9622cf8-cbe5-4e3c-b64c-dabd6c5b7fef',
     'owner_id': 'a6857628-b0aa-451e-a0a0-bbc1d6eea6e0',
     'status': 'packaging'}

    {'id': 'b951f7b8-0690-4004-86bf-cc9802359313',
     'workspace_id': 8,
     'sha': 'd3b93c9f280734106376e684792aa8b4285d527092fe87d89c74ec804f8e169e',
     'file_name': 'api_inference_orchestration.zip',
     'task_id': 'c9622cf8-cbe5-4e3c-b64c-dabd6c5b7fef',
     'owner_id': 'a6857628-b0aa-451e-a0a0-bbc1d6eea6e0',
     'status': 'ready'}

## Task Management Tutorial

Once an Orchestration has the status `ready`, it can be run as a task.  Tasks have three run options.

| Type | SDK Call |  How triggered |
|---|---|:---|
| Once       | `orchestration.run_once(name, json_args, timeout)` | Task runs once and exits.| Single batch, experimentation. |
| Scheduled  | `orchestration.run_scheduled(name, schedule, timeout, json_args)` | User provides schedule. Task runs exits whenever schedule dictates. | Recurrent batch. |

### Run Task Once via API

We'll do both a Run Once task and generate our Run Once Task from our orchestration.  Orchestrations are started as a run once task with the following request:

* **REQUEST**
  * POST `/v1/api/orchestration/task/run_once`
* **PARAMETERS**
  * **name** (*String* *Required*): The name to assign to the task.
  * **workspace_id** (*Integer* *Required*):  The numerical identifier of the workspace associated with the orchestration.
  * **orch_id**(*String* *Required*):  The orchestration ID represented by a UUID.
  * **json**(*Dict* *Required*):  The parameters to pass to the task.

Tasks are generated and run once with the Orchestration `run_once` method.  Any arguments for the orchestration are passed in as a `Dict`.  If there are no arguments, then an empty set `{}` is passed.

```python
# retrieve the authorization token
headers = wl.auth.auth_header()

data = {
    "name": "api run once task",
    "workspace_id": workspace_id,
    "orch_id": orchestration_id,
    "json": {
        "workspace_name": workspace_name,
        "pipeline_name": pipeline_name
    }
}

import datetime
task_start = datetime.datetime.now()

url=f"{APIURL}/v1/api/task/run_once"

response=requests.post(url, headers=headers, json=data).json()
display(response)
task_id = response['id']
```

    {'id': 'c868aa44-f7fe-4e3d-b11d-e1e6af3ec150'}

### Task Status via API

The list of tasks in the Wallaroo instance is retrieves through the Wallaroo MLOPs API request:

* **REQUEST**
  * POST `/v1/api/task/get_by_id`
* **PARAMETERS**
  * **task**:  The numerical identifier of the workspace associated with the orchestration.
  * **orch_id**:  The orchestration ID represented by a UUID.
  * **json**:  The parameters to pass to the task.

For this example, the status of the previously created task will be generated, then looped until it has reached status `started`.

```python
# retrieve the authorization token
headers = wl.auth.auth_header()

url=f"{APIURL}/v1/api/task/get_by_id"

data = {
    "id": task_id
}

status = None

while status != 'started':
    response=requests.post(url, headers=headers, json=data).json()
    display(response)
    status = response['status']
    time.sleep(10)
```

    {'name': 'api run once task',
     'id': 'c868aa44-f7fe-4e3d-b11d-e1e6af3ec150',
     'image': 'proxy.replicated.com/proxy/wallaroo/ghcr.io/wallaroolabs/arbex-orch-deploy',
     'image_tag': 'v2023.2.0-main-3271',
     'bind_secrets': ['minio'],
     'extra_env_vars': {'MINIO_URL': 'http://minio.wallaroo.svc.cluster.local:9000',
      'ORCH_OWNER_ID': 'a6857628-b0aa-451e-a0a0-bbc1d6eea6e0',
      'ORCH_SHA': 'd3b93c9f280734106376e684792aa8b4285d527092fe87d89c74ec804f8e169e',
      'TASK_DEBUG': 'false',
      'TASK_ID': 'c868aa44-f7fe-4e3d-b11d-e1e6af3ec150'},
     'auth_init': True,
     'workspace_id': 8,
     'flavor': 'exec_orch_oneshot',
     'reap_threshold_secs': 900,
     'exec_type': 'job',
     'status': 'pending',
     'input_data': {'pipeline_name': 'apipipelinegsze',
      'workspace_name': 'apiorchestrationworkspacegsze'},
     'killed': False,
     'created_at': '2023-05-22T21:08:31.099447+00:00',
     'updated_at': '2023-05-22T21:08:31.105312+00:00',
     'last_runs': []}

    {'name': 'api run once task',
     'id': 'c868aa44-f7fe-4e3d-b11d-e1e6af3ec150',
     'image': 'proxy.replicated.com/proxy/wallaroo/ghcr.io/wallaroolabs/arbex-orch-deploy',
     'image_tag': 'v2023.2.0-main-3271',
     'bind_secrets': ['minio'],
     'extra_env_vars': {'MINIO_URL': 'http://minio.wallaroo.svc.cluster.local:9000',
      'ORCH_OWNER_ID': 'a6857628-b0aa-451e-a0a0-bbc1d6eea6e0',
      'ORCH_SHA': 'd3b93c9f280734106376e684792aa8b4285d527092fe87d89c74ec804f8e169e',
      'TASK_DEBUG': 'false',
      'TASK_ID': 'c868aa44-f7fe-4e3d-b11d-e1e6af3ec150'},
     'auth_init': True,
     'workspace_id': 8,
     'flavor': 'exec_orch_oneshot',
     'reap_threshold_secs': 900,
     'exec_type': 'job',
     'status': 'started',
     'input_data': {'pipeline_name': 'apipipelinegsze',
      'workspace_name': 'apiorchestrationworkspacegsze'},
     'killed': False,
     'created_at': '2023-05-22T21:08:31.099447+00:00',
     'updated_at': '2023-05-22T21:08:36.585775+00:00',
     'last_runs': [{'run_id': '96a7f85f-e30c-40b5-9185-0dee5bd1a15e',
       'status': 'running',
       'created_at': '2023-05-22T21:08:33.112805+00:00',
       'updated_at': '2023-05-22T21:08:33.112805+00:00'}]}

### Task Results

We can view the inferences from our logs and verify that new entries were added from our task.  In our case, we'll assume the task once started takes about 1 minute to run (deploy the pipeline, run the inference, undeploy the pipeline).  We'll add in a wait of 1 minute, then display the logs during the time period the task was running.

```python
time.sleep(30)

task_end = datetime.datetime.now()
display(task_end)

pipeline.logs(start_datetime = task_start, end_datetime = task_end)
```

    datetime.datetime(2023, 5, 22, 21, 9, 32, 447321)

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
      <td>2023-05-22 21:08:37.779</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.7]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}

### Get Tasks by Orchestration SHA

Tasks tied to the same orchestration are retrieved through the following request.

* **REQUEST**
  * POST `/v1/api/task/get_tasks_by_orch_sha`
* **PARAMETERS**
  * **sha**:  The orchestrations SHA hash.
* **RETURNS**
  * `ids`: List[string] List of tasks by UUID.

```python
# retrieve the authorization token
headers = wl.auth.auth_header()

url=f"{APIURL}/v1/api/task/get_tasks_by_orch_sha"

data = {
    "sha": orchestration_sha
}

response=requests.post(url, headers=headers, json=data).json()
display(response)
```

    {'ids': ['2424a9a7-2331-42f6-bd84-90643386b130',
      'c868aa44-f7fe-4e3d-b11d-e1e6af3ec150',
      'a41fe4ae-b8a4-4e1f-a45a-114df64ae2bc']}

### Task Last Runs History

The history of a task, which each deployment of the task is known as a **task run** is retrieved with the Task `last_runs` method that takes the following arguments.  It returns the reverse chronological order of tasks runs listed by `updated_at`.

* **REQUEST**
  * POST `/v1/api/task/list_task_runs`
* **PARAMETERS**
  * **task_id**:  The numerical identifier of the task.
  * **status**:  Filters the task history by the `status`.  If `all`, returns all statuses.  Status values are: 
    * `running`: The task has started.
    * `failure`: The task failed.
    * `success`: The task completed.
  * **limit**:  The number of tasks runs to display.
* **RETURNS**
  * ids: List of task runs ids in UUID.

```python
# retrieve the authorization token
headers = wl.auth.auth_header()

url=f"{APIURL}/v1/api/task/list_task_runs"

data = {
    "task_id": task_id
}

response=requests.post(url, headers=headers, json=data).json()
task_run_id = response[0]['run_id']
display(response)
```

    [{'task': 'c868aa44-f7fe-4e3d-b11d-e1e6af3ec150',
      'run_id': '96a7f85f-e30c-40b5-9185-0dee5bd1a15e',
      'status': 'success',
      'created_at': '2023-05-22T21:08:33.112805+00:00',
      'updated_at': '2023-05-22T21:08:33.112805+00:00'}]

### Get Task Run Logs

Logs for a task run are retrieved through the following process. 

* **REQUEST**
  * POST `/v1/api/task/get_logs_for_run`
* **PARAMETERS**
  * **id**:  The numerical identifier of the task run associated with the orchestration.
  * **lines**:  The number of log lines to retrieve starting from the end of the log.
* **RETURNS**
  * **logs**: Array of log entries.

```python
# retrieve the authorization token
headers = wl.auth.auth_header()

url=f"{APIURL}/v1/api/task/get_logs_for_run"

data = {
    "id": task_run_id
}

response=requests.post(url, headers=headers, json=data).json()
display(response)
```

    {'logs': ["2023-05-22T21:09:17.683428502Z stdout F {'pipeline_name': 'apipipelinegsze', 'workspace_name': 'apiorchestrationworkspacegsze'}",
      '2023-05-22T21:09:17.683489102Z stdout F Getting the workspace apiorchestrationworkspacegsze',
      '2023-05-22T21:09:17.683497403Z stdout F Getting the pipeline apipipelinegsze',
      '2023-05-22T21:09:17.683504003Z stdout F Deploying the pipeline.',
      '2023-05-22T21:09:17.683510203Z stdout F Performing sample inference.',
      '2023-05-22T21:09:17.683516203Z stdout F                      time  ... check_failures',
      '2023-05-22T21:09:17.683521903Z stdout F 0 2023-05-22 21:08:37.779  ...              0',
      '2023-05-22T21:09:17.683527803Z stdout F ',
      '2023-05-22T21:09:17.683533603Z stdout F [1 rows x 4 columns]',
      '2023-05-22T21:09:17.683540103Z stdout F Undeploying the pipeline']}

### Run Task Scheduled via API

The other method of using tasks is as a **scheduled run** through the Orchestration `run_scheduled(name, schedule, timeout, json_args)`.  This sets up a task to run on an regular schedule as defined by the `schedule` parameter in the `cron` service format.  For example:

```python
schedule={'42 * * * *'}
```

Runs on the 42nd minute of every hour.

The following schedule runs every day at 12 noon from February 1 to February 15 2024 - and then ends.

```python
schedule={'0 0 12 1-15 2 2024'}
```

The Run Scheduled Task request is available at the following address:

`/v1/api/task/run_scheduled`

And takes the following parameters.

* **name** (*String* *Required*): The name to assign to the task.
* **orch_id** (*String* *Required*): The UUID orchestration ID to create the task from.
* **workspace_id**  (*Integer* *Required*):  The numberical identifier for the workspace.
* **schedule** (*String* *Required*): The schedule as a single string in `cron` format.
* **timeout**(*Integer* *Optional*):  The timeout to complete the task in seconds.
* **json** (*String* *Required*): The arguments to pass to the task.

For our example, we will create a scheduled task to run every 5 minutes, display the inference results, then use the Orchestration `kill` task to keep the task from running any further.

It is recommended that orchestrations that have pipeline deploy or undeploy commands be spaced out no less than 5 minutes to prevent colliding with other tasks that use the same pipeline.

```python
# retrieve the authorization token
headers = wl.auth.auth_header()

data = {
    "name": "scheduled api task",
    "workspace_id": workspace_id,
    "orch_id": orchestration_id,
    "schedule": "*/5 * * * *",
    "json": {
        "workspace_name": workspace_name,
        "pipeline_name": pipeline_name
    }
}

import datetime
task_start = datetime.datetime.now()

url=f"{APIURL}/v1/api/task/run_scheduled"

response=requests.post(url, headers=headers, json=data).json()
display(response)
scheduled_task_id = response['id']
```

    {'id': '87c2c7c0-e17e-4c00-9407-bfc44d632910'}

```python
# loop until the task is started

# retrieve the authorization token
headers = wl.auth.auth_header()

url=f"{APIURL}/v1/api/task/get_by_id"

data = {
    "id": scheduled_task_id
}

status = None

while status != 'started':
    response=requests.post(url, headers=headers, json=data).json()
    display(response)
    status = response['status']
    time.sleep(10)
```

    {'name': 'scheduled api task',
     'id': '87c2c7c0-e17e-4c00-9407-bfc44d632910',
     'image': 'proxy.replicated.com/proxy/wallaroo/ghcr.io/wallaroolabs/arbex-orch-deploy',
     'image_tag': 'v2023.2.0-main-3271',
     'bind_secrets': ['minio'],
     'extra_env_vars': {'MINIO_URL': 'http://minio.wallaroo.svc.cluster.local:9000',
      'ORCH_OWNER_ID': 'a6857628-b0aa-451e-a0a0-bbc1d6eea6e0',
      'ORCH_SHA': 'd3b93c9f280734106376e684792aa8b4285d527092fe87d89c74ec804f8e169e',
      'TASK_DEBUG': 'false',
      'TASK_ID': '87c2c7c0-e17e-4c00-9407-bfc44d632910'},
     'auth_init': True,
     'workspace_id': 8,
     'schedule': '*/5 * * * *',
     'reap_threshold_secs': 900,
     'status': 'started',
     'input_data': {'pipeline_name': 'apipipelinegsze',
      'workspace_name': 'apiorchestrationworkspacegsze'},
     'killed': False,
     'created_at': '2023-05-22T21:10:22.43957+00:00',
     'updated_at': '2023-05-22T21:10:22.871615+00:00',
     'last_runs': []}

```python
# retrieve the authorization token
headers = wl.auth.auth_header()

url=f"{APIURL}/v1/api/task/list_task_runs"

data = {
    "task_id": scheduled_task_id
}

# loop until we have a single run task
length = 0
while length == 0:
    response=requests.post(url, headers=headers, json=data).json()
    display(response)
    length = len(response)
    time.sleep(10)
task_run_id = response[0]['run_id']
```

    [{'task': '87c2c7c0-e17e-4c00-9407-bfc44d632910',
      'run_id': 'ad3d427a-3643-4f10-9b94-116688b32355',
      'status': 'running',
      'created_at': '2023-05-22T21:15:02.148274+00:00',
      'updated_at': '2023-05-22T21:15:02.148274+00:00'}]

```python
# retrieve the authorization token
headers = wl.auth.auth_header()

url=f"{APIURL}/v1/api/task/get_logs_for_run"

data = {
    "id": task_run_id
}

# loop until we get a log
response=requests.post(url, headers=headers, json=data).json()
display(response)
```

    {'logs': ["2023-05-22T21:15:55.548754679Z stdout F {'pipeline_name': 'apipipelinegsze', 'workspace_name': 'apiorchestrationworkspacegsze'}",
      '2023-05-22T21:15:55.54880928Z stdout F Getting the workspace apiorchestrationworkspacegsze',
      '2023-05-22T21:15:55.54881708Z stdout F Getting the pipeline apipipelinegsze',
      '2023-05-22T21:15:55.54882328Z stdout F Deploying the pipeline.',
      '2023-05-22T21:15:55.54883088Z stdout F Performing sample inference.',
      '2023-05-22T21:15:55.54883628Z stdout F                      time  ... check_failures',
      '2023-05-22T21:15:55.54884248Z stdout F 0 2023-05-22 21:15:17.420  ...              0',
      '2023-05-22T21:15:55.54884788Z stdout F ',
      '2023-05-22T21:15:55.54885348Z stdout F [1 rows x 4 columns]',
      '2023-05-22T21:15:55.54885938Z stdout F Undeploying the pipeline']}

## Cleanup

With the tutorial complete, we can undeploy the pipeline and return the resources back to the Wallaroo instance.

### Kill Task via API

Started tasks are killed through the following request:

`/v1/api/task/kill`

And takes the following parameters.

* **orch_id** (*String* *Required*): The UUID orchestration ID to create the task from.

```python
# retrieve the authorization token
headers = wl.auth.auth_header()

url=f"{APIURL}/v1/api/task/kill"

data = {
    "id": scheduled_task_id
}

response=requests.post(url, headers=headers, json=data).json()
display(response)
```

    {'name': 'scheduled api task',
     'id': '87c2c7c0-e17e-4c00-9407-bfc44d632910',
     'image': 'proxy.replicated.com/proxy/wallaroo/ghcr.io/wallaroolabs/arbex-orch-deploy',
     'image_tag': 'v2023.2.0-main-3271',
     'bind_secrets': ['minio'],
     'extra_env_vars': {'MINIO_URL': 'http://minio.wallaroo.svc.cluster.local:9000',
      'ORCH_OWNER_ID': 'a6857628-b0aa-451e-a0a0-bbc1d6eea6e0',
      'ORCH_SHA': 'd3b93c9f280734106376e684792aa8b4285d527092fe87d89c74ec804f8e169e',
      'TASK_DEBUG': 'false',
      'TASK_ID': '87c2c7c0-e17e-4c00-9407-bfc44d632910'},
     'auth_init': True,
     'workspace_id': 8,
     'schedule': '*/5 * * * *',
     'reap_threshold_secs': 900,
     'status': 'pending_kill',
     'input_data': {'pipeline_name': 'apipipelinegsze',
      'workspace_name': 'apiorchestrationworkspacegsze'},
     'killed': False,
     'created_at': '2023-05-22T21:10:22.43957+00:00',
     'updated_at': '2023-05-22T21:10:22.871615+00:00',
     'last_runs': [{'run_id': 'ad3d427a-3643-4f10-9b94-116688b32355',
       'status': 'success',
       'created_at': '2023-05-22T21:15:02.148274+00:00',
       'updated_at': '2023-05-22T21:15:02.148274+00:00'}]}

## Close Resources

With the tutorial complete, we'll verify the pipeline is closed so the resources are assigned back to the Wallaroo instance.

```python
pipeline.undeploy()
```

     ok

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>apipipelinegsze</td></tr><tr><th>created</th> <td>2023-05-22 20:48:30.700499+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-22 20:48:31.357336+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>aac61b5a-e4f4-4ea3-9347-6482c330b5f5, 101b252a-623c-4185-a24d-ec00593dda79</td></tr><tr><th>steps</th> <td>apiorchestrationmodelgsze</td></tr></table>
{{</table>}}

