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
```

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

workspace_name = 'apiorchestrationworkspace'
pipeline_name = 'apipipeline'
model_name = 'apiorchestrationmodel'
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
<table><tr><th>name</th> <td>apipipeline</td></tr><tr><th>created</th> <td>2023-05-17 16:15:51.394406+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 16:15:54.246572+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>0f774337-9a79-46f2-8a33-9580b389383e, b7b14b1a-2e2c-4a2e-b8c7-6f1640b9bb45</td></tr><tr><th>steps</th> <td>apiorchestrationmodel</td></tr></table>
{{</table>}}

```python
#deploy the pipeline
pipeline.deploy()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>apipipeline</td></tr><tr><th>created</th> <td>2023-05-17 16:15:51.394406+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 16:16:10.837047+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>a5a7c47a-c42d-41eb-88e1-615608e0a49b, 0f774337-9a79-46f2-8a33-9580b389383e, b7b14b1a-2e2c-4a2e-b8c7-6f1640b9bb45</td></tr><tr><th>steps</th> <td>apiorchestrationmodel</td></tr></table>
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

`/v1/api/orchestration/upload`

The following parameters are required:

* The upload is POST as a `multipart/form-data`.
* Included in the upload is:
  * The file with Content-Type as `application/octet-stream`.
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

    {'id': '97fc33e3-ed36-4460-b99c-d8e23cb1e339'}

### API List Orchestrations

A list of orchestrations retrieved via POST MLOps API route:

`/v1/api/orchestration/list`

The following parameters are required:

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

    [{'id': '886a75af-3e14-4120-bfb4-9bf29aa8e33d',
      'workspace_id': 9,
      'sha': 'd3b93c9f280734106376e684792aa8b4285d527092fe87d89c74ec804f8e169e',
      'name': 'apiorchestrationsample',
      'file_name': 'api_inference_orchestration.zip',
      'task_id': 'd19ebd7e-9bc3-47c6-b4ca-e0c958df7db2',
      'owner_id': '787453b1-fbd5-4b4e-90a5-8ccbe60778a9',
      'status': 'packaging',
      'created_at': '2023-05-17T16:25:17.34006+00:00',
      'updated_at': '2023-05-17T16:25:27.223682+00:00'},
     {'id': '97fc33e3-ed36-4460-b99c-d8e23cb1e339',
      'workspace_id': 9,
      'sha': 'd3b93c9f280734106376e684792aa8b4285d527092fe87d89c74ec804f8e169e',
      'name': 'apiorchestrationsample',
      'file_name': 'api_inference_orchestration.zip',
      'task_id': '6449f7fc-e123-476e-b872-b7854cc91a3b',
      'owner_id': '787453b1-fbd5-4b4e-90a5-8ccbe60778a9',
      'status': 'pending_packaging',
      'created_at': '2023-05-17T16:25:39.614228+00:00',
      'updated_at': '2023-05-17T16:25:39.614228+00:00'}]

### API Get Orchestration

A list of orchestrations retrieved via POST MLOps API route:

`/v1/api/orchestration/get_by_id`

The following parameters are required:

* **id**:  The UUID of the orchestration being retrieved.

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
    time.sleep(5)

```

    {'id': '97fc33e3-ed36-4460-b99c-d8e23cb1e339',
     'workspace_id': 9,
     'sha': 'd3b93c9f280734106376e684792aa8b4285d527092fe87d89c74ec804f8e169e',
     'file_name': 'api_inference_orchestration.zip',
     'task_id': '6449f7fc-e123-476e-b872-b7854cc91a3b',
     'owner_id': '787453b1-fbd5-4b4e-90a5-8ccbe60778a9',
     'status': 'packaging'}

    {'id': '97fc33e3-ed36-4460-b99c-d8e23cb1e339',
     'workspace_id': 9,
     'sha': 'd3b93c9f280734106376e684792aa8b4285d527092fe87d89c74ec804f8e169e',
     'file_name': 'api_inference_orchestration.zip',
     'task_id': '6449f7fc-e123-476e-b872-b7854cc91a3b',
     'owner_id': '787453b1-fbd5-4b4e-90a5-8ccbe60778a9',
     'status': 'packaging'}

    {'id': '97fc33e3-ed36-4460-b99c-d8e23cb1e339',
     'workspace_id': 9,
     'sha': 'd3b93c9f280734106376e684792aa8b4285d527092fe87d89c74ec804f8e169e',
     'file_name': 'api_inference_orchestration.zip',
     'task_id': '6449f7fc-e123-476e-b872-b7854cc91a3b',
     'owner_id': '787453b1-fbd5-4b4e-90a5-8ccbe60778a9',
     'status': 'packaging'}

    {'id': '97fc33e3-ed36-4460-b99c-d8e23cb1e339',
     'workspace_id': 9,
     'sha': 'd3b93c9f280734106376e684792aa8b4285d527092fe87d89c74ec804f8e169e',
     'file_name': 'api_inference_orchestration.zip',
     'task_id': '6449f7fc-e123-476e-b872-b7854cc91a3b',
     'owner_id': '787453b1-fbd5-4b4e-90a5-8ccbe60778a9',
     'status': 'packaging'}

    {'id': '97fc33e3-ed36-4460-b99c-d8e23cb1e339',
     'workspace_id': 9,
     'sha': 'd3b93c9f280734106376e684792aa8b4285d527092fe87d89c74ec804f8e169e',
     'file_name': 'api_inference_orchestration.zip',
     'task_id': '6449f7fc-e123-476e-b872-b7854cc91a3b',
     'owner_id': '787453b1-fbd5-4b4e-90a5-8ccbe60778a9',
     'status': 'packaging'}

    {'id': '97fc33e3-ed36-4460-b99c-d8e23cb1e339',
     'workspace_id': 9,
     'sha': 'd3b93c9f280734106376e684792aa8b4285d527092fe87d89c74ec804f8e169e',
     'file_name': 'api_inference_orchestration.zip',
     'task_id': '6449f7fc-e123-476e-b872-b7854cc91a3b',
     'owner_id': '787453b1-fbd5-4b4e-90a5-8ccbe60778a9',
     'status': 'packaging'}

    {'id': '97fc33e3-ed36-4460-b99c-d8e23cb1e339',
     'workspace_id': 9,
     'sha': 'd3b93c9f280734106376e684792aa8b4285d527092fe87d89c74ec804f8e169e',
     'file_name': 'api_inference_orchestration.zip',
     'task_id': '6449f7fc-e123-476e-b872-b7854cc91a3b',
     'owner_id': '787453b1-fbd5-4b4e-90a5-8ccbe60778a9',
     'status': 'packaging'}

    {'id': '97fc33e3-ed36-4460-b99c-d8e23cb1e339',
     'workspace_id': 9,
     'sha': 'd3b93c9f280734106376e684792aa8b4285d527092fe87d89c74ec804f8e169e',
     'file_name': 'api_inference_orchestration.zip',
     'task_id': '6449f7fc-e123-476e-b872-b7854cc91a3b',
     'owner_id': '787453b1-fbd5-4b4e-90a5-8ccbe60778a9',
     'status': 'packaging'}

    {'id': '97fc33e3-ed36-4460-b99c-d8e23cb1e339',
     'workspace_id': 9,
     'sha': 'd3b93c9f280734106376e684792aa8b4285d527092fe87d89c74ec804f8e169e',
     'file_name': 'api_inference_orchestration.zip',
     'task_id': '6449f7fc-e123-476e-b872-b7854cc91a3b',
     'owner_id': '787453b1-fbd5-4b4e-90a5-8ccbe60778a9',
     'status': 'packaging'}

    {'id': '97fc33e3-ed36-4460-b99c-d8e23cb1e339',
     'workspace_id': 9,
     'sha': 'd3b93c9f280734106376e684792aa8b4285d527092fe87d89c74ec804f8e169e',
     'file_name': 'api_inference_orchestration.zip',
     'task_id': '6449f7fc-e123-476e-b872-b7854cc91a3b',
     'owner_id': '787453b1-fbd5-4b4e-90a5-8ccbe60778a9',
     'status': 'packaging'}

    {'id': '97fc33e3-ed36-4460-b99c-d8e23cb1e339',
     'workspace_id': 9,
     'sha': 'd3b93c9f280734106376e684792aa8b4285d527092fe87d89c74ec804f8e169e',
     'file_name': 'api_inference_orchestration.zip',
     'task_id': '6449f7fc-e123-476e-b872-b7854cc91a3b',
     'owner_id': '787453b1-fbd5-4b4e-90a5-8ccbe60778a9',
     'status': 'packaging'}

    {'id': '97fc33e3-ed36-4460-b99c-d8e23cb1e339',
     'workspace_id': 9,
     'sha': 'd3b93c9f280734106376e684792aa8b4285d527092fe87d89c74ec804f8e169e',
     'file_name': 'api_inference_orchestration.zip',
     'task_id': '6449f7fc-e123-476e-b872-b7854cc91a3b',
     'owner_id': '787453b1-fbd5-4b4e-90a5-8ccbe60778a9',
     'status': 'packaging'}

    {'id': '97fc33e3-ed36-4460-b99c-d8e23cb1e339',
     'workspace_id': 9,
     'sha': 'd3b93c9f280734106376e684792aa8b4285d527092fe87d89c74ec804f8e169e',
     'file_name': 'api_inference_orchestration.zip',
     'task_id': '6449f7fc-e123-476e-b872-b7854cc91a3b',
     'owner_id': '787453b1-fbd5-4b4e-90a5-8ccbe60778a9',
     'status': 'ready'}

## Task Management Tutorial

Once an Orchestration has the status `ready`, it can be run as a task.  Tasks have three run options.

| Type | SDK Call |  How triggered |
|---|---|:---|
| Once       | `orchestration.run_once(name, json_args, timeout)` | Task runs once and exits.| Single batch, experimentation. |
| Scheduled  | `orchestration.run_scheduled(name, schedule, timeout, json_args)` | User provides schedule. Task runs exits whenever schedule dictates. | Recurrent batch. |

### Run Task Once via API

We'll do both a Run Once task and generate our Run Once Task from our orchestration.  Orchestrations are started as a run once task with the following request:

`v1/api//v1/api/orchestration/task/run_once`

The following parameters are required.

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
    "json": {}
}

import datetime
task_start = datetime.datetime.now()

url=f"{APIURL}/v1/api/task/run_once"

response=requests.post(url, headers=headers, json=data).json()
display(response)
task_id = response['id']
```

    {'id': '0830ddd0-cdec-44a4-8329-4ee50929cde3'}

### Task Status via API

The list of tasks in the Wallaroo instance is retrieves through the Wallaroo MLOPs API request:

The following parameters are required.

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
    time.sleep(5)
```

    {'name': 'api run once task',
     'id': '0830ddd0-cdec-44a4-8329-4ee50929cde3',
     'image': 'proxy.replicated.com/proxy/wallaroo/ghcr.io/wallaroolabs/arbex-orch-deploy',
     'image_tag': 'v2023.2.0-main-3233',
     'bind_secrets': ['minio'],
     'extra_env_vars': {'MINIO_URL': 'http://minio.wallaroo.svc.cluster.local:9000',
      'ORCH_OWNER_ID': '787453b1-fbd5-4b4e-90a5-8ccbe60778a9',
      'ORCH_SHA': 'd3b93c9f280734106376e684792aa8b4285d527092fe87d89c74ec804f8e169e',
      'TASK_ID': '0830ddd0-cdec-44a4-8329-4ee50929cde3'},
     'auth_init': True,
     'workspace_id': 9,
     'flavor': 'exec_orch_oneshot',
     'reap_threshold_secs': 900,
     'exec_type': 'job',
     'status': 'pending',
     'input_data': {},
     'killed': False,
     'created_at': '2023-05-17T16:27:34.024513+00:00',
     'updated_at': '2023-05-17T16:27:34.108342+00:00',
     'last_runs': [{'run_id': 'cd3bde62-0f0c-433b-94bb-658ce44b09c9',
       'status': 'running',
       'created_at': '2023-05-17T16:27:35.825154+00:00'}]}

    {'name': 'api run once task',
     'id': '0830ddd0-cdec-44a4-8329-4ee50929cde3',
     'image': 'proxy.replicated.com/proxy/wallaroo/ghcr.io/wallaroolabs/arbex-orch-deploy',
     'image_tag': 'v2023.2.0-main-3233',
     'bind_secrets': ['minio'],
     'extra_env_vars': {'MINIO_URL': 'http://minio.wallaroo.svc.cluster.local:9000',
      'ORCH_OWNER_ID': '787453b1-fbd5-4b4e-90a5-8ccbe60778a9',
      'ORCH_SHA': 'd3b93c9f280734106376e684792aa8b4285d527092fe87d89c74ec804f8e169e',
      'TASK_ID': '0830ddd0-cdec-44a4-8329-4ee50929cde3'},
     'auth_init': True,
     'workspace_id': 9,
     'flavor': 'exec_orch_oneshot',
     'reap_threshold_secs': 900,
     'exec_type': 'job',
     'status': 'started',
     'input_data': {},
     'killed': False,
     'created_at': '2023-05-17T16:27:34.024513+00:00',
     'updated_at': '2023-05-17T16:27:39.534322+00:00',
     'last_runs': [{'run_id': 'cd3bde62-0f0c-433b-94bb-658ce44b09c9',
       'status': 'running',
       'created_at': '2023-05-17T16:27:35.825154+00:00'}]}

### Task Results

We can view the inferences from our logs and verify that new entries were added from our task.  In our case, we'll assume the task once started takes about 1 minute to run (deploy the pipeline, run the inference, undeploy the pipeline).  We'll add in a wait of 1 minute, then display the logs during the time period the task was running.

```python
time.sleep(30)

task_end = datetime.datetime.now()
display(task_end)

pipeline.logs(start_datetime = task_start, end_datetime = task_end)
```

    datetime.datetime(2023, 5, 17, 10, 29, 10, 77147)

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
      <td>2023-05-17 16:27:40.460</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.7]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}

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
    "json": {}
}

import datetime
task_start = datetime.datetime.now()

url=f"{APIURL}/v1/api/task/run_scheduled"

response=requests.post(url, headers=headers, json=data).json()
display(response)
scheduled_task_id = response['id']
```

    {'id': 'e6ea2c20-71f0-4275-87a3-61b5cf561158'}

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
    time.sleep(5)
```

    {'name': 'scheduled api task',
     'id': 'e6ea2c20-71f0-4275-87a3-61b5cf561158',
     'image': 'proxy.replicated.com/proxy/wallaroo/ghcr.io/wallaroolabs/arbex-orch-deploy',
     'image_tag': 'v2023.2.0-main-3233',
     'bind_secrets': ['minio'],
     'extra_env_vars': {'MINIO_URL': 'http://minio.wallaroo.svc.cluster.local:9000',
      'ORCH_OWNER_ID': '787453b1-fbd5-4b4e-90a5-8ccbe60778a9',
      'ORCH_SHA': 'd3b93c9f280734106376e684792aa8b4285d527092fe87d89c74ec804f8e169e',
      'TASK_ID': 'e6ea2c20-71f0-4275-87a3-61b5cf561158'},
     'auth_init': True,
     'workspace_id': 9,
     'schedule': '*/1 * * * *',
     'reap_threshold_secs': 900,
     'status': 'started',
     'input_data': {},
     'killed': False,
     'created_at': '2023-05-17T16:30:12.401021+00:00',
     'updated_at': '2023-05-17T16:30:12.943188+00:00',
     'last_runs': []}

```python
# show the updated results
# wait 7 minutes to make sure the task started
time.sleep(420)

task_end = datetime.datetime.now()
display(task_end)

pipeline.logs(start_datetime = task_start, end_datetime = task_end)
```

    datetime.datetime(2023, 5, 17, 10, 31, 51, 573059)

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
      <td>2023-05-17 16:31:15.749</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.7]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}

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

## Close Resources

With the tutorial complete, we'll verify the pipeline is closed so the resources are assigned back to the Wallaroo instance.

```python
pipeline.undeploy()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>apipipeline</td></tr><tr><th>created</th> <td>2023-05-17 16:15:51.394406+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 16:16:10.837047+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>a5a7c47a-c42d-41eb-88e1-615608e0a49b, 0f774337-9a79-46f2-8a33-9580b389383e, b7b14b1a-2e2c-4a2e-b8c7-6f1640b9bb45</td></tr><tr><th>steps</th> <td>apiorchestrationmodel</td></tr></table>
{{</table>}}

