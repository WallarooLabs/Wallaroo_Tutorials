This can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/workload-orchestrations/orchestration_sdk_bigquery_houseprice_tutorial).

## Wallaroo ML Workload Orchestration House Price Model Tutorial

This tutorial provides a quick set of methods and examples regarding Wallaroo Connections and Wallaroo ML Workload Orchestration.  For full details, see the Wallaroo Documentation site.

Wallaroo provides Data Connections and ML Workload Orchestrations to provide organizations with a method of creating and managing automated tasks that can either be run on demand or a regular schedule.

## Definitions

* **Orchestration**: A set of instructions written as a python script with a requirements library.  Orchestrations are uploaded to the Wallaroo instance as a .zip file.
* **Task**: An implementation of an orchestration.  Tasks are run either once when requested, on a repeating schedule, or as a service.
* **Connection**: Definitions set by MLOps engineers that are used by other Wallaroo users for connection information to a data source.  Usually paired with orchestrations.

This tutorial will focus on using Google BigQuery as the data source.

## Tutorial Goals

The tutorial will demonstrate the following:

1. Create a Wallaroo connection to retrieving information from a Google BigQuery source table.
1. Create a Wallaroo connection to store inference results into a Google BigQuery destination table.
1. Upload Wallaroo ML Workload Orchestration that supports BigQuery connections with the connection details.
1. Run the orchestration once as a Run Once Task and verify that the inference request succeeded and the inference results were saved to the external data store.
1. Schedule the orchestration as a Scheduled Task and verify that the inference request succeeded and the inference results were saved to the external data store.

## Prerequisites

* An installed Wallaroo instance.
* The following Python libraries installed.  These are included by default in a Wallaroo instance's JupyterHub service.
  * `os`
  * [`wallaroo`](https://pypi.org/project/wallaroo/): The Wallaroo SDK. Included with the Wallaroo JupyterHub service by default.
  * [`pandas`](https://pypi.org/project/pandas/): Pandas, mainly used for Pandas DataFrame
  * [`pyarrow`](https://pypi.org/project/pyarrow/): PyArrow for Apache Arrow support
* The following Python libraries.  These are **not** included in a Wallaroo instance's JupyterHub service.
  * [`google-cloud-bigquery`](https://pypi.org/project/google-cloud-bigquery/): Specifically for its support for Google BigQuery.
  * [`google-auth`](https://pypi.org/project/google-auth/): Used to authenticate for bigquery.
  * [`db-dtypes`](https://pypi.org/project/db-dtypes/): Converts the BigQuery results to Apache Arrow table or pandas DataFrame.

## Tutorial Resources

* Models:
  * `models/rf_model.onnx`: A model that predicts house price values.
* Data:
  * `data/xtest-1.df.json` and `data/xtest-1k.df.json`:  DataFrame JSON inference inputs with 1 input and 1,000 inputs.
  * `data/xtest-1k.arrow`:  Apache Arrow inference inputs with 1 input and 1,000 inputs.
  * Sample inference inputs in `CSV` that can be imported into Google BigQuery.
    * `data/xtest-1k.df.json`: Random sample housing prices.
    * `data/smallinputs.df.json`: Sample housing prices that return results lower than $1.5 million.
    * `data/biginputs.df.json`: Sample housing prices that return results higher than $1.5 million.
  * SQL queries to create the inputs/outputs tables with schema.
    * `./resources/create_inputs_table.sql`: Inputs table with schema.
    * `./resources/create_outputs_table.sql`: Outputs table with schema.
    * `./resources/housrpricesga_inputs.avro`: Avro container of inputs table.

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
import json

# for Big Query connections
from google.cloud import bigquery
from google.oauth2 import service_account
import db_dtypes

# used for unique connection names

import string
import random

suffix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))
```

```python
wallaroo.__version__
```

    '2023.2.0+dfca0605e'

### Connect to the Wallaroo Instance

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
# Login through local Wallaroo instance

wl = wallaroo.Client()
```

## Variable Declaration

The following variables will be used for our big query testing.  

We'll use two connections:

* bigquery_input_connection: The connection that will draw inference input data from a BigQuery table.
* bigquery_output_connection:  The connection that will upload inference results into a BigQuery table.

Not that for the connection arguments, we'll retrieve the information from the files `./bigquery_service_account_input_key.json` and `./bigquery_service_account_output_key.json` that include the  [service account key file(SAK)](https://cloud.google.com/bigquery/docs/authentication/service-account-file) information, as well as the dataset and table used.

| Field | Included in SAK | 
|---|---|
| type | √ | 
| project_id | √ |
| private_key_id | √ |
| private_key | √ |
| client_email | √ |
| auth_uri | √ |
| token_uri | √ |
| auth_provider_x509_cert_url | √ |
| client_x509_cert_url | √ |
| database | 🚫 |
| table | 🚫 |

```python
# Setting variables for later steps

workspace_name = f'bigqueryworkspace{suffix}'
pipeline_name = f'bigquerypipeline{suffix}'
model_name = f'bigquerymodel{suffix}'
model_file_name = './models/rf_model.onnx'

bigquery_connection_input_name = 'bigqueryhouseinputs{suffix}'
bigquery_connection_input_type = "BIGQUERY"
bigquery_connection_input_argument = json.load(open("./bigquery_service_account_input_key.json"))

bigquery_connection_output_name = 'bigqueryhouseoutputs{suffix}'
bigquery_connection_output_type = "BIGQUERY"
bigquery_connection_output_argument = json.load(open("./bigquery_service_account_output_key.json"))
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

<table><tr><th>name</th> <td>bigquerypipelinechbp</td></tr><tr><th>created</th> <td>2023-05-23 15:17:30.123708+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-23 15:17:30.123708+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>fefa4278-04e0-4d1a-8d5b-9cc9c5275832</td></tr><tr><th>steps</th> <td></td></tr></table>

```python
#deploy the pipeline to set the pipeline steps
pipeline.deploy()
```

    Waiting for deployment - this will take up to 45s ............ ok

<table><tr><th>name</th> <td>bigquerypipelinechbp</td></tr><tr><th>created</th> <td>2023-05-23 15:17:30.123708+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-23 15:17:30.428928+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>3dd0653a-12b0-4298-b91b-1e0b712716c5, fefa4278-04e0-4d1a-8d5b-9cc9c5275832</td></tr><tr><th>steps</th> <td>bigquerymodelchbp</td></tr></table>

## Create Connections

We will create the data source connection via the Wallaroo client command `create_connection`.

Connections are created with the Wallaroo client command [`create_connection`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-ml-workload-orchestration/#create-orchestration) with the following parameters.

| Parameter | Type | Description |
| --- | --- | ---|
| **name** | string (Required) | The name of the connection. This must be unique - **if submitting the name of an existing** connection it will return an error. |
| **type** | string (Required) | The user defined type of connection. |
| **details** | Dict (Required) | User defined configuration details for the data connection.  These can be `{'username':'dataperson', 'password':'datapassword', 'port': 3339}`, or `{'token':'abcde123==', 'host':'example.com', 'port:1234'}`, or other user defined combinations.  |

* **IMPORTANT NOTE**:  Data connections names **must** be unique.  Attempting to create a data connection with the same `name` as an existing data connection will result in an error.

```python
connection_input = wl.create_connection(bigquery_connection_input_name, bigquery_connection_input_type, bigquery_connection_input_argument)
connection_output = wl.create_connection(bigquery_connection_output_name, bigquery_connection_output_type, bigquery_connection_output_argument)
wl.list_connections()
```

<table><tr><th>name</th><th>connection type</th><th>details</th><th>created at</th><th>linked workspaces</th></tr><tr><td>bigqueryhouseinputs</td><td>BIGQUERY</td><td>*****</td><td>2023-05-23T14:35:26.896064+00:00</td><td>['bigqueryworkspace']</td></tr><tr><td>bigqueryhouseoutputs</td><td>BIGQUERY</td><td>*****</td><td>2023-05-23T14:35:26.932685+00:00</td><td>['bigqueryworkspace']</td></tr><tr><td>bigqueryhouseinputs-jcw</td><td>BIGQUERY</td><td>*****</td><td>2023-05-23T14:37:22.103147+00:00</td><td>['bigqueryworkspace-jcw']</td></tr><tr><td>bigqueryhouseoutputs-jcw</td><td>BIGQUERY</td><td>*****</td><td>2023-05-23T14:37:22.141179+00:00</td><td>['bigqueryworkspace-jcw']</td></tr><tr><td>bigqueryhouseinputs{suffix}</td><td>BIGQUERY</td><td>*****</td><td>2023-05-23T15:05:29.850628+00:00</td><td>['bigqueryworkspacekbcy']</td></tr><tr><td>bigqueryhouseoutputs{suffix}</td><td>BIGQUERY</td><td>*****</td><td>2023-05-23T15:05:30.298941+00:00</td><td>['bigqueryworkspacekbcy']</td></tr><tr><td>bigqueryforecastinputsrklr</td><td>BIGQUERY</td><td>*****</td><td>2023-05-23T15:05:58.206726+00:00</td><td>['bigquerystatsmodelworkspacerklr']</td></tr><tr><td>bigqueryforecastoutputsrklr</td><td>BIGQUERY</td><td>*****</td><td>2023-05-23T15:05:58.673528+00:00</td><td>['bigquerystatsmodelworkspacerklr']</td></tr></table>

### Get Connection by Name

The Wallaroo client method `get_connection(name)` retrieves the connection that matches the `name` parameter.  We'll retrieve our connection and store it as `inference_source_connection`.

```python
big_query_input_connection = wl.get_connection(name=bigquery_connection_input_name)
big_query_output_connection = wl.get_connection(name=bigquery_connection_output_name)
display(big_query_input_connection)
display(big_query_output_connection)
```

<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Name</td><td>bigqueryhouseinputs{suffix}</td>
  </tr>
  <tr>
    <td>Connection Type</td><td>BIGQUERY</td>
  </tr>
  <tr>
    <td>Details</td><td>*****</td>
  </tr>
  <tr>
    <td>Created At</td><td>2023-05-23T15:05:29.850628+00:00</td>
  </tr>
  <tr>
    <td>Linked Workspaces</td><td>['bigqueryworkspacekbcy']</td>
  </tr>
</table>

<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Name</td><td>bigqueryhouseoutputs{suffix}</td>
  </tr>
  <tr>
    <td>Connection Type</td><td>BIGQUERY</td>
  </tr>
  <tr>
    <td>Details</td><td>*****</td>
  </tr>
  <tr>
    <td>Created At</td><td>2023-05-23T15:05:30.298941+00:00</td>
  </tr>
  <tr>
    <td>Linked Workspaces</td><td>['bigqueryworkspacekbcy']</td>
  </tr>
</table>

### Add Connection to Workspace

The method Workspace [`add_connection(connection_name)`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-dataconnections/#add-data-connection-to-workspace) adds a Data Connection to a workspace, and takes the following parameters.

| Parameter | Type | Description |
| --- | --- | ---|
| **name** | string (Required) | The name of the Data Connection |

We'll add both connections to our sample workspace, then list the connections available to the workspace to confirm.

```python
workspace.add_connection(bigquery_connection_input_name)
workspace.add_connection(bigquery_connection_output_name)

workspace.list_connections()
```

<table><tr><th>name</th><th>connection type</th><th>details</th><th>created at</th><th>linked workspaces</th></tr><tr><td>bigqueryhouseinputs{suffix}</td><td>BIGQUERY</td><td>*****</td><td>2023-05-23T15:05:29.850628+00:00</td><td>['bigqueryworkspacekbcy', 'bigqueryworkspacechbp']</td></tr><tr><td>bigqueryhouseoutputs{suffix}</td><td>BIGQUERY</td><td>*****</td><td>2023-05-23T15:05:30.298941+00:00</td><td>['bigqueryworkspacekbcy', 'bigqueryworkspacechbp']</td></tr></table>

## Big Query Connection Inference Example

We can test the BigQuery connection with a simple inference to our deployed pipeline.  We'll request the data, format the table into a pandas DataFrame, then submit it for an inference request.

### Create Google Credentials

From our BigQuery request, we'll create the credentials for our BigQuery connection.

```python
bigquery_input_credentials = service_account.Credentials.from_service_account_info(
    big_query_input_connection.details())

bigquery_output_credentials = service_account.Credentials.from_service_account_info(
    big_query_output_connection.details())
```

### Connect to Google BigQuery

We can now generate a client from our connection details, specifying the project that was included in the `big_query_connection` details.

```python
bigqueryinputclient = bigquery.Client(
    credentials=bigquery_input_credentials, 
    project=big_query_input_connection.details()['project_id']
)
bigqueryoutputclient = bigquery.Client(
    credentials=bigquery_output_credentials, 
    project=big_query_output_connection.details()['project_id']
)
```

### Query Data

Now we'll create our query and retrieve information from out dataset and table as defined in the file `bigquery_service_account_key.json`.  The table is expected to be in the format of the file `./data/xtest-1k.df.json`.

```python
inference_dataframe_input = bigqueryinputclient.query(
        f"""
        SELECT tensor
        FROM {big_query_input_connection.details()['dataset']}.{big_query_input_connection.details()['table']}"""
    ).to_dataframe()
```

```python
inference_dataframe_input.head(5)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[3.0, 1.75, 2200.0, 11520.0, 1.0, 0.0, 0.0, 4.0, 7.0, 2200.0, 0.0, 47.7659, -122.341, 1690.0, 8038.0, 62.0, 0.0, 0.0]</td>
    </tr>
  </tbody>
</table>

### Sample Inference

With our data retrieved, we'll perform an inference and display the results.

```python
result = pipeline.infer(inference_dataframe_input)
display(result.head(5))
```

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
      <td>2023-05-23 15:17:47.498</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.75]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-05-23 15:17:47.498</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[615094.56]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-05-23 15:17:47.498</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
      <td>[448627.72]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-05-23 15:17:47.498</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[758714.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-05-23 15:17:47.498</td>
      <td>[3.0, 1.75, 2200.0, 11520.0, 1.0, 0.0, 0.0, 4.0, 7.0, 2200.0, 0.0, 47.7659, -122.341, 1690.0, 8038.0, 62.0, 0.0, 0.0]</td>
      <td>[513264.7]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

### Upload the Results

With the query complete, we'll upload the results back to the BigQuery dataset.

```python
output_table = bigqueryoutputclient.get_table(f"{big_query_output_connection.details()['dataset']}.{big_query_output_connection.details()['table']}")

bigqueryoutputclient.insert_rows_from_dataframe(
    output_table, 
    dataframe=result.rename(columns={"in.tensor":"in_tensor", "out.variable":"out_variable"})
)
```

    [[], []]

## Wallaroo ML Workload Orchestration Example

With the pipeline deployed and our connections set, we will now generate our ML Workload Orchestration.  See the [Wallaroo ML Workload Orchestrations guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-ml-workload-orchestration/) for full details.

Orchestrations are uploaded to the Wallaroo instance as a ZIP file with the following requirements:

| Parameter | Type | Description |
|---|---|---|
| **User Code** | (*Required*) Python script as `.py` files | If `main.py` exists, then that will be used as the task entrypoint. Otherwise, the **first** `main.py` found in any subdirectory will be used as the entrypoint. |
| Python Library Requirements | (*Optional*) `requirements.txt` file in the [requirements file format](https://pip.pypa.io/en/stable/reference/requirements-file-format/).  A standard Python requirements.txt for any dependencies to be provided in the task environment. The Wallaroo SDK will already be present and **should not be included in the requirements.txt**. Multiple requirements.txt files are not allowed. |
| Other artifacts | &nbsp; | Other artifacts such as files, data, or code to support the orchestration.

For our example, our orchestration will:

1. Use the `bigquery_remote_inference` to open a connection to the input and output tables.
1. Deploy the pipeline.
1. Perform an inference with the input data.
1. Save the inference results to the output table.
1. Undeploy the pipeline.

This sample script is stored in `bigquery_remote_inference/main.py` with an `requirements.txt` file having the specific libraries for the Google BigQuery connection., and packaged into the orchestration as `./bigquery_remote_inference/bigquery_remote_inference.zip`.  We'll display the steps in uploading the orchestration to the Wallaroo instance.

Note that the orchestration assumes the pipeline is already deployed.

### Upload the Orchestration

Orchestrations are uploaded with the Wallaroo client `upload_orchestration(path)` method with the following parameters.

| Parameter | Type | Description |
| --- | --- | ---|
| **path** | string (Required) | The path to the ZIP file to be uploaded. |

Once uploaded, the deployment will be prepared and any requirements will be downloaded and installed.

For this example, the orchestration `./bigquery_remote_inference/bigquery_remote_inference.zip` will be uploaded and saved to the variable `orchestration`.  Then we will loop until the uploaded orchestration's `status` displays `ready`.

```python
orchestration = wl.upload_orchestration(path="./bigquery_remote_inference/bigquery_remote_inference.zip")

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

<table><tr><th>id</th><th>name</th><th>status</th><th>filename</th><th>sha</th><th>created at</th><th>updated at</th></tr><tr><td>78ea6222-c102-4eb1-9705-166233a12257</td><td>None</td><td>ready</td><td>bigquery_remote_inference.zip</td><td>582f33...a6957c</td><td>2023-23-May 15:17:48</td><td>2023-23-May 15:18:39</td></tr></table>

## Task Management Tutorial

Once an Orchestration has the status `ready`, it can be run as a task.  Tasks have three run options.

| Type | SDK Call |  How triggered |
|---|---|:---|
| Once       | `orchestration.run_once(name, json_args, timeout)` | Task runs once and exits.| Single batch, experimentation. |
| Scheduled  | `orchestration.run_scheduled(name, schedule, timeout, json_args)` | User provides schedule. Task runs exits whenever schedule dictates. | Recurrent batch. |

### Run Task Once

We'll do both a Run Once task and generate our Run Once Task from our orchestration.

Tasks are generated and run once with the Orchestration `run_once(name, json_args, timeout)` method.  Any arguments for the orchestration are passed in as a `Dict`.  If there are no arguments, then an empty set `{}` is passed.

We'll display the last 5 rows of our BigQuery output table, then start the task that will perform the same inference we did above.

```python
task_inference_results = bigqueryoutputclient.query(
        f"""
        SELECT *
        FROM {big_query_output_connection.details()['dataset']}.{big_query_output_connection.details()['table']}
        ORDER BY time DESC
        LIMIT 5
        """
    ).to_dataframe()

display(task_inference_results)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in_tensor</th>
      <th>out_variable</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-05-23 15:17:47.498</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[758714.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-05-23 15:17:47.498</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.75]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-05-23 15:17:47.498</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[615094.56]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-05-23 15:17:47.498</td>
      <td>[3.0, 1.75, 2200.0, 11520.0, 1.0, 0.0, 0.0, 4.0, 7.0, 2200.0, 0.0, 47.7659, -122.341, 1690.0, 8038.0, 62.0, 0.0, 0.0]</td>
      <td>[513264.7]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-05-23 15:17:47.498</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
      <td>[448627.72]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

```python
# Example: run once

import datetime
task_start = datetime.datetime.now()

task = orchestration.run_once(name="big query single run", json_args={})
task
```

<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>ID</td><td>59b20af0-4f2e-4371-9175-e0b69a94fb91</td>
  </tr>
  <tr>
    <td>Name</td><td>big query single run</td>
  </tr>
  <tr>
    <td>Last Run Status</td><td>unknown</td>
  </tr>
  <tr>
    <td>Type</td><td>Temporary Run</td>
  </tr>
  <tr>
    <td>Active</td><td>True</td>
  </tr>
  <tr>
    <td>Schedule</td><td>-</td>
  </tr>
  <tr>
    <td>Created At</td><td>2023-23-May 15:18:46</td>
  </tr>
  <tr>
    <td>Updated At</td><td>2023-23-May 15:18:46</td>
  </tr>
</table>

### Task Status

The list of tasks in the Wallaroo instance is retrieves through the Wallaroo Client `list_tasks()` method.  This returns an array list of the following.

| Parameter | Type | Description |
| --- | --- | ---|
| **id** | string | The UUID identifier for the task. |
| **last run status** | string | The last reported status the task.  Values are: <br><ul><li>`pending`: The task has not been started.</li><li>`started`: The task has been scheduled to execute.</li><li>`pending_kill`: The task kill command has been issued and the task is scheduled to be stopped.</li></ul> |
| **type** | string | The type of the task.  Values are: <br><ul><li>`Temporary Run`: The task runs once then stop.</li><li>`Scheduled Run`: The task repeats on a `cron` like schedule.</li></ul> |
| **schedule** | string | The schedule for the task.  If a run once task, the schedule will be `-`.|
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

```python
wl.list_tasks()
```

<table><tr><th>id</th><th>name</th><th>last run status</th><th>type</th><th>active</th><th>schedule</th><th>created at</th><th>updated at</th></tr><tr><td>59b20af0-4f2e-4371-9175-e0b69a94fb91</td><td>big query single run</td><td>failure</td><td>Temporary Run</td><td>True</td><td>-</td><td>2023-23-May 15:18:46</td><td>2023-23-May 15:18:51</td></tr></table>

### Task Results

We can view the inferences from our logs and verify that new entries were added from our task.  We'll query the last 5 rows of our inference output table after a wait of 60 seconds.

```python
time.sleep(60)

task_inference_results = bigqueryoutputclient.query(
        f"""
        SELECT *
        FROM {big_query_output_connection.details()['dataset']}.{big_query_output_connection.details()['table']}
        ORDER BY time DESC LIMIT 5"""
    ).to_dataframe()

display(task_inference_results)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in_tensor</th>
      <th>out_variable</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-05-23 15:17:47.498</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[758714.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-05-23 15:17:47.498</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.75]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-05-23 15:17:47.498</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[615094.56]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-05-23 15:17:47.498</td>
      <td>[3.0, 1.75, 2200.0, 11520.0, 1.0, 0.0, 0.0, 4.0, 7.0, 2200.0, 0.0, 47.7659, -122.341, 1690.0, 8038.0, 62.0, 0.0, 0.0]</td>
      <td>[513264.7]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-05-23 15:17:47.498</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
      <td>[448627.72]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

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
task_inference_results = bigqueryoutputclient.query(
        f"""
        SELECT *
        FROM {big_query_output_connection.details()['dataset']}.{big_query_output_connection.details()['table']}
        ORDER BY time DESC LIMIT 5"""
    ).to_dataframe()

display(task_inference_results.tail(5))

scheduled_task = orchestration.run_scheduled(name="simple_inference_schedule", schedule="*/5 * * * *", timeout=120, json_args={})
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in_tensor</th>
      <th>out_variable</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-05-23 15:17:47.498</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[758714.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-05-23 15:17:47.498</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.75]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-05-23 15:17:47.498</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[615094.56]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-05-23 15:17:47.498</td>
      <td>[3.0, 1.75, 2200.0, 11520.0, 1.0, 0.0, 0.0, 4.0, 7.0, 2200.0, 0.0, 47.7659, -122.341, 1690.0, 8038.0, 62.0, 0.0, 0.0]</td>
      <td>[513264.7]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-05-23 15:17:47.498</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
      <td>[448627.72]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

```python
while scheduled_task.status() != "started":
    display(scheduled_task.status())
    time.sleep(5)
```

    'pending'

```python
#wait 420 seconds to give the scheduled event time to finish
time.sleep(420)
task_inference_results = bigqueryoutputclient.query(
        f"""
        SELECT *
        FROM {big_query_output_connection.details()['dataset']}.{big_query_output_connection.details()['table']}
        ORDER BY time DESC LIMIT 5"""
    ).to_dataframe()

display(task_inference_results.tail(5))
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in_tensor</th>
      <th>out_variable</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-05-23 15:17:47.498</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[758714.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-05-23 15:17:47.498</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.75]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-05-23 15:17:47.498</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[615094.56]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-05-23 15:17:47.498</td>
      <td>[3.0, 1.75, 2200.0, 11520.0, 1.0, 0.0, 0.0, 4.0, 7.0, 2200.0, 0.0, 47.7659, -122.341, 1690.0, 8038.0, 62.0, 0.0, 0.0]</td>
      <td>[513264.7]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-05-23 15:17:47.498</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
      <td>[448627.72]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

## Kill Task

With our testing complete, we will kill the scheduled task so it will not run again.  First we'll show all the tasks to verify that our task is there, then issue it the kill command.

```python
scheduled_task.kill()
```

    <ArbexStatus.PENDING_KILL: 'pending_kill'>

## Cleanup

With the tutorial complete, we can undeploy the pipeline and return the resources back to the Wallaroo instance.

```python
pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s ..................................... ok

<table><tr><th>name</th> <td>bigquerypipelinechbp</td></tr><tr><th>created</th> <td>2023-05-23 15:17:30.123708+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-23 15:17:30.428928+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>3dd0653a-12b0-4298-b91b-1e0b712716c5, fefa4278-04e0-4d1a-8d5b-9cc9c5275832</td></tr><tr><th>steps</th> <td>bigquerymodelchbp</td></tr></table>

