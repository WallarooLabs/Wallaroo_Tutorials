This can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/blob/20230314_2023.2_updates/pipeline-orchestrators/connection_api_bigquery_tutorial).

## Wallaroo Connection with BigQuery House Price Model Tutorial

This tutorial provides a quick set of methods and examples regarding Wallaroo Connections.  For full details, see the Wallaroo Documentation site.

Wallaroo provides Data Connections to organizations with a method of creating and managing automated tasks that can either be run on demand or a regular schedule.

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

import requests

# used for unique connection names

import string
import random

suffix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))
```

```python
wallaroo.__version__
```

    '2023.2.0rc3'

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

wallarooPrefix = "YOUR PREFIX"
wallarooSuffix = "YOUR SUFFIX"
APIURL = f"https://{wallarooPrefix}.api.{wallarooSuffix}"

# Setting variables for later steps

workspace_name = 'bigqueryapiworkspace'
pipeline_name = 'bigqueryapipipeline'
model_name = 'bigqueryapimodel'
model_file_name = './models/rf_model.onnx'

bigquery_connection_input_name = f"bigqueryhouseapiinput{suffix}"
bigquery_connection_input_type = "BIGQUERY"
bigquery_connection_input_argument = json.load(open('./bigquery_service_account_input_key.json'))

bigquery_connection_output_name = f"bigqueryhouseapioutputs{suffix}"
bigquery_connection_output_type = "BIGQUERY"
bigquery_connection_output_argument = json.load(open('./bigquery_service_account_output_key.json'))
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
<table><tr><th>name</th> <td>bigqueryapipipeline</td></tr><tr><th>created</th> <td>2023-05-17 16:06:45.400005+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 16:06:48.150211+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>5e78de4f-51f1-43ce-8c2c-d724a05f856a, d2f03a9c-dfbb-4a03-a014-e70aa80902e3</td></tr><tr><th>steps</th> <td>bigqueryapimodel</td></tr></table>
{{</table>}}

```python
#deploy the pipeline
pipeline.deploy()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>bigqueryapipipeline</td></tr><tr><th>created</th> <td>2023-05-17 16:06:45.400005+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 16:11:23.317215+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>153a11e5-7968-450c-aef5-d1be17c6b173, 5e78de4f-51f1-43ce-8c2c-d724a05f856a, d2f03a9c-dfbb-4a03-a014-e70aa80902e3</td></tr><tr><th>steps</th> <td>bigqueryapimodel</td></tr></table>
{{</table>}}

## Connection Management via the Wallaroo MLOps API

The following steps will demonstration using the Wallaroo MLOps API to:

* Create the BigQuery connections
* Add the connections to the targeted workspace
* Use the connections for inference requests and uploading the results to a BigQuery dataset table.

### Create Connections via API

We will create the data source connection via the Wallaroo api request:

`/v1/api/connections/create`

This takes the following parameters:

* **name** (*String* *Required*): The name of the connection.
* **type** (*String* *Required*): The user defined type of connection.
* **details** (*String* *Required*):  User defined configuration details for the data connection.  These can be `{'username':'dataperson', 'password':'datapassword', 'port': 3339}`, or `{'token':'abcde123==', 'host':'example.com', 'port:1234'}`, or other user defined combinations.

* **IMPORTANT NOTE**:  Data connections names **must** be unique.  Attempting to create a data connection with the same `name` as an existing data connection will result in an error.

```python
# retrieve the authorization token
headers = wl.auth.auth_header()

url = f"{APIURL}/v1/api/connections/create"

# input connection
data = {
    'name': bigquery_connection_input_name,
    'type' : bigquery_connection_input_type,
    'details': bigquery_connection_input_argument
}

response=requests.post(url, headers=headers, json=data).json()
display(response)
# saved for later steps
connection_input_id = response['id']
```

    {'id': '839779d7-d4b3-4e1c-953a-2f5ef55f1bb2'}

```python
# retrieve the authorization token
headers = wl.auth.auth_header()

url = f"{APIURL}/v1/api/connections/create"

# output connection
data = {
    'name': bigquery_connection_output_name,
    'type' : bigquery_connection_output_type,
    'details': bigquery_connection_output_argument
}

response=requests.post(url, headers=headers, json=data).json()
display(response)
# saved for later steps
connection_output_id = response['id']
```

    {'id': 'ef0c62e7-e59f-4c43-bdff-05ed0976cffa'}

### Add Connections to Workspace via API

The connections will be added to the sample workspace with the MLOps API request:

`/v1/api/connections/add_to_workspace`

This takes the following parameters:

* **workspace_id** (*String* *Required*): The name of the connection.
* **connection_id** (*String* *Required*): The UUID connection ID

```python
# retrieve the authorization token
headers = wl.auth.auth_header()

url = f"{APIURL}/v1/api/connections/add_to_workspace"

data = {
    'workspace_id': workspace_id,
    'connection_id': connection_input_id
}

response=requests.post(url, headers=headers, json=data)
display(response.json())

data = {
    'workspace_id': workspace_id,
    'connection_id': connection_output_id
}

response=requests.post(url, headers=headers, json=data)
display(response.json())
```

    {'id': '3877da9d-e450-4b9a-85dd-7f68127b1313'}

    {'id': '15f247ab-833f-468a-b7c5-9b50831052e6'}

### Connect to Google BigQuery

With our connections set, we'll now use them for an inference request through the following steps:

1. Retrieve the input data from a BigQuery request from the input connection details.
1. Perform the inference.
1. Upload the inference results into another BigQuery table from the output connection details.

### Create Google Credentials

From our BigQuery request, we'll create the credentials for our BigQuery connection.

We will use the MLOps API call:

`/v1/api/connections/get`

to retrieve the connection. This request takes the following parameters:

* **name** (*String* *Required*): The name of the connection.

```python
# get the connection input details

# retrieve the authorization token
headers = wl.auth.auth_header()

url = f"{APIURL}/v1/api/connections/get"

data = {
    'name': bigquery_connection_input_name
}

connection_input_details=requests.post(url, headers=headers, json=data).json()['details']
```

```python
# get the connection output details

# retrieve the authorization token
headers = wl.auth.auth_header()

url = f"{APIURL}/v1/api/connections/get"

data = {
    'name': bigquery_connection_output_name
}

connection_output_details=requests.post(url, headers=headers, json=data).json()['details']
```

```python
# Set the bigquery credentials

bigquery_input_credentials = service_account.Credentials.from_service_account_info(
    connection_input_details)

bigquery_output_credentials = service_account.Credentials.from_service_account_info(
    connection_output_details)
```

### Connect to Google BigQuery

We can now generate a client from our connection details, specifying the project that was included in the `big_query_connection` details.

```python
bigqueryinputclient = bigquery.Client(
    credentials=bigquery_input_credentials, 
    project=connection_input_details['project_id']
)
bigqueryoutputclient = bigquery.Client(
    credentials=bigquery_output_credentials, 
    project=connection_output_details['project_id']
)
```

### Query Data

Now we'll create our query and retrieve information from out dataset and table as defined in the file `bigquery_service_account_key.json`.  The table is expected to be in the format of the file `./data/xtest-1k.df.json`.

```python
inference_dataframe_input = bigqueryinputclient.query(
        f"""
        SELECT tensor
        FROM {connection_input_details['dataset']}.{connection_input_details['table']}"""
    ).to_dataframe()
```

```python
inference_dataframe_input.head(5)
```

{{<table "table table-striped table-bordered" >}}
<table>
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
{{</table>}}

### Sample Inference

With our data retrieved, we'll perform an inference and display the results.

```python
result = pipeline.infer(inference_dataframe_input)
display(result.head(5))
```

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
      <td>2023-05-17 16:12:53.970</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.75]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-05-17 16:12:53.970</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[615094.56]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-05-17 16:12:53.970</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
      <td>[448627.72]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-05-17 16:12:53.970</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[758714.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-05-17 16:12:53.970</td>
      <td>[3.0, 1.75, 2200.0, 11520.0, 1.0, 0.0, 0.0, 4.0, 7.0, 2200.0, 0.0, 47.7659, -122.341, 1690.0, 8038.0, 62.0, 0.0, 0.0]</td>
      <td>[513264.7]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}

### Upload the Results

With the query complete, we'll upload the results back to the BigQuery dataset.

```python
output_table = bigqueryoutputclient.get_table(f"{connection_output_details['dataset']}.{connection_output_details['table']}")

bigqueryoutputclient.insert_rows_from_dataframe(
    output_table, 
    dataframe=result.rename(columns={"in.tensor":"in_tensor", "out.variable":"out_variable"})
)
```

    [[], []]

### Verify the Upload

We can verify the upload by requesting the last few rows of the output table.

```python
task_inference_results = bigqueryoutputclient.query(
        f"""
        SELECT *
        FROM {connection_output_details['dataset']}.{connection_output_details['table']}
        ORDER BY time DESC
        LIMIT 5
        """
    ).to_dataframe()

display(task_inference_results)
```

{{<table "table table-striped table-bordered" >}}
<table>
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
      <td>2023-05-17 16:12:53.970</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[758714.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-05-17 16:12:53.970</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.75]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-05-17 16:12:53.970</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[615094.56]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-05-17 16:12:53.970</td>
      <td>[3.0, 1.75, 2200.0, 11520.0, 1.0, 0.0, 0.0, 4.0, 7.0, 2200.0, 0.0, 47.7659, -122.341, 1690.0, 8038.0, 62.0, 0.0, 0.0]</td>
      <td>[513264.7]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-05-17 16:12:53.970</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
      <td>[448627.72]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}

## Cleanup

With the tutorial complete, we can undeploy the pipeline and return the resources back to the Wallaroo instance.

```python
pipeline.undeploy()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>bigqueryapipipeline</td></tr><tr><th>created</th> <td>2023-05-17 16:06:45.400005+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 16:11:23.317215+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>153a11e5-7968-450c-aef5-d1be17c6b173, 5e78de4f-51f1-43ce-8c2c-d724a05f856a, d2f03a9c-dfbb-4a03-a014-e70aa80902e3</td></tr><tr><th>steps</th> <td>bigqueryapimodel</td></tr></table>
{{</table>}}

