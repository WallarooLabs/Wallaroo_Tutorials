This can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/workload-orchestrations/orchestration_sdk_bigquery_statsmodel_tutorial).

## Wallaroo ML Workload Orchestrations with Statsmodel Tutorial

This tutorial provides a quick set of methods and examples regarding Wallaroo Connections and Wallaroo ML Workload Orchestration.  For full details, see the Wallaroo Documentation site.

Wallaroo provides Data Connections and ML Workload Orchestrations to provide organizations with a method of creating and managing automated tasks that can either be run on demand or a regular schedule.

## Definitions

* **Orchestration**: A set of instructions written as a python script with a requirements library.  Orchestrations are uploaded to the Wallaroo instance as a .zip file.
* **Task**: An implementation of an orchestration.  Tasks are run either once when requested, on a repeating schedule, or as a service.
* **Connection**: Definitions set by MLOps engineers that are used by other Wallaroo users for connection information to a data source.  Usually paired with orchestrations.

This tutorial will focus on using Google BigQuery as the data source for supplying the inference data to perform inferences through a Statsmodel ML model.

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
  * `models/bike_day_model.pkl`: The statsmodel model predicts how many bikes will be rented on each of the next 7 days, based on the previous 7 days' bike rentals, temperature, and wind speed.  This model only accepts shapes (7,4) - 7 rows (representing the last 7 days) and 4 columns (representing the fields `temp`, `holiday`, `workingday`, `windspeed`).  Additional files to support this example are:
  * `infer.py`: The inference script that is part of the `statsmodel`.
* Data:
  * `data/day.csv`: Data used to train the sample `statsmodel` example.
  * `data/bike_day_eval.json`: Data used for inferences.  This will be transferred to a BigQuery table as shown in the demonstration.
* Resources:
  * `resources/bigquery_service_account_input_key.json`: Example service key to authenticate to a Google BigQuery project with the dataset and table used for the inference data.
  * `resources/bigquery_service_account_output_key.json`: Example service key to authenticate to a Google BigQuery project with the dataset and table used for the inference result data.
  * `resources/statsmodel_forecast_inputs.sql`: SQL script to create the inputs table schema, populated with the values from `./data/day.csv`.
  * `resources/statsmodel_forecast_outputs.sql`: SQL script to create the outputs table schema.

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

import string
import random

suffix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))
```

```python
wallaroo.__version__
```

### Connect to the Wallaroo Instance

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  If logging in externally, update the `wallarooPrefix` and `wallarooSuffix` variables with the proper DNS information.  For more information on Wallaroo DNS settings, see the [Wallaroo DNS Integration Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-dns-guide/).

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

workspace_name = f'bigquerystatsmodelworkspace{suffix}'
pipeline_name = f'bigquerystatsmodelpipeline{suffix}'
model_name = f'bigquerystatsmodelmodel{suffix}'
model_file_name = './models/bike_day_model.pkl'

bigquery_connection_input_name = f'bigqueryforecastinputs{suffix}'
bigquery_connection_input_type = "BIGQUERY"
bigquery_connection_input_argument = json.load(open('./resources/bigquery_service_account_input_key.json.example'))

bigquery_connection_output_name = f'bigqueryforecastoutputs{suffix}'
bigquery_connection_output_type = "BIGQUERY"
bigquery_connection_output_argument = json.load(open('./resources/bigquery_service_account_output_key.json.example'))
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

bike_day_model = wl.upload_model(model_name, model_file_name).configure(runtime="python")

# Add the model as a pipeline step

pipeline.add_model_step(bike_day_model)
```

```python
#deploy the pipeline
pipeline.deploy()
```

### Sample Inferences

We'll perform some quick sample inferences with the local file data to verity the pipeline deployed and is ready for inferences.  Once done, we'll undeploy the pipeline.

```python
## perform inferences

results = pipeline.infer_from_file('./data/bike_day_eval.json', data_format="custom-json")
print(results)

```

## Create Connections

We will create the data source connection via the Wallaroo client command `create_connection`.

Connections are created with the Wallaroo client command [`create_connection`](https://staging.docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-ml-workload-orchestration/#create-orchestration) with the following parameters.

| Parameter | Type | Description |
| --- | --- | ---|
| **name** | string (Required) | The name of the connection. This must be unique - **if submitting the name of an existing** connection it will return an error. |
| **type** | string (Required) | The user defined type of connection. |
| **details** | Dict (Required) | User defined configuration details for the data connection.  These can be `{'username':'dataperson', 'password':'datapassword', 'port': 3339}`, or `{'token':'abcde123==', 'host':'example.com', 'port:1234'}`, or other user defined combinations.  |

* **IMPORTANT NOTE**:  Data connections names **must** be unique.  Attempting to create a data connection with the same `name` as an existing data connection will result in an error.

See the `statsmodel_forecast_inputs` and `statsmodel_forecast_outputs` details listed above for the table schema used for our example.

```python
connection_input = get_connection(bigquery_connection_input_name, bigquery_connection_input_type, bigquery_connection_input_argument)
connection_output = get_connection(bigquery_connection_output_name, bigquery_connection_output_type, bigquery_connection_output_argument)

display(connection_input)
display(connection_output)
```

{{<table "table table-striped table-bordered" >}}
<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Name</td><td>bigqueryforecastinputssxee</td>
  </tr>
  <tr>
    <td>Connection Type</td><td>BIGQUERY</td>
  </tr>
  <tr>
    <td>Details</td><td>*****</td>
  </tr>
  <tr>
    <td>Created At</td><td>2023-05-19T22:31:45.810930+00:00</td>
  </tr>
  <tr>
    <td>Linked Workspaces</td><td>['bigquerystatsmodelworkspacesxee']</td>
  </tr>
</table>
{{</table>}}

{{<table "table table-striped table-bordered" >}}
<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Name</td><td>bigqueryforecastoutputssxee</td>
  </tr>
  <tr>
    <td>Connection Type</td><td>BIGQUERY</td>
  </tr>
  <tr>
    <td>Details</td><td>*****</td>
  </tr>
  <tr>
    <td>Created At</td><td>2023-05-19T22:31:46.438522+00:00</td>
  </tr>
  <tr>
    <td>Linked Workspaces</td><td>['bigquerystatsmodelworkspacesxee']</td>
  </tr>
</table>
{{</table>}}

### Get Connection by Name

The Wallaroo client method `get_connection(name)` retrieves the connection that matches the `name` parameter.  We'll retrieve our connection and store it as `inference_source_connection`.

```python
big_query_input_connection = wl.get_connection(name=bigquery_connection_input_name)
big_query_output_connection = wl.get_connection(name=bigquery_connection_output_name)
display(big_query_input_connection)
display(big_query_output_connection)
```

{{<table "table table-striped table-bordered" >}}
<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Name</td><td>bigqueryforecastinputssxee</td>
  </tr>
  <tr>
    <td>Connection Type</td><td>BIGQUERY</td>
  </tr>
  <tr>
    <td>Details</td><td>*****</td>
  </tr>
  <tr>
    <td>Created At</td><td>2023-05-19T22:31:45.810930+00:00</td>
  </tr>
  <tr>
    <td>Linked Workspaces</td><td>['bigquerystatsmodelworkspacesxee']</td>
  </tr>
</table>
{{</table>}}

{{<table "table table-striped table-bordered" >}}
<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Name</td><td>bigqueryforecastoutputssxee</td>
  </tr>
  <tr>
    <td>Connection Type</td><td>BIGQUERY</td>
  </tr>
  <tr>
    <td>Details</td><td>*****</td>
  </tr>
  <tr>
    <td>Created At</td><td>2023-05-19T22:31:46.438522+00:00</td>
  </tr>
  <tr>
    <td>Linked Workspaces</td><td>['bigquerystatsmodelworkspacesxee']</td>
  </tr>
</table>
{{</table>}}

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

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th><th>connection type</th><th>details</th><th>created at</th><th>linked workspaces</th></tr><tr><td>bigqueryforecastinputssxee</td><td>BIGQUERY</td><td>*****</td><td>2023-05-19T22:31:45.810930+00:00</td><td>['bigquerystatsmodelworkspacesxee']</td></tr><tr><td>bigqueryforecastoutputssxee</td><td>BIGQUERY</td><td>*****</td><td>2023-05-19T22:31:46.438522+00:00</td><td>['bigquerystatsmodelworkspacesxee']</td></tr></table>
{{</table>}}

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

Now we'll create our query and retrieve information from out dataset and table as defined in the file `bigquery_service_account_key.json`.

We'll grab the last 7 days of data - with every record assumed to be one day - and use that for our inference request.

```python
inference_dataframe_input = bigqueryinputclient.query(
        f"""
        (select dteday, temp, holiday, workingday, windspeed
        FROM {big_query_input_connection.details()['dataset']}.{big_query_input_connection.details()['table']}
        ORDER BY dteday DESC LIMIT 7)
        ORDER BY dteday
        """
    ).to_dataframe().drop(columns=['dteday'])
```

```python
# convert to a dict, show the first 7 rows
display(inference_dataframe_input.to_dict())
```

    {'temp': {0: 0.291304,
      1: 0.243333,
      2: 0.254167,
      3: 0.253333,
      4: 0.253333,
      5: 0.255833,
      6: 0.215833},
     'holiday': {0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
     'workingday': {0: 0, 1: 1, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1},
     'windspeed': {0: 0.168726,
      1: 0.316546,
      2: 0.350133,
      3: 0.155471,
      4: 0.124383,
      5: 0.350754,
      6: 0.154846}}

### Sample Inference

With our data retrieved, we'll perform an inference and display the results.

```python
#deploy the pipeline
pipeline.deploy()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>bigquerystatsmodelpipelinesxee</td></tr><tr><th>created</th> <td>2023-05-19 22:31:29.309549+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-19 22:33:14.710969+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>a54faf95-e6b7-4246-9a40-920ce373919f, 6cb52e7d-4ac0-452a-9bc6-c42aa5c60995, 3ecf0b42-e69d-4ab0-a983-a0f105551523, a795521c-e965-4835-bad4-164cab59ec4a</td></tr><tr><th>steps</th> <td>bigquerystatsmodelmodelsxee</td></tr></table>
{{</table>}}

```python
results = pipeline.infer(inference_dataframe_input.to_dict())
display(results[0]['forecast'])
```

    [1231.2556997246595,
     1627.3643469089343,
     1674.3769827243134,
     1621.9273295873882,
     1140.7465817903185,
     1211.5223974364667,
     1457.1896450382922]

### Upload the Results

With the query complete, we'll upload the results back to the BigQuery dataset.

```python
output_table = bigqueryoutputclient.get_table(f"{big_query_output_connection.details()['dataset']}.{big_query_output_connection.details()['table']}")

job = bigqueryoutputclient.query(
        f"""
        INSERT {big_query_output_connection.details()['dataset']}.{big_query_output_connection.details()['table']}
        VALUES
        (current_timestamp(), "{results[0]['forecast']}")
        """
    )
```

```python
# Get the last insert to the output table to verify
# wait 10 seconds for the insert to finish
time.sleep(10)
task_inference_results = bigqueryoutputclient.query(
        f"""
        SELECT *
        FROM {big_query_output_connection.details()['dataset']}.{big_query_output_connection.details()['table']}
        ORDER BY date DESC
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
      <th>date</th>
      <th>forecast</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-05-19 22:33:27.655703+00:00</td>
      <td>[1231.2556997246595, 1627.3643469089343, 1674.3769827243134, 1621.9273295873882, 1140.7465817903185, 1211.5223974364667, 1457.1896450382922]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-05-17 17:25:05.452843+00:00</td>
      <td>[1231.2556997246595, 1627.3643469089343, 1674.3769827243134, 1621.9273295873882, 1140.7465817903185, 1211.5223974364667, 1457.1896450382922]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-05-11 20:13:57.448442+00:00</td>
      <td>[1231.2556997246595, 1627.3643469089343, 1674.3769827243134, 1621.9273295873882, 1140.7465817903185, 1211.5223974364667, 1457.1896450382922]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-05-11 20:11:04.519782+00:00</td>
      <td>[1231.2556997246595, 1627.3643469089343, 1674.3769827243134, 1621.9273295873882, 1140.7465817903185, 1211.5223974364667, 1457.1896450382922]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-05-11 20:04:49.821907+00:00</td>
      <td>[1231.2556997246595, 1627.3643469089343, 1674.3769827243134, 1621.9273295873882, 1140.7465817903185, 1211.5223974364667, 1457.1896450382922]</td>
    </tr>
  </tbody>
</table>
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

1. Use the `bigquery_remote_inference` to open a connection to the input and output tables.
1. Deploy the pipeline.
1. Perform an inference with the input data.
1. Save the inference results to the output table.
1. Undeploy the pipeline.

This sample script is stored in `bigquery_statsmodel_remote_inference/main.py` with an `requirements.txt` file having the specific libraries for the Google BigQuery connection., and packaged into the orchestration as `./bigquery_statsmodel_remote_inference/bigquery_statsmodel_remote_inference.zip`.  We'll display the steps in uploading the orchestration to the Wallaroo instance.

Note that the orchestration assumes the pipeline is already deployed.

### Upload the Orchestration

Orchestrations are uploaded with the Wallaroo client `upload_orchestration(path)` method with the following parameters.

| Parameter | Type | Description |
| --- | --- | ---|
| **path** | string (Required) | The path to the ZIP file to be uploaded. |

Once uploaded, the deployment will be prepared and any requirements will be downloaded and installed.

For this example, the orchestration `./bigquery_remote_inference/bigquery_remote_inference.zip` will be uploaded and saved to the variable `orchestration`.  Then we will loop until the orchestration status is `ready`.

```python
pipeline.deploy()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>bigquerystatsmodelpipelinesxee</td></tr><tr><th>created</th> <td>2023-05-19 22:31:29.309549+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-19 22:35:42.094232+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>eaa3e50d-f216-422c-86cc-1426b42d8ace, 1ee05b6f-ce95-4dd3-853a-b85c9ab4d2a6, a54faf95-e6b7-4246-9a40-920ce373919f, 6cb52e7d-4ac0-452a-9bc6-c42aa5c60995, 3ecf0b42-e69d-4ab0-a983-a0f105551523, a795521c-e965-4835-bad4-164cab59ec4a</td></tr><tr><th>steps</th> <td>bigquerystatsmodelmodelsxee</td></tr></table>
{{</table>}}

```python
orchestration = wl.upload_orchestration(path="./bigquery_remote_inference/bigquery_remote_inference.zip")

while orchestration.status() != 'ready':
    print(orchestration.status())
    time.sleep(5)
```

    ---------------------------------------------------------------------------

    OrchestrationUploadFailed                 Traceback (most recent call last)

    /var/folders/jf/_cj0q9d51s365wksymljdz4h0000gn/T/ipykernel_80766/16327239.py in <module>
    ----> 1 orchestration = wl.upload_orchestration(path="./bigquery_remote_inference/bigquery_remote_inference.zip")
          2 
          3 while orchestration.status() != 'ready':
          4     print(orchestration.status())
          5     time.sleep(5)

    /opt/homebrew/anaconda3/envs/wallaroosdk202302preview/lib/python3.9/site-packages/wallaroo/client.py in upload_orchestration(self, bytes_buffer, path, name, file_name)
       2123 
       2124         """
    -> 2125         return Orchestration.upload(
       2126             self, bytes_buffer=bytes_buffer, path=path, name=name, file_name=file_name
       2127         )

    /opt/homebrew/anaconda3/envs/wallaroosdk202302preview/lib/python3.9/site-packages/wallaroo/orchestration.py in upload(client, name, bytes_buffer, path, file_name)
        142 
        143         if ret is None:
    --> 144             raise OrchestrationUploadFailed("Internal service error")
        145 
        146         orch = Orchestration(client, dict({"id": ret.id}))

    OrchestrationUploadFailed: Orchestration upload failed: Internal service error

```python
wl.list_orchestrations()
```

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
# Get the last insert to the output table to verify

task_inference_results = bigqueryoutputclient.query(
        f"""
        SELECT *
        FROM {big_query_output_connection.details()['dataset']}.{big_query_output_connection.details()['table']}
        ORDER BY date DESC
        LIMIT 5
        """
    ).to_dataframe()

display(task_inference_results)
```

```python
# Example: run once
task = orchestration.run_once(name="big query statsmodel run once", json_args={})
task
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

```python
wl.list_tasks()
```

### Task Results

We can view the inferences from our logs and verify that new entries were added from our task.  We'll query the last 5 rows of our inference output table after a wait of 60 seconds.

```python
time.sleep(30)

# Get the last insert to the output table to verify

task_inference_results = bigqueryoutputclient.query(
        f"""
        SELECT *
        FROM {big_query_output_connection.details()['dataset']}.{big_query_output_connection.details()['table']}
        ORDER BY date DESC
        LIMIT 5
        """
    ).to_dataframe()

display(task_inference_results)
```

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

For our example, we will create a scheduled task to run every 1 minute, display the inference results, then use the Orchestration `kill` task to keep the task from running any further.

```python
task_inference_results = bigqueryoutputclient.query(
        f"""
        SELECT *
        FROM {big_query_output_connection.details()['dataset']}.{big_query_output_connection.details()['table']}
        ORDER BY date DESC
        LIMIT 5
        """
    ).to_dataframe()

display(task_inference_results)

scheduled_task = orchestration.run_scheduled(name="simple_statsmodel_inference_schedule", schedule="*/1 * * * *", timeout=120, json_args={})
```

```python
while scheduled_task.status() != "started":
    display(scheduled_task.status())
    time.sleep(5)
```

```python
#wait 120 seconds to give the scheduled event time to finish
time.sleep(60)
task_inference_results = bigqueryoutputclient.query(
        f"""
        SELECT *
        FROM {big_query_output_connection.details()['dataset']}.{big_query_output_connection.details()['table']}
        ORDER BY date DESC
        LIMIT 5
        """
    ).to_dataframe()

display(task_inference_results)
```

### Kill Task

With our testing complete, we will kill the scheduled task so it will not run again.  First we'll show all the tasks to verify that our task is there, then issue it the kill command.

```python
scheduled_task.kill()
```

### Running Task with Custom Parameters

Right now, our task assumes the workspace, pipeline, and connections all have the names we defined above.  For this example, we'll set up a new pipeline with the same pipeline step, but name it `bigquerystatsmodelpipeline02`.

When we create our task, we'll add that pipeline name as an argument to our task.  Within our orchestrations `main.py` there is a code block that takes in the task arguments, then sets the pipeline name:

```python
arguments = wl.task_args()
if "pipeline_name" in arguments:
        pipeline_name = arguments['pipeline_name']
    else:
        pipeline_name="bigquerystatsmodelpipeline"
```

We'll pass along our new pipeline name as `{ "pipeline_name": "bigquerystatsmodelpipeline02" }` and track the task progress as before.

```python
newpipeline_name = 'bigquerystatsmodelpipeline02'

pipeline02 = get_pipeline(newpipeline_name)
# add the model as the pipeline step
pipeline02.add_model_step(bike_day_model)

```

```python
# required to set the pipeline steps
pipeline02.deploy()
```

```python
# Get the last insert to the output table to verify

task_inference_results = bigqueryoutputclient.query(
        f"""
        SELECT *
        FROM {big_query_output_connection.details()['dataset']}.{big_query_output_connection.details()['table']}
        ORDER BY date DESC
        LIMIT 5
        """
    ).to_dataframe()

display(task_inference_results)

# Generate the run once task with the new parameter
task = orchestration.run_once(name="parameter sample", json_args={ "pipeline_name": newpipeline_name })
display(task)

# wait for the task to run
while task.status() != "started":
    display(task.status())
    time.sleep(5)
```

```python
# wait 60 seconds then display the results
time.sleep(30)

task_inference_results = bigqueryoutputclient.query(
        f"""
        SELECT *
        FROM {big_query_output_connection.details()['dataset']}.{big_query_output_connection.details()['table']}
        ORDER BY date DESC
        LIMIT 5
        """
    ).to_dataframe()

display(task_inference_results)
```

## Conclusion

With that, our tutorial is over.  Please feel free to use this tutorial code in your own Wallaroo related projects.  Our last task will be to undeploy our pipelines to restore the resources back to the Wallaroo instance.

```python
pipeline.undeploy()
pipeline02.undeploy()
```

```python

```
