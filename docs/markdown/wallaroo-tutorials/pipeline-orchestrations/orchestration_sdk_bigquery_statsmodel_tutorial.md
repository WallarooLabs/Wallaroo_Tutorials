This can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/blob/20230314_2023.2_updates/pipeline-orchestrators/orchestration_sdk_bigquery_statsmodel_tutorial).

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

import os
# Used for the Wallaroo SDK version 2023.1
os.environ["ARROW_ENABLED"]="True"

import time
import json

# for Big Query connections
from google.cloud import bigquery
from google.oauth2 import service_account
import db_dtypes
```

```python
wallaroo.__version__
```

    '2023.2.0rc2'

### Connect to the Wallaroo Instance

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  If logging in externally, update the `wallarooPrefix` and `wallarooSuffix` variables with the proper DNS information.  For more information on Wallaroo DNS settings, see the [Wallaroo DNS Integration Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-dns-guide/).

```python
# Login through local Wallaroo instance

# wl = wallaroo.Client()

# # SSO login through keycloak

# wallarooPrefix = "product-uat-ee"
# wallarooSuffix = "wallaroocommunity.ninja"

wallarooPrefix = "doc-test"
wallarooSuffix = "wallaroocommunity.ninja"

wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}.api.{wallarooSuffix}", 
                    auth_endpoint=f"https://{wallarooPrefix}.keycloak.{wallarooSuffix}", 
                    auth_type="sso")
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

workspace_name = 'bigquerystatsmodelworkspace'
pipeline_name = 'bigquerystatsmodelpipeline'
model_name = 'bigquerystatsmodelmodel'
model_file_name = './models/bike_day_model.pkl'

bigquery_connection_input_name = "bigqueryforecastinputs"
bigquery_connection_input_type = "BIGQUERY"
bigquery_connection_input_argument = json.load(open('./resources/bigquery_service_account_input_key.json.example'))

bigquery_connection_output_name = "bigqueryforecastoutputs"
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

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>bigquerystatsmodelpipeline</td></tr><tr><th>created</th> <td>2023-05-11 18:51:52.266084+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-11 18:53:05.836042+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>2ebd2911-47e8-4c10-8b59-f37d6d7ee948, 5ea9cdf0-d03e-4bba-acb6-e2dc071df842, 9984bdfb-fe27-40b5-8e4b-1bd46c09d30a</td></tr><tr><th>steps</th> <td>bigquerystatsmodelmodel</td></tr></table>
{{</table>}}

```python
#deploy the pipeline
pipeline.deploy()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>bigquerystatsmodelpipeline</td></tr><tr><th>created</th> <td>2023-05-11 18:51:52.266084+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-11 18:58:02.257796+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>68ee66b5-454e-4e2a-9524-d11f8629adfd, 2ebd2911-47e8-4c10-8b59-f37d6d7ee948, 5ea9cdf0-d03e-4bba-acb6-e2dc071df842, 9984bdfb-fe27-40b5-8e4b-1bd46c09d30a</td></tr><tr><th>steps</th> <td>bigquerystatsmodelmodel</td></tr></table>
{{</table>}}

### Sample Inferences

We'll perform some quick sample inferences with the local file data to verity the pipeline deployed and is ready for inferences.  Once done, we'll undeploy the pipeline.

```python
## perform inferences

results = pipeline.infer_from_file('./data/bike_day_eval.json', data_format="custom-json")
print(results)

```

    [{'forecast': [1882.3784555157672, 2130.607915701861, 2340.84005381799, 2895.754978552066, 2163.657515565616, 1509.1792126509536, 2431.1838923957016]}]

```python
# Undeploy the pipeline
pipeline.undeploy()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>bigquerystatsmodelpipeline</td></tr><tr><th>created</th> <td>2023-05-11 18:51:52.266084+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-11 18:58:02.257796+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>68ee66b5-454e-4e2a-9524-d11f8629adfd, 2ebd2911-47e8-4c10-8b59-f37d6d7ee948, 5ea9cdf0-d03e-4bba-acb6-e2dc071df842, 9984bdfb-fe27-40b5-8e4b-1bd46c09d30a</td></tr><tr><th>steps</th> <td>bigquerystatsmodelmodel</td></tr></table>
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

See the `statsmodel_forecast_inputs` and `statsmodel_forecast_outputs` details listed above for the table schema used for our example.

```python
connection_input = get_connection(bigquery_connection_input_name, bigquery_connection_input_type, bigquery_connection_input_argument)
connection_output = get_connection(bigquery_connection_output_name, bigquery_connection_output_type, bigquery_connection_output_argument)
#wl.list_connections()
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
    <td>Name</td><td>bigqueryforecastinputs</td>
  </tr>
  <tr>
    <td>Connection Type</td><td>BIGQUERY</td>
  </tr>
  <tr>
    <td>Details</td><td>*****</td>
  </tr>
  <tr>
    <td>Created At</td><td>2023-05-11T18:52:55.992411+00:00</td>
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
    <td>Name</td><td>bigqueryforecastoutputs</td>
  </tr>
  <tr>
    <td>Connection Type</td><td>BIGQUERY</td>
  </tr>
  <tr>
    <td>Details</td><td>*****</td>
  </tr>
  <tr>
    <td>Created At</td><td>2023-05-11T18:52:56.877371+00:00</td>
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
    <td>Name</td><td>bigqueryforecastinputs</td>
  </tr>
  <tr>
    <td>Connection Type</td><td>BIGQUERY</td>
  </tr>
  <tr>
    <td>Details</td><td>*****</td>
  </tr>
  <tr>
    <td>Created At</td><td>2023-05-11T18:52:55.992411+00:00</td>
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
    <td>Name</td><td>bigqueryforecastoutputs</td>
  </tr>
  <tr>
    <td>Connection Type</td><td>BIGQUERY</td>
  </tr>
  <tr>
    <td>Details</td><td>*****</td>
  </tr>
  <tr>
    <td>Created At</td><td>2023-05-11T18:52:56.877371+00:00</td>
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
# wl.list_connections()
```

```python
workspace.add_connection(bigquery_connection_input_name)
workspace.add_connection(bigquery_connection_output_name)

workspace.list_connections()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th><th>connection type</th><th>details</th><th>created at</th></tr><tr><td>bigqueryforecastinputs</td><td>BIGQUERY</td><td>*****</td><td>2023-05-11T18:52:55.992411+00:00</td></tr><tr><td>bigqueryforecastoutputs</td><td>BIGQUERY</td><td>*****</td><td>2023-05-11T18:52:56.877371+00:00</td></tr></table>
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
<table><tr><th>name</th> <td>bigquerystatsmodelpipeline</td></tr><tr><th>created</th> <td>2023-05-11 18:51:52.266084+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-11 18:59:01.260909+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>c3380b5c-1279-4af9-adde-4364dbef4b4b, 68ee66b5-454e-4e2a-9524-d11f8629adfd, 2ebd2911-47e8-4c10-8b59-f37d6d7ee948, 5ea9cdf0-d03e-4bba-acb6-e2dc071df842, 9984bdfb-fe27-40b5-8e4b-1bd46c09d30a</td></tr><tr><th>steps</th> <td>bigquerystatsmodelmodel</td></tr></table>
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
      <td>2023-05-11 20:13:57.448442+00:00</td>
      <td>[1231.2556997246595, 1627.3643469089343, 1674.3769827243134, 1621.9273295873882, 1140.7465817903185, 1211.5223974364667, 1457.1896450382922]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-05-11 20:11:04.519782+00:00</td>
      <td>[1231.2556997246595, 1627.3643469089343, 1674.3769827243134, 1621.9273295873882, 1140.7465817903185, 1211.5223974364667, 1457.1896450382922]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-05-11 20:04:49.821907+00:00</td>
      <td>[1231.2556997246595, 1627.3643469089343, 1674.3769827243134, 1621.9273295873882, 1140.7465817903185, 1211.5223974364667, 1457.1896450382922]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-05-11 20:03:25.094349+00:00</td>
      <td>[1231.2556997246595, 1627.3643469089343, 1674.3769827243134, 1621.9273295873882, 1140.7465817903185, 1211.5223974364667, 1457.1896450382922]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-05-11 20:02:08.732396+00:00</td>
      <td>[1231.2556997246595, 1627.3643469089343, 1674.3769827243134, 1621.9273295873882, 1140.7465817903185, 1211.5223974364667, 1457.1896450382922]</td>
    </tr>
  </tbody>
</table>
{{</table>}}

## Conclusion

With that, our tutorial is over.  Please feel free to use this tutorial code in your own Wallaroo related projects.
