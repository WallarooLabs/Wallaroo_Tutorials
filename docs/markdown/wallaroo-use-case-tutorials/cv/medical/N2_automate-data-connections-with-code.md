# Workshop Notebook 5: Automation with Wallaroo Connections

Wallaroo Connections are definitions set by MLOps engineers that are used by other Wallaroo users for connection information to a data source.

This provides MLOps engineers a method of creating and updating connection information for data stores:  databases, Kafka topics, etc.  Wallaroo Connections are composed of three main parts:

* Name:  The unique name of the connection.
* Type:  A user defined string that designates the type of connection.  This is used to organize connections.
* Details:  Details are a JSON object containing the information needed to make the connection.  This can include data sources, authentication tokens, etc.

Wallaroo Connections are only used to store the connection information used by other processes to create and use external connections.  The user still has to provide the libraries and other elements to actually make and use the conneciton.

The primary advantage is Wallaroo connections allow scripts and other code to retrieve the connection details directly from their Wallaroo instance, then refer to those connection details.  They don't need to know what those details actually - they can refer to them in their code to make their code more flexible.

For this step, we will use a Google BigQuery dataset to retrieve the inference information, predict the next month of sales, then store those predictions into another table.  This will use the Wallaroo Connection feature to create a Connection, assign it to our workspace, then perform our inferences by using the Connection details to connect to the BigQuery dataset and tables.

## Prerequisites

* A Wallaroo instance version 2023.2.1 or greater.

## References

* [Wallaroo SDK Essentials Guide: Model Uploads and Registrations: Python Models](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/wallaroo-sdk-model-upload-python/)
* [Wallaroo SDK Essentials Guide: Pipeline Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline/)
* [Wallaroo SDK Essentials Guide: Data Connections Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-dataconnections/)

## Preliminaries

In the blocks below we will preload some required libraries.

For convenience, the following `helper functions` are defined to retrieve previously created workspaces, models, and pipelines:

* `get_workspace(name, client)`: This takes in the name and the Wallaroo client being used in this session, and returns the workspace matching `name`.  If no workspaces are found matching the name, raises a `KeyError` and returns `None`.
* `get_model_version(model_name, workspace)`: Retrieves the most recent model version from the model matching the `model_name` within the provided `workspace`.  If no model matches that name, raises a `KeyError` and returns `None`.
* `get_pipeline(pipeline_name, workspace)`: Retrieves the most pipeline from the workspace matching the `pipeline_name` within the provided `workspace`.  If no model matches that name, raises a `KeyError` and returns `None`.

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

pd.set_option('display.max_colwidth', None)

import time
import pyarrow as pa
```

```python
## convenience functions from the previous notebooks

# return the workspace called <name> through the Wallaroo client.
def get_workspace(name, client):
    workspace = None
    for ws in client.list_workspaces():
        if ws.name() == name:
            workspace= ws
            return workspace
    # if no workspaces were found
    if workspace==None:
        raise KeyError(f"Workspace {name} was not found.")
    return workspace

# returns the most recent model version in a workspace for the matching `model_name`
def get_model_version(model_name, workspace):
    modellist = workspace.models()
    model_version = [m.versions()[-1] for m in modellist if m.name() == model_name]
    # if no models match, return None
    if len(modellist) <= 0:
        raise KeyError(f"Model {mname} not found in this workspace")
        return None
    return model_version[0]

# get a pipeline by name in the workspace
def get_pipeline(pipeline_name, workspace):
    plist = workspace.pipelines()
    pipeline = [p for p in plist if p.name() == pipeline_name]
    if len(pipeline) <= 0:
        raise KeyError(f"Pipeline {pipeline_name} not found in this workspace")
        return None
    return pipeline[0]

```

### Connect to the Wallaroo Instance

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
## blank space to log in 

wl = wallaroo.Client()

wallarooPrefix = "doc-test."
wallarooSuffix = "wallarooexample.ai"

wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}api.{wallarooSuffix}", 
                    auth_endpoint=f"https://{wallarooPrefix}keycloak.{wallarooSuffix}", 
                    auth_type="sso")
```

    Please log into the following URL in a web browser:
    
    	https://doc-test.keycloak.wallarooexample.ai/auth/realms/master/device?user_code=WVDT-DNFQ
    
    Login successful!

### Set Configurations

Set the workspace, pipeline, and model used from Notebook 1.  The helper functions will make this task easier.

#### Set Configurations References

* [Wallaroo SDK Essentials Guide: Workspace Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-workspace/)
* [Wallaroo SDK Essentials Guide: Pipeline Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline/)

```python
# retrieve the previous workspace, model, and pipeline version

workspace_name = "workshop-workspace-john-cv-medical"

workspace = get_workspace(workspace_name, wl)

# set your current workspace to the workspace that you just created
wl.set_current_workspace(workspace)

# optionally, examine your current workspace
wl.get_current_workspace()

model_name = 'mitochondria-detector'

prime_model_version = get_model_version(model_name, workspace)

pipeline_name = 'cv-mitochondria'

pipeline = get_pipeline(pipeline_name, workspace)

display(workspace)
display(prime_model_version)
display(pipeline)

```

    {'name': 'workshop-workspace-john-cv-medical', 'id': 10, 'archived': False, 'created_by': 'b030ff9c-41eb-49b4-afdf-2ccbecb6be5d', 'created_at': '2023-09-28T20:22:21.506043+00:00', 'models': [{'name': 'mitochondria-detector', 'versions': 2, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 9, 29, 17, 40, 6, 205146, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 9, 28, 20, 25, 2, 359367, tzinfo=tzutc())}], 'pipelines': [{'name': 'cv-mitochondria', 'create_time': datetime.datetime(2023, 9, 28, 20, 25, 17, 379732, tzinfo=tzutc()), 'definition': '[]'}]}

<table>
        <tr>
          <td>Name</td>
          <td>mitochondria-detector</td>
        </tr>
        <tr>
          <td>Version</td>
          <td>11a34cce-0eaa-4cdd-8739-43bd18b67a04</td>
        </tr>
        <tr>
          <td>File Name</td>
          <td>mitochondria_epochs_15.onnx</td>
        </tr>
        <tr>
          <td>SHA</td>
          <td>e80fcdaf563a183b0c32c027dcb3890a64e1764d6d7dcd29524cd270dd42e7bd</td>
        </tr>
        <tr>
          <td>Status</td>
          <td>ready</td>
        </tr>
        <tr>
          <td>Image Path</td>
          <td>None</td>
        </tr>
        <tr>
          <td>Updated At</td>
          <td>2023-29-Sep 17:40:06</td>
        </tr>
      </table>

<table><tr><th>name</th> <td>cv-mitochondria</td></tr><tr><th>created</th> <td>2023-09-28 20:25:17.379732+00:00</td></tr><tr><th>last_updated</th> <td>2023-09-29 18:08:12.573990+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>4b6dab7d-b3ff-4f14-8425-7d9b6de76637, 66e72bc0-a3e3-4872-bc20-19b992c194b4, cf4bdfb4-1eec-46f8-9af4-b16dea894de6, 507cb1eb-8034-4b5b-bc96-2427730a6407, 50ed1d2f-6dba-411c-9579-1090791b33bd, 6208c9cf-fcd6-4b20-bbfc-f6ce714596e3</td></tr><tr><th>steps</th> <td>mitochondria-detector</td></tr><tr><th>published</th> <td>False</td></tr></table>

## Deploy the Pipeline with the Model Version Step

As per the other workshops:

1. Clear the pipeline of all steps.
1. Add the model version as a pipeline step.
1. Deploy the pipeline with the following deployment configuration:

```python
deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(0.5).memory("1Gi").build()
```

```python
pipeline.clear()
pipeline.add_model_step(prime_model_version)

deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(0.5).memory("1Gi").build()
pipeline.deploy(deployment_config=deploy_config)
```

<table><tr><th>name</th> <td>cv-mitochondria</td></tr><tr><th>created</th> <td>2023-09-28 20:25:17.379732+00:00</td></tr><tr><th>last_updated</th> <td>2023-09-29 19:07:47.787981+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>ac4bd826-f47f-48b7-8319-01fbc6622899, 4b6dab7d-b3ff-4f14-8425-7d9b6de76637, 66e72bc0-a3e3-4872-bc20-19b992c194b4, cf4bdfb4-1eec-46f8-9af4-b16dea894de6, 507cb1eb-8034-4b5b-bc96-2427730a6407, 50ed1d2f-6dba-411c-9579-1090791b33bd, 6208c9cf-fcd6-4b20-bbfc-f6ce714596e3</td></tr><tr><th>steps</th> <td>mitochondria-detector</td></tr><tr><th>published</th> <td>False</td></tr></table>

## Create the Connection

For this demonstration, the connection set to a specific file on a GitHub repository.  The connection details can be anything that can be stored in JSON:  connection URLs, tokens, etc.

This connection will set a URL to pull a file from GitHub, then use the file contents to perform an inference.

Wallaroo connections are created through the Wallaroo Client `create_connection(name, type, details)` method.  See the [Wallaroo SDK Essentials Guide: Data Connections Management guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-dataconnections/) for full details.

Note that connection names must be unique across the Wallaroo instance - if needed, use random characters at the end to make sure your connection doesn't have the same name as a previously created connection.

Here's an example connection used to retrieve the same CSV file used in `./data/test_data.df.json`:  https://raw.githubusercontent.com/WallarooLabs/Workshops/main/Linear%20Regression/Real%20Estate/data/test_data.df.json

### Create the Connection Exercise

```python
# set the connection information for other steps
# suffix is used to create a unique data connection

forecast_connection_input_name = f'house-price-data'
forecast_connection_input_type = "HTTP"
forecast_connection_input_argument = { 
    "url": "https://raw.githubusercontent.com/WallarooLabs/Workshops/main/Linear%20Regression/Real%20Estate/data/test_data.df.json"
    }

wl.create_connection(forecast_connection_input_name, forecast_connection_input_type, forecast_connection_input_argument)
```

```python
# set the connection information for other steps
# suffix is used to create a unique data connection

medical_imaging_connection_input_name = f'house-price-data-source-medical-jch'
medical_imaging_input_type = "HTTP"
medical_imaging_input_argument = { 
    "url": "https://raw.githubusercontent.com/WallarooLabs/Workshops/20230927-foundations-v2/Computer%20Vision/Healthcare/data/image_0_21.tif.df.json"
    }

wl.create_connection(medical_imaging_connection_input_name, medical_imaging_input_type, medical_imaging_input_argument)
```

<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Name</td><td>house-price-data-source-medical-jch</td>
  </tr>
  <tr>
    <td>Connection Type</td><td>HTTP</td>
  </tr>
  <tr>
    <td>Details</td><td>*****</td>
  </tr>
  <tr>
    <td>Created At</td><td>2023-09-29T19:15:01.906357+00:00</td>
  </tr>
  <tr>
    <td>Linked Workspaces</td><td>[]</td>
  </tr>
</table>

## List Connections

Connections for the entire Wallaroo instance are listed with Wallaroo Client `list_connections()` method.

## List Connections Exercise

Here's an example of listing the connections when the Wallaroo client is `wl`.

```python
wl.list_connections()
```

```python
# list the connections here

wl.list_connections()
```

<table><tr><th>name</th><th>connection type</th><th>details</th><th>created at</th><th>linked workspaces</th></tr><tr><td>house-price-data-source-medical</td><td>HTTP</td><td>*****</td><td>2023-09-29T19:08:24.238046+00:00</td><td>[]</td></tr><tr><td>house-price-data-source-medical-john</td><td>HTTP</td><td>*****</td><td>2023-09-29T19:09:36.352483+00:00</td><td>['workshop-workspace-john-cv-medical']</td></tr><tr><td>house-price-data-source-medical-jch</td><td>HTTP</td><td>*****</td><td>2023-09-29T19:15:01.906357+00:00</td><td>[]</td></tr></table>

## Get Connection by Name

To retrieve a previosly created conneciton, we can assign it to a variable with the method Wallaroo `Client.get_connection(connection_name)`.  Then we can display the connection itself.  Notice that when displaying a connection, the `details` section will be hidden, but they are retrieved with `connection.details()`.  Here's an example:

```python
myconnection = client.get_connection("My amazing connection")
display(myconnection)
display(myconnection.details()
```

Use that code to retrieve your new connection.

### Get Connection by Name Example

Here's an example based on the Wallaroo client saved as `wl`.

```python
wl.get_connection(forecast_connection_input_name)
```

```python
# get the connection by name

this_connection = wl.get_connection(medical_imaging_connection_input_name)
this_connection
```

<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Name</td><td>house-price-data-source-medical-jch</td>
  </tr>
  <tr>
    <td>Connection Type</td><td>HTTP</td>
  </tr>
  <tr>
    <td>Details</td><td>*****</td>
  </tr>
  <tr>
    <td>Created At</td><td>2023-09-29T19:15:01.906357+00:00</td>
  </tr>
  <tr>
    <td>Linked Workspaces</td><td>[]</td>
  </tr>
</table>

## Add Connection to Workspace

We'll now add the connection to our workspace so it can be retrieved by other workspace users.  The method Workspace `add_connection(connection_name)` adds a Data Connection to a workspace.  The method Workspace `list_connections()` displays a list of connections attached to the workspace.

### Add Connection to Workspace Exercise

Use the connection we just created, and add it to the sample workspace.  Here's a code example where the workspace is saved to the variable `workspace` and the connection is saved as `forecast_connection_input_name`.

```python
workspace.add_connection(forecast_connection_input_name)
```

```python
workspace.add_connection(medical_imaging_connection_input_name)
workspace.list_connections()
```

<table><tr><th>name</th><th>connection type</th><th>details</th><th>created at</th><th>linked workspaces</th></tr><tr><td>house-price-data-source-medical-john</td><td>HTTP</td><td>*****</td><td>2023-09-29T19:09:36.352483+00:00</td><td>['workshop-workspace-john-cv-medical']</td></tr><tr><td>house-price-data-source-medical-jch</td><td>HTTP</td><td>*****</td><td>2023-09-29T19:15:01.906357+00:00</td><td>['workshop-workspace-john-cv-medical']</td></tr></table>

## Retrieve Connection from Workspace

To simulate a data scientist's procedural flow, we'll now retrieve the connection from the workspace.  Specific connections are retrieved by specifying their position in the returned list.

For example, if we have two connections in a workspace and we want the second one, we can assign it to a variable with `list_connections[1]`.

Create a new variable and retrieve the connection we just assigned to the workspace.

### Retrieve Connection from Workspace Exercise

Retrieve the connection that was just associated with the workspace.  You'll use the `list_connections` method, then assign a variable to the connection.  Here's an example if the connection is the most recently one added to the workspace `workspace`.

```python
forecast_connection = workspace.list_connections()[-1]
```

```python
medical_connection = workspace.list_connections()[-1]
display(medical_connection)
```

<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Name</td><td>house-price-data-source-medical-jch</td>
  </tr>
  <tr>
    <td>Connection Type</td><td>HTTP</td>
  </tr>
  <tr>
    <td>Details</td><td>*****</td>
  </tr>
  <tr>
    <td>Created At</td><td>2023-09-29T19:15:01.906357+00:00</td>
  </tr>
  <tr>
    <td>Linked Workspaces</td><td>['workshop-workspace-john-cv-medical']</td>
  </tr>
</table>

## Run Inference with Connection

Connections can be used for different purposes:  uploading new models, engine configurations - any place that data is needed.  This exercise will use the data connection to perform an inference through our deployed pipeline.

### Run Inference with Connection Exercise

We'll now retrieve sample data through the Wallaroo connection, and perform a sample inference.  The connection details are retrieved through the Connection `details()` method.  Use them to retrieve the pandas record file and convert it to a DataFrame, and use it with our sample model.

Here's a code example that uses the Python `requests` library to retrieve the file information, then turns it into a DataFrame for the inference request.

```python
display(forecast_connection.details()['url'])

import requests

response = requests.get(
                    forecast_connection.details()['url']
                )

# display(response.json())

df = pd.DataFrame(response.json())

pipeline.infer(df)
```

```python
display(medical_connection.details()['url'])

import requests

response = requests.get(
    medical_connection.details()['url']
)

df = pd.DataFrame(response.json())
display(df)

result = pipeline.infer(df)
display(result)
```

    'https://raw.githubusercontent.com/WallarooLabs/Workshops/20230927-foundations-v2/Computer%20Vision/Healthcare/data/image_0_21.tif.df.json'

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
      <td>[0.0572924651, 0.0555159546, 0.0621778691, 0.053739444, 0.04885404, 0.053739444, 0.0541835717, 0.0381949768, 0.0448568913, 0.0413038702, 0.0444127637, 0.0492981677, 0.0497422953, 0.0426362531, 0.0515188059, 0.0479657848, 0.0404156149, 0.039083232, 0.0470775295, 0.0413038702, 0.0461892742, 0.039083232, 0.0448568913, 0.0519629335, 0.0484099124, 0.043968636, 0.0453010189, 0.0448568913, 0.0519629335, 0.0444127637, 0.0484099124, 0.0479657848, 0.0546276993, 0.0461892742, 0.0635102521, 0.0515188059, 0.0612896139, 0.0519629335, 0.0546276993, 0.0666191455, 0.0541835717, 0.0577365928, 0.0568483375, 0.0555159546, 0.04885404, 0.053739444, 0.043968636, 0.0466334019, 0.0421921255, 0.0510746782, 0.0492981677, 0.0457451466, 0.0670632731, 0.0577365928, 0.0595131033, 0.0675074008, 0.0732810601, 0.0661750179, 0.0635102521, 0.0755016982, 0.0803871022, 0.0728369324, 0.0777223364, 0.0848283786, 0.084384251, 0.0866048892, 0.0768340812, 0.0821636128, 0.0803871022, 0.0866048892, 0.0848283786, 0.0777223364, 0.0834959957, 0.0786105917, 0.0799429746, 0.0808312299, 0.0817194852, 0.0839401233, 0.0808312299, 0.0786105917, 0.0777223364, 0.0759458259, 0.0715045495, 0.074613443, 0.064842635, 0.0595131033, 0.0617337415, 0.058624848, 0.059957231, 0.0652867626, 0.0715045495, 0.0675074008, 0.0728369324, 0.0728369324, 0.0732810601, 0.0750575706, 0.0728369324, 0.0723928048, 0.0723928048, 0.0675074008, ...]</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.conv2d_37</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-09-29 19:15:27.809</td>
      <td>[0.0572924651, 0.0555159546, 0.0621778691, 0.053739444, 0.04885404, 0.053739444, 0.0541835717, 0.0381949768, 0.0448568913, 0.0413038702, 0.0444127637, 0.0492981677, 0.0497422953, 0.0426362531, 0.0515188059, 0.0479657848, 0.0404156149, 0.039083232, 0.0470775295, 0.0413038702, 0.0461892742, 0.039083232, 0.0448568913, 0.0519629335, 0.0484099124, 0.043968636, 0.0453010189, 0.0448568913, 0.0519629335, 0.0444127637, 0.0484099124, 0.0479657848, 0.0546276993, 0.0461892742, 0.0635102521, 0.0515188059, 0.0612896139, 0.0519629335, 0.0546276993, 0.0666191455, 0.0541835717, 0.0577365928, 0.0568483375, 0.0555159546, 0.04885404, 0.053739444, 0.043968636, 0.0466334019, 0.0421921255, 0.0510746782, 0.0492981677, 0.0457451466, 0.0670632731, 0.0577365928, 0.0595131033, 0.0675074008, 0.0732810601, 0.0661750179, 0.0635102521, 0.0755016982, 0.0803871022, 0.0728369324, 0.0777223364, 0.0848283786, 0.084384251, 0.0866048892, 0.0768340812, 0.0821636128, 0.0803871022, 0.0866048892, 0.0848283786, 0.0777223364, 0.0834959957, 0.0786105917, 0.0799429746, 0.0808312299, 0.0817194852, 0.0839401233, 0.0808312299, 0.0786105917, 0.0777223364, 0.0759458259, 0.0715045495, 0.074613443, 0.064842635, 0.0595131033, 0.0617337415, 0.058624848, 0.059957231, 0.0652867626, 0.0715045495, 0.0675074008, 0.0728369324, 0.0728369324, 0.0732810601, 0.0750575706, 0.0728369324, 0.0723928048, 0.0723928048, 0.0675074008, ...]</td>
      <td>[0.073827654, 0.04537511, 0.022077948, 0.027527481, 0.021322101, 0.022883624, 0.02370426, 0.030836225, 0.042639196, 0.05585119, 0.07217789, 0.08134478, 0.08961046, 0.103625864, 0.12554172, 0.13468736, 0.16203326, 0.17829657, 0.21441683, 0.20686802, 0.22815448, 0.228313, 0.25762054, 0.24422944, 0.2527454, 0.24923241, 0.25063625, 0.22561434, 0.19528994, 0.1826241, 0.13989443, 0.11508188, 0.05944249, 0.0459204, 0.01766479, 0.011624932, 0.0019857585, 0.001327306, 0.0003760457, 0.00037619472, 5.8710575e-05, 6.005168e-05, 2.2798777e-05, 4.0113926e-05, 1.4781952e-05, 2.2858381e-05, 1.79708e-05, 3.990531e-05, 3.2246113e-05, 6.175041e-05, 4.014373e-05, 6.827712e-05, 2.8342009e-05, 4.389882e-05, 1.7046928e-05, 2.7179718e-05, 6.2286854e-06, 8.434057e-06, 2.5331974e-06, 5.453825e-06, 1.1920929e-06, 1.9669533e-06, 7.748604e-07, 1.9967556e-06, 7.748604e-07, 1.5497208e-06, 8.34465e-07, 2.2649765e-06, 1.1622906e-06, 2.503395e-06, 1.4305115e-06, 4.142523e-06, 2.4735928e-06, 5.096197e-06, 3.0696392e-06, 8.046627e-06, 5.185604e-06, 1.1056662e-05, 7.7188015e-06, 1.8835068e-05, 1.3649464e-05, 2.8192997e-05, 2.1010637e-05, 5.221367e-05, 4.3094158e-05, 8.9138746e-05, 6.3210726e-05, 0.00012665987, 6.726384e-05, 0.00010934472, 5.0485134e-05, 9.301305e-05, 3.7252903e-05, 5.8323145e-05, 2.5242567e-05, 4.926324e-05, 1.8715858e-05, 3.0636787e-05, 1.2457371e-05, 2.6792288e-05, ...]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

## Cleaning up.

Now that the workshop is complete, don't forget to undeploy your pipeline to free up the resources.

```python
pipeline.undeploy()
```

<table><tr><th>name</th> <td>cv-mitochondria</td></tr><tr><th>created</th> <td>2023-09-28 20:25:17.379732+00:00</td></tr><tr><th>last_updated</th> <td>2023-09-29 19:07:47.787981+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>ac4bd826-f47f-48b7-8319-01fbc6622899, 4b6dab7d-b3ff-4f14-8425-7d9b6de76637, 66e72bc0-a3e3-4872-bc20-19b992c194b4, cf4bdfb4-1eec-46f8-9af4-b16dea894de6, 507cb1eb-8034-4b5b-bc96-2427730a6407, 50ed1d2f-6dba-411c-9579-1090791b33bd, 6208c9cf-fcd6-4b20-bbfc-f6ce714596e3</td></tr><tr><th>steps</th> <td>mitochondria-detector</td></tr><tr><th>published</th> <td>False</td></tr></table>

## Congratulations!

In this workshop you have:

* Deployed a single step house price prediction pipeline and sent data to it.
* Create a new Wallaroo connection.
* Assigned the connection to a workspace.
* Retrieved the connection from the workspace.
* Used the data connection to retrieve information from outside of Wallaroo, and use it for an inference.

Great job!
