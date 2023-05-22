This can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/pipeline-orchestrators/orchestration_sdk_comprehensive_tutorial).

## Pipeline Orchestration Comprehensive Tutorial

This tutorial provides a complete set of methods and examples regarding Wallaroo Connections and Wallaroo ML Workload Orchestration.

Wallaroo provides data connections, orchestrations, and tasks to provide organizations with a method of creating and managing automated tasks that can either be run on demand, on a regular schedule, or as a service so they respond to requests.

| Object | Description |
|---|---|
| Orchestration | A set of instructions written as a python script with a requirements library.  Orchestrations are uploaded to the Wallaroo instance |
| Task | An implementation of an orchestration.  Tasks are run either once when requested, on a repeating schedule, or as a service. |
| Connection | Definitions set by MLOps engineers that are used by other Wallaroo users for connection information to a data source.  Usually paired with orchestrations. |

A typical flow in the orchestration, task and connection life cycle is:

1. (Optional) A connection is defined with information such as username, connection URL, tokens, etc.
1. One or more connections are applied to a workspace for users to implement in their code or orchestrations.
1. An orchestration is created to perform some set instructions.  For example:
    1. Deploy a pipeline, request data from an external service, store the results in an external database, then undeploy the pipeline.
    1. Download a ML Model then replace a current pipeline step with the new version.
    1. Collect log files from a deployed pipeline once every hour and submit it to a Kafka or other service.
1. A task is created that specifies the orchestration to perform and the schedule:
    1. Run once.
    1. Run on a schedule (based on `cron` like settings).
    1. Run as a service to be run whenever requested.
1. Once the use for a task is complete, it is killed and its schedule or service removed.

## Tutorial Goals

The tutorial will demonstrate the following:

1. Create a simple connection to retrieve an Apache Arrow table file from a GitHub registry.
1. Create an orchestration that retrieves the Apache Arrow table file from the location defined by the connection, deploy a pipeline, perform an inference, then undeploys the pipeline.
1. Implement the orchestration as a task that runs every minute.
1. Display the logs from the pipeline after 5 minutes to verify the task is running.

## Tutorial Required Libraries

The following libraries are required for this tutorial, and included by default in a Wallaroo instance's JupyterHub service.

* **IMPORTANT NOTE**:  These libraries are already installed in the Wallaroo JupyterHub service.  Do not uninstall and reinstall the Wallaroo SDK with the command below.

* [wallaroo](https://pypi.org/project/wallaroo/):  The Wallaroo SDK.
* [pandas](https://pypi.org/project/pandas/): The pandas data analysis library.
* [pyarrow](https://pypi.org/project/pyarrow/): The Apache Arrow Python library.

The specific versions used are set in the file `./resources/requirements.txt`.  Supported libraries are automatically installed with the `pypi` or `conda` commands.  For example, from the root of this tutorials folder:

```python
pip install -r ./resources/requirements.txt
```

## Initialization

The first step is to connect to a Wallaroo instance.  We'll load the libraries and set our client connection settings

### Workspace, Model and Pipeline Setup

For this tutorial, we'll create a workspace, upload our sample model and deploy a pipeline.  We'll perform some quick sample inferences to verify that everything it working.

```python
import wallaroo
from wallaroo.object import EntityNotFoundError

# to display dataframe tables
from IPython.display import display
# used to display dataframe information without truncating
import pandas as pd
pd.set_option('display.max_colwidth', None)
import pyarrow as pa

import requests

# Used to create unique workspace and pipeline names
import string
import random

# make a random 4 character prefix
suffix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))
display(suffix)
```

    'tgiq'

### Connect to the Wallaroo Instance

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  If logging in externally, update the `wallarooPrefix` and `wallarooSuffix` variables with the proper DNS information.  For more information on Wallaroo DNS settings, see the [Wallaroo DNS Integration Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-dns-guide/).

```python
# Login through local Wallaroo instance

wl = wallaroo.Client()
```

    Please log into the following URL in a web browser:
    
    	https://doc-test.keycloak.wallarooexample.ai/auth/realms/master/device?user_code=XJMM-APPM
    
    Login successful!

```python
# Setting variables for later steps

workspace_name = f'orchestrationworkspace{suffix}'
pipeline_name = f'orchestrationpipeline{suffix}'
model_name = f'orchestrationmodel{suffix}'
model_file_name = './models/rf_model.onnx'
connection_name = f'houseprice_arrow_table{suffix}'
```

### Helper Methods

The following helper methods are used to either create or get workspaces and pipelines.

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

housing_model_control = wl.upload_model(model_name, model_file_name).configure()

# Add the model as a pipeline step

pipeline.add_model_step(housing_model_control)
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>orchestrationpipelinetgiq</td></tr><tr><th>created</th> <td>2023-05-22 19:54:06.933674+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-22 19:54:06.933674+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>ed5bf4b1-1d5d-4ff9-8a23-2c1e44a8e672</td></tr><tr><th>steps</th> <td></td></tr></table>
{{</table>}}

```python
#deploy the pipeline
pipeline.deploy()
```

    Waiting for deployment - this will take up to 45s ........................ ok

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>orchestrationpipelinetgiq</td></tr><tr><th>created</th> <td>2023-05-22 19:54:06.933674+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-22 19:54:08.008312+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>a7336408-20ef-4b65-8167-c2f80c968a21, ed5bf4b1-1d5d-4ff9-8a23-2c1e44a8e672</td></tr><tr><th>steps</th> <td>orchestrationmodeltgiq</td></tr></table>
{{</table>}}

### Sample Inferences

We'll perform some quick sample inferences using an Apache Arrow table as the input.  Once that's finished, we'll undeploy the pipeline and return the resources back to the Wallaroo instance.

```python
# sample inferences

batch_inferences = pipeline.infer_from_file('./data/xtest-1k.arrow')

large_inference_result =  batch_inferences.to_pandas()
display(large_inference_result.head(20))
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
      <td>2023-05-22 19:54:33.671</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.75]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-05-22 19:54:33.671</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[615094.56]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-05-22 19:54:33.671</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
      <td>[448627.72]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-05-22 19:54:33.671</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[758714.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-05-22 19:54:33.671</td>
      <td>[3.0, 1.75, 2200.0, 11520.0, 1.0, 0.0, 0.0, 4.0, 7.0, 2200.0, 0.0, 47.7659, -122.341, 1690.0, 8038.0, 62.0, 0.0, 0.0]</td>
      <td>[513264.7]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2023-05-22 19:54:33.671</td>
      <td>[3.0, 2.0, 2140.0, 4923.0, 1.0, 0.0, 0.0, 4.0, 8.0, 1070.0, 1070.0, 47.6902, -122.339, 1470.0, 4923.0, 86.0, 0.0, 0.0]</td>
      <td>[668288.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2023-05-22 19:54:33.671</td>
      <td>[4.0, 3.5, 3590.0, 5334.0, 2.0, 0.0, 2.0, 3.0, 9.0, 3140.0, 450.0, 47.6763, -122.267, 2100.0, 6250.0, 9.0, 0.0, 0.0]</td>
      <td>[1004846.5]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2023-05-22 19:54:33.671</td>
      <td>[3.0, 2.0, 1280.0, 960.0, 2.0, 0.0, 0.0, 3.0, 9.0, 1040.0, 240.0, 47.602, -122.311, 1280.0, 1173.0, 0.0, 0.0, 0.0]</td>
      <td>[684577.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2023-05-22 19:54:33.671</td>
      <td>[4.0, 2.5, 2820.0, 15000.0, 2.0, 0.0, 0.0, 4.0, 9.0, 2820.0, 0.0, 47.7255, -122.101, 2440.0, 15000.0, 29.0, 0.0, 0.0]</td>
      <td>[727898.1]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2023-05-22 19:54:33.671</td>
      <td>[3.0, 2.25, 1790.0, 11393.0, 1.0, 0.0, 0.0, 3.0, 8.0, 1790.0, 0.0, 47.6297, -122.099, 2290.0, 11894.0, 36.0, 0.0, 0.0]</td>
      <td>[559631.1]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2023-05-22 19:54:33.671</td>
      <td>[3.0, 1.5, 1010.0, 7683.0, 1.5, 0.0, 0.0, 5.0, 7.0, 1010.0, 0.0, 47.72, -122.318, 1550.0, 7271.0, 61.0, 0.0, 0.0]</td>
      <td>[340764.53]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2023-05-22 19:54:33.671</td>
      <td>[3.0, 2.0, 1270.0, 1323.0, 3.0, 0.0, 0.0, 3.0, 8.0, 1270.0, 0.0, 47.6934, -122.342, 1330.0, 1323.0, 8.0, 0.0, 0.0]</td>
      <td>[442168.06]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2023-05-22 19:54:33.671</td>
      <td>[4.0, 1.75, 2070.0, 9120.0, 1.0, 0.0, 0.0, 4.0, 7.0, 1250.0, 820.0, 47.6045, -122.123, 1650.0, 8400.0, 57.0, 0.0, 0.0]</td>
      <td>[630865.6]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2023-05-22 19:54:33.671</td>
      <td>[4.0, 1.0, 1620.0, 4080.0, 1.5, 0.0, 0.0, 3.0, 7.0, 1620.0, 0.0, 47.6696, -122.324, 1760.0, 4080.0, 91.0, 0.0, 0.0]</td>
      <td>[559631.1]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2023-05-22 19:54:33.671</td>
      <td>[4.0, 3.25, 3990.0, 9786.0, 2.0, 0.0, 0.0, 3.0, 9.0, 3990.0, 0.0, 47.6784, -122.026, 3920.0, 8200.0, 10.0, 0.0, 0.0]</td>
      <td>[909441.1]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2023-05-22 19:54:33.671</td>
      <td>[4.0, 2.0, 1780.0, 19843.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1780.0, 0.0, 47.4414, -122.154, 2210.0, 13500.0, 52.0, 0.0, 0.0]</td>
      <td>[313096.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2023-05-22 19:54:33.671</td>
      <td>[4.0, 2.5, 2130.0, 6003.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2130.0, 0.0, 47.4518, -122.12, 1940.0, 4529.0, 11.0, 0.0, 0.0]</td>
      <td>[404040.8]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2023-05-22 19:54:33.671</td>
      <td>[3.0, 1.75, 1660.0, 10440.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1040.0, 620.0, 47.4448, -121.77, 1240.0, 10380.0, 36.0, 0.0, 0.0]</td>
      <td>[292859.5]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2023-05-22 19:54:33.671</td>
      <td>[3.0, 2.5, 2110.0, 4118.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2110.0, 0.0, 47.3878, -122.153, 2110.0, 4044.0, 25.0, 0.0, 0.0]</td>
      <td>[338357.88]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2023-05-22 19:54:33.671</td>
      <td>[4.0, 2.25, 2200.0, 11250.0, 1.5, 0.0, 0.0, 5.0, 7.0, 1300.0, 900.0, 47.6845, -122.201, 2320.0, 10814.0, 94.0, 0.0, 0.0]</td>
      <td>[682284.6]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}

## Create Wallaroo Connection

Connections are created at the Wallaroo instance level, typically by a MLOps or DevOps engineer, then applied to a workspace.

For this section:

1. We will create a sample connection that just has a URL to the same Arrow table file we used in the previous step.
1. We'll apply the data connection to the workspace above.
1. For a quick demonstration, we'll use the connection to retrieve the Arrow table file and use it for a quick sample inference.

### Create Connection

Connections are created with the Wallaroo client command [`create_connection`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-dataconnections/#create-data-connection) with the following parameters.

| Parameter | Type | Description |
| --- | --- | ---|
| **name** | string (Required) | The name of the connection. This must be unique - if submitting the name of an existing connection it will return an error. |
| **type** | string (Required) | The user defined type of connection. |
| **details** | Dict (Requires) | User defined configuration details for the data connection.  These can be `{'username':'dataperson', 'password':'datapassword', 'port': 3339}`, or `{'token':'abcde123==', 'host':'example.com', 'port:1234'}`, or other user defined combinations.  |

We'll create the connection named `houseprice_arrow_table`, set it to the type `HTTPFILE`, and provide the details as `'host':'https://github.com/WallarooLabs/Wallaroo_Tutorials/raw/main/wallaroo-testing-tutorials/houseprice-saga/data/xtest-1k.arrow'` - the location for our sample Arrow table inference input.

```python
wl.create_connection(connection_name, 
                  "HTTPFILE", 
                  {'host':'https://github.com/WallarooLabs/Wallaroo_Tutorials/raw/main/wallaroo-testing-tutorials/houseprice-saga/data/xtest-1k.arrow'}
                  )
```

{{<table "table table-striped table-bordered" >}}
<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Name</td><td>houseprice_arrow_tabletgiq</td>
  </tr>
  <tr>
    <td>Connection Type</td><td>HTTPFILE</td>
  </tr>
  <tr>
    <td>Details</td><td>*****</td>
  </tr>
  <tr>
    <td>Created At</td><td>2023-05-22T19:54:33.723860+00:00</td>
  </tr>
  <tr>
    <td>Linked Workspaces</td><td>[]</td>
  </tr>
</table>
{{</table>}}

### List Data Connections

The Wallaroo Client [`list_connections()`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-dataconnections/#list-data-connections) method lists all connections for the Wallaroo instance.

```python
wl.list_connections()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th><th>connection type</th><th>details</th><th>created at</th><th>linked workspaces</th></tr><tr><td>houseprice_arrow_tabletgiq</td><td>HTTPFILE</td><td>*****</td><td>2023-05-22T19:54:33.723860+00:00</td><td>[]</td></tr></table>
{{</table>}}

### Add Connection to Workspace

The method Workspace [`add_connection(connection_name)`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-dataconnections/#add-data-connection-to-workspace) adds a Data Connection to a workspace, and takes the following parameters.

| Parameter | Type | Description |
| --- | --- | ---|
| **name** | string (Required) | The name of the Data Connection |

We'll add this connection to our sample workspace.

```python
workspace.add_connection(connection_name)
```

### List Connections in a Workspace

The method Workspace `list_connections()` displays a list of connections attached to the workspace.  By default the `details` field is obfuscated.  We'll list the connections in our sample workspace, then use that to retrieve our specific connection.

```python
workspace.list_connections()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th><th>connection type</th><th>details</th><th>created at</th><th>linked workspaces</th></tr><tr><td>houseprice_arrow_tabletgiq</td><td>HTTPFILE</td><td>*****</td><td>2023-05-22T19:54:33.723860+00:00</td><td>['orchestrationworkspacetgiq']</td></tr></table>
{{</table>}}

### Get Connection

Connections are retrieved by the Wallaroo Client `get_connection(name)` method.

```python
connection = wl.get_connection(connection_name)
```

### Connection Details

The Connection method `details()` retrieves a the connection `details()` as a `dict`.

```python
display(connection.details())
```

    {'host': 'https://github.com/WallarooLabs/Wallaroo_Tutorials/raw/main/wallaroo-testing-tutorials/houseprice-saga/data/xtest-1k.arrow'}

### Using a Connection Example

For this example, the connection will be used to retrieve the Apache Arrow file referenced in the connection, and use that to turn it into an Apache Arrow table, then use that for a sample inference.

```python
# Deploy the pipeline 
pipeline.deploy()

# Retrieve the file
# set accept as apache arrow table
headers = {
    'Accept': 'application/vnd.apache.arrow.file'
}

response = requests.get(
                    connection.details()['host'], 
                    headers=headers
                )

# Arrow table is retrieved 
with pa.ipc.open_file(response.content) as reader:
    arrow_table = reader.read_all()

results = pipeline.infer(arrow_table)

result_table = results.to_pandas()
display(result_table.head(20))
```

     ok

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
      <td>2023-05-22 19:54:34.320</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.75]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-05-22 19:54:34.320</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[615094.56]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-05-22 19:54:34.320</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
      <td>[448627.72]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-05-22 19:54:34.320</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[758714.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-05-22 19:54:34.320</td>
      <td>[3.0, 1.75, 2200.0, 11520.0, 1.0, 0.0, 0.0, 4.0, 7.0, 2200.0, 0.0, 47.7659, -122.341, 1690.0, 8038.0, 62.0, 0.0, 0.0]</td>
      <td>[513264.7]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2023-05-22 19:54:34.320</td>
      <td>[3.0, 2.0, 2140.0, 4923.0, 1.0, 0.0, 0.0, 4.0, 8.0, 1070.0, 1070.0, 47.6902, -122.339, 1470.0, 4923.0, 86.0, 0.0, 0.0]</td>
      <td>[668288.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2023-05-22 19:54:34.320</td>
      <td>[4.0, 3.5, 3590.0, 5334.0, 2.0, 0.0, 2.0, 3.0, 9.0, 3140.0, 450.0, 47.6763, -122.267, 2100.0, 6250.0, 9.0, 0.0, 0.0]</td>
      <td>[1004846.5]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2023-05-22 19:54:34.320</td>
      <td>[3.0, 2.0, 1280.0, 960.0, 2.0, 0.0, 0.0, 3.0, 9.0, 1040.0, 240.0, 47.602, -122.311, 1280.0, 1173.0, 0.0, 0.0, 0.0]</td>
      <td>[684577.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2023-05-22 19:54:34.320</td>
      <td>[4.0, 2.5, 2820.0, 15000.0, 2.0, 0.0, 0.0, 4.0, 9.0, 2820.0, 0.0, 47.7255, -122.101, 2440.0, 15000.0, 29.0, 0.0, 0.0]</td>
      <td>[727898.1]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2023-05-22 19:54:34.320</td>
      <td>[3.0, 2.25, 1790.0, 11393.0, 1.0, 0.0, 0.0, 3.0, 8.0, 1790.0, 0.0, 47.6297, -122.099, 2290.0, 11894.0, 36.0, 0.0, 0.0]</td>
      <td>[559631.1]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2023-05-22 19:54:34.320</td>
      <td>[3.0, 1.5, 1010.0, 7683.0, 1.5, 0.0, 0.0, 5.0, 7.0, 1010.0, 0.0, 47.72, -122.318, 1550.0, 7271.0, 61.0, 0.0, 0.0]</td>
      <td>[340764.53]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2023-05-22 19:54:34.320</td>
      <td>[3.0, 2.0, 1270.0, 1323.0, 3.0, 0.0, 0.0, 3.0, 8.0, 1270.0, 0.0, 47.6934, -122.342, 1330.0, 1323.0, 8.0, 0.0, 0.0]</td>
      <td>[442168.06]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2023-05-22 19:54:34.320</td>
      <td>[4.0, 1.75, 2070.0, 9120.0, 1.0, 0.0, 0.0, 4.0, 7.0, 1250.0, 820.0, 47.6045, -122.123, 1650.0, 8400.0, 57.0, 0.0, 0.0]</td>
      <td>[630865.6]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2023-05-22 19:54:34.320</td>
      <td>[4.0, 1.0, 1620.0, 4080.0, 1.5, 0.0, 0.0, 3.0, 7.0, 1620.0, 0.0, 47.6696, -122.324, 1760.0, 4080.0, 91.0, 0.0, 0.0]</td>
      <td>[559631.1]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2023-05-22 19:54:34.320</td>
      <td>[4.0, 3.25, 3990.0, 9786.0, 2.0, 0.0, 0.0, 3.0, 9.0, 3990.0, 0.0, 47.6784, -122.026, 3920.0, 8200.0, 10.0, 0.0, 0.0]</td>
      <td>[909441.1]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2023-05-22 19:54:34.320</td>
      <td>[4.0, 2.0, 1780.0, 19843.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1780.0, 0.0, 47.4414, -122.154, 2210.0, 13500.0, 52.0, 0.0, 0.0]</td>
      <td>[313096.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2023-05-22 19:54:34.320</td>
      <td>[4.0, 2.5, 2130.0, 6003.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2130.0, 0.0, 47.4518, -122.12, 1940.0, 4529.0, 11.0, 0.0, 0.0]</td>
      <td>[404040.8]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2023-05-22 19:54:34.320</td>
      <td>[3.0, 1.75, 1660.0, 10440.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1040.0, 620.0, 47.4448, -121.77, 1240.0, 10380.0, 36.0, 0.0, 0.0]</td>
      <td>[292859.5]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2023-05-22 19:54:34.320</td>
      <td>[3.0, 2.5, 2110.0, 4118.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2110.0, 0.0, 47.3878, -122.153, 2110.0, 4044.0, 25.0, 0.0, 0.0]</td>
      <td>[338357.88]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2023-05-22 19:54:34.320</td>
      <td>[4.0, 2.25, 2200.0, 11250.0, 1.5, 0.0, 0.0, 5.0, 7.0, 1300.0, 900.0, 47.6845, -122.201, 2320.0, 10814.0, 94.0, 0.0, 0.0]</td>
      <td>[682284.6]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}

### Remove Connection from Workspace

The Workspace method `remove_connection(connection_name)` removes the connection from the workspace, but does not delete the connection from the Wallaroo instance.  This method takes the following parameters.

| Parameter | Type | Description |
|---|---|---|
| **name** | String (Required) | The name of the connection to be removed 

The previous connection will be removed from the workspace, then the workspace connections displayed to verify it has been removed.

```python
workspace.remove_connection(connection_name)

display(workspace.list_connections())
```

(no connections)

### Delete Connection

The Connection method `delete_connection()` removes the connection from the Wallaroo instance, and all attachments in workspaces they were connected to.

```python
connection.delete_connection()

wl.list_connections()
```

(no connections)

## Orchestration Tutorial

The next series of examples will build on what we just did.  So far we have:

* Deployed a pipeline, performed sample inferences with a local Apache Arrow file, displayed the results, then undeployed the pipeline.
* Deployed a pipeline, use a Wallaroo connection details to retrieve a remote Apache Arrow file, performed inferences and displayed the results, then undeployed the pipeline.

For the orchestration tutorial, we'll do the same thing only package it into a separate python script and upload it to the Wallaroo instance, then create a task from that orchestration and perform our sample inferences again.

### Orchestration Requirements

Orchestrations are uploaded to the Wallaroo instance as a ZIP file with the following requirements:

* The ZIP file should not contain any directories - only files at the top level.

| Parameter | Type | Description |
|---|---|---|
| **User Code** | (**Required**) Python script as `.py` files | Python scripts for the orchestration to run.  If the file `main.py` exists, that will be the entrypoint.  Otherwise, if only one .py exists, then that will be the entrypoint. |
| Python Library Requirements | (**Required**) `requirements.txt` file in the [requirements file format](https://pip.pypa.io/en/stable/reference/requirements-file-format/).  This is in the root of the zip file, and there can only be one `requirements.txt` file for the orchestration. |
| Other artifacts | &nbsp; | Other artifacts such as files, data, or code to support the orchestration.

#### Zip Instructions

In a terminal with the `zip` command, assemble artifacts as above and then create the archive.  The `zip` command is included by default with the Wallaroo JupyterHub service.

`zip` commands take the following format, with `{zipfilename}.zip` as the zip file to save the artifacts to, and each file thereafter as the files to add to the archive.

```bash
zip {zipfilename}.zip file1, file2, file3....
```

For example, the following command will add the files `main.py` and `requirements.txt` into the file `hello.zip`.

```shell
$ zip hello.zip main.py requirements.txt 
  adding: main.py (deflated 47%)
  adding: requirements.txt (deflated 52%)
```

### Orchestration Recommendations

The following recommendations will make using Wallaroo orchestrations 

* The version of Python used should match the same version as in the Wallaroo JupyterHub service.
* The same version of the [Wallaroo SDK](https://pypi.org/project/wallaroo/) should match the server.  For a 2023.2 Wallaroo instance, use the Wallaroo SDK version 2023.2.
* Specify the version of `pip` dependencies.
* The `wallaroo.Client` constructor `auth_type` argument is ignored.  Using `wallaroo.Client()` is sufficient.
* The following methods will assist with orchestrations:
  * `wallaroo.in_task()` :  Returns `True` if the code is running within an Orchestrator task.
  * `wallaroo.task_args()`:  Returns a `Dict` of invocation-specific arguments passed to the `run_` calls.
* Use `print` commands so outputs are saved to the task's log files.

### Example requirements.txt file

```python
dbt-bigquery==1.4.3
dbt-core==1.4.5
dbt-extractor==0.4.1
dbt-postgres==1.4.5
google-api-core==2.8.2
google-auth==2.11.0
google-auth-oauthlib==0.4.6
google-cloud-bigquery==3.3.2
google-cloud-bigquery-storage==2.15.0
google-cloud-core==2.3.2
google-cloud-storage==2.5.0
google-crc32c==1.5.0
google-pasta==0.2.0
google-resumable-media==2.3.3
googleapis-common-protos==1.56.4
```

### Sample Orchestrator

The following orchestrator artifacts are in the directory `./remote_inference` and includes the file `main.py` with the following code:

```python
import wallaroo
from wallaroo.object import EntityNotFoundError
import pandas as pd
import pyarrow as pa
import requests

wl = wallaroo.Client()

# Setting variables for later steps

# get the arguments
arguments = wl.task_args()

if "workspace_name" in arguments:
    workspace_name = arguments['workspace_name']
else:
    workspace_name="orchestrationworkspace"

if "pipeline_name" in arguments:
    pipeline_name = arguments['pipeline_name']
else:
    pipeline_name="orchestrationpipeline"

if "connection_name" in arguments:
    connection_name = arguments['connection_name']
else:
    connection_name = "houseprice_arrow_table"

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

print(f"Getting the workspace {workspace_name}")
workspace = get_workspace(workspace_name)
wl.set_current_workspace(workspace)

print(f"Getting the pipeline {pipeline_name}")
pipeline = get_pipeline(pipeline_name)
pipeline.deploy()
# Get the connection - assuming it will be the only one

inference_source_connection = wl.get_connection(name=connection_name)

print(f"Getting arrow table file")
# Retrieve the file
# set accept as apache arrow table
headers = {
    'Accept': 'application/vnd.apache.arrow.file'
}

response = requests.get(
                    inference_source_connection.details()['host'], 
                    headers=headers
                )

# Arrow table is retrieved 
with pa.ipc.open_file(response.content) as reader:
    arrow_table = reader.read_all()

print("Inference time.  Displaying results after.")
# Perform the inference
result = pipeline.infer(arrow_table)
print(result)

pipeline.undeploy()
```

This is saved to the file `./remote_inference/remote_inference.zip`.

### Preparing the Wallaroo Instance

To prepare the Wallaroo instance, we'll once again create the Wallaroo connection `houseprice_arrow_table` and apply it to the workspace.

```python
wl.create_connection(connection_name, 
                  "HTTPFILE", 
                  {'host':'https://github.com/WallarooLabs/Wallaroo_Tutorials/raw/main/wallaroo-testing-tutorials/houseprice-saga/data/xtest-1k.arrow'}
                  )

workspace.add_connection(connection_name)
```

### Upload the Orchestration

Orchestrations are uploaded with the Wallaroo client `upload_orchestration(path)` method with the following parameters.

| Parameter | Type | Description |
| --- | --- | ---|
| **path** | string (Required) | The path to the ZIP file to be uploaded. |

Once uploaded, the deployment will be prepared and any requirements will be downloaded and installed.

For this example, the orchestration `./remote_inference/remote_inference.zip` will be uploaded and saved to the variable `orchestration`.

```python
orchestration = wl.upload_orchestration(name="comprehensive sample", path="./remote_inference/remote_inference.zip")
```

### Orchestration Status

The Orchestration method `status()` displays the current status of the uploaded orchestration.

| Status | Description |
|---|---|
|---|---|
| `pending_packaging` | The orchestration is uploaded, but packaging hasn't started yet. |
| `packaging` | The orchestration is being packaged for use with the Wallaroo instance. |
| `ready` | The orchestration is ready for use. |

For this example, the status of the orchestration will be displayed then looped until it has reached status `ready`.

```python
import time

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

### List Orchestrations

Orchestrations are listed with the Wallaroo Client `list_orchestrations()` which returns a list of available orchestrations.

```python
wl.list_orchestrations()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>id</th><th>name</th><th>status</th><th>filename</th><th>sha</th><th>created at</th><th>updated at</th></tr><tr><td>0f90e606-09f8-409b-a306-cb04ec4c011a</td><td>comprehensive sample</td><td>ready</td><td>remote_inference.zip</td><td>b88e93...2396fb</td><td>2023-22-May 19:55:15</td><td>2023-22-May 19:56:09</td></tr></table>
{{</table>}}

## Task Management Tutorial

Once an Orchestration has the status `ready`, it can be run as a task.  Tasks have three run options.

| Type       | SDK Call |  How triggered                                                               | Purpose                                                       |
|------------|----------|:------------------------------------------------------------------------------|:---------------------------------------------------------------|
| Once       | `orchestration.run_once(name, json_args, timeout)` | Task runs once and exits.| Single batch, experimentation                                 |
| Scheduled  | `orchestration.run_scheduled()` | User provides schedule. Task runs exits whenever schedule dictates. | Recurrent batch ETL.|

### Task Run Once

Tasks are generated and run once with the Orchestration `run_once(name, json_args, timeout)` method.  Any arguments for the orchestration are passed in as a `Dict`.  If there are no arguments, then an empty set `{}` is passed.

For our example, we will pass the workspace, pipeline, and connection into our task.

```python
# Example: run once

import datetime
task_start = datetime.datetime.now()

task = orchestration.run_once(name="house price run once 2", json_args={"workspace_name": workspace_name, 
                                                                           "pipeline_name":pipeline_name,
                                                                           "connection_name": connection_name
                                                                           }
                            )
task
```

{{<table "table table-striped table-bordered" >}}
<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>ID</td><td>f0e27d6a-6a98-4d26-b240-266f08560c48</td>
  </tr>
  <tr>
    <td>Name</td><td>house price run once 2</td>
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
    <td>Created At</td><td>2023-22-May 19:58:32</td>
  </tr>
  <tr>
    <td>Updated At</td><td>2023-22-May 19:58:32</td>
  </tr>
</table>
{{</table>}}

```python
taskfail.last_runs()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>task id</th><th>pod id</th><th>status</th><th>created at</th><th>updated at</th></tr>
            <tr>
              <td>5ee51c78-a1c6-41e4-86a6-77110ce26161</td>
              <td>844902e0-5ff3-4c15-b497-e173aa3ce0d5</td>
              <td>running</td>
              <td>2023-22-May 20:15:08</td>
              <td>2023-22-May 20:15:08</td>
            </tr>
            </table>
{{</table>}}

### List Tasks

The list of tasks in the Wallaroo instance is retrieved through the Wallaroo Client `list_tasks()` method that accepts the following parameters.

| Parameter | Type | Description |
| --- | --- | ---|
| killed | **Boolean** (*Optional* *Default: `False`*) | Returns tasks depending on whether they have been issued the `kill` command.  `False` returns all tasks whether killed or not.  `True` only returns killed tasks. |

This returns an array list of the following in reverse chronological order from `updated at`.

| Parameter | Type | Description |
| --- | --- | ---|
| **id** | string | The UUID identifier for the task. |
| **last run status** | string | The last reported status the task.  Values are: <br><ul><li>`unknown`: The task has not been started or is being prepared.</li><li>`ready`: The task is scheduled to execute.</li><li>`running`: The task has started.</li><li>`failure`: The task failed.</li><li>`success`: The task completed.</ul> |
| **type** | string | The type of the task.  Values are: <br><ul><li>`Temporary Run`: The task runs once then stop.</li><li>`Scheduled Run`: The task repeats on a `cron` like schedule.</li><li>`Service Run`: The task runs as a service and executes when its service port is activated. |
| **active** | Boolean | `True`: The task is scheduled or running. `False`: The task has completed or has been issued the `kill` command. |
| **schedule** | string | The `cron` style schedule for the task.  If the task is not a scheduled one, then the schedule will be `-`. |
| **created at** | DateTime | The date and time the task was started. |
| **updated at** | DateTime | The date and time the task was updated. |

```python
wl.list_tasks()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>id</th><th>name</th><th>last run status</th><th>type</th><th>active</th><th>schedule</th><th>created at</th><th>updated at</th></tr><tr><td>f0e27d6a-6a98-4d26-b240-266f08560c48</td><td>house price run once 2</td><td>running</td><td>Temporary Run</td><td>True</td><td>-</td><td>2023-22-May 19:58:32</td><td>2023-22-May 19:58:38</td></tr><tr><td>36509ef8-98da-42a0-913f-e6e929dedb15</td><td>house price run once</td><td>success</td><td>Temporary Run</td><td>True</td><td>-</td><td>2023-22-May 19:56:37</td><td>2023-22-May 19:56:48</td></tr></table>
{{</table>}}

### Task Status

The status of the task is returned with the Task `status()` method that returned the tasks status.  Tasks can have the following status.

* `pending`: The task has not been started or is being prepared.
* `started`: The task has started to execute.

```python
while task.status() != "started":
    display(task.status())
    time.sleep(5)
```

### Task Last Runs History

The history of a task, which each deployment of the task is known as a **task run** is retrieved with the Task `last_runs` method that takes the following arguments.

| Parameter | Type | Description |
| --- | --- | ---|
| status | **String** (*Optional* *Default: `all`) | Filters the task history by the `status`.  If `all`, returns all statuses.  Status values are: <br><ul><li>`running`: The task has started.</li><li>`failure`: The task failed.</li><li>`success`: The task completed.</li></ul> | 
| limit | **Integer** (*Optional*) | Limits the number of task runs returned. |

This returns the following in reverse chronological order by `updated at`.

| Parameter | Type | Description |
| --- | --- | ---|
| task id | string | Task id in UUID format. |
| pod id | string | Pod id in UUID format. |
| status | string | Status of the task.  Status values are: <br><ul><li>`running`: The task has started.</li><li>`failure`: The task failed.</li><li>`success`: The task completed.</li></ul> |
| created at | DateTime | Date and time the task was created at. |
| updated at | DateTime | Date and time the task was updated. |

```python
task.last_runs()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>task id</th><th>pod id</th><th>status</th><th>created at</th><th>updated at</th></tr>
            <tr>
              <td>f0e27d6a-6a98-4d26-b240-266f08560c48</td>
              <td>7d9d73d5-df11-44ed-90c1-db0e64c7f9b8</td>
              <td>success</td>
              <td>2023-22-May 19:58:35</td>
              <td>2023-22-May 19:58:35</td>
            </tr>
            </table>
{{</table>}}

### Task Run Logs

The output of a task is displayed with the Task Run `logs()` method that takes the following parameters.

| Parameter | Type | Description |
| --- | --- | ---|
| limit | **Integer** (*Optional*) | Limits the lines returned from the task run log.  The `limit` parameter is based on the log tail - starting from the last line of the log file, then working up until the limit of lines is reached.  This is useful for viewing final outputs, exceptions, etc. |

The Task Run `logs()` returns the log entries as a string list, with each entry as an item in the list.

* **IMPORTANT NOTE**: It may take around a minute for task run logs to be integrated into the Wallaroo log database.

```python
# give time for the task to complete and the log files entered
time.sleep(60)
recent_run = task.last_runs()[0]
display(recent_run.logs())
```

<pre><code>2023-22-May 19:59:29 Getting the workspace orchestrationworkspacetgiq
2023-22-May 19:59:29 Getting the pipeline orchestrationpipelinetgiq
2023-22-May 19:59:29 Getting arrow table file
2023-22-May 19:59:29 Inference time.  Displaying results after.
2023-22-May 19:59:29 pyarrow.Table
2023-22-May 19:59:29 time: timestamp[ms]
2023-22-May 19:59:29 in.tensor: list<item: float> not null
2023-22-May 19:59:29   child 0, item: float
2023-22-May 19:59:29 out.variable: list<inner: float not null> not null
2023-22-May 19:59:29 check_failures: int8
2023-22-May 19:59:29   child 0, inner: float not null
2023-22-May 19:59:29 ----
2023-22-May 19:59:29 time: [[2023-05-22 19:58:49.767,2023-05-22 19:58:49.767,2023-05-22 19:58:49.767,2023-05-22 19:58:49.767,2023-05-22 19:58:49.767,...,2023-05-22 19:58:49.767,2023-05-22 19:58:49.767,2023-05-22 19:58:49.767,2023-05-22 19:58:49.767,2023-05-22 19:58:49.767]]
2023-22-May 19:59:29 in.tensor: [[[4,2.5,2900,5505,2,...,2970,5251,12,0,0],[2,2.5,2170,6361,1,...,2310,7419,6,0,0],...,[3,1.75,2910,37461,1,...,2520,18295,47,0,0],[3,2,2005,7000,1,...,1750,4500,34,0,0]]]
2023-22-May 19:59:29 check_failures: [[0,0,0,0,0,...,0,0,0,0,0]]
2023-22-May 19:59:29 out.variable: [[[718013.75],[615094.56],...,[706823.56],[581003]]]</code></pre>

#### Failed Task Logs

We can create a task that fails and show it in the `last_runs` list, then retrieve the logs to display why it failed.

```python
# Example: run once

import datetime
task_start = datetime.datetime.now()

taskfail = orchestration.run_once(name="house price run once 2", json_args={"workspace_name": "bob", 
                                                                           "pipeline_name":"does not exist",
                                                                           "connection_name": connection_name
                                                                           }
                            )

while taskfail.status() != "started":
    display(taskfail.status())
    time.sleep(5)
```

{{<table "table table-striped table-bordered" >}}
<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>ID</td><td>54229bd2-c388-4196-9ce2-f76503a27f99</td>
  </tr>
  <tr>
    <td>Name</td><td>house price run once 2</td>
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
    <td>Created At</td><td>2023-22-May 20:17:16</td>
  </tr>
  <tr>
    <td>Updated At</td><td>2023-22-May 20:17:16</td>
  </tr>
</table>
{{</table>}}

```python
# time.sleep(60)
taskfail.last_runs()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>task id</th><th>pod id</th><th>status</th><th>created at</th><th>updated at</th></tr>
            <tr>
              <td>54229bd2-c388-4196-9ce2-f76503a27f99</td>
              <td>79d7fe3e-ac8c-4f9c-8288-c9c207fb0a5e</td>
              <td>failure</td>
              <td>2023-22-May 20:17:18</td>
              <td>2023-22-May 20:17:18</td>
            </tr>
            </table>
{{</table>}}

```python
# time.sleep(60)
taskfaillogs = taskfail.last_runs()[0].logs()
display(taskfaillogs)
```

<pre><code>2023-22-May 20:17:22 Getting the workspace bob
2023-22-May 20:17:22   File "/home/jovyan/main.py", line 43, in get_pipeline
2023-22-May 20:17:22 Traceback (most recent call last):
2023-22-May 20:17:22 Getting the pipeline does not exist
2023-22-May 20:17:22     pipeline = wl.pipelines_by_name(name)[0]
2023-22-May 20:17:22   File "/home/jovyan/venv/lib/python3.9/site-packages/wallaroo/client.py", line 1064, in pipelines_by_name
2023-22-May 20:17:22     raise EntityNotFoundError("Pipeline", {"pipeline_name": pipeline_name})
2023-22-May 20:17:22 wallaroo.object.EntityNotFoundError: Pipeline not found: {'pipeline_name': 'does not exist'}
2023-22-May 20:17:22 
2023-22-May 20:17:22 During handling of the above exception, another exception occurred:
2023-22-May 20:17:22 Traceback (most recent call last):
2023-22-May 20:17:22 
2023-22-May 20:17:22   File "/home/jovyan/main.py", line 54, in <module>
2023-22-May 20:17:22     pipeline = get_pipeline(pipeline_name)
2023-22-May 20:17:22     pipeline = wl.build_pipeline(name)
2023-22-May 20:17:22   File "/home/jovyan/main.py", line 45, in get_pipeline
2023-22-May 20:17:22   File "/home/jovyan/venv/lib/python3.9/site-packages/wallaroo/client.py", line 1102, in build_pipeline
2023-22-May 20:17:22     require_dns_compliance(pipeline_name)
2023-22-May 20:17:22   File "/home/jovyan/venv/lib/python3.9/site-packages/wallaroo/checks.py", line 274, in require_dns_compliance
2023-22-May 20:17:22 wallaroo.object.InvalidNameError: Name 'does not exist is invalid: must be DNS-compatible (ASCII alpha-numeric plus dash (-))
2023-22-May 20:17:22     raise InvalidNameError(</code></pre>

### Task Results

We can view the inferences from our logs and verify that new entries were added from our task.  In our case, we'll assume the task once started takes about 1 minute to run (deploy the pipeline, run the inference, undeploy the pipeline).  We'll add in a wait of 1 minute, then display the logs during the time period the task was running.

```python
task_end = datetime.datetime.now()
display(task_end)

pipeline.logs(start_datetime = task_start, end_datetime = task_end)
```

    datetime.datetime(2023, 5, 22, 20, 1, 25, 418564)

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
      <td>2023-05-22 19:58:49.767</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.75]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-05-22 19:58:49.767</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[615094.56]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-05-22 19:58:49.767</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
      <td>[448627.72]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-05-22 19:58:49.767</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[758714.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-05-22 19:58:49.767</td>
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
      <th>501</th>
      <td>2023-05-22 19:58:49.767</td>
      <td>[3.0, 2.5, 1570.0, 1433.0, 3.0, 0.0, 0.0, 3.0, 8.0, 1570.0, 0.0, 47.6858, -122.336, 1570.0, 2652.0, 4.0, 0.0, 0.0]</td>
      <td>[557391.25]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>502</th>
      <td>2023-05-22 19:58:49.767</td>
      <td>[3.0, 2.5, 2390.0, 15669.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2390.0, 0.0, 47.7446, -122.193, 2640.0, 12500.0, 24.0, 0.0, 0.0]</td>
      <td>[741973.6]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>503</th>
      <td>2023-05-22 19:58:49.767</td>
      <td>[3.0, 0.75, 920.0, 20412.0, 1.0, 1.0, 2.0, 5.0, 6.0, 920.0, 0.0, 47.4781, -122.49, 1162.0, 54705.0, 64.0, 0.0, 0.0]</td>
      <td>[338418.8]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>504</th>
      <td>2023-05-22 19:58:49.767</td>
      <td>[4.0, 2.5, 2800.0, 246114.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2800.0, 0.0, 47.6586, -121.962, 2750.0, 60351.0, 15.0, 0.0, 0.0]</td>
      <td>[765468.75]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>505</th>
      <td>2023-05-22 19:58:49.767</td>
      <td>[2.0, 1.0, 1120.0, 9912.0, 1.0, 0.0, 0.0, 4.0, 6.0, 1120.0, 0.0, 47.3735, -122.43, 1540.0, 9750.0, 34.0, 0.0, 0.0]</td>
      <td>[309800.75]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}
<p>506 rows × 4 columns</p>

### Scheduled Tasks

Scheduled tasks are run with the Orchestration `run_scheduled` method.  We'll set it up to run every 5 minutes, then check the results.

It is recommended that orchestrations that have pipeline deploy or undeploy commands be spaced out no less than 5 minutes to prevent colliding with other tasks that use the same pipeline.

```python
task_start = datetime.datetime.now()
schedule = "*/5 * * * *"
task_scheduled = orchestration.run_scheduled(name="schedule example", 
                                             timeout=600, 
                                             schedule=schedule, 
                                             json_args={"workspace_name": workspace_name, 
                                                        "pipeline_name": pipeline_name,
                                                        "connection_name": connection_name
                                            })
while task_scheduled.status() != "started":
    display(task_scheduled.status())
    time.sleep(5)
task_scheduled
```

    'pending'

{{<table "table table-striped table-bordered" >}}
<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>ID</td><td>4af57c61-dfa9-43eb-944e-559135495df4</td>
  </tr>
  <tr>
    <td>Name</td><td>schedule example</td>
  </tr>
  <tr>
    <td>Last Run Status</td><td>unknown</td>
  </tr>
  <tr>
    <td>Type</td><td>Scheduled Run</td>
  </tr>
  <tr>
    <td>Active</td><td>True</td>
  </tr>
  <tr>
    <td>Schedule</td><td>*/5 * * * *</td>
  </tr>
  <tr>
    <td>Created At</td><td>2023-22-May 20:08:25</td>
  </tr>
  <tr>
    <td>Updated At</td><td>2023-22-May 20:08:25</td>
  </tr>
</table>
{{</table>}}

```python
time.sleep(420)
recent_run = task_scheduled.last_runs()[0]
display(recent_run.logs())
```

<pre><code>2023-22-May 20:11:02 Getting the workspace orchestrationworkspacetgiq
2023-22-May 20:11:02 Getting the pipeline orchestrationpipelinetgiq
2023-22-May 20:11:02 Inference time.  Displaying results after.
2023-22-May 20:11:02 Getting arrow table file
2023-22-May 20:11:02 pyarrow.Table
2023-22-May 20:11:02 time: timestamp[ms]
2023-22-May 20:11:02 in.tensor: list<item: float> not null
2023-22-May 20:11:02   child 0, item: float
2023-22-May 20:11:02 out.variable: list<inner: float not null> not null
2023-22-May 20:11:02   child 0, inner: float not null
2023-22-May 20:11:02 check_failures: int8
2023-22-May 20:11:02 ----
2023-22-May 20:11:02 time: [[2023-05-22 20:10:23.271,2023-05-22 20:10:23.271,2023-05-22 20:10:23.271,2023-05-22 20:10:23.271,2023-05-22 20:10:23.271,...,2023-05-22 20:10:23.271,2023-05-22 20:10:23.271,2023-05-22 20:10:23.271,2023-05-22 20:10:23.271,2023-05-22 20:10:23.271]]
2023-22-May 20:11:02 in.tensor: [[[4,2.5,2900,5505,2,...,2970,5251,12,0,0],[2,2.5,2170,6361,1,...,2310,7419,6,0,0],...,[3,1.75,2910,37461,1,...,2520,18295,47,0,0],[3,2,2005,7000,1,...,1750,4500,34,0,0]]]
2023-22-May 20:11:02 check_failures: [[0,0,0,0,0,...,0,0,0,0,0]]
2023-22-May 20:11:02 out.variable: [[[718013.75],[615094.56],...,[706823.56],[581003]]]</code></pre>

### Kill a Task

Killing a task removes the schedule or removes it from a service.  Tasks are killed with the Task `kill()` method, and returns a message with the status of the kill procedure.

If necessary, all tasks can be killed through the following script.

* **IMPORTANT NOTE**:  This command will kill all running tasks - scheduled or otherwise.  Only use this if required.

```python
# Kill all tasks
for t in wl.list_tasks(): t.kill()
```

When listed with Wallaroo client `task_list(killed=True)` , the field `active` displays tasks that are killed (`False`) or either completed running or still scheduled to run (`True`).

```python
task_scheduled.kill()
```

    <ArbexStatus.PENDING_KILL: 'pending_kill'>

```python
wl.list_tasks(killed=True)
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>id</th><th>name</th><th>last run status</th><th>type</th><th>active</th><th>schedule</th><th>created at</th><th>updated at</th></tr><tr><td>4af57c61-dfa9-43eb-944e-559135495df4</td><td>schedule example</td><td>success</td><td>Scheduled Run</td><td>False</td><td>*/5 * * * *</td><td>2023-22-May 20:08:25</td><td>2023-22-May 20:13:12</td></tr><tr><td>dc185e24-cf89-4a97-b6f0-33fc3d67da72</td><td>schedule example</td><td>unknown</td><td>Scheduled Run</td><td>False</td><td>*/5 * * * *</td><td>2023-22-May 20:05:47</td><td>2023-22-May 20:06:22</td></tr><tr><td>f0e27d6a-6a98-4d26-b240-266f08560c48</td><td>house price run once 2</td><td>success</td><td>Temporary Run</td><td>True</td><td>-</td><td>2023-22-May 19:58:32</td><td>2023-22-May 19:58:38</td></tr><tr><td>36509ef8-98da-42a0-913f-e6e929dedb15</td><td>house price run once</td><td>success</td><td>Temporary Run</td><td>True</td><td>-</td><td>2023-22-May 19:56:37</td><td>2023-22-May 19:56:48</td></tr></table>
{{</table>}}

## Cleaning Up

With the tutorial complete we will undeploy the pipeline and ensure the resources are returned back to the Wallaroo instance.

```python
pipeline.undeploy()
```
