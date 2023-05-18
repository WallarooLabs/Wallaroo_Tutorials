This can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/blob/20230314_2023.2_updates/pipeline-orchestrators/orchestration_sdk_comprehensive_tutorial).

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

workspace_name = 'orchestrationworkspace'
pipeline_name = 'orchestrationpipeline'
model_name = 'orchestrationmodel'
model_file_name = './models/rf_model.onnx'
connection_name = "houseprice_arrow_table"
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
<table><tr><th>name</th> <td>orchestrationpipeline</td></tr><tr><th>created</th> <td>2023-05-17 17:40:11.391198+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 17:40:11.391198+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>6964e793-b45d-41b8-9ef7-f773c1dc4fe5</td></tr><tr><th>steps</th> <td></td></tr></table>
{{</table>}}

```python
#deploy the pipeline
pipeline.deploy()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>orchestrationpipeline</td></tr><tr><th>created</th> <td>2023-05-17 17:40:11.391198+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 17:40:14.433826+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>0e260e71-8b80-4ed3-b23f-209c4eeb4b02, 6964e793-b45d-41b8-9ef7-f773c1dc4fe5</td></tr><tr><th>steps</th> <td>orchestrationmodel</td></tr></table>
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
      <td>2023-05-17 17:40:40.126</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.75]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-05-17 17:40:40.126</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[615094.56]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-05-17 17:40:40.126</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
      <td>[448627.72]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-05-17 17:40:40.126</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[758714.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-05-17 17:40:40.126</td>
      <td>[3.0, 1.75, 2200.0, 11520.0, 1.0, 0.0, 0.0, 4.0, 7.0, 2200.0, 0.0, 47.7659, -122.341, 1690.0, 8038.0, 62.0, 0.0, 0.0]</td>
      <td>[513264.7]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2023-05-17 17:40:40.126</td>
      <td>[3.0, 2.0, 2140.0, 4923.0, 1.0, 0.0, 0.0, 4.0, 8.0, 1070.0, 1070.0, 47.6902, -122.339, 1470.0, 4923.0, 86.0, 0.0, 0.0]</td>
      <td>[668288.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2023-05-17 17:40:40.126</td>
      <td>[4.0, 3.5, 3590.0, 5334.0, 2.0, 0.0, 2.0, 3.0, 9.0, 3140.0, 450.0, 47.6763, -122.267, 2100.0, 6250.0, 9.0, 0.0, 0.0]</td>
      <td>[1004846.5]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2023-05-17 17:40:40.126</td>
      <td>[3.0, 2.0, 1280.0, 960.0, 2.0, 0.0, 0.0, 3.0, 9.0, 1040.0, 240.0, 47.602, -122.311, 1280.0, 1173.0, 0.0, 0.0, 0.0]</td>
      <td>[684577.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2023-05-17 17:40:40.126</td>
      <td>[4.0, 2.5, 2820.0, 15000.0, 2.0, 0.0, 0.0, 4.0, 9.0, 2820.0, 0.0, 47.7255, -122.101, 2440.0, 15000.0, 29.0, 0.0, 0.0]</td>
      <td>[727898.1]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2023-05-17 17:40:40.126</td>
      <td>[3.0, 2.25, 1790.0, 11393.0, 1.0, 0.0, 0.0, 3.0, 8.0, 1790.0, 0.0, 47.6297, -122.099, 2290.0, 11894.0, 36.0, 0.0, 0.0]</td>
      <td>[559631.1]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2023-05-17 17:40:40.126</td>
      <td>[3.0, 1.5, 1010.0, 7683.0, 1.5, 0.0, 0.0, 5.0, 7.0, 1010.0, 0.0, 47.72, -122.318, 1550.0, 7271.0, 61.0, 0.0, 0.0]</td>
      <td>[340764.53]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2023-05-17 17:40:40.126</td>
      <td>[3.0, 2.0, 1270.0, 1323.0, 3.0, 0.0, 0.0, 3.0, 8.0, 1270.0, 0.0, 47.6934, -122.342, 1330.0, 1323.0, 8.0, 0.0, 0.0]</td>
      <td>[442168.06]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2023-05-17 17:40:40.126</td>
      <td>[4.0, 1.75, 2070.0, 9120.0, 1.0, 0.0, 0.0, 4.0, 7.0, 1250.0, 820.0, 47.6045, -122.123, 1650.0, 8400.0, 57.0, 0.0, 0.0]</td>
      <td>[630865.6]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2023-05-17 17:40:40.126</td>
      <td>[4.0, 1.0, 1620.0, 4080.0, 1.5, 0.0, 0.0, 3.0, 7.0, 1620.0, 0.0, 47.6696, -122.324, 1760.0, 4080.0, 91.0, 0.0, 0.0]</td>
      <td>[559631.1]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2023-05-17 17:40:40.126</td>
      <td>[4.0, 3.25, 3990.0, 9786.0, 2.0, 0.0, 0.0, 3.0, 9.0, 3990.0, 0.0, 47.6784, -122.026, 3920.0, 8200.0, 10.0, 0.0, 0.0]</td>
      <td>[909441.1]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2023-05-17 17:40:40.126</td>
      <td>[4.0, 2.0, 1780.0, 19843.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1780.0, 0.0, 47.4414, -122.154, 2210.0, 13500.0, 52.0, 0.0, 0.0]</td>
      <td>[313096.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2023-05-17 17:40:40.126</td>
      <td>[4.0, 2.5, 2130.0, 6003.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2130.0, 0.0, 47.4518, -122.12, 1940.0, 4529.0, 11.0, 0.0, 0.0]</td>
      <td>[404040.8]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2023-05-17 17:40:40.126</td>
      <td>[3.0, 1.75, 1660.0, 10440.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1040.0, 620.0, 47.4448, -121.77, 1240.0, 10380.0, 36.0, 0.0, 0.0]</td>
      <td>[292859.5]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2023-05-17 17:40:40.126</td>
      <td>[3.0, 2.5, 2110.0, 4118.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2110.0, 0.0, 47.3878, -122.153, 2110.0, 4044.0, 25.0, 0.0, 0.0]</td>
      <td>[338357.88]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2023-05-17 17:40:40.126</td>
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

Connections are created with the Wallaroo client command [`create_connection`](https://staging.docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-dataconnectors/#create-data-connection) with the following parameters.

| Parameter | Type | Description |
| --- | --- | ---|
| **name** | string (Required) | The name of the connection. This must be unique - if submitting the name of an existing connection it will return an error. |
| **type** | string (Required) | The user defined type of connection. |
| **details** | Dict (Requires) | User defined configuration details for the data connection.  These can be `{'username':'dataperson', 'password':'datapassword', 'port': 3339}`, or `{'token':'abcde123==', 'host':'example.com', 'port:1234'}`, or other user defined combinations.  |

We'll create the connection named `houseprice_arrow_table`, set it to the type `HTTPFILE`, and provide the details as `'host':'https://github.com/WallarooLabs/Wallaroo_Tutorials/blob/20230314_2023.2_updates/wallaroo-testing-tutorials/houseprice-saga/data/xtest-1k.arrow?raw=true'` - the location for our sample Arrow table inference input.

```python
wl.create_connection(connection_name, 
                  "HTTPFILE", 
                  {'host':'https://github.com/WallarooLabs/Wallaroo_Tutorials/blob/20230314_2023.2_updates/wallaroo-testing-tutorials/houseprice-saga/data/xtest-1k.arrow?raw=true'}
                  )
```

{{<table "table table-striped table-bordered" >}}
<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>Name</td><td>houseprice_arrow_table</td>
  </tr>
  <tr>
    <td>Connection Type</td><td>HTTPFILE</td>
  </tr>
  <tr>
    <td>Details</td><td>*****</td>
  </tr>
  <tr>
    <td>Created At</td><td>2023-05-17T17:42:20.215405+00:00</td>
  </tr>
</table>
{{</table>}}

### List Data Connections

The Wallaroo Client [`list_connections()`](https://staging.docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-dataconnectors/#list-data-connections) method lists all connections for the Wallaroo instance.

```python
wl.list_connections()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th><th>connection type</th><th>details</th><th>created at</th></tr><tr><td>external_inference_connection</td><td>HTTP</td><td>*****</td><td>2023-05-17T14:03:08.288200+00:00</td></tr><tr><td>bigqueryhouseapioutputsxmzn</td><td>BIGQUERY</td><td>*****</td><td>2023-05-17T16:07:05.669298+00:00</td></tr><tr><td>bigqueryhouseapiinputzjbz</td><td>BIGQUERY</td><td>*****</td><td>2023-05-17T16:12:10.751778+00:00</td></tr><tr><td>bigqueryhouseapioutputszjbz</td><td>BIGQUERY</td><td>*****</td><td>2023-05-17T16:12:12.247243+00:00</td></tr><tr><td>bigqueryhouseinputs</td><td>BIGQUERY</td><td>*****</td><td>2023-05-17T16:59:45.947103+00:00</td></tr><tr><td>bigqueryhouseoutputs</td><td>BIGQUERY</td><td>*****</td><td>2023-05-17T16:59:46.601720+00:00</td></tr><tr><td>bigqueryforecastinputs</td><td>BIGQUERY</td><td>*****</td><td>2023-05-17T17:24:32.276657+00:00</td></tr><tr><td>bigqueryforecastoutputs</td><td>BIGQUERY</td><td>*****</td><td>2023-05-17T17:24:32.876291+00:00</td></tr><tr><td>houseprice_arrow_table2</td><td>HTTPFILE</td><td>*****</td><td>2023-05-17T17:40:40.763136+00:00</td></tr><tr><td>houseprice_arrow_table3</td><td>HTTPFILE</td><td>*****</td><td>2023-05-17T17:40:41.517898+00:00</td></tr><tr><td>houseprice_arrow_table</td><td>HTTPFILE</td><td>*****</td><td>2023-05-17T17:42:20.215405+00:00</td></tr></table>
{{</table>}}

### Add Connection to Workspace

The method Workspace [`add_connection(connection_name)`](https://staging.docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-dataconnectors/#add-data-connection-to-workspace) adds a Data Connection to a workspace, and takes the following parameters.

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
<table><tr><th>name</th><th>connection type</th><th>details</th><th>created at</th></tr><tr><td>houseprice_arrow_table</td><td>HTTPFILE</td><td>*****</td><td>2023-05-17T17:42:20.215405+00:00</td></tr></table>
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

    {'host': 'https://github.com/WallarooLabs/Wallaroo_Tutorials/blob/20230314_2023.2_updates/wallaroo-testing-tutorials/houseprice-saga/data/xtest-1k.arrow?raw=true'}

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
      <td>2023-05-17 17:43:50.585</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.75]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-05-17 17:43:50.585</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[615094.56]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-05-17 17:43:50.585</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
      <td>[448627.72]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-05-17 17:43:50.585</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[758714.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-05-17 17:43:50.585</td>
      <td>[3.0, 1.75, 2200.0, 11520.0, 1.0, 0.0, 0.0, 4.0, 7.0, 2200.0, 0.0, 47.7659, -122.341, 1690.0, 8038.0, 62.0, 0.0, 0.0]</td>
      <td>[513264.7]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2023-05-17 17:43:50.585</td>
      <td>[3.0, 2.0, 2140.0, 4923.0, 1.0, 0.0, 0.0, 4.0, 8.0, 1070.0, 1070.0, 47.6902, -122.339, 1470.0, 4923.0, 86.0, 0.0, 0.0]</td>
      <td>[668288.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2023-05-17 17:43:50.585</td>
      <td>[4.0, 3.5, 3590.0, 5334.0, 2.0, 0.0, 2.0, 3.0, 9.0, 3140.0, 450.0, 47.6763, -122.267, 2100.0, 6250.0, 9.0, 0.0, 0.0]</td>
      <td>[1004846.5]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2023-05-17 17:43:50.585</td>
      <td>[3.0, 2.0, 1280.0, 960.0, 2.0, 0.0, 0.0, 3.0, 9.0, 1040.0, 240.0, 47.602, -122.311, 1280.0, 1173.0, 0.0, 0.0, 0.0]</td>
      <td>[684577.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2023-05-17 17:43:50.585</td>
      <td>[4.0, 2.5, 2820.0, 15000.0, 2.0, 0.0, 0.0, 4.0, 9.0, 2820.0, 0.0, 47.7255, -122.101, 2440.0, 15000.0, 29.0, 0.0, 0.0]</td>
      <td>[727898.1]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2023-05-17 17:43:50.585</td>
      <td>[3.0, 2.25, 1790.0, 11393.0, 1.0, 0.0, 0.0, 3.0, 8.0, 1790.0, 0.0, 47.6297, -122.099, 2290.0, 11894.0, 36.0, 0.0, 0.0]</td>
      <td>[559631.1]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2023-05-17 17:43:50.585</td>
      <td>[3.0, 1.5, 1010.0, 7683.0, 1.5, 0.0, 0.0, 5.0, 7.0, 1010.0, 0.0, 47.72, -122.318, 1550.0, 7271.0, 61.0, 0.0, 0.0]</td>
      <td>[340764.53]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2023-05-17 17:43:50.585</td>
      <td>[3.0, 2.0, 1270.0, 1323.0, 3.0, 0.0, 0.0, 3.0, 8.0, 1270.0, 0.0, 47.6934, -122.342, 1330.0, 1323.0, 8.0, 0.0, 0.0]</td>
      <td>[442168.06]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2023-05-17 17:43:50.585</td>
      <td>[4.0, 1.75, 2070.0, 9120.0, 1.0, 0.0, 0.0, 4.0, 7.0, 1250.0, 820.0, 47.6045, -122.123, 1650.0, 8400.0, 57.0, 0.0, 0.0]</td>
      <td>[630865.6]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2023-05-17 17:43:50.585</td>
      <td>[4.0, 1.0, 1620.0, 4080.0, 1.5, 0.0, 0.0, 3.0, 7.0, 1620.0, 0.0, 47.6696, -122.324, 1760.0, 4080.0, 91.0, 0.0, 0.0]</td>
      <td>[559631.1]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2023-05-17 17:43:50.585</td>
      <td>[4.0, 3.25, 3990.0, 9786.0, 2.0, 0.0, 0.0, 3.0, 9.0, 3990.0, 0.0, 47.6784, -122.026, 3920.0, 8200.0, 10.0, 0.0, 0.0]</td>
      <td>[909441.1]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2023-05-17 17:43:50.585</td>
      <td>[4.0, 2.0, 1780.0, 19843.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1780.0, 0.0, 47.4414, -122.154, 2210.0, 13500.0, 52.0, 0.0, 0.0]</td>
      <td>[313096.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2023-05-17 17:43:50.585</td>
      <td>[4.0, 2.5, 2130.0, 6003.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2130.0, 0.0, 47.4518, -122.12, 1940.0, 4529.0, 11.0, 0.0, 0.0]</td>
      <td>[404040.8]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2023-05-17 17:43:50.585</td>
      <td>[3.0, 1.75, 1660.0, 10440.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1040.0, 620.0, 47.4448, -121.77, 1240.0, 10380.0, 36.0, 0.0, 0.0]</td>
      <td>[292859.5]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2023-05-17 17:43:50.585</td>
      <td>[3.0, 2.5, 2110.0, 4118.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2110.0, 0.0, 47.3878, -122.153, 2110.0, 4044.0, 25.0, 0.0, 0.0]</td>
      <td>[338357.88]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2023-05-17 17:43:50.585</td>
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

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th><th>connection type</th><th>details</th><th>created at</th></tr><tr><td>external_inference_connection</td><td>HTTP</td><td>*****</td><td>2023-05-17T14:03:08.288200+00:00</td></tr><tr><td>bigqueryhouseapioutputsxmzn</td><td>BIGQUERY</td><td>*****</td><td>2023-05-17T16:07:05.669298+00:00</td></tr><tr><td>bigqueryhouseapiinputzjbz</td><td>BIGQUERY</td><td>*****</td><td>2023-05-17T16:12:10.751778+00:00</td></tr><tr><td>bigqueryhouseapioutputszjbz</td><td>BIGQUERY</td><td>*****</td><td>2023-05-17T16:12:12.247243+00:00</td></tr><tr><td>bigqueryhouseinputs</td><td>BIGQUERY</td><td>*****</td><td>2023-05-17T16:59:45.947103+00:00</td></tr><tr><td>bigqueryhouseoutputs</td><td>BIGQUERY</td><td>*****</td><td>2023-05-17T16:59:46.601720+00:00</td></tr><tr><td>bigqueryforecastinputs</td><td>BIGQUERY</td><td>*****</td><td>2023-05-17T17:24:32.276657+00:00</td></tr><tr><td>bigqueryforecastoutputs</td><td>BIGQUERY</td><td>*****</td><td>2023-05-17T17:24:32.876291+00:00</td></tr><tr><td>houseprice_arrow_table2</td><td>HTTPFILE</td><td>*****</td><td>2023-05-17T17:40:40.763136+00:00</td></tr><tr><td>houseprice_arrow_table3</td><td>HTTPFILE</td><td>*****</td><td>2023-05-17T17:40:41.517898+00:00</td></tr></table>
{{</table>}}

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

workspace_name = 'orchestrationworkspace'
pipeline_name = 'orchestrationpipeline'
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
connection = wl.list_connections()[0]
```

```python
wl.create_connection(connection_name, 
                  "HTTPFILE", 
                  {'host':'https://github.com/WallarooLabs/Wallaroo_Tutorials/blob/20230314_2023.2_updates/wallaroo-testing-tutorials/houseprice-saga/data/xtest-1k.arrow?raw=true'}
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
<table><tr><th>id</th><th>name</th><th>status</th><th>filename</th><th>sha</th><th>created at</th><th>updated at</th></tr><tr><td>be238470-86e1-49a7-83b9-2940f71b9f1d</td><td>comprehensive sample</td><td>ready</td><td>remote_inference.zip</td><td>805a93...7384bf</td><td>2023-17-May 18:15:09</td><td>2023-17-May 18:15:53</td></tr><tr><td>e9c8b5ff-be6a-4ceb-bd22-6c7cd63e24ba</td><td>bigquery example</td><td>ready</td><td>remote_inference.zip</td><td>bf26b4...ed87be</td><td>2023-17-May 17:44:01</td><td>2023-17-May 17:44:44</td></tr><tr><td>05539e2e-965c-4a9f-b953-5f82a8f96618</td><td>comprehensive example</td><td>ready</td><td>remote_inference.zip</td><td>6bac88...1947a5</td><td>2023-17-May 18:04:43</td><td>2023-17-May 18:05:26</td></tr><tr><td>b96dfa0a-0d5c-43cc-a7dd-376b732b2cda</td><td>comprehensive example</td><td>ready</td><td>remote_inference.zip</td><td>f1c032...9370f2</td><td>2023-17-May 18:09:49</td><td>2023-17-May 18:10:32</td></tr></table>
{{</table>}}

## Task Management Tutorial

Once an Orchestration has the status `ready`, it can be run as a task.  Tasks have three run options.

| Type       | SDK Call |  How triggered                                                               | Purpose                                                       |
|------------|----------|:------------------------------------------------------------------------------|:---------------------------------------------------------------|
| Once       | `orchestration.run_once(name, json_args, timeout)` | Task runs once and exits.| Single batch, experimentation                                 |
| Scheduled  | `orchestration.run_scheduled()` | User provides schedule. Task runs exits whenever schedule dictates. | Recurrent batch ETL.|

### Task Run Once

Tasks are generated and run once with the Orchestration `run_once(name, json_args, timeout)` method.  Any arguments for the orchestration are passed in as a `Dict`.  If there are no arguments, then an empty set `{}` is passed.

```python
# Example: run once

import datetime
task_start = datetime.datetime.now()

task = orchestration.run_once(name="house price run once task", json_args={})
task
```

{{<table "table table-striped table-bordered" >}}
<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>ID</td><td>d0d9f48a-2028-47f8-84e0-c6659bcbfcc3</td>
  </tr>
  <tr>
    <td>Name</td><td>house price run once task</td>
  </tr>
  <tr>
    <td>Status</td><td>pending</td>
  </tr>
  <tr>
    <td>Type</td><td>Temporary Run</td>
  </tr>
  <tr>
    <td>Created At</td><td>2023-17-May 18:15:57</td>
  </tr>
  <tr>
    <td>Updated At</td><td>2023-17-May 18:15:57</td>
  </tr>
</table>
{{</table>}}

### Task Status

The list of tasks in the Wallaroo instance is retrieves through the Wallaroo Client `list_tasks()` method.  This returns an array list of the following.

| Parameter | Type | Description |
| --- | --- | ---|
| **id** | string | The UUID identifier for the task. |
| **status** | enum | The status of the task.  Values are: <br><ul><li>`pending`: The task has not been started.</li><li>`started`: The task has been scheduled to execute.</li><li>`pending_kill`: The task kill command has been issued and the task is scheduled to be stopped.</li></ul> |
| **type** | string | The type of the task.  Values are: <br><ul><li>`Temporary Run`: The task runs once then stop.</li><li>`Scheduled Run`: The task repeats on a `cron` like schedule.</li><li>`Service Run`: The task runs as a service and executes when its service port is activated. |
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

    datetime.datetime(2023, 5, 17, 12, 17, 4, 443106)

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
      <td>2023-05-17 18:16:05.018</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.75]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-05-17 18:16:05.018</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[615094.56]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-05-17 18:16:05.018</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
      <td>[448627.72]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-05-17 18:16:05.018</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[758714.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-05-17 18:16:05.018</td>
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
      <th>512</th>
      <td>2023-05-17 18:16:05.018</td>
      <td>[4.0, 2.5, 2740.0, 43101.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2740.0, 0.0, 47.7649, -122.049, 2740.0, 33447.0, 21.0, 0.0, 0.0]</td>
      <td>[727898.1]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>513</th>
      <td>2023-05-17 18:16:05.018</td>
      <td>[4.0, 3.25, 3320.0, 8587.0, 3.0, 0.0, 0.0, 3.0, 11.0, 2950.0, 370.0, 47.691, -122.337, 1860.0, 5668.0, 6.0, 0.0, 0.0]</td>
      <td>[1130661.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>514</th>
      <td>2023-05-17 18:16:05.018</td>
      <td>[4.0, 2.5, 3130.0, 13202.0, 2.0, 0.0, 0.0, 3.0, 10.0, 3130.0, 0.0, 47.5878, -121.976, 2840.0, 10470.0, 19.0, 0.0, 0.0]</td>
      <td>[879083.4]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>515</th>
      <td>2023-05-17 18:16:05.018</td>
      <td>[2.0, 1.75, 1370.0, 5125.0, 1.0, 0.0, 0.0, 5.0, 6.0, 1370.0, 0.0, 47.6926, -122.346, 1200.0, 5100.0, 70.0, 0.0, 0.0]</td>
      <td>[444933.16]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>516</th>
      <td>2023-05-17 18:16:05.018</td>
      <td>[4.0, 2.5, 2040.0, 5508.0, 2.0, 0.0, 0.0, 4.0, 8.0, 2040.0, 0.0, 47.5719, -122.007, 2130.0, 5496.0, 18.0, 0.0, 0.0]</td>
      <td>[627853.5]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}
<p>517 rows × 4 columns</p>

### Scheduled Tasks

Scheduled tasks are run with the Orchestration `run_scheduled` method.  We'll set it up to run every minute, then check the results.

```python
task_start = datetime.datetime.now()
task_scheduled = orchestration.run_scheduled(name="schedule example", timeout=600, schedule="*/1, *, *, *, *", json_args={})
while task_scheduled.status() != "started":
    display(task_scheduled.status())
    time.sleep(5)
task_scheduled
```

{{<table "table table-striped table-bordered" >}}
<table>
  <tr>
    <th>Field</th>
    <th>Value</th>
  </tr>
  <tr>
    <td>ID</td><td>c6f1a4dd-f24f-42b5-87b2-db3dc5c9d817</td>
  </tr>
  <tr>
    <td>Name</td><td>schedule example</td>
  </tr>
  <tr>
    <td>Status</td><td>started</td>
  </tr>
  <tr>
    <td>Type</td><td>Scheduled Run</td>
  </tr>
  <tr>
    <td>Created At</td><td>2023-17-May 18:18:28</td>
  </tr>
  <tr>
    <td>Updated At</td><td>2023-17-May 18:18:28</td>
  </tr>
</table>
{{</table>}}

```python
time.sleep(90)

task_end = datetime.datetime.now()
display(task_end)

pipeline.logs(start_datetime = task_start, end_datetime = task_end)
```

    datetime.datetime(2023, 5, 17, 12, 19, 59, 485331)

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
      <td>2023-05-17 18:19:17.483</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.75]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-05-17 18:19:17.483</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[615094.56]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-05-17 18:19:17.483</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
      <td>[448627.72]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-05-17 18:19:17.483</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[758714.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-05-17 18:19:17.483</td>
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
      <th>512</th>
      <td>2023-05-17 18:19:17.483</td>
      <td>[4.0, 2.5, 2740.0, 43101.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2740.0, 0.0, 47.7649, -122.049, 2740.0, 33447.0, 21.0, 0.0, 0.0]</td>
      <td>[727898.1]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>513</th>
      <td>2023-05-17 18:19:17.483</td>
      <td>[4.0, 3.25, 3320.0, 8587.0, 3.0, 0.0, 0.0, 3.0, 11.0, 2950.0, 370.0, 47.691, -122.337, 1860.0, 5668.0, 6.0, 0.0, 0.0]</td>
      <td>[1130661.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>514</th>
      <td>2023-05-17 18:19:17.483</td>
      <td>[4.0, 2.5, 3130.0, 13202.0, 2.0, 0.0, 0.0, 3.0, 10.0, 3130.0, 0.0, 47.5878, -121.976, 2840.0, 10470.0, 19.0, 0.0, 0.0]</td>
      <td>[879083.4]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>515</th>
      <td>2023-05-17 18:19:17.483</td>
      <td>[2.0, 1.75, 1370.0, 5125.0, 1.0, 0.0, 0.0, 5.0, 6.0, 1370.0, 0.0, 47.6926, -122.346, 1200.0, 5100.0, 70.0, 0.0, 0.0]</td>
      <td>[444933.16]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>516</th>
      <td>2023-05-17 18:19:17.483</td>
      <td>[4.0, 2.5, 2040.0, 5508.0, 2.0, 0.0, 0.0, 4.0, 8.0, 2040.0, 0.0, 47.5719, -122.007, 2130.0, 5496.0, 18.0, 0.0, 0.0]</td>
      <td>[627853.5]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}
<p>517 rows × 4 columns</p>

### Kill a Task

Killing a task removes the schedule or removes it from a service.  Tasks are killed with the Task `kill()` method, and returns a message with the status of the kill procedure.

If necessary, all tasks can be killed through the following script.

* **IMPORTANT NOTE**:  This command will kill all running tasks - scheduled or otherwise.  Only use this if required.

```python
# Kill all tasks
for t in wl.list_tasks(): t.kill()
```

```python
task_scheduled.kill()
```

    <ArbexStatus.PENDING_KILL: 'pending_kill'>

## Cleaning Up

With the tutorial complete we will undeploy the pipeline and ensure the resources are returned back to the Wallaroo instance.

```python
pipeline.undeploy()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>orchestrationpipeline</td></tr><tr><th>created</th> <td>2023-05-17 17:40:11.391198+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 17:43:47.579379+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>945c0da9-8967-4f7b-8db5-b6161f86315a, 0e260e71-8b80-4ed3-b23f-209c4eeb4b02, 6964e793-b45d-41b8-9ef7-f773c1dc4fe5</td></tr><tr><th>steps</th> <td>orchestrationmodel</td></tr></table>
{{</table>}}

