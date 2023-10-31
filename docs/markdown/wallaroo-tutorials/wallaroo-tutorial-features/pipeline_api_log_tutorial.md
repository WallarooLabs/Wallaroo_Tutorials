This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-features/pipeline_api_log_tutorial).

## Pipeline API Log Tutorial

This tutorial demonstrates Wallaroo Pipeline MLOps API for pipeline log retrieval.

This tutorial will demonstrate how to:

1. Select or create a workspace, pipeline and upload the control model, and additional testing models.
1. Add a pipeline step with the champion model, then deploy the pipeline and perform sample inferences.
1. Retrieve the logs via the Wallaroo MLOps API.  These steps will be simplified to only show the API log retrieval method.  See the [Wallaroo Documentation site](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/) for full details.
1. Swap out the pipeline step with the champion model with a shadow deploy step that compares the champion model against two competitors.
1. Perform sample inferences with a shadow deployed step, then display the log files through the MLOps API for a shadow deployed pipeline.
1. Swap out the shadow deployed pipeline step with an A/B pipeline step.
1. Perform sample inferences with a A/B pipeline step, then display the log files through the MLOps API for an A/B pipeline step.
1. Undeploy the pipeline.

This tutorial provides the following:

* Models:
  * `models/rf_model.onnx`: The champion model that has been used in this environment for some time.
  * `models/xgb_model.onnx` and `models/gbr_model.onnx`: Rival models that will be tested against the champion.
* Data:
  * `data/xtest-1.df.json` and `data/xtest-1k.df.json`:  DataFrame JSON inference inputs with 1 input and 1,000 inputs.
  * `data/xtest-1k.arrow`:  Apache Arrow inference inputs with 1 input and 1,000 inputs.

## Prerequisites

* A deployed Wallaroo instance
* The following Python libraries installed:
  * [`wallaroo`](https://pypi.org/project/wallaroo/): The Wallaroo SDK. Included with the Wallaroo JupyterHub service by default.
  * [`pandas`](https://pypi.org/project/pandas/): Pandas, mainly used for Pandas DataFrame
  * [`pyarrow`](https://pypi.org/project/pyarrow/): Pyarrow for Apache Arrow support

## Initial Steps

### Import libraries

The first step is to import the libraries needed for this notebook.

```python
import wallaroo
from wallaroo.object import EntityNotFoundError

import pyarrow as pa

from IPython.display import display

# used to display DataFrame information without truncating
from IPython.display import display
import pandas as pd
pd.set_option('display.max_colwidth', None)

import datetime
import requests
```

### Connect to the Wallaroo Instance

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  If logging in externally, update the `wallarooPrefix` and `wallarooSuffix` variables with the proper DNS information.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
# Login through local Wallaroo instance

wl = wallaroo.Client()
```

## Wallaroo MLOps API URL

### API URL

The variable `APIURL` is used to specify the connection to the Wallaroo instance's MLOps API URL, and is composed of the Wallaroo DNS prefix and suffix.  For full details, see the [Wallaroo API Connection Guide
](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-connection-guide/).

The variables `wallarooPrefix.` and `wallarooSuffix` variables will be used to derive the API url.  For example, if the Wallaroo Prefix is `doc-test.` and the url is `example.com`, then the MLOps API URL would be `doc-test.api.example.com/v1/api/{request}`.  Note that the `.` is part of the prefix; if there is no prefix, then `wallarooPrefix = ""`.

Set the Wallaroo Prefix and Suffix in the code segment below based on your Wallaroo instance.

```python
wallarooPrefix = "YOUR PREFIX."
wallarooSuffix = "YOUR SUFFIX"

APIURL = f"https://{wallarooPrefix}api.{wallarooSuffix}"
```

### Create Workspace

We will create a workspace to manage our pipeline and models.  The following variables will set the name of our sample workspace then set it as the current workspace.

```python
workspace_name = 'logapiworkspace'
main_pipeline_name = 'logapipipeline'
model_name_control = 'logapicontrol'
model_file_name_control = './models/rf_model.onnx'
```

```python
def get_workspace(name):
    workspace = None
    for ws in wl.list_workspaces():
        if ws.name() == name:
            workspace= ws
    if(workspace == None):
        workspace = wl.create_workspace(name)
    return workspace
```

```python
workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)

workspace_id = workspace.id()
```

## Standard Pipeline

### Upload The Champion Model

For our example, we will upload the champion model that has been trained to derive house prices from a variety of inputs.  The model file is `rf_model.onnx`, and is uploaded with the name `housingcontrol`.

```python
housing_model_control = (wl.upload_model(model_name_control, 
                                         model_file_name_control, 
                                         framework=wallaroo.framework.Framework.ONNX)
                                         .configure(tensor_fields=["tensor"])
                        )
```

### Build the Pipeline

This pipeline is made to be an example of an existing situation where a model is deployed and being used for inferences in a production environment.  We'll call it `housepricepipeline`, set `housingcontrol` as a pipeline step, then run a few sample inferences.

```python
mainpipeline = wl.build_pipeline(main_pipeline_name)
mainpipeline.undeploy()
# in case this pipeline was run before
mainpipeline.clear()
mainpipeline.add_model_step(housing_model_control).deploy()
```

<table><tr><th>name</th> <td>logapipipeline</td></tr><tr><th>created</th> <td>2023-07-14 15:43:25.566285+00:00</td></tr><tr><th>last_updated</th> <td>2023-07-14 15:43:29.948989+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>762a7f50-d1cc-4912-ab1d-5ed87b985797, 29b7109c-1467-40e1-aa11-dcb96959bb3e</td></tr><tr><th>steps</th> <td>logapicontrol</td></tr></table>

### Testing

We'll pass in two DataFrame formatted inference requests which are returned as a pandas DataFrame.  Then roughly 1,000 inferences as a batch as an Apache Arrow table, which is returned as an arrow table, which we'll convert into a pandas DataFrame to display the first 20 results.

```python
dataframe_start = datetime.datetime.now(datetime.timezone.utc)

normal_input = pd.DataFrame.from_records({"tensor": [[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]]})
result = mainpipeline.infer(normal_input)
display(result)

large_house_input = pd.DataFrame.from_records({'tensor': [[4.0, 3.0, 3710.0, 20000.0, 2.0, 0.0, 2.0, 5.0, 10.0, 2760.0, 950.0, 47.6696, -122.261, 3970.0, 20000.0, 79.0, 0.0, 0.0]]})
large_house_result = mainpipeline.infer(large_house_input)
display(large_house_result)

import time
time.sleep(10)
dataframe_end = datetime.datetime.now(datetime.timezone.utc)

# generating multiple log entries
batch_inferences = mainpipeline.infer_from_file('./data/xtest-1k.arrow')
batch_inferences = mainpipeline.infer_from_file('./data/xtest-1k.arrow')
batch_inferences = mainpipeline.infer_from_file('./data/xtest-1k.arrow')

large_inference_result = batch_inferences.to_pandas()
display(large_inference_result.head(20))

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
      <td>2023-07-14 15:43:45.872</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.7]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

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
      <td>2023-07-14 15:43:46.983</td>
      <td>[4.0, 3.0, 3710.0, 20000.0, 2.0, 0.0, 2.0, 5.0, 10.0, 2760.0, 950.0, 47.6696, -122.261, 3970.0, 20000.0, 79.0, 0.0, 0.0]</td>
      <td>[1514079.4]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

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
      <td>2023-07-14 15:43:58.844</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.75]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-07-14 15:43:58.844</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[615094.56]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-07-14 15:43:58.844</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
      <td>[448627.72]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-07-14 15:43:58.844</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[758714.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-07-14 15:43:58.844</td>
      <td>[3.0, 1.75, 2200.0, 11520.0, 1.0, 0.0, 0.0, 4.0, 7.0, 2200.0, 0.0, 47.7659, -122.341, 1690.0, 8038.0, 62.0, 0.0, 0.0]</td>
      <td>[513264.7]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2023-07-14 15:43:58.844</td>
      <td>[3.0, 2.0, 2140.0, 4923.0, 1.0, 0.0, 0.0, 4.0, 8.0, 1070.0, 1070.0, 47.6902, -122.339, 1470.0, 4923.0, 86.0, 0.0, 0.0]</td>
      <td>[668288.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2023-07-14 15:43:58.844</td>
      <td>[4.0, 3.5, 3590.0, 5334.0, 2.0, 0.0, 2.0, 3.0, 9.0, 3140.0, 450.0, 47.6763, -122.267, 2100.0, 6250.0, 9.0, 0.0, 0.0]</td>
      <td>[1004846.5]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2023-07-14 15:43:58.844</td>
      <td>[3.0, 2.0, 1280.0, 960.0, 2.0, 0.0, 0.0, 3.0, 9.0, 1040.0, 240.0, 47.602, -122.311, 1280.0, 1173.0, 0.0, 0.0, 0.0]</td>
      <td>[684577.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2023-07-14 15:43:58.844</td>
      <td>[4.0, 2.5, 2820.0, 15000.0, 2.0, 0.0, 0.0, 4.0, 9.0, 2820.0, 0.0, 47.7255, -122.101, 2440.0, 15000.0, 29.0, 0.0, 0.0]</td>
      <td>[727898.1]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2023-07-14 15:43:58.844</td>
      <td>[3.0, 2.25, 1790.0, 11393.0, 1.0, 0.0, 0.0, 3.0, 8.0, 1790.0, 0.0, 47.6297, -122.099, 2290.0, 11894.0, 36.0, 0.0, 0.0]</td>
      <td>[559631.1]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2023-07-14 15:43:58.844</td>
      <td>[3.0, 1.5, 1010.0, 7683.0, 1.5, 0.0, 0.0, 5.0, 7.0, 1010.0, 0.0, 47.72, -122.318, 1550.0, 7271.0, 61.0, 0.0, 0.0]</td>
      <td>[340764.53]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2023-07-14 15:43:58.844</td>
      <td>[3.0, 2.0, 1270.0, 1323.0, 3.0, 0.0, 0.0, 3.0, 8.0, 1270.0, 0.0, 47.6934, -122.342, 1330.0, 1323.0, 8.0, 0.0, 0.0]</td>
      <td>[442168.06]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2023-07-14 15:43:58.844</td>
      <td>[4.0, 1.75, 2070.0, 9120.0, 1.0, 0.0, 0.0, 4.0, 7.0, 1250.0, 820.0, 47.6045, -122.123, 1650.0, 8400.0, 57.0, 0.0, 0.0]</td>
      <td>[630865.6]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2023-07-14 15:43:58.844</td>
      <td>[4.0, 1.0, 1620.0, 4080.0, 1.5, 0.0, 0.0, 3.0, 7.0, 1620.0, 0.0, 47.6696, -122.324, 1760.0, 4080.0, 91.0, 0.0, 0.0]</td>
      <td>[559631.1]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2023-07-14 15:43:58.844</td>
      <td>[4.0, 3.25, 3990.0, 9786.0, 2.0, 0.0, 0.0, 3.0, 9.0, 3990.0, 0.0, 47.6784, -122.026, 3920.0, 8200.0, 10.0, 0.0, 0.0]</td>
      <td>[909441.1]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2023-07-14 15:43:58.844</td>
      <td>[4.0, 2.0, 1780.0, 19843.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1780.0, 0.0, 47.4414, -122.154, 2210.0, 13500.0, 52.0, 0.0, 0.0]</td>
      <td>[313096.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2023-07-14 15:43:58.844</td>
      <td>[4.0, 2.5, 2130.0, 6003.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2130.0, 0.0, 47.4518, -122.12, 1940.0, 4529.0, 11.0, 0.0, 0.0]</td>
      <td>[404040.8]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2023-07-14 15:43:58.844</td>
      <td>[3.0, 1.75, 1660.0, 10440.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1040.0, 620.0, 47.4448, -121.77, 1240.0, 10380.0, 36.0, 0.0, 0.0]</td>
      <td>[292859.5]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2023-07-14 15:43:58.844</td>
      <td>[3.0, 2.5, 2110.0, 4118.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2110.0, 0.0, 47.3878, -122.153, 2110.0, 4044.0, 25.0, 0.0, 0.0]</td>
      <td>[338357.88]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2023-07-14 15:43:58.844</td>
      <td>[4.0, 2.25, 2200.0, 11250.0, 1.5, 0.0, 0.0, 5.0, 7.0, 1300.0, 900.0, 47.6845, -122.201, 2320.0, 10814.0, 94.0, 0.0, 0.0]</td>
      <td>[682284.6]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

### Standard Pipeline Logs

Pipeline logs are retrieved through the Wallaroo MLOps API with the following request.

* **REQUEST URL**
  * `v1/api/pipelines/get_logs`
* **Headers**
  * **Accept**:
    * `application/json; format=pandas-records`: For the logs returned as pandas DataFrame
    * `application/vnd.apache.arrow.file`: for the logs returned as Apache Arrow
* **PARAMETERS**
  * **pipeline_id** (*String* *Required*): The name of the pipeline.
  * **workspace_id** (*Integer* *Required*): The numerical identifier of the workspace.
  * **cursor** (*String* *Optional*): Cursor returned with a previous page of results from a pipeline log request, used to retrieve the next page of information.
  * **order**  (*String* *Optional* Default: `Desc`): The order for log inserts returned.  Valid values are:
    * `Asc`: In chronological order of inserts.
    * `Desc`: In reverse chronological order of inserts.
  * **page_size** (*Integer* *Optional* Default: `1000`.): Max records per page.
  * **start_time** (*String* *Optional*): The start time of the period to retrieve logs for in RFC 3339 format for DateTime.  **Must** be combined with `end_time`.
  * **end_time** (*String* *Optional*): The end time of the period to retrieve logs for in RFC 3339 format for DateTime.  **Must** be combined with `start_time`. 
* **RETURNS**
  * The logs are returned by default as  `'application/json; format=pandas-records'` format.  To request the logs as Apache Arrow tables, set the submission header `Accept` to `application/vnd.apache.arrow.file`.
  * Headers:
    * x-iteration-cursor: Used to retrieve the next page of results.  This is not included if `x-iteration-status` is `All`.
    * x-iteration-status: Informs whether there are more records available outside of this log request parameters.
      * All: This page includes all logs available from this request.  If `x-iteration-status` is `All`, then `x-iteration-cursor` is not provided.
      * SchemaChange: A change in the log schema caused by actions such as pipeline version, etc.
      * RecordLimited: The number of records exceeded from the page size, more records can be requested as the next page.  There **may** be more records available to retrieve OR the record limit was reached for this request even if no more records are available in next cursor request.
      * ByteLimited: The number of records exceeded the pipeline log limit which is around 100K. 

```python
# retrieve the authorization token
headers = wl.auth.auth_header()

url = f"{APIURL}/v1/api/pipelines/get_logs"

# Standard log retrieval

data = {
    'pipeline_id': main_pipeline_name,
    'workspace_id': workspace_id
}

response = requests.post(url, headers=headers, json=data)
standard_logs = pd.DataFrame.from_records(response.json())

display(len(standard_logs))
display(standard_logs.head(5).loc[:, ["time", "in", "out"]])
cursor = response.headers['x-iteration-cursor']
```

    2

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in</th>
      <th>out</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1689349425872</td>
      <td>{'tensor': [4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]}</td>
      <td>{'variable': [718013.7]}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1689349426983</td>
      <td>{'tensor': [4.0, 3.0, 3710.0, 20000.0, 2.0, 0.0, 2.0, 5.0, 10.0, 2760.0, 950.0, 47.6696, -122.261, 3970.0, 20000.0, 79.0, 0.0, 0.0]}</td>
      <td>{'variable': [1514079.4]}</td>
    </tr>
  </tbody>
</table>

```python
# Get next page of results as an arrow table

# retrieve the authorization token
headers = wl.auth.auth_header()
headers['Accept']="application/vnd.apache.arrow.file"

url = f"{APIURL}/v1/api/pipelines/get_logs"

# Standard log retrieval

data = {
    'pipeline_id': main_pipeline_name,
    'workspace_id': workspace_id,
    'cursor': cursor
}

response = requests.post(url, headers=headers, json=data)

# Arrow table is retrieved 
with pa.ipc.open_file(response.content) as reader:
    arrow_table = reader.read_all()

# convert to Polars DataFrame and display the first 5 rows
display(arrow_table.to_pandas().head(5).loc[:,["time", "out"]])
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1689349437595</td>
      <td>{'variable': [718013.75]}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1689349437595</td>
      <td>{'variable': [615094.56]}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1689349437595</td>
      <td>{'variable': [448627.72]}</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1689349437595</td>
      <td>{'variable': [758714.2]}</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1689349437595</td>
      <td>{'variable': [513264.7]}</td>
    </tr>
  </tbody>
</table>

```python
# Retrieve logs from specific date/time to only get the two DataFrame input inferences in ascending format

# retrieve the authorization token
headers = wl.auth.auth_header()

url = f"{APIURL}/v1/api/pipelines/get_logs"

# Standard log retrieval

data = {
    'pipeline_id': main_pipeline_name,
    'workspace_id': workspace_id,
    'order': 'Asc',
    'start_time': f'{dataframe_start.isoformat()}',
    'end_time': f'{dataframe_end.isoformat()}'
}

response = requests.post(url, headers=headers, json=data)
standard_logs = pd.DataFrame.from_records(response.json())

display(standard_logs.head(5).loc[:, ["time", "in", "out"]])
display(response.headers)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in</th>
      <th>out</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1689349425872</td>
      <td>{'tensor': [4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]}</td>
      <td>{'variable': [718013.7]}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1689349426983</td>
      <td>{'tensor': [4.0, 3.0, 3710.0, 20000.0, 2.0, 0.0, 2.0, 5.0, 10.0, 2760.0, 950.0, 47.6696, -122.261, 3970.0, 20000.0, 79.0, 0.0, 0.0]}</td>
      <td>{'variable': [1514079.4]}</td>
    </tr>
  </tbody>
</table>

    {'content-type': 'application/json; format=pandas-records', 'x-iteration-status': 'All', 'content-length': '867', 'date': 'Fri, 14 Jul 2023 15:44:37 GMT', 'x-envoy-upstream-service-time': '2', 'server': 'envoy'}

## Shadow Deploy Pipelines

Let's assume that after analyzing the assay information we want to test two challenger models to our control.  We do that with the Shadow Deploy pipeline step.

In Shadow Deploy, the pipeline step is added with the `add_shadow_deploy` method, with the champion model listed first, then an array of challenger models after.  **All** inference data is fed to **all** models, with the champion results displayed in the `out.variable` column, and the shadow results in the format `out_{model name}.variable`.  For example, since we named our challenger models `housingchallenger01` and `housingchallenger02`, the columns `out_housingchallenger01.variable` and `out_housingchallenger02.variable` have the shadow deployed model results.

For this example, we will remove the previous pipeline step, then replace it with a shadow deploy step with `rf_model.onnx` as our champion, and models `xgb_model.onnx` and `gbr_model.onnx` as the challengers.  We'll deploy the pipeline and prepare it for sample inferences.

```python
# Upload the challenger models

model_name_challenger01 = 'logcontrolchallenger01'
model_file_name_challenger01 = './models/xgb_model.onnx'

model_name_challenger02 = 'logcontrolchallenger02'
model_file_name_challenger02 = './models/gbr_model.onnx'

housing_model_challenger01 = (wl.upload_model(model_name_challenger01, 
                                              model_file_name_challenger01, 
                                              framework=wallaroo.framework.Framework.ONNX)
                                              .configure(tensor_fields=["tensor"])
                            )
housing_model_challenger02 = (wl.upload_model(model_name_challenger02, 
                                              model_file_name_challenger02, 
                                              framework=wallaroo.framework.Framework.ONNX).configure(tensor_fields=["tensor"])
                                )

```

```python
# Undeploy the pipeline
mainpipeline.undeploy()

mainpipeline.clear()

# Add the new shadow deploy step with our challenger models
mainpipeline.add_shadow_deploy(housing_model_control, [housing_model_challenger01, housing_model_challenger02])

# Deploy the pipeline with the new shadow step
mainpipeline.deploy()
```

<table><tr><th>name</th> <td>logapipipeline</td></tr><tr><th>created</th> <td>2023-07-14 15:43:25.566285+00:00</td></tr><tr><th>last_updated</th> <td>2023-07-14 15:45:23.038631+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>f2022a9f-1b94-4e23-9d19-05577f3d7010, 762a7f50-d1cc-4912-ab1d-5ed87b985797, 29b7109c-1467-40e1-aa11-dcb96959bb3e</td></tr><tr><th>steps</th> <td>logapicontrol</td></tr></table>

### Shadow Deploy Sample Inference

We'll now use our same sample data for an inference to our shadow deployed pipeline, then display the first 20 results with just the comparative outputs.

```python
shadow_date_start = datetime.datetime.now(datetime.timezone.utc)

shadow_result = mainpipeline.infer_from_file('./data/xtest-1k.arrow')

shadow_outputs =  shadow_result.to_pandas()
display(shadow_outputs.loc[0:20,['out.variable','out_logcontrolchallenger01.variable','out_logcontrolchallenger02.variable']])

shadow_date_end = datetime.datetime.now(datetime.timezone.utc)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out.variable</th>
      <th>out_logcontrolchallenger01.variable</th>
      <th>out_logcontrolchallenger02.variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[718013.75]</td>
      <td>[659806.0]</td>
      <td>[704901.9]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[615094.56]</td>
      <td>[732883.5]</td>
      <td>[695994.44]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[448627.72]</td>
      <td>[419508.84]</td>
      <td>[416164.8]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[758714.2]</td>
      <td>[634028.8]</td>
      <td>[655277.2]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[513264.7]</td>
      <td>[427209.44]</td>
      <td>[426854.66]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>[668288.0]</td>
      <td>[615501.9]</td>
      <td>[632556.1]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>[1004846.5]</td>
      <td>[1139732.5]</td>
      <td>[1100465.2]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>[684577.2]</td>
      <td>[498328.88]</td>
      <td>[528278.06]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>[727898.1]</td>
      <td>[722664.4]</td>
      <td>[659439.94]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>[559631.1]</td>
      <td>[525746.44]</td>
      <td>[534331.44]</td>
    </tr>
    <tr>
      <th>10</th>
      <td>[340764.53]</td>
      <td>[376337.1]</td>
      <td>[377187.2]</td>
    </tr>
    <tr>
      <th>11</th>
      <td>[442168.06]</td>
      <td>[382053.12]</td>
      <td>[403964.3]</td>
    </tr>
    <tr>
      <th>12</th>
      <td>[630865.6]</td>
      <td>[505608.97]</td>
      <td>[528991.3]</td>
    </tr>
    <tr>
      <th>13</th>
      <td>[559631.1]</td>
      <td>[603260.5]</td>
      <td>[612201.75]</td>
    </tr>
    <tr>
      <th>14</th>
      <td>[909441.1]</td>
      <td>[969585.4]</td>
      <td>[893874.7]</td>
    </tr>
    <tr>
      <th>15</th>
      <td>[313096.0]</td>
      <td>[313633.75]</td>
      <td>[318054.94]</td>
    </tr>
    <tr>
      <th>16</th>
      <td>[404040.8]</td>
      <td>[360413.56]</td>
      <td>[357816.75]</td>
    </tr>
    <tr>
      <th>17</th>
      <td>[292859.5]</td>
      <td>[316674.94]</td>
      <td>[294034.7]</td>
    </tr>
    <tr>
      <th>18</th>
      <td>[338357.88]</td>
      <td>[299907.44]</td>
      <td>[323254.3]</td>
    </tr>
    <tr>
      <th>19</th>
      <td>[682284.6]</td>
      <td>[811896.75]</td>
      <td>[770916.7]</td>
    </tr>
    <tr>
      <th>20</th>
      <td>[583765.94]</td>
      <td>[573618.5]</td>
      <td>[549141.4]</td>
    </tr>
  </tbody>
</table>

### Shadow Deploy Logs

Pipelines with a shadow deployed step include the shadow inference result in the same format as the inference result:  inference results from shadow deployed models are displayed as `out_{model name}.{output variable}`.

```python
# Retrieve logs from specific date/time to only get the two DataFrame input inferences in ascending format

# retrieve the authorization token
headers = wl.auth.auth_header()

url = f"{APIURL}/v1/api/pipelines/get_logs"

# Standard log retrieval

data = {
    'pipeline_id': main_pipeline_name,
    'workspace_id': workspace_id,
    'order': 'Asc',
    'start_time': f'{shadow_date_start.isoformat()}',
    'end_time': f'{shadow_date_end.isoformat()}'
}

response = requests.post(url, headers=headers, json=data)
standard_logs = pd.DataFrame.from_records(response.json())

display(standard_logs.head(5).loc[:, ["time", "out", "out_logcontrolchallenger01", "out_logcontrolchallenger02"]])
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out</th>
      <th>out_logcontrolchallenger01</th>
      <th>out_logcontrolchallenger02</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1689349535135</td>
      <td>{'variable': [718013.75]}</td>
      <td>{'variable': [659806.0]}</td>
      <td>{'variable': [704901.9]}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1689349535135</td>
      <td>{'variable': [615094.56]}</td>
      <td>{'variable': [732883.5]}</td>
      <td>{'variable': [695994.44]}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1689349535135</td>
      <td>{'variable': [448627.72]}</td>
      <td>{'variable': [419508.84]}</td>
      <td>{'variable': [416164.8]}</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1689349535135</td>
      <td>{'variable': [758714.2]}</td>
      <td>{'variable': [634028.8]}</td>
      <td>{'variable': [655277.2]}</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1689349535135</td>
      <td>{'variable': [513264.7]}</td>
      <td>{'variable': [427209.44]}</td>
      <td>{'variable': [426854.66]}</td>
    </tr>
  </tbody>
</table>

## A/B Testing Pipeline

A/B testing allows inference requests to be split between a control model and one or more challenger models.  For full details, see the [Pipeline Management Guide: A/B Testing](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipeline/#ab-testing).

When the inference results and log entries are displayed, they include the column `out._model_split` which displays:

| Field | Type | Description |
|---|---|---|
| `name` | String | The model name used for the inference.  |
| `version` | String| The version of the model. |
| `sha` | String | The sha hash of the model version. |

For this example, the shadow deployed step will be removed and replaced with an A/B Testing step with the ratio 1:1:1, so the control and each of the challenger models will be split randomly between inference requests.  A set of sample inferences will be run, then the pipeline logs displayed.

pipeline = (wl.build_pipeline("randomsplitpipeline-demo")
            .add_random_split([(2, control), (1, challenger)], "session_id"))

```python
ab_date_start = datetime.datetime.now(datetime.timezone.utc)
mainpipeline.undeploy()

# remove the shadow deploy steps
mainpipeline.clear()

# Add the a/b test step to the pipeline
mainpipeline.add_random_split([(1, housing_model_control), (1, housing_model_challenger01), (1, housing_model_challenger02)], "session_id")

mainpipeline.deploy()

# Perform sample inferences of 20 rows and display the results

abtesting_inputs = pd.read_json('./data/xtest-1k.df.json')

for index, row in abtesting_inputs.sample(20).iterrows():
    display(mainpipeline.infer(row.to_frame('tensor').reset_index()).loc[:,["out._model_split", "out.variable"]])

ab_date_end = datetime.datetime.now(datetime.timezone.utc)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out._model_split</th>
      <th>out.variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{"name":"logapicontrol","version":"448634a1-6f2b-438c-98fa-68268f151462","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}]</td>
      <td>[718013.7]</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out._model_split</th>
      <th>out.variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{"name":"logcontrolchallenger01","version":"bde52213-3828-4fd7-b286-09d2149d8a10","sha":"31e92d6ccb27b041a324a7ac22cf95d9d6cc3aa7e8263a229f7c4aec4938657c"}]</td>
      <td>[550902.5]</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out._model_split</th>
      <th>out.variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{"name":"logcontrolchallenger02","version":"52fdf218-5e90-457a-a956-d07d741d6dae","sha":"ed6065a79d841f7e96307bb20d5ef22840f15da0b587efb51425c7ad60589d6a"}]</td>
      <td>[329266.97]</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out._model_split</th>
      <th>out.variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{"name":"logapicontrol","version":"448634a1-6f2b-438c-98fa-68268f151462","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}]</td>
      <td>[450867.7]</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out._model_split</th>
      <th>out.variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{"name":"logapicontrol","version":"448634a1-6f2b-438c-98fa-68268f151462","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}]</td>
      <td>[499651.56]</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out._model_split</th>
      <th>out.variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{"name":"logcontrolchallenger02","version":"52fdf218-5e90-457a-a956-d07d741d6dae","sha":"ed6065a79d841f7e96307bb20d5ef22840f15da0b587efb51425c7ad60589d6a"}]</td>
      <td>[294921.5]</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out._model_split</th>
      <th>out.variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{"name":"logcontrolchallenger01","version":"bde52213-3828-4fd7-b286-09d2149d8a10","sha":"31e92d6ccb27b041a324a7ac22cf95d9d6cc3aa7e8263a229f7c4aec4938657c"}]</td>
      <td>[420434.13]</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out._model_split</th>
      <th>out.variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{"name":"logapicontrol","version":"448634a1-6f2b-438c-98fa-68268f151462","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}]</td>
      <td>[381737.6]</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out._model_split</th>
      <th>out.variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{"name":"logcontrolchallenger01","version":"bde52213-3828-4fd7-b286-09d2149d8a10","sha":"31e92d6ccb27b041a324a7ac22cf95d9d6cc3aa7e8263a229f7c4aec4938657c"}]</td>
      <td>[299659.7]</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out._model_split</th>
      <th>out.variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{"name":"logcontrolchallenger02","version":"52fdf218-5e90-457a-a956-d07d741d6dae","sha":"ed6065a79d841f7e96307bb20d5ef22840f15da0b587efb51425c7ad60589d6a"}]</td>
      <td>[349665.53]</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out._model_split</th>
      <th>out.variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{"name":"logapicontrol","version":"448634a1-6f2b-438c-98fa-68268f151462","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}]</td>
      <td>[293808.03]</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out._model_split</th>
      <th>out.variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{"name":"logcontrolchallenger01","version":"bde52213-3828-4fd7-b286-09d2149d8a10","sha":"31e92d6ccb27b041a324a7ac22cf95d9d6cc3aa7e8263a229f7c4aec4938657c"}]</td>
      <td>[186544.78]</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out._model_split</th>
      <th>out.variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{"name":"logapicontrol","version":"448634a1-6f2b-438c-98fa-68268f151462","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}]</td>
      <td>[294203.53]</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out._model_split</th>
      <th>out.variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{"name":"logapicontrol","version":"448634a1-6f2b-438c-98fa-68268f151462","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}]</td>
      <td>[289359.47]</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out._model_split</th>
      <th>out.variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{"name":"logcontrolchallenger01","version":"bde52213-3828-4fd7-b286-09d2149d8a10","sha":"31e92d6ccb27b041a324a7ac22cf95d9d6cc3aa7e8263a229f7c4aec4938657c"}]</td>
      <td>[589324.8]</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out._model_split</th>
      <th>out.variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{"name":"logapicontrol","version":"448634a1-6f2b-438c-98fa-68268f151462","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}]</td>
      <td>[271309.13]</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out._model_split</th>
      <th>out.variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{"name":"logapicontrol","version":"448634a1-6f2b-438c-98fa-68268f151462","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}]</td>
      <td>[465299.9]</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out._model_split</th>
      <th>out.variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{"name":"logapicontrol","version":"448634a1-6f2b-438c-98fa-68268f151462","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}]</td>
      <td>[247792.75]</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out._model_split</th>
      <th>out.variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{"name":"logcontrolchallenger01","version":"bde52213-3828-4fd7-b286-09d2149d8a10","sha":"31e92d6ccb27b041a324a7ac22cf95d9d6cc3aa7e8263a229f7c4aec4938657c"}]</td>
      <td>[413473.8]</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out._model_split</th>
      <th>out.variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{"name":"logcontrolchallenger01","version":"bde52213-3828-4fd7-b286-09d2149d8a10","sha":"31e92d6ccb27b041a324a7ac22cf95d9d6cc3aa7e8263a229f7c4aec4938657c"}]</td>
      <td>[442778.22]</td>
    </tr>
  </tbody>
</table>

### Retrieve A/B Testing Log Files through API

The log files for A/B Testing pipeline inference results contain the model information with the model outputs in the `out` field.

```python
# Retrieve logs from specific date/time to only get the two DataFrame input inferences in ascending format

# retrieve the authorization token
headers = wl.auth.auth_header()

url = f"{APIURL}/v1/api/pipelines/get_logs"

# Standard log retrieval

data = {
    'pipeline_id': main_pipeline_name,
    'workspace_id': workspace_id,
    'order': 'Asc',
    'start_time': f'{ab_date_start.isoformat()}',
    'end_time': f'{ab_date_end.isoformat()}'
}

response = requests.post(url, headers=headers, json=data)
standard_logs = pd.DataFrame.from_records(response.json())

display(standard_logs.head(5).loc[:, ["time", "out"]])
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1689349586459</td>
      <td>{'_model_split': ['{"name":"logapicontrol","version":"448634a1-6f2b-438c-98fa-68268f151462","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}'], 'variable': [718013.7]}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1689349586894</td>
      <td>{'_model_split': ['{"name":"logcontrolchallenger01","version":"bde52213-3828-4fd7-b286-09d2149d8a10","sha":"31e92d6ccb27b041a324a7ac22cf95d9d6cc3aa7e8263a229f7c4aec4938657c"}'], 'variable': [550902.5]}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1689349587285</td>
      <td>{'_model_split': ['{"name":"logcontrolchallenger02","version":"52fdf218-5e90-457a-a956-d07d741d6dae","sha":"ed6065a79d841f7e96307bb20d5ef22840f15da0b587efb51425c7ad60589d6a"}'], 'variable': [329266.97]}</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1689349587672</td>
      <td>{'_model_split': ['{"name":"logapicontrol","version":"448634a1-6f2b-438c-98fa-68268f151462","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}'], 'variable': [450867.7]}</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1689349588092</td>
      <td>{'_model_split': ['{"name":"logapicontrol","version":"448634a1-6f2b-438c-98fa-68268f151462","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}'], 'variable': [499651.56]}</td>
    </tr>
  </tbody>
</table>

### Undeploy Main Pipeline

With the examples and tutorial complete, we will undeploy the main pipeline and return the resources back to the Wallaroo instance.

```python
mainpipeline.undeploy()
```

<table><tr><th>name</th> <td>logapipipeline</td></tr><tr><th>created</th> <td>2023-07-14 15:43:25.566285+00:00</td></tr><tr><th>last_updated</th> <td>2023-07-14 15:46:15.685023+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>43cbb475-5eaf-4aaf-a6b1-63edc77f44a8, f2022a9f-1b94-4e23-9d19-05577f3d7010, 762a7f50-d1cc-4912-ab1d-5ed87b985797, 29b7109c-1467-40e1-aa11-dcb96959bb3e</td></tr><tr><th>steps</th> <td>logapicontrol</td></tr></table>

