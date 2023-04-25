This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-features/pipeline_log_tutorial/).

## Pipeline Log Tutorial

This tutorial demonstrates Wallaroo Pipeline logs and 

This tutorial will demonstrate how to:

1. Select or create a workspace, pipeline and upload the control model, then additional models for A/B Testing and Shadow Deploy.
1. Add a pipeline step with the champion model, then deploy the pipeline and perform sample inferences.
1. Display the various log types for a standard deployed pipeline.
1. Swap out the pipeline step with the champion model with a shadow deploy step that compares the champion model against two competitors.
1. Perform sample inferences with a shadow deployed step, then display the log files for a shadow deployed pipeline.
1. Swap out the shadow deployed pipeline step with an A/B pipeline step.
1. Perform sample inferences with a A/B pipeline step, then display the log files for an A/B pipeline step.
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

import os
# For Wallaroo SDK 2023.1
os.environ["ARROW_ENABLED"]="True"

import datetime
```

### Connect to Wallaroo Instance

The following command will create a connection to the Wallaroo instance and store it in the variable `wl`.

```python
# Login through local Wallaroo instance

wl = wallaroo.Client()

# SSO login through keycloak

# wallarooPrefix = "YOUR PREFIX"
# wallarooSuffix = "YOUR PREFIX"

# wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}.api.{wallarooSuffix}", 
#                     auth_endpoint=f"https://{wallarooPrefix}.keycloak.{wallarooSuffix}", 
#                     auth_type="sso")
```

### Create Workspace

We will create a workspace to manage our pipeline and models.  The following variables will set the name of our sample workspace then set it as the current workspace.

```python
workspace_name = 'pipelinelogworkspace'
main_pipeline_name = 'pipelinelogpipeline'
model_name_control = 'logcontrol'
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
```

    {'name': 'pipelinelogworkspace', 'id': 7, 'archived': False, 'created_by': '5c1a9977-8bad-414f-8a99-f72e7a4c2d5d', 'created_at': '2023-04-25T17:44:02.525825+00:00', 'models': [], 'pipelines': []}

## Standard Pipeline

### Upload The Champion Model

For our example, we will upload the champion model that has been trained to derive house prices from a variety of inputs.  The model file is `rf_model.onnx`, and is uploaded with the name `housingcontrol`.

```python
housing_model_control = wl.upload_model(model_name_control, model_file_name_control).configure()
```

### Build the Pipeline

This pipeline is made to be an example of an existing situation where a model is deployed and being used for inferences in a production environment.  We'll call it `housepricepipeline`, set `housingcontrol` as a pipeline step, then run a few sample inferences.

```python
mainpipeline = wl.build_pipeline(main_pipeline_name)
# in case this pipeline was run before
mainpipeline.clear()
mainpipeline.add_model_step(housing_model_control).deploy()
```

{{<table "table table-striped table-bordered">}}
<table><tr><th>name</th> <td>pipelinelogpipeline</td></tr><tr><th>created</th> <td>2023-04-25 17:44:05.525160+00:00</td></tr><tr><th>last_updated</th> <td>2023-04-25 17:44:06.188833+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>950589d2-0d7d-4bb4-bc0f-e63e1bf39e59, e68eb2bd-2ab4-4223-9e6b-9b570ba0f324</td></tr><tr><th>steps</th> <td>logcontrol</td></tr></table>
{{</table>}}

### Testing

We'll use two inferences as a quick sample test - one that has a house that should be determined around $700k, the other with a house determined to be around $1.5 million.  We'll also save the start and end periods for these events to for later log functionality.

```python
dataframe_start = datetime.datetime.now()

normal_input = pd.DataFrame.from_records({"tensor": [[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]]})
result = mainpipeline.infer(normal_input)
display(result)
```

{{<table "table table-striped table-bordered">}}
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
      <td>2023-04-25 17:44:17.455</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.7]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}

```python
large_house_input = pd.DataFrame.from_records({'tensor': [[4.0, 3.0, 3710.0, 20000.0, 2.0, 0.0, 2.0, 5.0, 10.0, 2760.0, 950.0, 47.6696, -122.261, 3970.0, 20000.0, 79.0, 0.0, 0.0]]})
large_house_result = mainpipeline.infer(large_house_input)
display(large_house_result)
dataframe_end = datetime.datetime.now()
```

{{<table "table table-striped table-bordered">}}
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
      <td>2023-04-25 17:44:17.871</td>
      <td>[4.0, 3.0, 3710.0, 20000.0, 2.0, 0.0, 2.0, 5.0, 10.0, 2760.0, 950.0, 47.6696, -122.261, 3970.0, 20000.0, 79.0, 0.0, 0.0]</td>
      <td>[1514079.4]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}

As one last sample, we'll run through roughly 1,000 inferences at once and show a few of the results.  For this example we'll use an Apache Arrow table, which has a smaller file size compared to uploading a pandas DataFrame JSON file.  The inference result is returned as an arrow table, which we'll convert into a pandas DataFrame to display the first 20 results.

```python
arrow_start = datetime.datetime.now()

batch_inferences = mainpipeline.infer_from_file('./data/xtest-1k.arrow')

arrow_end = datetime.datetime.now()
large_inference_result =  batch_inferences.to_pandas()
display(large_inference_result.head(20))
```

{{<table "table table-striped table-bordered">}}
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
      <td>2023-04-25 17:44:18.461</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.75]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[615094.56]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
      <td>[448627.72]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[758714.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 1.75, 2200.0, 11520.0, 1.0, 0.0, 0.0, 4.0, 7.0, 2200.0, 0.0, 47.7659, -122.341, 1690.0, 8038.0, 62.0, 0.0, 0.0]</td>
      <td>[513264.7]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 2.0, 2140.0, 4923.0, 1.0, 0.0, 0.0, 4.0, 8.0, 1070.0, 1070.0, 47.6902, -122.339, 1470.0, 4923.0, 86.0, 0.0, 0.0]</td>
      <td>[668288.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[4.0, 3.5, 3590.0, 5334.0, 2.0, 0.0, 2.0, 3.0, 9.0, 3140.0, 450.0, 47.6763, -122.267, 2100.0, 6250.0, 9.0, 0.0, 0.0]</td>
      <td>[1004846.5]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 2.0, 1280.0, 960.0, 2.0, 0.0, 0.0, 3.0, 9.0, 1040.0, 240.0, 47.602, -122.311, 1280.0, 1173.0, 0.0, 0.0, 0.0]</td>
      <td>[684577.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[4.0, 2.5, 2820.0, 15000.0, 2.0, 0.0, 0.0, 4.0, 9.0, 2820.0, 0.0, 47.7255, -122.101, 2440.0, 15000.0, 29.0, 0.0, 0.0]</td>
      <td>[727898.1]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 2.25, 1790.0, 11393.0, 1.0, 0.0, 0.0, 3.0, 8.0, 1790.0, 0.0, 47.6297, -122.099, 2290.0, 11894.0, 36.0, 0.0, 0.0]</td>
      <td>[559631.1]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 1.5, 1010.0, 7683.0, 1.5, 0.0, 0.0, 5.0, 7.0, 1010.0, 0.0, 47.72, -122.318, 1550.0, 7271.0, 61.0, 0.0, 0.0]</td>
      <td>[340764.53]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 2.0, 1270.0, 1323.0, 3.0, 0.0, 0.0, 3.0, 8.0, 1270.0, 0.0, 47.6934, -122.342, 1330.0, 1323.0, 8.0, 0.0, 0.0]</td>
      <td>[442168.06]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[4.0, 1.75, 2070.0, 9120.0, 1.0, 0.0, 0.0, 4.0, 7.0, 1250.0, 820.0, 47.6045, -122.123, 1650.0, 8400.0, 57.0, 0.0, 0.0]</td>
      <td>[630865.6]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[4.0, 1.0, 1620.0, 4080.0, 1.5, 0.0, 0.0, 3.0, 7.0, 1620.0, 0.0, 47.6696, -122.324, 1760.0, 4080.0, 91.0, 0.0, 0.0]</td>
      <td>[559631.1]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[4.0, 3.25, 3990.0, 9786.0, 2.0, 0.0, 0.0, 3.0, 9.0, 3990.0, 0.0, 47.6784, -122.026, 3920.0, 8200.0, 10.0, 0.0, 0.0]</td>
      <td>[909441.1]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[4.0, 2.0, 1780.0, 19843.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1780.0, 0.0, 47.4414, -122.154, 2210.0, 13500.0, 52.0, 0.0, 0.0]</td>
      <td>[313096.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[4.0, 2.5, 2130.0, 6003.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2130.0, 0.0, 47.4518, -122.12, 1940.0, 4529.0, 11.0, 0.0, 0.0]</td>
      <td>[404040.8]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 1.75, 1660.0, 10440.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1040.0, 620.0, 47.4448, -121.77, 1240.0, 10380.0, 36.0, 0.0, 0.0]</td>
      <td>[292859.5]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 2.5, 2110.0, 4118.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2110.0, 0.0, 47.3878, -122.153, 2110.0, 4044.0, 25.0, 0.0, 0.0]</td>
      <td>[338357.88]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[4.0, 2.25, 2200.0, 11250.0, 1.5, 0.0, 0.0, 5.0, 7.0, 1300.0, 900.0, 47.6845, -122.201, 2320.0, 10814.0, 94.0, 0.0, 0.0]</td>
      <td>[682284.6]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}

### Standard Pipeline Logs

Pipeline logs with standard pipeline steps are retrieved either with:

* Pipeline `logs` which returns either a pandas DataFrame or Apache Arrow table.
* Pipeline `export_logs` which saves the logs either a pandas DataFrame JSON file or Apache Arrow table.

For full details, see the Wallaroo Documentation Pipeline Log Management guide.

#### Pipeline Log Method

The Pipeline `logs` method accepts the following parameters.

| Parameter | Type | Description |
|---|---|---|
| `limit` | **Int** (*Optional*) | Limits how many log records to display.  Defaults to `100`.  If there are more pipeline logs than are being displayed, the **Warning** message `Pipeline log record limit exceeded` will be displayed.  For example, if 100 log files were requested and there are a total of 1,000, the warning message will be displayed. |
| `start_datetime` and `end_datetime` | **DateTime** (*Optional*) | Limits logs to all logs between the `start` and `end` DateTime parameters.  **Both parameters must be provided**. Submitting a `logs()` request with only `start_datetime` or `end_datetime` will generate an exception.<br />If `start_datetime` and `end_datetime` are provided as parameters, then the records are returned in **chronological** order, with the oldest record displayed first. |
| `arrow` | **Boolean** (*Optional*) | Defaults to **False**.  If `arrow` is set to `True`, then the logs are returned as an [Apache Arrow table](https://arrow.apache.org/).  If `arrow=False`, then the logs are returned as a pandas DataFrame. |

##### Pipeline Log Warnings

If the total number of logs the either the set limit or 10 MB in file size, the following warning is returned:

`Warning: There are more logs available. Please set a larger limit or request a file using export_logs.`

If the total number of logs **requested** either through the limit or through the `start_datetime` and `end_datetime` request is greater than 10 MB in size, the following error is displayed:

`Warning: Pipeline log size limit exceeded. Only displaying 509 log messages. Please request a file using export_logs.`

The following examples demonstrate displaying the logs, then displaying the logs between the `control_model_start` and `control_model_end` periods, then again retrieved as an Arrow table with the logs limited to only 5 entries.

```python
# pipeline log retrieval - reverse chronological order

regular_logs = mainpipeline.logs()

display(len(regular_logs))
display(regular_logs)

# Display arrow logs

display(arrow_start.isoformat() )
display(arrow_end.isoformat() )

arrow_logs = mainpipeline.logs(start_datetime=arrow_start, end_datetime=arrow_end)

display(len(arrow_logs))
display(arrow_logs)

# pipeline log retrieval limited to the last 5 an an arrow table

display(mainpipeline.logs(limit=5, arrow=True))
```

    Warning: There are more logs available. Please set a larger limit or request a file using export_logs.

    100

{{<table "table table-striped table-bordered">}}
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
      <td>2023-04-25 17:44:18.461</td>
      <td>[2.0, 1.5, 1780.0, 4750.0, 1.0, 0.0, 0.0, 4.0, 7.0, 1080.0, 700.0, 47.6859, -122.395, 1690.0, 5962.0, 67.0, 0.0, 0.0]</td>
      <td>[558463.3]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 2.75, 3010.0, 1842.0, 2.0, 0.0, 0.0, 3.0, 9.0, 3010.0, 0.0, 47.5836, -121.994, 2950.0, 4200.0, 3.0, 0.0, 0.0]</td>
      <td>[795841.06]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[4.0, 3.5, 4460.0, 16271.0, 2.0, 0.0, 2.0, 3.0, 11.0, 4460.0, 0.0, 47.5862, -121.97, 4540.0, 17122.0, 13.0, 0.0, 0.0]</td>
      <td>[1208638.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[5.0, 1.75, 2330.0, 6450.0, 1.0, 0.0, 1.0, 3.0, 8.0, 1330.0, 1000.0, 47.4959, -122.367, 2330.0, 8258.0, 57.0, 0.0, 0.0]</td>
      <td>[448720.28]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 2.5, 1750.0, 7208.0, 2.0, 0.0, 0.0, 3.0, 8.0, 1750.0, 0.0, 47.4315, -122.192, 2050.0, 7524.0, 20.0, 0.0, 0.0]</td>
      <td>[311909.6]</td>
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
      <th>95</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 2.5, 2430.0, 88426.0, 1.0, 0.0, 0.0, 4.0, 7.0, 1570.0, 860.0, 47.4828, -121.718, 1560.0, 56827.0, 29.0, 0.0, 0.0]</td>
      <td>[418823.4]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>96</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[2.0, 1.0, 810.0, 5100.0, 1.0, 0.0, 0.0, 3.0, 6.0, 810.0, 0.0, 47.7317, -122.343, 1500.0, 5100.0, 59.0, 0.0, 0.0]</td>
      <td>[340764.53]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>97</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[4.0, 1.75, 2180.0, 7876.0, 1.0, 0.0, 0.0, 4.0, 7.0, 1290.0, 890.0, 47.5157, -122.191, 1960.0, 7225.0, 38.0, 0.0, 0.0]</td>
      <td>[395096.03]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>98</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[4.0, 2.25, 2590.0, 12600.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2590.0, 0.0, 47.5566, -122.162, 2620.0, 11050.0, 36.0, 0.0, 0.0]</td>
      <td>[758714.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>99</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 2.0, 1080.0, 9225.0, 1.0, 0.0, 0.0, 2.0, 7.0, 1080.0, 0.0, 47.4842, -122.346, 1410.0, 9840.0, 59.0, 0.0, 0.0]</td>
      <td>[236238.66]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}
<p>100 rows × 4 columns</p>

    '2023-04-25T11:44:17.877271'

    '2023-04-25T11:44:18.623504'

    Warning: Pipeline log size limit exceeded. Only displaying 538 log messages. Please request a file using export_logs.

    538

{{<table "table table-striped table-bordered">}}
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
      <td>2023-04-25 17:44:18.461</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.75]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[615094.56]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
      <td>[448627.72]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[758714.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-04-25 17:44:18.461</td>
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
      <th>533</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 2.5, 1750.0, 7208.0, 2.0, 0.0, 0.0, 3.0, 8.0, 1750.0, 0.0, 47.4315, -122.192, 2050.0, 7524.0, 20.0, 0.0, 0.0]</td>
      <td>[311909.6]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>534</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[5.0, 1.75, 2330.0, 6450.0, 1.0, 0.0, 1.0, 3.0, 8.0, 1330.0, 1000.0, 47.4959, -122.367, 2330.0, 8258.0, 57.0, 0.0, 0.0]</td>
      <td>[448720.28]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>535</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[4.0, 3.5, 4460.0, 16271.0, 2.0, 0.0, 2.0, 3.0, 11.0, 4460.0, 0.0, 47.5862, -121.97, 4540.0, 17122.0, 13.0, 0.0, 0.0]</td>
      <td>[1208638.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>536</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 2.75, 3010.0, 1842.0, 2.0, 0.0, 0.0, 3.0, 9.0, 3010.0, 0.0, 47.5836, -121.994, 2950.0, 4200.0, 3.0, 0.0, 0.0]</td>
      <td>[795841.06]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>537</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[2.0, 1.5, 1780.0, 4750.0, 1.0, 0.0, 0.0, 4.0, 7.0, 1080.0, 700.0, 47.6859, -122.395, 1690.0, 5962.0, 67.0, 0.0, 0.0]</td>
      <td>[558463.3]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}
<p>538 rows × 4 columns</p>

    Warning: There are more logs available. Please set a larger limit or request a file using export_logs.

    pyarrow.Table
    time: timestamp[ms]
    in.tensor: list<item: float> not null
      child 0, item: float
    out.variable: list<inner: float not null> not null
      child 0, inner: float not null
    check_failures: int8
    ----
    time: [[2023-04-25 17:44:18.461,2023-04-25 17:44:18.461,2023-04-25 17:44:18.461,2023-04-25 17:44:18.461,2023-04-25 17:44:18.461]]
    in.tensor: [[[3,2,2005,7000,1,...,1750,4500,34,0,0],[3,1.75,2910,37461,1,...,2520,18295,47,0,0],...,[4,1.75,2700,7875,1.5,...,2220,7875,46,0,0],[3,2.5,2900,23550,1,...,2900,19604,27,0,0]]]
    out.variable: [[[581003],[706823.56],...,[441960.38],[827411]]]
    check_failures: [[0,0,0,0,0]]

#### Pipeline Limits

In a previous step we performed 10,000 inferences at once.  If we attempt to pull them at once, we'll likely run into the size limit for this pipeline and receive the following warning message indicating that the pipeline size limits were exceeded and we should use `export_logs` instead.

`Warning: Pipeline log size limit exceeded. Only displaying 1000 log messages (of 10000 requested). Please request a file using export_logs.`

```python
logs = mainpipeline.logs(limit=10000)
display(logs)
```

    Warning: Pipeline log size limit exceeded. Only displaying 1000 log messages (of 10000 requested). Please request a file using export_logs.

{{<table "table table-striped table-bordered">}}
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
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 2.0, 2005.0, 7000.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1605.0, 400.0, 47.6039, -122.298, 1750.0, 4500.0, 34.0, 0.0, 0.0]</td>
      <td>[581003.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 1.75, 2910.0, 37461.0, 1.0, 0.0, 0.0, 4.0, 7.0, 1530.0, 1380.0, 47.7015, -122.164, 2520.0, 18295.0, 47.0, 0.0, 0.0]</td>
      <td>[706823.56]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[4.0, 3.25, 2910.0, 1880.0, 2.0, 0.0, 3.0, 5.0, 9.0, 1830.0, 1080.0, 47.616, -122.282, 3100.0, 8200.0, 100.0, 0.0, 0.0]</td>
      <td>[1060847.5]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[4.0, 1.75, 2700.0, 7875.0, 1.5, 0.0, 0.0, 4.0, 8.0, 2700.0, 0.0, 47.454, -122.144, 2220.0, 7875.0, 46.0, 0.0, 0.0]</td>
      <td>[441960.38]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 2.5, 2900.0, 23550.0, 1.0, 0.0, 0.0, 3.0, 10.0, 1490.0, 1410.0, 47.5708, -122.153, 2900.0, 19604.0, 27.0, 0.0, 0.0]</td>
      <td>[827411.0]</td>
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
      <th>995</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 1.75, 2200.0, 11520.0, 1.0, 0.0, 0.0, 4.0, 7.0, 2200.0, 0.0, 47.7659, -122.341, 1690.0, 8038.0, 62.0, 0.0, 0.0]</td>
      <td>[513264.7]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>996</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[758714.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>997</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
      <td>[448627.72]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>998</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[615094.56]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>999</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.75]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}
<p>1000 rows × 4 columns</p>

#### Pipeline export_logs Method

The Pipeline method `export_logs` returns the Pipeline records as either a DataFrame JSON file, or an Apache Arrow table file.  For full details, see the Wallaroo Pipeline Log Management guide.

The `export_logs` method takes the following parameters:

| Parameter | Type | Description |
|---|---|---|
| `limit` | **Int** (*Optional*) | Limits how many log records to display.  Defaults to `100`.  If there are more pipeline logs than are being displayed, the **Warning** message `Pipeline log record limit exceeded` will be displayed.  For example, if 100 log files were requested and there are a total of 1,000, the warning message will be displayed. |
| `start` and `end` | **DateTime** (*Optional*) | Limits logs to all logs between the `start` and `end` DateTime parameters.  **Both parameters must be provided**. Submitting a `logs()` request with only `start` or `end` will generate an exception.<br />If `start` and `end` are provided as parameters, then the records are returned in **chronological** order, with the oldest record displayed first. |
| `filename` | **String** (*Required*) | The file name to save the log file to.  The requesting user must have write access to the file location. |
| `arrow` | **Boolean** (*Optional*) | Defaults to **False**.  If `arrow` is set to `True`, then the logs are returned as an [Apache Arrow table](https://arrow.apache.org/).  If `arrow=False`, then the logs are returned as JSON in pandas DataFrame format. |

The following examples demonstrate saving a DataFrame version of the `mainpipeline` logs, then an Arrow version.

```python
# Save the DataFrame version of the log file

mainpipeline.export_logs(filename="mainpipeline_logs.df.json")
data_df = pd.read_json("mainpipeline_logs.df.json", lines=True)
display(data_df)

# Save the Arrow version of the log file

mainpipeline.export_logs(filename="mainpipeline_logs.arrow", arrow=True)

with pa.ipc.open_file("mainpipeline_logs.arrow") as reader:
    results = reader.read_all()

results.to_pandas()
```

{{<table "table table-striped table-bordered">}}
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
      <td>2023-04-25T17:44:18.461Z</td>
      <td>[3.0, 2.0, 2005.0, 7000.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1605.0, 400.0, 47.6039009094, -122.297996521, 1750.0, 4500.0, 34.0, 0.0, 0.0]</td>
      <td>[581003.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-04-25T17:44:18.461Z</td>
      <td>[3.0, 1.75, 2910.0, 37461.0, 1.0, 0.0, 0.0, 4.0, 7.0, 1530.0, 1380.0, 47.701499939, -122.1640014648, 2520.0, 18295.0, 47.0, 0.0, 0.0]</td>
      <td>[706823.5625]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-04-25T17:44:18.461Z</td>
      <td>[4.0, 3.25, 2910.0, 1880.0, 2.0, 0.0, 3.0, 5.0, 9.0, 1830.0, 1080.0, 47.6160011292, -122.2819976807, 3100.0, 8200.0, 100.0, 0.0, 0.0]</td>
      <td>[1060847.5]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-04-25T17:44:18.461Z</td>
      <td>[4.0, 1.75, 2700.0, 7875.0, 1.5, 0.0, 0.0, 4.0, 8.0, 2700.0, 0.0, 47.4539985657, -122.1439971924, 2220.0, 7875.0, 46.0, 0.0, 0.0]</td>
      <td>[441960.375]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-04-25T17:44:18.461Z</td>
      <td>[3.0, 2.5, 2900.0, 23550.0, 1.0, 0.0, 0.0, 3.0, 10.0, 1490.0, 1410.0, 47.5708007812, -122.1529998779, 2900.0, 19604.0, 27.0, 0.0, 0.0]</td>
      <td>[827411.0]</td>
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
      <th>95</th>
      <td>2023-04-25T17:44:18.461Z</td>
      <td>[2.0, 1.5, 1070.0, 1236.0, 2.0, 0.0, 0.0, 3.0, 8.0, 1000.0, 70.0, 47.5619010925, -122.3820037842, 1170.0, 1888.0, 10.0, 0.0, 0.0]</td>
      <td>[435628.5625]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>96</th>
      <td>2023-04-25T17:44:18.461Z</td>
      <td>[3.0, 2.5, 2830.0, 6000.0, 1.0, 0.0, 3.0, 3.0, 9.0, 1730.0, 1100.0, 47.5750999451, -122.3779983521, 2040.0, 5300.0, 60.0, 0.0, 0.0]</td>
      <td>[981676.625]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>97</th>
      <td>2023-04-25T17:44:18.461Z</td>
      <td>[4.0, 1.75, 1720.0, 8750.0, 1.0, 0.0, 0.0, 3.0, 7.0, 860.0, 860.0, 47.7260017395, -122.2099990845, 1790.0, 8750.0, 43.0, 0.0, 0.0]</td>
      <td>[437177.84375]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>98</th>
      <td>2023-04-25T17:44:18.461Z</td>
      <td>[4.0, 2.25, 4470.0, 60373.0, 2.0, 0.0, 0.0, 3.0, 11.0, 4470.0, 0.0, 47.7289009094, -122.1269989014, 3210.0, 40450.0, 26.0, 0.0, 0.0]</td>
      <td>[1208638.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>99</th>
      <td>2023-04-25T17:44:18.461Z</td>
      <td>[3.0, 1.0, 1150.0, 3000.0, 1.0, 0.0, 0.0, 5.0, 6.0, 1150.0, 0.0, 47.6866989136, -122.3450012207, 1460.0, 3200.0, 108.0, 0.0, 0.0]</td>
      <td>[448627.71875]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}
<p>100 rows × 4 columns</p>

{{<table "table table-striped table-bordered">}}
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
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 2.0, 2005.0, 7000.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1605.0, 400.0, 47.6039, -122.298, 1750.0, 4500.0, 34.0, 0.0, 0.0]</td>
      <td>[581003.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 1.75, 2910.0, 37461.0, 1.0, 0.0, 0.0, 4.0, 7.0, 1530.0, 1380.0, 47.7015, -122.164, 2520.0, 18295.0, 47.0, 0.0, 0.0]</td>
      <td>[706823.56]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[4.0, 3.25, 2910.0, 1880.0, 2.0, 0.0, 3.0, 5.0, 9.0, 1830.0, 1080.0, 47.616, -122.282, 3100.0, 8200.0, 100.0, 0.0, 0.0]</td>
      <td>[1060847.5]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[4.0, 1.75, 2700.0, 7875.0, 1.5, 0.0, 0.0, 4.0, 8.0, 2700.0, 0.0, 47.454, -122.144, 2220.0, 7875.0, 46.0, 0.0, 0.0]</td>
      <td>[441960.38]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 2.5, 2900.0, 23550.0, 1.0, 0.0, 0.0, 3.0, 10.0, 1490.0, 1410.0, 47.5708, -122.153, 2900.0, 19604.0, 27.0, 0.0, 0.0]</td>
      <td>[827411.0]</td>
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
      <th>95</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[2.0, 1.5, 1070.0, 1236.0, 2.0, 0.0, 0.0, 3.0, 8.0, 1000.0, 70.0, 47.5619, -122.382, 1170.0, 1888.0, 10.0, 0.0, 0.0]</td>
      <td>[435628.56]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>96</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 2.5, 2830.0, 6000.0, 1.0, 0.0, 3.0, 3.0, 9.0, 1730.0, 1100.0, 47.5751, -122.378, 2040.0, 5300.0, 60.0, 0.0, 0.0]</td>
      <td>[981676.6]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>97</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[4.0, 1.75, 1720.0, 8750.0, 1.0, 0.0, 0.0, 3.0, 7.0, 860.0, 860.0, 47.726, -122.21, 1790.0, 8750.0, 43.0, 0.0, 0.0]</td>
      <td>[437177.84]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>98</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[4.0, 2.25, 4470.0, 60373.0, 2.0, 0.0, 0.0, 3.0, 11.0, 4470.0, 0.0, 47.7289, -122.127, 3210.0, 40450.0, 26.0, 0.0, 0.0]</td>
      <td>[1208638.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>99</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 1.0, 1150.0, 3000.0, 1.0, 0.0, 0.0, 5.0, 6.0, 1150.0, 0.0, 47.6867, -122.345, 1460.0, 3200.0, 108.0, 0.0, 0.0]</td>
      <td>[448627.72]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}
<p>100 rows × 4 columns</p>

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

housing_model_challenger01 = wl.upload_model(model_name_challenger01, model_file_name_challenger01).configure()
housing_model_challenger02 = wl.upload_model(model_name_challenger02, model_file_name_challenger02).configure()

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

{{<table "table table-striped table-bordered">}}
<table><tr><th>name</th> <td>pipelinelogpipeline</td></tr><tr><th>created</th> <td>2023-04-25 17:44:05.525160+00:00</td></tr><tr><th>last_updated</th> <td>2023-04-25 17:45:11.476267+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>122594e9-3ead-4187-866d-4eb9ff42be34, 950589d2-0d7d-4bb4-bc0f-e63e1bf39e59, e68eb2bd-2ab4-4223-9e6b-9b570ba0f324</td></tr><tr><th>steps</th> <td>logcontrol</td></tr></table>
{{</table>}}

### Shadow Deploy Sample Inference

We'll now use our same sample data for an inference to our shadow deployed pipeline, then display the first 20 results with just the comparative outputs.

```python
shadow_result = mainpipeline.infer_from_file('./data/xtest-1k.arrow')

shadow_outputs =  shadow_result.to_pandas()
display(shadow_outputs.loc[0:20,['out.variable','out_logcontrolchallenger01.variable','out_logcontrolchallenger02.variable']])
```

{{<table "table table-striped table-bordered">}}
<table>
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
{{</table>}}

### Shadow Deploy Logs

Pipelines with a shadow deployed step include the shadow inference result in the same format as the inference result:  inference results from shadow deployed models are displayed as `out_{model name}.{output variable}`.

```python
# display logs with shadow deployed steps

display(mainpipeline.logs())
```

    Warning: There are more logs available. Please set a larger limit or request a file using export_logs.

{{<table "table table-striped table-bordered">}}
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
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 2.0, 2005.0, 7000.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1605.0, 400.0, 47.6039, -122.298, 1750.0, 4500.0, 34.0, 0.0, 0.0]</td>
      <td>[581003.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 1.75, 2910.0, 37461.0, 1.0, 0.0, 0.0, 4.0, 7.0, 1530.0, 1380.0, 47.7015, -122.164, 2520.0, 18295.0, 47.0, 0.0, 0.0]</td>
      <td>[706823.56]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[4.0, 3.25, 2910.0, 1880.0, 2.0, 0.0, 3.0, 5.0, 9.0, 1830.0, 1080.0, 47.616, -122.282, 3100.0, 8200.0, 100.0, 0.0, 0.0]</td>
      <td>[1060847.5]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[4.0, 1.75, 2700.0, 7875.0, 1.5, 0.0, 0.0, 4.0, 8.0, 2700.0, 0.0, 47.454, -122.144, 2220.0, 7875.0, 46.0, 0.0, 0.0]</td>
      <td>[441960.38]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 2.5, 2900.0, 23550.0, 1.0, 0.0, 0.0, 3.0, 10.0, 1490.0, 1410.0, 47.5708, -122.153, 2900.0, 19604.0, 27.0, 0.0, 0.0]</td>
      <td>[827411.0]</td>
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
      <th>95</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[2.0, 1.5, 1070.0, 1236.0, 2.0, 0.0, 0.0, 3.0, 8.0, 1000.0, 70.0, 47.5619, -122.382, 1170.0, 1888.0, 10.0, 0.0, 0.0]</td>
      <td>[435628.56]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>96</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 2.5, 2830.0, 6000.0, 1.0, 0.0, 3.0, 3.0, 9.0, 1730.0, 1100.0, 47.5751, -122.378, 2040.0, 5300.0, 60.0, 0.0, 0.0]</td>
      <td>[981676.6]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>97</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[4.0, 1.75, 1720.0, 8750.0, 1.0, 0.0, 0.0, 3.0, 7.0, 860.0, 860.0, 47.726, -122.21, 1790.0, 8750.0, 43.0, 0.0, 0.0]</td>
      <td>[437177.84]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>98</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[4.0, 2.25, 4470.0, 60373.0, 2.0, 0.0, 0.0, 3.0, 11.0, 4470.0, 0.0, 47.7289, -122.127, 3210.0, 40450.0, 26.0, 0.0, 0.0]</td>
      <td>[1208638.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>99</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 1.0, 1150.0, 3000.0, 1.0, 0.0, 0.0, 5.0, 6.0, 1150.0, 0.0, 47.6867, -122.345, 1460.0, 3200.0, 108.0, 0.0, 0.0]</td>
      <td>[448627.72]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}
<p>100 rows × 4 columns</p>

```python
# Save shadow deployed log files as pandas DataFrame

mainpipeline.export_logs(filename="shadowdeploylogs_logs.df.json")

data_df = pd.read_json("shadowdeploylogs_logs.df.json", lines=True)
display(data_df)
```

{{<table "table table-striped table-bordered">}}
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
      <td>2023-04-25T17:44:18.461Z</td>
      <td>[3.0, 2.0, 2005.0, 7000.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1605.0, 400.0, 47.6039009094, -122.297996521, 1750.0, 4500.0, 34.0, 0.0, 0.0]</td>
      <td>[581003.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-04-25T17:44:18.461Z</td>
      <td>[3.0, 1.75, 2910.0, 37461.0, 1.0, 0.0, 0.0, 4.0, 7.0, 1530.0, 1380.0, 47.701499939, -122.1640014648, 2520.0, 18295.0, 47.0, 0.0, 0.0]</td>
      <td>[706823.5625]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-04-25T17:44:18.461Z</td>
      <td>[4.0, 3.25, 2910.0, 1880.0, 2.0, 0.0, 3.0, 5.0, 9.0, 1830.0, 1080.0, 47.6160011292, -122.2819976807, 3100.0, 8200.0, 100.0, 0.0, 0.0]</td>
      <td>[1060847.5]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-04-25T17:44:18.461Z</td>
      <td>[4.0, 1.75, 2700.0, 7875.0, 1.5, 0.0, 0.0, 4.0, 8.0, 2700.0, 0.0, 47.4539985657, -122.1439971924, 2220.0, 7875.0, 46.0, 0.0, 0.0]</td>
      <td>[441960.375]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-04-25T17:44:18.461Z</td>
      <td>[3.0, 2.5, 2900.0, 23550.0, 1.0, 0.0, 0.0, 3.0, 10.0, 1490.0, 1410.0, 47.5708007812, -122.1529998779, 2900.0, 19604.0, 27.0, 0.0, 0.0]</td>
      <td>[827411.0]</td>
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
      <th>95</th>
      <td>2023-04-25T17:44:18.461Z</td>
      <td>[2.0, 1.5, 1070.0, 1236.0, 2.0, 0.0, 0.0, 3.0, 8.0, 1000.0, 70.0, 47.5619010925, -122.3820037842, 1170.0, 1888.0, 10.0, 0.0, 0.0]</td>
      <td>[435628.5625]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>96</th>
      <td>2023-04-25T17:44:18.461Z</td>
      <td>[3.0, 2.5, 2830.0, 6000.0, 1.0, 0.0, 3.0, 3.0, 9.0, 1730.0, 1100.0, 47.5750999451, -122.3779983521, 2040.0, 5300.0, 60.0, 0.0, 0.0]</td>
      <td>[981676.625]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>97</th>
      <td>2023-04-25T17:44:18.461Z</td>
      <td>[4.0, 1.75, 1720.0, 8750.0, 1.0, 0.0, 0.0, 3.0, 7.0, 860.0, 860.0, 47.7260017395, -122.2099990845, 1790.0, 8750.0, 43.0, 0.0, 0.0]</td>
      <td>[437177.84375]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>98</th>
      <td>2023-04-25T17:44:18.461Z</td>
      <td>[4.0, 2.25, 4470.0, 60373.0, 2.0, 0.0, 0.0, 3.0, 11.0, 4470.0, 0.0, 47.7289009094, -122.1269989014, 3210.0, 40450.0, 26.0, 0.0, 0.0]</td>
      <td>[1208638.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>99</th>
      <td>2023-04-25T17:44:18.461Z</td>
      <td>[3.0, 1.0, 1150.0, 3000.0, 1.0, 0.0, 0.0, 5.0, 6.0, 1150.0, 0.0, 47.6866989136, -122.3450012207, 1460.0, 3200.0, 108.0, 0.0, 0.0]</td>
      <td>[448627.71875]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}
<p>100 rows × 4 columns</p>

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
```

{{<table "table table-striped table-bordered">}}
<table>
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
      <td>[{"name":"logcontrol","version":"a35b67e6-cd9d-4a09-b1c3-2efdb6cc5173","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}]</td>
      <td>[473287.25]</td>
    </tr>
  </tbody>
</table>
{{</table>}}

{{<table "table table-striped table-bordered">}}
<table>
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
      <td>[{"name":"logcontrolchallenger02","version":"fa87ef25-e790-4781-9570-2cd69ad75bab","sha":"ed6065a79d841f7e96307bb20d5ef22840f15da0b587efb51425c7ad60589d6a"}]</td>
      <td>[470371.28]</td>
    </tr>
  </tbody>
</table>
{{</table>}}

{{<table "table table-striped table-bordered">}}
<table>
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
      <td>[{"name":"logcontrol","version":"a35b67e6-cd9d-4a09-b1c3-2efdb6cc5173","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}]</td>
      <td>[473591.84]</td>
    </tr>
  </tbody>
</table>
{{</table>}}

{{<table "table table-striped table-bordered">}}
<table>
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
      <td>[{"name":"logcontrolchallenger02","version":"fa87ef25-e790-4781-9570-2cd69ad75bab","sha":"ed6065a79d841f7e96307bb20d5ef22840f15da0b587efb51425c7ad60589d6a"}]</td>
      <td>[268304.4]</td>
    </tr>
  </tbody>
</table>
{{</table>}}

{{<table "table table-striped table-bordered">}}
<table>
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
      <td>[{"name":"logcontrolchallenger01","version":"2372cd64-0079-4de9-a4bd-95e885417a50","sha":"31e92d6ccb27b041a324a7ac22cf95d9d6cc3aa7e8263a229f7c4aec4938657c"}]</td>
      <td>[354304.63]</td>
    </tr>
  </tbody>
</table>
{{</table>}}

{{<table "table table-striped table-bordered">}}
<table>
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
      <td>[{"name":"logcontrol","version":"a35b67e6-cd9d-4a09-b1c3-2efdb6cc5173","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}]</td>
      <td>[439977.53]</td>
    </tr>
  </tbody>
</table>
{{</table>}}

{{<table "table table-striped table-bordered">}}
<table>
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
      <td>[{"name":"logcontrol","version":"a35b67e6-cd9d-4a09-b1c3-2efdb6cc5173","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}]</td>
      <td>[583765.94]</td>
    </tr>
  </tbody>
</table>
{{</table>}}

{{<table "table table-striped table-bordered">}}
<table>
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
      <td>[{"name":"logcontrolchallenger01","version":"2372cd64-0079-4de9-a4bd-95e885417a50","sha":"31e92d6ccb27b041a324a7ac22cf95d9d6cc3aa7e8263a229f7c4aec4938657c"}]</td>
      <td>[568536.1]</td>
    </tr>
  </tbody>
</table>
{{</table>}}

{{<table "table table-striped table-bordered">}}
<table>
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
      <td>[{"name":"logcontrolchallenger02","version":"fa87ef25-e790-4781-9570-2cd69ad75bab","sha":"ed6065a79d841f7e96307bb20d5ef22840f15da0b587efb51425c7ad60589d6a"}]</td>
      <td>[615955.3]</td>
    </tr>
  </tbody>
</table>
{{</table>}}

{{<table "table table-striped table-bordered">}}
<table>
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
      <td>[{"name":"logcontrolchallenger01","version":"2372cd64-0079-4de9-a4bd-95e885417a50","sha":"31e92d6ccb27b041a324a7ac22cf95d9d6cc3aa7e8263a229f7c4aec4938657c"}]</td>
      <td>[410591.44]</td>
    </tr>
  </tbody>
</table>
{{</table>}}

{{<table "table table-striped table-bordered">}}
<table>
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
      <td>[{"name":"logcontrol","version":"a35b67e6-cd9d-4a09-b1c3-2efdb6cc5173","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}]</td>
      <td>[544392.06]</td>
    </tr>
  </tbody>
</table>
{{</table>}}

{{<table "table table-striped table-bordered">}}
<table>
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
      <td>[{"name":"logcontrol","version":"a35b67e6-cd9d-4a09-b1c3-2efdb6cc5173","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}]</td>
      <td>[987157.25]</td>
    </tr>
  </tbody>
</table>
{{</table>}}

{{<table "table table-striped table-bordered">}}
<table>
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
      <td>[{"name":"logcontrolchallenger02","version":"fa87ef25-e790-4781-9570-2cd69ad75bab","sha":"ed6065a79d841f7e96307bb20d5ef22840f15da0b587efb51425c7ad60589d6a"}]</td>
      <td>[399058.25]</td>
    </tr>
  </tbody>
</table>
{{</table>}}

{{<table "table table-striped table-bordered">}}
<table>
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
      <td>[{"name":"logcontrolchallenger01","version":"2372cd64-0079-4de9-a4bd-95e885417a50","sha":"31e92d6ccb27b041a324a7ac22cf95d9d6cc3aa7e8263a229f7c4aec4938657c"}]</td>
      <td>[386643.63]</td>
    </tr>
  </tbody>
</table>
{{</table>}}

{{<table "table table-striped table-bordered">}}
<table>
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
      <td>[{"name":"logcontrolchallenger02","version":"fa87ef25-e790-4781-9570-2cd69ad75bab","sha":"ed6065a79d841f7e96307bb20d5ef22840f15da0b587efb51425c7ad60589d6a"}]</td>
      <td>[249720.84]</td>
    </tr>
  </tbody>
</table>
{{</table>}}

{{<table "table table-striped table-bordered">}}
<table>
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
      <td>[{"name":"logcontrol","version":"a35b67e6-cd9d-4a09-b1c3-2efdb6cc5173","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}]</td>
      <td>[306159.38]</td>
    </tr>
  </tbody>
</table>
{{</table>}}

{{<table "table table-striped table-bordered">}}
<table>
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
      <td>[{"name":"logcontrolchallenger02","version":"fa87ef25-e790-4781-9570-2cd69ad75bab","sha":"ed6065a79d841f7e96307bb20d5ef22840f15da0b587efb51425c7ad60589d6a"}]</td>
      <td>[420780.5]</td>
    </tr>
  </tbody>
</table>
{{</table>}}

{{<table "table table-striped table-bordered">}}
<table>
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
      <td>[{"name":"logcontrolchallenger02","version":"fa87ef25-e790-4781-9570-2cd69ad75bab","sha":"ed6065a79d841f7e96307bb20d5ef22840f15da0b587efb51425c7ad60589d6a"}]</td>
      <td>[668267.25]</td>
    </tr>
  </tbody>
</table>
{{</table>}}

{{<table "table table-striped table-bordered">}}
<table>
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
      <td>[{"name":"logcontrolchallenger02","version":"fa87ef25-e790-4781-9570-2cd69ad75bab","sha":"ed6065a79d841f7e96307bb20d5ef22840f15da0b587efb51425c7ad60589d6a"}]</td>
      <td>[448610.4]</td>
    </tr>
  </tbody>
</table>
{{</table>}}

{{<table "table table-striped table-bordered">}}
<table>
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
      <td>[{"name":"logcontrolchallenger02","version":"fa87ef25-e790-4781-9570-2cd69ad75bab","sha":"ed6065a79d841f7e96307bb20d5ef22840f15da0b587efb51425c7ad60589d6a"}]</td>
      <td>[302680.94]</td>
    </tr>
  </tbody>
</table>
{{</table>}}

```python
## Get the logs with the a/b testing information

display(mainpipeline.logs(limit=20))
```

    Warning: There are more logs available. Please set a larger limit or request a file using export_logs.

{{<table "table table-striped table-bordered">}}
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
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 2.0, 2005.0, 7000.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1605.0, 400.0, 47.6039, -122.298, 1750.0, 4500.0, 34.0, 0.0, 0.0]</td>
      <td>[581003.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 1.75, 2910.0, 37461.0, 1.0, 0.0, 0.0, 4.0, 7.0, 1530.0, 1380.0, 47.7015, -122.164, 2520.0, 18295.0, 47.0, 0.0, 0.0]</td>
      <td>[706823.56]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[4.0, 3.25, 2910.0, 1880.0, 2.0, 0.0, 3.0, 5.0, 9.0, 1830.0, 1080.0, 47.616, -122.282, 3100.0, 8200.0, 100.0, 0.0, 0.0]</td>
      <td>[1060847.5]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[4.0, 1.75, 2700.0, 7875.0, 1.5, 0.0, 0.0, 4.0, 8.0, 2700.0, 0.0, 47.454, -122.144, 2220.0, 7875.0, 46.0, 0.0, 0.0]</td>
      <td>[441960.38]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 2.5, 2900.0, 23550.0, 1.0, 0.0, 0.0, 3.0, 10.0, 1490.0, 1410.0, 47.5708, -122.153, 2900.0, 19604.0, 27.0, 0.0, 0.0]</td>
      <td>[827411.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 1.75, 1300.0, 10030.0, 1.0, 0.0, 0.0, 4.0, 7.0, 1300.0, 0.0, 47.7359, -122.192, 1520.0, 7713.0, 48.0, 0.0, 0.0]</td>
      <td>[340764.53]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 2.5, 2140.0, 8925.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2140.0, 0.0, 47.6314, -122.027, 2310.0, 8956.0, 23.0, 0.0, 0.0]</td>
      <td>[669645.5]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[4.0, 2.5, 1690.0, 7260.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1080.0, 610.0, 47.3001, -122.368, 1690.0, 7700.0, 36.0, 0.0, 0.0]</td>
      <td>[284336.5]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[2.0, 1.0, 870.0, 8487.0, 1.0, 0.0, 0.0, 4.0, 6.0, 870.0, 0.0, 47.4955, -122.239, 1350.0, 6850.0, 71.0, 0.0, 0.0]</td>
      <td>[236238.66]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 1.75, 1600.0, 9579.0, 1.0, 0.0, 0.0, 3.0, 8.0, 1180.0, 420.0, 47.7662, -122.159, 1750.0, 9829.0, 38.0, 0.0, 0.0]</td>
      <td>[437177.84]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[4.0, 2.75, 2500.0, 4950.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2500.0, 0.0, 47.6964, -122.017, 2500.0, 4950.0, 4.0, 0.0, 0.0]</td>
      <td>[700271.56]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 2.5, 2010.0, 2287.0, 2.0, 0.0, 0.0, 3.0, 8.0, 1390.0, 620.0, 47.5517, -121.998, 1690.0, 1662.0, 0.0, 0.0, 0.0]</td>
      <td>[578362.9]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[5.0, 3.0, 2230.0, 6551.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1330.0, 900.0, 47.487, -122.32, 2230.0, 9476.0, 1.0, 0.0, 0.0]</td>
      <td>[399760.34]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[2.0, 1.0, 1380.0, 4390.0, 1.0, 0.0, 0.0, 4.0, 8.0, 880.0, 500.0, 47.6947, -122.323, 1390.0, 5234.0, 83.0, 0.0, 0.0]</td>
      <td>[441284.06]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[4.0, 2.5, 2230.0, 5650.0, 2.0, 0.0, 0.0, 3.0, 7.0, 2230.0, 0.0, 47.5073, -122.168, 1590.0, 7241.0, 3.0, 0.0, 0.0]</td>
      <td>[401263.9]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[2.0, 1.0, 1430.0, 3000.0, 1.5, 0.0, 0.0, 3.0, 7.0, 1300.0, 130.0, 47.6415, -122.303, 1750.0, 4000.0, 85.0, 0.0, 0.0]</td>
      <td>[455435.6]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 1.75, 1890.0, 93218.0, 1.0, 0.0, 0.0, 4.0, 7.0, 1890.0, 0.0, 47.2568, -122.07, 1690.0, 172062.0, 50.0, 0.0, 0.0]</td>
      <td>[291799.88]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[4.0, 2.5, 1980.0, 5909.0, 2.0, 0.0, 0.0, 3.0, 8.0, 1980.0, 0.0, 47.3913, -122.185, 2550.0, 5487.0, 11.0, 0.0, 0.0]</td>
      <td>[324875.06]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[2.0, 1.0, 1170.0, 7142.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1170.0, 0.0, 47.7497, -122.313, 1170.0, 7615.0, 63.0, 0.0, 0.0]</td>
      <td>[340764.53]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2023-04-25 17:44:18.461</td>
      <td>[3.0, 2.75, 2890.0, 12130.0, 2.0, 0.0, 3.0, 4.0, 10.0, 2830.0, 60.0, 47.6505, -122.203, 2415.0, 11538.0, 27.0, 0.0, 0.0]</td>
      <td>[987149.56]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}

```python
# Save a/b testing log files as DataFrame

mainpipeline.export_logs(filename="abtestinglogs_logs.df.json")

data_df = pd.read_json("abtestinglogs_logs.df.json", lines=True)
display(data_df)
```

{{<table "table table-striped table-bordered">}}
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
      <td>2023-04-25T17:44:18.461Z</td>
      <td>[3.0, 2.0, 2005.0, 7000.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1605.0, 400.0, 47.6039009094, -122.297996521, 1750.0, 4500.0, 34.0, 0.0, 0.0]</td>
      <td>[581003.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-04-25T17:44:18.461Z</td>
      <td>[3.0, 1.75, 2910.0, 37461.0, 1.0, 0.0, 0.0, 4.0, 7.0, 1530.0, 1380.0, 47.701499939, -122.1640014648, 2520.0, 18295.0, 47.0, 0.0, 0.0]</td>
      <td>[706823.5625]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-04-25T17:44:18.461Z</td>
      <td>[4.0, 3.25, 2910.0, 1880.0, 2.0, 0.0, 3.0, 5.0, 9.0, 1830.0, 1080.0, 47.6160011292, -122.2819976807, 3100.0, 8200.0, 100.0, 0.0, 0.0]</td>
      <td>[1060847.5]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-04-25T17:44:18.461Z</td>
      <td>[4.0, 1.75, 2700.0, 7875.0, 1.5, 0.0, 0.0, 4.0, 8.0, 2700.0, 0.0, 47.4539985657, -122.1439971924, 2220.0, 7875.0, 46.0, 0.0, 0.0]</td>
      <td>[441960.375]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-04-25T17:44:18.461Z</td>
      <td>[3.0, 2.5, 2900.0, 23550.0, 1.0, 0.0, 0.0, 3.0, 10.0, 1490.0, 1410.0, 47.5708007812, -122.1529998779, 2900.0, 19604.0, 27.0, 0.0, 0.0]</td>
      <td>[827411.0]</td>
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
      <th>95</th>
      <td>2023-04-25T17:44:18.461Z</td>
      <td>[2.0, 1.5, 1070.0, 1236.0, 2.0, 0.0, 0.0, 3.0, 8.0, 1000.0, 70.0, 47.5619010925, -122.3820037842, 1170.0, 1888.0, 10.0, 0.0, 0.0]</td>
      <td>[435628.5625]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>96</th>
      <td>2023-04-25T17:44:18.461Z</td>
      <td>[3.0, 2.5, 2830.0, 6000.0, 1.0, 0.0, 3.0, 3.0, 9.0, 1730.0, 1100.0, 47.5750999451, -122.3779983521, 2040.0, 5300.0, 60.0, 0.0, 0.0]</td>
      <td>[981676.625]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>97</th>
      <td>2023-04-25T17:44:18.461Z</td>
      <td>[4.0, 1.75, 1720.0, 8750.0, 1.0, 0.0, 0.0, 3.0, 7.0, 860.0, 860.0, 47.7260017395, -122.2099990845, 1790.0, 8750.0, 43.0, 0.0, 0.0]</td>
      <td>[437177.84375]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>98</th>
      <td>2023-04-25T17:44:18.461Z</td>
      <td>[4.0, 2.25, 4470.0, 60373.0, 2.0, 0.0, 0.0, 3.0, 11.0, 4470.0, 0.0, 47.7289009094, -122.1269989014, 3210.0, 40450.0, 26.0, 0.0, 0.0]</td>
      <td>[1208638.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>99</th>
      <td>2023-04-25T17:44:18.461Z</td>
      <td>[3.0, 1.0, 1150.0, 3000.0, 1.0, 0.0, 0.0, 5.0, 6.0, 1150.0, 0.0, 47.6866989136, -122.3450012207, 1460.0, 3200.0, 108.0, 0.0, 0.0]</td>
      <td>[448627.71875]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}
<p>100 rows × 4 columns</p>

### Undeploy Main Pipeline

With the examples and tutorial complete, we will undeploy the main pipeline and return the resources back to the Wallaroo instance.

```python
mainpipeline.undeploy()
```

{{<table "table table-striped table-bordered">}}
<table><tr><th>name</th> <td>pipelinelogpipeline</td></tr><tr><th>created</th> <td>2023-04-25 17:44:05.525160+00:00</td></tr><tr><th>last_updated</th> <td>2023-04-25 17:46:07.123288+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>59366a68-d79b-420e-bcd7-f17222bced97, 122594e9-3ead-4187-866d-4eb9ff42be34, 950589d2-0d7d-4bb4-bc0f-e63e1bf39e59, e68eb2bd-2ab4-4223-9e6b-9b570ba0f324</td></tr><tr><th>steps</th> <td>logcontrol</td></tr></table>
{{</table>}}

