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

import datetime

import os
```

### Connect to the Wallaroo Instance

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
# Login through local Wallaroo instance

wl = wallaroo.Client()
```

### Create Workspace

We will create a workspace to manage our pipeline and models.  The following variables will set the name of our sample workspace then set it as the current workspace.

```python
workspace_name = 'logworkspace'
main_pipeline_name = 'logpipeline-test'
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

    {'name': 'logworkspace', 'id': 27, 'archived': False, 'created_by': 'c97d480f-6064-4537-b18e-40fb1864b4cd', 'created_at': '2024-02-09T16:21:07.131681+00:00', 'models': [], 'pipelines': []}

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
# in case this pipeline was run before
mainpipeline.clear()
mainpipeline.add_model_step(housing_model_control)

deploy_config = wallaroo.deployment_config.DeploymentConfigBuilder() \
    .cpus(0.25)\
    .build()

mainpipeline.deploy(deployment_config=deploy_config)
```

    Waiting for deployment - this will take up to 45s .......... ok

<table><tr><th>name</th> <td>logpipeline-test</td></tr><tr><th>created</th> <td>2024-02-09 16:21:09.406182+00:00</td></tr><tr><th>last_updated</th> <td>2024-02-09 16:30:53.067304+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>e2b9d903-4015-4d09-902b-9150a7196cea, 9df38be1-d2f4-4be1-9022-8f0570a238b9, 3078b49f-3eff-48d1-8d9b-a8780b329ecc, 21bff9df-828f-40e7-8a22-449a2e636b44, f78a7030-bd25-4bf7-ba0d-a18cfe3790e0, 10c1ac25-d626-4413-8d5d-1bed42d0e65c, b179b693-b6b6-4ff9-b2a4-2a639d88bc9b, da7b9cf0-81e8-452b-8b70-689406dc9548, a9a9b62c-9d37-427f-99af-67725558bf9b, 1c14591a-96b4-4059-bb63-2d2bc4e308d5, add660ac-0ebf-4a24-bb6d-6cdc875866c8</td></tr><tr><th>steps</th> <td>logcontrol</td></tr><tr><th>published</th> <td>False</td></tr></table>

### Testing

We'll use two inferences as a quick sample test - one that has a house that should be determined around \\$700k, the other with a house determined to be around \\$1.5 million.  We'll also save the start and end periods for these events to for later log functionality.

```python
dataframe_start = datetime.datetime.now()

normal_input = pd.DataFrame.from_records({"tensor": [
            [
                4.0, 
                2.5, 
                2900.0, 
                5505.0, 
                2.0, 
                0.0, 
                0.0, 
                3.0, 
                8.0, 
                2900.0, 
                0.0, 
                47.6063, 
                -122.02, 
                2970.0, 
                5251.0, 
                12.0, 
                0.0, 
                0.0
            ]
        ]
    }
)
result = mainpipeline.infer(normal_input)
display(result)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.variable</th>
      <th>anomaly.count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-02-09 16:31:04.817</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.7]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

```python
large_house_input = pd.DataFrame.from_records(
    {
        'tensor': [
            [
                4.0, 
                3.0, 
                3710.0, 
                20000.0, 
                2.0, 
                0.0, 
                2.0, 
                5.0, 
                10.0, 
                2760.0, 
                950.0, 
                47.6696, 
                -122.261, 
                3970.0, 
                20000.0, 
                79.0, 
                0.0, 
                0.0
            ]
        ]
    }
)
large_house_result = mainpipeline.infer(large_house_input)
display(large_house_result)

import time
time.sleep(10)
dataframe_end = datetime.datetime.now()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.variable</th>
      <th>anomaly.count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-02-09 16:31:04.917</td>
      <td>[4.0, 3.0, 3710.0, 20000.0, 2.0, 0.0, 2.0, 5.0, 10.0, 2760.0, 950.0, 47.6696, -122.261, 3970.0, 20000.0, 79.0, 0.0, 0.0]</td>
      <td>[1514079.4]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

As one last sample, we'll run through roughly 1,000 inferences at once and show a few of the results.  For this example we'll use an Apache Arrow table, which has a smaller file size compared to uploading a pandas DataFrame JSON file.  The inference result is returned as an arrow table, which we'll convert into a pandas DataFrame to display the first 20 results.

```python
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
      <th>anomaly.count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-02-09 16:31:15.018</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.75]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-02-09 16:31:15.018</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[615094.56]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-02-09 16:31:15.018</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
      <td>[448627.72]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-02-09 16:31:15.018</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[758714.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-02-09 16:31:15.018</td>
      <td>[3.0, 1.75, 2200.0, 11520.0, 1.0, 0.0, 0.0, 4.0, 7.0, 2200.0, 0.0, 47.7659, -122.341, 1690.0, 8038.0, 62.0, 0.0, 0.0]</td>
      <td>[513264.7]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2024-02-09 16:31:15.018</td>
      <td>[3.0, 2.0, 2140.0, 4923.0, 1.0, 0.0, 0.0, 4.0, 8.0, 1070.0, 1070.0, 47.6902, -122.339, 1470.0, 4923.0, 86.0, 0.0, 0.0]</td>
      <td>[668288.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2024-02-09 16:31:15.018</td>
      <td>[4.0, 3.5, 3590.0, 5334.0, 2.0, 0.0, 2.0, 3.0, 9.0, 3140.0, 450.0, 47.6763, -122.267, 2100.0, 6250.0, 9.0, 0.0, 0.0]</td>
      <td>[1004846.5]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2024-02-09 16:31:15.018</td>
      <td>[3.0, 2.0, 1280.0, 960.0, 2.0, 0.0, 0.0, 3.0, 9.0, 1040.0, 240.0, 47.602, -122.311, 1280.0, 1173.0, 0.0, 0.0, 0.0]</td>
      <td>[684577.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2024-02-09 16:31:15.018</td>
      <td>[4.0, 2.5, 2820.0, 15000.0, 2.0, 0.0, 0.0, 4.0, 9.0, 2820.0, 0.0, 47.7255, -122.101, 2440.0, 15000.0, 29.0, 0.0, 0.0]</td>
      <td>[727898.1]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2024-02-09 16:31:15.018</td>
      <td>[3.0, 2.25, 1790.0, 11393.0, 1.0, 0.0, 0.0, 3.0, 8.0, 1790.0, 0.0, 47.6297, -122.099, 2290.0, 11894.0, 36.0, 0.0, 0.0]</td>
      <td>[559631.1]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2024-02-09 16:31:15.018</td>
      <td>[3.0, 1.5, 1010.0, 7683.0, 1.5, 0.0, 0.0, 5.0, 7.0, 1010.0, 0.0, 47.72, -122.318, 1550.0, 7271.0, 61.0, 0.0, 0.0]</td>
      <td>[340764.53]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2024-02-09 16:31:15.018</td>
      <td>[3.0, 2.0, 1270.0, 1323.0, 3.0, 0.0, 0.0, 3.0, 8.0, 1270.0, 0.0, 47.6934, -122.342, 1330.0, 1323.0, 8.0, 0.0, 0.0]</td>
      <td>[442168.06]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2024-02-09 16:31:15.018</td>
      <td>[4.0, 1.75, 2070.0, 9120.0, 1.0, 0.0, 0.0, 4.0, 7.0, 1250.0, 820.0, 47.6045, -122.123, 1650.0, 8400.0, 57.0, 0.0, 0.0]</td>
      <td>[630865.6]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2024-02-09 16:31:15.018</td>
      <td>[4.0, 1.0, 1620.0, 4080.0, 1.5, 0.0, 0.0, 3.0, 7.0, 1620.0, 0.0, 47.6696, -122.324, 1760.0, 4080.0, 91.0, 0.0, 0.0]</td>
      <td>[559631.1]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2024-02-09 16:31:15.018</td>
      <td>[4.0, 3.25, 3990.0, 9786.0, 2.0, 0.0, 0.0, 3.0, 9.0, 3990.0, 0.0, 47.6784, -122.026, 3920.0, 8200.0, 10.0, 0.0, 0.0]</td>
      <td>[909441.1]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2024-02-09 16:31:15.018</td>
      <td>[4.0, 2.0, 1780.0, 19843.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1780.0, 0.0, 47.4414, -122.154, 2210.0, 13500.0, 52.0, 0.0, 0.0]</td>
      <td>[313096.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2024-02-09 16:31:15.018</td>
      <td>[4.0, 2.5, 2130.0, 6003.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2130.0, 0.0, 47.4518, -122.12, 1940.0, 4529.0, 11.0, 0.0, 0.0]</td>
      <td>[404040.8]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2024-02-09 16:31:15.018</td>
      <td>[3.0, 1.75, 1660.0, 10440.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1040.0, 620.0, 47.4448, -121.77, 1240.0, 10380.0, 36.0, 0.0, 0.0]</td>
      <td>[292859.5]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2024-02-09 16:31:15.018</td>
      <td>[3.0, 2.5, 2110.0, 4118.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2110.0, 0.0, 47.3878, -122.153, 2110.0, 4044.0, 25.0, 0.0, 0.0]</td>
      <td>[338357.88]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2024-02-09 16:31:15.018</td>
      <td>[4.0, 2.25, 2200.0, 11250.0, 1.5, 0.0, 0.0, 5.0, 7.0, 1300.0, 900.0, 47.6845, -122.201, 2320.0, 10814.0, 94.0, 0.0, 0.0]</td>
      <td>[682284.6]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

### Standard Pipeline Logs

Pipeline logs with standard pipeline steps are retrieved either with:

* Pipeline `logs` which returns either a pandas DataFrame or Apache Arrow table.
* Pipeline `export_logs` which saves the logs either a pandas DataFrame JSON file or Apache Arrow table.

For full details, see the Wallaroo Documentation Pipeline Log Management guide.

#### Pipeline Log Method

The Pipeline `logs` method includes the following parameters.  For a complete list, see the [Wallaroo SDK Essentials Guide: Pipeline Log Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipeline-logs/).

| Parameter | Type | Description |
|---|---|---|
| `limit` | **Int** (*Optional*) | Limits how many log records to display.  Defaults to `100`.  If there are more pipeline logs than are being displayed, the **Warning** message `Pipeline log record limit exceeded` will be displayed.  For example, if 100 log files were requested and there are a total of 1,000, the warning message will be displayed. |
| `start_datetime` and `end_datetime` | **DateTime** (*Optional*) | Limits logs to all logs between the `start` and `end` DateTime parameters.  **Both parameters must be provided**. Submitting a `logs()` request with only `start_datetime` or `end_datetime` will generate an exception.<br />If `start_datetime` and `end_datetime` are provided as parameters, then the records are returned in **chronological** order, with the oldest record displayed first. |
| `dataset` | List (*OPTIONAL*) | The datasets to be returned. The datasets available are:<ul><li>`*`: Default. This translates to `["time", "in", "out", "anomaly"]`.</li><li>`time`: The DateTime of the inference request.</li><li>`in`: All inputs listed as `in_{variable_name}`.</li><li>`out`: All outputs listed as `out_variable_name`.</li><li>`anomaly`: Flags whether an [anomaly was detected](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline-validation/) was triggered. `0` indicates no checks were triggered, 1 or greater indicates a an anomaly was detected. was triggered.  Each validation is displayed in the returned logs as part of the `anomaly` dataset as `anomaly.{validation_name}`.  For more information on anomaly detection, see [Wallaroo SDK Essentials Guide: Anomaly Detection](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline-validation/)</li><li>`meta`: Returns metadata. **IMPORTANT NOTE**: See [Metadata RequestsRestrictions](#metadata-requests-restrictions) for specifications on how this dataset can be used with other datasets.<ul><li> Returns in the `metadata.elapsed` field:<ul><li>A list of time in nanoseconds for:<ul><li>The time to serialize the input.</li><li>How long each step took.</li></ul></li></ul></li><li>Returns in the `metadata.last_model` field:</li><ul><li>A dict with each Python step as:<ul><li>`model_name`: The name of the model in the pipeline step.</li><li>`model_sha` : The sha hash of the model in the pipeline step.</li></ul></li></ul></li><li>Returns in the `metadata.pipeline_version` field:<ul><li>The pipeline version as a UUID value.</li></ul></li></ul><li>`metadata.elapsed`: **IMPORTANT NOTE**: See [Metadata Requests Restrictions](#metadata-requests-restrictions)for specifications on how this dataset can be used with other datasets.<ul><li>Returns in the `metadata.elapsed` field:<ul><li>A list of time in nanoseconds for:<ul><li>The time to serialize the input.</li><li>How long each step took.</li></ul></li></ul></li></ul></ul> |
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

display("Standard Logs")
display(len(regular_logs))
display(regular_logs)

# Display metadata

metadatalogs = mainpipeline.logs(dataset=["time", "out.variable", "metadata"])
display("Metadata Logs")
# Only showing the pipeline version for space reasons
display(metadatalogs.loc[:, ["time", "out.variable", "metadata.pipeline_version"]])

# Display logs restricted by date and limit 

display("Logs restricted by date")
arrow_logs = mainpipeline.logs(start_datetime=dataframe_start, end_datetime=dataframe_end, limit=50)

display(len(arrow_logs))
display(arrow_logs)

# # pipeline log retrieval limited to arrow tables
display(mainpipeline.logs(arrow=True))
```

    Pipeline log schema has changed over the logs requested 1 newest records retrieved successfully, newest record seen was at <datetime>. Please request additional records separately

    'Standard Logs'

    1

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.variable</th>
      <th>anomaly.count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-02-09 16:28:44.753</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.7]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

    Pipeline log schema has changed over the logs requested 1 newest records retrieved successfully, newest record seen was at <datetime>. Please request additional records separately

    'Metadata Logs'

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out.variable</th>
      <th>metadata.pipeline_version</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-02-09 16:28:44.753</td>
      <td>[718013.7]</td>
      <td>21bff9df-828f-40e7-8a22-449a2e636b44</td>
    </tr>
  </tbody>
</table>

    'Logs restricted by date'

    2

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.variable</th>
      <th>anomaly.count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-02-09 16:31:04.817</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.7]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-02-09 16:31:04.917</td>
      <td>[4.0, 3.0, 3710.0, 20000.0, 2.0, 0.0, 2.0, 5.0, 10.0, 2760.0, 950.0, 47.6696, -122.261, 3970.0, 20000.0, 79.0, 0.0, 0.0]</td>
      <td>[1514079.4]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

    Pipeline log schema has changed over the logs requested 1 newest records retrieved successfully, newest record seen was at <datetime>. Please request additional records separately

    pyarrow.Table
    time: timestamp[ms]
    in.tensor: list<item: double> not null
      child 0, item: double
    out.variable: list<inner: float not null> not null
      child 0, inner: float not null
    anomaly.count: uint32 not null
    ----
    time: [[2024-02-09 16:28:44.753]]
    in.tensor: [[[4,2.5,2900,5505,2,...,2970,5251,12,0,0]]]
    out.variable: [[[718013.7]]]
    anomaly.count: [[0]]

```python
result = mainpipeline.infer(normal_input, dataset=["*", "metadata.pipeline_version"])
display(result)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.variable</th>
      <th>anomaly.count</th>
      <th>metadata.pipeline_version</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-02-09 16:31:30.617</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.7]</td>
      <td>0</td>
      <td></td>
    </tr>
  </tbody>
</table>

The following displays the pipeline metadata logs.

#### Standard Pipeline Steps Log Requests

Effected pipeline steps:

* `add_model_step`
* `replace_with_model_step`

For log file requests, the following metadata dataset requests for standard pipeline steps are available:

* `metadata`

These must be paired with specific columns.  `*` is **not** available when paired with `metadata`.

* `in`: All input fields.
* `out`: All output fields.
* `time`: The DateTime the inference request was made. 
* `in.{input_fields}`: Any input fields (`tensor`, etc.)
* `out.{output_fields}`: Any output fields (`out.house_price`, `out.variable`, etc.)
* `anomaly.count`:  Any anomalies detected from validations.
* `anomaly.{validation}`: The validation that triggered the anomaly detection and whether it is `True` (indicating an anomaly was detected) or `False`.

The following requests the metadata, and displays the output variable and last model from the metadata.

```python
# Display metadata

metadatalogs = mainpipeline.logs(dataset=['time', "out","metadata"])
display("Metadata Logs")
display(metadatalogs.loc[:, ['time', 'out.variable', 'metadata.last_model']])

```

    Pipeline log schema has changed over the logs requested 2 newest records retrieved successfully, newest record seen was at <datetime>. Please request additional records separately

    'Metadata Logs'

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out.variable</th>
      <th>metadata.last_model</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-02-09 16:28:44.753</td>
      <td>[718013.7]</td>
      <td>{"model_name":"logcontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-02-09 16:31:30.617</td>
      <td>[718013.7]</td>
      <td>{"model_name":"logcontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
    </tr>
  </tbody>
</table>

#### Pipeline Limits

In a previous step we performed 10,000 inferences at once.  If we attempt to pull them at once, we'll likely run into the size limit for this pipeline and receive the following warning message indicating that the pipeline size limits were exceeded and we should use `export_logs` instead.

`Warning: Pipeline log size limit exceeded. Only displaying 1000 log messages (of 10000 requested). Please request a file using export_logs.`

```python
logs = mainpipeline.logs(limit=10000)
display(logs)
```

    Pipeline log schema has changed over the logs requested 2 newest records retrieved successfully, newest record seen was at <datetime>. Please request additional records separately

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.variable</th>
      <th>anomaly.count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-02-09 16:28:44.753</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.7]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-02-09 16:31:30.617</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.7]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

#### Pipeline export_logs Method

The Pipeline method `export_logs` returns the Pipeline records as either a DataFrame JSON file, or an Apache Arrow table file.  For a complete list, see the [Wallaroo SDK Essentials Guide: Pipeline Log Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipeline-logs/).

The `export_logs` method takes the following parameters:

| Parameter | Type | Description |
|---|---|---|
| `directory` | **String** (*Optional*) (*Default*: `logs`) | Logs are exported to a file from current working directory to `directory`.|
| `data_size_limit` | **String** (*Optional*) ((*Default*: `100MB`) | The maximum size for the exported data in bytes.  Note that file size is approximate to the request; a request of `10MiB` may return 10.3MB of data.  The fields are in the format "{size as number} {unit value}", and can include a space so "10 MiB" and "10MiB" are the same.  The accepted unit values are:  <ul><li>`KiB` (for KiloBytes)</li><li>`MiB` (for MegaBytes)</li><li>`GiB` (for GigaBytes)</li><li>`TiB` (for TeraBytes)</li></ul>  |
| `file_prefix` | **String** (*Optional*) (*Default*: The name of the pipeline) | The name of the exported files.  By default, this will the name of the pipeline and is segmented by pipeline version between the limits or the start and end period.  For example:  'logpipeline-1.json`, etc. |
| `limit` | **Int** (*Optional*) | Limits how many log records to display.  Defaults to `100`.  If there are more pipeline logs than are being displayed, the **Warning** message `Pipeline log record limit exceeded` will be displayed.  For example, if 100 log files were requested and there are a total of 1,000, the warning message will be displayed. |
| `start` and `end` | **DateTime** (*Optional*) | Limits logs to all logs between the `start` and `end` DateTime parameters.  **Both parameters must be provided**. Submitting a `logs()` request with only `start` or `end` will generate an exception.<br />If `start` and `end` are provided as parameters, then the records are returned in **chronological** order, with the oldest record displayed first. |
| `dataset` | List (*OPTIONAL*) | The datasets to be returned. The datasets available are:<ul><li>`*`: Default. This translates to `["time", "in", "out", "anomaly"]`.</li><li>`time`: The DateTime of the inference request.</li><li>`in`: All inputs listed as `in_{variable_name}`.</li><li>`out`: All outputs listed as `out_variable_name`.</li><li>`anomaly`: Flags whether an [anomaly was detected](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline-validation/) was triggered. `0` indicates no checks were triggered, 1 or greater indicates a an anomaly was detected. was triggered.  Each validation is displayed in the returned logs as part of the `anomaly` dataset as `anomaly.{validation_name}`.  For more information on anomaly detection, see [Wallaroo SDK Essentials Guide: Anomaly Detection](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline-validation/)</li><li>`meta`: Returns metadata. **IMPORTANT NOTE**: See [Metadata RequestsRestrictions](#metadata-requests-restrictions) for specifications on how this dataset can be used with otherdatasets.<ul><li> Returns in the `metadata.elapsed` field:<ul><li>A list of time in nanoseconds for:<ul><li>The time to serialize the input.</li><li>How long each step took.</li></ul></li></ul></li><li>Returns in the `metadata.last_model` field:</li><ul><li>A dict with each Python step as:<ul><li>`model_name`: The name of the model in the pipeline step.</li><li>`model_sha` : The sha hash of the model in the pipeline step.</li></ul></li></ul></li><li>Returns in the `metadata.pipeline_version` field:<ul><li>The pipeline version as a UUID value.</li></ul></li></ul><li>`metadata.elapsed`: **IMPORTANT NOTE**: See [Metadata Requests Restrictions](#metadata-requests-restrictions)for specifications on how this dataset can be used with other datasets.<ul><li>Returns in the `metadata.elapsed` field:<ul><li>A list of time in nanoseconds for:<ul><li>The time to serialize the input.</li><li>How long each step took.</li></ul></li></ul></li></ul></ul> |
| `arrow` | **Boolean** (*Optional*) | Defaults to **False**.  If `arrow` is set to `True`, then the logs are returned as an [Apache Arrow table](https://arrow.apache.org/).  If `arrow=False`, then the logs are returned as JSON in pandas DataFrame format. |

The following examples demonstrate saving a DataFrame version of the `mainpipeline` logs, then an Arrow version.

```python
# Save the DataFrame version of the log file

mainpipeline.export_logs()
display(os.listdir('./logs'))

mainpipeline.export_logs(arrow=True)
display(os.listdir('./logs'))
```

    Warning: There are more logs available. Please set a larger limit to export more data.
    
    Note: The logs with different schemas are written to separate files in the provided directory.

    ['logpipeline-test-1.arrow',
     'logpipeline-test-2.arrow',
     'logpipeline-test-2.json',
     'logpipeline-1.json',
     'logpipeline-test-1.json',
     'logpipeline-1.arrow']

    Warning: There are more logs available. Please set a larger limit to export more data.
    
    Note: The logs with different schemas are written to separate files in the provided directory.

    ['logpipeline-test-1.arrow',
     'logpipeline-test-2.arrow',
     'logpipeline-test-2.json',
     'logpipeline-1.json',
     'logpipeline-test-1.json',
     'logpipeline-1.arrow']

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
                                              framework=wallaroo.framework.Framework.ONNX)
                                              .configure(tensor_fields=["tensor"])
                            )

```

```python
# Undeploy the pipeline
mainpipeline.undeploy()

mainpipeline.clear()

# Add the new shadow deploy step with our challenger models
mainpipeline.add_shadow_deploy(housing_model_control, [housing_model_challenger01, housing_model_challenger02])

# Deploy the pipeline with the new shadow step
deploy_config = wallaroo.deployment_config.DeploymentConfigBuilder() \
    .cpus(0.25)\
    .build()

mainpipeline.deploy(deployment_config=deploy_config)
```

    Waiting for undeployment - this will take up to 45s ................................... ok
    Waiting for deployment - this will take up to 45s ........ ok

<table><tr><th>name</th> <td>logpipeline-test</td></tr><tr><th>created</th> <td>2024-02-09 16:21:09.406182+00:00</td></tr><tr><th>last_updated</th> <td>2024-02-09 16:33:08.547068+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>e143a2d5-5641-4dcc-8ae4-786fd777a30a, e2b9d903-4015-4d09-902b-9150a7196cea, 9df38be1-d2f4-4be1-9022-8f0570a238b9, 3078b49f-3eff-48d1-8d9b-a8780b329ecc, 21bff9df-828f-40e7-8a22-449a2e636b44, f78a7030-bd25-4bf7-ba0d-a18cfe3790e0, 10c1ac25-d626-4413-8d5d-1bed42d0e65c, b179b693-b6b6-4ff9-b2a4-2a639d88bc9b, da7b9cf0-81e8-452b-8b70-689406dc9548, a9a9b62c-9d37-427f-99af-67725558bf9b, 1c14591a-96b4-4059-bb63-2d2bc4e308d5, add660ac-0ebf-4a24-bb6d-6cdc875866c8</td></tr><tr><th>steps</th> <td>logcontrol</td></tr><tr><th>published</th> <td>False</td></tr></table>

### Shadow Deploy Sample Inference

We'll now use our same sample data for an inference to our shadow deployed pipeline, then display the first 20 results with just the comparative outputs.

```python
shadow_date_start = datetime.datetime.now()

shadow_result = mainpipeline.infer_from_file('./data/xtest-1k.arrow')

shadow_outputs =  shadow_result.to_pandas()
display(shadow_outputs.loc[0:20,['out.variable','out_logcontrolchallenger01.variable','out_logcontrolchallenger02.variable']])

shadow_date_end = datetime.datetime.now()
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
# display logs with shadow deployed steps

display(mainpipeline.logs(start_datetime=shadow_date_start, end_datetime=shadow_date_end).loc[:, ["time", "out.variable", "out_logcontrolchallenger01.variable", "out_logcontrolchallenger02.variable"]])
```

    Warning: Pipeline log size limit exceeded. Please request logs using export_logs

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out.variable</th>
      <th>out_logcontrolchallenger01.variable</th>
      <th>out_logcontrolchallenger02.variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-02-09 16:33:18.093</td>
      <td>[718013.75]</td>
      <td>[659806.0]</td>
      <td>[704901.9]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-02-09 16:33:18.093</td>
      <td>[615094.56]</td>
      <td>[732883.5]</td>
      <td>[695994.44]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-02-09 16:33:18.093</td>
      <td>[448627.72]</td>
      <td>[419508.84]</td>
      <td>[416164.8]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-02-09 16:33:18.093</td>
      <td>[758714.2]</td>
      <td>[634028.8]</td>
      <td>[655277.2]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-02-09 16:33:18.093</td>
      <td>[513264.7]</td>
      <td>[427209.44]</td>
      <td>[426854.66]</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>495</th>
      <td>2024-02-09 16:33:18.093</td>
      <td>[873315.0]</td>
      <td>[779848.6]</td>
      <td>[771244.75]</td>
    </tr>
    <tr>
      <th>496</th>
      <td>2024-02-09 16:33:18.093</td>
      <td>[721143.6]</td>
      <td>[607252.1]</td>
      <td>[610430.56]</td>
    </tr>
    <tr>
      <th>497</th>
      <td>2024-02-09 16:33:18.093</td>
      <td>[1048372.4]</td>
      <td>[844343.56]</td>
      <td>[900959.4]</td>
    </tr>
    <tr>
      <th>498</th>
      <td>2024-02-09 16:33:18.093</td>
      <td>[244566.38]</td>
      <td>[251694.84]</td>
      <td>[246188.81]</td>
    </tr>
    <tr>
      <th>499</th>
      <td>2024-02-09 16:33:18.093</td>
      <td>[518869.0]</td>
      <td>[482136.66]</td>
      <td>[547725.56]</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 4 columns</p>

For log file requests, the following metadata dataset requests for testing pipeline steps are available:

* `metadata`

These must be paired with specific columns.  `*` is **not** available when paired with `metadata`.

* `in`: All input fields.
* `out`: All output fields.
* `time`: The DateTime the inference request was made.
* `in.{input_fields}`: Any input fields (`tensor`, etc.).
* `out.{output_fields}`: Any output fields matching the specific `output_field` (`out.house_price`, `out.variable`, etc.).
* `out_`: All shadow deployed challenger steps Any output fields matching the specific `output_field` (`out.house_price`, `out.variable`, etc.).
* `anomaly.count`:  Any anomalies detected from validations.
* `anomaly.{validation}`: The validation that triggered the anomaly detection and whether it is `True` (indicating an anomaly was detected) or `False`.

The following example retrieves the logs from a pipeline with shadow deployed models, and displays the specific shadow deployed model outputs and the `metadata.elasped` field.

```python
# display logs with shadow deployed steps

display(mainpipeline.logs(start_datetime=shadow_date_start, end_datetime=shadow_date_end).loc[:, ["time", 
                                                                                                  "out.variable", 
                                                                                                  "out_logcontrolchallenger01.variable", 
                                                                                                  "out_logcontrolchallenger02.variable"
                                                                                                  ]
                                                                                        ])
```

    Warning: Pipeline log size limit exceeded. Please request logs using export_logs

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out.variable</th>
      <th>out_logcontrolchallenger01.variable</th>
      <th>out_logcontrolchallenger02.variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-02-09 16:33:18.093</td>
      <td>[718013.75]</td>
      <td>[659806.0]</td>
      <td>[704901.9]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-02-09 16:33:18.093</td>
      <td>[615094.56]</td>
      <td>[732883.5]</td>
      <td>[695994.44]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-02-09 16:33:18.093</td>
      <td>[448627.72]</td>
      <td>[419508.84]</td>
      <td>[416164.8]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-02-09 16:33:18.093</td>
      <td>[758714.2]</td>
      <td>[634028.8]</td>
      <td>[655277.2]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-02-09 16:33:18.093</td>
      <td>[513264.7]</td>
      <td>[427209.44]</td>
      <td>[426854.66]</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>495</th>
      <td>2024-02-09 16:33:18.093</td>
      <td>[873315.0]</td>
      <td>[779848.6]</td>
      <td>[771244.75]</td>
    </tr>
    <tr>
      <th>496</th>
      <td>2024-02-09 16:33:18.093</td>
      <td>[721143.6]</td>
      <td>[607252.1]</td>
      <td>[610430.56]</td>
    </tr>
    <tr>
      <th>497</th>
      <td>2024-02-09 16:33:18.093</td>
      <td>[1048372.4]</td>
      <td>[844343.56]</td>
      <td>[900959.4]</td>
    </tr>
    <tr>
      <th>498</th>
      <td>2024-02-09 16:33:18.093</td>
      <td>[244566.38]</td>
      <td>[251694.84]</td>
      <td>[246188.81]</td>
    </tr>
    <tr>
      <th>499</th>
      <td>2024-02-09 16:33:18.093</td>
      <td>[518869.0]</td>
      <td>[482136.66]</td>
      <td>[547725.56]</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 4 columns</p>

```python
metadatalogs = mainpipeline.logs(dataset=["time",
                                          "out_logcontrolchallenger01.variable", 
                                          "out_logcontrolchallenger02.variable", 
                                          "metadata",
                                          'anomaly.count'
                                          ],
                                start_datetime=shadow_date_start, 
                                end_datetime=shadow_date_end
                                )

display(metadatalogs.loc[:, ['out_logcontrolchallenger01.variable',	
                             'out_logcontrolchallenger02.variable', 
                             'metadata.elapsed',
                             'anomaly.count'
                             ]
                        ])
```

    Warning: Pipeline log size limit exceeded. Please request logs using export_logs

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out_logcontrolchallenger01.variable</th>
      <th>out_logcontrolchallenger02.variable</th>
      <th>metadata.elapsed</th>
      <th>anomaly.count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[659806.0]</td>
      <td>[704901.9]</td>
      <td>[325472, 124071]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[732883.5]</td>
      <td>[695994.44]</td>
      <td>[325472, 124071]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[419508.84]</td>
      <td>[416164.8]</td>
      <td>[325472, 124071]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[634028.8]</td>
      <td>[655277.2]</td>
      <td>[325472, 124071]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[427209.44]</td>
      <td>[426854.66]</td>
      <td>[325472, 124071]</td>
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
      <th>495</th>
      <td>[779848.6]</td>
      <td>[771244.75]</td>
      <td>[325472, 124071]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>496</th>
      <td>[607252.1]</td>
      <td>[610430.56]</td>
      <td>[325472, 124071]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>497</th>
      <td>[844343.56]</td>
      <td>[900959.4]</td>
      <td>[325472, 124071]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>498</th>
      <td>[251694.84]</td>
      <td>[246188.81]</td>
      <td>[325472, 124071]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>499</th>
      <td>[482136.66]</td>
      <td>[547725.56]</td>
      <td>[325472, 124071]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 4 columns</p>

The following demonstrates exporting the shadow deployed logs to the directory `shadow`.

```python
# Save shadow deployed log files as pandas DataFrame

mainpipeline.export_logs(directory="shadow", file_prefix="shadowdeploylogs")
display(os.listdir('./shadow'))
```

    Warning: There are more logs available. Please set a larger limit to export more data.
    
    Note: The logs with different schemas are written to separate files in the provided directory.

    ['shadowdeploylogs-2.json', 'shadowdeploylogs-1.json']

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

deploy_config = wallaroo.deployment_config.DeploymentConfigBuilder() \
    .cpus(0.25)\
    .build()

mainpipeline.deploy(deployment_config=deploy_config)

# Perform sample inferences of 20 rows and display the results
ab_date_start = datetime.datetime.now()
abtesting_inputs = pd.read_json('./data/xtest-1k.df.json')

for index, row in abtesting_inputs.sample(20).iterrows():
    display(mainpipeline.infer(row.to_frame('tensor').reset_index()).loc[:,["out._model_split", "out.variable"]])

ab_date_end = datetime.datetime.now()
```

    Waiting for undeployment - this will take up to 45s ..................................... ok
    Waiting for deployment - this will take up to 45s ......... ok

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
      <td>[{"name":"logcontrolchallenger01","version":"5b63884e-3f09-4e90-9f09-213350b9c445","sha":"31e92d6ccb27b041a324a7ac22cf95d9d6cc3aa7e8263a229f7c4aec4938657c"}]</td>
      <td>[300542.5]</td>
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
      <td>[{"name":"logcontrolchallenger01","version":"5b63884e-3f09-4e90-9f09-213350b9c445","sha":"31e92d6ccb27b041a324a7ac22cf95d9d6cc3aa7e8263a229f7c4aec4938657c"}]</td>
      <td>[580584.3]</td>
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
      <td>[{"name":"logcontrol","version":"1f93edce-3f3e-4d29-be29-6a4e9303da05","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}]</td>
      <td>[447162.84]</td>
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
      <td>[{"name":"logcontrol","version":"1f93edce-3f3e-4d29-be29-6a4e9303da05","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}]</td>
      <td>[581002.94]</td>
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
      <td>[{"name":"logcontrolchallenger01","version":"5b63884e-3f09-4e90-9f09-213350b9c445","sha":"31e92d6ccb27b041a324a7ac22cf95d9d6cc3aa7e8263a229f7c4aec4938657c"}]</td>
      <td>[944906.25]</td>
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
      <td>[{"name":"logcontrolchallenger02","version":"6fc54099-7151-48d7-9e57-6d989fb9bb1c","sha":"ed6065a79d841f7e96307bb20d5ef22840f15da0b587efb51425c7ad60589d6a"}]</td>
      <td>[488997.9]</td>
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
      <td>[{"name":"logcontrol","version":"1f93edce-3f3e-4d29-be29-6a4e9303da05","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}]</td>
      <td>[373955.94]</td>
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
      <td>[{"name":"logcontrolchallenger01","version":"5b63884e-3f09-4e90-9f09-213350b9c445","sha":"31e92d6ccb27b041a324a7ac22cf95d9d6cc3aa7e8263a229f7c4aec4938657c"}]</td>
      <td>[868765.4]</td>
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
      <td>[{"name":"logcontrolchallenger02","version":"6fc54099-7151-48d7-9e57-6d989fb9bb1c","sha":"ed6065a79d841f7e96307bb20d5ef22840f15da0b587efb51425c7ad60589d6a"}]</td>
      <td>[499459.2]</td>
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
      <td>[{"name":"logcontrol","version":"1f93edce-3f3e-4d29-be29-6a4e9303da05","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}]</td>
      <td>[559631.06]</td>
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
      <td>[{"name":"logcontrolchallenger01","version":"5b63884e-3f09-4e90-9f09-213350b9c445","sha":"31e92d6ccb27b041a324a7ac22cf95d9d6cc3aa7e8263a229f7c4aec4938657c"}]</td>
      <td>[344156.25]</td>
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
      <td>[{"name":"logcontrolchallenger02","version":"6fc54099-7151-48d7-9e57-6d989fb9bb1c","sha":"ed6065a79d841f7e96307bb20d5ef22840f15da0b587efb51425c7ad60589d6a"}]</td>
      <td>[296829.75]</td>
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
      <td>[{"name":"logcontrolchallenger02","version":"6fc54099-7151-48d7-9e57-6d989fb9bb1c","sha":"ed6065a79d841f7e96307bb20d5ef22840f15da0b587efb51425c7ad60589d6a"}]</td>
      <td>[532923.94]</td>
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
      <td>[{"name":"logcontrolchallenger02","version":"6fc54099-7151-48d7-9e57-6d989fb9bb1c","sha":"ed6065a79d841f7e96307bb20d5ef22840f15da0b587efb51425c7ad60589d6a"}]</td>
      <td>[878232.2]</td>
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
      <td>[{"name":"logcontrolchallenger02","version":"6fc54099-7151-48d7-9e57-6d989fb9bb1c","sha":"ed6065a79d841f7e96307bb20d5ef22840f15da0b587efb51425c7ad60589d6a"}]</td>
      <td>[996693.6]</td>
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
      <td>[{"name":"logcontrolchallenger02","version":"6fc54099-7151-48d7-9e57-6d989fb9bb1c","sha":"ed6065a79d841f7e96307bb20d5ef22840f15da0b587efb51425c7ad60589d6a"}]</td>
      <td>[544343.3]</td>
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
      <td>[{"name":"logcontrol","version":"1f93edce-3f3e-4d29-be29-6a4e9303da05","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}]</td>
      <td>[379076.28]</td>
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
      <td>[{"name":"logcontrolchallenger01","version":"5b63884e-3f09-4e90-9f09-213350b9c445","sha":"31e92d6ccb27b041a324a7ac22cf95d9d6cc3aa7e8263a229f7c4aec4938657c"}]</td>
      <td>[585684.3]</td>
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
      <td>[{"name":"logcontrolchallenger01","version":"5b63884e-3f09-4e90-9f09-213350b9c445","sha":"31e92d6ccb27b041a324a7ac22cf95d9d6cc3aa7e8263a229f7c4aec4938657c"}]</td>
      <td>[573976.44]</td>
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
      <td>[{"name":"logcontrol","version":"1f93edce-3f3e-4d29-be29-6a4e9303da05","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}]</td>
      <td>[310164.06]</td>
    </tr>
  </tbody>
</table>

```python
## Get the logs with the a/b testing information

metadatalogs = mainpipeline.logs(dataset=["time",
                                          "out", 
                                          "metadata"
                                          ]
                                )

display(metadatalogs.loc[:, ['out.variable', 'metadata.last_model']])
```

    Pipeline log schema has changed over the logs requested 2 newest records retrieved successfully, newest record seen was at <datetime>. Please request additional records separately

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out.variable</th>
      <th>metadata.last_model</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[718013.7]</td>
      <td>{"model_name":"logcontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[718013.7]</td>
      <td>{"model_name":"logcontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
    </tr>
  </tbody>
</table>

```python
# Save a/b testing log files as DataFrame

mainpipeline.export_logs(directory="abtesting", 
                         file_prefix="abtests", 
                         start_datetime=ab_date_start, 
                         end_datetime=ab_date_end)
display(os.listdir('./abtesting'))
```

    ['abtests-1.json']

The following exports the metadata with the log files.

```python
# Save a/b testing log files as DataFrame

mainpipeline.export_logs(directory="abtesting-metadata", 
                         file_prefix="abtests", 
                         start_datetime=ab_date_start, 
                         end_datetime=ab_date_end,
                         dataset=["time", "out", "metadata"])
display(os.listdir('./abtesting-metadata'))
```

    ['abtests-1.json']

## Anomaly Detection Logs

Wallaroo provides **validations** to detect anomalous data from inference inputs and outputs.  Validations are added to a Wallaroo pipeline with the `wallaroo.pipeline.add_validations` method.

Adding validations takes the format:

```python
pipeline.add_validations(
    validation_name_01 = polars.col(in|out.{column_name}) EXPRESSION,
    validation_name_02 = polars.col(in|out.{column_name}) EXPRESSION
    ...{additional rules}
)
```

* `validation_name`: The user provided name of the validation.  The names must match Python variable naming requirements.
  * **IMPORTANT NOTE**: Using the name `count` as a validation name **returns an error**.  Any validation rules named `count` are dropped upon request and an error returned.
* `polars.col(in|out.{column_name})`: Specifies the **input** or **output** for a specific field aka "column" in an inference result.  Wallaroo inference requests are in the format `in.{field_name}` for **inputs**, and `out.{field_name}` for **outputs**.
  * More than one field can be selected, as long as they follow the rules of the [polars 0.18 Expressions library](https://docs.pola.rs/docs/python/version/0.18/reference/expressions/index.html).
* `EXPRESSION`:  The expression to validate. When the expression returns **True**, that indicates an anomaly detected.

The [`polars` library version 0.18.5](https://docs.pola.rs/docs/python/version/0.18/index.html) is used to create the validation rule.  This is installed by default with the Wallaroo SDK.  This provides a powerful range of comparisons to organizations tracking anomalous data from their ML models.

When validations are added to a pipeline, inference request outputs return the following fields:

| Field | Type | Description |
|---|---|---|
| **anomaly.count** | **Integer** | The total of all validations that returned **True**. |
| **anomaly.{validation name}** | **Bool** | The output of the validation `{validation_name}`. |

When validation returns `True`, **an anomaly is detected**.

For example, adding the validation `fraud` to the following pipeline returns `anomaly.count` of `1` when the validation `fraud` returns `True`.  The validation `fraud` returns `True` when the **output** field **dense_1** at index **0** is greater than 0.9.

```python
sample_pipeline = wallaroo.client.build_pipeline("sample-pipeline")
sample_pipeline.add_model_step(model)

# add the validation
sample_pipeline.add_validations(
    fraud=pl.col("out.dense_1").list.get(0) > 0.9,
    )

# deploy the pipeline
sample_pipeline.deploy()

# sample inference
display(sample_pipeline.infer_from_file("dev_high_fraud.json", data_format='pandas-records'))
```

|&nbsp;|time|in.tensor|out.dense_1|anomaly.count|anomaly.fraud|
|---|---|---|---|---|---|
|0|2024-02-02 16:05:42.152|[1.0678324729, 18.1555563975, -1.6589551058, 5...]|[0.981199]|1|True|

### Anomaly Detection Inference Requests Example

For this example, we create the validation rule `too_high` which detects houses with a value greater than 1,000,000 and show the output for houses that trigger that validation.

For these examples we'll create a new pipeline to ensure the logs are "clean" for the samples.

```python
import polars as pl

mainpipeline.undeploy()
mainpipeline.clear()
mainpipeline.add_model_step(housing_model_control)
mainpipeline.add_validations(
    too_high=pl.col("out.variable").list.get(0) > 1000000.0
)

deploy_config = wallaroo.deployment_config.DeploymentConfigBuilder() \
    .cpus(0.25)\
    .build()

mainpipeline.deploy(deployment_config=deploy_config)
```

    Waiting for undeployment - this will take up to 45s ...................................... ok
    Waiting for deployment - this will take up to 45s ......... ok

<table><tr><th>name</th> <td>logpipeline-test</td></tr><tr><th>created</th> <td>2024-02-09 16:21:09.406182+00:00</td></tr><tr><th>last_updated</th> <td>2024-02-09 16:53:37.061953+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>764c7706-c996-42e9-90ff-87b1b496f98d, 05c46dbc-9d72-40d5-bc4c-7fee7bc3e971, 9a4d76f5-9905-4063-8bf8-47e103987515, d5e4882a-3c17-4965-b059-66432a50a3cd, 00b3d5e7-4644-4138-b73d-b0511b3c9e2a, e143a2d5-5641-4dcc-8ae4-786fd777a30a, e2b9d903-4015-4d09-902b-9150a7196cea, 9df38be1-d2f4-4be1-9022-8f0570a238b9, 3078b49f-3eff-48d1-8d9b-a8780b329ecc, 21bff9df-828f-40e7-8a22-449a2e636b44, f78a7030-bd25-4bf7-ba0d-a18cfe3790e0, 10c1ac25-d626-4413-8d5d-1bed42d0e65c, b179b693-b6b6-4ff9-b2a4-2a639d88bc9b, da7b9cf0-81e8-452b-8b70-689406dc9548, a9a9b62c-9d37-427f-99af-67725558bf9b, 1c14591a-96b4-4059-bb63-2d2bc4e308d5, add660ac-0ebf-4a24-bb6d-6cdc875866c8</td></tr><tr><th>steps</th> <td>logcontrol</td></tr><tr><th>published</th> <td>False</td></tr></table>

```python
import datetime
import time
import pytz

inference_start = datetime.datetime.now(pytz.utc)

# adding sleep to ensure log distinction
time.sleep(15)

results = mainpipeline.infer_from_file('./data/test-1000.df.json')

inference_end = datetime.datetime.now(pytz.utc)

# first 20 results
display(results.head(20))

# only results that trigger the anomaly too_high
results.loc[results['anomaly.too_high'] == True]
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.variable</th>
      <th>anomaly.count</th>
      <th>anomaly.too_high</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.75]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[615094.56]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
      <td>[448627.72]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[758714.2]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[3.0, 1.75, 2200.0, 11520.0, 1.0, 0.0, 0.0, 4.0, 7.0, 2200.0, 0.0, 47.7659, -122.341, 1690.0, 8038.0, 62.0, 0.0, 0.0]</td>
      <td>[513264.7]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[3.0, 2.0, 2140.0, 4923.0, 1.0, 0.0, 0.0, 4.0, 8.0, 1070.0, 1070.0, 47.6902, -122.339, 1470.0, 4923.0, 86.0, 0.0, 0.0]</td>
      <td>[668288.0]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 3.5, 3590.0, 5334.0, 2.0, 0.0, 2.0, 3.0, 9.0, 3140.0, 450.0, 47.6763, -122.267, 2100.0, 6250.0, 9.0, 0.0, 0.0]</td>
      <td>[1004846.5]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[3.0, 2.0, 1280.0, 960.0, 2.0, 0.0, 0.0, 3.0, 9.0, 1040.0, 240.0, 47.602, -122.311, 1280.0, 1173.0, 0.0, 0.0, 0.0]</td>
      <td>[684577.2]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 2.5, 2820.0, 15000.0, 2.0, 0.0, 0.0, 4.0, 9.0, 2820.0, 0.0, 47.7255, -122.101, 2440.0, 15000.0, 29.0, 0.0, 0.0]</td>
      <td>[727898.1]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[3.0, 2.25, 1790.0, 11393.0, 1.0, 0.0, 0.0, 3.0, 8.0, 1790.0, 0.0, 47.6297, -122.099, 2290.0, 11894.0, 36.0, 0.0, 0.0]</td>
      <td>[559631.1]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[3.0, 1.5, 1010.0, 7683.0, 1.5, 0.0, 0.0, 5.0, 7.0, 1010.0, 0.0, 47.72, -122.318, 1550.0, 7271.0, 61.0, 0.0, 0.0]</td>
      <td>[340764.53]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[3.0, 2.0, 1270.0, 1323.0, 3.0, 0.0, 0.0, 3.0, 8.0, 1270.0, 0.0, 47.6934, -122.342, 1330.0, 1323.0, 8.0, 0.0, 0.0]</td>
      <td>[442168.06]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 1.75, 2070.0, 9120.0, 1.0, 0.0, 0.0, 4.0, 7.0, 1250.0, 820.0, 47.6045, -122.123, 1650.0, 8400.0, 57.0, 0.0, 0.0]</td>
      <td>[630865.6]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 1.0, 1620.0, 4080.0, 1.5, 0.0, 0.0, 3.0, 7.0, 1620.0, 0.0, 47.6696, -122.324, 1760.0, 4080.0, 91.0, 0.0, 0.0]</td>
      <td>[559631.1]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 3.25, 3990.0, 9786.0, 2.0, 0.0, 0.0, 3.0, 9.0, 3990.0, 0.0, 47.6784, -122.026, 3920.0, 8200.0, 10.0, 0.0, 0.0]</td>
      <td>[909441.1]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 2.0, 1780.0, 19843.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1780.0, 0.0, 47.4414, -122.154, 2210.0, 13500.0, 52.0, 0.0, 0.0]</td>
      <td>[313096.0]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 2.5, 2130.0, 6003.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2130.0, 0.0, 47.4518, -122.12, 1940.0, 4529.0, 11.0, 0.0, 0.0]</td>
      <td>[404040.8]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[3.0, 1.75, 1660.0, 10440.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1040.0, 620.0, 47.4448, -121.77, 1240.0, 10380.0, 36.0, 0.0, 0.0]</td>
      <td>[292859.5]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[3.0, 2.5, 2110.0, 4118.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2110.0, 0.0, 47.3878, -122.153, 2110.0, 4044.0, 25.0, 0.0, 0.0]</td>
      <td>[338357.88]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 2.25, 2200.0, 11250.0, 1.5, 0.0, 0.0, 5.0, 7.0, 1300.0, 900.0, 47.6845, -122.201, 2320.0, 10814.0, 94.0, 0.0, 0.0]</td>
      <td>[682284.6]</td>
      <td>0</td>
      <td>False</td>
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
      <th>anomaly.count</th>
      <th>anomaly.too_high</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 3.5, 3590.0, 5334.0, 2.0, 0.0, 2.0, 3.0, 9.0, 3140.0, 450.0, 47.6763, -122.267, 2100.0, 6250.0, 9.0, 0.0, 0.0]</td>
      <td>[1004846.5]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 3.0, 3710.0, 20000.0, 2.0, 0.0, 2.0, 5.0, 10.0, 2760.0, 950.0, 47.6696, -122.261, 3970.0, 20000.0, 79.0, 0.0, 0.0]</td>
      <td>[1514079.8]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>40</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 4.5, 5120.0, 41327.0, 2.0, 0.0, 0.0, 3.0, 10.0, 3290.0, 1830.0, 47.7009, -122.059, 3360.0, 82764.0, 6.0, 0.0, 0.0]</td>
      <td>[1204324.8]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>63</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 3.0, 4040.0, 19700.0, 2.0, 0.0, 0.0, 3.0, 11.0, 4040.0, 0.0, 47.7205, -122.127, 3930.0, 21887.0, 27.0, 0.0, 0.0]</td>
      <td>[1028923.06]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>110</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 2.5, 3470.0, 20445.0, 2.0, 0.0, 0.0, 4.0, 10.0, 3470.0, 0.0, 47.547, -122.219, 3360.0, 21950.0, 51.0, 0.0, 0.0]</td>
      <td>[1412215.3]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>130</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 2.75, 2620.0, 13777.0, 1.5, 0.0, 2.0, 4.0, 9.0, 1720.0, 900.0, 47.58, -122.285, 3530.0, 9287.0, 88.0, 0.0, 0.0]</td>
      <td>[1223839.1]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>133</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[5.0, 2.25, 3320.0, 13138.0, 1.0, 0.0, 2.0, 4.0, 9.0, 1900.0, 1420.0, 47.759, -122.269, 2820.0, 13138.0, 51.0, 0.0, 0.0]</td>
      <td>[1108000.1]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>154</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 2.75, 3800.0, 9606.0, 2.0, 0.0, 0.0, 3.0, 9.0, 3800.0, 0.0, 47.7368, -122.208, 3400.0, 9677.0, 6.0, 0.0, 0.0]</td>
      <td>[1039781.25]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>160</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[5.0, 3.5, 4150.0, 13232.0, 2.0, 0.0, 0.0, 3.0, 11.0, 4150.0, 0.0, 47.3417, -122.182, 3840.0, 15121.0, 9.0, 0.0, 0.0]</td>
      <td>[1042119.1]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>210</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 3.5, 4300.0, 70407.0, 2.0, 0.0, 0.0, 3.0, 10.0, 2710.0, 1590.0, 47.4472, -122.092, 3520.0, 26727.0, 22.0, 0.0, 0.0]</td>
      <td>[1115275.0]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>239</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 3.25, 5010.0, 49222.0, 2.0, 0.0, 0.0, 5.0, 9.0, 3710.0, 1300.0, 47.5489, -122.092, 3140.0, 54014.0, 36.0, 0.0, 0.0]</td>
      <td>[1092274.1]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>248</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 3.75, 4410.0, 8112.0, 3.0, 0.0, 4.0, 3.0, 11.0, 3570.0, 840.0, 47.5888, -122.392, 2770.0, 5750.0, 12.0, 0.0, 0.0]</td>
      <td>[1967344.1]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>255</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 3.0, 4750.0, 21701.0, 1.5, 0.0, 0.0, 5.0, 11.0, 4750.0, 0.0, 47.6454, -122.218, 3120.0, 18551.0, 38.0, 0.0, 0.0]</td>
      <td>[2002393.5]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>271</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[5.0, 3.25, 5790.0, 13726.0, 2.0, 0.0, 3.0, 3.0, 10.0, 4430.0, 1360.0, 47.5388, -122.114, 5790.0, 13726.0, 0.0, 0.0, 0.0]</td>
      <td>[1189654.4]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>281</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[3.0, 3.0, 3570.0, 6250.0, 2.0, 0.0, 2.0, 3.0, 10.0, 2710.0, 860.0, 47.5624, -122.399, 2550.0, 7596.0, 30.0, 0.0, 0.0]</td>
      <td>[1124493.3]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>282</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[3.0, 2.75, 3170.0, 34850.0, 1.0, 0.0, 0.0, 5.0, 9.0, 3170.0, 0.0, 47.6611, -122.169, 3920.0, 36740.0, 58.0, 0.0, 0.0]</td>
      <td>[1227073.8]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>283</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 2.75, 3260.0, 19542.0, 1.0, 0.0, 0.0, 4.0, 10.0, 2170.0, 1090.0, 47.6245, -122.236, 3480.0, 19863.0, 46.0, 0.0, 0.0]</td>
      <td>[1364650.3]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>285</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 2.75, 4020.0, 18745.0, 2.0, 0.0, 4.0, 4.0, 10.0, 2830.0, 1190.0, 47.6042, -122.21, 3150.0, 20897.0, 26.0, 0.0, 0.0]</td>
      <td>[1322835.9]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>323</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[3.0, 3.0, 2480.0, 5500.0, 2.0, 0.0, 3.0, 3.0, 10.0, 1730.0, 750.0, 47.6466, -122.404, 2950.0, 5670.0, 64.0, 1.0, 55.0]</td>
      <td>[1100884.1]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>351</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[5.0, 4.0, 4660.0, 9900.0, 2.0, 0.0, 2.0, 4.0, 9.0, 2600.0, 2060.0, 47.5135, -122.2, 3380.0, 9900.0, 35.0, 0.0, 0.0]</td>
      <td>[1058105.0]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>360</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 3.5, 3770.0, 8501.0, 2.0, 0.0, 0.0, 3.0, 10.0, 3770.0, 0.0, 47.6744, -122.196, 1520.0, 9660.0, 6.0, 0.0, 0.0]</td>
      <td>[1169643.0]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>398</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[3.0, 2.25, 2390.0, 7875.0, 1.0, 0.0, 1.0, 3.0, 10.0, 1980.0, 410.0, 47.6515, -122.278, 3720.0, 9075.0, 66.0, 0.0, 0.0]</td>
      <td>[1364149.9]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>414</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[5.0, 3.5, 5430.0, 10327.0, 2.0, 0.0, 2.0, 3.0, 10.0, 4010.0, 1420.0, 47.5476, -122.116, 4340.0, 10324.0, 7.0, 0.0, 0.0]</td>
      <td>[1207858.6]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>443</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[5.0, 4.0, 4360.0, 8030.0, 2.0, 0.0, 0.0, 3.0, 10.0, 4360.0, 0.0, 47.5923, -121.973, 3570.0, 6185.0, 0.0, 0.0, 0.0]</td>
      <td>[1160512.8]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>497</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 2.5, 4090.0, 11225.0, 2.0, 0.0, 0.0, 3.0, 10.0, 4090.0, 0.0, 47.581, -121.971, 3510.0, 8762.0, 9.0, 0.0, 0.0]</td>
      <td>[1048372.4]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>513</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 3.25, 3320.0, 8587.0, 3.0, 0.0, 0.0, 3.0, 11.0, 2950.0, 370.0, 47.691, -122.337, 1860.0, 5668.0, 6.0, 0.0, 0.0]</td>
      <td>[1130661.0]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>520</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[5.0, 3.75, 4170.0, 8142.0, 2.0, 0.0, 2.0, 3.0, 10.0, 4170.0, 0.0, 47.5354, -122.181, 3030.0, 7980.0, 9.0, 0.0, 0.0]</td>
      <td>[1098628.8]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>530</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 4.25, 3500.0, 8750.0, 1.0, 0.0, 4.0, 5.0, 9.0, 2140.0, 1360.0, 47.7222, -122.367, 3110.0, 8750.0, 63.0, 0.0, 0.0]</td>
      <td>[1140733.8]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>535</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 3.5, 4460.0, 16271.0, 2.0, 0.0, 2.0, 3.0, 11.0, 4460.0, 0.0, 47.5862, -121.97, 4540.0, 17122.0, 13.0, 0.0, 0.0]</td>
      <td>[1208638.0]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>556</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 3.5, 4285.0, 9567.0, 2.0, 0.0, 1.0, 5.0, 10.0, 3485.0, 800.0, 47.6434, -122.409, 2960.0, 6902.0, 68.0, 0.0, 0.0]</td>
      <td>[1886959.4]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>623</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 3.25, 4240.0, 25639.0, 2.0, 0.0, 3.0, 3.0, 10.0, 3550.0, 690.0, 47.3241, -122.378, 3590.0, 24967.0, 25.0, 0.0, 0.0]</td>
      <td>[1156651.3]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>624</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 3.5, 3440.0, 9776.0, 2.0, 0.0, 0.0, 3.0, 10.0, 3440.0, 0.0, 47.5374, -122.216, 2400.0, 11000.0, 9.0, 0.0, 0.0]</td>
      <td>[1124493.3]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>634</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 3.25, 4700.0, 38412.0, 2.0, 0.0, 0.0, 3.0, 10.0, 3420.0, 1280.0, 47.6445, -122.167, 3640.0, 35571.0, 36.0, 0.0, 0.0]</td>
      <td>[1164589.4]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>651</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[3.0, 3.0, 3920.0, 13085.0, 2.0, 1.0, 4.0, 4.0, 11.0, 3920.0, 0.0, 47.5716, -122.204, 3450.0, 13287.0, 18.0, 0.0, 0.0]</td>
      <td>[1452224.5]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>658</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[3.0, 3.25, 3230.0, 7800.0, 2.0, 0.0, 3.0, 3.0, 10.0, 3230.0, 0.0, 47.6348, -122.403, 3030.0, 6600.0, 9.0, 0.0, 0.0]</td>
      <td>[1077279.3]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>671</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[3.0, 3.5, 3080.0, 6495.0, 2.0, 0.0, 3.0, 3.0, 11.0, 2530.0, 550.0, 47.6321, -122.393, 4120.0, 8620.0, 18.0, 1.0, 10.0]</td>
      <td>[1122811.8]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>685</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 2.5, 4200.0, 35267.0, 2.0, 0.0, 0.0, 3.0, 11.0, 4200.0, 0.0, 47.7108, -122.071, 3540.0, 22234.0, 24.0, 0.0, 0.0]</td>
      <td>[1181336.0]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>686</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 3.25, 4160.0, 47480.0, 2.0, 0.0, 0.0, 3.0, 10.0, 4160.0, 0.0, 47.7266, -122.115, 3400.0, 40428.0, 19.0, 0.0, 0.0]</td>
      <td>[1082353.3]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>698</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 4.5, 5770.0, 10050.0, 1.0, 0.0, 3.0, 5.0, 9.0, 3160.0, 2610.0, 47.677, -122.275, 2950.0, 6700.0, 65.0, 0.0, 0.0]</td>
      <td>[1689843.3]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>711</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[3.0, 2.5, 5403.0, 24069.0, 2.0, 1.0, 4.0, 4.0, 12.0, 5403.0, 0.0, 47.4169, -122.348, 3980.0, 104374.0, 39.0, 0.0, 0.0]</td>
      <td>[1946437.3]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>720</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[5.0, 3.0, 3420.0, 18129.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2540.0, 880.0, 47.5333, -122.217, 3750.0, 16316.0, 62.0, 1.0, 53.0]</td>
      <td>[1325961.0]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>722</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[3.0, 3.25, 4560.0, 13363.0, 1.0, 0.0, 4.0, 3.0, 11.0, 2760.0, 1800.0, 47.6205, -122.214, 4060.0, 13362.0, 20.0, 0.0, 0.0]</td>
      <td>[2005883.1]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>726</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[5.0, 3.5, 4200.0, 5400.0, 2.0, 0.0, 0.0, 3.0, 9.0, 3140.0, 1060.0, 47.7077, -122.12, 3300.0, 5564.0, 2.0, 0.0, 0.0]</td>
      <td>[1052898.0]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>737</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 3.25, 2980.0, 7000.0, 2.0, 0.0, 3.0, 3.0, 10.0, 2140.0, 840.0, 47.5933, -122.292, 2200.0, 4800.0, 114.0, 1.0, 114.0]</td>
      <td>[1156206.5]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>740</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 4.5, 6380.0, 88714.0, 2.0, 0.0, 0.0, 3.0, 12.0, 6380.0, 0.0, 47.5592, -122.015, 3040.0, 7113.0, 8.0, 0.0, 0.0]</td>
      <td>[1355747.1]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>782</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[5.0, 4.25, 4860.0, 9453.0, 1.5, 0.0, 1.0, 5.0, 10.0, 3100.0, 1760.0, 47.6196, -122.286, 3150.0, 8557.0, 109.0, 0.0, 0.0]</td>
      <td>[1910823.8]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>798</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 2.5, 2790.0, 5450.0, 2.0, 0.0, 0.0, 3.0, 10.0, 1930.0, 860.0, 47.6453, -122.303, 2320.0, 5450.0, 89.0, 1.0, 75.0]</td>
      <td>[1097757.4]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>818</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 4.0, 4620.0, 130208.0, 2.0, 0.0, 0.0, 3.0, 10.0, 4620.0, 0.0, 47.5885, -121.939, 4620.0, 131007.0, 1.0, 0.0, 0.0]</td>
      <td>[1164589.4]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>827</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 2.5, 3340.0, 10422.0, 2.0, 0.0, 0.0, 3.0, 10.0, 3340.0, 0.0, 47.6515, -122.197, 1770.0, 9490.0, 18.0, 0.0, 0.0]</td>
      <td>[1103101.4]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>828</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[5.0, 3.5, 3760.0, 10207.0, 2.0, 0.0, 0.0, 3.0, 10.0, 3150.0, 610.0, 47.5605, -122.225, 3550.0, 12118.0, 46.0, 0.0, 0.0]</td>
      <td>[1489624.5]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>901</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 2.25, 4470.0, 60373.0, 2.0, 0.0, 0.0, 3.0, 11.0, 4470.0, 0.0, 47.7289, -122.127, 3210.0, 40450.0, 26.0, 0.0, 0.0]</td>
      <td>[1208638.0]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>912</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[3.0, 2.25, 2960.0, 8330.0, 1.0, 0.0, 3.0, 4.0, 10.0, 2260.0, 700.0, 47.7035, -122.385, 2960.0, 8840.0, 62.0, 0.0, 0.0]</td>
      <td>[1178314.0]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>919</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 3.25, 5180.0, 19850.0, 2.0, 0.0, 3.0, 3.0, 12.0, 3540.0, 1640.0, 47.562, -122.162, 3160.0, 9750.0, 9.0, 0.0, 0.0]</td>
      <td>[1295531.3]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>941</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 3.75, 3770.0, 4000.0, 2.5, 0.0, 0.0, 5.0, 9.0, 2890.0, 880.0, 47.6157, -122.287, 2800.0, 5000.0, 98.0, 0.0, 0.0]</td>
      <td>[1182821.0]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>965</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[6.0, 4.0, 5310.0, 12741.0, 2.0, 0.0, 2.0, 3.0, 10.0, 3600.0, 1710.0, 47.5696, -122.213, 4190.0, 12632.0, 48.0, 0.0, 0.0]</td>
      <td>[2016006.0]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>973</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[5.0, 2.0, 3540.0, 9970.0, 2.0, 0.0, 3.0, 3.0, 9.0, 3540.0, 0.0, 47.7108, -122.277, 2280.0, 7195.0, 44.0, 0.0, 0.0]</td>
      <td>[1085835.8]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>997</th>
      <td>2024-02-09 16:54:02.507</td>
      <td>[4.0, 3.25, 2910.0, 1880.0, 2.0, 0.0, 3.0, 5.0, 9.0, 1830.0, 1080.0, 47.616, -122.282, 3100.0, 8200.0, 100.0, 0.0, 0.0]</td>
      <td>[1060847.5]</td>
      <td>1</td>
      <td>True</td>
    </tr>
  </tbody>
</table>

```python
### Anomaly Detection Logs

Pipeline logs retrieves with `wallaroo.pipeline.logs` include the `anomaly` dataset.
```

```python
logs = mainpipeline.logs(limit=1000)
display(logs)
display(logs.loc[logs['anomaly.too_high'] == True])
```

    Warning: There are more logs available. Please set a larger limit or request a file using export_logs.

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.variable</th>
      <th>anomaly.count</th>
      <th>anomaly.too_high</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[3.0, 2.0, 2005.0, 7000.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1605.0, 400.0, 47.6039, -122.298, 1750.0, 4500.0, 34.0, 0.0, 0.0]</td>
      <td>[581003.0]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[3.0, 1.75, 2910.0, 37461.0, 1.0, 0.0, 0.0, 4.0, 7.0, 1530.0, 1380.0, 47.7015, -122.164, 2520.0, 18295.0, 47.0, 0.0, 0.0]</td>
      <td>[706823.56]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[4.0, 3.25, 2910.0, 1880.0, 2.0, 0.0, 3.0, 5.0, 9.0, 1830.0, 1080.0, 47.616, -122.282, 3100.0, 8200.0, 100.0, 0.0, 0.0]</td>
      <td>[1060847.5]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[4.0, 1.75, 2700.0, 7875.0, 1.5, 0.0, 0.0, 4.0, 8.0, 2700.0, 0.0, 47.454, -122.144, 2220.0, 7875.0, 46.0, 0.0, 0.0]</td>
      <td>[441960.38]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[3.0, 2.5, 2900.0, 23550.0, 1.0, 0.0, 0.0, 3.0, 10.0, 1490.0, 1410.0, 47.5708, -122.153, 2900.0, 19604.0, 27.0, 0.0, 0.0]</td>
      <td>[827411.0]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[3.0, 1.75, 2200.0, 11520.0, 1.0, 0.0, 0.0, 4.0, 7.0, 2200.0, 0.0, 47.7659, -122.341, 1690.0, 8038.0, 62.0, 0.0, 0.0]</td>
      <td>[513264.7]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>996</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[758714.2]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>997</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
      <td>[448627.72]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>998</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[615094.56]</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>999</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.75]</td>
      <td>0</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 5 columns</p>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.variable</th>
      <th>anomaly.count</th>
      <th>anomaly.too_high</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[4.0, 3.25, 2910.0, 1880.0, 2.0, 0.0, 3.0, 5.0, 9.0, 1830.0, 1080.0, 47.616, -122.282, 3100.0, 8200.0, 100.0, 0.0, 0.0]</td>
      <td>[1060847.5]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[5.0, 2.0, 3540.0, 9970.0, 2.0, 0.0, 3.0, 3.0, 9.0, 3540.0, 0.0, 47.7108, -122.277, 2280.0, 7195.0, 44.0, 0.0, 0.0]</td>
      <td>[1085835.8]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[6.0, 4.0, 5310.0, 12741.0, 2.0, 0.0, 2.0, 3.0, 10.0, 3600.0, 1710.0, 47.5696, -122.213, 4190.0, 12632.0, 48.0, 0.0, 0.0]</td>
      <td>[2016006.0]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>58</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[4.0, 3.75, 3770.0, 4000.0, 2.5, 0.0, 0.0, 5.0, 9.0, 2890.0, 880.0, 47.6157, -122.287, 2800.0, 5000.0, 98.0, 0.0, 0.0]</td>
      <td>[1182821.0]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>80</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[4.0, 3.25, 5180.0, 19850.0, 2.0, 0.0, 3.0, 3.0, 12.0, 3540.0, 1640.0, 47.562, -122.162, 3160.0, 9750.0, 9.0, 0.0, 0.0]</td>
      <td>[1295531.2]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>87</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[3.0, 2.25, 2960.0, 8330.0, 1.0, 0.0, 3.0, 4.0, 10.0, 2260.0, 700.0, 47.7035, -122.385, 2960.0, 8840.0, 62.0, 0.0, 0.0]</td>
      <td>[1178314.0]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>98</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[4.0, 2.25, 4470.0, 60373.0, 2.0, 0.0, 0.0, 3.0, 11.0, 4470.0, 0.0, 47.7289, -122.127, 3210.0, 40450.0, 26.0, 0.0, 0.0]</td>
      <td>[1208638.0]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>171</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[5.0, 3.5, 3760.0, 10207.0, 2.0, 0.0, 0.0, 3.0, 10.0, 3150.0, 610.0, 47.5605, -122.225, 3550.0, 12118.0, 46.0, 0.0, 0.0]</td>
      <td>[1489624.5]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>172</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[4.0, 2.5, 3340.0, 10422.0, 2.0, 0.0, 0.0, 3.0, 10.0, 3340.0, 0.0, 47.6515, -122.197, 1770.0, 9490.0, 18.0, 0.0, 0.0]</td>
      <td>[1103101.4]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>181</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[4.0, 4.0, 4620.0, 130208.0, 2.0, 0.0, 0.0, 3.0, 10.0, 4620.0, 0.0, 47.5885, -121.939, 4620.0, 131007.0, 1.0, 0.0, 0.0]</td>
      <td>[1164589.4]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>201</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[4.0, 2.5, 2790.0, 5450.0, 2.0, 0.0, 0.0, 3.0, 10.0, 1930.0, 860.0, 47.6453, -122.303, 2320.0, 5450.0, 89.0, 1.0, 75.0]</td>
      <td>[1097757.4]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>217</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[5.0, 4.25, 4860.0, 9453.0, 1.5, 0.0, 1.0, 5.0, 10.0, 3100.0, 1760.0, 47.6196, -122.286, 3150.0, 8557.0, 109.0, 0.0, 0.0]</td>
      <td>[1910823.8]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>259</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[4.0, 4.5, 6380.0, 88714.0, 2.0, 0.0, 0.0, 3.0, 12.0, 6380.0, 0.0, 47.5592, -122.015, 3040.0, 7113.0, 8.0, 0.0, 0.0]</td>
      <td>[1355747.1]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>262</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[4.0, 3.25, 2980.0, 7000.0, 2.0, 0.0, 3.0, 3.0, 10.0, 2140.0, 840.0, 47.5933, -122.292, 2200.0, 4800.0, 114.0, 1.0, 114.0]</td>
      <td>[1156206.5]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>273</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[5.0, 3.5, 4200.0, 5400.0, 2.0, 0.0, 0.0, 3.0, 9.0, 3140.0, 1060.0, 47.7077, -122.12, 3300.0, 5564.0, 2.0, 0.0, 0.0]</td>
      <td>[1052898.0]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>277</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[3.0, 3.25, 4560.0, 13363.0, 1.0, 0.0, 4.0, 3.0, 11.0, 2760.0, 1800.0, 47.6205, -122.214, 4060.0, 13362.0, 20.0, 0.0, 0.0]</td>
      <td>[2005883.1]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>279</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[5.0, 3.0, 3420.0, 18129.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2540.0, 880.0, 47.5333, -122.217, 3750.0, 16316.0, 62.0, 1.0, 53.0]</td>
      <td>[1325961.0]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>288</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[3.0, 2.5, 5403.0, 24069.0, 2.0, 1.0, 4.0, 4.0, 12.0, 5403.0, 0.0, 47.4169, -122.348, 3980.0, 104374.0, 39.0, 0.0, 0.0]</td>
      <td>[1946437.2]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>301</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[4.0, 4.5, 5770.0, 10050.0, 1.0, 0.0, 3.0, 5.0, 9.0, 3160.0, 2610.0, 47.677, -122.275, 2950.0, 6700.0, 65.0, 0.0, 0.0]</td>
      <td>[1689843.2]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>313</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[4.0, 3.25, 4160.0, 47480.0, 2.0, 0.0, 0.0, 3.0, 10.0, 4160.0, 0.0, 47.7266, -122.115, 3400.0, 40428.0, 19.0, 0.0, 0.0]</td>
      <td>[1082353.2]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>314</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[4.0, 2.5, 4200.0, 35267.0, 2.0, 0.0, 0.0, 3.0, 11.0, 4200.0, 0.0, 47.7108, -122.071, 3540.0, 22234.0, 24.0, 0.0, 0.0]</td>
      <td>[1181336.0]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>328</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[3.0, 3.5, 3080.0, 6495.0, 2.0, 0.0, 3.0, 3.0, 11.0, 2530.0, 550.0, 47.6321, -122.393, 4120.0, 8620.0, 18.0, 1.0, 10.0]</td>
      <td>[1122811.8]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>341</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[3.0, 3.25, 3230.0, 7800.0, 2.0, 0.0, 3.0, 3.0, 10.0, 3230.0, 0.0, 47.6348, -122.403, 3030.0, 6600.0, 9.0, 0.0, 0.0]</td>
      <td>[1077279.2]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>348</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[3.0, 3.0, 3920.0, 13085.0, 2.0, 1.0, 4.0, 4.0, 11.0, 3920.0, 0.0, 47.5716, -122.204, 3450.0, 13287.0, 18.0, 0.0, 0.0]</td>
      <td>[1452224.5]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>365</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[4.0, 3.25, 4700.0, 38412.0, 2.0, 0.0, 0.0, 3.0, 10.0, 3420.0, 1280.0, 47.6445, -122.167, 3640.0, 35571.0, 36.0, 0.0, 0.0]</td>
      <td>[1164589.4]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>375</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[4.0, 3.5, 3440.0, 9776.0, 2.0, 0.0, 0.0, 3.0, 10.0, 3440.0, 0.0, 47.5374, -122.216, 2400.0, 11000.0, 9.0, 0.0, 0.0]</td>
      <td>[1124493.2]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>376</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[4.0, 3.25, 4240.0, 25639.0, 2.0, 0.0, 3.0, 3.0, 10.0, 3550.0, 690.0, 47.3241, -122.378, 3590.0, 24967.0, 25.0, 0.0, 0.0]</td>
      <td>[1156651.2]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>443</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[4.0, 3.5, 4285.0, 9567.0, 2.0, 0.0, 1.0, 5.0, 10.0, 3485.0, 800.0, 47.6434, -122.409, 2960.0, 6902.0, 68.0, 0.0, 0.0]</td>
      <td>[1886959.4]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>464</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[4.0, 3.5, 4460.0, 16271.0, 2.0, 0.0, 2.0, 3.0, 11.0, 4460.0, 0.0, 47.5862, -121.97, 4540.0, 17122.0, 13.0, 0.0, 0.0]</td>
      <td>[1208638.0]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>469</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[4.0, 4.25, 3500.0, 8750.0, 1.0, 0.0, 4.0, 5.0, 9.0, 2140.0, 1360.0, 47.7222, -122.367, 3110.0, 8750.0, 63.0, 0.0, 0.0]</td>
      <td>[1140733.8]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>479</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[5.0, 3.75, 4170.0, 8142.0, 2.0, 0.0, 2.0, 3.0, 10.0, 4170.0, 0.0, 47.5354, -122.181, 3030.0, 7980.0, 9.0, 0.0, 0.0]</td>
      <td>[1098628.8]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>486</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[4.0, 3.25, 3320.0, 8587.0, 3.0, 0.0, 0.0, 3.0, 11.0, 2950.0, 370.0, 47.691, -122.337, 1860.0, 5668.0, 6.0, 0.0, 0.0]</td>
      <td>[1130661.0]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>502</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[4.0, 2.5, 4090.0, 11225.0, 2.0, 0.0, 0.0, 3.0, 10.0, 4090.0, 0.0, 47.581, -121.971, 3510.0, 8762.0, 9.0, 0.0, 0.0]</td>
      <td>[1048372.4]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>556</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[5.0, 4.0, 4360.0, 8030.0, 2.0, 0.0, 0.0, 3.0, 10.0, 4360.0, 0.0, 47.5923, -121.973, 3570.0, 6185.0, 0.0, 0.0, 0.0]</td>
      <td>[1160512.8]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>585</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[5.0, 3.5, 5430.0, 10327.0, 2.0, 0.0, 2.0, 3.0, 10.0, 4010.0, 1420.0, 47.5476, -122.116, 4340.0, 10324.0, 7.0, 0.0, 0.0]</td>
      <td>[1207858.6]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>601</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[3.0, 2.25, 2390.0, 7875.0, 1.0, 0.0, 1.0, 3.0, 10.0, 1980.0, 410.0, 47.6515, -122.278, 3720.0, 9075.0, 66.0, 0.0, 0.0]</td>
      <td>[1364149.9]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>639</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[4.0, 3.5, 3770.0, 8501.0, 2.0, 0.0, 0.0, 3.0, 10.0, 3770.0, 0.0, 47.6744, -122.196, 1520.0, 9660.0, 6.0, 0.0, 0.0]</td>
      <td>[1169643.0]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>648</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[5.0, 4.0, 4660.0, 9900.0, 2.0, 0.0, 2.0, 4.0, 9.0, 2600.0, 2060.0, 47.5135, -122.2, 3380.0, 9900.0, 35.0, 0.0, 0.0]</td>
      <td>[1058105.0]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>676</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[3.0, 3.0, 2480.0, 5500.0, 2.0, 0.0, 3.0, 3.0, 10.0, 1730.0, 750.0, 47.6466, -122.404, 2950.0, 5670.0, 64.0, 1.0, 55.0]</td>
      <td>[1100884.1]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>714</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[4.0, 2.75, 4020.0, 18745.0, 2.0, 0.0, 4.0, 4.0, 10.0, 2830.0, 1190.0, 47.6042, -122.21, 3150.0, 20897.0, 26.0, 0.0, 0.0]</td>
      <td>[1322835.9]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>716</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[4.0, 2.75, 3260.0, 19542.0, 1.0, 0.0, 0.0, 4.0, 10.0, 2170.0, 1090.0, 47.6245, -122.236, 3480.0, 19863.0, 46.0, 0.0, 0.0]</td>
      <td>[1364650.2]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>717</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[3.0, 2.75, 3170.0, 34850.0, 1.0, 0.0, 0.0, 5.0, 9.0, 3170.0, 0.0, 47.6611, -122.169, 3920.0, 36740.0, 58.0, 0.0, 0.0]</td>
      <td>[1227073.8]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>718</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[3.0, 3.0, 3570.0, 6250.0, 2.0, 0.0, 2.0, 3.0, 10.0, 2710.0, 860.0, 47.5624, -122.399, 2550.0, 7596.0, 30.0, 0.0, 0.0]</td>
      <td>[1124493.2]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>728</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[5.0, 3.25, 5790.0, 13726.0, 2.0, 0.0, 3.0, 3.0, 10.0, 4430.0, 1360.0, 47.5388, -122.114, 5790.0, 13726.0, 0.0, 0.0, 0.0]</td>
      <td>[1189654.4]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>744</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[4.0, 3.0, 4750.0, 21701.0, 1.5, 0.0, 0.0, 5.0, 11.0, 4750.0, 0.0, 47.6454, -122.218, 3120.0, 18551.0, 38.0, 0.0, 0.0]</td>
      <td>[2002393.5]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>751</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[4.0, 3.75, 4410.0, 8112.0, 3.0, 0.0, 4.0, 3.0, 11.0, 3570.0, 840.0, 47.5888, -122.392, 2770.0, 5750.0, 12.0, 0.0, 0.0]</td>
      <td>[1967344.1]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>760</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[4.0, 3.25, 5010.0, 49222.0, 2.0, 0.0, 0.0, 5.0, 9.0, 3710.0, 1300.0, 47.5489, -122.092, 3140.0, 54014.0, 36.0, 0.0, 0.0]</td>
      <td>[1092274.1]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>789</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[4.0, 3.5, 4300.0, 70407.0, 2.0, 0.0, 0.0, 3.0, 10.0, 2710.0, 1590.0, 47.4472, -122.092, 3520.0, 26727.0, 22.0, 0.0, 0.0]</td>
      <td>[1115275.0]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>839</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[5.0, 3.5, 4150.0, 13232.0, 2.0, 0.0, 0.0, 3.0, 11.0, 4150.0, 0.0, 47.3417, -122.182, 3840.0, 15121.0, 9.0, 0.0, 0.0]</td>
      <td>[1042119.1]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>845</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[4.0, 2.75, 3800.0, 9606.0, 2.0, 0.0, 0.0, 3.0, 9.0, 3800.0, 0.0, 47.7368, -122.208, 3400.0, 9677.0, 6.0, 0.0, 0.0]</td>
      <td>[1039781.25]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>866</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[5.0, 2.25, 3320.0, 13138.0, 1.0, 0.0, 2.0, 4.0, 9.0, 1900.0, 1420.0, 47.759, -122.269, 2820.0, 13138.0, 51.0, 0.0, 0.0]</td>
      <td>[1108000.1]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>869</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[4.0, 2.75, 2620.0, 13777.0, 1.5, 0.0, 2.0, 4.0, 9.0, 1720.0, 900.0, 47.58, -122.285, 3530.0, 9287.0, 88.0, 0.0, 0.0]</td>
      <td>[1223839.1]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>889</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[4.0, 2.5, 3470.0, 20445.0, 2.0, 0.0, 0.0, 4.0, 10.0, 3470.0, 0.0, 47.547, -122.219, 3360.0, 21950.0, 51.0, 0.0, 0.0]</td>
      <td>[1412215.2]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>936</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[4.0, 3.0, 4040.0, 19700.0, 2.0, 0.0, 0.0, 3.0, 11.0, 4040.0, 0.0, 47.7205, -122.127, 3930.0, 21887.0, 27.0, 0.0, 0.0]</td>
      <td>[1028923.06]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>959</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[4.0, 4.5, 5120.0, 41327.0, 2.0, 0.0, 0.0, 3.0, 10.0, 3290.0, 1830.0, 47.7009, -122.059, 3360.0, 82764.0, 6.0, 0.0, 0.0]</td>
      <td>[1204324.8]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>969</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[4.0, 3.0, 3710.0, 20000.0, 2.0, 0.0, 2.0, 5.0, 10.0, 2760.0, 950.0, 47.6696, -122.261, 3970.0, 20000.0, 79.0, 0.0, 0.0]</td>
      <td>[1514079.8]</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>993</th>
      <td>2024-02-09 16:49:26.521</td>
      <td>[4.0, 3.5, 3590.0, 5334.0, 2.0, 0.0, 2.0, 3.0, 9.0, 3140.0, 450.0, 47.6763, -122.267, 2100.0, 6250.0, 9.0, 0.0, 0.0]</td>
      <td>[1004846.5]</td>
      <td>1</td>
      <td>True</td>
    </tr>
  </tbody>
</table>

### Undeploy Main Pipeline

With the examples and tutorial complete, we will undeploy the main pipeline and return the resources back to the Wallaroo instance.

```python
mainpipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s ..................................... ok

<table><tr><th>name</th> <td>logpipeline-test</td></tr><tr><th>created</th> <td>2024-02-09 16:21:09.406182+00:00</td></tr><tr><th>last_updated</th> <td>2024-02-09 16:53:37.061953+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>764c7706-c996-42e9-90ff-87b1b496f98d, 05c46dbc-9d72-40d5-bc4c-7fee7bc3e971, 9a4d76f5-9905-4063-8bf8-47e103987515, d5e4882a-3c17-4965-b059-66432a50a3cd, 00b3d5e7-4644-4138-b73d-b0511b3c9e2a, e143a2d5-5641-4dcc-8ae4-786fd777a30a, e2b9d903-4015-4d09-902b-9150a7196cea, 9df38be1-d2f4-4be1-9022-8f0570a238b9, 3078b49f-3eff-48d1-8d9b-a8780b329ecc, 21bff9df-828f-40e7-8a22-449a2e636b44, f78a7030-bd25-4bf7-ba0d-a18cfe3790e0, 10c1ac25-d626-4413-8d5d-1bed42d0e65c, b179b693-b6b6-4ff9-b2a4-2a639d88bc9b, da7b9cf0-81e8-452b-8b70-689406dc9548, a9a9b62c-9d37-427f-99af-67725558bf9b, 1c14591a-96b4-4059-bb63-2d2bc4e308d5, add660ac-0ebf-4a24-bb6d-6cdc875866c8</td></tr><tr><th>steps</th> <td>logcontrol</td></tr><tr><th>published</th> <td>False</td></tr></table>

