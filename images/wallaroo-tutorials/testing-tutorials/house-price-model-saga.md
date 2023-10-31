This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-testing-tutorials/anomaly_detection).

## House Price Testing Life Cycle Comprehensive Tutorial

This tutorial simulates using Wallaroo for testing a model for inference outliers, potential model drift, and methods to test competitive models against each other and deploy the final version to use.  This demonstrates using assays to detect model or data drift, then Wallaroo Shadow Deploy to compare different models to determine which one is most fit for an organization's needs.  These features allow organizations to monitor model performance and accuracy then swap out models as needed.

* **IMPORTANT NOTE**: This tutorial assumes that the House Price Model Life Cycle Preparation notebook was run before this notebook, and that the workspace, pipeline and models used are the same.  This is **critical** for the section on Assays below.  If the preparation notebook has not been run, skip the Assays section as there will be no historical data for the assays to function on.

This tutorial will demonstrate how to:

1. Select or create a workspace, pipeline and upload the champion model.
1. Add a pipeline step with the champion model, then deploy the pipeline and perform sample inferences.
1. Create an assay and set a baseline, then demonstrate inferences that trigger the assay alert threshold.
1. Swap out the pipeline step with the champion model with a shadow deploy step that compares the champion model against two competitors.
1. Evaluate the results of the champion versus competitor models.
1. Change the pipeline step from a shadow deploy step to an A/B testing step, and show the different results.
1. Change the A/B testing step back to standard pipeline step with the original control model, then demonstrate hot swapping the control model with a challenger model without undeploying the pipeline.
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

## Initial Steps

### Import libraries

The first step is to import the libraries needed for this notebook.

```python
import wallaroo
from wallaroo.object import EntityNotFoundError
from wallaroo.framework import Framework

from IPython.display import display

# used to display DataFrame information without truncating
from IPython.display import display
import pandas as pd
pd.set_option('display.max_colwidth', None)

import datetime
import time

# used for unique connection names

import string
import random

suffix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))
suffix='baselines'
import json
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

Workspace, pipeline, and model names should be unique to each user, so we'll add in a randomly generated suffix so multiple people can run this tutorial in a Wallaroo instance without effecting each other.

```python
workspace_name = f'housepricesagaworkspace{suffix}'
main_pipeline_name = f'housepricesagapipeline'
model_name_control = f'housepricesagacontrol'
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

def get_pipeline(name, workspace):
    pipelines = workspace.pipelines()
    pipe_filter = filter(lambda x: x.name() == name, pipelines)
    pipes = list(pipe_filter)
    # we can't have a pipe in the workspace with the same name, so it's always the first
    if pipes:
        pipeline = pipes[0]
    else:
        pipeline = wl.build_pipeline(name)
    return pipeline
```

```python
workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)
```

    {'name': 'housepricesagaworkspacebaselines', 'id': 87, 'archived': False, 'created_by': 'd6a42dd8-1da9-4405-bb80-7c4b42e38b52', 'created_at': '2023-10-31T16:56:03.395454+00:00', 'models': [], 'pipelines': []}

### Upload The Champion Model

For our example, we will upload the champion model that has been trained to derive house prices from a variety of inputs.  The model file is `rf_model.onnx`, and is uploaded with the name `housingcontrol`.

```python
housing_model_control = (wl.upload_model(model_name_control, 
                                        model_file_name_control, 
                                        framework=Framework.ONNX)
                                        .configure(tensor_fields=["tensor"])
                        )
```

## Standard Pipeline Steps

### Build the Pipeline

This pipeline is made to be an example of an existing situation where a model is deployed and being used for inferences in a production environment.  We'll call it `housepricepipeline`, set `housingcontrol` as a pipeline step, then run a few sample inferences.

This pipeline will be a simple one - just a single pipeline step.

```python
mainpipeline = get_pipeline(main_pipeline_name, workspace)

# clearing from previous runs and verifying it is undeployed
mainpipeline.clear()
mainpipeline.undeploy()
mainpipeline.add_model_step(housing_model_control)

#minimum deployment config
deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(0.5).memory("1Gi").build()
mainpipeline.deploy(deployment_config = deploy_config)
```

    Waiting for deployment - this will take up to 45s ............. ok

<table><tr><th>name</th> <td>housepricesagapipeline</td></tr><tr><th>created</th> <td>2023-10-31 16:56:07.345831+00:00</td></tr><tr><th>last_updated</th> <td>2023-10-31 16:56:07.444932+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>2ea0cc42-c955-4fc3-bac9-7a2c7c22ddc1, dca84ab2-274a-4391-95dd-a99bda7621e1</td></tr><tr><th>steps</th> <td>housepricesagacontrol</td></tr><tr><th>published</th> <td>False</td></tr></table>

### Testing

We'll use two inferences as a quick sample test - one that has a house that should be determined around $700k, the other with a house determined to be around $1.5 million.  We'll also save the start and end periods for these events to for later log functionality.

```python
normal_input = pd.DataFrame.from_records({"tensor": [[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]]})
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
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-10-31 16:57:07.307</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.7]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

```python
large_house_input = pd.DataFrame.from_records({'tensor': [[4.0, 3.0, 3710.0, 20000.0, 2.0, 0.0, 2.0, 5.0, 10.0, 2760.0, 950.0, 47.6696, -122.261, 3970.0, 20000.0, 79.0, 0.0, 0.0]]})
large_house_result = mainpipeline.infer(large_house_input)
display(large_house_result)
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
      <td>2023-10-31 16:57:07.746</td>
      <td>[4.0, 3.0, 3710.0, 20000.0, 2.0, 0.0, 2.0, 5.0, 10.0, 2760.0, 950.0, 47.6696, -122.261, 3970.0, 20000.0, 79.0, 0.0, 0.0]</td>
      <td>[1514079.4]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

As one last sample, we'll run through roughly 1,000 inferences at once and show a few of the results.  For this example we'll use an Apache Arrow table, which has a smaller file size compared to uploading a pandas DataFrame JSON file.  The inference result is returned as an arrow table, which we'll convert into a pandas DataFrame to display the first 20 results.

```python
time.sleep(5)
control_model_start = datetime.datetime.now()
batch_inferences = mainpipeline.infer_from_file('./data/xtest-1k.arrow')

large_inference_result =  batch_inferences.to_pandas()
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
      <td>2023-10-31 16:57:13.771</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.7]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-10-31 16:57:13.771</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[615094.6]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-10-31 16:57:13.771</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
      <td>[448627.8]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-10-31 16:57:13.771</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[758714.3]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-10-31 16:57:13.771</td>
      <td>[3.0, 1.75, 2200.0, 11520.0, 1.0, 0.0, 0.0, 4.0, 7.0, 2200.0, 0.0, 47.7659, -122.341, 1690.0, 8038.0, 62.0, 0.0, 0.0]</td>
      <td>[513264.66]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2023-10-31 16:57:13.771</td>
      <td>[3.0, 2.0, 2140.0, 4923.0, 1.0, 0.0, 0.0, 4.0, 8.0, 1070.0, 1070.0, 47.6902, -122.339, 1470.0, 4923.0, 86.0, 0.0, 0.0]</td>
      <td>[668287.94]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2023-10-31 16:57:13.771</td>
      <td>[4.0, 3.5, 3590.0, 5334.0, 2.0, 0.0, 2.0, 3.0, 9.0, 3140.0, 450.0, 47.6763, -122.267, 2100.0, 6250.0, 9.0, 0.0, 0.0]</td>
      <td>[1004846.56]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2023-10-31 16:57:13.771</td>
      <td>[3.0, 2.0, 1280.0, 960.0, 2.0, 0.0, 0.0, 3.0, 9.0, 1040.0, 240.0, 47.602, -122.311, 1280.0, 1173.0, 0.0, 0.0, 0.0]</td>
      <td>[684577.25]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2023-10-31 16:57:13.771</td>
      <td>[4.0, 2.5, 2820.0, 15000.0, 2.0, 0.0, 0.0, 4.0, 9.0, 2820.0, 0.0, 47.7255, -122.101, 2440.0, 15000.0, 29.0, 0.0, 0.0]</td>
      <td>[727898.25]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2023-10-31 16:57:13.771</td>
      <td>[3.0, 2.25, 1790.0, 11393.0, 1.0, 0.0, 0.0, 3.0, 8.0, 1790.0, 0.0, 47.6297, -122.099, 2290.0, 11894.0, 36.0, 0.0, 0.0]</td>
      <td>[559631.06]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2023-10-31 16:57:13.771</td>
      <td>[3.0, 1.5, 1010.0, 7683.0, 1.5, 0.0, 0.0, 5.0, 7.0, 1010.0, 0.0, 47.72, -122.318, 1550.0, 7271.0, 61.0, 0.0, 0.0]</td>
      <td>[340764.53]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2023-10-31 16:57:13.771</td>
      <td>[3.0, 2.0, 1270.0, 1323.0, 3.0, 0.0, 0.0, 3.0, 8.0, 1270.0, 0.0, 47.6934, -122.342, 1330.0, 1323.0, 8.0, 0.0, 0.0]</td>
      <td>[442168.12]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2023-10-31 16:57:13.771</td>
      <td>[4.0, 1.75, 2070.0, 9120.0, 1.0, 0.0, 0.0, 4.0, 7.0, 1250.0, 820.0, 47.6045, -122.123, 1650.0, 8400.0, 57.0, 0.0, 0.0]</td>
      <td>[630865.5]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2023-10-31 16:57:13.771</td>
      <td>[4.0, 1.0, 1620.0, 4080.0, 1.5, 0.0, 0.0, 3.0, 7.0, 1620.0, 0.0, 47.6696, -122.324, 1760.0, 4080.0, 91.0, 0.0, 0.0]</td>
      <td>[559631.06]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2023-10-31 16:57:13.771</td>
      <td>[4.0, 3.25, 3990.0, 9786.0, 2.0, 0.0, 0.0, 3.0, 9.0, 3990.0, 0.0, 47.6784, -122.026, 3920.0, 8200.0, 10.0, 0.0, 0.0]</td>
      <td>[909441.25]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2023-10-31 16:57:13.771</td>
      <td>[4.0, 2.0, 1780.0, 19843.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1780.0, 0.0, 47.4414, -122.154, 2210.0, 13500.0, 52.0, 0.0, 0.0]</td>
      <td>[313096.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2023-10-31 16:57:13.771</td>
      <td>[4.0, 2.5, 2130.0, 6003.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2130.0, 0.0, 47.4518, -122.12, 1940.0, 4529.0, 11.0, 0.0, 0.0]</td>
      <td>[404040.78]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2023-10-31 16:57:13.771</td>
      <td>[3.0, 1.75, 1660.0, 10440.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1040.0, 620.0, 47.4448, -121.77, 1240.0, 10380.0, 36.0, 0.0, 0.0]</td>
      <td>[292859.44]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2023-10-31 16:57:13.771</td>
      <td>[3.0, 2.5, 2110.0, 4118.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2110.0, 0.0, 47.3878, -122.153, 2110.0, 4044.0, 25.0, 0.0, 0.0]</td>
      <td>[338357.88]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2023-10-31 16:57:13.771</td>
      <td>[4.0, 2.25, 2200.0, 11250.0, 1.5, 0.0, 0.0, 5.0, 7.0, 1300.0, 900.0, 47.6845, -122.201, 2320.0, 10814.0, 94.0, 0.0, 0.0]</td>
      <td>[682284.56]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

### Graph of Prices

Here's a distribution plot of the inferences to view the values, with the X axis being the house price in millions, and the Y axis the number of houses fitting in a bin grouping.  The majority of houses are in the \$250,000 to \$500,000 range, with some outliers in the far end.

```python
import matplotlib.pyplot as plt
houseprices = pd.DataFrame({'sell_price': large_inference_result['out.variable'].apply(lambda x: x[0])})

houseprices.hist(column='sell_price', bins=75, grid=False, figsize=(12,8))
plt.axvline(x=0, color='gray', ls='--')
_ = plt.title('Distribution of predicted home sales price')
time.sleep(5)
control_model_end = datetime.datetime.now()
```

    
{{<figure src="/images/2023.4.0/wallaroo-tutorials/testing-tutorials/house-price-model-saga_files/house-price-model-saga_19_0.png" width="800" label="png">}}
    

### Pipeline Logs

Pipeline logs with standard pipeline steps are retrieved either with:

* Pipeline `logs` which returns either a pandas DataFrame or Apache Arrow table.
* Pipeline `export_logs` which saves the logs either a pandas DataFrame JSON file or Apache Arrow table.

For full details, see the Wallaroo Documentation Pipeline Log Management guide.

#### Pipeline Log Methods

The Pipeline `logs` method accepts the following parameters.

| **Parameter** | **Type** | **Description** |
|---|---|---|
| `limit` | **Int** (*Optional*) | Limits how many log records to display.  Defaults to `100`.  If there are more pipeline logs than are being displayed, the **Warning** message `Pipeline log record limit exceeded` will be displayed.  For example, if 100 log files were requested and there are a total of 1,000, the warning message will be displayed. |
| `start_datetimert` and `end_datetime` | **DateTime** (*Optional*) | Limits logs to all logs between the `start_datetime` and `end_datetime` DateTime parameters.  **Both parameters must be provided**. Submitting a `logs()` request with only `start_datetime` or `end_datetime` will generate an exception.<br />If `start_datetime` and `end_datetime` are provided as parameters, then the records are returned in **chronological** order, with the oldest record displayed first. |
| `arrow` | **Boolean** (*Optional*) | Defaults to **False**.  If `arrow` is set to `True`, then the logs are returned as an [Apache Arrow table](https://arrow.apache.org/).  If `arrow=False`, then the logs are returned as a pandas DataFrame. |

The following examples demonstrate displaying the logs, then displaying the logs between the `control_model_start` and `control_model_end` periods, then again retrieved as an Arrow table.

```python
# pipeline log retrieval - reverse chronological order

display(mainpipeline.logs())

# pipeline log retrieval between two dates - chronological order

display(mainpipeline.logs(start_datetime=control_model_start, end_datetime=control_model_end))

# pipeline log retrieval limited to the last 5 an an arrow table

display(mainpipeline.logs(arrow=True))
```

    Warning: There are more logs available. Please set a larger limit or request a file using export_logs.

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
      <td>2023-10-31 16:57:13.771</td>
      <td>[3.0, 2.0, 2005.0, 7000.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1605.0, 400.0, 47.6039, -122.298, 1750.0, 4500.0, 34.0, 0.0, 0.0]</td>
      <td>[581002.94]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-10-31 16:57:13.771</td>
      <td>[3.0, 1.75, 2910.0, 37461.0, 1.0, 0.0, 0.0, 4.0, 7.0, 1530.0, 1380.0, 47.7015, -122.164, 2520.0, 18295.0, 47.0, 0.0, 0.0]</td>
      <td>[706823.6]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-10-31 16:57:13.771</td>
      <td>[4.0, 3.25, 2910.0, 1880.0, 2.0, 0.0, 3.0, 5.0, 9.0, 1830.0, 1080.0, 47.616, -122.282, 3100.0, 8200.0, 100.0, 0.0, 0.0]</td>
      <td>[1060847.5]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-10-31 16:57:13.771</td>
      <td>[4.0, 1.75, 2700.0, 7875.0, 1.5, 0.0, 0.0, 4.0, 8.0, 2700.0, 0.0, 47.454, -122.144, 2220.0, 7875.0, 46.0, 0.0, 0.0]</td>
      <td>[441960.3]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-10-31 16:57:13.771</td>
      <td>[3.0, 2.5, 2900.0, 23550.0, 1.0, 0.0, 0.0, 3.0, 10.0, 1490.0, 1410.0, 47.5708, -122.153, 2900.0, 19604.0, 27.0, 0.0, 0.0]</td>
      <td>[827411.25]</td>
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
      <td>2023-10-31 16:57:13.771</td>
      <td>[2.0, 1.5, 1070.0, 1236.0, 2.0, 0.0, 0.0, 3.0, 8.0, 1000.0, 70.0, 47.5619, -122.382, 1170.0, 1888.0, 10.0, 0.0, 0.0]</td>
      <td>[435628.72]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>96</th>
      <td>2023-10-31 16:57:13.771</td>
      <td>[3.0, 2.5, 2830.0, 6000.0, 1.0, 0.0, 3.0, 3.0, 9.0, 1730.0, 1100.0, 47.5751, -122.378, 2040.0, 5300.0, 60.0, 0.0, 0.0]</td>
      <td>[981676.7]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>97</th>
      <td>2023-10-31 16:57:13.771</td>
      <td>[4.0, 1.75, 1720.0, 8750.0, 1.0, 0.0, 0.0, 3.0, 7.0, 860.0, 860.0, 47.726, -122.21, 1790.0, 8750.0, 43.0, 0.0, 0.0]</td>
      <td>[437177.97]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>98</th>
      <td>2023-10-31 16:57:13.771</td>
      <td>[4.0, 2.25, 4470.0, 60373.0, 2.0, 0.0, 0.0, 3.0, 11.0, 4470.0, 0.0, 47.7289, -122.127, 3210.0, 40450.0, 26.0, 0.0, 0.0]</td>
      <td>[1208638.1]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>99</th>
      <td>2023-10-31 16:57:13.771</td>
      <td>[3.0, 1.0, 1150.0, 3000.0, 1.0, 0.0, 0.0, 5.0, 6.0, 1150.0, 0.0, 47.6867, -122.345, 1460.0, 3200.0, 108.0, 0.0, 0.0]</td>
      <td>[448627.8]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 4 columns</p>

    Warning: Pipeline log size limit exceeded. Please request logs using export_logs

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
      <td>2023-10-31 16:57:13.771</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.7]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-10-31 16:57:13.771</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[615094.6]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-10-31 16:57:13.771</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
      <td>[448627.8]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-10-31 16:57:13.771</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[758714.3]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-10-31 16:57:13.771</td>
      <td>[3.0, 1.75, 2200.0, 11520.0, 1.0, 0.0, 0.0, 4.0, 7.0, 2200.0, 0.0, 47.7659, -122.341, 1690.0, 8038.0, 62.0, 0.0, 0.0]</td>
      <td>[513264.66]</td>
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
      <td>2023-10-31 16:57:13.771</td>
      <td>[4.0, 2.5, 3550.0, 35689.0, 2.0, 0.0, 0.0, 4.0, 9.0, 3550.0, 0.0, 47.7503, -122.074, 3350.0, 35711.0, 23.0, 0.0, 0.0]</td>
      <td>[873315.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>496</th>
      <td>2023-10-31 16:57:13.771</td>
      <td>[4.0, 2.5, 2510.0, 47044.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2510.0, 0.0, 47.7699, -122.085, 2600.0, 42612.0, 27.0, 0.0, 0.0]</td>
      <td>[721143.6]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>497</th>
      <td>2023-10-31 16:57:13.771</td>
      <td>[4.0, 2.5, 4090.0, 11225.0, 2.0, 0.0, 0.0, 3.0, 10.0, 4090.0, 0.0, 47.581, -121.971, 3510.0, 8762.0, 9.0, 0.0, 0.0]</td>
      <td>[1048372.44]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>498</th>
      <td>2023-10-31 16:57:13.771</td>
      <td>[2.0, 1.0, 720.0, 5000.0, 1.0, 0.0, 0.0, 5.0, 6.0, 720.0, 0.0, 47.5195, -122.374, 810.0, 5000.0, 63.0, 0.0, 0.0]</td>
      <td>[244566.39]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>499</th>
      <td>2023-10-31 16:57:13.771</td>
      <td>[4.0, 2.75, 2930.0, 22000.0, 1.0, 0.0, 3.0, 4.0, 9.0, 1580.0, 1350.0, 47.3227, -122.384, 2930.0, 9758.0, 36.0, 0.0, 0.0]</td>
      <td>[518869.03]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 4 columns</p>

    Warning: There are more logs available. Please set a larger limit or request a file using export_logs.

    pyarrow.Table
    time: timestamp[ms]
    in.tensor: list<item: float> not null
      child 0, item: float
    out.variable: list<inner: float not null> not null
      child 0, inner: float not null
    check_failures: int8
    ----
    time: [[2023-10-31 16:57:13.771,2023-10-31 16:57:13.771,2023-10-31 16:57:13.771,2023-10-31 16:57:13.771,2023-10-31 16:57:13.771,...,2023-10-31 16:57:13.771,2023-10-31 16:57:13.771,2023-10-31 16:57:13.771,2023-10-31 16:57:13.771,2023-10-31 16:57:13.771]]
    in.tensor: [[[3,2,2005,7000,1,...,1750,4500,34,0,0],[3,1.75,2910,37461,1,...,2520,18295,47,0,0],...,[4,2.25,4470,60373,2,...,3210,40450,26,0,0],[3,1,1150,3000,1,...,1460,3200,108,0,0]]]
    out.variable: [[[581002.94],[706823.6],...,[1208638.1],[448627.8]]]
    check_failures: [[0,0,0,0,0,...,0,0,0,0,0]]

## Anomaly Detection through Validations

Anomaly detection allows organizations to set validation parameters in a pipeline. A validation is added to a pipeline to test data based on an expression, and flag any inferences where the validation failed inference result and the pipeline logs.

Validations are added through the Pipeline `add_validation(name, validation)` command which uses the following parameters:

| Parameter | Type | Description |
|---|---|---|
| name | String (**Required**) | The name of the validation. |
| Validation | [Expression](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/checks/#Expression) (**Required**) | The validation test command in the format `model_name.outputs][field][index] {Operation} {Value}`. |

For this example, we want to detect the outputs of `housing_model_control` and validate that values are less than `1,500,000`.  Any outputs greater than that will trigger a `check_failure` which is shown in the output.

```python
## Add the validation to the pipeline

mainpipeline = mainpipeline.add_validation('price too high', housing_model_control.outputs[0][0] < 1500000.0)

#minimum deployment config
deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(0.5).memory("1Gi").build()
mainpipeline.deploy(deployment_config = deploy_config)
```

     ok

<table><tr><th>name</th> <td>housepricesagapipeline</td></tr><tr><th>created</th> <td>2023-10-31 16:56:07.345831+00:00</td></tr><tr><th>last_updated</th> <td>2023-10-31 16:57:24.026392+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>f7e0deaf-63a2-491a-8ff9-e8148d3cabcb, 2ea0cc42-c955-4fc3-bac9-7a2c7c22ddc1, dca84ab2-274a-4391-95dd-a99bda7621e1</td></tr><tr><th>steps</th> <td>housepricesagacontrol</td></tr><tr><th>published</th> <td>False</td></tr></table>

### Validation Testing

Two validations will be tested:

* One that should return a house value lower than 1,500,000.  The validation will pass so `check_failure` will be 0.
* The other than should return a house value greater than 1,500,000.  The validation will fail, so `check_failure` will be 1.

```python
validation_start = datetime.datetime.now()

# Small value home

normal_input = pd.DataFrame.from_records({
        "tensor": [[
            3.0,
            2.25,
            1620.0,
            997.0,
            2.5,
            0.0,
            0.0,
            3.0,
            8.0,
            1540.0,
            80.0,
            47.5400009155,
            -122.0260009766,
            1620.0,
            1068.0,
            4.0,
            0.0,
            0.0
        ]]
    }
)

small_result = mainpipeline.infer(normal_input)

display(small_result.loc[:,["time", "out.variable", "check_failures"]])
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out.variable</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-10-31 16:57:26.205</td>
      <td>[544392.06]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

```python
# Big value home

big_input = pd.DataFrame.from_records({
        "tensor": [[
            4.0,
            4.5,
            5770.0,
            10050.0,
            1.0,
            0.0,
            3.0,
            5.0,
            9.0,
            3160.0,
            2610.0,
            47.6769981384,
            -122.2750015259,
            2950.0,
            6700.0,
            65.0,
            0.0,
            0.0
        ]]
    }
)

big_result = mainpipeline.infer(big_input)

display(big_result.loc[:,["time", "out.variable", "check_failures"]])
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out.variable</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-10-31 16:57:27.138</td>
      <td>[1689843.1]</td>
      <td>1</td>
    </tr>
  </tbody>
</table>

### Anomaly Results

We'll run through our previous batch, this time showing only those results outside of the validation, and a graph showing where the anomalies are against the other results.

```python
batch_inferences = mainpipeline.infer_from_file('./data/xtest-1k.arrow')

large_inference_result =  batch_inferences.to_pandas()
# Display only the anomalous results

display(large_inference_result[large_inference_result["check_failures"] > 0].loc[:,["time", "out.variable", "check_failures"]])
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out.variable</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>30</th>
      <td>2023-10-31 16:57:28.974</td>
      <td>[1514079.4]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>248</th>
      <td>2023-10-31 16:57:28.974</td>
      <td>[1967344.1]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>255</th>
      <td>2023-10-31 16:57:28.974</td>
      <td>[2002393.6]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>556</th>
      <td>2023-10-31 16:57:28.974</td>
      <td>[1886959.2]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>698</th>
      <td>2023-10-31 16:57:28.974</td>
      <td>[1689843.1]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>711</th>
      <td>2023-10-31 16:57:28.974</td>
      <td>[1946437.8]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>722</th>
      <td>2023-10-31 16:57:28.974</td>
      <td>[2005883.1]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>782</th>
      <td>2023-10-31 16:57:28.974</td>
      <td>[1910824.0]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>965</th>
      <td>2023-10-31 16:57:28.974</td>
      <td>[2016006.1]</td>
      <td>1</td>
    </tr>
  </tbody>
</table>

```python
import matplotlib.pyplot as plt
houseprices = pd.DataFrame({'sell_price': large_inference_result['out.variable'].apply(lambda x: x[0])})

houseprices.hist(column='sell_price', bins=75, grid=False, figsize=(12,8))
plt.axvline(x=1500000, color='red', ls='--')
_ = plt.title('Distribution of predicted home sales price')
```

    
{{<figure src="/images/2023.4.0/wallaroo-tutorials/testing-tutorials/house-price-model-saga_files/house-price-model-saga_30_0.png" width="800" label="png">}}
    

## Assays

Wallaroo assays provide a method for detecting input or model drift.  These can be triggered either when unexpected input is provided for the inference, or when the model needs to be retrained from changing environment conditions.

Wallaroo assays can track either an input field and its index, or an output field and its index.  For full details, see the [Wallaroo Assays Management Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-pipeline-management/wallaroo-pipeline-assays/).

For this example, we will:

* Perform sample inferences based on lower priced houses.
* Create an assay with the baseline set off those lower priced houses.
* Generate inferences spread across all house values, plus specific set of high priced houses to trigger the assay alert.
* Run an interactive assay to show the detection of values outside the established baseline.

### Assay Generation

To start the demonstration, we'll create a baseline of values from houses with small estimated prices and set that as our baseline. Assays are typically run on a 24 hours interval based on a 24 hour window of data, but we'll bypass that by setting our baseline time even shorter.

```python
small_houses_inputs = pd.read_json('./data/smallinputs.df.json')
baseline_size = 500

# Where the baseline data will start
baseline_start = datetime.datetime.now()

# These inputs will be random samples of small priced houses.  Around 30,000 is a good number
small_houses = small_houses_inputs.sample(baseline_size, replace=True).reset_index(drop=True)

small_results = mainpipeline.infer(small_houses)

# Set the baseline end

baseline_end = datetime.datetime.now()
```

```python
# turn the inference results into a numpy array for the baseline

# set the results to a non-array value
small_results_baseline_df = small_results.copy()
small_results_baseline_df['variable']=small_results['out.variable'].map(lambda x: x[0])

# get the numpy values
small_results_baseline = small_results_baseline_df['variable'].to_numpy()
```

```python
assay_baseline_from_numpy_name = "house price saga assay from numpy"

# assay builder by baseline
assay_builder_from_numpy = wl.build_assay(assay_name=assay_baseline_from_numpy_name, 
                               pipeline=mainpipeline, 
                               model_name=model_name_control, 
                               iopath="output variable 0", 
                               baseline_data = small_results_baseline)
```

```python
# set the width from the recent results
assay_builder_from_numpy.window_builder().add_width(minutes=1)
assay_config_from_numpy = assay_builder_from_numpy.build()
assay_analysis_from_numpy = assay_config_from_numpy.interactive_run()
```

```python
# get the histogram from the numpy baseline
assay_builder_from_numpy.baseline_histogram()
```

    
{{<figure src="/images/2023.4.0/wallaroo-tutorials/testing-tutorials/house-price-model-saga_files/house-price-model-saga_37_0.png" width="800" label="png">}}
    

```python
# show the baseline stats
assay_analysis_from_numpy[0].baseline_stats()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Baseline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>500</td>
    </tr>
    <tr>
      <th>min</th>
      <td>236238.67</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1489624.3</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>513762.85694</td>
    </tr>
    <tr>
      <th>median</th>
      <td>448627.8</td>
    </tr>
    <tr>
      <th>std</th>
      <td>235726.284713</td>
    </tr>
    <tr>
      <th>start</th>
      <td>None</td>
    </tr>
    <tr>
      <th>end</th>
      <td>None</td>
    </tr>
  </tbody>
</table>

Now we'll perform some inferences with a spread of values, then a larger set with a set of larger house values to trigger our assay alert.

Because our assay windows are 1 minutes, we'll need to stagger our inference values to be set into the proper windows.  This will take about 4 minutes.

```python
# Get a spread of house values

time.sleep(35)
# regular_houses_inputs = pd.read_json('./data/xtest-1k.df.json', orient="records")
inference_size = 1000

# regular_houses = regular_houses_inputs.sample(inference_size, replace=True).reset_index(drop=True)

# And a spread of large house values

big_houses_inputs = pd.read_json('./data/biginputs.df.json', orient="records")
big_houses = big_houses_inputs.sample(inference_size, replace=True).reset_index(drop=True)

# Set the start for our assay window period.
assay_window_start = datetime.datetime.now()

mainpipeline.infer(big_houses)

# End our assay window period
time.sleep(35)
assay_window_end = datetime.datetime.now()
```

```python
assay_builder_from_numpy.add_run_until(assay_window_end)
assay_builder_from_numpy.window_builder().add_width(minutes=1).add_interval(minutes=1)
assay_config_from_dates = assay_builder_from_numpy.build()
assay_analysis_from_numpy = assay_config_from_numpy.interactive_run()
```

```python
# Show how many assay windows were analyzed, then show the chart
print(f"Generated {len(assay_analysis_from_numpy)} analyses")
assay_analysis_from_numpy.chart_scores()
```

    Generated 5 analyses

    
{{<figure src="/images/2023.4.0/wallaroo-tutorials/testing-tutorials/house-price-model-saga_files/house-price-model-saga_42_1.png" width="800" label="png">}}
    

```python
# Display the results as a DataFrame - we're mainly interested in the score and whether the 
# alert threshold was triggered
display(assay_analysis_from_numpy.to_dataframe().loc[:, ["score", "start", "alert_threshold", "status"]])
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>score</th>
      <th>start</th>
      <th>alert_threshold</th>
      <th>status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.016079</td>
      <td>2023-10-31T16:56:21.399000+00:00</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.006042</td>
      <td>2023-10-31T16:57:21.399000+00:00</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.868832</td>
      <td>2023-10-31T16:58:21.399000+00:00</td>
      <td>0.25</td>
      <td>Alert</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.868832</td>
      <td>2023-10-31T16:59:21.399000+00:00</td>
      <td>0.25</td>
      <td>Alert</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8.868832</td>
      <td>2023-10-31T17:00:21.399000+00:00</td>
      <td>0.25</td>
      <td>Alert</td>
    </tr>
  </tbody>
</table>

```python
assay_builder_from_numpy.upload()
```

    18

The assay is now visible through the Wallaroo UI by selecting the workspace, then the pipeline, then **Insights**.

{{<figure src="/images/2023.4.0/housepricesaga-sample-assay.png" width="800" label="Sample assay in the UI">}}

## Shadow Deploy

Let's assume that after analyzing the assay information we want to test two challenger models to our control.  We do that with the Shadow Deploy pipeline step.

In Shadow Deploy, the pipeline step is added with the `add_shadow_deploy` method, with the champion model listed first, then an array of challenger models after.  **All** inference data is fed to **all** models, with the champion results displayed in the `out.variable` column, and the shadow results in the format `out_{model name}.variable`.  For example, since we named our challenger models `housingchallenger01` and `housingchallenger02`, the columns `out_housingchallenger01.variable` and `out_housingchallenger02.variable` have the shadow deployed model results.

For this example, we will remove the previous pipeline step, then replace it with a shadow deploy step with `rf_model.onnx` as our champion, and models `xgb_model.onnx` and `gbr_model.onnx` as the challengers.  We'll deploy the pipeline and prepare it for sample inferences.

```python
# Upload the challenger models

model_name_challenger01 = 'housingchallenger01'
model_file_name_challenger01 = './models/xgb_model.onnx'

model_name_challenger02 = 'housingchallenger02'
model_file_name_challenger02 = './models/gbr_model.onnx'

housing_model_challenger01 = (wl.upload_model(model_name_challenger01, 
                                              model_file_name_challenger01, 
                                              framework=Framework.ONNX)
                                              .configure(tensor_fields=["tensor"])
                            )
housing_model_challenger02 = (wl.upload_model(model_name_challenger02, 
                                              model_file_name_challenger02, 
                                              framework=Framework.ONNX)
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
#minimum deployment config
deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(0.5).memory("1Gi").build()
mainpipeline.deploy(deployment_config = deploy_config)
```

    Waiting for undeployment - this will take up to 45s ................................... ok
    Waiting for deployment - this will take up to 45s ................................. ok

<table><tr><th>name</th> <td>housepricesagapipeline</td></tr><tr><th>created</th> <td>2023-10-31 16:56:07.345831+00:00</td></tr><tr><th>last_updated</th> <td>2023-10-31 17:04:57.709641+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>44d28e3e-094f-4507-8aa1-909c1f151dd5, b9f46ba3-3f14-4e5b-989e-e0b8d166392f, f7e0deaf-63a2-491a-8ff9-e8148d3cabcb, 2ea0cc42-c955-4fc3-bac9-7a2c7c22ddc1, dca84ab2-274a-4391-95dd-a99bda7621e1</td></tr><tr><th>steps</th> <td>housepricesagacontrol</td></tr><tr><th>published</th> <td>False</td></tr></table>

### Shadow Deploy Sample Inference

We'll now use our same sample data for an inference to our shadow deployed pipeline, then display the first 20 results with just the comparative outputs.

```python
shadow_result = mainpipeline.infer_from_file('./data/xtest-1k.arrow')

shadow_outputs =  shadow_result.to_pandas()
display(shadow_outputs.loc[0:20,['out.variable','out_housingchallenger01.variable','out_housingchallenger02.variable']])
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out.variable</th>
      <th>out_housingchallenger01.variable</th>
      <th>out_housingchallenger02.variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[718013.7]</td>
      <td>[659806.0]</td>
      <td>[704901.9]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[615094.6]</td>
      <td>[732883.5]</td>
      <td>[695994.44]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[448627.8]</td>
      <td>[419508.84]</td>
      <td>[416164.8]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[758714.3]</td>
      <td>[634028.75]</td>
      <td>[655277.2]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[513264.66]</td>
      <td>[427209.47]</td>
      <td>[426854.66]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>[668287.94]</td>
      <td>[615501.9]</td>
      <td>[632556.06]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>[1004846.56]</td>
      <td>[1139732.4]</td>
      <td>[1100465.2]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>[684577.25]</td>
      <td>[498328.88]</td>
      <td>[528278.06]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>[727898.25]</td>
      <td>[722664.4]</td>
      <td>[659439.94]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>[559631.06]</td>
      <td>[525746.44]</td>
      <td>[534331.44]</td>
    </tr>
    <tr>
      <th>10</th>
      <td>[340764.53]</td>
      <td>[376337.06]</td>
      <td>[377187.2]</td>
    </tr>
    <tr>
      <th>11</th>
      <td>[442168.12]</td>
      <td>[382053.12]</td>
      <td>[403964.3]</td>
    </tr>
    <tr>
      <th>12</th>
      <td>[630865.5]</td>
      <td>[505608.97]</td>
      <td>[528991.3]</td>
    </tr>
    <tr>
      <th>13</th>
      <td>[559631.06]</td>
      <td>[603260.5]</td>
      <td>[612201.75]</td>
    </tr>
    <tr>
      <th>14</th>
      <td>[909441.25]</td>
      <td>[969585.44]</td>
      <td>[893874.7]</td>
    </tr>
    <tr>
      <th>15</th>
      <td>[313096.0]</td>
      <td>[313633.7]</td>
      <td>[318054.94]</td>
    </tr>
    <tr>
      <th>16</th>
      <td>[404040.78]</td>
      <td>[360413.62]</td>
      <td>[357816.7]</td>
    </tr>
    <tr>
      <th>17</th>
      <td>[292859.44]</td>
      <td>[316674.88]</td>
      <td>[294034.62]</td>
    </tr>
    <tr>
      <th>18</th>
      <td>[338357.88]</td>
      <td>[299907.47]</td>
      <td>[323254.28]</td>
    </tr>
    <tr>
      <th>19</th>
      <td>[682284.56]</td>
      <td>[811896.75]</td>
      <td>[770916.6]</td>
    </tr>
    <tr>
      <th>20</th>
      <td>[583765.94]</td>
      <td>[573618.5]</td>
      <td>[549141.4]</td>
    </tr>
  </tbody>
</table>

## A/B Testing

A/B Testing is another method of comparing and testing models.  Like shadow deploy, multiple models are compared against the champion or control models.  The difference is that instead of submitting the inference data to all models, then tracking the outputs of all of the models, the inference inputs are off of a ratio and other conditions.

For this example, we'll be using a 1:1:1 ratio with a random split between the champion model and the two challenger models.  Each time an inference request is made, there is a random equal chance of any one of them being selected.

When the inference results and log entries are displayed, they include the column `out._model_split` which displays:

| Field | Type | Description |
|---|---|---|
| `name` | String | The model name used for the inference.  |
| `version` | String| The version of the model. |
| `sha` | String | The sha hash of the model version. |

This is used to determine which model was used for the inference request.

```python
# remove the shadow deploy steps
mainpipeline.clear()

# Add the a/b test step to the pipeline
mainpipeline.add_random_split([(1, housing_model_control), (1, housing_model_challenger01), (1, housing_model_challenger02)], "session_id")

mainpipeline.deploy()

# Perform sample inferences of 20 rows and display the results
ab_date_start = datetime.datetime.now()
abtesting_inputs = pd.read_json('./data/xtest-1k.df.json')

df = pd.DataFrame(columns=["model", "value"])

for index, row in abtesting_inputs.sample(20).iterrows():
    result = mainpipeline.infer(row.to_frame('tensor').reset_index())
    value = result.loc[0]["out.variable"]
    model = json.loads(result.loc[0]["out._model_split"][0])['name']
    df = df.append({'model': model, 'value': value}, ignore_index=True)

display(df)
ab_date_end = datetime.datetime.now()
```

     ok

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>housingchallenger01</td>
      <td>[278554.44]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>housingchallenger02</td>
      <td>[615955.3]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>housepricesagacontrol</td>
      <td>[1092273.9]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>housepricesagacontrol</td>
      <td>[683845.75]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>housepricesagacontrol</td>
      <td>[682284.56]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>housepricesagacontrol</td>
      <td>[247792.75]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>housingchallenger02</td>
      <td>[315142.44]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>housingchallenger02</td>
      <td>[530408.94]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>housepricesagacontrol</td>
      <td>[340764.53]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>housepricesagacontrol</td>
      <td>[421153.16]</td>
    </tr>
    <tr>
      <th>10</th>
      <td>housingchallenger01</td>
      <td>[395150.63]</td>
    </tr>
    <tr>
      <th>11</th>
      <td>housingchallenger02</td>
      <td>[544343.3]</td>
    </tr>
    <tr>
      <th>12</th>
      <td>housingchallenger01</td>
      <td>[395284.4]</td>
    </tr>
    <tr>
      <th>13</th>
      <td>housepricesagacontrol</td>
      <td>[701940.7]</td>
    </tr>
    <tr>
      <th>14</th>
      <td>housepricesagacontrol</td>
      <td>[448627.8]</td>
    </tr>
    <tr>
      <th>15</th>
      <td>housepricesagacontrol</td>
      <td>[320863.72]</td>
    </tr>
    <tr>
      <th>16</th>
      <td>housingchallenger01</td>
      <td>[558485.3]</td>
    </tr>
    <tr>
      <th>17</th>
      <td>housingchallenger02</td>
      <td>[236329.28]</td>
    </tr>
    <tr>
      <th>18</th>
      <td>housepricesagacontrol</td>
      <td>[559631.06]</td>
    </tr>
    <tr>
      <th>19</th>
      <td>housingchallenger01</td>
      <td>[281437.56]</td>
    </tr>
  </tbody>
</table>

## Model Swap

Now that we've completed our testing, we can swap our deployed model in the original `housepricingpipeline` with one we feel works better.

We'll start by removing the A/B Testing pipeline step, then going back to the single pipeline step with the champion model and perform a test inference.

When going from a testing step such as A/B Testing or Shadow Deploy, it is best to undeploy the pipeline, change the steps, then deploy the pipeline.  In a production environment, there should be two pipelines:  One for production, the other for testing models.  Since this example uses one pipeline for simplicity, we will undeploy our main pipeline and reset it back to a one-step pipeline with the current champion model as our pipeline step.

Once done, we'll perform the hot swap with the model `gbr_model.onnx`, which was labeled `housing_model_challenger02` in a previous step.  We'll do an inference with the same data as used with the challenger model.  Note that previously, the inference through the original model returned `[718013.7]`.

```python
mainpipeline.undeploy()

# remove the shadow deploy steps
mainpipeline.clear()

mainpipeline.add_model_step(housing_model_control).deploy()

# Inference test
normal_input = pd.DataFrame.from_records({"tensor": [[4.0,
            2.25,
            2200.0,
            11250.0,
            1.5,
            0.0,
            0.0,
            5.0,
            7.0,
            1300.0,
            900.0,
            47.6845,
            -122.201,
            2320.0,
            10814.0,
            94.0,
            0.0,
            0.0]]})
controlresult = mainpipeline.infer(normal_input)
display(controlresult)
```

     ok
    Waiting for deployment - this will take up to 45s ........ ok

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
      <td>2023-10-31 17:08:19.579</td>
      <td>[4.0, 2.25, 2200.0, 11250.0, 1.5, 0.0, 0.0, 5.0, 7.0, 1300.0, 900.0, 47.6845, -122.201, 2320.0, 10814.0, 94.0, 0.0, 0.0]</td>
      <td>[682284.56]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

Now we'll "hot swap" the control model.  We don't have to deploy the pipeline - we can just swap the model out in that pipeline step and continue with only a millisecond or two lost while the swap was performed.

```python
# Perform hot swap

mainpipeline.replace_with_model_step(0, housing_model_challenger02).deploy()
# wait a moment for the database to be updated.  The swap is near instantaneous but database writes may take a moment
import time
time.sleep(15)

# inference after model swap
normal_input = pd.DataFrame.from_records({"tensor": [[4.0,
            2.25,
            2200.0,
            11250.0,
            1.5,
            0.0,
            0.0,
            5.0,
            7.0,
            1300.0,
            900.0,
            47.6845,
            -122.201,
            2320.0,
            10814.0,
            94.0,
            0.0,
            0.0]]})
challengerresult = mainpipeline.infer(normal_input)
display(challengerresult)
```

     ok

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
      <td>2023-10-31 17:09:23.932</td>
      <td>[4.0, 2.25, 2200.0, 11250.0, 1.5, 0.0, 0.0, 5.0, 7.0, 1300.0, 900.0, 47.6845, -122.201, 2320.0, 10814.0, 94.0, 0.0, 0.0]</td>
      <td>[770916.6]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

```python
# Display the difference between the two

display(f'Original model output: {controlresult.loc[0]["out.variable"]}')
display(f'Hot swapped model  output: {challengerresult.loc[0]["out.variable"]}')
```

    'Original model output: [682284.56]'

    'Hot swapped model  output: [770916.6]'

### Undeploy Main Pipeline

With the examples and tutorial complete, we will undeploy the main pipeline and return the resources back to the Wallaroo instance.

```python
mainpipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s ...................................... ok

<table><tr><th>name</th> <td>housepricesagapipeline</td></tr><tr><th>created</th> <td>2023-10-31 16:56:07.345831+00:00</td></tr><tr><th>last_updated</th> <td>2023-10-31 17:09:08.758964+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>51034ba5-b58e-4475-908d-ec8fae069745, cd0f20e9-6923-4e6f-8114-1137676da5c5, 58274aab-0045-4557-a740-7d085af8574d, 4cf4f0dc-2c42-4f8b-975c-1f5c5d939a98, c5ae5a56-1c7f-4a94-9d4e-f08bf35e7f4f, f652a9d8-a4f6-428a-9b36-b29eec6b5198, 0e7d9d77-02d7-4c13-8062-9c77492145f8, 42b97784-fc16-43be-b89e-2d3dad20dd2b, 7d35285a-f8af-4bed-855b-60258c3435ee, 8b0dc4bb-3340-44f9-85d6-fbd091e19412, 06169fed-59e9-41f8-9dbd-0fc9f80ebb73, 44d28e3e-094f-4507-8aa1-909c1f151dd5, b9f46ba3-3f14-4e5b-989e-e0b8d166392f, f7e0deaf-63a2-491a-8ff9-e8148d3cabcb, 2ea0cc42-c955-4fc3-bac9-7a2c7c22ddc1, dca84ab2-274a-4391-95dd-a99bda7621e1</td></tr><tr><th>steps</th> <td>housepricesagacontrol</td></tr><tr><th>published</th> <td>False</td></tr></table>

