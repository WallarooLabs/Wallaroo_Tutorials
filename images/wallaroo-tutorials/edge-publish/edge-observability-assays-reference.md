This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-testing-tutorials/anomaly_detection).

## House Price Testing Life Cycle Preparation

This notebook is used in preparation for the House Price Life Cycle tutorial.  This includes:

* Setting up a workspace, pipeline, and model for deriving the price of a house based on inputs.
* Creating an assay from a sample of inferences.
* Display the inference result and upload the assay to the Wallaroo instance where it can be referenced later.

This preparation is used for the House Price Life Cycle Comprehensive and Short tutorials, included in this folder.

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

## Preparation

It is recommended that this preparation is run at least an hour or so before a demonstration of the House Price Saga Comprehensive and Short tutorials.

This will require that the same workspace, pipeline, model name, and assay name are used from this preparation notebook to the tutorial.  Those variables are stored directly below.  

```python
# used for unique connection names

import string
import random

suffix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))
suffix=''

# used to make a unique workspace
suffix='baseline-examples'

workspace_name = f'edge-observability-assays{suffix}'
main_pipeline_name = f'housepricesagapipeline'
model_name_control = f'housepricesagacontrol'
model_file_name_control = './models/rf_model.onnx'

# Set the name of the assay
assay_name=f"house price test{suffix}"
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
workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)
```

    {'name': 'edge-observability-assaysbaseline-examples', 'id': 89, 'archived': False, 'created_by': 'd6a42dd8-1da9-4405-bb80-7c4b42e38b52', 'created_at': '2023-10-31T18:15:18.554323+00:00', 'models': [], 'pipelines': []}

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
mainpipeline.clear()

mainpipeline.add_model_step(housing_model_control)

#minimum deployment config
deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(0.5).memory("1Gi").build()

mainpipeline.deploy(deployment_config = deploy_config)
```

    Waiting for deployment - this will take up to 45s ................................. ok

<table><tr><th>name</th> <td>housepricesagapipeline</td></tr><tr><th>created</th> <td>2023-10-31 18:15:22.006805+00:00</td></tr><tr><th>last_updated</th> <td>2023-10-31 18:15:22.061320+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>e96d2fde-84a2-44a0-aa1e-eaa3018da51a, 515bb61b-545b-4816-8e93-ab376472cd10</td></tr><tr><th>steps</th> <td>housepricesagacontrol</td></tr><tr><th>published</th> <td>False</td></tr></table>

### Testing

We'll use two inferences as a quick sample test - one that has a house that should be determined around `$700k`, the other with a house determined to be around `$1.5` million.  We'll also save the start and end periods for these events to for later log functionality.

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
      <td>2023-10-31 18:15:57.159</td>
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
      <td>2023-10-31 18:15:57.195</td>
      <td>[4.0, 3.0, 3710.0, 20000.0, 2.0, 0.0, 2.0, 5.0, 10.0, 2760.0, 950.0, 47.6696, -122.261, 3970.0, 20000.0, 79.0, 0.0, 0.0]</td>
      <td>[1514079.4]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

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
assay_baseline_start = datetime.datetime.now()

# These inputs will be random samples of small priced houses.
small_houses = small_houses_inputs.sample(baseline_size, replace=True).reset_index(drop=True)

# Wait 30 seconds to set this data apart from the rest
time.sleep(30)
small_results = mainpipeline.infer(small_houses)

# Set the baseline end

assay_baseline_end = datetime.datetime.now()
```

```python
display(small_results)
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
      <td>2023-10-31 18:16:27.272</td>
      <td>[3.0, 2.5, 2360.0, 4080.0, 2.0, 0.0, 0.0, 3.0, 7.0, 2360.0, 0.0, 47.6824989319, -122.0380020142, 2290.0, 4080.0, 11.0, 0.0, 0.0]</td>
      <td>[701940.7]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-10-31 18:16:27.272</td>
      <td>[2.0, 1.0, 920.0, 43560.0, 1.0, 0.0, 0.0, 4.0, 5.0, 920.0, 0.0, 47.5245018005, -121.9309997559, 1530.0, 11875.0, 91.0, 0.0, 0.0]</td>
      <td>[243300.83]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-10-31 18:16:27.272</td>
      <td>[4.0, 2.5, 2632.0, 4117.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2632.0, 0.0, 47.3428001404, -122.2779998779, 2040.0, 5195.0, 1.0, 0.0, 0.0]</td>
      <td>[368504.3]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-10-31 18:16:27.272</td>
      <td>[4.0, 2.25, 3310.0, 8540.0, 1.0, 0.0, 4.0, 4.0, 9.0, 1660.0, 1650.0, 47.5602989197, -122.1579971313, 3450.0, 9566.0, 41.0, 0.0, 0.0]</td>
      <td>[921561.56]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-10-31 18:16:27.272</td>
      <td>[4.0, 2.5, 3140.0, 12792.0, 2.0, 0.0, 0.0, 4.0, 9.0, 3140.0, 0.0, 47.3862991333, -122.15599823, 2510.0, 12792.0, 37.0, 0.0, 0.0]</td>
      <td>[513583.06]</td>
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
      <td>2023-10-31 18:16:27.272</td>
      <td>[4.0, 2.25, 2580.0, 7344.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2580.0, 0.0, 47.5647010803, -122.0899963379, 2390.0, 7507.0, 37.0, 0.0, 0.0]</td>
      <td>[701940.7]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>496</th>
      <td>2023-10-31 18:16:27.272</td>
      <td>[3.0, 1.75, 1960.0, 8136.0, 1.0, 0.0, 0.0, 3.0, 7.0, 980.0, 980.0, 47.5208015442, -122.3639984131, 1070.0, 7480.0, 66.0, 0.0, 0.0]</td>
      <td>[365436.22]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>497</th>
      <td>2023-10-31 18:16:27.272</td>
      <td>[4.0, 2.5, 2230.0, 4372.0, 2.0, 0.0, 0.0, 5.0, 8.0, 1540.0, 690.0, 47.6697998047, -122.3339996338, 2020.0, 4372.0, 79.0, 0.0, 0.0]</td>
      <td>[682284.56]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>498</th>
      <td>2023-10-31 18:16:27.272</td>
      <td>[3.0, 2.75, 2340.0, 16500.0, 1.0, 0.0, 0.0, 4.0, 8.0, 1500.0, 840.0, 47.595199585, -122.0510025024, 2210.0, 15251.0, 42.0, 0.0, 0.0]</td>
      <td>[687786.44]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>499</th>
      <td>2023-10-31 18:16:27.272</td>
      <td>[3.0, 2.25, 2960.0, 8330.0, 1.0, 0.0, 3.0, 4.0, 10.0, 2260.0, 700.0, 47.7034988403, -122.3850021362, 2960.0, 8840.0, 62.0, 0.0, 0.0]</td>
      <td>[1178313.9]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 4 columns</p>

```python
# get the numpy values

# set the results to a non-array value
small_results_baseline_df = small_results.copy()
small_results_baseline_df['variable']=small_results['out.variable'].map(lambda x: x[0])
small_results_baseline_df
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.variable</th>
      <th>check_failures</th>
      <th>variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-10-31 18:16:27.272</td>
      <td>[3.0, 2.5, 2360.0, 4080.0, 2.0, 0.0, 0.0, 3.0, 7.0, 2360.0, 0.0, 47.6824989319, -122.0380020142, 2290.0, 4080.0, 11.0, 0.0, 0.0]</td>
      <td>[701940.7]</td>
      <td>0</td>
      <td>701940.70</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-10-31 18:16:27.272</td>
      <td>[2.0, 1.0, 920.0, 43560.0, 1.0, 0.0, 0.0, 4.0, 5.0, 920.0, 0.0, 47.5245018005, -121.9309997559, 1530.0, 11875.0, 91.0, 0.0, 0.0]</td>
      <td>[243300.83]</td>
      <td>0</td>
      <td>243300.83</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-10-31 18:16:27.272</td>
      <td>[4.0, 2.5, 2632.0, 4117.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2632.0, 0.0, 47.3428001404, -122.2779998779, 2040.0, 5195.0, 1.0, 0.0, 0.0]</td>
      <td>[368504.3]</td>
      <td>0</td>
      <td>368504.30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-10-31 18:16:27.272</td>
      <td>[4.0, 2.25, 3310.0, 8540.0, 1.0, 0.0, 4.0, 4.0, 9.0, 1660.0, 1650.0, 47.5602989197, -122.1579971313, 3450.0, 9566.0, 41.0, 0.0, 0.0]</td>
      <td>[921561.56]</td>
      <td>0</td>
      <td>921561.56</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-10-31 18:16:27.272</td>
      <td>[4.0, 2.5, 3140.0, 12792.0, 2.0, 0.0, 0.0, 4.0, 9.0, 3140.0, 0.0, 47.3862991333, -122.15599823, 2510.0, 12792.0, 37.0, 0.0, 0.0]</td>
      <td>[513583.06]</td>
      <td>0</td>
      <td>513583.06</td>
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
      <th>495</th>
      <td>2023-10-31 18:16:27.272</td>
      <td>[4.0, 2.25, 2580.0, 7344.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2580.0, 0.0, 47.5647010803, -122.0899963379, 2390.0, 7507.0, 37.0, 0.0, 0.0]</td>
      <td>[701940.7]</td>
      <td>0</td>
      <td>701940.70</td>
    </tr>
    <tr>
      <th>496</th>
      <td>2023-10-31 18:16:27.272</td>
      <td>[3.0, 1.75, 1960.0, 8136.0, 1.0, 0.0, 0.0, 3.0, 7.0, 980.0, 980.0, 47.5208015442, -122.3639984131, 1070.0, 7480.0, 66.0, 0.0, 0.0]</td>
      <td>[365436.22]</td>
      <td>0</td>
      <td>365436.22</td>
    </tr>
    <tr>
      <th>497</th>
      <td>2023-10-31 18:16:27.272</td>
      <td>[4.0, 2.5, 2230.0, 4372.0, 2.0, 0.0, 0.0, 5.0, 8.0, 1540.0, 690.0, 47.6697998047, -122.3339996338, 2020.0, 4372.0, 79.0, 0.0, 0.0]</td>
      <td>[682284.56]</td>
      <td>0</td>
      <td>682284.56</td>
    </tr>
    <tr>
      <th>498</th>
      <td>2023-10-31 18:16:27.272</td>
      <td>[3.0, 2.75, 2340.0, 16500.0, 1.0, 0.0, 0.0, 4.0, 8.0, 1500.0, 840.0, 47.595199585, -122.0510025024, 2210.0, 15251.0, 42.0, 0.0, 0.0]</td>
      <td>[687786.44]</td>
      <td>0</td>
      <td>687786.44</td>
    </tr>
    <tr>
      <th>499</th>
      <td>2023-10-31 18:16:27.272</td>
      <td>[3.0, 2.25, 2960.0, 8330.0, 1.0, 0.0, 3.0, 4.0, 10.0, 2260.0, 700.0, 47.7034988403, -122.3850021362, 2960.0, 8840.0, 62.0, 0.0, 0.0]</td>
      <td>[1178313.9]</td>
      <td>0</td>
      <td>1178313.90</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 5 columns</p>

```python
# get the numpy values
small_results_baseline = small_results_baseline_df['variable'].to_numpy()
small_results_baseline
```

    array([ 701940.7 ,  243300.83,  368504.3 ,  921561.56,  513583.06,
           1004846.56,  287576.4 ,  291903.97,  283759.94,  291239.75,
            448627.8 ,  448627.8 ,  793214.3 ,  559631.06,  437177.97,
            320395.38,  340764.53,  365436.22,  480151.44,  557391.25,
            530288.94,  480151.44,  375012.  ,  320863.72,  886958.6 ,
            413013.38, 1115275.  ,  713485.7 ,  365436.22, 1100884.3 ,
           1160512.8 ,  444885.6 ,  294203.53,  473287.25,  523576.2 ,
            391459.97,  444408.  ,  795841.06,  283759.94, 1227073.8 ,
            284336.47,  383833.88,  349102.75,  444408.  ,  448627.8 ,
            684577.25,  712309.9 ,  281058.75,  341649.34,  437177.97,
            340764.53,  349102.75,  736751.3 ,  340764.53,  448627.8 ,
            536388.2 , 1227073.8 ,  423382.72,  827411.25,  261886.94,
           1325960.8 ,  300988.8 ,  713979.  ,  236238.67,  448627.8 ,
           1189654.5 ,  682284.56,  921695.4 ,  559631.06,  937359.6 ,
            718445.7 ,  498579.5 , 1189654.5 , 1322835.6 ,  784103.56,
            703282.7 ,  784103.56,  559923.9 ,  236238.67,  675545.44,
            404676.1 ,  437177.97,  411000.13,  498579.5 ,  718445.7 ,
            448627.8 ,  559139.4 ,  450867.7 ,  557391.25,  244351.92,
            444408.  ,  727923.2 ,  446769.  ,  332134.97,  276709.06,
            424966.6 , 1322835.6 ,  354512.44,  657905.75,  450867.7 ,
           1052897.9 ,  544392.06,  437177.97,  837085.5 ,  380009.28,
            448627.8 ,  464057.38,  340764.53,  294203.53,  475971.5 ,
            827411.25,  557391.25,  324875.06,  391459.97,  555231.94,
            448627.8 ,  546631.94,  712309.9 ,  340764.53,  675545.44,
            683869.56,  879092.9 ,  340764.53,  721143.6 ,  437177.97,
            723934.9 ,  263051.63,  244566.39,  727923.2 ,  281823.13,
            437929.84,  701940.7 ,  448627.8 , 1060847.5 ,  236238.67,
            332134.97,  287576.4 ,  448627.8 ,  718445.7 ,  435628.72,
            266405.63,  725184.1 ,  320395.38,  559631.06,  523576.2 ,
            680620.7 ,  328513.6 ,  937359.6 ,  448627.8 ,  324875.06,
            829775.3 ,  630865.5 ,  557391.25, 1085835.4 ,  758714.3 ,
            682284.56,  400561.2 , 1227073.8 ,  498579.5 ,  403520.16,
            706407.4 ,  467484.3 ,  442168.13,  536175.7 ,  559631.06,
            450867.7 ,  448180.78,  779810.06,  448627.8 ,  249227.83,
            450867.7 ,  657905.75,  447162.84,  539867.1 ,  448627.8 ,
            404676.1 ,  705013.5 ,  544392.06,  758714.3 ,  400561.2 ,
            448627.8 ,  684577.25,  450867.7 ,  438346.38,  765468.9 ,
            320863.72,  559631.06,  827411.25,  675545.44,  559452.94,
            384558.4 ,  421402.1 ,  244380.27,  438346.38,  675545.44,
            341472.13,  684577.25,  682284.56,  544392.06,  341649.34,
            536371.2 ,  435628.72,  449229.97,  536371.2 ,  437177.97,
            480151.44,  431929.2 ,  559923.9 ,  320863.72,  758714.3 ,
            713485.7 ,  358668.2 ,  467484.3 ,  736751.3 ,  400561.2 ,
            288798.1 ,  358668.2 ,  267013.97,  453195.8 ,  267013.97,
            266405.63,  318011.4 ,  718445.7 ,  536371.2 ,  550275.1 ,
            704672.25,  411000.13,  559631.06,  551223.44,  306037.63,
            442168.13, 1208638.1 ,  687786.44,  450867.7 ,  348616.63,
           1295531.8 ,  340764.53,  261886.94,  450996.34,  784103.56,
            258377.  ,  784103.56,  682284.56,  359947.78,  495822.63,
            513264.66,  404676.1 ,  430252.3 ,  291903.97,  559631.06,
            244380.27,  886958.6 ,  450996.34,  630865.5 ,  243063.13,
            557391.25,  519346.94,  276709.06,  437177.97,  340764.53,
            340764.53,  340764.53,  424966.6 ,  538436.75,  324875.06,
            450867.7 ,  450928.94,  711565.44,  684577.25,  403520.16,
            403520.16,  701940.7 ,  444931.28,  236238.67,  437177.97,
            437929.84,  296202.7 ,  557391.25,  450867.7 ,  290987.34,
            831483.56,  532234.1 ,  758714.3 ,  434534.22,  411000.13,
            551223.44,  244380.27,  355371.1 ,  682284.56,  333878.22,
            435628.72,  950176.7 ,  442856.4 ,  513264.66,  758714.3 ,
            308049.63,  400561.2 ,  725184.1 ,  713485.7 ,  718445.7 ,
            274207.16,  723867.7 ,  718445.7 ,  283759.94,  437177.97,
            363491.63,  458858.44,  728707.7 ,  437177.97,  437177.97,
            413013.38,  431929.2 ,  559631.06,  303002.25,  379076.28,
            713485.7 ,  701940.7 ,  285253.13,  450867.7 ,  987157.25,
            725184.1 ,  437177.97,  682284.56,  450867.7 ,  241657.14,
            920795.94,  444931.28,  921695.4 , 1060847.5 ,  450867.7 ,
            718013.7 ,  716776.56,  701940.7 ,  393833.97,  559631.06,
            461279.1 ,  334257.7 ,  458858.44,  283759.94,  557391.25,
            559631.06,  686890.94,  723867.7 ,  350049.3 ,  448627.8 ,
            937359.6 ,  711565.44, 1082353.1 ,  243300.83,  713003.4 ,
            288798.1 ,  435628.72,  480151.44,  363491.63,  313096.  ,
            292859.44,  448627.8 ,  444408.  ,  368504.3 ,  431929.2 ,
            437177.97,  529302.44,  323856.28,  538316.4 ,  448627.8 ,
            721143.6 ,  404040.78,  879083.56,  673288.6 , 1004846.56,
            684577.25,  243063.13,  450867.7 ,  555231.94,  276709.06,
            725184.1 ,  559452.94,  529302.44,  684577.25,  384558.4 ,
            431992.22,  276709.06,  595497.3 ,  836230.2 ,  937359.6 ,
            236238.67,  964052.6 ,  246901.14,  946325.75,  730767.94,
            438346.38, 1364149.9 ,  340764.53,  296411.7 ,  701940.7 ,
            340764.53,  879083.56,  424966.6 ,  713979.  ,  713485.7 ,
            287576.4 ,  340764.53,  559631.06,  718445.7 ,  236238.67,
            320863.72,  921561.56,  559631.06,  328513.6 ,  437177.97,
            434534.22,  438514.28,  421402.1 ,  557391.25,  243585.28,
            241852.33, 1178313.9 ,  498579.5 ,  413013.38,  544392.06,
            252192.9 , 1489624.3 ,  448627.8 ,  437177.97,  637377.  ,
            243300.83,  267013.97, 1322835.6 ,  236238.67,  701940.7 ,
            435628.72,  450867.7 ,  258321.66,  380009.28,  238078.03,
           1060847.5 ,  340764.53,  320863.72,  453195.8 ,  400561.2 ,
            239734.08,  557391.25,  239734.08,  559452.94,  725184.1 ,
            684577.25,  508746.63,  289684.22,  379076.28,  340764.53,
            242591.38,  447192.22,  559452.94,  431992.22,  700294.25,
            795841.06,  555231.94,  238078.03,  444933.25,  354512.44,
            317765.63,  450867.7 ,  350835.9 ,  723934.9 ,  340764.53,
            292859.44,  713979.  ,  342604.47,  544392.06,  413013.38,
            437929.84,  448627.8 ,  236238.67,  728707.7 ,  559923.9 ,
            684577.25,  846775.06,  536371.2 ,  241330.17,  243063.13,
            340764.53,  921561.56,  682284.56,  542342.  ,  400561.2 ,
            450867.7 ,  311909.6 ,  437177.97,  448627.8 ,  542342.  ,
            300480.3 ,  288798.1 ,  446769.  ,  350835.9 ,  450867.7 ,
            701940.7 ,  365436.22,  682284.56,  687786.44, 1178313.9 ])

```python
assay_baseline_from_numpy_name = "edge assays from numpy"

# assay builder by baseline
assay_builder_from_numpy = wl.build_assay(assay_name=assay_baseline_from_numpy_name, 
                               pipeline=mainpipeline, 
                               model_name=model_name_control, 
                               iopath="output variable 0", 
                               baseline_data = small_results_baseline)

# for brand new instances, this provides time for the pipeline logs to finish writing
time.sleep(60)
```

```python
# set the width
assay_builder_from_numpy.window_builder().add_width(minutes=1)
assay_builder_from_numpy.add_location_filter(['engine-6c6ccb9cf-ngg6g', 'houseprice-edgebaseline-examples'])
```

    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    /tmp/ipykernel_2056/3420244867.py in <module>
          1 # set the width
          2 assay_builder_from_numpy.window_builder().add_width(minutes=1)
    ----> 3 assay_builder_from_numpy.add_location_filter(['engine-6c6ccb9cf-ngg6g', 'houseprice-edgebaseline-examples'])
    

    AttributeError: 'AssayBuilder' object has no attribute 'add_location_filter'

```python
assay_config_from_numpy = assay_builder_from_numpy.build()
```

```python
assay_analysis_from_numpy = assay_config_from_numpy.interactive_run()
```

```python
# get the histogram from the numpy baseline
assay_builder_from_numpy.baseline_histogram()
```

    
{{<figure src="/images/2023.4.0/wallaroo-tutorials/edge-publish/edge-observability-assays-reference_files/edge-observability-assays-reference_26_0.png" width="800" label="png">}}
    

### Assay Testing

Now we'll perform some inferences with a spread of values, then a larger set with a set of larger house values to trigger our assay alert.  We'll use our assay created from the numpy baseline values to demonstrate.

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

    Generated 1 analyses

    
{{<figure src="/images/2023.4.0/wallaroo-tutorials/edge-publish/edge-observability-assays-reference_files/edge-observability-assays-reference_30_1.png" width="800" label="png">}}
    

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
      <td>0.0142</td>
      <td>2023-10-31T18:15:57.159000+00:00</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
  </tbody>
</table>

## Edge Deployment

### Publish Pipeline

### Add Edge

### DevOps Deployment

## Add Edge to Assay

```python
assay_pub = mainpipeline.publish()
```

    Waiting for pipeline publish... It may take up to 600 sec.
    Pipeline is Publishing.......Published.

```python
edge_name = f'houseprice-edge{suffix}'

edge_publish = assay_pub.add_edge(edge_name)
display(edge_publish)
```

<table>
    <tr><td>ID</td><td>41</td></tr>
    <tr><td>Pipeline Version</td><td>2d2ca802-4fb8-4a67-8081-a9242144d664</td></tr>
    <tr><td>Status</td><td>Published</td></tr>
    <tr><td>Engine URL</td><td><a href='https://us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/engines/proxy/wallaroo/ghcr.io/wallaroolabs/standalone-mini:v2023.4.0-main-4079'>us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/engines/proxy/wallaroo/ghcr.io/wallaroolabs/standalone-mini:v2023.4.0-main-4079</a></td></tr>
    <tr><td>Pipeline URL</td><td><a href='https://us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/pipelines/housepricesagapipeline:2d2ca802-4fb8-4a67-8081-a9242144d664'>us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/pipelines/housepricesagapipeline:2d2ca802-4fb8-4a67-8081-a9242144d664</a></td></tr>
    <tr><td>Helm Chart URL</td><td><a href='https://us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/charts/housepricesagapipeline'>us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/charts/housepricesagapipeline</a></td></tr>
    <tr><td>Helm Chart Reference</td><td>us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/charts@sha256:25d1e0391b50345abe10915aae39d9e1cf8b8ca6d8351c52a129ccdd01e20748</td></tr>
    <tr><td>Helm Chart Version</td><td>0.0.1-2d2ca802-4fb8-4a67-8081-a9242144d664</td></tr>
    <tr><td>Engine Config</td><td>{'engine': {'resources': {'limits': {'cpu': 4.0, 'memory': '3Gi'}, 'requests': {'cpu': 4.0, 'memory': '3Gi'}}}, 'engineAux': {}, 'enginelb': {'resources': {'limits': {'cpu': 1.0, 'memory': '512Mi'}, 'requests': {'cpu': 0.2, 'memory': '512Mi'}}}}</td></tr>
    <tr><td>User Images</td><td>[]</td></tr>
    <tr><td>Created By</td><td>john.hummel@wallaroo.ai</td></tr>
    <tr><td>Created At</td><td>2023-10-31 18:18:37.979662+00:00</td></tr>
    <tr><td>Updated At</td><td>2023-10-31 18:18:37.979662+00:00</td></tr>
    <tr><td>Docker Run Variables</td><td>{'EDGE_BUNDLE': 'abcde'}</td></tr>
</table>

```python
# create docker run 

docker_command = f'''
docker run -p 8080:8080 \\
    -e DEBUG=true \\
    -e OCI_REGISTRY=$REGISTRYURL \\
    -e EDGE_BUNDLE={edge_publish.docker_run_variables['EDGE_BUNDLE']} \\
    -e CONFIG_CPUS=1 \\
    -e OCI_USERNAME=$REGISTRYUSERNAME \\
    -e OCI_PASSWORD=$REGISTRYPASSWORD \\
    -e PIPELINE_URL={edge_publish.pipeline_url} \\
    {edge_publish.engine_url}
'''

print(docker_command)
```

    
    docker run -p 8080:8080 \
        -e DEBUG=true \
        -e OCI_REGISTRY=$REGISTRYURL \
        -e EDGE_BUNDLE=ZXhwb3J0IEJVTkRMRV9WRVJTSU9OPTEKZXhwb3J0IEVER0VfTkFNRT1ob3VzZXByaWNlLWVkZ2ViYXNlbGluZS1leGFtcGxlcwpleHBvcnQgSk9JTl9UT0tFTj02MWFlNDA1ZC04NjU3LTQ3YWEtYjFjNi0zZjJlODYwYTZkNTcKZXhwb3J0IE9QU0NFTlRFUl9IT1NUPXByb2R1Y3QtdWF0LWVlLmVkZ2Uud2FsbGFyb29jb21tdW5pdHkubmluamEKZXhwb3J0IFBJUEVMSU5FX1VSTD11cy1jZW50cmFsMS1kb2NrZXIucGtnLmRldi93YWxsYXJvby1kZXYtMjUzODE2L3VhdC9waXBlbGluZXMvaG91c2VwcmljZXNhZ2FwaXBlbGluZToyZDJjYTgwMi00ZmI4LTRhNjctODA4MS1hOTI0MjE0NGQ2NjQKZXhwb3J0IFdPUktTUEFDRV9JRD04OQ== \
        -e CONFIG_CPUS=1 \
        -e OCI_USERNAME=$REGISTRYUSERNAME \
        -e OCI_PASSWORD=$REGISTRYPASSWORD \
        -e PIPELINE_URL=us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/pipelines/housepricesagapipeline:2d2ca802-4fb8-4a67-8081-a9242144d664 \
        us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/engines/proxy/wallaroo/ghcr.io/wallaroolabs/standalone-mini:v2023.4.0-main-4079
    

```python
!curl testboy.local:8080/pipelines
```

    {"pipelines":[{"id":"housepricesagapipeline","status":"Running"}]}

```python
!curl -X POST testboy.local:8080/pipelines/housepricesagapipeline \
    -H "Content-Type: application/vnd.apache.arrow.file" \
    --data-binary @./data/xtest-1k.arrow > curl_response_edge.df.json
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  516k  100  445k  100 73100  19.9M  3272k --:--:-- --:--:-- --:--:-- 29.6M

```python
mainpipeline.export_logs(limit=50000,
    directory='partition-edge-observability',
    file_prefix='edge-logs',
    dataset=['time', 'out.variable', 'metadata'])
```

```python
# display the head 20 results

df_logs = pd.read_json('./partition-edge-observability/edge-logs-1.json', orient="records", lines=True)
display(df_logs.tail(20))

display(pd.unique(df_logs['metadata.partition']).tolist())
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out.variable</th>
      <th>metadata.last_model</th>
      <th>metadata.pipeline_version</th>
      <th>metadata.elapsed</th>
      <th>metadata.dropped</th>
      <th>metadata.partition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2482</th>
      <td>1698777454921</td>
      <td>[292859.5]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td></td>
      <td>[356937, 1404771]</td>
      <td>[]</td>
      <td>houseprice-edgebaseline-examples</td>
    </tr>
    <tr>
      <th>2483</th>
      <td>1698777454921</td>
      <td>[404040.8125]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td></td>
      <td>[356937, 1404771]</td>
      <td>[]</td>
      <td>houseprice-edgebaseline-examples</td>
    </tr>
    <tr>
      <th>2484</th>
      <td>1698777454921</td>
      <td>[313096.0]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td></td>
      <td>[356937, 1404771]</td>
      <td>[]</td>
      <td>houseprice-edgebaseline-examples</td>
    </tr>
    <tr>
      <th>2485</th>
      <td>1698777454921</td>
      <td>[909441.125]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td></td>
      <td>[356937, 1404771]</td>
      <td>[]</td>
      <td>houseprice-edgebaseline-examples</td>
    </tr>
    <tr>
      <th>2486</th>
      <td>1698777454921</td>
      <td>[559631.125]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td></td>
      <td>[356937, 1404771]</td>
      <td>[]</td>
      <td>houseprice-edgebaseline-examples</td>
    </tr>
    <tr>
      <th>2487</th>
      <td>1698777454921</td>
      <td>[630865.625]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td></td>
      <td>[356937, 1404771]</td>
      <td>[]</td>
      <td>houseprice-edgebaseline-examples</td>
    </tr>
    <tr>
      <th>2488</th>
      <td>1698777454921</td>
      <td>[442168.0625]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td></td>
      <td>[356937, 1404771]</td>
      <td>[]</td>
      <td>houseprice-edgebaseline-examples</td>
    </tr>
    <tr>
      <th>2489</th>
      <td>1698777454921</td>
      <td>[340764.53125]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td></td>
      <td>[356937, 1404771]</td>
      <td>[]</td>
      <td>houseprice-edgebaseline-examples</td>
    </tr>
    <tr>
      <th>2490</th>
      <td>1698777454921</td>
      <td>[559631.125]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td></td>
      <td>[356937, 1404771]</td>
      <td>[]</td>
      <td>houseprice-edgebaseline-examples</td>
    </tr>
    <tr>
      <th>2491</th>
      <td>1698777454921</td>
      <td>[727898.125]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td></td>
      <td>[356937, 1404771]</td>
      <td>[]</td>
      <td>houseprice-edgebaseline-examples</td>
    </tr>
    <tr>
      <th>2492</th>
      <td>1698777454921</td>
      <td>[684577.125]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td></td>
      <td>[356937, 1404771]</td>
      <td>[]</td>
      <td>houseprice-edgebaseline-examples</td>
    </tr>
    <tr>
      <th>2493</th>
      <td>1698777454921</td>
      <td>[1004846.625]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td></td>
      <td>[356937, 1404771]</td>
      <td>[]</td>
      <td>houseprice-edgebaseline-examples</td>
    </tr>
    <tr>
      <th>2494</th>
      <td>1698777454921</td>
      <td>[668287.9375]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td></td>
      <td>[356937, 1404771]</td>
      <td>[]</td>
      <td>houseprice-edgebaseline-examples</td>
    </tr>
    <tr>
      <th>2495</th>
      <td>1698777454921</td>
      <td>[513264.625]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td></td>
      <td>[356937, 1404771]</td>
      <td>[]</td>
      <td>houseprice-edgebaseline-examples</td>
    </tr>
    <tr>
      <th>2496</th>
      <td>1698777454921</td>
      <td>[758714.1875]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td></td>
      <td>[356937, 1404771]</td>
      <td>[]</td>
      <td>houseprice-edgebaseline-examples</td>
    </tr>
    <tr>
      <th>2497</th>
      <td>1698777454921</td>
      <td>[448627.6875]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td></td>
      <td>[356937, 1404771]</td>
      <td>[]</td>
      <td>houseprice-edgebaseline-examples</td>
    </tr>
    <tr>
      <th>2498</th>
      <td>1698777454921</td>
      <td>[615094.5625]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td></td>
      <td>[356937, 1404771]</td>
      <td>[]</td>
      <td>houseprice-edgebaseline-examples</td>
    </tr>
    <tr>
      <th>2499</th>
      <td>1698777454921</td>
      <td>[718013.6875]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td></td>
      <td>[356937, 1404771]</td>
      <td>[]</td>
      <td>houseprice-edgebaseline-examples</td>
    </tr>
    <tr>
      <th>2500</th>
      <td>1698776157195</td>
      <td>[1514079.375]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td>e96d2fde-84a2-44a0-aa1e-eaa3018da51a</td>
      <td>[42600, 275002]</td>
      <td>[]</td>
      <td>engine-6c6ccb9cf-ngg6g</td>
    </tr>
    <tr>
      <th>2501</th>
      <td>1698776157159</td>
      <td>[718013.6875]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td>e96d2fde-84a2-44a0-aa1e-eaa3018da51a</td>
      <td>[107801, 325003]</td>
      <td>[]</td>
      <td>engine-6c6ccb9cf-ngg6g</td>
    </tr>
  </tbody>
</table>

    ['engine-6c6ccb9cf-ngg6g', 'houseprice-edgebaseline-examples']

```python
assay_builder_from_numpy.add_location_filter(['engine-6c6ccb9cf-ngg6g', 'houseprice-edgebaseline-examples'])
```

    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    /tmp/ipykernel_2056/4052956881.py in <module>
    ----> 1 assay_builder_from_numpy.add_location_filter(locations=pd.unique(df_logs['metadata.partition']).tolist())
    

    AttributeError: 'AssayBuilder' object has no attribute 'add_location_filter'

### Upload Assay

With the assay created and fully tested, we will upload it to the Wallaroo instance.  This will make it available for future demonstrations and visible through the Wallaroo UI.

```python

```

```python
assay_builder_from_numpy.upload()
```

### Undeploy Main Pipeline

With the examples and tutorial complete, we will undeploy the main pipeline and return the resources back to the Wallaroo instance.

```python
mainpipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s .................................... ok

<table><tr><th>name</th> <td>housepricesagapipeline</td></tr><tr><th>created</th> <td>2023-10-31 18:15:22.006805+00:00</td></tr><tr><th>last_updated</th> <td>2023-10-31 18:18:37.854174+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>2d2ca802-4fb8-4a67-8081-a9242144d664, e96d2fde-84a2-44a0-aa1e-eaa3018da51a, 515bb61b-545b-4816-8e93-ab376472cd10</td></tr><tr><th>steps</th> <td>housepricesagacontrol</td></tr><tr><th>published</th> <td>False</td></tr></table>

```python

```
