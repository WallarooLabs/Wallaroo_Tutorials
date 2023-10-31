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

    {'name': 'edge-observability-assaysbaseline-examples', 'id': 13, 'archived': False, 'created_by': 'b3deff28-04d0-41b8-a04f-b5cf610d6ce9', 'created_at': '2023-10-31T20:04:58.466872+00:00', 'models': [{'name': 'housepricesagacontrol', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 10, 31, 20, 4, 58, 809138, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 10, 31, 20, 4, 58, 809138, tzinfo=tzutc())}], 'pipelines': [{'name': 'housepricesagapipeline', 'create_time': datetime.datetime(2023, 10, 31, 20, 4, 58, 893166, tzinfo=tzutc()), 'definition': '[]'}]}

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

     ok

<table><tr><th>name</th> <td>housepricesagapipeline</td></tr><tr><th>created</th> <td>2023-10-31 20:04:58.893166+00:00</td></tr><tr><th>last_updated</th> <td>2023-10-31 20:38:39.352660+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>c49b4fcd-521f-4435-a731-8c155e34eab7, 0056edf3-730f-452d-a6ed-2dfa47ff5567, 8bc714ea-8257-4512-a102-402baf3143b3, 76006480-b145-4d6a-9e95-9b2e7a4f8d8e</td></tr><tr><th>steps</th> <td>housepricesagacontrol</td></tr><tr><th>published</th> <td>True</td></tr></table>

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
      <td>2023-10-31 20:38:39.742</td>
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
      <td>2023-10-31 20:38:39.775</td>
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
      <td>2023-10-31 20:39:09.851</td>
      <td>[4.0, 2.0, 1780.0, 19843.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1780.0, 0.0, 47.4413986206, -122.1539993286, 2210.0, 13500.0, 52.0, 0.0, 0.0]</td>
      <td>[313096.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-10-31 20:39:09.851</td>
      <td>[3.0, 1.75, 1530.0, 7245.0, 1.0, 0.0, 0.0, 4.0, 7.0, 1530.0, 0.0, 47.7309989929, -122.1910018921, 1530.0, 7490.0, 31.0, 0.0, 0.0]</td>
      <td>[431929.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-10-31 20:39:09.851</td>
      <td>[5.0, 2.0, 2300.0, 7897.0, 2.5, 0.0, 0.0, 4.0, 8.0, 2300.0, 0.0, 47.7555999756, -122.3560028076, 2030.0, 7902.0, 59.0, 0.0, 0.0]</td>
      <td>[523576.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-10-31 20:39:09.851</td>
      <td>[5.0, 2.5, 2600.0, 3839.0, 2.0, 0.0, 0.0, 3.0, 7.0, 2600.0, 0.0, 47.4323997498, -122.1449966431, 2180.0, 4800.0, 9.0, 0.0, 0.0]</td>
      <td>[400676.47]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-10-31 20:39:09.851</td>
      <td>[5.0, 4.0, 4360.0, 8030.0, 2.0, 0.0, 0.0, 3.0, 10.0, 4360.0, 0.0, 47.592300415, -121.9729995728, 3570.0, 6185.0, 0.0, 0.0, 0.0]</td>
      <td>[1160512.8]</td>
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
      <td>2023-10-31 20:39:09.851</td>
      <td>[4.0, 3.75, 2690.0, 4000.0, 2.0, 0.0, 3.0, 4.0, 9.0, 2120.0, 570.0, 47.6417999268, -122.3720016479, 2830.0, 4000.0, 105.0, 1.0, 80.0]</td>
      <td>[999203.06]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>496</th>
      <td>2023-10-31 20:39:09.851</td>
      <td>[2.0, 1.5, 1170.0, 5248.0, 1.0, 0.0, 0.0, 5.0, 6.0, 1170.0, 0.0, 47.5317993164, -122.3740005493, 1170.0, 5120.0, 74.0, 0.0, 0.0]</td>
      <td>[260266.5]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>497</th>
      <td>2023-10-31 20:39:09.851</td>
      <td>[4.0, 2.5, 2630.0, 5701.0, 2.0, 0.0, 0.0, 3.0, 7.0, 2630.0, 0.0, 47.375, -122.1600036621, 2770.0, 5939.0, 4.0, 0.0, 0.0]</td>
      <td>[368504.3]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>498</th>
      <td>2023-10-31 20:39:09.851</td>
      <td>[3.0, 2.25, 2960.0, 8330.0, 1.0, 0.0, 3.0, 4.0, 10.0, 2260.0, 700.0, 47.7034988403, -122.3850021362, 2960.0, 8840.0, 62.0, 0.0, 0.0]</td>
      <td>[1178313.9]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>499</th>
      <td>2023-10-31 20:39:09.851</td>
      <td>[3.0, 2.25, 1780.0, 9969.0, 1.0, 0.0, 0.0, 3.0, 8.0, 1450.0, 330.0, 47.7285995483, -122.1679992676, 1950.0, 7974.0, 29.0, 0.0, 0.0]</td>
      <td>[437177.97]</td>
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
      <td>2023-10-31 20:39:09.851</td>
      <td>[4.0, 2.0, 1780.0, 19843.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1780.0, 0.0, 47.4413986206, -122.1539993286, 2210.0, 13500.0, 52.0, 0.0, 0.0]</td>
      <td>[313096.0]</td>
      <td>0</td>
      <td>313096.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-10-31 20:39:09.851</td>
      <td>[3.0, 1.75, 1530.0, 7245.0, 1.0, 0.0, 0.0, 4.0, 7.0, 1530.0, 0.0, 47.7309989929, -122.1910018921, 1530.0, 7490.0, 31.0, 0.0, 0.0]</td>
      <td>[431929.2]</td>
      <td>0</td>
      <td>431929.20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-10-31 20:39:09.851</td>
      <td>[5.0, 2.0, 2300.0, 7897.0, 2.5, 0.0, 0.0, 4.0, 8.0, 2300.0, 0.0, 47.7555999756, -122.3560028076, 2030.0, 7902.0, 59.0, 0.0, 0.0]</td>
      <td>[523576.2]</td>
      <td>0</td>
      <td>523576.20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-10-31 20:39:09.851</td>
      <td>[5.0, 2.5, 2600.0, 3839.0, 2.0, 0.0, 0.0, 3.0, 7.0, 2600.0, 0.0, 47.4323997498, -122.1449966431, 2180.0, 4800.0, 9.0, 0.0, 0.0]</td>
      <td>[400676.47]</td>
      <td>0</td>
      <td>400676.47</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-10-31 20:39:09.851</td>
      <td>[5.0, 4.0, 4360.0, 8030.0, 2.0, 0.0, 0.0, 3.0, 10.0, 4360.0, 0.0, 47.592300415, -121.9729995728, 3570.0, 6185.0, 0.0, 0.0, 0.0]</td>
      <td>[1160512.8]</td>
      <td>0</td>
      <td>1160512.80</td>
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
      <td>2023-10-31 20:39:09.851</td>
      <td>[4.0, 3.75, 2690.0, 4000.0, 2.0, 0.0, 3.0, 4.0, 9.0, 2120.0, 570.0, 47.6417999268, -122.3720016479, 2830.0, 4000.0, 105.0, 1.0, 80.0]</td>
      <td>[999203.06]</td>
      <td>0</td>
      <td>999203.06</td>
    </tr>
    <tr>
      <th>496</th>
      <td>2023-10-31 20:39:09.851</td>
      <td>[2.0, 1.5, 1170.0, 5248.0, 1.0, 0.0, 0.0, 5.0, 6.0, 1170.0, 0.0, 47.5317993164, -122.3740005493, 1170.0, 5120.0, 74.0, 0.0, 0.0]</td>
      <td>[260266.5]</td>
      <td>0</td>
      <td>260266.50</td>
    </tr>
    <tr>
      <th>497</th>
      <td>2023-10-31 20:39:09.851</td>
      <td>[4.0, 2.5, 2630.0, 5701.0, 2.0, 0.0, 0.0, 3.0, 7.0, 2630.0, 0.0, 47.375, -122.1600036621, 2770.0, 5939.0, 4.0, 0.0, 0.0]</td>
      <td>[368504.3]</td>
      <td>0</td>
      <td>368504.30</td>
    </tr>
    <tr>
      <th>498</th>
      <td>2023-10-31 20:39:09.851</td>
      <td>[3.0, 2.25, 2960.0, 8330.0, 1.0, 0.0, 3.0, 4.0, 10.0, 2260.0, 700.0, 47.7034988403, -122.3850021362, 2960.0, 8840.0, 62.0, 0.0, 0.0]</td>
      <td>[1178313.9]</td>
      <td>0</td>
      <td>1178313.90</td>
    </tr>
    <tr>
      <th>499</th>
      <td>2023-10-31 20:39:09.851</td>
      <td>[3.0, 2.25, 1780.0, 9969.0, 1.0, 0.0, 0.0, 3.0, 8.0, 1450.0, 330.0, 47.7285995483, -122.1679992676, 1950.0, 7974.0, 29.0, 0.0, 0.0]</td>
      <td>[437177.97]</td>
      <td>0</td>
      <td>437177.97</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 5 columns</p>

```python
# get the numpy values
small_results_baseline = small_results_baseline_df['variable'].to_numpy()
small_results_baseline
```

    array([ 313096.  ,  431929.2 ,  523576.2 ,  400676.47, 1160512.8 ,
            288798.1 ,  258377.  ,  404676.1 ,  309800.8 ,  448627.8 ,
            475971.5 ,  276046.53,  530288.94,  290323.1 ,  311515.1 ,
            450867.7 ,  444408.  ,  261201.22,  276709.06,  343304.63,
            244380.27,  275408.03,  252192.9 ,  431992.22,  539867.1 ,
            573403.2 ,  348616.63,  453195.8 ,  385561.78,  725572.56,
            637377.  ,  684577.25,  950678.06,  557391.25,  793214.3 ,
            388243.13,  536371.2 ,  243585.28,  449699.88,  437177.97,
            784103.56, 1227073.8 ,  448627.8 ,  448627.8 ,  950176.7 ,
            516278.63,  713979.  ,  274207.16,  359614.66,  450867.7 ,
            473287.25,  358668.2 ,  508926.25,  450928.94,  597475.75,
            716173.94, 1115275.  ,  444408.  ,  437177.97,  430252.3 ,
            536388.2 ,  706407.4 ,  256630.36,  400561.2 ,  701940.7 ,
            846775.06,  349102.75,  291239.75,  642519.7 ,  630865.5 ,
            778197.7 ,  546631.94,  317132.63,  340764.53,  424966.6 ,
            241330.17,  718445.7 ,  725184.1 ,  553463.25,  328513.6 ,
            947561.9 ,  328513.6 ,  363491.63,  448627.8 ,  886958.6 ,
            292859.44,  473287.25,  448627.8 ,  375012.  ,  241809.88,
            630865.5 ,  553463.25,  544392.06,  519346.94,  448627.8 ,
            529302.44,  236238.67,  363491.63,  473287.25,  353912.  ,
            765468.9 ,  553463.25,  450867.7 ,  559923.9 ,  243300.83,
            244380.27,  675545.44,  291799.84,  320395.38,  450867.7 ,
            448627.8 ,  437177.97,  317132.63,  450867.7 ,  293808.03,
            343304.63,  303936.78,  437177.97,  236238.67,  630865.5 ,
            296202.7 ,  594678.75,  673288.6 ,  713979.  ,  236815.8 ,
            598725.  ,  349102.75,  349102.75,  921561.56,  432908.6 ,
            988481.9 ,  418823.38,  340764.53,  557391.25,  340764.53,
            630865.5 ,  448627.8 ,  689450.3 ,  444931.28,  557391.25,
            557391.25,  482485.6 , 1115275.  ,  458858.44,  437177.97,
            421402.1 ,  721518.44,  404676.1 ,  276046.53,  448627.8 ,
            717051.8 ,  244380.27,  572709.94,  438346.38,  249227.83,
           1077279.1 ,  573403.2 ,  473287.25,  291239.75,  516278.63,
            332134.97,  442856.4 ,  296202.7 ,  236238.67,  917346.2 ,
            630865.5 ,  448627.8 ,  656923.44,  682284.56,  784103.56,
            266405.63,  291857.06,  758714.3 ,  236238.67,  567502.06,
            252192.9 ,  718588.94,  628260.9 ,  550184.  ,  448627.8 ,
            274207.16,  713485.7 ,  559923.9 ,  276046.53,  244380.27,
            267013.97,  513583.06,  559631.06,  243063.13,  311515.1 ,
            536388.2 ,  718445.7 ,  673519.6 ,  784103.56,  557391.25,
            987157.25,  498579.5 ,  687786.44,  498579.5 ,  276046.53,
            581002.94,  730767.94,  380009.28,  435628.72,  450996.34,
            524275.4 ,  437177.97,  548006.06,  723867.7 ,  727923.2 ,
            448627.8 ,  368504.3 ,  964052.6 ,  950678.06,  581002.94,
           1160512.8 , 1208638.1 ,  288798.1 ,  448627.8 ,  340764.53,
            437177.97,  689450.3 ,  713979.  ,  711565.44,  557391.25,
            306159.38,  450867.7 ,  267013.97,  448627.8 ,  919031.5 ,
            682181.9 ,  363491.63,  236238.67,  498579.5 ,  296202.7 ,
           1039781.2 ,  448627.8 ,  340764.53,  400561.2 ,  464057.38,
            418823.38, 1489624.3 ,  448627.8 ,  293560.06,  403520.16,
            236238.67,  340764.53,  567502.06,  437177.97,  342604.47,
            291799.84,  836230.2 ,  657905.75,  246525.2 ,  546631.94,
            340764.53,  557391.25,  765468.9 ,  276709.06,  704672.25,
            673288.6 ,  879092.9 ,  634865.7 ,  519346.94,  630865.5 ,
            438346.38,  701940.7 ,  453195.8 ,  284081.53,  683869.56,
            249455.83,  435628.72,  964052.6 ,  276709.06,  413013.38,
            437177.97,  340764.53,  482485.6 ,  291239.75,  559452.94,
            267013.97,  550275.1 ,  758714.3 ,  437177.97,  546631.94,
            299854.75,  383833.88,  713358.8 ,  448627.8 ,  448627.8 ,
            244380.27,  964052.6 ,  421402.1 ,  758714.3 ,  338357.88,
            437929.84,  300446.66,  277145.63,  716173.94,  937359.6 ,
            236238.67,  457449.06,  320863.72,  426066.38,  469038.13,
            281823.13,  368504.3 ,  291799.84,  450928.94,  827411.25,
            291903.97,  293808.03,  437753.4 ,  947561.9 ,  713485.7 ,
            553463.25,  236238.67,  637377.  ,  303002.25,  557391.25,
            384558.4 ,  559631.06,  400561.2 ,  657905.75,  559923.9 ,
            725572.56,  266405.63,  721518.44,  437177.97,  278475.66,
            450867.7 ,  291239.75,  268856.88,  438346.38,  404676.1 ,
           1004846.56,  464060.2 ,  921695.4 ,  438346.38,  575724.7 ,
            475270.97,  340764.53,  555231.94,  563844.44,  523152.63,
            416774.6 ,  290323.1 ,  559631.06,  559631.06,  557391.25,
            438346.38,  450867.7 ,  340764.53,  256630.36,  557391.25,
            684577.25,  921561.56,  376762.4 ,  306037.63,  400561.2 ,
            886958.6 ,  557391.25,  244351.92,  711565.44,  557391.25,
            594678.75,  243300.83,  508746.63,  559631.06, 1060847.5 ,
            241330.17,  310992.94,  448627.8 ,  536371.2 ,  404676.1 ,
            567502.06,  474010.4 ,  642519.7 ,  238078.03,  675545.44,
            246901.14,  340764.53,  450867.7 ,  448627.8 ,  711565.44,
            251194.58,  550184.  ,  553463.25,  241330.17,  630865.5 ,
            291857.06,  340764.53,  448627.8 ,  726181.75,  555231.94,
            557391.25, 1325960.8 ,  236238.67,  306037.63,  705013.5 ,
            683845.75,  795841.06,  448627.8 ,  757403.4 ,  261201.22,
            675545.44,  846775.06,  431929.2 ,  352864.1 ,  260266.5 ,
            246088.6 ,  434534.22,  988481.9 ,  400676.47,  236238.67,
           1223838.8 ,  236815.8 ,  311515.1 ,  957189.7 ,  450867.7 ,
            393833.97,  448627.8 ,  957189.7 , 1322835.6 ,  575724.7 ,
           1052897.9 ,  765468.9 ,  437177.97,  448627.8 ,  557391.25,
            368504.3 ,  340764.53,  557391.25,  332134.97,  448627.8 ,
            296202.7 ,  400561.2 ,  559923.9 ,  559139.4 ,  544392.06,
            718445.7 ,  322015.03,  353912.  ,  400561.2 ,  538436.75,
            313096.  ,  630865.5 ,  437177.97,  559631.06,  873848.44,
            559631.06, 1295531.8 ,  947561.9 ,  446769.  ,  423382.72,
            682284.56,  435628.72,  323856.28,  236815.8 ,  448627.8 ,
            276709.06,  630865.5 ,  303002.25,  508926.25,  701940.7 ,
            513583.06,  450867.7 ,  394707.2 ,  340764.53,  349102.75,
            725875.94,  581002.94,  448627.8 ,  444408.  ,  308049.63,
            421402.1 ,  782434.44,  846775.06,  236815.8 ,  293560.06,
            310992.94,  278475.66,  317132.63,  391459.97,  437002.88,
            244351.92,  348616.63,  706407.4 ,  358668.2 ,  846775.06,
            448627.8 ,  765468.9 ,  557391.25,  921561.56,  736751.3 ,
            999203.06,  260266.5 ,  368504.3 , 1178313.9 ,  437177.97])

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
```

    <wallaroo.assay_config.WindowBuilder at 0x7f439dba5a00>

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

    Generated 6 analyses

    
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
      <td>0.015281</td>
      <td>2023-10-31T20:05:18.132000+00:00</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.012015</td>
      <td>2023-10-31T20:09:18.132000+00:00</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.871443</td>
      <td>2023-10-31T20:11:18.132000+00:00</td>
      <td>0.25</td>
      <td>Alert</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.039158</td>
      <td>2023-10-31T20:16:18.132000+00:00</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.002826</td>
      <td>2023-10-31T20:38:18.132000+00:00</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>5</th>
      <td>8.871443</td>
      <td>2023-10-31T20:40:18.132000+00:00</td>
      <td>0.25</td>
      <td>Alert</td>
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
    Pipeline is Publishing....Published.

```python
edge_name = f'houseprice-edge{suffix}'

edge_publish = assay_pub.add_edge(edge_name)
display(edge_publish)
```

<table>
    <tr><td>ID</td><td>4</td></tr>
    <tr><td>Pipeline Version</td><td>abd69aa0-10f5-4787-977c-8ce62a112b74</td></tr>
    <tr><td>Status</td><td>Published</td></tr>
    <tr><td>Engine URL</td><td><a href='https://us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/engines/proxy/wallaroo/ghcr.io/wallaroolabs/standalone-mini:v2023.4.0-4092'>us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/engines/proxy/wallaroo/ghcr.io/wallaroolabs/standalone-mini:v2023.4.0-4092</a></td></tr>
    <tr><td>Pipeline URL</td><td><a href='https://us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/pipelines/housepricesagapipeline:abd69aa0-10f5-4787-977c-8ce62a112b74'>us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/pipelines/housepricesagapipeline:abd69aa0-10f5-4787-977c-8ce62a112b74</a></td></tr>
    <tr><td>Helm Chart URL</td><td>oci://<a href='https://us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/charts/housepricesagapipeline'>us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/charts/housepricesagapipeline</a></td></tr>
    <tr><td>Helm Chart Reference</td><td>us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/charts@sha256:b5a745c792f2845473a7b863791af5639e6a9ca1de6669c74245bafd903b6eaf</td></tr>
    <tr><td>Helm Chart Version</td><td>0.0.1-abd69aa0-10f5-4787-977c-8ce62a112b74</td></tr>
    <tr><td>Engine Config</td><td>{'engine': {'resources': {'limits': {'cpu': 4.0, 'memory': '3Gi'}, 'requests': {'cpu': 4.0, 'memory': '3Gi'}}}, 'engineAux': {}, 'enginelb': {'resources': {'limits': {'cpu': 1.0, 'memory': '512Mi'}, 'requests': {'cpu': 0.2, 'memory': '512Mi'}}}}</td></tr>
    <tr><td>User Images</td><td>[]</td></tr>
    <tr><td>Created By</td><td>john.hummel@wallaroo.ai</td></tr>
    <tr><td>Created At</td><td>2023-10-31 20:41:21.203601+00:00</td></tr>
    <tr><td>Updated At</td><td>2023-10-31 20:41:21.203601+00:00</td></tr>
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
        -e EDGE_BUNDLE=ZXhwb3J0IEJVTkRMRV9WRVJTSU9OPTEKZXhwb3J0IEVER0VfTkFNRT1ob3VzZXByaWNlLWVkZ2ViYXNlbGluZS1leGFtcGxlcwpleHBvcnQgSk9JTl9UT0tFTj1iMTVkMWFjYS03NzIwLTQwZWYtOTYwMC0wMGY4NzMxZDE2NjkKZXhwb3J0IE9QU0NFTlRFUl9IT1NUPXByb2R1Y3QtdWF0LWVlLmVkZ2Uud2FsbGFyb29jb21tdW5pdHkubmluamEKZXhwb3J0IFBJUEVMSU5FX1VSTD11cy1jZW50cmFsMS1kb2NrZXIucGtnLmRldi93YWxsYXJvby1kZXYtMjUzODE2L3VhdC9waXBlbGluZXMvaG91c2VwcmljZXNhZ2FwaXBlbGluZTphYmQ2OWFhMC0xMGY1LTQ3ODctOTc3Yy04Y2U2MmExMTJiNzQKZXhwb3J0IFdPUktTUEFDRV9JRD0xMw== \
        -e CONFIG_CPUS=1 \
        -e OCI_USERNAME=$REGISTRYUSERNAME \
        -e OCI_PASSWORD=$REGISTRYPASSWORD \
        -e PIPELINE_URL=us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/pipelines/housepricesagapipeline:abd69aa0-10f5-4787-977c-8ce62a112b74 \
        us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/engines/proxy/wallaroo/ghcr.io/wallaroolabs/standalone-mini:v2023.4.0-4092
    

```bash
curl testboy.local:8080/pipelines
{"pipelines":[{"id":"housepricesagapipeline","status":"Running"}]}
```

```bash
curl -X POST testboy.local:8080/pipelines/housepricesagapipeline \
    -H "Content-Type: application/vnd.apache.arrow.file" \
    --data-binary @./data/xtest-1k.arrow > curl_response_edge.df.json
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100  513k  100  442k  100 73100  19.8M  3287k --:--:-- --:--:-- --:--:-- 23.8M
```

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

display(pd.unique(df_logs['metadata.partition']).tolist()[1])
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
      <th>5484</th>
      <td>1698782748239</td>
      <td>[778197.6875]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td></td>
      <td>[1801705, 2182407]</td>
      <td>[]</td>
      <td>engine-54c865f485-dwhjf</td>
    </tr>
    <tr>
      <th>5485</th>
      <td>1698782748239</td>
      <td>[236238.671875]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td></td>
      <td>[1801705, 2182407]</td>
      <td>[]</td>
      <td>engine-54c865f485-dwhjf</td>
    </tr>
    <tr>
      <th>5486</th>
      <td>1698782748239</td>
      <td>[448627.8125]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td></td>
      <td>[1801705, 2182407]</td>
      <td>[]</td>
      <td>engine-54c865f485-dwhjf</td>
    </tr>
    <tr>
      <th>5487</th>
      <td>1698782748239</td>
      <td>[683869.5625]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td></td>
      <td>[1801705, 2182407]</td>
      <td>[]</td>
      <td>engine-54c865f485-dwhjf</td>
    </tr>
    <tr>
      <th>5488</th>
      <td>1698782748239</td>
      <td>[684577.25]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td></td>
      <td>[1801705, 2182407]</td>
      <td>[]</td>
      <td>engine-54c865f485-dwhjf</td>
    </tr>
    <tr>
      <th>5489</th>
      <td>1698782748239</td>
      <td>[253679.625]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td></td>
      <td>[1801705, 2182407]</td>
      <td>[]</td>
      <td>engine-54c865f485-dwhjf</td>
    </tr>
    <tr>
      <th>5490</th>
      <td>1698782748239</td>
      <td>[287576.40625]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td></td>
      <td>[1801705, 2182407]</td>
      <td>[]</td>
      <td>engine-54c865f485-dwhjf</td>
    </tr>
    <tr>
      <th>5491</th>
      <td>1698782748239</td>
      <td>[553463.25]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td></td>
      <td>[1801705, 2182407]</td>
      <td>[]</td>
      <td>engine-54c865f485-dwhjf</td>
    </tr>
    <tr>
      <th>5492</th>
      <td>1698782748239</td>
      <td>[498579.5]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td></td>
      <td>[1801705, 2182407]</td>
      <td>[]</td>
      <td>engine-54c865f485-dwhjf</td>
    </tr>
    <tr>
      <th>5493</th>
      <td>1698782748239</td>
      <td>[682284.5625]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td></td>
      <td>[1801705, 2182407]</td>
      <td>[]</td>
      <td>engine-54c865f485-dwhjf</td>
    </tr>
    <tr>
      <th>5494</th>
      <td>1698782748239</td>
      <td>[827411.25]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td></td>
      <td>[1801705, 2182407]</td>
      <td>[]</td>
      <td>engine-54c865f485-dwhjf</td>
    </tr>
    <tr>
      <th>5495</th>
      <td>1698782748239</td>
      <td>[544392.0625]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td></td>
      <td>[1801705, 2182407]</td>
      <td>[]</td>
      <td>engine-54c865f485-dwhjf</td>
    </tr>
    <tr>
      <th>5496</th>
      <td>1698782748239</td>
      <td>[311515.09375]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td></td>
      <td>[1801705, 2182407]</td>
      <td>[]</td>
      <td>engine-54c865f485-dwhjf</td>
    </tr>
    <tr>
      <th>5497</th>
      <td>1698782748239</td>
      <td>[263051.625]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td></td>
      <td>[1801705, 2182407]</td>
      <td>[]</td>
      <td>engine-54c865f485-dwhjf</td>
    </tr>
    <tr>
      <th>5498</th>
      <td>1698782748239</td>
      <td>[712309.875]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td></td>
      <td>[1801705, 2182407]</td>
      <td>[]</td>
      <td>engine-54c865f485-dwhjf</td>
    </tr>
    <tr>
      <th>5499</th>
      <td>1698782748239</td>
      <td>[349102.75]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td></td>
      <td>[1801705, 2182407]</td>
      <td>[]</td>
      <td>engine-54c865f485-dwhjf</td>
    </tr>
    <tr>
      <th>5500</th>
      <td>1698782748239</td>
      <td>[475270.96875]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td></td>
      <td>[1801705, 2182407]</td>
      <td>[]</td>
      <td>engine-54c865f485-dwhjf</td>
    </tr>
    <tr>
      <th>5501</th>
      <td>1698782748239</td>
      <td>[290323.09375]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td></td>
      <td>[1801705, 2182407]</td>
      <td>[]</td>
      <td>engine-54c865f485-dwhjf</td>
    </tr>
    <tr>
      <th>5502</th>
      <td>1698782718168</td>
      <td>[1514079.375]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td></td>
      <td>[69600, 886202]</td>
      <td>[]</td>
      <td>engine-54c865f485-dwhjf</td>
    </tr>
    <tr>
      <th>5503</th>
      <td>1698782718132</td>
      <td>[718013.6875]</td>
      <td>{"model_name":"housepricesagacontrol","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}</td>
      <td></td>
      <td>[173700, 340301]</td>
      <td>[]</td>
      <td>engine-54c865f485-dwhjf</td>
    </tr>
  </tbody>
</table>

    ['engine-54c865f485-dwhjf', 'houseprice-edgebaseline-examples']

    'houseprice-edgebaseline-examples'

```python
assay_builder_from_numpy.window_builder().add_location_filter([pd.unique(df_logs['metadata.partition']).tolist()[1]])
```

    <wallaroo.assay_config.WindowBuilder at 0x7f439dba5a00>

```python
assay_config_from_numpy = assay_builder_from_numpy.build()
```

```python
assay_analysis_from_numpy = assay_config_from_numpy.interactive_run()
```

```python
# Show how many assay windows were analyzed, then show the chart
print(f"Generated {len(assay_analysis_from_numpy)} analyses")
assay_analysis_from_numpy.chart_scores()
```

    Generated 1 analyses

    
{{<figure src="/images/2023.4.0/wallaroo-tutorials/edge-publish/edge-observability-assays-reference_files/edge-observability-assays-reference_42_1.png" width="800" label="png">}}
    

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
      <td>0.039158</td>
      <td>2023-10-31T20:16:18.132000+00:00</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
  </tbody>
</table>

### Upload Assay

With the assay created and fully tested, we will upload it to the Wallaroo instance.  This will make it available for future demonstrations and visible through the Wallaroo UI.

```python
# left unrun as exercise to the reader
# assay_builder_from_numpy.upload()
```

### Undeploy Main Pipeline

With the examples and tutorial complete, we will undeploy the main pipeline and return the resources back to the Wallaroo instance.

```python
mainpipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s ...................................
