This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-observe-tutorials/wallaro-model-observability-assays).

## Model Drift Observability with Assays

The Model Insights feature lets you monitor how the environment that your model operates within may be changing in ways that affect it's predictions so that you can intervene (retrain) in an efficient and timely manner. Changes in the inputs, **data drift**, can occur due to errors in the data processing pipeline or due to changes in the environment such as user preference or behavior. 

This notebook focuses on interactive exploration over historical data. After you are comfortable with how your data has behaved historically, you can schedule this same analysis (called an *assay*) to automatically run periodically, looking for indications of data drift or concept drift.

In this notebook, we will be running a drift assay on an ONNX model pre-trained to predict house prices.

## Goal

Model insights monitors the output of the spam classifier model over a designated time window and compares it to an expected baseline distribution. We measure the difference between  the window distribution and the baseline distribution; large differences indicate that the behavior of the model (or its inputs) has changed from what we expect. This possibly indicates a change that should be accounted for, possibly by retraining the models.

### Resources

This tutorial provides the following:

* Models:
  * `models/rf_model.onnx`: The champion model that has been used in this environment for some time.
  * Various inputs:
    * `smallinputs.df.json`: A set of house inputs that tends to generate low house price values.
    * `biginputs.df.json`: A set of house inputs that tends to generate high house price values.

### Prerequisites

* A deployed Wallaroo instance
* The following Python libraries installed:
  * [`wallaroo`](https://pypi.org/project/wallaroo/): The Wallaroo SDK. Included with the Wallaroo JupyterHub service by default.
  * [`pandas`](https://pypi.org/project/pandas/): Pandas, mainly used for Pandas DataFrame

## Steps

* Deploying a sample ML model used to determine house prices based on a set of input parameters.
* Build an assay baseline from a set of baseline start and end dates, and an assay baseline from a numpy array.
* Preview the assay and show different assay configurations.
* Upload the assay.
* View assay results.
* Pause and resume the assay.

### Import Libraries

The first step will be to import our libraries, and set variables used through this tutorial.

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
suffix=''

# used to make a unique workspace
suffix=''

workspace_name = f'assay-demonstration-tutorial{suffix}'
main_pipeline_name = f'assay-demonstration-tutorial'
model_name_control = f'house-price-estimator'
model_file_name_control = './models/rf_model.onnx'

# Set the name of the assay
assay_name=f"house price test{suffix}"

# ignoring warnings for demonstration
import warnings
warnings.filterwarnings('ignore')

# used to display DataFrame information without truncating
from IPython.display import display
import pandas as pd
pd.set_option('display.max_colwidth', None)
```

```python
def get_workspace(name, client):
    workspace = None
    for ws in client.list_workspaces():
        if ws.name() == name:
            workspace= ws
    if(workspace == None):
        workspace = client.create_workspace(name)
    return workspace
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
workspace = get_workspace(workspace_name, wl)

wl.set_current_workspace(workspace)
```

    {'name': 'assay-demonstration-tutorial', 'id': 5, 'archived': False, 'created_by': '755d34dd-e562-454e-82ed-593d6f3244b7', 'created_at': '2024-02-15T16:18:22.083904+00:00', 'models': [], 'pipelines': []}

### Upload The Champion Model

For our example, we will upload the champion model that has been trained to derive house prices from a variety of inputs.  The model file is `rf_model.onnx`, and is uploaded with the name `house-price-estimator`.

```python
housing_model_control = (wl.upload_model(model_name_control, 
                                        model_file_name_control, 
                                        framework=Framework.ONNX)
                                        .configure(tensor_fields=["tensor"])
                        )
```

### Build the Pipeline

This pipeline is made to be an example of an existing situation where a model is deployed and being used for inferences in a production environment.  We'll call it `assay-demonstration-tutorial`, set `housing_model_control` as a pipeline step, then run a few sample inferences.

This pipeline will be a simple one - just a single pipeline step.

```python
mainpipeline = wl.build_pipeline(main_pipeline_name)
# clear the steps if used before
mainpipeline.clear()

mainpipeline.add_model_step(housing_model_control)

#minimum deployment config
deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(0.5).memory("1Gi").build()

mainpipeline.deploy(deployment_config = deploy_config)
```

<table><tr><th>name</th> <td>assay-demonstration-tutorial</td></tr><tr><th>created</th> <td>2024-02-15 16:18:25.122619+00:00</td></tr><tr><th>last_updated</th> <td>2024-02-15 16:18:26.007576+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>53ea84b7-21d0-4aad-9e20-5d8f368abc63, 8de360dd-a086-4b7f-a5ec-84bae69e13ac</td></tr><tr><th>steps</th> <td>house-price-estimator</td></tr><tr><th>published</th> <td>False</td></tr></table>

### Testing

We'll use two inferences as a quick sample test - one that has a house that should be determined around `$700k`, the other with a house determined to be around `$1.5` million.

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
      <td>2024-02-15 16:18:42.466</td>
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
      <td>2024-02-15 16:18:42.866</td>
      <td>[4.0, 3.0, 3710.0, 20000.0, 2.0, 0.0, 2.0, 5.0, 10.0, 2760.0, 950.0, 47.6696, -122.261, 3970.0, 20000.0, 79.0, 0.0, 0.0]</td>
      <td>[1514079.4]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

### Generate Sample Data

Before creating the assays, we must generate data for the assays to build from.

For this example, we will:

* Perform sample inferences based on lower priced houses and use that as our baseline.
* Generate inferences from specific set of high priced houses create inference outputs that will be outside the baseline.  This is used in later steps to demonstrate baseline comparison against assay analyses.

#### Inference Results History Generation

To start the demonstration, we'll create a baseline of values from houses with small estimated prices and set that as our baseline.

We will save the beginning and end periods of our baseline data to the variables `assay_baseline_start` and `assay_baseline_end`.

```python
small_houses_inputs = pd.read_json('./data/smallinputs.df.json')
baseline_size = 500

# Where the baseline data will start
assay_baseline_start = datetime.datetime.now()

# These inputs will be random samples of small priced houses.  Around 30,000 is a good number
small_houses = small_houses_inputs.sample(baseline_size, replace=True).reset_index(drop=True)

# Wait 60 seconds to set this data apart from the rest
time.sleep(60)
small_results = mainpipeline.infer(small_houses)

# Set the baseline end

assay_baseline_end = datetime.datetime.now()
```

#### Generate Numpy Baseline Values

This process generates a numpy array of the inference results used as baseline data in later steps.

```python
# get the numpy values

# set the results to a non-array value
small_results_baseline_df = small_results.copy()
small_results_baseline_df['variable']=small_results['out.variable'].map(lambda x: x[0])
small_results_baseline_df

# set the numpy array
small_results_baseline = small_results_baseline_df['variable'].to_numpy()
```

#### Assay Test Data

The following will generate inference data for us to test against the assay baseline.  For this, we will add in house data that generate higher house prices than the baseline data we used earlier.

This process should take 6 minutes to generate the historical data we'll later use in our assays.  We store the DateTime `assay_window_start` to determine where to start out assay analyses.

```python
# Get a spread of house values

# # Set the start for our assay window period.
assay_window_start = datetime.datetime.now()

time.sleep(65)
inference_size = 1000

# And a spread of large house values

small_houses_inputs = pd.read_json('./data/smallinputs.df.json', orient="records")
small_houses = small_houses_inputs.sample(inference_size, replace=True).reset_index(drop=True)

mainpipeline.infer(small_houses)

time.sleep(65)
```

```python
# Get a spread of large house values

time.sleep(65)
inference_size = 1000

# And a spread of large house values

big_houses_inputs = pd.read_json('./data/biginputs.df.json', orient="records")
big_houses = big_houses_inputs.sample(inference_size, replace=True).reset_index(drop=True)

mainpipeline.infer(big_houses)

time.sleep(65)
```

## Model Insights via the Wallaroo Dashboard SDK

Assays generated through the Wallaroo SDK can be previewed, configured, and uploaded to the Wallaroo Ops instance.  The following is a condensed version of this process.  For full details see the [Wallaroo SDK Essentials Guide: Assays Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-assays/) guide.

Model drift detection with assays using the Wallaroo SDK follows this general process.

* Define the Baseline: From either historical inference data for a specific model in a pipeline, or from a pre-determine array of data, a **baseline** is formed.
* Assay Preview:  Once the baseline is formed, we **preview the assay** and configure the different options until we have the the best method of detecting environment or model drift.
* Create Assay:  With the previews and configuration complete, we **upload** the assay.  The assay will perform an analysis on a regular scheduled based on the configuration.
* Get Assay Results:  Retrieve the analyses and use them to detect model drift and possible sources.
* Pause/Resume Assay:  Pause or restart an assay as needed.

### Define the Baseline

Assay baselines are defined with the [`wallaroo.client.build_assay`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/client/#Client.build_assay) method. Through this process we define the baseline from either a range of dates or pre-generated values.

`wallaroo.client.build_assay` take the following parameters:

| Parameter | Type | Description |
|---|---|---|
| **assay_name** | *String* (*Required*) - required | The name of the assay.  Assay names **must** be unique across the Wallaroo instance. |
| **pipeline** | *wallaroo.pipeline.Pipeline* (*Required*) | The pipeline the assay is monitoring. |
| **model_name** | *String* (*Required*)  | The name of the model to monitor.
| **iopath** | *String* (*Required*) | The input/output data for the model being tracked in the format `input/output field index`.  Only one value is tracked for any assay.  For example, to track the **output** of the model's field `house_value` at index `0`, the `iopath` is `'output house_value 0`. |
| **baseline_start** | *datetime.datetime* (*Optional*) | The start time for the inferences to use as the baseline.  **Must be included with `baseline_end`.  Cannot be included with `baseline_data`**. |
| **baseline_end** | *datetime.datetime* (*Optional*) | The end time of the baseline window. the baseline. Windows start immediately after the baseline window and are run at regular intervals continuously until the assay is deactivated or deleted.  **Must be included with `baseline_start`.  Cannot be included with `baseline_data`.**. |
| **baseline_data** | *numpy.array* (*Optional*) | The baseline data in numpy array format.  **Cannot be included with either `baseline_start` or `baseline_data`**. |

Baselines are created in one of two ways:

* **Date Range**:  The `baseline_start` and `baseline_end` retrieves the inference requests and results for the pipeline from the start and end period.  This data is summarized and used to create the baseline.
* **Numpy Values**:  The `baseline_data` sets the baseline from a provided numpy array.

#### Define the Baseline Example

This example shows two methods of defining the baseline for an assay:

* `"assays from date baseline"`: This assay uses historical inference requests to define the baseline.  This assay is saved to the variable `assay_builder_from_dates`.
* `"assays from numpy"`:  This assay uses a pre-generated numpy array to define the baseline.  This assay is saved to the variable `assay_builder_from_numpy`.

In both cases, the following parameters are used:

| Parameter | Value |
|---|---|
| **assay_name** | `"assays from date baseline"` and `"assays from numpy"` |
| **pipeline** | `mainpipeline`:  A pipeline with a ML model that predicts house prices.  The output field for this model is `variable`. |
| **model_name** | `"houseprice-predictor"` - the model name set during model upload. |
| **iopath** | These assays monitor the model's **output** field **variable** at index 0.  From this, the `iopath` setting is `"output variable 0"`.  |

The difference between the two assays' parameters determines how the baseline is generated.

* `"assays from date baseline"`: Uses the `baseline_start` and `baseline_end` to set the time period of inference requests and results to gather data from.
* `"assays from numpy"`:  Uses a pre-generated numpy array as for the baseline data.

For each of our assays, we will set the time period of inference data to compare against the baseline data.

```python
# Build the assay, based on the start and end of our baseline time, 
# and tracking the output variable index 0
assay_builder_from_dates = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=mainpipeline, 
                                          model_name="house-price-estimator", 
                                          iopath="output variable 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# set the width, interval, and time period 
assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)
assay_config_from_dates = assay_builder_from_dates.build()
assay_results_from_dates = assay_config_from_dates.interactive_run()

# assay builder by baseline
assay_builder_from_numpy = wl.build_assay(assay_name="assays from numpy", 
                               pipeline=mainpipeline, 
                               model_name="house-price-estimator", 
                               iopath="output variable 0", 
                               baseline_data = small_results_baseline)

# set the width, interval, and time period 
assay_builder_from_numpy.add_run_until(datetime.datetime.now())
assay_builder_from_numpy.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

assay_config_from_numpy = assay_builder_from_numpy.build()
assay_results_from_numpy = assay_config_from_numpy.interactive_run()
```

#### Baseline DataFrame

The method [`wallaroo.assay_config.AssayBuilder.baseline_dataframe`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/assay_config/#AssayBuilder.baseline_dataframe) returns a DataFrame of the assay baseline generated from the provided parameters.  This includes:

* `metadata`:  The inference metadata with the model information, inference time, and other related factors.
* `in` data:  Each input field assigned with the label `in.{input field name}`.
* `out` data:  Each output field assigned with the label `out.{output field name}`

Note that for assays generated from numpy values, there is only the `out` data based on the supplied baseline data.

In the following example, the baseline DataFrame is retrieved.  

```python
display(assay_builder_from_dates.baseline_dataframe())
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>metadata</th>
      <th>input_tensor_0</th>
      <th>input_tensor_1</th>
      <th>input_tensor_2</th>
      <th>input_tensor_3</th>
      <th>input_tensor_4</th>
      <th>input_tensor_5</th>
      <th>input_tensor_6</th>
      <th>input_tensor_7</th>
      <th>...</th>
      <th>input_tensor_9</th>
      <th>input_tensor_10</th>
      <th>input_tensor_11</th>
      <th>input_tensor_12</th>
      <th>input_tensor_13</th>
      <th>input_tensor_14</th>
      <th>input_tensor_15</th>
      <th>input_tensor_16</th>
      <th>input_tensor_17</th>
      <th>output_variable_0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1708013922866</td>
      <td>{'last_model': '{"model_name":"house-price-estimator","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}', 'pipeline_version': '', 'elapsed': [49610, 341497], 'dropped': [], 'partition': 'engine-58b565bf45-29xnm'}</td>
      <td>4.0</td>
      <td>3.00</td>
      <td>3710.0</td>
      <td>20000.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>...</td>
      <td>2760.0</td>
      <td>950.0</td>
      <td>47.669600</td>
      <td>-122.261000</td>
      <td>3970.0</td>
      <td>20000.0</td>
      <td>79.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.514079e+06</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1708013983808</td>
      <td>{'last_model': '{"model_name":"house-price-estimator","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}', 'pipeline_version': '', 'elapsed': [2517407, 2287534], 'dropped': [], 'partition': 'engine-58b565bf45-29xnm'}</td>
      <td>3.0</td>
      <td>2.50</td>
      <td>1500.0</td>
      <td>7420.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>1000.0</td>
      <td>500.0</td>
      <td>47.723598</td>
      <td>-122.174004</td>
      <td>1840.0</td>
      <td>7272.0</td>
      <td>42.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.196772e+05</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1708013983808</td>
      <td>{'last_model': '{"model_name":"house-price-estimator","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}', 'pipeline_version': '', 'elapsed': [2517407, 2287534], 'dropped': [], 'partition': 'engine-58b565bf45-29xnm'}</td>
      <td>4.0</td>
      <td>2.50</td>
      <td>2009.0</td>
      <td>5000.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>2009.0</td>
      <td>0.0</td>
      <td>47.257702</td>
      <td>-122.197998</td>
      <td>2009.0</td>
      <td>5182.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.208637e+05</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1708013983808</td>
      <td>{'last_model': '{"model_name":"house-price-estimator","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}', 'pipeline_version': '', 'elapsed': [2517407, 2287534], 'dropped': [], 'partition': 'engine-58b565bf45-29xnm'}</td>
      <td>3.0</td>
      <td>1.75</td>
      <td>1530.0</td>
      <td>7245.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>1530.0</td>
      <td>0.0</td>
      <td>47.730999</td>
      <td>-122.191002</td>
      <td>1530.0</td>
      <td>7490.0</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.319292e+05</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1708013983808</td>
      <td>{'last_model': '{"model_name":"house-price-estimator","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}', 'pipeline_version': '', 'elapsed': [2517407, 2287534], 'dropped': [], 'partition': 'engine-58b565bf45-29xnm'}</td>
      <td>3.0</td>
      <td>1.75</td>
      <td>1480.0</td>
      <td>4800.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>1140.0</td>
      <td>340.0</td>
      <td>47.656700</td>
      <td>-122.397003</td>
      <td>1810.0</td>
      <td>4800.0</td>
      <td>70.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.361757e+05</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>496</th>
      <td>1708013983808</td>
      <td>{'last_model': '{"model_name":"house-price-estimator","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}', 'pipeline_version': '', 'elapsed': [2517407, 2287534], 'dropped': [], 'partition': 'engine-58b565bf45-29xnm'}</td>
      <td>4.0</td>
      <td>2.25</td>
      <td>2560.0</td>
      <td>12100.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>1760.0</td>
      <td>800.0</td>
      <td>47.631001</td>
      <td>-122.108002</td>
      <td>2240.0</td>
      <td>12100.0</td>
      <td>38.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.019407e+05</td>
    </tr>
    <tr>
      <th>497</th>
      <td>1708013983808</td>
      <td>{'last_model': '{"model_name":"house-price-estimator","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}', 'pipeline_version': '', 'elapsed': [2517407, 2287534], 'dropped': [], 'partition': 'engine-58b565bf45-29xnm'}</td>
      <td>2.0</td>
      <td>1.00</td>
      <td>1160.0</td>
      <td>5000.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>1160.0</td>
      <td>0.0</td>
      <td>47.686501</td>
      <td>-122.399002</td>
      <td>1750.0</td>
      <td>5000.0</td>
      <td>77.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.508677e+05</td>
    </tr>
    <tr>
      <th>498</th>
      <td>1708013983808</td>
      <td>{'last_model': '{"model_name":"house-price-estimator","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}', 'pipeline_version': '', 'elapsed': [2517407, 2287534], 'dropped': [], 'partition': 'engine-58b565bf45-29xnm'}</td>
      <td>4.0</td>
      <td>2.50</td>
      <td>1910.0</td>
      <td>5000.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>1910.0</td>
      <td>0.0</td>
      <td>47.360802</td>
      <td>-122.036003</td>
      <td>2020.0</td>
      <td>5000.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.962027e+05</td>
    </tr>
    <tr>
      <th>499</th>
      <td>1708013983808</td>
      <td>{'last_model': '{"model_name":"house-price-estimator","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}', 'pipeline_version': '', 'elapsed': [2517407, 2287534], 'dropped': [], 'partition': 'engine-58b565bf45-29xnm'}</td>
      <td>3.0</td>
      <td>1.50</td>
      <td>1590.0</td>
      <td>8911.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>1590.0</td>
      <td>0.0</td>
      <td>47.739399</td>
      <td>-122.251999</td>
      <td>1590.0</td>
      <td>9625.0</td>
      <td>58.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.371780e+05</td>
    </tr>
    <tr>
      <th>500</th>
      <td>1708013983808</td>
      <td>{'last_model': '{"model_name":"house-price-estimator","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}', 'pipeline_version': '', 'elapsed': [2517407, 2287534], 'dropped': [], 'partition': 'engine-58b565bf45-29xnm'}</td>
      <td>4.0</td>
      <td>2.75</td>
      <td>2640.0</td>
      <td>4000.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>...</td>
      <td>1730.0</td>
      <td>910.0</td>
      <td>47.672699</td>
      <td>-122.296997</td>
      <td>1530.0</td>
      <td>3740.0</td>
      <td>89.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.184457e+05</td>
    </tr>
  </tbody>
</table>
<p>501 rows × 21 columns</p>

```python
assay_builder_from_numpy.baseline_dataframe()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>output_variable_0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>419677.20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>320863.72</td>
    </tr>
    <tr>
      <th>2</th>
      <td>431929.20</td>
    </tr>
    <tr>
      <th>3</th>
      <td>536175.70</td>
    </tr>
    <tr>
      <th>4</th>
      <td>343304.63</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>495</th>
      <td>701940.70</td>
    </tr>
    <tr>
      <th>496</th>
      <td>450867.70</td>
    </tr>
    <tr>
      <th>497</th>
      <td>296202.70</td>
    </tr>
    <tr>
      <th>498</th>
      <td>437177.97</td>
    </tr>
    <tr>
      <th>499</th>
      <td>718445.70</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 1 columns</p>

#### Baseline Stats

The method `wallaroo.assay.AssayAnalysis.baseline_stats()` returns a `pandas.core.frame.DataFrame` of the baseline stats.

The baseline stats for each assay are displayed in the examples below.

```python
assay_results_from_dates[0].baseline_stats()
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
      <td>501</td>
    </tr>
    <tr>
      <th>min</th>
      <td>236238.671875</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1514079.375</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>495193.231786</td>
    </tr>
    <tr>
      <th>median</th>
      <td>442168.125</td>
    </tr>
    <tr>
      <th>std</th>
      <td>226075.814267</td>
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

```python
assay_results_from_numpy[0].baseline_stats()
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
      <td>493155.46054</td>
    </tr>
    <tr>
      <th>median</th>
      <td>441840.425</td>
    </tr>
    <tr>
      <th>std</th>
      <td>221657.583536</td>
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

#### Baseline Bins

The method `wallaroo.assay.AssayAnalysis.baseline_bins` a simple dataframe to with the edge/bin data for a baseline.

```python
assay_results_from_dates[0].baseline_bins()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>b_edges</th>
      <th>b_edge_names</th>
      <th>b_aggregated_values</th>
      <th>b_aggregation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.362387e+05</td>
      <td>left_outlier</td>
      <td>0.000000</td>
      <td>Density</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.962027e+05</td>
      <td>q_20</td>
      <td>0.203593</td>
      <td>Density</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.159643e+05</td>
      <td>q_40</td>
      <td>0.195609</td>
      <td>Density</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.640602e+05</td>
      <td>q_60</td>
      <td>0.203593</td>
      <td>Density</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6.821819e+05</td>
      <td>q_80</td>
      <td>0.197605</td>
      <td>Density</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.514079e+06</td>
      <td>q_100</td>
      <td>0.199601</td>
      <td>Density</td>
    </tr>
    <tr>
      <th>6</th>
      <td>inf</td>
      <td>right_outlier</td>
      <td>0.000000</td>
      <td>Density</td>
    </tr>
  </tbody>
</table>

```python
assay_results_from_numpy[0].baseline_bins()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>b_edges</th>
      <th>b_edge_names</th>
      <th>b_aggregated_values</th>
      <th>b_aggregation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>236238.67</td>
      <td>left_outlier</td>
      <td>0.000</td>
      <td>Density</td>
    </tr>
    <tr>
      <th>1</th>
      <td>296202.70</td>
      <td>q_20</td>
      <td>0.204</td>
      <td>Density</td>
    </tr>
    <tr>
      <th>2</th>
      <td>415964.30</td>
      <td>q_40</td>
      <td>0.196</td>
      <td>Density</td>
    </tr>
    <tr>
      <th>3</th>
      <td>464057.38</td>
      <td>q_60</td>
      <td>0.200</td>
      <td>Density</td>
    </tr>
    <tr>
      <th>4</th>
      <td>675545.44</td>
      <td>q_80</td>
      <td>0.200</td>
      <td>Density</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1489624.30</td>
      <td>q_100</td>
      <td>0.200</td>
      <td>Density</td>
    </tr>
    <tr>
      <th>6</th>
      <td>inf</td>
      <td>right_outlier</td>
      <td>0.000</td>
      <td>Density</td>
    </tr>
  </tbody>
</table>

#### Baseline Histogram Chart

The method [`wallaroo.assay_config.AssayBuilder.baseline_histogram`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/assay_config/#AssayBuilder.baseline_dataframe) returns a histogram chart of the assay baseline generated from the provided parameters.

```python
assay_builder_from_dates.baseline_histogram()
```

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_35_0.png" width="800" label="png">}}
    

#### Baseline KDE Chart

The method [`wallaroo.assay_config.AssayBuilder.baseline_kde`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/assay_config/#AssayBuilder.baseline_dataframe) returns a Kernel Density Estimation (KDE) chart of the assay baseline generated from the provided parameters.

```python
assay_builder_from_dates.baseline_kde()
```

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_37_0.png" width="800" label="png">}}
    

#### Baseline ECDF Chart

The method [`wallaroo.assay_config.AssayBuilder.baseline_ecdf`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/assay_config/#AssayBuilder.baseline_dataframe) returns a Empirical Cumulative Distribution Function (CDF) chart of the assay baseline generated from the provided parameters.

```python
assay_builder_from_dates.baseline_ecdf()
```

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_39_0.png" width="800" label="png">}}
    

### Assay Preview

Now that the baseline is defined, we look at different configuration options and view how the assay baseline and results changes.  Once we determine what gives us the best method of determining model drift, we can create the assay.

#### Analysis List Chart Scores

Analysis List scores show the assay scores for each assay result interval in one chart.  Values that are outside of the alert threshold are colored red, while scores within the alert threshold are green.

Assay chart scores are displayed with the method [`wallaroo.assay.AssayAnalysisList.chart_scores(title: Optional[str] = None)`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/assay/#AssayAnalysisList.chart_scores), with ability to display an optional title with the chart.

The following example shows retrieving the assay results and displaying the chart scores.  From our example, we have two windows - the first should be green, and the second is red showing that values were outside the alert threshold.

```python
# Build the assay, based on the start and end of our baseline time, 
# and tracking the output variable index 0
assay_builder_from_dates = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=mainpipeline, 
                                          model_name="house-price-estimator", 
                                          iopath="output variable 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# set the width, interval, and time period 
assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)
assay_config_from_dates = assay_builder_from_dates.build()
assay_results_from_dates = assay_config_from_dates.interactive_run()

assay_results_from_dates.chart_scores()
```

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_42_0.png" width="800" label="png">}}
    

#### Analysis Chart

The method `wallaroo.assay.AssayAnalysis.chart()` displays a comparison between the baseline and an interval of inference data.

This is compared to the [Chart Scores](#analysis-list-chart-scores), which is a **list** of all of the inference data split into intervals, while the **Analysis Chart** shows the breakdown of one set of inference data against the baseline.

Score from the [Analysis List Chart Scores](#analysis-list-chart-scores) and each element from the [Analysis List DataFrame](#analysis-list-dataframe) generates 

The following fields are included.

| Field | Type | Description |
|---|---|---|
| **baseline mean** | **Float** | The mean of the baseline values. |
| **window mean** | **Float** | The mean of the window values. |
| **baseline median** | **Float** | The median of the baseline values. |
| **window median** | **Float** | The median of the window values. |
| **bin_mode** | **String** | The binning mode used for the assay. |
| **aggregation** | **String** | The aggregation mode used for the assay. |
| **metric** | **String** | The metric mode used for the assay. |
| **weighted** | **Bool** | Whether the bins were manually weighted. |
| **score** | **Float** | The score from the assay window. |
| **scores** | **List(Float)** | The score from each assay window bin. |
| **index** | **Integer/None** | The window index.  Interactive assay runs are `None`. |

```python
# Build the assay, based on the start and end of our baseline time, 
# and tracking the output variable index 0
assay_builder_from_dates = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=mainpipeline, 
                                          model_name="house-price-estimator", 
                                          iopath="output variable 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# set the width, interval, and time period 
assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)
assay_config_from_dates = assay_builder_from_dates.build()
assay_results_from_dates = assay_config_from_dates.interactive_run()

assay_results_from_dates[0].chart()
```

    baseline mean = 495193.23178642715
    window mean = 517763.394625
    baseline median = 442168.125
    window median = 448627.8125
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.0363497101644573
    scores = [0.0, 0.027271477163285655, 0.003847844548034077, 0.000217023993714693, 0.002199485350158766, 0.0028138791092641195, 0.0]
    index = None

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_44_1.png" width="800" label="png">}}
    

#### Analysis List DataFrame

`wallaroo.assay.AssayAnalysisList.to_dataframe()` returns a DataFrame showing the assay results for each window aka individual analysis.  This DataFrame contains the following fields:

| Field | Type | Description |
|---|---|---|
| **assay_id** | *Integer/None* | The assay id.  Only provided from uploaded and executed assays. |
| **name** | *String/None* | The name of the assay.  Only provided from uploaded and executed assays. |
| **iopath** | *String/None* | The iopath of the assay.  Only provided from uploaded and executed assays. |
| **score** | *Float* | The assay score. |
| **start** | *DateTime* | The DateTime start of the assay window.
| **min** | *Float* | The minimum value in the assay window.
| **max**  | *Float* | The maximum value in the assay window.
| **mean** | *Float* | The mean value in the assay window.
| **median** | *Float* | The median value in the assay window.
| **std** | *Float* | The standard deviation value in the assay window.
| **warning_threshold** | *Float/None* | The warning threshold of the assay window.
| **alert_threshold** | *Float/None* | The alert threshold of the assay window.
| **status** | *String* | The assay window status.  Values are:  <ul><li>`OK`: The score is within accepted thresholds.</li><li>`Warning`: The score has triggered the `warning_threshold` if exists, but not the `alert_threshold`.</li><li>`Alert`: The score has triggered the the `alert_threshold`.</li></ul> |

For this example, the assay analysis list DataFrame is listed.  

From this tutorial, we should have 2 windows of dta to look at, each one minute apart.  The first window should show `status: OK`, with the second window with the very large house prices will show `status: alert`

```python
# Build the assay, based on the start and end of our baseline time, 
# and tracking the output variable index 0
assay_builder_from_dates = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=mainpipeline, 
                                          model_name="house-price-estimator", 
                                          iopath="output variable 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# set the width, interval, and time period 
assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)
assay_config_from_dates = assay_builder_from_dates.build()
assay_results_from_dates = assay_config_from_dates.interactive_run()

assay_results_from_dates.to_dataframe()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>assay_id</th>
      <th>name</th>
      <th>iopath</th>
      <th>score</th>
      <th>start</th>
      <th>min</th>
      <th>max</th>
      <th>mean</th>
      <th>median</th>
      <th>std</th>
      <th>warning_threshold</th>
      <th>alert_threshold</th>
      <th>status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>None</td>
      <td></td>
      <td></td>
      <td>0.036350</td>
      <td>2024-02-15T16:20:43.976756+00:00</td>
      <td>2.362387e+05</td>
      <td>1489624.250</td>
      <td>5.177634e+05</td>
      <td>4.486278e+05</td>
      <td>227729.030050</td>
      <td>None</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>1</th>
      <td>None</td>
      <td></td>
      <td></td>
      <td>8.868614</td>
      <td>2024-02-15T16:22:43.976756+00:00</td>
      <td>1.514079e+06</td>
      <td>2016006.125</td>
      <td>1.885772e+06</td>
      <td>1.946438e+06</td>
      <td>160046.727324</td>
      <td>None</td>
      <td>0.25</td>
      <td>Alert</td>
    </tr>
  </tbody>
</table>

#### Analysis List Full DataFrame

`wallaroo.assay.AssayAnalysisList.to_full_dataframe()` returns a DataFrame showing all values, including the inputs and outputs from the assay results for each window aka individual analysis.  This DataFrame contains the following fields:

	pipeline_id	warning_threshold	bin_index	created_at

| Field | Type | Description |
|---|---|---|
| **window_start** | *DateTime* | The date and time when the window period began. |
| **analyzed_at** | *DateTime* | The date and time when the assay analysis was performed. |
| **elapsed_millis** | *Integer* | How long the analysis took to perform in milliseconds. |
| **baseline_summary_count** | *Integer* | The number of data elements from the baseline. |
| **baseline_summary_min** | *Float* | The minimum value from the baseline summary. |
| **baseline_summary_max** | *Float* | The maximum value from the baseline summary. |
| **baseline_summary_mean** | *Float* | The mean value of the baseline summary. |
| **baseline_summary_median** | *Float* | The median value of the baseline summary. |
| **baseline_summary_std** | *Float* | The standard deviation value of the baseline summary. |
| **baseline_summary_edges_{0...n}** | *Float* | The baseline summary edges for each baseline edge from 0 to number of edges. |
| **summarizer_type** | *String* | The type of summarizer used for the baseline.  See [`wallaroo.assay_config` for other summarizer types](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/assay_config/). |
| **summarizer_bin_weights** | *List / None* | If baseline bin weights were provided, the list of those weights.  Otherwise, `None`. |
| **summarizer_provided_edges** | *List / None* | If baseline bin edges were provided, the list of those edges.  Otherwise, `None`. |
| **status** | *String* | The assay window status.  Values are:  <ul><li>`OK`: The score is within accepted thresholds.</li><li>`Warning`: The score has triggered the `warning_threshold` if exists, but not the `alert_threshold`.</li><li>`Alert`: The score has triggered the the `alert_threshold`.</li></ul> |
| **id** | *Integer/None* | The id for the window aka analysis.  Only provided from uploaded and executed assays. |
| **assay_id** | *Integer/None* | The assay id.  Only provided from uploaded and executed assays. |
| **pipeline_id** | *Integer/None* | The pipeline id.  Only provided from uploaded and executed assays. |
| **warning_threshold** | *Float* | The warning threshold set for the assay. |
| **warning_threshold** | *Float* | The warning threshold set for the assay.
| **bin_index** | *Integer/None* | The bin index for the window aka analysis.|
| **created_at** | *Datetime/None* | The date and time the window aka analysis was generated.  Only provided from uploaded and executed assays. |

For this example, full DataFrame from an assay preview is generated.

From this tutorial, we should have 2 windows of dta to look at, each one minute apart.  The first window should show `status: OK`, with the second window with the very large house prices will show `status: alert`

```python
# Build the assay, based on the start and end of our baseline time, 
# and tracking the output variable index 0
assay_builder_from_dates = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=mainpipeline, 
                                          model_name="house-price-estimator", 
                                          iopath="output variable 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# set the width, interval, and time period 
assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)
assay_config_from_dates = assay_builder_from_dates.build()
assay_results_from_dates = assay_config_from_dates.interactive_run()

assay_results_from_dates.to_full_dataframe()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>window_start</th>
      <th>analyzed_at</th>
      <th>elapsed_millis</th>
      <th>baseline_summary_count</th>
      <th>baseline_summary_min</th>
      <th>baseline_summary_max</th>
      <th>baseline_summary_mean</th>
      <th>baseline_summary_median</th>
      <th>baseline_summary_std</th>
      <th>baseline_summary_edges_0</th>
      <th>...</th>
      <th>summarizer_type</th>
      <th>summarizer_bin_weights</th>
      <th>summarizer_provided_edges</th>
      <th>status</th>
      <th>id</th>
      <th>assay_id</th>
      <th>pipeline_id</th>
      <th>warning_threshold</th>
      <th>bin_index</th>
      <th>created_at</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-02-15T16:20:43.976756+00:00</td>
      <td>2024-02-15T16:26:42.266029+00:00</td>
      <td>82</td>
      <td>501</td>
      <td>236238.671875</td>
      <td>1514079.375</td>
      <td>495193.231786</td>
      <td>442168.125</td>
      <td>226075.814267</td>
      <td>236238.671875</td>
      <td>...</td>
      <td>UnivariateContinuous</td>
      <td>None</td>
      <td>None</td>
      <td>Ok</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-02-15T16:22:43.976756+00:00</td>
      <td>2024-02-15T16:26:42.266134+00:00</td>
      <td>83</td>
      <td>501</td>
      <td>236238.671875</td>
      <td>1514079.375</td>
      <td>495193.231786</td>
      <td>442168.125</td>
      <td>226075.814267</td>
      <td>236238.671875</td>
      <td>...</td>
      <td>UnivariateContinuous</td>
      <td>None</td>
      <td>None</td>
      <td>Alert</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 86 columns</p>

#### Analysis Compare Basic Stats

The method `wallaroo.assay.AssayAnalysis.compare_basic_stats` returns a DataFrame comparing one set of inference data against the baseline.

This is compared to the [Analysis List DataFrame](#analysis-list-dataframe), which is a **list** of all of the inference data split into intervals, while the **Analysis Compare Basic Stats** shows the breakdown of one set of inference data against the baseline.

```python
# Build the assay, based on the start and end of our baseline time, 
# and tracking the output variable index 0
assay_builder_from_dates = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=mainpipeline, 
                                          model_name="house-price-estimator", 
                                          iopath="output variable 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# set the width, interval, and time period 
assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)
assay_config_from_dates = assay_builder_from_dates.build()
assay_results_from_dates = assay_config_from_dates.interactive_run()

assay_results_from_dates[0].compare_basic_stats()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Baseline</th>
      <th>Window</th>
      <th>diff</th>
      <th>pct_diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>501.0</td>
      <td>1000.0</td>
      <td>499.000000</td>
      <td>99.600798</td>
    </tr>
    <tr>
      <th>min</th>
      <td>236238.671875</td>
      <td>236238.671875</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1514079.375</td>
      <td>1489624.25</td>
      <td>-24455.125000</td>
      <td>-1.615181</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>495193.231786</td>
      <td>517763.394625</td>
      <td>22570.162839</td>
      <td>4.557850</td>
    </tr>
    <tr>
      <th>median</th>
      <td>442168.125</td>
      <td>448627.8125</td>
      <td>6459.687500</td>
      <td>1.460912</td>
    </tr>
    <tr>
      <th>std</th>
      <td>226075.814267</td>
      <td>227729.03005</td>
      <td>1653.215783</td>
      <td>0.731266</td>
    </tr>
    <tr>
      <th>start</th>
      <td>None</td>
      <td>2024-02-15T16:20:43.976756+00:00</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>end</th>
      <td>None</td>
      <td>2024-02-15T16:21:43.976756+00:00</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>

#### Configure Assays

Before creating the assay, **configure** the assay and continue to preview it until the best method for detecting drift is set.  The following options are available.

##### Score Metric

The `score` is a distance between the baseline and the analysis window.  The larger the score, the greater the difference between the baseline and the analysis window.  The following methods are provided determining the score:

* `PSI` (*Default*) - Population Stability Index (PSI).
* `MAXDIFF`: Maximum difference between corresponding bins.
* `SUMDIFF`: Mum of differences between corresponding bins.

The metric type used is updated with the [`wallaroo.assay_config.AssayBuilder.add_metric(metric: wallaroo.assay_config.Metric)`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/assay_config/#UnivariateContinousSummarizerBuilder.add_metric) method.

The following three charts use each of the metrics.  Note how the scores change based on the score type used.

```python
# Build the assay, based on the start and end of our baseline time, 
# and tracking the output variable index 0
assay_builder_from_dates = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=mainpipeline, 
                                          model_name="house-price-estimator", 
                                          iopath="output variable 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# set metric PSI mode
assay_builder_from_dates.summarizer_builder.add_metric(wallaroo.assay_config.Metric.PSI)

# set the width, interval, and time period 
assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)
assay_config_from_dates = assay_builder_from_dates.build()
assay_results_from_dates = assay_config_from_dates.interactive_run()

assay_results_from_dates[0].chart()
```

    baseline mean = 495193.23178642715
    window mean = 517763.394625
    baseline median = 442168.125
    window median = 448627.8125
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.0363497101644573
    scores = [0.0, 0.027271477163285655, 0.003847844548034077, 0.000217023993714693, 0.002199485350158766, 0.0028138791092641195, 0.0]
    index = None

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_53_1.png" width="800" label="png">}}
    

```python
# Build the assay, based on the start and end of our baseline time, 
# and tracking the output variable index 0
assay_builder_from_dates = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=mainpipeline, 
                                          model_name="house-price-estimator", 
                                          iopath="output variable 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# set metric MAXDIFF mode
assay_builder_from_dates.summarizer_builder.add_metric(wallaroo.assay_config.Metric.MAXDIFF)

# set the width, interval, and time period 
assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)
assay_config_from_dates = assay_builder_from_dates.build()
assay_results_from_dates = assay_config_from_dates.interactive_run()

assay_results_from_dates[0].chart()
```

    baseline mean = 495193.23178642715
    window mean = 517763.394625
    baseline median = 442168.125
    window median = 448627.8125
    bin_mode = Quantile
    aggregation = Density
    metric = MaxDiff
    weighted = False
    score = 0.06759281437125747
    scores = [0.0, 0.06759281437125747, 0.028391217564870255, 0.006592814371257472, 0.02139520958083832, 0.02439920159680639, 0.0]
    index = 1

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_54_1.png" width="800" label="png">}}
    

```python
# Build the assay, based on the start and end of our baseline time, 
# and tracking the output variable index 0
assay_builder_from_dates = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=mainpipeline, 
                                          model_name="house-price-estimator", 
                                          iopath="output variable 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# set metric SUMDIFF mode
assay_builder_from_dates.summarizer_builder.add_metric(wallaroo.assay_config.Metric.SUMDIFF)

# set the width, interval, and time period 
assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)
assay_config_from_dates = assay_builder_from_dates.build()
assay_results_from_dates = assay_config_from_dates.interactive_run()

assay_results_from_dates[0].chart()
```

    baseline mean = 495193.23178642715
    window mean = 517763.394625
    baseline median = 442168.125
    window median = 448627.8125
    bin_mode = Quantile
    aggregation = Density
    metric = SumDiff
    weighted = False
    score = 0.07418562874251496
    scores = [0.0, 0.06759281437125747, 0.028391217564870255, 0.006592814371257472, 0.02139520958083832, 0.02439920159680639, 0.0]
    index = None

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_55_1.png" width="800" label="png">}}
    

##### Alert Threshold

Assay alert thresholds are modified with the [`wallaroo.assay_config.AssayBuilder.add_alert_threshold(alert_threshold: float)`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/assay_config/) method.  By default alert thresholds are `0.1`.

The following example updates the alert threshold to `0.5`.

```python
# Build the assay, based on the start and end of our baseline time, 
# and tracking the output variable index 0
assay_builder_from_dates = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=mainpipeline, 
                                          model_name="house-price-estimator", 
                                          iopath="output variable 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

assay_builder_from_dates.add_alert_threshold(0.5)

# set the width, interval, and time period 
assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)
assay_config_from_dates = assay_builder_from_dates.build()
assay_results_from_dates = assay_config_from_dates.interactive_run()

assay_results_from_dates.to_dataframe()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>assay_id</th>
      <th>name</th>
      <th>iopath</th>
      <th>score</th>
      <th>start</th>
      <th>min</th>
      <th>max</th>
      <th>mean</th>
      <th>median</th>
      <th>std</th>
      <th>warning_threshold</th>
      <th>alert_threshold</th>
      <th>status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>None</td>
      <td></td>
      <td></td>
      <td>0.036350</td>
      <td>2024-02-15T16:20:43.976756+00:00</td>
      <td>2.362387e+05</td>
      <td>1489624.250</td>
      <td>5.177634e+05</td>
      <td>4.486278e+05</td>
      <td>227729.030050</td>
      <td>None</td>
      <td>0.5</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>1</th>
      <td>None</td>
      <td></td>
      <td></td>
      <td>8.868614</td>
      <td>2024-02-15T16:22:43.976756+00:00</td>
      <td>1.514079e+06</td>
      <td>2016006.125</td>
      <td>1.885772e+06</td>
      <td>1.946438e+06</td>
      <td>160046.727324</td>
      <td>None</td>
      <td>0.5</td>
      <td>Alert</td>
    </tr>
  </tbody>
</table>

##### Number of Bins

Number of bins sets how the baseline data is partitioned.  The total number of bins includes the set number plus the left_outlier and the right_outlier, so the total number of bins will be the total set + 2.

The number of bins is set with the [`wallaroo.assay_config.UnivariateContinousSummarizerBuilder.add_num_bins(num_bins: int)`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/assay_config/#UnivariateContinousSummarizerBuilder.add_num_bins) method.

```python
# Build the assay, based on the start and end of our baseline time, 
# and tracking the output variable index 0
assay_builder_from_dates = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=mainpipeline, 
                                          model_name="house-price-estimator", 
                                          iopath="output variable 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# Set the number of bins
# update number of bins here
assay_builder_from_dates.summarizer_builder.add_num_bins(10)

# set the width, interval, and time period 
assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)
assay_config_from_dates = assay_builder_from_dates.build()
assay_results_from_dates = assay_config_from_dates.interactive_run()

assay_results_from_dates[0].chart()
```

    baseline mean = 495193.23178642715
    window mean = 517763.394625
    baseline median = 442168.125
    window median = 448627.8125
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.05250979748389363
    scores = [0.0, 0.009076998929542533, 0.01924002322223739, 0.0021945246367443406, 0.0016700458183385653, 0.005779503770625584, 0.002393429678215835, 0.002942858220315506, 0.00010651192741915124, 0.00046961759334670583, 0.008636283687108028, 0.0]
    index = None

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_59_1.png" width="800" label="png">}}
    

##### Binning Mode

Binning Mode defines how the bins are separated.  Binning modes are modified through the `wallaroo.assay_config.UnivariateContinousSummarizerBuilder.add_bin_mode(bin_mode: bin_mode: wallaroo.assay_config.BinMode, edges: Optional[List[float]] = None)`.

Available `bin_mode` values from `wallaroo.assay_config.Binmode` are the following:

* `QUANTILE` (*Default*): Based on percentages. If `num_bins` is 5 then quintiles so bins are created at the 20%, 40%, 60%, 80% and 100% points.
* `EQUAL`: Evenly spaced bins where each bin is set with the formula `min - max / num_bins`
* `PROVIDED`: The user provides the edge points for the bins.

If `PROVIDED` is supplied, then a List of float values must be provided for the `edges` parameter that matches the number of bins.

The following examples are used to show how each of the binning modes effects the bins.

```python
# Build the assay, based on the start and end of our baseline time, 
# and tracking the output variable index 0
assay_builder_from_dates = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=mainpipeline, 
                                          model_name="house-price-estimator", 
                                          iopath="output variable 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# update binning mode here
assay_builder_from_dates.summarizer_builder.add_bin_mode(wallaroo.assay_config.BinMode.QUANTILE)

# set the width, interval, and time period 
assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)
assay_config_from_dates = assay_builder_from_dates.build()
assay_results_from_dates = assay_config_from_dates.interactive_run()

assay_results_from_dates[0].chart()
```

    baseline mean = 495193.23178642715
    window mean = 517763.394625
    baseline median = 442168.125
    window median = 448627.8125
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.0363497101644573
    scores = [0.0, 0.027271477163285655, 0.003847844548034077, 0.000217023993714693, 0.002199485350158766, 0.0028138791092641195, 0.0]
    index = None

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_61_1.png" width="800" label="png">}}
    

```python
# Build the assay, based on the start and end of our baseline time, 
# and tracking the output variable index 0
assay_builder_from_dates = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=mainpipeline, 
                                          model_name="house-price-estimator", 
                                          iopath="output variable 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# update binning mode here
assay_builder_from_dates.summarizer_builder.add_bin_mode(wallaroo.assay_config.BinMode.EQUAL)

# set the width, interval, and time period 
assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)
assay_config_from_dates = assay_builder_from_dates.build()
assay_results_from_dates = assay_config_from_dates.interactive_run()

assay_results_from_dates[0].chart()
```

    baseline mean = 495193.23178642715
    window mean = 517763.394625
    baseline median = 442168.125
    window median = 448627.8125
    bin_mode = Equal
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.013362603453760629
    scores = [0.0, 0.0016737762070682225, 1.1166481947075492e-06, 0.011233704798893194, 1.276169365380064e-07, 0.00045387818266796784, 0.0]
    index = None

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_62_1.png" width="800" label="png">}}
    

The following example manually sets the bin values.

The values in this dataset run from 200000 to 1500000. We can specify the bins with the `BinMode.PROVIDED` and specifying a list of floats with the right hand / upper edge of each bin and optionally the lower edge of the smallest bin. If the lowest edge is not specified the threshold for left outliers is taken from the smallest value in the baseline dataset.

```python
# Build the assay, based on the start and end of our baseline time, 
# and tracking the output variable index 0
assay_builder_from_dates = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=mainpipeline, 
                                          model_name="house-price-estimator", 
                                          iopath="output variable 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

edges = [200000.0, 400000.0, 600000.0, 800000.0, 1500000.0, 2000000.0]

# update binning mode here
assay_builder_from_dates.summarizer_builder.add_bin_mode(wallaroo.assay_config.BinMode.PROVIDED, edges)

# set the width, interval, and time period 
assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)
assay_config_from_dates = assay_builder_from_dates.build()
assay_results_from_dates = assay_config_from_dates.interactive_run()

assay_results_from_dates[0].chart()
```

    baseline mean = 495193.23178642715
    window mean = 517763.394625
    baseline median = 442168.125
    window median = 448627.8125
    bin_mode = Provided
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.01005936099521711
    scores = [0.0, 0.0030207963288415803, 0.00011480201840874194, 0.00045327555974347976, 0.0037119550613212583, 0.0027585320269020493, 0.0]
    index = None

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_64_1.png" width="800" label="png">}}
    

##### Aggregation Options

Assay aggregation options are modified with the [`wallaroo.assay_config.AssayBuilder.add_aggregation(aggregation: wallaroo.assay_config.Aggregation)`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/assay_config/#UnivariateContinousSummarizerBuilder.add_aggregation) method.  The following options are provided:

* `Aggregation.DENSITY` (*Default*): Count the number/percentage of values that fall in each bin. 
* `Aggregation.CUMULATIVE`: Empirical Cumulative Density Function style, which keeps a cumulative count of the values/percentages that fall in each bin.

The following example demonstrate the different results between the two.

```python
#Aggregation.DENSITY - the default

# Build the assay, based on the start and end of our baseline time, 
# and tracking the output variable index 0
assay_builder_from_dates = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=mainpipeline, 
                                          model_name="house-price-estimator", 
                                          iopath="output variable 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

assay_builder_from_dates.summarizer_builder.add_aggregation(wallaroo.assay_config.Aggregation.DENSITY)

# set the width, interval, and time period 
assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)
assay_config_from_dates = assay_builder_from_dates.build()
assay_results_from_dates = assay_config_from_dates.interactive_run()

assay_results_from_dates[0].chart()
```

    baseline mean = 495193.23178642715
    window mean = 517763.394625
    baseline median = 442168.125
    window median = 448627.8125
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.0363497101644573
    scores = [0.0, 0.027271477163285655, 0.003847844548034077, 0.000217023993714693, 0.002199485350158766, 0.0028138791092641195, 0.0]
    index = None

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_66_1.png" width="800" label="png">}}
    

```python
#Aggregation.CUMULATIVE

# Build the assay, based on the start and end of our baseline time, 
# and tracking the output variable index 0
assay_builder_from_dates = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=mainpipeline, 
                                          model_name="house-price-estimator", 
                                          iopath="output variable 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

assay_builder_from_dates.summarizer_builder.add_aggregation(wallaroo.assay_config.Aggregation.CUMULATIVE)

# set the width, interval, and time period 
assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)
assay_config_from_dates = assay_builder_from_dates.build()
assay_results_from_dates = assay_config_from_dates.interactive_run()

assay_results_from_dates[0].chart()
```

    baseline mean = 495193.23178642715
    window mean = 517763.394625
    baseline median = 442168.125
    window median = 448627.8125
    bin_mode = Quantile
    aggregation = Cumulative
    metric = PSI
    weighted = False
    score = 0.17698802395209584
    scores = [0.0, 0.06759281437125747, 0.03920159680638724, 0.04579441117764471, 0.02439920159680642, 0.0, 0.0]
    index = None

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_67_1.png" width="800" label="png">}}
    

##### Inference Interval and Inference Width

The inference interval aka window interval sets how often to run the assay analysis.  This is set from the [`wallaroo.assay_config.AssayBuilder.window_builder.add_interval`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/assay_config/#WindowBuilder.add_interval) method to collect data expressed in time units:  "hours=24", "minutes=1", etc.

For example, with an interval of 1 minute, the assay collects data every minute.  Within an hour, 60 intervals of data is collected.

We can adjust the interval and see how the assays change based on how **frequently** they are run.

The width sets the time period from the [`wallaroo.assay_config.AssayBuilder.window_builder.add_width`](/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/assay_config/#WindowBuilder.add_width) method to collect data expressed in time units:  "hours=24", "minutes=1", etc.

For example, an interval of 1 minute and a width of 1 minute collects 1 minutes worth of data every minute.  An interval of 1 minute with a width of 5 minutes collects 5 minute of inference data every minute.

By default, the interval and width is **24 hours**.

For this example, we'll adjust the width and interval from 1 minute to 5 minutes and see how the number of analyses and their score changes.

```python
# Build the assay, based on the start and end of our baseline time, 
# and tracking the output variable index 0
assay_builder_from_dates = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=mainpipeline, 
                                          model_name="house-price-estimator", 
                                          iopath="output variable 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# set the width, interval, and time period 
assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)
assay_config_from_dates = assay_builder_from_dates.build()
assay_results_from_dates = assay_config_from_dates.interactive_run()

assay_results_from_dates.chart_scores()
```

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_69_0.png" width="800" label="png">}}
    

```python
# Build the assay, based on the start and end of our baseline time, 
# and tracking the output variable index 0
assay_builder_from_dates = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=mainpipeline, 
                                          model_name="house-price-estimator", 
                                          iopath="output variable 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# set the width, interval, and time period 
assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=5).add_interval(minutes=5).add_start(assay_window_start)
assay_config_from_dates = assay_builder_from_dates.build()
assay_results_from_dates = assay_config_from_dates.interactive_run()

assay_results_from_dates.chart_scores()
```

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_70_0.png" width="800" label="png">}}
    

##### Add Run Until and Add Inference Start

For previewing assays, setting [`wallaroo.assay_config.AssayBuilder.add_run_until`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/assay_config/#AssayBuilder.add_run_until) sets the **end date and time** for collecting inference data.  When an assay is uploaded, this setting is no longer valid - assays run at the [Inference Interval](#inference-interval-and-inference-width) until the [assay is paused](#pause-and-resume-assay).

Setting the [`wallaroo.assay_config.WindowBuilder.add_start`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/assay_config/#WindowBuilder.add_start) sets the **start date and time** to collect inference data.   When an assay is uploaded, this setting **is included**, and assay results will be displayed starting from that start date at the [Inference Interval](#inference-interval-and-inference-width) until the [assay is paused](#pause-and-resume-assay).  By default, `add_start` begins 24 hours after the assay is uploaded unless set in the assay configuration manually.

For the following example, the `add_run_until` setting is set to `datetime.datetime.now()` to collect all inference data from `assay_window_start` up until now, and the second example limits that example to only two minutes of data.

```python
# inference data that includes all of the data until now 

assay_builder_from_dates = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=mainpipeline, 
                                          model_name="house-price-estimator", 
                                          iopath="output variable 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# set the width, interval, and time period 
assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)
assay_config_from_dates = assay_builder_from_dates.build()
assay_results_from_dates = assay_config_from_dates.interactive_run()
assay_results_from_dates.chart_scores()
```

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_72_0.png" width="800" label="png">}}
    

```python
# inference data that includes all of the data until now 

assay_builder_from_dates = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=mainpipeline, 
                                          model_name="house-price-estimator", 
                                          iopath="output variable 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# set the width, interval, and time period 
assay_builder_from_dates.add_run_until(assay_window_start+datetime.timedelta(seconds=120))
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)
assay_config_from_dates = assay_builder_from_dates.build()
assay_results_from_dates = assay_config_from_dates.interactive_run()

assay_results_from_dates.chart_scores()
```

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_73_0.png" width="800" label="png">}}
    

### Create Assay

With the assay previewed and configuration options determined, we officially create it by uploading it to the Wallaroo instance.

Once it is uploaded, the assay runs an analysis based on the window width, interval, and the other settings configured.

Assays are uploaded with the `wallaroo.assay_config.upload()` method. This uploads the assay into the Wallaroo database with the configurations applied and returns the assay id. Note that assay names **must be unique across the Wallaroo instance**; attempting to upload an assay with the same name as an existing one will return an error.

`wallaroo.assay_config.upload()` returns the assay id for the assay.

Typically we would just call `wallaroo.assay_config.upload()` after configuring the assay.  For the example below, we will perform the complete configuration in one window to show all of the configuration steps at once before creating the assay.

```python
# Build the assay, based on the start and end of our baseline time, 
# and tracking the output variable index 0
assay_builder_from_dates = wl.build_assay(assay_name="assays creation example", 
                                          pipeline=mainpipeline, 
                                          model_name="house-price-estimator", 
                                          iopath="output variable 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# set the width, interval, and assay start date and time
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

# add other options
assay_builder_from_dates.summarizer_builder.add_aggregation(wallaroo.assay_config.Aggregation.CUMULATIVE)
assay_builder_from_dates.summarizer_builder.add_metric(wallaroo.assay_config.Metric.MAXDIFF)
assay_builder_from_dates.add_alert_threshold(0.5)

assay_id = assay_builder_from_dates.upload()

# wait 65 seconds for the first analysis run performed
time.sleep(65)

```

The assay is now visible through the Wallaroo UI by selecting the workspace, then the pipeline, then **Insights**.

{{<figure src="/images/housepricesaga-sample-assay.png" width="800" label="Sample assay in the UI">}}

### Get Assay Results

Once an assay is created the assay runs an analysis based on the window width, interval, and the other settings configured.

Assay results are retrieved with the `wallaroo.client.get_assay_results` method, which takes the following parameters:

| Parameter | Type | Description |
|---|---|---|
| **assay_id** | *Integer* (*Required*) | The numerical id of the assay. |
| **start** | *Datetime.Datetime* (*Required*) | The start date and time of historical data from the pipeline to start analyses from. |
| **end** | *Datetime.Datetime* (*Required*) | The end date and time of historical data from the pipeline to limit analyses to. |

* **IMPORTANT NOTE**:  This process requires that additional historical data is generated from the time the assay is created to when the results are available. To add additional inference data, use the [Assay Test Data](#assay-test-data) section above.

```python
assay_results = wl.get_assay_results(assay_id=assay_id,
                     start=assay_window_start,
                     end=datetime.datetime.now())

assay_results.chart_scores()
```

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_78_0.png" width="800" label="png">}}
    

```python
assay_results[0].chart()
```

    baseline mean = 495193.23178642715
    window mean = 517763.394625
    baseline median = 442168.125
    window median = 448627.8125
    bin_mode = Quantile
    aggregation = Cumulative
    metric = MaxDiff
    weighted = False
    score = 0.067592815
    scores = [0.0, 0.06759281437125747, 0.03920159680638724, 0.04579441117764471, 0.02439920159680642, 0.0, 0.0]
    index = 1

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_79_1.png" width="800" label="png">}}
    

#### List and Retrieve Assay

If the assay id is not already know, it is retrieved from the `wallaroo.client.list_assays()` method.  Select the assay to retrieve data for and retrieve its id with `wallaroo.assay.Assay._id` method.

```python
wl.list_assays()
```

<table><tr><th>name</th><th>active</th><th>status</th><th>warning_threshold</th><th>alert_threshold</th><th>pipeline_name</th></tr><tr><td>assays creation example</td><td>True</td><td>{"run_at": "2024-02-15T16:40:53.212979206+00:00",  "num_ok": 0, "num_warnings": 0, "num_alerts": 0}</td><td>None</td><td>0.5</td><td>assay-demonstration-tutorial</td></tr></table>

```python
retrieved_assay = wl.list_assays()[0]

live_assay_results = wl.get_assay_results(assay_id=retrieved_assay._id,
                     start=assay_window_start,
                     end=datetime.datetime.now())

live_assay_results.chart_scores()
```

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_82_0.png" width="800" label="png">}}
    

```python
live_assay_results[0].chart()
```

    baseline mean = 495193.23178642715
    window mean = 517763.394625
    baseline median = 442168.125
    window median = 448627.8125
    bin_mode = Quantile
    aggregation = Cumulative
    metric = MaxDiff
    weighted = False
    score = 0.067592815
    scores = [0.0, 0.06759281437125747, 0.03920159680638724, 0.04579441117764471, 0.02439920159680642, 0.0, 0.0]
    index = 1

    
{{<figure src="/images/2023.4.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_83_1.png" width="800" label="png">}}
    

### Pause and Resume Assay

Assays are paused and started with the `wallaroo.assay.Assay.turn_off` and `wallaroo.assay.Assay.turn_on` methods.

For the following, we retrieve an assay from the wallaroo instance and pause it, then list the assays to verify its setting `Active` is `False`.

```python
display(wl.list_assays())
retrieved_assay = wl.list_assays()[0]
```

<table><tr><th>name</th><th>active</th><th>status</th><th>warning_threshold</th><th>alert_threshold</th><th>pipeline_name</th></tr><tr><td>assays creation example</td><td>True</td><td>{"run_at": "2024-02-15T16:40:53.212979206+00:00",  "num_ok": 0, "num_warnings": 0, "num_alerts": 0}</td><td>None</td><td>0.5</td><td>assay-demonstration-tutorial</td></tr></table>

Now we pause the assay, and show the assay list to verify it is no longer active.

```python
retrieved_assay.turn_off()
display(wl.list_assays())
```

<table><tr><th>name</th><th>active</th><th>status</th><th>warning_threshold</th><th>alert_threshold</th><th>pipeline_name</th></tr><tr><td>assays creation example</td><td>False</td><td>{"run_at": "2024-02-15T16:40:53.212979206+00:00",  "num_ok": 0, "num_warnings": 0, "num_alerts": 0}</td><td>None</td><td>0.5</td><td>assay-demonstration-tutorial</td></tr></table>

We resume the assay and verify its setting `Active` is `True`.

```python
retrieved_assay.turn_on()
display(wl.list_assays())
```

<table><tr><th>name</th><th>active</th><th>status</th><th>warning_threshold</th><th>alert_threshold</th><th>pipeline_name</th></tr><tr><td>assays creation example</td><td>True</td><td>{"run_at": "2024-02-15T16:40:53.212979206+00:00",  "num_ok": 0, "num_warnings": 0, "num_alerts": 0}</td><td>None</td><td>0.5</td><td>assay-demonstration-tutorial</td></tr></table>

### Undeploy Main Pipeline

With the examples and tutorial complete, we will undeploy the main pipeline and return the resources back to the Wallaroo instance.

```python
mainpipeline.undeploy()
```
