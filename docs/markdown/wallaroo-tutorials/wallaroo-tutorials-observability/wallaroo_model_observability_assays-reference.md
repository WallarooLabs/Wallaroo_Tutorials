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

wl = wallaroo.Client()

wallarooPrefix = "doc-test."
wallarooSuffix = "wallarooexample.ai"

wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}api.{wallarooSuffix}", 
                    auth_endpoint=f"https://{wallarooPrefix}keycloak.{wallarooSuffix}", 
                    auth_type="sso")
```

### Create Workspace

We will create a workspace to manage our pipeline and models.  The following variables will set the name of our sample workspace then set it as the current workspace.

Workspace, pipeline, and model names should be unique to each user, so we'll add in a randomly generated suffix so multiple people can run this tutorial in a Wallaroo instance without effecting each other.

```python
workspace = get_workspace(workspace_name, wl)

wl.set_current_workspace(workspace)
```

    {'name': 'assay-demonstration-tutorial', 'id': 7, 'archived': False, 'created_by': '784e4c99-ee08-4aab-9eaa-0d8ad8e1af53', 'created_at': '2024-02-12T18:15:52.065725+00:00', 'models': [{'name': 'house-price-estimator', 'versions': 2, 'owner_id': '""', 'last_update_time': datetime.datetime(2024, 2, 13, 15, 18, 54, 437014, tzinfo=tzutc()), 'created_at': datetime.datetime(2024, 2, 12, 18, 15, 57, 462966, tzinfo=tzutc())}], 'pipelines': [{'name': 'assay-demonstration-tutorial', 'create_time': datetime.datetime(2024, 2, 12, 18, 16, 18, 335326, tzinfo=tzutc()), 'definition': '[]'}]}

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

<table><tr><th>name</th> <td>assay-demonstration-tutorial</td></tr><tr><th>created</th> <td>2024-02-12 18:16:18.335326+00:00</td></tr><tr><th>last_updated</th> <td>2024-02-14 15:42:59.384256+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>accel</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>d333456d-fd11-4652-935f-6b3e44bba1dc, 645ff11d-ca3b-49d1-83c0-ae5fffc81054, 4f108788-ff21-4ad1-83bf-7e92935d3b90, f3d5f531-86cd-4287-adc2-dfee9c9bf4c0, b6525c2a-c79d-4f36-a8d7-0f515d45e4fc, b65e67fb-135e-43ca-8ff0-96ee04d6de02</td></tr><tr><th>steps</th> <td>house-price-estimator</td></tr><tr><th>published</th> <td>False</td></tr></table>

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
      <th>anomaly.count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-02-14 15:43:16.711</td>
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
      <th>anomaly.count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-02-14 15:43:17.018</td>
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

For each of our assays, we will set the time period of inferences as part of our baseline creation.

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
      <td>1707925457831</td>
      <td>{'last_model': '{"model_name":"house-price-estimator","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}', 'pipeline_version': 'd333456d-fd11-4652-935f-6b3e44bba1dc', 'elapsed': [3357700, 2341849], 'dropped': [], 'partition': 'engine-65c8d9cf67-t529s'}</td>
      <td>3.0</td>
      <td>1.75</td>
      <td>2040.0</td>
      <td>12065.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>2040.0</td>
      <td>0.0</td>
      <td>47.375599</td>
      <td>-122.043999</td>
      <td>2010.0</td>
      <td>11717.0</td>
      <td>27.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>328513.59375</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1707925457831</td>
      <td>{'last_model': '{"model_name":"house-price-estimator","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}', 'pipeline_version': 'd333456d-fd11-4652-935f-6b3e44bba1dc', 'elapsed': [3357700, 2341849], 'dropped': [], 'partition': 'engine-65c8d9cf67-t529s'}</td>
      <td>4.0</td>
      <td>1.50</td>
      <td>1540.0</td>
      <td>7168.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>1160.0</td>
      <td>380.0</td>
      <td>47.455002</td>
      <td>-122.197998</td>
      <td>1540.0</td>
      <td>7176.0</td>
      <td>51.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>293808.03125</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1707925457831</td>
      <td>{'last_model': '{"model_name":"house-price-estimator","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}', 'pipeline_version': 'd333456d-fd11-4652-935f-6b3e44bba1dc', 'elapsed': [3357700, 2341849], 'dropped': [], 'partition': 'engine-65c8d9cf67-t529s'}</td>
      <td>4.0</td>
      <td>3.50</td>
      <td>3270.0</td>
      <td>15704.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>2110.0</td>
      <td>1160.0</td>
      <td>47.625599</td>
      <td>-122.042000</td>
      <td>3020.0</td>
      <td>8582.0</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>846775.06250</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1707925457831</td>
      <td>{'last_model': '{"model_name":"house-price-estimator","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}', 'pipeline_version': 'd333456d-fd11-4652-935f-6b3e44bba1dc', 'elapsed': [3357700, 2341849], 'dropped': [], 'partition': 'engine-65c8d9cf67-t529s'}</td>
      <td>3.0</td>
      <td>2.25</td>
      <td>1790.0</td>
      <td>11393.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>1790.0</td>
      <td>0.0</td>
      <td>47.629700</td>
      <td>-122.098999</td>
      <td>2290.0</td>
      <td>11894.0</td>
      <td>36.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>559631.06250</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1707925457831</td>
      <td>{'last_model': '{"model_name":"house-price-estimator","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}', 'pipeline_version': 'd333456d-fd11-4652-935f-6b3e44bba1dc', 'elapsed': [3357700, 2341849], 'dropped': [], 'partition': 'engine-65c8d9cf67-t529s'}</td>
      <td>5.0</td>
      <td>3.25</td>
      <td>4115.0</td>
      <td>7910.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>4115.0</td>
      <td>0.0</td>
      <td>47.684700</td>
      <td>-122.015999</td>
      <td>3950.0</td>
      <td>6765.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>988481.87500</td>
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
      <th>495</th>
      <td>1707925457831</td>
      <td>{'last_model': '{"model_name":"house-price-estimator","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}', 'pipeline_version': 'd333456d-fd11-4652-935f-6b3e44bba1dc', 'elapsed': [3357700, 2341849], 'dropped': [], 'partition': 'engine-65c8d9cf67-t529s'}</td>
      <td>4.0</td>
      <td>2.50</td>
      <td>1820.0</td>
      <td>3899.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>1820.0</td>
      <td>0.0</td>
      <td>47.735001</td>
      <td>-121.985001</td>
      <td>1820.0</td>
      <td>3899.0</td>
      <td>16.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>437177.96875</td>
    </tr>
    <tr>
      <th>496</th>
      <td>1707925457831</td>
      <td>{'last_model': '{"model_name":"house-price-estimator","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}', 'pipeline_version': 'd333456d-fd11-4652-935f-6b3e44bba1dc', 'elapsed': [3357700, 2341849], 'dropped': [], 'partition': 'engine-65c8d9cf67-t529s'}</td>
      <td>3.0</td>
      <td>2.50</td>
      <td>1520.0</td>
      <td>4170.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>1520.0</td>
      <td>0.0</td>
      <td>47.384201</td>
      <td>-122.040001</td>
      <td>1560.0</td>
      <td>4237.0</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>261886.93750</td>
    </tr>
    <tr>
      <th>497</th>
      <td>1707925457831</td>
      <td>{'last_model': '{"model_name":"house-price-estimator","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}', 'pipeline_version': 'd333456d-fd11-4652-935f-6b3e44bba1dc', 'elapsed': [3357700, 2341849], 'dropped': [], 'partition': 'engine-65c8d9cf67-t529s'}</td>
      <td>4.0</td>
      <td>3.50</td>
      <td>3450.0</td>
      <td>7832.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>3450.0</td>
      <td>0.0</td>
      <td>47.563702</td>
      <td>-122.123001</td>
      <td>3220.0</td>
      <td>8567.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>921695.37500</td>
    </tr>
    <tr>
      <th>498</th>
      <td>1707925457831</td>
      <td>{'last_model': '{"model_name":"house-price-estimator","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}', 'pipeline_version': 'd333456d-fd11-4652-935f-6b3e44bba1dc', 'elapsed': [3357700, 2341849], 'dropped': [], 'partition': 'engine-65c8d9cf67-t529s'}</td>
      <td>3.0</td>
      <td>1.75</td>
      <td>1680.0</td>
      <td>8100.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>1680.0</td>
      <td>0.0</td>
      <td>47.721199</td>
      <td>-122.363998</td>
      <td>1880.0</td>
      <td>7750.0</td>
      <td>65.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>437177.96875</td>
    </tr>
    <tr>
      <th>499</th>
      <td>1707925457831</td>
      <td>{'last_model': '{"model_name":"house-price-estimator","model_sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}', 'pipeline_version': 'd333456d-fd11-4652-935f-6b3e44bba1dc', 'elapsed': [3357700, 2341849], 'dropped': [], 'partition': 'engine-65c8d9cf67-t529s'}</td>
      <td>3.0</td>
      <td>1.50</td>
      <td>1430.0</td>
      <td>9600.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>1430.0</td>
      <td>0.0</td>
      <td>47.473701</td>
      <td>-122.150002</td>
      <td>1590.0</td>
      <td>10240.0</td>
      <td>48.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>260603.87500</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 21 columns</p>

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
      <td>328513.60</td>
    </tr>
    <tr>
      <th>1</th>
      <td>293808.03</td>
    </tr>
    <tr>
      <th>2</th>
      <td>846775.06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>559631.06</td>
    </tr>
    <tr>
      <th>4</th>
      <td>988481.90</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>495</th>
      <td>437177.97</td>
    </tr>
    <tr>
      <th>496</th>
      <td>261886.94</td>
    </tr>
    <tr>
      <th>497</th>
      <td>921695.40</td>
    </tr>
    <tr>
      <th>498</th>
      <td>437177.97</td>
    </tr>
    <tr>
      <th>499</th>
      <td>260603.88</td>
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
      <td>500</td>
    </tr>
    <tr>
      <th>min</th>
      <td>236238.671875</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1489624.25</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>528747.089375</td>
    </tr>
    <tr>
      <th>median</th>
      <td>450867.6875</td>
    </tr>
    <tr>
      <th>std</th>
      <td>233850.215184</td>
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
      <td>528747.0917</td>
    </tr>
    <tr>
      <th>median</th>
      <td>450867.7</td>
    </tr>
    <tr>
      <th>std</th>
      <td>233850.218373</td>
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

#### Baseline Histogram Chart

The method [`wallaroo.assay_config.AssayBuilder.baseline_histogram`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/assay_config/#AssayBuilder.baseline_dataframe) returns a histogram chart of the assay baseline generated from the provided parameters.

```python
assay_builder_from_dates.baseline_histogram()
```

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_32_0.png" width="800" label="png">}}
    

#### Baseline KDE Chart

The method [`wallaroo.assay_config.AssayBuilder.baseline_kde`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/assay_config/#AssayBuilder.baseline_dataframe) returns a Kernel Density Estimation (KDE) chart of the assay baseline generated from the provided parameters.

```python
assay_builder_from_dates.baseline_kde()
```

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_34_0.png" width="800" label="png">}}
    

#### Baseline ECDF Chart

The method [`wallaroo.assay_config.AssayBuilder.baseline_ecdf`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/assay_config/#AssayBuilder.baseline_dataframe) returns a Empirical Cumulative Distribution Function (CDF) chart of the assay baseline generated from the provided parameters.

```python
assay_builder_from_dates.baseline_ecdf()
```

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_36_0.png" width="800" label="png">}}
    

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

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_39_0.png" width="800" label="png">}}
    

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

For this example, the assay analysis list DataFrame is listed.  For space reasons, only the `score`, `start`, `alert_threshold` and `status` are shown.

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
      <td>0.015143</td>
      <td>2024-02-14T15:45:17.990389+00:00</td>
      <td>2.362387e+05</td>
      <td>1489624.250</td>
      <td>5.031280e+05</td>
      <td>4.486278e+05</td>
      <td>209423.814215</td>
      <td>None</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>1</th>
      <td>None</td>
      <td></td>
      <td></td>
      <td>8.869797</td>
      <td>2024-02-14T15:47:17.990389+00:00</td>
      <td>1.514079e+06</td>
      <td>2016006.125</td>
      <td>1.888533e+06</td>
      <td>1.946438e+06</td>
      <td>157326.685330</td>
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
      <td>2024-02-14T15:45:17.990389+00:00</td>
      <td>2024-02-14T15:50:42.192173+00:00</td>
      <td>106</td>
      <td>500</td>
      <td>236238.671875</td>
      <td>1489624.25</td>
      <td>528747.089375</td>
      <td>450867.6875</td>
      <td>233850.215184</td>
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
      <td>2024-02-14T15:47:17.990389+00:00</td>
      <td>2024-02-14T15:50:42.192332+00:00</td>
      <td>111</td>
      <td>500</td>
      <td>236238.671875</td>
      <td>1489624.25</td>
      <td>528747.089375</td>
      <td>450867.6875</td>
      <td>233850.215184</td>
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

#### Assay Analysis Chart

The method `wallaroo.assay.AssayAnalysis.chart()` displays a comparison between the baseline and an analysis window.  The following fields are included.

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

    baseline mean = 528747.089375
    window mean = 503127.956625
    baseline median = 450867.6875
    window median = 448627.8125
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.01514337711118979
    scores = [0.0, 0.0, 0.0021338298517208126, 0.004531207722385977, 0.0013341057430248191, 0.00714423379405818, 0.0]
    index = None

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_45_1.png" width="800" label="png">}}
    

#### Score Metric

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

    baseline mean = 528747.089375
    window mean = 503127.956625
    baseline median = 450867.6875
    window median = 448627.8125
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.01514337711118979
    scores = [0.0, 0.0, 0.0021338298517208126, 0.004531207722385977, 0.0013341057430248191, 0.00714423379405818, 0.0]
    index = None

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_47_1.png" width="800" label="png">}}
    

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

    baseline mean = 528747.089375
    window mean = 503127.956625
    baseline median = 450867.6875
    window median = 448627.8125
    bin_mode = Quantile
    aggregation = Density
    metric = MaxDiff
    weighted = False
    score = 0.036000000000000004
    scores = [0.0, 0.0, 0.021999999999999992, 0.03, 0.016000000000000014, 0.036000000000000004, 0.0]
    index = 5

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_48_1.png" width="800" label="png">}}
    

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

    baseline mean = 497321.3875686128
    window mean = 522256.02471875
    baseline median = 448627.8125
    window median = 450867.6875
    bin_mode = Quantile
    aggregation = Density
    metric = SumDiff
    weighted = False
    score = 0.045395209580838314
    scores = [0.0, 0.02060079840319362, 0.004596806387225538, 0.014600798403193616, 0.005596806387225539, 0.045395209580838314, 0.0]
    index = None

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_49_1.png" width="800" label="png">}}
    

#### Alert Threshold

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
      <td>0.015143</td>
      <td>2024-02-14T15:45:17.990389+00:00</td>
      <td>2.362387e+05</td>
      <td>1489624.250</td>
      <td>5.031280e+05</td>
      <td>4.486278e+05</td>
      <td>209423.814215</td>
      <td>None</td>
      <td>0.5</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>1</th>
      <td>None</td>
      <td></td>
      <td></td>
      <td>8.869797</td>
      <td>2024-02-14T15:47:17.990389+00:00</td>
      <td>1.514079e+06</td>
      <td>2016006.125</td>
      <td>1.888533e+06</td>
      <td>1.946438e+06</td>
      <td>157326.685330</td>
      <td>None</td>
      <td>0.5</td>
      <td>Alert</td>
    </tr>
  </tbody>
</table>

#### Number of Bins

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

    baseline mean = 528747.089375
    window mean = 503127.956625
    baseline median = 450867.6875
    window median = 448627.8125
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.0263329428544279
    scores = [0.0, 0.00047823470784233907, 0.0005027801419987897, 0.0073846243328338414, 0.0004356961621405474, 0.0008399874581299729, 0.004317584074929309, 0.0022886938139282556, 1.0695289116747927e-05, 0.0006534242481141364, 0.00942122262539396, 0.0]
    index = None

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_53_1.png" width="800" label="png">}}
    

#### Binning Mode

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

    baseline mean = 528747.089375
    window mean = 503127.956625
    baseline median = 450867.6875
    window median = 448627.8125
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.01514337711118979
    scores = [0.0, 0.0, 0.0021338298517208126, 0.004531207722385977, 0.0013341057430248191, 0.00714423379405818, 0.0]
    index = None

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_55_1.png" width="800" label="png">}}
    

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

    baseline mean = 497321.3875686128
    window mean = 522256.02471875
    baseline median = 448627.8125
    window median = 450867.6875
    bin_mode = Equal
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.02539819897479689
    scores = [0.0, 0.002342946191184832, 0.00010214878249991722, 0.0001346137635990256, 0.021852018042416495, 0.0009664721950966164, 0.0]
    index = None

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_56_1.png" width="800" label="png">}}
    

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

    baseline mean = 528747.089375
    window mean = 503127.956625
    baseline median = 450867.6875
    window median = 448627.8125
    bin_mode = Provided
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.01967715317306405
    scores = [0.0, 0.0029546155739340755, 0.00015764056863465774, 5.865119452398063e-06, 0.016559031911042917, 0.0, 0.0]
    index = None

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_58_1.png" width="800" label="png">}}
    

#### Aggregation Options

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

    baseline mean = 528747.089375
    window mean = 503127.956625
    baseline median = 450867.6875
    window median = 448627.8125
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.01514337711118979
    scores = [0.0, 0.0, 0.0021338298517208126, 0.004531207722385977, 0.0013341057430248191, 0.00714423379405818, 0.0]
    index = None

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_60_1.png" width="800" label="png">}}
    

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

    baseline mean = 528747.089375
    window mean = 503127.956625
    baseline median = 450867.6875
    window median = 448627.8125
    bin_mode = Quantile
    aggregation = Cumulative
    metric = PSI
    weighted = False
    score = 0.11
    scores = [0.0, 0.0, 0.02200000000000002, 0.052000000000000046, 0.03599999999999992, 0.0, 0.0]
    index = None

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_61_1.png" width="800" label="png">}}
    

#### Compare Basic Stats

The method `wallaroo.assay.AssayAnalysis.compare_basic_stats` returns a DataFrame comparing the baseline with the window.

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

    baseline mean = 528747.089375
    window mean = 503127.956625
    baseline median = 450867.6875
    window median = 448627.8125
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.01514337711118979
    scores = [0.0, 0.0, 0.0021338298517208126, 0.004531207722385977, 0.0013341057430248191, 0.00714423379405818, 0.0]
    index = None

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_63_1.png" width="800" label="png">}}
    

#### Window Interval and Window Width

The window interval sets how often to run the assay analysis.  This is set from the [`wallaroo.assay_config.AssayBuilder.window_builder.add_interval`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/assay_config/#WindowBuilder.add_interval) method to collect data expressed in time units:  "hours=24", "minutes=1", etc.

We can adjust the interval and see how the assays change based on how **frequently** they are run.

The width sets the time period from the [`wallaroo.assay_config.AssayBuilder.window_builder.add_width`](/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/assay_config/#WindowBuilder.add_width) method to collect data expressed in time units:  "hours=24", "minutes=1", etc.

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

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_65_0.png" width="800" label="png">}}
    

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

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_66_0.png" width="800" label="png">}}
    

### Create Assay

With the assay created and fully tested, we officially create it by uploading it to the Wallaroo instance.  Once it is uploaded, the assay runs an analysis based on the window width, interval, and the other settings configured.

Assays are uploaded with the `wallaroo.assay_config.upload()` method. This uploads the assay into the Wallaroo database with the configurations applied and returns the assay id. Note that assay names **must be unique across the Wallaroo instance**; attempting to upload an assay with the same name as an existing one will return an error.

`wallaroo.assay_config.upload()` returns the assay id for the assay. 

```python
# Build the assay, based on the start and end of our baseline time, 
# and tracking the output variable index 0
assay_builder_from_dates = wl.build_assay(assay_name="assays creation example", 
                                          pipeline=mainpipeline, 
                                          model_name="house-price-estimator", 
                                          iopath="output variable 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# set the width, interval, and time period 
assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

assay_id = assay_builder_from_dates.upload()

# wait 60 seconds for the first analysis run performed
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

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_71_0.png" width="800" label="png">}}
    

```python
assay_results[0].chart()
```

    baseline mean = 528747.089375
    window mean = 503127.956625
    baseline median = 450867.6875
    window median = 448627.8125
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.015143377
    scores = [0.0, 0.0, 0.0021338298517208126, 0.004531207722385977, 0.0013341057430248191, 0.00714423379405818, 0.0]
    index = None

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_72_1.png" width="800" label="png">}}
    

#### List and Retrieve Assay

If the assay id is not already know, it is retrieved from the `wallaroo.client.list_assays()` method.  Select the assay to retrieve data for and retrieve its id with `wallaroo.assay.Assay._id` method.

```python
wl.list_assays()
```

<table><tr><th>Assay ID</th><th>Assay Name</th><th>Active</th><th>Status</th><th>Warning Threshold</th><th>Alert Threshold</th><th>Pipeline ID</th><th>Pipeline Name</th></tr><tr><td>4</td><td>assays creation example</td><td>True</td><td>{"run_at": "2024-02-14T16:01:40.292769112+00:00",  "num_ok": 0, "num_warnings": 0, "num_alerts": 0}</td><td>None</td><td>0.25</td><td>7</td><td>assay-demonstration-tutorial</td></tr><tr><td>3</td><td>assays from date sample</td><td>True</td><td>{"run_at": "2024-02-14T16:01:40.318747428+00:00",  "num_ok": 0, "num_warnings": 0, "num_alerts": 0}</td><td>None</td><td>0.25</td><td>7</td><td>assay-demonstration-tutorial</td></tr><tr><td>1</td><td>assays from date baseline create sample</td><td>True</td><td>{"run_at": "2024-02-14T16:01:40.305696152+00:00",  "num_ok": 0, "num_warnings": 0, "num_alerts": 0}</td><td>None</td><td>0.25</td><td>7</td><td>assay-demonstration-tutorial</td></tr></table>

```python
retrieved_assay = wl.list_assays()[0]

assay_results = wl.get_assay_results(assay_id=retrieved_assay._id,
                     start=assay_window_start,
                     end=datetime.datetime.now())

assay_results.chart_scores()

```

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_75_0.png" width="800" label="png">}}
    

### Pause and Resume Assay

Assays are paused and started with the `wallaroo.assay.Assay.turn_off` and `wallaroo.assay.Assay.turn_on` methods.

For the following, we retrieve an assay from the wallaroo instance and pause it, then list the assays to verify its setting `Active` is `False`.

```python
display(wl.list_assays())
retrieved_assay = wl.list_assays()[0]

retrieved_assay.turn_off()
display(wl.list_assays())
```

<table><tr><th>Assay ID</th><th>Assay Name</th><th>Active</th><th>Status</th><th>Warning Threshold</th><th>Alert Threshold</th><th>Pipeline ID</th><th>Pipeline Name</th></tr><tr><td>4</td><td>assays creation example</td><td>True</td><td>{"run_at": "2024-02-14T16:02:40.314790278+00:00",  "num_ok": 0, "num_warnings": 0, "num_alerts": 0}</td><td>None</td><td>0.25</td><td>7</td><td>assay-demonstration-tutorial</td></tr><tr><td>3</td><td>assays from date sample</td><td>True</td><td>{"run_at": "2024-02-14T16:02:40.295722964+00:00",  "num_ok": 0, "num_warnings": 0, "num_alerts": 0}</td><td>None</td><td>0.25</td><td>7</td><td>assay-demonstration-tutorial</td></tr><tr><td>1</td><td>assays from date baseline create sample</td><td>True</td><td>{"run_at": "2024-02-14T16:02:40.328304342+00:00",  "num_ok": 0, "num_warnings": 0, "num_alerts": 0}</td><td>None</td><td>0.25</td><td>7</td><td>assay-demonstration-tutorial</td></tr></table>

<table><tr><th>Assay ID</th><th>Assay Name</th><th>Active</th><th>Status</th><th>Warning Threshold</th><th>Alert Threshold</th><th>Pipeline ID</th><th>Pipeline Name</th></tr><tr><td>4</td><td>assays creation example</td><td>False</td><td>{"run_at": "2024-02-14T16:02:40.314790278+00:00",  "num_ok": 0, "num_warnings": 0, "num_alerts": 0}</td><td>None</td><td>0.25</td><td>7</td><td>assay-demonstration-tutorial</td></tr><tr><td>3</td><td>assays from date sample</td><td>True</td><td>{"run_at": "2024-02-14T16:02:40.295722964+00:00",  "num_ok": 0, "num_warnings": 0, "num_alerts": 0}</td><td>None</td><td>0.25</td><td>7</td><td>assay-demonstration-tutorial</td></tr><tr><td>1</td><td>assays from date baseline create sample</td><td>True</td><td>{"run_at": "2024-02-14T16:02:40.328304342+00:00",  "num_ok": 0, "num_warnings": 0, "num_alerts": 0}</td><td>None</td><td>0.25</td><td>7</td><td>assay-demonstration-tutorial</td></tr></table>

We resume the assay and verify its setting `Active` is `True`.

```python
retrieved_assay.turn_on()
display(wl.list_assays())
```

<table><tr><th>Assay ID</th><th>Assay Name</th><th>Active</th><th>Status</th><th>Warning Threshold</th><th>Alert Threshold</th><th>Pipeline ID</th><th>Pipeline Name</th></tr><tr><td>4</td><td>assays creation example</td><td>True</td><td>{"run_at": "2024-02-14T16:02:40.314790278+00:00",  "num_ok": 0, "num_warnings": 0, "num_alerts": 0}</td><td>None</td><td>0.25</td><td>7</td><td>assay-demonstration-tutorial</td></tr><tr><td>3</td><td>assays from date sample</td><td>True</td><td>{"run_at": "2024-02-14T16:02:40.295722964+00:00",  "num_ok": 0, "num_warnings": 0, "num_alerts": 0}</td><td>None</td><td>0.25</td><td>7</td><td>assay-demonstration-tutorial</td></tr><tr><td>1</td><td>assays from date baseline create sample</td><td>True</td><td>{"run_at": "2024-02-14T16:02:40.328304342+00:00",  "num_ok": 0, "num_warnings": 0, "num_alerts": 0}</td><td>None</td><td>0.25</td><td>7</td><td>assay-demonstration-tutorial</td></tr></table>

### Undeploy Main Pipeline

With the examples and tutorial complete, we will undeploy the main pipeline and return the resources back to the Wallaroo instance.

```python
mainpipeline.undeploy()
```

<table><tr><th>name</th> <td>assay-demonstration-tutorial</td></tr><tr><th>created</th> <td>2024-02-12 18:16:18.335326+00:00</td></tr><tr><th>last_updated</th> <td>2024-02-14 15:42:59.384256+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>accel</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>d333456d-fd11-4652-935f-6b3e44bba1dc, 645ff11d-ca3b-49d1-83c0-ae5fffc81054, 4f108788-ff21-4ad1-83bf-7e92935d3b90, f3d5f531-86cd-4287-adc2-dfee9c9bf4c0, b6525c2a-c79d-4f36-a8d7-0f515d45e4fc, b65e67fb-135e-43ca-8ff0-96ee04d6de02</td></tr><tr><th>steps</th> <td>house-price-estimator</td></tr><tr><th>published</th> <td>False</td></tr></table>

