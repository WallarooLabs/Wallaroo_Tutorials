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

    {'name': 'assay-demonstration-tutorial', 'id': 7, 'archived': False, 'created_by': '784e4c99-ee08-4aab-9eaa-0d8ad8e1af53', 'created_at': '2024-02-12T18:15:52.065725+00:00', 'models': [], 'pipelines': []}

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

<table><tr><th>name</th> <td>assay-demonstration-tutorial</td></tr><tr><th>created</th> <td>2024-02-12 18:16:18.335326+00:00</td></tr><tr><th>last_updated</th> <td>2024-02-12 18:16:19.100769+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>accel</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>b6525c2a-c79d-4f36-a8d7-0f515d45e4fc, b65e67fb-135e-43ca-8ff0-96ee04d6de02</td></tr><tr><th>steps</th> <td>house-price-estimator</td></tr><tr><th>published</th> <td>False</td></tr></table>

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
      <td>2024-02-12 18:16:35.463</td>
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
      <td>2024-02-12 18:16:35.773</td>
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

```python
# Get a regular spread of values

time.sleep(65)
inference_size = 1000

# And a spread of regular values

big_houses_inputs = pd.read_json('./data/houseprice_5000_data.df.json', orient="records")
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
| **baseline_start** | *datetime.datetime* (*Optional*) | The start time for the inferences to use as the baseline.  **Must be included with `baseline_end`.  Cannot be included with `baseline_data`.** |
| **baseline_end** | *datetime.datetime* (*Optional*) | The end time of the baseline window. the baseline. Windows start immediately after the baseline window and are run at regular intervals continuously until the assay is deactivated or deleted.  **Must be included with `baseline_start`.  Cannot be included with `baseline_data`.**. |
| **baseline_data** | *numpy.array* (*Optional*) | The baseline data in numpy array format.  **Cannot be included with either `baseline_start` or `baseline_data`. |

Baselines are created in one of two ways:

* **Date Range**:  The `baseline_start` and `baseline_end` retrieves the inference requests and results for the pipeline from the start and end period.  This data is summarized and used to create the baseline.
* **Numpy Values**:  The `baseline_data` sets the baseline from a provided numpy array.

#### Define the Baseline Example

This example shows two methods of defining the baseline for an assay:

* `"assays from date baseline"`: This assay uses historical inference requests to define the baseline.  This assay is saved to the variable `assay_builder_from_dates`.
* `"assays from numpy"`:  This assay uses a pre-generated numpy array to define the baseline.  This assay is saved to the variable `assay_builder_from_numpy`.

In both cases, the following parameters are used:

| Parameter | Value |
|---|---|---|
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

# set the width, interval, and time period for the assay interactive run
assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)
baseline_from_dates_run = assay_builder_from_dates.build().interactive_run()[0]

# assay builder by baseline
assay_builder_from_numpy = wl.build_assay(assay_name="assays from numpy", 
                               pipeline=mainpipeline, 
                               model_name="house-price-estimator", 
                               iopath="output variable 0", 
                               baseline_data = small_results_baseline)

# set the width, interval, and time period for the assay interactive run
assay_builder_from_numpy.add_run_until(datetime.datetime.now())
assay_builder_from_numpy.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

baseline_from_numpy_run = assay_builder_from_numpy.build().interactive_run()[0]
```

With the baseline's of each assay generated, examine the data and generate some visual representations.

#### Baseline Histogram Chart

The method [`wallaroo.assay_config.AssayBuilder.baseline_histogram`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/assay_config/#AssayBuilder.baseline_dataframe) returns a histogram chart of the assay baseline generated from the provided parameters.

```python
assay_builder_from_dates.baseline_histogram()
```

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_28_0.png" width="800" label="png">}}
    

```python
assay_builder_from_numpy.baseline_histogram()
```

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_29_0.png" width="800" label="png">}}
    

#### Baseline KDE Chart

The method [`wallaroo.assay_config.AssayBuilder.baseline_kde`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/assay_config/#AssayBuilder.baseline_dataframe) returns a Kernel Density Estimation (KDE) chart of the assay baseline generated from the provided parameters.

```python
assay_builder_from_dates.baseline_kde()
```

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_31_0.png" width="800" label="png">}}
    

```python
assay_builder_from_numpy.baseline_kde()
```

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_32_0.png" width="800" label="png">}}
    

#### Baseline ECDF Chart

The method [`wallaroo.assay_config.AssayBuilder.baseline_ecdf`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/assay_config/#AssayBuilder.baseline_dataframe) returns a Empirical Cumulative Distribution Function (CDF) chart of the assay baseline generated from the provided parameters.

```python
assay_builder_from_dates.baseline_ecdf()
```

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_34_0.png" width="800" label="png">}}
    

```python
assay_builder_from_numpy.baseline_ecdf()
```

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_35_0.png" width="800" label="png">}}
    

#### Baseline DataFrame

The method [`wallaroo.assay_config.AssayBuilder.baseline_ecdf`](/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/assay_config/#AssayBuilder.baseline_dataframe) returns a dataframe of the assay baseline generated from the provided parameters.  This includes:

* `metadata`:  The inference metadata with the model information, inference time, and other related factors.
* `in` data:  Each input field assigned with the label `in.{input field name}`.
* `out` data:  Each output field assigned with the label `out.{output field name}`

Note that for assays generated from numpy values, there is only the `out` data based on the supplied baseline data.

In the following example, the baseline DataFrame is retrieved.  For space purposes, only the `time` and output variable is displayed for the baseline generated by dates.

```python
assay_builder_from_dates.baseline_dataframe().loc[:, ['time', 'output_variable_0']]
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>output_variable_0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1707761795773</td>
      <td>1.514079e+06</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1707761856628</td>
      <td>3.553711e+05</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1707761856628</td>
      <td>4.467690e+05</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1707761856628</td>
      <td>2.425914e+05</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1707761856628</td>
      <td>2.380780e+05</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>496</th>
      <td>1707761856628</td>
      <td>4.958226e+05</td>
    </tr>
    <tr>
      <th>497</th>
      <td>1707761856628</td>
      <td>4.356287e+05</td>
    </tr>
    <tr>
      <th>498</th>
      <td>1707761856628</td>
      <td>8.467751e+05</td>
    </tr>
    <tr>
      <th>499</th>
      <td>1707761856628</td>
      <td>3.407645e+05</td>
    </tr>
    <tr>
      <th>500</th>
      <td>1707761856628</td>
      <td>3.407645e+05</td>
    </tr>
  </tbody>
</table>
<p>501 rows × 2 columns</p>

```python
assay_builder_from_numpy.baseline_dataframe().loc[:, ['output_variable_0']]
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
      <td>355371.10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>446769.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>242591.38</td>
    </tr>
    <tr>
      <th>3</th>
      <td>238078.03</td>
    </tr>
    <tr>
      <th>4</th>
      <td>467484.30</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>495</th>
      <td>495822.63</td>
    </tr>
    <tr>
      <th>496</th>
      <td>435628.72</td>
    </tr>
    <tr>
      <th>497</th>
      <td>846775.06</td>
    </tr>
    <tr>
      <th>498</th>
      <td>340764.53</td>
    </tr>
    <tr>
      <th>499</th>
      <td>340764.53</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 1 columns</p>

#### Baseline Stats

The method `wallaroo.assay.AssayAnalysis.baseline_stats()` returns a `pandas.core.frame.DataFrame` of the baseline stats.

The baseline stats for each assay are displayed in the examples below.

```python
baseline_from_dates_run.baseline_stats()
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
      <td>514078.818363</td>
    </tr>
    <tr>
      <th>median</th>
      <td>450867.6875</td>
    </tr>
    <tr>
      <th>std</th>
      <td>231664.973507</td>
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
baseline_from_numpy_run.baseline_stats()
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
      <td>1322835.6</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>512078.81888</td>
    </tr>
    <tr>
      <th>median</th>
      <td>450867.7</td>
    </tr>
    <tr>
      <th>std</th>
      <td>227534.604401</td>
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

### Assay Preview

Now that the baseline is defined, we look at different configuration options and view how the assay baseline and results changes.  Once we determine what gives us the best method of determining model drift, we can create the assay.

For each example two analyses are displayed.  Both are generated from sample data with a width and interval of one minute.

* `assay_builder_from_dates.build().interactive_run()[0]`: This analysis is similar to the baseline and generates a low score.
* `assay_builder_from_dates.build().interactive_run()[1]`: This analysis is generated from values that are very different than the baseline, which generates a high score.

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

# set the width, interval, and time period for the assay interactive run
assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

assay_builder_from_dates.build().interactive_run().chart_scores()
```

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_44_0.png" width="800" label="png">}}
    

#### Analysis List DataFrame

`wallaroo.assay.AssayAnalysisList.to_dataframe()` returns a DataFrame showing the assay results for each window aka individual analysis.  This DataFrame contains the following fields:

| Field | Type | Description |
|---|---|---|
| assay_id | **Integer/None** | The assay id.  If this is from an interactive run and not an uploaded assay, the id is `None`.
| name | **String/None** | The name of the assay.  If this is from an interactive run and not an uploaded assay, the name is `None`.
| iopath | **String/None** | The iopath of the assay.  If this is from an interactive run and not an uploaded assay, the iopath is `None`.
| score | **Float** | The assay score. |
| start | **DateTime** | The DateTime start of the assay window.
| min | **Float** | The minimum value in the assay window.
| max  | **Float** | The maximum value in the assay window.
| mean | **Float** | The mean value in the assay window.
| median | **Float** | The median value in the assay window.
| std | **Float** | The standard deviation value in the assay window.
| warning_threshold | **Float/None** | The warning threshold of the assay window.
| alert_threshold | **Float/None** | The alert threshold of the assay window.
| status | **String** | The assay window status.  Values are:  <ul><li>`OK`: The score is within accepted thresholds.</li><li>`Warning`: The score has triggered the `warning_threshold` if exists, but not the `alert_threshold`.</li><li>`Alert`: The score has triggered the the `alert_threshold`.</li></ul> |

For this example, the assay analysis list DataFrame is listed from an interactive run.  For space reasons, only the `score`, `start`, `alert_threshold` and `status` are shown.

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

# set the width, interval, and time period for the assay interactive run
assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)
baseline_from_dates_run = assay_builder_from_dates.build().interactive_run()[1]

assay_builder_from_dates.build().interactive_run().to_dataframe().loc[:, ['score', 'start', 'alert_threshold', 'status']]
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
      <td>0.010825</td>
      <td>2024-02-12T18:18:36.905868+00:00</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.868920</td>
      <td>2024-02-12T18:20:36.905868+00:00</td>
      <td>0.25</td>
      <td>Alert</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.027985</td>
      <td>2024-02-12T18:28:36.905868+00:00</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
  </tbody>
</table>

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

# set the width, interval, and time period for the assay interactive run
assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

assay_builder_from_dates.build().interactive_run()[0].chart()
assay_builder_from_dates.build().interactive_run()[1].chart()
```

    baseline mean = 514078.81836327346
    window mean = 515981.794078125
    baseline median = 450867.6875
    window median = 448627.8125
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.010824679818960396
    scores = [0.0, 0.002969815172154558, 0.005578645912763821, 0.0002298040928465568, 0.0013956367196008871, 0.0006507779215945722, 0.0]
    index = None

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_48_1.png" width="800" label="png">}}
    

    baseline mean = 514078.81836327346
    window mean = 1883044.252
    baseline median = 450867.6875
    window median = 1946437.75
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 8.868919890469542
    scores = [0.0, 0.7174700740703411, 0.7548517748433896, 0.6896479439992407, 0.7454763738420332, 0.6896479439992407, 5.2718257797152965]
    index = None

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_48_3.png" width="800" label="png">}}
    

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

# set the width, interval, and time period for the assay interactive run
assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

assay_builder_from_dates.build().interactive_run()[0].chart()
assay_builder_from_dates.build().interactive_run()[1].chart()
```

    baseline mean = 514078.81836327346
    window mean = 515981.794078125
    baseline median = 450867.6875
    window median = 448627.8125
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.010824679818960396
    scores = [0.0, 0.002969815172154558, 0.005578645912763821, 0.0002298040928465568, 0.0013956367196008871, 0.0006507779215945722, 0.0]
    index = None

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_50_1.png" width="800" label="png">}}
    

    baseline mean = 514078.81836327346
    window mean = 1883044.252
    baseline median = 450867.6875
    window median = 1946437.75
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 8.868919890469542
    scores = [0.0, 0.7174700740703411, 0.7548517748433896, 0.6896479439992407, 0.7454763738420332, 0.6896479439992407, 5.2718257797152965]
    index = None

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_50_3.png" width="800" label="png">}}
    

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

# set the width, interval, and time period for the assay interactive run
assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

assay_builder_from_dates.build().interactive_run()[0].chart()
assay_builder_from_dates.build().interactive_run()[1].chart()
```

    baseline mean = 514078.81836327346
    window mean = 515981.794078125
    baseline median = 450867.6875
    window median = 448627.8125
    bin_mode = Quantile
    aggregation = Density
    metric = MaxDiff
    weighted = False
    score = 0.03541516966067865
    scores = [0.0, 0.023600798403193624, 0.03541516966067865, 0.006612774451097814, 0.016588822355289412, 0.011387225548902176, 0.0]
    index = 2

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_51_1.png" width="800" label="png">}}
    

    baseline mean = 514078.81836327346
    window mean = 1883044.252
    baseline median = 450867.6875
    window median = 1946437.75
    bin_mode = Quantile
    aggregation = Density
    metric = MaxDiff
    weighted = False
    score = 1.0
    scores = [0.0, 0.1996007984031936, 0.20758483033932135, 0.1936127744510978, 0.2055888223552894, 0.1936127744510978, 1.0]
    index = 6

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_51_3.png" width="800" label="png">}}
    

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

# set the width, interval, and time period for the assay interactive run
assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

assay_builder_from_dates.build().interactive_run()[0].chart()
assay_builder_from_dates.build().interactive_run()[1].chart()
```

    baseline mean = 514078.81836327346
    window mean = 515981.794078125
    baseline median = 450867.6875
    window median = 448627.8125
    bin_mode = Quantile
    aggregation = Density
    metric = SumDiff
    weighted = False
    score = 0.046802395209580835
    scores = [0.0, 0.023600798403193624, 0.03541516966067865, 0.006612774451097814, 0.016588822355289412, 0.011387225548902176, 0.0]
    index = None

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_52_1.png" width="800" label="png">}}
    

    baseline mean = 514078.81836327346
    window mean = 1883044.252
    baseline median = 450867.6875
    window median = 1946437.75
    bin_mode = Quantile
    aggregation = Density
    metric = SumDiff
    weighted = False
    score = 1.0
    scores = [0.0, 0.1996007984031936, 0.20758483033932135, 0.1936127744510978, 0.2055888223552894, 0.1936127744510978, 1.0]
    index = None

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_52_3.png" width="800" label="png">}}
    

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

# set the width, interval, and time period for the assay interactive run
assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

# get the dataframe from the interactive run
assay_builder_from_dates.build().interactive_run().to_dataframe().loc[:, ['score', 'start', 'alert_threshold', 'status']]
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
      <td>0.010825</td>
      <td>2024-02-12T18:18:36.905868+00:00</td>
      <td>0.5</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.868920</td>
      <td>2024-02-12T18:20:36.905868+00:00</td>
      <td>0.5</td>
      <td>Alert</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.027985</td>
      <td>2024-02-12T18:28:36.905868+00:00</td>
      <td>0.5</td>
      <td>Ok</td>
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

# set the width, interval, and time period for the assay interactive run
assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

assay_builder_from_dates.build().interactive_run()[0].chart()
assay_builder_from_dates.build().interactive_run()[1].chart()
```

    baseline mean = 514078.81836327346
    window mean = 515981.794078125
    baseline median = 450867.6875
    window median = 448627.8125
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.02621254075808954
    scores = [0.0, 0.010569514437627874, 0.00037363078823020166, 0.0033781949078495045, 0.0022725850767459357, 0.0010566612077523828, 0.004511714130505298, 1.4147243707877562e-05, 0.0033457853900346513, 0.00017948501813203413, 0.0005108225575037834, 0.0]
    index = None

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_56_1.png" width="800" label="png">}}
    

    baseline mean = 514078.81836327346
    window mean = 1883044.252
    baseline median = 450867.6875
    window median = 1946437.75
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 8.12154551791522
    scores = [0.0, 0.2838072039839983, 0.2838072039839983, 0.2838072039839983, 0.31561967084401865, 0.33988809231947975, 0.20735189427254225, 0.2916995446658736, 0.2996329479746256, 0.2681491896477076, 0.27595678652368016, 5.2718257797152965]
    index = None

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_56_3.png" width="800" label="png">}}
    

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

# set the width, interval, and time period for the assay interactive run
assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

assay_builder_from_dates.build().interactive_run()[0].chart()
assay_builder_from_dates.build().interactive_run()[1].chart()
```

    baseline mean = 514078.81836327346
    window mean = 515981.794078125
    baseline median = 450867.6875
    window median = 448627.8125
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.010824679818960396
    scores = [0.0, 0.002969815172154558, 0.005578645912763821, 0.0002298040928465568, 0.0013956367196008871, 0.0006507779215945722, 0.0]
    index = None

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_58_1.png" width="800" label="png">}}
    

    baseline mean = 514078.81836327346
    window mean = 1883044.252
    baseline median = 450867.6875
    window median = 1946437.75
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 8.868919890469542
    scores = [0.0, 0.7174700740703411, 0.7548517748433896, 0.6896479439992407, 0.7454763738420332, 0.6896479439992407, 5.2718257797152965]
    index = None

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_58_3.png" width="800" label="png">}}
    

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

# set the width, interval, and time period for the assay interactive run
assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

assay_builder_from_dates.build().interactive_run()[0].chart()
assay_builder_from_dates.build().interactive_run()[1].chart()
```

    baseline mean = 514078.81836327346
    window mean = 515981.794078125
    baseline median = 450867.6875
    window median = 448627.8125
    bin_mode = Equal
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.0035028394332936013
    scores = [0.0, 2.980229110305778e-05, 0.000622915228940027, 0.0007762707808988463, 0.00074530078189759, 0.0013285503504540802, 0.0]
    index = None

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_59_1.png" width="800" label="png">}}
    

    baseline mean = 514078.81836327346
    window mean = 1883044.252
    baseline median = 450867.6875
    window median = 1946437.75
    bin_mode = Equal
    aggregation = Density
    metric = PSI
    weighted = False
    score = 9.44553462565066
    scores = [0.0, 2.71520515570787, 1.1845568469857768, 0.2147793051333076, 0.04994766134541187, 0.009219876762995191, 5.2718257797152965]
    index = None

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_59_3.png" width="800" label="png">}}
    

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

# set the width, interval, and time period for the assay interactive run
assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

assay_builder_from_dates.build().interactive_run()[0].chart()
assay_builder_from_dates.build().interactive_run()[1].chart()
```

    baseline mean = 514078.81836327346
    window mean = 515981.794078125
    baseline median = 450867.6875
    window median = 448627.8125
    bin_mode = Provided
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.0044289194274896694
    scores = [0.0, 2.0883795503916425e-05, 8.129210375338785e-06, 0.0007993288996432606, 0.0008420454950651045, 0.0027585320269020493, 0.0]
    index = None

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_61_1.png" width="800" label="png">}}
    

    baseline mean = 514078.81836327346
    window mean = 1883044.252
    baseline median = 450867.6875
    window median = 1946437.75
    bin_mode = Provided
    aggregation = Density
    metric = PSI
    weighted = False
    score = 9.174813366112216
    scores = [0.0, 1.44121120281246, 1.6306076742577449, 0.6073469373145474, 0.2681491896477076, 3.803547720179949, 1.4239506418998062]
    index = None

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_61_3.png" width="800" label="png">}}
    

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

# set the width, interval, and time period for the assay interactive run
assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

assay_builder_from_dates.build().interactive_run()[0].chart()
assay_builder_from_dates.build().interactive_run()[1].chart()
```

    baseline mean = 514078.81836327346
    window mean = 515981.794078125
    baseline median = 450867.6875
    window median = 448627.8125
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.010824679818960396
    scores = [0.0, 0.002969815172154558, 0.005578645912763821, 0.0002298040928465568, 0.0013956367196008871, 0.0006507779215945722, 0.0]
    index = None

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_63_1.png" width="800" label="png">}}
    

    baseline mean = 514078.81836327346
    window mean = 1883044.252
    baseline median = 450867.6875
    window median = 1946437.75
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 8.868919890469542
    scores = [0.0, 0.7174700740703411, 0.7548517748433896, 0.6896479439992407, 0.7454763738420332, 0.6896479439992407, 5.2718257797152965]
    index = None

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_63_3.png" width="800" label="png">}}
    

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

# set the width, interval, and time period for the assay interactive run
assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

assay_builder_from_dates.build().interactive_run()[0].chart()
assay_builder_from_dates.build().interactive_run()[1].chart()
```

    baseline mean = 514078.81836327346
    window mean = 515981.794078125
    baseline median = 450867.6875
    window median = 448627.8125
    bin_mode = Quantile
    aggregation = Cumulative
    metric = PSI
    weighted = False
    score = 0.052003992015968
    scores = [0.0, 0.023600798403193624, 0.011814371257485023, 0.00520159680638721, 0.011387225548902149, 0.0, 0.0]
    index = None

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_64_1.png" width="800" label="png">}}
    

    baseline mean = 514078.81836327346
    window mean = 1883044.252
    baseline median = 450867.6875
    window median = 1946437.75
    bin_mode = Quantile
    aggregation = Cumulative
    metric = PSI
    weighted = False
    score = 3.0139720558882237
    scores = [0.0, 0.1996007984031936, 0.40718562874251496, 0.6007984031936128, 0.8063872255489022, 1.0, 0.0]
    index = None

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_64_3.png" width="800" label="png">}}
    

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

# set the width, interval, and time period for the assay interactive run
assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)
```

    <wallaroo.assay_config.WindowBuilder at 0x288671880>

```python
assay_builder_from_dates.build().interactive_run()[0].compare_basic_stats()
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
      <td>1412215.125</td>
      <td>-101864.250000</td>
      <td>-6.727801</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>514078.818363</td>
      <td>515981.794078</td>
      <td>1902.975715</td>
      <td>0.370172</td>
    </tr>
    <tr>
      <th>median</th>
      <td>450867.6875</td>
      <td>448627.8125</td>
      <td>-2239.875000</td>
      <td>-0.496792</td>
    </tr>
    <tr>
      <th>std</th>
      <td>231664.973507</td>
      <td>228993.031693</td>
      <td>-2671.941814</td>
      <td>-1.153365</td>
    </tr>
    <tr>
      <th>start</th>
      <td>None</td>
      <td>2024-02-12T18:18:36.905868+00:00</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>end</th>
      <td>None</td>
      <td>2024-02-12T18:19:36.905868+00:00</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>

```python
assay_builder_from_dates.build().interactive_run()[1].compare_basic_stats()
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
      <td>4.990000e+02</td>
      <td>99.600798</td>
    </tr>
    <tr>
      <th>min</th>
      <td>236238.671875</td>
      <td>1514079.375</td>
      <td>1.277841e+06</td>
      <td>540.910890</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1514079.375</td>
      <td>2016006.125</td>
      <td>5.019268e+05</td>
      <td>33.150623</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>514078.818363</td>
      <td>1883044.252</td>
      <td>1.368965e+06</td>
      <td>266.294853</td>
    </tr>
    <tr>
      <th>median</th>
      <td>450867.6875</td>
      <td>1946437.75</td>
      <td>1.495570e+06</td>
      <td>331.709303</td>
    </tr>
    <tr>
      <th>std</th>
      <td>231664.973507</td>
      <td>159158.122868</td>
      <td>-7.250685e+04</td>
      <td>-31.298150</td>
    </tr>
    <tr>
      <th>start</th>
      <td>None</td>
      <td>2024-02-12T18:20:36.905868+00:00</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>end</th>
      <td>None</td>
      <td>2024-02-12T18:21:36.905868+00:00</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>

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

# set the width, interval, and time period for the assay interactive run
assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

assay_builder_from_dates.build().interactive_run().chart_scores()
```

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_70_0.png" width="800" label="png">}}
    

```python
# Build the assay, based on the start and end of our baseline time, 
# and tracking the output variable index 0
assay_builder_from_dates = wl.build_assay(assay_name="assays from date baseline", 
                                          pipeline=mainpipeline, 
                                          model_name="house-price-estimator", 
                                          iopath="output variable 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# set the width, interval, and time period for the assay interactive run
assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=5).add_interval(minutes=5).add_start(assay_window_start)

assay_builder_from_dates.build().interactive_run().chart_scores()
```

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_71_0.png" width="800" label="png">}}
    

### Create Assay

With the assay created and fully tested, we officially create it by uploading it to the Wallaroo instance.  Once it is uploaded, the assay runs an analysis based on the window width, interval, and the other settings configured.

Assays are uploaded with the `wallaroo.assay_config.upload()` method. This uploads the assay into the Wallaroo database with the configurations applied and returns the assay id. Note that assay names **must be unique across the Wallaroo instance**; attempting to upload an assay with the same name as an existing one will return an error.

`wallaroo.assay_config.upload()` returns the assay id for the assay. 

```python
# Build the assay, based on the start and end of our baseline time, 
# and tracking the output variable index 0
assay_builder_from_dates = wl.build_assay(assay_name="assays from date baseline create sample", 
                                          pipeline=mainpipeline, 
                                          model_name="house-price-estimator", 
                                          iopath="output variable 0",
                                          baseline_start=assay_baseline_start, 
                                          baseline_end=assay_baseline_end)

# set the width, interval, and time period for the assay interactive run
assay_builder_from_dates.add_run_until(datetime.datetime.now())
assay_builder_from_dates.window_builder().add_width(minutes=1).add_interval(minutes=1).add_start(assay_window_start)

assay_builder_from_dates.build()

assay_id = assay_builder_from_dates.upload()
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
assay_results.to_dataframe().loc[:, ['score', 'start', 'alert_threshold', 'status']]
```

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_76_0.png" width="800" label="png">}}
    

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
      <td>0.014419</td>
      <td>2024-02-12T18:33:36.905868+00:00</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.868920</td>
      <td>2024-02-12T18:35:36.905868+00:00</td>
      <td>0.25</td>
      <td>Alert</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.005493</td>
      <td>2024-02-12T18:37:36.905868+00:00</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
  </tbody>
</table>

```python
for assay_analysis in assay_results:
    assay_analysis.chart()
```

    baseline mean = 513975.7783183633
    window mean = 516338.589359375
    baseline median = 448627.8125
    window median = 448627.8125
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 0.005207728
    scores = [0.0, 0.0008040331034294781, 0.00010009324303378128, 0.0005128718504664744, 0.0037778397193669257, 1.2890162113860805e-05, 0.0]
    index = None

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_77_1.png" width="800" label="png">}}
    

    baseline mean = 513975.7783183633
    window mean = 1883684.608375
    baseline median = 448627.8125
    window median = 1946437.75
    bin_mode = Quantile
    aggregation = Density
    metric = PSI
    weighted = False
    score = 8.869396
    scores = [0.0, 0.7361208231835864, 0.7736613502280227, 0.6528484073872128, 0.7174700740703411, 0.7174700740703411, 5.2718257797152965]
    index = None

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_77_3.png" width="800" label="png">}}
    

#### List and Retrieve Assay

If the assay id is not already know, it is retrieved from the `wallaroo.client.list_assays()` method.  Select the assay to retrieve data for and retrieve its id with `wallaroo.assay.Assay._id` method.

```python
wl.list_assays()
```

<table><tr><th>name</th><th>active</th><th>status</th><th>warning_threshold</th><th>alert_threshold</th><th>pipeline_name</th></tr><tr><td>assays from date baseline create sample</td><td>True</td><td>{"run_at": "2024-02-07T22:48:57.585048639+00:00",  "num_ok": 0, "num_warnings": 0, "num_alerts": 0}</td><td>None</td><td>0.25</td><td>assay-demonstration-tutorial</td></tr><tr><td>assays from date sample</td><td>True</td><td>{"run_at": "2024-02-07T22:48:57.560023292+00:00",  "num_ok": 0, "num_warnings": 0, "num_alerts": 0}</td><td>None</td><td>0.25</td><td>assay-demonstration-tutorial</td></tr><tr><td>assays from date baseline</td><td>True</td><td>{"run_at": "2024-02-07T22:48:57.573609045+00:00",  "num_ok": 0, "num_warnings": 0, "num_alerts": 0}</td><td>None</td><td>0.25</td><td>assay-demonstration-tutorial</td></tr><tr><td>assays from numpy</td><td>True</td><td>created</td><td>None</td><td>0.25</td><td>assay-demonstration-tutorial</td></tr></table>

```python
retrieved_assay = wl.list_assays()[0]

assay_results = wl.get_assay_results(assay_id=retrieved_assay._id,
                     start=assay_window_start,
                     end=datetime.datetime.now())

assay_results.chart_scores()

```

    
{{<figure src="/images/2024.1/wallaroo-tutorials/wallaroo-tutorials-observability/wallaroo_model_observability_assays-reference_files/wallaroo_model_observability_assays-reference_80_0.png" width="800" label="png">}}
    

### Pause and Resume Assay

Assays are paused and started with the `wallaroo.assay.Assay.turn_off` and `wallaroo.assay.Assay.turn_on` methods.

For the following, we retrieve an assay from the wallaroo instance and pause it, then list the assays to verify its setting `Active` is `False`.

```python
display(wl.list_assays())
retrieved_assay = wl.list_assays()[0]

retrieved_assay.turn_off()
display(wl.list_assays())
```

<table><tr><th>Assay ID</th><th>Assay Name</th><th>Active</th><th>Status</th><th>Warning Threshold</th><th>Alert Threshold</th><th>Pipeline ID</th><th>Pipeline Name</th></tr><tr><td>1</td><td>assays from date baseline create sample</td><td>True</td><td>{"run_at": "2024-02-12T18:38:36.972894731+00:00",  "num_ok": 1, "num_warnings": 0, "num_alerts": 0}</td><td>None</td><td>0.25</td><td>7</td><td>assay-demonstration-tutorial</td></tr></table>

<table><tr><th>Assay ID</th><th>Assay Name</th><th>Active</th><th>Status</th><th>Warning Threshold</th><th>Alert Threshold</th><th>Pipeline ID</th><th>Pipeline Name</th></tr><tr><td>1</td><td>assays from date baseline create sample</td><td>False</td><td>{"run_at": "2024-02-12T18:38:36.972894731+00:00",  "num_ok": 1, "num_warnings": 0, "num_alerts": 0}</td><td>None</td><td>0.25</td><td>7</td><td>assay-demonstration-tutorial</td></tr></table>

We resume the assay and verify its setting `Active` is `True`.

```python
retrieved_assay.turn_on()
display(wl.list_assays())
```

<table><tr><th>Assay ID</th><th>Assay Name</th><th>Active</th><th>Status</th><th>Warning Threshold</th><th>Alert Threshold</th><th>Pipeline ID</th><th>Pipeline Name</th></tr><tr><td>1</td><td>assays from date baseline create sample</td><td>True</td><td>{"run_at": "2024-02-12T18:38:36.972894731+00:00",  "num_ok": 1, "num_warnings": 0, "num_alerts": 0}</td><td>None</td><td>0.25</td><td>7</td><td>assay-demonstration-tutorial</td></tr></table>

### Undeploy Main Pipeline

With the examples and tutorial complete, we will undeploy the main pipeline and return the resources back to the Wallaroo instance.

```python
mainpipeline.undeploy()
```

<table><tr><th>name</th> <td>assay-demonstration-tutorial</td></tr><tr><th>created</th> <td>2024-02-12 18:16:18.335326+00:00</td></tr><tr><th>last_updated</th> <td>2024-02-12 18:16:19.100769+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>accel</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>b6525c2a-c79d-4f36-a8d7-0f515d45e4fc, b65e67fb-135e-43ca-8ff0-96ee04d6de02</td></tr><tr><th>steps</th> <td>house-price-estimator</td></tr><tr><th>published</th> <td>False</td></tr></table>

