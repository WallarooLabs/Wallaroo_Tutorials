This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/model_conversion/xgboost-autoconversion).

## Introduction

The following tutorial is a brief example of how to convert a [XGBoost](https://xgboost.readthedocs.io/en/stable/index.html) Classification ML model with the `convert_model` method and upload it into your Wallaroo instance.

This tutorial assumes that you have a Wallaroo instance and are running this Notebook from the Wallaroo Jupyter Hub service.

* Convert a `XGBoost` Classification ML model and upload it into the Wallaroo engine.
* Run a sample inference on the converted model in a Wallaroo instance.

This tutorial provides the following:

* `xgb_class.pickle`: A pretrained `XGBoost` Classification model with 25 columns.
* `xgb_class_eval.json`: Test data to perform a sample inference.

## Prerequisites

Wallaroo supports the following model versions:

* XGBoost:  Version 1.6.0
* SKLearn: 1.1.2

## Conversion Steps

To use the Wallaroo autoconverter `convert_model(path, source_type, conversion_arguments)` method takes 3 parameters.  The parameters for `XGBoost` conversions are:

* `path` (STRING):  The path to the ML model file.
* `source_type` (ModelConversionSource): The type of ML model to be converted.  As of this time Wallaroo auto-conversion supports the following source types and their associated `ModelConversionSource`:
  * **sklearn**: `ModelConversionSource.SKLEARN`
  * **xgboost**: `ModelConversionSource.XGBOOST`
  * **keras**: `ModelConversionSource.KERAS`
* `conversion_arguments`:  The arguments for the conversion based on the type of model being converted.  These are:
    * `wallaroo.ModelConversion.ConvertXGBoostArgs`: Used for `XGBoost` models and takes the following parameters:
    * `name`: The name of the model being converted.
    * `comment`: Any comments for the model.
    * `number_of_columns`: The number of columns the model was trained for.
    * `input_type`: A [tensorflow Dtype](https://www.tensorflow.org/api_docs/python/tf/dtypes/DType) called in the format `ModelConversionInputType.{type}`, where `{type}` is `Float`, `Double`, etc depending on the model.

### Import Libraries

The first step is to import the libraries needed.

```python
import wallaroo

from wallaroo.ModelConversion import ConvertXGBoostArgs, ModelConversionSource, ModelConversionInputType
from wallaroo.object import EntityNotFoundError

# used to display dataframe information without truncating
from IPython.display import display
import pandas as pd
pd.set_option('display.max_colwidth', None)
```

### Connect to the Wallaroo Instance

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
# Login through local Wallaroo instance

wl = wallaroo.Client()
```

### Configuration and Methods

The following will set the workspace, pipeline, model name, the model file name used when uploading and converting the `keras` model, and the sample data.

The functions `get_workspace(name)` will either set the current workspace to the requested name, or create it if it does not exist.  The function `get_pipeline(name)` will either set the pipeline used to the name requested, or create it in the current workspace if it does not exist.

```python
workspace_name = 'xgboost-classification-autoconvert-workspace'
pipeline_name = 'xgboost-classification-autoconvert-pipeline'
model_name = 'xgb-class-model'
model_file_name = 'xgb_class.pickle'

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

### Set the Workspace and Pipeline

Set or create the workspace and pipeline based on the names configured earlier.

```python
workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)

pipeline = get_pipeline(pipeline_name)
pipeline
```

<table><tr><th>name</th> <td>xgboost-classification-autoconvert-pipeline</td></tr><tr><th>created</th> <td>2023-05-17 21:21:19.962450+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 21:21:19.962450+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>bbe4dce4-f62a-4f4f-a45c-aebbfce23304</td></tr><tr><th>steps</th> <td></td></tr></table>

### Set the Model Autoconvert Parameters

Set the paramters for converting the `xgb-class-model`.

```python
#the number of columns
NF = 25

model_conversion_args = ConvertXGBoostArgs(
    name=model_name,
    comment="xgboost classification model test",
    number_of_columns=NF,
    input_type=ModelConversionInputType.Float32
)
model_conversion_type = ModelConversionSource.XGBOOST
```

### Upload and Convert the Model

Now we can upload the convert the model.  Once finished, it will be stored as `{unique-file-id}-converted.onnx`.

```python
# convert and upload
model_wl = wl.convert_model(model_file_name, model_conversion_type, model_conversion_args)
```

## Test Inference

With the model uploaded and converted, we can run a sample inference.

### Deploy the Pipeline

Add the uploaded and converted `model_wl` as a step in the pipeline, then deploy it.

```python
pipeline.add_model_step(model_wl).deploy()
```

<table><tr><th>name</th> <td>xgboost-classification-autoconvert-pipeline</td></tr><tr><th>created</th> <td>2023-05-17 21:21:19.962450+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 21:21:22.906665+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>5f7bb0cc-f60d-4cee-8425-c5e85331ae2f, bbe4dce4-f62a-4f4f-a45c-aebbfce23304</td></tr><tr><th>steps</th> <td>xgb-class-model</td></tr></table>

### Run the Inference

Use the evaluation data to verify the process completed successfully.

```python
sample_data = 'xgb_class_eval.df.json'
result = pipeline.infer_from_file(sample_data)
display(result)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.label</th>
      <th>out.probabilities</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-05-17 21:21:34.273</td>
      <td>[-0.9650039837, 1.7162569382, 1.8570196174, 0.7225873636, 1.4614264692, 1.9567455469, 3.1280554236, 2.4737274835, 2.045634687, 0.0697759683, -0.7334890238, 1.4661397464, -1.7339080123, -0.3295498275, -0.5405674404, 0.9325072938, -0.1753815275, 0.8389569878, 0.2995238298, 2.020354449, 0.307715435, -0.786562628, 1.6198295619, -3.1550540615, 2.4493095715]</td>
      <td>[1]</td>
      <td>[0.45164853, 0.54835147]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-05-17 21:21:34.273</td>
      <td>[-0.24290676, -2.7621478465, -1.0460044448, -0.4367771067, 0.7114974086, 3.1152360132, 0.8780655791, 1.5959052391, 0.1291853603, -0.4705432269, -0.2870965835, 0.2758634598, -2.5296629025, -0.8581708475, -0.0447250952, -0.8147113092, 0.3394927614, 0.1165005518, 0.5214230106, 1.0323965467, 0.824008803, -0.2602068525, -2.5164397098, -2.2480625668, 0.7147467132]</td>
      <td>[0]</td>
      <td>[0.76527536, 0.23472464]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-05-17 21:21:34.273</td>
      <td>[0.3261925153, -1.1340263025, -0.0210165684, -0.402436985, 0.1136841647, 1.9756910921, -1.6567823116, -3.0377564302, 1.0839562248, 1.535350752, -1.5641493986, -0.4037836272, -0.0502258358, -1.383033319, -2.1692714889, 0.5474654104, 0.5884733316, -0.6575750129, -0.4456088906, 1.9450809267, -0.5395060067, 0.0020371202, -2.0035740797, 5.3368805176, -1.3683109303]</td>
      <td>[1]</td>
      <td>[0.011478066, 0.98852193]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-05-17 21:21:34.273</td>
      <td>[0.7071268106, -1.1177500788, 0.1311702635, -0.0342823916, 1.4166474292, -0.7600812269, -1.643252821, 1.1809622308, 1.1552655664, -1.4616319423, -1.3196760448, -0.3871231717, -1.0052010294, 0.3757483273, 0.8164121104, 0.6636194102, 0.2054206669, 0.3971757239, 1.0712736575, 0.5687901164, 0.545534547, -0.4022272078, 0.5202183853, -1.1450692638, -1.6687803276]</td>
      <td>[1]</td>
      <td>[0.40538806, 0.59461194]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-05-17 21:21:34.273</td>
      <td>[-0.4753271684, 0.9648567582, 4.1002801029, -0.3474129796, 0.5912316716, -0.3616544697, -2.9339075495, 0.8583809009, -0.7625328481, -1.447786717, -0.0183969915, -0.1028844583, -1.9931308252, -0.6141588978, 1.5368353642, -0.5482829279, 2.1576770706, 0.4772412627, 0.9956210462, 1.7124754134, -0.7415852899, -0.3876944367, 5.7178008466, 7.1237030134, 0.1815704771]</td>
      <td>[1]</td>
      <td>[0.0016139746, 0.998386]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

### Undeploy the Pipeline

With the tests complete, we will undeploy the pipeline to return the resources back to the Wallaroo instance.

```python
pipeline.undeploy()
```

<table><tr><th>name</th> <td>xgboost-classification-autoconvert-pipeline</td></tr><tr><th>created</th> <td>2023-05-17 21:21:19.962450+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 21:21:22.906665+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>5f7bb0cc-f60d-4cee-8425-c5e85331ae2f, bbe4dce4-f62a-4f4f-a45c-aebbfce23304</td></tr><tr><th>steps</th> <td>xgb-class-model</td></tr></table>

