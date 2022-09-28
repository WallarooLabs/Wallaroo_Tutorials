This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/releases).

## Introduction

The following tutorial is a brief example of how to convert a [XGBoost](https://xgboost.readthedocs.io/en/stable/index.html) Classification ML model with the `convert_model` method and upload it into your Wallaroo instance.

This tutorial assumes that you have a Wallaroo instance and are running this Notebook from the Wallaroo Jupyter Hub service.

* Convert a `XGBoost` Classification ML model and upload it into the Wallaroo engine.
* Run a sample inference on the converted model in a Wallaroo instance.

This tutorial provides the following:

* `xgb_class.pickle`: A pretrained `XGBoost` Classification model with 25 columns.
* `xgb_class_eval.json`: Test data to perform a sample inference.

## Conversion Steps

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
```

### Configuration and Methods

The following will set the workspace, pipeline, model name, the model file name used when uploading and converting the `keras` model, and the sample data.

The functions `get_workspace(name)` will either set the current workspace to the requested name, or create it if it does not exist.  The function `get_pipeline(name)` will either set the pipeline used to the name requested, or create it in the current workspace if it does not exist.


```python
workspace_name = 'xgboost-classification-autoconvert-workspace'
pipeline_name = 'xgboost-classification-autoconvert-pipeline'
model_name = 'xgb-class-model'
model_file_name = 'xgb_class.pickle'
sample_data = 'xgb_class_eval.json'


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
        pipeline = wl.pipelines_by_name(pipeline_name)[0]
    except EntityNotFoundError:
        pipeline = wl.build_pipeline(pipeline_name)
    return pipeline
```

### Connect to Wallaroo

Connect to your Wallaroo instance and store the connection into the variable `wl`.


```python
wl = wallaroo.Client()
```

### Set the Workspace and Pipeline

Set or create the workspace and pipeline based on the names configured earlier.


```python
workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)

pipeline = get_pipeline(pipeline_name)
pipeline
```




<table><tr><th>name</th> <td>xgboost-classification-autoconvert-pipeline</td></tr><tr><th>created</th> <td>2022-08-03 15:35:20.889178+00:00</td></tr><tr><th>last_updated</th> <td>2022-08-03 15:35:20.889178+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>steps</th> <td></td></tr></table>



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

    Waiting for deployment - this will take up to 45s .... ok





<table><tr><th>name</th> <td>xgboost-classification-autoconvert-pipeline</td></tr><tr><th>created</th> <td>2022-08-03 15:35:20.889178+00:00</td></tr><tr><th>last_updated</th> <td>2022-08-03 15:35:23.597027+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>steps</th> <td>xgb-class-model</td></tr></table>



### Run the Inference

Use the `test_class_eval.json` as set earlier as our `sample_data` and perform the inference.


```python
result = pipeline.infer_from_file(sample_data)
result[0].data()
```

    Waiting for inference response - this will take up to 45s ....... ok





    [array([0.e+000, 0.e+000, 5.e-324, 0.e+000, 5.e-324]),
     array([[0.99668795, 0.00331205],
            [0.52999395, 0.47000605],
            [0.14704436, 0.85295564],
            [0.995507  , 0.004493  ],
            [0.19796491, 0.80203509]])]



### Undeploy the Pipeline

With the tests complete, we will undeploy the pipeline to return the resources back to the Wallaroo instance.


```python
pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s .............................. ok





<table><tr><th>name</th> <td>xgboost-classification-autoconvert-pipeline</td></tr><tr><th>created</th> <td>2022-08-03 15:35:20.889178+00:00</td></tr><tr><th>last_updated</th> <td>2022-08-03 15:35:23.597027+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>steps</th> <td>xgb-class-model</td></tr></table>


