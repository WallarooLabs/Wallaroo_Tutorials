This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/model_conversion/keras-to-onnx).

## Introduction

Machine Learning (ML) models can be converted into a Wallaroo Model object and uploaded into Wallaroo workspace using the Wallaroo Client `convert_model(path, source_type, conversion_arguments)` method.  This conversion process transforms the model into an open format that can be run across different frameworks at compiled C-language speeds.

The following tutorial is a brief example of how to convert a [Keras](https://keras.io/) or Tensor ML model to [ONNX](https://onnx.ai/ ).  This allows organizations that have trained Keras or Tensor models to convert them and use them with Wallaroo.

This tutorial assumes that you have a Wallaroo instance and are running this Notebook from the Wallaroo Jupyter Hub service.

This tutorial demonstrates how to:

* Convert a `keras` ML model and upload it into the Wallaroo engine.
* Run a sample inference on the converted model in a Wallaroo instance.

This tutorial provides the following:

* `simple_sentiment_model.zip`: A pre-trained `keras` sentiment model to be converted.  This has 100 columns.

## Conversion Steps

To use the Wallaroo autoconverter `convert_model(path, source_type, conversion_arguments)` method takes 3 parameters.  The paramters for `keras` conversions are:

* `path` (STRING):  The path to the ML model file.
* `source_type` (ModelConversionSource): The type of ML model to be converted.  As of this time Wallaroo auto-conversion supports the following source types and their associated `ModelConversionSource`:
  * **sklearn**: `ModelConversionSource.SKLEARN`
  * **xgboost**: `ModelConversionSource.XGBOOST`
  * **keras**: `ModelConversionSource.KERAS`
* `conversion_arguments`:  The arguments for the conversion based on the type of model being converted.  These are:
  * `wallaroo.ModelConversion.ConvertKerasArguments`: Used for converting `keras` type models and takes the following parameters:
    * `name`: The name of the model being converted.
    * `comment`: Any comments for the model.
    * `input_type`: A [tensorflow Dtype](https://www.tensorflow.org/api_docs/python/tf/dtypes/DType) called in the format `ModelConversionInputType.{type}`, where `{type}` is `Float`, `Double`, etc depending on the model.
    * `dimensions`: Corresponds to the keras `xtrain` in the format List[Union[None, int, float]].

### Import Libraries

The first step is to import the libraries needed.


```python
import wallaroo

from wallaroo.ModelConversion import ConvertKerasArguments, ModelConversionSource, ModelConversionInputType
from wallaroo.object import EntityNotFoundError
import pandas as pd

# used to display dataframe information without truncating
from IPython.display import display
pd.set_option('display.max_colwidth', None)
```

### Configuration and Methods

The following will set the workspace, pipeline, model name, the model file name used when uploading and converting the `keras` model, and the sample data.

The functions `get_workspace(name)` will either set the current workspace to the requested name, or create it if it does not exist.  The function `get_pipeline(name)` will either set the pipeline used to the name requested, or create it in the current workspace if it does not exist.


```python
workspace_name = 'externalkerasautoconvertworkspace'
pipeline_name = 'externalkerasautoconvertpipeline'
model_name = 'externalsimple-sentiment-model'
model_file_name = 'simple_sentiment_model.zip'
sample_data = 'simple_sentiment_testdata.json'


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
# SSO login through keycloak

wallarooPrefix = "YOUR PREFIX"
wallarooSuffix = "YOUR SUFFIX"

wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}.api.{wallarooSuffix}", 
                auth_endpoint=f"https://{wallarooPrefix}.keycloak.{wallarooSuffix}", 
                auth_type="sso")
```

### Arrow Support

As of the 2023.1 release, Wallaroo provides support for dataframe and Arrow for inference inputs.  This tutorial allows users to adjust their experience based on whether they have enabled Arrow support in their Wallaroo instance or not.

If Arrow support has been enabled, `arrowEnabled=True`. If disabled or you're not sure, set it to `arrowEnabled=False`

The examples below will be shown in an arrow enabled environment.


```python
import os
# Only set the below to make the OS environment ARROW_ENABLED to TRUE.  Otherwise, leave as is.
# os.environ["ARROW_ENABLED"]="True"

if "ARROW_ENABLED" not in os.environ or os.environ["ARROW_ENABLED"] == "False":
    arrowEnabled = False
else:
    arrowEnabled = True
print(arrowEnabled)
```

### Set the Workspace and Pipeline

Set or create the workspace and pipeline based on the names configured earlier.


```python
workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)

pipeline = get_pipeline(pipeline_name)
pipeline
```




<table><tr><th>name</th> <td>externalkerasautoconvertpipeline</td></tr><tr><th>created</th> <td>2023-02-21 18:16:17.818879+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-21 18:16:17.818879+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>d0be28f6-4a0b-4b1c-9196-d809e8e21379</td></tr><tr><th>steps</th> <td></td></tr></table>



### Set the Model Autoconvert Parameters

Set the paramters for converting the `simple-sentiment-model`.  This includes the shape of the model.


```python
model_columns = 100

model_conversion_args = ConvertKerasArguments(
    name=model_name,
    comment="simple keras model",
    input_type=ModelConversionInputType.Float32,
    dimensions=(None, model_columns)
)
model_conversion_type = ModelConversionSource.KERAS
```

### Upload and Convert the Model

Now we can upload the convert the model.  Once finished, it will be stored as `{unique-file-id}-converted.onnx`.

![converted model](./images/wallaroo-tutorials/wallaroo-keras-converted-model.png)


```python
# converts and uploads model.
model_wl = wl.convert_model('simple_sentiment_model.zip', model_conversion_type, model_conversion_args)
model_wl
```




    {'name': 'externalsimple-sentiment-model', 'version': '7ac5e3a0-62f4-406e-87bb-185dd4f26fb6', 'file_name': '2f0a2876-fe21-44ac-9e1a-ba2462c77692-converted.onnx', 'image_path': None, 'last_update_time': datetime.datetime(2023, 2, 21, 18, 16, 26, 708440, tzinfo=tzutc())}



## Test Inference

With the model uploaded and converted, we can run a sample inference.

### Add Pipeline Step and Deploy

We will add the model as a step into our pipeline, then deploy it.


```python
pipeline.add_model_step(model_wl).deploy()
```




<table><tr><th>name</th> <td>externalkerasautoconvertpipeline</td></tr><tr><th>created</th> <td>2023-02-21 18:16:17.818879+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-21 18:16:30.388009+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>f7141b40-c610-42f7-873c-6fcb2b6e9ada, d0be28f6-4a0b-4b1c-9196-d809e8e21379</td></tr><tr><th>steps</th> <td>externalsimple-sentiment-model</td></tr></table>




```python
pipeline.status()
```




    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.48.0.50',
       'name': 'engine-579bdcf5bd-9j2mk',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'externalkerasautoconvertpipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'externalsimple-sentiment-model',
          'version': '7ac5e3a0-62f4-406e-87bb-185dd4f26fb6',
          'sha': '88f8118f5e9ea7368dde563413c77738e64b4e3f5856c3c9323b02bcf0dd1fd5',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.48.0.49',
       'name': 'engine-lb-74b4969486-gmwzg',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}



### Run a Test Inference

We can run a test inference from the `simple_sentiment_testdata.json` file, then display just the results.


```python
if arrowEnabled is True:
    sample_data = 'simple_sentiment_testdata.df.json'
    result = pipeline.infer_from_file(sample_data)
    display(result["out.dense"])
else:
    sample_data = 'simple_sentiment_testdata.json'
    result = pipeline.infer_from_file(sample_data)
    display(result[0].data())
```


    0    [0.094697624]
    1       [0.991031]
    2     [0.93407357]
    3     [0.56030995]
    4      [0.9964503]
    Name: out.dense, dtype: object


### Undeploy the Pipeline

With the tests complete, we will undeploy the pipeline to return the resources back to the Wallaroo instance.


```python
pipeline.undeploy()
```




<table><tr><th>name</th> <td>externalkerasautoconvertpipeline</td></tr><tr><th>created</th> <td>2023-02-21 18:16:17.818879+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-21 18:16:30.388009+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>f7141b40-c610-42f7-873c-6fcb2b6e9ada, d0be28f6-4a0b-4b1c-9196-d809e8e21379</td></tr><tr><th>steps</th> <td>externalsimple-sentiment-model</td></tr></table>


