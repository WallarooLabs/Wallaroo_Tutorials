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
```

### Configuration and Methods

The following will set the workspace, pipeline, model name, the model file name used when uploading and converting the `keras` model, and the sample data.

The functions `get_workspace(name)` will either set the current workspace to the requested name, or create it if it does not exist.  The function `get_pipeline(name)` will either set the pipeline used to the name requested, or create it in the current workspace if it does not exist.


```python
workspace_name = 'keras-autoconvert-workspace'
pipeline_name = 'keras-autoconvert-pipeline'
model_name = 'simple-sentiment-model'
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
# Client connection from local Wallaroo instance

# wl = wallaroo.Client()

# SSO login through keycloak

wallarooPrefix = "YOUR PREFIX"
wallarooSuffix = "YOUR SUFFIX"

wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}.api.{wallarooSuffix}", 
                    auth_endpoint=f"https://{wallarooPrefix}.keycloak.{wallarooSuffix}", 
                    auth_type="sso")
```

### Set the Workspace and Pipeline

Set or create the workspace and pipeline based on the names configured earlier.


```python
workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)

pipeline = get_pipeline(pipeline_name)
pipeline
```




<table><tr><th>name</th> <td>keras-autoconvert-pipeline</td></tr><tr><th>created</th> <td>2022-07-07 16:27:57.437207+00:00</td></tr><tr><th>last_updated</th> <td>2022-07-07 16:28:42.403022+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>steps</th> <td>simple-sentiment-model</td></tr></table>



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




    {'name': 'simple-sentiment-model', 'version': 'c76870f8-e16b-4534-bb17-e18a3e3806d5', 'file_name': '14d9ab8d-47f4-4557-82a7-6b26cb67ab05-converted.onnx', 'last_update_time': datetime.datetime(2022, 7, 7, 16, 41, 22, 528430, tzinfo=tzutc())}



## Test Inference

With the model uploaded and converted, we can run a sample inference.

### Add Pipeline Step and Deploy

We will add the model as a step into our pipeline, then deploy it.


```python
pipeline.add_model_step(model_wl).deploy()
```

    Waiting for deployment - this will take up to 45s .... ok





<table><tr><th>name</th> <td>keras-autoconvert-pipeline</td></tr><tr><th>created</th> <td>2022-07-07 16:27:57.437207+00:00</td></tr><tr><th>last_updated</th> <td>2022-07-07 16:41:23.615423+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>steps</th> <td>simple-sentiment-model</td></tr></table>



### Run a Test Inference

We can run a test inference from the `simple_sentiment_testdata.json` file, then display just the results.


```python
sample_data = 'simple_sentiment_testdata.json'
result = pipeline.infer_from_file(sample_data)
result[0].data()
```

    Waiting for inference response - this will take up to 45s .... ok





    [array([[0.09469762],
            [0.99103099],
            [0.93407357],
            [0.56030995],
            [0.9964503 ]])]



### Undeploy the Pipeline

With the tests complete, we will undeploy the pipeline to return the resources back to the Wallaroo instance.


```python
pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s ................................... ok





<table><tr><th>name</th> <td>keras-autoconvert-pipeline</td></tr><tr><th>created</th> <td>2022-07-07 16:27:57.437207+00:00</td></tr><tr><th>last_updated</th> <td>2022-07-07 16:57:06.402657+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>steps</th> <td>simple-sentiment-model</td></tr></table>




```python

```
