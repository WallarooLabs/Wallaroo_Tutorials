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

## Prerequisites

* An installed Wallaroo instance.
* The following Python libraries installed:
  * `os`
  * [`wallaroo`](https://pypi.org/project/wallaroo/): The Wallaroo SDK. Included with the Wallaroo JupyterHub service by default.
  * [`pandas`](https://pypi.org/project/pandas/): Pandas, mainly used for Pandas DataFrame

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
        pipeline = wl.pipelines_by_name(name)[0]
    except EntityNotFoundError:
        pipeline = wl.build_pipeline(name)
    return pipeline
```

### Connect to the Wallaroo Instance

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
# Client connection from local Wallaroo instance

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

<table><tr><th>name</th> <td>externalkerasautoconvertpipeline</td></tr><tr><th>created</th> <td>2023-05-17 21:13:27.523527+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 21:13:27.523527+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>3948e0dc-d591-4ff5-a48f-b8d17195a806</td></tr><tr><th>steps</th> <td></td></tr></table>

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

{{<figure src="/images/2023.3.0/wallaroo-tutorials/wallaroo-keras-converted-model.png" width="800" label="converted model">}}

```python
# converts and uploads model.
model_wl = wl.convert_model('simple_sentiment_model.zip', model_conversion_type, model_conversion_args)
model_wl
```

    {'name': 'externalsimple-sentiment-model', 'version': 'c378425b-b70f-465a-a15b-d9e662b15263', 'file_name': '19ec5f96-d3a6-47af-ae6f-928187735de2-converted.onnx', 'image_path': None, 'last_update_time': datetime.datetime(2023, 5, 17, 21, 13, 29, 933149, tzinfo=tzutc())}

## Test Inference

With the model uploaded and converted, we can run a sample inference.

### Add Pipeline Step and Deploy

We will add the model as a step into our pipeline, then deploy it.

```python
pipeline.add_model_step(model_wl).deploy()
```

<table><tr><th>name</th> <td>externalkerasautoconvertpipeline</td></tr><tr><th>created</th> <td>2023-05-17 21:13:27.523527+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 21:13:30.959401+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>7be0dd01-ef82-4335-b60d-6f1cd5287e5b, 3948e0dc-d591-4ff5-a48f-b8d17195a806</td></tr><tr><th>steps</th> <td>externalsimple-sentiment-model</td></tr></table>

```python
pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.3.139',
       'name': 'engine-59fb67fcc6-tns2j',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'externalkerasautoconvertpipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'externalsimple-sentiment-model',
          'version': 'c378425b-b70f-465a-a15b-d9e662b15263',
          'sha': '49f7367eede690b369aef322569c5b54c4133692610a11dc29b14d4c49ea983c',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.4.170',
       'name': 'engine-lb-584f54c899-fnntk',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}

### Run a Test Inference

We can run a test inference from the `simple_sentiment_testdata.json` file, then display just the results.

```python
sample_data = 'simple_sentiment_testdata.df.json'
result = pipeline.infer_from_file(sample_data)
display(result["out.dense"])
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

<table><tr><th>name</th> <td>externalkerasautoconvertpipeline</td></tr><tr><th>created</th> <td>2023-05-17 21:13:27.523527+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 21:13:30.959401+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>7be0dd01-ef82-4335-b60d-6f1cd5287e5b, 3948e0dc-d591-4ff5-a48f-b8d17195a806</td></tr><tr><th>steps</th> <td>externalsimple-sentiment-model</td></tr></table>

