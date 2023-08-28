## Auto-Conversion And Upload Tutorial

Machine Learning (ML) models can be converted into a Wallaroo and uploaded into Wallaroo workspace using the Wallaroo Client `convert_model(path, source_type, conversion_arguments)` method.  This conversion process transforms the model into an open format that can be run across different frameworks at compiled C-language speeds.

The three input parameters are:

* `path` (STRING):  The path to the ML model file.
* `source_type` (ModelConversionSource): The type of ML model to be converted.  As of this time Wallaroo auto-conversion supports the following source types and their associated `ModelConversionSource`:
  * **sklearn**: `ModelConversionSource.SKLEARN`
  * **xgboost**: `ModelConversionSource.XGBOOST`
* `conversion_arguments`:  The arguments for the conversion:
  * `name`: The name of the model being converted.
  * `comment`: Any comments for the model.
  * `number_of_columns`: The number of columns the model was trained for.
  * `input_type`: The ModelConversationInputType, typically `Float` or `Double` depending on the model.
  
The following tutorial demonstrates how to convert a **sklearn** Linear Model and a **XGBoost** Regression Model, and upload them into a Wallaroo Workspace.  The following is provided for the tutorial:

* `sklearn-linear-model.pickle`: A sklearn linear model.  An example of training the model is provided in the Jupyter Notebook `sklearn-linear-model-example.ipynb`.  It has 25 columns.
* `xgb_reg.pickle`:  A XGBoost regression model.  An example of training the model is provided in the Jupyter Notebook `xgboost-regression-model-example.ipynb`.  It has 25 columns.

## Prerequisites

* An installed Wallaroo instance.
* The following Python libraries installed:
  * `os`
  * [`wallaroo`](https://pypi.org/project/wallaroo/): The Wallaroo SDK. Included with the Wallaroo JupyterHub service by default.
  * [`scikit-learn`](https://pypi.org/project/scikit-learn/) Version 1.1.1
  * [`xgboost`](https://pypi.org/project/xgboost/) Version 1.6.2
  * `pickle`

## Steps

### Import Libraries

Import the libraries that will be used for the auto-conversion process.

```python
import pickle
import json

import wallaroo

from wallaroo.ModelConversion import ConvertSKLearnArguments, ConvertXGBoostArgs, ModelConversionSource, ModelConversionInputType
from wallaroo.object import EntityNotFoundError
```

```python
# Verify the version of XGBoost used to generate the models

import sklearn
import sklearn.datasets

import xgboost as xgb

print(xgb.__version__)
print(sklearn.__version__)
```

    1.6.2
    1.1.2

The following code is used to either connect to an existing workspace or to create a new one.  For more details on working with workspaces, see the [Wallaroo Workspace Management Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-workspace-management/).

### Connect to the Wallaroo Instance

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
# Client connection from local Wallaroo instance

wl = wallaroo.Client()

```

### Set the Workspace

We'll connect or create the workspace `testautoconversion` and use it for our model testing.

```python
workspace_name = 'testautoconversion'

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
```

```python
workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)

wl.get_current_workspace()
```

    {'name': 'testautoconversion', 'id': 11, 'archived': False, 'created_by': '028c8b48-c39b-4578-9110-0b5bdd3824da', 'created_at': '2023-05-17T21:11:40.856672+00:00', 'models': [], 'pipelines': []}

### Set the Model Conversion Arguments

We'll create two different configurations, one for each of our models:

* `sklearn_model_conversion_args`: Used for our sklearn model.
* `xgboost_model_converstion_args`: Used for our XGBoost model.

```python
# The number of columns
NF=25

sklearn_model_conversion_args = ConvertSKLearnArguments(
    name="sklearntest",
    comment="test linear regression",
    number_of_columns=NF,
    input_type=ModelConversionInputType.Double
)
sklearn_model_conversion_type = ModelConversionSource.SKLEARN

xgboost_model_conversion_args = ConvertXGBoostArgs(
    name="xgbtestreg",
    comment="xgboost regression model test",
    number_of_columns=NF,
    input_type=ModelConversionInputType.Float32
)
xgboost_model_conversion_type = ModelConversionSource.XGBOOST
```

### Convert the Models

The `convert_model` method converts the model using the arguments, and uploads it into the current workspace - in this case, `testconversion`.  Once complete, we can run `get_current_workspace` to verify that the models were uploaded.

```python
# converts and uploads the sklearn model.
wl.convert_model('sklearn-linear-model.pickle', sklearn_model_conversion_type, sklearn_model_conversion_args)

# converts and uploads the XGBoost model.
wl.convert_model('xgb_reg.pickle', xgboost_model_conversion_type, xgboost_model_conversion_args)
```

    {'name': 'xgbtestreg', 'version': '07b1ff68-0a7b-4687-b64f-76ddc995e7c6', 'file_name': '957178f8-fcd1-496d-b985-29beb89ba46e-converted.onnx', 'image_path': None, 'last_update_time': datetime.datetime(2023, 5, 17, 21, 11, 43, 585229, tzinfo=tzutc())}

```python
wl.get_current_workspace()
```

    {'name': 'testautoconversion', 'id': 11, 'archived': False, 'created_by': '028c8b48-c39b-4578-9110-0b5bdd3824da', 'created_at': '2023-05-17T21:11:40.856672+00:00', 'models': [{'name': 'sklearntest', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 5, 17, 21, 11, 42, 463507, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 5, 17, 21, 11, 42, 463507, tzinfo=tzutc())}, {'name': 'xgbtestreg', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 5, 17, 21, 11, 43, 585229, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 5, 17, 21, 11, 43, 585229, tzinfo=tzutc())}], 'pipelines': []}

