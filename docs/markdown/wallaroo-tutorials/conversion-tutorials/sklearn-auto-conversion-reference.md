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
    1.1.1


The following code is used to either connect to an existing workspace or to create a new one.  For more details on working with workspaces, see the [Wallaroo Workspace Management Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-workspace-management/).

### Connect to Wallaroo

Connect to your Wallaroo instance.


```python
# Client connection from local Wallaroo instance

wl = wallaroo.Client()

# SSO login through keycloak

# wallarooPrefix = "YOUR PREFIX"
# wallarooSuffix = "YOUR SUFFIX"

# wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}.api.{wallarooSuffix}", 
#                     auth_endpoint=f"https://{wallarooPrefix}.keycloak.{wallarooSuffix}", 
#                     auth_type="sso")
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




    {'name': 'testautoconversion', 'id': 22, 'archived': False, 'created_by': '56d97480-bb64-4575-acb6-f93d05652e86', 'created_at': '2023-03-29T16:33:09.14146+00:00', 'models': [], 'pipelines': []}



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




    {'name': 'xgbtestreg', 'version': 'ac5601d9-02fd-4817-857e-7cee8de7dd93', 'file_name': 'd2aa8c58-4e5f-4e6a-8ede-782bfd5ac68f-converted.onnx', 'image_path': None, 'last_update_time': datetime.datetime(2023, 3, 29, 16, 33, 11, 977787, tzinfo=tzutc())}




```python
wl.get_current_workspace()
```




    {'name': 'testautoconversion', 'id': 22, 'archived': False, 'created_by': '56d97480-bb64-4575-acb6-f93d05652e86', 'created_at': '2023-03-29T16:33:09.14146+00:00', 'models': [{'name': 'sklearntest', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 3, 29, 16, 33, 10, 860213, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 3, 29, 16, 33, 10, 860213, tzinfo=tzutc())}, {'name': 'xgbtestreg', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 3, 29, 16, 33, 11, 977787, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 3, 29, 16, 33, 11, 977787, tzinfo=tzutc())}], 'pipelines': []}


