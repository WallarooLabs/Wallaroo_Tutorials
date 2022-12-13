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

## Steps

### Prerequisites

Before starting, the following must be available:

* The model to upload into a workspace.
* The number of columns the model was trained for.

### Import Libraries

Import the libraries that will be used for the auto-conversion process.

```python
import pickle
import json

import wallaroo

from wallaroo.ModelConversion import ConvertSKLearnArguments, ConvertXGBoostArgs, ModelConversionSource, ModelConversionInputType
from wallaroo.object import EntityNotFoundError
```

The following code is used to either connect to an existing workspace or to create a new one.  For more details on working with workspaces, see the [Wallaroo Workspace Management Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-workspace-management/).

### Connect to Wallaroo

Connect to your Wallaroo instance.

```python
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



    {'name': 'testautoconversion', 'id': 12, 'archived': False, 'created_by': '13f4ce0d-cb22-4a5c-b07b-c65e4d730315', 'created_at': '2022-08-02T22:16:30.552476+00:00', 'models': [], 'pipelines': []}


### Set the Model Conversion Arguments

We'll create two different configurations, one for each of our models:

* `sklearn_model_conversion_args`: Used for our sklearn model.
* `xgboost_model_converstion_args`: Used for our XGBoost model.

```python
# The number of columns
NF=25

sklearn_model_conversion_args = ConvertSKLearnArguments(
    name="lm-test",
    comment="test linear regression",
    number_of_columns=NF,
    input_type=ModelConversionInputType.Double
)
sklearn_model_conversion_type = ModelConversionSource.SKLEARN

xgboost_model_conversion_args = ConvertXGBoostArgs(
    name="xgb-test-reg",
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



    {'name': 'xgb-test-reg', 'version': '9ade0e7a-dc3f-4935-8974-ed8bda12d148', 'file_name': '39c215bb-ae23-4a05-b520-aa0b8d94ba42-converted.onnx', 'last_update_time': datetime.datetime(2022, 8, 3, 14, 26, 58, 413122, tzinfo=tzutc())}



```python
wl.get_current_workspace()
```



    {'name': 'testautoconversion', 'id': 12, 'archived': False, 'created_by': '13f4ce0d-cb22-4a5c-b07b-c65e4d730315', 'created_at': '2022-08-02T22:16:30.552476+00:00', 'models': [{'name': 'lm-test', 'version': '2227f4a5-3139-4bc8-844c-3587546f326a', 'file_name': '2fb7d46d-d92f-4371-872c-5300c52188bb-converted.onnx', 'last_update_time': datetime.datetime(2022, 8, 3, 14, 26, 55, 892457, tzinfo=tzutc())}, {'name': 'xgb-test-reg', 'version': '9ade0e7a-dc3f-4935-8974-ed8bda12d148', 'file_name': '39c215bb-ae23-4a05-b520-aa0b8d94ba42-converted.onnx', 'last_update_time': datetime.datetime(2022, 8, 3, 14, 26, 58, 413122, tzinfo=tzutc())}], 'pipelines': []}

