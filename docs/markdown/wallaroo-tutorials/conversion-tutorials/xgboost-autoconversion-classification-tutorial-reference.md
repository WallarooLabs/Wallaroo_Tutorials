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

import os
# For Wallaroo SDK 2023.1
os.environ["ARROW_ENABLED"]="True"

# used to display dataframe information without truncating
from IPython.display import display
import pandas as pd
pd.set_option('display.max_colwidth', None)
```

### Connect to Wallaroo

Connect to your Wallaroo instance and store the connection into the variable `wl`.


```python
# Login through local Wallaroo instance

wl = wallaroo.Client()

# SSO login through keycloak

# wallarooPrefix = "YOUR PREFIX"
# wallarooSuffix = "YOUR PREFIX"

# wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}.api.{wallarooSuffix}", 
#                     auth_endpoint=f"https://{wallarooPrefix}.keycloak.{wallarooSuffix}", 
#                     auth_type="sso")
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
        pipeline = wl.pipelines_by_name(pipeline_name)[0]
    except EntityNotFoundError:
        pipeline = wl.build_pipeline(pipeline_name)
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




<table><tr><th>name</th> <td>xgboost-classification-autoconvert-pipeline</td></tr><tr><th>created</th> <td>2023-02-22 17:28:53.153447+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-27 20:53:34.397083+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>6017f564-ab0d-4a67-afde-facf2eb601b3, 9020e9aa-ce33-4dcb-a222-6c732d6aeb83, e97e4287-b077-4405-879c-82af763b7b18, 94d9c2f9-45b6-489e-9ceb-912791873086, 84a06ef1-6959-47b3-bacb-3af325d4612e, d57f666c-c958-4eec-a46e-b7e28b145614, 5314a152-8120-4f32-9663-6b01601864e2</td></tr><tr><th>steps</th> <td>xgb-class-model</td></tr></table>



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


    ---------------------------------------------------------------------------

    HTTPError                                 Traceback (most recent call last)

    /opt/homebrew/anaconda3/envs/arrowtests/lib/python3.8/site-packages/wallaroo/client.py in convert_model(self, path, source_type, conversion_arguments)
       1626                 )
    -> 1627                 model_id = _handle_response(response)
       1628             else:


    /opt/homebrew/anaconda3/envs/arrowtests/lib/python3.8/site-packages/wallaroo/client.py in _handle_response(http_response)
       1569         def _handle_response(http_response) -> int:
    -> 1570             http_response.raise_for_status()
       1571             response_record = http_response.json()


    /opt/homebrew/anaconda3/envs/arrowtests/lib/python3.8/site-packages/requests/models.py in raise_for_status(self)
       1020         if http_error_msg:
    -> 1021             raise HTTPError(http_error_msg, response=self)
       1022 


    HTTPError: 500 Server Error: Internal Server Error for url: https://wallaroo.api.example.com/v1/convert/xgboost?name=xgb-class-model&number_of_columns=25&input_type=float32&comment=xgboost+classification+model+test&workspace_id=63

    
    During handling of the above exception, another exception occurred:


    ModelConversionGenericException           Traceback (most recent call last)

    /var/folders/jf/_cj0q9d51s365wksymljdz4h0000gn/T/ipykernel_10455/2322016990.py in <module>
          1 # convert and upload
    ----> 2 model_wl = wl.convert_model(model_file_name, model_conversion_type, model_conversion_args)
    

    /opt/homebrew/anaconda3/envs/arrowtests/lib/python3.8/site-packages/wallaroo/client.py in convert_model(self, path, source_type, conversion_arguments)
       1634             return Model(self, {"id": model_id})
       1635         except Exception:
    -> 1636             raise ModelConversionGenericException(
       1637                 "This model type could not be deployed successfully. Please contact your Wallaroo support team at community@wallaroo.ai"
       1638             )


    ModelConversionGenericException: This model type could not be deployed successfully. Please contact your Wallaroo support team at community@wallaroo.ai


## Test Inference

With the model uploaded and converted, we can run a sample inference.

### Deploy the Pipeline

Add the uploaded and converted `model_wl` as a step in the pipeline, then deploy it.


```python
pipeline.add_model_step(model_wl).deploy()
```

### Run the Inference

Use the evaluation data to verify the process completed successfully.


```python
sample_data = 'xgb_class_eval.df.json'
result = pipeline.infer_from_file(sample_data)
display(result)
```

### Undeploy the Pipeline

With the tests complete, we will undeploy the pipeline to return the resources back to the Wallaroo instance.


```python
#pipeline.undeploy()
```
