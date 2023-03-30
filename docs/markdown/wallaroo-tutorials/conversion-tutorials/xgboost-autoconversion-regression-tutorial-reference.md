This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/model_conversion/xgboost-autoconversion).

## Introduction

The following tutorial is a brief example of how to convert a [XGBoost](https://xgboost.readthedocs.io/en/stable/index.html) Regression ML model with the `convert_model` method and upload it into your Wallaroo instance.

This tutorial assumes that you have a Wallaroo instance and are running this Notebook from the Wallaroo Jupyter Hub service.

* Convert a `XGBoost` Regression ML model and upload it into the Wallaroo engine.
* Run a sample inference on the converted model in a Wallaroo instance.

This tutorial provides the following:

* `xgb_reg.pickle`: A pretrained `XGBoost` Regression model with 25 columns.
* `xgb_regression_eval.json`: Test data to perform a sample inference.

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
workspace_name = 'xgboost-regression-autoconvert-workspace'
pipeline_name = 'xgboost-regression-autoconvert-pipeline'
model_name = 'xgb-regression-model'
model_file_name = 'xgb_reg.pickle'

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




<table><tr><th>name</th> <td>xgboost-regression-autoconvert-pipeline</td></tr><tr><th>created</th> <td>2023-02-22 17:36:31.143795+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-24 16:26:10.105654+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>73345b3e-e141-497e-bcab-e6145cb8b6ec, 6f6a7cf1-b328-4a67-a37d-4047b1082516, 00bb4dd6-3061-4619-9491-a3b7c307f784, 45081f0c-1991-4ce0-907c-6135ca328084, 1220c119-45e9-4ff4-bdfd-8ff2f95486d5</td></tr><tr><th>steps</th> <td>xgb-regression-model</td></tr></table>



### Set the Model Autoconvert Parameters

Set the paramters for converting the `xgb-class-model`.


```python
#the number of columns
NF = 25

model_conversion_args = ConvertXGBoostArgs(
    name=model_name,
    comment="xgboost regression model test",
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


    HTTPError: 500 Server Error: Internal Server Error for url: https://wallaroo.api.example.com/v1/convert/xgboost?name=xgb-regression-model&number_of_columns=25&input_type=float32&comment=xgboost+regression+model+test&workspace_id=64

    
    During handling of the above exception, another exception occurred:


    ModelConversionGenericException           Traceback (most recent call last)

    /var/folders/jf/_cj0q9d51s365wksymljdz4h0000gn/T/ipykernel_10651/2322016990.py in <module>
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

    Waiting for deployment - this will take up to 45s ............. ok





<table><tr><th>name</th> <td>xgboost-regression-autoconvert-pipeline</td></tr><tr><th>created</th> <td>2023-02-22 17:36:31.143795+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-24 16:26:10.105654+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>73345b3e-e141-497e-bcab-e6145cb8b6ec, 6f6a7cf1-b328-4a67-a37d-4047b1082516, 00bb4dd6-3061-4619-9491-a3b7c307f784, 45081f0c-1991-4ce0-907c-6135ca328084, 1220c119-45e9-4ff4-bdfd-8ff2f95486d5</td></tr><tr><th>steps</th> <td>xgb-regression-model</td></tr></table>




```python
pipeline.status()
```




    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.48.1.19',
       'name': 'engine-749bb6f4fd-4qt2x',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'xgboost-regression-autoconvert-pipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'xgb-regression-model',
          'version': '2f30475f-6385-4c5e-bfbb-9d47fa84f8ae',
          'sha': '7414d26cb5495269bc54bcfeebd269d7c74412cbfca07562fc7cb184c55b6f8e',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.48.1.20',
       'name': 'engine-lb-74b4969486-7jzwh',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}



### Run the Inference

Use the `test_class_eval.json` as set earlier as our `sample_data` and perform the inference.


```python
if arrowEnabled is True:
    sample_data = 'xgb_regression_eval.df.json'
    result = pipeline.infer_from_file(sample_data)
    display(result)
else:
    sample_data = 'xgb_regression_eval.json'
    result = pipeline.infer_from_file(sample_data)
    result[0].data()
    
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.variable</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-02-24 16:26:24.157</td>
      <td>[-0.0337420814, -0.1876901281, 0.3183056488, 1.1831088244, -0.3047963287, 1.0713634828, 0.4679136198, 1.1382147115, 2.8101110944, -0.9981048796, -0.2543715265, 0.2845195171, -0.6477265924, -1.2198006181, 2.0592129832, -1.586429512, 0.1884164743, -0.3816011585, 1.0781704305, -0.2251253601, 0.6067409459, 0.9659944831, -0.690207203, -0.3849078305, -1.7806555641]</td>
      <td>[124.11397]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-02-24 16:26:24.157</td>
      <td>[-0.6374335428, 0.9713713274, -0.3899847809, -1.8685333445, 0.6264452739, 1.0778638153, -1.1687273967, -1.9366353171, -0.7583260267, -0.1288186991, 2.2018769654, -0.9383105208, -0.0959982166, 0.6889112707, 1.0172067951, -0.1988865499, 1.3461760224, -0.5692275708, 0.0112450486, -1.0244657911, -0.0065034946, -0.888033574, 2.5997682335, -0.6593191496, 0.4554196997]</td>
      <td>[-238.03009]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-02-24 16:26:24.157</td>
      <td>[0.9847406173, -0.6887896553, -0.9483266359, -0.6146245598, 0.395195321, 0.2237676197, -2.1580851068, -0.8124396117, 0.8795326949, 1.0463472648, -0.2343060791, 1.9127900859, -0.0636431887, 2.7055743269, 1.424242505, 0.1486958646, -0.7771892138, -0.6720552548, 0.9127712446, 0.680721406, 1.5207886874, 1.9579334337, -0.9336538468, -0.2942243461, 0.8563934417]</td>
      <td>[253.06976]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-02-24 16:26:24.157</td>
      <td>[-0.0894312686, 2.0916777545, 0.155086745, 0.8335388277, 0.4376497549, -0.2875695352, -1.272466627, -0.8226918076, -0.8637972417, -0.4856051115, -0.978749107, 0.2675108269, 0.5246808262, -0.96869578, 0.8475004997, 1.0027495438, 0.4704188579, 2.6906210825, 1.34454675, -1.4987055653, 0.680752942, -2.6459314502, 0.6274277031, 1.3640818416, -0.8077878088]</td>
      <td>[141.34639]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-02-24 16:26:24.157</td>
      <td>[-0.9200220805, -1.8760634694, -0.8277296049, 0.6511561005, 1.5066237509, -1.1236118386, -0.3776053288, -0.0445487434, -1.4965713379, -0.1756118518, 0.0317408338, 0.2496108303, 1.6857141605, 0.0339106658, -0.3340227553, -0.3428326984, -0.5932644698, -0.4395685475, -0.6870452688, -0.4132149028, -0.7352879532, 0.2080507404, 0.4575261189, -2.0175947284, 1.154633581]</td>
      <td>[42.79154]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


### Undeploy the Pipeline

With the tests complete, we will undeploy the pipeline to return the resources back to the Wallaroo instance.


```python
pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s ..................................... ok





<table><tr><th>name</th> <td>xgboost-regression-autoconvert-pipeline</td></tr><tr><th>created</th> <td>2023-02-22 17:36:31.143795+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-24 16:26:10.105654+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>73345b3e-e141-497e-bcab-e6145cb8b6ec, 6f6a7cf1-b328-4a67-a37d-4047b1082516, 00bb4dd6-3061-4619-9491-a3b7c307f784, 45081f0c-1991-4ce0-907c-6135ca328084, 1220c119-45e9-4ff4-bdfd-8ff2f95486d5</td></tr><tr><th>steps</th> <td>xgb-regression-model</td></tr></table>


