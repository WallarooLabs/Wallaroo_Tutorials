This tutorial can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/model_uploads/xgboost-upload-tutorials).

## Wallaroo Model Upload via the Wallaroo SDK: XGBoost RF Regressor

The following tutorial demonstrates how to upload a XGBoost RF Regressor model to a Wallaroo instance.

### Tutorial Goals

Demonstrate the following:

* Upload a XGBoost RF Regressor model to a Wallaroo instance.
* Create a pipeline and add the model as a pipeline step.
* Perform a sample inference.

### Prerequisites

* A Wallaroo version 2023.2.1 or above instance.

### References

* [Wallaroo MLOps API Essentials Guide: Model Upload and Registrations](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-api-essential-guide/wallaroo-mlops-api-essentials-guide-model-uploads/)
* [Wallaroo API Connection Guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-connection-guide/)
* [DNS Integration Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-dns-guide/)

## Tutorial Steps

### Import Libraries

The first step is to import the libraries we'll be using.  These are included by default in the Wallaroo instance's JupyterHub service.

```python
import json
import os
import pickle

import wallaroo
from wallaroo.pipeline import Pipeline
from wallaroo.deployment_config import DeploymentConfigBuilder
from wallaroo.object import EntityNotFoundError
from wallaroo.framework import Framework

import os
os.environ["MODELS_ENABLED"] = "true"

import pyarrow as pa
import numpy as np
import pandas as pd
```

### Open a Connection to Wallaroo

The next step is connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  If logging in externally, update the `wallarooPrefix` and `wallarooSuffix` variables with the proper DNS information.  For more information on Wallaroo DNS settings, see the [Wallaroo DNS Integration Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-dns-guide/).

```python
wl = wallaroo.Client()

wallarooPrefix = ""
wallarooSuffix = "autoscale-uat-ee.wallaroo.dev"

wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}api.{wallarooSuffix}", 
                    auth_endpoint=f"https://{wallarooPrefix}keycloak.{wallarooSuffix}", 
                    auth_type="sso")
```

### Set Variables and Helper Functions

We'll set the name of our workspace, pipeline, models and files.  Workspace names must be unique across the Wallaroo workspace.  For this, we'll add in a randomly generated 4 characters to the workspace name to prevent collisions with other users' workspaces.  If running this tutorial, we recommend hard coding the workspace name so it will function in the same workspace each time it's run.

We'll set up some helper functions that will either use existing workspaces and pipelines, or create them if they do not already exist.

```python
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

import string
import random

# make a random 4 character suffix to prevent overwriting other user's workspaces
suffix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))
workspace_name = f'xgboost-rf-regressor{suffix}'
pipeline_name = f'xgboost-rf-regressor'

model_name = 'xgboost-rf-regressor'
model_file_name = 'models/model-auto-conversion_xgboost_xgb_rf_regressor_diabetes.pkl'
```

### Create Workspace and Pipeline

We will now create the Wallaroo workspace to store our model and set it as the current workspace.  Future commands will default to this workspace for pipeline creation, model uploads, etc.  We'll create our Wallaroo pipeline to deploy our model.

```python
workspace = get_workspace(workspace_name)
wl.set_current_workspace(workspace)

pipeline = get_pipeline(pipeline_name)
```

### Configure Data Schemas

XGBoost models are uploaded to Wallaroo through the Wallaroo Client `upload_model` method.

### Upload XGBoost Model Parameters

The following parameters are required for XGBoost models.  Note that while some fields are considered as **optional** for the `upload_model` method, they are required for proper uploading of a XGBoost model to Wallaroo.

| Parameter | Type | Description |
|---|---|---|
|`name` | `string` (*Required*) | The name of the model.  Model names are unique per workspace.  Models that are uploaded with the same name are assigned as a new **version** of the model. |
|`path` | `string` (*Required*) | The path to the model file being uploaded. 
|`framework` |`string` (*Upload Method Optional, SKLearn model Required*) | Set as the `Framework.XGBOOST`. |
|`input_schema` | `pyarrow.lib.Schema` (*Upload Method Optional, SKLearn model Required*) | The input schema in Apache Arrow schema format. |
|`output_schema` | `pyarrow.lib.Schema` (*Upload Method Optional, SKLearn model Required*) | The output schema in Apache Arrow schema format. |
| `convert_wait` | `bool` (*Upload Method Optional, SKLearn model Optional*) (*Default: True*) | <ul><li>**True**: Waits in the script for the model conversion completion.</li><li>**False**:  Proceeds with the script without waiting for the model conversion process to display complete. |

Once the upload process starts, the model is containerized by the Wallaroo instance.  This process may take up to 10 minutes.

#### XGBoost Schema Inputs

XGBoost schema follows a different format than other models.  To prevent inputs from being out of order, the inputs should be submitted in a single row in the order the model is trained to accept, with **all of the data types being the same**.  If a model is originally trained to accept inputs of different data types, it will need to be retrained to only accept one data type for each column - typically `pa.float64()` is a good choice.

For example, the following DataFrame has 4 columns, each column a `float`.

|&nbsp;|sepal length (cm)|sepal width (cm)|petal length (cm)|petal width (cm)|
|---|---|---|---|---|
|0|5.1|3.5|1.4|0.2|
|1|4.9|3.0|1.4|0.2|

For submission to an XGBoost model, the data input schema will be a single array with 4 float values.

```python
input_schema = pa.schema([
    pa.field('inputs', pa.list_(pa.float64(), list_size=4))
])
```

When submitting as an inference, the DataFrame is converted to rows with the column data expressed as a single array.  The data **must** be in the same order as the model expects, which is why the data is submitted as a single array rather than JSON labeled columns:  this insures that the data is submitted in the exact order as the model is trained to accept.

Original DataFrame:

&nbsp;|sepal length (cm)|sepal width (cm)|petal length (cm)|petal width (cm)
|---|---|---|---|---|
0|5.1|3.5|1.4|0.2
1|4.9|3.0|1.4|0.2

Converted DataFrame:

|&nbsp;|inputs|
|---|---|
|0|[5.1, 3.5, 1.4, 0.2]|
|1|[4.9, 3.0, 1.4, 0.2]|

#### XGBoost Schema Outputs

Outputs for XGBoost are labeled based on the trained model outputs.  For this example, the output is simply a single output listed as `output`.  In the Wallaroo inference result, it is grouped with the metadata `out` as `out.output`.

```python
output_schema = pa.schema([
    pa.field('output', pa.int32())
])
```

```python
pipeline.infer(dataframe)
```

|&nbsp;|time|in.inputs|out.output|check_failures|
|---|---|---|---|---|
|0|2023-07-05 15:11:29.776|[5.1, 3.5, 1.4, 0.2]|0|0|
|1|2023-07-05 15:11:29.776|[4.9, 3.0, 1.4, 0.2]|0|0|

```python
input_schema = pa.schema([
    pa.field('inputs', pa.list_(pa.float64(), list_size=10))
])

output_schema = pa.schema([
    pa.field('output', pa.float64())
])
```

### Upload Model

The model will be uploaded with the framework set as `Framework.XGBOOST`.

```python
model = wl.upload_model(model_name, 
                        model_file_name, 
                        framework=Framework.XGBOOST, 
                        input_schema=input_schema, 
                        output_schema=output_schema)
model
```

    Waiting for model conversion... It may take up to 10.0min.
    Model is Pending conversion.Converting..Pending conversion.Converting........Ready.

<table>
        <tr>
          <td>Name</td>
          <td>xgboost-rf-regressor</td>
        </tr>
        <tr>
          <td>Version</td>
          <td>18786ea1-2964-4406-ad1b-2eea413326d8</td>
        </tr>
        <tr>
          <td>File Name</td>
          <td>model-auto-conversion_xgboost_xgb_rf_regressor_diabetes.pkl</td>
        </tr>
        <tr>
          <td>SHA</td>
          <td>461341d78d54a9bfc8e4faa94be6037aef15217974ba59bad92d31ef48e6bd99</td>
        </tr>
        <tr>
          <td>Status</td>
          <td>ready</td>
        </tr>
        <tr>
          <td>Image Path</td>
          <td>proxy.replicated.com/proxy/wallaroo/ghcr.io/wallaroolabs/mlflow-deploy:v2023.3.0-main-3509</td>
        </tr>
        <tr>
          <td>Updated At</td>
          <td>2023-13-Jul 20:02:19</td>
        </tr>
      </table>

```python
model.config().runtime()
```

    'mlflow'

### Deploy Pipeline

The model is uploaded and ready for use.  We'll add it as a step in our pipeline, then deploy the pipeline.  For this example we're allocated 0.25 cpu and 4 Gi RAM to the pipeline through the pipeline's deployment configuration.

```python
deployment_config = DeploymentConfigBuilder() \
    .cpus(0.25).memory('1Gi') \
    .build()
```

```python
# clear the pipeline if it was used before
pipeline.undeploy()
pipeline.clear()

pipeline.add_model_step(model)

pipeline.deploy(deployment_config=deployment_config)
pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.9.3',
       'name': 'engine-68746578dc-zh47j',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'xgboost-rf-regressor',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'xgboost-rf-regressor',
          'version': '18786ea1-2964-4406-ad1b-2eea413326d8',
          'sha': '461341d78d54a9bfc8e4faa94be6037aef15217974ba59bad92d31ef48e6bd99',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.9.5',
       'name': 'engine-lb-584f54c899-kxfbn',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': [{'ip': '10.244.9.6',
       'name': 'engine-sidekick-xgboost-rf-regressor-288-cfcc58bb4-b47cl',
       'status': 'Running',
       'reason': None,
       'details': [],
       'statuses': '\n'}]}

### Run Inference

A sample inference will be run.  First the pandas DataFrame used for the inference is created, then the inference run through the pipeline's `infer` method.

```python
data = pd.read_json('./data/test_xgb_rf-regressor.json')
display(data)

dataframe = pd.DataFrame({"inputs": data[:2].values.tolist()})
display(dataframe)

pipeline.infer(dataframe)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>bp</th>
      <th>s1</th>
      <th>s2</th>
      <th>s3</th>
      <th>s4</th>
      <th>s5</th>
      <th>s6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.038076</td>
      <td>0.050680</td>
      <td>0.061696</td>
      <td>0.021872</td>
      <td>-0.044223</td>
      <td>-0.034821</td>
      <td>-0.043401</td>
      <td>-0.002592</td>
      <td>0.019907</td>
      <td>-0.017646</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.001882</td>
      <td>-0.044642</td>
      <td>-0.051474</td>
      <td>-0.026328</td>
      <td>-0.008449</td>
      <td>-0.019163</td>
      <td>0.074412</td>
      <td>-0.039493</td>
      <td>-0.068332</td>
      <td>-0.092204</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>inputs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[0.0380759064, 0.0506801187, 0.0616962065, 0.0...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[-0.0018820165, -0.0446416365, -0.051474061200...</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.inputs</th>
      <th>out.output</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-07-13 20:02:41.843</td>
      <td>[0.0380759064, 0.0506801187, 0.0616962065, 0.0...</td>
      <td>166.618774</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-07-13 20:02:41.843</td>
      <td>[-0.0018820165, -0.0446416365, -0.0514740612, ...</td>
      <td>76.189583</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

### Undeploy Pipelines

With the tutorial complete, the pipeline is undeployed to return the resources back to the cluster.

```python
pipeline.undeploy()
```

<table><tr><th>name</th> <td>xgboost-rf-regressor</td></tr><tr><th>created</th> <td>2023-07-13 20:01:14.889724+00:00</td></tr><tr><th>last_updated</th> <td>2023-07-13 20:02:25.945653+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>c9d392fa-96d7-4709-810f-beae105c879f, cb4f88e6-2c8c-4fef-beb5-5718bc64a2ea</td></tr><tr><th>steps</th> <td>xgboost-rf-regressor</td></tr></table>

