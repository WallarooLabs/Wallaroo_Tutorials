This tutorial can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/blob/2023.2.1_prerelease/model_uploads/xgboost-upload-tutorials).

## Wallaroo Model Upload via the Wallaroo SDK: XGBoost Classification

The following tutorial demonstrates how to upload a XGBoost Classification model to a Wallaroo instance.

### Tutorial Goals

Demonstrate the following:

* Upload a XGBoost Classification model to a Wallaroo instance.
* Create a pipeline and add the model as a pipeline step.
* Perform a sample inference.

### Prerequisites

* A Wallaroo version 2023.2.1 or above instance.

### References

* [Wallaroo MLOps API Essentials Guide: Model Upload and Registrations](https://staging.docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-api-essential-guide/wallaroo-mlops-api-essentials-guide-model-uploads/)
* [Wallaroo API Connection Guide](https://staging.docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-connection-guide/)
* [DNS Integration Guide](https://staging.docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-dns-guide/)

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
workspace_name = f'xgboost-classification{suffix}'
pipeline_name = f'xgboost-classification'

model_name = 'xgboost-classification'
model_file_name = './models/model-auto-conversion_xgboost_xgb_classification_iris.pkl'
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
    pa.field('inputs', pa.list_(pa.float64(), list_size=4))
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
    Model is Pending conversion........Converting.............Ready.

<table>
        <tr>
          <td>Name</td>
          <td>xgboost-classification</td>
        </tr>
        <tr>
          <td>Version</td>
          <td>35b2bba5-cfe5-46f0-8a17-242e022fa9d1</td>
        </tr>
        <tr>
          <td>File Name</td>
          <td>model-auto-conversion_xgboost_xgb_classification_iris.pkl</td>
        </tr>
        <tr>
          <td>SHA</td>
          <td>4a1844c460e8c8503207305fb807e3a28e788062588925021807c54ee80cc7f9</td>
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
          <td>2023-13-Jul 19:29:33</td>
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
     'engines': [{'ip': '10.244.9.234',
       'name': 'engine-cf56c9c99-l76rt',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'xgboost-classification',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'xgboost-classification',
          'version': '35b2bba5-cfe5-46f0-8a17-242e022fa9d1',
          'sha': '4a1844c460e8c8503207305fb807e3a28e788062588925021807c54ee80cc7f9',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.9.235',
       'name': 'engine-lb-584f54c899-c29jw',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': [{'ip': '10.244.9.233',
       'name': 'engine-sidekick-xgboost-classification-284-5fddbdc4c8-6vttd',
       'status': 'Running',
       'reason': None,
       'details': [],
       'statuses': '\n'}]}

### Run Inference

A sample inference will be run.  First the pandas DataFrame used for the inference is created, then the inference run through the pipeline's `infer` method.

```python
data = pd.read_json('data/test-xgboost-classification-data.json')
display(data)

dataframe = pd.DataFrame({"inputs": data[:2].values.tolist()})
display(dataframe)

results = pipeline.infer(dataframe)
display(results)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
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
      <td>[5.1, 3.5, 1.4, 0.2]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[4.9, 3.0, 1.4, 0.2]</td>
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
      <td>2023-07-13 19:29:51.614</td>
      <td>[5.1, 3.5, 1.4, 0.2]</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-07-13 19:29:51.614</td>
      <td>[4.9, 3.0, 1.4, 0.2]</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

### Undeploy Pipelines

With the tutorial complete, the pipeline is undeployed to return the resources back to the cluster.

```python
pipeline.undeploy()
```

<table><tr><th>name</th> <td>xgboost-classification</td></tr><tr><th>created</th> <td>2023-07-13 19:27:34.750372+00:00</td></tr><tr><th>last_updated</th> <td>2023-07-13 19:29:35.910554+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>4d693511-388f-48a5-b0d1-e5aa5b11538a, 20e79a32-0739-413a-84c3-6695adc901ec</td></tr><tr><th>steps</th> <td>xgboost-classification</td></tr></table>

