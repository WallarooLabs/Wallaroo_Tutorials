This tutorial can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/model_uploads/keras-upload-tutorials).

## Wallaroo Model Upload via the Wallaroo SDK: TensorFlow keras Sequential Single IO

The following tutorial demonstrates how to upload a TensorFlow keras Sequential Single IO model to a Wallaroo instance.

### Tutorial Goals

Demonstrate the following:

* Upload a TensorFlow keras Sequential Single IO to a Wallaroo instance.
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
from wallaroo.pipeline   import Pipeline
from wallaroo.deployment_config import DeploymentConfigBuilder
from wallaroo.framework import Framework
from wallaroo.object import EntityNotFoundError

import os
os.environ["MODELS_ENABLED"] = "true"

import pyarrow as pa
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

import datetime
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
workspace_name = f'keras-sequential-single-io{suffix}'
pipeline_name = f'keras-sequential-single-io'

model_name = 'keras-sequential-single-io'
model_file_name = 'models/model-auto-conversion_keras_single_io_keras_sequential_model.h5'
```

### Create Workspace and Pipeline

We will now create the Wallaroo workspace to store our model and set it as the current workspace.  Future commands will default to this workspace for pipeline creation, model uploads, etc.  We'll create our Wallaroo pipeline to deploy our model.

```python
workspace = get_workspace(workspace_name)
wl.set_current_workspace(workspace)

pipeline = get_pipeline(pipeline_name)
```

### Configure Data Schemas

The following parameters are required for TensorFlow keras models.  Note that while some fields are considered as **optional** for the `upload_model` method, they are required for proper uploading of a TensorFlow Keras model to Wallaroo.

| Parameter | Type | Description |
|---|---|---|
|`name` | `string` (*Required*) | The name of the model.  Model names are unique per workspace.  Models that are uploaded with the same name are assigned as a new **version** of the model. |
|`path` | `string` (*Required*) | The path to the model file being uploaded. 
|`framework` |`string` (*Upload Method Optional, TensorFlow keras model Required*) | Set as the `Framework.KERAS`. |
|`input_schema` | `pyarrow.lib.Schema` (*Upload Method Optional, TensorFlow Keras model Required*) | The input schema in Apache Arrow schema format. |
|`output_schema` | `pyarrow.lib.Schema` (*Upload Method Optional, TensorFlow Keras model Required*) | The output schema in Apache Arrow schema format. |
| `convert_wait` | `bool` (*Upload Method Optional, TensorFlow model Optional*) (*Default: True*) | <ul><li>**True**: Waits in the script for the model conversion completion.</li><li>**False**:  Proceeds with the script without waiting for the model conversion process to display complete. |

Once the upload process starts, the model is containerized by the Wallaroo instance.  This process may take up to 10 minutes.

```python
input_schema = pa.schema([
    pa.field('input', pa.list_(pa.float64(), list_size=10))
])
output_schema = pa.schema([
    pa.field('output', pa.list_(pa.float64(), list_size=32))
])
```

### Upload Model

The model will be uploaded with the framework set as `Framework.KERAS`.

```python
framework=Framework.KERAS

model = wl.upload_model(model_name, 
                        model_file_name, 
                        framework=framework, 
                        input_schema=input_schema, 
                        output_schema=output_schema)
model
```

    Waiting for model conversion... It may take up to 10.0min.
    Model is Pending conversion.Converting..Pending conversion.Converting.............Ready.

<table>
        <tr>
          <td>Name</td>
          <td>keras-sequential-single-io</td>
        </tr>
        <tr>
          <td>Version</td>
          <td>79e34aa4-ce71-4c3a-90fc-79bfa0a40052</td>
        </tr>
        <tr>
          <td>File Name</td>
          <td>model-auto-conversion_keras_single_io_keras_sequential_model.h5</td>
        </tr>
        <tr>
          <td>SHA</td>
          <td>f7e49538e38bebe066ce8df97bac8be239ae8c7d2733e500c8cd633706ae95a8</td>
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
          <td>2023-13-Jul 17:50:07</td>
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
# clear the pipeline if used in a previous tutorial
pipeline.undeploy()
pipeline.clear()
pipeline.add_model_step(model)

pipeline.deploy(deployment_config=deployment_config)
pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.9.178',
       'name': 'engine-75cb64bc94-lc74f',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'keras-sequential-single-io',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'keras-sequential-single-io',
          'version': '79e34aa4-ce71-4c3a-90fc-79bfa0a40052',
          'sha': 'f7e49538e38bebe066ce8df97bac8be239ae8c7d2733e500c8cd633706ae95a8',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.9.179',
       'name': 'engine-lb-584f54c899-96vb6',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': [{'ip': '10.244.9.180',
       'name': 'engine-sidekick-keras-sequential-single-io-268-84cff895cf-bm2tl',
       'status': 'Running',
       'reason': None,
       'details': [],
       'statuses': '\n'}]}

### Run Inference

A sample inference will be run.  First the pandas DataFrame used for the inference is created, then the inference run through the pipeline's `infer` method.

```python
input_data = np.random.rand(10, 10)
mock_dataframe = pd.DataFrame({
    "input": input_data.tolist()
})
mock_dataframe
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>input</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[0.5809853116232783, 0.14701285145269583, 0.58...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[0.7257919427827948, 0.7589800713851653, 0.297...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[0.3764542634917982, 0.5494748793973108, 0.485...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[0.5025570851921953, 0.8837007217828465, 0.406...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[0.0866396068940275, 0.10670979528669655, 0.09...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>[0.8860315511881905, 0.6706861365257704, 0.412...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>[0.37994954016981175, 0.7429705751348403, 0.12...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>[0.49027013691203447, 0.7105289734919781, 0.99...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>[0.4446469438043267, 0.09139454740740094, 0.24...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>[0.932824358657356, 0.3388034065847041, 0.0416...</td>
    </tr>
  </tbody>
</table>

```python
pipeline.infer(mock_dataframe)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.input</th>
      <th>out.output</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-07-13 17:50:42.609</td>
      <td>[0.5809853116, 0.1470128515, 0.5859677386, 0.2...</td>
      <td>[0.025315184146165848, 0.023196307942271233, 0...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-07-13 17:50:42.609</td>
      <td>[0.7257919428, 0.7589800714, 0.297258173, 0.39...</td>
      <td>[0.022579584270715714, 0.026824792847037315, 0...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-07-13 17:50:42.609</td>
      <td>[0.3764542635, 0.5494748794, 0.4852001553, 0.8...</td>
      <td>[0.02744304947555065, 0.03327963128685951, 0.0...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-07-13 17:50:42.609</td>
      <td>[0.5025570852, 0.8837007218, 0.4064710644, 0.5...</td>
      <td>[0.03851581737399101, 0.021599330008029938, 0....</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-07-13 17:50:42.609</td>
      <td>[0.0866396069, 0.1067097953, 0.0918865633, 0.2...</td>
      <td>[0.020835522562265396, 0.034067943692207336, 0...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2023-07-13 17:50:42.609</td>
      <td>[0.8860315512, 0.6706861365, 0.4123840879, 0.2...</td>
      <td>[0.034137945622205734, 0.01922944001853466, 0....</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2023-07-13 17:50:42.609</td>
      <td>[0.3799495402, 0.7429705751, 0.1207460912, 0.3...</td>
      <td>[0.03986137732863426, 0.019290560856461525, 0....</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2023-07-13 17:50:42.609</td>
      <td>[0.4902701369, 0.7105289735, 0.9948842471, 0.2...</td>
      <td>[0.026285773143172264, 0.02646280638873577, 0....</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2023-07-13 17:50:42.609</td>
      <td>[0.4446469438, 0.0913945474, 0.24660973, 0.456...</td>
      <td>[0.023244783282279968, 0.033836156129837036, 0...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2023-07-13 17:50:42.609</td>
      <td>[0.9328243587, 0.3388034066, 0.0416730168, 0.4...</td>
      <td>[0.02200852520763874, 0.027223799377679825, 0....</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

### Undeploy Pipelines

With the tutorial complete, the pipeline is undeployed to return the resources back to the cluster.

```python
pipeline.undeploy()
```

<table><tr><th>name</th> <td>keras-sequential-single-io</td></tr><tr><th>created</th> <td>2023-07-13 17:50:13.553180+00:00</td></tr><tr><th>last_updated</th> <td>2023-07-13 17:50:13.553180+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>e04712da-19d6-40d1-88d3-2dab8ab950e1</td></tr><tr><th>steps</th> <td>keras-sequential-single-io</td></tr></table>

