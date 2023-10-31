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

suffix=''

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

    Waiting for model loading - this will take up to 10.0min.
    Model is pending loading to a native runtime.
    Model is pending loading to a container runtime..
    Model is attempting loading to a container runtime.....................successful
    
    Ready

<table>
        <tr>
          <td>Name</td>
          <td>keras-sequential-single-io</td>
        </tr>
        <tr>
          <td>Version</td>
          <td>3aa90e63-f61b-4418-a70a-1631edfcb7e6</td>
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
          <td>proxy.replicated.com/proxy/wallaroo/ghcr.io/wallaroolabs/mlflow-deploy:v2023.4.0-main-4005</td>
        </tr>
        <tr>
          <td>Architecture</td>
          <td>None</td>
        </tr>
        <tr>
          <td>Updated At</td>
          <td>2023-20-Oct 17:52:05</td>
        </tr>
      </table>

```python
model.config().runtime()
```

    'flight'

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
      <td>[0.1991693928323468, 0.7669068394222905, 0.310...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[0.5845813884987883, 0.9974851461171536, 0.641...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[0.932340919861926, 0.5378812209722794, 0.1672...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[0.40557019984163867, 0.49709830678629907, 0.6...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[0.044116334060004925, 0.8634686667900255, 0.2...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>[0.9516672781081118, 0.5804880585685775, 0.858...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>[0.6956141885467453, 0.7382529966340766, 0.392...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>[0.21926011010552227, 0.6843926552276196, 0.78...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>[0.9941171972816318, 0.45451319048527616, 0.95...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>[0.3712231387032263, 0.08633906733980612, 0.87...</td>
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
      <td>2023-10-20 17:58:27.143</td>
      <td>[0.1991693928, 0.7669068394, 0.3105885385, 0.2...</td>
      <td>[0.028634641, 0.023895813, 0.040356454, 0.0244...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-10-20 17:58:27.143</td>
      <td>[0.5845813885, 0.9974851461, 0.6415301842, 0.8...</td>
      <td>[0.039276116, 0.01871082, 0.047833905, 0.01654...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-10-20 17:58:27.143</td>
      <td>[0.9323409199, 0.537881221, 0.1672841103, 0.79...</td>
      <td>[0.016344877, 0.039013684, 0.03176823, 0.03096...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-10-20 17:58:27.143</td>
      <td>[0.4055701998, 0.4970983068, 0.6517088712, 0.2...</td>
      <td>[0.03115139, 0.019364284, 0.03944681, 0.024314...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-10-20 17:58:27.143</td>
      <td>[0.0441163341, 0.8634686668, 0.212879749, 0.62...</td>
      <td>[0.024984468, 0.031172493, 0.06482109, 0.02364...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2023-10-20 17:58:27.143</td>
      <td>[0.9516672781, 0.5804880586, 0.8585948355, 0.8...</td>
      <td>[0.03174607, 0.024760708, 0.051566537, 0.02118...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2023-10-20 17:58:27.143</td>
      <td>[0.6956141885, 0.7382529966, 0.3924137787, 0.9...</td>
      <td>[0.031668007, 0.025940748, 0.06299812, 0.02237...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2023-10-20 17:58:27.143</td>
      <td>[0.2192601101, 0.6843926552, 0.78983548, 0.939...</td>
      <td>[0.03280156, 0.025590684, 0.0276087, 0.0241528...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2023-10-20 17:58:27.143</td>
      <td>[0.9941171973, 0.4545131905, 0.95580094, 0.179...</td>
      <td>[0.024180355, 0.02735067, 0.051267277, 0.03041...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2023-10-20 17:58:27.143</td>
      <td>[0.3712231387, 0.0863390673, 0.8778721234, 0.0...</td>
      <td>[0.02604158, 0.02733598, 0.04093318, 0.0291493...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

### Undeploy Pipelines

With the tutorial complete, the pipeline is undeployed to return the resources back to the cluster.

```python
pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s .............
