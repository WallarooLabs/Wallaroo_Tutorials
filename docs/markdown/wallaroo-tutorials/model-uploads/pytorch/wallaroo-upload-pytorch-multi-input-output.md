This tutorial can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/model_uploads/pytorch-upload-tutorials).

## Wallaroo Model Upload via the Wallaroo SDK: Pytorch Multiple Input Output

The following tutorial demonstrates how to upload a Pytorch Multiple Input Output model to a Wallaroo instance.

### Tutorial Goals

Demonstrate the following:

* Upload a Pytorch Multiple Input Output to a Wallaroo instance.
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

suffix=''

workspace_name = f'pytorch-multi-io{suffix}'
pipeline_name = f'pytorch-multi-io'

model_name = 'pytorch-multi-io'
model_file_name = "./models/model-auto-conversion_pytorch_multi_io_model.pt"
```

### Create Workspace and Pipeline

We will now create the Wallaroo workspace to store our model and set it as the current workspace.  Future commands will default to this workspace for pipeline creation, model uploads, etc.  We'll create our Wallaroo pipeline to deploy our model.

```python
workspace = get_workspace(workspace_name)
wl.set_current_workspace(workspace)

pipeline = get_pipeline(pipeline_name)
```

### Configure Data Schemas

The following parameters are required for PyTorch models.  Note that while some fields are considered as **optional** for the `upload_model` method, they are required for proper uploading of a PyTorch model to Wallaroo.

| Parameter | Type | Description |
|---|---|---|
|`name` | `string` (*Required*) | The name of the model.  Model names are unique per workspace.  Models that are uploaded with the same name are assigned as a new **version** of the model. |
|`path` | `string` (*Required*) | The path to the model file being uploaded. 
|`framework` |`string` (*Upload Method Optional, PyTorch model Required*) | Set as the `Framework.PyTorch`. |
|`input_schema` | `pyarrow.lib.Schema` (*Upload Method Optional, PyTorch model Required*) | The input schema in Apache Arrow schema format. Note that float values **must** be `pyarrow.float32()`. |
|`output_schema` | `pyarrow.lib.Schema` (*Upload Method Optional, PyTorch model Required*) | The output schema in Apache Arrow schema format. Note that float values **must** be `pyarrow.float32()`. |
| `convert_wait` | `bool` (*Upload Method Optional, PyTorch model Optional*) (*Default: True*) | <ul><li>**True**: Waits in the script for the model conversion completion.</li><li>**False**:  Proceeds with the script without waiting for the model conversion process to display complete. |

Once the upload process starts, the model is containerized by the Wallaroo instance.  This process may take up to 10 minutes.

```python
input_schema = pa.schema([
    pa.field('input_1', pa.list_(pa.float32(), list_size=10)),
    pa.field('input_2', pa.list_(pa.float32(), list_size=5))
])
output_schema = pa.schema([
    pa.field('output_1', pa.list_(pa.float32(), list_size=3)),
    pa.field('output_2', pa.list_(pa.float32(), list_size=2))
])
```

### Upload Model

The model will be uploaded with the framework set as `Framework.PYTORCH`.

```python
model = wl.upload_model(model_name, 
                        model_file_name, 
                        framework=Framework.PYTORCH, 
                        input_schema=input_schema, 
                        output_schema=output_schema
                       )
model
```

    Waiting for model loading - this will take up to 10.0min.
    Model is pending loading to a native runtime..
    Ready

<table>
        <tr>
          <td>Name</td>
          <td>pytorch-multi-io</td>
        </tr>
        <tr>
          <td>Version</td>
          <td>f8df148e-a006-42c5-ac99-796f115897b8</td>
        </tr>
        <tr>
          <td>File Name</td>
          <td>model-auto-conversion_pytorch_multi_io_model.pt</td>
        </tr>
        <tr>
          <td>SHA</td>
          <td>792db9ee9f41aded3c1d4705f50ccdedd21cafb8b6232c03e4a849b6da1050a8</td>
        </tr>
        <tr>
          <td>Status</td>
          <td>ready</td>
        </tr>
        <tr>
          <td>Image Path</td>
          <td>None</td>
        </tr>
        <tr>
          <td>Architecture</td>
          <td>None</td>
        </tr>
        <tr>
          <td>Updated At</td>
          <td>2023-23-Oct 19:08:41</td>
        </tr>
      </table>

```python
model.config().runtime()
```

    'onnx'

### Deploy Pipeline

The model is uploaded and ready for use.  We'll add it as a step in our pipeline, then deploy the pipeline.  For this example we're allocated 0.25 cpu and 4 Gi RAM to the pipeline through the pipeline's deployment configuration.

```python
deployment_config = DeploymentConfigBuilder() \
    .cpus(0.25).memory('1Gi') \
    .build()
```

```python
# clear the pipeline if it was used before
pipeline.clear()

pipeline.add_model_step(model)

pipeline.deploy(deployment_config=deployment_config)
pipeline.status()
```

    Waiting for deployment - this will take up to 45s .................................. ok

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.3.164',
       'name': 'engine-8549d6985f-qfgsp',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'pytorch-multi-io',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'pytorch-multi-io',
          'version': 'f8df148e-a006-42c5-ac99-796f115897b8',
          'sha': '792db9ee9f41aded3c1d4705f50ccdedd21cafb8b6232c03e4a849b6da1050a8',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.2.215',
       'name': 'engine-lb-584f54c899-n8t26',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}

### Run Inference

A sample inference will be run.  First the pandas DataFrame used for the inference is created, then the inference run through the pipeline's `infer` method.

```python
mock_inference_data = [np.random.rand(10, 10), np.random.rand(10, 5)]
mock_dataframe = pd.DataFrame(
    {
        "input_1": mock_inference_data[0].tolist(),
        "input_2": mock_inference_data[1].tolist(),
    }
)
```

```python
pipeline.infer(mock_dataframe)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.input_1</th>
      <th>in.input_2</th>
      <th>out.output_1</th>
      <th>out.output_2</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-10-23 19:09:20.035</td>
      <td>[0.5520269716, 0.353031825, 0.5972010785, 0.77...</td>
      <td>[0.2850280321, 0.8368284642, 0.532692657, 0.53...</td>
      <td>[-0.12692596, -0.048615545, 0.16396174]</td>
      <td>[0.037088655, -0.07631089]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-10-23 19:09:20.035</td>
      <td>[0.665654034, 0.2721328048, 0.611313055, 0.742...</td>
      <td>[0.2229083231, 0.0462179945, 0.6249161412, 0.4...</td>
      <td>[0.05110594, -0.0646694, 0.26961502]</td>
      <td>[0.14888486, -0.011880934]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-10-23 19:09:20.035</td>
      <td>[0.7482700902, 0.867766789, 0.7562958282, 0.14...</td>
      <td>[0.5123290316, 0.619602395, 0.6079586226, 0.67...</td>
      <td>[0.025596283, -0.04604797, 0.33752537]</td>
      <td>[0.20393606, 0.020352483]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-10-23 19:09:20.035</td>
      <td>[0.7405163748, 0.8286719088, 0.165741416, 0.89...</td>
      <td>[0.3027072867, 0.3416387734, 0.2969483802, 0.1...</td>
      <td>[0.007636443, -0.09208804, 0.23095742]</td>
      <td>[0.09487003, 0.08022867]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-10-23 19:09:20.035</td>
      <td>[0.1383739731, 0.1535340384, 0.5779792315, 0.0...</td>
      <td>[0.3376164512, 0.969855208, 0.1542470748, 0.25...</td>
      <td>[-0.14555773, 0.1504708, 0.07537982]</td>
      <td>[0.049346283, -0.15848272]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2023-10-23 19:09:20.035</td>
      <td>[0.6635943228, 0.1642769142, 0.5543163791, 0.4...</td>
      <td>[0.822826152, 0.4580905361, 0.9199056247, 0.64...</td>
      <td>[-0.018602632, -0.057865076, 0.31042594]</td>
      <td>[-0.028693462, -0.06809359]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2023-10-23 19:09:20.035</td>
      <td>[0.4162411861, 0.0306481021, 0.9126412322, 0.8...</td>
      <td>[0.8228113533, 0.351936158, 0.5182544133, 0.74...</td>
      <td>[0.030514084, -0.063929394, 0.24837358]</td>
      <td>[-0.05689506, 0.047190517]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2023-10-23 19:09:20.035</td>
      <td>[0.5263343174, 0.3983799972, 0.6999939214, 0.6...</td>
      <td>[0.3230482528, 0.2033379246, 0.6991399275, 0.0...</td>
      <td>[-0.008261979, 0.06275801, 0.16552104]</td>
      <td>[0.15914191, 0.044295207]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2023-10-23 19:09:20.035</td>
      <td>[0.5231957471, 0.2019007135, 0.7899041536, 0.2...</td>
      <td>[0.633605919, 0.1939292389, 0.0518512061, 0.28...</td>
      <td>[0.0014905855, 0.072615415, 0.21212187]</td>
      <td>[0.16694427, -0.0021356493]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2023-10-23 19:09:20.035</td>
      <td>[0.0287641209, 0.9260014076, 0.540519311, 0.10...</td>
      <td>[0.9722058029, 0.8047130284, 0.671538585, 0.61...</td>
      <td>[-0.07756777, -0.11604313, 0.25187707]</td>
      <td>[-0.014747229, 0.03463669]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

### Undeploy Pipelines

With the tutorial complete, the pipeline is undeployed to return the resources back to the cluster.

```python
pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s .................................... ok

<table><tr><th>name</th> <td>pytorch-multi-io</td></tr><tr><th>created</th> <td>2023-10-19 21:47:11.881873+00:00</td></tr><tr><th>last_updated</th> <td>2023-10-23 19:08:43.867731+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>75b823f2-7f60-4423-ae54-3a52c0de67c4, 39168af2-bd41-49f8-8f5e-b1aba608ee68, d543d8ac-ba93-4dd5-a8d4-8bd2b405eb18, c55a8525-f108-404a-b0da-2f78f2ea2e34, c0358f0b-22b1-431f-8eee-73eef2ce8bbc</td></tr><tr><th>steps</th> <td>pytorch-multi-io</td></tr><tr><th>published</th> <td>False</td></tr></table>

```python

```
