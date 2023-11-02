This tutorial can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/model_uploads/pytorch-upload-tutorials).

## Wallaroo Model Upload via the Wallaroo SDK: Pytorch Single Input Output

The following tutorial demonstrates how to upload a Pytorch Single Input Output model to a Wallaroo instance.

### Tutorial Goals

Demonstrate the following:

* Upload a Pytorch Single Input Output to a Wallaroo instance.
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
workspace_name = f'pytorch-single-io{suffix}'
pipeline_name = 'pytorch-single-io'

model_name = 'pytorch-single-io'
model_file_name = "./models/model-auto-conversion_pytorch_single_io_model.pt"
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
    pa.field('input', pa.list_(pa.float32(), list_size=10))
])
output_schema = pa.schema([
    pa.field('output', pa.list_(pa.float32(), list_size=1))
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
          <td>pytorch-single-io</td>
        </tr>
        <tr>
          <td>Version</td>
          <td>7e8be792-3b25-4be0-b0b8-e687e501f8df</td>
        </tr>
        <tr>
          <td>File Name</td>
          <td>model-auto-conversion_pytorch_single_io_model.pt</td>
        </tr>
        <tr>
          <td>SHA</td>
          <td>23bdbafc51c3df7ac84e5f8b2833c592d7da2b27715a7da3e45bf732ea85b8bb</td>
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
          <td>2023-23-Oct 19:14:00</td>
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
pipeline.undeploy()
pipeline.clear()

pipeline.add_model_step(model)

pipeline.deploy(deployment_config=deployment_config)
```

     ok
    Waiting for deployment - this will take up to 45s ................................... ok

<table><tr><th>name</th> <td>pytorch-single-io</td></tr><tr><th>created</th> <td>2023-10-19 21:44:22.896716+00:00</td></tr><tr><th>last_updated</th> <td>2023-10-23 19:14:02.453910+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>8221fbd5-2dc6-4761-82a7-1744aa05d81a, 0f430725-0252-4585-80ea-ec0e2028316d, 64edbfa5-d1bb-44e2-beb2-c6ee87d1a4d9</td></tr><tr><th>steps</th> <td>pytorch-single-io</td></tr><tr><th>published</th> <td>False</td></tr></table>

### Run Inference

A sample inference will be run.  First the pandas DataFrame used for the inference is created, then the inference run through the pipeline's `infer` method.

```python
mock_inference_data = np.random.rand(10, 10)
mock_dataframe = pd.DataFrame({"input": mock_inference_data.tolist()})
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
      <td>2023-10-23 19:14:39.424</td>
      <td>[0.4474957532, 0.1724440529, 0.9187094437, 0.7...</td>
      <td>[-0.07964326]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-10-23 19:14:39.424</td>
      <td>[0.4042578849, 0.0492679987, 0.8344617226, 0.0...</td>
      <td>[0.04861839]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-10-23 19:14:39.424</td>
      <td>[0.5826077484, 0.1242290539, 0.8257548413, 0.2...</td>
      <td>[-0.040291503]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-10-23 19:14:39.424</td>
      <td>[0.5482710709, 0.8289840296, 0.4642728375, 0.8...</td>
      <td>[-0.1577661]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-10-23 19:14:39.424</td>
      <td>[0.7409150425, 0.093049813, 0.7461292107, 0.70...</td>
      <td>[-0.07514663]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2023-10-23 19:14:39.424</td>
      <td>[0.1703855753, 0.7154656468, 0.7717122989, 0.5...</td>
      <td>[-0.027950048]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2023-10-23 19:14:39.424</td>
      <td>[0.8858337747, 0.3015204777, 0.5810049274, 0.9...</td>
      <td>[-0.20391503]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2023-10-23 19:14:39.424</td>
      <td>[0.0857726256, 0.17188405, 0.1614009234, 0.095...</td>
      <td>[-0.058097966]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2023-10-23 19:14:39.424</td>
      <td>[0.0874623336, 0.1636057129, 0.5464519563, 0.0...</td>
      <td>[-0.027925014]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2023-10-23 19:14:39.424</td>
      <td>[0.4850890686, 0.7002499257, 0.6851349698, 0.0...</td>
      <td>[0.088012084]</td>
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

<table><tr><th>name</th> <td>pytorch-single-io</td></tr><tr><th>created</th> <td>2023-10-19 21:44:22.896716+00:00</td></tr><tr><th>last_updated</th> <td>2023-10-23 19:14:02.453910+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>8221fbd5-2dc6-4761-82a7-1744aa05d81a, 0f430725-0252-4585-80ea-ec0e2028316d, 64edbfa5-d1bb-44e2-beb2-c6ee87d1a4d9</td></tr><tr><th>steps</th> <td>pytorch-single-io</td></tr><tr><th>published</th> <td>False</td></tr></table>

