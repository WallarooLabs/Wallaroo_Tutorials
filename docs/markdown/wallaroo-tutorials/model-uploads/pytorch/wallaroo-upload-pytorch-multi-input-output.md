This tutorial can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/blob/2023.2.1_prerelease/model_uploads/pytorch-upload-tutorials).

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
|`input_schema` | `pyarrow.lib.Schema` (*Upload Method Optional, PyTorch model Required*) | The input schema in Apache Arrow schema format. |
|`output_schema` | `pyarrow.lib.Schema` (*Upload Method Optional, PyTorch model Required*) | The output schema in Apache Arrow schema format. |
| `convert_wait` | `bool` (*Upload Method Optional, PyTorch model Optional*) (*Default: True*) | <ul><li>**True**: Waits in the script for the model conversion completion.</li><li>**False**:  Proceeds with the script without waiting for the model conversion process to display complete. |

Once the upload process starts, the model is containerized by the Wallaroo instance.  This process may take up to 10 minutes.

```python
input_schema = pa.schema([
    pa.field('input_1', pa.list_(pa.float64(), list_size=10)),
    pa.field('input_2', pa.list_(pa.float64(), list_size=5))
])
output_schema = pa.schema([
    pa.field('output_1', pa.list_(pa.float64(), list_size=3)),
    pa.field('output_2', pa.list_(pa.float64(), list_size=2))
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

    Waiting for model conversion... It may take up to 10.0min.
    Model is Pending conversion.Converting..........Ready.

<table>
        <tr>
          <td>Name</td>
          <td>pytorch-multi-io</td>
        </tr>
        <tr>
          <td>Version</td>
          <td>d503b511-7a0c-4c90-9cbc-022467886dcd</td>
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
          <td>proxy.replicated.com/proxy/wallaroo/ghcr.io/wallaroolabs/mlflow-deploy:v2023.3.0-main-3509</td>
        </tr>
        <tr>
          <td>Updated At</td>
          <td>2023-13-Jul 17:40:39</td>
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
     'engines': [{'ip': '10.244.9.169',
       'name': 'engine-7d8fbd4b74-kc5l5',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'pytorch-multi-io',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'pytorch-multi-io',
          'version': 'd503b511-7a0c-4c90-9cbc-022467886dcd',
          'sha': '792db9ee9f41aded3c1d4705f50ccdedd21cafb8b6232c03e4a849b6da1050a8',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.9.167',
       'name': 'engine-lb-584f54c899-xb4wd',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': [{'ip': '10.244.9.168',
       'name': 'engine-sidekick-pytorch-multi-io-266-c9fdfd57c-r26jj',
       'status': 'Running',
       'reason': None,
       'details': [],
       'statuses': '\n'}]}

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
      <td>2023-07-13 17:41:15.828</td>
      <td>[0.7193870373, 0.0442841786, 0.0050186971, 0.9...</td>
      <td>[0.5665563078, 0.2719984279, 0.8020988158, 0.9...</td>
      <td>[-0.054150406271219254, -0.08067788183689117, ...</td>
      <td>[-0.0867157131433487, 0.03102545440196991]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-07-13 17:41:15.828</td>
      <td>[0.3303183352, 0.7338588864, 0.4901244227, 0.4...</td>
      <td>[0.0020920459, 0.7473108704, 0.4701280309, 0.8...</td>
      <td>[-0.0814041793346405, -0.008595120161771774, 0...</td>
      <td>[0.22059735655784607, -0.0954931378364563]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-07-13 17:41:15.828</td>
      <td>[0.8350163099, 0.9301095252, 0.4634737661, 0.0...</td>
      <td>[0.2042988348, 0.3131013315, 0.2396516618, 0.8...</td>
      <td>[-0.0936204195022583, 0.07057543098926544, 0.2...</td>
      <td>[0.07758928835391998, 0.02205061912536621]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-07-13 17:41:15.828</td>
      <td>[0.9560662259, 0.7334543871, 0.8347215148, 0.5...</td>
      <td>[0.0846331092, 0.4104567348, 0.9964352268, 0.7...</td>
      <td>[-0.07443580776453018, 0.00262654572725296, 0....</td>
      <td>[0.08881741762161255, 0.184173583984375]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-07-13 17:41:15.828</td>
      <td>[0.8893996022, 0.0422634898, 0.094115839, 0.30...</td>
      <td>[0.7091750984, 0.2677670739, 0.797334875, 0.60...</td>
      <td>[0.011080481112003326, -0.03954530879855156, 0...</td>
      <td>[0.02079789713025093, -0.023370355367660522]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2023-07-13 17:41:15.828</td>
      <td>[0.360097173, 0.5326510416, 0.7427586985, 0.73...</td>
      <td>[0.4059651678, 0.8209608747, 0.6650853071, 0.5...</td>
      <td>[-0.041532136499881744, -0.02947094291448593, ...</td>
      <td>[0.19275487959384918, -0.09685075283050537]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2023-07-13 17:41:15.828</td>
      <td>[0.8493053911, 0.9772701228, 0.1395685377, 0.6...</td>
      <td>[0.5020254829, 0.664290124, 0.0900637878, 0.27...</td>
      <td>[0.023206554353237152, -0.05458785593509674, 0...</td>
      <td>[0.252506822347641, -0.056555017828941345]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2023-07-13 17:41:15.828</td>
      <td>[0.1783516801, 0.7383490632, 0.8826853, 0.8707...</td>
      <td>[0.197614553, 0.7345261372, 0.3909055798, 0.12...</td>
      <td>[-0.007199503481388092, -0.008408397436141968,...</td>
      <td>[0.15768739581108093, 0.10399264097213745]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2023-07-13 17:41:15.828</td>
      <td>[0.7671391397, 0.4079364465, 0.763576349, 0.26...</td>
      <td>[0.3169436757, 0.3800284784, 0.1143413322, 0.2...</td>
      <td>[-0.027935631573200226, 0.08666972815990448, 0...</td>
      <td>[0.25775399804115295, -0.042692944407463074]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2023-07-13 17:41:15.828</td>
      <td>[0.8885134534, 0.2440000822, 0.56551096, 0.780...</td>
      <td>[0.1742113245, 0.624024604, 0.267043414, 0.153...</td>
      <td>[-0.004899069666862488, 0.04411523416638374, 0...</td>
      <td>[0.22322586178779602, -0.12406840920448303]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

### Undeploy Pipelines

With the tutorial complete, the pipeline is undeployed to return the resources back to the cluster.

```python
pipeline.undeploy()
```

<table><tr><th>name</th> <td>pytorch-multi-io</td></tr><tr><th>created</th> <td>2023-07-13 17:38:02.341959+00:00</td></tr><tr><th>last_updated</th> <td>2023-07-13 17:40:44.329100+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>6867e6c5-7193-44d8-9756-4dfbc8b7db5c, 7b21d715-e95e-4557-abd1-a856eaea6e42</td></tr><tr><th>steps</th> <td>pytorch-multi-io</td></tr></table>

