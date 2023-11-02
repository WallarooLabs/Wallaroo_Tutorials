This tutorial can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/model_uploads/sklearn-upload-tutorials).

## Wallaroo Model Upload via the Wallaroo SDK: Sklearn Clustering SVM

The following tutorial demonstrates how to upload a SKLearn Clustering Support Vector Machine(SVM) model to a Wallaroo instance.

### Tutorial Goals

Demonstrate the following:

* Upload a Sklearn Clustering SVM model to a Wallaroo instance.
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

workspace_name = f'sklearn-clustering-svm{suffix}'
pipeline_name = f'sklearn-clustering-svm'

model_name = 'sklearn-clustering-svm'
model_file_name = './models/model-auto-conversion_sklearn_svm_pipeline.pkl'
```

### Create Workspace and Pipeline

We will now create the Wallaroo workspace to store our model and set it as the current workspace.  Future commands will default to this workspace for pipeline creation, model uploads, etc.  We'll create our Wallaroo pipeline to deploy our model.

```python
workspace = get_workspace(workspace_name)
wl.set_current_workspace(workspace)

pipeline = get_pipeline(pipeline_name)
```

### Configure Data Schemas

SKLearn models are uploaded to Wallaroo through the Wallaroo Client [`upload_model`](/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/client/#Client.upload_model) method.

### Upload SKLearn Model Parameters

The following parameters are required for SKLearn models.  Note that while some fields are considered as **optional** for the `upload_model` method, they are required for proper uploading of a SKLearn model to Wallaroo.

| Parameter | Type | Description |
|---|---|---|
|`name` | `string` (*Required*) | The name of the model.  Model names are unique per workspace.  Models that are uploaded with the same name are assigned as a new **version** of the model. |
|`path` | `string` (*Required*) | The path to the model file being uploaded. 
|`framework` |`string` (*Upload Method Optional, SKLearn model Required*) | Set as the `Framework.SKLEARN`. |
|`input_schema` | `pyarrow.lib.Schema` (*Upload Method Optional, SKLearn model Required*) | The input schema in Apache Arrow schema format. |
|`output_schema` | `pyarrow.lib.Schema` (*Upload Method Optional, SKLearn model Required*) | The output schema in Apache Arrow schema format. |
| `convert_wait` | `bool` (*Upload Method Optional, SKLearn model Optional*) (*Default: True*) | <ul><li>**True**: Waits in the script for the model conversion completion.</li><li>**False**:  Proceeds with the script without waiting for the model conversion process to display complete. |

Once the upload process starts, the model is containerized by the Wallaroo instance.  This process may take up to 10 minutes.

```python
input_schema = pa.schema([
    pa.field('inputs', pa.list_(pa.float64(), list_size=4))
])

output_schema = pa.schema([
    pa.field('predictions', pa.int32())
])
```

### Upload Model

The model will be uploaded with the framework set as `Framework.SKLEARN`.

```python
model = wl.upload_model(model_name, 
                        model_file_name, 
                        framework=Framework.SKLEARN, 
                        input_schema=input_schema, 
                        output_schema=output_schema)
model
```

    Waiting for model loading - this will take up to 10.0min.
    Model is pending loading to a native runtime..
    Model is attempting loading to a native runtime..incompatible
    
    Model is pending loading to a container runtime.
    Model is attempting loading to a container runtime.........successful
    
    Ready

<table>
        <tr>
          <td>Name</td>
          <td>sklearn-clustering-svm</td>
        </tr>
        <tr>
          <td>Version</td>
          <td>9e9f6913-999d-4a19-894c-ecc372debcaf</td>
        </tr>
        <tr>
          <td>File Name</td>
          <td>model-auto-conversion_sklearn_svm_pipeline.pkl</td>
        </tr>
        <tr>
          <td>SHA</td>
          <td>c6eec69d96f7eeb3db034600dea6b12da1d2b832c39252ec4942d02f68f52f40</td>
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
          <td>2023-20-Oct 21:10:50</td>
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
# clear the pipeline if it was used before
pipeline.undeploy()
pipeline.clear()

pipeline.add_model_step(model)

pipeline.deploy(deployment_config=deployment_config)
```

     ok
    Waiting for deployment - this will take up to 45s ...........................................

<strong>*** An error occurred while deploying your pipeline.</strong>

    Deployment failed. See status for details.
    Status: {'status': 'Error', 'details': [], 'engines': [{'ip': '10.244.3.151', 'name': 'engine-8444db4dcb-md57f', 'status': 'Running', 'reason': None, 'details': [], 'pipeline_statuses': {'pipelines': [{'id': 'sklearn-clustering-svm', 'status': 'Running'}]}, 'model_statuses': {'models': [{'name': 'sklearn-clustering-svm', 'version': '9e9f6913-999d-4a19-894c-ecc372debcaf', 'sha': 'c6eec69d96f7eeb3db034600dea6b12da1d2b832c39252ec4942d02f68f52f40', 'status': 'Running'}]}}], 'engine_lbs': [{'ip': '10.244.2.207', 'name': 'engine-lb-584f54c899-nkj9h', 'status': 'Running', 'reason': None, 'details': []}], 'sidekicks': [{'ip': '10.244.3.150', 'name': 'engine-sidekick-sklearn-clustering-svm-71-59bc7cf755-vkq92', 'status': 'Running', 'reason': None, 'details': [], 'statuses': None}]}

## Inference

SKLearn models must have all of the data as one line to prevent columns from being read out of order when submitting in JSON.  The following will take in the data, convert the rows into a single `inputs` for the table, then perform the inference.  From the `output_schema` we have defined the output as `predictions` which will be displayed in our inference result output as `out.predictions`.

```python
data = pd.read_json('./data/test_cluster-svm.json')
display(data)

# move the column values to a single array input
dataframe = pd.DataFrame({"inputs": data[:2].values.tolist()})
display(dataframe)
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

```python
pipeline.infer(dataframe)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.inputs</th>
      <th>out.predictions</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-10-20 21:11:44.879</td>
      <td>[5.1, 3.5, 1.4, 0.2]</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-10-20 21:11:44.879</td>
      <td>[4.9, 3.0, 1.4, 0.2]</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

### Undeploy Pipelines

With the tutorial complete, the pipeline is undeployed to return the resources back to the cluster.

```python
pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s ..................................... ok

<table><tr><th>name</th> <td>sklearn-clustering-svm</td></tr><tr><th>created</th> <td>2023-10-20 21:00:38.923149+00:00</td></tr><tr><th>last_updated</th> <td>2023-10-20 21:10:52.412340+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>bf6b7315-b5c8-47fe-b088-ded36b18a3c6, 1758d8ac-6f86-4863-b194-1d7baf20fc1f, 8a5fbd0d-028a-478d-990d-4aa34a8e2b1b</td></tr><tr><th>steps</th> <td>sklearn-clustering-svm</td></tr><tr><th>published</th> <td>False</td></tr></table>

