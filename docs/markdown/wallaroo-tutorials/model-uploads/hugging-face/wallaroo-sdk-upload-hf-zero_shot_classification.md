The Wallaroo 101 tutorial can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/model_uploads/hugging-face-upload-tutorials).

## Wallaroo Model Upload via the Wallaroo SDK: Hugging Face Zero Shot Classification

The following tutorial demonstrates how to upload a Hugging Face Zero Shot model to a Wallaroo instance.

### Tutorial Goals

Demonstrate the following:

* Upload a Hugging Face Zero Shot Model to a Wallaroo instance.
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
workspace_name = f'hf-zero-shot-classification{suffix}'
pipeline_name = f'hf-zero-shot-classification'

model_name = 'hf-zero-shot-classification'
model_file_name = './models/model-auto-conversion_hugging-face_dummy-pipelines_zero-shot-classification-pipeline.zip'

```

### Create Workspace and Pipeline

We will now create the Wallaroo workspace to store our model and set it as the current workspace.  Future commands will default to this workspace for pipeline creation, model uploads, etc.  We'll create our Wallaroo pipeline to deploy our model.

```python
workspace = get_workspace(workspace_name)
wl.set_current_workspace(workspace)

pipeline = get_pipeline(pipeline_name)
```

### Configure Data Schemas

The following parameters are required for Hugging Face models.  Note that while some fields are considered as **optional** for the `upload_model` method, they are required for proper uploading of a Hugging Face model to Wallaroo.

| Parameter | Type | Description |
|---|---|---|
|`name` | `string` (*Required*) | The name of the model.  Model names are unique per workspace.  Models that are uploaded with the same name are assigned as a new **version** of the model. |
|`path` | `string` (*Required*) | The path to the model file being uploaded. 
|`framework` |`string` (*Upload Method Optional, Hugging Face model Required*) | Set as the framework. |
|`input_schema` | `pyarrow.lib.Schema` (*Upload Method Optional, Hugging Face model Required*) | The input schema in Apache Arrow schema format. |
|`output_schema` | `pyarrow.lib.Schema` (*Upload Method Optional, Hugging Face model Required*) | The output schema in Apache Arrow schema format. |
| `convert_wait` | `bool` (*Upload Method Optional, Hugging Face model Optional*) (*Default: True*) | <ul><li>**True**: Waits in the script for the model conversion completion.</li><li>**False**:  Proceeds with the script without waiting for the model conversion process to display complete. |

The input and output schemas will be configured for the data inputs and outputs.  More information on the available inputs under the [official 🤗 Hugging Face source code](https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/pipelines/zero_shot_classification.py#L172).

```python
input_schema = pa.schema([
    pa.field('inputs', pa.string()), # required
    pa.field('candidate_labels', pa.list_(pa.string(), list_size=2)), # required
    pa.field('hypothesis_template', pa.string()), # optional
    pa.field('multi_label', pa.bool_()), # optional
])

output_schema = pa.schema([
    pa.field('sequence', pa.string()),
    pa.field('scores', pa.list_(pa.float64(), list_size=2)), # same as number of candidate labels, list_size can be skipped by may result in slightly worse performance
    pa.field('labels', pa.list_(pa.string(), list_size=2)), # same as number of candidate labels, list_size can be skipped by may result in slightly worse performance
])
```

### Upload Model

The model will be uploaded with the framework set as `Framework.HUGGING_FACE_ZERO_SHOT_CLASSIFICATION`.

```python
framework=Framework.HUGGING_FACE_ZERO_SHOT_CLASSIFICATION

model = wl.upload_model(model_name,
                        model_file_name,
                        framework=framework,
                        input_schema=input_schema,
                        output_schema=output_schema,
                        convert_wait=True)
model
```

    Waiting for model conversion... It may take up to 10.0min.
    Model is Pending conversion..Converting.............Ready.

<table>
        <tr>
          <td>Name</td>
          <td>hf-zero-shot-classification</td>
        </tr>
        <tr>
          <td>Version</td>
          <td>56bd73be-74d1-46c9-81ad-f2aa7b0c3466</td>
        </tr>
        <tr>
          <td>File Name</td>
          <td>model-auto-conversion_hugging-face_dummy-pipelines_zero-shot-classification-pipeline.zip</td>
        </tr>
        <tr>
          <td>SHA</td>
          <td>3dcc14dd925489d4f0a3960e90a7ab5917ab685ce955beca8924aa7bb9a69398</td>
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
          <td>2023-13-Jul 16:24:31</td>
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
pipeline = get_pipeline(pipeline_name)
# clear the pipeline if used previously
pipeline.undeploy()
pipeline.clear()
pipeline.add_model_step(model)

pipeline.deploy(deployment_config=deployment_config)
pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.9.82',
       'name': 'engine-7c9767fbcc-t9f9d',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'hf-zero-shot-classification',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'hf-zero-shot-classification',
          'version': '56bd73be-74d1-46c9-81ad-f2aa7b0c3466',
          'sha': '3dcc14dd925489d4f0a3960e90a7ab5917ab685ce955beca8924aa7bb9a69398',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.9.81',
       'name': 'engine-lb-584f54c899-kz67n',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': [{'ip': '10.244.9.79',
       'name': 'engine-sidekick-hf-zero-shot-classification-251-5874655d474f5zt',
       'status': 'Running',
       'reason': None,
       'details': [],
       'statuses': '\n'}]}

### Run Inference

A sample inference will be run.  First the pandas DataFrame used for the inference is created, then the inference run through the pipeline's `infer` method.

```python
input_data = {
        "inputs": ["this is a test", "this is another test"], # required
        "candidate_labels": [["english", "german"], ["english", "german"]], # optional: using the defaults, similar to not passing this parameter
        "hypothesis_template": ["This example is {}.", "This example is {}."], # optional: using the defaults, similar to not passing this parameter
        "multi_label": [False, False], # optional: using the defaults, similar to not passing this parameter
}
dataframe = pd.DataFrame(input_data)
dataframe
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>inputs</th>
      <th>candidate_labels</th>
      <th>hypothesis_template</th>
      <th>multi_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>this is a test</td>
      <td>[english, german]</td>
      <td>This example is {}.</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>this is another test</td>
      <td>[english, german]</td>
      <td>This example is {}.</td>
      <td>False</td>
    </tr>
  </tbody>
</table>

```python
%time
pipeline.infer(dataframe)
```

    CPU times: user 3 µs, sys: 0 ns, total: 3 µs
    Wall time: 3.81 µs

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.candidate_labels</th>
      <th>in.hypothesis_template</th>
      <th>in.inputs</th>
      <th>in.multi_label</th>
      <th>out.labels</th>
      <th>out.scores</th>
      <th>out.sequence</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-07-13 16:25:46.100</td>
      <td>[english, german]</td>
      <td>This example is {}.</td>
      <td>this is a test</td>
      <td>False</td>
      <td>[english, german]</td>
      <td>[0.5040545463562012, 0.49594542384147644]</td>
      <td>this is a test</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-07-13 16:25:46.100</td>
      <td>[english, german]</td>
      <td>This example is {}.</td>
      <td>this is another test</td>
      <td>False</td>
      <td>[english, german]</td>
      <td>[0.5037839412689209, 0.4962160289287567]</td>
      <td>this is another test</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

### Undeploy Pipelines

With the tutorial complete, the pipeline is undeployed to return the resources back to the cluster.

```python
pipeline.undeploy()
```

<table><tr><th>name</th> <td>hf-zero-shot-classification</td></tr><tr><th>created</th> <td>2023-07-13 16:25:22.793477+00:00</td></tr><tr><th>last_updated</th> <td>2023-07-13 16:25:22.793477+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>dedeaa61-8c3f-4a7d-8ac4-642b52d7afde</td></tr><tr><th>steps</th> <td>hf-zero-shot-classification</td></tr></table>

