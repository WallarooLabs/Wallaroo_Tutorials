This tutorial can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/pipeline-architecture/wallaroo-arm-byop-vgg16).

## Arbitrary Python Tutorial Deploy Model in Wallaroo Upload and Deploy

This tutorial demonstrates how to use arbitrary python as a ML Model in Wallaroo.  Arbitrary Python allows organizations to use Python scripts that require specific libraries and artifacts as models in the Wallaroo engine.  This allows for highly flexible use of ML models with supporting scripts.

### Arbitrary Python Script Requirements

The entry point of the arbitrary python aka BYOP (Bring Your Own Predict) model is any python script that **must** include the following.

* `class ImageClustering(Inference)`:  The default inference class.  This is used to perform the actual inferences.  Wallaroo uses the `_predict` method to receive the inference data and call the appropriate functions for the inference.
  * `def __init__`:  Used to initialize this class and load in any other classes or other required settings.
  * `def expected_model_types`: Used by Wallaroo to anticipate what model types are used by the script.
  * `def model(self, model)`: Defines the model used for the inference.  Accepts the model instance used in the inference.
    * `self._raise_error_if_model_is_wrong_type(model)`: Returns the error if the wrong model type is used.  This verifies that only the anticipated model type is used for the inference.
    * `self._model = model`: Sets the submitted model as the model for this class, provided `_raise_error_if_model_is_wrong_type` is not raised.
  * `def _predict(self, input_data: InferenceData)`:  This is the entry point for Wallaroo to perform the inference.  This will receive the inference data, then perform whatever steps and return a dictionary of numpy arrays.
* `class ImageClusteringBuilder(InferenceBuilder)`: Loads the model and prepares it for inferencing.
  * `def inference(self) -> ImageClustering`: Sets the inference class being used for the inferences.
  * `def create(self, config: CustomInferenceConfig) -> ImageClustering`: Creates an inference subclass, assigning the model and any attributes required for it to function.

All other methods used for the functioning of these classes are optional, as long as they meet the requirements listed above.

### Tutorial Prerequisites

* A Wallaroo version 2023.2.1 or above instance.

### References

* [Wallaroo SDK Essentials Guide: Model Uploads and Registrations: Arbitrary Python](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/wallaroo-sdk-model-arbitrary-python/)

## Tutorial Steps

### Import Libraries

The first step is to import the libraries we'll be using.  These are included by default in the Wallaroo instance's JupyterHub service.

```python
import numpy as np
import pandas as pd
import json
import os
import pickle
import pyarrow as pa
import tensorflow as tf
import wallaroo
from wallaroo.object import EntityNotFoundError

from sklearn.cluster import KMeans
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten
from wallaroo.pipeline   import Pipeline
from wallaroo.deployment_config import DeploymentConfigBuilder
from wallaroo.framework import Framework
```

    2023-09-18 14:57:12.681704: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
    2023-09-18 14:57:12.681746: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.

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
import string
import random

# make a random 4 character suffix to prevent overwriting other user's workspaces
suffix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))

workspace_name = f'vgg16-clustering-workspace{suffix}'
pipeline_name = f'vgg16-clustering-pipeline'

model_name = 'arm-vgg16-clustering'
model_file_name = './models/model-auto-conversion-BYOP-vgg16-clustering_arm.zip'
```

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
```

### Create Workspace and Pipeline

We will now create the Wallaroo workspace to store our model and set it as the current workspace.  Future commands will default to this workspace for pipeline creation, model uploads, etc.  We'll create our Wallaroo pipeline that is used to deploy our arbitrary Python model.

```python
workspace = get_workspace(workspace_name)
wl.set_current_workspace(workspace)

pipeline = get_pipeline(pipeline_name)
```

### Upload Arbitrary Python Model

Arbitrary Python models are uploaded to Wallaroo through the Wallaroo Client `upload_model` method.

#### Upload Arbitrary Python Model Parameters

The following parameters are required for Arbitrary Python models.  Note that while some fields are considered as **optional** for the `upload_model` method, they are required for proper uploading of a Arbitrary Python model to Wallaroo.

| Parameter | Type | Description |
|---|---|---|
|`name` | `string` (*Required*) | The name of the model.  Model names are unique per workspace.  Models that are uploaded with the same name are assigned as a new **version** of the model. |
|`path` | `string` (*Required*) | The path to the model file being uploaded. |
|`framework` |`string` (*Upload Method Optional, Arbitrary Python model Required*) | Set as `Framework.CUSTOM`. |
|`input_schema` | `pyarrow.lib.Schema` (*Upload Method Optional, Arbitrary Python model Required*) | The input schema in Apache Arrow schema format. |
|`output_schema` | `pyarrow.lib.Schema` (*Upload Method Optional, Arbitrary Python model Required*) | The output schema in Apache Arrow schema format. |
| `convert_wait` | `bool` (*Upload Method Optional, Arbitrary Python model Optional*) (*Default: True*) | <ul><li>**True**: Waits in the script for the model conversion completion.</li><li>**False**:  Proceeds with the script without waiting for the model conversion process to display complete. |

Once the upload process starts, the model is containerized by the Wallaroo instance.  This process may take up to 10 minutes.

#### Upload Arbitrary Python Model Return

The following is returned with a successful model upload and conversion.

| Field | Type | Description |
|---|---|---|
| `name` | string | The name of the model. |
| `version` | string | The model version as a unique UUID. |
| `file_name` | string | The file name of the model as stored in Wallaroo. |
| `image_path` | string | The image used to deploy the model in the Wallaroo engine. |
| `last_update_time` | DateTime | When the model was last updated. |

For our example, we'll start with setting the `input_schema` and `output_schema` that is expected by our `ImageClustering._predict()` method.

```python
input_schema = pa.schema([
    pa.field('images', pa.list_(
        pa.list_(
            pa.list_(
                pa.int64(),
                list_size=3
            ),
            list_size=32
        ),
        list_size=32
    )),
])

output_schema = pa.schema([
    pa.field('predictions', pa.int64()),
])
```

### Upload Model

Now we'll upload our model.  The framework is `Framework.CUSTOM` for arbitrary Python models, and we'll specify the input and output schemas for the upload.

An important note for the ARM version of the VGG16 model is verifying that the Python modules used will operate in an ARM environment.  For example, the x86 model has the following as its `requirements.txt`:

```python
tensorflow==2.8.0
scikit-learn==1.2.2
```

The ARM version replaces the `tensorflow` module with the `tensorflow-aarch64` that runs in the ARM archtecture.  Otherwise, the model and Python script is the same.

```python
tensorflow-aarch64==2.8.4
scikit-learn==1.2.2
```

```python
from wallaroo.engine_config import Architecture

model = wl.upload_model(model_name, 
                        model_file_name, 
                        framework=Framework.CUSTOM, 
                        input_schema=input_schema, 
                        output_schema=output_schema,
                        arch=Architecture.ARM, 
                        convert_wait=True)
model
```

    Waiting for model loading - this will take up to 10.0min.
    Model is pending loading to a container runtime..
    Model is attempting loading to a container runtime.......................successful
    
    Ready

<table>
        <tr>
          <td>Name</td>
          <td>arm-vgg16-clustering</td>
        </tr>
        <tr>
          <td>Version</td>
          <td>14f9929d-cb37-4e44-aacd-2120b349ef31</td>
        </tr>
        <tr>
          <td>File Name</td>
          <td>model-auto-conversion-BYOP-vgg16-clustering_arm.zip</td>
        </tr>
        <tr>
          <td>SHA</td>
          <td>e53d1775766567eda5fd7ecb1618ea073fc18ffd6298c75da67be3b704029f15</td>
        </tr>
        <tr>
          <td>Status</td>
          <td>ready</td>
        </tr>
        <tr>
          <td>Image Path</td>
          <td>proxy.replicated.com/proxy/wallaroo/ghcr.io/wallaroolabs/mlflow-deploy:v2023.4.0-main-3827</td>
        </tr>
        <tr>
          <td>Updated At</td>
          <td>2023-18-Sep 14:59:39</td>
        </tr>
      </table>

```python
print(model)
```

    {'name': 'arm-vgg16-clustering', 'version': '14f9929d-cb37-4e44-aacd-2120b349ef31', 'file_name': 'model-auto-conversion-BYOP-vgg16-clustering_arm.zip', 'image_path': 'proxy.replicated.com/proxy/wallaroo/ghcr.io/wallaroolabs/mlflow-deploy:v2023.4.0-main-3827', 'last_update_time': datetime.datetime(2023, 9, 18, 14, 59, 39, 729445, tzinfo=tzutc())}

### Deploy Pipeline

The model is uploaded and ready for use.  We'll add it as a step in our pipeline, then deploy the pipeline.  For this example we're allocated 0.25 cpu and 4 Gi RAM to the pipeline through the pipeline's deployment configuration.

```python
pipeline.add_model_step(model)
```

<table><tr><th>name</th> <td>vgg16-clustering-pipeline</td></tr><tr><th>created</th> <td>2023-09-18 14:57:31.537378+00:00</td></tr><tr><th>last_updated</th> <td>2023-09-18 14:57:31.537378+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>d42d5272-86dd-414b-8928-6dea07c866c1</td></tr><tr><th>steps</th> <td></td></tr><tr><th>published</th> <td>False</td></tr></table>

```python
deployment_config = DeploymentConfigBuilder() \
    .cpus(1).memory('4Gi') \
    .build()

pipeline.deploy(deployment_config=deployment_config)
pipeline.status()
```

    Waiting for deployment - this will take up to 90s ........................ ok

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.2.13',
       'name': 'engine-d665b9c7d-jhr7s',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'vgg16-clustering-pipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'arm-vgg16-clustering',
          'version': '14f9929d-cb37-4e44-aacd-2120b349ef31',
          'sha': 'e53d1775766567eda5fd7ecb1618ea073fc18ffd6298c75da67be3b704029f15',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.3.39',
       'name': 'engine-lb-584f54c899-ksj6c',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': [{'ip': '10.244.0.40',
       'name': 'engine-sidekick-arm-vgg16-clustering-2-6f5975c9f6-6dgct',
       'status': 'Running',
       'reason': None,
       'details': [],
       'statuses': '\n'}]}

### Run inference

Everything is in place - we'll now run a sample inference with some toy data.  In this case we're randomly generating some values in the data shape the model expects, then submitting an inference request through our deployed pipeline.

```python
input_data = {
        "images": [np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)] * 2,
}
dataframe = pd.DataFrame(input_data)
dataframe
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>images</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[[[142, 30, 10], [182, 46, 219], [233, 142, 22...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[[[142, 30, 10], [182, 46, 219], [233, 142, 22...</td>
    </tr>
  </tbody>
</table>

```python
pipeline.infer(dataframe, timeout=10000)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.images</th>
      <th>out.predictions</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-09-18 15:00:07.019</td>
      <td>[142, 30, 10, 182, 46, 219, 233, 142, 223, 94,...</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-09-18 15:00:07.019</td>
      <td>[142, 30, 10, 182, 46, 219, 233, 142, 223, 94,...</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

### Undeploy Pipelines

The inference is successful, so we will undeploy the pipeline and return the resources back to the cluster.

```python
pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s .................................... ok

<table><tr><th>name</th> <td>vgg16-clustering-pipeline</td></tr><tr><th>created</th> <td>2023-09-18 14:57:31.537378+00:00</td></tr><tr><th>last_updated</th> <td>2023-09-18 14:59:40.816423+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>65299c7d-79dd-477e-b54e-adb0239f60eb, d42d5272-86dd-414b-8928-6dea07c866c1</td></tr><tr><th>steps</th> <td>arm-vgg16-clustering</td></tr><tr><th>published</th> <td>False</td></tr></table>

