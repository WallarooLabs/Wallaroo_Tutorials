This tutorial can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/model_uploads/mlflow-registries-upload-tutorials).

## MLFLow Registry Model Upload Demonstration

Wallaroo users can register their trained machine learning models from a model registry into their Wallaroo instance and perform inferences with it through a Wallaroo pipeline.

This guide details how to add ML Models from a model registry service into a Wallaroo instance.

## Artifact Requirements

Models are uploaded to the Wallaroo instance as the specific **artifact** - the "file" or other data that represents the file itself.  This **must** comply with the [Wallaroo model requirements framework and version](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/) or it will not be deployed.

This tutorial will:

* Create a Wallaroo workspace and pipeline.
* Show how to connect a Wallaroo Registry that connects to a Model Registry Service.
* Use the registry connection details to upload a sample model to Wallaroo.
* Perform a sample inference.

### Prerequisites

* A Wallaroo version 2023.2.1 or above instance.
* A Model (aka Artifact) Registry Service

### References

* [Wallaroo SDK Essentials Guide: Pipeline Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline/)
* [Python Models](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/wallaroo-sdk-model-upload-python/)

## Tutorial Steps

### Import Libraries

We'll start with importing the libraries we need for the tutorial.

```python
import wallaroo
from wallaroo.object import EntityNotFoundError
from wallaroo.framework import Framework
from wallaroo.deployment_config import DeploymentConfigBuilder

import pandas as pd

#import os
# import json
import pyarrow as pa
import os
```

    '2023.2.1+903061ec8'

### Connect to the Wallaroo Instance through the User Interface

The next step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
import os
```

```python
# Login through local Wallaroo instance

wl = wallaroo.Client()
```

### Create the Wallaroo Registry

The Wallaroo Registry stores the URL and authentication token to the Model Registry service, with the assigned name.  Note that in this demonstration all URLs and token are examples.

```python
registry = wl.create_model_registry(name="JamieRegistry", 
                                    token="abcdefg", 
                                    url="https://model.registry.wallaroo.ai"
                                    )
registry
```

### Configure Data Schemas

To upload a ML Model to Wallaroo, the input and output schemas must be defined in `pyarrow.lib.Schema` format.

```python
from wallaroo.framework import Framework
import pyarrow as pa

input_schema = pa.schema([
    pa.field('inputs', pa.list_(pa.float64(), list_size=4))
])

output_schema = pa.schema([
    pa.field('predictions', pa.int32()),
    pa.field('probabilities', pa.list_(pa.float64(), list_size=3))
])
```

### Upload a Model from a Registry

Models uploaded to the Wallaroo workspace are uploaded from a MLFlow Registry with the `Wallaroo.Registry.upload` method.

#### Upload a Model from a Registry Parameters

| Parameter | Type | Description |
|---|---|---|
| `name` | string (*Required*) | The name to assign the model once uploaded.  Model names are unique within a workspace.  Models assigned the same name as an existing model will be uploaded as a new model version.|
| `path` | string (*Required*) | The full path to the model artifact in the registry. |
| `framework` | string (*Required*) | The Wallaroo model `Framework`.  See [Model Uploads and Registrations Supported Frameworks]({{<ref "wallaroo-sdk-model-uploads#supported-frameworks">}}) |
|`input_schema` | `pyarrow.lib.Schema` (*Required for non-native runtimes*) | The input schema in Apache Arrow schema format. |
|`output_schema` | `pyarrow.lib.Schema` (*Required for non-native runtimes*) | The output schema in Apache Arrow schema format. |

```python
model = registry.upload_model(
  name="sample-sklearn-model", 
  path="https://model.registry.wallaroo.ai/api/2.0/dbfs/read?path=/databricks/mlflow-registry/9168792a16cb40a88de6959ef31e42a2/models/√erified-working/model.pkl", 
    framework=Framework.SKLEARN,
  input_schema=input_schema,
  output_schema=output_schema)
```

### Model Runtime

Once uploaded and converted, the model runtime is derived.  This determines whether to allocate resources to pipeline's native runtime environment or containerized runtime environment.  For more details, see the [Wallaroo SDK Essentials Guide: Pipeline Deployment Configuration guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline-deployment-config/).

```python
model.config().runtime()
```

    'mlflow'

### Verify the Model Status

Once uploaded, the model will undergo conversion.  The following will loop through the model status until it is ready.  Once ready, it is available for deployment.

```python
import time
while model.status() != "ready" and model.status() != "error":
    print(model.status())
    time.sleep(3)
print(model.status())
```

    converting
    converting
    converting
    pending_conversion
    pending_conversion
    converting
    converting
    converting
    converting
    converting
    converting
    converting
    converting
    converting
    converting
    converting
    ready

### Deploy Pipeline

The model is uploaded and ready for use.  We'll add it as a step in our pipeline, then deploy the pipeline.  For this example we're allocated 0.5 cpu to the runtime environment and 1 CPU to the containerized runtime environment.

```python
import os, json
from wallaroo.deployment_config import DeploymentConfigBuilder
deployment_config = (DeploymentConfigBuilder().cpus(0.5).sidekick_cpus(model, 1).build())

pipeline = wl.build_pipeline("sklearn-jamie-test-1")

pipeline = pipeline.add_model_step(model)

deployment = pipeline.deploy(deployment_config=deployment_config)
```

    Waiting for deployment - this will take up to 90s ........ ok

```python
pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.2.230',
       'name': 'engine-76b8d9d6f9-7c627',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'sklearn-jamie-test-1',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'jamie-reg-12',
          'version': 'd0784ad9-61d0-403c-9337-79da6d049255',
          'sha': '5f4c25b0b564ab9fe0ea437424323501a460aa74463e81645a6419be67933ca4',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.0.103',
       'name': 'engine-lb-584f54c899-ppkms',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': [{'ip': '10.244.2.229',
       'name': 'engine-sidekick-jamie-reg-12-14-67cb7f8b86-sv5dd',
       'status': 'Running',
       'reason': None,
       'details': [],
       'statuses': '\n'}]}

### Run Inference

A sample inference will be run.  First the pandas DataFrame used for the inference is created, then the inference run through the pipeline's `infer` method.

```python
import pandas as pd
from sklearn.datasets import load_iris

data = load_iris(as_frame=True)

X = data['data'].values
dataframe = pd.DataFrame({"inputs": data['data'][:2].values.tolist()})
dataframe
```

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
deployment.infer(dataframe)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.inputs</th>
      <th>out.predictions</th>
      <th>out.probabilities</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-07-14 21:55:22.827</td>
      <td>[5.1, 3.5, 1.4, 0.2]</td>
      <td>0</td>
      <td>[0.981814913291491, 0.018185072312411506, 1.43...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-07-14 21:55:22.827</td>
      <td>[4.9, 3.0, 1.4, 0.2]</td>
      <td>0</td>
      <td>[0.9717552971628304, 0.02824467272952288, 3.01...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

### Undeploy Pipelines

With the tutorial complete, the pipeline is undeployed to return the resources back to the cluster.

```python
pipeline.undeploy()
```
