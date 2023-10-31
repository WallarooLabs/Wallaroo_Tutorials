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
import os
import wallaroo

```

### Connect to the Wallaroo Instance through the User Interface

The next step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
wl=wallaroo.Client()
```

### Connect to Model Registry

The Wallaroo Registry stores the URL and authentication token to the Model Registry service, with the assigned name.  Note that in this demonstration all URLs and token are examples.

```python
registry = wl.create_model_registry(name="JeffRegistry45", 
                                    token="dapi67c8c0b04606f730e78b7ae5e3221015-3", 
                                    url="https://sample.registry.service.azuredatabricks.net")
registry
```

<table>
          <tr>
            <th>Field</th>
            <th>Value</th>
          </tr>
          <tr>
            <td>Name</td><td>JeffRegistry45</td>
          </tr>
          <tr>
            <td>URL</td><td>https://sample.registry.service.azuredatabricks.net</td>
          </tr>
          <tr>
            <td>Workspaces</td><td>john.hummel@wallaroo.ai - Default Workspace</td>
          </tr>
          <tr>
              <td>Created At</td><td>2023-17-Jul 19:54:49</td>
          </tr>
          <tr>
              <td>Updated At</td><td>2023-17-Jul 19:54:49</td>
          </tr>
        </table>

### List Model Registries

Registries associated with a workspace are listed with the `Wallaroo.Client.list_model_registries()` method.

```python
# List all registries in this workspace
registries = wl.list_model_registries()
registries
```

<table><tr><th>name</th><th>registry url</th><th>created at</th><th>updated at</th></tr><tr><td>JeffRegistry45</td><td>https://sample.registry.service.azuredatabricks.net</td><td>2023-17-Jul 17:56:52</td><td>2023-17-Jul 17:56:52</td></tr><tr><td>JeffRegistry45</td><td>https://sample.registry.service.azuredatabricks.net</td><td>2023-17-Jul 19:54:49</td><td>2023-17-Jul 19:54:49</td></tr></table>

### Create Workspace

For this demonstration, we will create a random Wallaroo workspace, then attach our registry to the workspace so it is accessible by other workspace users.

### Add Registry to Workspace

Registries are assigned to a Wallaroo workspace with the `Wallaroo.registry.add_registry_to_workspace` method.  This allows members of the workspace to access the registry connection.  A registry can be associated with one or more workspaces.

#### Add Registry to Workspace Parameters

| Parameter | Type | Description |
|---|---|---|
| `name` | string (*Required*) | The numerical identifier of the workspace. |

```python
# Make a random new workspace
import math
import random
num = math.floor(random.random()* 1000)
workspace_id = wl.create_workspace(f"test{num}").id()

registry.add_registry_to_workspace(workspace_id=workspace_id)
```

<table>
          <tr>
            <th>Field</th>
            <th>Value</th>
          </tr>
          <tr>
            <td>Name</td><td>JeffRegistry45</td>
          </tr>
          <tr>
            <td>URL</td><td>https://sample.registry.service.azuredatabricks.net</td>
          </tr>
          <tr>
            <td>Workspaces</td><td>test68, john.hummel@wallaroo.ai - Default Workspace</td>
          </tr>
          <tr>
              <td>Created At</td><td>2023-17-Jul 19:54:49</td>
          </tr>
          <tr>
              <td>Updated At</td><td>2023-17-Jul 19:54:49</td>
          </tr>
        </table>

### Remove Registry from Workspace

Registries are removed from a Wallaroo workspace with the Registry `remove_registry_from_workspace` method.

#### Remove Registry from Workspace Parameters

| Parameter | Type | Description |
|---|---|---|
| `workspace_id` | Integer (*Required*) | The numerical identifier of the workspace. |

```python
registry.remove_registry_from_workspace(workspace_id=workspace_id)
```

<table>
          <tr>
            <th>Field</th>
            <th>Value</th>
          </tr>
          <tr>
            <td>Name</td><td>JeffRegistry45</td>
          </tr>
          <tr>
            <td>URL</td><td>https://sample.registry.service.azuredatabricks.net</td>
          </tr>
          <tr>
            <td>Workspaces</td><td>john.hummel@wallaroo.ai - Default Workspace</td>
          </tr>
          <tr>
              <td>Created At</td><td>2023-17-Jul 19:54:49</td>
          </tr>
          <tr>
              <td>Updated At</td><td>2023-17-Jul 19:54:49</td>
          </tr>
        </table>

### List Models in a Registry

A **List** of models available to the Wallaroo instance through the MLFlow Registry is performed with the `Wallaroo.Registry.list_models()` method.

```python
registry_models = registry.list_models()
registry_models
```

<table>
  <tr>
    <td>Name</td>
    <td>Registry User</td>
    <td>Versions</td>
    <td>Created At</td>
    <td>Updated At</td>
  </tr>

      <tr>
        <td>logreg1</td>
        <td>gib.bhojraj@wallaroo.ai</td>
        <td>1</td>
        <td>2023-06-Jul 14:36:54</td>
        <td>2023-06-Jul 14:36:56</td>
      </tr>

      <tr>
        <td>sidekick-test</td>
        <td>gib.bhojraj@wallaroo.ai</td>
        <td>1</td>
        <td>2023-11-Jul 14:42:14</td>
        <td>2023-11-Jul 14:42:14</td>
      </tr>

      <tr>
        <td>testmodel</td>
        <td>gib.bhojraj@wallaroo.ai</td>
        <td>1</td>
        <td>2023-16-Jun 12:38:42</td>
        <td>2023-06-Jul 15:03:41</td>
      </tr>

      <tr>
        <td>testmodel2</td>
        <td>gib.bhojraj@wallaroo.ai</td>
        <td>1</td>
        <td>2023-16-Jun 12:41:04</td>
        <td>2023-29-Jun 18:08:33</td>
      </tr>

      <tr>
        <td>verified-working</td>
        <td>gib.bhojraj@wallaroo.ai</td>
        <td>1</td>
        <td>2023-11-Jul 16:18:03</td>
        <td>2023-11-Jul 16:57:54</td>
      </tr>

      <tr>
        <td>wine_quality</td>
        <td>gib.bhojraj@wallaroo.ai</td>
        <td>2</td>
        <td>2023-16-Jun 13:05:53</td>
        <td>2023-16-Jun 13:09:57</td>
      </tr>

</table>

### Select Model from Registry

Registry models are selected from the `Wallaroo.Registry.list_models()` method, then specifying the model to use.

```python
single_registry_model = registry_models[4]
single_registry_model
```

<table>
  <tr>
    <td>Name</td>
    <td>verified-working</td>
  </tr>
  <tr>
    <td>Registry User</td>
    <td>gib.bhojraj@wallaroo.ai</td>
  </tr>
  <tr>
    <td>Versions</td>
    <td>1</td>
  </tr>
  <tr>
    <td>Created At</td>
    <td>2023-11-Jul 16:18:03</td>
  </tr>
  <tr>
    <td>Updated At</td>
    <td>2023-11-Jul 16:57:54</td>
  </tr>
</table>

### List Model Versions

The Registry Model attribute `versions` shows the complete list of versions for the particular model.

```python
single_registry_model.versions()
```

<table>
  <tr>
    <td>Name</td>
    <td>Version</td>
    <td>Description</td>
  </tr>

  <tr>
    <td>verified-working</td>
    <td>3</td>
    <td>None</td>
  </tr>

</table>

### List Model Version Artifacts

Artifacts belonging to a MLFlow registry model are listed with the Model Version `list_artifacts()` method.  This returns all artifacts for the model.

```python
single_registry_model.versions()[1].list_artifacts()
```

<table>
  <tr>
    <th>File Name</th>
    <th>File Size</th>
    <th>Full Path</th>
  </tr>

  <tr>
    <td>MLmodel</td>
    <td>559B</td>
    <td>https://sample.registry.service.azuredatabricks.net/api/2.0/dbfs/read?path=/databricks/mlflow-registry/9168792a16cb40a88de6959ef31e42a2/models/√erified-working/MLmodel</td>
  </tr>

  <tr>
    <td>conda.yaml</td>
    <td>182B</td>
    <td>https://sample.registry.service.azuredatabricks.net/api/2.0/dbfs/read?path=/databricks/mlflow-registry/9168792a16cb40a88de6959ef31e42a2/models/√erified-working/conda.yaml</td>
  </tr>

  <tr>
    <td>model.pkl</td>
    <td>829B</td>
    <td>https://sample.registry.service.azuredatabricks.net/api/2.0/dbfs/read?path=/databricks/mlflow-registry/9168792a16cb40a88de6959ef31e42a2/models/√erified-working/model.pkl</td>
  </tr>

  <tr>
    <td>python_env.yaml</td>
    <td>122B</td>
    <td>https://sample.registry.service.azuredatabricks.net/api/2.0/dbfs/read?path=/databricks/mlflow-registry/9168792a16cb40a88de6959ef31e42a2/models/√erified-working/python_env.yaml</td>
  </tr>

  <tr>
    <td>requirements.txt</td>
    <td>73B</td>
    <td>https://sample.registry.service.azuredatabricks.net/api/2.0/dbfs/read?path=/databricks/mlflow-registry/9168792a16cb40a88de6959ef31e42a2/models/√erified-working/requirements.txt</td>
  </tr>

</table>

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
| `framework` | string (*Required*) | The Wallaroo model `Framework`.  See [Model Uploads and Registrations Supported Frameworks](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/#list-wallaroo-frameworks) |
|`input_schema` | `pyarrow.lib.Schema` (*Required for non-native runtimes*) | The input schema in Apache Arrow schema format. |
|`output_schema` | `pyarrow.lib.Schema` (*Required for non-native runtimes*) | The output schema in Apache Arrow schema format. |

```python
model = registry.upload_model(
  name="verified-working", 
  path="https://sample.registry.service.azuredatabricks.net/api/2.0/dbfs/read?path=/databricks/mlflow-registry/9168792a16cb40a88de6959ef31e42a2/models/√erified-working/model.pkl", 
  framework=Framework.SKLEARN,
  input_schema=input_schema,
  output_schema=output_schema)
model
```

<table>
        <tr>
          <td>Name</td>
          <td>verified-working</td>
        </tr>
        <tr>
          <td>Version</td>
          <td>cf194b65-65b2-4d42-a4e2-6ca6fa5bfc42</td>
        </tr>
        <tr>
          <td>File Name</td>
          <td>model.pkl</td>
        </tr>
        <tr>
          <td>SHA</td>
          <td>5f4c25b0b564ab9fe0ea437424323501a460aa74463e81645a6419be67933ca4</td>
        </tr>
        <tr>
          <td>Status</td>
          <td>pending_conversion</td>
        </tr>
        <tr>
          <td>Image Path</td>
          <td>None</td>
        </tr>
        <tr>
          <td>Updated At</td>
          <td>2023-17-Jul 17:57:23</td>
        </tr>
      </table>

### Verify the Model Status

Once uploaded, the model will undergo conversion.  The following will loop through the model status until it is ready.  Once ready, it is available for deployment.

```python
import time
while model.status() != "ready" and model.status() != "error":
    print(model.status())
    time.sleep(3)
print(model.status())
```

    pending_conversion
    pending_conversion
    pending_conversion
    pending_conversion
    pending_conversion
    pending_conversion
    pending_conversion
    pending_conversion
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
    converting
    converting
    converting
    converting
    ready

### Model Runtime

Once uploaded and converted, the model runtime is derived.  This determines whether to allocate resources to pipeline's native runtime environment or containerized runtime environment.  For more details, see the [Wallaroo SDK Essentials Guide: Pipeline Deployment Configuration guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline-deployment-config/).

```python
model.config().runtime()
```

    'mlflow'

### Deploy Pipeline

The model is uploaded and ready for use.  We'll add it as a step in our pipeline, then deploy the pipeline.  For this example we're allocated 0.5 cpu to the runtime environment and 1 CPU to the containerized runtime environment.

```python
import os, json
from wallaroo.deployment_config import DeploymentConfigBuilder
deployment_config = DeploymentConfigBuilder().cpus(0.5).sidekick_cpus(model, 1).build()
pipeline = wl.build_pipeline("jefftest1")
pipeline = pipeline.add_model_step(model)
deployment = pipeline.deploy(deployment_config=deployment_config)
```

```python
pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.3.148',
       'name': 'engine-86c7fc5c95-8kwh5',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'jefftest1',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'verified-working',
          'version': 'cf194b65-65b2-4d42-a4e2-6ca6fa5bfc42',
          'sha': '5f4c25b0b564ab9fe0ea437424323501a460aa74463e81645a6419be67933ca4',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.4.203',
       'name': 'engine-lb-584f54c899-tpv5b',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': [{'ip': '10.244.0.225',
       'name': 'engine-sidekick-verified-working-43-74f957566d-9zdfh',
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
      <td>2023-07-17 17:59:18.840</td>
      <td>[5.1, 3.5, 1.4, 0.2]</td>
      <td>0</td>
      <td>[0.981814913291491, 0.018185072312411506, 1.43...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-07-17 17:59:18.840</td>
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

<table><tr><th>name</th> <td>jefftest1</td></tr><tr><th>created</th> <td>2023-07-17 17:59:05.922172+00:00</td></tr><tr><th>last_updated</th> <td>2023-07-17 17:59:06.684060+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>c2cca319-fcad-47b2-9de0-ad5b2852d1a2, f1e6d1b5-96ee-46a1-bfdf-174310ff4270</td></tr><tr><th>steps</th> <td>verified-working</td></tr></table>

