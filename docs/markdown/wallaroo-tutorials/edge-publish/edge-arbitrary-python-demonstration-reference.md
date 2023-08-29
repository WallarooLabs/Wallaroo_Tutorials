This tutorial can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/pipeline-edge-publish/edge-arbitrary-python).

## Arbitrary Python Tutorial Deploy Model in Wallaroo Upload and Deploy

This tutorial demonstrates how to use arbitrary python as a ML Model in Wallaroo and deploy the pipeline, then deploy that same pipeline to edge devices through an Open Container Initiative Registry, registered in the Wallaroo instance as the Edge Deployment Registry.

### Tutorial Goals

* **Arbitrary Python Tutorial Deploy Model in Wallaroo Upload and Deploy**: Deploys the `KMeans` model in an arbitrary Python package in Wallaroo, and perform sample inferences.  The file `models/model-auto-conversion-BYOP-vgg16-clustering.zip` is provided so users can go right to testing deployment.
* **Publish pipeline to Open Container Initiative Registry, registered in the Wallaroo instance as the Edge Deployment Registry.**:  This will containerize the pipeline and copy the engine, python steps, model, and deployment configuration to the registry service.
* **Deploy the Pipeline to an Edge Device**:  The pipeline will be deployed to an Edge device through Docker.

### Arbitrary Python Script Requirements

The entry point of the arbitrary python model is any python script that **must** include the following.

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

from sklearn.cluster import KMeans
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten
from wallaroo.pipeline   import Pipeline
from wallaroo.deployment_config import DeploymentConfigBuilder
from wallaroo.framework import Framework
from wallaroo.object import EntityNotFoundError

import requests
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
import string
import random

# make a random 4 character suffix to prevent overwriting other user's workspaces
suffix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))
workspace_name = f'vgg16-clustering-workspace{suffix}'
pipeline_name = f'vgg16-clustering-pipeline'

model_name = 'vgg16-clustering'
model_file_name = './models/model-auto-conversion-BYOP-vgg16-clustering.zip'
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

```python
model = wl.upload_model(model_name, 
                        model_file_name, 
                        framework=Framework.CUSTOM, 
                        input_schema=input_schema, 
                        output_schema=output_schema, 
                        convert_wait=True)
model
```

    Waiting for model loading - this will take up to 10.0min.
    Model is pending loading to a container runtime.
    Model is attempting loading to a container runtime....................successful
    
    Ready

<table>
        <tr>
          <td>Name</td>
          <td>vgg16-clustering</td>
        </tr>
        <tr>
          <td>Version</td>
          <td>90c3ea47-0e4f-4de6-9bea-af47e3c9da6f</td>
        </tr>
        <tr>
          <td>File Name</td>
          <td>model-auto-conversion-BYOP-vgg16-clustering.zip</td>
        </tr>
        <tr>
          <td>SHA</td>
          <td>7bb3362b1768c92ea7e593451b2b8913d3b7616c19fd8d25b73fb6990f9283e0</td>
        </tr>
        <tr>
          <td>Status</td>
          <td>ready</td>
        </tr>
        <tr>
          <td>Image Path</td>
          <td>proxy.replicated.com/proxy/wallaroo/ghcr.io/wallaroolabs/mlflow-deploy:v2023.3.0-main-3731</td>
        </tr>
        <tr>
          <td>Updated At</td>
          <td>2023-28-Aug 20:29:44</td>
        </tr>
      </table>

### Deploy Pipeline

The model is uploaded and ready for use.  We'll add it as a step in our pipeline, then deploy the pipeline.  For this example we're allocated 0.25 cpu and 4 Gi RAM to the pipeline through the pipeline's deployment configuration.

```python
pipeline.add_model_step(model)
```

<table><tr><th>name</th> <td>vgg16-clustering-pipeline</td></tr><tr><th>created</th> <td>2023-08-28 20:26:59.888454+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-28 20:26:59.888454+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>af9b1b8e-578c-4d72-8bf7-97724301fb45</td></tr><tr><th>steps</th> <td></td></tr><tr><th>published</th> <td>False</td></tr></table>

```python
deployment_config = DeploymentConfigBuilder() \
    .cpus(0.25).memory('4Gi') \
    .build()

pipeline.deploy(deployment_config=deployment_config)
pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.3.100',
       'name': 'engine-76bbb56bcc-rnnhl',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'vgg16-clustering-pipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'vgg16-clustering',
          'version': '90c3ea47-0e4f-4de6-9bea-af47e3c9da6f',
          'sha': '7bb3362b1768c92ea7e593451b2b8913d3b7616c19fd8d25b73fb6990f9283e0',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.4.115',
       'name': 'engine-lb-584f54c899-sxpws',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': [{'ip': '10.244.3.101',
       'name': 'engine-sidekick-vgg16-clustering-16-56b9b4bb88-mxl7w',
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
      <td>[[[146, 99, 233], [130, 180, 165], [18, 140, 2...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[[[146, 99, 233], [130, 180, 165], [18, 140, 2...</td>
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
      <th>in.images</th>
      <th>out.predictions</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-08-28 20:30:07.086</td>
      <td>[146, 99, 233, 130, 180, 165, 18, 140, 203, 58...</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-08-28 20:30:07.086</td>
      <td>[146, 99, 233, 130, 180, 165, 18, 140, 203, 58...</td>
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

<table><tr><th>name</th> <td>vgg16-clustering-pipeline</td></tr><tr><th>created</th> <td>2023-08-28 20:26:59.888454+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-28 20:29:47.161847+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>9ef719d2-e4ad-4f45-9740-1ea82a7008b8, af9b1b8e-578c-4d72-8bf7-97724301fb45</td></tr><tr><th>steps</th> <td>vgg16-clustering</td></tr><tr><th>published</th> <td>False</td></tr></table>

### Publish the Pipeline for Edge Deployment

It worked! For a demo, we'll take working once as "tested". So now that we've tested our pipeline, we are ready to publish it for edge deployment.

Publishing it means assembling all of the configuration files and model assets and pushing them to an Open Container Initiative (OCI) repository set in the Wallaroo instance as the Edge Registry service.  DevOps engineers then retrieve that image and deploy it through Docker, Kubernetes, or similar deployments.

See [Edge Deployment Registry Guide](https://staging.docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-edge-deployment/) for details on adding an OCI Registry Service to Wallaroo as the Edge Deployment Registry.

This is done through the SDK command `wallaroo.pipeline.publish(deployment_config)` which has the following parameters and returns.

#### Publish a Pipeline Parameters

The `publish` method takes the following parameters.  The containerized pipeline will be pushed to the Edge registry service with the model, pipeline configurations, and other artifacts needed to deploy the pipeline.

| Parameter | Type | Description |
|---|---|---|
| `deployment_config` | `wallaroo.deployment_config.DeploymentConfig` (*Optional*) | Sets the pipeline deployment configuration.  For example:    For more information on pipeline deployment configuration, see the [Wallaroo SDK Essentials Guide: Pipeline Deployment Configuration]({{<ref "wallaroo-sdk-essentials-pipeline-deployment-config">}}).

#### Publish a Pipeline Returns

| Field | Type | Description |
|---|---|---|
| id | integer | Numerical Wallaroo id of the published pipeline. |
| pipeline version id | integer | Numerical Wallaroo id of the pipeline version published. |
| status | string | The status of the pipeline publication.  Values include:  <ul><li>PendingPublish: The pipeline publication is about to be uploaded or is in the process of being uploaded.</li><li>Published:  The pipeline is published and ready for use.</li></ul> |
| Engine URL | string | The URL of the published pipeline engine in the edge registry. |
| Pipeline URL | string | The URL of the published pipeline in the edge registry. |
| Helm Chart URL | string | The URL of the helm chart for the published pipeline in the edge registry. |
| Helm Chart Reference | string | The help chart reference. |
| Helm Chart Version | string | The version of the Helm Chart of the published pipeline.  This is also used as the Docker tag. |
| Engine Config | `wallaroo.deployment_config.DeploymentConfig` | The pipeline configuration included with the published pipeline. |
| Created At | DateTime | When the published pipeline was created. |
| Updated At | DateTime | When the published pipeline was updated. |

### Publish Example

We will now publish the pipeline to our Edge Deployment Registry with the `pipeline.publish(deployment_config)` command.  `deployment_config` is an optional field that specifies the pipeline deployment.  This can be overridden by the DevOps engineer during deployment.

```python
pub = pipeline.publish(deployment_config)
pub
```

    Waiting for pipeline publish... It may take up to 600 sec.
    Pipeline is Publishing......Published.

<table>
    <tr><td>ID</td><td>11</td></tr>
    <tr><td>Pipeline Version</td><td>2c5968b7-62fa-49f8-9489-aaf84cfb80aa</td></tr>
    <tr><td>Status</td><td>Published</td></tr>
    <tr><td>Engine URL</td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/engine:v2023.3.0-main-3731'>ghcr.io/wallaroolabs/doc-samples/engine:v2023.3.0-main-3731</a></td></tr>
    <tr><td>Pipeline URL</td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/pipelines/vgg16-clustering-pipeline:2c5968b7-62fa-49f8-9489-aaf84cfb80aa'>ghcr.io/wallaroolabs/doc-samples/pipelines/vgg16-clustering-pipeline:2c5968b7-62fa-49f8-9489-aaf84cfb80aa</a></td></tr>
    <tr><td>Helm Chart URL</td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/charts/vgg16-clustering-pipeline'>ghcr.io/wallaroolabs/doc-samples/charts/vgg16-clustering-pipeline</a></td></tr>
    <tr><td>Helm Chart Reference</td><td>ghcr.io/wallaroolabs/doc-samples/charts@sha256:4fb4799235f09b7bd548fe2397a67a6d119d2cb284750945f86915376ccd0127</td></tr>
    <tr><td>Helm Chart Version</td><td>0.0.1-2c5968b7-62fa-49f8-9489-aaf84cfb80aa</td></tr>
    <tr><td>Engine Config</td><td>{'engine': {'resources': {'limits': {'cpu': 0.25, 'memory': '4Gi'}, 'requests': {'cpu': 0.25, 'memory': '4Gi'}}}, 'engineAux': {'images': {}}, 'enginelb': {}}</td></tr>
    <tr><td>Created By</td><td>john.hummel@wallaroo.ai</td></tr>
    <tr><td>Created At</td><td>2023-08-28 20:30:48.863741+00:00</td></tr>
    <tr><td>Updated At</td><td>2023-08-28 20:30:48.863741+00:00</td></tr>
</table>

### List Published Pipeline

The method `wallaroo.client.list_pipelines()` shows a list of all pipelines in the Wallaroo instance, and includes the `published` field that indicates whether the pipeline was published to the registry (`True`), or has not yet been published (`False`).

```python
wl.list_pipelines()
```

<table><tr><th>name</th><th>created</th><th>last_updated</th><th>deployed</th><th>tags</th><th>versions</th><th>steps</th><th>published</th></tr><tr><td>vgg16-clustering-pipeline</td><td>2023-28-Aug 20:26:59</td><td>2023-28-Aug 20:30:48</td><td>False</td><td></td><td>2c5968b7-62fa-49f8-9489-aaf84cfb80aa, 9ef719d2-e4ad-4f45-9740-1ea82a7008b8, af9b1b8e-578c-4d72-8bf7-97724301fb45</td><td>vgg16-clustering</td><td>True</td></tr><tr><td>alohapipelineudjo</td><td>2023-28-Aug 17:10:05</td><td>2023-28-Aug 17:10:10</td><td>False</td><td></td><td>750fec3b-f178-4abd-9643-2208e0fd1fbd, e3f6ddff-9a25-48cc-9f94-fbc7bd67db29</td><td>alohamodeludjo</td><td>False</td></tr><tr><td>ejatalohapipeline</td><td>2023-28-Aug 16:59:39</td><td>2023-28-Aug 16:59:44</td><td>False</td><td></td><td>7f4c15e6-f6e3-4145-bed7-02ab9018bd67, 71611219-8559-410b-9bae-5f55a2ae245e</td><td>ejatalohamodel</td><td>False</td></tr><tr><td>housing-pipe</td><td>2023-28-Aug 16:53:31</td><td>2023-28-Aug 16:54:44</td><td>False</td><td></td><td>fede64f0-2a46-4a7f-8a52-59acfbaf7678, 63da191e-93f5-4933-afe5-1c552121d8a0, 45768a6d-0115-4b71-870f-865d479f1b5c</td><td>preprocess</td><td>False</td></tr><tr><td>edge-pipeline</td><td>2023-28-Aug 14:47:11</td><td>2023-28-Aug 14:47:11</td><td>False</td><td></td><td>ab7ab48f-9aab-4c3d-9f66-097477bd34bf</td><td>ccfraud</td><td>True</td></tr><tr><td>edgebiolabspipeline</td><td>2023-27-Aug 19:29:38</td><td>2023-27-Aug 21:43:20</td><td>False</td><td></td><td>d73b969e-7327-4e1c-b756-ae2fd044deb2, e051c3ff-8729-4c22-8939-51b3213c32fa, abe60490-b086-4043-9313-b5564159359c, 3679df28-cf72-4e1f-bee7-7cf3af29a68c, 2a85eb39-b920-4b79-91e0-e11b40ec2324, 8d65156f-3528-49da-8a8a-291fe5d7678e, e4707610-281c-4930-a3d2-338443e567c0, cb23d16b-5812-4401-9fda-fcfaa4ecf4f3, e45f7703-105f-4b6f-a4b0-33905be1bffd, d73c9e45-fed4-4c6b-8d53-e2fd509a043b, bcf8db22-378f-4ebf-bba6-3dc76e3e2d17, 83d3865f-e0e0-4907-9fd9-ef753e7a7c72, ebf39d16-bba6-46fc-b164-b56ac2a06a9f, cc1657d8-9cb8-4ea4-8c6a-9175ccba5c97</td><td>edgebiolabsmodel</td><td>True</td></tr><tr><td>edge-hf-summarization</td><td>2023-25-Aug 18:53:19</td><td>2023-25-Aug 18:54:49</td><td>False</td><td></td><td>af77957c-6af6-4332-aeeb-a4d9e1a22963, ad77bf95-36c9-4669-9fca-69163f5de601, e0ad32af-21e4-42bd-9eb1-273532cf4f15</td><td>hf-summarization</td><td>True</td></tr><tr><td>hf-summarization-pipeline-edge</td><td>2023-25-Aug 15:52:02</td><td>2023-25-Aug 16:24:04</td><td>False</td><td></td><td>c8d94cce-b237-4d03-bef4-eca89d8d5c88, c7a067bc-997b-47c2-89c7-29ddd507cf7d, c1164da4-e044-49d3-a079-2c6c6a8cdc3f, 28176ea4-5717-4c60-b9c0-91a695bfb78d, 2d55d49d-45d6-4d88-9c6b-a3225a2ba565, 55760fa6-3919-4790-93a2-121be29d1962</td><td>hf-summarization-demoyns2</td><td>True</td></tr><tr><td>houseprice-estimator</td><td>2023-24-Aug 21:16:00</td><td>2023-24-Aug 21:22:22</td><td>False</td><td></td><td>72dbd1e6-2852-4790-885b-9c24865e5126, acd2c0fb-107f-49d6-9041-77ffb8c6979a</td><td>house-price-prime</td><td>False</td></tr><tr><td>edge-pipeline</td><td>2023-24-Aug 16:57:29</td><td>2023-24-Aug 17:05:01</td><td>True</td><td></td><td>710aad65-1437-487b-b6db-0f732b5d2d73, 44c71e77-d8fa-4015-aeee-1cdbccfeb0ef, 7c4383d2-b79d-4179-91a1-b592803e1373, d0a55f2b-0938-45a0-ae58-7d78b9b590d6</td><td>ccfraud</td><td>True</td></tr></table>

### List Publishes from a Pipeline

All publishes created from a pipeline are displayed with the `wallaroo.pipeline.publishes` method.  The `pipeline_version_id` is used to know what version of the pipeline was used in that specific publish.  This allows for pipelines to be updated over time, and newer versions to be sent and tracked to the Edge Deployment Registry service.

#### List Publishes Parameters

N/A

#### List Publishes Returns

A List of the following fields:

| Field | Type | Description |
|---|---|---|
| id | integer | Numerical Wallaroo id of the published pipeline. |
| pipeline_version_id | integer | Numerical Wallaroo id of the pipeline version published. |
| engine_url | string | The URL of the published pipeline engine in the edge registry. |
| pipeline_url | string | The URL of the published pipeline in the edge registry. |
| created_by | string | The email address of the user that published the pipeline.
| Created At | DateTime | When the published pipeline was created. |
| Updated At | DateTime | When the published pipeline was updated. |

```python
pipeline.publishes()
```

<table><tr><th>id</th><th>pipeline_version_name</th><th>engine_url</th><th>pipeline_url</th><th>created_by</th><th>created_at</th><th>updated_at</th></tr><tr><td>11</td><td>2c5968b7-62fa-49f8-9489-aaf84cfb80aa</td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/engine:v2023.3.0-main-3731'>ghcr.io/wallaroolabs/doc-samples/engine:v2023.3.0-main-3731</a></td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/pipelines/vgg16-clustering-pipeline:2c5968b7-62fa-49f8-9489-aaf84cfb80aa'>ghcr.io/wallaroolabs/doc-samples/pipelines/vgg16-clustering-pipeline:2c5968b7-62fa-49f8-9489-aaf84cfb80aa</a></td><td>john.hummel@wallaroo.ai</td><td>2023-28-Aug 20:30:48</td><td>2023-28-Aug 20:30:48</td></tr></table>

## DevOps Deployment

We now have our pipeline published to our Edge Registry service.  We can deploy this in a x86 environment running Docker that is logged into the same registry service that we deployed to.

For more details, check with the documentation on your artifact service.  The following are provided for the three major cloud services:

* [Set up authentication for Docker](https://cloud.google.com/artifact-registry/docs/docker/authentication)
* [Authenticate with an Azure container registry](https://learn.microsoft.com/en-us/azure/container-registry/container-registry-authentication?tabs=azure-cli)
* [Authenticating Amazon ECR Repositories for Docker CLI with Credential Helper](https://aws.amazon.com/blogs/compute/authenticating-amazon-ecr-repositories-for-docker-cli-with-credential-helper/)

## DevOps - Pipeline Edge Deployment

Once a pipeline is deployed to the Edge Registry service, it can be deployed in environments such as Docker, Kubernetes, or similar container running services by a DevOps engineer.

### Docker Deployment

First, the DevOps engineer must authenticate to the same OCI Registry service used for the Wallaroo Edge Deployment registry.

For more details, check with the documentation on your artifact service.  The following are provided for the three major cloud services:

* [Set up authentication for Docker](https://cloud.google.com/artifact-registry/docs/docker/authentication)
* [Authenticate with an Azure container registry](https://learn.microsoft.com/en-us/azure/container-registry/container-registry-authentication?tabs=azure-cli)
* [Authenticating Amazon ECR Repositories for Docker CLI with Credential Helper](https://aws.amazon.com/blogs/compute/authenticating-amazon-ecr-repositories-for-docker-cli-with-credential-helper/)

For the deployment, the engine URL is specified with the following environmental variables:

```bash
{published engine url}
-e DEBUG=true -e OCI_REGISTRY={your registry server} \
-e CONFIG_CPUS=4 \ # optional number of CPUs to use
-e OCI_USERNAME={registry username} \
-e OCI_PASSWORD={registry token here} \
-e PIPELINE_URL={published pipeline url}
```

#### Docker Deployment Example

Using our sample environment, here's sample deployment using Docker with a computer vision ML model, the same used in the [Wallaroo Use Case Tutorials Computer Vision: Retail]({{<ref "use-case-computer-vision-retail">}}) tutorials.

```bash
docker run -p 8080:8080 \
    -e DEBUG=true -e OCI_REGISTRY={your registry server} \
    -e CONFIG_CPUS=4 \
    -e OCI_USERNAME=oauth2accesstoken \
    -e OCI_PASSWORD={registry token here} \
    -e PIPELINE_URL={your registry server}/pipelines/edge-cv-retail:bf70eaf7-8c11-4b46-b751-916a43b1a555 \
    {your registry server}/engine:v2023.3.0-main-3707
```

### Docker Compose Deployment

For users who prefer to use `docker compose`, the following sample `compose.yaml` file is used to launch the Wallaroo Edge pipeline.  This is the same used in the [Wallaroo Use Case Tutorials Computer Vision: Retail]({{<ref "use-case-computer-vision-retail">}}) tutorials.

```yml
services:
  engine:
    image: {Your Engine URL}
    ports:
      - 8080:8080
    environment:
      PIPELINE_URL: {Your Pipeline URL}
      OCI_REGISTRY: {Your Edge Registry URL}
      OCI_USERNAME:  {Your Registry Username}
      OCI_PASSWORD: {Your Token or Password}
      CONFIG_CPUS: 4
```

For example:

```yml
services:
  engine:
    image: sample-registry.com/engine:v2023.3.0-main-3707
    ports:
      - 8080:8080
    environment:
      PIPELINE_URL: sample-registry.com/pipelines/edge-cv-retail:bf70eaf7-8c11-4b46-b751-916a43b1a555
      OCI_REGISTRY: sample-registry.com
      OCI_USERNAME:  _json_key_base64
      OCI_PASSWORD: abc123
      CONFIG_CPUS: 4
```

#### Docker Compose Deployment Example

The deployment and undeployment is then just a simple `docker compose up` and `docker compose down`.  The following shows an example of deploying the Wallaroo edge pipeline using `docker compose`.

```bash
docker compose up
[+] Running 1/1
 ✔ Container cv_data-engine-1  Recreated                                                                                                                                                                 0.5s
Attaching to cv_data-engine-1
cv_data-engine-1  | Wallaroo Engine - Standalone mode
cv_data-engine-1  | Login Succeeded
cv_data-engine-1  | Fetching manifest and config for pipeline: sample-registry.com/pipelines/edge-cv-retail:bf70eaf7-8c11-4b46-b751-916a43b1a555
cv_data-engine-1  | Fetching model layers
cv_data-engine-1  | digest: sha256:c6c8869645962e7711132a7e17aced2ac0f60dcdc2c7faa79b2de73847a87984
cv_data-engine-1  |   filename: c6c8869645962e7711132a7e17aced2ac0f60dcdc2c7faa79b2de73847a87984
cv_data-engine-1  |   name: resnet-50
cv_data-engine-1  |   type: model
cv_data-engine-1  |   runtime: onnx
cv_data-engine-1  |   version: 693e19b5-0dc7-4afb-9922-e3f7feefe66d
cv_data-engine-1  |
cv_data-engine-1  | Fetched
cv_data-engine-1  | Starting engine
cv_data-engine-1  | Looking for preexisting `yaml` files in //modelconfigs
cv_data-engine-1  | Looking for preexisting `yaml` files in //pipelines
```

### Helm Deployment

Published pipelines can be deployed through the use of helm charts.

Helm deployments take up to two steps - the first step is in retrieving the required `values.yaml` and making updates to override.

1. Pull the helm charts from the published pipeline.  The two fields are the Helm Chart URL and the Helm Chart version to specify the OCI .    This typically takes the format of:

  ```bash
  helm pull oci://{published.helm_chart_url} --version {published.helm_chart_version}
  ```

1. Extract the `tgz` file and copy the `values.yaml` and copy the values used to edit engine allocations, etc.  The following are **required** for the deployment to run:

  ```yml
  ociRegistry:
    registry: {your registry service}
    username:  {registry username here}
    password: {registry token here}
  ```

  Store this into another file, suc as `local-values.yaml`.

1. Create the namespace to deploy the pipeline to.  For example, the namespace `wallaroo-edge-pipeline` would be:

  ```bash
  kubectl create -n wallaroo-edge-pipeline
  ```

1. Deploy the `helm` installation with `helm install` through one of the following options:
    1. Specify the `tgz` file that was downloaded and the local values file.  For example:

        ```bash
        helm install --namespace {namespace} --values {local values file} {helm install name} {tgz path}
        ```

    1. Specify the expended directory from the downloaded `tgz` file.

        ```bash
        helm install --namespace {namespace} --values {local values file} {helm install name} {helm directory path}
        ```

    1. Specify the Helm Pipeline Helm Chart and the Pipeline Helm Version.

        ```bash
        helm install --namespace {namespace} --values {local values file} {helm install name} oci://{published.helm_chart_url} --version {published.helm_chart_version}
        ```

1. Once deployed, the DevOps engineer will have to forward the appropriate ports to the `svc/engine-svc` service in the specific pipeline.  For example, using `kubectl port-forward` to the namespace `ccfraud` that would be:

    ```bash
    kubectl port-forward svc/engine-svc -n ccfraud01 8080 --address 0.0.0.0`
    ```

The following code segment generates a `docker compose` template based on the previously published pipeline.

```python
docker_compose = f'''
services:
  engine:
    image: {pub.engine_url}
    ports:
      - 8080:8080
    environment:
      PIPELINE_URL: {pub.pipeline_url}
      OCI_USERNAME: YOUR USERNAME 
      OCI_PASSWORD: YOUR PASSWORD OR TOKEN
      OCI_REGISTRY: ghcr.io
      CONFIG_CPUS: 4
'''

print(docker_compose)
```

    
    services:
      engine:
        image: ghcr.io/wallaroolabs/doc-samples/engine:v2023.3.0-main-3731
        ports:
          - 8080:8080
        environment:
          PIPELINE_URL: ghcr.io/wallaroolabs/doc-samples/pipelines/vgg16-clustering-pipeline:2c5968b7-62fa-49f8-9489-aaf84cfb80aa
          OCI_USERNAME: YOUR USERNAME 
          OCI_PASSWORD: YOUR PASSWORD OR TOKEN
          OCI_REGISTRY: ghcr.io
          CONFIG_CPUS: 4
    

## Edge Deployed Pipeline API Endpoints

Once deployed, we can check the pipelines and models available.  We'll use a `curl` command, but any HTTP based request will work the same way.

The endpoint `/pipelines` returns:

* **id** (*String*):  The name of the pipeline.
* **status** (*String*):  The status as either `Running`, or `Error` if there are any issues.

For this example, the deployment is made on a machine called `testboy.local`.  Replace this URL with the URL of you edge deployment.

```python
!curl testboy.local:8080/pipelines
```

    {"pipelines":[{"id":"vgg16-clustering-pipeline","status":"Running"}]}

The endpoint `/models` returns a List of models with the following fields:

* **name** (*String*): The model name.
* **sha** (*String*): The sha hash value of the ML model.
* **status** (*String*):  The status of either Running or Error if there are any issues.
* **version** (*String*):  The model version.  This matches the version designation used by Wallaroo to track model versions in UUID format.

```python
!curl testboy.local:8080/models
```

    {"models":[{"name":"vgg16-clustering","sha":"7bb3362b1768c92ea7e593451b2b8913d3b7616c19fd8d25b73fb6990f9283e0","status":"Running","version":"90c3ea47-0e4f-4de6-9bea-af47e3c9da6f"}]}

### Edge Deployed Inference

The inference endpoint takes the following pattern:

* `/pipelines/{pipeline-name}`:  The `pipeline-name` is the same as returned from the [`/pipelines`](#list-pipelines) endpoint as `id`.

Wallaroo inference endpoint URLs accept the following data inputs through the `Content-Type` header:

* `Content-Type: application/vnd.apache.arrow.file`: For Apache Arrow tables.
* `Content-Type: application/json; format=pandas-records`: For pandas DataFrame in record format.
* `Content-Type: application/json`: JSON.

The `Accept` header will determine what kind format of the data is returned.

* `Accept: application/json`: Returns a JSON object in the following format.

* **check_failures** (*List[Integer]*): Any Validations that failed.  For more information, see [Wallaroo SDK Essentials Guide: Pipeline Management: Anomaly Testing](https://staging.docs.wallaroo.ai/20230201/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline/#anomaly-testing)
* **elapsed** (*List[Integer]*): A List of time in nanoseconds for:
  * [0]: The time to serialize the input.
  * [1...n]: How long each step took.
* **model_name** (*String*): The name of the model.
* **model_version** (*String*): The UUID identifier of the model version from Wallaroo.
* **original_data** (*List*): The original submitted data.
* **outputs** (*List*): A List of outputs with each output field corresponding to the input.  This is in the format for each data type returned:
  * **{data-type}**: The data type being returned.
    * **data** (*List*): The data from this data type.
    * **dim** (*List*): The shape of the data for this data type.
* **dim** (*List*): The shape of the data received.
* **pipeline_name** (*String*): The name of the pipeline.
* **shadow_data** (*List*): Any data returned from a shadow deployed model.  For more information, see [Wallaroo SDK Essentials Guide: Pipeline Management: Shadow Deployments](https://staging.docs.wallaroo.ai/20230201/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline/#pipeline-shadow-deployments).
* **time** (*Integer*): The Epoch time of the inference.
  

```python
numpy = [np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)] * 2

input_data = {
        "images": numpy,
}
dataframe = pd.DataFrame(input_data)
dataframe

# set the content type and accept headers
headers = {
    'Content-Type': 'application/json; format=pandas-records'
}

data = dataframe.to_json(orient="records")

host = 'http://testboy.local:8080'

deployurl = f'{host}/pipelines/vgg16-clustering-pipeline'

response = requests.post(
                deployurl, 
                headers=headers, 
                data=data, 
                verify=True
            )

display(response.json())

```

    [{'check_failures': [],
      'elapsed': [4575572, 140490262],
      'model_name': 'vgg16-clustering',
      'model_version': '90c3ea47-0e4f-4de6-9bea-af47e3c9da6f',
      'original_data': None,
      'outputs': [{'Int64': {'data': [1, 1], 'dim': [2, 1], 'v': 1}}],
      'pipeline_name': 'vgg16-clustering-pipeline',
      'shadow_data': {},
      'time': 1693255638345}]

