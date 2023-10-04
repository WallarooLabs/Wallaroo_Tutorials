The following tutorial is available on the [Wallaroo Github Repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/pipeline-edge-publish/edge-cv-retail).

## Computer Vision Edge Deployment Demonstration

This notebook will walk through building a computer vision (CV) pipeline in Wallaroo, deploying it to the local cluster for testing, and then publishing it for edge deployment.

This demonstration will focus on deployment to the edge.  For further examples of using Wallaroo with this computer vision models, see [Use Case Tutorials: Computer Vision: Retail](https://docs.wallaroo.ai/wallaroo-use-case-tutorials/wallaroo-use-case-computer-vision/use-case-computer-vision-retail/).

This demonstration will perform the following:

1. As a Data Scientist:
    1. Upload a computer vision model to Wallaroo, deploy it in a Wallaroo pipeline, then perform a sample inference.
    1. Publish the pipeline to an Open Container Initiative (OCI) Registry service.  This is configured in the Wallaroo instance.  See [Edge Deployment Registry Guide](https://staging.docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-edge-deployment/) for details on adding an OCI Registry Service to Wallaroo as the Edge Deployment Registry.
    1. View the pipeline publish details.
1. As a DevOps Engineer:
    1. Deploy the published pipeline into an edge instance.  This example will use Docker.
    1. Perform a sample inference into the deployed pipeline with the same data used in the data scientist example.

## Data Scientist Pipeline Publish Steps

### Load Libraries

The first step is to import the libraries used in this notebook.

```python
import wallaroo
from wallaroo.object import EntityNotFoundError

import pyarrow as pa
import pandas as pd

# used to display dataframe information without truncating
from IPython.display import display
pd.set_option('display.max_colwidth', None)
```

### Connect to the Wallaroo Instance through the User Interface

The next step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
wl = wallaroo.Client()
```

### Create a New Workspace

We'll use the SDK below to create our workspace , assign as our **current workspace**, then display all of the workspaces we have at the moment.  We'll also set up variables for our models and pipelines down the road, so we have one spot to change names to whatever fits your organization's standards best.

To allow this tutorial to be run by multiple users in the same Wallaroo instance, a random 4 character prefix will be added to the workspace, pipeline, and model.  Feel free to set `suffix=''` if this is not required.

```python
import string
import random

# make a random 4 character prefix
suffix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))

workspace_name = f'edge-cv-demo{suffix}'
pipeline_name = 'edge-cv-demo'
model_name = 'resnet-50'
model_file_name = './models/resnet50_v1.onnx'
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

```python
workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)
```

    {'name': 'edge-cv-demojohn', 'id': 9, 'archived': False, 'created_by': 'aa707604-ec80-495a-a9a1-87774c8086d5', 'created_at': '2023-09-08T18:25:19.398959+00:00', 'models': [], 'pipelines': []}

### Upload the Model

When a model is uploaded to a Wallaroo cluster, it is optimized and packaged to make it ready to run as part of a pipeline. In many times, the Wallaroo Server can natively run a model without any Python overhead. In other cases, such as a Python script, a custom Python environment will be automatically generated. This is comparable to the process of "containerizing" a model by adding a small HTTP server and other wrapping around it.

Our pretrained model is in ONNX format, which is specified in the `framework` parameter.

```python
model = wl.upload_model(model_name, 
                        model_file_name, 
                        framework = wallaroo.framework.Framework.ONNX)
model
```

<table>
        <tr>
          <td>Name</td>
          <td>resnet-50</td>
        </tr>
        <tr>
          <td>Version</td>
          <td>5fef1823-79fe-4efe-a7d1-7d2eda6f7802</td>
        </tr>
        <tr>
          <td>File Name</td>
          <td>resnet50_v1.onnx</td>
        </tr>
        <tr>
          <td>SHA</td>
          <td>c6c8869645962e7711132a7e17aced2ac0f60dcdc2c7faa79b2de73847a87984</td>
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
          <td>Updated At</td>
          <td>2023-08-Sep 18:37:14</td>
        </tr>
      </table>

### Reserve Pipeline Resources

Before deploying an inference engine we need to tell wallaroo what resources it will need.
To do this we will use the wallaroo DeploymentConfigBuilder() and fill in the options listed below to determine what the properties of our inference engine will be.

We will be testing this deployment for an edge scenario, so the resource specifications are kept small -- what's the minimum needed to meet the expected load on the planned hardware.

- cpus - 4 => allow the engine to use 4 CPU cores when running the neural net
- memory - 4Gi => each inference engine will have 4 GB of memory, which is plenty for processing a single image at a time.
- arch - we will specify the X86 architecture.

```python
deployment_config = wallaroo.DeploymentConfigBuilder() \
    .cpus(4)\
    .memory("4Gi")\
    .arch(wallaroo.engine_config.Architecture.X86)\
    .build()
```

### Simulated Edge Deployment

We will now deploy our pipeline into the current Kubernetes environment using the specified resource constraints. This is a "simulated edge" deploy in that we try to mimic the edge hardware as closely as possible.

```python
pipeline = wl.build_pipeline(pipeline_name)
# clear if the tutorial was run before
pipeline.clear()
pipeline.add_model_step(model)
```

<table><tr><th>name</th> <td>edge-cv-demo</td></tr><tr><th>created</th> <td>2023-09-08 18:25:24.382997+00:00</td></tr><tr><th>last_updated</th> <td>2023-09-08 18:37:14.531293+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>482fc033-00a6-42e7-b359-90611b76f74d, 32805f9a-40eb-4366-b444-635ab466ef76, b412ff15-c87b-46ea-8d96-48868b7867f0, aaf2c947-af26-4b0e-9819-f8aca5657017, 7ad0a22c-6472-4390-8f33-a8b3eccc7877, c73bbf20-8fe3-4714-be5e-e35773fe4779, fc431a83-22dc-43db-8610-cde3095af584</td></tr><tr><th>steps</th> <td>resnet-50</td></tr><tr><th>published</th> <td>True</td></tr></table>

```python
pipeline.deploy(deployment_config = deployment_config)
```

    Waiting for deployment - this will take up to 45s ........ ok

<table><tr><th>name</th> <td>edge-cv-demo</td></tr><tr><th>created</th> <td>2023-09-08 18:25:24.382997+00:00</td></tr><tr><th>last_updated</th> <td>2023-09-08 18:37:14.791209+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>69a912fb-47da-4049-98d5-aa024e7d66b2, 482fc033-00a6-42e7-b359-90611b76f74d, 32805f9a-40eb-4366-b444-635ab466ef76, b412ff15-c87b-46ea-8d96-48868b7867f0, aaf2c947-af26-4b0e-9819-f8aca5657017, 7ad0a22c-6472-4390-8f33-a8b3eccc7877, c73bbf20-8fe3-4714-be5e-e35773fe4779, fc431a83-22dc-43db-8610-cde3095af584</td></tr><tr><th>steps</th> <td>resnet-50</td></tr><tr><th>published</th> <td>True</td></tr></table>

```python
pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.3.159',
       'name': 'engine-7f84547f5c-8t87n',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'edge-cv-demo',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'resnet-50',
          'version': '5fef1823-79fe-4efe-a7d1-7d2eda6f7802',
          'sha': 'c6c8869645962e7711132a7e17aced2ac0f60dcdc2c7faa79b2de73847a87984',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.4.191',
       'name': 'engine-lb-584f54c899-7b7kv',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}

### Run Single Image Inference

A single image, encoded using the Apache Arrow format, is sent to the deployed pipeline. Arrow is used here because, as a binary protocol, there is far lower network and compute overhead than using JSON. The Wallaroo Server engine accepts both JSON, pandas DataFrame, and Apache Arrow formats.

The sample Apache Arrow table is in the file `./data/image_224x224.arrow`.

Once complete, we'll display how long the inference request took.

```python
with pa.ipc.open_file("./data/image_224x224.arrow") as f:
    image = f.read_all()

for _ in range(10):
    results = pipeline.infer(image, dataset=["*", "metadata.elapsed"])

iter = 3
elapsed = 0
for _ in range(iter):
    results = pipeline.infer(image, dataset=["*", "metadata.elapsed"])
    elapsed += results['metadata.elapsed'][0].as_py()[1] / 1000000.0

print(f"Average elapsed: {elapsed/iter} ms")
```

    Average elapsed: 28.956216 ms

### Undeploy Pipeline

We have tested out the inferences, so we'll undeploy the pipeline to retrieve the system resources.

```python
pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s .................................... ok

<table><tr><th>name</th> <td>edge-cv-demo</td></tr><tr><th>created</th> <td>2023-09-08 18:25:24.382997+00:00</td></tr><tr><th>last_updated</th> <td>2023-09-08 18:37:14.791209+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>69a912fb-47da-4049-98d5-aa024e7d66b2, 482fc033-00a6-42e7-b359-90611b76f74d, 32805f9a-40eb-4366-b444-635ab466ef76, b412ff15-c87b-46ea-8d96-48868b7867f0, aaf2c947-af26-4b0e-9819-f8aca5657017, 7ad0a22c-6472-4390-8f33-a8b3eccc7877, c73bbf20-8fe3-4714-be5e-e35773fe4779, fc431a83-22dc-43db-8610-cde3095af584</td></tr><tr><th>steps</th> <td>resnet-50</td></tr><tr><th>published</th> <td>True</td></tr></table>

### Publish the Pipeline for Edge Deployment

It worked! For a demo, we'll take working once as "tested". So now that we've tested our pipeline, we are ready to publish it for edge deployment.

Publishing it means assembling all of the configuration files and model assets and pushing them to an Open Container Initiative (OCI) repository set in the Wallaroo instance as the Edge Registry service.  DevOps engineers then retrieve that image and deploy it through Docker, Kubernetes, or similar deployments.

See [Edge Deployment Registry Guide](https://staging.docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-edge-deployment/) for details on adding an OCI Registry Service to Wallaroo as the Edge Deployment Registry.

This is done through the SDK command `wallaroo.pipeline.publish(deployment_config)`.  This publishes the most recent **pipeline version**.  The alternate method is to use the `wallaroo.pipeline_variant.publish(deployment_config)`, which specifies the pipeline version to publish.

#### Publish a Pipeline Parameters

The `wallaroo.pipeline.publish(deployment_config)` method takes the following parameters.  The containerized pipeline will be pushed to the Edge registry service with the model, pipeline configurations, and other artifacts needed to deploy the pipeline.

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

We will first specify the most recent version of our pipeline, and publish that to our Edge Registry service. 

```python
pub = pipeline.publish(deployment_config)
pub
```

    Waiting for pipeline publish... It may take up to 600 sec.
    Pipeline is Publishing....Published.

<table>
    <tr><td>ID</td><td>9</td></tr>
    <tr><td>Pipeline Version</td><td>7ad0a22c-6472-4390-8f33-a8b3eccc7877</td></tr>
    <tr><td>Status</td><td>Published</td></tr>
    <tr><td>Engine URL</td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/engines/proxy/wallaroo/ghcr.io/wallaroolabs/standalone-mini:v2023.3.0-3798'>ghcr.io/wallaroolabs/doc-samples/engines/proxy/wallaroo/ghcr.io/wallaroolabs/standalone-mini:v2023.3.0-3798</a></td></tr>
    <tr><td>Pipeline URL</td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/pipelines/edge-cv-demo:7ad0a22c-6472-4390-8f33-a8b3eccc7877'>ghcr.io/wallaroolabs/doc-samples/pipelines/edge-cv-demo:7ad0a22c-6472-4390-8f33-a8b3eccc7877</a></td></tr>
    <tr><td>Helm Chart URL</td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/charts/edge-cv-demo'>ghcr.io/wallaroolabs/doc-samples/charts/edge-cv-demo</a></td></tr>
    <tr><td>Helm Chart Reference</td><td>ghcr.io/wallaroolabs/doc-samples/charts@sha256:4b03a85769455ea9aa607ce8f04f847afe83c73d5b2203d5d40ac54161f1ecf4</td></tr>
    <tr><td>Helm Chart Version</td><td>0.0.1-7ad0a22c-6472-4390-8f33-a8b3eccc7877</td></tr>
    <tr><td>Engine Config</td><td>{'engine': {'resources': {'limits': {'cpu': 1.0, 'memory': '512Mi'}, 'requests': {'cpu': 1.0, 'memory': '512Mi'}}}, 'engineAux': {'images': {}}, 'enginelb': {'resources': {'limits': {'cpu': 1.0, 'memory': '512Mi'}, 'requests': {'cpu': 1.0, 'memory': '512Mi'}}}}</td></tr>
    <tr><td>User Images</td><td>[]</td></tr>
    <tr><td>Created By</td><td>john.hummel@wallaroo.ai</td></tr>
    <tr><td>Created At</td><td>2023-09-08 18:26:12.822334+00:00</td></tr>
    <tr><td>Updated At</td><td>2023-09-08 18:26:12.822334+00:00</td></tr>
</table>

### List Published Pipeline

The method `wallaroo.client.list_pipelines()` shows a list of all pipelines in the Wallaroo instance, and includes the `published` field that indicates whether the pipeline was published to the registry (`True`), or has not yet been published (`False`).

```python
wl.list_pipelines()
```

<table><tr><th>name</th><th>created</th><th>last_updated</th><th>deployed</th><th>tags</th><th>versions</th><th>steps</th><th>published</th></tr><tr><td>edge-cv-demo</td><td>2023-08-Sep 18:25:24</td><td>2023-08-Sep 18:26:12</td><td>False</td><td></td><td>7ad0a22c-6472-4390-8f33-a8b3eccc7877, c73bbf20-8fe3-4714-be5e-e35773fe4779, fc431a83-22dc-43db-8610-cde3095af584</td><td>resnet-50</td><td>True</td></tr><tr><td>edge-pipeline</td><td>2023-08-Sep 17:30:28</td><td>2023-08-Sep 18:21:00</td><td>False</td><td></td><td>2d8f9e1d-dc65-4e90-a5ce-ee619162d8cd, 1ea2d089-1127-464d-a980-e087d1f052e2</td><td>ccfraud</td><td>True</td></tr><tr><td>edge-pipeline</td><td>2023-08-Sep 17:24:44</td><td>2023-08-Sep 17:24:59</td><td>True</td><td></td><td>873582f4-4b39-4a69-a2b9-536a0e29927c, 079cf5a1-7e95-4cb7-ae40-381b538371db</td><td>ccfraud</td><td>True</td></tr><tr><td>edge-pipeline-classification-cybersecurity</td><td>2023-08-Sep 15:40:28</td><td>2023-08-Sep 15:45:22</td><td>False</td><td></td><td>60222730-4fb5-4179-b8bf-fa53762fecd1, 86040216-0bbb-4715-b08f-da461857c515, 34204277-bdbd-4ae2-9ce9-86dabe4be5f5, 729ccaa2-41b5-4c8f-89f4-fe1e98f2b303, 216bb86b-f6e8-498f-b8a5-020347355715</td><td>aloha</td><td>True</td></tr><tr><td>edge-pipeline</td><td>2023-08-Sep 15:36:03</td><td>2023-08-Sep 15:36:03</td><td>False</td><td></td><td>83b49e9e-f43d-4459-bb2a-7fa144352307, 73a5d31f-75f5-42c4-9a9d-3ee524113b6c</td><td>aloha</td><td>False</td></tr><tr><td>vgg16-clustering-pipeline</td><td>2023-08-Sep 14:52:44</td><td>2023-08-Sep 14:56:09</td><td>False</td><td></td><td>50d6586a-0661-4f26-802d-c71da2ceea2e, d94e44b3-7ff6-4138-8b76-be1795cb6690, 8d2a8143-2255-408a-bd09-e3008a5bde0b</td><td>vgg16-clustering</td><td>True</td></tr></table>

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

<table><tr><th>id</th><th>pipeline_version_name</th><th>engine_url</th><th>pipeline_url</th><th>created_by</th><th>created_at</th><th>updated_at</th></tr><tr><td>9</td><td>7ad0a22c-6472-4390-8f33-a8b3eccc7877</td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/engines/proxy/wallaroo/ghcr.io/wallaroolabs/standalone-mini:v2023.3.0-3798'>ghcr.io/wallaroolabs/doc-samples/engines/proxy/wallaroo/ghcr.io/wallaroolabs/standalone-mini:v2023.3.0-3798</a></td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/pipelines/edge-cv-demo:7ad0a22c-6472-4390-8f33-a8b3eccc7877'>ghcr.io/wallaroolabs/doc-samples/pipelines/edge-cv-demo:7ad0a22c-6472-4390-8f33-a8b3eccc7877</a></td><td>john.hummel@wallaroo.ai</td><td>2023-08-Sep 18:26:12</td><td>2023-08-Sep 18:26:12</td></tr></table>

## DevOps - Pipeline Edge Deployment

Once a pipeline is deployed to the Edge Registry service, it can be deployed in environments such as Docker, Kubernetes, or similar container running services by a DevOps engineer.

### Docker Deployment

First, the DevOps engineer must authenticate to the same OCI Registry service used for the Wallaroo Edge Deployment registry.

For more details, check with the documentation on your artifact service.  The following are provided for the three major cloud services:

* [Set up authentication for Docker](https://cloud.google.com/artifact-registry/docs/docker/authentication)
* [Authenticate with an Azure container registry](https://learn.microsoft.com/en-us/azure/container-registry/container-registry-authentication?tabs=azure-cli)
* [Authenticating Amazon ECR Repositories for Docker CLI with Credential Helper](https://aws.amazon.com/blogs/compute/authenticating-amazon-ecr-repositories-for-docker-cli-with-credential-helper/)

For the deployment, the engine URL is specified with the following environmental variables:

* `DEBUG` (true|false): Whether to include debug output.
* `OCI_REGISTRY`: The URL of the registry service.
* `CONFIG_CPUS`: The number of CPUs to use.
* `OCI_USERNAME`: The edge registry username.
* `OCI_PASSWORD`:  The edge registry password or token.
* `PIPELINE_URL`: The published pipeline URL.

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
        image: ghcr.io/wallaroolabs/doc-samples/engines/proxy/wallaroo/ghcr.io/wallaroolabs/standalone-mini:v2023.3.0-3798
        ports:
          - 8080:8080
        environment:
          PIPELINE_URL: ghcr.io/wallaroolabs/doc-samples/pipelines/edge-cv-demo:7ad0a22c-6472-4390-8f33-a8b3eccc7877
          OCI_USERNAME: YOUR USERNAME 
          OCI_PASSWORD: YOUR PASSWORD OR TOKEN
          OCI_REGISTRY: ghcr.io
          CONFIG_CPUS: 4
    

## Edge Deployed Pipeline API Endpoints

Once deployed, we can check the pipelines and models available.  We'll use a `curl` command, but any HTTP based request will work the same way.

The endpoint `/pipelines` returns:

* **id** (*String*):  The name of the pipeline.
* **status** (*String*):  The status as either `Running`, or `Error` if there are any issues.

```bash
curl localhost:8080/pipelines
{"pipelines":[{"id":"edge-cv-retail","status":"Running"}]}
```

```python
!curl testboy.local:8080/pipelines
```

    {"pipelines":[{"id":"edge-cv-demo","status":"Running"}]}

The endpoint `/models` returns a List of models with the following fields:

* **name** (*String*): The model name.
* **sha** (*String*): The sha hash value of the ML model.
* **status** (*String*):  The status of either Running or Error if there are any issues.
* **version** (*String*):  The model version.  This matches the version designation used by Wallaroo to track model versions in UUID format.

```bash
curl localhost:8080/models
{"models":[{"name":"resnet-50","sha":"c6c8869645962e7711132a7e17aced2ac0f60dcdc2c7faa79b2de73847a87984","status":"Running","version":"693e19b5-0dc7-4afb-9922-e3f7feefe66d"}]}
```

The following example uses the host `testboy.local`.  Replace with your own host name of your Edge deployed pipeline.

```python
!curl testboy.local:8080/models
```

    {"models":[{"name":"resnet-50","sha":"c6c8869645962e7711132a7e17aced2ac0f60dcdc2c7faa79b2de73847a87984","status":"Running","version":"2e05e1d0-fcb3-4213-bba8-4bac13f53e8d"}]}

### Edge Inference Endpoint

The inference endpoint takes the following pattern:

* `/pipelines/{pipeline-name}`:  The `pipeline-name` is the same as returned from the [`/pipelines`](#list-pipelines) endpoint as `id`.

Wallaroo inference endpoint URLs accept the following data inputs through the `Content-Type` header:

* `Content-Type: application/vnd.apache.arrow.file`: For Apache Arrow tables.
* `Content-Type: application/json; format=pandas-records`: For pandas DataFrame in record format.

Once deployed, we can perform an inference through the deployment URL.

The endpoint returns `Content-Type: application/json; format=pandas-records` by default with the following fields:

* **check_failures** (*List[Integer]*): Whether any validation checks were triggered.  For more information, see [Wallaroo SDK Essentials Guide: Pipeline Management: Anomaly Testing]({{<ref "wallaroo-sdk-essentials-pipeline#anomaly-testing">}}).
* **elapsed** (*List[Integer]*): A list of time in nanoseconds for:
  * [0] The time to serialize the input.
  * [1...n] How long each step took.
* **model_name** (*String*): The name of the model used.
* **model_version** (*String*): The version of the model in UUID format.
* **original_data**: The original input data.  Returns `null` if the input may be too long for a proper return.
* **outputs** (*List*): The outputs of the inference result separated by data type, where each data type includes:
  * **data**: The returned values.
  * **dim** (*List[Integer]*): The dimension shape returned.
  * **v** (*Integer*): The vector shape of the data.
* **pipeline_name**  (*String*): The name of the pipeline.
* **shadow_data**: Any shadow deployed data inferences in the same format as **outputs**.
* **time** (*Integer*): The time since UNIX epoch.
  
Once deployed, we can perform an inference through the deployment URL.  We'll assume we're running the inference request through the localhost and submitting the local file `./data/image_224x224.arrow`.  Note that our inference endpoint is `pipelines/edge-cv-retail` - the same as our pipeline name.

The following example demonstrates sending the same inference requests as above, only via API to the Edge inference endpoints.

```python
!curl -X POST testboy.local:8080/pipelines/edge-cv-demo -H "Content-Type: application/vnd.apache.arrow.file" --data-binary @./data/image_224x224.arrow
```

    [{"check_failures":[],"elapsed":[1067541,21209776],"model_name":"resnet-50","model_version":"2e05e1d0-fcb3-4213-bba8-4bac13f53e8d","original_data":null,"outputs":[{"Int64":{"data":[535],"dim":[1],"v":1}},{"Float":{"data":[0.00009498586587142199,0.00009141524787992239,0.0004606838047038764,0.00007667174941161647,0.00008047101437114179,0.00006355856748996302,0.0001758082798914984,0.000014166356777423061,0.00004344096305430867,0.00004225136217428371,0.0002540049026720226,0.005299815908074379,0.00016666953160893172,0.0001903128286357969,0.00020846890402026474,0.00014618519344367087,0.00034408149076625705,0.0008281364571303129,0.000119782991532702,0.00020627757476177067,0.00014886556891724467,0.0002607095520943403,0.000900866580195725,0.0014754909789189696,0.0008267511730082333,0.00030276484903879464,0.0001936637272592634,0.0005283929640427232,0.0001492276060162112,0.0002412181202089414,0.00041593582136556506,0.00003615637979237363,0.00024112094251904637,0.0001600235846126452,0.00012632399739231914,0.00026500714011490345,0.00007694320811424404,0.00016679221880622208,0.00003494401244097389,0.0017466217977926135,0.00015653071750421077,0.00019110953144263476,0.00011712997365975752,0.0005060972762294114,0.0005225501372478902,0.00018326807185076177,0.0003845964965876192,0.00006664674583589658,0.0006007050978951156,0.000031040912290336564,0.00008364469249499962,0.0005029476596973836,0.00021304779511410743,0.0006510639796033502,0.00027023922302760184,0.0001951652520801872,0.00004173403431195766,0.00018414676014799625,0.00005605358455795795,0.000153101296746172,0.0029485111590474844,0.00010002857743529603,0.00008645151683595031,0.000371249538147822,0.0010456130839884281,0.0016248043393716216,0.00036552781239151955,0.0005579001735895872,0.00011697380250552669,0.0004602446570061147,0.00009498831786913797,0.00018202984938398004,0.0003983260248787701,0.00015528377844020724,0.002486655255779624,0.00014153339725453407,0.0004125907726120204,0.00010047079558717087,0.00038154804497025907,0.001140037551522255,0.0008210585219785571,0.001470172661356628,0.0004218766698613763,0.00032689858926460147,0.0004138693620916456,0.00009274575131712481,0.0002862788096535951,0.0005950824706815183,0.0003809730114880949,0.00035864440724253654,0.000749282306060195,0.0002762994263321161,0.00031715870136395097,0.0016836447175592184,0.000391888665035367,0.0003621992073021829,0.00044199329568073153,0.0002009006857406348,0.00010593135084491223,0.0009713732288219035,0.0005836604977957904,0.00027098090504296124,0.00007530449511250481,0.00006718681834172457,0.0016714695375412703,0.00012960520689375699,0.00002487235724402126,0.0001577018410898745,0.000090404661023058,0.00006928551738383248,0.00004740867007058114,0.0002472156484145671,0.03141512721776962,0.00021109479712322354,0.00030715856701135635,0.00008931805496104062,0.00011660819291137159,0.0001696061808615923,0.0006639185012318194,9.541420695313718e-6,0.0000359770237992052,0.0001311398227699101,0.000038840589695610106,0.00019295488891657442,0.00001744666587910615,0.00004296033512218855,0.00016763315943535417,0.0012729717418551445,0.0006275437772274017,0.007415024098008871,0.0009311998728662729,0.00008604206959716976,0.00011408662248868495,0.0005248250090517104,0.00016869320825207978,0.00007456653838744387,0.0005433708429336548,0.00021447183098644018,0.00016884137585293502,0.0001395243452861905,0.00023141947167459875,0.0002781787479761988,0.0006277807988226414,0.0004487009427975863,0.0005733187426812947,0.00025018330779857934,0.00038505616248585284,0.0009276813943870366,0.00022452330449596047,0.00022443864145316184,0.0004369820817373693,0.00023927201982587576,0.0003679501824080944,0.00035244139144197106,0.0007338618161156774,0.00007371179526671767,0.00012836324458476156,0.00021939525322522968,0.00017031336028594524,0.0003373581275809556,0.0002003818517550826,0.0001500487996963784,0.00015647034160792828,0.00011027725849999115,0.0008106996538117528,0.00017924379790201783,0.00012851452629547566,0.0005347997648641467,0.00027042083092965186,0.0003134775906801224,0.00008673530828673393,0.00012084542686352506,0.00021569391537923366,0.0000919712329050526,0.0003878172137774527,0.00014786726387683302,0.00022733310470357537,0.00007891463610576466,0.0001840223849285394,0.00039511467912234366,0.000033926200558198616,0.00007368090155068785,0.00024970099912025034,0.00008823492680676281,0.00034549637348391116,0.000151101456140168,0.00005533627700060606,0.00008519103721482679,0.000214536368730478,0.000717825023457408,0.00013769639190286398,0.0018266920233145356,0.00017284344357904047,0.00006754085916327313,0.00015001595602370799,0.000780453730840236,0.0002401376113994047,0.00008981378050521016,0.00010351323726354167,0.00016653115744702518,0.00022738341067451984,0.00017854891484603286,0.00005073372813058086,0.00022161378001328558,0.00008385455294046551,0.00007755459955660626,0.00006896461127325892,0.00021314450714271516,0.00015701152733527124,0.00017486776050645858,0.00007180214743129909,0.00022100603382568806,0.0002457557711750269,0.00016138350474648178,0.00007672946230741218,0.00015351278125308454,0.00011571095819817856,0.0003623465308919549,0.00024095601111184806,0.000046604818635387346,0.00007692135113757104,0.0004624855355359614,0.00025198038201779127,0.0002291945565957576,0.00017196634144056588,0.0001292970555368811,0.00012269521539565176,0.00013313270756043494,0.00011138558329548687,0.00023298188170883805,0.00010513139568502083,0.000028025242500007153,0.0000748007878428325,0.00008719053585082293,0.00006468429637607187,0.00008809879363980144,0.00027453622897155583,0.000174011685885489,0.00023895299818832427,0.00036519349669106305,0.00025978428311645985,0.00010121530067408457,0.00012700716615654528,0.00008325914677698165,0.00014043778355699033,0.00006267145363381132,0.0002136369002982974,0.00014604236639570445,0.00015019369311630726,0.00011775943858083338,0.00026230255025438964,0.00023179285926744342,0.0003367597528267652,0.00025973093579523265,0.0001699720451142639,0.00020027677237521857,0.00009662854427006096,0.00008577309927204624,0.00020530540496110916,0.00041713067912496626,0.00022278742108028382,0.0002722055360209197,0.00010114110773429275,0.0002827951975632459,0.00010052188008558005,0.00020146783208474517,0.0004074680618941784,0.00011438109504524618,0.00007723412272753194,0.00018511175585445017,0.0001034958113450557,0.0002637268917169422,0.00016025023069232702,0.00008702953346073627,0.00013277711695991457,0.0003259497170802206,0.00013994183973409235,0.00017213822866324335,0.00016190540918614715,0.00014610638027079403,0.0003115086001344025,0.00015091041859705,0.00034696690272539854,0.0007292924565263093,0.0021730386652052402,0.0007040738128125668,0.0005063159624114633,0.000057511686463840306,0.00004752138556796126,0.00013986059639137238,0.00007857958553358912,0.00009013563249027357,0.0001743465691106394,0.00008407327550230548,0.00037599020288325846,0.00012815414811484516,0.00005326466271071695,0.00015554114361293614,0.0002470168692525476,0.00015018251724541187,0.00039412567275576293,0.00014846271369606256,0.00014515273505821824,0.00022573399473913014,0.0004045003734063357,0.00010239946277579293,0.00004118842480238527,0.0005723059293814003,0.0004498929774854332,0.00026333684218116105,0.00012577157758641988,0.0007940831710584462,0.00019071380665991455,0.0004915734170936048,0.00016221955593209714,0.0008366875699721277,0.0006234603351913393,0.00018929754151031375,0.00009834141383180395,0.0011205484624952078,0.0015215011080726981,0.0005397516069933772,0.00004581292887451127,0.00009577121090842411,0.00008625280315754935,0.00010854312859009951,0.00007731887308182195,0.00018301131785847247,0.00008435356721747667,0.00009233105811290443,0.00007363689655903727,0.00012422216241247952,0.000692436471581459,0.00032979081152006984,0.00005544300438486971,0.00006338889215840027,0.00028664490673691034,0.00010128327994607389,0.00016046807286329567,0.00006382978608598933,0.00009259012585971504,0.00012247092672623694,0.00019537298067007214,0.000042449264583410695,0.000028542715881485492,0.00023056937789078802,0.00011885951244039461,0.00006731662870151922,0.00005814673568238504,0.00013051605492364615,0.00015240366337820888,0.00044564835843630135,0.0007350511732511222,0.00011247357178945094,0.0001914570457302034,0.00025515921879559755,0.0000685819104546681,0.0005027491715736687,0.0002097149845212698,0.0006290936726145446,0.0003259264340158552,0.00005935063018114306,0.0002843359543476254,0.00014050309255253524,0.00010275054955855012,0.00010872588609345257,0.00028568546986207366,0.000048066969611682,0.00016030800179578364,0.00013734235835727304,0.0001407296658726409,0.00018133342382498085,0.00035974333877675235,0.00003342011768836528,0.00020700889581348747,0.0001783567713573575,0.00012609988334588706,0.00035061853122897446,0.00028656359063461423,0.00025325710885226727,0.0000648176865070127,0.00011849744623759761,0.000041553219489287585,0.00017081378609873354,0.00022553828603122383,0.00022503039508592337,0.0000931524918996729,0.000313989061396569,0.00010199484677286819,0.00007633002678630874,0.0005536979879252613,0.00005425702693173662,0.000020687606593128294,0.00023762721684761345,0.00006069021765142679,0.00004019628977403045,0.00017379583732690662,0.00003340014518471435,0.00008545353193767369,0.0002544641029089689,0.002068242058157921,0.0005671332473866642,0.00011009509762516245,0.0011866018176078796,0.0003142678178846836,0.00041002873331308365,0.0007057273178361356,0.000123012374388054,0.00008512793283443898,0.00003627919068094343,0.001111032790504396,0.00004476181129575707,0.00035930838203057647,0.0014156019315123558,0.004740036558359861,0.00030492417863570154,0.000017281434338656254,0.003115355037152767,0.0005614555557258427,0.0056387088261544704,0.0014335709856823087,0.0010917949257418513,0.0003833889204543084,0.00022159960644785315,0.0004050695861224085,0.00007660030678380281,0.000056392855185549706,0.0008680583559907973,0.0010005292715504766,0.0001236498646903783,0.0008075831574387848,0.00013508096162695438,0.0001181716361315921,0.006565396673977375,0.00030239636544138193,0.00021670969726983458,0.0005258044111542404,0.00018677936168387532,0.0005671478575095534,0.0020488083828240633,0.0004142673860769719,0.000468193378765136,0.001288025639951229,0.00017147391918115318,0.0006056931451894343,0.00006368015601765364,0.00037113376311026514,0.02096373960375786,0.0005652746767736971,0.00021039317653048784,0.00014829063729848713,0.000028620210287044756,0.02560994029045105,0.00013037299504503608,0.00010839566675713286,0.000019789582438534126,0.00022670374892186373,0.007652463391423225,0.00018585636280477047,0.00004537768836598843,0.0016958207124844193,0.00006311608740361407,0.0003554021823219955,0.01116266380995512,0.0006776303634978831,0.0007829949609003961,0.003253006609156728,0.00014230640954338014,0.00003876191840390675,0.000052642779337475076,0.0004918651538901031,0.0007090230938047171,0.00013544323155656457,0.00006474388646893203,0.017759324982762337,0.00013648145250044763,0.0000720634707249701,0.000030715913453605026,0.0001699164422461763,0.0007694593514315784,0.00004094455289305188,0.00042997923446819186,0.00069183181039989,0.00042795788613148034,0.00013007057714276016,0.0002349170099478215,0.000690680171828717,0.0003230010624974966,0.004702151753008366,0.002261379035189748,0.000011289697795291431,0.00041112431790679693,0.003011296270415187,0.0005294234142638743,0.0006860736175440252,0.0008129160269163549,0.00025201018434017897,0.0002725508820731193,0.0001608116872375831,0.00004076231562066823,0.02143017388880253,0.00015415706729982048,0.0005084978765808046,0.0007915119058452547,0.011835966259241104,0.0009027220075950027,0.0005889892345294356,0.00028480790206231177,0.002500005066394806,0.000319988263072446,0.000029753980925306678,0.0004997228388674557,0.00007795573037583381,0.008496513590216637,0.000675845192745328,0.0010315540712326765,0.00028518529143184423,0.0005604343023151159,0.0007420527981594205,0.00026658442220650613,0.000670255976729095,0.000549429445527494,0.000486265926156193,0.0008852951577864587,0.0010208644671365619,0.0005504823056980968,0.00006643193773925304,0.00016707296890672296,0.0018395154038444161,0.00020792883879039437,0.000672424619551748,0.001722491579130292,0.0004128349828533828,0.0007139745284803212,0.00011114005610579625,0.033084772527217865,0.00010199168900726363,0.0000802336071501486,0.00004980604717275128,0.00042539965943433344,0.0007217631209641695,0.001107925781980157,0.0006939570885151625,0.0013572226744145155,0.0015684914542362094,0.00006747210136381909,0.001139119267463684,0.0010041212663054466,0.00007046740938676521,0.00024608318926766515,0.008800563402473927,0.00035966665018349886,0.003451175056397915,0.00009517156286165118,0.0008660504827275872,0.00023326478549279273,0.00008802556112641469,0.003872094675898552,0.0002133018133463338,0.0025237654335796833,0.0023425938561558723,0.00009528759255772457,0.00006937789294170216,0.0000788723846198991,0.0009900140576064587,0.00008783437078818679,0.00006110726826591417,0.0001720589498290792,0.0026126252487301826,0.00008416661876253784,0.00010105230467161164,0.0004147965519223362,0.00007185471622506157,0.0010353204561397433,0.00008421491656918079,0.0026522218249738216,0.00005155754115548916,0.00011890973837580532,0.000894684053491801,0.0015078024007380009,0.00021264120005071163,0.00003852442023344338,0.00005637478534481488,0.000014754580661247019,0.008194838650524616,0.025179794058203697,0.003381428774446249,0.00012250264990143478,0.0015897983685135841,0.001364692929200828,0.0037181139923632145,0.0004787960206158459,0.000597482779994607,0.0009444899624213576,0.004496522713452578,0.00030056259129196405,0.00009878808486973867,0.019855067133903503,0.0002870318421628326,0.0003189035633113235,0.000463134580058977,0.005991595331579447,0.001109348377212882,0.0007891665445640683,0.00005801042425446212,0.004181323107331991,0.0008145738393068314,0.0011338794138282537,0.0002629592490848154,0.0003492420946713537,0.00006442250014515594,0.0027946687769144773,0.00026692228857427835,0.00010071288124890998,0.002296908525750041,0.0004214767541270703,0.003413868835195899,0.0039136060513556,0.00034663931000977755,0.002451243344694376,0.003369604703038931,0.0005183491157367826,0.0007979461806826293,0.0004219271650072187,0.012866662815213203,0.0000348803041561041,0.00026076199719682336,0.005895869806408882,0.0001772861141944304,0.00021437151008285582,0.003132300218567252,0.0006888847565278411,0.0031361605506390333,0.0018961209570989013,0.0009908544598147273,0.00003762157939490862,0.0026160927955061197,0.0007033298606984317,0.00014954865036997944,0.0006599962944164872,0.00013283664884511381,0.0001075687978300266,0.00577824329957366,0.0004715236136689782,0.00035480462247505784,0.003188936971127987,0.00006447486521210521,0.0002320961357327178,0.0012023726012557745,0.0008255292195826769,0.00007059726340230554,0.0016246122540906072,0.0007324085454456508,0.00007100572838680819,0.0014354230370372534,0.00003152315184706822,0.00010502287477720529,0.00003169905539834872,0.0007149870507419109,0.00004233977961121127,0.00010430769179947674,0.0001382120535708964,0.000017395004761056043,0.004259190522134304,0.00011460930545581505,0.0010657149832695723,0.00011586499749682844,0.0008855801424942911,0.00026814601733349264,0.0001403211208526045,0.0001127241994254291,0.00026420882204547524,0.00009847767796600237,0.00030476285610347986,0.00023018191859591752,0.001593792694620788,0.0006807689205743372,0.002463536337018013,0.010918837040662766,0.0011293744901195168,0.0005581072764471173,0.0029545058496296406,0.00118914560880512,0.00015714613255113363,0.00834642630070448,0.002323272405192256,0.0000477357316412963,0.009146459400653839,0.00004678943150793202,0.0002719701733440161,0.00036042684223502874,0.000027832909836433828,0.00020616379333660007,0.00026969736791215837,0.0010894923470914364,0.00012293286272324622,0.0004029597039334476,0.0050276415422558784,0.0002885114517994225,0.00007667397585464641,0.005518381949514151,0.0008715179865248501,0.0032090998720377684,0.0007127528078854084,0.00008145146421156824,0.00006068714719731361,0.0000233462833421072,0.00008884121780283749,0.0001942482340382412,0.004549853038042784,0.00014450379239860922,0.0003629896673373878,0.00017611074144952,0.0006488843355327845,0.001840242650359869,0.0021717173513025045,0.0027140495367348194,0.00018598066526465118,0.00009417089313501492,0.00044673620141111314,0.0010807416401803493,0.0004274447856005281,0.00004169759631622583,0.001155734877102077,0.0006934652337804437,0.0001971444144146517,0.00021102074242662638,0.00255193910561502,0.00016984394460450858,0.00014682677283417434,0.004810339771211147,0.00014269481471274048,0.016139861196279526,0.00027973760734312236,0.0024672106374055147,0.00003682959868456237,0.00006452423258451745,0.0007991477614268661,0.00008228424121625721,0.00018033738888334483,0.0002582163433544338,0.003253732342272997,0.0005689526442438364,0.001316008041612804,0.00017250659584533423,0.0014401093358173966,0.0010854782303795218,0.00028539696359075606,0.004279123153537512,0.000606553687248379,0.00410532345995307,0.000699279538821429,0.00013334535469766706,0.0003922379110008478,0.0036946171894669533,0.0008572317892685533,0.00013983626558911055,0.0004357346915639937,0.00003284418198745698,0.002029589144513011,0.00064713234314695,0.0015882503939792514,0.001108865486457944,0.00003452750024734996,0.010173125192523003,0.0014297740999609232,0.00038526937714777887,0.000475258071674034,0.0009320021490566432,0.000130272819660604,0.005989774130284786,0.0006256481283344328,0.0022420701570808887,0.007279571145772934,0.00024401461996603757,0.0007026214152574539,0.00020484825654421002,0.0018617674941197038,0.0073440195992589,0.00108207983430475,0.00008027631702134386,0.0005128596094436944,0.00031330916681326926,0.0016340112779289484,0.006782021839171648,0.006025397218763828,0.000020351779312477447,0.00017984061560127884,0.00027681243955157697,0.00003426772673265077,0.0018119995947927237,0.0006264462717808783,0.00027333630714565516,0.0021321766544133425,0.0006001786096021533,0.0004534905019681901,0.0009496444836258888,0.00038760388270020485,0.00036731260479427874,0.001763870008289814,0.0006593285361304879,0.00015793211059644818,0.00017143605509772897,0.00020174508972559124,0.00020106321608182043,0.0037068817764520645,0.00018330870079807937,0.00038726229104213417,0.00044403309584595263,0.00020527085871435702,0.000026126906959689222,0.0012235183967277408,0.0017890000017359853,0.0017822254449129105,0.028902743011713028,0.0004932042793370783,0.00021147049847058952,0.00009399655391462147,0.00025304852169938385,0.006250543519854546,0.000049950078391702846,0.000037343135772971436,0.0001774897100403905,0.00010433236457174644,0.0010067168623209,0.00011981612624367699,0.00006331669283099473,0.003840849967673421,0.0009726308635435998,0.0019491907441988587,0.00005032110493630171,0.0006725862622261047,0.000916636607144028,0.00015008336049504578,0.00047226098831743,0.0003718054504133761,0.00010026849486166611,0.000554478436242789,0.0004076138138771057,0.0014444665284827352,0.0001920631038956344,0.0030017290264368057,0.0005599703290499747,0.0005014916532672942,0.00002119927557941992,0.0040903291665017605,0.0006480140727944672,0.0068014939315617085,0.00003470692900009453,0.0008900888497009873,0.000303997949231416,0.00007904773519840091,0.0031371102668344975,0.004015175625681877,0.00006729082815581933,0.0027373251505196095,0.0013298088451847434,0.00014384483802132308,0.00040448474464938045,0.00025029690004885197,0.003962395712733269,0.00020060480164829642,0.002549479715526104,0.0005042953998781741,0.00015140711911953986,0.000023598391635459848,0.000060946174926357344,0.00008476997027173638,0.0000808067707112059,0.0003207217960152775,0.00037995382444933057,0.00017761738854460418,0.00021202430070843548,0.0038306494243443012,0.00010710978676797822,0.00002533388942538295,0.0014211576199159026,0.000416610884713009,0.0008034217171370983,0.00020861040684394538,0.0004556515777949244,0.00019095840980298817,0.0009474422549828887,0.00033907717443071306,0.003743865992873907,0.00020461737585719675,0.00237844861112535,0.00027825901634059846,0.00011422026000218466,0.00009886994666885585,0.0008324604714289308,0.00010915553866652772,0.0014770248671993613,0.00438002310693264,0.00330581353046,0.0004904119414277375,0.000939111749175936,0.004294249694794416,0.004433757625520229,0.0011093608336523175,0.0014735679142177105,0.00013492221478372812,0.0031802221201360226,0.009836934506893158,0.00016089445853140205,0.0014605469768866897,0.0011098716640844941,0.00295627792365849,0.0007235891534946859,0.00012282557145226747,0.0001760651357471943,0.00043235180783085525,0.0003031664527952671,0.0001412580895703286,0.0002472866326570511,0.0004931027069687843,0.00004568049553199671,0.0056058987975120544,0.0001193483840324916,0.00020548298198264092,0.000508841301780194,0.00030003293068148196,0.002218998735770583,0.0000868467177497223,0.00015559833263978362,0.000030328314096550457,0.00020001422672066838,0.00016394644626416266,0.00018518617434892803,0.00005766172398580238,0.000023466711354558356,0.000652167946100235,0.00019569970027077943,0.00003589003244997002,0.00004542778697214089,0.000029700044251512736,0.00017792297876439989,0.00010494553134776652,0.00008614901162218302,0.00005918580791330896,0.00008496203372487798,0.0003192806034348905,0.00005834848707308993,0.00032498640939593315,0.00014573335647583008,0.00009598095493856817,0.00034375712857581675,0.00016984314424917102,0.00004552476093522273,0.0001578650262672454,0.0004491916042752564,0.00008291343692690134,0.00017383544764015824,0.00019677006639540195,0.00017128414765466005,0.0005099810659885406,0.00004297650230000727,0.00023316980514209718,0.00006352208583848551,0.00010151478636544198,0.000051395731134107336,0.00033004459692165256,0.0001497933699283749,0.00007846359221730381,0.00003207534609828144,0.00009726793359732255,0.00007835646829335019,0.0009405238670296967,0.0006174164591357112,0.00045560835860669613,0.0010927320690825582,0.0002573138917796314,0.00025607412680983543,0.0000861210428411141,0.00005662925468641333,0.0006073255790397525,0.000041197814425686374,0.00019046534725930542,0.000269181007752195,0.0003623024676926434,0.00009973799024010077,0.0003084538329858333,0.0001476453908253461,0.00002188755024690181,0.00014684967754874378,0.00012759050878230482,0.0004005460941698402,0.00008555448584957048,0.00011063445708714426,0.00024984890478663146,0.00019454439461696893,0.00034877064172178507,0.000070881076680962,0.00003814319643424824,0.00012102609616704285,0.00023900199448689818,0.00008068171882769093,0.00013133625907357782,0.0001693719532340765,0.00012985065404791385,0.0007559973746538162],"dim":[1,1001],"v":1}}],"pipeline_name":"edge-cv-demo","shadow_data":{},"time":1694205578428}]

```python
import requests
import json
import pandas as pd

# set the content type and accept headers
headers = {
    'Content-Type': 'application/vnd.apache.arrow.file'
}

# Submit arrow file
dataFile="./data/image_224x224.arrow"

data = open(dataFile,'rb').read()

host = 'http://testboy.local:8080'

deployurl = f'{host}/pipelines/edge-cv-demo'

response = requests.post(
                    deployurl, 
                    headers=headers, 
                    data=data, 
                    verify=True
                )
display(pd.DataFrame(response.json()))
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>check_failures</th>
      <th>elapsed</th>
      <th>model_name</th>
      <th>model_version</th>
      <th>original_data</th>
      <th>outputs</th>
      <th>pipeline_name</th>
      <th>shadow_data</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[]</td>
      <td>[804580, 22115674]</td>
      <td>resnet-50</td>
      <td>2e05e1d0-fcb3-4213-bba8-4bac13f53e8d</td>
      <td>None</td>
      <td>[{'Int64': {'data': [535], 'dim': [1], 'v': 1}...</td>
      <td>edge-cv-demo</td>
      <td>{}</td>
      <td>1694205665513</td>
    </tr>
  </tbody>
</table>
