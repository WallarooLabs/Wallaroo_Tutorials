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
# import string
# import random

# # make a random 4 character prefix
# suffix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))

suffix=''

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

    {'name': 'edge-cv-demo', 'id': 15, 'archived': False, 'created_by': 'b030ff9c-41eb-49b4-afdf-2ccbecb6be5d', 'created_at': '2023-10-10T16:51:54.492798+00:00', 'models': [], 'pipelines': []}

### Upload the Model

When a model is uploaded to a Wallaroo cluster, it is optimized and packaged to make it ready to run as part of a pipeline. In many times, the Wallaroo Server can natively run a model without any Python overhead. In other cases, such as a Python script, a custom Python environment will be automatically generated. This is comparable to the process of "containerizing" a model by adding a small HTTP server and other wrapping around it.

Our pretrained model is in ONNX format, which is specified in the `framework` parameter.

```python
model = wl.upload_model(model_name, 
                        model_file_name, 
                        framework = wallaroo.framework.Framework.ONNX).configure(tensor_fields=["tensor"])
model
```

<table>
        <tr>
          <td>Name</td>
          <td>resnet-50</td>
        </tr>
        <tr>
          <td>Version</td>
          <td>231233d2-837d-4b8d-bc2d-35b04f387fa8</td>
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
          <td>2023-10-Oct 16:52:01</td>
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

<table><tr><th>name</th> <td>edge-cv-demo</td></tr><tr><th>created</th> <td>2023-10-10 16:52:02.601870+00:00</td></tr><tr><th>last_updated</th> <td>2023-10-10 16:52:02.601870+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>5320bc9c-e64f-4bc1-ac97-7d2b40eeb53e</td></tr><tr><th>steps</th> <td></td></tr><tr><th>published</th> <td>False</td></tr></table>

```python
pipeline.deploy(deployment_config = deployment_config)
```

<table><tr><th>name</th> <td>edge-cv-demo</td></tr><tr><th>created</th> <td>2023-10-10 16:52:02.601870+00:00</td></tr><tr><th>last_updated</th> <td>2023-10-10 16:52:04.159627+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>a91a8b17-f587-439f-9001-292892e94336, 5320bc9c-e64f-4bc1-ac97-7d2b40eeb53e</td></tr><tr><th>steps</th> <td>resnet-50</td></tr><tr><th>published</th> <td>False</td></tr></table>

```python
pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.3.3',
       'name': 'engine-77b56d8d4f-29zf6',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'edge-cv-demo',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'resnet-50',
          'version': '231233d2-837d-4b8d-bc2d-35b04f387fa8',
          'sha': 'c6c8869645962e7711132a7e17aced2ac0f60dcdc2c7faa79b2de73847a87984',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.4.26',
       'name': 'engine-lb-584f54c899-t5ht4',
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

    Average elapsed: 18.49911266666667 ms

### Undeploy Pipeline

We have tested out the inferences, so we'll undeploy the pipeline to retrieve the system resources.

```python
pipeline.undeploy()
```

<table><tr><th>name</th> <td>edge-cv-demo</td></tr><tr><th>created</th> <td>2023-10-10 16:52:02.601870+00:00</td></tr><tr><th>last_updated</th> <td>2023-10-10 16:52:04.159627+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>a91a8b17-f587-439f-9001-292892e94336, 5320bc9c-e64f-4bc1-ac97-7d2b40eeb53e</td></tr><tr><th>steps</th> <td>resnet-50</td></tr><tr><th>published</th> <td>False</td></tr></table>

### Publish the Pipeline for Edge Deployment

It worked! For a demo, we'll take working once as "tested". So now that we've tested our pipeline, we are ready to publish it for edge deployment.

Publishing it means assembling all of the configuration files and model assets and pushing them to an Open Container Initiative (OCI) repository set in the Wallaroo instance as the Edge Registry service.  DevOps engineers then retrieve that image and deploy it through Docker, Kubernetes, or similar deployments.

See [Edge Deployment Registry Guide](https://staging.docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-edge-deployment/) for details on adding an OCI Registry Service to Wallaroo as the Edge Deployment Registry.

This is done through the SDK command `wallaroo.pipeline.publish(deployment_config)`.  This publishes the most recent **pipeline version**.  The alternate method is to use the `wallaroo.pipeline_variant.publish(deployment_config)`, which specifies the pipeline version to publish.

#### Publish a Pipeline Parameters

The `wallaroo.pipeline.publish(deployment_config)` method takes the following parameters.  The containerized pipeline will be pushed to the Edge registry service with the model, pipeline configurations, and other artifacts needed to deploy the pipeline.

| Parameter | Type | Description |
|---|---|---|
| `deployment_config` | `wallaroo.deployment_config.DeploymentConfig` (*Optional*) | Sets the pipeline deployment configuration.  For example:    For more information on pipeline deployment configuration, see the [Wallaroo SDK Essentials Guide: Pipeline Deployment Configuration](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline-deployment-config/).

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
    Pipeline is Publishing...Published.

<table>
    <tr><td>ID</td><td>9</td></tr>
    <tr><td>Pipeline Version</td><td>4f116503-6506-47d6-b427-1e7056a8c62e</td></tr>
    <tr><td>Status</td><td>Published</td></tr>
    <tr><td>Engine URL</td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/engines/proxy/wallaroo/ghcr.io/wallaroolabs/standalone-mini:v2023.3.0-3854'>ghcr.io/wallaroolabs/doc-samples/engines/proxy/wallaroo/ghcr.io/wallaroolabs/standalone-mini:v2023.3.0-3854</a></td></tr>
    <tr><td>Pipeline URL</td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/pipelines/edge-cv-demo:4f116503-6506-47d6-b427-1e7056a8c62e'>ghcr.io/wallaroolabs/doc-samples/pipelines/edge-cv-demo:4f116503-6506-47d6-b427-1e7056a8c62e</a></td></tr>
    <tr><td>Helm Chart URL</td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/charts/edge-cv-demo'>ghcr.io/wallaroolabs/doc-samples/charts/edge-cv-demo</a></td></tr>
    <tr><td>Helm Chart Reference</td><td>ghcr.io/wallaroolabs/doc-samples/charts@sha256:411901e66f7b00b9ce16f32a5cd0d7277fd81fa88a443f70791ca2ff25e1c5a4</td></tr>
    <tr><td>Helm Chart Version</td><td>0.0.1-4f116503-6506-47d6-b427-1e7056a8c62e</td></tr>
    <tr><td>Engine Config</td><td>{'engine': {'resources': {'limits': {'cpu': 1.0, 'memory': '512Mi'}, 'requests': {'cpu': 1.0, 'memory': '512Mi'}}}, 'engineAux': {'images': {}}, 'enginelb': {'resources': {'limits': {'cpu': 1.0, 'memory': '512Mi'}, 'requests': {'cpu': 1.0, 'memory': '512Mi'}}}}</td></tr>
    <tr><td>User Images</td><td>[]</td></tr>
    <tr><td>Created By</td><td>john.hummel@wallaroo.ai</td></tr>
    <tr><td>Created At</td><td>2023-10-10 16:53:10.157330+00:00</td></tr>
    <tr><td>Updated At</td><td>2023-10-10 16:53:10.157330+00:00</td></tr>
</table>

### List Published Pipeline

The method `wallaroo.client.list_pipelines()` shows a list of all pipelines in the Wallaroo instance, and includes the `published` field that indicates whether the pipeline was published to the registry (`True`), or has not yet been published (`False`).

```python
wl.list_pipelines()
```

<table><tr><th>name</th><th>created</th><th>last_updated</th><th>deployed</th><th>tags</th><th>versions</th><th>steps</th><th>published</th></tr><tr><td>edge-cv-demo</td><td>2023-10-Oct 16:52:02</td><td>2023-10-Oct 16:53:08</td><td>False</td><td></td><td>4f116503-6506-47d6-b427-1e7056a8c62e, a91a8b17-f587-439f-9001-292892e94336, 5320bc9c-e64f-4bc1-ac97-7d2b40eeb53e</td><td>resnet-50</td><td>True</td></tr><tr><td>edge04yolo8n</td><td>2023-10-Oct 14:48:43</td><td>2023-10-Oct 16:35:13</td><td>False</td><td></td><td>9b40cc1b-af1c-4521-9354-4e33e4f9d9c5, b742ddbf-2c69-4c70-b59e-bb33a6f7979c, bab6e409-af82-4678-8ba7-0f0e49997529, 4812d72c-a0ca-4432-aa6d-8a12d9a7fd02</td><td>yolov8n</td><td>False</td></tr><tr><td>edge-pipeline-classification-cybersecurity</td><td>2023-10-Oct 14:36:15</td><td>2023-10-Oct 14:37:08</td><td>False</td><td></td><td>fe99cad9-dc32-4846-bcbc-27de68975784, b53618b7-191e-44cb-b38d-bbfd9ffc7748, e1a9f56c-17f5-45f8-86bf-69ebf6c446aa</td><td>aloha</td><td>True</td></tr><tr><td>hf-summarizer</td><td>2023-05-Oct 16:31:44</td><td>2023-05-Oct 20:24:57</td><td>False</td><td></td><td>6c591132-5ba7-413d-87a6-f4221ef972a6, 60bb46b0-52b8-464a-a379-299db4ea26c0, c4c1213a-6b6e-4a98-b397-c7903e8faae4, 25ef3557-d73b-4e8b-874e-1e126693eff8, cc4bd9e0-b661-48c9-a0a9-29dafddeedcb, d359aafc-843d-4e32-9439-e365b8095d65, 8bd92035-2894-4fe4-8112-f5f3512dc8ea</td><td>hf-summarizer</td><td>True</td></tr><tr><td>houseprice-estimator</td><td>2023-03-Oct 18:07:05</td><td>2023-03-Oct 18:10:31</td><td>False</td><td></td><td>eac1e82a-e5c0-4f4b-a7fc-6583719f4a62, be1fc3f0-a769-4ce9-94e1-ba01898d91eb, 9007b5ba-d8a8-4cbe-aef7-2e9b24ee010a, d683431f-4074-4ba1-9d27-71361bd4ffd2, aaa216e0-94af-4173-b52a-b9d7c8118f17</td><td>house-price-prime</td><td>True</td></tr><tr><td>cv-mitochondria</td><td>2023-28-Sep 20:25:17</td><td>2023-29-Sep 19:26:53</td><td>False</td><td></td><td>63f71352-93bc-4e4a-85f6-0a0bf603124c, d271be10-fadd-4408-97aa-c57e6ae4e35a, ac4bd826-f47f-48b7-8319-01fbc6622899, 4b6dab7d-b3ff-4f14-8425-7d9b6de76637, 66e72bc0-a3e3-4872-bc20-19b992c194b4, cf4bdfb4-1eec-46f8-9af4-b16dea894de6, 507cb1eb-8034-4b5b-bc96-2427730a6407, 50ed1d2f-6dba-411c-9579-1090791b33bd, 6208c9cf-fcd6-4b20-bbfc-f6ce714596e3</td><td>mitochondria-detector</td><td>True</td></tr><tr><td>retailimage</td><td>2023-28-Sep 19:44:33</td><td>2023-28-Sep 19:54:59</td><td>False</td><td></td><td>26afe601-6515-48ca-9a37-d063ab1e1ea2, 1d806c89-ecc6-4207-b98f-c56eccd16c43, 11835eda-4e10-49c0-baab-63862c16d1ef, 57bf2bfb-009b-42b9-b926-742f8bbb8d3c, 891fe58d-902b-49bd-94d3-c2196a8efd3b, db0d489b-d8fa-41d3-b46f-a9623b28e336, f039eaf3-d0dd-4ab7-a767-852db5241ff0, 2f5cd92d-ecc8-4e75-aee5-1605c1f23f0e</td><td>v5s6</td><td>False</td></tr><tr><td>retailimage</td><td>2023-28-Sep 18:55:14</td><td>2023-28-Sep 19:23:05</td><td>True</td><td></td><td>d64dabed-7f7a-4f41-a307-e7995d7b8144, 8d257d18-2ca1-46b9-a40e-1f3d7f308dc1, e84586a7-05bb-4d67-a696-f04e80df8b58, 95c2157a-2722-4a5b-b564-d3a709c6238f, fa351ab0-fe77-4fc0-b521-ba15e92a91d7</td><td>v5s6</td><td>False</td></tr><tr><td>cv-yolo</td><td>2023-28-Sep 16:07:29</td><td>2023-28-Sep 18:47:35</td><td>True</td><td></td><td>5f889757-89c5-4475-a579-937639779ab3, f9981617-7734-4f2d-905a-62333c600fe7, b21ac721-49e3-402d-b6c0-af139d51299a, 3f277cc7-351d-4d10-bdb2-c770c0dc1ac2</td><td>house-price-prime</td><td>False</td></tr><tr><td>houseprice-estimator</td><td>2023-27-Sep 16:51:15</td><td>2023-27-Sep 16:53:56</td><td>False</td><td></td><td>07cac6a2-140d-4a5e-b7ec-264f5fbf9dc3, bd389561-2c4f-492b-a82b-896cf76c2acf, 37bcce00-28d9-4d28-b637-33acf4021103, 146a3e4a-057b-4bd2-94f7-ebadc133df3d, 996a9877-142f-4934-aa4a-7696d3662297, a79802b5-42f4-4fb6-bd6b-3da560d39d73</td><td>house-price-prime</td><td>False</td></tr><tr><td>aloha-fraud-detector</td><td>2023-27-Sep 16:29:55</td><td>2023-27-Sep 18:28:05</td><td>False</td><td></td><td>e2a42011-d319-476f-bc32-9b6cccae4870, be15dcad-5a78-4493-b568-ee4502fa1791, b74a8b3a-8128-4356-a6ff-434c2b283cc8, 6d72feb7-76b5-4121-b401-9dfd4b978745, c22e3aa7-8efa-41c1-8844-cc4e7d1147c5, 739269a7-7890-4774-9597-fda5f80a3a6d, aa362e18-7f7e-4dc6-9069-3207e9d2f605, 79865932-5b89-4b2a-bfb1-cb9ebeb5125f, 4727b985-db36-44f7-a1a3-7f1886bbf894, 07cbfcae-1fa2-4746-b585-55349d230b45, 03824313-6bbb-4ccd-95ea-64340f789b9c, 9ce54998-a667-43b3-8198-b2d95e0d2879, 8a416842-5675-455a-b638-29fe7dbb5ba1</td><td>aloha-prime</td><td>True</td></tr><tr><td>cv-arm-edge</td><td>2023-27-Sep 15:20:15</td><td>2023-27-Sep 15:20:15</td><td>(unknown)</td><td></td><td>86dd133a-c12f-478b-af9a-30a7e4850fc4</td><td></td><td>True</td></tr><tr><td>cv-arm-edge</td><td>2023-27-Sep 15:17:45</td><td>2023-27-Sep 15:17:45</td><td>(unknown)</td><td></td><td>97a92779-0a5d-4c2b-bcf1-7afd60ac83d5</td><td></td><td>False</td></tr></table>

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

<table><tr><th>id</th><th>pipeline_version_name</th><th>engine_url</th><th>pipeline_url</th><th>created_by</th><th>created_at</th><th>updated_at</th></tr><tr><td>9</td><td>4f116503-6506-47d6-b427-1e7056a8c62e</td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/engines/proxy/wallaroo/ghcr.io/wallaroolabs/standalone-mini:v2023.3.0-3854'>ghcr.io/wallaroolabs/doc-samples/engines/proxy/wallaroo/ghcr.io/wallaroolabs/standalone-mini:v2023.3.0-3854</a></td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/pipelines/edge-cv-demo:4f116503-6506-47d6-b427-1e7056a8c62e'>ghcr.io/wallaroolabs/doc-samples/pipelines/edge-cv-demo:4f116503-6506-47d6-b427-1e7056a8c62e</a></td><td>john.hummel@wallaroo.ai</td><td>2023-10-Oct 16:53:10</td><td>2023-10-Oct 16:53:10</td></tr></table>

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

Using our sample environment, here's sample deployment using Docker with a computer vision ML model, the same used in the [Wallaroo Use Case Tutorials Computer Vision: Retail](https://docs.wallaroo.ai/wallaroo-use-case-tutorials/wallaroo-use-case-computer-vision/use-case-computer-vision-retail/) tutorials.

For `docker run` commands, the persistent volume for storing session data is stored with `-v ./data:/persist`.  Updated as required for your deployments.

```bash
docker run -p 8080:8080 \
    -v ./data:/persist \
    -e DEBUG=true -e OCI_REGISTRY={your registry server} \
    -e CONFIG_CPUS=4 \
    -e OCI_USERNAME=oauth2accesstoken \
    -e OCI_PASSWORD={registry token here} \
    -e PIPELINE_URL={your registry server}/pipelines/edge-cv-retail:bf70eaf7-8c11-4b46-b751-916a43b1a555 \
    {your registry server}/engine:v2023.3.0-main-3707
```

### Docker Compose Deployment

For users who prefer to use `docker compose`, the following sample `compose.yaml` file is used to launch the Wallaroo Edge pipeline.  This is the same used in the [Wallaroo Use Case Tutorials Computer Vision: Retail](https://docs.wallaroo.ai/wallaroo-use-case-tutorials/wallaroo-use-case-computer-vision/use-case-computer-vision-retail/) tutorials.  The session and other data is stored with the `volumes` entry to add a persistent volume.

```yml
services:
  engine:
    image: {Your Engine URL}
    ports:
      - 8080:8080
    volumes:
      - ./data:/persist
    environment:
      PIPELINE_URL: {Your Pipeline URL}
      OCI_REGISTRY: {Your Edge Registry URL}
      OCI_USERNAME:  {Your Registry Username}
      OCI_PASSWORD: {Your Token or Password}
      CONFIG_CPUS: 1
```

For example:

```yml
services:
  engine:
    image: sample-registry.com/engine:v2023.3.0-main-3707
    ports:
      - 8080:8080
    volumes:
      - ./data:/persist
    environment:
      PIPELINE_URL: sample-registry.com/pipelines/edge-cv-retail:bf70eaf7-8c11-4b46-b751-916a43b1a555
      OCI_REGISTRY: sample-registry.com
      OCI_USERNAME:  _json_key_base64
      OCI_PASSWORD: abc123
      CONFIG_CPUS: 1
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
    volumes:
      - ./data:/persist
    environment:
      PIPELINE_URL: {pub.pipeline_url}
      OCI_USERNAME: YOUR USERNAME 
      OCI_PASSWORD: YOUR PASSWORD OR TOKEN
      OCI_REGISTRY: ghcr.io
      CONFIG_CPUS: 4
'''

print(docker_compose)
```

```python
docker_deploy = f'''
docker run -p 8080:8080 \\
    -e OCI_REGISTRY=$REGISTRYURL \\
    -e CONFIG_CPUS=4 \\
    -e OCI_USERNAME=$REGISTRYUSERNAME \\
    -e OCI_PASSWORD=$REGISTRYPASSWORD \\
    -e PIPELINE_URL={pub.pipeline_url} \\
    {pub.engine_url}
'''

print(docker_deploy)
```

    
    docker run -p 8080:8080 \
        -e OCI_REGISTRY=$REGISTRYURL \
        -e CONFIG_CPUS=4 \
        -e OCI_USERNAME=$REGISTRYUSERNAME \
        -e OCI_PASSWORD=$REGISTRYPASSWORD \
        -e PIPELINE_URL=ghcr.io/wallaroolabs/doc-samples/pipelines/edge-cv-demo:4f116503-6506-47d6-b427-1e7056a8c62e \
        ghcr.io/wallaroolabs/doc-samples/engines/proxy/wallaroo/ghcr.io/wallaroolabs/standalone-mini:v2023.3.0-3854
    

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
!curl localhost:8080/pipelines
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
!curl localhost:8080/models
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

* **check_failures** (*List[Integer]*): Whether any validation checks were triggered.  For more information, see [Wallaroo SDK Essentials Guide: Pipeline Management: Anomaly Testing](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline/#anomaly-testing).
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
!curl -X POST localhost:8080/pipelines/edge-cv-demo \
    -H "Content-Type: application/vnd.apache.arrow.file" \
        --data-binary @./data/image_224x224.arrow
```

    [{"check_failures":[],"elapsed":[6414875,452202208],"model_name":"resnet-50","model_version":"2e05e1d0-fcb3-4213-bba8-4bac13f53e8d","original_data":null,"outputs":[{"Int64":{"data":[535],"dim":[1],"v":1}},{"Float":{"data":[0.00009498585131950676,0.00009141524060396478,0.00046068374649621546,0.00007667177851544693,0.00008047104347497225,0.00006355856021400541,0.00017580816347617656,0.000014166347682476044,0.00004344095941632986,0.000042251358536304906,0.00025400498998351395,0.005299815908074379,0.0001666695170570165,0.00019031290139537305,0.0002084688749164343,0.00014618523709941655,0.00034408163628540933,0.0008281365735456347,0.00011978298425674438,0.0002062775456579402,0.00014886555436532944,0.00026070952299050987,0.0009008666384033859,0.001475491444580257,0.0008267512894235551,0.0003027648199349642,0.00019366369815543294,0.0005283929058350623,0.00014922766422387213,0.00024121809110511094,0.00041593576315790415,0.000036156357964500785,0.0002411209134152159,0.00016002357006072998,0.00012632398284040391,0.0002650072274263948,0.00007694320083828643,0.00016679218970239162,0.00003494399425107986,0.0017466224962845445,0.00015653077571187168,0.00019110951689071953,0.00011712996638379991,0.0005060969851911068,0.0005225498462095857,0.00018326804274693131,0.00038459646748378873,0.00006664673855993897,0.0006007053307257593,0.00003104090865235776,0.00008364472159883007,0.0005029478925280273,0.00021304766414687037,0.0006510640378110111,0.00027023907750844955,0.00019516532483976334,0.00004173403067397885,0.00018414674559608102,0.00005605360638583079,0.00015310128219425678,0.0029485123232007027,0.0001000285628833808,0.00008645150228403509,0.00037124930531717837,0.001045612501911819,0.0016248042229562998,0.00036552795791067183,0.0005579001153819263,0.00011697378795361146,0.0004602445987984538,0.00009498830331722274,0.00018202982028014958,0.00039832599577493966,0.00015528367657680064,0.002486653858795762,0.0001415333681507036,0.00041259071440435946,0.00010047078103525564,0.0003815478412434459,0.0011400374351069331,0.0008210586383938789,0.0014701721956953406,0.00042187661165371537,0.000326898560160771,0.0004138693038839847,0.00009274573676520959,0.0002862789260689169,0.0005950826453045011,0.000380972953280434,0.0003586443781387061,0.0007492825970984995,0.0002762995718512684,0.0003171586722601205,0.0016836452996358275,0.0003918888105545193,0.0003621992946136743,0.0004419932665769011,0.00020090065663680434,0.00010593134356895462,0.0009713731124065816,0.0005836604395881295,0.0002709808759391308,0.0000753044878365472,0.00006718681106576696,0.0016714693047106266,0.00012960519234184176,0.000024872353606042452,0.0001577017392264679,0.00009040465374710038,0.00006928551010787487,0.00004740866643260233,0.0002472156484145671,0.03141514211893082,0.00021109476801939309,0.0003071585379075259,0.000089318047685083,0.00011660818563541397,0.00016960615175776184,0.0006639184430241585,9.541419785819016e-6,0.00003597702016122639,0.00013113980821799487,0.0000388405860576313,0.00019295485981274396,0.000017446653146180324,0.00004296032784623094,0.00016763313033152372,0.0012729722075164318,0.0006275440682657063,0.007415026426315308,0.0009311999892815948,0.00008604201866546646,0.00011408660793676972,0.0005248249508440495,0.00016869328101165593,0.00007456648745574057,0.000543370726518333,0.00021447180188260972,0.0001688413613010198,0.0001395243889419362,0.00023141945712268353,0.0002781786024570465,0.0006277807406149805,0.0004487008845899254,0.0005733186844736338,0.0002501832786947489,0.0003850561333820224,0.000927681743633002,0.00022452339180745184,0.00022443861234933138,0.0004369820235297084,0.0002392719907220453,0.00036795015330426395,0.0003524413623381406,0.0007338618743233383,0.00007371178799076006,0.00012836323003284633,0.0002193951077060774,0.00017031333118211478,0.00033735792385414243,0.00020038172078784555,0.00015004878514446318,0.00015647031250409782,0.00011027724394807592,0.0008106993627734482,0.0001792437833501026,0.0001285144389839843,0.0005347997066564858,0.0002704208018258214,0.0003134773869533092,0.00008673530101077631,0.00012084541231160983,0.0002156938862754032,0.00009197117469739169,0.000387817359296605,0.00014786732208449394,0.00022733309015166014,0.00007891462882980704,0.00018402237037662417,0.00039511482464149594,0.00003392618236830458,0.00007368085789494216,0.0002497009700164199,0.00008823491225484759,0.00034549631527625024,0.00015110144158825278,0.000055336295190500095,0.00008519102266291156,0.00021453633962664753,0.0007178247324191034,0.00013769637735094875,0.0018266914412379265,0.00017284350178670138,0.00006754085188731551,0.00015001594147179276,0.0007804536144249141,0.00024013758229557425,0.00008981376595329493,0.00010351322271162644,0.00016653114289510995,0.00022738338157068938,0.00017854890029411763,0.00005073372085462324,0.00022161376546137035,0.0000838545456645079,0.00007755455590086058,0.00006896460399730131,0.00021314439072739333,0.000157011512783356,0.00017486774595454335,0.00007180216925917193,0.00022100601927377284,0.00024575585848651826,0.0001613835629541427,0.00007672945503145456,0.0001535127667011693,0.00011571094364626333,0.00036234650178812444,0.0002409559820080176,0.00004660481135942973,0.0000769213802414015,0.0004624854773283005,0.0002519803529139608,0.00022919464390724897,0.00017196632688865066,0.00012929704098496586,0.00012269520084373653,0.0001331326930085197,0.00011138556874357164,0.00023298172163777053,0.00010513138840906322,0.000028025238862028345,0.00007480078056687489,0.0000871905212989077,0.00006468428910011426,0.00008809878636384383,0.00027453619986772537,0.00017401167133357376,0.00023895308549981564,0.0003651934675872326,0.0002597842540126294,0.00010121529339812696,0.00012700709339696914,0.00008325917588081211,0.0001404377690050751,0.00006267144635785371,0.00021363687119446695,0.00014604235184378922,0.00015019359125290066,0.00011775942402891815,0.000262302637565881,0.0002317928447155282,0.0003367597237229347,0.0002597309066914022,0.00016997203056234866,0.0002002767432713881,0.00009662853699410334,0.00008577309199608862,0.0002053053758572787,0.000417130853747949,0.00022278739197645336,0.00027220562333241105,0.00010114109318237752,0.0002827950520440936,0.00010052191646536812,0.0002014678029809147,0.00040746803279034793,0.00011438108049333096,0.00007723410817561671,0.0001851117267506197,0.00010349575313739479,0.00026372686261311173,0.00016025020158849657,0.00008702948980499059,0.00013277710240799934,0.0003259495133534074,0.0001399417669745162,0.00017213821411132812,0.00016190539463423193,0.00014610629295930266,0.00031150857103057206,0.00015091047680471092,0.0003469668445177376,0.0007292923983186483,0.0021730384323745966,0.0007040736963972449,0.0005063159042038023,0.000057511650084052235,0.000047521378292003646,0.000139860509079881,0.0000785795709816739,0.00009013566887006164,0.00017434645269531757,0.00008407326822634786,0.000375990173779428,0.00012815413356292993,0.00005326463360688649,0.00015554105630144477,0.00024701684014871716,0.00015018250269349664,0.000394125614548102,0.00014846269914414734,0.000145152720506303,0.0002257339801872149,0.0004045003151986748,0.00010239940456813201,0.0000411884393543005,0.0005723058711737394,0.00044989315210841596,0.0002633366675581783,0.00012577150482684374,0.0007940830546431243,0.0001907138794194907,0.0004915733588859439,0.00016221952682826668,0.0008366872789338231,0.000623460509814322,0.0001892975124064833,0.00009834139927988872,0.001120548346079886,0.001521500525996089,0.0005397515487857163,0.00004581294706440531,0.0000957711527007632,0.00008625279588159174,0.00010854311403818429,0.00007731885852990672,0.00018301130330655724,0.00008435355994151905,0.00009233105083694682,0.00007363688928307965,0.0001242221478605643,0.0006924364133737981,0.0003297907824162394,0.0000554430007468909,0.0000633888557786122,0.00028664484852924943,0.00010128327267011628,0.00016046804375946522,0.00006382977881003171,0.00009259011130779982,0.0001224709121743217,0.00019537295156624168,0.00004244926094543189,0.000028542712243506685,0.00023056945065036416,0.00011885943240486085,0.00006731662142556161,0.00005814672840642743,0.00013051609857939184,0.00015240356151480228,0.00044564830022864044,0.0007350510568358004,0.00011247355723753572,0.00019145703117828816,0.00025515907327644527,0.00006858190317871049,0.0005027491133660078,0.00020971505728084594,0.0006290936144068837,0.0003259263758081943,0.000059350622905185446,0.0002843360707629472,0.0001405030197929591,0.00010275053500663489,0.00010872587154153734,0.0002856853243429214,0.00004806696597370319,0.00016030789993237704,0.00013734234380535781,0.00014072957856114954,0.00018133330740965903,0.0003597433096729219,0.000033420099498471245,0.000207008866709657,0.00017835674225352705,0.00012609986879397184,0.000350618502125144,0.0002865635324269533,0.0002532572252675891,0.00006481764285126701,0.00011849743168568239,0.00004155319038545713,0.00017081368423532695,0.00022553815506398678,0.00022503036598209292,0.00009315248462371528,0.00031398903229273856,0.00010199478856520727,0.00007633001951035112,0.0005536979297176003,0.00005425699055194855,0.00002068760477413889,0.0002376270858803764,0.00006069021037546918,0.000040196267946157604,0.0001737958227749914,0.000033400123356841505,0.00008545352466171607,0.00025446407380513847,0.00206824135966599,0.0005671331891790032,0.00011009508307324722,0.0011866017011925578,0.00031426778878085315,0.00041002867510542274,0.0007057274342514575,0.00012301235983613878,0.00008512791828252375,0.00003627915066317655,0.0011110326740890741,0.000044761782191926613,0.000359308352926746,0.0014156022807583213,0.004740038421005011,0.00030492403311654925,0.00001728143251966685,0.0031153562013059855,0.0005614557303488255,0.005638706963509321,0.0014335708692669868,0.0010917945764958858,0.00038338889135047793,0.00022159959189593792,0.0004050697316415608,0.00007660026312805712,0.00005639284790959209,0.0008680580649524927,0.0010005293879657984,0.00012364985013846308,0.000807583041023463,0.000135080874315463,0.00011817162157967687,0.006565399002283812,0.000302396307233721,0.00021670977002941072,0.0005258043529465795,0.00018677933258004487,0.0005671480903401971,0.002048808615654707,0.00041426735697314143,0.0004681935824919492,0.001288025756366551,0.00017147396283689886,0.0006056927959434688,0.00006368011963786557,0.0003711339086294174,0.020963728427886963,0.0005652746185660362,0.00021039314742665738,0.00014829054998699576,0.00002862020664906595,0.0256099384278059,0.00013037298049312085,0.00010839565948117524,0.000019789567886618897,0.00022670361795462668,0.007652466185390949,0.00018585634825285524,0.000045377684728009626,0.0016958210617303848,0.000063116051023826,0.00035540215321816504,0.011162667535245419,0.0006776305963285267,0.0007829952519387007,0.0032530068419873714,0.0001423063949914649,0.000038761932955821976,0.00005264277569949627,0.0004918648046441376,0.0007090235012583435,0.00013544321700464934,0.00006474387919297442,0.017759323120117188,0.0001364814379485324,0.00007206340524135157,0.00003071592436754145,0.0001699163403827697,0.0007694592350162566,0.00004094452742720023,0.0004299794090911746,0.000691831752192229,0.0004279576241970062,0.00013007050438318402,0.00023491699539590627,0.0006906801136210561,0.00032300103339366615,0.004702153615653515,0.0022613792680203915,0.000011289695976302028,0.0004111242888029665,0.0030112951062619686,0.0005294233560562134,0.0006860735593363643,0.0008129157358780503,0.0002520101552363485,0.00027255096938461065,0.00016081167268566787,0.00004076233017258346,0.021430160850286484,0.00015415705274790525,0.0005084978183731437,0.0007915114401839674,0.011835964396595955,0.0009027221240103245,0.0005889891763217747,0.0002848078729584813,0.002500004367902875,0.0003199880593456328,0.000029753964554402046,0.0004997227806597948,0.00007795571582391858,0.008496510796248913,0.000675845134537667,0.0010315539548173547,0.0002851853787433356,0.0005604341858997941,0.0007420529727824032,0.0002665843931026757,0.0006702560931444168,0.000549429387319833,0.0004862656642217189,0.0008852946339175105,0.001020865049213171,0.0005504822474904358,0.0000664319668430835,0.0001670729398028925,0.0018395151710137725,0.0002079288096865639,0.0006724245613440871,0.0017224918119609356,0.0004128347209189087,0.000713974644895643,0.00011113999062217772,0.033084768801927567,0.0001019916744553484,0.00008023356349440292,0.000049806061724666506,0.00042539960122667253,0.0007217634120024741,0.001107925665564835,0.0006939568556845188,0.0013572219759225845,0.0015684913378208876,0.00006747209408786148,0.001139119383879006,0.001004121731966734,0.0000704674021108076,0.0002460831601638347,0.008800562471151352,0.0003596664173528552,0.0034511748235672712,0.00009517150465399027,0.000866050599142909,0.00023326485825236887,0.00008802555385045707,0.0038720942102372646,0.00021330187155399472,0.0025237652007490396,0.0023425943218171597,0.00009528757800580934,0.00006937788566574454,0.00007887237734394148,0.0009900141740217805,0.00008783435623627156,0.00006110726098995656,0.00017205893527716398,0.002612625015899539,0.00008416660421062261,0.00010105229011969641,0.0004147964937146753,0.00007185470894910395,0.0010353205725550652,0.00008421490201726556,0.0026522227562963963,0.00005155753387953155,0.00011890972382389009,0.0008946839952841401,0.0015078018186613917,0.0002126411854987964,0.00003852439840557054,0.000056374781706836075,0.000014754578842257615,0.008194835856556892,0.025179803371429443,0.0033814297057688236,0.00012250257714185864,0.0015897982520982623,0.0013646928127855062,0.0037181153893470764,0.00047879572957754135,0.0005974830128252506,0.0009444896131753922,0.004496521782130003,0.0003005625621881336,0.00009878807759378105,0.0198550745844841,0.00028703181305900216,0.00031890367972664535,0.0004631343181245029,0.0059915948659181595,0.001109347678720951,0.0007891662535257638,0.00005801041697850451,0.004181322641670704,0.0008145736064761877,0.00113387918099761,0.00026295933639630675,0.00034924206556752324,0.00006442249286919832,0.0027946685440838337,0.0002669223758857697,0.00010071286669699475,0.002296909224241972,0.0004214771033730358,0.0034138686023652554,0.003913606982678175,0.00034663925180211663,0.0024512442760169506,0.0033696044702082872,0.0005183485918678343,0.0007979461224749684,0.0004219271067995578,0.012866660952568054,0.000034880300518125296,0.0002607619680929929,0.005895871669054031,0.00017728608509059995,0.0002143714955309406,0.0031323006842285395,0.0006888844072818756,0.003136160783469677,0.001896119792945683,0.0009908534120768309,0.00003762159394682385,0.002616092562675476,0.0007033292786218226,0.0001495486358180642,0.0006599962362088263,0.00013283650332596153,0.00010756878327811137,0.005778242368251085,0.0004715237591881305,0.00035480473889037967,0.0031889358069747686,0.0000644748579361476,0.00023209610662888736,0.0012023727176710963,0.0008255299180746078,0.00007059722702251747,0.0016246120212599635,0.0007324084872379899,0.00007100576476659626,0.001435423269867897,0.00003152314820908941,0.00010502280929358676,0.000031699037208454683,0.0007149871671572328,0.00004233977597323246,0.00010430767724756151,0.00013821203901898116,0.00001739500294206664,0.0042591881938278675,0.00011460929090389982,0.0010657146340236068,0.00011586498294491321,0.0008855802589096129,0.00026814587181434035,0.00014032117906026542,0.00011272418487351388,0.0002642087929416448,0.00009847767069004476,0.0003047628269996494,0.00023018188949208707,0.0015937925782054663,0.0006807688623666763,0.0024635361041873693,0.010918836109340191,0.0011293742572888732,0.0005581068689934909,0.002954506315290928,0.0011891447938978672,0.0001571461179992184,0.008346425369381905,0.002323271008208394,0.00004773572800331749,0.009146460331976414,0.00004678942423197441,0.00027197026065550745,0.000360426667612046,0.00002783290619845502,0.00020616388064809144,0.0002696973388083279,0.001089491997845471,0.000122932848171331,0.0004029596457257867,0.005027643404901028,0.00028851156821474433,0.00007667400495847687,0.005518380086869001,0.0008715178701095283,0.003209100104868412,0.0007127526332624257,0.00008145140600390732,0.00006068711445550434,0.000023346281523117796,0.00008884121052687988,0.00019424821948632598,0.00454985536634922,0.0001445038360543549,0.0003629894636105746,0.00017611072689760476,0.0006488841609098017,0.0018402438145130873,0.0021717161871492863,0.002714048605412245,0.00018598065071273595,0.00009417088585905731,0.0004467363760340959,0.0010807417565956712,0.0004274447273928672,0.00004169758904026821,0.001155734178610146,0.0006934652919881046,0.00019714429799932986,0.00021102061145938933,0.002551938407123089,0.00016984384274110198,0.0001468267000745982,0.00481033930554986,0.00014269480016082525,0.016139870509505272,0.0002797374618239701,0.0024672113358974457,0.00003682959504658356,0.00006452422530855983,0.0007991478778421879,0.0000822842339402996,0.00018033746164292097,0.0002582163142506033,0.0032537321094423532,0.0005689522949978709,0.0013160079251974821,0.0001725064794300124,0.001440109801478684,0.0010854781139642,0.0002853967889677733,0.004279119893908501,0.0006065535126253963,0.004105324856936932,0.0006992792477831244,0.00013334539835341275,0.0003922378527931869,0.0036946169566363096,0.0008572319056838751,0.00013983616372570395,0.0004357346333563328,0.00003284416379756294,0.0020295889116823673,0.0006471322849392891,0.001588250626809895,0.0011088656028732657,0.000034527496609371156,0.010173124261200428,0.0014297745656222105,0.0003852693480439484,0.0004752578097395599,0.0009320020326413214,0.00013027280510868877,0.005989773664623499,0.0006256482447497547,0.0022420703899115324,0.007279570680111647,0.00024401459086220711,0.0007026211242191494,0.00020484822744037956,0.0018617683090269566,0.007344024255871773,0.0010820794850587845,0.00008027628064155579,0.0005128595512360334,0.0003133092832285911,0.0016340103466063738,0.006782020907849073,0.006025396287441254,0.00002035178658843506,0.00017984058649744838,0.0002768124104477465,0.000034267723094671965,0.0018119999440386891,0.0006264462135732174,0.0002733362780418247,0.002132176887243986,0.0006001784931868315,0.00045349044376052916,0.0009496446582488716,0.00038760382449254394,0.00036731240106746554,0.0017638689605519176,0.000659328477922827,0.0001579321688041091,0.00017143611330538988,0.00020174506062176079,0.00020106318697798997,0.003706882940605283,0.00018330868624616414,0.00038726223283447325,0.0004440330667421222,0.0002052708441624418,0.000026126903321710415,0.0012235178146511316,0.0017890010494738817,0.0017822252120822668,0.028902767226099968,0.0004932042211294174,0.00021147046936675906,0.00009399654663866386,0.0002530484925955534,0.006250547245144844,0.00004995004928787239,0.000037343113945098594,0.0001774896081769839,0.00010433235001983121,0.0010067167459055781,0.00011981611169176176,0.00006331668555503711,0.0038408495020121336,0.000972631445620209,0.0019491915591061115,0.000050321097660344094,0.0006725863204337656,0.000916636548936367,0.00015008334594313055,0.0004722609301097691,0.0003718055959325284,0.00010026848030975088,0.0005544783780351281,0.00040761378477327526,0.0014444667613133788,0.00019206308934371918,0.0030017292592674494,0.0005599699215963483,0.000501491129398346,0.00002119926102750469,0.004090329632163048,0.0006480138981714845,0.006801491137593985,0.000034706925362115726,0.0008900891407392919,0.00030399792012758553,0.0000790477279224433,0.0031371114309877157,0.004015177953988314,0.00006729082087986171,0.002737324917688966,0.0013298087287694216,0.00014384482346940786,0.0004044845118187368,0.00025029698736034334,0.003962397109717131,0.00020060477254446596,0.002549479017034173,0.0005042953416705132,0.00015140716277528554,0.000023598389816470444,0.00006094616765039973,0.00008477000665152445,0.00008080676343524829,0.00032072176691144705,0.0003799535916186869,0.0001776174467522651,0.00021202428615652025,0.0038306480273604393,0.00010710977949202061,0.000025333885787404142,0.0014211578527465463,0.0004166108265053481,0.0008034216589294374,0.00020861027587670833,0.00045565151958726346,0.00019095839525107294,0.0009474421385675669,0.0003390771453268826,0.003743864828720689,0.0002046173467533663,0.0023784483782947063,0.0002782591327559203,0.00011422019451856613,0.00009886993211694062,0.0008324605878442526,0.00010915552411461249,0.0014770242851227522,0.004380023572593927,0.0033058125991374254,0.0004904121160507202,0.0009391116909682751,0.004294251557439566,0.004433758091181517,0.0011093614157289267,0.0014735684962943196,0.0001349222584394738,0.00318022258579731,0.009836940094828606,0.0001608944294275716,0.0014605475589632988,0.0011098711984232068,0.0029562783893197775,0.0007235890370793641,0.00012282555690035224,0.00017606512119527906,0.000432351982453838,0.0003031663072761148,0.00014125801681075245,0.00024728660355322063,0.0004931026487611234,0.000045680491894017905,0.005605900660157204,0.00011934831854887307,0.00020548295287881047,0.0005088415346108377,0.0003000329015776515,0.0022189978044480085,0.000086846666818019,0.00015559824532829225,0.000030328295906656422,0.00020001412485726178,0.00016394634440075606,0.0001851861597970128,0.00005766171670984477,0.00002346670771657955,0.0006521676550619304,0.00019569967116694897,0.000035890028811991215,0.000045427757868310437,0.000029700055165449157,0.00017792286234907806,0.00010494551679585129,0.00008614900434622541,0.0000591857751714997,0.00008496202644892037,0.00031928057433106005,0.000058348454331280664,0.0003249863802921027,0.00014573334192391485,0.00009598094766261056,0.0003437568957451731,0.00016984311514534056,0.00004552475729724392,0.00015786508447490633,0.0004491915460675955,0.00008291342965094373,0.00017383552039973438,0.00019677005184348673,0.00017128413310274482,0.0005099810077808797,0.00004297649866202846,0.00023316977603826672,0.0000635220785625279,0.00010151477181352675,0.00005139570566825569,0.00033004439319483936,0.00014979335537645966,0.00007846357766538858,0.000032075342460302636,0.00009726791904540733,0.00007835646101739258,0.0009405241580680013,0.0006174164009280503,0.0004556083003990352,0.0010927317198365927,0.0002573138626758009,0.000256074097706005,0.00008612103556515649,0.000056629276514286175,0.0006073255208320916,0.00004119781078770757,0.0001904652308439836,0.0002691808622330427,0.00036230243858881295,0.00009973798296414316,0.00030845380388200283,0.00014764531806576997,0.000021887548427912407,0.00014684964844491333,0.0001275904942303896,0.0004005460359621793,0.00008555447129765525,0.00011063444253522903,0.000249848875682801,0.0001945443800650537,0.00034877043799497187,0.00007088106940500438,0.00003814319279626943,0.00012102608161512762,0.00023900196538306773,0.00008068174793152139,0.0001313362445216626,0.00016937193868216127,0.00012985063949599862,0.0007559973164461553],"dim":[1,1001],"v":1}}],"pipeline_name":"edge-cv-demo","shadow_data":{},"time":1696958898804}]

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

host = 'http://localhost:8080'

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
      <td>[1165083, 364076625]</td>
      <td>resnet-50</td>
      <td>2e05e1d0-fcb3-4213-bba8-4bac13f53e8d</td>
      <td>None</td>
      <td>[{'Int64': {'data': [535], 'dim': [1], 'v': 1}}, {'Float': {'data': [9.498585131950676e-05, 9.141524060396478e-05, 0.00046068374649621546, 7.667177851544693e-05, 8.047104347497225e-05, 6.355856021400541e-05, 0.00017580816347617656, 1.4166347682476044e-05, 4.344095941632986e-05, 4.2251358536304906e-05, 0.00025400498998351395, 0.005299815908074379, 0.0001666695170570165, 0.00019031290139537305, 0.0002084688749164343, 0.00014618523709941655, 0.00034408163628540933, 0.0008281365735456347, 0.00011978298425674438, 0.0002062775456579402, 0.00014886555436532944, 0.00026070952299050987, 0.0009008666384033859, 0.001475491444580257, 0.0008267512894235551, 0.0003027648199349642, 0.00019366369815543294, 0.0005283929058350623, 0.00014922766422387213, 0.00024121809110511094, 0.00041593576315790415, 3.6156357964500785e-05, 0.0002411209134152159, 0.00016002357006072998, 0.00012632398284040391, 0.0002650072274263948, 7.694320083828643e-05, 0.00016679218970239162, 3.494399425107986e-05, 0.0017466224962845445, 0.00015653077571187168, 0.00019110951689071953, 0.00011712996638379991, 0.0005060969851911068, 0.0005225498462095857, 0.00018326804274693131, 0.00038459646748378873, 6.664673855993897e-05, 0.0006007053307257593, 3.104090865235776e-05, 8.364472159883007e-05, 0.0005029478925280273, 0.00021304766414687037, 0.0006510640378110111, 0.00027023907750844955, 0.00019516532483976334, 4.173403067397885e-05, 0.00018414674559608102, 5.605360638583079e-05, 0.00015310128219425678, 0.0029485123232007027, 0.0001000285628833808, 8.645150228403509e-05, 0.00037124930531717837, 0.001045612501911819, 0.0016248042229562998, 0.00036552795791067183, 0.0005579001153819263, 0.00011697378795361146, 0.0004602445987984538, 9.498830331722274e-05, 0.00018202982028014958, 0.00039832599577493966, 0.00015528367657680064, 0.002486653858795762, 0.0001415333681507036, 0.00041259071440435946, 0.00010047078103525564, 0.0003815478412434459, 0.0011400374351069331, 0.0008210586383938789, 0.0014701721956953406, 0.00042187661165371537, 0.000326898560160771, 0.0004138693038839847, 9.274573676520959e-05, 0.0002862789260689169, 0.0005950826453045011, 0.000380972953280434, 0.0003586443781387061, 0.0007492825970984995, 0.0002762995718512684, 0.0003171586722601205, 0.0016836452996358275, 0.0003918888105545193, 0.0003621992946136743, 0.0004419932665769011, 0.00020090065663680434, 0.00010593134356895462, 0.0009713731124065816, 0.0005836604395881295, 0.0002709808759391308, 7.53044878365472e-05, 6.718681106576696e-05, 0.0016714693047106266, 0.00012960519234184176, 2.4872353606042452e-05, 0.0001577017392264679, 9.040465374710038e-05, 6.928551010787487e-05, 4.740866643260233e-05, 0.0002472156484145671, 0.03141514211893082, 0.00021109476801939309, 0.0003071585379075259, 8.9318047685083e-05, 0.00011660818563541397, 0.00016960615175776184, 0.0006639184430241585, 9.541419785819016e-06, 3.597702016122639e-05, 0.00013113980821799487, 3.88405860576313e-05, 0.00019295485981274396, 1.7446653146180324e-05, 4.296032784623094e-05, 0.00016763313033152372, 0.0012729722075164318, 0.0006275440682657063, 0.007415026426315308, 0.0009311999892815948, 8.604201866546646e-05, 0.00011408660793676972, 0.0005248249508440495, 0.00016869328101165593, 7.456648745574057e-05, 0.000543370726518333, 0.00021447180188260972, 0.0001688413613010198, 0.0001395243889419362, 0.00023141945712268353, 0.0002781786024570465, 0.0006277807406149805, 0.0004487008845899254, 0.0005733186844736338, 0.0002501832786947489, 0.0003850561333820224, 0.000927681743633002, 0.00022452339180745184, 0.00022443861234933138, 0.0004369820235297084, 0.0002392719907220453, 0.00036795015330426395, 0.0003524413623381406, 0.0007338618743233383, 7.371178799076006e-05, 0.00012836323003284633, 0.0002193951077060774, 0.00017031333118211478, 0.00033735792385414243, 0.00020038172078784555, 0.00015004878514446318, 0.00015647031250409782, 0.00011027724394807592, 0.0008106993627734482, 0.0001792437833501026, 0.0001285144389839843, 0.0005347997066564858, 0.0002704208018258214, 0.0003134773869533092, 8.673530101077631e-05, 0.00012084541231160983, 0.0002156938862754032, 9.197117469739169e-05, 0.000387817359296605, 0.00014786732208449394, 0.00022733309015166014, 7.891462882980704e-05, 0.00018402237037662417, 0.00039511482464149594, 3.392618236830458e-05, 7.368085789494216e-05, 0.0002497009700164199, 8.823491225484759e-05, 0.00034549631527625024, 0.00015110144158825278, 5.5336295190500095e-05, 8.519102266291156e-05, 0.00021453633962664753, 0.0007178247324191034, 0.00013769637735094875, 0.0018266914412379265, 0.00017284350178670138, 6.754085188731551e-05, 0.00015001594147179276, 0.0007804536144249141, 0.00024013758229557425, 8.981376595329493e-05, 0.00010351322271162644, 0.00016653114289510995, 0.00022738338157068938, 0.00017854890029411763, 5.073372085462324e-05, 0.00022161376546137035, 8.38545456645079e-05, 7.755455590086058e-05, 6.896460399730131e-05, 0.00021314439072739333, 0.000157011512783356, 0.00017486774595454335, 7.180216925917193e-05, 0.00022100601927377284, 0.00024575585848651826, 0.0001613835629541427, 7.672945503145456e-05, 0.0001535127667011693, 0.00011571094364626333, 0.00036234650178812444, 0.0002409559820080176, 4.660481135942973e-05, 7.69213802414015e-05, 0.0004624854773283005, 0.0002519803529139608, 0.00022919464390724897, 0.00017196632688865066, 0.00012929704098496586, 0.00012269520084373653, 0.0001331326930085197, 0.00011138556874357164, 0.00023298172163777053, 0.00010513138840906322, 2.8025238862028345e-05, 7.480078056687489e-05, 8.71905212989077e-05, 6.468428910011426e-05, 8.809878636384383e-05, 0.00027453619986772537, 0.00017401167133357376, 0.00023895308549981564, 0.0003651934675872326, 0.0002597842540126294, 0.00010121529339812696, 0.00012700709339696914, 8.325917588081211e-05, 0.0001404377690050751, 6.267144635785371e-05, 0.00021363687119446695, 0.00014604235184378922, 0.00015019359125290066, 0.00011775942402891815, 0.000262302637565881, 0.0002317928447155282, 0.0003367597237229347, 0.0002597309066914022, 0.00016997203056234866, 0.0002002767432713881, 9.662853699410334e-05, 8.577309199608862e-05, 0.0002053053758572787, 0.000417130853747949, 0.00022278739197645336, 0.00027220562333241105, 0.00010114109318237752, 0.0002827950520440936, 0.00010052191646536812, 0.0002014678029809147, 0.00040746803279034793, 0.00011438108049333096, 7.723410817561671e-05, 0.0001851117267506197, 0.00010349575313739479, 0.00026372686261311173, 0.00016025020158849657, 8.702948980499059e-05, 0.00013277710240799934, 0.0003259495133534074, 0.0001399417669745162, 0.00017213821411132812, 0.00016190539463423193, 0.00014610629295930266, 0.00031150857103057206, 0.00015091047680471092, 0.0003469668445177376, 0.0007292923983186483, 0.0021730384323745966, 0.0007040736963972449, 0.0005063159042038023, 5.7511650084052235e-05, 4.7521378292003646e-05, 0.000139860509079881, 7.85795709816739e-05, 9.013566887006164e-05, 0.00017434645269531757, 8.407326822634786e-05, 0.000375990173779428, 0.00012815413356292993, 5.326463360688649e-05, 0.00015554105630144477, 0.00024701684014871716, 0.00015018250269349664, 0.000394125614548102, 0.00014846269914414734, 0.000145152720506303, 0.0002257339801872149, 0.0004045003151986748, 0.00010239940456813201, 4.11884393543005e-05, 0.0005723058711737394, 0.00044989315210841596, 0.0002633366675581783, 0.00012577150482684374, 0.0007940830546431243, 0.0001907138794194907, 0.0004915733588859439, 0.00016221952682826668, 0.0008366872789338231, 0.000623460509814322, 0.0001892975124064833, 9.834139927988872e-05, 0.001120548346079886, 0.001521500525996089, 0.0005397515487857163, 4.581294706440531e-05, 9.57711527007632e-05, 8.625279588159174e-05, 0.00010854311403818429, 7.731885852990672e-05, 0.00018301130330655724, 8.435355994151905e-05, 9.233105083694682e-05, 7.363688928307965e-05, 0.0001242221478605643, 0.0006924364133737981, 0.0003297907824162394, 5.54430007468909e-05, 6.33888557786122e-05, 0.00028664484852924943, 0.00010128327267011628, 0.00016046804375946522, 6.382977881003171e-05, 9.259011130779982e-05, 0.0001224709121743217, 0.00019537295156624168, 4.244926094543189e-05, 2.8542712243506685e-05, 0.00023056945065036416, 0.00011885943240486085, 6.731662142556161e-05, 5.814672840642743e-05, 0.00013051609857939184, 0.00015240356151480228, 0.00044564830022864044, 0.0007350510568358004, 0.00011247355723753572, 0.00019145703117828816, 0.00025515907327644527, 6.858190317871049e-05, 0.0005027491133660078, 0.00020971505728084594, 0.0006290936144068837, 0.0003259263758081943, 5.9350622905185446e-05, 0.0002843360707629472, 0.0001405030197929591, 0.00010275053500663489, 0.00010872587154153734, 0.0002856853243429214, 4.806696597370319e-05, 0.00016030789993237704, 0.00013734234380535781, 0.00014072957856114954, 0.00018133330740965903, 0.0003597433096729219, 3.3420099498471245e-05, 0.000207008866709657, 0.00017835674225352705, 0.00012609986879397184, 0.000350618502125144, 0.0002865635324269533, 0.0002532572252675891, 6.481764285126701e-05, 0.00011849743168568239, 4.155319038545713e-05, 0.00017081368423532695, 0.00022553815506398678, 0.00022503036598209292, 9.315248462371528e-05, 0.00031398903229273856, 0.00010199478856520727, 7.633001951035112e-05, 0.0005536979297176003, 5.425699055194855e-05, 2.068760477413889e-05, 0.0002376270858803764, 6.069021037546918e-05, 4.0196267946157604e-05, 0.0001737958227749914, 3.3400123356841505e-05, 8.545352466171607e-05, 0.00025446407380513847, 0.00206824135966599, 0.0005671331891790032, 0.00011009508307324722, 0.0011866017011925578, 0.00031426778878085315, 0.00041002867510542274, 0.0007057274342514575, 0.00012301235983613878, 8.512791828252375e-05, 3.627915066317655e-05, 0.0011110326740890741, 4.4761782191926613e-05, 0.000359308352926746, 0.0014156022807583213, 0.004740038421005011, 0.00030492403311654925, 1.728143251966685e-05, 0.0031153562013059855, 0.0005614557303488255, 0.005638706963509321, 0.0014335708692669868, 0.0010917945764958858, 0.00038338889135047793, 0.00022159959189593792, 0.0004050697316415608, 7.660026312805712e-05, 5.639284790959209e-05, 0.0008680580649524927, 0.0010005293879657984, 0.00012364985013846308, 0.000807583041023463, 0.000135080874315463, 0.00011817162157967687, 0.006565399002283812, 0.000302396307233721, 0.00021670977002941072, 0.0005258043529465795, 0.00018677933258004487, 0.0005671480903401971, 0.002048808615654707, 0.00041426735697314143, 0.0004681935824919492, 0.001288025756366551, 0.00017147396283689886, 0.0006056927959434688, 6.368011963786557e-05, 0.0003711339086294174, 0.020963728427886963, 0.0005652746185660362, 0.00021039314742665738, 0.00014829054998699576, 2.862020664906595e-05, 0.0256099384278059, 0.00013037298049312085, 0.00010839565948117524, 1.9789567886618897e-05, 0.00022670361795462668, 0.007652466185390949, 0.00018585634825285524, 4.5377684728009626e-05, 0.0016958210617303848, 6.3116051023826e-05, 0.00035540215321816504, 0.011162667535245419, 0.0006776305963285267, 0.0007829952519387007, 0.0032530068419873714, 0.0001423063949914649, 3.8761932955821976e-05, 5.264277569949627e-05, 0.0004918648046441376, 0.0007090235012583435, 0.00013544321700464934, 6.474387919297442e-05, 0.017759323120117188, 0.0001364814379485324, 7.206340524135157e-05, 3.071592436754145e-05, 0.0001699163403827697, 0.0007694592350162566, 4.094452742720023e-05, 0.0004299794090911746, 0.000691831752192229, 0.0004279576241970062, 0.00013007050438318402, 0.00023491699539590627, 0.0006906801136210561, 0.00032300103339366615, 0.004702153615653515, 0.0022613792680203915, 1.1289695976302028e-05, 0.0004111242888029665, 0.0030112951062619686, 0.0005294233560562134, 0.0006860735593363643, 0.0008129157358780503, 0.0002520101552363485, 0.00027255096938461065, 0.00016081167268566787, 4.076233017258346e-05, 0.021430160850286484, 0.00015415705274790525, 0.0005084978183731437, 0.0007915114401839674, 0.011835964396595955, 0.0009027221240103245, 0.0005889891763217747, 0.0002848078729584813, 0.002500004367902875, 0.0003199880593456328, 2.9753964554402046e-05, 0.0004997227806597948, 7.795571582391858e-05, 0.008496510796248913, 0.000675845134537667, 0.0010315539548173547, 0.0002851853787433356, 0.0005604341858997941, 0.0007420529727824032, 0.0002665843931026757, 0.0006702560931444168, 0.000549429387319833, 0.0004862656642217189, 0.0008852946339175105, 0.001020865049213171, 0.0005504822474904358, 6.64319668430835e-05, 0.0001670729398028925, 0.0018395151710137725, 0.0002079288096865639, 0.0006724245613440871, 0.0017224918119609356, 0.0004128347209189087, 0.000713974644895643, 0.00011113999062217772, 0.033084768801927567, 0.0001019916744553484, 8.023356349440292e-05, 4.9806061724666506e-05, 0.00042539960122667253, 0.0007217634120024741, 0.001107925665564835, 0.0006939568556845188, 0.0013572219759225845, 0.0015684913378208876, 6.747209408786148e-05, 0.001139119383879006, 0.001004121731966734, 7.04674021108076e-05, 0.0002460831601638347, 0.008800562471151352, 0.0003596664173528552, 0.0034511748235672712, 9.517150465399027e-05, 0.000866050599142909, 0.00023326485825236887, 8.802555385045707e-05, 0.0038720942102372646, 0.00021330187155399472, 0.0025237652007490396, 0.0023425943218171597, 9.528757800580934e-05, 6.937788566574454e-05, 7.887237734394148e-05, 0.0009900141740217805, 8.783435623627156e-05, 6.110726098995656e-05, 0.00017205893527716398, 0.002612625015899539, 8.416660421062261e-05, 0.00010105229011969641, 0.0004147964937146753, 7.185470894910395e-05, 0.0010353205725550652, 8.421490201726556e-05, 0.0026522227562963963, 5.155753387953155e-05, 0.00011890972382389009, 0.0008946839952841401, 0.0015078018186613917, 0.0002126411854987964, 3.852439840557054e-05, 5.6374781706836075e-05, 1.4754578842257615e-05, 0.008194835856556892, 0.025179803371429443, 0.0033814297057688236, 0.00012250257714185864, 0.0015897982520982623, 0.0013646928127855062, 0.0037181153893470764, 0.00047879572957754135, 0.0005974830128252506, 0.0009444896131753922, 0.004496521782130003, 0.0003005625621881336, 9.878807759378105e-05, 0.0198550745844841, 0.00028703181305900216, 0.00031890367972664535, 0.0004631343181245029, 0.0059915948659181595, 0.001109347678720951, 0.0007891662535257638, 5.801041697850451e-05, 0.004181322641670704, 0.0008145736064761877, 0.00113387918099761, 0.00026295933639630675, 0.00034924206556752324, 6.442249286919832e-05, 0.0027946685440838337, 0.0002669223758857697, 0.00010071286669699475, 0.002296909224241972, 0.0004214771033730358, 0.0034138686023652554, 0.003913606982678175, 0.00034663925180211663, 0.0024512442760169506, 0.0033696044702082872, 0.0005183485918678343, 0.0007979461224749684, 0.0004219271067995578, 0.012866660952568054, 3.4880300518125296e-05, 0.0002607619680929929, 0.005895871669054031, 0.00017728608509059995, 0.0002143714955309406, 0.0031323006842285395, 0.0006888844072818756, 0.003136160783469677, 0.001896119792945683, 0.0009908534120768309, 3.762159394682385e-05, 0.002616092562675476, 0.0007033292786218226, 0.0001495486358180642, 0.0006599962362088263, 0.00013283650332596153, 0.00010756878327811137, 0.005778242368251085, 0.0004715237591881305, 0.00035480473889037967, 0.0031889358069747686, 6.44748579361476e-05, 0.00023209610662888736, 0.0012023727176710963, 0.0008255299180746078, 7.059722702251747e-05, 0.0016246120212599635, 0.0007324084872379899, 7.100576476659626e-05, 0.001435423269867897, 3.152314820908941e-05, 0.00010502280929358676, 3.1699037208454683e-05, 0.0007149871671572328, 4.233977597323246e-05, 0.00010430767724756151, 0.00013821203901898116, 1.739500294206664e-05, 0.0042591881938278675, 0.00011460929090389982, 0.0010657146340236068, 0.00011586498294491321, 0.0008855802589096129, 0.00026814587181434035, 0.00014032117906026542, 0.00011272418487351388, 0.0002642087929416448, 9.847767069004476e-05, 0.0003047628269996494, 0.00023018188949208707, 0.0015937925782054663, 0.0006807688623666763, 0.0024635361041873693, 0.010918836109340191, 0.0011293742572888732, 0.0005581068689934909, 0.002954506315290928, 0.0011891447938978672, 0.0001571461179992184, 0.008346425369381905, 0.002323271008208394, 4.773572800331749e-05, 0.009146460331976414, 4.678942423197441e-05, 0.00027197026065550745, 0.000360426667612046, 2.783290619845502e-05, 0.00020616388064809144, 0.0002696973388083279, 0.001089491997845471, 0.000122932848171331, 0.0004029596457257867, 0.005027643404901028, 0.00028851156821474433, 7.667400495847687e-05, 0.005518380086869001, 0.0008715178701095283, 0.003209100104868412, 0.0007127526332624257, 8.145140600390732e-05, 6.068711445550434e-05, 2.3346281523117796e-05, 8.884121052687988e-05, 0.00019424821948632598, 0.00454985536634922, 0.0001445038360543549, 0.0003629894636105746, 0.00017611072689760476, 0.0006488841609098017, 0.0018402438145130873, 0.0021717161871492863, 0.002714048605412245, 0.00018598065071273595, 9.417088585905731e-05, 0.0004467363760340959, 0.0010807417565956712, 0.0004274447273928672, 4.169758904026821e-05, 0.001155734178610146, 0.0006934652919881046, 0.00019714429799932986, 0.00021102061145938933, 0.002551938407123089, 0.00016984384274110198, 0.0001468267000745982, 0.00481033930554986, 0.00014269480016082525, 0.016139870509505272, 0.0002797374618239701, 0.0024672113358974457, 3.682959504658356e-05, 6.452422530855983e-05, 0.0007991478778421879, 8.22842339402996e-05, 0.00018033746164292097, 0.0002582163142506033, 0.0032537321094423532, 0.0005689522949978709, 0.0013160079251974821, 0.0001725064794300124, 0.001440109801478684, 0.0010854781139642, 0.0002853967889677733, 0.004279119893908501, 0.0006065535126253963, 0.004105324856936932, 0.0006992792477831244, 0.00013334539835341275, 0.0003922378527931869, 0.0036946169566363096, 0.0008572319056838751, 0.00013983616372570395, 0.0004357346333563328, 3.284416379756294e-05, 0.0020295889116823673, 0.0006471322849392891, 0.001588250626809895, 0.0011088656028732657, 3.4527496609371156e-05, 0.010173124261200428, 0.0014297745656222105, 0.0003852693480439484, 0.0004752578097395599, 0.0009320020326413214, 0.00013027280510868877, 0.005989773664623499, 0.0006256482447497547, 0.0022420703899115324, 0.007279570680111647, 0.00024401459086220711, 0.0007026211242191494, 0.00020484822744037956, 0.0018617683090269566, 0.007344024255871773, 0.0010820794850587845, 8.027628064155579e-05, 0.0005128595512360334, 0.0003133092832285911, 0.0016340103466063738, 0.006782020907849073, 0.006025396287441254, 2.035178658843506e-05, 0.00017984058649744838, 0.0002768124104477465, 3.4267723094671965e-05, 0.0018119999440386891, 0.0006264462135732174, 0.0002733362780418247, 0.002132176887243986, 0.0006001784931868315, 0.00045349044376052916, 0.0009496446582488716, 0.00038760382449254394, 0.00036731240106746554, 0.0017638689605519176, 0.000659328477922827, 0.0001579321688041091, 0.00017143611330538988, 0.00020174506062176079, 0.00020106318697798997, 0.003706882940605283, 0.00018330868624616414, 0.00038726223283447325, 0.0004440330667421222, 0.0002052708441624418, 2.6126903321710415e-05, 0.0012235178146511316, 0.0017890010494738817, 0.0017822252120822668, 0.028902767226099968, 0.0004932042211294174, 0.00021147046936675906, 9.399654663866386e-05, 0.0002530484925955534, 0.006250547245144844, 4.995004928787239e-05, 3.7343113945098594e-05, 0.0001774896081769839, 0.00010433235001983121, 0.0010067167459055781, 0.00011981611169176176, 6.331668555503711e-05, 0.0038408495020121336, 0.000972631445620209, 0.0019491915591061115, 5.0321097660344094e-05, 0.0006725863204337656, 0.000916636548936367, 0.00015008334594313055, 0.0004722609301097691, 0.0003718055959325284, 0.00010026848030975088, 0.0005544783780351281, 0.00040761378477327526, 0.0014444667613133788, 0.00019206308934371918, 0.0030017292592674494, 0.0005599699215963483, 0.000501491129398346, 2.119926102750469e-05, 0.004090329632163048, 0.0006480138981714845, 0.006801491137593985, 3.4706925362115726e-05, 0.0008900891407392919, 0.00030399792012758553, 7.90477279224433e-05, 0.0031371114309877157, 0.004015177953988314, 6.729082087986171e-05, 0.002737324917688966, 0.0013298087287694216, 0.00014384482346940786, 0.0004044845118187368, 0.00025029698736034334, 0.003962397109717131, 0.00020060477254446596, 0.002549479017034173, 0.0005042953416705132, 0.00015140716277528554, 2.3598389816470444e-05, 6.094616765039973e-05, 8.477000665152445e-05, 8.080676343524829e-05, 0.00032072176691144705, 0.0003799535916186869, 0.0001776174467522651, 0.00021202428615652025, 0.0038306480273604393, 0.00010710977949202061, 2.5333885787404142e-05, 0.0014211578527465463, 0.0004166108265053481, 0.0008034216589294374, 0.00020861027587670833, 0.00045565151958726346, 0.00019095839525107294, 0.0009474421385675669, 0.0003390771453268826, 0.003743864828720689, 0.0002046173467533663, 0.0023784483782947063, 0.0002782591327559203, 0.00011422019451856613, 9.886993211694062e-05, 0.0008324605878442526, 0.00010915552411461249, 0.0014770242851227522, 0.004380023572593927, 0.0033058125991374254, 0.0004904121160507202, 0.0009391116909682751, 0.004294251557439566, 0.004433758091181517, 0.0011093614157289267, 0.0014735684962943196, 0.0001349222584394738, 0.00318022258579731, 0.009836940094828606, 0.0001608944294275716, 0.0014605475589632988, 0.0011098711984232068, 0.0029562783893197775, 0.0007235890370793641, 0.00012282555690035224, 0.00017606512119527906, 0.000432351982453838, 0.0003031663072761148, 0.00014125801681075245, 0.00024728660355322063, 0.0004931026487611234, 4.5680491894017905e-05, 0.005605900660157204, 0.00011934831854887307, 0.00020548295287881047, 0.0005088415346108377, 0.0003000329015776515, 0.0022189978044480085, 8.6846666818019e-05, 0.00015559824532829225, 3.0328295906656422e-05, 0.00020001412485726178, 0.00016394634440075606, 0.0001851861597970128, 5.766171670984477e-05, 2.346670771657955e-05, 0.0006521676550619304, 0.00019569967116694897, 3.5890028811991215e-05, 4.5427757868310437e-05, 2.9700055165449157e-05, 0.00017792286234907806, 0.00010494551679585129, 8.614900434622541e-05, 5.91857751714997e-05, 8.496202644892037e-05, 0.00031928057433106005, 5.8348454331280664e-05, 0.0003249863802921027, 0.00014573334192391485, 9.598094766261056e-05, 0.0003437568957451731, 0.00016984311514534056, 4.552475729724392e-05, 0.00015786508447490633, 0.0004491915460675955, 8.291342965094373e-05, 0.00017383552039973438, 0.00019677005184348673, 0.00017128413310274482, 0.0005099810077808797, 4.297649866202846e-05, 0.00023316977603826672, 6.35220785625279e-05, 0.00010151477181352675, 5.139570566825569e-05, 0.00033004439319483936, 0.00014979335537645966, 7.846357766538858e-05, 3.2075342460302636e-05, 9.726791904540733e-05, 7.835646101739258e-05, 0.0009405241580680013, 0.0006174164009280503, 0.0004556083003990352, 0.0010927317198365927, 0.0002573138626758009, 0.000256074097706005, 8.612103556515649e-05, 5.6629276514286175e-05, 0.0006073255208320916, 4.119781078770757e-05, 0.0001904652308439836, 0.0002691808622330427, 0.00036230243858881295, 9.973798296414316e-05, 0.00030845380388200283, 0.00014764531806576997, 2.1887548427912407e-05, 0.00014684964844491333, 0.0001275904942303896, 0.0004005460359621793, 8.555447129765525e-05, 0.00011063444253522903, 0.000249848875682801, 0.0001945443800650537, 0.00034877043799497187, 7.088106940500438e-05, 3.814319279626943e-05, 0.00012102608161512762, 0.00023900196538306773, 8.068174793152139e-05, 0.0001313362445216626, 0.00016937193868216127, 0.00012985063949599862, 0.0007559973164461553], 'dim': [1, 1001], 'v': 1}}]</td>
      <td>edge-cv-demo</td>
      <td>{}</td>
      <td>1696958917070</td>
    </tr>
  </tbody>
</table>
