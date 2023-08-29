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

workspace_name = f'edge-cv-retail{suffix}'
pipeline_name = 'edge-cv-retail'
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

    {'name': 'edge-cv-retailjohn', 'id': 13, 'archived': False, 'created_by': '66d3b2c4-9b22-4429-b16e-3bcdc1ac28e3', 'created_at': '2023-08-23T17:06:48.016472+00:00', 'models': [], 'pipelines': []}

### Upload the Model

When a model is uploaded to a Wallaroo cluster, it is optimized and packaged to make it ready to run as part of a pipeline. In many times, the Wallaroo Server can natively run a model without any Python overhead. In other cases, such as a Python script, a custom Python environment will be automatically generated. This is comparable to the process of "containerizing" a model by adding a small HTTP server and other wrapping around it.

Our pretrained model is in ONNX format, which is specified in the `framework` parameter.

```python
model = wl.upload_model(model_name, 
                        model_file_name, 
                        framework = wallaroo.framework.Framework.ONNX)
```

### Reserve Pipeline Resources

Before deploying an inference engine we need to tell wallaroo what resources it will need.
To do this we will use the wallaroo DeploymentConfigBuilder() and fill in the options listed below to determine what the properties of our inference engine will be.

We will be testing this deployment for an edge scenario, so the resource specifications are kept small -- what's the minimum needed to meet the expected load on the planned hardware.

- cpus - 4 => allow the engine to use 4 CPU cores when running the neural net
- memory - 2Gi => each inference engine will have 2 GB of memory, which is plenty for processing a single image at a time.
- arch - we will specify the X86 architecture.

```python
deployment_config = wallaroo.DeploymentConfigBuilder() \
    .cpus(4)\
    .memory("2Gi")\
    .arch(wallaroo.engine_config.Architecture.X86)\
    .build()
```

### Simulated Edge Deployment

We will now deploy our pipeline into the current Kubernetes environment using the specified resource constraints. This is a "simulated edge" deploy in that we try to mimic the edge hardware as closely as possible.

```python
pipeline = wl.build_pipeline(pipeline_name)
pipeline.add_model_step(model)

pipeline.deploy(deployment_config = deployment_config)
```

    Waiting for deployment - this will take up to 45s ............. ok

<table><tr><th>name</th> <td>edge-cv-retail</td></tr><tr><th>created</th> <td>2023-08-23 17:07:09.438570+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-23 17:07:09.502996+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>5eacb4cd-948f-4b82-9206-1eab30bd5488, 38b2bfc8-843d-4449-9935-ec82d9f86c49</td></tr><tr><th>steps</th> <td>resnet-50</td></tr><tr><th>published</th> <td>False</td></tr></table>

```python
pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.3.84',
       'name': 'engine-dff78c9bc-mg4xs',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'edge-cv-retail',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'resnet-50',
          'version': '693e19b5-0dc7-4afb-9922-e3f7feefe66d',
          'sha': 'c6c8869645962e7711132a7e17aced2ac0f60dcdc2c7faa79b2de73847a87984',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.4.101',
       'name': 'engine-lb-584f54c899-pkr6s',
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

    Average elapsed: 34.79213433333334 ms

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

    Waiting for pipeline publish... It may take up to 60 sec.
    Pipeline is Publishing....Published.

<table>
    <tr><td>ID</td><td>6</td></tr>
    <tr><td>Pipeline Version ID</td><td>26</td></tr>
    <tr><td>Status</td><td>Published</td></tr>
    <tr><td>Engine URL</td><td><a href='https://us-west1-docker.pkg.dev/wallaroo-dev-253816/doc-test-registry/engine:v2023.3.0-main-3707'>us-west1-docker.pkg.dev/wallaroo-dev-253816/doc-test-registry/engine:v2023.3.0-main-3707</a></td></tr>
    <tr><td>Pipeline URL</td><td><a href='https://us-west1-docker.pkg.dev/wallaroo-dev-253816/doc-test-registry/pipelines/edge-cv-retail:bf70eaf7-8c11-4b46-b751-916a43b1a555'>us-west1-docker.pkg.dev/wallaroo-dev-253816/doc-test-registry/pipelines/edge-cv-retail:bf70eaf7-8c11-4b46-b751-916a43b1a555</a></td></tr>
    <tr><td>Helm Chart URL</td><td><a href='https://us-west1-docker.pkg.dev/wallaroo-dev-253816/doc-test-registry/charts/edge-cv-retail'>us-west1-docker.pkg.dev/wallaroo-dev-253816/doc-test-registry/charts/edge-cv-retail</a></td></tr>
    <tr><td>Helm Chart Reference</td><td>us-west1-docker.pkg.dev/wallaroo-dev-253816/doc-test-registry/charts@sha256:04e73fe893c5bf8f646f1a4559b9bb91f6657793479ff9d67e9b91ef20f92514</td></tr>
    <tr><td>Helm Chart Version</td><td>0.0.1-bf70eaf7-8c11-4b46-b751-916a43b1a555</td></tr>
    <tr><td>Engine Config</td><td>{'engine': {'resources': {'limits': {'cpu': 4.0, 'memory': '2Gi'}, 'requests': {'cpu': 4.0, 'memory': '2Gi'}}}, 'engineAux': {'images': {}}, 'enginelb': {}}</td></tr>
    <tr><td>Created By</td><td>66d3b2c4-9b22-4429-b16e-3bcdc1ac28e3</td></tr>
    <tr><td>Created At</td><td>2023-08-23 17:07:38.914639+00:00</td></tr>
    <tr><td>Updated At</td><td>2023-08-23 17:07:38.914639+00:00</td></tr>
</table>

### List Published Pipeline

The method `wallaroo.client.list_pipelines()` shows a list of all pipelines in the Wallaroo instance, and includes the `published` field that indicates whether the pipeline was published to the registry (`True`), or has not yet been published (`False`).

```python
wl.list_pipelines()
```

<table><tr><th>name</th><th>created</th><th>last_updated</th><th>deployed</th><th>tags</th><th>versions</th><th>steps</th><th>published</th></tr><tr><td>edge-cv-retail</td><td>2023-23-Aug 17:07:09</td><td>2023-23-Aug 17:07:38</td><td>True</td><td></td><td>bf70eaf7-8c11-4b46-b751-916a43b1a555, 5eacb4cd-948f-4b82-9206-1eab30bd5488, 38b2bfc8-843d-4449-9935-ec82d9f86c49</td><td>resnet-50</td><td>True</td></tr><tr><td>houseprice-estimator</td><td>2023-22-Aug 20:39:35</td><td>2023-23-Aug 14:43:10</td><td>True</td><td></td><td>a9fbe04e-4ec3-48fc-ac5f-998035c98f5a, 751aea9c-222d-4eac-8f4f-318fd4019db0, f42c0457-e4f3-4370-b152-0a220347de11</td><td>house-price-prime</td><td>False</td></tr><tr><td>biolabspipeline</td><td>2023-22-Aug 16:07:20</td><td>2023-22-Aug 16:24:40</td><td>False</td><td></td><td>4c6dceb7-e692-4b8b-b615-4f7873eb020b, 59d0babe-bc1d-4dbb-959f-711c74f7b05d, ae834c0d-7a5b-4f87-9e2e-1f06f3cd25e7, 7c438222-28d8-4fca-9a70-eabee8a0fac5</td><td>biolabsmodel</td><td>False</td></tr><tr><td>biolabspipeline</td><td>2023-22-Aug 16:03:33</td><td>2023-22-Aug 16:03:38</td><td>False</td><td></td><td>4e103a7d-cd4d-464b-b182-61d4041518a8, ec2a0fd6-21d4-4843-b7c3-65b1e5be1b85, 516f3848-be98-40d7-8564-a1e48eecb7a8</td><td>biolabsmodel</td><td>False</td></tr><tr><td>biolabspipelinegomj</td><td>2023-22-Aug 15:11:12</td><td>2023-22-Aug 15:42:44</td><td>False</td><td></td><td>1dc9f89f-82aa-4a71-b21a-75dc8d5e4e51, 152d12f2-1200-46ad-ad04-60078c5aa284, 6ca59ffd-802e-4ad5-bd9a-35146b9fbda5, bdab08cc-3e99-4afc-b22d-657e33b76f29, 3c8feb0d-3124-4018-8dfa-06162156d51e</td><td>biolabsmodelgomj</td><td>False</td></tr><tr><td>edge-pipeline</td><td>2023-21-Aug 20:54:37</td><td>2023-22-Aug 19:06:46</td><td>False</td><td></td><td>2be013d9-a438-453c-a013-3fd8e6218394, a02b6af5-4235-42af-92c6-5ae678b35be4, e721ccad-11d8-4874-8388-4211c4957d18, d642e766-cffb-451f-b197-e058bedbdd5f, eb586aba-4908-4bff-84e1-bdeb1fa4b7d3, 2163d718-a5ea-41e3-b69f-095efa858462</td><td>ccfraud</td><td>True</td></tr><tr><td>p1</td><td>2023-21-Aug 19:38:44</td><td>2023-21-Aug 19:38:44</td><td>(unknown)</td><td></td><td>5f93e90a-e8d6-4e8a-8a1a-22eee80a3e13, 5f78247f-7bf9-445b-98a6-e146fb22b8e9</td><td></td><td>True</td></tr></table>

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

<table><tr><th>id</th><th>pipeline_version_id</th><th>engine_url</th><th>pipeline_url</th><th>created_by</th><th>created_at</th><th>updated_at</th></tr><tr><td>6</td><td>26</td><td><a href='https://us-west1-docker.pkg.dev/wallaroo-dev-253816/doc-test-registry/engine:v2023.3.0-main-3707'>us-west1-docker.pkg.dev/wallaroo-dev-253816/doc-test-registry/engine:v2023.3.0-main-3707</a></td><td><a href='https://us-west1-docker.pkg.dev/wallaroo-dev-253816/doc-test-registry/pipelines/edge-cv-retail:bf70eaf7-8c11-4b46-b751-916a43b1a555'>us-west1-docker.pkg.dev/wallaroo-dev-253816/doc-test-registry/pipelines/edge-cv-retail:bf70eaf7-8c11-4b46-b751-916a43b1a555</a></td><td>66d3b2c4-9b22-4429-b16e-3bcdc1ac28e3</td><td>2023-23-Aug 17:07:38</td><td>2023-23-Aug 17:07:38</td></tr></table>

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

## Edge Deployed Pipeline API Endpoints

Once deployed, we can check the pipelines and models available.  We'll use a `curl` command, but any HTTP based request will work the same way.

The endpoint `/pipelines` returns:

* **id** (*String*):  The name of the pipeline.
* **status** (*String*):  The status as either `Running`, or `Error` if there are any issues.

```bash
curl localhost:8080/pipelines
{"pipelines":[{"id":"edge-cv-retail","status":"Running"}]}
```

The endpoint `/models` returns a List of models with the following fields:

* **name** (*String*): The model name.
* **sha** (*String*): The sha hash value of the ML model.
* **status** (*String*):  The status of either Running or Error if there are any issues.
* **version** (*String*):  The model version.  This matches the version designation used by Wallaroo to track model versions in UUID format.

```bash
curl localhost:8080/models
{"models":[{"name":"resnet-50","sha":"c6c8869645962e7711132a7e17aced2ac0f60dcdc2c7faa79b2de73847a87984","status":"Running","version":"693e19b5-0dc7-4afb-9922-e3f7feefe66d"}]}
```

### Edge Deployed Inference

The inference endpoint takes the following pattern:

* `/pipelines/{pipeline-name}`:  The `pipeline-name` is the same as returned from the [`/pipelines`](#list-pipelines) endpoint as `id`.

Wallaroo inference endpoint URLs accept the following data inputs through the `Content-Type` header:

* `Content-Type: application/vnd.apache.arrow.file`: For Apache Arrow tables.
* `Content-Type: application/json; format=pandas-records`: For pandas DataFrame in record format.
* `Content-Type: application/json`: JSON.

The `Accept` header will determine what kind format of the data is returned.
  * `Accept: application/json` Returns a JSON object in the following format.

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
  
Once deployed, we can perform an inference through the deployment URL.  We'll assume we're running the inference request through the localhost and submitting the local file `./data/image_224x224.arrow`.  Note that our inference endpoint is `pipelines/edge-cv-retail` - the same as our pipeline name.

The following example demonstrates sending an Apache Arrow table to the Edge deployed pipeline, requesting the inference results back in a pandas DataFrame records format.

```bash
curl -X POST localhost:8080/pipelines/edge-cv-retail -H "Content-Type: application/vnd.apache.arrow.file" -H 'Accept: application/json; format=pandas-records'  --data-binary @./data/image_224x224.arrow
```

Returns:

```json
{"check_failures":[],"elapsed":[777138,22000001],"model_name":"resnet-50","model_version":"693e19b5-0dc7-4afb-9922-e3f7feefe66d","original_data":null,"outputs":[{"Int64":{"data":[535],"dim":[1],"v":1}},{"Float":{"data":[0.00009498586587142199,0.00009141524787992239,0.0004606838047038764,0.00007667174941161647,0.00008047101437114179,0.00006355856748996302,0.0001758082798914984,0.000014166356777423061,0.00004344096305430867,0.00004225136217428371,0.0002540049026720226,0.005299815908074379...],"dim":[1,1001],"v":1}}],"pipeline_name":"edge-cv-retail","shadow_data":{},"time":1692817794503}]
```

