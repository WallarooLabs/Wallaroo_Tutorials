The following tutorial is available on the [Wallaroo Github Repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/pipeline-edge-publish/edge-observability-classification-finserv).

## Classification Financial Services Edge Deployment Demonstration

This notebook will walk through building Wallaroo pipeline with a a Classification model deployed to detect the likelihood of credit card fraud, then publishing that pipeline to an Open Container Initiative (OCI) Registry where it can be deployed in other Docker and Kubernetes environments.

This demonstration will focus on deployment to the edge.  For further examples of using Wallaroo with this computer vision models, see [Wallaroo 101](https://docs.wallaroo.ai/wallaroo-101/).

This demonstration will perform the following:

1. As a Data Scientist:
    1. Upload a computer vision model to Wallaroo, deploy it in a Wallaroo pipeline, then perform a sample inference.
    1. Publish the pipeline to an Open Container Initiative (OCI) Registry service.  This is configured in the Wallaroo instance.  See [Edge Deployment Registry Guide](https://staging.docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-edge-deployment/) for details on adding an OCI Registry Service to Wallaroo as the Edge Deployment Registry.  This demonstration uses a GitHub repository - see [Introduction to GitHub Packages](https://docs.github.com/en/packages/learn-github-packages/introduction-to-github-packages) for setting up your own package repository using GitHub, which can then be used with this tutorial.
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
import requests

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

# make a random 4 character suffix to verify uniqueness in tutorials
random_suffix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))

suffix='jch'

workspace_name = f'edge-observability-demo{suffix}'
pipeline_name = 'edge-observability-pipeline'
xgboost_model_name = 'ccfraud-xgboost'
xgboost_model_file_name = './models/xgboost_ccfraud.onnx'
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

    {'name': 'edge-observability-demojch', 'id': 14, 'archived': False, 'created_by': 'b3deff28-04d0-41b8-a04f-b5cf610d6ce9', 'created_at': '2023-10-31T20:12:45.456363+00:00', 'models': [], 'pipelines': []}

### Upload the Model

When a model is uploaded to a Wallaroo cluster, it is optimized and packaged to make it ready to run as part of a pipeline. In many times, the Wallaroo Server can natively run a model without any Python overhead. In other cases, such as a Python script, a custom Python environment will be automatically generated. This is comparable to the process of "containerizing" a model by adding a small HTTP server and other wrapping around it.

Our pretrained model is in ONNX format, which is specified in the `framework` parameter.

```python
xgboost_edge_demo_model = wl.upload_model(
    xgboost_model_name,
    xgboost_model_file_name,
    framework=wallaroo.framework.Framework.ONNX,
).configure(tensor_fields=["tensor"])
```

### Reserve Pipeline Resources

Before deploying an inference engine we need to tell wallaroo what resources it will need.
To do this we will use the wallaroo DeploymentConfigBuilder() and fill in the options listed below to determine what the properties of our inference engine will be.

We will be testing this deployment for an edge scenario, so the resource specifications are kept small -- what's the minimum needed to meet the expected load on the planned hardware.

- cpus - 1 => allow the engine to use 4 CPU cores when running the neural net
- memory - 900Mi => each inference engine will have 2 GB of memory, which is plenty for processing a single image at a time.
- arch - we will specify the X86 architecture.

```python
deploy_config = (wallaroo
                 .DeploymentConfigBuilder()
                 .replica_count(1)
                 .cpus(1)
                 .memory("900Mi")
                 .build()
                 )
```

### Simulated Edge Deployment

We will now deploy our pipeline into the current Kubernetes environment using the specified resource constraints. This is a "simulated edge" deploy in that we try to mimic the edge hardware as closely as possible.

```python
pipeline = get_pipeline(pipeline_name)
display(pipeline)

# clear the pipeline if previously run
pipeline.clear()
pipeline.add_model_step(xgboost_edge_demo_model)

pipeline.deploy(deployment_config = deploy_config)
```

<table><tr><th>name</th> <td>edge-observability-pipeline</td></tr><tr><th>created</th> <td>2023-10-31 20:12:45.895294+00:00</td></tr><tr><th>last_updated</th> <td>2023-10-31 20:12:45.895294+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>5d440e01-f2db-440c-a4f9-cd28bb491b6c</td></tr><tr><th>steps</th> <td></td></tr><tr><th>published</th> <td>False</td></tr></table>

    Waiting for deployment - this will take up to 45s ....... ok

<table><tr><th>name</th> <td>edge-observability-pipeline</td></tr><tr><th>created</th> <td>2023-10-31 20:12:45.895294+00:00</td></tr><tr><th>last_updated</th> <td>2023-10-31 20:12:46.009712+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>2b335efd-5593-422c-9ee9-52542b59601a, 5d440e01-f2db-440c-a4f9-cd28bb491b6c</td></tr><tr><th>steps</th> <td>ccfraud-xgboost</td></tr><tr><th>published</th> <td>False</td></tr></table>

### Run Single Image Inference

A single image, encoded using the Apache Arrow format, is sent to the deployed pipeline. Arrow is used here because, as a binary protocol, there is far lower network and compute overhead than using JSON. The Wallaroo Server engine accepts both JSON, pandas DataFrame, and Apache Arrow formats.

The sample DataFrames and arrow tables are in the `./data` directory.  We'll use the Apache Arrow table `cc_data_10k.arrow`.

Once complete, we'll display how long the inference request took.

```python
import datetime
import time

deploy_url = pipeline._deployment._url()

headers = wl.auth.auth_header()

headers['Content-Type']='application/vnd.apache.arrow.file'
# headers['Content-Type']='application/json; format=pandas-records'
headers['Accept']='application/json; format=pandas-records'

dataFile = './data/cc_data_1k.arrow'
local_inference_start = datetime.datetime.now()
```

```python
!curl -X POST {deploy_url} \
     -H "Authorization:{headers['Authorization']}" \
     -H "Content-Type:{headers['Content-Type']}" \
     -H "Accept:{headers['Accept']}" \
     --data-binary @{dataFile} > curl_response_xgboost.df.json
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  799k  100  685k  100  114k  39.3M  6733k --:--:-- --:--:-- --:--:-- 45.9M

We will import the inference output, and isolate the metadata `partition` to store where the inference results are stored in the pipeline logs.

```python
# display the first 20 results

df_results = pd.read_json('./curl_response_xgboost.df.json', orient="records")
# get just the partition
df_results['partition'] = df_results['metadata'].map(lambda x: x['partition'])
# display(df_results.head(20))
display(df_results.head(20).loc[:, ['time', 'out', 'partition']])
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out</th>
      <th>partition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1698783174953</td>
      <td>{'variable': [1.0094898]}</td>
      <td>engine-5d9b58dbd9-v5rvw</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1698783174953</td>
      <td>{'variable': [1.0094898]}</td>
      <td>engine-5d9b58dbd9-v5rvw</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1698783174953</td>
      <td>{'variable': [1.0094898]}</td>
      <td>engine-5d9b58dbd9-v5rvw</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1698783174953</td>
      <td>{'variable': [1.0094898]}</td>
      <td>engine-5d9b58dbd9-v5rvw</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1698783174953</td>
      <td>{'variable': [-1.9073485999999998e-06]}</td>
      <td>engine-5d9b58dbd9-v5rvw</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1698783174953</td>
      <td>{'variable': [-4.4882298e-05]}</td>
      <td>engine-5d9b58dbd9-v5rvw</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1698783174953</td>
      <td>{'variable': [-9.36985e-05]}</td>
      <td>engine-5d9b58dbd9-v5rvw</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1698783174953</td>
      <td>{'variable': [-8.3208084e-05]}</td>
      <td>engine-5d9b58dbd9-v5rvw</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1698783174953</td>
      <td>{'variable': [-8.332728999999999e-05]}</td>
      <td>engine-5d9b58dbd9-v5rvw</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1698783174953</td>
      <td>{'variable': [0.0004896521599999999]}</td>
      <td>engine-5d9b58dbd9-v5rvw</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1698783174953</td>
      <td>{'variable': [0.0006609559]}</td>
      <td>engine-5d9b58dbd9-v5rvw</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1698783174953</td>
      <td>{'variable': [7.57277e-05]}</td>
      <td>engine-5d9b58dbd9-v5rvw</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1698783174953</td>
      <td>{'variable': [-0.000100553036]}</td>
      <td>engine-5d9b58dbd9-v5rvw</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1698783174953</td>
      <td>{'variable': [-0.0005198717]}</td>
      <td>engine-5d9b58dbd9-v5rvw</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1698783174953</td>
      <td>{'variable': [-3.695488e-06]}</td>
      <td>engine-5d9b58dbd9-v5rvw</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1698783174953</td>
      <td>{'variable': [-0.00010883808]}</td>
      <td>engine-5d9b58dbd9-v5rvw</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1698783174953</td>
      <td>{'variable': [-0.00017666817]}</td>
      <td>engine-5d9b58dbd9-v5rvw</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1698783174953</td>
      <td>{'variable': [-2.8312206e-05]}</td>
      <td>engine-5d9b58dbd9-v5rvw</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1698783174953</td>
      <td>{'variable': [2.1755695e-05]}</td>
      <td>engine-5d9b58dbd9-v5rvw</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1698783174953</td>
      <td>{'variable': [-8.493661999999999e-05]}</td>
      <td>engine-5d9b58dbd9-v5rvw</td>
    </tr>
  </tbody>
</table>

### Undeploy Pipeline

With our testing complete, we will undeploy the pipeline and return the resources back to the cluster.

```python
pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s .................................... ok

<table><tr><th>name</th> <td>edge-observability-pipeline</td></tr><tr><th>created</th> <td>2023-10-31 20:12:45.895294+00:00</td></tr><tr><th>last_updated</th> <td>2023-10-31 20:12:46.009712+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>2b335efd-5593-422c-9ee9-52542b59601a, 5d440e01-f2db-440c-a4f9-cd28bb491b6c</td></tr><tr><th>steps</th> <td>ccfraud-xgboost</td></tr><tr><th>published</th> <td>False</td></tr></table>

### Publish the Pipeline for Edge Deployment

It worked! For a demo, we'll take working once as "tested". So now that we've tested our pipeline, we are ready to publish it for edge deployment.

Publishing it means assembling all of the configuration files and model assets and pushing them to an Open Container Initiative (OCI) repository set in the Wallaroo instance as the Edge Registry service.  DevOps engineers then retrieve that image and deploy it through Docker, Kubernetes, or similar deployments.

See [Edge Deployment Registry Guide](https://staging.docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-edge-deployment/) for details on adding an OCI Registry Service to Wallaroo as the Edge Deployment Registry.

This is done through the SDK command `wallaroo.pipeline.publish(deployment_config)` which has the following parameters and returns.

#### Publish a Pipeline Parameters

The `publish` method takes the following parameters.  The containerized pipeline will be pushed to the Edge registry service with the model, pipeline configurations, and other artifacts needed to deploy the pipeline.

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

```python
pub=pipeline.publish(deploy_config)
display(pub)
```

    Waiting for pipeline publish... It may take up to 600 sec.
    Pipeline is Publishing....Published.

<table>
    <tr><td>ID</td><td>2</td></tr>
    <tr><td>Pipeline Version</td><td>1a5a21e2-f0e8-4148-9ee2-23c877c0ac90</td></tr>
    <tr><td>Status</td><td>Published</td></tr>
    <tr><td>Engine URL</td><td><a href='https://us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/engines/proxy/wallaroo/ghcr.io/wallaroolabs/standalone-mini:v2023.4.0-4092'>us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/engines/proxy/wallaroo/ghcr.io/wallaroolabs/standalone-mini:v2023.4.0-4092</a></td></tr>
    <tr><td>Pipeline URL</td><td><a href='https://us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/pipelines/edge-observability-pipeline:1a5a21e2-f0e8-4148-9ee2-23c877c0ac90'>us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/pipelines/edge-observability-pipeline:1a5a21e2-f0e8-4148-9ee2-23c877c0ac90</a></td></tr>
    <tr><td>Helm Chart URL</td><td>oci://<a href='https://us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/charts/edge-observability-pipeline'>us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/charts/edge-observability-pipeline</a></td></tr>
    <tr><td>Helm Chart Reference</td><td>us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/charts@sha256:8bf628174b5ff87d913590d34a4c3d5eaa846b8b2a52bcf6a76295cd588cb6e8</td></tr>
    <tr><td>Helm Chart Version</td><td>0.0.1-1a5a21e2-f0e8-4148-9ee2-23c877c0ac90</td></tr>
    <tr><td>Engine Config</td><td>{'engine': {'resources': {'limits': {'cpu': 1.0, 'memory': '512Mi'}, 'requests': {'cpu': 1.0, 'memory': '512Mi'}}}, 'engineAux': {'images': {}}, 'enginelb': {'resources': {'limits': {'cpu': 1.0, 'memory': '512Mi'}, 'requests': {'cpu': 1.0, 'memory': '512Mi'}}}}</td></tr>
    <tr><td>User Images</td><td>[]</td></tr>
    <tr><td>Created By</td><td>john.hummel@wallaroo.ai</td></tr>
    <tr><td>Created At</td><td>2023-10-31 20:13:32.751657+00:00</td></tr>
    <tr><td>Updated At</td><td>2023-10-31 20:13:32.751657+00:00</td></tr>
    <tr><td>Docker Run Variables</td><td>{}</td></tr>
</table>

### List Published Pipelines

The method `wallaroo.client.list_pipelines()` shows a list of all pipelines in the Wallaroo instance, and includes the `published` field that indicates whether the pipeline was published to the registry (`True`), or has not yet been published (`False`).

```python
wl.list_pipelines()
```

<table><tr><th>name</th><th>created</th><th>last_updated</th><th>deployed</th><th>arch</th><th>tags</th><th>versions</th><th>steps</th><th>published</th></tr><tr><td>edge-observability-pipeline</td><td>2023-31-Oct 20:12:45</td><td>2023-31-Oct 20:13:32</td><td>False</td><td>None</td><td></td><td>1a5a21e2-f0e8-4148-9ee2-23c877c0ac90, 2b335efd-5593-422c-9ee9-52542b59601a, 5d440e01-f2db-440c-a4f9-cd28bb491b6c</td><td>ccfraud-xgboost</td><td>True</td></tr><tr><td>housepricesagapipeline</td><td>2023-31-Oct 20:04:58</td><td>2023-31-Oct 20:13:02</td><td>True</td><td>None</td><td></td><td>0056edf3-730f-452d-a6ed-2dfa47ff5567, 8bc714ea-8257-4512-a102-402baf3143b3, 76006480-b145-4d6a-9e95-9b2e7a4f8d8e</td><td>housepricesagacontrol</td><td>True</td></tr></table>

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

<table><tr><th>id</th><th>pipeline_version_name</th><th>engine_url</th><th>pipeline_url</th><th>created_by</th><th>created_at</th><th>updated_at</th></tr><tr><td>2</td><td>1a5a21e2-f0e8-4148-9ee2-23c877c0ac90</td><td><a href='https://us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/engines/proxy/wallaroo/ghcr.io/wallaroolabs/standalone-mini:v2023.4.0-4092'>us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/engines/proxy/wallaroo/ghcr.io/wallaroolabs/standalone-mini:v2023.4.0-4092</a></td><td><a href='https://us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/pipelines/edge-observability-pipeline:1a5a21e2-f0e8-4148-9ee2-23c877c0ac90'>us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/pipelines/edge-observability-pipeline:1a5a21e2-f0e8-4148-9ee2-23c877c0ac90</a></td><td>john.hummel@wallaroo.ai</td><td>2023-31-Oct 20:13:32</td><td>2023-31-Oct 20:13:32</td></tr></table>

## DevOps - Pipeline Edge Deployment

Once a pipeline is deployed to the Edge Registry service, it can be deployed in environments such as Docker, Kubernetes, or similar container running services by a DevOps engineer as a Wallaroo Server.

The following guides will demonstrate publishing a Wallaroo Pipeline as a Wallaroo Server.

### Add Edge Location

Wallaroo Servers can optionally connect to the Wallaroo Ops instance and transmit their inference results.  These are added to the pipeline logs for the published pipeline the Wallaroo Server is associated with.

Wallaroo Servers are added with the `wallaroo.pipeline_publish.add_edge(name: string)` method.  The `name` is the unique primary key for each edge added to the pipeline publish and must be unique.

This returns a Publish Edge with the following fields:

| Field | Type | Description |
|---|---|---|
| `id` | Integer | The integer ID of the pipeline publish.|
| `created_at` | DateTime | The DateTime of the pipeline publish. |
| `docker_run_variables` | String | The Docker variables in UUID format that include the following: The `BUNDLE_VERSION`, `EDGE_NAME`, `JOIN_TOKEN_`, `OPSCENTER_HOST`, `PIPELINE_URL`, and `WORKSPACE_ID`.
| `engine_config` | String | The Wallaroo `wallaroo.deployment_config.DeploymentConfig` for the pipeline. |
| `pipeline_version_id` | Integer | The integer identifier of the pipeline version published. |
| `status` | String | The status of the publish.  `Published` is a successful publish.|
| `updated_at` | DateTime | The DateTime when the pipeline publish was updated. |
| `user_images` | List(String) | User images used in the pipeline publish. |
| `created_by` | String | The UUID of the Wallaroo user that created the pipeline publish. |
| `engine_url` | String | The URL for the published pipeline's Wallaroo engine in the OCI registry. |
| `error` | String | Any errors logged. |
| `helm` | String | The helm chart, helm reference and helm version. |
| `pipeline_url` | String | The URL for the published pipeline's container in the OCI registry. |
| `pipeline_version_name` | String | The UUID identifier of the pipeline version published. |
| `additional_properties` | String | Any other identities. |

Two edge publishes will be created so we can demonstrate removing an edge shortly.

```python
edge_01_name = f'edge-ccfraud-observability{random_suffix}'
edge01 = pub.add_edge(edge_01_name)
display(edge01)

edge_02_name = f'edge-ccfraud-observability-02{random_suffix}'
edge02 = pub.add_edge(edge_02_name)
display(edge02)
```

<table>
    <tr><td>ID</td><td>2</td></tr>
    <tr><td>Pipeline Version</td><td>1a5a21e2-f0e8-4148-9ee2-23c877c0ac90</td></tr>
    <tr><td>Status</td><td>Published</td></tr>
    <tr><td>Engine URL</td><td><a href='https://us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/engines/proxy/wallaroo/ghcr.io/wallaroolabs/standalone-mini:v2023.4.0-4092'>us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/engines/proxy/wallaroo/ghcr.io/wallaroolabs/standalone-mini:v2023.4.0-4092</a></td></tr>
    <tr><td>Pipeline URL</td><td><a href='https://us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/pipelines/edge-observability-pipeline:1a5a21e2-f0e8-4148-9ee2-23c877c0ac90'>us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/pipelines/edge-observability-pipeline:1a5a21e2-f0e8-4148-9ee2-23c877c0ac90</a></td></tr>
    <tr><td>Helm Chart URL</td><td>oci://<a href='https://us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/charts/edge-observability-pipeline'>us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/charts/edge-observability-pipeline</a></td></tr>
    <tr><td>Helm Chart Reference</td><td>us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/charts@sha256:8bf628174b5ff87d913590d34a4c3d5eaa846b8b2a52bcf6a76295cd588cb6e8</td></tr>
    <tr><td>Helm Chart Version</td><td>0.0.1-1a5a21e2-f0e8-4148-9ee2-23c877c0ac90</td></tr>
    <tr><td>Engine Config</td><td>{'engine': {'resources': {'limits': {'cpu': 1.0, 'memory': '512Mi'}, 'requests': {'cpu': 1.0, 'memory': '512Mi'}}}, 'engineAux': {'images': {}}, 'enginelb': {'resources': {'limits': {'cpu': 1.0, 'memory': '512Mi'}, 'requests': {'cpu': 1.0, 'memory': '512Mi'}}}}</td></tr>
    <tr><td>User Images</td><td>[]</td></tr>
    <tr><td>Created By</td><td>john.hummel@wallaroo.ai</td></tr>
    <tr><td>Created At</td><td>2023-10-31 20:13:32.751657+00:00</td></tr>
    <tr><td>Updated At</td><td>2023-10-31 20:13:32.751657+00:00</td></tr>
    <tr><td>Docker Run Variables</td><td>{'EDGE_BUNDLE': 'abcde'}</td></tr>
</table>

<table>
    <tr><td>ID</td><td>2</td></tr>
    <tr><td>Pipeline Version</td><td>1a5a21e2-f0e8-4148-9ee2-23c877c0ac90</td></tr>
    <tr><td>Status</td><td>Published</td></tr>
    <tr><td>Engine URL</td><td><a href='https://us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/engines/proxy/wallaroo/ghcr.io/wallaroolabs/standalone-mini:v2023.4.0-4092'>us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/engines/proxy/wallaroo/ghcr.io/wallaroolabs/standalone-mini:v2023.4.0-4092</a></td></tr>
    <tr><td>Pipeline URL</td><td><a href='https://us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/pipelines/edge-observability-pipeline:1a5a21e2-f0e8-4148-9ee2-23c877c0ac90'>us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/pipelines/edge-observability-pipeline:1a5a21e2-f0e8-4148-9ee2-23c877c0ac90</a></td></tr>
    <tr><td>Helm Chart URL</td><td>oci://<a href='https://us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/charts/edge-observability-pipeline'>us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/charts/edge-observability-pipeline</a></td></tr>
    <tr><td>Helm Chart Reference</td><td>us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/charts@sha256:8bf628174b5ff87d913590d34a4c3d5eaa846b8b2a52bcf6a76295cd588cb6e8</td></tr>
    <tr><td>Helm Chart Version</td><td>0.0.1-1a5a21e2-f0e8-4148-9ee2-23c877c0ac90</td></tr>
    <tr><td>Engine Config</td><td>{'engine': {'resources': {'limits': {'cpu': 1.0, 'memory': '512Mi'}, 'requests': {'cpu': 1.0, 'memory': '512Mi'}}}, 'engineAux': {'images': {}}, 'enginelb': {'resources': {'limits': {'cpu': 1.0, 'memory': '512Mi'}, 'requests': {'cpu': 1.0, 'memory': '512Mi'}}}}</td></tr>
    <tr><td>User Images</td><td>[]</td></tr>
    <tr><td>Created By</td><td>john.hummel@wallaroo.ai</td></tr>
    <tr><td>Created At</td><td>2023-10-31 20:13:32.751657+00:00</td></tr>
    <tr><td>Updated At</td><td>2023-10-31 20:13:32.751657+00:00</td></tr>
    <tr><td>Docker Run Variables</td><td>{'EDGE_BUNDLE': 'abcde'}</td></tr>
</table>

```python
pipeline.list_edges()
```

<table><tr><th>ID</th><th>Name</th><th>Tags</th><th>Pipeline Version</th><th>SPIFFE ID</th></tr><tr><td>898bb58c-77c2-4164-b6cc-f004dc39e125</td><td>edge-ccfraud-observabilityymgy</td><td>[]</td><td>6</td><td>wallaroo.ai/ns/deployments/edge/898bb58c-77c2-4164-b6cc-f004dc39e125</td></tr><tr><td>1f35731a-f4f6-4cd0-a23a-c4a326b73277</td><td>edge-ccfraud-observability-02ymgy</td><td>[]</td><td>6</td><td>wallaroo.ai/ns/deployments/edge/1f35731a-f4f6-4cd0-a23a-c4a326b73277</td></tr></table>

### Remove Edge Location

Wallaroo Servers are removed with the `wallaroo.pipeline_publish.remove_edge(name: string)` method.

This returns a Publish Edge with the following fields:

| Field | Type | Description |
|---|---|---|
| `id` | Integer | The integer ID of the pipeline publish.|
| `created_at` | DateTime | The DateTime of the pipeline publish. |
| `docker_run_variables` | String | The Docker variables in UUID format that include the following: The `BUNDLE_VERSION`, `EDGE_NAME`, `JOIN_TOKEN_`, `OPSCENTER_HOST`, `PIPELINE_URL`, and `WORKSPACE_ID`.
| `engine_config` | String | The Wallaroo `wallaroo.deployment_config.DeploymentConfig` for the pipeline. |
| `pipeline_version_id` | Integer | The integer identifier of the pipeline version published. |
| `status` | String | The status of the publish.  `Published` is a successful publish.|
| `updated_at` | DateTime | The DateTime when the pipeline publish was updated. |
| `user_images` | List(String) | User images used in the pipeline publish. |
| `created_by` | String | The UUID of the Wallaroo user that created the pipeline publish. |
| `engine_url` | String | The URL for the published pipeline's Wallaroo engine in the OCI registry. |
| `error` | String | Any errors logged. |
| `helm` | String | The helm chart, helm reference and helm version. |
| `pipeline_url` | String | The URL for the published pipeline's container in the OCI registry. |
| `pipeline_version_name` | String | The UUID identifier of the pipeline version published. |
| `additional_properties` | String | Any other identities. |

Two edge publishes will be created so we can demonstrate removing an edge shortly.

```python
sample = pub.remove_edge(edge_02_name)
display(sample)

```

    None

```python
pipeline.list_edges()
```

<table><tr><th>ID</th><th>Name</th><th>Tags</th><th>Pipeline Version</th><th>SPIFFE ID</th></tr><tr><td>898bb58c-77c2-4164-b6cc-f004dc39e125</td><td>edge-ccfraud-observabilityymgy</td><td>[]</td><td>6</td><td>wallaroo.ai/ns/deployments/edge/898bb58c-77c2-4164-b6cc-f004dc39e125</td></tr></table>

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

```bash
docker run -p 8080:8080 \
    -e DEBUG=true \
    -e OCI_REGISTRY={your registry server} \
    -e EDGE_BUNDLE={Your edge bundle['EDGE_BUNDLE']}
    -e CONFIG_CPUS=1 \
    -e OCI_USERNAME=oauth2accesstoken \
    -e OCI_PASSWORD={registry token here} \
    -e PIPELINE_URL={your registry server}{Your Wallaroo Server pipeline} \
    {your registry server}/{Your Wallaroo Server engine}
```

### Docker Compose Deployment

For users who prefer to use `docker compose`, the following sample `compose.yaml` file is used to launch the Wallaroo Edge pipeline.  This is the same used in the [Wallaroo Use Case Tutorials Computer Vision: Retail](https://docs.wallaroo.ai/wallaroo-use-case-tutorials/wallaroo-use-case-computer-vision/use-case-computer-vision-retail/) tutorials.

```yml
services:
  engine:
    image: {Your Engine URL}
    ports:
      - 8080:8080
    environment:
      EDGE_BUNDLE: {Your Edge Bundle['EDGE_BUNDLE']}
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

The following code segment generates a `docker run` template based on the previously published pipeline.  Replace the `$REGISTRYURL`, `$REGISTRYUSERNAME` and `$REGISTRYPASSWORD` with your OCI registry values.

```python
docker_command = f'''
docker run -p 8080:8080 \\
    -e DEBUG=true \\
    -e OCI_REGISTRY=$REGISTRYURL \\
    -e EDGE_BUNDLE={edge01.docker_run_variables['EDGE_BUNDLE']} \\
    -e CONFIG_CPUS=1 \\
    -e OCI_USERNAME=$REGISTRYUSERNAME \\
    -e OCI_PASSWORD=$REGISTRYPASSWORD \\
    -e PIPELINE_URL={edge01.pipeline_url} \\
    {edge01.engine_url}
'''

print(docker_command)
```

## Edge Deployed Pipeline API Endpoints

Once deployed, we can check the pipelines and models available.  We'll use a `curl` command, but any HTTP based request will work the same way.

The endpoint `/pipelines` returns:

* **id** (*String*):  The name of the pipeline.
* **status** (*String*):  The status as either `Running`, or `Error` if there are any issues.

For this example, the deployment is made on a machine called `localhost`.  Replace this URL with the URL of you edge deployment.

```python
!curl testboy.local:8080/pipelines
```

The endpoint `/models` returns a List of models with the following fields:

* **name** (*String*): The model name.
* **sha** (*String*): The sha hash value of the ML model.
* **status** (*String*):  The status of either Running or Error if there are any issues.
* **version** (*String*):  The model version.  This matches the version designation used by Wallaroo to track model versions in UUID format.

```python
!curl testboy.local:8080/models
```

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

```python
!curl -X POST testboy.local:8080/pipelines/edge-observability-pipeline \
    -H "Content-Type: application/vnd.apache.arrow.file" \
    --data-binary @./data/cc_data_1k.arrow > curl_response_edge.df.json
```

```python
# display the first 20 results

df_results = pd.read_json('./curl_response_edge.df.json', orient="records")
# display(df_results.head(20))
display(df_results.head(20).loc[:, ['time', 'out', 'metadata']])
```

```python
pipeline.export_logs(
    limit=30000,
    directory='partition-edge-observability',
    file_prefix='edge-logs',
    dataset=['time', 'out', 'metadata']
)
```

```python
# display the head 20 results

df_logs = pd.read_json('./partition-edge-observability/edge-logs-1.json', orient="records", lines=True)
# get just the partition
# df_results['partition'] = df_results['metadata'].map(lambda x: x['partition'])
# display(df_results.head(20))
display(df_logs.head(20).loc[:, ['time', 'out.variable', 'metadata.partition']])
```

```python
display(pd.unique(df_logs['metadata.partition']))
```
