The following tutorial is available on the [Wallaroo Github Repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/pipeline-edge-publish/edge-observability-classification-finserv-api).

## Classification Financial Services Edge Deployment Demonstration via API

This notebook will walk through building Wallaroo pipeline with a a Classification model deployed to detect the likelihood of credit card fraud, then publishing that pipeline to an Open Container Initiative (OCI) Registry where it can be deployed in other Docker and Kubernetes environments.  This example focuses on using the Wallaroo MLOps API to publish the pipeline and retrieve the relevant information.

This demonstration will focus on deployment to the edge.  For further examples of using Wallaroo with this computer vision models, see [Wallaroo 101](https://docs.wallaroo.ai/wallaroo-101/).

For this demonstration, logging into the Wallaroo MLOps API will be done through the Wallaroo SDK.  For more details on authenticating and connecting to the Wallaroo MLOps API, see the [Wallaroo API Connection Guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-connection-guide/).

This demonstration performs the following:

* In Wallaroo Ops:
  * Setting up a workspace, pipeline, and model for deriving the price of a house based on inputs.
  * Creating an assay from a sample of inferences.
  * Display the inference result and upload the assay to the Wallaroo instance where it can be referenced later.
* In a remote aka edge location:
  * Deploying the Wallaroo pipeline as a Wallaroo Inference Server deployed on an edge device with observability features.
* In Wallaroo Ops:
  * Observe the Wallaroo Ops and remote Wallaroo Inference Server inference results as part of the pipeline logs.

## Prerequisites

* A deployed Wallaroo Ops instance.
* A location with Docker or Kubernetes with `helm` for Wallaroo Inference server deployments.
* The following Python libraries installed:
  * [`wallaroo`](https://pypi.org/project/wallaroo/): The Wallaroo SDK. Included with the Wallaroo JupyterHub service by default.
  * [`pandas`](https://pypi.org/project/pandas/): Pandas, mainly used for Pandas DataFrame

## References

* [Wallaroo SDK Essentials Guide: Pipeline Edge Publication](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline-publication/)

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
suffix_random= ''.join(random.choice(string.ascii_lowercase) for i in range(4))
suffix=''

workspace_name = f'edge-publish-api-demo{suffix}'
pipeline_name = 'edge-pipeline'
model_name = 'ccfraud'
model_file_name = './models/xgboost_ccfraud.onnx'
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

    {'name': 'edge-publish-api-demo', 'id': 21, 'archived': False, 'created_by': 'b3deff28-04d0-41b8-a04f-b5cf610d6ce9', 'created_at': '2023-10-31T20:36:33.093439+00:00', 'models': [], 'pipelines': []}

### Upload the Model

When a model is uploaded to a Wallaroo cluster, it is optimized and packaged to make it ready to run as part of a pipeline. In many times, the Wallaroo Server can natively run a model without any Python overhead. In other cases, such as a Python script, a custom Python environment will be automatically generated. This is comparable to the process of "containerizing" a model by adding a small HTTP server and other wrapping around it.

Our pretrained model is in ONNX format, which is specified in the `framework` parameter.

```python
edge_demo_model = wl.upload_model(
    model_name,
    model_file_name,
    framework=wallaroo.framework.Framework.ONNX,
).configure(tensor_fields=["tensor"])
```

### Reserve Pipeline Resources

Before deploying an inference engine we need to tell wallaroo what resources it will need.
To do this we will use the wallaroo DeploymentConfigBuilder() and fill in the options listed below to determine what the properties of our inference engine will be.

We will be testing this deployment for an edge scenario, so the resource specifications are kept small -- what's the minimum needed to meet the expected load on the planned hardware.

- cpus - 0.5 => allow the engine to use 4 CPU cores when running the neural net
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

pipeline.add_model_step(edge_demo_model)

pipeline.deploy(deployment_config = deploy_config)
```

<table><tr><th>name</th> <td>edge-pipeline</td></tr><tr><th>created</th> <td>2023-10-31 20:36:33.486078+00:00</td></tr><tr><th>last_updated</th> <td>2023-10-31 20:36:33.486078+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>a5752acc-9334-4cac-8b34-471979b55d61</td></tr><tr><th>steps</th> <td></td></tr><tr><th>published</th> <td>False</td></tr></table>

    Waiting for deployment - this will take up to 45s ............. ok

<table><tr><th>name</th> <td>edge-pipeline</td></tr><tr><th>created</th> <td>2023-10-31 20:36:33.486078+00:00</td></tr><tr><th>last_updated</th> <td>2023-10-31 20:36:33.629569+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>932e9cd2-978e-40e6-ab84-2ed5e5d63408, a5752acc-9334-4cac-8b34-471979b55d61</td></tr><tr><th>steps</th> <td>ccfraud</td></tr><tr><th>published</th> <td>False</td></tr></table>

### Run Sample Inference

A sample input will not be provided to test the inference.

The sample DataFrames and arrow tables are in the `./data` directory.  We'll use the Apache Arrow table `cc_data_1k.arrow` with 1,000 records.

```python
deploy_url = pipeline._deployment._url()

headers = wl.auth.auth_header()

headers['Content-Type']='application/vnd.apache.arrow.file'
# headers['Content-Type']='application/json; format=pandas-records'
headers['Accept']='application/json; format=pandas-records'

dataFile = './data/cc_data_1k.arrow'
```

```python
!curl -X POST {deploy_url} \
     -H "Authorization:{headers['Authorization']}" \
     -H "Content-Type:{headers['Content-Type']}" \
     -H "Accept:{headers['Accept']}" \
     --data-binary @{dataFile} > curl_response.df.json
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  755k  100  641k  100  114k  44.7M  8176k --:--:-- --:--:-- --:--:-- 52.7M

```python
# display the first 20 results

df_results = pd.read_json('./curl_response.df.json', orient="records")
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
      <td>1698784607697</td>
      <td>{'variable': [1.0094898]}</td>
      <td>engine-fc7d5c445-bw7hf</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1698784607697</td>
      <td>{'variable': [1.0094898]}</td>
      <td>engine-fc7d5c445-bw7hf</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1698784607697</td>
      <td>{'variable': [1.0094898]}</td>
      <td>engine-fc7d5c445-bw7hf</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1698784607697</td>
      <td>{'variable': [1.0094898]}</td>
      <td>engine-fc7d5c445-bw7hf</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1698784607697</td>
      <td>{'variable': [-1.9073485999999998e-06]}</td>
      <td>engine-fc7d5c445-bw7hf</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1698784607697</td>
      <td>{'variable': [-4.4882298e-05]}</td>
      <td>engine-fc7d5c445-bw7hf</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1698784607697</td>
      <td>{'variable': [-9.36985e-05]}</td>
      <td>engine-fc7d5c445-bw7hf</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1698784607697</td>
      <td>{'variable': [-8.3208084e-05]}</td>
      <td>engine-fc7d5c445-bw7hf</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1698784607697</td>
      <td>{'variable': [-8.332728999999999e-05]}</td>
      <td>engine-fc7d5c445-bw7hf</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1698784607697</td>
      <td>{'variable': [0.0004896521599999999]}</td>
      <td>engine-fc7d5c445-bw7hf</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1698784607697</td>
      <td>{'variable': [0.0006609559]}</td>
      <td>engine-fc7d5c445-bw7hf</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1698784607697</td>
      <td>{'variable': [7.57277e-05]}</td>
      <td>engine-fc7d5c445-bw7hf</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1698784607697</td>
      <td>{'variable': [-0.000100553036]}</td>
      <td>engine-fc7d5c445-bw7hf</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1698784607697</td>
      <td>{'variable': [-0.0005198717]}</td>
      <td>engine-fc7d5c445-bw7hf</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1698784607697</td>
      <td>{'variable': [-3.695488e-06]}</td>
      <td>engine-fc7d5c445-bw7hf</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1698784607697</td>
      <td>{'variable': [-0.00010883808]}</td>
      <td>engine-fc7d5c445-bw7hf</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1698784607697</td>
      <td>{'variable': [-0.00017666817]}</td>
      <td>engine-fc7d5c445-bw7hf</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1698784607697</td>
      <td>{'variable': [-2.8312206e-05]}</td>
      <td>engine-fc7d5c445-bw7hf</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1698784607697</td>
      <td>{'variable': [2.1755695e-05]}</td>
      <td>engine-fc7d5c445-bw7hf</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1698784607697</td>
      <td>{'variable': [-8.493661999999999e-05]}</td>
      <td>engine-fc7d5c445-bw7hf</td>
    </tr>
  </tbody>
</table>

### Undeploy the Pipeline

With the testing complete, we can undeploy the pipeline.  Note that deploying and undeploying a pipeline is not required for publishing a pipeline to the Edge Registry - this is done just for this demonstration.

```python
pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s .................................... ok

<table><tr><th>name</th> <td>edge-pipeline</td></tr><tr><th>created</th> <td>2023-10-31 20:36:33.486078+00:00</td></tr><tr><th>last_updated</th> <td>2023-10-31 20:36:33.629569+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>932e9cd2-978e-40e6-ab84-2ed5e5d63408, a5752acc-9334-4cac-8b34-471979b55d61</td></tr><tr><th>steps</th> <td>ccfraud</td></tr><tr><th>published</th> <td>False</td></tr></table>

### Publish the Pipeline for Edge Deployment

It worked! For a demo, we'll take working once as "tested". So now that we've tested our pipeline, we are ready to publish it for edge deployment.

Publishing it means assembling all of the configuration files and model assets and pushing them to an Open Container Initiative (OCI) repository set in the Wallaroo instance as the Edge Registry service.  DevOps engineers then retrieve that image and deploy it through Docker, Kubernetes, or similar deployments.

See [Edge Deployment Registry Guide](https://staging.docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-edge-deployment/) for details on adding an OCI Registry Service to Wallaroo as the Edge Deployment Registry.

This is done through the SDK command `wallaroo.pipeline.publish(deployment_config)` which has the following parameters and returns.

### Publish a Pipeline Endpoint

Pipelines are published as images to the edge registry set in the [Enable Wallaroo Edge Registry](#enable-wallaroo-edge-registry) through the following endpoint.

* Endpoint:
  * `/v1/api/pipelines/publish`
* Parameters
  * **pipeline_id** (*Integer* *Required*): The numerical id of the pipeline to publish to the edge registry.
  * **pipeline_version_id** (*Integer* *Required*): The numerical id of the pipeline's version to publish to the edge registry.
  * **model_config_ids** (*List* *Optional*): The list of model config ids to include.
* Returns
  * id (*Integer*): Numerical Wallaroo id of the published pipeline.
  * pipeline_version_id (*Integer*): Numerical Wallaroo id of the pipeline version published.
  * status: The status of the pipeline publish.  Values include:
    * PendingPublish: The pipeline publication is about to be uploaded or is in the process of being uploaded.
    * Published:  The pipeline is published and ready for use.
  * created_at (*String*): When the published pipeline was created.
  * updated_at (*String*): When the published pipeline was updated.
  * created_by (*String*): The email address of the Wallaroo user that created the pipeline publish.
  * pipeline_url (*String*): The URL of the published pipeline in the edge registry.  May be `null` until the status is `Published`)
  * engine_url (*String*): The URL of the published pipeline engine in the edge registry.  May be `null` until the status is `Published`.
   * **helm** (*String*): The helm chart, helm reference and helm version.
  * engine_config (*`wallaroo.deployment_config.DeploymentConfig`) | The pipeline configuration included with the published pipeline.

#### Publish Example

We will now publish the pipeline to our Edge Deployment Registry with the `pipeline.publish(deployment_config)` command.  `deployment_config` is an optional field that specifies the pipeline deployment.  This can be overridden by the DevOps engineer during deployment.  For this demonstration, we will be retrieving the most recent pipeline version.

For this, we will require the pipeline version id, the workspace id, and the model config ids (which will be empty as not required).

```python
# get the pipeline version
pipeline_version_id = pipeline.versions()[-1].id()
display(pipeline_version_id)

pipeline_id = pipeline.id()
display(pipeline_id)
headers = {
    "Authorization": wl.auth._bearer_token_str(),
    "Content-Type": "application/json",
}

url = f"{wl.api_endpoint}/v1/api/pipelines/publish"
response = requests.post(
    url,
    headers=headers,
    json={"pipeline_id": 1, 
          "pipeline_version_id": pipeline_version_id, 
          "model_config_ids": [edge_demo_model.id()]
          }
    )
display(response.json())
```

    7

    7

    {'id': 3,
     'pipeline_version_id': 7,
     'pipeline_version_name': 'a5752acc-9334-4cac-8b34-471979b55d61',
     'status': 'PendingPublish',
     'created_at': '2023-10-31T20:37:25.516351+00:00',
     'updated_at': '2023-10-31T20:37:25.516351+00:00',
     'created_by': 'b3deff28-04d0-41b8-a04f-b5cf610d6ce9',
     'pipeline_url': None,
     'engine_url': None,
     'helm': None,
     'engine_config': {'engine': {'resources': {'limits': {'cpu': 4.0,
         'memory': '3Gi'},
        'requests': {'cpu': 4.0, 'memory': '3Gi'}}},
      'enginelb': {'resources': {'limits': {'cpu': 1.0, 'memory': '512Mi'},
        'requests': {'cpu': 0.2, 'memory': '512Mi'}}},
      'engineAux': {}},
     'error': None,
     'user_images': [],
     'docker_run_variables': {}}

```python
publish_id = response.json()['id']
display(publish_id)
```

    3

### Get Publish Status

The status of a created Wallaroo pipeline publish is available through the following endpoint.

* Endpoint:
  * `/v1/api/pipelines/get_publish_status`
* Parameters
  * **publish_id** (*Integer* *Required*): The numerical id of the pipeline **publish** to retrieve.
* Returns
  * id (*Integer*): Numerical Wallaroo id of the published pipeline.
  * pipeline_version_id (*Integer*): Numerical Wallaroo id of the pipeline version published.
  * status: The status of the pipeline publish.  Values include:
    * PendingPublish: The pipeline publication is about to be uploaded or is in the process of being uploaded.
    * Published:  The pipeline is published and ready for use.
  * created_at (*String*): When the published pipeline was created.
  * updated_at (*String*): When the published pipeline was updated.
  * created_by (*String*): The email address of the Wallaroo user that created the pipeline publish.
  * pipeline_url (*String*): The URL of the published pipeline in the edge registry.  May be `null` until the status is `Published`)
  * engine_url (*String*): The URL of the published pipeline engine in the edge registry.  May be `null` until the status is `Published`.
  * **helm**:  The Helm chart information including the following fields:
    * **reference** (*String*): The Helm reference.
    * **chart** (*String*): The Helm chart URL.
    * **version** (*String*): The Helm chart version.
  * engine_config (*`wallaroo.deployment_config.DeploymentConfig`) | The pipeline configuration included with the published pipeline.

#### Get Publish Status Example

The following example shows retrieving the status of a recently created pipeline publish.  Once the published status is `Published` we know the pipeline is ready for deployment.

```python
headers = {
    "Authorization": wl.auth._bearer_token_str(),
    "Content-Type": "application/json",
}
url = f"{wl.api_endpoint}/v1/api/pipelines/get_publish_status"
response = requests.post(
    url,
    headers=headers,
    json={
        "id": publish_id
        },
)
display(response.json())
pub = response.json()
```

    {'id': 3,
     'pipeline_version_id': 7,
     'pipeline_version_name': 'a5752acc-9334-4cac-8b34-471979b55d61',
     'status': 'Publishing',
     'created_at': '2023-10-31T20:37:25.516351+00:00',
     'updated_at': '2023-10-31T20:37:25.516351+00:00',
     'created_by': 'b3deff28-04d0-41b8-a04f-b5cf610d6ce9',
     'pipeline_url': None,
     'engine_url': None,
     'helm': None,
     'engine_config': {'engine': {'resources': {'limits': {'cpu': 4.0,
         'memory': '3Gi'},
        'requests': {'cpu': 4.0, 'memory': '3Gi'}}},
      'enginelb': {'resources': {'limits': {'cpu': 1.0, 'memory': '512Mi'},
        'requests': {'cpu': 0.2, 'memory': '512Mi'}}},
      'engineAux': {}},
     'error': None,
     'user_images': [],
     'docker_run_variables': {}}

### List Publishes for a Pipeline

A list of publishes created for a specific pipeline is retrieved through the following endpoint.

* Endpoint:
  * `/v1/api/pipelines/list_publishes_for_pipeline`
* Parameters
  * **publish_id** (*Integer* *Required*): The numerical id of the pipeline to retrieve all publishes for.
* Returns a List of pipeline publishes with the following fields:
  * **id** (*Integer*): Numerical Wallaroo id of the published pipeline.
  * **pipeline_version_id** (*Integer*): Numerical Wallaroo id of the pipeline version published.
  * **status**: The status of the pipeline publish.  Values include:
    * `PendingPublish`: The pipeline publication is about to be uploaded or is in the process of being uploaded.
    * `Published`:  The pipeline is published and ready for use.
  * **created_at** (*String*): When the published pipeline was created.
  * **updated_at** (*String*): When the published pipeline was updated.
  * **created_by** (*String*): The email address of the Wallaroo user that created the pipeline publish.
  * **pipeline_url** (*String*): The URL of the published pipeline in the edge registry.  May be `null` until the status is `Published`)
  * **engine_url** (*String*): The URL of the published pipeline engine in the edge registry.  May be `null` until the status is `Published`.
  * **helm**:  The Helm chart information including the following fields:
    * **reference** (*String*): The Helm reference.
    * **chart** (*String*): The Helm chart URL.
    * **version** (*String*): The Helm chart version.
  * **engine_config** (*`wallaroo.deployment_config.DeploymentConfig`) | The pipeline configuration included with the published pipeline.

#### List Publishes for a Pipeline Example

```python
headers = {
    "Authorization": wl.auth._bearer_token_str(),
    "Content-Type": "application/json",
}

url = f"{wl.api_endpoint}/v1/api/pipelines/list_publishes_for_pipeline"

response = requests.post(
    url,
    headers=headers,
    json={"pipeline_id": pipeline_id},
)
display(response.json())
```

    {'publishes': [{'id': 3,
       'pipeline_version_id': 7,
       'pipeline_version_name': 'a5752acc-9334-4cac-8b34-471979b55d61',
       'status': 'Publishing',
       'created_at': '2023-10-31T20:37:25.516351+00:00',
       'updated_at': '2023-10-31T20:37:25.516351+00:00',
       'created_by': 'b3deff28-04d0-41b8-a04f-b5cf610d6ce9',
       'pipeline_url': None,
       'engine_url': None,
       'helm': None,
       'engine_config': {'engine': {'resources': {'limits': {'cpu': 4.0,
           'memory': '3Gi'},
          'requests': {'cpu': 4.0, 'memory': '3Gi'}}},
        'enginelb': {'resources': {'limits': {'cpu': 1.0, 'memory': '512Mi'},
          'requests': {'cpu': 0.2, 'memory': '512Mi'}}},
        'engineAux': {}},
       'error': None,
       'user_images': [],
       'docker_run_variables': {}}],
     'edges': []}

## DevOps Deployment

We now have our pipeline published to our Edge Registry service.  We can deploy this in a x86 environment running Docker that is logged into the same registry service that we deployed to.

For more details, check with the documentation on your artifact service.  The following are provided for the three major cloud services:

* [Set up authentication for Docker](https://cloud.google.com/artifact-registry/docs/docker/authentication)
* [Authenticate with an Azure container registry](https://learn.microsoft.com/en-us/azure/container-registry/container-registry-authentication?tabs=azure-cli)
* [Authenticating Amazon ECR Repositories for Docker CLI with Credential Helper](https://aws.amazon.com/blogs/compute/authenticating-amazon-ecr-repositories-for-docker-cli-with-credential-helper/)

## Add Edge Location

Wallaroo Servers can optionally connect to the Wallaroo Ops instance and transmit their inference results.  These are added to the pipeline logs for the published pipeline the Wallaroo Server is associated with.

### Add Publish Edge

Edges are added to an existing pipeline publish with the following endpoint.

* Endpoint
  * `/v1/api/pipelines/add_edge_to_publish`
* Parameters
  * **name** (*String* *Required*): The name of the edge.  This **must** be a unique value across all edges in the Wallaroo instance.
  * **pipeline_publish_id** (*Integer* *Required*): The numerical identifier of the pipeline publish to add this edge to.
  * **tags** (*List(String)* *Required*): A list of optional tags.
* Returns
  * **id** (*Integer*): The integer ID of the pipeline publish.
  * **created_at**: (*String*): The DateTime of the pipeline publish.
  * **docker_run_variables** (*String*) The Docker variables in base64 encoded format that include the following: The `BUNDLE_VERSION`, `EDGE_NAME`, `JOIN_TOKEN_`, `OPSCENTER_HOST`, `PIPELINE_URL`, and `WORKSPACE_ID`.
  * **engine_config** (*String*): The Wallaroo `wallaroo.deployment_config.DeploymentConfig` for the pipeline.
  * **pipeline_version_id** (*Integer*): The integer identifier of the pipeline version published.
  * **status** (*String*): The status of the publish.  `Published` is a successful publish.
  * **updated_at** (*DateTime*): The DateTime when the pipeline publish was updated.
  * **user_images** (*List(String)*):  User images used in the pipeline publish.
  * **created_by** (*String*): The UUID of the Wallaroo user that created the pipeline publish.
  * **engine_url** (*String*): The URL for the published pipeline's Wallaroo engine in the OCI registry.
  * **error** (*String*): Any errors logged.
  * **helm** (*String*): The helm chart, helm reference and helm version.
  * **pipeline_url** (*String*): The URL for the published pipeline's container in the OCI registry.
  * **pipeline_version_name** (*String*): The UUID identifier of the pipeline version published.
  * **additional_properties** (*String*): Any additional properties.

```python
headers = {
    "Authorization": wl.auth._bearer_token_str(),
    "Content-Type": "application/json",
}

edge_name = f"ccfraud-observability-api{suffix_random}"

url = f"{wl.api_endpoint}/v1/api/pipelines/add_edge_to_publish"

response = requests.post(
    url,
    headers=headers,
    json={
        "pipeline_publish_id": publish_id,
        "name": edge_name,
        "tags": []
    }
)
display(response.json())
```

    {'id': 3,
     'pipeline_version_id': 7,
     'pipeline_version_name': 'a5752acc-9334-4cac-8b34-471979b55d61',
     'status': 'Published',
     'created_at': '2023-10-31T20:37:25.516351+00:00',
     'updated_at': '2023-10-31T20:37:25.516351+00:00',
     'created_by': 'b3deff28-04d0-41b8-a04f-b5cf610d6ce9',
     'pipeline_url': 'us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/pipelines/edge-pipeline:a5752acc-9334-4cac-8b34-471979b55d61',
     'engine_url': 'us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/engines/proxy/wallaroo/ghcr.io/wallaroolabs/standalone-mini:v2023.4.0-4092',
     'helm': {'reference': 'us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/charts@sha256:4197aaf0beccfc7f935e93c60338aa043d84dddaf4deaeadb000a5cdb4c7ab33',
      'chart': 'us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/charts/edge-pipeline',
      'version': '0.0.1-a5752acc-9334-4cac-8b34-471979b55d61',
      'values': {'edgeBundle': 'ZXhwb3J0IEJVTkRMRV9WRVJTSU9OPTEKZXhwb3J0IEVER0VfTkFNRT1jY2ZyYXVkLW9ic2VydmFiaWxpdHktYXBpbnZ3aApleHBvcnQgSk9JTl9UT0tFTj1jMmFmNjgxNC0yZGI4LTRmNDgtOGJmMS03NzIwMTkxODhhMzEKZXhwb3J0IE9QU0NFTlRFUl9IT1NUPXByb2R1Y3QtdWF0LWVlLmVkZ2Uud2FsbGFyb29jb21tdW5pdHkubmluamEKZXhwb3J0IFBJUEVMSU5FX1VSTD11cy1jZW50cmFsMS1kb2NrZXIucGtnLmRldi93YWxsYXJvby1kZXYtMjUzODE2L3VhdC9waXBlbGluZXMvZWRnZS1waXBlbGluZTphNTc1MmFjYy05MzM0LTRjYWMtOGIzNC00NzE5NzliNTVkNjEKZXhwb3J0IFdPUktTUEFDRV9JRD0yMQ=='}},
     'engine_config': {'engine': {'resources': {'limits': {'cpu': 4.0,
         'memory': '3Gi'},
        'requests': {'cpu': 4.0, 'memory': '3Gi'}}},
      'enginelb': {'resources': {'limits': {'cpu': 1.0, 'memory': '512Mi'},
        'requests': {'cpu': 0.2, 'memory': '512Mi'}}},
      'engineAux': {}},
     'error': None,
     'user_images': [],
     'docker_run_variables': {'EDGE_BUNDLE': 'abcde'}}

```python
# used for deployment later
edge_location_publish=response.json()
```

### Remove Edge from Publish

Edges are removed from an existing pipeline publish with the following endpoint.

* Endpoint
  * `/v1/api/pipelines/remove_edge_from_publish`
* Parameters
  * **name** (*String* *Required*): The name of the edge.  This **must** be a unique value across all edges in the Wallaroo instance.
* Returns
  * null

```python
# adding another edge to remove

headers = {
    "Authorization": wl.auth._bearer_token_str(),
    "Content-Type": "application/json",
}

edge_name2 = f"ccfraud-observability-api2{suffix_random}"

url = f"{wl.api_endpoint}/v1/api/pipelines/add_edge_to_publish"

response = requests.post(
    url,
    headers=headers,
    json={
        "pipeline_publish_id": publish_id,
        "name": edge_name2,
        "tags": []
    }
)
display(response.json())
```

    {'id': 3,
     'pipeline_version_id': 7,
     'pipeline_version_name': 'a5752acc-9334-4cac-8b34-471979b55d61',
     'status': 'Published',
     'created_at': '2023-10-31T20:37:25.516351+00:00',
     'updated_at': '2023-10-31T20:37:25.516351+00:00',
     'created_by': 'b3deff28-04d0-41b8-a04f-b5cf610d6ce9',
     'pipeline_url': 'us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/pipelines/edge-pipeline:a5752acc-9334-4cac-8b34-471979b55d61',
     'engine_url': 'us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/engines/proxy/wallaroo/ghcr.io/wallaroolabs/standalone-mini:v2023.4.0-4092',
     'helm': {'reference': 'us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/charts@sha256:4197aaf0beccfc7f935e93c60338aa043d84dddaf4deaeadb000a5cdb4c7ab33',
      'chart': 'us-central1-docker.pkg.dev/wallaroo-dev-253816/uat/charts/edge-pipeline',
      'version': '0.0.1-a5752acc-9334-4cac-8b34-471979b55d61',
      'values': {'edgeBundle': 'ZXhwb3J0IEJVTkRMRV9WRVJTSU9OPTEKZXhwb3J0IEVER0VfTkFNRT1jY2ZyYXVkLW9ic2VydmFiaWxpdHktYXBpMm52d2gKZXhwb3J0IEpPSU5fVE9LRU49NDZhODVmZGYtMzYwZC00MjUxLTk2NTEtNTBkYmRiNDFkNTA1CmV4cG9ydCBPUFNDRU5URVJfSE9TVD1wcm9kdWN0LXVhdC1lZS5lZGdlLndhbGxhcm9vY29tbXVuaXR5Lm5pbmphCmV4cG9ydCBQSVBFTElORV9VUkw9dXMtY2VudHJhbDEtZG9ja2VyLnBrZy5kZXYvd2FsbGFyb28tZGV2LTI1MzgxNi91YXQvcGlwZWxpbmVzL2VkZ2UtcGlwZWxpbmU6YTU3NTJhY2MtOTMzNC00Y2FjLThiMzQtNDcxOTc5YjU1ZDYxCmV4cG9ydCBXT1JLU1BBQ0VfSUQ9MjE='}},
     'engine_config': {'engine': {'resources': {'limits': {'cpu': 4.0,
         'memory': '3Gi'},
        'requests': {'cpu': 4.0, 'memory': '3Gi'}}},
      'enginelb': {'resources': {'limits': {'cpu': 1.0, 'memory': '512Mi'},
        'requests': {'cpu': 0.2, 'memory': '512Mi'}}},
      'engineAux': {}},
     'error': None,
     'user_images': [],
     'docker_run_variables': {'EDGE_BUNDLE': 'abcde'}}

```python
edge_name2
```

    'ccfraud-observability-api2nvwh'

```python
# remove the edge

headers = {
    "Authorization": wl.auth._bearer_token_str(),
    "Content-Type": "application/json",
}

edge_name2 = f"ccfraud-observability-api2{suffix_random}"

url = f"{wl.api_endpoint}/v1/api/pipelines/remove_edge"

response = requests.post(
    url,
    headers=headers,
    json={
        "name": "ccfraud-observability-api2nvwh",
    }
)
display(response)
```

    <Response [200]>

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

```bash
docker run -p 8080:8080 \
    -e DEBUG=true -e OCI_REGISTRY={your registry server} \
    -e CONFIG_CPUS=1 \
    -e OCI_USERNAME=oauth2accesstoken \
    -e OCI_PASSWORD={registry token here} \
    -e PIPELINE_URL={your registry server}/pipelines/edge-cv-retail:bf70eaf7-8c11-4b46-b751-916a43b1a555 \
    {your registry server}/engine:v2023.3.0-main-3707
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

The following generates a `docker run` command based on the added edge location example above.  Replace `$REGISTRYURL`, `$REGISTRYUSERNAME` and `$REGISTRYPASSWORD` with the appropriate values.

```python
docker_command = f'''
docker run -p 8080:8080 \\
    -e DEBUG=true \\
    -e OCI_REGISTRY=$REGISTRYURL \\
    -e EDGE_BUNDLE={edge_location_publish['docker_run_variables']['EDGE_BUNDLE']} \\
    -e CONFIG_CPUS=1 \\
    -e OCI_USERNAME=$REGISTRYUSERNAME \\
    -e OCI_PASSWORD=$REGISTRYPASSWORD \\
    -e PIPELINE_URL={edge_location_publish['pipeline_url']} \\
    {edge_location_publish['engine_url']}
'''

print(docker_command)

```

## Edge Deployed Pipeline API Endpoints

Once deployed, we can check the pipelines and models available.  We'll use a `curl` command, but any HTTP based request will work the same way.

The endpoint `/pipelines` returns:

* **id** (*String*):  The name of the pipeline.
* **status** (*String*):  The status as either `Running`, or `Error` if there are any issues.

For this example, the deployment is made on a machine called `testboy.local`.  Replace this URL with the URL of you edge deployment.

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
!curl -X POST testboy.local:8080/pipelines/edge-pipeline \
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
    file_prefix='edge-logs-api',
    dataset=['time', 'out', 'metadata']
)
```

```python
# display the head 20 results

df_logs = pd.read_json('./partition-edge-observability/edge-logs-api-1.json', orient="records", lines=True)
# get just the partition
# df_results['partition'] = df_results['metadata'].map(lambda x: x['partition'])
# display(df_results.head(20))
display(df_logs.head(20).loc[:, ['time', 'out.variable', 'metadata.partition']])
```

```python
display(pd.unique(df_logs['metadata.partition']))
```
