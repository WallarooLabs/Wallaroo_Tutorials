This tutorial is available on the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/pipeline-architecture/wallaroo-arm-classification-finserv-api).

## Classification Financial Services with Arm Architecture via the Wallaroo MLOps API

This tutorial demonstrates how to use the Wallaroo combined with ARM processors to perform inferences with pre-trained classification financial services ML models.  For this demonstration, we will focus on the using the [Wallaroo MLOps API](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/) to deploy the pipeline with the specified pipeline deployments, alternating between x86 architecture and ARM architecture.

* A Wallaroo version 2023.3 or above instance is installed.
* A nodepools with ARM architecture virtual machines are part of the Kubernetes cluster.  For example, Azure supports Ampere® Altra® Arm-based processor included with the following virtual machines:
  * [Dpsv5 and Dpdsv5-series](https://learn.microsoft.com/en-us/azure/virtual-machines/dpsv5-dpdsv5-series)
  * [Epsv5 and Epdsv5-series](https://learn.microsoft.com/en-us/azure/virtual-machines/epsv5-epdsv5-series)

In this notebook we will the example model and sample data from the Machine Learning Group's demonstration on [Credit Card Fraud detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

### Tutorial Goals

For our example, we will perform the following:

* Create a workspace for our work.
* Upload the CCFraud model.
* Create a pipeline using the default architecture that can ingest our submitted data, submit it to the model, and export the results while tracking how long the inference took.
* Redeploy the same pipeline on the ARM architecture, then perform the same inference on the same data and model and track how long the inference took.
* Compare the inference timing through the default architecture versus the ARM architecture.

```python
import wallaroo
from wallaroo.object import EntityNotFoundError

import pyarrow as pa

from wallaroo.framework import Framework

# used to display dataframe information without truncating
from IPython.display import display
import pandas as pd
pd.set_option('display.max_colwidth', None)
import time

import requests
```

### Connect to the Wallaroo Instance through the User Interface

The next step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
# Login through local Wallaroo instance

wl = wallaroo.Client()
```

### Create Workspace

Now we'll use the SDK below to create our workspace , assign as our **current workspace**, then display all of the workspaces we have at the moment.  We'll also set up for our models and pipelines down the road, so we have one spot to change names to whatever fits your organization's standards best.

To allow this tutorial to be run multiple times or by multiple users in the same Wallaroo instance, a random 4 character prefix will be added to the workspace.  Feel free to change this `suffix` variable to `''` if not required.

When we create our new workspace, we'll save it in the Python variable `workspace` so we can refer to it as needed.

For more information, see the [Wallaroo SDK Essentials Guide: Workspace Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-workspace/).

```python
import string
import random

# make a random 4 character prefix
suffix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))

suffix='john'

workspace_name = f'arm-classification-finserv{suffix}'
pipeline_name = 'arm-classification-example'
model_name = 'ccfraudmodel'
model_file_name = './models/ccfraud.onnx'
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

    {'name': 'arm-classification-finservjohn', 'id': 49, 'archived': False, 'created_by': '6d75da2e-3913-4acd-b1bb-06dd1eb3d0df', 'created_at': '2023-08-29T15:23:58.056984+00:00', 'models': [], 'pipelines': []}

## Upload a model

Our workspace is created.  Let's upload our credit card fraud model to it.  This is the file name `ccfraud.onnx`, and we'll upload it as `ccfraudmodel`.  The credit card fraud model is trained to detect credit card fraud based on a 0 to 1 model:  The closer to 0 the less likely the transactions indicate fraud, while the closer to 1 the more likely the transactions indicate fraud.

For more information, see the [Wallaroo SDK Essentials Guide: Model Uploads and Registrations](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/).

```python
ccfraud_model = (wl.upload_model(model_name, 
                                 model_file_name, 
                                 wallaroo.framework.Framework.ONNX)
                                 .configure()
                )
```

We can verify that our model was uploaded by listing the models uploaded to our Wallaroo instance with the `list_models()` command.  Note that since we uploaded this model before, we now have different versions of it we can use for our testing.

### Create a Pipeline

With our model uploaded, time to create our pipeline and deploy it so it can accept data and process it through our `ccfraudmodel`.  We'll call our pipeline `ccfraudpipeline`.

* **NOTE**:  Pipeline names must be unique.  If two pipelines are assigned the same name, the new pipeline is created as a new **version** of the pipeline.

For more information, see the [Wallaroo SDK Essentials Guide: Pipeline Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline/)

```python
pipeline = get_pipeline(pipeline_name)
pipeline.clear()
```

<table><tr><th>name</th> <td>arm-classification-example</td></tr><tr><th>created</th> <td>2023-08-29 15:24:03.060543+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-29 15:24:03.060543+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>6ed66747-f597-459c-bd37-bfe8329e619f</td></tr><tr><th>steps</th> <td></td></tr><tr><th>published</th> <td>False</td></tr></table>

Now our pipeline is set.  Let's add a single **step** to it - in this case, our `ccfraud_model` that we uploaded to our workspace.

```python
pipeline.add_model_step(ccfraud_model)
```

<table><tr><th>name</th> <td>arm-classification-example</td></tr><tr><th>created</th> <td>2023-08-29 15:24:03.060543+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-29 15:24:03.060543+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>6ed66747-f597-459c-bd37-bfe8329e619f</td></tr><tr><th>steps</th> <td></td></tr><tr><th>published</th> <td>False</td></tr></table>

### Set Standard Pipeline Deployment Architecture

Now we can set our pipeline deployment architecture with the `arch(wallaroo.engine_config.Architecture)` parameter.  By default, the deployment configuration architecture default to `wallaroo.engine_config.Architecture.X86`.

For this example, we will create a pipeline deployment and leave the `arch` out so it will default to `X86`.

This deployment will be applied to the pipeline deployment.

For more information, see the [Wallaroo SDK Essentials Guide: Pipeline Deployment Configuration](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline-deployment-config/).

```python

deployment_config = (wallaroo.deployment_config
                     .DeploymentConfigBuilder()
                     .cpus(1)
                     .memory('1Gi')
                     .build()
                     )
deployment_config
```

    {'engine': {'cpu': 1,
      'resources': {'limits': {'cpu': 1, 'memory': '1Gi'},
       'requests': {'cpu': 1, 'memory': '1Gi'}}},
     'enginelb': {},
     'engineAux': {'images': {}},
     'node_selector': {}}

### Deploy Pipeline via the Wallaroo MLOps API

For this tutorial, we will switch over to the Wallaroo MLOps API to deploy the pipeline.  This is through the following endpoint:

### Deploy a Pipeline

Deploy a an existing pipeline.  Note that for any pipeline that has model steps, they must be included either in `model_configs`, `model_ids` or `models`.

* **Endpoint**
  * `/pipelines/deploy`
* **Parameters**
  * **deploy_id** (*REQUIRED string*): The name for the pipeline deployment.
  * **engine_config** (*OPTIONAL string*): Additional configuration options for the pipeline.  These set the memory, replicas, and other settings.  For example: `{"cpus": 1, "replica_count": 1, "memory": "999Mi"}` Available parameters include the following.
    * `cpus`: The number of CPUs to apply to the native runtime models in the pipeline.  `cpus` can be a fraction of a cpu, for example `"cpus": 0.25`.
    * `gpus`:  The number of GPUs to apply to the native runtime models.  GPUs can only be allocated in whole numbers.  Organizations should monitor how many GPUs are allocated to a pipelines to verify they have enough GPUs for all pipelines.
    * `arch`: The architecture to use.  The available values are:
      * `x64` (**Default**): x86 architecture.
      * `arm`: ARM based architecture such as Ampere® Altra® Arm-based processor included with the following Azure virtual machines:
        * [Dpsv5 and Dpdsv5-series](https://learn.microsoft.com/en-us/azure/virtual-machines/dpsv5-dpdsv5-series)
        * [Epsv5 and Epdsv5-series](https://learn.microsoft.com/en-us/azure/virtual-machines/epsv5-epdsv5-series)
    * `replica_count`: The number of replicas of the pipeline to deploy. This allows for multiple deployments of the same models to be deployed to increase inferences through parallelization.
    * `replica_autoscale_min_max`: Provides replicas to be scaled from 0 to some maximum number of replicas. This allows pipelines to spin up additional replicas as more resources are required, then spin them back down to save on resources and costs.  For example:  `"replica_autoscale_min_max": {"maximum": 2, "minimum":0}`
    * `autoscale_cpu_utilization`: Sets the average CPU percentage metric for when to load or unload another replica.
    * `disable_autoscale`: Disables autoscaling in the deployment configuration.
    * `memory`: Sets the amount of RAM to allocate the pipeline. The memory_spec string is in the format “{size as number}{unit value}”. The accepted unit values are:
      * `KiB` (for KiloBytes)
      * `MiB` (for MegaBytes)
      * `GiB` (for GigaBytes)
      * `TiB` (for TeraBytes)
    * `lb_cpus`: Sets the number or fraction of CPUs to use for the pipeline’s load balancer, for example: 0.25, 1, 1.5, etc. The units, similar to the Kubernetes CPU definitions.
    * `lb_memory`: Sets the amount of RAM to allocate the pipeline’s load balancer. The memory_spec string is in the format “{size as number}{unit value}”. The accepted unit values are:
      * `KiB` (for KiloBytes)
      * `MiB` (for MegaBytes)
      * `GiB` (for GigaBytes)
      * `TiB` (for TeraBytes)
    * `deployment_label`: Label used for Kubernetes labels.
    * `sidekick_cpus`:  Sets the number of CPUs to be used for the model’s sidekick container. Only affects image-based models (e.g. MLFlow models) in a deployment. The parameters are as follows:
      * `model`: The sidekick model to configure.
      * `core_count`: Sets the number or fraction of CPUs to use.
    * `sidekick_memory`: Sets the memory available to for the model’s sidekick container. Only affects image-based models (e.g. MLFlow models) in a deployment. The parameters are as follows:
      * `model`: The sidekick model to configure.
      * `memory_spec`: The amount of memory to allocated as memory unit values. The accepted unit values are:
        * `KiB` (for KiloBytes)
        * `MiB` (for MegaBytes)
        * `GiB` (for GigaBytes)
        * `TiB` (for TeraBytes)
    * `sidekick_env`: Environment variables submitted to the model’s sidekick container. Only affects image-based models (e.g. MLFlow models) in a deployment. These are used specifically for containerized models that have environment variables that effect their performance.  This takes the following parameters:
      * `model`: The sidekick model to configure.
      * `environment`: Dictionary inputs
    * `sidekick_gpus`: Sets the number of GPUs to allocate for containerized runtimes. GPUs are only allocated in whole units, not as fractions. Organizations should be aware of the total number of GPUs available to the cluster, and monitor which pipeline deployment configurations have GPUs allocated to ensure they do not run out. If there are not enough GPUs to allocate to a pipeline deployment configuration, and error message will be deployed when the pipeline is deployed.  This takes the following parameters:
      * `model`: The sidekick model to configure.
      * `core_count`: The number of GPUs to allocate.
    * `sidekick_arch`: The architecture to use for containerized runtimes.  The available values are:
      * `x64` (**Default**): x86 architecture.
      * `arm`: ARM based architecture such as Ampere® Altra® Arm-based processor included with the following Azure virtual machines:
        * [Dpsv5 and Dpdsv5-series](https://learn.microsoft.com/en-us/azure/virtual-machines/dpsv5-dpdsv5-series)
        * [Epsv5 and Epdsv5-series](https://learn.microsoft.com/en-us/azure/virtual-machines/epsv5-epdsv5-series)
  * **pipeline_version_pk_id** (*REQUIRED int*): Pipeline version id.
  * **model_configs** (*OPTIONAL Array int*): Ids of model configs to apply.
  * **model_ids** (*OPTIONAL Array int*): Ids of models to apply to the pipeline.  If passed in, model_configs will be created automatically.
  * **models** (*OPTIONAL Array models*):  If the model ids are not available as a pipeline step, the models' data can be passed to it through this method.  The options below are only required if `models` are provided as a parameter.
    * **name** (*REQUIRED string*): Name of the uploaded model that is in the same workspace as the pipeline.
    * **version** (*REQUIRED string*): Version of the model to use.
    * **sha** (*REQUIRED string*): SHA value of the model.
  * **pipeline_id** (*REQUIRED int*): Numerical value of the pipeline to deploy.
* **Returns**
  * **id** (*int*): The deployment id.

```python
pipeline_version_id = pipeline.versions()[-1].id()
display(pipeline_version_id)

pipeline_id = pipeline.id()
display(pipeline_id)

model_version_id = ccfraud_model.id()
display(model_version_id)

model_version = ccfraud_model.version()
display(model_version)

model_sha = ccfraud_model.sha()
display(model_sha)

deploy_id = pipeline.name()
display(deploy_id)

display(deployment_config)
```

    33

    33

    12

    '482dba36-b9c2-424d-80bc-487a6c2839dc'

    'bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507'

    'arm-classification-example'

    {'engine': {'cpu': 1,
      'resources': {'limits': {'cpu': 1, 'memory': '1Gi'},
       'requests': {'cpu': 1, 'memory': '1Gi'}}},
     'enginelb': {},
     'engineAux': {'images': {}},
     'node_selector': {}}

```python
headers = {
    "Authorization": wl.auth._bearer_token_str(),
    "Content-Type": "application/json",
}

url = f"{wl.api_endpoint}/v1/api/pipelines/deploy"

data = {
    "deploy_id": deploy_id,
    "pipeline_version_pk_id": pipeline_version_id,
    "engine_config": deployment_config,
    "models": [
        {
            "name":model_name,
            "version":model_version,
            "sha":model_sha
        }
    ],
    "pipeline_id": pipeline_id
}

response = requests.post(
    url,
    headers=headers,
    json=data,
)
display(response.json())

```

    {'id': 14}

We'll switch back to the Wallaroo SDK to display the status of our pipeline.  Once it is "Running" we can continue.

```python
import time
status = None

while status != 'Running':
    print(pipeline.status())
    status = pipeline.status()['status']
    time.sleep(5)
```

    {'status': 'Running', 'details': [], 'engines': [{'ip': '10.244.1.27', 'name': 'engine-674c4785c4-b5xvw', 'status': 'Running', 'reason': None, 'details': [], 'pipeline_statuses': {'pipelines': [{'id': 'arm-classification-example', 'status': 'Running'}]}, 'model_statuses': {'models': [{'name': 'ccfraudmodel', 'version': '482dba36-b9c2-424d-80bc-487a6c2839dc', 'sha': 'bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507', 'status': 'Running'}]}}], 'engine_lbs': [{'ip': '10.244.1.26', 'name': 'engine-lb-584f54c899-d22j2', 'status': 'Running', 'reason': None, 'details': []}], 'sidekicks': []}

### Inference on Standard Architecture

We will now perform an inference on 10,000 records by specifying an Apache Arrow input file, and track the time it takes to perform the inference and display the first 5 results.

For more information, see the [Wallaroo SDK Essentials Guide: Inference Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-inferences/)

```python
start_time = time.time()
result = pipeline.infer_from_file('./data/cc_data_10k.arrow')
end_time = time.time()
x86_time = end_time - start_time

outputs =  result.to_pandas()
display(outputs.head(5))
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.tensor</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-08-29 15:25:17.561</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-08-29 15:25:17.561</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-08-29 15:25:17.561</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-08-29 15:25:17.561</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-08-29 15:25:17.561</td>
      <td>[0.5817662, 0.09788155, 0.15468194, 0.4754102, -0.19788623, -0.45043448, 0.016654044, -0.025607055, 0.09205616, -0.27839172, 0.059329946, -0.019658541, -0.42250833, -0.12175389, 1.5473095, 0.23916228, 0.3553975, -0.76851654, -0.7000849, -0.11900433, -0.3450517, -1.1065114, 0.25234112, 0.020944182, 0.21992674, 0.25406894, -0.04502251, 0.10867739, 0.25471792]</td>
      <td>[0.5817662, 0.09788155, 0.15468194, 0.4754102, -0.19788623, -0.45043448, 0.016654044, -0.025607055, 0.09205616, -0.27839172, 0.059329946, -0.019658541, -0.42250833, -0.12175389, 1.5473095, 0.23916228, 0.3553975, -0.76851654, -0.7000849, -0.11900433, -0.3450517, -1.1065114, 0.25234112, 0.020944182, 0.21992674, 0.25406894, -0.04502251, 0.10867739, 0.25471792]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

### Set Arm Pipeline Deployment Architecture

Now we can set our pipeline deployment architecture with the `arch(wallaroo.engine_config.Architecture)` parameter to `wallaroo.engine_config.Architecture.ARM`, then redeploy our pipeline.  This will allocate ARM based virtual machines with the same configuration as earlier in the number of CPUs and RAM - the only difference will be the underlying architecture.

```python
from wallaroo.engine_config import Architecture
deployment_config_arm = (wallaroo.deployment_config
                         .DeploymentConfigBuilder()
                         .cpus(1)
                         .memory('1Gi')
                         .arch(Architecture.ARM)
                         .build()
                         )
deployment_config_arm
```

    {'engine': {'cpu': 1,
      'resources': {'limits': {'cpu': 1, 'memory': '1Gi'},
       'requests': {'cpu': 1, 'memory': '1Gi'}},
      'arch': 'arm'},
     'enginelb': {},
     'engineAux': {'images': {}},
     'node_selector': {}}

```python
pipeline.undeploy()
headers = {
    "Authorization": wl.auth._bearer_token_str(),
    "Content-Type": "application/json",
}

url = f"{wl.api_endpoint}/v1/api/pipelines/deploy"

data = {
    "deploy_id": deploy_id,
    "pipeline_version_pk_id": pipeline_version_id,
    "engine_config": deployment_config_arm,
    "models": [
        {
            "name":model_name,
            "version":model_version,
            "sha":model_sha
        }
    ],
    "pipeline_id": pipeline_id
}

response = requests.post(
    url,
    headers=headers,
    json=data,
)
display(response.json())

```

    {'id': 14}

```python
import time
status = None

while status != 'Running':
    print(pipeline.status())
    status = pipeline.status()['status']
    time.sleep(5)
```

    {'status': 'Running', 'details': [], 'engines': [{'ip': '10.244.2.3', 'name': 'engine-7845df649-xrkvp', 'status': 'Running', 'reason': None, 'details': [], 'pipeline_statuses': {'pipelines': [{'id': 'arm-classification-example', 'status': 'Running'}]}, 'model_statuses': {'models': [{'name': 'ccfraudmodel', 'version': '482dba36-b9c2-424d-80bc-487a6c2839dc', 'sha': 'bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507', 'status': 'Running'}]}}], 'engine_lbs': [{'ip': '10.244.2.2', 'name': 'engine-lb-584f54c899-pt2jd', 'status': 'Running', 'reason': None, 'details': []}], 'sidekicks': []}

### Inference with ARM

We'll do the exact same inference with the exact same file and display the same results, storing how long it takes to perform the inference under the arm processors.

```python
start_time = time.time()
result = pipeline.infer_from_file('./data/cc_data_10k.arrow')
end_time = time.time()
arm_time = end_time - start_time

outputs =  result.to_pandas()
display(outputs.head(5))
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.tensor</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-08-29 15:26:56.862</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-08-29 15:26:56.862</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-08-29 15:26:56.862</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-08-29 15:26:56.862</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-08-29 15:26:56.862</td>
      <td>[0.5817662, 0.09788155, 0.15468194, 0.4754102, -0.19788623, -0.45043448, 0.016654044, -0.025607055, 0.09205616, -0.27839172, 0.059329946, -0.019658541, -0.42250833, -0.12175389, 1.5473095, 0.23916228, 0.3553975, -0.76851654, -0.7000849, -0.11900433, -0.3450517, -1.1065114, 0.25234112, 0.020944182, 0.21992674, 0.25406894, -0.04502251, 0.10867739, 0.25471792]</td>
      <td>[0.5817662, 0.09788155, 0.15468194, 0.4754102, -0.19788623, -0.45043448, 0.016654044, -0.025607055, 0.09205616, -0.27839172, 0.059329946, -0.019658541, -0.42250833, -0.12175389, 1.5473095, 0.23916228, 0.3553975, -0.76851654, -0.7000849, -0.11900433, -0.3450517, -1.1065114, 0.25234112, 0.020944182, 0.21992674, 0.25406894, -0.04502251, 0.10867739, 0.25471792]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

### Compare Differences

We will now compare the results on the standard architecture versus ARM.  Typically, ARM delivers a 15% improvement on inference times while requireing less power and cost requirements.

```python
display(f"Standard architecture: {x86_time}")
display(f"ARM architecture: {arm_time}")
```

    'Standard architecture: 1.5043368339538574'

    'ARM architecture: 1.1869759559631348'

With our work in the pipeline done, we'll undeploy it to get back our resources from the Kubernetes cluster.  If we keep the same settings we can redeploy the pipeline with the same configuration in the future.

```python
pipeline.undeploy()
```

<table><tr><th>name</th> <td>arm-classification-example</td></tr><tr><th>created</th> <td>2023-08-29 15:24:03.060543+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-29 15:24:03.060543+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>6ed66747-f597-459c-bd37-bfe8329e619f</td></tr><tr><th>steps</th> <td>ccfraudmodel</td></tr><tr><th>published</th> <td>False</td></tr></table>

