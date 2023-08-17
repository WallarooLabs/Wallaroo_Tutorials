This tutorial is available on the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/blob/2023.3.0-updates/pipeline-architecture/wallaroo-arm-classification-cybersecurity).

## Classification Cybersecurity with Arm Architecture

This tutorial demonstrates how to use the Wallaroo combined with ARM processors to perform inferences with pre-trained classification cybersecurity ML models.  This demonstration assumes that:

* A Wallaroo version 2023.3 or above instance is installed.
* A nodepools with ARM architecture virtual machines are part of the Kubernetes cluster.  For example, Azure supports Ampere® Altra® Arm-based processor included with the following virtual machines:
  * [Dpsv5 and Dpdsv5-series](https://learn.microsoft.com/en-us/azure/virtual-machines/dpsv5-dpdsv5-series)
  * [Epsv5 and Epdsv5-series](https://learn.microsoft.com/en-us/azure/virtual-machines/epsv5-epdsv5-series)

In this notebook we will walk through a simple pipeline deployment to inference on a model. For this example we will be using an open source model that uses an [Aloha CNN LSTM model](https://www.researchgate.net/publication/348920204_Using_Auxiliary_Inputs_in_Deep_Learning_Models_for_Detecting_DGA-based_Domain_Names) for classifying Domain names as being either legitimate or being used for nefarious purposes such as malware distribution.

## Tutorial Goals

For our example, we will perform the following:

* Create a workspace for our work.
* Upload the Aloha model.
* Create a pipeline using the default architecture that can ingest our submitted data, submit it to the model, and export the results while tracking how long the inference took.
* Redeploy the same pipeline on the ARM architecture, then perform the same inference on the same data and model and track how long the inference took.
* Compare the inference timing through the default architecture versus the ARM architecture.

All sample data and models are available through the [Wallaroo Quick Start Guide Samples repository](https://github.com/WallarooLabs/quickstartguide_samples).

### Connect to the Wallaroo Instance

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  If logging in externally, update the `wallarooPrefix` and `wallarooSuffix` variables with the proper DNS information.  For more information on Wallaroo DNS settings, see the [Wallaroo DNS Integration Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-dns-guide/).

```python
import wallaroo
from wallaroo.object import EntityNotFoundError

from wallaroo.framework import Framework

# to display dataframe tables
from IPython.display import display
# used to display dataframe information without truncating
import pandas as pd
pd.set_option('display.max_colwidth', None)
import pyarrow as pa
import time
```

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

# make a random 4 character suffix to verify uniqueness in tutorials
suffix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))

suffix='john'

workspace_name = f'arm-classification-security{suffix}'
pipeline_name = 'alohapipeline'
model_name = 'alohamodel'
model_file_name = './models/alohacnnlstm.zip'
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

    {'name': 'arm-classification-securityjohn', 'id': 31, 'archived': False, 'created_by': '26a792cb-25d0-470a-aba2-9b4c4ba373ac', 'created_at': '2023-08-15T20:56:52.658573+00:00', 'models': [{'name': 'alohamodel', 'versions': 4, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 8, 16, 16, 23, 6, 709123, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 8, 15, 20, 57, 8, 117141, tzinfo=tzutc())}], 'pipelines': [{'name': 'alohapipeline', 'create_time': datetime.datetime(2023, 8, 15, 20, 56, 52, 700421, tzinfo=tzutc()), 'definition': '[]'}]}

We can verify the workspace is created the current default workspace with the `get_current_workspace()` command.

```python
wl.get_current_workspace()
```

    {'name': 'arm-classification-securityjohn', 'id': 31, 'archived': False, 'created_by': '26a792cb-25d0-470a-aba2-9b4c4ba373ac', 'created_at': '2023-08-15T20:56:52.658573+00:00', 'models': [{'name': 'alohamodel', 'versions': 4, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 8, 16, 16, 23, 6, 709123, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 8, 15, 20, 57, 8, 117141, tzinfo=tzutc())}], 'pipelines': [{'name': 'alohapipeline', 'create_time': datetime.datetime(2023, 8, 15, 20, 56, 52, 700421, tzinfo=tzutc()), 'definition': '[]'}]}

# Upload the Models

Now we will upload our models.  Note that for this example we are applying the model from a .ZIP file.  The Aloha model is a [protobuf](https://developers.google.com/protocol-buffers) file that has been defined for evaluating web pages, and we will configure it to use data in the `tensorflow` format.

For more information, see the [Wallaroo SDK Essentials Guide: Model Uploads and Registrations](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/).

```python
model = wl.upload_model(model_name, model_file_name, framework=Framework.TENSORFLOW).configure("tensorflow")
```

## Deploy a model

Now that we have a model that we want to use we will create a deployment for it.  We will create a pipeline, then add the model to the pipeline as a pipeline step.

For more information, see the [Wallaroo SDK Essentials Guide: Pipeline Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline/)

```python
aloha_pipeline = get_pipeline(pipeline_name)
aloha_pipeline

aloha_pipeline.add_model_step(model)
```

<table><tr><th>name</th> <td>alohapipeline</td></tr><tr><th>created</th> <td>2023-08-15 21:45:11.378943+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-15 23:02:01.114999+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>6f2e0f93-03de-40f9-9921-2d75a5acbd81, 64fa00ca-bf81-46d0-b319-cdf219bd7ae4, f9af8905-33dd-40c4-b5b6-7ac58b727b20, 25e95e10-7191-490f-a36b-d2041e218d5e, 16f7f894-ec4c-40eb-8604-8496d867c329, 7f029018-8b6a-4fe8-bb50-d65d0fdd3b00, 60c072a8-2493-451b-b174-a026869b8efd, a9bb9165-ef77-4368-84c3-0f255e66215f</td></tr><tr><th>steps</th> <td>alohamodel</td></tr></table>

### Set Standard Pipeline Deployment Architecture

Now we can set our pipeline deployment architecture with the `arch(wallaroo.engine_config.Architecture)` parameter.  By default, the deployment configuration architecture default to `wallaroo.engine_config.Architecture.X86`.

For this example, we will create a pipeline deployment and leave the `arch` out so it will default to `X86`.

This deployment will be applied to the pipeline deployment.

For more information, see the [Wallaroo SDK Essentials Guide: Pipeline Deployment Configuration](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline-deployment-config/).

```python
deployment_config = (wallaroo.deployment_config
                     .DeploymentConfigBuilder()
                     .cpus(4)
                     .memory('8Gi')
                     .build()
                    )
display(deployment_config)
aloha_pipeline.deploy(deployment_config=deployment_config)
```

    {'engine': {'cpu': 4,
      'resources': {'limits': {'cpu': 4, 'memory': '8Gi'},
       'requests': {'cpu': 4, 'memory': '8Gi'}}},
     'enginelb': {},
     'engineAux': {'images': {}},
     'node_selector': {}}

    Waiting for deployment - this will take up to 45s ........ ok

<table><tr><th>name</th> <td>alohapipeline</td></tr><tr><th>created</th> <td>2023-08-15 20:56:52.700421+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-16 16:30:42.841855+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>5f6f1123-b371-40ba-ac44-b1ebb45598f7, 7d331276-d17b-4d54-bd5f-d65b11f2148e, 88f3dc23-41b3-407d-9852-ad6c797c16aa, 996d1e65-dfa3-424c-a996-cb8e88e2087c, 0a5efd20-1b75-4695-be5d-d5f2ed48810d, 18c6c824-1003-45e0-89cd-0e09e8c829f1, 8e367dac-1980-4e72-9fea-620b8120ee3d, e0a49f43-b43e-4251-9d30-ad245b27c1aa, d55825d3-1c47-465d-ab6a-543f271d5a5b, 1d54f862-9404-4aa9-8eea-b7aa77e3f919</td></tr><tr><th>steps</th> <td>alohamodel</td></tr></table>

We can verify that the pipeline is running and list what models are associated with it.

```python
aloha_pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.3.18',
       'name': 'engine-79487c4ccb-s85gt',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'alohapipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'alohamodel',
          'version': 'a4f4560d-72ad-46c3-82d4-9c331c4d3aba',
          'sha': 'd71d9ffc61aaac58c2b1ed70a2db13d1416fb9d3f5b891e5e4e2e97180fe22f8',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.2.10',
       'name': 'engine-lb-584f54c899-m9gjl',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}

### Inference on Standard Architecture

We will now perform an inference on 25,000 records by specifying an Apache Arrow input file, and track the time it takes to perform the inference and display the first 5 results.

For more information, see the [Wallaroo SDK Essentials Guide: Inference Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-inferences/)

```python
startTime = time.time()
result = aloha_pipeline.infer_from_file('./data/data_25k.arrow')
endTime = time.time()
x64_time = endTime-startTime
display(result.to_pandas().loc[:, ["time","out.main"]])
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out.main</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-08-16 16:30:55.891</td>
      <td>[0.997564]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-08-16 16:30:55.891</td>
      <td>[0.99999994]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-08-16 16:30:55.891</td>
      <td>[1.0]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-08-16 16:30:55.891</td>
      <td>[0.9999997]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-08-16 16:30:55.891</td>
      <td>[0.9999989]</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>24949</th>
      <td>2023-08-16 16:30:55.891</td>
      <td>[0.9996881]</td>
    </tr>
    <tr>
      <th>24950</th>
      <td>2023-08-16 16:30:55.891</td>
      <td>[0.99981505]</td>
    </tr>
    <tr>
      <th>24951</th>
      <td>2023-08-16 16:30:55.891</td>
      <td>[0.9999919]</td>
    </tr>
    <tr>
      <th>24952</th>
      <td>2023-08-16 16:30:55.891</td>
      <td>[1.0]</td>
    </tr>
    <tr>
      <th>24953</th>
      <td>2023-08-16 16:30:55.891</td>
      <td>[0.99999803]</td>
    </tr>
  </tbody>
</table>
<p>24954 rows × 2 columns</p>

### Deploy with ARM

We have demonstrated performing our sample inference using a standard pipeline deployment.  Now we will redeploy the same pipeline with the ARM architecture with the Wallaroo deployment setting `wallaroo.engine_config.Architecture.ARM` setting and applying it to the deployment configurations `arch` parameter.

```python
from wallaroo.engine_config import Architecture
deployment_config_arm = (wallaroo.deployment_config
                         .DeploymentConfigBuilder()
                         .cpus(4)
                         .memory('8Gi')
                         .arch(Architecture.ARM)
                         .build()
                        )
display(deployment_config_arm)
aloha_pipeline.deploy(deployment_config = deployment_config_arm)
```

    {'engine': {'cpu': 4,
      'resources': {'limits': {'cpu': 4, 'memory': '8Gi'},
       'requests': {'cpu': 4, 'memory': '8Gi'}},
      'arch': 'arm'},
     'enginelb': {},
     'engineAux': {'images': {}},
     'node_selector': {}}

     ok

<table><tr><th>name</th> <td>alohapipeline</td></tr><tr><th>created</th> <td>2023-08-15 20:56:52.700421+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-16 16:31:01.773252+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>7aab9fa8-730a-4fc5-abfe-78c4489a012f, 5f6f1123-b371-40ba-ac44-b1ebb45598f7, 7d331276-d17b-4d54-bd5f-d65b11f2148e, 88f3dc23-41b3-407d-9852-ad6c797c16aa, 996d1e65-dfa3-424c-a996-cb8e88e2087c, 0a5efd20-1b75-4695-be5d-d5f2ed48810d, 18c6c824-1003-45e0-89cd-0e09e8c829f1, 8e367dac-1980-4e72-9fea-620b8120ee3d, e0a49f43-b43e-4251-9d30-ad245b27c1aa, d55825d3-1c47-465d-ab6a-543f271d5a5b, 1d54f862-9404-4aa9-8eea-b7aa77e3f919</td></tr><tr><th>steps</th> <td>alohamodel</td></tr></table>

### Infer with ARM

We will now perform the same inference request, this time through the pipeline with the ARM architecture.  The same data, the same model - just on ARM.

```python
startTime = time.time()
result = aloha_pipeline.infer_from_file('./data/data_25k.arrow',timeout=2500)
endTime = time.time()
arm_time = endTime-startTime
display(result.to_pandas().loc[:, ["time","out.main"]])
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out.main</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-08-16 16:31:14.151</td>
      <td>[0.997564]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-08-16 16:31:14.151</td>
      <td>[0.99999994]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-08-16 16:31:14.151</td>
      <td>[1.0]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-08-16 16:31:14.151</td>
      <td>[0.9999997]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-08-16 16:31:14.151</td>
      <td>[0.9999989]</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>24949</th>
      <td>2023-08-16 16:31:14.151</td>
      <td>[0.9996881]</td>
    </tr>
    <tr>
      <th>24950</th>
      <td>2023-08-16 16:31:14.151</td>
      <td>[0.99981505]</td>
    </tr>
    <tr>
      <th>24951</th>
      <td>2023-08-16 16:31:14.151</td>
      <td>[0.9999919]</td>
    </tr>
    <tr>
      <th>24952</th>
      <td>2023-08-16 16:31:14.151</td>
      <td>[1.0]</td>
    </tr>
    <tr>
      <th>24953</th>
      <td>2023-08-16 16:31:14.151</td>
      <td>[0.99999803]</td>
    </tr>
  </tbody>
</table>
<p>24954 rows × 2 columns</p>

### Compare Standard against Arm

With the two inferences complete, we'll compare the standard deployment architecture time against the ARM architecture.

```python
display(f"Standard architecture: {x64_time}")
display(f"ARM architecture: {arm_time}")
```

    'Standard architecture: 5.773933172225952'

    'ARM architecture: 5.696603536605835'

## Undeploy Pipeline

When finished with our tests, we will undeploy the pipeline so we have the Kubernetes resources back for other tasks.  Note that if the deployment variable is unchanged aloha_pipeline.deploy() will restart the inference engine in the same configuration as before.

```python
aloha_pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s .................................... ok

<table><tr><th>name</th> <td>alohapipeline</td></tr><tr><th>created</th> <td>2023-08-15 20:56:52.700421+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-16 16:31:01.773252+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>7aab9fa8-730a-4fc5-abfe-78c4489a012f, 5f6f1123-b371-40ba-ac44-b1ebb45598f7, 7d331276-d17b-4d54-bd5f-d65b11f2148e, 88f3dc23-41b3-407d-9852-ad6c797c16aa, 996d1e65-dfa3-424c-a996-cb8e88e2087c, 0a5efd20-1b75-4695-be5d-d5f2ed48810d, 18c6c824-1003-45e0-89cd-0e09e8c829f1, 8e367dac-1980-4e72-9fea-620b8120ee3d, e0a49f43-b43e-4251-9d30-ad245b27c1aa, d55825d3-1c47-465d-ab6a-543f271d5a5b, 1d54f862-9404-4aa9-8eea-b7aa77e3f919</td></tr><tr><th>steps</th> <td>alohamodel</td></tr></table>

