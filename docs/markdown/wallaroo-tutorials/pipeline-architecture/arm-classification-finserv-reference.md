This tutorial is available on the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/blob/2023.3.0-updates/pipeline-architecture/wallaroo-arm-classification-finserv).

## Classification Financial Services with Arm Architecture

This tutorial demonstrates how to use the Wallaroo combined with ARM processors to perform inferences with pre-trained classification financial services ML models.  This demonstration assumes that:

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
```

```python
wallaroo.__version__
```

    '2023.3.0+f2e5950e8'

### Connect to the Wallaroo Instance through the User Interface

The next step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
# Login through local Wallaroo instance

wl = wallaroo.Client()
```

    Please log into the following URL in a web browser:
    
    	https://peters-arm-test.keycloak.wallarooexample.ai/auth/realms/master/device?user_code=JKPL-UBPD
    
    Login successful!

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

    {'name': 'arm-classification-finservjohn', 'id': 32, 'archived': False, 'created_by': '26a792cb-25d0-470a-aba2-9b4c4ba373ac', 'created_at': '2023-08-15T21:27:27.693789+00:00', 'models': [{'name': 'ccfraudmodel', 'versions': 2, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 8, 16, 16, 41, 11, 549537, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 8, 15, 21, 27, 29, 931499, tzinfo=tzutc())}], 'pipelines': [{'name': 'arm-classification-example', 'create_time': datetime.datetime(2023, 8, 15, 21, 27, 31, 128328, tzinfo=tzutc()), 'definition': '[]'}]}

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

<table><tr><th>name</th> <td>arm-classification-example</td></tr><tr><th>created</th> <td>2023-08-15 21:27:31.128328+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-16 16:41:50.403174+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>b6383a02-ed45-4f84-a638-1d27e62bc4e8, 14e18386-e73e-4883-9da9-32be9d9b8b64, f0d39e54-1a3a-4d98-8350-938ecd315119, 3f2b6bf2-14eb-4005-80a9-6b0f06ded120, a8d55b45-2331-4d44-aeba-051eaf20e98c</td></tr><tr><th>steps</th> <td>ccfraudmodel</td></tr></table>

Now our pipeline is set.  Let's add a single **step** to it - in this case, our `ccfraud_model` that we uploaded to our workspace.

```python
pipeline.add_model_step(ccfraud_model)
```

<table><tr><th>name</th> <td>arm-classification-example</td></tr><tr><th>created</th> <td>2023-08-15 21:27:31.128328+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-16 16:41:50.403174+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>b6383a02-ed45-4f84-a638-1d27e62bc4e8, 14e18386-e73e-4883-9da9-32be9d9b8b64, f0d39e54-1a3a-4d98-8350-938ecd315119, 3f2b6bf2-14eb-4005-80a9-6b0f06ded120, a8d55b45-2331-4d44-aeba-051eaf20e98c</td></tr><tr><th>steps</th> <td>ccfraudmodel</td></tr></table>

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

```python
pipeline.deploy(deployment_config=deployment_config)
```

     ok

<table><tr><th>name</th> <td>arm-classification-example</td></tr><tr><th>created</th> <td>2023-08-15 21:27:31.128328+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-16 17:36:06.084003+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>76a5fa70-7272-489e-aee7-338a9abb9852, b6383a02-ed45-4f84-a638-1d27e62bc4e8, 14e18386-e73e-4883-9da9-32be9d9b8b64, f0d39e54-1a3a-4d98-8350-938ecd315119, 3f2b6bf2-14eb-4005-80a9-6b0f06ded120, a8d55b45-2331-4d44-aeba-051eaf20e98c</td></tr><tr><th>steps</th> <td>ccfraudmodel</td></tr></table>

### Inference on Standard Architecture

We will now perform an inference on 10,000 records by specifying an Apache Arrow input file, and track the time it takes to perform the inference and display the first 5 results.

For more information, see the [Wallaroo SDK Essentials Guide: Inference Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-inferences/)

```python
start_time = time.time()
result = pipeline.infer_from_file('./data/cc_data_10k.arrow')
end_time = time.time()
x64_time = end_time - start_time

outputs =  result.to_pandas()
display(outputs.head(5))
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.dense_1</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-08-16 17:36:06.669</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>[0.99300325]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-08-16 17:36:06.669</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>[0.99300325]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-08-16 17:36:06.669</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>[0.99300325]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-08-16 17:36:06.669</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>[0.99300325]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-08-16 17:36:06.669</td>
      <td>[0.5817662, 0.09788155, 0.15468194, 0.4754102, -0.19788623, -0.45043448, 0.016654044, -0.025607055, 0.09205616, -0.27839172, 0.059329946, -0.019658541, -0.42250833, -0.12175389, 1.5473095, 0.23916228, 0.3553975, -0.76851654, -0.7000849, -0.11900433, -0.3450517, -1.1065114, 0.25234112, 0.020944182, 0.21992674, 0.25406894, -0.04502251, 0.10867739, 0.25471792]</td>
      <td>[0.0010916889]</td>
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
pipeline.deploy(deployment_config=deployment_config_arm)
```

    Waiting for undeployment - this will take up to 45s ..................................... ok
    Waiting for deployment - this will take up to 45s ............. ok

<table><tr><th>name</th> <td>arm-classification-example</td></tr><tr><th>created</th> <td>2023-08-15 21:27:31.128328+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-16 17:36:45.689357+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>cd769d93-a989-4927-978f-d36ff5588b3b, 76a5fa70-7272-489e-aee7-338a9abb9852, b6383a02-ed45-4f84-a638-1d27e62bc4e8, 14e18386-e73e-4883-9da9-32be9d9b8b64, f0d39e54-1a3a-4d98-8350-938ecd315119, 3f2b6bf2-14eb-4005-80a9-6b0f06ded120, a8d55b45-2331-4d44-aeba-051eaf20e98c</td></tr><tr><th>steps</th> <td>ccfraudmodel</td></tr></table>

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
      <th>out.dense_1</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-08-16 17:36:59.807</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>[0.99300325]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-08-16 17:36:59.807</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>[0.99300325]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-08-16 17:36:59.807</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>[0.99300325]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-08-16 17:36:59.807</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>[0.99300325]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-08-16 17:36:59.807</td>
      <td>[0.5817662, 0.09788155, 0.15468194, 0.4754102, -0.19788623, -0.45043448, 0.016654044, -0.025607055, 0.09205616, -0.27839172, 0.059329946, -0.019658541, -0.42250833, -0.12175389, 1.5473095, 0.23916228, 0.3553975, -0.76851654, -0.7000849, -0.11900433, -0.3450517, -1.1065114, 0.25234112, 0.020944182, 0.21992674, 0.25406894, -0.04502251, 0.10867739, 0.25471792]</td>
      <td>[0.0010916889]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

### Compare Differences

We will now compare the results on the standard architecture versus ARM.  Typically, ARM delivers a 15% improvement on inference times while requireing less power and cost requirements.

```python
display(f"Standard architecture: {x64_time}")
display(f"ARM architecture: {arm_time}")
```

    'Standard architecture: 0.11516523361206055'

    'ARM architecture: 0.06871604919433594'

With our work in the pipeline done, we'll undeploy it to get back our resources from the Kubernetes cluster.  If we keep the same settings we can redeploy the pipeline with the same configuration in the future.

```python
pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s ...................................... ok

<table><tr><th>name</th> <td>arm-classification-example</td></tr><tr><th>created</th> <td>2023-08-15 21:27:31.128328+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-16 17:36:45.689357+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>cd769d93-a989-4927-978f-d36ff5588b3b, 76a5fa70-7272-489e-aee7-338a9abb9852, b6383a02-ed45-4f84-a638-1d27e62bc4e8, 14e18386-e73e-4883-9da9-32be9d9b8b64, f0d39e54-1a3a-4d98-8350-938ecd315119, 3f2b6bf2-14eb-4005-80a9-6b0f06ded120, a8d55b45-2331-4d44-aeba-051eaf20e98c</td></tr><tr><th>steps</th> <td>ccfraudmodel</td></tr></table>

