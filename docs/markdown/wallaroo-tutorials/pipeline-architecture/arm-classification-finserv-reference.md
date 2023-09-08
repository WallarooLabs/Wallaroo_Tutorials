This tutorial is available on the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/pipeline-architecture/wallaroo-arm-classification-finserv).

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

    '2023.3.0+8a3f2fb38'

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

    {'name': 'arm-classification-finservjohn', 'id': 25, 'archived': False, 'created_by': '0e5060a5-218c-47c1-9678-e83337494184', 'created_at': '2023-09-08T21:40:58.35659+00:00', 'models': [], 'pipelines': []}

## Upload a model

Our workspace is created.  Let's upload our credit card fraud model to it.  This is the file name `ccfraud.onnx`, and we'll upload it as `ccfraudmodel`.  The credit card fraud model is trained to detect credit card fraud based on a 0 to 1 model:  The closer to 0 the less likely the transactions indicate fraud, while the closer to 1 the more likely the transactions indicate fraud.

We will create two versions of the model:  one that defaults to the x86 architecture, the other that will use the ARM architecture.

For more information, see the [Wallaroo SDK Essentials Guide: Model Uploads and Registrations](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/).

```python
from wallaroo.engine_config import Architecture

x86_ccfraud_model = (wl.upload_model(model_name, 
                                 model_file_name, 
                                 wallaroo.framework.Framework.ONNX)
                                 .configure()
                )

arm_ccfraud_model = (wl.upload_model(model_name, 
                                 model_file_name, 
                                 wallaroo.framework.Framework.ONNX,
                                 arch=Architecture.ARM)
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
```

### Deploy Pipeline for x86 Architecture

Now our pipeline is set.  Let's add a single **step** to it - in this case, our `x86_ccfraud_model` that we uploaded to our workspace to use the default x86 architecture.

```python
# clear the steps if used before
pipeline.undeploy()
pipeline.clear()
pipeline.add_model_step(x86_ccfraud_model)
```

    Waiting for undeployment - this will take up to 45s ..................................... ok

<table><tr><th>name</th> <td>arm-classification-example</td></tr><tr><th>created</th> <td>2023-09-08 21:40:58.819251+00:00</td></tr><tr><th>last_updated</th> <td>2023-09-08 21:45:48.571799+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>7800fd1c-edc1-4177-99eb-29100c2908b9, 285566fa-5596-48c9-b2e8-81ede445bc80, 24fa4009-3ed7-4887-857d-9e2617433962, 6a2beaff-2bfd-41ab-abd6-b4ccfcab0f77, 7c2b0cce-82d7-4fb7-a052-50e2b13dea55</td></tr><tr><th>steps</th> <td>ccfraudmodel</td></tr><tr><th>published</th> <td>False</td></tr></table>

### Deploy Pipeline

We will now deploy the pipeline.

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

    Waiting for deployment - this will take up to 45s ........ ok

<table><tr><th>name</th> <td>arm-classification-example</td></tr><tr><th>created</th> <td>2023-09-08 21:40:58.819251+00:00</td></tr><tr><th>last_updated</th> <td>2023-09-08 21:47:16.136633+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>1704c888-8b7a-4903-aaea-168194105baa, 7800fd1c-edc1-4177-99eb-29100c2908b9, 285566fa-5596-48c9-b2e8-81ede445bc80, 24fa4009-3ed7-4887-857d-9e2617433962, 6a2beaff-2bfd-41ab-abd6-b4ccfcab0f77, 7c2b0cce-82d7-4fb7-a052-50e2b13dea55</td></tr><tr><th>steps</th> <td>ccfraudmodel</td></tr><tr><th>published</th> <td>False</td></tr></table>

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
      <th>out.dense_1</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-09-08 21:48:02.799</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>[0.99300325]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-09-08 21:48:02.799</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>[0.99300325]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-09-08 21:48:02.799</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>[0.99300325]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-09-08 21:48:02.799</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>[0.99300325]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-09-08 21:48:02.799</td>
      <td>[0.5817662, 0.09788155, 0.15468194, 0.4754102, -0.19788623, -0.45043448, 0.016654044, -0.025607055, 0.09205616, -0.27839172, 0.059329946, -0.019658541, -0.42250833, -0.12175389, 1.5473095, 0.23916228, 0.3553975, -0.76851654, -0.7000849, -0.11900433, -0.3450517, -1.1065114, 0.25234112, 0.020944182, 0.21992674, 0.25406894, -0.04502251, 0.10867739, 0.25471792]</td>
      <td>[0.0010916889]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

```python
x86_time
```

    0.0525670051574707

### Set Arm Pipeline Deployment Architecture

Now we can set our pipeline deployment architecture by specifying the ARM version of the model, then deploy our ipeline again.

```python
pipeline.undeploy()
pipeline.clear()
# clear the steps if used before
pipeline.clear()
pipeline.add_model_step(arm_ccfraud_model)
pipeline.deploy(deployment_config=deployment_config)
```

    Waiting for undeployment - this will take up to 45s ...................................... ok
    Waiting for deployment - this will take up to 45s ........ ok

<table><tr><th>name</th> <td>arm-classification-example</td></tr><tr><th>created</th> <td>2023-09-08 21:40:58.819251+00:00</td></tr><tr><th>last_updated</th> <td>2023-09-08 21:48:46.549206+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>32315379-fae5-44bd-84e8-78a9d7f57b30, 1704c888-8b7a-4903-aaea-168194105baa, 7800fd1c-edc1-4177-99eb-29100c2908b9, 285566fa-5596-48c9-b2e8-81ede445bc80, 24fa4009-3ed7-4887-857d-9e2617433962, 6a2beaff-2bfd-41ab-abd6-b4ccfcab0f77, 7c2b0cce-82d7-4fb7-a052-50e2b13dea55</td></tr><tr><th>steps</th> <td>ccfraudmodel</td></tr><tr><th>published</th> <td>False</td></tr></table>

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
      <td>2023-09-08 21:48:55.214</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>[0.99300325]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-09-08 21:48:55.214</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>[0.99300325]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-09-08 21:48:55.214</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>[0.99300325]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-09-08 21:48:55.214</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>[0.99300325]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-09-08 21:48:55.214</td>
      <td>[0.5817662, 0.09788155, 0.15468194, 0.4754102, -0.19788623, -0.45043448, 0.016654044, -0.025607055, 0.09205616, -0.27839172, 0.059329946, -0.019658541, -0.42250833, -0.12175389, 1.5473095, 0.23916228, 0.3553975, -0.76851654, -0.7000849, -0.11900433, -0.3450517, -1.1065114, 0.25234112, 0.020944182, 0.21992674, 0.25406894, -0.04502251, 0.10867739, 0.25471792]</td>
      <td>[0.0010916889]</td>
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

    'Standard architecture: 0.0525670051574707'

    'ARM architecture: 0.04808354377746582'

With our work in the pipeline done, we'll undeploy it to get back our resources from the Kubernetes cluster.  If we keep the same settings we can redeploy the pipeline with the same configuration in the future.

```python
pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s .................................... ok

<table><tr><th>name</th> <td>arm-classification-example</td></tr><tr><th>created</th> <td>2023-09-08 21:40:58.819251+00:00</td></tr><tr><th>last_updated</th> <td>2023-09-08 21:48:46.549206+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>32315379-fae5-44bd-84e8-78a9d7f57b30, 1704c888-8b7a-4903-aaea-168194105baa, 7800fd1c-edc1-4177-99eb-29100c2908b9, 285566fa-5596-48c9-b2e8-81ede445bc80, 24fa4009-3ed7-4887-857d-9e2617433962, 6a2beaff-2bfd-41ab-abd6-b4ccfcab0f77, 7c2b0cce-82d7-4fb7-a052-50e2b13dea55</td></tr><tr><th>steps</th> <td>ccfraudmodel</td></tr><tr><th>published</th> <td>False</td></tr></table>

