This tutorial is available on the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/pipeline-architecture/wallaroo-arm-cv-demonstration).

This tutorial demonstrates how to use the Wallaroo combined with ARM processors to perform inferences with pre-trained computer vision ML models.  This demonstration assumes that:

* A Wallaroo version 2023.3 or above instance is installed.
* A nodepools with ARM architecture virtual machines are part of the Kubernetes cluster.  For example, Azure supports Ampere® Altra® Arm-based processor included with the following virtual machines:
  * [Dpsv5 and Dpdsv5-series](https://learn.microsoft.com/en-us/azure/virtual-machines/dpsv5-dpdsv5-series)
  * [Epsv5 and Epdsv5-series](https://learn.microsoft.com/en-us/azure/virtual-machines/epsv5-epdsv5-series)

### Tutorial Goals

For our example, we will perform the following:

* Create a workspace for our work.
* Upload the the resnet computer vision model model.
* Create a pipeline using the default architecture that can ingest our submitted data, submit it to the model, and export the results while tracking how long the inference took.
* Redeploy the same pipeline on the ARM architecture, then perform the same inference on the same data and model and track how long the inference took.
* Compare the inference timing through the default architecture versus the ARM architecture.

## Steps

### Import Libraries

The first step will be to import our libraries.

```python
import torch
import pickle
import wallaroo
from wallaroo.object import EntityNotFoundError
from wallaroo.framework import Framework

import numpy as np
import json
import requests
import time
import pandas as pd

# used to display dataframe information without truncating
from IPython.display import display
import pandas as pd
pd.set_option('display.max_colwidth', None)

# used for unique connection names

import string
import random
suffix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))
```

    /opt/conda/lib/python3.9/site-packages/tqdm/auto.py:22: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm

### Connect to the Wallaroo Instance

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
# Login through local service

wl = wallaroo.Client()
```

### Set Variables

The following variables and methods are used later to create or connect to an existing workspace, pipeline, and model.

The `suffix` is used to ensure unique workspace names across the Wallaroo instance.  Set this to '' if not required.

```python
suffix=''

workspace_name = f'cv-arm-example{suffix}'
pipeline_name = 'cv-sample'

x86_resnet_model_name = 'x86-resnet50'
resnet_model_file_name = 'models/resnet50_v1.onnx'

arm_resnet_model_name = 'arm-resnet50'
resnet_model_file_name = 'models/resnet50_v1.onnx'
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

### Create Workspace

The workspace will be created or connected to, and set as the default workspace for this session.  Once that is done, then all models and pipelines will be set in that workspace.

```python
workspace = get_workspace(workspace_name)
wl.set_current_workspace(workspace)
wl.get_current_workspace()
```

    {'name': 'cv-arm-example', 'id': 26, 'archived': False, 'created_by': '0e5060a5-218c-47c1-9678-e83337494184', 'created_at': '2023-09-08T21:54:56.56663+00:00', 'models': [{'name': 'x86-resnet50', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 9, 8, 21, 55, 1, 675188, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 9, 8, 21, 55, 1, 675188, tzinfo=tzutc())}, {'name': 'arm-resnet50', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 9, 8, 21, 55, 6, 69116, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 9, 8, 21, 55, 6, 69116, tzinfo=tzutc())}], 'pipelines': [{'name': 'cv-sample', 'create_time': datetime.datetime(2023, 9, 8, 21, 54, 57, 62345, tzinfo=tzutc()), 'definition': '[]'}]}

### Create Pipeline and Upload Model

We will now create or connect to our pipeline, then create two versions of the model:  one that defaults to the x86 architecture, the other that will use the ARM architecture.

```python
pipeline = get_pipeline(pipeline_name)

from wallaroo.engine_config import Architecture

x86_resnet_model = wl.upload_model(x86_resnet_model_name, 
                                   resnet_model_file_name, 
                                   framework=Framework.ONNX)
arm_resnet_model = wl.upload_model(arm_resnet_model_name, 
                                   resnet_model_file_name, 
                                   framework=Framework.ONNX,
                                   arch=Architecture.ARM)
```

### Deploy Pipeline

With the model uploaded, we can add it is as a step in the pipeline, then deploy it.

For this deployment we will be using the default deployment which uses the x86 architecture.

Once deployed, resources from the Wallaroo instance will be reserved and the pipeline will be ready to use the model to perform inference requests. 

```python
deployment_config = (wallaroo.deployment_config
                            .DeploymentConfigBuilder()
                            .cpus(2)
                            .memory('2Gi')
                            .build()
                        )
# clear previous steps
pipeline.undeploy()
pipeline.clear()
pipeline.add_model_step(x86_resnet_model)

pipeline.deploy(deployment_config = deployment_config)
```

    Waiting for deployment - this will take up to 45s .............. ok

<table><tr><th>name</th> <td>cv-sample</td></tr><tr><th>created</th> <td>2023-09-08 21:54:57.062345+00:00</td></tr><tr><th>last_updated</th> <td>2023-09-08 21:55:33.858777+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>ccc676d6-019c-4f9a-8866-033950a5907b, 68297806-92bb-4dce-8c10-a1f1d278ab2a</td></tr><tr><th>steps</th> <td>x86-resnet50</td></tr><tr><th>published</th> <td>False</td></tr></table>

```python
pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.3.60',
       'name': 'engine-7645b79695-c78gv',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'cv-sample',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'x86-resnet50',
          'version': '801b73a0-a6e3-45d7-b518-120c826fa718',
          'sha': 'c6c8869645962e7711132a7e17aced2ac0f60dcdc2c7faa79b2de73847a87984',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.3.59',
       'name': 'engine-lb-584f54c899-2w6j9',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}

### Run Inference

With that done, we can have the model detect the objects on the image by running an inference through the pipeline.  For this example, we will use a prepared Apache Arrow table `./data/image_224x224.arrow`

```python
startTime = time.time()
# pass the table in 
results = pipeline.infer_from_file('./data/image_224x224.arrow')
endTime = time.time()
x86_time = endTime-startTime
```

### Deploy with ARM

We have demonstrated performing our sample inference using a standard pipeline deployment.  Now we will redeploy the same pipeline with the ARM architecture version of the model.

```python

# clear previous steps
pipeline.undeploy()
pipeline.clear()
pipeline.add_model_step(arm_resnet_model)

pipeline.deploy(deployment_config = deployment_config)
```

    Waiting for undeployment - this will take up to 45s .................................... ok
    Waiting for deployment - this will take up to 45s ............... ok

<table><tr><th>name</th> <td>cv-sample</td></tr><tr><th>created</th> <td>2023-09-08 21:54:57.062345+00:00</td></tr><tr><th>last_updated</th> <td>2023-09-08 21:56:26.218871+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>3aff896f-52cb-478b-9cd7-64c3212d768f, ccc676d6-019c-4f9a-8866-033950a5907b, 68297806-92bb-4dce-8c10-a1f1d278ab2a</td></tr><tr><th>steps</th> <td>x86-resnet50</td></tr><tr><th>published</th> <td>False</td></tr></table>

```python
pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.3.61',
       'name': 'engine-57548b6596-qx8k8',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'cv-sample',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'arm-resnet50',
          'version': 'dec621e2-8b13-44cf-a330-4fdada1f518e',
          'sha': 'c6c8869645962e7711132a7e17aced2ac0f60dcdc2c7faa79b2de73847a87984',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.3.62',
       'name': 'engine-lb-584f54c899-56c8l',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}

### ARM Inference

We will now perform the same inference we did with the standard deployment architecture, only this time through the ARM virtual machines.

```python
startTime = time.time()
# pass the table in 
results = pipeline.infer_from_file('./data/image_224x224.arrow')
endTime = time.time()
arm_time = endTime-startTime
```

### Compare Standard against Arm

With the two inferences complete, we'll compare the standard deployment architecture against the ARM architecture.

```python
display(f"Standard architecture: {x86_time}")
display(f"ARM architecture: {arm_time}")
```

    'Standard architecture: 0.06283044815063477'

    'ARM architecture: 0.06364631652832031'

### Undeploy the Pipeline

With the inference complete, we can undeploy the pipeline and return the resources back to the Wallaroo instance.

```python
pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s ...................................... ok

<table><tr><th>name</th> <td>cv-sample</td></tr><tr><th>created</th> <td>2023-09-08 21:54:57.062345+00:00</td></tr><tr><th>last_updated</th> <td>2023-09-08 21:56:26.218871+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>3aff896f-52cb-478b-9cd7-64c3212d768f, ccc676d6-019c-4f9a-8866-033950a5907b, 68297806-92bb-4dce-8c10-a1f1d278ab2a</td></tr><tr><th>steps</th> <td>x86-resnet50</td></tr><tr><th>published</th> <td>False</td></tr></table>

