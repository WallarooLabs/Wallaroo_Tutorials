## Statsmodel Forecast with Wallaroo Features: Deploy and Test Infer

This tutorial series demonstrates how to use Wallaroo to create a Statsmodel forecasting model based on bike rentals.  This tutorial series is broken down into the following:

* Create and Train the Model:  This first notebook shows how the model is trained from existing data.
* Deploy and Sample Inference:  With the model developed, we will deploy it into Wallaroo and perform a sample inference.
* Parallel Infer:  A sample of multiple weeks of data will be retrieved and submitted as an asynchronous parallel inference.  The results will be collected and uploaded to a sample database.
* External Connection:  A sample data connection to Google BigQuery to retrieve input data and store the results in a table.
* ML Workload Orchestration:  Take all of the previous steps and automate the request into a single Wallaroo ML Workload Orchestration.

In the previous step "Statsmodel Forecast with Wallaroo Features: Model Creation", the statsmodel was trained and saved to the Python file `forecast.py`.  This file will now be uploaded to a Wallaroo instance as a Python model, then used for sample inferences.

## Prerequisites

* A Wallaroo instance version 2023.2.1 or greater.

## References

* [Wallaroo SDK Essentials Guide: Model Uploads and Registrations: Python Models](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/wallaroo-sdk-model-upload-python/)
* [Wallaroo SDK Essentials Guide: Pipeline Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline/)
* [Wallaroo SDK Essentials: Inference Guide: Parallel Inferences](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-inferences/#parallel-inferences)

## Tutorial Steps

### Import Libraries

The first step is to import the libraries that we will need.

```python
import json
import os
import datetime

import wallaroo
from wallaroo.object import EntityNotFoundError
from wallaroo.framework import Framework

# used to display dataframe information without truncating
from IPython.display import display
import pandas as pd
pd.set_option('display.max_colwidth', None)
```

```python
wallaroo.__version__
```

    '2023.2.1rc2'

### Initialize connection

Start a connect to the Wallaroo instance and save the connection into the variable `wl`.

```python
# Login through local Wallaroo instance

wl = wallaroo.Client()
```

### Set Configurations

The following will set the workspace, model name, and pipeline that will be used for this example.  If the workspace or pipeline already exist, then they will assigned for use in this example.  If they do not exist, they will be created based on the names listed below.

Workspace names must be unique.  To allow this tutorial to run in the same Wallaroo instance for multiple users, the `suffix` variable is generated from a random set of 4 ASCII characters.  To use the same workspace each time, hard code `suffix` and verify the workspace name created is is unique across the Wallaroo instance.

```python
# used for unique connection names

import string
import random

suffix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))

workspace_name = f'multiple-replica-forecast-tutorial-{suffix}'
pipeline_name = 'bikedaypipe'
model_name = 'bikedaymodel'
```

### Set the Workspace and Pipeline

The workspace will be either used or created if it does not exist, along with the pipeline.

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

workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)

pipeline = get_pipeline(pipeline_name)
```

### Upload Model

The Python model created in "Forecast and Parallel Infer with Statsmodel: Model Creation" will now be uploaded.  Note that the Framework and runtime are set to `python`.

```python
model_file_name = 'forecast.py'

bike_day_model = wl.upload_model(model_name, model_file_name, Framework.PYTHON).configure(runtime="python")
```

### Deploy the Pipeline

We will now add the uploaded model as a step for the pipeline, then deploy it.  The pipeline configuration will allow for multiple replicas of the pipeline to be deployed and spooled up in the cluster.  Each pipeline replica will use 0.25 cpu and 512 Gi RAM.

```python
# Set the deployment to allow for additional engines to run
deploy_config = (wallaroo.DeploymentConfigBuilder()
                        .replica_count(1)
                        .replica_autoscale_min_max(minimum=2, maximum=5)
                        .cpus(0.25)
                        .memory("512Mi")
                        .build()
                    )

pipeline.add_model_step(bike_day_model).deploy(deployment_config = deploy_config)
```

<table><tr><th>name</th> <td>bikedaypipe</td></tr><tr><th>created</th> <td>2023-07-14 15:50:50.014326+00:00</td></tr><tr><th>last_updated</th> <td>2023-07-14 15:50:52.029628+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>7aae4653-9e9f-468c-b266-4433be652313, 48983f9b-7c43-41fe-9688-df72a6aa55e9</td></tr><tr><th>steps</th> <td>bikedaymodel</td></tr></table>

### Run Inference

Run a test inference to verify the pipeline is operational from the sample test data stored in `./data/testdata_dict.json`.

```python
inferencedata = json.load(open("./data/testdata_dict.json"))

results = pipeline.infer(inferencedata)

display(results)
```

    [{'forecast': [1764, 1749, 1743, 1741, 1740, 1740, 1740]}]

### Undeploy the Pipeline

Undeploy the pipeline and return the resources back to the Wallaroo instance.

```python
pipeline.undeploy()
```

<table><tr><th>name</th> <td>bikedaypipe</td></tr><tr><th>created</th> <td>2023-07-14 15:50:50.014326+00:00</td></tr><tr><th>last_updated</th> <td>2023-07-14 15:50:52.029628+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>7aae4653-9e9f-468c-b266-4433be652313, 48983f9b-7c43-41fe-9688-df72a6aa55e9</td></tr><tr><th>steps</th> <td>bikedaymodel</td></tr></table>

