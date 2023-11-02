This tutorial can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/model_uploads/python-upload-tutorials).

## Python Model Upload to Wallaroo

Python scripts can be deployed to Wallaroo as Python Models.  These are treated like other models, and are used for:

* ML Models: Models written entirely in Python script.
* Data Formatting:  Typically preprocess or post process modules that shape incoming data into what a ML model expects, or receives data output by a ML model and changes the data for other processes to accept.

Models are added to Wallaroo pipelines as pipeline steps, with the data from the previous step submitted to the next one.  Python steps require the entry method `wallaroo_json`.  These methods should be structured to receive and send pandas DataFrames as the inputs and outputs.

This allows inference requests to a Wallaroo pipeline to receive pandas DataFrames or Apache Arrow tables, and return the same for consistent results.

This tutorial will:

* Create a Wallaroo workspace and pipeline.
* Upload the sample Python model and ONNX model.
* Demonstrate the outputs of the ONNX model to an inference request.
* Demonstrate the functionality of the Python model in reshaping data after an inference request.
* Use both the ONNX model and the Python model together as pipeline steps to perform an inference request and export the data for use.

### Prerequisites

* A Wallaroo version 2023.2.1 or above instance.

### References

* [Wallaroo SDK Essentials Guide: Pipeline Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline/)
* [Python Models](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/wallaroo-sdk-model-upload-python/)

## Tutorial Steps

### Import Libraries

We'll start with importing the libraries we need for the tutorial.  The main libraries used are:

* Wallaroo: To connect with the Wallaroo instance and perform the MLOps commands.
* `pyarrow`: Used for formatting the data.
* `pandas`: Used for pandas DataFrame tables.

```python
import wallaroo
from wallaroo.object import EntityNotFoundError
from wallaroo.framework import Framework
from wallaroo.deployment_config import DeploymentConfigBuilder

import pandas as pd

#import os
# import json
import pyarrow as pa
```

### Connect to the Wallaroo Instance through the User Interface

The next step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
# Login through local Wallaroo instance

wl = wallaroo.Client()
```

### Set Variables and Helper Functions

We'll set the name of our workspace, pipeline, models and files.  Workspace names must be unique across the Wallaroo workspace.  For this, we'll add in a randomly generated 4 characters to the workspace name to prevent collisions with other users' workspaces.  If running this tutorial, we recommend hard coding the workspace name so it will function in the same workspace each time it's run.

We'll set up some helper functions that will either use existing workspaces and pipelines, or create them if they do not already exist.

```python
import string
import random

# make a random 4 character suffix to prevent overwriting other user's workspaces
suffix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))
suffix='jch'
workspace_name = f'python-demo{suffix}'
pipeline_name = f'python-step-demo-pipeline'

onnx_model_name = 'house-price-sample'
onnx_model_file_name = './models/house_price_keras.onnx'
python_model_name = 'python-step'
python_model_file_name = './models/step.py'
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

### Create a New Workspace

For our tutorial, we'll create the workspace, set it as the current workspace, then the pipeline we'll add our models to.

#### Create New Workspace References

* [Wallaroo SDK Essentials Guide: Workspace Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-workspace/)
* [Wallaroo SDK Essentials Guide: Pipeline Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline/)

```python
workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)

pipeline = get_pipeline(pipeline_name)
pipeline
```

### Model Descriptions

We have two models we'll be using.

* `./models/house_price_keras.onnx`:  A ML model trained to forecast hour prices based on inputs.  This forecast is stored in the column `dense_2`.
* `./models/step.py`: A Python script that accepts the data from the house price model, and reformats the output. We'll be using it as a post-processing step.

For the Python step, it contains the method `wallaroo_json` as the entry point used by Wallaroo when deployed as a pipeline step.  Our sample script has the following:

```python
# take a dataframe output of the house price model, and reformat the `dense_2`
# column as `output`
def wallaroo_json(data: pd.DataFrame):
    print(data)
    return [{"output": [data["dense_2"].to_list()[0][0]]}]
```

As seen from the description, all those function will do it take the DataFrame output of the house price model, and output a DataFrame replacing the first element in the list from column `dense_2` with `output`.

### Upload Models

Both of these models will be uploaded to our current workspace using the method `upload_model(name, path, framework).configure(framework, input_schema, output_schema)`.

* For `./models/house_price_keras.onnx`, we will specify it as `Framework.ONNX`.  We do not need to specify the input and output schemas.
* For `./models/step.py`, we will set the input and output schemas in the required `pyarrow.lib.Schema` format.

#### Upload Model References

* [Wallaroo SDK Essentials Guide: Model Uploads and Registrations: ONNX](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/wallaroo-sdk-model-upload-onnx/)
* [Wallaroo SDK Essentials Guide: Model Uploads and Registrations: Python Models](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/wallaroo-sdk-model-upload-python/)

```python
house_price_model = (wl.upload_model(onnx_model_name, 
                                    onnx_model_file_name, 
                                    framework=Framework.ONNX)
                                    .configure('onnx', 
                                               tensor_fields=["tensor"]
                                              )
                    )

input_schema = pa.schema([
    pa.field('dense_2', pa.list_(pa.float64()))
])
output_schema = pa.schema([
    pa.field('output', pa.list_(pa.float64()))
])

step = (wl.upload_model(python_model_name, 
                        python_model_file_name, 
                        framework=Framework.PYTHON)
                        .configure(
                            'python', 
                            input_schema=input_schema, 
                            output_schema=output_schema
                        )
        )
```

### Pipeline Steps

With our models uploaded, we'll perform different configurations of the pipeline steps.

First we'll add just the house price model to the pipeline, deploy it, and submit a sample inference.

```python
# used to restrict the resources needed for this demonstration
deployment_config = DeploymentConfigBuilder() \
    .cpus(0.25).memory('1Gi') \
    .build()
```

```python
# clear the pipeline if this tutorial was run before
pipeline.undeploy()
pipeline.clear()

pipeline.add_model_step(house_price_model).deploy(deployment_config=deployment_config)
```

```python
## sample inference data

data = pd.DataFrame.from_dict({"tensor": [[0.6878518042239091,
                                            0.17607340208535074,
                                            -0.8695140830357148,
                                            0.34638762962802144,
                                            -0.0916270832672289,
                                            -0.022063226781124278,
                                            -0.13969884765926363,
                                            1.002792335666138,
                                            -0.3067449033633758,
                                            0.9272000630461978,
                                            0.28326687982544635,
                                            0.35935375728372815,
                                            -0.682562654045523,
                                            0.532642794275658,
                                            -0.22705189652659302,
                                            0.5743846356405602,
                                            -0.18805086358065454
                                            ]]})

results = pipeline.infer(data)
display(results)
```

### Inference with Pipeline Step

Our inference result had the results in the `out.dense_2` column.  We'll clear the pipeline, then add in as the pipeline step just the Python postprocessing step we've created.  Then for our inference request, we'll just submit the output of the house price model.  Our result should be the first element in the array returned in the `out.output` column.

```python
pipeline.clear()
pipeline.add_model_step(step)

pipeline.deploy(deployment_config=deployment_config)
```

```python
data = pd.DataFrame.from_dict({"dense_2": [12.886651]})
python_result = pipeline.infer(data)
display(python_result)
```

### Putting Both Models Together

Now we'll do one last pipeline deployment with 2 steps:

* First the house price model that outputs the inference result into `dense_2`.
* Second the python step so it will accept the output of the house price model, and reshape it into `output`.

```python
import datetime
inference_start = datetime.datetime.now()

pipeline.undeploy()
pipeline.clear()
pipeline.add_model_step(house_price_model)
pipeline.add_model_step(step)
```

```python
pipeline.deploy(deployment_config=deployment_config)
```

```python
pipeline.status()
```

```python
data = pd.DataFrame.from_dict({"tensor": [[0.6878518042239091,
                                            0.17607340208535074,
                                            -0.8695140830357148,
                                            0.34638762962802144,
                                            -0.0916270832672289,
                                            -0.022063226781124278,
                                            -0.13969884765926363,
                                            1.002792335666138,
                                            -0.3067449033633758,
                                            0.9272000630461978,
                                            0.28326687982544635,
                                            0.35935375728372815,
                                            -0.682562654045523,
                                            0.532642794275658,
                                            -0.22705189652659302,
                                            0.5743846356405602,
                                            -0.18805086358065454
                                        ]]})

results = pipeline.infer(data)
display(results)
```

### Pipeline Logs

As the data was exported by the pipeline step as a pandas DataFrame, it will be reflected in the pipeline logs.  We'll retrieve the most recent log from our most recent inference.

```python
inference_end = datetime.datetime.now()

pipeline.logs(start_datetime=inference_start, end_datetime=inference_end)
```

### Undeploy the Pipeline

With our tutorial complete, we'll undeploy the pipeline and return the resources back to the cluster.

This process demonstrated how to structure a postprocessing Python script as a Wallaroo Pipeline step.  This can be used for pre or post processing, Python based models, and other use cases.

```python
pipeline.undeploy()
```
