This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/model_uploads/tensorflow-upload-tutorials).

## Wallaroo SDK Upload Tutorial: Tensorflow

In this notebook we will walk through uploading a Tensorflow model to a Wallaroo instance and performing sample inferences.  For this example we will be using an open source model that uses an [Aloha CNN LSTM model](https://www.researchgate.net/publication/348920204_Using_Auxiliary_Inputs_in_Deep_Learning_Models_for_Detecting_DGA-based_Domain_Names) for classifying Domain names as being either legitimate or being used for nefarious purposes such as malware distribution.  

## Prerequisites

* An installed Wallaroo instance.
* The following Python libraries installed:
  * `os`
  * [`wallaroo`](https://pypi.org/project/wallaroo/): The Wallaroo SDK. Included with the Wallaroo JupyterHub service by default.
  * [`pandas`](https://pypi.org/project/pandas/): Pandas, mainly used for Pandas DataFrame
  * [`pyarrow`](https://pypi.org/project/pyarrow/): PyArrow for Apache Arrow support

## Tutorial Goals

For our example, we will perform the following:

* Create a workspace for our work.
* Upload the Aloha model.
* Create a pipeline that can ingest our submitted data, submit it to the model, and export the results.

All sample data and models are available through the [Wallaroo Quick Start Guide Samples repository](https://github.com/WallarooLabs/quickstartguide_samples).

### Connect to the Wallaroo Instance

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).e/wallaroo-sdk-essentials-client/).

```python
import wallaroo
from wallaroo.object import EntityNotFoundError
from wallaroo.framework import Framework

# to display dataframe tables
from IPython.display import display
# used to display dataframe information without truncating

import os
os.environ["MODELS_ENABLED"] = "true"

import pandas as pd
pd.set_option('display.max_colwidth', None)
import pyarrow as pa
```

```python
# Login through local Wallaroo instance

wl = wallaroo.Client()
```

## Create the Workspace

We will create a workspace to work in and call it the "alohaworkspace", then set it as current workspace environment.  We'll also create our pipeline in advance as `alohapipeline`.  The model name and the model file will be specified for use in later steps.

To allow this tutorial to be run multiple times or by multiple users in the same Wallaroo instance, a random 4 character prefix will be added to the workspace, pipeline, and model.

```python
import string
import random

# make a random 4 character suffix to verify uniqueness in tutorials
suffix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))

suffix=''

workspace_name = f'tensorflowuploadexampleworkspace{suffix}'
pipeline_name = f'tensorflowuploadexample{suffix}'
model_name = f'tensorflowuploadexample{suffix}'
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

aloha_pipeline = get_pipeline(pipeline_name)
aloha_pipeline
```

<table><tr><th>name</th> <td>tensorflowuploadexample</td></tr><tr><th>created</th> <td>2023-10-20 21:15:17.379310+00:00</td></tr><tr><th>last_updated</th> <td>2023-10-20 21:15:17.379310+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>dce97e55-97d7-4bf9-8c75-8f9115e40e23</td></tr><tr><th>steps</th> <td></td></tr><tr><th>published</th> <td>False</td></tr></table>

We can verify the workspace is created the current default workspace with the `get_current_workspace()` command.

```python
wl.get_current_workspace()
```

    {'name': 'tensorflowuploadexampleworkspace', 'id': 83, 'archived': False, 'created_by': '420ff140-d2a0-4425-9c12-cac5ed602472', 'created_at': '2023-10-20T21:15:17.207568+00:00', 'models': [], 'pipelines': [{'name': 'tensorflowuploadexample', 'create_time': datetime.datetime(2023, 10, 20, 21, 15, 17, 379310, tzinfo=tzutc()), 'definition': '[]'}]}

# Upload the Models

Now we will upload our models.  Note that for this example we are applying the model from a .ZIP file.  The Aloha model is a [protobuf](https://developers.google.com/protocol-buffers) file that has been defined for evaluating web pages, and we will configure it to use data in the `tensorflow` format.

The following parameters are required for TensorFlow models.  Tensorflow models are native runtimes in Wallaroo, so the `input_schema` and `output_schema` parameters are optional.

| Parameter | Type | Description |
|---|---|---|
|`name` | `string` (*Required*) | The name of the model.  Model names are unique per workspace.  Models that are uploaded with the same name are assigned as a new **version** of the model. |
|`path` | `string` (*Required*) | The path to the model file being uploaded. 
|`framework` |`string` (*Required*) | Set as the `Framework.TENSORFLOW`. |
|`input_schema` | `pyarrow.lib.Schema` (*Optional*) | The input schema in Apache Arrow schema format. |
|`output_schema` | `pyarrow.lib.Schema` (*Optional*) | The output schema in Apache Arrow schema format. |
| `convert_wait` | `bool` (*Optional*) (*Default: True*) | Not required for native runtimes. <ul><li>**True**: Waits in the script for the model conversion completion.</li><li>**False**:  Proceeds with the script without waiting for the model conversion process to display complete. |

### TensorFlow File Format

TensorFlow models are .zip file of the SavedModel format.  For example, the Aloha sample TensorFlow model is stored in the directory `alohacnnlstm`:

```bash
├── saved_model.pb
└── variables
    ├── variables.data-00000-of-00002
    ├── variables.data-00001-of-00002
    └── variables.index
```

This is compressed into the .zip file `alohacnnlstm.zip` with the following command:

```python
zip -r alohacnnlstm.zip alohacnnlstm/
```

```python
model = wl.upload_model(model_name, model_file_name, Framework.TENSORFLOW).configure("tensorflow")
```

```python
model.config().runtime()
```

    'tensorflow'

## Deploy a model

Now that we have a model that we want to use we will create a deployment for it. 

We will tell the deployment we are using a tensorflow model and give the deployment name and the configuration we want for the deployment.

```python
aloha_pipeline.add_model_step(model)
```

<table><tr><th>name</th> <td>tensorflowuploadexample</td></tr><tr><th>created</th> <td>2023-10-20 21:15:17.379310+00:00</td></tr><tr><th>last_updated</th> <td>2023-10-20 21:15:17.379310+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>dce97e55-97d7-4bf9-8c75-8f9115e40e23</td></tr><tr><th>steps</th> <td></td></tr><tr><th>published</th> <td>False</td></tr></table>

```python
aloha_pipeline.deploy()
```

    Waiting for deployment - this will take up to 45s ........ ok

<table><tr><th>name</th> <td>tensorflowuploadexample</td></tr><tr><th>created</th> <td>2023-10-20 21:15:17.379310+00:00</td></tr><tr><th>last_updated</th> <td>2023-10-20 21:15:17.786848+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>ceffdd27-2887-4a32-ba90-4eec496aafb1, dce97e55-97d7-4bf9-8c75-8f9115e40e23</td></tr><tr><th>steps</th> <td>tensorflowuploadexample</td></tr><tr><th>published</th> <td>False</td></tr></table>

We can verify that the pipeline is running and list what models are associated with it.

```python
aloha_pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.0.33',
       'name': 'engine-69b5b9c587-cg8rx',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'tensorflowuploadexample',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'tensorflowuploadexample',
          'version': '5d6ca253-93b4-4f16-b117-6d116450b108',
          'sha': 'd71d9ffc61aaac58c2b1ed70a2db13d1416fb9d3f5b891e5e4e2e97180fe22f8',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.2.210',
       'name': 'engine-lb-584f54c899-z5pcp',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}

## Interferences

### Infer 1 row

Now that the pipeline is deployed and our Aloha model is in place, we'll perform a smoke test to verify the pipeline is up and running properly.  We'll use the `infer_from_file` command to load a single encoded URL into the inference engine and print the results back out.

The result should tell us that the tokenized URL is legitimate (0) or fraud (1).  This sample data should return close to 1 in `out.main`.

```python
smoke_test = pd.DataFrame.from_records(
    [
    {
        "text_input":[
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            28,
            16,
            32,
            23,
            29,
            32,
            30,
            19,
            26,
            17
        ]
    }
]
)

result = aloha_pipeline.infer(smoke_test)
display(result.loc[:, ["time","out.main"]])
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
      <td>2023-10-20 21:15:26.393</td>
      <td>[0.997564]</td>
    </tr>
  </tbody>
</table>

## Undeploy Pipeline

When finished with our tests, we will undeploy the pipeline so we have the Kubernetes resources back for other tasks.  Note that if the deployment variable is unchanged aloha_pipeline.deploy() will restart the inference engine in the same configuration as before.

```python
aloha_pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s ..................................... ok

<table><tr><th>name</th> <td>tensorflowuploadexample</td></tr><tr><th>created</th> <td>2023-10-20 21:15:17.379310+00:00</td></tr><tr><th>last_updated</th> <td>2023-10-20 21:15:17.786848+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>ceffdd27-2887-4a32-ba90-4eec496aafb1, dce97e55-97d7-4bf9-8c75-8f9115e40e23</td></tr><tr><th>steps</th> <td>tensorflowuploadexample</td></tr><tr><th>published</th> <td>False</td></tr></table>

