This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-features/wallaroo-model-endpoints).

## External Pipeline Inference URL Tutorial

Wallaroo provides the ability to perform inferences through deployed pipelines via both internal and external URLs.  These URLs allow inferences to be performed by submitting data to the internal or external URL, with the inference results returned in the same format as the [InferenceResult Object](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/#run-inference-through-a-pipeline).

**Internal URLs** are available only through the internal Kubernetes environment hosting the Wallaroo instance as demonstrated in this tutorial.
**External URLs** are available outside of the Kubernetes environment, such as the public internet.  These are demonstrated in the External Pipeline Deployment URL Tutorial.

**IMPORTANT NOTE**:  Before starting this tutorial, the Internal Pipeline Deployment URL Tutorial must be completed to establish the Wallaroo workspace, pipeline and model to be used 

The following tutorial shows how to set up an environment and demonstrates how to use the External Deployment URL.  This example provides the following:

* For Arrow enabled instances:
  * `data_1.df.json`, `data_1k.df.json` and `data_25k.df.json`:  Sample data used for testing inferences with the sample model.
* For Arrow distabled instances:
  * `data_1.json`, `data_1k.json` and `data_25k.json`:  Sample data used for testing inferences with the sample model.

## Prerequisites

1. Before running this sample notebook, verify that the Internal Pipeline Deployment URL Tutorial has been run.  This will create the workspace, pipeline, etc for the below example to run.
1. Enable external URl inference endpoints through the Wallaroo Administrative Dashboard.  This can be accessed through the `kots` application as detailed in the [Wallaroo Install Guildes](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-install-guides/).

### Retrieve Token

There are two methods of retrieving the JWT token used to authenticate to the Wallaroo instance's API service:

* [Wallaroo SDK](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-api-essential-guide/#through-the-wallaroo-sdk).  This method requires a Wallaroo based user.
* [API Clent Secret](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-api-essential-guide/#through-keycloak).  This is the recommended method as it is user independent.  It allows any valid user to make an inference request.

This tutorial will use the Wallaroo SDK method for convenience, with examples on using the API Client Secret method.

### API Request Methods

All Wallaroo API endpoints follow the format:

* `https://$URLPREFIX.api.$URLSUFFIX/v1/api$COMMAND`

Where `$COMMAND` is the specific endpoint.  For example, for the command to list of workspaces in the Wallaroo instance would use the above format based on these settings:

* `$URLPREFIX`: `smooth-moose-1617`
* `$URLSUFFIX`: `example.wallaroo.ai`
* `$COMMAND`: `/workspaces/list`

This would create the following API endpoint:

* `https://smooth-moose-1617.api.example.wallaroo.ai/v1/api/workspaces/list`

### Connect to Wallaroo

For this example, a connection to the Wallaroo SDK is used.  This will be used to retrieve the JWT token as describe above.  Update `wallarooPrefix = "YOUR PREFIX"` and `wallarooSuffix = "YOUR SUFFIX"` to match the Wallaroo instance used for this demonstration.

```python
import wallaroo
from wallaroo.object import EntityNotFoundError
import pandas as pd

import requests
from requests.auth import HTTPBasicAuth

import json

# used to display dataframe information without truncating
from IPython.display import display
pd.set_option('display.max_colwidth', None)
```

```python
# Client connection from local Wallaroo instance

wl = wallaroo.Client()

# SSO login through keycloak

# wallarooPrefix = "YOUR PREFIX"
# wallarooSuffix = "YOUR SUFFIX"

# wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}.api.{wallarooSuffix}", 
#                     auth_endpoint=f"https://{wallarooPrefix}.keycloak.{wallarooSuffix}", 
#                     auth_type="sso")
```

```python
APIURL=f"https://{wallarooPrefix}.api.{wallarooSuffix}/v1/api"
```

### Arrow Support

As of the 2023.1 release, Wallaroo provides support for dataframe and Arrow for inference inputs.  This tutorial allows users to adjust their experience based on whether they have enabled Arrow support in their Wallaroo instance or not.

If Arrow support has been enabled, `arrowEnabled=True`. If disabled or you're not sure, set it to `arrowEnabled=False`

The examples below will be shown in an arrow enabled environment.

```python
import os
# Only set the below to make the OS environment ARROW_ENABLED to TRUE.  Otherwise, leave as is.
# os.environ["ARROW_ENABLED"]="True"

if "ARROW_ENABLED" not in os.environ or os.environ["ARROW_ENABLED"] == "False":
    arrowEnabled = False
else:
    arrowEnabled = True
print(arrowEnabled)
```

    True

## Verify Pipeline Deployed

We're using the same pipeline from the Internal Pipeline Inference URL Tutorial.  We'll deploy it via the SDK to verify it is running for the later tests.

```python
workspace_name = 'urldemoworkspace'
pipeline_name = 'urldemopipeline'
model_name = 'urldemomodel'
model_file_name = './alohacnnlstm.zip'

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
        pipeline = wl.pipelines_by_name(pipeline_name)[0]
    except EntityNotFoundError:
        pipeline = wl.build_pipeline(pipeline_name)
    return pipeline

workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)

pipeline = get_pipeline(pipeline_name)
pipeline

pipeline.deploy()
```

<table><tr><th>name</th> <td>urldemopipeline</td></tr><tr><th>created</th> <td>2023-02-27 17:55:12.813456+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-27 17:59:03.244180+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>6db21694-9e11-42cb-914c-1528549cedca, 930fe54d-9503-4768-8bf9-499f72272098, 54158104-c71d-4980-a6a3-25564c909b44</td></tr><tr><th>steps</th> <td>urldemomodel</td></tr></table>
{{</table>}}

### Get External Inference URL

The API command `/admin/get_pipeline_external_url` retrieves the external inference URL for a specific pipeline in a workspace.

* **Parameters**
  * **workspace_id** (*REQUIRED integer*):  The workspace integer id.
  * **pipeline_name** (*REQUIRED string*): The name of the pipeline.

In this example, a list of the workspaces will be retrieved.  Based on the setup from the Internal Pipeline Deployment URL Tutorial, the workspace matching `urlworkspace` will have it's **workspace id** stored and used for the `/admin/get_pipeline_external_url` request with the pipeline `urlpipeline`.

The External Inference URL will be stored as a variable for the next step.

**Modify these values to match the ones used in the Internal Pipeline Deployment URL Tutorial.**

```python
# Retrieve the token
connection =wl.mlops().__dict__
token = connection['token']

## Start with the a lists of the workspaces to verify the ID

workspaceName = "urldemoworkspace"
urlPipeline = "urldemopipeline"

# List workspaces

apiRequest = f"{APIURL}/workspaces/list"

data = {
}

headers= {
    'Authorization': 'Bearer ' + token,
    'Content-Type':'application/json'
}

response = requests.post(apiRequest, json=data, headers=headers, verify=True)
workspaces = response.json()
```

```python
workspaceList = workspaces['workspaces']
#print(workspaceList)
workspaceId = list(filter(lambda x:x["name"]==workspaceName,workspaceList))[0]['id']
```

```python
# Retrieve the token
connection =wl.mlops().__dict__
token = connection['token']

## Retrieve the pipeline's External Inference URL

apiRequest = f"{APIURL}/admin/get_pipeline_external_url"

data = {
    "workspace_id": workspaceId,
    "pipeline_name": urlPipeline
}

headers= {
    'Authorization': 'Bearer ' + token,
    'Content-Type':'application/json'
}

response = requests.post(apiRequest, json=data, headers=headers, verify=True).json()
externalUrl = response['url']
externalUrl
```

    'https://doc-test.api.wallaroocommunity.ninja/v1/api/pipelines/infer/urldemopipeline-11'

### Perform Inference Through External URL

The inference can now be performed through the External Inference URL.  This URL will accept the same inference data file that is used with the Wallaroo SDK, or with an Internal Inference URL as used in the Internal Pipeline Inference URL Tutorial.

For this example, the `externalUrl` retrieved through the [Get External Inference URL](#get-external-inference-url) is used to submit a single inference request through the data file `data-1.json`.

```python
# Retrieve the token
connection =wl.mlops().__dict__
token = connection['token']

## Inference through external URL

if arrowEnabled is True:
    # retrieve the json data to submit
    data = json.load(open('./data/data_1k.df.json','rb'))
    # set the headers
    headers= {
        'Authorization': 'Bearer ' + token,
        'Content-Type':'application/json; format=pandas-records'
    }
else:
    data = json.load(open('./data/data_1k.json','rb'))
    # set the headers
    headers= {
        'Authorization': 'Bearer ' + token
    }

# submit the request via POST
response = requests.post(externalUrl, json=data, headers=headers)

# Only the first 300 characters will be displayed for brevity
printResponse = json.dumps(response.json())
print(printResponse[0:300])

```

    [{"time": 1677520761514, "in": {"text_input": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 16, 32, 23, 29, 32, 30, 19, 26, 17]}, "out": {"corebot": [0.9829148], "banjori": [0.0015195871], "suppobox": [1.3889898e-27], "ma

### Undeploy the Pipeline

With the tutorial complete, we'll use the SDK to undeploy the pipeline and return the resources back to the Wallaroo instance.

```python
pipeline.undeploy()
```

<table><tr><th>name</th> <td>urldemopipeline</td></tr><tr><th>created</th> <td>2023-02-27 17:55:12.813456+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-27 17:59:03.244180+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>6db21694-9e11-42cb-914c-1528549cedca, 930fe54d-9503-4768-8bf9-499f72272098, 54158104-c71d-4980-a6a3-25564c909b44</td></tr><tr><th>steps</th> <td>urldemomodel</td></tr></table>
{{</table>}}

Wallaroo supports the ability to perform inferences through the SDK and through the API for each deployed pipeline.  For more information on how to use Wallaroo, see the [Wallaroo Documentation Site](https://docs.wallaroo.ai) for full details.

##
