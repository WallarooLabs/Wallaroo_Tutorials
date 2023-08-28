This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-features/wallaroo-model-endpoints).

## Wallaroo API Inference Tutorial

Wallaroo provides the ability to perform inferences through deployed pipelines via the Wallaroo SDK and the Wallaroo MLOps API.  This tutorial demonstrates performing inferences using the Wallaroo MLOps API.

This tutorial provides the following:

* `ccfraud.onnx`:  A pre-trained credit card fraud detection model.
* `data/cc_data_1k.arrow`, `data/cc_data_10k.arrow`: Sample testing data in Apache Arrow format with 1,000 and 10,000 records respectively.
* `wallaroo-model-endpoints-api.py`:  A code-only version of this tutorial as a Python script.

This tutorial and sample data comes from the Machine Learning Group's demonstration on [Credit Card Fraud detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

### Prerequisites

The following is required for this tutorial:

* A [deployed Wallaroo instance](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-install-guides/) with [Model Endpoints Enabled](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-model-endpoints-guide/)
* The following Python libraries:
  * `os`
  * `requests`
  * [`pandas`](https://pypi.org/project/pandas/)
  * [`pyarrow`](https://pypi.org/project/pyarrow/)
  * [`wallaroo`](https://pypi.org/project/wallaroo/) (Installed in the Wallaroo JupyterHub service by default).

### Tutorial Goals

This demonstration provides a quick tutorial on performing inferences using the Wallaroo MLOps API using a deployed pipeline's Inference URL.  This following steps will be performed:

* Connect to a Wallaroo instance using the Wallaroo SDK and environmental variables.  This bypasses the browser link confirmation for a seamless login, and provides a simple method of retrieving the JWT token used for Wallaroo MLOps API calls.  For more information, see the [Wallaroo SDK Essentials Guide:  Client Connection](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/) and the [Wallaroo MLOps API Essentials Guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-api-essential-guide/).
* Create a workspace for our models and pipelines.
* Upload the `ccfraud` model.
* Create a pipeline and add the `ccfraud` model as a pipeline step.
* Run sample inferences with pandas DataFrame inputs and Apache Arrow inputs.

### Retrieve Token

There are two methods of retrieving the JWT token used to authenticate to the Wallaroo instance's API service:

* [Wallaroo SDK](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-api-essential-guide/#through-the-wallaroo-sdk).  This method requires a Wallaroo based user.
* [API Client Secret](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-api-essential-guide/#through-keycloak).  This is the recommended method as it is user independent.  It allows any valid user to make an inference request.

This tutorial will use the Wallaroo SDK method for convenience with environmental variables for a seamless login without browser validation.  For more information, see the [Wallaroo SDK Essentials Guide: Client Connection](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

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

For this example, a connection to the Wallaroo SDK is used.  This will be used to retrieve the JWT token for the MLOps API calls.  

This example will store the user's credentials into the file `./creds.json` which contains the following:

```json
{
    "username": "{Connecting User's Username}", 
    "password": "{Connecting User's Password}", 
    "email": "{Connecting User's Email Address}"
}
```

Replace the `username`, `password`, and `email` fields with the user account connecting to the Wallaroo instance.  This allows a seamless connection to the Wallaroo instance and bypasses the standard browser based confirmation link.  For more information, see the [Wallaroo SDK Essentials Guide:  Client Connection](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

Update `wallarooPrefix = "YOUR PREFIX."` and `wallarooSuffix = "YOUR SUFFIX"` to match the Wallaroo instance used for this demonstration.  Note the `.` is part of the prefix.  If there is no prefix, then `wallarooPrefix = ""`

```python
import wallaroo
from wallaroo.object import EntityNotFoundError

import pandas as pd
import os
import base64

import pyarrow as pa

import requests
from requests.auth import HTTPBasicAuth

# Used to create unique workspace and pipeline names
import string
import random

# make a random 4 character prefix
suffix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))
display(suffix)

import json

# used to display dataframe information without truncating
from IPython.display import display
pd.set_option('display.max_colwidth', None)
```

    'atwc'

```python
# Retrieve the login credentials.
os.environ["WALLAROO_SDK_CREDENTIALS"] = './creds.json.example'

# wl = wallaroo.Client(auth_type="user_password")

# Client connection from local Wallaroo instance
wallarooPrefix = ""
wallarooSuffix = "autoscale-uat-ee.wallaroo.dev"

wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}api.{wallarooSuffix}", 
                    auth_endpoint=f"https://{wallarooPrefix}keycloak.{wallarooSuffix}", 
                    auth_type="user_password")

```

```python
wallarooPrefix = "YOUR PREFIX."
wallarooPrefix = "YOUR SUFFIX"

wallarooPrefix = ""
wallarooSuffix = "autoscale-uat-ee.wallaroo.dev"

APIURL=f"https://{wallarooPrefix}api.{wallarooSuffix}"
APIURL
```

    'https://api.autoscale-uat-ee.wallaroo.dev'

## Retrieve the JWT Token

As mentioned earlier, there are multiple methods of authenticating to the Wallaroo instance for MLOps API calls.  This tutorial will use the Wallaroo SDK method Wallaroo Client `wl.auth.auth_header()` method, extracting the token from the response.

Reference:  [MLOps API Retrieve Token Through Wallaroo SDK](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-api-essential-guide/#through-the-wallaroo-sdk)

```python
# Retrieve the token
headers = wl.auth.auth_header()
display(headers)
```

    {'Authorization': 'Bearer eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJEWkc4UE4tOHJ0TVdPdlVGc0V0RWpacXNqbkNjU0tJY3Zyak85X3FxcXc0In0.eyJleHAiOjE2ODg3NTE2NjQsImlhdCI6MTY4ODc1MTYwNCwianRpIjoiNGNmNmFjMzQtMTVjMy00MzU0LWI0ZTYtMGYxOWIzNjg3YmI2IiwiaXNzIjoiaHR0cHM6Ly9rZXljbG9hay5hdXRvc2NhbGUtdWF0LWVlLndhbGxhcm9vLmRldi9hdXRoL3JlYWxtcy9tYXN0ZXIiLCJhdWQiOlsibWFzdGVyLXJlYWxtIiwiYWNjb3VudCJdLCJzdWIiOiJkOWE3MmJkOS0yYTFjLTQ0ZGQtOTg5Zi0zYzdjMTUxMzA4ODUiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJzZGstY2xpZW50Iiwic2Vzc2lvbl9zdGF0ZSI6Ijk0MjkxNTAwLWE5MDgtNGU2Ny1hMzBiLTA4MTczMzNlNzYwOCIsImFjciI6IjEiLCJyZWFsbV9hY2Nlc3MiOnsicm9sZXMiOlsiZGVmYXVsdC1yb2xlcy1tYXN0ZXIiLCJvZmZsaW5lX2FjY2VzcyIsInVtYV9hdXRob3JpemF0aW9uIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsibWFzdGVyLXJlYWxtIjp7InJvbGVzIjpbIm1hbmFnZS11c2VycyIsInZpZXctdXNlcnMiLCJxdWVyeS1ncm91cHMiLCJxdWVyeS11c2VycyJdfSwiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwic2NvcGUiOiJwcm9maWxlIGVtYWlsIiwic2lkIjoiOTQyOTE1MDAtYTkwOC00ZTY3LWEzMGItMDgxNzMzM2U3NjA4IiwiZW1haWxfdmVyaWZpZWQiOmZhbHNlLCJodHRwczovL2hhc3VyYS5pby9qd3QvY2xhaW1zIjp7IngtaGFzdXJhLXVzZXItaWQiOiJkOWE3MmJkOS0yYTFjLTQ0ZGQtOTg5Zi0zYzdjMTUxMzA4ODUiLCJ4LWhhc3VyYS1kZWZhdWx0LXJvbGUiOiJ1c2VyIiwieC1oYXN1cmEtYWxsb3dlZC1yb2xlcyI6WyJ1c2VyIl0sIngtaGFzdXJhLXVzZXItZ3JvdXBzIjoie30ifSwibmFtZSI6IkpvaG4gSGFuc2FyaWNrIiwicHJlZmVycmVkX3VzZXJuYW1lIjoiam9obi5odW1tZWxAd2FsbGFyb28uYWkiLCJnaXZlbl9uYW1lIjoiSm9obiIsImZhbWlseV9uYW1lIjoiSGFuc2FyaWNrIiwiZW1haWwiOiJqb2huLmh1bW1lbEB3YWxsYXJvby5haSJ9.QE5WJ6NI5bQob0p2M7KsVXxrAiUUxnsIjZPuHIx7_6kTsDt4zarcCu2b5X6s6wg0EZQDX22oANWUAXnkWRTQd_E6zE7DkKF7H5kodtyu90ewiFM8ULx2iOWy2GkafQTdiuW90-BGDIjAcOiQtOkdHNaNHqJ9go2Lsom1t_b4-FOhh8bAGhMM3aDS0w-Y8dGKClxW_xFSTmOjNLaPxbFs5NCib-_QAsR_PiyfSFNJ_kjIV8f2mdzeyOauj0YOE-w5nXjhbrDvhS1kJ3n_8C2J2eOnEg85OGd3m6VKVzoR7oPzoZH15Jtl8shKTDS6BEUWpzZNfjYjwZdy1KTenCbzAQ'}

## Create Workspace

In a production environment, the Wallaroo workspace that contains the pipeline and models would be created and deployed.  We will quickly recreate those steps using the MLOps API.  If the workspace and pipeline have already been created through the [Wallaroo SDK Inference Tutorial](https://docs.wallaroo.ai/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-endpoints/wallaroo-external-inference-tutorial/), then we can skip directly to [Deploy Pipeline](#deploy-pipeline).

Workspaces are created through the MLOps API with the `/v1/api/workspaces/create` command.  This requires the workspace name be provided, and that the workspace not already exist in the Wallaroo instance.

Reference: [MLOps API Create Workspace](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-api-essential-guide/wallaroo-mlops-api-essential-guide-workspaces/#create-workspace)

```python
# Retrieve the token
headers = wl.auth.auth_header()

# set Content-Type type
headers['Content-Type']='application/json'

# Create workspace
apiRequest = f"{APIURL}/v1/api/workspaces/create"

workspace_name = f"apiinferenceexampleworkspace{suffix}"

data = {
  "workspace_name": workspace_name
}

response = requests.post(apiRequest, json=data, headers=headers, verify=True).json()
display(response)
# Stored for future examples
workspaceId = response['workspace_id']
```

    {'workspace_id': 374}

## Upload Model

The model is uploaded using the `/v1/api/models/upload_and_convert` command.  This uploads a ML Model to a Wallaroo workspace via POST with `Content-Type: multipart/form-data` and takes the following parameters:

* **Parameters**
  * **name** - (*REQUIRED string*): Name of the model
  * **visibility** - (*OPTIONAL string*): The visibility of the model as either `public` or `private`.
  * **workspace_id** - (*REQUIRED int*): The numerical id of the workspace to upload the model to.  Stored earlier as `workspaceId`.

Directly after we will use the `/models/list_versions` to retrieve model details used for later steps.

Reference: [Wallaroo MLOps API Essentials Guide: Model Management: Upload Model to Workspace](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-api-essential-guide/wallaroo-mlops-api-essential-guide-models/)

```python
## upload model

# Retrieve the token
headers = wl.auth.auth_header()

apiRequest = f"{APIURL}/v1/api/models/upload_and_convert"

framework='onnx'

model_name = f"{suffix}ccfraud"

data = {
    "name": model_name,
    "visibility": "public",
    "workspace_id": workspaceId,
    "conversion": {
        "framework": framework,
        "python_version": "3.8",
        "requirements": []
    }
}

files = {
    "metadata": (None, json.dumps(data), "application/json"),
    'file': (model_name, open('./ccfraud.onnx', 'rb'), "application/octet-stream")
    }

response = requests.post(apiRequest, files=files, headers=headers).json()
display(response)
modelId=response['insert_models']['returning'][0]['models'][0]['id']
```

    {'insert_models': {'returning': [{'models': [{'id': 176}]}]}}

```python
# Get the model details

# Retrieve the token
headers = wl.auth.auth_header()

# set Content-Type type
headers['Content-Type']='application/json'

apiRequest = f"{APIURL}/v1/api/models/get_by_id"

data = {
  "id": modelId
}

response = requests.post(apiRequest, json=data, headers=headers, verify=True).json()
display(response)
```

    {'msg': 'The provided model id was not found.', 'code': 400}

```python
# Get the model details

# Retrieve the token
headers = wl.auth.auth_header()

# set Content-Type type
headers['Content-Type']='application/json'

apiRequest = f"{APIURL}/v1/api/models/list_versions"

data = {
  "model_id": model_name,
  "models_pk_id" : modelId
}

response = requests.post(apiRequest, json=data, headers=headers, verify=True).json()
display(response)
```

    [{'sha': 'bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507',
      'models_pk_id': 175,
      'model_version': 'fa4c2f8c-769e-4ee1-9a91-fe029a4beffc',
      'owner_id': '""',
      'model_id': 'vsnaccfraud',
      'id': 176,
      'file_name': 'vsnaccfraud',
      'image_path': 'proxy.replicated.com/proxy/wallaroo/ghcr.io/wallaroolabs/mlflow-deploy:v2023.3.0-main-3481',
      'status': 'ready'},
     {'sha': 'bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507',
      'models_pk_id': 175,
      'model_version': '701be439-8702-4896-88b5-644bb5cb4d61',
      'owner_id': '""',
      'model_id': 'vsnaccfraud',
      'id': 175,
      'file_name': 'vsnaccfraud',
      'image_path': 'proxy.replicated.com/proxy/wallaroo/ghcr.io/wallaroolabs/mlflow-deploy:v2023.3.0-main-3481',
      'status': 'ready'}]

```python
model_version = response[0]['model_version']
display(model_version)
model_sha = response[0]['sha']
display(model_sha)
```

    'fa4c2f8c-769e-4ee1-9a91-fe029a4beffc'

    'bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507'

## Create Pipeline

Create Pipeline in a Workspace with the `/v1/api/pipelines/create` command.  This creates a new pipeline in the specified workspace.

* **Parameters**
  * **pipeline_id** - (REQUIRED string): Name of the new pipeline.
  * **workspace_id** - (REQUIRED int): Numerical id of the workspace for the new pipeline.  Stored earlier as `workspaceId`.
  * **definition** - (REQUIRED string): Pipeline definitions, can be `{}` for none.

For our example, we are setting the pipeline steps through the `definition` field.  This will direct inference requests to the model before output.

Reference: [Wallaroo MLOps API Essentials Guide: Pipeline Management: Create Pipeline in a Workspace](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-api-essential-guide/wallaroo-mlops-api-essential-guide-pipelines/#create-pipeline-in-a-workspace)

```python
# Create pipeline

# Retrieve the token
headers = wl.auth.auth_header()

# set Content-Type type
headers['Content-Type']='application/json'

apiRequest = f"{APIURL}/v1/api/pipelines/create"

pipeline_name=f"{suffix}apiinferenceexamplepipeline"

data = {
  "pipeline_id": pipeline_name,
  "workspace_id": workspaceId,
  "definition": {'steps': [{'ModelInference': {'models': [{'name': f'{model_name}', 'version': model_version, 'sha': model_sha}]}}]}
}

response = requests.post(apiRequest, json=data, headers=headers, verify=True).json()

pipeline_id = response['pipeline_pk_id']
pipeline_variant_id=response['pipeline_variant_pk_id']
pipeline_variant_version=['pipeline_variant_version']
```

## Deploy Pipeline

With the pipeline created and the model uploaded into the workspace, the pipeline can be deployed.  This will allocate resources from the Kubernetes cluster hosting the Wallaroo instance and prepare the pipeline to process inference requests.

Pipelines are deployed through the MLOps API command `/v1/api/pipelines/deploy` which takes the following parameters:

* **Parameters**
  * **deploy_id** (*REQUIRED string*): The name for the pipeline deployment.
  * **engine_config** (*OPTIONAL string*): Additional configuration options for the pipeline.
  * **pipeline_version_pk_id** (*REQUIRED int*): Pipeline version id.  Captured earlier as `pipeline_variant_id`.
  * **model_configs** (*OPTIONAL Array int*): Ids of model configs to apply.
  * **model_ids** (*OPTIONAL Array int*): Ids of models to apply to the pipeline.  If passed in, model_configs will be created automatically.
  * **models** (*OPTIONAL Array models*):  If the model ids are not available as a pipeline step, the models' data can be passed to it through this method.  The options below are only required if `models` are provided as a parameter.
    * **name** (*REQUIRED string*): Name of the uploaded model that is in the same workspace as the pipeline.  Captured earlier as the `model_name` variable.
    * **version** (*REQUIRED string*): Version of the model to use.  
    * **sha** (*REQUIRED string*): SHA value of the model.
  * **pipeline_id** (*REQUIRED int*): Numerical value of the pipeline to deploy.
* **Returns**
  * **id** (*int*): The deployment id.

Reference: [Wallaroo MLOps API Essentials Guide: Pipeline Management: Deploy a Pipeline](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-api-essential-guide/wallaroo-mlops-api-essential-guide-pipelines/#deploy-a-pipeline)

```python
# Deploy Pipeline

# Retrieve the token
headers = wl.auth.auth_header()

# set Content-Type type
headers['Content-Type']='application/json'

apiRequest = f"{APIURL}/v1/api/pipelines/deploy"

exampleModelDeployId=pipeline_name

data = {
    "deploy_id": exampleModelDeployId,
    "pipeline_version_pk_id": pipeline_variant_id,
    "model_ids": [
        modelId
    ],
    "pipeline_id": pipeline_id
}

response = requests.post(apiRequest, json=data, headers=headers, verify=True).json()
display(response)
exampleModelDeploymentId=response['id']

# wait 45 seconds for the pipeline to complete deployment
import time
time.sleep(45)
```

    {'id': 260}

### Get Deployment Status

This returns the deployment status - we're waiting until the deployment has the status "Ready."

* **Parameters**
  * **name** - (REQUIRED string): The deployment in the format {deployment_name}-{deploymnent-id}.
  
Example: The deployed empty and model pipelines status will be displayed.

```python
# Retrieve the token
headers = wl.auth.auth_header()

# set Content-Type type
headers['Content-Type']='application/json'

# Get model pipeline deployment

api_request = f"{APIURL}/v1/api/status/get_deployment"

data = {
  "name": f"{pipeline_name}-{exampleModelDeploymentId}"
}

response = requests.post(api_request, json=data, headers=headers, verify=True).json()
response
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.17.3',
       'name': 'engine-f77b5c44b-4j2n5',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'vsnaapiinferenceexamplepipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'vsnaccfraud',
          'version': 'fa4c2f8c-769e-4ee1-9a91-fe029a4beffc',
          'sha': 'bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.17.4',
       'name': 'engine-lb-584f54c899-q877m',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}

## Get External Inference URL

The API command `/admin/get_pipeline_external_url` retrieves the external inference URL for a specific pipeline in a workspace.

* **Parameters**
  * **workspace_id** (*REQUIRED integer*):  The workspace integer id.
  * **pipeline_name** (*REQUIRED string*): The name of the pipeline.

In this example, a list of the workspaces will be retrieved.  Based on the setup from the Internal Pipeline Deployment URL Tutorial, the workspace matching `urlworkspace` will have it's **workspace id** stored and used for the `/admin/get_pipeline_external_url` request with the pipeline `urlpipeline`.

The External Inference URL will be stored as a variable for the next step.

Reference: [Wallaroo MLOps API Essentials Guide: Pipeline Management: Get External Inference URL](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-api-essential-guide/wallaroo-mlops-api-essential-guide-pipelines/#get-external-inference-url)

```python
# Retrieve the token
headers = wl.auth.auth_header()

# set Content-Type type
headers['Content-Type']='application/json'

## Retrieve the pipeline's External Inference URL

apiRequest = f"{APIURL}/v1/api/admin/get_pipeline_external_url"

data = {
    "workspace_id": workspaceId,
    "pipeline_name": pipeline_name
}

response = requests.post(apiRequest, json=data, headers=headers, verify=True).json()
deployurl = response['url']
deployurl

```

    'https://api.autoscale-uat-ee.wallaroo.dev/v1/api/pipelines/infer/vsnaapiinferenceexamplepipeline-260/vsnaapiinferenceexamplepipeline'

### Perform Inference Through External URL

The inference can now be performed through the External Inference URL.  This URL will accept the same inference data file that is used with the Wallaroo SDK, or with an Internal Inference URL as used in the Internal Pipeline Inference URL Tutorial.

For this example, the `externalUrl` retrieved through the [Get External Inference URL](#get-external-inference-url) is used to submit a single inference request through the data file `data-1.json`.

Reference: [Wallaroo MLOps API Essentials Guide: Pipeline Management: Perform Inference Through External URL](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-api-essential-guide/wallaroo-mlops-api-essential-guide-pipelines/#perform-inference-through-external-url)

```python
# Retrieve the token
headers = wl.auth.auth_header()

# set Content-Type type
headers['Content-Type']='application/json; format=pandas-records'

## Inference through external URL using dataframe

# retrieve the json data to submit
data = [
    {
        "tensor":[
            1.0678324729,
            0.2177810266,
            -1.7115145262,
            0.682285721,
            1.0138553067,
            -0.4335000013,
            0.7395859437,
            -0.2882839595,
            -0.447262688,
            0.5146124988,
            0.3791316964,
            0.5190619748,
            -0.4904593222,
            1.1656456469,
            -0.9776307444,
            -0.6322198963,
            -0.6891477694,
            0.1783317857,
            0.1397992467,
            -0.3554220649,
            0.4394217877,
            1.4588397512,
            -0.3886829615,
            0.4353492889,
            1.7420053483,
            -0.4434654615,
            -0.1515747891,
            -0.2668451725,
            -1.4549617756
        ]
    }
]

# submit the request via POST, import as pandas DataFrame
response = pd.DataFrame.from_records(
    requests.post(
        deployurl, 
        json=data, 
        headers=headers)
        .json()
    )

display(response.loc[:,["time", "out"]])
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1688750664105</td>
      <td>{'dense_1': [0.0014974177]}</td>
    </tr>
  </tbody>
</table>

```python
# Retrieve the token
headers = wl.auth.auth_header()

# set Content-Type type
headers['Content-Type']='application/vnd.apache.arrow.file'

# set accept as apache arrow table
headers['Accept']="application/vnd.apache.arrow.file"

# Submit arrow file
dataFile="./data/cc_data_10k.arrow"

data = open(dataFile,'rb').read()

response = requests.post(
                    deployurl, 
                    headers=headers, 
                    data=data, 
                    verify=True
                )

# Arrow table is retrieved 
with pa.ipc.open_file(response.content) as reader:
    arrow_table = reader.read_all()

# convert to Polars DataFrame and display the first 5 rows
display(arrow_table.to_pandas().head(5).loc[:,["time", "out"]])
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1688750664889</td>
      <td>{'dense_1': [0.99300325]}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1688750664889</td>
      <td>{'dense_1': [0.99300325]}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1688750664889</td>
      <td>{'dense_1': [0.99300325]}</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1688750664889</td>
      <td>{'dense_1': [0.99300325]}</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1688750664889</td>
      <td>{'dense_1': [0.0010916889]}</td>
    </tr>
  </tbody>
</table>

### Undeploy the Pipeline

With the tutorial complete, we'll undeploy the pipeline with `/v1/api/pipelines/undeploy` and return the resources back to the Wallaroo instance.

Reference: [Wallaroo MLOps API Essentials Guide: Pipeline Management: Undeploy a Pipeline](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-api-essential-guide/wallaroo-mlops-api-essential-guide-pipelines/#undeploy-a-pipeline)

```python
# Retrieve the token
headers = wl.auth.auth_header()

# set Content-Type type
headers['Content-Type']='application/json'

apiRequest = f"{APIURL}/v1/api/pipelines/undeploy"

data = {
    "pipeline_id": pipeline_id,
    "deployment_id":exampleModelDeploymentId
}

response = requests.post(apiRequest, json=data, headers=headers, verify=True).json()
display(response)
```

    None

Wallaroo supports the ability to perform inferences through the SDK and through the API for each deployed pipeline.  For more information on how to use Wallaroo, see the [Wallaroo Documentation Site](https://docs.wallaroo.ai) for full details.

##
