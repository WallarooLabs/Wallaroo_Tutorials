This tutorial and the assets are available as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/development/mlops_api).

## Wallaroo MLOps API Pipeline Management Tutorial

This tutorial focuses on using the Wallaroo MLOps API for Wallaroo pipeline management.  For this tutorial, we will be using the Wallaroo SDK to provide authentication credentials for ease of use examples.  See the [Wallaroo API Guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/) for full details on using the Wallaroo MLOps API.

### References

The following references are available for more information about Wallaroo and the Wallaroo MLOps API:

* [Wallaroo Documentation Site](https://docs.wallaroo.ai):  The Wallaroo Documentation Site
* Wallaroo MLOps API Documentation from a Wallaroo instance:  A Swagger UI based documentation is available from your Wallaroo instance at `https://{Wallaroo Prefix.}api.{Wallaroo Suffix}/v1/api/docs`.  For example, if the Wallaroo Instance suffix is `example.wallaroo.ai` with the prefix `{lovely-rhino-5555.}`, then the Wallaroo MLOps API Documentation would be available at `https://lovely-rhino-5555.api.example.wallaroo.ai/v1/api/docs`.  Note the `.` is part of the prefix.
* For another example, a Wallaroo Enterprise users who do not use a prefix and has the suffix `wallaroo.example.wallaroo.ai`, the the Wallaroo MLOps API Documentation would be available at `https://api.wallaroo.example.wallaroo.ai/v1/api/docs`.  For more information, see the [Wallaroo Documentation Site](https://docs.wallaroo.ai).

**IMPORTANT NOTE**:  The Wallaroo MLOps API is provided as an early access features.  Future iterations may adjust the methods and returns to provide a better user experience.  Please refer to this guide for updates.

## Prerequisites

* An installed Wallaroo instance.
* The following Python libraries installed:
  * `requests`
  * `json`
  * [`wallaroo`](https://pypi.org/project/wallaroo/): The Wallaroo SDK. Included with the Wallaroo JupyterHub service by default.

## Connection Steps

### Import Libraries

For these examples, we will rely on the `wallaroo` SDK and `requests` library for making connections to our sample Wallaroo Ops instance.

```python
import wallaroo

import requests

import json

import pandas as pd
```

### Connect to the Wallaroo Instance

The next step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
# Login through local Wallaroo instance

wl = wallaroo.Client()

## blank space to log in 

wl = wallaroo.Client()

wallarooPrefix = "doc-test."
wallarooSuffix = "wallarooexample.ai"

wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}api.{wallarooSuffix}", 
                    auth_endpoint=f"https://{wallarooPrefix}keycloak.{wallarooSuffix}", 
                    auth_type="sso")
```

### Retrieve API Service URL

The Wallaroo SDK provides the API endpoint through the `wallaroo.client.api_endpoint` variable.  This is derived from the Wallaroo OPs DNS settings.

The method `wallaroo.client.auth.auth_header()` retrieves the HTTP authorization headers for the API connection.

Both of these are used to authenticate for the Wallaroo MLOps API calls used in the future examples.

* References
  * [Wallaroo API Connection Guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-connection-guide/)

```python
display(wl.api_endpoint)
display(wl.auth.auth_header())
```

    'https://doc-test.api.wallarooexample.ai'

    {'Authorization': 'Bearer eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJKNXZOUVVIajVNa3lOV2pLUkwtUGZZSXJ2S3Z5YUx3eThJZFB2dktrZnRnIn0.eyJleHAiOjE3MDEzNjg0ODQsImlhdCI6MTcwMTM2ODQyNCwiYXV0aF90aW1lIjoxNzAxMzY4NDE4LCJqdGkiOiIyNTA3NjRjMy0yNTVjLTQyNDYtYjNkMy1kNTA3OTFmZWNhOGUiLCJpc3MiOiJodHRwczovL2RvYy10ZXN0LmtleWNsb2FrLndhbGxhcm9vY29tbXVuaXR5Lm5pbmphL2F1dGgvcmVhbG1zL21hc3RlciIsImF1ZCI6WyJtYXN0ZXItcmVhbG0iLCJhY2NvdW50Il0sInN1YiI6IjEyZWEwOWQxLTBmNDktNDA1ZS1iZWQxLTI3ZWI2ZDEwZmRlNCIsInR5cCI6IkJlYXJlciIsImF6cCI6InNkay1jbGllbnQiLCJzZXNzaW9uX3N0YXRlIjoiZTU2YWNhM2QtN2JiOC00NzNmLTk2ZjUtNjMzMmEyZThlMDZhIiwiYWNyIjoiMSIsInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJkZWZhdWx0LXJvbGVzLW1hc3RlciIsIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iXX0sInJlc291cmNlX2FjY2VzcyI6eyJtYXN0ZXItcmVhbG0iOnsicm9sZXMiOlsibWFuYWdlLXVzZXJzIiwidmlldy11c2VycyIsInF1ZXJ5LWdyb3VwcyIsInF1ZXJ5LXVzZXJzIl19LCJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6InByb2ZpbGUgZW1haWwiLCJzaWQiOiJlNTZhY2EzZC03YmI4LTQ3M2YtOTZmNS02MzMyYTJlOGUwNmEiLCJlbWFpbF92ZXJpZmllZCI6ZmFsc2UsImh0dHBzOi8vaGFzdXJhLmlvL2p3dC9jbGFpbXMiOnsieC1oYXN1cmEtdXNlci1pZCI6IjEyZWEwOWQxLTBmNDktNDA1ZS1iZWQxLTI3ZWI2ZDEwZmRlNCIsIngtaGFzdXJhLWRlZmF1bHQtcm9sZSI6InVzZXIiLCJ4LWhhc3VyYS1hbGxvd2VkLXJvbGVzIjpbInVzZXIiXSwieC1oYXN1cmEtdXNlci1ncm91cHMiOiJ7fSJ9LCJuYW1lIjoiSm9obiBIYW5zYXJpY2siLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJqb2huLmh1bW1lbEB3YWxsYXJvby5haSIsImdpdmVuX25hbWUiOiJKb2huIiwiZmFtaWx5X25hbWUiOiJIYW5zYXJpY2siLCJlbWFpbCI6ImpvaG4uaHVtbWVsQHdhbGxhcm9vLmFpIn0.RUe1fRpHnb8Dp0QnQkCiAJpnwNoHYgV1TdpFNQRLIVoykvwUpzX5ECC7MDBljd1uHrSdRAmOSvnUaXEEUeEopPekQ2vrcPY4Wk7TP9DlvxLL_iGTEd8C-CrnlQs8YG7TlSEAZd8ohSLeDLHedP0Z6vNV9z8fTKfcPogdijXIXVOv1Jhp8N3vmU_iQL9BKSJ3W9H2YeB-sNFqAkL9fjrtjkg3Qui5BbYwsfbubC0T-Xf0lYFZJ2COLxyOUS5jWYwFZDAJx7tFvTkuy9smJA8LNbHq2-QG8BtgFQMaHM8IBeXMJocOQzmm7YsHngdQjL-ezzlFN-5VjnPD_Rpjx3K5yQ'}

## Pipeline Management

Pipelines are managed through the Wallaroo API or the Wallaroo SDK.  Pipelines are the vehicle used for deploying, serving, and monitoring ML models.  For more information, see the [Wallaroo Glossary](https://docs.wallaroo.ai/wallaroo-glossary/).

### Create Pipeline in a Workspace

* **Endpoint**: `/v1/api/pipelines/create`

Creates a new pipeline in the specified workspace.

#### Create Pipeline in a Workspace Parameters

| Field | &nbsp; | Type | Description |
|---|---|---|---|
| **pipeline_id** | &nbsp; | *String* (*Required*) | Name of the new pipeline. |
| **workspace_id** | &nbsp; | *Integer* (*Required*) | Numerical id of the workspace for the new pipeline. |
| **definition** | &nbsp; | *String* (*Required*) | Pipeline definitions, can be `{}` for none.  This is where the pipeline steps are set. |
| &nbsp; | **steps** | *List[steps]* | The pipeline steps to add to the pipeline. |

##### Model Inference Pipeline Step

Pipeline steps from models follow the following schema.

```json
{
  "ModelInference": {
    "models": [
      {
        "name": "{name of model: String}",
        "version": {model version: Integer},
        "sha": "{model sha: String}"
      }
    ]
  }
}

```

### Create Pipeline in a Workspace Returns

| Field | Type | Description |
|---|---|---|
| **pipeline_pk_id** | *Integer* | The pipeline id. |
| **pipeline_variant_pk_id** | *Integer* | The pipeline version id. |
| **pipeline_variant_version** | *String* | The pipeline version UUID identifier. |

### Create Pipeline in a Workspace Examples

Two pipelines are created in the workspace id `10`.  This assumes that the workspace is created and has models uploaded to it.

One pipeline is an empty pipeline without any models.  

For the other pipeline, sample models are uploaded then added pipeline.

The pipeline id, variant id, and variant version of each pipeline will be stored for later examples.

Create empty pipeline via Requests.

```python
# Create pipeline in a workspace
# Retrieve the token 
headers = wl.auth.auth_header()

endpoint = f"{wl.api_endpoint}/v1/api/pipelines/create"

example_workspace_id = 10

data = {
  "pipeline_id": "api-empty-pipeline",
  "workspace_id": example_workspace_id,
  "definition": {}
}

response = requests.post(endpoint, 
                         json=data, 
                         headers=headers, 
                         verify=True).json()

empty_pipeline_id = response['pipeline_pk_id']
empty_pipeline_variant_id=response['pipeline_variant_pk_id']
example_pipeline_variant_version=['pipeline_variant_version']
display(json.dumps(response))
```

    '{"pipeline_pk_id": 25, "pipeline_variant_pk_id": 26, "pipeline_variant_version": "c29a277a-10b9-48f9-a738-aafb296df8c2"}'

Create empty pipeline via curl.

```python
!curl {wl.api_endpoint}/v1/api/pipelines/create \
    -H "Authorization: {wl.auth.auth_header()['Authorization']}" \
    -H "Content-Type: application/json" \
    --data '{{"pipeline_id": "api-empty-pipeline", "workspace_id": {example_workspace_id},"definition": {{}} }}'
```

    {"pipeline_pk_id":25,"pipeline_variant_pk_id":27,"pipeline_variant_version":"f6241f32-85a8-4ad8-9e71-da2763717811"}

Create pipeline with model steps via Requests.

```python
# Create pipeline in a workspace with models

# First upload a model
# Retrieve the token 
headers = wl.auth.auth_header()

endpoint = f"{wl.api_endpoint}/v1/api/models/upload_and_convert"

workspace_id = 10

framework='onnx'

example_model_name = f"api-sample-model"

metadata = {
    "name": example_model_name,
    "visibility": "public",
    "workspace_id": workspace_id,
    "conversion": {
        "framework": framework,
        "python_version": "3.8",
        "requirements": []
    }
}

files = {
    "metadata": (None, json.dumps(metadata), "application/json"),
    'file': (example_model_name, open('./models/ccfraud.onnx', 'rb'), "application/octet-stream")
    }

response = requests.post(endpoint, files=files, headers=headers).json()

example_model_id = response['insert_models']['returning'][0]['models'][0]['id']

# Second, get the model version

# Retrieve the token 
headers = wl.auth.auth_header()
endpoint = f"{wl.api_endpoint}/v1/api/models/list_versions"

data = {
  "model_id": example_model_name,
  "models_pk_id": example_model_id
}

response = requests.post(endpoint, json=data, headers=headers, verify=True).json()
example_model_sha = response[-1]['sha']
example_model_version = response[-1]['model_version']

# Now create the pipeline with the new model
# Retrieve the token 
headers = wl.auth.auth_header()

endpoint = f"{wl.api_endpoint}/v1/api/pipelines/create"

data = {
  "pipeline_id": "api-pipeline-with-models",
  "workspace_id": example_workspace_id,
  "definition": {
      'steps': [
          {
            'ModelInference': 
              {
                  'models': 
                    [
                        {
                            'name': example_model_name, 
                            'version': example_model_version, 
                            'sha': example_model_sha
                        }
                    ]
              }
          }
        ]
      }
    }

response = requests.post(endpoint, 
                         json=data, 
                         headers=headers, 
                         verify=True).json()
display(json.dumps(response))

# saved for later steps

model_pipeline_id = response['pipeline_pk_id']
model_pipeline_variant_id=response['pipeline_variant_pk_id']
model_pipeline_variant_version=['pipeline_variant_version']
```

    '{"pipeline_pk_id": 28, "pipeline_variant_pk_id": 29, "pipeline_variant_version": "5d0326fa-6753-4252-bb56-3b2106a8c671"}'

Create pipeline with model steps via curl.

```python
!curl {wl.api_endpoint}/v1/api/pipelines/create \
    -H "Authorization: {wl.auth.auth_header()['Authorization']}" \
    -H "Content-Type: application/json" \
    --data '{{ \
    "pipeline_id": "api-pipeline-with-models", \
    "workspace_id": {example_workspace_id}, \
    "definition": {{ \
        "steps": [ \
            {{ \
              "ModelInference": \
                {{ \
                    "models": \
                      [ \
                          {{ \
                              "name": "{example_model_name}", \
                              "version": "{example_model_version}", \
                              "sha": "{example_model_sha}" \
                          }} \
                      ] \
                }} \
            }} \
          ] \
        }} \
      }}'
```

    {"pipeline_pk_id":28,"pipeline_variant_pk_id":30,"pipeline_variant_version":"82148e63-3950-4ec4-99f7-b9a22212bfdf"}

### Deploy a Pipeline

* **Endpoint**: `/v1/api/pipelines/deploy`

Deploy a an existing pipeline.  Note that for any pipeline that has model steps, they must be included either in `model_configs`, `model_ids` or `models`.

#### Deploy a Pipeline Parameters

| Field | &nbsp; | Type | Description |
|---|---|---|---|
| **deploy_id** | &nbsp; | *String* (*REQUIRED*) | The name for the pipeline deployment.  This **must** match the name of the pipeline being deployed. |
| **engine_config** | &nbsp; | *String* (*OPTIONAL*) | Additional configuration options for the pipeline. |
| **pipeline_version_pk_id** | &nbsp; | *Integer* *REQUIRED*) | Pipeline version id. |
| **model_configs** | &nbsp; | *List[Integer]* (*OPTIONAL*) | Ids of model configs to apply. |
| **model_ids** | &nbsp; | *List[Integer]* (*OPTIONAL*) | Ids of models to apply to the pipeline.  If passed in, model_configs will be created automatically. |
| **models** | &nbsp; | *List[models]* (*OPTIONAL*) | If the model ids are not available as a pipeline step, the models' data can be passed to it through this method.  The options below are only required if `models` are provided as a parameter. |
| &nbsp; | **name** | *String* (*REQUIRED*) | Name of the uploaded model that is in the same workspace as the pipeline. |
| &nbsp; | **version** | *String* (*REQUIRED*) | Version of the model to use. |
| &nbsp; | **sha** | *String* (*REQUIRED*) | SHA value of the model. |
| **pipeline_id** | &nbsp; | *Integer (*REQUIRED*) | Numerical value of the pipeline to deploy. |

#### Deploy a Pipeline Returns

| Field | Type | Description |
|---|---|---|
| **id** | *Integer* | The deployment id. |

#### Deploy a Pipeline Returns

The pipeline with models created in the step [Create Pipeline in a Workspace](#create-pipeline-in-a-workspace) will be deployed and their deployment information saved for later examples.

Deploy pipeline via Requests.

```python
# Deploy a pipeline with models

# Retrieve the token 
headers = wl.auth.auth_header()
endpoint = f"{wl.api_endpoint}/v1/api/pipelines/deploy"

# verify this matches the pipeline with model created earlier
pipeline_with_models_id = "api-pipeline-with-models"

data = {
    "deploy_id": pipeline_with_models_id,
    "pipeline_version_pk_id": model_pipeline_variant_id,
    "models": [
        {
            "name": example_model_name,
            "version":example_model_version,
            "sha":example_model_sha
        }
    ],
    "pipeline_id": model_pipeline_id
}

response = requests.post(endpoint, json=data, headers=headers, verify=True).json()
display(response)
model_deployment_id=response['id']

```

    {'id': 14}

Deploy pipeline via curl.

```python
!curl {wl.api_endpoint}/v1/api/pipelines/deploy \
    -H "Authorization: {wl.auth.auth_header()['Authorization']}" \
    -H "Content-Type: application/json" \
    --data '{{ \
        "deploy_id": "{pipeline_with_models_id}", \
        "pipeline_version_pk_id": {model_pipeline_variant_id}, \
        "models": [ \
            {{ \
                "name": "{example_model_name}", \
                "version": "{example_model_version}", \
                "sha": "{example_model_sha}" \
            }} \
        ], \
        "pipeline_id": {model_pipeline_id} \
    }}'
```

    {"id":14}

### Get Deployment Status

* **Endpoint**: `/v1/api/status/get_deployment`

#### Get Deployment Status Parameters

| Field | Type | Description |
|---|---|---|
| **id** | *String* (*Required*) | The deployment in the format `{deployment_name}-{deployment-id}`. |

#### Get Deployment Status Returns

| Field | Type | Description |
|---|---|---|
| **status** | *String* | Status of the pipeline deployment. Values are:  `Running`: the Deployment successfully started.  `Starting`:  The Deployment is still loading.  `Error`:  There is an error with the deployment. |
| **details** | *List[details]* | The list of deployment details. |
| **engines** | *List[engines]* | A list of engines deployed in the pipeline.
| **engine_lbs** | *List[engine_lbs]* | A list of engine load balancers. |
| **sidekicks** | *List[sidekicks]* | A list of deployment sidekicks.  These are used for Containerized Deployment Runtimes. |

#### Get Deployment Status Examples

The deployed pipeline with model details from the previous step is displayed.

Get deployment status via Requests.

```python
# Retrieve the token 
headers = wl.auth.auth_header()

# Get model pipeline deployment

endpoint = f"{wl.api_endpoint}/v1/api/status/get_deployment"

data = {
  "name": f"{pipeline_with_models_id}-{model_deployment_id}"
}

response = requests.post(endpoint, json=data, headers=headers, verify=True).json()
response
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.3.145',
       'name': 'engine-797d8958d9-fsszh',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'api-pipeline-with-models',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'api-sample-model',
          'version': 'bdfc8c60-b5bc-4c0e-aa87-157cd52895b6',
          'sha': 'bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.4.157',
       'name': 'engine-lb-584f54c899-qfjgp',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}

Get deployment status via curl.

```python
!curl {wl.api_endpoint}/v1/api/status/get_deployment \
    -H "Authorization: {wl.auth.auth_header()['Authorization']}" \
    -H "Content-Type: application/json" \
    --data '{{ \
        "name": "{pipeline_with_models_id}-{model_deployment_id}" \
}}'
```

    {"status":"Running","details":[],"engines":[{"ip":"10.244.3.145","name":"engine-797d8958d9-fsszh","status":"Running","reason":null,"details":[],"pipeline_statuses":{"pipelines":[{"id":"api-pipeline-with-models","status":"Running"}]},"model_statuses":{"models":[{"name":"api-sample-model","version":"bdfc8c60-b5bc-4c0e-aa87-157cd52895b6","sha":"bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507","status":"Running"}]}}],"engine_lbs":[{"ip":"10.244.4.157","name":"engine-lb-584f54c899-qfjgp","status":"Running","reason":null,"details":[]}],"sidekicks":[]}

### Get External Inference URL

* **Endpoint**: `/v1/api/admin/get_pipeline_external_url`

Retrieves the external inference URL for a specific pipeline in a workspace.

#### Get External Inference URL Parameters

| Field | Type | Description |
|---|---|---|
| **workspace_id** | *Integer* (*REQUIRED) | The workspace integer id. |
| **pipeline_name** | *String* (*REQUIRED*) | The name of the deployment. |

#### Get External Inference URL Returns

| Field | Type | Description |
|---|---|---|
| **url** | *String* | The pipeline's external inference URL. |

#### Get External Inference URL Examples

In this example, the pipeline's external inference URL from the previous example is retrieved.

Get external inference url via Requests.

```python
## Retrieve the pipeline's External Inference URL

# Retrieve the token 
headers = wl.auth.auth_header()

endpoint = f"{wl.api_endpoint}/v1/api/admin/get_pipeline_external_url"

data = {
    "workspace_id": example_workspace_id,
    "pipeline_name": pipeline_with_models_id
}

response = requests.post(endpoint, json=data, headers=headers, verify=True).json()
print(response)
deployurl = response['url']
```

    {'url': 'https://doc-test.api.wallarooexample.ai/v1/api/pipelines/infer/api-pipeline-with-models-14/api-pipeline-with-models'}

Get external inference url via Requests.

```python
!curl {wl.api_endpoint}/v1/api/admin/get_pipeline_external_url \
    -H "Authorization: {wl.auth.auth_header()['Authorization']}" \
    -H "Content-Type: application/json" \
    --data '{{ \
        "workspace_id": {example_workspace_id}, \
        "pipeline_name": "{pipeline_with_models_id}" \
}}'
```

    {"url":"https://doc-test.api.wallarooexample.ai/v1/api/pipelines/infer/api-pipeline-with-models-14/api-pipeline-with-models"}

### Perform Inference Through External URL

The inference can now be performed through the External Inference URL.  This URL will accept the same inference data file that is used with the Wallaroo SDK, or with an Internal Inference URL as used in the Internal Pipeline Inference URL Tutorial.

Deployed pipelines have their own Inference URL that accepts HTTP POST submissions.

For connections that are external to the Kubernetes cluster hosting the Wallaroo instance, [model endpoints must be enabled](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-model-endpoints-guide/).

#### HTTP Headers

The following headers are required for connecting the the Pipeline Deployment URL:

* **Authorization**: This requires the JWT token in the format `'Bearer ' + token`.  For example:

    ```bash
    Authorization: Bearer abcdefg==
    ```

* **Content-Type**:
* For DataFrame formatted JSON:

    ```bash
    Content-Type:application/json; format=pandas-records
    ```

* For Arrow binary files, the `Content-Type` is `application/vnd.apache.arrow.file`.

    ```bash
    Content-Type:application/vnd.apache.arrow.file
    ```

* **IMPORTANT NOTE**:  Verify that the pipeline deployed has status `Running` before attempting an inference.

Perform inference via external url via Requests.

```python
# Retrieve the token
headers = wl.auth.auth_header()

# set Content-Type type
headers['Content-Type']='application/json; format=pandas-records'

## Inference through external URL using dataframe

# retrieve the json data to submit
data = [
    {
        "dense_input":[
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

display(response)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in</th>
      <th>out</th>
      <th>check_failures</th>
      <th>metadata</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1701376879347</td>
      <td>{'dense_input': [1.0678324729, 0.2177810266, -...</td>
      <td>{'dense_1': [0.0014974177]}</td>
      <td>[]</td>
      <td>{'last_model': '{"model_name":"api-sample-mode...</td>
    </tr>
  </tbody>
</table>

Perform inference via external url via curl.

```python
!curl {deployurl} \
    -H "Content-Type: application/json; format=pandas-records" \
    -H "Authorization: {wl.auth.auth_header()['Authorization']}" \
    --data '[ \
    {{ \
        "dense_input":[ \
            1.0678324729, \
            0.2177810266, \
            -1.7115145262, \
            0.682285721, \
            1.0138553067, \
            -0.4335000013, \
            0.7395859437, \
            -0.2882839595, \
            -0.447262688, \
            0.5146124988, \
            0.3791316964, \
            0.5190619748, \
            -0.4904593222, \
            1.1656456469, \
            -0.9776307444, \
            -0.6322198963, \
            -0.6891477694, \
            0.1783317857, \
            0.1397992467, \
            -0.3554220649, \
            0.4394217877, \
            1.4588397512, \
            -0.3886829615, \
            0.4353492889, \
            1.7420053483, \
            -0.4434654615, \
            -0.1515747891, \
            -0.2668451725, \
            -1.4549617756 \
        ] \
    }} \
]'
```

    [{"time":1701377058894,"in":{"dense_input":[1.0678324729,0.2177810266,-1.7115145262,0.682285721,1.0138553067,-0.4335000013,0.7395859437,-0.2882839595,-0.447262688,0.5146124988,0.3791316964,0.5190619748,-0.4904593222,1.1656456469,-0.9776307444,-0.6322198963,-0.6891477694,0.1783317857,0.1397992467,-0.3554220649,0.4394217877,1.4588397512,-0.3886829615,0.4353492889,1.7420053483,-0.4434654615,-0.1515747891,-0.2668451725,-1.4549617756]},"out":{"dense_1":[0.0014974177]},"check_failures":[],"metadata":{"last_model":"{\"model_name\":\"api-sample-model\",\"model_sha\":\"bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507\"}","pipeline_version":"","elapsed":[53101,318104],"dropped":[],"partition":"engine-797d8958d9-fsszh"}}]

### Undeploy a Pipeline

* **Endpoint**: `/v1/api/pipelines/undeploy`

Undeploys a deployed pipeline.

#### Undeploy a Pipeline Parameters

| Field | Type | Description |
|---|---|---|
| **pipeline_id** | *Integer* (*REQUIRED) | The numerical id of the pipeline. |
| **deployment_id** | *Integer* (*REQUIRED) | The numerical id of the deployment. |

#### Undeploy a Pipeline Returns

Nothing if the call is successful.

#### Undeploy a Pipeline Examples

The pipeline with models deployed is undeployed.

Undeploy the pipeline via Requests.

```python
# Undeploy pipeline with models
# Retrieve the token 
headers = wl.auth.auth_header()
endpoint = f"{wl.api_endpoint}/v1/api/pipelines/undeploy"

data = {
    "pipeline_id": model_pipeline_id,
    "deployment_id":model_deployment_id
}

response = requests.post(endpoint, json=data, headers=headers, verify=True).json()
display(response)
```

Undeploy the pipeline via curl.

```python
!curl {wl.api_endpoint}/v1/api/pipelines/undeploy \
    -H "Authorization: {wl.auth.auth_header()['Authorization']}" \
    -H "Content-Type: application/json" \
    --data '{{ \
        "pipeline_id": {model_pipeline_id}, \
        "deployment_id": {model_deployment_id} \
}}'
```

    null

### Copy a Pipeline

Copies an existing pipeline into a new one in the same workspace.  A new engine configuration can be set for the copied pipeline.

#### Copy a Pipeline Parameters

#### Copy a Pipeline Returns

* **Parameters**
  * **name** - (REQUIRED string): The name of the new pipeline.
  * **workspace_id** - (REQUIRED int): The numerical id of the workspace to copy the source pipeline from.
  * **source_pipeline** - (REQUIRED int): The numerical id of the pipeline to copy from.
  * **deploy** - (OPTIONAL string): Name of the deployment.
  * **engine_config** - (OPTIONAL string): Engine configuration options.
  * **pipeline_version** - (OPTIONAL string): Optional version of the copied pipeline to create.

#### Copy a Pipeline Examples

The pipeline with models created in the step Create Pipeline in a Workspace will be copied into a new one.

```python
Copy a pipeline via Requests.
```

```python
## Copy a pipeline

# Retrieve the token 
headers = wl.auth.auth_header()

endpoint = f"{wl.api_endpoint}/v1/api/pipelines/copy"

data = {
  "name": "api-copied-pipeline-requests",
  "workspace_id": example_workspace_id,
  "source_pipeline": model_pipeline_id
}

response = requests.post(endpoint, json=data, headers=headers, verify=True).json()
response
```

    {'pipeline_pk_id': 9,
     'pipeline_variant_pk_id': 9,
     'pipeline_version': None,
     'deployment': None}

Copy a pipeline via curl.

```python
!curl {wl.api_endpoint}/v1/api/pipelines/copy \
    -H "Authorization: {wl.auth.auth_header()['Authorization']}" \
    -H "Content-Type: application/json" \
    --data '{{ \
        "name": "api-copied-pipeline-curl", \
        "workspace_id": {example_workspace_id}, \
        "source_pipeline": {model_pipeline_id} \
}}'
```

    {"pipeline_pk_id":32,"pipeline_variant_pk_id":32,"pipeline_version":null,"deployment":null}
