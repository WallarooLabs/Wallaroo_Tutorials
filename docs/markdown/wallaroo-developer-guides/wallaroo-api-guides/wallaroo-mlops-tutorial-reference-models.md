## Models

### Upload Model to Workspace

Uploads a ML Model to a Wallaroo workspace via POST with `Content-Type: multipart/form-data`.

* **Parameters**
  * **name** - (*REQUIRED string*): Name of the model
  * **visibility** - (*OPTIONAL string*): The visibility of the model as either `public` or `private`.
  * **workspace_id** - (*REQUIRED int*): The numerical id of the workspace to upload the model to.
  
Example:  This example will upload the sample file `ccfraud.onnx` to the workspace created in the [Create Workspace](#create-workspace) step as `apitestmodel`.  The model name will be saved as `exampleModelName` for use in other examples.  The id of the uploaded model will be saved as `exampleModelId` for use in later examples.

```python
# upload model - uses multiform data through a Python `request`

apiRequest = "/models/upload"

exampleModelName = f"apitestmodel-{uuid.uuid4()}"

data = {
    "name":exampleModelName,
    "visibility":"public",
    "workspace_id": exampleWorkspaceId
}

files = {
    "file": ('ccfraud.onnx', open('./models/ccfraud.onnx','rb'))
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data, files)
response
```

    {'insert_models': {'returning': [{'models': [{'id': 68}]}]}}

```python
exampleModelId=response['insert_models']['returning'][0]['models'][0]['id']
exampleModelId
```

    10

### Stream Upload Model to Workspace

Streams a potentially large ML Model to a Wallaroo workspace via POST with `Content-Type: multipart/form-data`.

* **Parameters**
  * **name** - (*REQUIRED string*): Name of the model
  * **filename** - (*REQUIRED string*): Name of the file being uploaded.
  * **visibility** - (*OPTIONAL string*): The visibility of the model as either `public` or `private`.
  * **workspace_id** - (*REQUIRED int*): The numerical id of the workspace to upload the model to.
  
Example:  This example will upload the sample file `ccfraud.onnx` to the workspace created in the [Create Workspace](#create-workspace) step as `apitestmodel`.

```python
# stream upload model - next test is adding arbitrary chunks to the stream

apiRequest = "/models/upload_stream"
exampleModelName = f"apitestmodel-{uuid.uuid4()}"
filename = 'streamfile.onnx'

data = {
    "name":exampleModelName,
    "filename": 'streamfile.onnx',
    "visibility":"public",
    "workspace_id": exampleWorkspaceId
}

contentType='application/octet-stream'

file = open('./models/ccfraud.onnx','rb')

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data=None, files=file, contentType='application/octet-stream', params=data)
response
```

    {'insert_models': {'returning': [{'models': [{'id': 11}]}]}}

### List Models in Workspace

Returns a list of models added to a specific workspace.

* **Parameters**
  * **workspace_id** - (*REQUIRED int*): The workspace id to list.
  
Example:  Display the models for the workspace used in the Upload Model to Workspace step.  The model id and model name will be saved as `exampleModelId` and `exampleModelName` variables for other examples.

```python
# List models in a workspace

apiRequest = "/models/list"

data = {
  "workspace_id": exampleWorkspaceId
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```

    {'models': [{'id': 68,
       'name': 'apitestmodel-dfa7e9fd-df72-4b28-93f6-3d147c9f962f',
       'owner_id': '""',
       'created_at': '2022-12-20T19:34:47.014072+00:00',
       'updated_at': '2022-12-20T19:34:47.014072+00:00'}]}

```python
exampleModelId = response['models'][0]['id']
exampleModelName = response['models'][0]['name']
```

### Get Model Details

Returns the model details by the specific model id.

* **Parameters**
  * **workspace_id** - (*REQUIRED int*): The workspace id to list.
* **Returns**
  * **id** - (*int*):  Numerical id of the model.
  * **owner_id** - (*string*): Id of the owner of the model.
  * **workspace_id** - (*int*): Numerical of the id the model is in.
  * **name** - (*string*): Name of the model.
  * **updated_at** - (*DateTime*): Date and time of the model's last update.
  * **created_at** - (*DateTime*): Date and time of the model's creation.
  * **model_config** - (*string*): Details of the model's configuration.
  
Example:  Retrieve the details for the model uploaded in the Upload Model to Workspace step.

```python
# Get model details by id

apiRequest = "/models/get_by_id"

data = {
  "id": exampleModelId
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```

    {'id': 11,
     'owner_id': '""',
     'workspace_id': 5,
     'name': 'apitestmodel-08b16b74-837c-4b4b-a6b7-49475ddece98',
     'updated_at': '2022-11-28T21:30:05.071826+00:00',
     'created_at': '2022-11-28T21:30:05.071826+00:00',
     'model_config': None}

### Get Model Versions

Retrieves all versions of a model based on either the name of the model or the `model_pk_id`.

* **Parameters**
  * **model_id** - (*REQUIRED String*): The model name.
  * **models_pk_id** - (*REQUIRED int*): The model integer pk id.
* **Returns**
  * Array(Model Details)
    * **sha** - (*String*): The `sha` hash of the model version.
    * **models_pk_id**- (*int*): The pk id of the model.
    * **model_version** - (*String*): The UUID identifier of the model version.
    * **owner_id** - (*String*): The Keycloak user id of the model's owner.
    * **model_id** - (*String*): The name of the model.
    * **id** - (*int*): The integer id of the model.
    * **file_name** - (*String*): The filename used when uploading the model.
    * **image_path** - (*String*): The image path of the model.

Example:  Retrieve the versions for a previously uploaded model. The variables `exampleModelVersion` and `exampleModelSha` will store the model's version and SHA values for use in other examples.

```python
# List models in a workspace

apiRequest = "/models/list_versions"

data = {
  "model_id": exampleModelName,
  "models_pk_id": exampleModelId
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```

    [{'sha': 'bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507',
      'models_pk_id': 68,
      'model_version': '87476d85-b8ee-4714-81ba-53041b26f50f',
      'owner_id': '""',
      'model_id': 'apitestmodel-dfa7e9fd-df72-4b28-93f6-3d147c9f962f',
      'id': 68,
      'file_name': 'ccfraud.onnx',
      'image_path': None}]

```python
# Stored for future examples

exampleModelVersion = response[0]['model_version']
exampleModelSha = response[0]['sha']
```

### Get Model Configuration by Id

Returns the model's configuration details.

* **Parameters**
  * **model_id** - (*REQUIRED int*): The numerical value of the model's id.
  
Example:  Submit the model id for the model uploaded in the Upload Model to Workspace step to retrieve configuration details.

```python
# Get model config by id

apiRequest = "/models/get_config_by_id"

data = {
  "model_id": exampleModelId
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```

    {'model_config': None}

### Get Model Details

Returns details regarding a single model, including versions.

Returns the model's configuration details.

* **Parameters**
  * **model_id** - (*REQUIRED int*): The numerical value of the model's id.
  
Example:  Submit the model id for the model uploaded in the Upload Model to Workspace step to retrieve configuration details.

```python
# Get model config by id

apiRequest = "/models/get"

data = {
  "id": exampleModelId
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```

    {'id': 11,
     'name': 'apitestmodel-08b16b74-837c-4b4b-a6b7-49475ddece98',
     'owner_id': '""',
     'created_at': '2022-11-28T21:30:05.071826+00:00',
     'updated_at': '2022-11-28T21:30:05.071826+00:00',
     'models': [{'sha': 'bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507',
       'models_pk_id': 11,
       'model_version': '2589d7e4-bc61-4cb7-8278-5399f28ad001',
       'owner_id': '""',
       'model_id': 'apitestmodel-08b16b74-837c-4b4b-a6b7-49475ddece98',
       'id': 11,
       'file_name': 'streamfile.onnx',
       'image_path': None}]}

### Create Pipeline in a Workspace

Creates a new pipeline in the specified workspace.

* **Parameters**
  * **pipeline_id** - (REQUIRED string): Name of the new pipeline.
  * **workspace_id** - (REQUIRED int): Numerical id of the workspace for the new pipeline.
  * **definition** - (REQUIRED string): Pipeline definitions, can be `{}` for none.

Example:  Two pipelines are created in the workspace created in the step Create Workspace.  One will be an empty pipeline without any models, the other will be created using the uploaded models in the Upload Model to Workspace step and no configuration details.  The pipeline id, variant id, and variant version of each pipeline will be stored for later examples.

```python
# Create pipeline in a workspace

apiRequest = "/pipelines/create"

exampleEmptyPipelineName=f"emptypipeline-{uuid.uuid4()}"

data = {
  "pipeline_id": exampleEmptyPipelineName,
  "workspace_id": exampleWorkspaceId,
  "definition": {}
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
exampleEmptyPipelineId = response['pipeline_pk_id']
exampleEmptyPipelineVariantId=response['pipeline_variant_pk_id']
emptyExamplePipelineVariantVersion=['pipeline_variant_version']
response
```

    {'pipeline_pk_id': 3,
     'pipeline_variant_pk_id': 3,
     'pipeline_variant_version': '84730f78-7b89-4420-bdcb-3c5abac0dd10'}

```python
# Create pipeline in a workspace with models

apiRequest = "/pipelines/create"

exampleModelPipelineName=f"pipelinewithmodel-{uuid.uuid4()}"
exampleModelDeployName = f"deploywithmodel-{uuid.uuid4()}"

data = {
  "pipeline_id": exampleModelPipelineName,
  "workspace_id": exampleWorkspaceId,
  "definition": {
      "id":exampleModelDeployName,
      "steps":
      [
          {
          "ModelInference":
          {
              "models": [
                    {
                        "name":exampleModelName,
                        "version":exampleModelVersion,
                        "sha":exampleModelSha
                    }
                ]
          }
          }
      ]
  }
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
exampleModelPipelineId = response['pipeline_pk_id']
exampleModelPipelineVariantId=response['pipeline_variant_pk_id']
emptyModelPipelineVariantVersion=['pipeline_variant_version']
response
```

    {'pipeline_pk_id': 108,
     'pipeline_variant_pk_id': 108,
     'pipeline_variant_version': '246685fd-258a-49a6-b431-ae4847890eee'}

### Deploy a Pipeline

Deploy a an existing pipeline.  Note that for any pipeline that has model steps, they must be included either in `model_configs`, `model_ids` or `models`.

* **Parameters**
  * **deploy_id** (*REQUIRED string*): The name for the pipeline deployment.
  * **engine_config** (*OPTIONAL string*): Additional configuration options for the pipeline.
  * **pipeline_version_pk_id** (*REQUIRED int*): Pipeline version id.
  * **model_configs** (*OPTIONALArray int*): Ids of model configs to apply.
  * **model_ids** (*OPTIONALArray int*): Ids of models to apply to the pipeline.  If passed in, model_configs will be created automatically.
  * **models** (*OPTIONAL Array models*):  If the model ids are not available as a pipeline step, the models' data can be passed to it through this method.  The options below are only required if `models` are provided as a parameter.
    * **name** (*REQUIRED string*): Name of the uploaded model that is in the same workspace as the pipeline.
    * **version** (*REQUIRED string*): Version of the model to use.
    * **sha** (*REQUIRED string*): SHA value of the model.
  * **pipeline_id** (*REQUIRED int*): Numerical value of the pipeline to deploy.
* **Returns**
  * **id** (*int*): The deployment id.

Examples:  Both the empty pipeline and pipeline with model created in the step [Create Pipeline in a Workspace](#create-pipeline-in-a-workspace) will be deployed and their deployment information saved for later examples.

```python
# Deploy empty pipeline

apiRequest = "/pipelines/deploy"

exampleEmptyDeployId = f"emptydeploy-{uuid.uuid4()}"

data = {
    "deploy_id": exampleEmptyDeployId,
    "pipeline_version_pk_id": exampleEmptyPipelineVariantId,
    "pipeline_id": exampleEmptyPipelineId
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
exampleEmptyDeploymentId=response['id']
response

```

    {'id': 2}

```python
# Deploy a pipeline with models

apiRequest = "/pipelines/deploy"
exampleModelDeployId=f"modeldeploy-{uuid.uuid4()}"

data = {
    "deploy_id": exampleModelDeployId,
    "pipeline_version_pk_id": exampleModelPipelineVariantId,
    "models": [
        {
            "name":exampleModelName,
            "version":exampleModelVersion,
            "sha":exampleModelSha
        }
    ],
    "pipeline_id": exampleModelPipelineId
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
exampleModelDeploymentId=response['id']
response
```

    {'id': 60}

### Get Deployment Status

Returns the deployment status.

* **Parameters**
  * **name** - (REQUIRED string): The deployment in the format {deployment_name}-{deploymnent-id}.
  
Example: The deployed empty and model pipelines status will be displayed.

```python
# Get empty pipeline deployment

apiRequest = "/status/get_deployment"

data = {
  "name": f"{exampleEmptyDeployId}-{exampleEmptyDeploymentId}"
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```

```python
# Get model pipeline deployment

apiRequest = "/status/get_deployment"

data = {
  "name": f"{exampleModelDeployId}-{exampleModelDeploymentId}"
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.4.1.151',
       'name': 'engine-577db84597-x7bm4',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'pipelinewithmodel-94676967-b018-4002-89ef-1d69defc6273',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'apitestmodel-dfa7e9fd-df72-4b28-93f6-3d147c9f962f',
          'version': '87476d85-b8ee-4714-81ba-53041b26f50f',
          'sha': 'bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.4.2.59',
       'name': 'engine-lb-7d6f4bfdd-6hxj5',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}

### Get External Inference URL

The API command `/admin/get_pipeline_external_url` retrieves the external inference URL for a specific pipeline in a workspace.

* **Parameters**
  * **workspace_id** (*REQUIRED integer*):  The workspace integer id.
  * **pipeline_name** (*REQUIRED string*): The name of the pipeline.

In this example, a list of the workspaces will be retrieved.  Based on the setup from the Internal Pipeline Deployment URL Tutorial, the workspace matching `urlworkspace` will have it's **workspace id** stored and used for the `/admin/get_pipeline_external_url` request with the pipeline `urlpipeline`.

The External Inference URL will be stored as a variable for the next step.

```python
## Retrieve the pipeline's External Inference URL

apiRequest = "/admin/get_pipeline_external_url"

data = {
    "workspace_id": exampleWorkspaceId,
    "pipeline_name": exampleModelPipelineName
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
print(response)
externalUrl = response['url']
externalUrl
```

```python
exampleModelPipelineName
```

### Perform Inference Through External URL

The inference can now be performed through the External Inference URL.  This URL will accept the same inference data file that is used with the Wallaroo SDK, or with an Internal Inference URL as used in the Internal Pipeline Inference URL Tutorial.

For this example, the `externalUrl` retrieved through the [Get External Inference URL](#get-external-inference-url) is used to submit a single inference request through the data file `data-1.json`.

```python
TOKEN2 = "eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJEaXlQVjFQMUxHdV81WkpseXR3X2RZSnZwUjdodE9EN3V5cGxocjhacHNrIn0.eyJleHAiOjE2NzE1NjY5NDEsImlhdCI6MTY3MTU2MzM0MSwianRpIjoiN2Y2MTgyZmQtYTIyMS00Y2ZlLThjZDUtN2Y1ZDM5Y2MwZTAwIiwiaXNzIjoiaHR0cHM6Ly9hcGl0ZXN0LmtleWNsb2FrLndhbGxhcm9vY29tbXVuaXR5Lm5pbmphL2F1dGgvcmVhbG1zL21hc3RlciIsImF1ZCI6WyJtYXN0ZXItcmVhbG0iLCJhY2NvdW50Il0sInN1YiI6IjEyMmI2MGE1LWRkYzctNDNjOS05ZmM2LTIwZTljMmI3ZDg1NCIsInR5cCI6IkJlYXJlciIsImF6cCI6ImFwaS1jbGllbnQiLCJzZXNzaW9uX3N0YXRlIjoiNGY4ZDk0NDYtZmE1Ny00MGM3LTk1ZDAtM2E1OGY0YzVlMTQxIiwiYWNyIjoiMSIsImFsbG93ZWQtb3JpZ2lucyI6WyIqIl0sInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJkZWZhdWx0LXJvbGVzLW1hc3RlciIsIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iXX0sInJlc291cmNlX2FjY2VzcyI6eyJtYXN0ZXItcmVhbG0iOnsicm9sZXMiOlsibWFuYWdlLXVzZXJzIiwidmlldy11c2VycyIsInF1ZXJ5LWdyb3VwcyIsInF1ZXJ5LXVzZXJzIl19LCJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6InByb2ZpbGUgZW1haWwiLCJzaWQiOiI0ZjhkOTQ0Ni1mYTU3LTQwYzctOTVkMC0zYTU4ZjRjNWUxNDEiLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiaHR0cHM6Ly9oYXN1cmEuaW8vand0L2NsYWltcyI6eyJ4LWhhc3VyYS11c2VyLWlkIjoiMTIyYjYwYTUtZGRjNy00M2M5LTlmYzYtMjBlOWMyYjdkODU0IiwieC1oYXN1cmEtZGVmYXVsdC1yb2xlIjoidXNlciIsIngtaGFzdXJhLWFsbG93ZWQtcm9sZXMiOlsidXNlciJdLCJ4LWhhc3VyYS11c2VyLWdyb3VwcyI6Int9In0sInByZWZlcnJlZF91c2VybmFtZSI6InNjb3R0IiwiZW1haWwiOiJzY290dEB3YWxsYXJvby5haSJ9.jNw6LJrimK9HxCENWPkKQPVQXMmk2nGP_VGxSUv9vMdxuzYRONLRaTqrXoxz380IlPQT_QGoiMtA0vjMk40-jg0dAmTcWlDA0sGQyKQq2qIctutGCTbiVhOdmc_69kYyePXlYk3fvl16_L_ZcuhGjlS1Gwl1FtrI6jNAjCrBzTQf1MHHZPa77gxZrqK8D95EGYs34WDu359i5S-stPmH0D9OSTFad0UMZvmuTql4YBOIwWpDzgtU8nC_6XLkWYnzlasABDsfsLMggyWntULHeNxDd9wWK0QSdvGedRZdMhy8gRd0DrV3SFTwF9YvlR2q4t6zNbxWXDR-iF8I7Vog4w"

externalURL2 = "https://apitest.api.YOUR SUFFIX/v1/api/pipelines/infer/test1-6"

import json
## Inference through external URL

# retrieve the json data to submit
data = json.load(open('./data/cc_data_1k.json','rb'))

# set the headers
headers= {
        'Authorization': 'Bearer ' + TOKEN2
    }

# submit the request via POST
response = requests.post(externalURL2, json=data, headers=headers)

# Only the first 300 characters will be displayed for brevity
printResponse = json.dumps(response.json())
print(printResponse[0:300])

```

    [{"model_name": "apitestmodel-418716fb-f3da-4e11-bde5-be323c9382fe", "model_version": "1e794d4e-21d1-4140-9a52-7a5252094e7b", "pipeline_name": "test1", "outputs": [{"Float": {"v": 1, "dim": [1001, 1], "data": [0.993003249168396, 0.993003249168396, 0.993003249168396, 0.993003249168396, 0.001091688871

### Undeploy a Pipeline

Undeploys a deployed pipeline.

* **Parameters**
  * **pipeline_id** - (*REQUIRED int*): The numerical id of the pipeline.
  * **deployment_id** - (*REQUIRED int*): The numerical id of the deployment.
* **Returns**
  * Nothing if the call is successful.

Example:  Both the empty pipeline and pipeline with models deployed in the step Deploy a Pipeline will be undeployed.

```python
# Undeploy an empty pipeline

apiRequest = "/pipelines/undeploy"

data = {
    "pipeline_id": exampleEmptyPipelineId,
    "deployment_id":exampleEmptyDeploymentId
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```

```python
# Undeploy pipeline with models

apiRequest = "/pipelines/undeploy"

data = {
    "pipeline_id": exampleModelPipelineId,
    "deployment_id":exampleModelDeploymentId
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```

### Copy a Pipeline

Copies an existing pipeline into a new one in the same workspace.  A new engine configuration can be set for the copied pipeline.

* **Parameters**
  * **name** - (REQUIRED string): The name of the new pipeline.
  * **workspace_id** - (REQUIRED int): The numerical id of the workspace to copy the source pipeline from.
  * **source_pipeline** - (REQUIRED int): The numerical id of the pipeline to copy from.
  * **deploy** - (OPTIONAL string): Name of the deployment.
  * **engine_config** - (OPTIONAL string): Engine configuration options.
  * **pipeline_version** - (OPTIONAL string): Optional version of the copied pipeline to create.

Example:  The pipeline with models created in the step Create Pipeline in a Workspace will be copied into a new one.

```python
# Copy a pipeline

apiRequest = "/pipelines/copy"

exampleCopiedPipelineName=f"copiedmodelpipeline-{uuid.uuid4()}"

data = {
  "name": exampleCopiedPipelineName,
  "workspace_id": exampleWorkspaceId,
  "source_pipeline": exampleModelPipelineId
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```

    {'pipeline_pk_id': 5,
     'pipeline_variant_pk_id': 5,
     'pipeline_version': None,
     'deployment': None}

