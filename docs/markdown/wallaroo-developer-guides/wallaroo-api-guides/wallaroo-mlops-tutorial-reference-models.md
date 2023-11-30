This tutorial and the assets are available as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/development/mlops_api).

## Wallaroo MLOps API User Management Tutorial

This tutorial focuses on using the Wallaroo MLOps API for model management.  For this tutorial, we will be using the Wallaroo SDK to provide authentication credentials for ease of use examples.  See the [Wallaroo API Guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/) for full details on using the Wallaroo MLOps API.

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

`pyarrow` is the Apache Arrow library used for data schemas in Wallaroo, while `base64` is used to convert data schemas to base64 format for model uploads.

```python
import wallaroo

import requests

import json

import pyarrow as pa

import base64
```

### Connect to the Wallaroo Instance

The next step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
# Login through local Wallaroo instance

wl = wallaroo.Client()
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

    {'Authorization': 'Bearer eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJKNXZOUVVIajVNa3lOV2pLUkwtUGZZSXJ2S3Z5YUx3eThJZFB2dktrZnRnIn0.eyJleHAiOjE3MDEyOTQxMTcsImlhdCI6MTcwMTI5NDA1NywiYXV0aF90aW1lIjoxNzAxMjkyNzYzLCJqdGkiOiIxZjdiM2Q4My0yOGJlLTRjMjQtYjMzNy01ZWRlNGRiYjg0ZjEiLCJpc3MiOiJodHRwczovL2RvYy10ZXN0LmtleWNsb2FrLndhbGxhcm9vY29tbXVuaXR5Lm5pbmphL2F1dGgvcmVhbG1zL21hc3RlciIsImF1ZCI6WyJtYXN0ZXItcmVhbG0iLCJhY2NvdW50Il0sInN1YiI6IjEyZWEwOWQxLTBmNDktNDA1ZS1iZWQxLTI3ZWI2ZDEwZmRlNCIsInR5cCI6IkJlYXJlciIsImF6cCI6InNkay1jbGllbnQiLCJzZXNzaW9uX3N0YXRlIjoiMDE0ZGRlZDgtMzUxYi00MDMyLWI1NWMtMDQ2MWY3NzIzZDNkIiwiYWNyIjoiMSIsInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJkZWZhdWx0LXJvbGVzLW1hc3RlciIsIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iXX0sInJlc291cmNlX2FjY2VzcyI6eyJtYXN0ZXItcmVhbG0iOnsicm9sZXMiOlsibWFuYWdlLXVzZXJzIiwidmlldy11c2VycyIsInF1ZXJ5LWdyb3VwcyIsInF1ZXJ5LXVzZXJzIl19LCJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6InByb2ZpbGUgZW1haWwiLCJzaWQiOiIwMTRkZGVkOC0zNTFiLTQwMzItYjU1Yy0wNDYxZjc3MjNkM2QiLCJlbWFpbF92ZXJpZmllZCI6ZmFsc2UsImh0dHBzOi8vaGFzdXJhLmlvL2p3dC9jbGFpbXMiOnsieC1oYXN1cmEtdXNlci1pZCI6IjEyZWEwOWQxLTBmNDktNDA1ZS1iZWQxLTI3ZWI2ZDEwZmRlNCIsIngtaGFzdXJhLWRlZmF1bHQtcm9sZSI6InVzZXIiLCJ4LWhhc3VyYS1hbGxvd2VkLXJvbGVzIjpbInVzZXIiXSwieC1oYXN1cmEtdXNlci1ncm91cHMiOiJ7fSJ9LCJuYW1lIjoiSm9obiBIYW5zYXJpY2siLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJqb2huLmh1bW1lbEB3YWxsYXJvby5haSIsImdpdmVuX25hbWUiOiJKb2huIiwiZmFtaWx5X25hbWUiOiJIYW5zYXJpY2siLCJlbWFpbCI6ImpvaG4uaHVtbWVsQHdhbGxhcm9vLmFpIn0.eLmnWABd0bxLF5pDlp3ZAUuJ_veDqGGsBaXP5fyIo5_HrjDvRKLenxhX1exOIOZuC0V5EPjHx7XiCQx898rgrYOE_ouKx73ccN3mmPIyjpV-NlrXWFMbVOklSDaI_PigJO8Ja7rnWFPYUHw0G3MHx_smhFVIi8mMpW-712qZCMPVSKfi4xKbmgdypp6BDkJBeUw9IaLeLV9SLh4ERiEZ9ZkJHGI9puIiA3W1kZx6fW2ZKeNBV1WI-03QmJg6REyC3g5pl0BQzsuUh3AWW8TXExIAO4pvetHzZJwGM__UbMnNkMypI6DVx7use9TnY_PW4r08hDaf6XLGoz61auWEJw'}

## Models

The Wallaroo MLOps API allows users to upload different types of ML models and frameworks into Wallaroo.

### Upload Model to Workspace

* **Endpoint**: `/v1/api/models/upload_and_convert`
* **Content-Type**: `multipart/form-data`

Models uploaded through this method that are not Wallaroo Native Runtimes (ONNX, Tensorflow, and Python script) are containerized within the Wallaroo instance then run by the Wallaroo engine.  See [Wallaroo MLOps API Essentials Guide: Pipeline Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-api-essential-guide/wallaroo-mlops-api-essential-guide-pipelines/) for details on pipeline configurations and deployments.

#### Upload Model to Workspace Parameters

| Field | &nbsp; | Type | Description |
|---|---|---|---|
| **name** | &nbsp; | *String* (*Required*) | The model name. |
| **visibility** | &nbsp; | *String* (*Required*) | Either `public` or `private`. |
| **workspace_id** | &nbsp; | *String* (*Required*) | The numerical ID of the workspace to upload the model to. |
| **conversion** | &nbsp; | *String* (*Required*) |  The conversion parameters that include the following: |
| &nbsp; | **framework** | *String* (*Required*) |  The framework of the model being uploaded.  See the list of supported models for more details. |
| &nbsp; | **python_version** | *String* (*Required*) | The version of Python required for model. |
| &nbsp; | **requirements** | *String* (*Required*) | Required libraries.  Can be `[]` if the requirements are default Wallaroo JupyterHub libraries. |
| &nbsp; | **input_schema** | *String* (*Optional*) | The input schema from the Apache Arrow `pyarrow.lib.Schema` format, encoded with `base64.b64encode`.  **Only required for Containerized Wallaroo Runtime models.** |
| &nbsp; | **output_schema** | *String* (*Optional*) |  The output schema from the Apache Arrow `pyarrow.lib.Schema` format, encoded with `base64.b64encode`.  **Only required for non-native runtime models.** |

Files are uploaded in the `multipart/form-data` format with two parts:

* `metadata`: Contains the parameters listed above as `application/json`.
* `file`: The binary file (ONNX, .zip, etc) as Content-Type `application/octet-stream`.

#### Upload Model to Workspace Returns

| Field | &nbsp; | Type | Description |
|---|---|---|---|
| **insert_models{'returning': [models]}** | &nbsp; | *List[models]* | The uploaded models details. |
| &nbsp; | **id** | *Integer* | The model's numerical id. |

#### Upload Model to Workspace Examples

The following example shows uploading an ONNX model to a Wallaroo instance.  Note that the `input_schema` and `output_schema` encoded details are not required.

This example assumes the workspace id of `10`.  Modify this code block based on your Wallaroo Ops instance.

Upload model via Requests library.

```python
# Retrieve the token 
headers = wl.auth.auth_header()

endpoint = f"{wl.api_endpoint}/v1/api/models/upload_and_convert"

workspace_id = 10

framework='onnx'

model_name = f"api-sample-model"

metadata = {
    "name": model_name,
    "visibility": "public",
    "workspace_id": workspace_id,
    "conversion": {
        "framework": framework,
        "python_version": "3.8",
        "requirements": []
    }
}

files = {
    "metadata": (None, json.dumps(data), "application/json"),
    'file': (model_name, open('./models/ccfraud.onnx', 'rb'), "application/octet-stream")
    }

response = requests.post(endpoint, files=files, headers=headers).json()

display(f"Uploaded Model Name: {model_name}.")
display(f"Sample model file: ./models/ccfraud.onnx")
display(response)
```

    'Uploaded Model Name: api-sample-model.'

    'Sample model file: ./models/ccfraud.onnx'

    {'insert_models': {'returning': [{'models': [{'id': 14}]}]}}

Upload ONNX model via curl.

```python
metadata = {
    "name": model_name,
    "visibility": "public",
    "workspace_id": workspace_id,
    "conversion": {
        "framework": framework,
        "python_version": "3.8",
        "requirements": []
    }
}

# save metadata to a file
with open("data/onnx_file_upload.json", "w") as outfile:
    json.dump(metadata, outfile)

!curl {wl.api_endpoint}/v1/api/models/upload_and_convert \
    -H "Authorization: {wl.auth.auth_header()['Authorization']}" \
    --form 'metadata=@./data/onnx_file_upload.json;type=application/json' \
    --form 'file=@./models/ccfraud.onnx;type=application/octet-stream'
```

    {"insert_models":{"returning":[{"models":[{"id":18}]}]}}

The following example shows uploading a Pytorch model to a Wallaroo instance.  Note that the `input_schema` and `output_schema` encoded details are required.

Upload Pytorch via Requests.

```python
input_schema = pa.schema([
    pa.field('input_1', pa.list_(pa.float32(), list_size=10)),
    pa.field('input_2', pa.list_(pa.float32(), list_size=5))
])
output_schema = pa.schema([
    pa.field('output_1', pa.list_(pa.float32(), list_size=3)),
    pa.field('output_2', pa.list_(pa.float32(), list_size=2))
])

encoded_input_schema = base64.b64encode(
                bytes(input_schema.serialize())
            ).decode("utf8")

encoded_output_schema = base64.b64encode(
                bytes(output_schema.serialize())
            ).decode("utf8")

framework = 'pytorch'

model_name = 'api-upload-pytorch-multi-io'

metadata = {
    "name": model_name,
    "visibility": "private",
    "workspace_id": workspace_id,
    "conversion": {
        "framework": framework,
        "python_version": "3.8",
        "requirements": []
    },
    "input_schema": encoded_input_schema,
    "output_schema": encoded_output_schema,
}

headers = wl.auth.auth_header()

files = {
    'metadata': (None, json.dumps(metadata), "application/json"),
    'file': (model_name, open('./models/model-auto-conversion_pytorch_multi_io_model.pt','rb'),'application/octet-stream')
}

response = requests.post(endpoint, files=files, headers=headers).json()

display(f"Uploaded Model Name: {model_name}.")
display(f"Sample model file: ./models/model-auto-conversion_pytorch_multi_io_model.pt")
display(response)
```

    'Uploaded Model Name: api-upload-pytorch-multi-io.'

    'Sample model file: ./models/model-auto-conversion_pytorch_multi_io_model.pt'

    {'insert_models': {'returning': [{'models': [{'id': 15}]}]}}

Upload Pytorch via curl.

```python
# save metadata to a file
with open("./data/pytorch_file_upload.json", "w") as outfile:
    json.dump(metadata, outfile)

!curl {wl.api_endpoint}/v1/api/models/upload_and_convert \
    -H "Authorization: {wl.auth.auth_header()['Authorization']}" \
    --form 'metadata=@./data/pytorch_file_upload.json;type=application/json' \
    --form 'file=@./models/model-auto-conversion_pytorch_multi_io_model.pt;type=application/octet-stream'
```

    {"insert_models":{"returning":[{"models":[{"id":19}]}]}}

### List Models in Workspace

* **Endpoint**: `/v1/api/models/list`

Returns a list of models added to a specific workspace.

#### List Models in Workspace Parameters

| Field | Type | Description |
|---|---|---|
| **workspace_id** | *Integer* (*REQUIRED*) | The workspace id to list. |
  
#### List Models in Workspace Returns

| Field | &nbsp; | Type | Description |
|---|---|---|---|
| **models** | &nbsp; | *List[models]* | List of models in the workspace. |
| &nbsp; | **id** | *Integer* | The numerical id of the model. |
| &nbsp; | **owner_id | *String* | Identifer of the model owner. |
| &nbsp; | *created_at* | *String* | DateTime of the model's creation. |
| &nbsp; | *updated_at* | *String* | DateTime of the model's last update. |

#### List Models in Workspace Examples

Display the models for the workspace.  This is assumed to be workspace_id of `10`.  Adjust the script for your own use.

List models in workspace via Requests.

```python
# Retrieve the token 
headers = wl.auth.auth_header()

endpoint = f"{wl.api_endpoint}/v1/api/models/list"

data = {
  "workspace_id": workspace_id
}

response = requests.post(endpoint, 
                         json=data, 
                         headers=headers, 
                         verify=True).json()
display(response)
```

    {'models': [{'id': 15,
       'name': 'api-upload-pytorch-multi-io',
       'owner_id': '""',
       'created_at': '2023-11-29T16:30:53.716526+00:00',
       'updated_at': '2023-11-29T18:20:39.610964+00:00'},
      {'id': 14,
       'name': 'api-sample-model',
       'owner_id': '""',
       'created_at': '2023-11-29T16:26:06.011817+00:00',
       'updated_at': '2023-11-29T16:26:06.011817+00:00'}]}

List models in workspace via curl.

```python
!curl {wl.api_endpoint}/v1/api/models/list \
    -H "Authorization: {wl.auth.auth_header()['Authorization']}" \
    -H "Content-Type: application/json" \
    --data '{{"workspace_id": {workspace_id}}}'
```

    {"models":[{"id":15,"name":"api-upload-pytorch-multi-io","owner_id":"\"\"","created_at":"2023-11-29T16:30:53.716526+00:00","updated_at":"2023-11-29T18:20:39.610964+00:00"},{"id":14,"name":"api-sample-model","owner_id":"\"\"","created_at":"2023-11-29T16:26:06.011817+00:00","updated_at":"2023-11-29T16:26:06.011817+00:00"}]}

### Get Model Details By ID

* **Endpoint**:  `/v1/api/models/get_by_id`

Returns the model details by the specific model id.

#### Get Model Details By ID Parameters

| Field | Type | Description |
|---|---|---|
| **workspace_id** | *Integer* (*REQUIRED*) | The workspace id to list. |

#### Get Model Details By ID Returns

| Field | Type | Description |
|---|---|---|
| **id** | *Integer* | Numerical id of the model. |
| **owner_id** | *String* | Id of the owner of the model. |
| **workspace_id** | *Integer* | Numerical of the id the model is in. |
| **name** | *String* | Name of the model. |
| **updated_at** | *String* | DateTime of the model's last update.|
| **created_at** | *String* | DateTime of the model's creation. |
| **model_config** | *String* | Details of the model's configuration. |

#### Get Model Details By ID Examples
  
Retrieve the details for the model uploaded in the Upload Model to Workspace step.  This will first list the models in the workspace with the id `10`, then use that first model to display information.  This assumes the workspace id and that there is at least one model uploaded to it.

Get Model Details By ID via Requests.

```python
# Retrieve the token 
headers = wl.auth.auth_header()

endpoint = f"{wl.api_endpoint}/v1/api/models/list"

data = {
  "workspace_id": workspace_id
}

# first get the list of models in the workspace
response = requests.post(endpoint, 
                         json=data, 
                         headers=headers, 
                         verify=True).json()
example_model_id = response['models'][0]['id']
example_model_name = response['models'][0]['name']

# Get model details by id
# Retrieve the token 
headers = wl.auth.auth_header()

endpoint = f"{wl.api_endpoint}/v1/api/models/get_by_id"

data = {
  "id": example_model_id
}

response = requests.post(endpoint, json=data, headers=headers, verify=True).json()
display(response)
```

    {'id': 15,
     'owner_id': '""',
     'workspace_id': 10,
     'name': 'api-upload-pytorch-multi-io',
     'updated_at': '2023-11-29T18:20:39.610964+00:00',
     'created_at': '2023-11-29T16:30:53.716526+00:00',
     'model_config': {'id': 25,
      'runtime': 'flight',
      'tensor_fields': None,
      'filter_threshold': None}}

Get Model Details By ID via curl.

```python
!curl {wl.api_endpoint}/v1/api/models/get_by_id \
    -H "Authorization: {wl.auth.auth_header()['Authorization']}" \
    -H "Content-Type: application/json" \
    --data '{{"id": {example_model_id}}}'
```

    {"id":15,"owner_id":"\"\"","workspace_id":10,"name":"api-upload-pytorch-multi-io","updated_at":"2023-11-29T18:20:39.610964+00:00","created_at":"2023-11-29T16:30:53.716526+00:00","model_config":{"id":25,"runtime":"flight","tensor_fields":null,"filter_threshold":null}}

### Get Model Versions

* **Endpoint**: `/v1/api/models/list_versions`

Retrieves all versions of a model based on either the name of the model or the `model_pk_id`.

#### Get Model Versions Parameters

| Field | Type | Description |
|---|---|---|
| **model_id** | *String* (*REQUIRED*) | The model name. |
| **models_pk_id** | *Integer* (*REQUIRED*) | The model's numerical id. |

#### Get Model Versions Returns

| Field |&nbsp;| Type | Description |
|---|---|---|---|
| Unnamed | &nbsp; | *List[models]* | A list of model versions for the requested model. |
| &nbsp; | **sha** | *String* | The `sha` hash of the model version. |
| &nbsp; | **models_pk_id** | *Integer* | The pk id of the model. |
| &nbsp; | **model_version** | *String* | The UUID identifier of the model version. |
| &nbsp; | **owner_id** | *String* | The Keycloak user id of the model's owner. |
| &nbsp; | **model_id**  | *String* | The name of the model. |
| &nbsp; | **id** | *Integer* | The integer id of the model. |
| &nbsp; | **file_name**  | *String* | The filename used when uploading the model. |
| &nbsp; | **image_path** | *String* | The image path of the model. |

Retrieve the versions for a previously uploaded model.  This assumes a workspace with id `10` has models already loaded into it.

Retrieve model versions via Requests.

```python

# Retrieve the token 
headers = wl.auth.auth_header()

endpoint = f"{wl.api_endpoint}/v1/api/models/list"

data = {
  "workspace_id": workspace_id
}

# first get the list of models in the workspace
response = requests.post(endpoint, 
                         json=data, 
                         headers=headers, 
                         verify=True).json()
example_model_id = response['models'][0]['id']
example_model_name = response['models'][0]['name']

## List model versions

# Retrieve the token 
headers = wl.auth.auth_header()
endpoint = f"{wl.api_endpoint}/v1/api/models/list_versions"

data = {
  "model_id": example_model_name,
  "models_pk_id": example_model_id
}

response = requests.post(endpoint, json=data, headers=headers, verify=True).json()
display(response)
```

    [{'sha': '792db9ee9f41aded3c1d4705f50ccdedd21cafb8b6232c03e4a849b6da1050a8',
      'models_pk_id': 15,
      'model_version': '271875ae-92ee-4137-b54f-c2ce1e88121c',
      'owner_id': '""',
      'model_id': 'api-upload-pytorch-multi-io',
      'id': 19,
      'file_name': 'model-auto-conversion_pytorch_multi_io_model.pt',
      'image_path': None,
      'status': 'error'},
     {'sha': 'bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507',
      'models_pk_id': 15,
      'model_version': 'ce648f2a-faee-4c8d-8a7b-e2789f3ab919',
      'owner_id': '""',
      'model_id': 'api-upload-pytorch-multi-io',
      'id': 17,
      'file_name': 'ccfraud.onnx',
      'image_path': None,
      'status': 'error'},
     {'sha': '792db9ee9f41aded3c1d4705f50ccdedd21cafb8b6232c03e4a849b6da1050a8',
      'models_pk_id': 15,
      'model_version': 'a6762893-be27-4142-ba09-4ce1b87b74a8',
      'owner_id': '""',
      'model_id': 'api-upload-pytorch-multi-io',
      'id': 15,
      'file_name': 'api-upload-pytorch-multi-io',
      'image_path': None,
      'status': 'error'},
     {'sha': 'bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507',
      'models_pk_id': 15,
      'model_version': '0f96fae1-8ebc-41d8-a11d-3eaa3bc26526',
      'owner_id': '""',
      'model_id': 'api-upload-pytorch-multi-io',
      'id': 18,
      'file_name': 'ccfraud.onnx',
      'image_path': None,
      'status': 'error'},
     {'sha': 'bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507',
      'models_pk_id': 15,
      'model_version': '9150b046-5df0-4c41-a60b-3016355f89d5',
      'owner_id': '""',
      'model_id': 'api-upload-pytorch-multi-io',
      'id': 16,
      'file_name': 'ccfraud.onnx',
      'image_path': 'proxy.replicated.com/proxy/wallaroo/ghcr.io/wallaroolabs/mlflow-deploy:v2023.4.0-4103',
      'status': 'ready'}]

Retrieve model versions via curl.

```python
!curl {wl.api_endpoint}/v1/api/models/list_versions \
    -H "Authorization: {wl.auth.auth_header()['Authorization']}" \
    -H "Content-Type: application/json" \
    -d '{json.dumps(data)}'
```

    [{"sha":"792db9ee9f41aded3c1d4705f50ccdedd21cafb8b6232c03e4a849b6da1050a8","models_pk_id":15,"model_version":"271875ae-92ee-4137-b54f-c2ce1e88121c","owner_id":"\"\"","model_id":"api-upload-pytorch-multi-io","id":19,"file_name":"model-auto-conversion_pytorch_multi_io_model.pt","image_path":null,"status":"error"},{"sha":"bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507","models_pk_id":15,"model_version":"ce648f2a-faee-4c8d-8a7b-e2789f3ab919","owner_id":"\"\"","model_id":"api-upload-pytorch-multi-io","id":17,"file_name":"ccfraud.onnx","image_path":null,"status":"error"},{"sha":"792db9ee9f41aded3c1d4705f50ccdedd21cafb8b6232c03e4a849b6da1050a8","models_pk_id":15,"model_version":"a6762893-be27-4142-ba09-4ce1b87b74a8","owner_id":"\"\"","model_id":"api-upload-pytorch-multi-io","id":15,"file_name":"api-upload-pytorch-multi-io","image_path":null,"status":"error"},{"sha":"bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507","models_pk_id":15,"model_version":"0f96fae1-8ebc-41d8-a11d-3eaa3bc26526","owner_id":"\"\"","model_id":"api-upload-pytorch-multi-io","id":18,"file_name":"ccfraud.onnx","image_path":null,"status":"error"},{"sha":"bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507","models_pk_id":15,"model_version":"9150b046-5df0-4c41-a60b-3016355f89d5","owner_id":"\"\"","model_id":"api-upload-pytorch-multi-io","id":16,"file_name":"ccfraud.onnx","image_path":"proxy.replicated.com/proxy/wallaroo/ghcr.io/wallaroolabs/mlflow-deploy:v2023.4.0-4103","status":"ready"}]

### Get Model Configuration by Id

* **Endpoints**: `/v1/api/models/get_config_by_id`

Returns the model's configuration details.

#### Get Model Configuration by Id Parameters

| Field | Type | Description |
|---|---|---|
| **model_id** | *Integer* (*Required*) | The numerical value of the model's id. |

#### Get Model Configuration by Id Returns

| Field | Type | Description |
|---|---|---|
  
#### Get Model Configuration by Id Examples

Submit the model id for the model uploaded in the Upload Model to Workspace step to retrieve configuration details.

Retrieve model configuration via Requests.

```python
## Get model config by id

# Retrieve the token 
headers = wl.auth.auth_header()
endpoint = f"{wl.api_endpoint}/v1/api/models/get_config_by_id"

data = {
  "model_id": example_model_id
}

response = requests.post(endpoint, json=data, headers=headers, verify=True).json()
response
```

    {'model_config': {'id': 25,
      'runtime': 'flight',
      'tensor_fields': None,
      'filter_threshold': None}}

Retrieve model configuration via curl.

```python
!curl {wl.api_endpoint}/v1/api/models/get_config_by_id \
    -H "Authorization: {wl.auth.auth_header()['Authorization']}" \
    -H "Content-Type: application/json" \
    -d '{json.dumps(data)}'
```

    {"model_config":{"id":25,"runtime":"flight","tensor_fields":null,"filter_threshold":null}}

### Get Model Details

* **Endpoint**: `/v1/api/models/get`

#### Get Model Details Parameters

Returns details regarding a single model, including versions.

| Field | Type | Description |
|---|---|---|
| **model_id** | *Integer* (*REQUIRED*) | The numerical value of the model's id. |

#### Get Model Details Returns

| Field | &nbsp; | Type | Description |
|---|---|---|---|
| **id** | &nbsp;| *Integer* | The numerical value of the model's id. |
| **name** | &nbsp;| *String* | The name of the model. |
| **owner_id** | &nbsp;| *String* | The model owner. |
| **created_at** | &nbsp;| *String* | DateTime of the model's creation. |
| **updated_at** | &nbsp;| *String* | DateTime of the model's last update. |
| models | &nbsp; | *List[models]* | The list of model versions associated with this model. |
| &nbsp; | **sha** | *String* | The sha hash of the model version. |
| &nbsp; |**models_pk_id**| *Integer* | The model id. |
| &nbsp; |**model_version**| *String* | The UUID identifier of the model version.|
| &nbsp; |**owner_id** | *String* | The model owner. |
| &nbsp; |**model_id** | *String* | The name of the model. |
| &nbsp; |**id**| *Integer* | The numerical identifier of the model version. |
| &nbsp; |**file_name** | *String* | The file name used when uploading the model version |
| &nbsp; |**image_path** | *String* or *None* | The image path of the model verison. |

#### Get Model Details Examples

Submit the model id for the model uploaded in the Upload Model to Workspace step to retrieve configuration details.

Get model details via Requests.

```python
# Get model config by id
# Retrieve the token 
headers = wl.auth.auth_header()
endpoint = f"{wl.api_endpoint}/v1/api/models/get"

data = {
  "id": example_model_id
}

response = requests.post(endpoint, json=data, headers=headers, verify=True).json()
response
```

    {'id': 15,
     'name': 'api-upload-pytorch-multi-io',
     'owner_id': '""',
     'created_at': '2023-11-29T16:30:53.716526+00:00',
     'updated_at': '2023-11-29T18:20:39.610964+00:00',
     'models': [{'sha': '792db9ee9f41aded3c1d4705f50ccdedd21cafb8b6232c03e4a849b6da1050a8',
       'models_pk_id': 15,
       'model_version': '271875ae-92ee-4137-b54f-c2ce1e88121c',
       'owner_id': '""',
       'model_id': 'api-upload-pytorch-multi-io',
       'id': 19,
       'file_name': 'model-auto-conversion_pytorch_multi_io_model.pt',
       'image_path': None},
      {'sha': 'bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507',
       'models_pk_id': 15,
       'model_version': 'ce648f2a-faee-4c8d-8a7b-e2789f3ab919',
       'owner_id': '""',
       'model_id': 'api-upload-pytorch-multi-io',
       'id': 17,
       'file_name': 'ccfraud.onnx',
       'image_path': None},
      {'sha': '792db9ee9f41aded3c1d4705f50ccdedd21cafb8b6232c03e4a849b6da1050a8',
       'models_pk_id': 15,
       'model_version': 'a6762893-be27-4142-ba09-4ce1b87b74a8',
       'owner_id': '""',
       'model_id': 'api-upload-pytorch-multi-io',
       'id': 15,
       'file_name': 'api-upload-pytorch-multi-io',
       'image_path': None},
      {'sha': 'bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507',
       'models_pk_id': 15,
       'model_version': '0f96fae1-8ebc-41d8-a11d-3eaa3bc26526',
       'owner_id': '""',
       'model_id': 'api-upload-pytorch-multi-io',
       'id': 18,
       'file_name': 'ccfraud.onnx',
       'image_path': None},
      {'sha': 'bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507',
       'models_pk_id': 15,
       'model_version': '9150b046-5df0-4c41-a60b-3016355f89d5',
       'owner_id': '""',
       'model_id': 'api-upload-pytorch-multi-io',
       'id': 16,
       'file_name': 'ccfraud.onnx',
       'image_path': 'proxy.replicated.com/proxy/wallaroo/ghcr.io/wallaroolabs/mlflow-deploy:v2023.4.0-4103'}]}

Get model details via curl.

```python
!curl {wl.api_endpoint}/v1/api/models/get \
    -H "Authorization: {wl.auth.auth_header()['Authorization']}" \
    -H "Content-Type: application/json" \
    -d '{json.dumps(data)}'
```

    {"id":15,"name":"api-upload-pytorch-multi-io","owner_id":"\"\"","created_at":"2023-11-29T16:30:53.716526+00:00","updated_at":"2023-11-29T18:20:39.610964+00:00","models":[{"sha":"792db9ee9f41aded3c1d4705f50ccdedd21cafb8b6232c03e4a849b6da1050a8","models_pk_id":15,"model_version":"271875ae-92ee-4137-b54f-c2ce1e88121c","owner_id":"\"\"","model_id":"api-upload-pytorch-multi-io","id":19,"file_name":"model-auto-conversion_pytorch_multi_io_model.pt","image_path":null},{"sha":"bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507","models_pk_id":15,"model_version":"ce648f2a-faee-4c8d-8a7b-e2789f3ab919","owner_id":"\"\"","model_id":"api-upload-pytorch-multi-io","id":17,"file_name":"ccfraud.onnx","image_path":null},{"sha":"792db9ee9f41aded3c1d4705f50ccdedd21cafb8b6232c03e4a849b6da1050a8","models_pk_id":15,"model_version":"a6762893-be27-4142-ba09-4ce1b87b74a8","owner_id":"\"\"","model_id":"api-upload-pytorch-multi-io","id":15,"file_name":"api-upload-pytorch-multi-io","image_path":null},{"sha":"bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507","models_pk_id":15,"model_version":"0f96fae1-8ebc-41d8-a11d-3eaa3bc26526","owner_id":"\"\"","model_id":"api-upload-pytorch-multi-io","id":18,"file_name":"ccfraud.onnx","image_path":null},{"sha":"bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507","models_pk_id":15,"model_version":"9150b046-5df0-4c41-a60b-3016355f89d5","owner_id":"\"\"","model_id":"api-upload-pytorch-multi-io","id":16,"file_name":"ccfraud.onnx","image_path":"proxy.replicated.com/proxy/wallaroo/ghcr.io/wallaroolabs/mlflow-deploy:v2023.4.0-4103"}]}
