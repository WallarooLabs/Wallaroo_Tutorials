## Models

Models can be uploaded and managed through the Wallaroo API.

* [Upload Model to Workspace](#upload-model-to-workspace)
* [Stream Upload Model to Workspace](#stream-upload-model-to-workspace)
* [List Models in Workspace](#list-models-in-workspace)
* [Get Model Details by ID](#get-model-details-by-id)
* [Get Model Versions](#get-model-versions)
* [Get Model Configuration by Id](#get-model-configuration-by-id)
* [Get Model Details](#get-model-details)

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

### Get Model Details by ID

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
