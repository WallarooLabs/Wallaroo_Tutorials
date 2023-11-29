This tutorial and the assets are available as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/development/mlops_api).

## Wallaroo MLOps API Workspace Management Tutorial

This tutorial focuses on using the Wallaroo MLOps API for Wallaroo workspace management.  For this tutorial, we will be using the Wallaroo SDK to provide authentication credentials for ease of use examples.  See the [Wallaroo API Guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/) for full details on using the Wallaroo MLOps API.

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

    {'Authorization': 'Bearer eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJKNXZOUVVIajVNa3lOV2pLUkwtUGZZSXJ2S3Z5YUx3eThJZFB2dktrZnRnIn0.eyJleHAiOjE3MDEyMDUyMjQsImlhdCI6MTcwMTIwNTE2NCwiYXV0aF90aW1lIjoxNzAxMjAyMjg1LCJqdGkiOiIwNWQxNTdlNS0xMjJkLTRjZmMtODA4Ny1jNDJmNmZkMTY5MjUiLCJpc3MiOiJodHRwczovL2RvYy10ZXN0LmtleWNsb2FrLndhbGxhcm9vY29tbXVuaXR5Lm5pbmphL2F1dGgvcmVhbG1zL21hc3RlciIsImF1ZCI6WyJtYXN0ZXItcmVhbG0iLCJhY2NvdW50Il0sInN1YiI6IjEyZWEwOWQxLTBmNDktNDA1ZS1iZWQxLTI3ZWI2ZDEwZmRlNCIsInR5cCI6IkJlYXJlciIsImF6cCI6InNkay1jbGllbnQiLCJzZXNzaW9uX3N0YXRlIjoiYjg5ZTY0YzAtNmM4ZS00ZjQ2LWExMDAtMzI5ZTczZDc0MzYyIiwiYWNyIjoiMSIsInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJkZWZhdWx0LXJvbGVzLW1hc3RlciIsIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iXX0sInJlc291cmNlX2FjY2VzcyI6eyJtYXN0ZXItcmVhbG0iOnsicm9sZXMiOlsibWFuYWdlLXVzZXJzIiwidmlldy11c2VycyIsInF1ZXJ5LWdyb3VwcyIsInF1ZXJ5LXVzZXJzIl19LCJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6InByb2ZpbGUgZW1haWwiLCJzaWQiOiJiODllNjRjMC02YzhlLTRmNDYtYTEwMC0zMjllNzNkNzQzNjIiLCJlbWFpbF92ZXJpZmllZCI6ZmFsc2UsImh0dHBzOi8vaGFzdXJhLmlvL2p3dC9jbGFpbXMiOnsieC1oYXN1cmEtdXNlci1pZCI6IjEyZWEwOWQxLTBmNDktNDA1ZS1iZWQxLTI3ZWI2ZDEwZmRlNCIsIngtaGFzdXJhLWRlZmF1bHQtcm9sZSI6InVzZXIiLCJ4LWhhc3VyYS1hbGxvd2VkLXJvbGVzIjpbInVzZXIiXSwieC1oYXN1cmEtdXNlci1ncm91cHMiOiJ7fSJ9LCJuYW1lIjoiSm9obiBIYW5zYXJpY2siLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJqb2huLmh1bW1lbEB3YWxsYXJvby5haSIsImdpdmVuX25hbWUiOiJKb2huIiwiZmFtaWx5X25hbWUiOiJIYW5zYXJpY2siLCJlbWFpbCI6ImpvaG4uaHVtbWVsQHdhbGxhcm9vLmFpIn0.T2kwa5d_277tLUIxtW5qXRhbT-gUR_AGgTqcbjFtAhOUDQOYgDryajFvmLxLGX0ye0ZpkLCNsxbGdLtxMyqJtQ5m4I0Ojp5c64yGuTo6ugt61PE-W0-Y9XYzfqdu5_ePz8OjblR16FCCzPTco_2L9HYXY_dy3NXpavHQsY8UVQqzyqioqoaIE23j5VY6my_bzNBOqxfOwOLq-zshqtuEFrM-d5fWlSpE7OmbpOoLCIbe_BfZfUh_ZUtDzy8Dt6iDAiCUkb_hv4803S-fJP-p7JDlOcFup2OCGyMrGfdidEhwiCBJHLcoikBnuu36hFhnmGb5S07H0-H9yreEY5FjBw'}

## Workspaces

### List User Workspaces

* **Endpoint**:  /v1/api/workspaces/list

List the workspaces for specified users.

#### List User Workspaces Parameters

| Field | Type | Description |
|---|---|---|
| **user_ids** | *List[Keycloak user ids]* (*Optional*) | An array of Keycloak user ids, typically in UUID format. |

If an empty set `{}` is submitted as a parameter, then the workspaces for users are returned.

#### List User Workspaces Returns

| Field | &nbsp; | Type | Description |
|---|---|---|---|
| **workspaces** | &nbsp; | *List[workspaces]* | A List of workspaces for the specified users.
| &nbsp; | **id** | *Integer* | The numerical ID of the workspace. |
| &nbsp; | **name** | *String* | The assigned name of the workspace. |
| &nbsp; | **create_at** | *String* | The DateTime the workspace was created. |
| &nbsp; | **create_by** | *String* | The Keycloak ID of the user who created the workspace. |
| &nbsp; | **archived** | *Boolean* | Whether the workspace is archived or not. |
| &nbsp; | **models** | *List[Integer]* | The model ids uploaded to the workspace. |
| &nbsp; | **pipelines** | *List[Integer]* | The pipeline ids built within the workspace. |

#### List User Workspaces Examples

In these example, the workspaces for all users will be displayed.

List all workspaces via Requests.

```python
# Retrieve the token 
headers = wl.auth.auth_header()

endpoint = f"{wl.api_endpoint}/v1/api/workspaces/list"

data = {
}

response = requests.post(endpoint, 
                         json=data, 
                         headers=headers, 
                         verify=True).json()
response
```

    {'workspaces': [{'id': 1,
       'name': 'john.hummel@wallaroo.ai - Default Workspace',
       'created_at': '2023-11-20T16:05:06.323911+00:00',
       'created_by': '12ea09d1-0f49-405e-bed1-27eb6d10fde4',
       'archived': False,
       'models': [],
       'pipelines': []},
      {'id': 5,
       'name': 'mobilenetworkspacetest',
       'created_at': '2023-11-20T16:05:48.271364+00:00',
       'created_by': '12ea09d1-0f49-405e-bed1-27eb6d10fde4',
       'archived': False,
       'models': [1, 2],
       'pipelines': [1]},
      {'id': 6,
       'name': 'edge-observability-assaysbaseline-examples',
       'created_at': '2023-11-20T16:09:31.950532+00:00',
       'created_by': '12ea09d1-0f49-405e-bed1-27eb6d10fde4',
       'archived': False,
       'models': [3],
       'pipelines': [4]},
      {'id': 7,
       'name': 'edge-observability-houseprice-demo',
       'created_at': '2023-11-20T17:36:07.131292+00:00',
       'created_by': '12ea09d1-0f49-405e-bed1-27eb6d10fde4',
       'archived': False,
       'models': [4, 5, 6, 7, 8, 9, 12],
       'pipelines': [7, 14, 16]},
      {'id': 8,
       'name': 'clip-demo',
       'created_at': '2023-11-20T18:57:53.667873+00:00',
       'created_by': '12ea09d1-0f49-405e-bed1-27eb6d10fde4',
       'archived': False,
       'models': [10, 11],
       'pipelines': [19]},
      {'id': 9,
       'name': 'onnx-tutorial',
       'created_at': '2023-11-22T16:24:47.786643+00:00',
       'created_by': '12ea09d1-0f49-405e-bed1-27eb6d10fde4',
       'archived': False,
       'models': [13],
       'pipelines': [22]}]}

List all workspaces via curl.

```python
!curl {wl.api_endpoint}/v1/api/workspaces/list \
    -H "Authorization: {wl.auth.auth_header()['Authorization']}" \
    -H "Content-Type: application/json" \
    --data '{{}}'
```

    {"workspaces":[{"id":1,"name":"john.hummel@wallaroo.ai - Default Workspace","created_at":"2023-11-20T16:05:06.323911+00:00","created_by":"12ea09d1-0f49-405e-bed1-27eb6d10fde4","archived":false,"models":[],"pipelines":[]},{"id":5,"name":"mobilenetworkspacetest","created_at":"2023-11-20T16:05:48.271364+00:00","created_by":"12ea09d1-0f49-405e-bed1-27eb6d10fde4","archived":false,"models":[1,2],"pipelines":[1]},{"id":6,"name":"edge-observability-assaysbaseline-examples","created_at":"2023-11-20T16:09:31.950532+00:00","created_by":"12ea09d1-0f49-405e-bed1-27eb6d10fde4","archived":false,"models":[3],"pipelines":[4]},{"id":7,"name":"edge-observability-houseprice-demo","created_at":"2023-11-20T17:36:07.131292+00:00","created_by":"12ea09d1-0f49-405e-bed1-27eb6d10fde4","archived":false,"models":[4,5,6,7,8,9,12],"pipelines":[7,14,16]},{"id":8,"name":"clip-demo","created_at":"2023-11-20T18:57:53.667873+00:00","created_by":"12ea09d1-0f49-405e-bed1-27eb6d10fde4","archived":false,"models":[10,11],"pipelines":[19]},{"id":9,"name":"onnx-tutorial","created_at":"2023-11-22T16:24:47.786643+00:00","created_by":"12ea09d1-0f49-405e-bed1-27eb6d10fde4","archived":false,"models":[13],"pipelines":[22]}]}

### Create Workspace

* **Endpoint**:  `/v1/api/workspaces/create`

A new workspace will be created in the Wallaroo instance.  Upon creating, the workspace owner will be assigned as the user making the MLOps API request.

#### Create Workspace Parameters

| Field | Type | Description |
|---|---|---|
| **workspace_name** | *String* (*REQUIRED*) | The name of the new workspace with the following requirements: <ul><li>Must be unique.</li>DNS compliant with only lowercase characters.</li></ul> |

#### Create Workspace Returns

| Field | Type | Description |
|---|---|---|
| **workspace_id** | *Integer* | The ID of the new workspace. |

#### Create Workspace Examples

In this example, workspaces named `testapiworkspace-requests` and `testapiworkspace-curl` will be created.

After the request is complete, the [List Workspaces](#list-workspaces) command will be issued to demonstrate the new workspace has been created.

Create workspace via Requests.

```python
# Retrieve the token 
headers = wl.auth.auth_header()

endpoint = f"{wl.api_endpoint}/v1/api/workspaces/create"

data = {
  "workspace_name": "testapiworkspace-requests"
}

response = requests.post(endpoint, json=data, headers=headers, verify=True).json()
display(response)

# Stored for future examples
example_workspace_id = response['workspace_id']
```

    {'workspace_id': 10}

```python
## List workspaces

# Retrieve the token 
headers = wl.auth.auth_header()

endpoint = f"{wl.api_endpoint}/v1/api/workspaces/list"

data = {
}

response = requests.post(endpoint, 
                         json=data, 
                         headers=headers, 
                         verify=True).json()
response
```

    {'workspaces': [{'id': 1,
       'name': 'john.hummel@wallaroo.ai - Default Workspace',
       'created_at': '2023-11-20T16:05:06.323911+00:00',
       'created_by': '12ea09d1-0f49-405e-bed1-27eb6d10fde4',
       'archived': False,
       'models': [],
       'pipelines': []},
      {'id': 5,
       'name': 'mobilenetworkspacetest',
       'created_at': '2023-11-20T16:05:48.271364+00:00',
       'created_by': '12ea09d1-0f49-405e-bed1-27eb6d10fde4',
       'archived': False,
       'models': [1, 2],
       'pipelines': [1]},
      {'id': 6,
       'name': 'edge-observability-assaysbaseline-examples',
       'created_at': '2023-11-20T16:09:31.950532+00:00',
       'created_by': '12ea09d1-0f49-405e-bed1-27eb6d10fde4',
       'archived': False,
       'models': [3],
       'pipelines': [4]},
      {'id': 7,
       'name': 'edge-observability-houseprice-demo',
       'created_at': '2023-11-20T17:36:07.131292+00:00',
       'created_by': '12ea09d1-0f49-405e-bed1-27eb6d10fde4',
       'archived': False,
       'models': [4, 5, 6, 7, 8, 9, 12],
       'pipelines': [7, 14, 16]},
      {'id': 8,
       'name': 'clip-demo',
       'created_at': '2023-11-20T18:57:53.667873+00:00',
       'created_by': '12ea09d1-0f49-405e-bed1-27eb6d10fde4',
       'archived': False,
       'models': [10, 11],
       'pipelines': [19]},
      {'id': 9,
       'name': 'onnx-tutorial',
       'created_at': '2023-11-22T16:24:47.786643+00:00',
       'created_by': '12ea09d1-0f49-405e-bed1-27eb6d10fde4',
       'archived': False,
       'models': [13],
       'pipelines': [22]},
      {'id': 10,
       'name': 'testapiworkspace-requests',
       'created_at': '2023-11-28T21:16:09.891951+00:00',
       'created_by': '12ea09d1-0f49-405e-bed1-27eb6d10fde4',
       'archived': False,
       'models': [],
       'pipelines': []}]}

Create workspace via curl.

```python
!curl {wl.api_endpoint}/v1/api/workspaces/create \
    -H "Authorization: {wl.auth.auth_header()['Authorization']}" \
    -H "Content-Type: application/json" \
    --data '{{"workspace_name": "testapiworkspace-curl"}}'
```

    {"workspace_id":12}

```python
!curl {wl.api_endpoint}/v1/api/workspaces/list \
    -H "Authorization: {wl.auth.auth_header()['Authorization']}" \
    -H "Content-Type: application/json" \
    --data '{{}}'
```

    {"workspaces":[{"id":1,"name":"john.hummel@wallaroo.ai - Default Workspace","created_at":"2023-11-20T16:05:06.323911+00:00","created_by":"12ea09d1-0f49-405e-bed1-27eb6d10fde4","archived":false,"models":[],"pipelines":[]},{"id":5,"name":"mobilenetworkspacetest","created_at":"2023-11-20T16:05:48.271364+00:00","created_by":"12ea09d1-0f49-405e-bed1-27eb6d10fde4","archived":false,"models":[1,2],"pipelines":[1]},{"id":6,"name":"edge-observability-assaysbaseline-examples","created_at":"2023-11-20T16:09:31.950532+00:00","created_by":"12ea09d1-0f49-405e-bed1-27eb6d10fde4","archived":false,"models":[3],"pipelines":[4]},{"id":7,"name":"edge-observability-houseprice-demo","created_at":"2023-11-20T17:36:07.131292+00:00","created_by":"12ea09d1-0f49-405e-bed1-27eb6d10fde4","archived":false,"models":[4,5,6,7,8,9,12],"pipelines":[7,14,16]},{"id":8,"name":"clip-demo","created_at":"2023-11-20T18:57:53.667873+00:00","created_by":"12ea09d1-0f49-405e-bed1-27eb6d10fde4","archived":false,"models":[10,11],"pipelines":[19]},{"id":9,"name":"onnx-tutorial","created_at":"2023-11-22T16:24:47.786643+00:00","created_by":"12ea09d1-0f49-405e-bed1-27eb6d10fde4","archived":false,"models":[13],"pipelines":[22]},{"id":10,"name":"testapiworkspace-requests","created_at":"2023-11-28T21:16:09.891951+00:00","created_by":"12ea09d1-0f49-405e-bed1-27eb6d10fde4","archived":false,"models":[],"pipelines":[]},{"id":12,"name":"testapiworkspace-curl","created_at":"2023-11-28T21:19:46.829351+00:00","created_by":"12ea09d1-0f49-405e-bed1-27eb6d10fde4","archived":false,"models":[],"pipelines":[]}]}

### Add User to Workspace

* **Endpoint**: `/v1/api/workspaces/add_user`

Existing users of the Wallaroo instance can be added to an existing workspace.

#### Add User to Workspace Parameters

| Field | Type | Description |
|---|---|---|
| **email** | *String* (*REQUIRED*) | The email address of the user to add to the workspace.  **This user must already exist in the Wallaroo instance.** |
| **workspace_id** | *Integer* (*REQUIRED*): The numerical id of the workspace.

#### Add User to Workspace Returns

Returns `{}` on a successful request.

#### Add User to Workspace Examples
  
The following example adds the user "john.hansarick@wallaroo.ai" to the workspace created in the previous step.

Add existing user to existing workspace via Requests.

```python
# Retrieve the token 
headers = wl.auth.auth_header()

endpoint = f"{wl.api_endpoint}/v1/api/workspaces/add_user"

data = {
  "email": "john.hansarick@wallaroo.ai",
  "workspace_id": example_workspace_id
}

response = requests.post(endpoint, 
                         json=data, 
                         headers=headers, 
                         verify=True).json()
response
```

    {}

Add existing user to existing workspace via curl.

```python
!curl {wl.api_endpoint}/v1/api/workspaces/add_user \
    -H "Authorization: {wl.auth.auth_header()['Authorization']}" \
    -H "Content-Type: application/json" \
    --data '{{"email": "john.hansarick@wallaroo.ai","workspace_id": {example_workspace_id}}}'
```

    {}

### List Users in a Workspace

* **Endpoint**: `/v1/api/workspaces/list_users`

Lists the users who are either owners or collaborators of a workspace.

#### List Users in a Workspace Parameters

| Field | Type | Description |
|---|---|---|
| **workspace_id** | *Integer (*REQUIRED*) | The id of the workspace. |

#### List Users in a Workspace Returns

| Field | &nbsp; | Type | Description |
|---|---|---|---|
| **users** | &nbsp; | *List[users]* | The list of users and attributes in the workspace.
| &nbsp; | **user_id** | *String* | The user's Keycloak id. |
| &nbsp; | **user_type** | *String* | The user's workspace type of `OWNER` or `COLLABORATOR`. |

#### List Users in a Workspace Examples

The following examples list all users part a workspace created in a previous request.

List users in a workspace via Requests.

```python
# Retrieve the token 

headers = wl.auth.auth_header()

endpoint = f"{wl.api_endpoint}/v1/api/workspaces/list_users"

data = {
  "workspace_id": example_workspace_id
}

response = requests.post(endpoint, json=data, headers=headers, verify=True).json()
response
```

    {'users': [{'user_id': '12ea09d1-0f49-405e-bed1-27eb6d10fde4',
       'user_type': 'OWNER'},
      {'user_id': '57d61aed-3058-4327-9e65-a5d39a9718c0',
       'user_type': 'COLLABORATOR'}]}

List users in a workspace via curl.

```python
!curl {wl.api_endpoint}/v1/api/workspaces/list_users \
    -H "Authorization: {wl.auth.auth_header()['Authorization']}" \
    -H "Content-Type: application/json" \
    --data '{{"workspace_id": {example_workspace_id}}}'
```

    {"users":[{"user_id":"12ea09d1-0f49-405e-bed1-27eb6d10fde4","user_type":"OWNER"},{"user_id":"57d61aed-3058-4327-9e65-a5d39a9718c0","user_type":"COLLABORATOR"}]}

### Remove User from a Workspace

Removes the user from the given workspace.  In this request, either the user's Keycloak ID is required **OR** the user's email address is required.

#### Remove User from a Workspace Parameters

| Field | Type | Description |
|---|---|---|
| **workspace_id** | *Integer* (*Required*) | The id of the workspace. |
| **user_id** | *String* (*Optional*) |  The Keycloak ID of the user.  If `email` is not provided, then this parameter is **REQUIRED**. |
| **email** | *String* (*Optional*) | The user's email address.  If `user_id` is not provided, then this parameter is **REQUIRED**. |

#### Remove User from a Workspace Returns

| Field | Type | Description |
|---|---|---|
| **affected_rows** | *Integer* | The number of workspaces effected by the change. |

#### Remove User from a Workspace Examples

The following example will remove the user `john.hansarick@wallaroo.ai` from a workspace created the previous steps.  Then the list of users for the workspace is retrieved to verify the change.

Remove existing user from an existing workspace via Requests.

```python
# Retrieve the token 
headers = wl.auth.auth_header()

endpoint = f"{wl.api_endpoint}/v1/api/workspaces/remove_user"

data = {
  "email": "john.hansarick@wallaroo.ai",
  "workspace_id": example_workspace_id
}

response = requests.post(endpoint, json=data, headers=headers, verify=True).json()
response
```

    {'affected_rows': 1}

```python
# Retrieve the token 

headers = wl.auth.auth_header()

endpoint = f"{wl.api_endpoint}/v1/api/workspaces/list_users"

data = {
  "workspace_id": example_workspace_id
}

response = requests.post(endpoint, json=data, headers=headers, verify=True).json()
response
```

    {'users': [{'user_id': '12ea09d1-0f49-405e-bed1-27eb6d10fde4',
       'user_type': 'OWNER'}]}

Remove existing user from an existing workspace via curl.

```python
!curl {wl.api_endpoint}/v1/api/workspaces/remove_user \
    -H "Authorization: {wl.auth.auth_header()['Authorization']}" \
    -H "Content-Type: application/json" \
    --data '{{"email": "john.hansarick@wallaroo.ai","workspace_id": {example_workspace_id}}}'
```

    {"affected_rows":0}

```python
!curl {wl.api_endpoint}/v1/api/workspaces/list_users \
    -H "Authorization: {wl.auth.auth_header()['Authorization']}" \
    -H "Content-Type: application/json" \
    --data '{{"workspace_id": {example_workspace_id}}}'
```

    {"users":[{"user_id":"12ea09d1-0f49-405e-bed1-27eb6d10fde4","user_type":"OWNER"}]}
