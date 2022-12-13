This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/mlops_api).

## Wallaroo MLOps API Tutorial

The Wallaroo MLOps API allows organizations to submit requests to their Wallaroo instance to perform such actions as:

* Create a new user and invite them to the instance.
* Create workspaces and list their configuration details.
* Upload a model.
* Deploy and undeploy a pipeline.

The following examples will show how to submit queries to the Wallaroo MLOps API and the types of responses returned.

### References

The following references are available for more information about Wallaroo and the Wallaroo MLOps API:

* [Wallaroo Documentation Site](https://docs.wallaroo.ai):  The Wallaroo Documentation Site
* Wallaroo MLOps API Documentation from a Wallaroo instance:  A Swagger UI based documentation is available from your Wallaroo instance at `https://{Wallaroo Prefix}.api.{Wallaroo Suffix}/v1/api/docs`.  For example, if the Wallaroo Instance is Wallaroo Community with the prefix `{lovely-rhino-5555}`, then the Wallaroo MLOps API Documentation would be available at `https://lovely-rhino-5555.api.wallaroo.community/v1/api/docs`.  For another example, a Wallaroo Enterprise users who do not use a prefix and has the suffix `wallaroo.example.com`, the the Wallaroo MLOps API Documentation would be available at `https://api.wallaroo.example.com/v1/api/docs`.  For more information, see the [Wallaroo Documentation Site](https://docs.wallaroo.ai).

**IMPORTANT NOTE**:  This tutorial is in beta while the MLOps API is updated.  Some commands may not work as currently documented and will be updated in an upcoming release.  Other commands may undergo changes to be more user friendly.

## OpenAPI Steps

The following demonstrates how to use each command in the Wallaroo MLOps API, and can be modified as best fits your organization's needs.

### Import Libraries

For the examples, the Python `requests` library will be used to make the REST HTTP(S) connections.  `import uuid` will be used to create workspaces, pipelines, assays and other items uniquely so we don't go clobbering over existing items.


```python
# Requires requests and requests-toolbelt with either:
# pip install requests-toolbelt
# conda install -c conda-forge requests-toolbelt

import requests
from requests.auth import HTTPBasicAuth

import uuid
```

### Set Variables

The following variables are used for the example and should be modified to fit your organization.

Wallaroo comes pre-installed with a confidential OpenID Connect client.  The default client is `api-client`, but other clients may be created and configured.

As it is a confidential client, api-client requires its secret to be supplied when requesting a token. Administrators may obtain their API client credentials from Keycloak from the Keycloak Service URL as listed above and the prefix `/auth/admin/master/console/#/realms/master/clients`.

For example, if the Wallaroo Community instance DNS address is `https://magical-rhino-5555.wallaroo.community`, then the direct path to the Keycloak API client credentials would be:

`https://magical-rhino-5555.keycloak.wallaroo.community/auth/admin/master/console/#/realms/master/clients`

Then select the client, in this case **api-client**, then **Credentials**.

![Wallaroo Keycloak Service](../images/wallaroo-developer-guides/wallaroo-api/wallaroo-api-keycloak-service.png)

![Wallaroo Components](../images/wallaroo-developer-guides/wallaroo-api/wallaroo-api-keycloak-credentials.png)

By default, tokens issued for api-client are valid for up to 60 minutes. Refresh tokens are supported.


```python
## Variables

URLPREFIX='YOUR PREFIX'
URLSUFFIX='YOUR SUFFIX'
SECRET="YOUR SECRET"
TOKENURL=f'https://{URLPREFIX}.keycloak.{URLSUFFIX}/auth/realms/master/protocol/openid-connect/token'
CLIENT="api-client"
USERNAME="YOUR EMAIL"
PASSWORD="YOUR PASSWORD"
APIURL=f"https://{URLPREFIX}.api.{URLSUFFIX}/v1/api"
newUser="NEW USER EMAIL"
newPassword="NEW USER PASSWORD"
```

The following is an output of the `TOKENURL` variable to verify it matches your Wallaroo instance's Keycloak API client credentials URL.


```python
TOKENURL
```




    'https://magical-bear-3782.keycloak.wallaroo.community/auth/realms/master/protocol/openid-connect/token'



### API Example Methods

The following methods are used to retrieve the MLOPs API Token from the Wallaroo instance's Keycloak service, and submit MLOps API requests through the Wallaroo instance's MLOps API.

MLOps API requests are always `POST`, and are either submitted as `'Content-Type':'application/json'` or as a multipart submission including a file.


```python
def get_jwt_token(url, client, secret, username, password):
    auth = HTTPBasicAuth(client, secret)
    data = {
        'grant_type': 'password',
        'username': username,
        'password': password
    }
    response = requests.post(url, auth=auth, data=data, verify=True)
    return response.json()['access_token']


# This can either submit a plain POST request ('Content-Type':'application/json'), or with a file.

def get_wallaroo_response(url, api_request, token, data, files=None, contentType='application/json', params=None):
    apiUrl=f"{url}{api_request}"
    if files is None:
        # Regular POST request
        headers= {
            'Authorization': 'Bearer ' + token,
            'Content-Type':contentType
        }
        response = requests.post(apiUrl, json=data, headers=headers, verify=True)
    elif contentType == 'application/octet-stream':
        # Post request as octet-stream
        headers= {
            'Authorization': 'Bearer ' + token,
            'Content-Type':contentType
        }
        response = requests.post(apiUrl, data=files, headers=headers, params=params)
        #response = requests.post(apiUrl, data=data, headers=headers, files=files, verify=True)
    else:
        # POST request with file
        headers= {
            'Authorization': 'Bearer ' + token
        }
        response = requests.post(apiUrl, data=data, headers=headers, files=files, verify=True)
    return response.json()

```

### Retrieve MLOps API Token

To retrieve an API token for a specific user with the Client Secret, request the token from the Wallaroo instance using the client secret and provide the following:

* Token Request URL: The Keycloak token retrieval URL.
* OpenID Connect client name: The name of the OpenID Connect client.
* OpenID Connect client Secret:  The secret for the OpenID Connect client.
* UserName:  The username of the Wallaroo instance user, usually their email address.
* Password:  The password of the Wallaroo instance user.

The following sample uses the variables set above to request the token, then displays it.


```python
TOKEN=get_jwt_token(TOKENURL, CLIENT, SECRET, USERNAME, PASSWORD)
```


```python
TOKEN
```




    'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJBbV9ESjd5VjJKeVVOQUd6SGFPczdpeENTNlIzX180RGNQZDRVOGQxNkJzIn0.eyJleHAiOjE2Njk2NzQ1MzUsImlhdCI6MTY2OTY3MDkzNSwianRpIjoiODk2MDE0YjctZjJhNS00OTFkLTg5YWItMGVlNWQwNjdlZjE3IiwiaXNzIjoiaHR0cHM6Ly9tYWdpY2FsLWJlYXItMzc4Mi5rZXljbG9hay53YWxsYXJvby5jb21tdW5pdHkvYXV0aC9yZWFsbXMvbWFzdGVyIiwiYXVkIjpbIm1hc3Rlci1yZWFsbSIsImFjY291bnQiXSwic3ViIjoiNWU5YzlhMmItN2E3Zi00NTRhLWI4ZTctOTFlM2MyZDg2YzlmIiwidHlwIjoiQmVhcmVyIiwiYXpwIjoiYXBpLWNsaWVudCIsInNlc3Npb25fc3RhdGUiOiI4OWJkOWQ5MC0zOTg3LTQ4ZmMtYTQwMC1jMzYzOWQ4MGFhYmYiLCJhY3IiOiIxIiwiYWxsb3dlZC1vcmlnaW5zIjpbIioiXSwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbImNyZWF0ZS1yZWFsbSIsImRlZmF1bHQtcm9sZXMtbWFzdGVyIiwib2ZmbGluZV9hY2Nlc3MiLCJhZG1pbiIsInVtYV9hdXRob3JpemF0aW9uIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsibWFzdGVyLXJlYWxtIjp7InJvbGVzIjpbInZpZXctaWRlbnRpdHktcHJvdmlkZXJzIiwidmlldy1yZWFsbSIsIm1hbmFnZS1pZGVudGl0eS1wcm92aWRlcnMiLCJpbXBlcnNvbmF0aW9uIiwiY3JlYXRlLWNsaWVudCIsIm1hbmFnZS11c2VycyIsInF1ZXJ5LXJlYWxtcyIsInZpZXctYXV0aG9yaXphdGlvbiIsInF1ZXJ5LWNsaWVudHMiLCJxdWVyeS11c2VycyIsIm1hbmFnZS1ldmVudHMiLCJtYW5hZ2UtcmVhbG0iLCJ2aWV3LWV2ZW50cyIsInZpZXctdXNlcnMiLCJ2aWV3LWNsaWVudHMiLCJtYW5hZ2UtYXV0aG9yaXphdGlvbiIsIm1hbmFnZS1jbGllbnRzIiwicXVlcnktZ3JvdXBzIl19LCJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6InByb2ZpbGUgZW1haWwiLCJzaWQiOiI4OWJkOWQ5MC0zOTg3LTQ4ZmMtYTQwMC1jMzYzOWQ4MGFhYmYiLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiaHR0cHM6Ly9oYXN1cmEuaW8vand0L2NsYWltcyI6eyJ4LWhhc3VyYS11c2VyLWlkIjoiNWU5YzlhMmItN2E3Zi00NTRhLWI4ZTctOTFlM2MyZDg2YzlmIiwieC1oYXN1cmEtZGVmYXVsdC1yb2xlIjoidXNlciIsIngtaGFzdXJhLWFsbG93ZWQtcm9sZXMiOlsidXNlciJdLCJ4LWhhc3VyYS11c2VyLWdyb3VwcyI6Int9In0sInByZWZlcnJlZF91c2VybmFtZSI6ImpvaG4uaGFuc2FyaWNrQHdhbGxhcm9vLmFpIiwiZW1haWwiOiJqb2huLmhhbnNhcmlja0B3YWxsYXJvby5haSJ9.N4aqBvNcqVv3Iwr3uhGCzXd6C4tpD3D-fSJKfe_fYoXEJPTeLL84sw1Ady9kJLmynj8cWkVAhps0GveDB6mkAyYH7uUOaM-mXhBn7kphM-q4PaDWUwRZ2xIViamTh1IVzFpQbCKu6olrDHHqrTn78mZuSd_OXosIll5vXSLyuqk-Ob8HaBry_n-JVHYBqmnnSJIPNUkwQqiiOXya3Rl90I68vQrsvgt_lF_IUsDBgP8tALofoS5U2k3cAxFVhZ6jVIeZ6w6Vgl93T0xNOIeKd7Nc16_VEJOiWmr_MsHr8---EwIfqgYzgAb3YSidHiY7bR6_9PjcZSofhohVzkrokQ'



## Users

### Get Users

Users can be retrieved either by their Keycloak user id, or return all users if an empty set `{}` is submitted.

* **Parameters**
  * `{}`: Empty set, returns all users.
  * **user_ids** *Array[Keycloak user ids]*: An array of Keycloak user ids, typically in UUID format.

Example:  The first example will submit an empty set `{}` to return all users, then submit the first user's user id and request only that user's details.


```python
# Get all users

apiRequest = "/users/query"
data = {
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```




    {'users': {'5e9c9a2b-7a7f-454a-b8e7-91e3c2d86c9f': {'access': {'impersonate': True,
        'manageGroupMembership': True,
        'manage': True,
        'mapRoles': True,
        'view': True},
       'createdTimestamp': 1669221287375,
       'disableableCredentialTypes': [],
       'email': 'john.hansarick@wallaroo.ai',
       'emailVerified': True,
       'enabled': True,
       'id': '5e9c9a2b-7a7f-454a-b8e7-91e3c2d86c9f',
       'notBefore': 0,
       'requiredActions': [],
       'username': 'john.hansarick@wallaroo.ai'},
      '941937b3-7dc8-4abe-8bb1-bd23c816421e': {'access': {'view': True,
        'manage': True,
        'manageGroupMembership': True,
        'mapRoles': True,
        'impersonate': True},
       'createdTimestamp': 1669221214282,
       'disableableCredentialTypes': [],
       'emailVerified': False,
       'enabled': True,
       'id': '941937b3-7dc8-4abe-8bb1-bd23c816421e',
       'notBefore': 0,
       'requiredActions': [],
       'username': 'admin'},
      'da7c2f4c-822e-49eb-93d7-a4b90af9b4ca': {'access': {'mapRoles': True,
        'impersonate': True,
        'manage': True,
        'manageGroupMembership': True,
        'view': True},
       'createdTimestamp': 1669654086172,
       'disableableCredentialTypes': [],
       'email': 'kilvin.mitchell@wallaroo.ai',
       'emailVerified': True,
       'enabled': True,
       'id': 'da7c2f4c-822e-49eb-93d7-a4b90af9b4ca',
       'notBefore': 0,
       'requiredActions': [],
       'username': 'kilvin.mitchell@wallaroo.ai'}}}




```python
# Get first user Keycloak id
firstUserKeycloak = list(response['users'])[0]

apiRequest = "/users/query"
data = {
  "user_ids": [
    firstUserKeycloak
  ]
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```




    {'users': {'5e9c9a2b-7a7f-454a-b8e7-91e3c2d86c9f': {'access': {'view': True,
        'manage': True,
        'manageGroupMembership': True,
        'mapRoles': True,
        'impersonate': True},
       'createdTimestamp': 1669221287375,
       'disableableCredentialTypes': [],
       'email': 'john.hansarick@wallaroo.ai',
       'emailVerified': True,
       'enabled': True,
       'id': '5e9c9a2b-7a7f-454a-b8e7-91e3c2d86c9f',
       'notBefore': 0,
       'requiredActions': [],
       'username': 'john.hansarick@wallaroo.ai'}}}



### Invite Users

**IMPORTANT NOTE**:  This command is for Wallaroo Community only.  For more details on user management, see [Wallaroo User Management](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-user-management/).

Users can be invited through `/users/invite`.  When using Wallaroo Community, this will send an invitation email to the email address listed.  Note that the user must not already be a member of the Wallaroo instance, and email addresses must be unique.  If the email address is already in use for another user, the request will generate an error.

* **Parameters**
  * **email** *(REQUIRED string): The email address of the new user to invite.
  * **password** *(OPTIONAL string)*: The assigned password of the new user to invite.  If not provided, the Wallaroo instance will provide the new user a temporary password that must be changed upon initial login.

Example:  In this example, a new user will be invited to the Wallaroo instance and assigned a password.


```python
# invite users
apiRequest = "/users/invite"
data = {
    "email": newUser,
    "password":newPassword
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```

### Deactivate User

Users can be deactivated so they can not login to their Wallaroo instance.  Deactivated users do not count against the Wallaroo license count.

* **Parameters**
  * **email** (*REQUIRED string*):  The email address of the user to deactivate.

Example:  In this example, the `newUser` will be deactivated.


```python
# Deactivate users

apiRequest = "/users/deactivate"

data = {
    "email": newUser
}
```


```python
response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```




    {}



### Activate User

A deactivated user can be reactivated to allow them access to their Wallaroo instance.  Activated users count against the Wallaroo license count.

* **Parameters**
  * **email** (*REQUIRED string*):  The email address of the user to activate.

Example:  In this example, the `newUser` will be activated.


```python
# Activate users

apiRequest = "/users/activate"

data = {
    "email": newUser
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```




    {}



## Workspaces

### List Workspaces

List the workspaces for a specific user.

* **Parameters**
  * **user_id** - (*OPTIONAL string*): The Keycloak ID.
  
Example:  In this example, the workspaces for the a specific user will be displayed, then workspaces for all users will be displayed.


```python
# List workspaces by user id

apiRequest = "/workspaces/list"

data = {
    "user_id":firstUserKeycloak
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```




    {'workspaces': [{'id': 1,
       'name': 'john.hansarick@wallaroo.ai - Default Workspace',
       'created_at': '2022-11-23T16:34:47.914362+00:00',
       'created_by': '5e9c9a2b-7a7f-454a-b8e7-91e3c2d86c9f',
       'archived': False,
       'models': [],
       'pipelines': []},
      {'id': 2,
       'name': 'alohaworkspace',
       'created_at': '2022-11-23T16:44:28.782225+00:00',
       'created_by': '5e9c9a2b-7a7f-454a-b8e7-91e3c2d86c9f',
       'archived': False,
       'models': [1],
       'pipelines': [1]},
      {'id': 4,
       'name': 'testapiworkspace-cdf86c3c-8c9a-4bf4-865d-fe0ec00fad7c',
       'created_at': '2022-11-28T16:48:29.622794+00:00',
       'created_by': '5e9c9a2b-7a7f-454a-b8e7-91e3c2d86c9f',
       'archived': False,
       'models': [2, 3, 5, 4, 6, 7, 8, 9],
       'pipelines': []}]}




```python
# List workspaces

apiRequest = "/workspaces/list"

data = {
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```




    {'workspaces': [{'id': 1,
       'name': 'john.hansarick@wallaroo.ai - Default Workspace',
       'created_at': '2022-11-23T16:34:47.914362+00:00',
       'created_by': '5e9c9a2b-7a7f-454a-b8e7-91e3c2d86c9f',
       'archived': False,
       'models': [],
       'pipelines': []},
      {'id': 2,
       'name': 'alohaworkspace',
       'created_at': '2022-11-23T16:44:28.782225+00:00',
       'created_by': '5e9c9a2b-7a7f-454a-b8e7-91e3c2d86c9f',
       'archived': False,
       'models': [1],
       'pipelines': [1]},
      {'id': 4,
       'name': 'testapiworkspace-cdf86c3c-8c9a-4bf4-865d-fe0ec00fad7c',
       'created_at': '2022-11-28T16:48:29.622794+00:00',
       'created_by': '5e9c9a2b-7a7f-454a-b8e7-91e3c2d86c9f',
       'archived': False,
       'models': [2, 3, 5, 4, 6, 7, 8, 9],
       'pipelines': []}]}



### Create Workspace

A new workspace will be created in the Wallaroo instance.  Upon creating, the workspace owner will be assigned as the user making the MLOps API request.

* **Parameters**:
  * **workspace_name** - (*REQUIRED string*):  The name of the new workspace.
* **Returns**:
  * **workspace_id** - (*int*):  The ID of the new workspace.
  
Example:  In this example, a workspace with the name `testapiworkspace-` with a randomly generated UUID will be created, and the newly created workspace's `workspace_id` saved for use in other code examples.  After the request is complete, the [List Workspaces](#list-workspaces) command will be issued to demonstrate the new workspace has been created.


```python
# Create workspace

apiRequest = "/workspaces/create"

exampleWorkspaceName = f"testapiworkspace-{uuid.uuid4()}"
data = {
  "workspace_name": exampleWorkspaceName
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
# Stored for future examples
exampleWorkspaceId = response['workspace_id']
response
```




    {'workspace_id': 5}




```python
# List workspaces

apiRequest = "/workspaces/list"

data = {
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```




    {'workspaces': [{'id': 1,
       'name': 'john.hansarick@wallaroo.ai - Default Workspace',
       'created_at': '2022-11-23T16:34:47.914362+00:00',
       'created_by': '5e9c9a2b-7a7f-454a-b8e7-91e3c2d86c9f',
       'archived': False,
       'models': [],
       'pipelines': []},
      {'id': 2,
       'name': 'alohaworkspace',
       'created_at': '2022-11-23T16:44:28.782225+00:00',
       'created_by': '5e9c9a2b-7a7f-454a-b8e7-91e3c2d86c9f',
       'archived': False,
       'models': [1],
       'pipelines': [1]},
      {'id': 4,
       'name': 'testapiworkspace-cdf86c3c-8c9a-4bf4-865d-fe0ec00fad7c',
       'created_at': '2022-11-28T16:48:29.622794+00:00',
       'created_by': '5e9c9a2b-7a7f-454a-b8e7-91e3c2d86c9f',
       'archived': False,
       'models': [2, 3, 5, 4, 6, 7, 8, 9],
       'pipelines': []},
      {'id': 5,
       'name': 'testapiworkspace-e8f42d00-b3a6-40a3-9309-1f1a0f4dba3f',
       'created_at': '2022-11-28T21:29:36.268636+00:00',
       'created_by': '5e9c9a2b-7a7f-454a-b8e7-91e3c2d86c9f',
       'archived': False,
       'models': [],
       'pipelines': []}]}



### Add User to Workspace

Existing users of the Wallaroo instance can be added to an existing workspace.

* **Parameters**
  * **email** - (*REQUIRED string*):  The email address of the user to add to the workspace.
  * **workspace_id** - (*REQUIRED int*): The id of the workspace.
  
Example:  The following example adds the user created in Invite Users request to the workspace created in the [Create Workspace](#create-workspace) request.


```python
# Add existing user to existing workspace

apiRequest = "/workspaces/add_user"

data = {
  "email":newUser,
  "workspace_id": exampleWorkspaceId
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```




    {}



### List Users in a Workspace

Lists the users who are either owners or collaborators of a workspace.

* **Parameters**
  * **workspace_id** - (*REQUIRED int*): The id of the workspace.
* **Returns**
  * **user_id**:  The user's identification.
  * **user_type**:  The user's workspace type (owner, co-owner, etc).
  
Example:  The following example will list all users part of the workspace created in the [Create Workspace](#create-workspace) request.


```python
# List users in a workspace

apiRequest = "/workspaces/list_users"

data = {
  "workspace_id": exampleWorkspaceId
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```




    {'users': [{'user_id': '5e9c9a2b-7a7f-454a-b8e7-91e3c2d86c9f',
       'user_type': 'OWNER'},
      {'user_id': 'da7c2f4c-822e-49eb-93d7-a4b90af9b4ca',
       'user_type': 'COLLABORATOR'}]}



### Remove User from a Workspace

Removes the user from the given workspace.  In this request, either the user's Keycloak ID is required **OR** the user's email address is required.

* **Parameters**
  * **workspace_id** - (*REQUIRED int*): The id of the workspace.
  * **user_id** - (*string*): The Keycloak ID of the user.  If `email` is not provided, then this parameter is **REQUIRED**.
  * **email** - (*string*): The user's email address.  If `user_id` is not provided, then this parameter is **REQUIRED**.
* **Returns**
  * **user_id**:  The user's identification.
  * **user_type**:  The user's workspace type (owner, co-owner, etc).
  
Example:  The following example will remove the `newUser` from workspace created in the [Create Workspace](#create-workspace) request.  Then the users for that workspace will be listed to verify `newUser` has been removed.


```python
# Remove existing user from an existing workspace

apiRequest = "/workspaces/remove_user"

data = {
  "email":newUser,
  "workspace_id": exampleWorkspaceId
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```




    {'affected_rows': 1}




```python
# List users in a workspace

apiRequest = "/workspaces/list_users"

data = {
  "workspace_id": exampleWorkspaceId
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```




    {'users': [{'user_id': '5e9c9a2b-7a7f-454a-b8e7-91e3c2d86c9f',
       'user_type': 'OWNER'}]}



## Models

### Upload Model to Workspace

Uploads a ML Model to a Wallaroo workspace via POST with `Content-Type: multipart/form-data`.

* **Parameters**
  * **name** - (*REQUIRED string*): Name of the model
  * **visibility** - (*OPTIONAL string*): The visibility of the model as either `public` or `private`.
  * **workspace_id** - (*REQUIRED int*): The numerical id of the workspace to upload the model to.
  
Example:  This example will upload the sample file `ccfraud.onnx` to the workspace created in the [Create Workspace](#create-workspace) step as `apitestmodel`.  The id of the uploaded model will be saved for later examples.


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




    {'insert_models': {'returning': [{'models': [{'id': 10}]}]}}




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
  
Example:  This example will upload the sample file `ccfraud.onnx` to the workspace created in the [Create Workspace](#create-workspace) step as `apitestmodel`.  The id of the uploaded model will be saved for later examples.


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
  
Example:  Display the models for the workspace used in the Upload Model to Workspace step.  The details of the models will be saved as variables for other examples.


```python
# List models in a workspace

apiRequest = "/models/list"

data = {
  "workspace_id": exampleWorkspaceId
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```




    {'models': [{'id': 11,
       'name': 'apitestmodel-08b16b74-837c-4b4b-a6b7-49475ddece98',
       'owner_id': '""',
       'created_at': '2022-11-28T21:30:05.071826+00:00',
       'updated_at': '2022-11-28T21:30:05.071826+00:00'},
      {'id': 10,
       'name': 'apitestmodel-8e5d6d2d-37cb-4819-9c91-e123b034cf1e',
       'owner_id': '""',
       'created_at': '2022-11-28T21:29:58.270976+00:00',
       'updated_at': '2022-11-28T21:29:58.270976+00:00'}]}




```python
#exampleModelSha = response['models'][0]['models']['sha']
#exampleModelVersion = response['models'][0]['models']['model_version']
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


Example:  Retrieve the versions for a previously uploaded model.


```python
# List models in a workspace

apiRequest = "/models/list_versions"

data = {
  "model_id": "apitestmodel-290e6e7f-691d-46d2-9e41-9a4b8115c00b",
  "models_pk_id": 0
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```




    [{'sha': 'bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507',
      'models_pk_id': 9,
      'model_version': '5fb09950-b69a-4585-a1f0-4992c945b80f',
      'owner_id': '""',
      'model_id': 'apitestmodel-290e6e7f-691d-46d2-9e41-9a4b8115c00b',
      'id': 9,
      'file_name': 'streamfile.onnx',
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

Example:  Two pipelines are created in the workspace created in the step Create Workspace.  One will be an empty pipeline without any models, the other will be created using the uploaded models in the Upload Model to Workspace step and no configuration details.  The pipeline details will be stored for later examples.


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




    {'pipeline_pk_id': 4,
     'pipeline_variant_pk_id': 4,
     'pipeline_variant_version': '6526328c-5b13-4430-9ba2-ece4971515fc'}



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

Examples:  Both the empty pipeline and pipeline with model created in the step Create Pipeline in a Workspace will be deployed and their deployment information saved for later examples.



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




    {'id': 3}



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




    {'status': 'Starting',
     'details': [],
     'engines': [{'ip': None,
       'name': 'engine-6f7bfc658f-c9nln',
       'status': 'Pending',
       'reason': None,
       'details': ['containers with unready status: [engine]',
        'containers with unready status: [engine]'],
       'pipeline_statuses': None,
       'model_statuses': None}],
     'engine_lbs': [{'ip': '10.244.4.13',
       'name': 'engine-lb-c6485cfd5-45ppc',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}




```python
# Get model pipeline deployment

apiRequest = "/status/get_deployment"

data = {
  "name": f"{exampleModelDeployId}-{exampleModelDeploymentId}"
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```




    {'status': 'Starting',
     'details': [],
     'engines': [{'ip': None,
       'name': 'engine-958b7888d-48pqj',
       'status': 'Pending',
       'reason': None,
       'details': ['containers with unready status: [engine]',
        'containers with unready status: [engine]'],
       'pipeline_statuses': None,
       'model_statuses': None}],
     'engine_lbs': [{'ip': '10.244.0.7',
       'name': 'engine-lb-c6485cfd5-p89jr',
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

Modify these values to match the ones used in the Internal Pipeline Deployment URL Tutorial.


```python
## Start with the a lists of the workspaces to verify the ID

# List workspaces

apiRequest = "/workspaces/list"

data = {
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```


```python
workspaceList = response['workspaces']
workspaceId = list(filter(lambda x:x["name"]=="urldemoworkspace",workspaceList))[0]['id']
workspaceId
```


```python
## Retrieve the pipeline's External Inference URL

apiRequest = "/admin/get_pipeline_external_url"

data = {
    "workspace_id": workspaceId,
    "pipeline_name": 'urldemopipeline'
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
externalUrl = response['url']
externalUrl
```

### Perform Inference Through External URL

The inference can now be performed through the External Inference URL.  This URL will accept the same inference data file that is used with the Wallaroo SDK, or with an Internal Inference URL as used in the Internal Pipeline Inference URL Tutorial.

For this example, the `externalUrl` retrieved through the [Get External Inference URL](#get-external-inference-url) is used to submit a single inference request through the data file `data-1.json`.


```python
## Inference through external URL

# retrieve the json data to submit
data = json.load(open('./data/data-1k.json','rb'))

# set the headers
headers= {
        'Authorization': 'Bearer ' + TOKEN
    }

# submit the request via POST
response = requests.post(externalUrl, json=data, headers=headers)

# Only the first 300 characters will be displayed for brevity
printResponse = json.dumps(response.json())
print(printResponse[0:300])

```

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



## List Enablement Features

Lists the enablement features for the Wallaroo instance.

* **PARAMETERS**
  * null:  An empty set `{}`
* **RETURNS**
  * **features** - (*string*): Enabled features.
  * **name** - (*string*): Name of the Wallaroo instance.
  * **is_auth_enabled** - (*bool*): Whether authentication is enabled.


```python
# List enablement features

apiRequest = "/features/list"

data = {
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```




    {'features': {'plateau': 'true'},
     'name': 'Wallaroo Dev',
     'is_auth_enabled': True}



## Assays

Note: These assays were run in a Wallaroo environment with canned historical data.  See the [Wallaroo Assay Tutorial](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/model_insights) for details on setting up this environment.

### Create Assay

Create a new array in a specified pipeline.

* **PARAMETERS**
  * **id** - (*OPTIONAL int*):  The numerical identifier for the assay.
  * **name** - (*REQUIRED string*): The name of the assay.
  * **pipeline_id** - (*REQUIRED int*): The numerical idenfifier the assay will be placed into.
  * **pipeline_name** - (*REQUIRED string*): The name of the pipeline
  * **active** - (*REQUIRED bool*): Indicates whether the assay will be active upon creation or not.
  * **status** - (*REQUIRED string*): The status of the assay upon creation.
  * **iopath** - (*REQUIRED string*): The iopath of the assay.
  * **baseline** - (*REQUIRED baseline*): The baseline for the assay.
    * **Fixed** - (*REQUIRED AssayFixConfiguration*): The fixed configuration for the assay.
      * **pipeline** - (*REQUIRED string*): The name of the pipeline with the baseline data.
      * **model** - (*REQUIRED string*): The name of the model used.
      * **start_at** - (*REQUIRED string*): The DateTime of the baseline start date.
      * **end_at** - (*REQUIRED string*): The DateTime of the baseline end date.
  * **window** (*REQUIRED AssayWindow*): Assay window.
    * **pipeline** - (*REQUIRED string*): The name of the pipeline for the assay window.
    * **model** - (*REQUIRED string*): The name of the model used for the assay window.
    * **width** - (*REQUIRED string*): The width of the assay window.
    * **start** - (*OPTIONAL string*): The DateTime of when to start the assay window.
    * **interval** - (*OPTIONAL string*): The assay window interval.
  * **summarizer** - (*REQUIRED AssaySummerizer*): The summarizer type for the array aka "advanced settings" in the Wallaroo Dashboard UI.
    * **type** - (*REQUIRED string*): Type of summarizer.
    * **bin_mode** - (*REQUIRED string*): The binning model type.  Values can be:
      * Quantile
      * Equal
    * **aggregation** - (*REQUIRED string*): Aggregation type.
    * **metric** - (*REQUIRED string*): Metric type.  Values can be:
      * PSI
      * Maximum Difference of Bins
      * Sum of the Difference of Bins
    * **num_bins** - (*REQUIRED int*): The number of bins.  Recommanded values are between 5 and 14.
    * **bin_weights** - (*OPTIONAL AssayBinWeight*): The weights assigned to the assay bins.
    * **bin_width** - (*OPTIONAL AssayBinWidth*): The width assigned to the assay bins.
    * **provided_edges** - (*OPTIONAL AssayProvidedEdges*): The edges used for the assay bins.
    * **add_outlier_edges** - (*REQUIRED bool*): Indicates whether to add outlier edges or not.
  * **warning_threshold** - (*OPTIONAL number*): Optional warning threshold.
  * **alert_threshold** - (*REQUIRED number*): Alert threshold.
  * **run_until** - (*OPTIONAL string*): DateTime of when to end the assay.
  * **workspace_id** - (*REQUIRED integer*): The workspace the assay is part of.
  * **model_insights_url** - (*OPTIONAL string*): URL for model insights.
* **RETURNS**
  * **assay_id** - (*integer*): The id of the new assay.



```python
# Create assay

apiRequest = "/assays/create"

exampleAssayName = "api_assay_test2"

## Now get all of the assays for the pipeline in workspace 4 `housepricedrift`

exampleAssayPipelineId = 4
exampleAssayPipelineName = "housepricepipe"
exampleAssayModelName = "housepricemodel"
exampleAssayWorkspaceId = 4

# iopath can be input 00 or output 0 0
data = {
    'name': exampleAssayName,
    'pipeline_id': exampleAssayPipelineId,
    'pipeline_name': exampleAssayPipelineName,
    'active': True,
    'status': 'active',
    'iopath': "input 0 0",
    'baseline': {
        'Fixed': {
            'pipeline': exampleAssayPipelineName,
            'model': 'houseprice-model-yns',
            'start_at': '2022-01-01T00:00:00-05:00',
            'end_at': '2022-01-02T00:00:00-05:00'
        }
    },
    'window': {
        'pipeline': exampleAssayPipelineName,
        'model': exampleAssayModelName,
        'width': '24 hours',
        'start': None,
        'interval': None
    },
    'summarizer': {
        'type': 'UnivariateContinuous',
        'bin_mode': 'Quantile',
        'aggregation': 'Density',
        'metric': 'PSI',
        'num_bins': 5,
        'bin_weights': None,
        'bin_width': None,
        'provided_edges': None,
        'add_outlier_edges': True
    },
    'warning_threshold': 0,
    'alert_threshold': 0.1,
    'run_until': None,
    'workspace_id': exampleAssayWorkspaceId
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
example_assay_id = response['assay_id']
response
```




    {'assay_id': 2}



### List Assays

Lists all assays in the specified pipeline.

* **PARAMETERS**
  * **pipeline_id** - (*REQUIRED int*):  The numerical ID of the pipeline.
* **RETURNS**
  * **assays** - (*Array assays*): A list of all assays.

Example:  Display a list of all assays in a workspace.  This will assume we have a workspace with an existing Assay and the associated data has been upload.  See the tutorial [Wallaroo Assays Tutorial](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/model_insights).

For this reason, these values are hard coded for now.


```python
## First list all of the workspaces and the list of pipelines

# List workspaces

apiRequest = "/workspaces/list"

data = {
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```




    {'workspaces': [{'id': 1,
       'name': 'john.hansarick@wallaroo.ai - Default Workspace',
       'created_at': '2022-10-10T16:32:45.355874+00:00',
       'created_by': 'f68760ad-a27c-4f9b-808f-0b512f07571f',
       'archived': False,
       'models': [],
       'pipelines': []},
      {'id': 3,
       'name': 'testapiworkspace-e87e543f-25f1-4f6d-82c6-4eb48902575a',
       'created_at': '2022-10-10T18:25:27.926919+00:00',
       'created_by': 'f68760ad-a27c-4f9b-808f-0b512f07571f',
       'archived': False,
       'models': [1],
       'pipelines': [1, 2, 3]},
      {'id': 4,
       'name': 'housepricedrift',
       'created_at': '2022-10-10T18:38:50.748057+00:00',
       'created_by': 'f68760ad-a27c-4f9b-808f-0b512f07571f',
       'archived': False,
       'models': [2],
       'pipelines': [4]},
      {'id': 5,
       'name': 'housepricedrifts',
       'created_at': '2022-10-10T18:45:00.152716+00:00',
       'created_by': 'f68760ad-a27c-4f9b-808f-0b512f07571f',
       'archived': False,
       'models': [],
       'pipelines': []}]}




```python
# Get assays

apiRequest = "/assays/list"

data = {
    "pipeline_id": exampleAssayPipelineId
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```




    [{'id': 3,
      'name': 'example assay',
      'active': True,
      'status': 'created',
      'warning_threshold': None,
      'alert_threshold': 0.1,
      'pipeline_id': 4,
      'pipeline_name': 'housepricepipe',
      'last_run': None,
      'next_run': '2022-10-10T19:00:43.941894+00:00',
      'run_until': None,
      'updated_at': '2022-10-10T19:00:43.945411+00:00',
      'baseline': {'Fixed': {'pipeline': 'housepricepipe',
        'model': 'housepricemodel',
        'start_at': '2022-01-01T00:00:00+00:00',
        'end_at': '2022-01-02T00:00:00+00:00'}},
      'window': {'pipeline': 'housepricepipe',
       'model': 'housepricemodel',
       'width': '24 hours',
       'start': None,
       'interval': None},
      'summarizer': {'type': 'UnivariateContinuous',
       'bin_mode': 'Quantile',
       'aggregation': 'Density',
       'metric': 'PSI',
       'num_bins': 5,
       'bin_weights': None,
       'bin_width': None,
       'provided_edges': None,
       'add_outlier_edges': True}},
     {'id': 2,
      'name': 'api_assay_test2',
      'active': True,
      'status': 'created',
      'warning_threshold': 0.0,
      'alert_threshold': 0.1,
      'pipeline_id': 4,
      'pipeline_name': 'housepricepipe',
      'last_run': None,
      'next_run': '2022-10-10T18:53:16.444786+00:00',
      'run_until': None,
      'updated_at': '2022-10-10T18:53:16.450269+00:00',
      'baseline': {'Fixed': {'pipeline': 'housepricepipe',
        'model': 'houseprice-model-yns',
        'start_at': '2022-01-01T00:00:00-05:00',
        'end_at': '2022-01-02T00:00:00-05:00'}},
      'window': {'pipeline': 'housepricepipe',
       'model': 'housepricemodel',
       'width': '24 hours',
       'start': None,
       'interval': None},
      'summarizer': {'type': 'UnivariateContinuous',
       'bin_mode': 'Quantile',
       'aggregation': 'Density',
       'metric': 'PSI',
       'num_bins': 5,
       'bin_weights': None,
       'bin_width': None,
       'provided_edges': None,
       'add_outlier_edges': True}},
     {'id': 1,
      'name': 'api_assay_test',
      'active': True,
      'status': 'created',
      'warning_threshold': 0.0,
      'alert_threshold': 0.1,
      'pipeline_id': 4,
      'pipeline_name': 'housepricepipe',
      'last_run': None,
      'next_run': '2022-10-10T18:48:00.829479+00:00',
      'run_until': None,
      'updated_at': '2022-10-10T18:48:00.833336+00:00',
      'baseline': {'Fixed': {'pipeline': 'housepricepipe',
        'model': 'houseprice-model-yns',
        'start_at': '2022-01-01T00:00:00-05:00',
        'end_at': '2022-01-02T00:00:00-05:00'}},
      'window': {'pipeline': 'housepricepipe',
       'model': 'housepricemodel',
       'width': '24 hours',
       'start': None,
       'interval': None},
      'summarizer': {'type': 'UnivariateContinuous',
       'bin_mode': 'Quantile',
       'aggregation': 'Density',
       'metric': 'PSI',
       'num_bins': 5,
       'bin_weights': None,
       'bin_width': None,
       'provided_edges': None,
       'add_outlier_edges': True}}]



## Activate or Deactivate Assay

Activates or deactivates an existing assay.

* **Parameters**
  * **id** - (*REQUIRED int*): The numerical id of the assay.
  * **active** - (*REQUIRED bool*): True to activate the assay, False to deactivate it.
* **Returns**
  * * **id** - (*integer*): The numerical id of the assay.
  * **active** - (*bool*): True to activate the assay, False to deactivate it.

Example:  Assay 8 "House Output Assay" will be deactivated then activated.


```python
# Deactivate assay

apiRequest = "/assays/set_active"

data = {
    'id': example_assay_id,
    'active': False
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```




    {'id': 2, 'active': False}




```python
# Activate assay

apiRequest = "/assays/set_active"

data = {
    'id': example_assay_id,
    'active': True
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```




    {'id': 2, 'active': True}



### Create Interactive Baseline

Creates an interactive assay baseline.

* **PARAMETERS**
  * **id** - (*REQUIRED int*):  The numerical identifier for the assay.
  * **name** - (*REQUIRED string*): The name of the assay.
  * **pipeline_id** - (*REQUIRED int*): The numerical idenfifier the assay will be placed into.
  * **pipeline_name** - (*REQUIRED string*): The name of the pipeline
  * **active** - (*REQUIRED bool*): Indicates whether the assay will be active upon creation or not.
  * **status** - (*REQUIRED string*): The status of the assay upon creation.
  * **iopath** - (*REQUIRED string*): The iopath of the assay.
  * **baseline** - (*REQUIRED baseline*): The baseline for the assay.
    * **Fixed** - (*REQUIRED AssayFixConfiguration*): The fixed configuration for the assay.
      * **pipeline** - (*REQUIRED string*): The name of the pipeline with the baseline data.
      * **model** - (*REQUIRED string*): The name of the model used.
      * **start_at** - (*REQUIRED string*): The DateTime of the baseline start date.
      * **end_at** - (*REQUIRED string*): The DateTime of the baseline end date.
  * **window** (*REQUIRED AssayWindow*): Assay window.
    * **pipeline** - (*REQUIRED string*): The name of the pipeline for the assay window.
    * **model** - (*REQUIRED string*): The name of the model used for the assay window.
    * **width** - (*REQUIRED string*): The width of the assay window.
    * **start** - (*OPTIONAL string*): The DateTime of when to start the assay window.
    * **interval** - (*OPTIONAL string*): The assay window interval.
  * **summarizer** - (*REQUIRED AssaySummerizer*): The summarizer type for the array aka "advanced settings" in the Wallaroo Dashboard UI.
    * **type** - (*REQUIRED string*): Type of summarizer.
    * **bin_mode** - (*REQUIRED string*): The binning model type.  Values can be:
      * Quantile
      * Equal
    * **aggregation** - (*REQUIRED string*): Aggregation type.
    * **metric** - (*REQUIRED string*): Metric type.  Values can be:
      * PSI
      * Maximum Difference of Bins
      * Sum of the Difference of Bins
    * **num_bins** - (*REQUIRED int*): The number of bins.  Recommanded values are between 5 and 14.
    * **bin_weights** - (*OPTIONAL AssayBinWeight*): The weights assigned to the assay bins.
    * **bin_width** - (*OPTIONAL AssayBinWidth*): The width assigned to the assay bins.
    * **provided_edges** - (*OPTIONAL AssayProvidedEdges*): The edges used for the assay bins.
    * **add_outlier_edges** - (*REQUIRED bool*): Indicates whether to add outlier edges or not.
  * **warning_threshold** - (*OPTIONAL number*): Optional warning threshold.
  * **alert_threshold** - (*REQUIRED number*): Alert threshold.
  * **run_until** - (*OPTIONAL string*): DateTime of when to end the assay.
  * **workspace_id** - (*REQUIRED integer*): The workspace the assay is part of.
  * **model_insights_url** - (*OPTIONAL string*): URL for model insights.
* **RETURNS**
  * {} when successful.

Example:  An interactive assay baseline will be set for the assay "Test Assay" on Pipeline 4.


```python
# Run interactive baseline

apiRequest = "/assays/run_interactive_baseline"

exampleAssayPipelineId = 4
exampleAssayPipelineName = "housepricepipe"
exampleAssayModelName = "housepricemodel"
exampleAssayWorkspaceId = 4
exampleAssayId = 3
exampleAssayName = "example assay"

data = {
    'id': exampleAssayId,
    'name': exampleAssayName,
    'pipeline_id': exampleAssayPipelineId,
    'pipeline_name': exampleAssayPipelineName,
    'active': True,
    'status': 'active',
    'iopath': "input 0 0",
    'baseline': {
        'Fixed': {
            'pipeline': exampleAssayPipelineName,
            'model': exampleAssayModelName,
            'start_at': '2022-01-01T00:00:00-05:00',
            'end_at': '2022-01-02T00:00:00-05:00'
        }
    },
    'window': {
        'pipeline': exampleAssayPipelineName,
        'model': exampleAssayModelName,
        'width': '24 hours',
        'start': None,
        'interval': None
    },
    'summarizer': {
        'type': 'UnivariateContinuous',
        'bin_mode': 'Quantile',
        'aggregation': 'Density',
        'metric': 'PSI',
        'num_bins': 5,
        'bin_weights': None,
        'bin_width': None,
        'provided_edges': None,
        'add_outlier_edges': True
    },
    'warning_threshold': 0,
    'alert_threshold': 0.1,
    'run_until': None,
    'workspace_id': exampleAssayWorkspaceId
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```




    {'assay_id': 3,
     'name': 'example assay',
     'created_at': 1665428974654,
     'elapsed_millis': 3,
     'pipeline_id': 4,
     'pipeline_name': 'housepricepipe',
     'iopath': 'input 0 0',
     'baseline_summary': {'count': 1812,
      'min': -3.6163810435665233,
      'max': 7.112734553641073,
      'mean': 0.03518936967736047,
      'median': -0.39764636440424433,
      'std': 0.9885006118746916,
      'edges': [-3.6163810435665233,
       -0.39764636440424433,
       -0.39764636440424433,
       0.6752651953165153,
       0.6752651953165153,
       7.112734553641073,
       None],
      'edge_names': ['left_outlier',
       'q_20',
       'q_40',
       'q_60',
       'q_80',
       'q_100',
       'right_outlier'],
      'aggregated_values': [0.0,
       0.5739514348785872,
       0.0,
       0.3383002207505519,
       0.0,
       0.08774834437086093,
       0.0],
      'aggregation': 'Density',
      'start': '2022-01-01T05:00:00Z',
      'end': '2022-01-02T05:00:00Z'},
     'window_summary': {'count': 1812,
      'min': -3.6163810435665233,
      'max': 7.112734553641073,
      'mean': 0.03518936967736047,
      'median': -0.39764636440424433,
      'std': 0.9885006118746916,
      'edges': [-3.6163810435665233,
       -0.39764636440424433,
       -0.39764636440424433,
       0.6752651953165153,
       0.6752651953165153,
       7.112734553641073,
       None],
      'edge_names': ['left_outlier',
       'e_-3.98e-1',
       'e_-3.98e-1',
       'e_6.75e-1',
       'e_6.75e-1',
       'e_7.11e0',
       'right_outlier'],
      'aggregated_values': [0.0,
       0.5739514348785872,
       0.0,
       0.3383002207505519,
       0.0,
       0.08774834437086093,
       0.0],
      'aggregation': 'Density',
      'start': '2022-01-01T05:00:00Z',
      'end': '2022-01-02T05:00:00Z'},
     'warning_threshold': 0.0,
     'alert_threshold': 0.1,
     'score': 0.0,
     'scores': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     'index': None,
     'summarizer_meta': '{"type":"UnivariateContinuous","bin_mode":"Quantile","aggregation":"Density","metric":"PSI","num_bins":5,"bin_weights":null,"provided_edges":null}',
     'status': 'BaselineRun'}



### Get Assay Baseline

Retrieve an assay baseline.

* **Parameters**
  * **workspace_id** - (*REQUIRED integer*): Numerical id for the workspace the assay is in.
  * **pipeline_name** - (*REQUIRED string*): Name of the pipeline the assay is in.
  * **start** - (*OPTIONAL string*): DateTime for when the baseline starts.
  * **end** - (*OPTIONAL string*): DateTime for when the baseline ends.
  * **model_name** - (*OPTIONAL string*): Name of the model.
  * **limit** - (*OPTIONAL integer*): Maximum number of baselines to return.
* **Returns**
  * Assay Baseline
  
Example:  3 assay baselines for Workspace 6 and pipeline `houseprice-pipe-yns` will be retrieved.


```python
# Get Assay Baseline

apiRequest = "/assays/get_baseline"

data = {
    'workspace_id': exampleAssayWorkspaceId,
    'pipeline_name': exampleAssayPipelineName,
    'limit': 3
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```




    [{'check_failures': [],
      'elapsed': 138,
      'model_name': 'housepricemodel',
      'model_version': 'test_version',
      'original_data': {'tensor': [[0.6752651953165153,
         0.4999342471069234,
         -0.1508359058521761,
         0.20024994573167013,
         -0.08666382440547035,
         0.009116407905326388,
         -0.002872821251696453,
         -0.9179715198382244,
         -0.305653139057544,
         2.4393894526979074,
         0.29288456205300767,
         -0.3485179782510063,
         1.1121054807107582,
         0.20193559456886756,
         -0.20817781526102327,
         1.0279052268485522,
         -0.0196096612880121]]},
      'outputs': [{'Float': {'data': [13.262725830078123],
         'dim': [1, 1],
         'v': 1}}],
      'pipeline_name': 'housepricepipe',
      'time': 1643673456974},
     {'check_failures': [],
      'elapsed': 136,
      'model_name': 'housepricemodel',
      'model_version': 'test_version',
      'original_data': {'tensor': [[-0.39764636440424433,
         -1.4463372267359147,
         0.044822346031326635,
         -0.4259897870655369,
         -0.08666382440547035,
         -0.009153974747246364,
         -0.2568455220872559,
         0.005746226275241667,
         -0.305653139057544,
         -0.6285378875598833,
         -0.5584151415472702,
         -0.9748223338538442,
         -0.65605032361317,
         -1.5328599554165074,
         -0.20817781526102327,
         0.06504981348446033,
         -0.20382525042318508]]},
      'outputs': [{'Float': {'data': [12.82761001586914], 'dim': [1, 1], 'v': 1}}],
      'pipeline_name': 'housepricepipe',
      'time': 1643673504654},
     {'check_failures': [],
      'elapsed': 93,
      'model_name': 'housepricemodel',
      'model_version': 'test_version',
      'original_data': {'tensor': [[-1.470557924125004,
         -0.4732014898144956,
         1.0989221532266944,
         1.3317512811267456,
         -0.08666382440547035,
         0.006116141374609494,
         -0.21472817109954076,
         -0.9179715198382244,
         -0.305653139057544,
         -0.6285378875598833,
         0.29288456205300767,
         -0.14376463122700162,
         -0.65605032361317,
         1.1203567680905366,
         -0.20817781526102327,
         0.2692918708647222,
         -0.23870674508328787]]},
      'outputs': [{'Float': {'data': [13.03465175628662], 'dim': [1, 1], 'v': 1}}],
      'pipeline_name': 'housepricepipe',
      'time': 1643673552333}]



### Run Assay Interactively

Runs an assay.

* **Parameters**
  * **id** - (*REQUIRED int*):  The numerical identifier for the assay.
  * **name** - (*REQUIRED string*): The name of the assay.
  * **pipeline_id** - (*REQUIRED int*): The numerical idenfifier the assay will be placed into.
  * **pipeline_name** - (*REQUIRED string*): The name of the pipeline
  * **active** - (*REQUIRED bool*): Indicates whether the assay will be active upon creation or not.
  * **status** - (*REQUIRED string*): The status of the assay upon creation.
  * **iopath** - (*REQUIRED string*): The iopath of the assay.
  * **baseline** - (*REQUIRED baseline*): The baseline for the assay.
    * **Fixed** - (*REQUIRED AssayFixConfiguration*): The fixed configuration for the assay.
      * **pipeline** - (*REQUIRED string*): The name of the pipeline with the baseline data.
      * **model** - (*REQUIRED string*): The name of the model used.
      * **start_at** - (*REQUIRED string*): The DateTime of the baseline start date.
      * **end_at** - (*REQUIRED string*): The DateTime of the baseline end date.
  * **window** (*REQUIRED AssayWindow*): Assay window.
    * **pipeline** - (*REQUIRED string*): The name of the pipeline for the assay window.
    * **model** - (*REQUIRED string*): The name of the model used for the assay window.
    * **width** - (*REQUIRED string*): The width of the assay window.
    * **start** - (*OPTIONAL string*): The DateTime of when to start the assay window.
    * **interval** - (*OPTIONAL string*): The assay window interval.
  * **summarizer** - (*REQUIRED AssaySummerizer*): The summarizer type for the array aka "advanced settings" in the Wallaroo Dashboard UI.
    * **type** - (*REQUIRED string*): Type of summarizer.
    * **bin_mode** - (*REQUIRED string*): The binning model type.  Values can be:
      * Quantile
      * Equal
    * **aggregation** - (*REQUIRED string*): Aggregation type.
    * **metric** - (*REQUIRED string*): Metric type.  Values can be:
      * PSI
      * Maximum Difference of Bins
      * Sum of the Difference of Bins
    * **num_bins** - (*REQUIRED int*): The number of bins.  Recommanded values are between 5 and 14.
    * **bin_weights** - (*OPTIONAL AssayBinWeight*): The weights assigned to the assay bins.
    * **bin_width** - (*OPTIONAL AssayBinWidth*): The width assigned to the assay bins.
    * **provided_edges** - (*OPTIONAL AssayProvidedEdges*): The edges used for the assay bins.
    * **add_outlier_edges** - (*REQUIRED bool*): Indicates whether to add outlier edges or not.
  * **warning_threshold** - (*OPTIONAL number*): Optional warning threshold.
  * **alert_threshold** - (*REQUIRED number*): Alert threshold.
  * **run_until** - (*OPTIONAL string*): DateTime of when to end the assay.
  * **workspace_id** - (*REQUIRED integer*): The workspace the assay is part of.
  * **model_insights_url** - (*OPTIONAL string*): URL for model insights.
* **Returns**
  * Assay
  
Example:  An interactive assay will be run for Assay exampleAssayId exampleAssayName.  Depending on the number of assay results and the data window, this may take some time.  This returns *all* of the results for this assay at this time.  The total number of responses will be displayed after.


```python
# Run interactive assay

apiRequest = "/assays/run_interactive"

data = {
    'id': exampleAssayId,
    'name': exampleAssayName,
    'pipeline_id': exampleAssayPipelineId,
    'pipeline_name': exampleAssayPipelineName,
    'active': True,
    'status': 'active',
    'iopath': "input 0 0",
    'baseline': {
        'Fixed': {
            'pipeline': exampleAssayPipelineName,
            'model': exampleAssayModelName,
            'start_at': '2022-01-01T00:00:00-05:00',
            'end_at': '2022-01-02T00:00:00-05:00'
        }
    },
    'window': {
        'pipeline': exampleAssayPipelineName,
        'model': exampleAssayModelName,
        'width': '24 hours',
        'start': None,
        'interval': None
    },
    'summarizer': {
        'type': 'UnivariateContinuous',
        'bin_mode': 'Quantile',
        'aggregation': 'Density',
        'metric': 'PSI',
        'num_bins': 5,
        'bin_weights': None,
        'bin_width': None,
        'provided_edges': None,
        'add_outlier_edges': True
    },
    'warning_threshold': 0,
    'alert_threshold': 0.1,
    'run_until': None,
    'workspace_id': exampleAssayWorkspaceId
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response[0]
```




    {'assay_id': 3,
     'name': 'example assay',
     'created_at': 1665429281268,
     'elapsed_millis': 178,
     'pipeline_id': 4,
     'pipeline_name': 'housepricepipe',
     'iopath': 'input 0 0',
     'baseline_summary': {'count': 1812,
      'min': -3.6163810435665233,
      'max': 7.112734553641073,
      'mean': 0.03518936967736047,
      'median': -0.39764636440424433,
      'std': 0.9885006118746916,
      'edges': [-3.6163810435665233,
       -0.39764636440424433,
       -0.39764636440424433,
       0.6752651953165153,
       0.6752651953165153,
       7.112734553641073,
       None],
      'edge_names': ['left_outlier',
       'q_20',
       'q_40',
       'q_60',
       'q_80',
       'q_100',
       'right_outlier'],
      'aggregated_values': [0.0,
       0.5739514348785872,
       0.0,
       0.3383002207505519,
       0.0,
       0.08774834437086093,
       0.0],
      'aggregation': 'Density',
      'start': '2022-01-01T05:00:00Z',
      'end': '2022-01-02T05:00:00Z'},
     'window_summary': {'count': 1812,
      'min': -3.6163810435665233,
      'max': 3.8939998744787943,
      'mean': 0.006175756859303479,
      'median': -0.39764636440424433,
      'std': 0.9720429128755866,
      'edges': [-3.6163810435665233,
       -0.39764636440424433,
       -0.39764636440424433,
       0.6752651953165153,
       0.6752651953165153,
       7.112734553641073,
       None],
      'edge_names': ['left_outlier',
       'e_-3.98e-1',
       'e_-3.98e-1',
       'e_6.75e-1',
       'e_6.75e-1',
       'e_7.11e0',
       'right_outlier'],
      'aggregated_values': [0.0,
       0.5883002207505519,
       0.0,
       0.3162251655629139,
       0.0,
       0.09547461368653422,
       0.0],
      'aggregation': 'Density',
      'start': '2022-01-02T05:00:00Z',
      'end': '2022-01-03T05:00:00Z'},
     'warning_threshold': 0.0,
     'alert_threshold': 0.1,
     'score': 0.002495916218595029,
     'scores': [0.0,
      0.0003543090106786176,
      0.0,
      0.0014896074883327124,
      0.0,
      0.0006519997195836994,
      0.0],
     'index': None,
     'summarizer_meta': {'type': 'UnivariateContinuous',
      'bin_mode': 'Quantile',
      'aggregation': 'Density',
      'metric': 'PSI',
      'num_bins': 5,
      'bin_weights': None,
      'provided_edges': None},
     'status': 'Warning'}




```python
print(len(response))
```

    30


### Get Assay Results

Retrieve the results for an assay.

* **Parameters**
  * **assay_id** - (*REQUIRED integer*): Numerical id for the assay.
  * **start** - (*OPTIONAL string*): DateTime for when the baseline starts.
  * **end** - (*OPTIONAL string*): DateTime for when the baseline ends.
  * **limit** - (*OPTIONAL integer*): Maximum number of results to return.
  * **pipeline_id** - (*OPTIONAL integer*): Numerical id of the pipeline the assay is in.
* **Returns**
  * Assay Baseline
  
Example:  Results for Assay 3 "example assay" will be retrieved for January 2 to January 3.  For the sake of time, only the first record will be displayed.


```python
# Get Assay Results

apiRequest = "/assays/get_results"

data = {
    'assay_id': exampleAssayId,
    'pipeline_id': exampleAssayPipelineId
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```
