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
  
Example:  In this example, a workspace with the name `testapiworkspace-` with a randomly generated UUID will be created, and the newly created workspace's `workspace_id` saved as the variable `exampleWorkspaceId` for use in other code examples.  After the request is complete, the [List Workspaces](#list-workspaces) command will be issued to demonstrate the new workspace has been created.

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

    {'workspace_id': 618489}

```python
# List workspaces

apiRequest = "/workspaces/list"

data = {
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```

    {'workspaces': [{'id': 15,
       'name': 'john.hansarick@wallaroo.ai - Default Workspace',
       'created_at': '2022-12-16T20:23:23.150058+00:00',
       'created_by': '01a797f9-1357-4506-a4d2-8ab9c4681103',
       'archived': False,
       'models': [],
       'pipelines': []},
      {'id': 28,
       'name': 'alohaworkspace',
       'created_at': '2022-12-16T21:00:01.614796+00:00',
       'created_by': '01a797f9-1357-4506-a4d2-8ab9c4681103',
       'archived': False,
       'models': [2],
       'pipelines': [4]},
      {'id': 29,
       'name': 'abtestworkspace',
       'created_at': '2022-12-16T21:03:08.785538+00:00',
       'created_by': '01a797f9-1357-4506-a4d2-8ab9c4681103',
       'archived': False,
       'models': [3, 5, 4, 6],
       'pipelines': [6]},
      {'id': 618487,
       'name': 'sdkquickworkspace',
       'created_at': '2022-12-20T15:56:22.088161+00:00',
       'created_by': '01a797f9-1357-4506-a4d2-8ab9c4681103',
       'archived': False,
       'models': [48],
       'pipelines': [76]},
      {'id': 618489,
       'name': 'testapiworkspace-e9e386a7-8146-4ead-b4c6-a2580af70083',
       'created_at': '2022-12-20T19:34:30.392835+00:00',
       'created_by': '01a797f9-1357-4506-a4d2-8ab9c4681103',
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

