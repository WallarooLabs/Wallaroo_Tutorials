This tutorial and the assets are available as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/development/mlops_api).

## Wallaroo MLOps API User Management Tutorial

This tutorial focuses on using the Wallaroo MLOps API for user management.  For this tutorial, we will be using the Wallaroo SDK to provide authentication credentials for ease of use examples.  See the [Wallaroo API Guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/) for full details on using the Wallaroo MLOps API.

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

    {'Authorization': 'Bearer eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJKNXZOUVVIajVNa3lOV2pLUkwtUGZZSXJ2S3Z5YUx3eThJZFB2dktrZnRnIn0.eyJleHAiOjE3MDExODk4MTIsImlhdCI6MTcwMTE4OTc1MiwiYXV0aF90aW1lIjoxNzAxMTg5NDEyLCJqdGkiOiJiMGQzMDI3ZC00MTVkLTRjNjYtOWMzNy1jNzJjNDM5ODNjMzUiLCJpc3MiOiJodHRwczovL2RvYy10ZXN0LmtleWNsb2FrLndhbGxhcm9vY29tbXVuaXR5Lm5pbmphL2F1dGgvcmVhbG1zL21hc3RlciIsImF1ZCI6WyJtYXN0ZXItcmVhbG0iLCJhY2NvdW50Il0sInN1YiI6IjEyZWEwOWQxLTBmNDktNDA1ZS1iZWQxLTI3ZWI2ZDEwZmRlNCIsInR5cCI6IkJlYXJlciIsImF6cCI6InNkay1jbGllbnQiLCJzZXNzaW9uX3N0YXRlIjoiOTBhMmVmYzMtM2JlNy00M2FmLTkyZTItYWI3MDIzYWYyODA2IiwiYWNyIjoiMSIsInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJkZWZhdWx0LXJvbGVzLW1hc3RlciIsIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iXX0sInJlc291cmNlX2FjY2VzcyI6eyJtYXN0ZXItcmVhbG0iOnsicm9sZXMiOlsibWFuYWdlLXVzZXJzIiwidmlldy11c2VycyIsInF1ZXJ5LWdyb3VwcyIsInF1ZXJ5LXVzZXJzIl19LCJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6InByb2ZpbGUgZW1haWwiLCJzaWQiOiI5MGEyZWZjMy0zYmU3LTQzYWYtOTJlMi1hYjcwMjNhZjI4MDYiLCJlbWFpbF92ZXJpZmllZCI6ZmFsc2UsImh0dHBzOi8vaGFzdXJhLmlvL2p3dC9jbGFpbXMiOnsieC1oYXN1cmEtdXNlci1pZCI6IjEyZWEwOWQxLTBmNDktNDA1ZS1iZWQxLTI3ZWI2ZDEwZmRlNCIsIngtaGFzdXJhLWRlZmF1bHQtcm9sZSI6InVzZXIiLCJ4LWhhc3VyYS1hbGxvd2VkLXJvbGVzIjpbInVzZXIiXSwieC1oYXN1cmEtdXNlci1ncm91cHMiOiJ7fSJ9LCJuYW1lIjoiSm9obiBIYW5zYXJpY2siLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJqb2huLmh1bW1lbEB3YWxsYXJvby5haSIsImdpdmVuX25hbWUiOiJKb2huIiwiZmFtaWx5X25hbWUiOiJIYW5zYXJpY2siLCJlbWFpbCI6ImpvaG4uaHVtbWVsQHdhbGxhcm9vLmFpIn0.K5sVjfogVTelgmm2zQyNO9h3Z6eRFvaZWWFqLuJtbU45ROMZbY8kJD6Cf5S0hDgXhfbgSMFnO4IpycBcMzFDrkmox7-6wB8sp2gA6c8di5clboB92pYpJvu8kNKc6kdk13kXUTKBk1YRziHKQ-uOOLY5MszNd3XJShroJlKjB3Ms1dPa6XA60dyTi2mL_31plp4Xnrjpf0bTjTwYXUOnpVYNolM30bsALj_5w-efNsHdwj6jDxELtKgfambPdcZlBq7YFU2HI701Tf7hCBJjJ7zPLNg5CZk3Kt7Egsr3sN0h_6WcDrS2Grz3sVys33MMyIkNYGX-JYV9TbLOtpewFA'}

## Users

### Get Users

* **Endpoint**: `/v1/api/users/query`

Users are retrieved either by their Keycloak user id, or return all users if an empty set `{}` is submitted.

#### Get Users Parameters

| Field | Type | Description |
|---|---|---|
| **user_ids** | *List[Keycloak user ids]* (*Optional*) | An array of Keycloak user ids, typically in UUID format. |

If an empty set `{}` is submitted as a parameter, then all users are returned.

#### Get Users Returns

Full details are available from the [Keycloak UserRepresentation site](https://www.keycloak.org/docs-api/21.1.1/javadocs/org/keycloak/representations/idm/UserRepresentation.html).  The following represents the most relevant values.

| Field | &nbsp; | &nbsp; | Type | Description |
|---|---|---|---|---|
| **users** | | | *List[user]* | A list of users and their information with the Keycloak ID as the primary key. |
| &nbsp; | **`{keycloak id}`** | | *user* | User details. |
| &nbsp; | &nbsp; | **createdTimeTamp** | *Integer* | The Unix Epoc Timestamp of when the user was created. |
| &nbsp; | &nbsp; | **email** | *String* | The user's email address. |
| &nbsp; | &nbsp; | **enabled** | *Boolean* | Whether the user is verified or not. |
| &nbsp; | &nbsp; | **firstName** | *String* | The user's first name. |
| &nbsp; | &nbsp; | **lastName** | *String* | The user's last name. |
| &nbsp; | &nbsp; | **id** | *String* | The user's keycloak id in UUID format. |
| &nbsp; | &nbsp; | **username** | *String* | The user's username as an email address. |

#### Get Users Examples

The first example will submit an empty set `{}` to return all users, then submit the first user's user id and request only that user's details.

```python
# Get All Users via Requests

# Retrieve the token 
headers = wl.auth.auth_header()

endpoint = f"{wl.api_endpoint}/v1/api/users/query"
data = {
}

response = requests.post(endpoint, 
                         json=data, 
                         headers=headers, 
                         verify=True).json()
response
```

    {'users': {'12ea09d1-0f49-405e-bed1-27eb6d10fde4': {'access': {'view': True,
        'manage': True,
        'manageGroupMembership': True,
        'mapRoles': True,
        'impersonate': False},
       'createdTimestamp': 1700496282637,
       'disableableCredentialTypes': [],
       'email': 'john.hummel@wallaroo.ai',
       'emailVerified': False,
       'enabled': True,
       'firstName': 'John',
       'id': '12ea09d1-0f49-405e-bed1-27eb6d10fde4',
       'lastName': 'Hansarick',
       'notBefore': 0,
       'requiredActions': [],
       'username': 'john.hummel@wallaroo.ai'},
      '57d61aed-3058-4327-9e65-a5d39a9718c0': {'access': {'view': True,
        'manage': True,
        'impersonate': False,
        'manageGroupMembership': True,
        'mapRoles': True},
       'createdTimestamp': 1701202186276,
       'disableableCredentialTypes': [],
       'email': 'john.hansarick@wallaroo.ai',
       'emailVerified': True,
       'enabled': True,
       'firstName': 'John',
       'id': '57d61aed-3058-4327-9e65-a5d39a9718c0',
       'lastName': 'Hansarick',
       'notBefore': 0,
       'requiredActions': [],
       'username': 'john.hansarick@wallaroo.ai'},
      '1bc88c6d-b476-4e4d-aa1a-a5a3554591d3': {'access': {'view': True,
        'mapRoles': True,
        'manage': True,
        'manageGroupMembership': True,
        'impersonate': False},
       'createdTimestamp': 1700495081278,
       'disableableCredentialTypes': [],
       'emailVerified': False,
       'enabled': True,
       'id': '1bc88c6d-b476-4e4d-aa1a-a5a3554591d3',
       'notBefore': 0,
       'requiredActions': [],
       'username': 'admin'}}}

```python
# Get All Users via curl

!curl {wl.api_endpoint}/v1/api/users/query \
    -H "Authorization: {wl.auth.auth_header()['Authorization']}" \
    -H "Content-Type: application/json" \
    --data '{{}}'
```

    {"users":{"57d61aed-3058-4327-9e65-a5d39a9718c0":{"access":{"manage":true,"impersonate":false,"view":true,"manageGroupMembership":true,"mapRoles":true},"createdTimestamp":1701202186276,"disableableCredentialTypes":[],"email":"john.hansarick@wallaroo.ai","emailVerified":true,"enabled":true,"firstName":"John","id":"57d61aed-3058-4327-9e65-a5d39a9718c0","lastName":"Hansarick","notBefore":0,"requiredActions":[],"username":"john.hansarick@wallaroo.ai"},"12ea09d1-0f49-405e-bed1-27eb6d10fde4":{"access":{"impersonate":false,"manageGroupMembership":true,"view":true,"manage":true,"mapRoles":true},"createdTimestamp":1700496282637,"disableableCredentialTypes":[],"email":"john.hummel@wallaroo.ai","emailVerified":false,"enabled":true,"firstName":"John","id":"12ea09d1-0f49-405e-bed1-27eb6d10fde4","lastName":"Hansarick","notBefore":0,"requiredActions":[],"username":"john.hummel@wallaroo.ai"},"1bc88c6d-b476-4e4d-aa1a-a5a3554591d3":{"access":{"manage":true,"manageGroupMembership":true,"mapRoles":true,"view":true,"impersonate":false},"createdTimestamp":1700495081278,"disableableCredentialTypes":[],"emailVerified":false,"enabled":true,"id":"1bc88c6d-b476-4e4d-aa1a-a5a3554591d3","notBefore":0,"requiredActions":[],"username":"admin"}}}

```python
# Get first user via Keycloak ID

# Retrieve the token 
headers = wl.auth.auth_header()

endpoint = f"{wl.api_endpoint}/v1/api/users/query"

# retrieved from the previous request - get the 2nd user since the first will always be `admin`
first_user_keycloak = list(response['users'])[1]

data = {
  "user_ids": [
    first_user_keycloak
  ]
}

user_response = requests.post(endpoint, 
                         json=data, 
                         headers=headers, 
                         verify=True).json()
user_response
```

    {'user_ids': ['57d61aed-3058-4327-9e65-a5d39a9718c0']}

    {'users': {'57d61aed-3058-4327-9e65-a5d39a9718c0': {'access': {'manageGroupMembership': True,
        'mapRoles': True,
        'manage': True,
        'impersonate': False,
        'view': True},
       'createdTimestamp': 1701202186276,
       'disableableCredentialTypes': [],
       'email': 'john.hansarick@wallaroo.ai',
       'emailVerified': True,
       'enabled': True,
       'federatedIdentities': [],
       'firstName': 'John',
       'id': '57d61aed-3058-4327-9e65-a5d39a9718c0',
       'lastName': 'Hansarick',
       'notBefore': 0,
       'requiredActions': [],
       'username': 'john.hansarick@wallaroo.ai'}}}

```python
# Get first user via curl

!curl {wl.api_endpoint}/v1/api/users/query \
    -H "Authorization: {wl.auth.auth_header()['Authorization']}" \
    -H "Content-Type: application/json" \
    --data '{json.dumps(data)}'
```

    {"users":{"57d61aed-3058-4327-9e65-a5d39a9718c0":{"access":{"manageGroupMembership":true,"mapRoles":true,"impersonate":false,"view":true,"manage":true},"createdTimestamp":1701202186276,"disableableCredentialTypes":[],"email":"john.hansarick@wallaroo.ai","emailVerified":true,"enabled":true,"federatedIdentities":[],"firstName":"John","id":"57d61aed-3058-4327-9e65-a5d39a9718c0","lastName":"Hansarick","notBefore":0,"requiredActions":[],"username":"john.hansarick@wallaroo.ai"}}}

### Invite Users

* **Endpoint**:  `/v1/api/users/invite`

**IMPORTANT NOTE**:  This command is for Wallaroo Community only.  For more details on user management, see [Wallaroo User Management](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-user-management/).

Users are invited through `/users/invite`.  When using Wallaroo Community, this will send an invitation email to the email address listed.  Note that the user must not already be a member of the Wallaroo instance, and email addresses must be unique.  If the email address is already in use for another user, the request will generate an error.

#### Invite Users Parameters

| Field | Type | Description |
|---|---|---|
| **email** | *String* (*Required*) | The email address of the new user to invite.|
| **password** | *String* (*Optional*) | The assigned password of the new user to invite.  If not provided, the Wallaroo instance will provide the new user a temporary password that must be changed upon initial login. |

#### Invite Users Returns

| Field | Type | Description |
|---|---|---|
| **id** | *String* | The email address of the new user to invite.|
| **password** | *String* | The assigned password of the new user. |

#### Invite Users Examples

Example:  In this example, a new user will be invited to the Wallaroo instance and assigned a password.

```python
# invite users

# 
headers = wl.auth.auth_header()

endpoint = f"{wl.api_endpoint}/v1/api/users/invite"

data = {
    "email": "example.person@wallaroo.ai",
    "password":"Example-Password"
}

response = requests.post(endpoint, 
                         json=data, 
                         headers=headers, 
                         verify=True).json()
response
```

```python
!curl {wl.api_endpoint}/v1/api/users/invite \
    -H "Authorization: {wl.auth.auth_header()['Authorization']}" \
    -H "Content-Type: application/json" \
    --data '{{"email": "example.person@wallaroo.ai","password":"Example-Password"}}'
```

### Deactivate User

* **Endpoint**: `/v1/api/users/deactivate`

Users can be deactivated so they can not login to their Wallaroo instance.  Deactivated users do not count against the Wallaroo license count.

#### Deactivate User Parameters

| Field | Type | Description |
|---|---|---|
| **email** | *String* (*Required*) | The email address user to deactivate.|

#### Deactivate User Returns

`{}` on a successful request.

### Deactivate User Examples

Example:  In this example, a user will be deactivated.

```python
## Deactivate users

# Retrieve the token 
headers = wl.auth.auth_header()

endpoint = f"{wl.api_endpoint}/v1/api/users/deactivate"

data = {
    "email": "john.hansarick@wallaroo.ai"
}

response = requests.post(endpoint, 
                         json=data, 
                         headers=headers, 
                         verify=True).json()
response
```

    {}

```python
!curl {wl.api_endpoint}/v1/api/users/deactivate \
    -H "Authorization: {wl.auth.auth_header()['Authorization']}" \
    -H "Content-Type: application/json" \
    --data '{{"email": "john.hansarick@wallaroo.ai"}}'
```

    {}

### Activate User

* **Endpoint**: `/v1/api/users/activate`

A deactivated user can be reactivated to allow them access to their Wallaroo instance.  Activated users count against the Wallaroo license count.

#### Activate User Parameters

| Field | Type | Description |
|---|---|---|
| **email** | *String* (*Required*) | The email address user to activate.|

#### Activate User Returns

`{}` on a successful request.

#### Activate User Examples

In this example, the user `john.hansarick@wallaroo.ai` will be activated.

```python
## Activate users

# Retrieve the token 
headers = wl.auth.auth_header()
endpoint = f"{wl.api_endpoint}/v1/api/users/activate"

data = {
    "email": "john.hansarick@wallaroo.ai"
}

response = requests.post(endpoint, 
                         json=data, 
                         headers=headers, 
                         verify=True).json()
response
```

    {}

```python
!curl {wl.api_endpoint}/v1/api/users/activate \
    -H "Authorization: {wl.auth.auth_header()['Authorization']}" \
    -H "Content-Type: application/json" \
    --data '{{"email": "john.hansarick@wallaroo.ai"}}'
```

    {}
