This tutorial and the assets are available as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/development/mlops_api).

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
* Wallaroo MLOps API Documentation from a Wallaroo instance:  A Swagger UI based documentation is available from your Wallaroo instance at `https://{Wallaroo Prefix}.api.{Wallaroo Suffix}/v1/api/docs`.  For example, if the Wallaroo Instance is YOUR SUFFIX with the prefix `{lovely-rhino-5555}`, then the Wallaroo MLOps API Documentation would be available at `https://lovely-rhino-5555.api.example.wallaroo.ai/v1/api/docs`.  For another example, a Wallaroo Enterprise users who do not use a prefix and has the suffix `wallaroo.example.wallaroo.ai`, the the Wallaroo MLOps API Documentation would be available at `https://api.wallaroo.example.wallaroo.ai/v1/api/docs`.  For more information, see the [Wallaroo Documentation Site](https://docs.wallaroo.ai).

**IMPORTANT NOTE**:  The Wallaroo MLOps API is provided as an early access features.  Future iterations may adjust the methods and returns to provide a better user experience.  Please refer to this guide for updates.

## Prerequisites

* An installed Wallaroo instance.
* The following Python libraries installed:
  * `requests`
  * `json`
  * [`wallaroo`](https://pypi.org/project/wallaroo/): The Wallaroo SDK. Included with the Wallaroo JupyterHub service by default.

## OpenAPI Steps

The following demonstrates how to use each command in the Wallaroo MLOps API, and can be modified as best fits your organization's needs.

### Import Libraries

For the examples, the Python `requests` library will be used to make the REST HTTP(S) connections.


```python
# Requires requests and requests-toolbelt with either:
# pip install requests-toolbelt
# conda install -c conda-forge requests-toolbelt

import requests
import json
from requests.auth import HTTPBasicAuth
```

## Retrieve Credentials

### Through Keycloak

Wallaroo comes pre-installed with a confidential OpenID Connect client.  The default client is `api-client`, but other clients may be created and configured.

As it is a confidential client, api-client requires its secret to be supplied when requesting a token. Administrators may obtain their API client credentials from Keycloak from the Keycloak Service URL as listed above and the prefix `/auth/admin/master/console/#/realms/master/clients`.

For example, if the YOUR SUFFIX instance DNS address is `https://magical-rhino-5555.example.wallaroo.ai`, then the direct path to the Keycloak API client credentials would be:

`https://magical-rhino-5555.keycloak.example.wallaroo.ai/auth/admin/master/console/#/realms/master/clients`

Then select the client, in this case **api-client**, then **Credentials**.

![Wallaroo Keycloak Service](./images/wallaroo-developer-guides/wallaroo-api/wallaroo-api-keycloak-service.png)

![Wallaroo Components](./images/wallaroo-developer-guides/wallaroo-api/wallaroo-api-keycloak-credentials.png)

By default, tokens issued for api-client are valid for up to 60 minutes. Refresh tokens are supported.

### Through the Wallaroo SDK

The API token is retrieved using the [Wallaroo SDK](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-install-guide) through the `wallaroo.client.mlops()` command. In the following example, the token will be retrieved and stored to the variable `TOKEN`:

```python
connection =wl.mlops().__dict__
TOKEN = connection['token']
print(TOKEN)
'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJnTHBSY1B6QkhjQ1k1RTFHTVZoTlQtelI0VDY2YUM0QWh2eXVORmpVOTBjIn0.eyJleHAiOjE2NzEwMzMzMzUsImlhdCI6MTY3MTAzMzI3NSwiYXV0aF90aW1lIjoxNjcxMDMyODgyLCJqdGkiOiJiNDk3YmM3Yy1kMTc5LTRhYWQtODdmZC0yZGJiYTBlZDI4ZDYiLCJpc3MiOiJodHRwczovL21hZ2ljYWwtYmVhci0zNzgyLmtleWNsb2FrLndhbGxhcm9vLmNvbW11bml0eS9hdXRoL3JlYWxtcy9tYXN0ZXIiLCJhdWQiOlsibWFzdGVyLXJlYWxtIiwiYWNjb3VudCJdLCJzdWIiOiJmMWYzMmJkZi05YmQ5LTQ1OTUtYTUzMS1hY2E1Nzc4Y2VhZjAiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJzZGstY2xpZW50Iiwic2Vzc2lvbl9zdGF0ZSI6IjYzYzNiZjYwLTNmNjMtNDBjNC05NmI1LWNiYTk4ZjZhOGNmNyIsImFjciI6IjEiLCJyZWFsbV9hY2Nlc3MiOnsicm9sZXMiOlsiY3JlYXRlLXJlYWxtIiwiZGVmYXVsdC1yb2xlcy1tYXN0ZXIiLCJvZmZsaW5lX2FjY2VzcyIsImFkbWluIiwidW1hX2F1dGhvcml6YXRpb24iXX0sInJlc291cmNlX2FjY2VzcyI6eyJtYXN0ZXItcmVhbG0iOnsicm9sZXMiOlsidmlldy1pZGVudGl0eS1wcm92aWRlcnMiLCJ2aWV3LXJlYWxtIiwibWFuYWdlLWlkZW50aXR5LXByb3ZpZGVycyIsImltcGVyc29uYXRpb24iLCJjcmVhdGUtY2xpZW50IiwibWFuYWdlLXVzZXJzIiwicXVlcnktcmVhbG1zIiwidmlldy1hdXRob3JpemF0aW9uIiwicXVlcnktY2xpZW50cyIsInF1ZXJ5LXVzZXJzIiwibWFuYWdlLWV2ZW50cyIsIm1hbmFnZS1yZWFsbSIsInZpZXctZXZlbnRzIiwidmlldy11c2VycyIsInZpZXctY2xpZW50cyIsIm1hbmFnZS1hdXRob3JpemF0aW9uIiwibWFuYWdlLWNsaWVudHMiLCJxdWVyeS1ncm91cHMiXX0sImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJtYW5hZ2UtYWNjb3VudC1saW5rcyIsInZpZXctcHJvZmlsZSJdfX0sInNjb3BlIjoiZW1haWwgcHJvZmlsZSIsInNpZCI6IjYzYzNiZjYwLTNmNjMtNDBjNC05NmI1LWNiYTk4ZjZhOGNmNyIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJodHRwczovL2hhc3VyYS5pby9qd3QvY2xhaW1zIjp7IngtaGFzdXJhLXVzZXItaWQiOiJmMWYzMmJkZi05YmQ5LTQ1OTUtYTUzMS1hY2E1Nzc4Y2VhZjAiLCJ4LWhhc3VyYS1kZWZhdWx0LXJvbGUiOiJ1c2VyIiwieC1oYXN1cmEtYWxsb3dlZC1yb2xlcyI6WyJ1c2VyIl0sIngtaGFzdXJhLXVzZXItZ3JvdXBzIjoie30ifSwicHJlZmVycmVkX3VzZXJuYW1lIjoiam9obi5oYW5zYXJpY2tAd2FsbGFyb28uYWkiLCJlbWFpbCI6ImpvaG4uaGFuc2FyaWNrQHdhbGxhcm9vLmFpIn0.EEt9UK1jxvO1DYg_hiy1ne4s9iK8mJtqbVfE7MPQfMRYhzXqDU4gFpP3Nwzlo0iW9fSLDiCxPg303Rz-l4it3oPFu5SaS1S8pQpqvtMAJqy8V_CNPp5H5ggQFYm4Z50aAPdPzOOOkVQOZUhupRsEeUERvK1-eFqtG1bb-IUV6DpQO_XaRVcQbIVubFi48C0_im5Tb3i4WFCNA_1pRrEBKFbZLWgzSCu8fglBQ27mODqfmRQVbTeXLjxsQX5O8meErSfibEGmsJKQytGCJ3NYdnXfal3YhWEqp6A4dG0tkoRW1eD-aKBpsHf9nKKzxcSsjeXDQF6iQAONCGmC40oqHQ'
```

### Set Variables

The following variables are used for the example and should be modified to fit your organization.


```python
## Variables

URLPREFIX='YOURPREFIX'
URLSUFFIX='YOURSUFFIX'
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




    'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJKaDZyX3BKUGhheVhSSlc0U1F3ckc0QUEwUmkyMHNBMTNxYmNhTVJ1d1hrIn0.eyJleHAiOjE2Nzc3ODk4MzEsImlhdCI6MTY3Nzc4NjIzMSwianRpIjoiMjg1MTU1NmItZjhkNC00OWZkLWJjMjEtOGFlNDI2OTJiM2FiIiwiaXNzIjoiaHR0cHM6Ly9kb2MtdGVzdC5rZXljbG9hay53YWxsYXJvb2NvbW11bml0eS5uaW5qYS9hdXRoL3JlYWxtcy9tYXN0ZXIiLCJhdWQiOlsibWFzdGVyLXJlYWxtIiwiYWNjb3VudCJdLCJzdWIiOiJjYTdkNzA0My04ZTk0LTQyZDUtOWYzYS04ZjU1YzJlNDI4MTQiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJhcGktY2xpZW50Iiwic2Vzc2lvbl9zdGF0ZSI6IjFiNTEyOTZiLTMwNjAtNGUwYy1hZDMwLTNhYjczYmNiMDYzNyIsImFjciI6IjEiLCJhbGxvd2VkLW9yaWdpbnMiOlsiKiJdLCJyZWFsbV9hY2Nlc3MiOnsicm9sZXMiOlsiZGVmYXVsdC1yb2xlcy1tYXN0ZXIiLCJvZmZsaW5lX2FjY2VzcyIsInVtYV9hdXRob3JpemF0aW9uIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsibWFzdGVyLXJlYWxtIjp7InJvbGVzIjpbIm1hbmFnZS11c2VycyIsInZpZXctdXNlcnMiLCJxdWVyeS1ncm91cHMiLCJxdWVyeS11c2VycyJdfSwiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwic2NvcGUiOiJwcm9maWxlIGVtYWlsIiwic2lkIjoiMWI1MTI5NmItMzA2MC00ZTBjLWFkMzAtM2FiNzNiY2IwNjM3IiwiZW1haWxfdmVyaWZpZWQiOmZhbHNlLCJodHRwczovL2hhc3VyYS5pby9qd3QvY2xhaW1zIjp7IngtaGFzdXJhLXVzZXItaWQiOiJjYTdkNzA0My04ZTk0LTQyZDUtOWYzYS04ZjU1YzJlNDI4MTQiLCJ4LWhhc3VyYS1kZWZhdWx0LXJvbGUiOiJ1c2VyIiwieC1oYXN1cmEtYWxsb3dlZC1yb2xlcyI6WyJ1c2VyIl0sIngtaGFzdXJhLXVzZXItZ3JvdXBzIjoie30ifSwibmFtZSI6IkpvaG4gSGFuc2FyaWNrIiwicHJlZmVycmVkX3VzZXJuYW1lIjoiam9obi5odW1tZWxAd2FsbGFyb28uYWkiLCJnaXZlbl9uYW1lIjoiSm9obiIsImZhbWlseV9uYW1lIjoiSGFuc2FyaWNrIiwiZW1haWwiOiJqb2huLmh1bW1lbEB3YWxsYXJvby5haSJ9.Qxhsu1lbhWpVZyUjKLqsr47j-ybjVB28jEXPcyb8m4NlzYDSfWHH2Wc7i1RMLV4IUe4td8ujPQJjkan2zatoHhSNqWYwEziwgFwIcP-uYqDcBhIIkNIu3Shw8f9FxAt3UtEc0twTXNED4ak2cfTs9nNwF2v_ZRcKMsrWObAfm2Iuly2tKuu6TlK_3Nbi6DTip4rXTO5AavIhjqKZn7ofuJ-NhOBh9s9gZPIZpWQ-klk-zeM7mzzulD8THBTCITvEpmMSJf9qI24-QXQWhpRFEpmUh8gy6GkQs1lEcjvt8NzLP5mf9L7fmgQZCgvETLwuA9dmp7BPYS_G3pamDGqDoA'



## Users

### Get Users

Users can be retrieved either by their Keycloak user id, or return all users if an empty set `{}` is submitted.

* **Parameters**
  * `{}`: Empty set, returns all users.
  * **user_ids** *Array[Keycloak user ids]*: An array of Keycloak user ids, typically in UUID format.

Example:  The first example will submit an empty set `{}` to return all users, then submit the first user's user id and request only that user's details.


```python
# Get all users

TOKEN=get_jwt_token(TOKENURL, CLIENT, SECRET, USERNAME, PASSWORD)

apiRequest = "/users/query"
data = {
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```




    {'users': {'9727c4a8-d6fc-4aad-894f-9ee69801d2dd': {'access': {'manageGroupMembership': True,
        'impersonate': False,
        'view': True,
        'mapRoles': True,
        'manage': True},
       'createdTimestamp': 1677704075554,
       'disableableCredentialTypes': [],
       'emailVerified': False,
       'enabled': True,
       'id': '9727c4a8-d6fc-4aad-894f-9ee69801d2dd',
       'notBefore': 0,
       'requiredActions': [],
       'username': 'admin'},
      'ca7d7043-8e94-42d5-9f3a-8f55c2e42814': {'access': {'impersonate': False,
        'manage': True,
        'mapRoles': True,
        'manageGroupMembership': True,
        'view': True},
       'createdTimestamp': 1677704179667,
       'disableableCredentialTypes': [],
       'email': 'john.hummel@wallaroo.ai',
       'emailVerified': False,
       'enabled': True,
       'firstName': 'John',
       'id': 'ca7d7043-8e94-42d5-9f3a-8f55c2e42814',
       'lastName': 'Hansarick',
       'notBefore': 0,
       'requiredActions': [],
       'username': 'john.hummel@wallaroo.ai'}}}




```python
# Get first user Keycloak id
TOKEN=get_jwt_token(TOKENURL, CLIENT, SECRET, USERNAME, PASSWORD)
firstUserKeycloak = list(response['users'])[1]

apiRequest = "/users/query"
data = {
  "user_ids": [
    firstUserKeycloak
  ]
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```




    {'users': {'ca7d7043-8e94-42d5-9f3a-8f55c2e42814': {'access': {'impersonate': False,
        'view': True,
        'mapRoles': True,
        'manage': True,
        'manageGroupMembership': True},
       'createdTimestamp': 1677704179667,
       'disableableCredentialTypes': [],
       'email': 'john.hummel@wallaroo.ai',
       'emailVerified': False,
       'enabled': True,
       'federatedIdentities': [{'identityProvider': 'google',
         'userId': '117610299312093432527',
         'userName': 'john.hummel@wallaroo.ai'}],
       'firstName': 'John',
       'id': 'ca7d7043-8e94-42d5-9f3a-8f55c2e42814',
       'lastName': 'Hansarick',
       'notBefore': 0,
       'requiredActions': [],
       'username': 'john.hummel@wallaroo.ai'}}}



### Invite Users

**IMPORTANT NOTE**:  This command is for YOUR SUFFIX only.  For more details on user management, see [Wallaroo User Management](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-user-management/).

Users are invited through `/users/invite`.  When using YOUR SUFFIX, this will send an invitation email to the email address listed.  Note that the user must not already be a member of the Wallaroo instance, and email addresses must be unique.  If the email address is already in use for another user, the request will generate an error.

* **Parameters**
  * **email** *(REQUIRED string): The email address of the new user to invite.
  * **password** *(OPTIONAL string)*: The assigned password of the new user to invite.  If not provided, the Wallaroo instance will provide the new user a temporary password that must be changed upon initial login.

Example:  In this example, a new user will be invited to the Wallaroo instance and assigned a password.


```python
# invite users
TOKEN=get_jwt_token(TOKENURL, CLIENT, SECRET, USERNAME, PASSWORD)
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
TOKEN=get_jwt_token(TOKENURL, CLIENT, SECRET, USERNAME, PASSWORD)
apiRequest = "/users/deactivate"

data = {
    "email": newUser
}
```


```python
response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```

### Activate User

A deactivated user can be reactivated to allow them access to their Wallaroo instance.  Activated users count against the Wallaroo license count.

* **Parameters**
  * **email** (*REQUIRED string*):  The email address of the user to activate.

Example:  In this example, the `newUser` will be activated.


```python
# Activate users
TOKEN=get_jwt_token(TOKENURL, CLIENT, SECRET, USERNAME, PASSWORD)
apiRequest = "/users/activate"

data = {
    "email": newUser
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```

## Workspaces

### List Workspaces

List the workspaces for a specific user.

* **Parameters**
  * **user_id** - (*OPTIONAL string*): The Keycloak ID.
  
Example:  In this example, the workspaces for the a specific user will be displayed, then workspaces for all users will be displayed.


```python
# List workspaces by user id
TOKEN=get_jwt_token(TOKENURL, CLIENT, SECRET, USERNAME, PASSWORD)
apiRequest = "/workspaces/list"

data = {
    "user_id":firstUserKeycloak
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```




    {'workspaces': [{'id': 1,
       'name': 'john.hummel@wallaroo.ai - Default Workspace',
       'created_at': '2023-03-01T20:56:22.658436+00:00',
       'created_by': 'ca7d7043-8e94-42d5-9f3a-8f55c2e42814',
       'archived': False,
       'models': [],
       'pipelines': []},
      {'id': 4,
       'name': 'anomalyexampletest3',
       'created_at': '2023-03-01T20:56:32.632146+00:00',
       'created_by': 'ca7d7043-8e94-42d5-9f3a-8f55c2e42814',
       'archived': False,
       'models': [1],
       'pipelines': [1]},
      {'id': 5,
       'name': 'ccfraudcomparisondemo',
       'created_at': '2023-03-01T21:02:40.955593+00:00',
       'created_by': 'ca7d7043-8e94-42d5-9f3a-8f55c2e42814',
       'archived': False,
       'models': [2, 3, 4],
       'pipelines': [3]},
      {'id': 6,
       'name': 'rlhxccfraudworkspace',
       'created_at': '2023-03-01T21:30:28.848609+00:00',
       'created_by': 'ca7d7043-8e94-42d5-9f3a-8f55c2e42814',
       'archived': False,
       'models': [5],
       'pipelines': [5]},
      {'id': 7,
       'name': 'mlflowstatsmodelworkspace',
       'created_at': '2023-03-02T18:06:42.074341+00:00',
       'created_by': 'ca7d7043-8e94-42d5-9f3a-8f55c2e42814',
       'archived': False,
       'models': [6, 7],
       'pipelines': [8]},
      {'id': 8,
       'name': 'mobilenetworkspace',
       'created_at': '2023-03-02T18:24:27.304478+00:00',
       'created_by': 'ca7d7043-8e94-42d5-9f3a-8f55c2e42814',
       'archived': False,
       'models': [8, 9],
       'pipelines': [10]},
      {'id': 9,
       'name': 'mobilenetworkspacetest',
       'created_at': '2023-03-02T19:21:36.309503+00:00',
       'created_by': 'ca7d7043-8e94-42d5-9f3a-8f55c2e42814',
       'archived': False,
       'models': [10, 12],
       'pipelines': [13]},
      {'id': 10,
       'name': 'resnetworkspace',
       'created_at': '2023-03-02T19:22:28.371499+00:00',
       'created_by': 'ca7d7043-8e94-42d5-9f3a-8f55c2e42814',
       'archived': False,
       'models': [11],
       'pipelines': [14]},
      {'id': 11,
       'name': 'resnetworkspacetest',
       'created_at': '2023-03-02T19:35:30.236438+00:00',
       'created_by': 'ca7d7043-8e94-42d5-9f3a-8f55c2e42814',
       'archived': False,
       'models': [13],
       'pipelines': [18]},
      {'id': 12,
       'name': 'shadowimageworkspacetest',
       'created_at': '2023-03-02T19:37:23.348346+00:00',
       'created_by': 'ca7d7043-8e94-42d5-9f3a-8f55c2e42814',
       'archived': False,
       'models': [14, 15],
       'pipelines': [20]}]}




```python
# List workspaces
TOKEN=get_jwt_token(TOKENURL, CLIENT, SECRET, USERNAME, PASSWORD)
apiRequest = "/workspaces/list"

data = {
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```




    {'workspaces': [{'id': 1,
       'name': 'john.hummel@wallaroo.ai - Default Workspace',
       'created_at': '2023-03-01T20:56:22.658436+00:00',
       'created_by': 'ca7d7043-8e94-42d5-9f3a-8f55c2e42814',
       'archived': False,
       'models': [],
       'pipelines': []},
      {'id': 4,
       'name': 'anomalyexampletest3',
       'created_at': '2023-03-01T20:56:32.632146+00:00',
       'created_by': 'ca7d7043-8e94-42d5-9f3a-8f55c2e42814',
       'archived': False,
       'models': [1],
       'pipelines': [1]},
      {'id': 5,
       'name': 'ccfraudcomparisondemo',
       'created_at': '2023-03-01T21:02:40.955593+00:00',
       'created_by': 'ca7d7043-8e94-42d5-9f3a-8f55c2e42814',
       'archived': False,
       'models': [2, 3, 4],
       'pipelines': [3]},
      {'id': 6,
       'name': 'rlhxccfraudworkspace',
       'created_at': '2023-03-01T21:30:28.848609+00:00',
       'created_by': 'ca7d7043-8e94-42d5-9f3a-8f55c2e42814',
       'archived': False,
       'models': [5],
       'pipelines': [5]},
      {'id': 7,
       'name': 'mlflowstatsmodelworkspace',
       'created_at': '2023-03-02T18:06:42.074341+00:00',
       'created_by': 'ca7d7043-8e94-42d5-9f3a-8f55c2e42814',
       'archived': False,
       'models': [6, 7],
       'pipelines': [8]},
      {'id': 8,
       'name': 'mobilenetworkspace',
       'created_at': '2023-03-02T18:24:27.304478+00:00',
       'created_by': 'ca7d7043-8e94-42d5-9f3a-8f55c2e42814',
       'archived': False,
       'models': [8, 9],
       'pipelines': [10]},
      {'id': 9,
       'name': 'mobilenetworkspacetest',
       'created_at': '2023-03-02T19:21:36.309503+00:00',
       'created_by': 'ca7d7043-8e94-42d5-9f3a-8f55c2e42814',
       'archived': False,
       'models': [10, 12],
       'pipelines': [13]},
      {'id': 10,
       'name': 'resnetworkspace',
       'created_at': '2023-03-02T19:22:28.371499+00:00',
       'created_by': 'ca7d7043-8e94-42d5-9f3a-8f55c2e42814',
       'archived': False,
       'models': [11],
       'pipelines': [14]},
      {'id': 11,
       'name': 'resnetworkspacetest',
       'created_at': '2023-03-02T19:35:30.236438+00:00',
       'created_by': 'ca7d7043-8e94-42d5-9f3a-8f55c2e42814',
       'archived': False,
       'models': [13],
       'pipelines': [18]},
      {'id': 12,
       'name': 'shadowimageworkspacetest',
       'created_at': '2023-03-02T19:37:23.348346+00:00',
       'created_by': 'ca7d7043-8e94-42d5-9f3a-8f55c2e42814',
       'archived': False,
       'models': [14, 15],
       'pipelines': [20]}]}



### Create Workspace

A new workspace will be created in the Wallaroo instance.  Upon creating, the workspace owner will be assigned as the user making the MLOps API request.

* **Parameters**:
  * **workspace_name** - (*REQUIRED string*):  The name of the new workspace.
* **Returns**:
  * **workspace_id** - (*int*):  The ID of the new workspace.
  
Example:  In this example, a workspace with the name `testapiworkspace` will be created, and the newly created workspace's `workspace_id` saved as the variable `exampleWorkspaceId` for use in other code examples.  After the request is complete, the [List Workspaces](#list-workspaces) command will be issued to demonstrate the new workspace has been created.


```python
# Create workspace
TOKEN=get_jwt_token(TOKENURL, CLIENT, SECRET, USERNAME, PASSWORD)
apiRequest = "/workspaces/create"

exampleWorkspaceName = "testapiworkspace"

data = {
  "workspace_name": exampleWorkspaceName
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
# Stored for future examples
exampleWorkspaceId = response['workspace_id']
response
```




    {'workspace_id': 13}




```python
# List workspaces
TOKEN=get_jwt_token(TOKENURL, CLIENT, SECRET, USERNAME, PASSWORD)
apiRequest = "/workspaces/list"

data = {
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```




    {'workspaces': [{'id': 1,
       'name': 'john.hummel@wallaroo.ai - Default Workspace',
       'created_at': '2023-03-01T20:56:22.658436+00:00',
       'created_by': 'ca7d7043-8e94-42d5-9f3a-8f55c2e42814',
       'archived': False,
       'models': [],
       'pipelines': []},
      {'id': 4,
       'name': 'anomalyexampletest3',
       'created_at': '2023-03-01T20:56:32.632146+00:00',
       'created_by': 'ca7d7043-8e94-42d5-9f3a-8f55c2e42814',
       'archived': False,
       'models': [1],
       'pipelines': [1]},
      {'id': 5,
       'name': 'ccfraudcomparisondemo',
       'created_at': '2023-03-01T21:02:40.955593+00:00',
       'created_by': 'ca7d7043-8e94-42d5-9f3a-8f55c2e42814',
       'archived': False,
       'models': [2, 3, 4],
       'pipelines': [3]},
      {'id': 6,
       'name': 'rlhxccfraudworkspace',
       'created_at': '2023-03-01T21:30:28.848609+00:00',
       'created_by': 'ca7d7043-8e94-42d5-9f3a-8f55c2e42814',
       'archived': False,
       'models': [5],
       'pipelines': [5]},
      {'id': 7,
       'name': 'mlflowstatsmodelworkspace',
       'created_at': '2023-03-02T18:06:42.074341+00:00',
       'created_by': 'ca7d7043-8e94-42d5-9f3a-8f55c2e42814',
       'archived': False,
       'models': [6, 7],
       'pipelines': [8]},
      {'id': 8,
       'name': 'mobilenetworkspace',
       'created_at': '2023-03-02T18:24:27.304478+00:00',
       'created_by': 'ca7d7043-8e94-42d5-9f3a-8f55c2e42814',
       'archived': False,
       'models': [8, 9],
       'pipelines': [10]},
      {'id': 9,
       'name': 'mobilenetworkspacetest',
       'created_at': '2023-03-02T19:21:36.309503+00:00',
       'created_by': 'ca7d7043-8e94-42d5-9f3a-8f55c2e42814',
       'archived': False,
       'models': [10, 12],
       'pipelines': [13]},
      {'id': 10,
       'name': 'resnetworkspace',
       'created_at': '2023-03-02T19:22:28.371499+00:00',
       'created_by': 'ca7d7043-8e94-42d5-9f3a-8f55c2e42814',
       'archived': False,
       'models': [11],
       'pipelines': [14]},
      {'id': 11,
       'name': 'resnetworkspacetest',
       'created_at': '2023-03-02T19:35:30.236438+00:00',
       'created_by': 'ca7d7043-8e94-42d5-9f3a-8f55c2e42814',
       'archived': False,
       'models': [13],
       'pipelines': [18]},
      {'id': 12,
       'name': 'shadowimageworkspacetest',
       'created_at': '2023-03-02T19:37:23.348346+00:00',
       'created_by': 'ca7d7043-8e94-42d5-9f3a-8f55c2e42814',
       'archived': False,
       'models': [14, 15],
       'pipelines': [20]},
      {'id': 13,
       'name': 'testapiworkspace',
       'created_at': '2023-03-02T19:44:20.279346+00:00',
       'created_by': 'ca7d7043-8e94-42d5-9f3a-8f55c2e42814',
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
TOKEN=get_jwt_token(TOKENURL, CLIENT, SECRET, USERNAME, PASSWORD)
apiRequest = "/workspaces/add_user"

data = {
  "email":newUser,
  "workspace_id": exampleWorkspaceId
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```

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
TOKEN=get_jwt_token(TOKENURL, CLIENT, SECRET, USERNAME, PASSWORD)
apiRequest = "/workspaces/list_users"

data = {
  "workspace_id": exampleWorkspaceId
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```




    {'users': [{'user_id': 'ca7d7043-8e94-42d5-9f3a-8f55c2e42814',
       'user_type': 'OWNER'}]}



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
TOKEN=get_jwt_token(TOKENURL, CLIENT, SECRET, USERNAME, PASSWORD)
apiRequest = "/workspaces/remove_user"

data = {
  "email":newUser,
  "workspace_id": exampleWorkspaceId
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```


```python
# List users in a workspace
TOKEN=get_jwt_token(TOKENURL, CLIENT, SECRET, USERNAME, PASSWORD)
apiRequest = "/workspaces/list_users"

data = {
  "workspace_id": exampleWorkspaceId
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```




    {'users': [{'user_id': '5abe33ef-90d2-49bd-8f6a-21ef20c383e8',
       'user_type': 'OWNER'}]}



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
TOKEN=get_jwt_token(TOKENURL, CLIENT, SECRET, USERNAME, PASSWORD)
apiRequest = "/models/upload"

exampleModelName = "apitestmodel"

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




    {'insert_models': {'returning': [{'models': [{'id': 16}]}]}}




```python
exampleModelId=response['insert_models']['returning'][0]['models'][0]['id']
exampleModelId
```




    16



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
TOKEN=get_jwt_token(TOKENURL, CLIENT, SECRET, USERNAME, PASSWORD)
apiRequest = "/models/upload_stream"
exampleModelName = "apitestmodelstream"
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




    {'insert_models': {'returning': [{'models': [{'id': 17}]}]}}



### List Models in Workspace

Returns a list of models added to a specific workspace.

* **Parameters**
  * **workspace_id** - (*REQUIRED int*): The workspace id to list.
  
Example:  Display the models for the workspace used in the Upload Model to Workspace step.  The model id and model name will be saved as `exampleModelId` and `exampleModelName` variables for other examples.


```python
# List models in a workspace
TOKEN=get_jwt_token(TOKENURL, CLIENT, SECRET, USERNAME, PASSWORD)
apiRequest = "/models/list"

data = {
  "workspace_id": exampleWorkspaceId
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```




    {'models': [{'id': 17,
       'name': 'apitestmodelstream',
       'owner_id': '""',
       'created_at': '2023-03-02T19:44:56.549555+00:00',
       'updated_at': '2023-03-02T19:44:56.549555+00:00'},
      {'id': 16,
       'name': 'apitestmodel',
       'owner_id': '""',
       'created_at': '2023-03-02T19:44:53.173913+00:00',
       'updated_at': '2023-03-02T19:44:53.173913+00:00'}]}




```python
exampleModelId = response['models'][0]['id']
exampleModelName = response['models'][0]['name']
```

### Get Model Details By ID

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
TOKEN=get_jwt_token(TOKENURL, CLIENT, SECRET, USERNAME, PASSWORD)
apiRequest = "/models/get_by_id"

data = {
  "id": exampleModelId
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```




    {'id': 17,
     'owner_id': '""',
     'workspace_id': 13,
     'name': 'apitestmodelstream',
     'updated_at': '2023-03-02T19:44:56.549555+00:00',
     'created_at': '2023-03-02T19:44:56.549555+00:00',
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
TOKEN=get_jwt_token(TOKENURL, CLIENT, SECRET, USERNAME, PASSWORD)
apiRequest = "/models/list_versions"

data = {
  "model_id": exampleModelName,
  "models_pk_id": exampleModelId
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```




    [{'sha': 'bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507',
      'models_pk_id': 17,
      'model_version': '0396de99-0a55-4880-9f53-8fdcd1b3357a',
      'owner_id': '""',
      'model_id': 'apitestmodelstream',
      'id': 17,
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
TOKEN=get_jwt_token(TOKENURL, CLIENT, SECRET, USERNAME, PASSWORD)
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
TOKEN=get_jwt_token(TOKENURL, CLIENT, SECRET, USERNAME, PASSWORD)
apiRequest = "/models/get"

data = {
  "id": exampleModelId
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```




    {'id': 17,
     'name': 'apitestmodelstream',
     'owner_id': '""',
     'created_at': '2023-03-02T19:44:56.549555+00:00',
     'updated_at': '2023-03-02T19:44:56.549555+00:00',
     'models': [{'sha': 'bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507',
       'models_pk_id': 17,
       'model_version': '0396de99-0a55-4880-9f53-8fdcd1b3357a',
       'owner_id': '""',
       'model_id': 'apitestmodelstream',
       'id': 17,
       'file_name': 'streamfile.onnx',
       'image_path': None}]}



## Pipeline Management

Pipelines are managed through the Wallaroo API or the Wallaroo SDK.  Pipelines are the vehicle used for deploying, serving, and monitoring ML models.  For more information, see the [Wallaroo Glossary](https://docs.wallaroo.ai/wallaroo-glossary/).

### Create Pipeline in a Workspace

Creates a new pipeline in the specified workspace.

* **Parameters**
  * **pipeline_id** - (REQUIRED string): Name of the new pipeline.
  * **workspace_id** - (REQUIRED int): Numerical id of the workspace for the new pipeline.
  * **definition** - (REQUIRED string): Pipeline definitions, can be `{}` for none.

Example:  Two pipelines are created in the workspace created in the step Create Workspace.  One will be an empty pipeline without any models, the other will be created using the uploaded models in the Upload Model to Workspace step and no configuration details.  The pipeline id, variant id, and variant version of each pipeline will be stored for later examples.


```python
# Create pipeline in a workspace
TOKEN=get_jwt_token(TOKENURL, CLIENT, SECRET, USERNAME, PASSWORD)
apiRequest = "/pipelines/create"

exampleEmptyPipelineName="emptypipeline"

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




    {'pipeline_pk_id': 22,
     'pipeline_variant_pk_id': 22,
     'pipeline_variant_version': 'e4c3a3dc-97ee-4020-88ce-3bf059b772ef'}




```python
# Create pipeline in a workspace with models
TOKEN=get_jwt_token(TOKENURL, CLIENT, SECRET, USERNAME, PASSWORD)
apiRequest = "/pipelines/create"

exampleModelPipelineName="pipelinewithmodel"

data = {
  "pipeline_id": exampleModelPipelineName,
  "workspace_id": exampleWorkspaceId,
  "definition": {}
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
exampleModelPipelineId = response['pipeline_pk_id']
exampleModelPipelineVariantId=response['pipeline_variant_pk_id']
emptyModelPipelineVariantVersion=['pipeline_variant_version']
response
```




    {'pipeline_pk_id': 25,
     'pipeline_variant_pk_id': 25,
     'pipeline_variant_version': '8eaa146e-1bfb-4786-9969-4264877db7d2'}



### Deploy a Pipeline

Deploy a an existing pipeline.  Note that for any pipeline that has model steps, they must be included either in `model_configs`, `model_ids` or `models`.

* **Parameters**
  * **deploy_id** (*REQUIRED string*): The name for the pipeline deployment.
  * **engine_config** (*OPTIONAL string*): Additional configuration options for the pipeline.
  * **pipeline_version_pk_id** (*REQUIRED int*): Pipeline version id.
  * **model_configs** (*OPTIONAL Array int*): Ids of model configs to apply.
  * **model_ids** (*OPTIONAL Array int*): Ids of models to apply to the pipeline.  If passed in, model_configs will be created automatically.
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
TOKEN=get_jwt_token(TOKENURL, CLIENT, SECRET, USERNAME, PASSWORD)
apiRequest = "/pipelines/deploy"

exampleEmptyDeployId = "emptydeploy"

data = {
    "deploy_id": exampleEmptyDeployId,
    "pipeline_version_pk_id": exampleEmptyPipelineVariantId,
    "pipeline_id": exampleEmptyPipelineId
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
exampleEmptyDeploymentId=response['id']
response

```




    {'id': 14}




```python
# Deploy a pipeline with models
TOKEN=get_jwt_token(TOKENURL, CLIENT, SECRET, USERNAME, PASSWORD)
apiRequest = "/pipelines/deploy"
exampleModelDeployId="modeldeploy"

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




    {'id': 17}



### Get Deployment Status

Returns the deployment status.

* **Parameters**
  * **name** - (REQUIRED string): The deployment in the format {deployment_name}-{deploymnent-id}.
  
Example: The deployed empty and model pipelines status will be displayed.


```python
# Get empty pipeline deployment
TOKEN=get_jwt_token(TOKENURL, CLIENT, SECRET, USERNAME, PASSWORD)
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
       'name': 'engine-7c44d857cb-995p7',
       'status': 'Pending',
       'reason': None,
       'details': ['containers with unready status: [engine]',
        'containers with unready status: [engine]'],
       'pipeline_statuses': None,
       'model_statuses': None}],
     'engine_lbs': [{'ip': '10.244.12.53',
       'name': 'engine-lb-ddd995646-vjz7f',
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




    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.13.27',
       'name': 'engine-7df9567698-m7zdx',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'pipelinewithmodel',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'apitestmodelstream',
          'version': '0396de99-0a55-4880-9f53-8fdcd1b3357a',
          'sha': 'bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.12.55',
       'name': 'engine-lb-ddd995646-qk6tj',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}



### Get External Inference URL

The API command `/admin/get_pipeline_external_url` retrieves the external inference URL for a specific pipeline in a workspace.

* **Parameters**
  * **workspace_id** (*REQUIRED integer*):  The workspace integer id.
  * **pipeline_name** (*REQUIRED string*): The name of the deployment.

In this example, a list of the workspaces will be retrieved.  Based on the setup from the Internal Pipeline Deployment URL Tutorial, the workspace matching `urlworkspace` will have it's **workspace id** stored and used for the `/admin/get_pipeline_external_url` request with the pipeline `urlpipeline`.

The External Inference URL will be stored as a variable for the next step.


```python
## Retrieve the pipeline's External Inference URL
TOKEN=get_jwt_token(TOKENURL, CLIENT, SECRET, USERNAME, PASSWORD)

apiRequest = "/admin/get_pipeline_external_url"

data = {
    "workspace_id": exampleWorkspaceId,
    "pipeline_name": exampleModelDeployId
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
print(response)
externalUrl = response['url']
externalUrl
```

    {'url': 'https://wallaroo.api.example.com/v1/api/pipelines/infer/modeldeploy-15'}





    'https://wallaroo.api.example.com/v1/api/pipelines/infer/modeldeploy-15'



### Perform Inference Through External URL

The inference can now be performed through the External Inference URL.  This URL will accept the same inference data file that is used with the Wallaroo SDK, or with an Internal Inference URL as used in the Internal Pipeline Inference URL Tutorial.

Deployed pipelines have their own Inference URL that accepts HTTP POST submissions.

For connections that are external to the Kubernetes cluster hosting the Wallaroo instance, [model endpoints must be enabled](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-model-endpoints-guide/).

## HTTP Headers

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


```python
TOKEN=get_jwt_token(TOKENURL, CLIENT, SECRET, USERNAME, PASSWORD)

dataFile = './data/cc_data_1k.df.json'
data = json.load(open('./data/cc_data_1k.df.json','rb'))
contentType="application/json; format=pandas-records"

# set the headers
headers= {
    'Authorization': 'Bearer ' + TOKEN,
    'Content-Type': contentType
}

# submit the request via POST
response = requests.post(externalUrl, json=data, headers=headers)

# Only the first 300 characters will be displayed for brevity
printResponse = json.dumps(response.json())
print(printResponse[0:300])

```

    [{"time": 1677788050393, "in": {"tensor": [-1.0603297501, 2.3544967095, -3.5638788326, 5.1387348926, -1.2308457019, -0.7687824608, -3.5881228109, 1.8880837663, -3.2789674274, -3.9563254554, 4.0993439118, -5.6539176395, -0.8775733373, -9.131571192, -0.6093537873, -3.7480276773, -5.0309125017, -0.8748


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
TOKEN=get_jwt_token(TOKENURL, CLIENT, SECRET, USERNAME, PASSWORD)
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
TOKEN=get_jwt_token(TOKENURL, CLIENT, SECRET, USERNAME, PASSWORD)
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
TOKEN=get_jwt_token(TOKENURL, CLIENT, SECRET, USERNAME, PASSWORD)
apiRequest = "/pipelines/copy"

exampleCopiedPipelineName="copiedmodelpipeline"

data = {
  "name": exampleCopiedPipelineName,
  "workspace_id": exampleWorkspaceId,
  "source_pipeline": exampleModelPipelineId
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```




    {'pipeline_pk_id': 26,
     'pipeline_variant_pk_id': 26,
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
TOKEN=get_jwt_token(TOKENURL, CLIENT, SECRET, USERNAME, PASSWORD)
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

**IMPORTANT NOTE**: These assays were run in a Wallaroo environment with canned historical data.  See the [Wallaroo Assay Tutorial](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/model_insights) for details on setting up this environment.  This historical data is **required** for these examples.

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

As noted this example requires the [Wallaroo Assay Tutorial](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/model_insights) for historical data.  Before running this example, set the sample pipeline id, pipeline, name, model name, and workspace id in the code sample below.  For more information on retrieving this information, see the [Wallaroo Developer Guides](https://docs.wallaroo.ai/wallaroo-developer-guides/).


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


```python
# Get assays

apiRequest = "/assays/list"

data = {
    "pipeline_id": exampleAssayPipelineId
}

response = get_wallaroo_response(APIURL, apiRequest, TOKEN, data)
response
```

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


```python
print(len(response))
```

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
