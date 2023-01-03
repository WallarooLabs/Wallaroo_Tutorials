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
* Wallaroo MLOps API Documentation from a Wallaroo instance:  A Swagger UI based documentation is available from your Wallaroo instance at `https://{Wallaroo Prefix}.api.{Wallaroo Suffix}/v1/api/docs`.  For example, if the Wallaroo Instance is Wallaroo Community with the prefix `{lovely-rhino-5555}`, then the Wallaroo MLOps API Documentation would be available at `https://lovely-rhino-5555.api.example.wallaroo.ai/v1/api/docs`.  For another example, a Wallaroo Enterprise users who do not use a prefix and has the suffix `wallaroo.example.wallaroo.ai`, the the Wallaroo MLOps API Documentation would be available at `https://api.wallaroo.example.wallaroo.ai/v1/api/docs`.  For more information, see the [Wallaroo Documentation Site](https://docs.wallaroo.ai).

**IMPORTANT NOTE**:  The Wallaroo MLOps API is provided as an early access features.  Future iterations may adjust the methods and returns to provide a better user experience.  Please refer to this guide for updates.

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

## Retrieve Credentials

### Through Keycloak

Wallaroo comes pre-installed with a confidential OpenID Connect client.  The default client is `api-client`, but other clients may be created and configured.

As it is a confidential client, api-client requires its secret to be supplied when requesting a token. Administrators may obtain their API client credentials from Keycloak from the Keycloak Service URL as listed above and the prefix `/auth/admin/master/console/#/realms/master/clients`.

For example, if the Wallaroo Community instance DNS address is `https://magical-rhino-5555.example.wallaroo.ai`, then the direct path to the Keycloak API client credentials would be:

`https://magical-rhino-5555.keycloak.example.wallaroo.ai/auth/admin/master/console/#/realms/master/clients`

Then select the client, in this case **api-client**, then **Credentials**.

![Wallaroo Keycloak Service](/images/wallaroo-developer-guides/wallaroo-api/wallaroo-api-keycloak-service.png)

![Wallaroo Components](/images/wallaroo-developer-guides/wallaroo-api/wallaroo-api-keycloak-credentials.png)

By default, tokens issued for api-client are valid for up to 60 minutes. Refresh tokens are supported.

### Through the Wallaroo SDK

The API token can be retrieved using the [Wallaroo SDK](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-install-guide) through the `wallaroo.client.mlops()` command. In the following example, the token will be retrieved and stored to the variable `TOKEN`:

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

URLPREFIX='YOUR PREFIX'
URLSUFFIX='YOUR SUFFIX'
SECRET="YOUR SECRET"
TOKENURL=f'https://{URLPREFIX}.keycloak.{URLSUFFIX}/auth/realms/master/protocol/openid-connect/token'
CLIENT="api-client"
USERNAME="YOUR USERNAME"
PASSWORD="YOUR PASSWORD"
APIURL=f"https://{URLPREFIX}.api.{URLSUFFIX}/v1/api"
newUser="NEW USER EMAIL"
newPassword="NEW USER PASSWORD"
```

The following is an output of the `TOKENURL` variable to verify it matches your Wallaroo instance's Keycloak API client credentials URL.

```python
TOKENURL
```

    'https://YOUR PREFIX.keycloak.YOUR SUFFIX/auth/realms/master/protocol/openid-connect/token'

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

    'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJWalFITFhMMThub3BXNWVHM2hMOVJ5MDZ1SFVWMko1dHREUkVxSGtBT2VzIn0.eyJleHAiOjE2NzE1NjY2MzYsImlhdCI6MTY3MTU2MzAzNiwianRpIjoiNTdjZTkzMmYtYjZmYS00MGFkLThhNzMtMWMyMzE4OTAwNWJjIiwiaXNzIjoiaHR0cHM6Ly9zcXVpc2h5LXdhbGxhcm9vLTYxODcua2V5Y2xvYWsud2FsbGFyb28uZGV2L2F1dGgvcmVhbG1zL21hc3RlciIsImF1ZCI6WyJtYXN0ZXItcmVhbG0iLCJhY2NvdW50Il0sInN1YiI6IjAxYTc5N2Y5LTEzNTctNDUwNi1hNGQyLThhYjljNDY4MTEwMyIsInR5cCI6IkJlYXJlciIsImF6cCI6ImFwaS1jbGllbnQiLCJzZXNzaW9uX3N0YXRlIjoiZDZlYzdkODUtNWExMC00ZDJjLWE3NTUtYjI4YzI1NjJlNDYyIiwiYWNyIjoiMSIsImFsbG93ZWQtb3JpZ2lucyI6WyIqIl0sInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJjcmVhdGUtcmVhbG0iLCJkZWZhdWx0LXJvbGVzLW1hc3RlciIsIm9mZmxpbmVfYWNjZXNzIiwiYWRtaW4iLCJ1bWFfYXV0aG9yaXphdGlvbiJdfSwicmVzb3VyY2VfYWNjZXNzIjp7Im1hc3Rlci1yZWFsbSI6eyJyb2xlcyI6WyJ2aWV3LWlkZW50aXR5LXByb3ZpZGVycyIsInZpZXctcmVhbG0iLCJtYW5hZ2UtaWRlbnRpdHktcHJvdmlkZXJzIiwiaW1wZXJzb25hdGlvbiIsImNyZWF0ZS1jbGllbnQiLCJtYW5hZ2UtdXNlcnMiLCJxdWVyeS1yZWFsbXMiLCJ2aWV3LWF1dGhvcml6YXRpb24iLCJxdWVyeS1jbGllbnRzIiwicXVlcnktdXNlcnMiLCJtYW5hZ2UtZXZlbnRzIiwibWFuYWdlLXJlYWxtIiwidmlldy1ldmVudHMiLCJ2aWV3LXVzZXJzIiwidmlldy1jbGllbnRzIiwibWFuYWdlLWF1dGhvcml6YXRpb24iLCJtYW5hZ2UtY2xpZW50cyIsInF1ZXJ5LWdyb3VwcyJdfSwiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwic2NvcGUiOiJwcm9maWxlIGVtYWlsIiwic2lkIjoiZDZlYzdkODUtNWExMC00ZDJjLWE3NTUtYjI4YzI1NjJlNDYyIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsImh0dHBzOi8vaGFzdXJhLmlvL2p3dC9jbGFpbXMiOnsieC1oYXN1cmEtdXNlci1pZCI6IjAxYTc5N2Y5LTEzNTctNDUwNi1hNGQyLThhYjljNDY4MTEwMyIsIngtaGFzdXJhLWRlZmF1bHQtcm9sZSI6InVzZXIiLCJ4LWhhc3VyYS1hbGxvd2VkLXJvbGVzIjpbInVzZXIiXSwieC1oYXN1cmEtdXNlci1ncm91cHMiOiJ7fSJ9LCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJqb2huLmhhbnNhcmlja0B3YWxsYXJvby5haSIsImVtYWlsIjoiam9obi5oYW5zYXJpY2tAd2FsbGFyb28uYWkifQ.Au0X-XRMaWQVMVBnVt19rFt6xqtImkWrA5Cbc0hFSG4msLRJPjq8OtulxcxKU_SbXP3hNGAkHCPG1-nnh3l45IQlhhBqdBQViyHnggnRVt_B-MVKMauvV1oCtFVLKFDkY11EVr1GCdFOeFLmY99imccQr7J99yF82bNmp3XUrDHHy7BOJ2Dn2NJJC3yJCoyo3wLwPwF2yD0O3Hpj5e5_zVLHABPO97eybX3CZUanoVL8nTrumgfBHCG1I6RzX4PhRTxp-nbTf9ArX33qeMoZrPPYrn9ZryCgbiSjdlnacCmAnwVGpgXbwCU2sxEGvn4bGhWqFMkXm1aZ_bNNadRxBw'

