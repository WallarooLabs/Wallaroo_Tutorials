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
* Wallaroo MLOps API Documentation from a Wallaroo instance:  A Swagger UI based documentation is available from your Wallaroo instance at `https://{Wallaroo Prefix.}api.{Wallaroo Suffix}/v1/api/docs`.  For example, if the Wallaroo Instance suffix is `example.wallaroo.ai` with the prefix `{lovely-rhino-5555.}`, then the Wallaroo MLOps API Documentation would be available at `https://lovely-rhino-5555.api.example.wallaroo.ai/v1/api/docs`.  Note the `.` is part of the prefix.
* For another example, a Wallaroo Enterprise users who do not use a prefix and has the suffix `wallaroo.example.wallaroo.ai`, the the Wallaroo MLOps API Documentation would be available at `https://api.wallaroo.example.wallaroo.ai/v1/api/docs`.  For more information, see the [Wallaroo Documentation Site](https://docs.wallaroo.ai).

**IMPORTANT NOTE**:  The Wallaroo MLOps API is provided as an early access features.  Future iterations may adjust the methods and returns to provide a better user experience.  Please refer to this guide for updates.

## Prerequisites

* An installed Wallaroo instance.
* The following Python libraries installed:
  * `requests`
  * `json`
  * [`wallaroo`](https://pypi.org/project/wallaroo/): The Wallaroo SDK. Included with the Wallaroo JupyterHub service by default.
  * [`pandas`](https://pypi.org/project/pandas/): Pandas, mainly used for Pandas DataFrame.  Included with the Wallaroo JupyterHub service by default.
  * [`pyarrow`](https://pypi.org/project/pyarrow/): PyArrow for Apache Arrow support.  Included with the Wallaroo JupyterHub service by default.
  * [`polars`](https://pypi.org/project/polars/): Polars for DataFrame with native Apache Arrow support

## OpenAPI Steps

The following demonstrates how to use each command in the Wallaroo MLOps API, and can be modified as best fits your organization's needs.

### Import Libraries

For the examples, the Python `requests` library will be used to make the REST HTTP(S) connections.

```python
import wallaroo
from wallaroo.object import EntityNotFoundError
import pandas as pd
import os

import pyarrow as pa

import requests
from requests.auth import HTTPBasicAuth

import json

# used to display dataframe information without truncating
from IPython.display import display
pd.set_option('display.max_colwidth', None)
```

## Notes About This Guide

The following guide was established with set names for workspaces, pipelines, and models.  Note that some commands, such as creating a workspace, will fail **if another workspace is already created with the same name**.  Similar, if a user is already established with the same email address as in the examples below, etc.

To reduce errors, the following variables are declared.  Please change them as required to avoid issues in an established Wallaroo environment.

For `wallarooPrefix = "YOUR PREFIX."` and `wallarooSuffix = "YOUR SUFFIX"`, enter the prefix and suffix for your Wallaroo instance DNS name.  If the prefix instance is blank, then it can be `wallarooPrefix = ""`.  **Note that the prefix includes the `.` for proper formatting.**

```python
## Sample Variables List

new_user = "john.hansarick@wallaroo.ai"
new_user_password = "Snugglebunnies"

example_workspace_name = "apiworkspaces"
model_name = "apimodel"
model_file_name = "./models/ccfraud.onnx"

stream_model_name = "apiteststreammodel"
stream_model_file_name = "./models/ccfraud.onnx"

empty_pipeline_name="pipelinenomodel"

model_pipeline_name="pipelinemodels"

example_copied_pipeline_name="copiedmodelpipeline"

wallarooPrefix = "YOUR PREFIX."
wallarooSuffix = "YOUR SUFFIX"

# Retrieving login data through credential file
f = open('./creds.json')
login_data = json.load(f)
```

## Retrieve Credentials

### Through Keycloak

Wallaroo comes pre-installed with a confidential OpenID Connect client.  The default client is `api-client`, but other clients may be created and configured.

Confidential clients require its secret to be supplied when requesting a token. Administrators may obtain their API client credentials from Keycloak from the Keycloak Service URL as listed above and the prefix `/auth/admin/master/console/#/realms/master/clients`.

For example, if the Wallaroo DNS address is in the format `https://{WALLAROO PREFIX.}{WALLAROO SUFFIX}`, then the direct path to the Keycloak API client credentials would be:

`https://{WALLAROO PREFIX.}keycloak.{WALLAROO SUFFIX}/auth/admin/master/console/#/realms/master/clients`

If the there is no prefix, then the address would simply be:

`https://keycloak.{WALLAROO SUFFIX}/auth/admin/master/console/#/realms/master/clients`

Then select the client, in this case **api-client**, then **Credentials**.

{{<figure src="/images/2023.3.0/wallaroo-developer-guides/wallaroo-api/wallaroo-api-keycloak-service.png" width="800" label="Wallaroo Keycloak Service">}}

{{<figure src="/images/2023.3.0/wallaroo-developer-guides/wallaroo-api/wallaroo-api-keycloak-credentials.png" width="800" label="Wallaroo Components">}}

By default, tokens issued for api-client are valid for up to 60 minutes. Refresh tokens are supported.

### Token Types

There are two tokens used with Wallaroo API services:

* MLOps tokens:  User tokens are generated with the confidential client credentials and the username/password of the Wallaroo user making the MLOps API request and requires:
  * The Wallaroo instance Keycloak address.
  * The confidential client, `api-client` by default.
  * The confidential client secret.
  * The Wallaroo username making the MLOps API request.
  * The Wallaroo user's password.

    This request return includes the `access_token` and the `refresh_token`.  The `access_token` is used to authenticate.  The `refresh_token` can be used to create a new token without submitting the original username and password.

    A sample `curl` version of that request is:

    ```bash
    eval $(curl "https://${URL_PREFIX}keycloak.${URL_SUFFIX}/auth/realms/master/protocol/openid-connect/token" -u "${CONFIDENTIAL_CLIENT}:${CONFIDENTIAL_CLIENT_SECRET}" -d "grant_type=password&username=${USERNAME}&password=${PASSWORD}&scope=offline_access' -s | jq -r '"TOKEN=\(.access_token) REFRESH=\(.refresh_token)"')
    ```

    * Tokens can be refreshed via a refresh request and require:
      * The confidential client, `api-client` by default.
      * The confidential client secret.
      * The refresh token retrieved from the initial access token request.  A `curl` version of that request is:

        ```bash
        TOKEN=$(curl "https://${URL_PREFIX}keycloak.${URL_SUFFIX}/auth/realms/master/protocol/openid-connect/token" -u "${CONFIDENTIAL_CLIENT}:${CONFIDENTIAL_CLIENT_SECRET}" -d "grant_type=refresh_token&refresh_token=${REFRESH}" -s | jq -r '.access_token')
        ```

* Inference Token:  Tokens used as part of a Pipeline Inference URL request.  These do **not** require a Wallaroo user credentials.  Inference token request require the following:
  * The Wallaroo instance Keycloak address.
  * The confidential client, `api-client` by default.
  * The confidential client secret.

    A `curl` version of that command is:

    ```bash
    TOKEN=$(curl "https://${URL_PREFIX}keycloak.${URL_SUFFIX}/auth/realms/master/protocol/openid-connect/token" -u "${CONFIDENTIAL_CLIENT}:${CONFIDENTIAL_CLIENT_SECRET}" -d 'grant_type=client_credentials' -s | jq -r '.access_token')
    ```

The following examples demonstrate:

* Generating a MLOps API token with the confidential client, client secret, username, and password.
* Refreshing a MLOps API token with the confidential client and client secret (the username and password are not required for refreshing the token).
* Generate a Pipeline Inference URl token with the confidential client and client secret (username and password are not required).

The username and password for the user are stored in the file `./creds.json` to prevent them from being displayed in a demonstration.

```python
## Generating token with confidential client, client secret, username, password

TOKENURL=f'https://{wallarooPrefix}keycloak.{wallarooSuffix}/auth/realms/master/protocol/openid-connect/token'

USERNAME = login_data["username"]
PASSWORD = login_data["password"]
CONFIDENTIAL_CLIENT=login_data["confidentialClient"]
CONFIDENTIAL_CLIENT_SECRET=login_data["confidentialPassword"]

auth = HTTPBasicAuth(CONFIDENTIAL_CLIENT, CONFIDENTIAL_CLIENT_SECRET)
data = {
    'grant_type': 'password',
    'username': USERNAME,
    'password': PASSWORD
}
response = requests.post(TOKENURL, auth=auth, data=data, verify=True)
access_token = response.json()['access_token']
refresh_token = response.json()['refresh_token']
display(access_token)
```

    'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJDYkFqN19QY0xCWTFkWmJiUDZ6Q3BsbkNBYTd6US0tRHlyNy0yLXlQb25nIn0.eyJleHAiOjE2ODQzNjAxNjUsImlhdCI6MTY4NDM1NjU2NSwianRpIjoiZGQxMDFkODMtMzk5ZC00N2M2LThlZDMtNjQxMGRmNThhYmViIiwiaXNzIjoiaHR0cHM6Ly9kb2MtdGVzdC5rZXljbG9hay53YWxsYXJvb2NvbW11bml0eS5uaW5qYS9hdXRoL3JlYWxtcy9tYXN0ZXIiLCJhdWQiOlsibWFzdGVyLXJlYWxtIiwiYWNjb3VudCJdLCJzdWIiOiIwMjhjOGI0OC1jMzliLTQ1NzgtOTExMC0wYjViZGQzODI0ZGEiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJhcGktY2xpZW50Iiwic2Vzc2lvbl9zdGF0ZSI6Ijk4MDhkZTA5LWU2NjYtNGIyNC05ZWQ4LTc2MmUxZjllODk0ZSIsImFjciI6IjEiLCJhbGxvd2VkLW9yaWdpbnMiOlsiKiJdLCJyZWFsbV9hY2Nlc3MiOnsicm9sZXMiOlsiZGVmYXVsdC1yb2xlcy1tYXN0ZXIiLCJvZmZsaW5lX2FjY2VzcyIsInVtYV9hdXRob3JpemF0aW9uIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsibWFzdGVyLXJlYWxtIjp7InJvbGVzIjpbIm1hbmFnZS11c2VycyIsInZpZXctdXNlcnMiLCJxdWVyeS1ncm91cHMiLCJxdWVyeS11c2VycyJdfSwiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwic2NvcGUiOiJwcm9maWxlIGVtYWlsIiwic2lkIjoiOTgwOGRlMDktZTY2Ni00YjI0LTllZDgtNzYyZTFmOWU4OTRlIiwiZW1haWxfdmVyaWZpZWQiOmZhbHNlLCJodHRwczovL2hhc3VyYS5pby9qd3QvY2xhaW1zIjp7IngtaGFzdXJhLXVzZXItaWQiOiIwMjhjOGI0OC1jMzliLTQ1NzgtOTExMC0wYjViZGQzODI0ZGEiLCJ4LWhhc3VyYS1kZWZhdWx0LXJvbGUiOiJ1c2VyIiwieC1oYXN1cmEtYWxsb3dlZC1yb2xlcyI6WyJ1c2VyIl0sIngtaGFzdXJhLXVzZXItZ3JvdXBzIjoie30ifSwibmFtZSI6IkpvaG4gSGFuc2FyaWNrIiwicHJlZmVycmVkX3VzZXJuYW1lIjoiam9obi5odW1tZWxAd2FsbGFyb28uYWkiLCJnaXZlbl9uYW1lIjoiSm9obiIsImZhbWlseV9uYW1lIjoiSGFuc2FyaWNrIiwiZW1haWwiOiJqb2huLmh1bW1lbEB3YWxsYXJvby5haSJ9.lfesLVvWM21kbWIhQtT2ap-ruT5_qt7CVcPUt1mAS8KoksuiJIb4QxPV9FwGB1I7sPiWjXR60cR-cjNLLoTgCX9GZZbfISDR4NvqN5ZBANDzYCx64WtTZCaDPeWROClHvmmE6Mfs1mAdgC3fIxkDe6Ns5-S6wnDqW7v6-yaNo5gBywftaCFyD3lFsmpmBvcyXphtn7sUlX_W4Ku9xmaalUkLv1F8528thZAARN5Jl-_uTHNKCe5wYGiEpQkbeIZ_Rjzqnctx-onw3cVKgbS6_wATr0TZQxgR2AY459OkCJ3rcuJTTTI5PihEGKlQUX5GmDIGG8DqE3iAPJ-xCY-OBQ'

```python
## Refresh the token

TOKENURL=f'https://{wallarooPrefix}keycloak.{wallarooSuffix}/auth/realms/master/protocol/openid-connect/token'

# Retrieving through os environmental variables 
f = open('./creds.json')
login_data = json.load(f)

CONFIDENTIAL_CLIENT=login_data["confidentialClient"]
CONFIDENTIAL_CLIENT_SECRET=login_data["confidentialPassword"]

auth = HTTPBasicAuth(CONFIDENTIAL_CLIENT, CONFIDENTIAL_CLIENT_SECRET)
data = {
    'grant_type': 'refresh_token',
    'refresh_token': refresh_token
}
response = requests.post(TOKENURL, auth=auth, data=data, verify=True)
access_token = response.json()['access_token']
refresh_token = response.json()['refresh_token']
display(access_token)
```

    'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJDYkFqN19QY0xCWTFkWmJiUDZ6Q3BsbkNBYTd6US0tRHlyNy0yLXlQb25nIn0.eyJleHAiOjE2ODQzNjAxNjcsImlhdCI6MTY4NDM1NjU2NywianRpIjoiZDJlNTNlNzEtYjYzMi00MzNmLThjY2UtOGIxMDI0ZjFmYzliIiwiaXNzIjoiaHR0cHM6Ly9kb2MtdGVzdC5rZXljbG9hay53YWxsYXJvb2NvbW11bml0eS5uaW5qYS9hdXRoL3JlYWxtcy9tYXN0ZXIiLCJhdWQiOlsibWFzdGVyLXJlYWxtIiwiYWNjb3VudCJdLCJzdWIiOiIwMjhjOGI0OC1jMzliLTQ1NzgtOTExMC0wYjViZGQzODI0ZGEiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJhcGktY2xpZW50Iiwic2Vzc2lvbl9zdGF0ZSI6Ijk4MDhkZTA5LWU2NjYtNGIyNC05ZWQ4LTc2MmUxZjllODk0ZSIsImFjciI6IjEiLCJhbGxvd2VkLW9yaWdpbnMiOlsiKiJdLCJyZWFsbV9hY2Nlc3MiOnsicm9sZXMiOlsiZGVmYXVsdC1yb2xlcy1tYXN0ZXIiLCJvZmZsaW5lX2FjY2VzcyIsInVtYV9hdXRob3JpemF0aW9uIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsibWFzdGVyLXJlYWxtIjp7InJvbGVzIjpbIm1hbmFnZS11c2VycyIsInZpZXctdXNlcnMiLCJxdWVyeS1ncm91cHMiLCJxdWVyeS11c2VycyJdfSwiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwic2NvcGUiOiJwcm9maWxlIGVtYWlsIiwic2lkIjoiOTgwOGRlMDktZTY2Ni00YjI0LTllZDgtNzYyZTFmOWU4OTRlIiwiZW1haWxfdmVyaWZpZWQiOmZhbHNlLCJodHRwczovL2hhc3VyYS5pby9qd3QvY2xhaW1zIjp7IngtaGFzdXJhLXVzZXItaWQiOiIwMjhjOGI0OC1jMzliLTQ1NzgtOTExMC0wYjViZGQzODI0ZGEiLCJ4LWhhc3VyYS1kZWZhdWx0LXJvbGUiOiJ1c2VyIiwieC1oYXN1cmEtYWxsb3dlZC1yb2xlcyI6WyJ1c2VyIl0sIngtaGFzdXJhLXVzZXItZ3JvdXBzIjoie30ifSwibmFtZSI6IkpvaG4gSGFuc2FyaWNrIiwicHJlZmVycmVkX3VzZXJuYW1lIjoiam9obi5odW1tZWxAd2FsbGFyb28uYWkiLCJnaXZlbl9uYW1lIjoiSm9obiIsImZhbWlseV9uYW1lIjoiSGFuc2FyaWNrIiwiZW1haWwiOiJqb2huLmh1bW1lbEB3YWxsYXJvby5haSJ9.Va5OiLkedj-mFuI7UhsxXshmaSbhXthLv-PU56f2JiQi5-wiWXXxRk3pUioavIzKbi-VjmYQfbR95VY5QSWpUuD3scPuRhHDkSeslz6390phYiFygK_PmXMviQnL2q1mwdGzwh69htOjUWLf7MGWjNmkNdzjYyBy8gfD3V2O2MCfN3onVVCqr1aA1aAQXe9y_JswhjooxAQGit1xzNicvm3IW3QhHtOrDKj7gXNuSlc5vKqe52RQYEgElltqIOV4PVe12UGthKMfvdlDIeUEpTzXVFRH8XHJCrO_YQ_W9m-Rt1_9kelBl3SksdYKOisZaGwo6lv7hhapembH0iD29Q'

```python
## Pipeline Inference URL token - does not require Wallaroo username/password.

TOKENURL=f'https://{wallarooPrefix}keycloak.{wallarooSuffix}/auth/realms/master/protocol/openid-connect/token'

# Retrieving through os environmental variables 
f = open('./creds.json')
login_data = json.load(f)

CONFIDENTIAL_CLIENT=login_data["confidentialClient"]
CLIENT_SECRET=login_data["confidentialPassword"]

auth = HTTPBasicAuth(CONFIDENTIAL_CLIENT, CLIENT_SECRET)
data = {
    'grant_type': 'client_credentials'
}
response = requests.post(TOKENURL, auth=auth, data=data, verify=True)
inference_access_token = response.json()['access_token']
display(inference_access_token)
```

    'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJDYkFqN19QY0xCWTFkWmJiUDZ6Q3BsbkNBYTd6US0tRHlyNy0yLXlQb25nIn0.eyJleHAiOjE2ODQzNjAxNjgsImlhdCI6MTY4NDM1NjU2OCwianRpIjoiNjIyOTViYmUtYzVlMi00NDQ2LWFmMDctNzY5MDAwNmI2NzI3IiwiaXNzIjoiaHR0cHM6Ly9kb2MtdGVzdC5rZXljbG9hay53YWxsYXJvb2NvbW11bml0eS5uaW5qYS9hdXRoL3JlYWxtcy9tYXN0ZXIiLCJhdWQiOlsibWFzdGVyLXJlYWxtIiwiYWNjb3VudCJdLCJzdWIiOiJmNmI5ODg5NC1iZTVjLTQyZDUtYTZhNS02ZjE5ZTY1YmNiNGEiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJhcGktY2xpZW50IiwiYWNyIjoiMSIsImFsbG93ZWQtb3JpZ2lucyI6WyIqIl0sInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJkZWZhdWx0LXJvbGVzLW1hc3RlciIsIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iXX0sInJlc291cmNlX2FjY2VzcyI6eyJtYXN0ZXItcmVhbG0iOnsicm9sZXMiOlsiaW1wZXJzb25hdGlvbiIsIm1hbmFnZS11c2VycyIsInZpZXctdXNlcnMiLCJxdWVyeS1ncm91cHMiLCJxdWVyeS11c2VycyJdfSwiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwic2NvcGUiOiJwcm9maWxlIGVtYWlsIiwiZW1haWxfdmVyaWZpZWQiOmZhbHNlLCJjbGllbnRJZCI6ImFwaS1jbGllbnQiLCJjbGllbnRIb3N0IjoiMTAuMjQ0LjEuNzQiLCJodHRwczovL2hhc3VyYS5pby9qd3QvY2xhaW1zIjp7IngtaGFzdXJhLXVzZXItaWQiOiJmNmI5ODg5NC1iZTVjLTQyZDUtYTZhNS02ZjE5ZTY1YmNiNGEiLCJ4LWhhc3VyYS1kZWZhdWx0LXJvbGUiOiJ1c2VyIiwieC1oYXN1cmEtYWxsb3dlZC1yb2xlcyI6WyJ1c2VyIl0sIngtaGFzdXJhLXVzZXItZ3JvdXBzIjoie30ifSwicHJlZmVycmVkX3VzZXJuYW1lIjoic2VydmljZS1hY2NvdW50LWFwaS1jbGllbnQiLCJjbGllbnRBZGRyZXNzIjoiMTAuMjQ0LjEuNzQifQ.fde1-NsmXqCen71sRcIarscK1j4oFGATf8jh834aAUSb_UGXEmxEnUDDGMegu7KmpbeOi2ogIGY0ndACaZqS21lvVpzWHyVdsQGXCtl1mjwgLt0kzq6U5uR8znMIV-2Babw-9eE65F9I3TdUKRlnh8J5SAPvbOj_Hv_Y3u4cNj1b_Hk_o9lAEg-m2V0ZL7UDxgnVyitbWChiP4DE3q6yBBSVoORiBXDrfUiwIpCXyVKJIO_HrowEA8bYVOhh8PcbywmVa1kZaPcMuAOzsaysE361NCvJqbikVf4KX5Ii9k7lk90v3c-9VX24bIC67HFG8TwvWVnKRBAawwXcQ2ZTIA'

### Through the Wallaroo SDK

The Wallaroo SDK method Wallaroo Client `wl.auth.auth_header()` method provides the token with the `Authorization` header.

```python
# Retrieve the token
headers = wl.auth.auth_header()
display(headers)

{'Authorization': 'Bearer abcdefg'}
```

### Connect to Wallaroo

For this example, a connection to the Wallaroo SDK is used.  This will be used to retrieve the JWT token for the MLOps API calls.  

This example will store the user's credentials into the file `./creds.json` which contains the following:

```json
{
    "username": "{Connecting User's Username}", 
    "password": "{Connecting User's Password}", 
    "email": "{Connecting User's Email Address}"
}
```

Replace the `username`, `password`, and `email` fields with the user account connecting to the Wallaroo instance.  This allows a seamless connection to the Wallaroo instance and bypasses the standard browser based confirmation link.  For more information, see the [Wallaroo SDK Essentials Guide:  Client Connection](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

For `wallarooPrefix = "YOUR PREFIX."` and `wallarooSuffix = "YOUR SUFFIX"`, enter the prefix and suffix for your Wallaroo instance DNS name.  If the prefix instance is blank, then it can be `wallarooPrefix = ""`.  **Note that the prefix includes the `.` for proper formatting.**

```python
# Retrieve the login credentials.
os.environ["WALLAROO_SDK_CREDENTIALS"] = './creds.json'

# Client connection from local Wallaroo instance

wallarooPrefix = "YOUR PREFIX."
wallarooSuffix = "YOUR SUFFIX"

wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}api.{wallarooSuffix}", 
                     auth_endpoint=f"https://{wallarooPrefix}keycloak.{wallarooSuffix}", 
                     auth_type="user_password")
```

### API URL

The variable `APIURL` is used to specify the connection to the Wallaroo instance's MLOps API URL.

```python
APIURL=f"https://{wallarooPrefix}api.{wallarooSuffix}"
```

### API Request Methods

This tutorial relies on the Python `requests` library, and the Wallaroo Wallaroo Client `wl.auth.auth_header()` method.

MLOps API requests are always `POST`.  Most are submitted with the header `'Content-Type':'application/json'` unless specified otherwise.

## Users

### Get Users

Users can be retrieved either by their Keycloak user id, or return all users if an empty set `{}` is submitted.

* **Parameters**
  * `{}`: Empty set, returns all users.
  * **user_ids** *Array[Keycloak user ids]*: An array of Keycloak user ids, typically in UUID format.

Example:  The first example will submit an empty set `{}` to return all users, then submit the first user's user id and request only that user's details.

```python
# Get all users

# Retrieve the token 
headers = wl.auth.auth_header()

api_request = f"{APIURL}/v1/api/users/query"
data = {
}

response = requests.post(api_request, json=data, headers=headers, verify=True).json()
response
```

    {'users': {'028c8b48-c39b-4578-9110-0b5bdd3824da': {'access': {'manageGroupMembership': True,
        'impersonate': False,
        'view': True,
        'manage': True,
        'mapRoles': True},
       'createdTimestamp': 1684355671859,
       'disableableCredentialTypes': [],
       'email': 'john.hummel@wallaroo.ai',
       'emailVerified': False,
       'enabled': True,
       'firstName': 'John',
       'id': '028c8b48-c39b-4578-9110-0b5bdd3824da',
       'lastName': 'Hansarick',
       'notBefore': 0,
       'requiredActions': [],
       'username': 'john.hummel@wallaroo.ai'},
      'de777519-2963-423a-92d2-e6e26d687527': {'access': {'manage': True,
        'manageGroupMembership': True,
        'mapRoles': True,
        'impersonate': False,
        'view': True},
       'createdTimestamp': 1684355337295,
       'disableableCredentialTypes': [],
       'emailVerified': False,
       'enabled': True,
       'id': 'de777519-2963-423a-92d2-e6e26d687527',
       'notBefore': 0,
       'requiredActions': [],
       'username': 'admin'}}}

```python
# Get first user Keycloak id
# Retrieve the token 
headers = wl.auth.auth_header()

# retrieved from the previous request
first_user_keycloak = list(response['users'])[0]

api_request = f"{APIURL}/v1/api/users/query"

data = {
  "user_ids": [
    first_user_keycloak
  ]
}

response = requests.post(api_request, json=data, headers=headers, verify=True).json()
response
```

    {'users': {'028c8b48-c39b-4578-9110-0b5bdd3824da': {'access': {'mapRoles': True,
        'view': True,
        'manage': True,
        'manageGroupMembership': True,
        'impersonate': False},
       'createdTimestamp': 1684355671859,
       'disableableCredentialTypes': [],
       'email': 'john.hummel@wallaroo.ai',
       'emailVerified': False,
       'enabled': True,
       'federatedIdentities': [{'identityProvider': 'google',
         'userId': '117610299312093432527',
         'userName': 'john.hummel@wallaroo.ai'}],
       'firstName': 'John',
       'id': '028c8b48-c39b-4578-9110-0b5bdd3824da',
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

Retrieve the token 
headers = wl.auth.auth_header()

api_request = f"{APIURL}/v1/api/users/invite"

data = {
    "email": new_user,
    "password":new_user_password
}

response = requests.post(api_request, json=data, headers=headers, verify=True).json()
response
```

### Deactivate User

Users can be deactivated so they can not login to their Wallaroo instance.  Deactivated users do not count against the Wallaroo license count.

* **Parameters**
  * **email** (*REQUIRED string*):  The email address of the user to deactivate.

Example:  In this example, the `deactivated_user` will be deactivated.

```python
## Deactivate users

# Retrieve the token 
headers = wl.auth.auth_header()

api_request = f"{APIURL}/v1/api/users/deactivate"

data = {
    "email": new_user
}

response = requests.post(api_request, json=data, headers=headers, verify=True).json()
response
```

### Activate User

A deactivated user can be reactivated to allow them access to their Wallaroo instance.  Activated users count against the Wallaroo license count.

* **Parameters**
  * **email** (*REQUIRED string*):  The email address of the user to activate.

Example:  In this example, the `activated_user` will be activated.

```python
## Activate users

# Retrieve the token 
headers = wl.auth.auth_header()
api_request = f"{APIURL}/v1/api/users/activate"

data = {
    "email": new_user
}

response = requests.post(api_request, json=data, headers=headers, verify=True).json()
response
```

## Workspaces

### List Workspaces

List the workspaces for a specific user.

* **Parameters**
  * **user_id** - (*OPTIONAL string*): The Keycloak ID.
  
Example:  In this example, the workspaces for all users will be displayed.

```python
# List workspaces

# Retrieve the token 
headers = wl.auth.auth_header()

api_request = f"{APIURL}/v1/api/workspaces/list"

data = {
}

response = requests.post(api_request, json=data, headers=headers, verify=True).json()
response
```

    {'workspaces': [{'id': 1,
       'name': 'john.hummel@wallaroo.ai - Default Workspace',
       'created_at': '2023-05-17T20:36:36.312003+00:00',
       'created_by': '028c8b48-c39b-4578-9110-0b5bdd3824da',
       'archived': False,
       'models': [],
       'pipelines': []},
      {'id': 5,
       'name': 'housepricedrift',
       'created_at': '2023-05-17T20:41:50.351766+00:00',
       'created_by': '028c8b48-c39b-4578-9110-0b5bdd3824da',
       'archived': False,
       'models': [1],
       'pipelines': [1]},
      {'id': 6,
       'name': 'sdkquickworkspace',
       'created_at': '2023-05-17T20:43:36.727099+00:00',
       'created_by': '028c8b48-c39b-4578-9110-0b5bdd3824da',
       'archived': False,
       'models': [2, 3],
       'pipelines': [3]}]}

### Create Workspace

A new workspace will be created in the Wallaroo instance.  Upon creating, the workspace owner will be assigned as the user making the MLOps API request.

* **Parameters**:
  * **workspace_name** - (*REQUIRED string*):  The name of the new workspace with the following requirements:
    * Must be unique.
    * DNS compliant with only lowercase characters.
* **Returns**:
  * **workspace_id** - (*int*):  The ID of the new workspace.
  
Example:  In this example, a workspace with the name `testapiworkspace` will be created, and the newly created workspace's `workspace_id` saved as the variable `example_workspace_id` for use in other code examples.  After the request is complete, the [List Workspaces](#list-workspaces) command will be issued to demonstrate the new workspace has been created.

```python
# Create workspace
# Retrieve the token 
headers = wl.auth.auth_header()

api_request = f"{APIURL}/v1/api/workspaces/create"

data = {
  "workspace_name": example_workspace_name
}

response = requests.post(api_request, json=data, headers=headers, verify=True).json()
display(response)
# Stored for future examples
example_workspace_id = response['workspace_id']
```

    {'workspace_id': 7}

```python
## List workspaces

# Retrieve the token 
headers = wl.auth.auth_header()

api_request = f"{APIURL}/v1/api/workspaces/list"

data = {
}

response = requests.post(api_request, json=data, headers=headers, verify=True).json()
response
```

    {'workspaces': [{'id': 1,
       'name': 'john.hummel@wallaroo.ai - Default Workspace',
       'created_at': '2023-05-17T20:36:36.312003+00:00',
       'created_by': '028c8b48-c39b-4578-9110-0b5bdd3824da',
       'archived': False,
       'models': [],
       'pipelines': []},
      {'id': 5,
       'name': 'housepricedrift',
       'created_at': '2023-05-17T20:41:50.351766+00:00',
       'created_by': '028c8b48-c39b-4578-9110-0b5bdd3824da',
       'archived': False,
       'models': [1],
       'pipelines': [1]},
      {'id': 6,
       'name': 'sdkquickworkspace',
       'created_at': '2023-05-17T20:43:36.727099+00:00',
       'created_by': '028c8b48-c39b-4578-9110-0b5bdd3824da',
       'archived': False,
       'models': [2, 3],
       'pipelines': [3]},
      {'id': 7,
       'name': 'apiworkspaces',
       'created_at': '2023-05-17T20:50:36.298217+00:00',
       'created_by': '028c8b48-c39b-4578-9110-0b5bdd3824da',
       'archived': False,
       'models': [],
       'pipelines': []}]}

### Add User to Workspace

Existing users of the Wallaroo instance can be added to an existing workspace.

* **Parameters**
  * **email** - (*REQUIRED string*):  The email address of the user to add to the workspace.  This user must already exist in the Wallaroo instance.
  * **workspace_id** - (*REQUIRED int*): The id of the workspace.
  
Example:  The following example adds the user created in Invite Users request to the workspace created in the [Create Workspace](#create-workspace) request.

```python
# Add existing user to existing workspace

# Retrieve the token 
headers = wl.auth.auth_header()

api_request = f"{APIURL}/v1/api/workspaces/add_user"

data = {
  "email":new_user,
  "workspace_id": example_workspace_id
}

response = requests.post(api_request, json=data, headers=headers, verify=True).json()
response
```

    {}

### List Users in a Workspace

Lists the users who are either owners or collaborators of a workspace.

* **Parameters**
  * **workspace_id** - (*REQUIRED int*): The id of the workspace.
* **Returns**
  * **user_id**:  The user's Keycloak identification.
  * **user_type**:  The user's workspace type (owner, co-owner, etc).
  
Example:  The following example will list all users part of the workspace created in the [Create Workspace](#create-workspace) request.

```python
# List users in a workspace

# Retrieve the token 

headers = wl.auth.auth_header()

api_request = f"{APIURL}/v1/api/workspaces/list_users"

data = {
  "workspace_id": example_workspace_id
}

response = requests.post(api_request, json=data, headers=headers, verify=True).json()
response
```

    {'users': [{'user_id': '028c8b48-c39b-4578-9110-0b5bdd3824da',
       'user_type': 'OWNER'},
      {'user_id': 'c64d26bc-5d30-4d0d-9ae9-9c0089bc4b80',
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

# Retrieve the token 
headers = wl.auth.auth_header()

api_request = f"{APIURL}/v1/api/workspaces/remove_user"

data = {
  "email":new_user,
  "workspace_id": example_workspace_id
}

response = requests.post(api_request, json=data, headers=headers, verify=True).json()
response
```

    {'affected_rows': 1}

```python
## List users in a workspace

# Retrieve the token 
headers = wl.auth.auth_header()

api_request = f"{APIURL}/v1/api/users/query"

data = {
  "workspace_id": example_workspace_id
}

response = requests.post(api_request, json=data, headers=headers, verify=True).json()
response
```

    {'users': {'de777519-2963-423a-92d2-e6e26d687527': {'access': {'mapRoles': True,
        'view': True,
        'manageGroupMembership': True,
        'impersonate': False,
        'manage': True},
       'createdTimestamp': 1684355337295,
       'disableableCredentialTypes': [],
       'emailVerified': False,
       'enabled': True,
       'id': 'de777519-2963-423a-92d2-e6e26d687527',
       'notBefore': 0,
       'requiredActions': [],
       'username': 'admin'},
      'c64d26bc-5d30-4d0d-9ae9-9c0089bc4b80': {'access': {'manage': True,
        'mapRoles': True,
        'manageGroupMembership': True,
        'impersonate': False,
        'view': True},
       'createdTimestamp': 1684356685177,
       'disableableCredentialTypes': [],
       'email': 'john.hansarick@wallaroo.ai',
       'emailVerified': True,
       'enabled': True,
       'firstName': 'John',
       'id': 'c64d26bc-5d30-4d0d-9ae9-9c0089bc4b80',
       'lastName': 'Hansarick',
       'notBefore': 0,
       'requiredActions': [],
       'username': 'john.hansarick@wallaroo.ai'},
      '028c8b48-c39b-4578-9110-0b5bdd3824da': {'access': {'mapRoles': True,
        'view': True,
        'impersonate': False,
        'manage': True,
        'manageGroupMembership': True},
       'createdTimestamp': 1684355671859,
       'disableableCredentialTypes': [],
       'email': 'john.hummel@wallaroo.ai',
       'emailVerified': False,
       'enabled': True,
       'firstName': 'John',
       'id': '028c8b48-c39b-4578-9110-0b5bdd3824da',
       'lastName': 'Hansarick',
       'notBefore': 0,
       'requiredActions': [],
       'username': 'john.hummel@wallaroo.ai'}}}

## Models

### Upload Model to Workspace

ML Models are uploaded to Wallaroo through the following endpoint:

Models uploaded through this method that are not native runtimes are containerized within the Wallaroo instance then run by the Wallaroo engine.  See [Wallaroo MLOps API Essentials Guide: Pipeline Management]({{<ref "wallaroo-mlops-api-essential-guide-pipelines">}}) for details on pipeline configurations and deployments.

For these models, the following inputs are required.

* Endpoint:
  * `/v1/api/models/upload_and_convert`
* Headers:
  * **Content-Type**: `multipart/form-data`
* Parameters
  * **name** (*String* *Required*): The model name.
  * **visibility** (*String* *Required*): Either `public` or `private`.
  * **workspace_id** (*String* *Required*): The numerical ID of the workspace to upload the model to.
  * **conversion** (*String* *Required*):  The conversion parameters that include the following:
    * **framework** (*String* *Required*): The framework of the model being uploaded.  See the list of supported models for more details.
    * **python_version** (*String* *Required*):  The version of Python required for model.
    * **requirements**  (*String* *Required*):  Required libraries.  Can be `[]` if the requirements are default Wallaroo JupyterHub libraries.
    * **input_schema**  (*String* *Optional*): The input schema from the Apache Arrow `pyarrow.lib.Schema` format, encoded with `base64.b64encode`.  Only required for non-native runtime models.
    * **output_schema** (*String* *Optional*): The output schema from the Apache Arrow `pyarrow.lib.Schema` format, encoded with `base64.b64encode`.  Only required for non-native runtime models.

#### Upload Native Runtime Model Example

ONNX are always native runtimes.  The following example shows uploading an ONNX model to a Wallaroo instance using the `requests` library.  Note that the `input_schema` and `output_schema` encoded details are not required.

```python
 authorization header
headers = {'Authorization': 'Bearer abcdefg'}

apiRequest = f"{APIURL}/v1/api/models/upload_and_convert"

framework='onnx'

model_name = f"{suffix}ccfraud"

data = {
    "name": model_name,
    "visibility": "public",
    "workspace_id": workspaceId,
    "conversion": {
        "framework": framework,
        "python_version": "3.8",
        "requirements": []
    }
}

files = {
    "metadata": (None, json.dumps(data), "application/json"),
    'file': (model_name, open('./ccfraud.onnx', 'rb'), "application/octet-stream")
    }

response = requests.post(apiRequest, files=files, headers=headers).json()
```

    'Sample model name: apimodel'

    'Sample model file: ./models/ccfraud.onnx'

    {'insert_models': {'returning': [{'models': [{'id': 4}]}]}}

#### Upload Converted Model Examples

The following example shows uploading a Hugging Face model to a Wallaroo instance using the `requests` library.  Note that the `input_schema` and `output_schema` encoded details are required.

```python
input_schema = pa.schema([
    pa.field('inputs', pa.string()), # required
    pa.field('candidate_labels', pa.list_(pa.string(), list_size=2)), # required
    pa.field('hypothesis_template', pa.string()), # optional
    pa.field('multi_label', pa.bool_()), # optional
])

output_schema = pa.schema([
    pa.field('sequence', pa.string()),
    pa.field('scores', pa.list_(pa.float64(), list_size=2)), # same as number of candidate labels, list_size can be skipped by may result in slightly worse performance
    pa.field('labels', pa.list_(pa.string(), list_size=2)), # same as number of candidate labels, list_size can be skipped by may result in slightly worse performance
])

encoded_input_schema = base64.b64encode(
                bytes(input_schema.serialize())
            ).decode("utf8")

encoded_output_schema = base64.b64encode(
                bytes(output_schema.serialize())
            ).decode("utf8")

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
    'file': (model_name, open(model_path,'rb'),'application/octet-stream')
}

response = requests.post('https://{APIURL}/v1/api/models/upload_and_convert', 
                         headers=headers, 
                         files=files).json()
```

### Stream Upload Model to Workspace

Streams a potentially large ML Model to a Wallaroo workspace via POST with `Content-Type: multipart/form-data`.

* **Parameters**
  * **name** - (*REQUIRED string*): Name of the model.  Must only include alphanumeric characters.
  * **filename** - (*REQUIRED string*): Name of the file being uploaded.
  * **visibility** - (*OPTIONAL string*): The visibility of the model as either `public` or `private`.
  * **workspace_id** - (*REQUIRED int*): The numerical id of the workspace to upload the model to.
  
Example:  This example will upload the sample file `ccfraud.onnx` to the workspace created in the [Create Workspace](#create-workspace) step as `apitestmodel`.

```python
# stream upload model - next test is adding arbitrary chunks to the stream

# Retrieve the token 
headers = wl.auth.auth_header()

# Set the contentType
headers['contentType']='application/octet-stream'

api_request = f"{APIURL}/v1/api/models/upload_stream"

# Model name and file to use
display(f"Sample stream model name: {stream_model_name}")
display(f"Sample model file: {stream_model_file_name}")

data = {
    "name":stream_model_name,
    "filename": stream_model_file_name,
    "visibility":"public",
    "workspace_id": example_workspace_id
}

files = {
    'file': (stream_model_name, open(stream_model_file_name, 'rb'))
    }

response = requests.post(apiRequest, files=files, data=data, headers=headers).json()
response
```

    'Sample stream model name: apiteststreammodel'

    'Sample model file: ./models/ccfraud.onnx'

    {'insert_models': {'returning': [{'models': [{'id': 5}]}]}}

### List Models in Workspace

Returns a list of models added to a specific workspace.

* **Parameters**
  * **workspace_id** - (*REQUIRED int*): The workspace id to list.
  
Example:  Display the models for the workspace used in the Upload Model to Workspace step.  The model id and model name will be saved as `example_model_id` and `exampleModelName` variables for other examples.

```python
# List models in a workspace
# Retrieve the token 
headers = wl.auth.auth_header()

api_request = f"{APIURL}/v1/api/models/list"

data = {
  "workspace_id": example_workspace_id
}

response = requests.post(api_request, json=data, headers=headers, verify=True).json()
response
```

    {'models': [{'id': 5,
       'name': 'apiteststreammodel',
       'owner_id': '""',
       'created_at': '2023-05-17T20:51:53.077997+00:00',
       'updated_at': '2023-05-17T20:51:53.077997+00:00'},
      {'id': 4,
       'name': 'apimodel',
       'owner_id': '""',
       'created_at': '2023-05-17T20:51:51.092416+00:00',
       'updated_at': '2023-05-17T20:51:51.092416+00:00'}]}

```python
model = next(model for model in response["models"] if model["name"] == "apimodel")
example_model_id = model['id']
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
# Retrieve the token 
headers = wl.auth.auth_header()

api_request = f"{APIURL}/v1/api/models/get_by_id"

data = {
  "id": example_model_id
}

response = requests.post(api_request, json=data, headers=headers, verify=True).json()
response
```

    {'id': 4,
     'owner_id': '""',
     'workspace_id': 7,
     'name': 'apimodel',
     'updated_at': '2023-05-17T20:51:51.092416+00:00',
     'created_at': '2023-05-17T20:51:51.092416+00:00',
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

Example:  Retrieve the versions for a previously uploaded model. The variables `example_model_version` and `example_model_sha` will store the model's version and SHA values for use in other examples.

```python
## List model versions

# Retrieve the token 
headers = wl.auth.auth_header()
api_request = f"{APIURL}/v1/api/models/list_versions"

data = {
  "model_id": model_name,
  "models_pk_id": example_model_id
}

response = requests.post(api_request, json=data, headers=headers, verify=True).json()
response
```

    [{'sha': 'bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507',
      'models_pk_id': 4,
      'model_version': '0b989008-5f1d-453e-8085-98d97be1b722',
      'owner_id': '""',
      'model_id': 'apimodel',
      'id': 4,
      'file_name': 'apimodel',
      'image_path': None,
      'status': 'ready'}]

```python
# Stored for future examples

example_model_version = response[-1]['model_version']
example_model_sha = response[-1]['sha']
```

### Get Model Configuration by Id

Returns the model's configuration details.

* **Parameters**
  * **model_id** - (*REQUIRED int*): The numerical value of the model's id.
  
Example:  Submit the model id for the model uploaded in the Upload Model to Workspace step to retrieve configuration details.

```python
## Get model config by id

# Retrieve the token 
headers = wl.auth.auth_header()
api_request = f"{APIURL}/v1/api/models/get_config_by_id"

data = {
  "model_id": example_model_id
}

response = requests.post(api_request, json=data, headers=headers, verify=True).json()
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
# Retrieve the token 
headers = wl.auth.auth_header()
api_request = f"{APIURL}/v1/api/models/get"

data = {
  "id": example_model_id
}

response = requests.post(api_request, json=data, headers=headers, verify=True).json()
response
```

    {'id': 4,
     'name': 'apimodel',
     'owner_id': '""',
     'created_at': '2023-05-17T20:51:51.092416+00:00',
     'updated_at': '2023-05-17T20:51:51.092416+00:00',
     'models': [{'sha': 'bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507',
       'models_pk_id': 4,
       'model_version': '0b989008-5f1d-453e-8085-98d97be1b722',
       'owner_id': '""',
       'model_id': 'apimodel',
       'id': 4,
       'file_name': 'apimodel',
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

The variable `example_workspace_id` was created in a previous example.

```python
# Create pipeline in a workspace
# Retrieve the token 
headers = wl.auth.auth_header()

api_request = f"{APIURL}/v1/api/pipelines/create"

data = {
  "pipeline_id": empty_pipeline_name,
  "workspace_id": example_workspace_id,
  "definition": {}
}

response = requests.post(api_request, json=data, headers=headers, verify=True).json()

empty_pipeline_id = response['pipeline_pk_id']
empty_pipeline_variant_id=response['pipeline_variant_pk_id']
example_pipeline_variant_version=['pipeline_variant_version']
response
```

    {'pipeline_pk_id': 7,
     'pipeline_variant_pk_id': 7,
     'pipeline_variant_version': 'a6dd2cee-58d6-4d24-9e25-f531dbbb95ad'}

```python
# Create pipeline in a workspace with models
# Retrieve the token 
headers = wl.auth.auth_header()
api_request = f"{APIURL}/v1/api/pipelines/create"

data = {
  "pipeline_id": model_pipeline_name,
  "workspace_id": example_workspace_id,
  "definition": {
      'steps': [
          {
            'ModelInference': 
              {
                  'models': 
                    [
                        {
                            'name': model_name, 
                            'version': example_model_version, 
                            'sha': example_model_sha
                        }
                    ]
              }
          }
        ]
      }
    }

response = requests.post(api_request, json=data, headers=headers, verify=True).json()
model_pipeline_id = response['pipeline_pk_id']
model_pipeline_variant_id=response['pipeline_variant_pk_id']
model_pipeline_variant_version=['pipeline_variant_version']
response
```

    {'pipeline_pk_id': 8,
     'pipeline_variant_pk_id': 8,
     'pipeline_variant_version': '55f45c16-591e-4a16-8082-3ab6d843b484'}

### Deploy a Pipeline

Deploy a an existing pipeline.  Note that for any pipeline that has model steps, they must be included either in `model_configs`, `model_ids` or `models`.

* **Parameters**
  * **deploy_id** (*REQUIRED string*): The name for the pipeline deployment.  This **must** match the name of the pipeline being deployed.
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

Examples:  Both the pipeline with model created in the step [Create Pipeline in a Workspace](#create-pipeline-in-a-workspace) will be deployed and their deployment information saved for later examples.

```python
# Deploy a pipeline with models

# Retrieve the token 
headers = wl.auth.auth_header()
api_request = f"{APIURL}/v1/api/pipelines/deploy"

model_deploy_id=model_pipeline_name

# example_model_deploy_id="test deployment name"

data = {
    "deploy_id": model_deploy_id,
    "pipeline_version_pk_id": model_pipeline_variant_id,
    "models": [
        {
            "name": model_name,
            "version":example_model_version,
            "sha":example_model_sha
        }
    ],
    "pipeline_id": model_pipeline_id
}

response = requests.post(api_request, json=data, headers=headers, verify=True).json()
display(response)
model_deployment_id=response['id']

# wait 45 seconds for the pipeline to complete deployment
import time
time.sleep(45)

```

    {'id': 5}

### Get Deployment Status

Returns the deployment status.

* **Parameters**
  * **name** - (REQUIRED string): The deployment in the format {deployment_name}-{deploymnent-id}.
  
Example: The deployed empty and model pipelines status will be displayed.

```python
# Retrieve the token 
headers = wl.auth.auth_header()

# Get model pipeline deployment

api_request = f"{APIURL}/v1/api/status/get_deployment"

data = {
  "name": f"{model_deploy_id}-{model_deployment_id}"
}

response = requests.post(api_request, json=data, headers=headers, verify=True).json()
response
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.3.136',
       'name': 'engine-76b8f76d58-vbwqs',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'pipelinemodels',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'apimodel',
          'version': '0b989008-5f1d-453e-8085-98d97be1b722',
          'sha': 'bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.4.166',
       'name': 'engine-lb-584f54c899-rdfkl',
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

# Retrieve the token 
headers = wl.auth.auth_header()

api_request = f"{APIURL}/v1/api/admin/get_pipeline_external_url"

data = {
    "workspace_id": example_workspace_id,
    "pipeline_name": model_pipeline_name
}

response = requests.post(api_request, json=data, headers=headers, verify=True).json()
print(response)
deployurl = response['url']
```

    {'url': 'https://doc-test.api.wallarooexample.ai/v1/api/pipelines/infer/pipelinemodels-5/pipelinemodels'}

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

```python
# Retrieve the token
headers = wl.auth.auth_header()

# set Content-Type type
headers['Content-Type']='application/json; format=pandas-records'

## Inference through external URL using dataframe

# retrieve the json data to submit
data = [
    {
        "tensor":[
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
      <td>1684356836285</td>
      <td>{'tensor': [1.0678324729, 0.2177810266, -1.7115145262, 0.682285721, 1.0138553067, -0.4335000013, 0.7395859437, -0.2882839595, -0.447262688, 0.5146124988, 0.3791316964, 0.5190619748, -0.4904593222, 1.1656456469, -0.9776307444, -0.6322198963, -0.6891477694, 0.1783317857, 0.1397992467, -0.3554220649, 0.4394217877, 1.4588397512, -0.3886829615, 0.4353492889, 1.7420053483, -0.4434654615, -0.1515747891, -0.2668451725, -1.4549617756]}</td>
      <td>{'dense_1': [0.0014974177]}</td>
      <td>[]</td>
      <td>{'last_model': '{"model_name":"apimodel","model_sha":"bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507"}', 'pipeline_version': '', 'elapsed': [163502, 309804]}</td>
    </tr>
  </tbody>
</table>

### Undeploy a Pipeline

Undeploys a deployed pipeline.

* **Parameters**
  * **pipeline_id** - (*REQUIRED int*): The numerical id of the pipeline.
  * **deployment_id** - (*REQUIRED int*): The numerical id of the deployment.
* **Returns**
  * Nothing if the call is successful.

Example:  Both the empty pipeline and pipeline with models deployed in the step Deploy a Pipeline will be undeployed.

```python
# Undeploy pipeline with models
# Retrieve the token 
headers = wl.auth.auth_header()
api_request = f"{APIURL}/v1/api/pipelines/undeploy"

data = {
    "pipeline_id": model_pipeline_id,
    "deployment_id":model_deployment_id
}

response = requests.post(api_request, json=data, headers=headers, verify=True).json()
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
## Copy a pipeline

# Retrieve the token 
headers = wl.auth.auth_header()

api_request = f"{APIURL}/v1/api/pipelines/copy"

data = {
  "name": example_copied_pipeline_name,
  "workspace_id": example_workspace_id,
  "source_pipeline": model_pipeline_id
}

response = requests.post(api_request, json=data, headers=headers, verify=True).json()
response
```

    {'pipeline_pk_id': 9,
     'pipeline_variant_pk_id': 9,
     'pipeline_version': None,
     'deployment': None}

## Enablement Management

Enablement Management allows users to see what Wallaroo features have been activated.

### List Enablement Features

Lists the enablement features for the Wallaroo instance.

* **PARAMETERS**
  * null:  An empty set `{}`
* **RETURNS**
  * **features** - (*string*): Enabled features.
  * **name** - (*string*): Name of the Wallaroo instance.
  * **is_auth_enabled** - (*bool*): Whether authentication is enabled.

```python
# List enablement features
# Retrieve the token 
headers = wl.auth.auth_header()
api_request = f"{APIURL}/v1/api/features/list"

data = {
}

response = requests.post(api_request, json=data, headers=headers, verify=True).json()
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
  * **iopath** - (*REQUIRED string*): The iopath of the assay in the format `"input|output field_name field_index`.
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

As noted this example requires the [Wallaroo Assay Tutorial](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/model_insights) for historical data.  Before running this example, set the sample pipeline id, pipeline, name, model name, and workspace id in the code sample below.

For our example, we will be using the output of the field `dense_2` at the index 0 for the iopath.

For more information on retrieving this information, see the [Wallaroo Developer Guides](https://docs.wallaroo.ai/wallaroo-developer-guides/).

```python
# Retrieve information for the housepricedrift workspace

# List workspaces

# Retrieve the token 
headers = wl.auth.auth_header()

api_request = f"{APIURL}/v1/api/workspaces/list"

data = {
}

response = requests.post(api_request, json=data, headers=headers, verify=True).json()
```

```python
assay_workspace = next(workspace for workspace in response["workspaces"] if workspace["name"] == "housepricedrift")
assay_workspace_id = assay_workspace['id']
assay_pipeline_id = assay_workspace['pipelines'][0]
```

```python
## Create assay

# Retrieve the token 
headers = wl.auth.auth_header()

api_request = f"{APIURL}/v1/api/assays/create"

exampleAssayName = "api assay"

## Now get all of the assays for the pipeline in workspace 4 `housepricedrift`

exampleAssayPipelineId = assay_pipeline_id
exampleAssayPipelineName = "housepricepipe"
exampleAssayModelName = "housepricemodel"
exampleAssayWorkspaceId = assay_workspace_id

# iopath can be input 00 or output 0 0
data = {
    'name': exampleAssayName,
    'pipeline_id': exampleAssayPipelineId,
    'pipeline_name': exampleAssayPipelineName,
    'active': True,
    'status': 'active',
    'iopath': "output dense_2 0",
    'baseline': {
        'Fixed': {
            'pipeline': exampleAssayPipelineName,
            'model': exampleAssayModelName,
            'start_at': '2023-01-01T00:00:00-05:00',
            'end_at': '2023-01-02T00:00:00-05:00'
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

response = requests.post(api_request, json=data, headers=headers, verify=True).json()
example_assay_id = response['assay_id']
response
```

    {'assay_id': 5}

### List Assays

Lists all assays in the specified pipeline.

* **PARAMETERS**
  * **pipeline_id** - (*REQUIRED int*):  The numerical ID of the pipeline.
* **RETURNS**
  * **assays** - (*Array assays*): A list of all assays.

Example:  Display a list of all assays in a workspace.  This will assume we have a workspace with an existing Assay and the associated data has been upload.  See the tutorial [Wallaroo Assays Tutorial](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/model_insights).

For this reason, these values are hard coded for now.

```python
# Get assays
# Retrieve the token 
headers = wl.auth.auth_header()

api_request = f"{APIURL}/v1/api/assays/list"

data = {
    "pipeline_id": exampleAssayPipelineId
}

response = requests.post(api_request, json=data, headers=headers, verify=True).json()
response
```

    [{'id': 5,
      'name': 'api assay',
      'active': True,
      'status': 'created',
      'warning_threshold': 0.0,
      'alert_threshold': 0.1,
      'pipeline_id': 1,
      'pipeline_name': 'housepricepipe',
      'last_run': None,
      'next_run': '2023-05-19T17:26:51.743327+00:00',
      'run_until': None,
      'updated_at': '2023-05-19T17:26:51.745495+00:00',
      'baseline': {'Fixed': {'pipeline': 'housepricepipe',
        'model': 'housepricemodel',
        'start_at': '2023-01-01T00:00:00-05:00',
        'end_at': '2023-01-02T00:00:00-05:00'}},
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
       'provided_edges': None}},
     {'id': 2,
      'name': 'onmyexample assay',
      'active': True,
      'status': 'created',
      'warning_threshold': None,
      'alert_threshold': 0.5,
      'pipeline_id': 1,
      'pipeline_name': 'housepricepipe',
      'last_run': None,
      'next_run': '2023-05-17T21:46:58.633746+00:00',
      'run_until': None,
      'updated_at': '2023-05-17T21:46:58.636786+00:00',
      'baseline': {'Fixed': {'pipeline': 'housepricepipe',
        'model': 'housepricemodel',
        'start_at': '2023-01-01T00:00:00+00:00',
        'end_at': '2023-01-02T00:00:00+00:00'}},
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
       'provided_edges': None}},
     {'id': 1,
      'name': 'api_assay',
      'active': True,
      'status': 'created',
      'warning_threshold': 0.0,
      'alert_threshold': 0.1,
      'pipeline_id': 1,
      'pipeline_name': 'housepricepipe',
      'last_run': None,
      'next_run': '2023-05-17T20:54:09.27658+00:00',
      'run_until': None,
      'updated_at': '2023-05-17T20:54:09.27932+00:00',
      'baseline': {'Fixed': {'pipeline': 'housepricepipe',
        'model': 'housepricemodel',
        'start_at': '2023-01-01T00:00:00-05:00',
        'end_at': '2023-01-02T00:00:00-05:00'}},
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
       'provided_edges': None}}]

### Activate or Deactivate Assay

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

# Retrieve the token 
headers = wl.auth.auth_header()

api_request = f"{APIURL}/v1/api/assays/set_active"

data = {
    'id': example_assay_id,
    'active': False
}

response = requests.post(api_request, json=data, headers=headers, verify=True).json()
response
```

    {'id': 5, 'active': False}

```python
# Activate assay

# Retrieve the token 
headers = wl.auth.auth_header()

api_request = f"{APIURL}/v1/api/assays/set_active"

data = {
    'id': example_assay_id,
    'active': True
}

response = requests.post(api_request, json=data, headers=headers, verify=True).json()
response
```

    {'id': 5, 'active': True}

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
## Run interactive baseline

# Retrieve the token 
headers = wl.auth.auth_header()

api_request = f"{APIURL}/v1/api/assays/run_interactive_baseline"

data = {
    'id': example_assay_id,
    'name': exampleAssayName,
    'pipeline_id': exampleAssayPipelineId,
    'pipeline_name': exampleAssayPipelineName,
    'active': True,
    'status': 'active',
    'iopath': "output dense_2 0",
    'baseline': {
        'Fixed': {
            'pipeline': exampleAssayPipelineName,
            'model': exampleAssayModelName,
            'start_at': '2023-01-01T00:00:00-05:00',
            'end_at': '2023-01-02T00:00:00-05:00'
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

response = requests.post(api_request, json=data, headers=headers, verify=True).json()
response
```

    {'id': None,
     'assay_id': 5,
     'window_start': '2023-01-01T05:00:00Z',
     'analyzed_at': '2023-05-19T17:26:59.664583293Z',
     'elapsed_millis': 0,
     'iopath': 'output dense_2 0',
     'pipeline_id': None,
     'baseline_summary': {'count': 181,
      'min': 12.002464294433594,
      'max': 14.095687866210938,
      'mean': 12.892810610776449,
      'median': 12.862584114074709,
      'std': 0.4259400394014014,
      'edges': [12.002464294433594,
       12.525982856750488,
       12.772802352905272,
       12.960931777954102,
       13.246906280517578,
       14.095687866210938,
       None],
      'edge_names': ['left_outlier',
       'q_20',
       'q_40',
       'q_60',
       'q_80',
       'q_100',
       'right_outlier'],
      'aggregated_values': [0.0,
       0.19889502762430936,
       0.19889502762430936,
       0.20441988950276244,
       0.19889502762430936,
       0.19889502762430936,
       0.0],
      'aggregation': 'Density',
      'start': '2023-01-01T05:00:00Z',
      'end': '2023-01-02T05:00:00Z'},
     'window_summary': {'count': 181,
      'min': 12.002464294433594,
      'max': 14.095687866210938,
      'mean': 12.892810610776449,
      'median': 12.862584114074709,
      'std': 0.4259400394014014,
      'edges': [12.002464294433594,
       12.525982856750488,
       12.772802352905272,
       12.960931777954102,
       13.246906280517578,
       14.095687866210938,
       None],
      'edge_names': ['left_outlier',
       'e_1.25e1',
       'e_1.28e1',
       'e_1.30e1',
       'e_1.32e1',
       'e_1.41e1',
       'right_outlier'],
      'aggregated_values': [0.0,
       0.19889502762430936,
       0.19889502762430936,
       0.20441988950276244,
       0.19889502762430936,
       0.19889502762430936,
       0.0],
      'aggregation': 'Density',
      'start': '2023-01-01T05:00:00Z',
      'end': '2023-01-02T05:00:00Z'},
     'warning_threshold': 0.0,
     'alert_threshold': 0.1,
     'score': 0.0,
     'scores': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     'bin_index': None,
     'summarizer': {'type': 'UnivariateContinuous',
      'bin_mode': 'Quantile',
      'aggregation': 'Density',
      'metric': 'PSI',
      'num_bins': 5,
      'bin_weights': None,
      'provided_edges': None},
     'status': 'BaselineRun',
     'created_at': None}

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
  
Example:  3 assay baselines for Workspace 6 and pipeline `houseprice-pipe` will be retrieved.

```python
## Get Assay Baseline

# Retrieve the token 
headers = wl.auth.auth_header()

api_request = f"{APIURL}/v1/api/assays/get_baseline"

data = {
    'workspace_id': exampleAssayWorkspaceId,
    'pipeline_name': exampleAssayPipelineName,
    'limit': 3
}

response = requests.post(api_request, json=data, headers=headers, verify=True).json()
response[0:2]
```

    [{'time': 1672531200000,
      'in.tensor': [0.6752651953165153,
       -0.4732014898144956,
       -1.0785881334179752,
       0.25006446993148707,
       -0.08666382440547035,
       0.012211745933432551,
       -0.1904726364343265,
       -0.9179715198382244,
       -0.305653139057544,
       0.905425782569012,
       -0.5584151415472702,
       -0.8905121321380776,
       1.7014907488187343,
       -0.03617359856638,
       -0.20817781526102327,
       -0.4017891748132812,
       -0.19176790501742016],
      'out.dense_2': [12.529610633850098],
      'metadata.last_model': '{"model_name": "housepricemodel", "model_sha": "test_version"}',
      'metadata.profile': '{"elapsed_ns": 243}'},
     {'time': 1672531676753,
      'in.tensor': [-0.39764636440424433,
       -0.4732014898144956,
       0.5769261528142077,
       0.07215545493232875,
       -0.08666382440547035,
       0.5668723158705202,
       0.0035716408873876734,
       -0.9179715198382244,
       -0.305653139057544,
       0.905425782569012,
       0.29288456205300767,
       -0.10763168763453018,
       1.3841294506067472,
       -0.13822039562434324,
       -0.20817781526102327,
       1.392623186456163,
       0.0831911918963078],
      'out.dense_2': [13.355737686157228],
      'metadata.last_model': '{"model_name": "housepricemodel", "model_sha": "test_version"}',
      'metadata.profile': '{"elapsed_ns": 216}'}]

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
## Run interactive assay

# Retrieve the token 
headers = wl.auth.auth_header()

api_request = f"{APIURL}/v1/api/assays/run_interactive"

data = {
    'id': example_assay_id,
    'name': exampleAssayName,
    'pipeline_id': exampleAssayPipelineId,
    'pipeline_name': exampleAssayPipelineName,
    'active': True,
    'status': 'active',
    'iopath': "output dense_2 0",
    'baseline': {
        'Fixed': {
            'pipeline': exampleAssayPipelineName,
            'model': exampleAssayModelName,
            'start_at': '2023-01-01T00:00:00-05:00',
            'end_at': '2023-01-02T00:00:00-05:00'
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

response = requests.post(api_request, json=data, headers=headers, verify=True).json()
response[0]
```

    {'id': None,
     'assay_id': 1,
     'window_start': '2023-01-02T05:00:00Z',
     'analyzed_at': '2023-05-17T20:54:19.568121901Z',
     'elapsed_millis': 578,
     'iopath': 'output dense_2 0',
     'pipeline_id': None,
     'baseline_summary': {'count': 181,
      'min': 12.002464294433594,
      'max': 14.095687866210938,
      'mean': 12.892810610776449,
      'median': 12.862584114074709,
      'std': 0.4259400394014014,
      'edges': [12.002464294433594,
       12.525982856750488,
       12.772802352905272,
       12.960931777954102,
       13.246906280517578,
       14.095687866210938,
       None],
      'edge_names': ['left_outlier',
       'q_20',
       'q_40',
       'q_60',
       'q_80',
       'q_100',
       'right_outlier'],
      'aggregated_values': [0.0,
       0.19889502762430936,
       0.19889502762430936,
       0.20441988950276244,
       0.19889502762430936,
       0.19889502762430936,
       0.0],
      'aggregation': 'Density',
      'start': '2023-01-01T05:00:00Z',
      'end': '2023-01-02T05:00:00Z'},
     'window_summary': {'count': 182,
      'min': 12.037200927734377,
      'max': 14.712774276733398,
      'mean': 12.966292286967184,
      'median': 12.895143508911133,
      'std': 0.4705339357836451,
      'edges': [12.002464294433594,
       12.525982856750488,
       12.772802352905272,
       12.960931777954102,
       13.246906280517578,
       14.095687866210938,
       None],
      'edge_names': ['left_outlier',
       'e_1.25e1',
       'e_1.28e1',
       'e_1.30e1',
       'e_1.32e1',
       'e_1.41e1',
       'right_outlier'],
      'aggregated_values': [0.0,
       0.17032967032967034,
       0.17582417582417584,
       0.23626373626373623,
       0.1978021978021978,
       0.1978021978021978,
       0.02197802197802198],
      'aggregation': 'Density',
      'start': '2023-01-02T05:00:00Z',
      'end': '2023-01-03T05:00:00Z'},
     'warning_threshold': 0.0,
     'alert_threshold': 0.1,
     'score': 0.037033182069201614,
     'scores': [0.0,
      0.0044288126945783235,
      0.00284446741288289,
      0.004610114809454446,
      6.021116179797982e-06,
      6.021116179797982e-06,
      0.02513774491992636],
     'bin_index': None,
     'summarizer': {'type': 'UnivariateContinuous',
      'bin_mode': 'Quantile',
      'aggregation': 'Density',
      'metric': 'PSI',
      'num_bins': 5,
      'bin_weights': None,
      'provided_edges': None},
     'status': 'Warning',
     'created_at': None}

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

```python
# Get Assay Results

# Retrieve the token 
headers = wl.auth.auth_header()

api_request = f"{APIURL}/v1/api/assays/get_results"

data = {
    'assay_id': example_assay_id,
    'pipeline_id': exampleAssayPipelineId
}

response = requests.post(api_request, json=data, headers=headers, verify=True).json()
response
```
