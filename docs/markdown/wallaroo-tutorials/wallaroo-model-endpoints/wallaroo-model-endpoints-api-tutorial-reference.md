This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-model-endpoints).

## External Pipeline Inference URL Tutorial

Wallaroo provides the ability to perform inferences through deployed pipelines via both internal and external URLs.  These URLs allow inferences to be performed by submitting data to the internal or external URL, with the inference results returned in the same format as the [InferenceResult Object](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/#run-inference-through-a-pipeline).

**Internal URLs** are available only through the internal Kubernetes environment hosting the Wallaroo instance as demonstrated in this tutorial.
**External URLs** are available outside of the Kubernetes environment, such as the public internet.  These are demonstrated in the External Pipeline Deployment URL Tutorial.

**IMPORTANT NOTE**:  Before starting this tutorial, the Internal Pipeline Deployment URL Tutorial must be completed to establish the Wallaroo workspace, pipeline and model to be used 

The following tutorial shows how to set up an environment and demonstrates how to use the External Deployment URL.  This example provides the following:

* `data-1.json`, `data-1k.json` and `data-25k.json`:  Sample data used for testing inferences with the sample model.

## Prerequisites

1. Before running this sample notebook, verify that the Internal Pipeline Deployment URL Tutorial has been run.  This will create the workspace, pipeline, etc for the below example to run.
1. Enable external URl inference endpoints through the Wallaroo Administrative Dashboard.  This can be accessed through the `kots` application as detailed in the [Wallaroo Install Guildes](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-install-guides/).  To access the Wallaroo Administrative Dashboard:
    1. From a terminal shell connected to the Kubernetes environment hosting the Wallaroo instance, run the following `kots` command:

      ```bash
      kubectl kots admin-console --namespace wallaroo
      ```

      This provides the following standard output:

      ```bash
        • Press Ctrl+C to exit
        • Go to http://localhost:8800 to access the Admin Console
      ```

      This will host a `http` connection to the Wallaroo Administrative Dashboard, by default at `http://localhost:8800`.

    1. Open a browser at the URL detailed in the step above and authenticate using the console password set as described in the as detailed in the [Wallaroo Install Guildes](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-install-guides/).
    1. From the top menu, select **Config** then verify that **Networking Configuration -> Ingress Mode for Wallaroo interactive services -> Enable external URL inference endpoints** is enabled.

      ![](/images/wallaroo-tutorials/wallaroo-api/enable-external-url-config.png)
 
    1. Save the updated configuration, then deploy it.  Once complete, the external URL inference endpoints will be enabled.

### Set Variables

The following variables are used for the example and should be modified to fit your organization.

Wallaroo comes pre-installed with a confidential OpenID Connect client.  The default client is `api-client`, but other clients may be created and configured.

As it is a confidential client, api-client requires its secret to be supplied when requesting a token. Administrators may obtain their API client credentials from Keycloak from the Keycloak Service URL as listed above and the prefix `/auth/admin/master/console/#/realms/master/clients`.

For example, if the Wallaroo Community instance DNS address is `https://magical-rhino-5555.wallaroo.community`, then the direct path to the Keycloak API client credentials would be:

`https://magical-rhino-5555.keycloak.wallaroo.community/auth/admin/master/console/#/realms/master/clients`

Then select the client, in this case **api-client**, then **Credentials**.

![Wallaroo Keycloak Service](/images/wallaroo-tutorials/wallaroo-api/wallaroo-api-keycloak-service.png)

![Wallaroo Components](/images/wallaroo-tutorials/wallaroo-api/wallaroo-api-keycloak-credentials.png)

By default, tokens issued for api-client are valid for up to 60 minutes. Refresh tokens are supported.

Set the following variables from the list below:

* `URLPREFIX`: The prefix for your Wallaroo instance.  For more information, see the [DNS Integration Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-dns-guide/).
* `URLSUFFIX`: The suffix for your Wallaroo instance.  For more information, see the [DNS Integration Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-dns-guide/).
* `CLIENT`: The client used as defined in the steps above.
* `SECRET`: The credentials as defined in the steps above.
* `USERNAME`: The Wallaroo user username, typically your email address.
* `PASSWORD`: The Wallaroo user password.

```python
import requests
from requests.auth import HTTPBasicAuth

import json

# User Set Variables

URLPREFIX='YOUR-PREFIX-HERE'
URLSUFFIX='YOUR-SUFFIX-HERE'
SECRET="YOUR-API-CREDENTIALS-HERE"
CLIENT="api-client"
USERNAME="WALLAROO-USERNAME-HERE"
PASSWORD="WALLAROO-PASSWORD-HERE"

# Derived variables
TOKENURL=f'https://{URLPREFIX}.keycloak.{URLSUFFIX}/auth/realms/master/protocol/openid-connect/token'
APIURL=f"https://{URLPREFIX}.api.{URLSUFFIX}/v1/api"

def get_jwt_token(url, client, secret, username, password):
    auth = HTTPBasicAuth(client, secret)
    data = {
        'grant_type': 'password',
        'username': username,
        'password': password
    }
    response = requests.post(url, auth=auth, data=data, verify=True)
    return response.json()['access_token']

TOKEN=get_jwt_token(TOKENURL, CLIENT, SECRET, USERNAME, PASSWORD)
print(TOKEN)
```

    eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJibVMxOWtsa2NLZlFyMzUxTlVPeVlKSzdJNzhkQ3Rza0lYTENOOWx2SUpJIn0.eyJleHAiOjE2Njk4MjUzNjMsImlhdCI6MTY2OTgyMTc2MywianRpIjoiZTdmZmVlY2EtNTNiOC00Yjk1LTk5MDEtYzA4ODUwZGEyYjMyIiwiaXNzIjoiaHR0cHM6Ly9zcGFya2x5LWFwcGxlLTMwMjYua2V5Y2xvYWsud2FsbGFyb28uY29tbXVuaXR5L2F1dGgvcmVhbG1zL21hc3RlciIsImF1ZCI6WyJtYXN0ZXItcmVhbG0iLCJhY2NvdW50Il0sInN1YiI6Ijc0YTQxMDlhLTk3OTgtNGQ3Yy05OGJlLTYyZDkzODBjOTYwNiIsInR5cCI6IkJlYXJlciIsImF6cCI6ImFwaS1jbGllbnQiLCJzZXNzaW9uX3N0YXRlIjoiNDgxMWZkOGQtMjgzNy00MWE4LWEyNjctNjA3NGQ1YTU3ZTg3IiwiYWNyIjoiMSIsImFsbG93ZWQtb3JpZ2lucyI6WyIqIl0sInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJjcmVhdGUtcmVhbG0iLCJkZWZhdWx0LXJvbGVzLW1hc3RlciIsIm9mZmxpbmVfYWNjZXNzIiwiYWRtaW4iLCJ1bWFfYXV0aG9yaXphdGlvbiJdfSwicmVzb3VyY2VfYWNjZXNzIjp7Im1hc3Rlci1yZWFsbSI6eyJyb2xlcyI6WyJ2aWV3LXJlYWxtIiwidmlldy1pZGVudGl0eS1wcm92aWRlcnMiLCJtYW5hZ2UtaWRlbnRpdHktcHJvdmlkZXJzIiwiaW1wZXJzb25hdGlvbiIsImNyZWF0ZS1jbGllbnQiLCJtYW5hZ2UtdXNlcnMiLCJxdWVyeS1yZWFsbXMiLCJ2aWV3LWF1dGhvcml6YXRpb24iLCJxdWVyeS1jbGllbnRzIiwicXVlcnktdXNlcnMiLCJtYW5hZ2UtZXZlbnRzIiwibWFuYWdlLXJlYWxtIiwidmlldy1ldmVudHMiLCJ2aWV3LXVzZXJzIiwidmlldy1jbGllbnRzIiwibWFuYWdlLWF1dGhvcml6YXRpb24iLCJtYW5hZ2UtY2xpZW50cyIsInF1ZXJ5LWdyb3VwcyJdfSwiYWNjb3VudCI6eyJyb2xlcyI6WyJtYW5hZ2UtYWNjb3VudCIsIm1hbmFnZS1hY2NvdW50LWxpbmtzIiwidmlldy1wcm9maWxlIl19fSwic2NvcGUiOiJlbWFpbCBwcm9maWxlIiwic2lkIjoiNDgxMWZkOGQtMjgzNy00MWE4LWEyNjctNjA3NGQ1YTU3ZTg3IiwiZW1haWxfdmVyaWZpZWQiOnRydWUsImh0dHBzOi8vaGFzdXJhLmlvL2p3dC9jbGFpbXMiOnsieC1oYXN1cmEtdXNlci1pZCI6Ijc0YTQxMDlhLTk3OTgtNGQ3Yy05OGJlLTYyZDkzODBjOTYwNiIsIngtaGFzdXJhLWRlZmF1bHQtcm9sZSI6InVzZXIiLCJ4LWhhc3VyYS1hbGxvd2VkLXJvbGVzIjpbInVzZXIiXSwieC1oYXN1cmEtdXNlci1ncm91cHMiOiJ7fSJ9LCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJqb2huLmhhbnNhcmlja0B3YWxsYXJvby5haSIsImVtYWlsIjoiam9obi5oYW5zYXJpY2tAd2FsbGFyb28uYWkifQ.Hg-ChAQijlKcHzTGkBWC9xf8EYnt8Vsc7qN_DGptMd-GYk_cGS5T0CN8k1XetN8RrFiTE_bWNmnDx5eqEHafa0P4eA_hu-omgwhjkC0VgacayxdgmvFMBgVlWVb5xR8KQBVWc_OeDACvVFRUzrl21tYNy3IW3KWo084bhKDWUJRRvAN5pmwPeN3ese9tia7ZfFbKnEliVCBT74fLmsiJICxpd8tM2Vk9NE7mpMrMsNQI3YgsYJjpntvAiTir6VIYhPAzccrvtJWozuYdPVmrqfkf5ILEzeS1Y81A3W6NsLGeND38twC5LHOLxd4m3op1hzb-RKDgXhvF9uMVjrHuOA

### API Request Methods

All Wallaroo API endpoints follow the format:

* `https://$URLPREFIX.api.$URLSUFFIX/v1/api$COMMAND`

Where `$COMMAND` is the specific endpoint.  For example, for the command to list of workspaces in the Wallaroo instance would use the above format based on these settings:

* `$URLPREFIX`: `smooth-moose-1617`
* `$URLSUFFIX`: `wallaroo.community`
* `$COMMAND`: `/workspaces/list`

This would create the following API endpoint:

* `https://smooth-moose-1617.api.wallaroo.community/v1/api/workspaces/list`

The following methods are used to connect to the Wallaroo API, and the external URL inference endpoints.

```python
# This can either submit a plain POST request ('Content-Type':'application/json'), or with a file.

def get_wallaroo_response(url, api_request, token, data, files=None):
    apiUrl=f"{url}{api_request}"
    if files is None:
        # Regular POST request
        headers= {
            'Authorization': 'Bearer ' + token,
            'Content-Type':'application/json'
        }
        response = requests.post(apiUrl, json=data, headers=headers, verify=True)
    else:
        # POST request with file
        headers= {
            'Authorization': 'Bearer ' + token
        }
        response = requests.post(apiUrl, data=data, headers=headers, files=files, verify=True)
    return response.json()
```

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

    {'workspaces': [{'id': 5,
       'name': 'john.hansarick@wallaroo.ai - Default Workspace',
       'created_at': '2022-11-29T19:01:47.558267+00:00',
       'created_by': '74a4109a-9798-4d7c-98be-62d9380c9606',
       'archived': False,
       'models': [],
       'pipelines': []},
      {'id': 15,
       'name': 'alohaworkspace',
       'created_at': '2022-11-29T19:24:16.056468+00:00',
       'created_by': '74a4109a-9798-4d7c-98be-62d9380c9606',
       'archived': False,
       'models': [3],
       'pipelines': [5]},
      {'id': 16,
       'name': 'urldemoworkspace',
       'created_at': '2022-11-30T15:19:57.293347+00:00',
       'created_by': '74a4109a-9798-4d7c-98be-62d9380c9606',
       'archived': False,
       'models': [4],
       'pipelines': [8]}]}

```python
workspaceList = response['workspaces']
workspaceId = list(filter(lambda x:x["name"]=="urldemoworkspace",workspaceList))[0]['id']
workspaceId
```

    16

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

    'https://sparkly-apple-3026.api.wallaroo.community/v1/api/pipelines/infer/urldemopipeline-5'

### Perform Inference Through External URL

The inference can now be performed through the External Inference URL.  This URL will accept the same inference data file that is used with the Wallaroo SDK, or with an Internal Inference URL as used in the Internal Pipeline Inference URL Tutorial.

For this example, the `externalUrl` retrieved through the [Get External Inference URL](#get-external-inference-url) is used to submit a single inference request through the data file `data-1.json`.

```python
## Inference through external URL

# retrieve the json data to submit
data = json.load(open('./data-1k.json','rb'))

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

    [{"model_name": "urldemomodel", "model_version": "47299047-a3e0-4637-9b87-cc243f4552f3", "pipeline_name": "urldemopipeline", "outputs": [{"Float": {"v": 1, "dim": [1000, 1], "data": [0.001519581419415772, 2.8375030524330214e-05, 3.0770573289373715e-07, 8.822828535468008e-13, 5.48706066183513e-06, 8.

Wallaroo supports the ability to perform inferences through the SDK and through the API for each deployed pipeline.  For more information on how to use Wallaroo, see the [Wallaroo Documentation Site](https://docs.wallaroo.ai) for full details.

##
