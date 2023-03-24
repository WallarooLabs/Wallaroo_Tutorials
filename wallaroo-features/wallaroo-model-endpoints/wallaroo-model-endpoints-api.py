# %% [markdown]
# This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-features/wallaroo-model-endpoints).
# 
# ## Wallaroo API Inference Tutorial
# 
# Wallaroo provides the ability to perform inferences through deployed pipelines via the Wallaroo SDK and the Wallaroo MLOps API.  This tutorial demonstrates performing inferences using the Wallaroo MLOps API.
# 
# This tutorial provides the following:
# 
# * `ccfraud.onnx`:  A pre-trained credit card fraud detection model.
# * `data/cc_data_1k.arrow`, `data/cc_data_10k.arrow`: Sample testing data in Apache Arrow format with 1,000 and 10,000 records respectively.
# * `wallaroo-model-endpoints-api.py`:  A code-only version of this tutorial as a Python script.
# 
# This tutorial and sample data comes from the Machine Learning Group's demonstration on [Credit Card Fraud detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
# 
# ### Prerequisites
# 
# The following is required for this tutorial:
# 
# * A [deployed Wallaroo instance](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-install-guides/) with [Model Endpoints Enabled](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-model-endpoints-guide/)
# * The following Python libraries:
#   * `os`
#   * `requests`
#   * [`pandas`](https://pypi.org/project/pandas/)
#   * [`polars`](https://pypi.org/project/polars/)
#   * [`pyarrow`](https://pypi.org/project/pyarrow/)
#   * [`wallaroo`](https://pypi.org/project/wallaroo/) (Installed in the Wallaroo JupyterHub service by default).
# 
# ### Tutorial Goals
# 
# This demonstration provides a quick tutorial on performing inferences using the Wallaroo MLOps API using a deployed pipeline's Inference URL.  This following steps will be performed:
# 
# * Connect to a Wallaroo instance using the Wallaroo SDK and environmental variables.  This bypasses the browser link confirmation for a seamless login, and provides a simple method of retrieving the JWT token used for Wallaroo MLOps API calls.  For more information, see the [Wallaroo SDK Essentials Guide:  Client Connection](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/) and the [Wallaroo MLOps API Essentials Guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-api-essential-guide/).
# * Create a workspace for our models and pipelines.
# * Upload the `ccfraud` model.
# * Create a pipeline and add the `ccfraud` model as a pipeline step.
# * Run sample inferences with pandas DataFrame inputs and Apache Arrow inputs.

# %% [markdown]
# ### Retrieve Token
# 
# There are two methods of retrieving the JWT token used to authenticate to the Wallaroo instance's API service:
# 
# * [Wallaroo SDK](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-api-essential-guide/#through-the-wallaroo-sdk).  This method requires a Wallaroo based user.
# * [API Clent Secret](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-api-essential-guide/#through-keycloak).  This is the recommended method as it is user independent.  It allows any valid user to make an inference request.
# 
# This tutorial will use the Wallaroo SDK method for convenience with environmental variables for a seamless login without browser validation.  For more information, see the [Wallaroo SDK Essentials Guide: Client Connection](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

# %% [markdown]
# ### API Request Methods
# 
# All Wallaroo API endpoints follow the format:
# 
# * `https://$URLPREFIX.api.$URLSUFFIX/v1/api$COMMAND`
# 
# Where `$COMMAND` is the specific endpoint.  For example, for the command to list of workspaces in the Wallaroo instance would use the above format based on these settings:
# 
# * `$URLPREFIX`: `smooth-moose-1617`
# * `$URLSUFFIX`: `example.wallaroo.ai`
# * `$COMMAND`: `/workspaces/list`
# 
# This would create the following API endpoint:
# 
# * `https://smooth-moose-1617.api.example.wallaroo.ai/v1/api/workspaces/list`

# %% [markdown]
# ### Connect to Wallaroo
# 
# For this example, a connection to the Wallaroo SDK is used.  This will be used to retrieve the JWT token for the MLOps API calls.  
# 
# This example will store the user's credentials into the file `./creds.json` which contains the following:
# 
# ```json
# {
#     "username": "{Connecting User's Username}", 
#     "password": "{Connecting User's Password}", 
#     "email": "{Connecting User's Email Address}"
# }
# ```
# 
# Replace the `username`, `password`, and `email` fields with the user account connecting to the Wallaroo instance.  This allows a seamless connection to the Wallaroo instance and bypasses the standard browser based confirmation link.  For more information, see the [Wallaroo SDK Essentials Guide:  Client Connection](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).
# 
# Update `wallarooPrefix = "YOUR PREFIX"` and `wallarooSuffix = "YOUR SUFFIX"` to match the Wallaroo instance used for this demonstration.

# %%
import wallaroo
from wallaroo.object import EntityNotFoundError
import pandas as pd
import os

import requests
from requests.auth import HTTPBasicAuth

# Used to create unique workspace and pipeline names
import string
import random

# make a random 4 character prefix
prefix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))

import json

# used to display dataframe information without truncating
from IPython.display import display
pd.set_option('display.max_colwidth', None)

# %%
# Retrieve the login credentials.
os.environ["WALLAROO_SDK_CREDENTIALS"] = './creds.json'

# Client connection from local Wallaroo instance

wl = wallaroo.Client(auth_type="user_password")

# Login from external connection

# wallarooPrefix = "YOUR PREFIX"
# wallarooSuffix = "YOUR SUFFIX"

# wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}.api.{wallarooSuffix}", 
#                     auth_endpoint=f"https://{wallarooPrefix}.keycloak.{wallarooSuffix}", 
#                     auth_type="user_password")

# %%
APIURL=f"https://{wallarooPrefix}.api.{wallarooSuffix}"

# %% [markdown]
# ## Retrieve the JWT Token
# 
# As mentioned earlier, there are multiple methods of authenticating to the Wallaroo instance for MLOps API calls.  This tutorial will use the Wallaroo SDK method Wallaroo Client `wl.mlops().__dict__` method, extracting the token from the response.
# 
# Reference:  [MLOps API Retrieve Token Through Wallaroo SDK](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-api-essential-guide/#through-the-wallaroo-sdk)

# %%
# Retrieve the token
connection =wl.mlops().__dict__
token = connection['token']
display(token)

# %% [markdown]
# ## Create Workspace
# 
# In a production environment, the Wallaroo workspace that contains the pipeline and models would be created and deployed.  We will quickly recreate those steps using the MLOps API.  If the workspace and pipeline have already been created through the [Wallaroo SDK Inference Tutorial](https://staging.docs.wallaroo.ai/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-endpoints/wallaroo-external-inference-tutorial/), then we can skip directly to [Deploy Pipeline](#deploy-pipeline).
# 
# Workspaces are created through the MLOps API with the `/v1/api/workspaces/create` command.  This requires the workspace name be provided, and that the workspace not already exist in the Wallaroo instance.
# 
# Reference: [MLOps API Create Workspace](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-api-essential-guide/wallaroo-mlops-api-essential-guide-workspaces/#create-workspace)

# %%
# Retrieve the token
connection =wl.mlops().__dict__
token = connection['token']

# Create workspace
apiRequest = f"{APIURL}/v1/api/workspaces/create"

workspace_name = f"{prefix}apiinferenceexampleworkspace"

data = {
  "workspace_name": workspace_name
}

headers= {
    'Authorization': 'Bearer ' + token,
    'Content-Type':'application/json'
}

response = requests.post(apiRequest, json=data, headers=headers, verify=True).json()
display(response)
# Stored for future examples
workspaceId = response['workspace_id']

# %% [markdown]
# ## Upload Model
# 
# The model is uploaded using the `/v1/api/models/upload` command.  This uploads a ML Model to a Wallaroo workspace via POST with `Content-Type: multipart/form-data` and takes the following parameters:
# 
# * **Parameters**
#   * **name** - (*REQUIRED string*): Name of the model
#   * **visibility** - (*OPTIONAL string*): The visibility of the model as either `public` or `private`.
#   * **workspace_id** - (*REQUIRED int*): The numerical id of the workspace to upload the model to.  Stored earlier as `workspaceId`.
# 
# Directly after we will use the `/models/list_versions` to retrieve model details used for later steps.
# 
# Reference: [Wallaroo MLOps API Essentials Guide: Model Management: Upload Model to Workspace](https://staging.docs.wallaroo.ai/202301/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-api-essential-guide/wallaroo-mlops-api-essential-guide-models/#upload-model-to-workspace)

# %%
# Upload Model

# Retrieve the token
connection =wl.mlops().__dict__
token = connection['token']

apiRequest = f"{APIURL}/v1/api/models/upload"

model_name = f"{prefix}ccfraud"

data = {
    "name":model_name,
    "visibility":"public",
    "workspace_id": workspaceId
}

headers= {
    'Authorization': 'Bearer ' + token
}

files = {
    "file": ('ccfraud.onnx', open('./ccfraud.onnx','rb'))
}

response = requests.post(apiRequest, data=data, headers=headers, files=files, verify=True).json()
display(response)

modelId=response['insert_models']['returning'][0]['models'][0]['id'] # Stored for later steps.

# %%
# Retrieve uploaded model details

# Retrieve the token
connection =wl.mlops().__dict__
token = connection['token']

apiRequest = f"{APIURL}/v1/api/models/list_versions"

headers= {
    'Authorization': 'Bearer ' + token,
    'Content-Type':'application/json'
}

data = {
  "model_id": model_name,
  "models_pk_id": modelId
}

response = requests.post(apiRequest, json=data, headers=headers, verify=True).json()
display(response)

# Stored for future examples - get the last model id information

exampleModelVersion = response[-1]['model_version']
exampleModelSha = response[-1]['sha']

# %% [markdown]
# ## Create Pipeline
# 
# Create Pipeline in a Workspace with the `/v1/api/pipelines/create` command.  This creates a new pipeline in the specified workspace.
# 
# * **Parameters**
#   * **pipeline_id** - (REQUIRED string): Name of the new pipeline.
#   * **workspace_id** - (REQUIRED int): Numerical id of the workspace for the new pipeline.  Stored earlier as `workspaceId`.
#   * **definition** - (REQUIRED string): Pipeline definitions, can be `{}` for none.
# 
# Reference: [Wallaroo MLOps API Essentials Guide: Pipeline Management: Create Pipeline in a Workspace](https://staging.docs.wallaroo.ai/202301/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-api-essential-guide/wallaroo-mlops-api-essential-guide-pipelines/#create-pipeline-in-a-workspace)

# %%
# Create pipeline

# Retrieve the token
connection =wl.mlops().__dict__
token = connection['token']

apiRequest = f"{APIURL}/v1/api/pipelines/create"

pipeline_name=f"{prefix}apiinferenceexamplepipeline"

data = {
  "pipeline_id": pipeline_name,
  "workspace_id": workspaceId,
  "definition": {}
}

headers= {
    'Authorization': 'Bearer ' + token,
    'Content-Type':'application/json'
}

response = requests.post(apiRequest, json=data, headers=headers, verify=True).json()

pipeline_id = response['pipeline_pk_id']
pipeline_variant_id=response['pipeline_variant_pk_id']
pipeline_variant_version=['pipeline_variant_version']

# %% [markdown]
# ## Deploy Pipeline
# 
# With the pipeline created and the model uploaded into the workspace, the pipeline can be deployed.  This will allocate resources from the Kubernetes cluster hosting the Wallaroo instance and prepare the pipeline to process inference requests.
# 
# Pipelines are deployed through the MLOps API command `/v1/api/pipelines/deploy` which takes the following parameters:
# 
# * **Parameters**
#   * **deploy_id** (*REQUIRED string*): The name for the pipeline deployment.
#   * **engine_config** (*OPTIONAL string*): Additional configuration options for the pipeline.
#   * **pipeline_version_pk_id** (*REQUIRED int*): Pipeline version id.  Captured earlier as `pipeline_variant_id`.
#   * **model_configs** (*OPTIONAL Array int*): Ids of model configs to apply.
#   * **model_ids** (*OPTIONAL Array int*): Ids of models to apply to the pipeline.  If passed in, model_configs will be created automatically.
#   * **models** (*OPTIONAL Array models*):  If the model ids are not available as a pipeline step, the models' data can be passed to it through this method.  The options below are only required if `models` are provided as a parameter.
#     * **name** (*REQUIRED string*): Name of the uploaded model that is in the same workspace as the pipeline.  Captured earlier as the `model_name` variable.
#     * **version** (*REQUIRED string*): Version of the model to use.  
#     * **sha** (*REQUIRED string*): SHA value of the model.
#   * **pipeline_id** (*REQUIRED int*): Numerical value of the pipeline to deploy.
# * **Returns**
#   * **id** (*int*): The deployment id.
# 
# Reference: [Wallaroo MLOps API Essentials Guide: Pipeline Management: Deploy a Pipeline](https://staging.docs.wallaroo.ai/202301/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-api-essential-guide/wallaroo-mlops-api-essential-guide-pipelines/#deploy-a-pipeline)

# %%
# Deploy Pipeline

# Retrieve the token
connection =wl.mlops().__dict__
token = connection['token']

apiRequest = f"{APIURL}/v1/api/pipelines/deploy"

exampleModelDeployId=pipeline_name

headers= {
    'Authorization': 'Bearer ' + token,
    'Content-Type':'application/json'
}

data = {
    "deploy_id": exampleModelDeployId,
    "pipeline_version_pk_id": pipeline_variant_id,
    "models": [
        {
            "name":model_name,
            "version":exampleModelVersion,
            "sha":exampleModelSha
        }
    ],
    "pipeline_id": pipeline_id
}


response = requests.post(apiRequest, json=data, headers=headers, verify=True).json()
display(response)
exampleModelDeploymentId=response['id']

# %% [markdown]
# ## Get External Inference URL
# 
# The API command `/admin/get_pipeline_external_url` retrieves the external inference URL for a specific pipeline in a workspace.
# 
# * **Parameters**
#   * **workspace_id** (*REQUIRED integer*):  The workspace integer id.
#   * **pipeline_name** (*REQUIRED string*): The name of the pipeline.
# 
# In this example, a list of the workspaces will be retrieved.  Based on the setup from the Internal Pipeline Deployment URL Tutorial, the workspace matching `urlworkspace` will have it's **workspace id** stored and used for the `/admin/get_pipeline_external_url` request with the pipeline `urlpipeline`.
# 
# The External Inference URL will be stored as a variable for the next step.
# 
# Reference: [Wallaroo MLOps API Essentials Guide: Pipeline Management: Get External Inference URL](https://staging.docs.wallaroo.ai/202301/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-api-essential-guide/wallaroo-mlops-api-essential-guide-pipelines/#get-external-inference-url)

# %%
# Retrieve the token
connection =wl.mlops().__dict__
token = connection['token']

## Retrieve the pipeline's External Inference URL

apiRequest = f"{APIURL}/v1/api/admin/get_pipeline_external_url"

data = {
    "workspace_id": workspaceId,
    "pipeline_name": pipeline_name
}

headers= {
    'Authorization': 'Bearer ' + token,
    'Content-Type':'application/json'
}

response = requests.post(apiRequest, json=data, headers=headers, verify=True).json()
externalUrl = response['url']
externalUrl

# %% [markdown]
# ### Perform Inference Through External URL
# 
# The inference can now be performed through the External Inference URL.  This URL will accept the same inference data file that is used with the Wallaroo SDK, or with an Internal Inference URL as used in the Internal Pipeline Inference URL Tutorial.
# 
# For this example, the `externalUrl` retrieved through the [Get External Inference URL](#get-external-inference-url) is used to submit a single inference request through the data file `data-1.json`.
# 
# Reference: [Wallaroo MLOps API Essentials Guide: Pipeline Management: Perform Inference Through External URL](https://staging.docs.wallaroo.ai/202301/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-api-essential-guide/wallaroo-mlops-api-essential-guide-pipelines/#perform-inference-through-external-url)

# %%
# Retrieve the token
connection =wl.mlops().__dict__
token = connection['token']

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

# set the headers
headers= {
    'Authorization': 'Bearer ' + token,
    'Content-Type':'application/json; format=pandas-records'
}

# submit the request via POST, import as pandas DataFrame
response = pd.DataFrame.from_records(requests.post(externalUrl, json=data, headers=headers).json())
display(response)

# %%
# Retrieve the token
connection =wl.mlops().__dict__
token = connection['token']

# Submit arrow file
dataFile="./data/cc_data_10k.arrow"

data = open(dataFile,'rb').read()


contentType="application/vnd.apache.arrow.file"

# set the headers
headers= {
    'Authorization': 'Bearer ' + token,
    'Content-Type':'application/vnd.apache.arrow.file'
}

response = pd.DataFrame.from_records(requests.post(externalUrl, headers=headers, data=data, verify=True).json())
display(response.head(5))

# %% [markdown]
# ### Undeploy the Pipeline
# 
# With the tutorial complete, we'll undeploy the pipeline with `/v1/api/pipelines/undeploy` and return the resources back to the Wallaroo instance.
# 
# Reference: [Wallaroo MLOps API Essentials Guide: Pipeline Management: Undeploy a Pipeline](https://staging.docs.wallaroo.ai/202301/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-api-essential-guide/wallaroo-mlops-api-essential-guide-pipelines/#undeploy-a-pipeline)

# %%
# Retrieve the token
connection =wl.mlops().__dict__
token = connection['token']

apiRequest = f"{APIURL}/v1/api/pipelines/undeploy"

data = {
    "pipeline_id": pipeline_id,
    "deployment_id":exampleModelDeploymentId
}

# set the headers
headers= {
    'Authorization': 'Bearer ' + token,
    'Content-Type':'application/json; format=pandas-records'
}

response = requests.post(apiRequest, json=data, headers=headers, verify=True).json()
display(response)

# %% [markdown]
# Wallaroo supports the ability to perform inferences through the SDK and through the API for each deployed pipeline.  For more information on how to use Wallaroo, see the [Wallaroo Documentation Site](https://docs.wallaroo.ai) for full details.

# %% [markdown]
# ##


