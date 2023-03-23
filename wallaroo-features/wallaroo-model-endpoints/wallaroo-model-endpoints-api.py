#!/usr/bin/env python

import wallaroo
from wallaroo.object import EntityNotFoundError
import pandas as pd
import os

import requests
from requests.auth import HTTPBasicAuth

import json

# used to display dataframe information without truncating
from IPython.display import display
pd.set_option('display.max_colwidth', None)

# Used to create unique workspace and pipeline names
import string
import random

# make a random 4 character prefix
prefix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))

# Retrieve the login credentials.
os.environ["WALLAROO_SDK_CREDENTIALS"] = './creds.json.example'

# Client connection from local Wallaroo instance

wl = wallaroo.Client(auth_type="user_password")

# Login from external connection

# wallarooPrefix = "YOUR PREFIX"
# wallarooSuffix = "YOUR SUFFIX"

# wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}.api.{wallarooSuffix}", 
#                     auth_endpoint=f"https://{wallarooPrefix}.keycloak.{wallarooSuffix}", 
#                     auth_type="user_password")

APIURL=f"https://{wallarooPrefix}.api.{wallarooSuffix}"

# Retrieve the token
connection =wl.mlops().__dict__
token = connection['token']
display(token)


# Create sample workspace

# Retrieve the token
connection =wl.mlops().__dict__
token = connection['token']

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

# Create pipeline

# Retrieve the token
connection =wl.mlops().__dict__
token = connection['token']

apiRequest = f"{APIURL}/v1/api/pipelines/create"

pipeline_name= f"{prefix}apiinferenceexamplepipeline"

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
display(response)

# Stored for later examples
pipeline_id = response['pipeline_pk_id']
pipeline_variant_id=response['pipeline_variant_pk_id']
pipeline_variant_version=['pipeline_variant_version']

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

# Stored for later examples
exampleModelDeploymentId=response['id']

# Wait 60 seconds for the pipeline to complete deploying. 
# Other methods would be to loop until the pipeline deployment status shows running.

import time

time.sleep(60)

## Retrieve the pipeline's External Inference URL

# Retrieve the token
connection =wl.mlops().__dict__
token = connection['token']

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
display(externalUrl)

## Inference with DataFrame Input

# Retrieve the token
connection =wl.mlops().__dict__
token = connection['token']

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

## Inference with DataFrame Input

# Retrieve the token
connection =wl.mlops().__dict__
token = connection['token']


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

# Undeploy the Pipeline

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
