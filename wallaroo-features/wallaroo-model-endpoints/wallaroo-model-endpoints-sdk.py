#!/usr/bin/env python

import wallaroo
from wallaroo.object import EntityNotFoundError
import pandas as pd
import os

# used for the Wallaroo 2023.1 Wallaroo SDK for Arrow support
os.environ["ARROW_ENABLED"]="True"

# used to display dataframe information without truncating
from IPython.display import display
pd.set_option('display.max_colwidth', None)

import requests

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

workspace_name = 'sdkinferenceexampleworkspace'
pipeline_name = 'sdkinferenceexamplepipeline'
model_name = 'ccfraud'
model_file_name = './ccfraud.onnx'

def get_workspace(name):
    workspace = None
    for ws in wl.list_workspaces():
        if ws.name() == name:
            workspace= ws
    if(workspace == None):
        workspace = wl.create_workspace(name)
    return workspace

def get_pipeline(name):
    try:
        pipeline = wl.pipelines_by_name(pipeline_name)[0]
    except EntityNotFoundError:
        pipeline = wl.build_pipeline(pipeline_name)
    return pipeline

workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)

# Create or select the current pipeline

ccfraudpipeline = get_pipeline(pipeline_name)

# Add ccfraud model as the pipeline step

ccfraud_model = wl.upload_model(model_name, model_file_name).configure()

ccfraudpipeline.add_model_step(ccfraud_model).deploy()

# List the pipelines by name in the current workspace - just the first several to save space.

display("\nList first five pipelines.")
display(wl.list_pipelines()[:5])

# Set the `pipeline` variable to our sample pipeline.

pipeline = wl.pipelines_by_name(pipeline_name)[0]
display("\nCurrent pipeline")
display(pipeline)

# inference via `infer` method
smoke_test = pd.DataFrame.from_records([
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
])
result = pipeline.infer(smoke_test)
display("\nInfer result")
display(result)

#inference via infer_from_file method

result = pipeline.infer_from_file('./data/cc_data_10k.arrow')

display("\nInfer from file result.")
display(result)

# use pyarrow to convert results to a pandas DataFrame and display only the results with > 0.75

import pyarrow as pa

list = [0.75]

outputs =  result.to_pandas()
# display(outputs)
filter = [elt[0] > 0.75 for elt in outputs['out.dense_1']]
outputs = outputs.loc[filter]

display("\nFiltered inferences with pandas")
display(outputs)

# use polars to convert results to a polars DataFrame and display only the results with > 0.75

import polars as pl

outputs =  pl.from_arrow(result)

display("\nFiltered inferences with polars")
display(outputs.filter(pl.col("out.dense_1").apply(lambda x: x[0]) > 0.75))

# Inferences through HTTP POST

# Retrieve the pipeline inference URL

deploy_url = pipeline._deployment._url()
display("\nPipeline Inference URL")
print(deploy_url)

# Retrieve the token
connection =wl.mlops().__dict__
token = connection['token']

## Inference through external URL using DataFrame

# retrieve the json data to submit
data = pd.DataFrame.from_records([
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
])

contentType = 'application/json; format=pandas-records'

# set the headers
headers= {
    'Authorization': 'Bearer ' + token,
    'Content-Type': contentType
}

# submit the request via POST, import as pandas DataFrame
response = pd.DataFrame.from_records(requests.post(deploy_url, data=data.to_json(orient="records"), headers=headers).json())
display("\nInference with DataFrame result")
display(response)

# HTTP inference with Apache Arrow as input

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
    'Content-Type': contentType
}

response = pd.DataFrame.from_records(requests.post(deploy_url, headers=headers, data=data, verify=True).json())
display("\nInference with Arrow table, first 5 results")
display(response.head(5))

# Undeploy the pipeline

pipeline.undeploy()