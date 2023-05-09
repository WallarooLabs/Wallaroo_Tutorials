import wallaroo
from wallaroo.object import EntityNotFoundError
import pandas as pd
import pyarrow as pa
import requests
import os
#Used for the Wallaroo SDK version 2023.1
os.environ["ARROW_ENABLED"]="True"

wl = wallaroo.Client()

# Setting variables for later steps

workspace_name = 'simpleorchestrationworkspace'
pipeline_name = 'simpleorchestrationpipeline'

# helper methods to retrieve workspaces and pipelines

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
        pipeline = wl.pipelines_by_name(name)[0]
    except EntityNotFoundError:
        pipeline = wl.build_pipeline(name)
    return pipeline

print(f"Getting the workspace {workspace_name}")
workspace = get_workspace(workspace_name)
wl.set_current_workspace(workspace)


print(f"Getting the pipeline {pipeline_name}")
pipeline = get_pipeline(pipeline_name)

# Get the connection - assuming it will be the only one

inference_source_connection = wl.get_connection(name="external_inference_connection")

inference_results_connection = wl.get_connection(name="inference_results_connection")

print(f"Getting arrow table file")
# Retrieve the file
# set accept as apache arrow table
headers = {
    'Accept': 'application/vnd.apache.arrow.file'
}

response = requests.get(
                    inference_source_connection.details()['host'], 
                    headers=headers
                )

# Arrow table is retrieved 
with pa.ipc.open_file(response.content) as reader:
    arrow_table = reader.read_all()

print("Inference time.  Displaying results after.")
# Perform the inference
result = pipeline.infer(arrow_table)

result_dataframe = result.to_pandas()

# # Save result to local file - should be /home/jovyen

# result_dataframe.to_json(inference_results_connection.details()['location'], orient="records")

print(result_dataframe.head(5))
