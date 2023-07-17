import wallaroo
from wallaroo.object import EntityNotFoundError
import pandas as pd
import pyarrow as pa
import requests

wl = wallaroo.Client()

# Setting variables for later steps

# get the arguments
arguments = wl.task_args()

if "workspace_name" in arguments:
    workspace_name = arguments['workspace_name']
else:
    workspace_name="orchestrationworkspace"

if "pipeline_name" in arguments:
    pipeline_name = arguments['pipeline_name']
else:
    pipeline_name="orchestrationpipeline"

if "connection_name" in arguments:
    connection_name = arguments['connection_name']
else:
    connection_name = "houseprice_arrow_table"


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
pipeline.deploy()
# Get the connection - assuming it will be the only one

inference_source_connection = wl.get_connection(name=connection_name)

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
print(result)

pipeline.undeploy()