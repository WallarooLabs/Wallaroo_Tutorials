import wallaroo
from wallaroo.object import EntityNotFoundError
import pandas as pd
import pyarrow as pa
import requests

wl = wallaroo.Client()

# Setting variables for later steps

workspace_name = 'orchestrationworkspace'
pipeline_name = 'orchestrationpipeline'
model_name = 'orchestrationmodel'
model_file_name = './models/rf_model.onnx'
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

workspace = get_workspace(workspace_name)
wl.set_current_workspace(workspace)

pipeline = get_pipeline(pipeline_name)

#deploy the pipeline
pipeline.deploy()

# Get the connection - assuming it will be the only one

connection = workspace.list_connections()[0]

# Deploy the pipeline 
pipeline.deploy()

# Retrieve the file
# set accept as apache arrow table
headers = {
    'Accept': 'application/vnd.apache.arrow.file'
}

response = requests.get(
                    connection.details()['host'], 
                    headers=headers
                )

# Arrow table is retrieved 
with pa.ipc.open_file(response.content) as reader:
    arrow_table = reader.read_all()

# Perform the inference
pipeline.infer(arrow_table)

# Undeploy the pipeline and return the resources back to the Wallaroo instance
pipeline.undeploy()

