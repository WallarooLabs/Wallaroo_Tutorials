import wallaroo
import pandas as pd
import pyarrow as pa
import requests
import time

wl = wallaroo.Client()

# Setting variables for later steps

# get the arguments
arguments = wl.task_args()

if "workspace_name" in arguments:
    workspace_name = arguments['workspace_name']
else:
    workspace_name="simpleorchestrationworkspace"

if "pipeline_name" in arguments:
    pipeline_name = arguments['pipeline_name']
else:
    pipeline_name="simpleorchestrationpipeline"

if "connection_name" in arguments:
    connection_name = arguments['connection_name']
else:
    connection_name = "external_inference_connection"

print(f"Getting the workspace {workspace_name}", flush=True)
workspace = wl.get_workspace(workspace_name)
wl.set_current_workspace(workspace)


print(f"Getting the pipeline {pipeline_name}", flush=True)
pipeline = wl.get_pipeline(pipeline_name)

print("Deploy the pipeline.")
deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(0.5).memory("1Gi").build()

pipeline.deploy(deployment_config=deploy_config, wait_for_status=False)

# verify the pipeline is running

while pipeline.status()['status'] != 'Running':
    time.sleep(15)
    print("Waiting for deployment.", flush=True)

print(pipeline.status())

# Get the connection - assuming it will be the only one

inference_source_connection = wl.get_connection(name=connection_name)

# our continuous loop - check every minute for the file.
# in a real example, this would be a database with a search filter to inference on the latest
# updates, then store the inference results in another data store

while True:
    print(f"Getting arrow table file", flush=True)
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

    print("Inference time.  Displaying results after.", flush=True)
    # Perform the inference
    result = pipeline.infer(arrow_table)

    result_dataframe = result.to_pandas()

    print(result_dataframe.head(5), flush=True)
    
    time.sleep(60)


