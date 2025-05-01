import wallaroo
from wallaroo.object import EntityNotFoundError
import pandas as pd
import pyarrow as pa
import requests

wl = wallaroo.Client()

if wl.in_task():
    arguments = wl.task_args()
    print(wl.task_args())

    if "workspace_name" in arguments:
        workspace_name = arguments['workspace_name']
    else:
        workspace_name = 'apiorchestrationworkspace'

    if "pipeline_name" in arguments:
        pipeline_name = arguments['pipeline_name']
    else:
        pipeline_name = 'apipipeline'

else:
    # we're not in the task, so use the default values
    workspace_name = 'apiorchestrationworkspace'
    pipeline_name = 'apipipeline'

# helper methods to retrieve workspaces and pipelines
# 2024.1 - no longer needed.

# def get_workspace(name):
#     workspace = None
#     for ws in wl.list_workspaces():
#         if ws.name() == name:
#             workspace= ws
#     if(workspace == None):
#         workspace = wl.create_workspace(name)
#     return workspace

# def get_pipeline(name):
#     try:
#         pipeline = wl.pipelines_by_name(name)[0]
#     except EntityNotFoundError:
#         pipeline = wl.build_pipeline(name)
#     return pipeline

print(f"Getting the workspace {workspace_name}")
workspace = wl.get_workspace(workspace_name)
wl.set_current_workspace(workspace)


print(f"Getting the pipeline {pipeline_name}")
# this will get the most recent version with the model steps set up
pipeline = wl.get_pipeline(pipeline_name)

# deploy the pipeline

print("Deploying the pipeline.")
pipeline.deploy()

# sample inference
print("Performing sample inference.")
normal_input = pd.DataFrame.from_records({"tensor": [[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]]})
result = pipeline.infer(normal_input)
print(result)

print("Undeploying the pipeline")
pipeline.undeploy()