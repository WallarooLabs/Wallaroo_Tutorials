import wallaroo
from wallaroo.object import EntityNotFoundError
import pandas as pd
import pyarrow as pa
import requests
import time

wl = wallaroo.Client()

# Setting variables for later steps

# get the arguments
arguments = wl.task_args()

pipeline_name_a = 'house-price-zone-a'
pipeline_name_b = 'house-price-zone-b'
pipeline_name_c = 'house-price-zone-c'

if "workspace_name" in arguments:
    workspace_name = arguments['workspace_name']
else:
    workspace_name = 'multiple_pipeline_deployment_tutorial'

if "pipeline_name_a" in arguments:
    pipeline_name_a = arguments['pipeline_name_a']
else:
    pipeline_name_a ="house-price-zone-a"

if "pipeline_name_b" in arguments:
    pipeline_name_b = arguments['pipeline_name_b']
else:
    pipeline_name_b="house-price-zone-b"
    
if "pipeline_name_c" in arguments:
    pipeline_name_c = arguments['pipeline_name_c']
else:
    pipeline_name_c="house-price-zone-c"

    
print(f"Getting the workspace {workspace_name}")
workspace = wl.get_workspace(workspace_name)
wl.set_current_workspace(workspace)


print(f"Getting the pipelines")
pipeline_a = wl.get_pipeline(pipeline_name_a)
pipeline_b = wl.get_pipeline(pipeline_name_b)
pipeline_c = wl.get_pipeline(pipeline_name_c)

deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(0.5).memory("1Gi").build()

pipeline_a.deploy(deployment_config=deploy_config, wait_for_status=False)
pipeline_b.deploy(deployment_config=deploy_config, wait_for_status=False)
pipeline_c.deploy(deployment_config=deploy_config, wait_for_status=False)

while pipeline_a.status()['status'] != 'Running':
    time.sleep(15)
    print(f"Waiting for deployment of {pipeline_name_a}.")

while pipeline_b.status()['status'] != 'Running':
    time.sleep(15)
    print(f"Waiting for deployment of {pipeline_name_b}.")

while pipeline_c.status()['status'] != 'Running':
    time.sleep(15)
    print(f"Waiting for deployment of {pipeline_name_c}.")

# perform the inference

print(pipeline_a.infer_from_file('./data/zonea.df.json').head(20))
print(pipeline_b.infer_from_file('./data/zoneb.df.json').head(20))
print(pipeline_c.infer_from_file('./data/zonec.df.json').head(20))

# undeploy the pipelines

pipeline_a.undeploy()
pipeline_b.undeploy()
pipeline_c.undeploy()

