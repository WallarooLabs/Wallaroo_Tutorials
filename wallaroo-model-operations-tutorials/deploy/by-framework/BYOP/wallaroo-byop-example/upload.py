# disable logging output from byop imports
import logging

import numpy as np
import pandas as pd
import pyarrow as pa
import wallaroo

from wallaroo.deployment_config import DeploymentConfigBuilder
from wallaroo.framework import Framework

# connect to Wallaroo
wl = wallaroo.Client()

# set up the workspace
workspace_name = f'sample-byop-best-practices'
workspace = wl.get_workspace(name=workspace_name, create_if_not_exist=True)
wl.set_current_workspace(workspace)

# upload the model
model_name = "sample-byop-model"
model_file_name = "./byop-sample.zip"

input_schema = pa.schema([
    pa.field('input_number', pa.int64()),
    pa.field('id', pa.int64())
])

output_schema = pa.schema([
    pa.field('result', pa.int64()),
    pa.field('id', pa.int64())
])

model = wl.upload_model(model_name, 
                        model_file_name, 
                        framework=Framework.CUSTOM, 
                        input_schema=input_schema, 
                        output_schema=output_schema,
                        convert_wait=True)

print(model)

# create the pipeline
pipeline = wl.build_pipeline("byop-sample-pipeline")


pipeline.clear()
pipeline.add_model_step(model)

# deploy the pipeline
deployment_config = DeploymentConfigBuilder() \
    .cpus(0.25).memory('512Mi') \
    .sidekick_cpus(model, 1) \
    .sidekick_memory(model, '1Gi') \
    .build()

pipeline.deploy(deployment_config=deployment_config, wait_for_status=False)

# wait until the pipeline is deployed

import time
time.sleep(15)

while pipeline.status()['status'] != 'Running':
    time.sleep(15)
    print("Waiting for deployment.")
print(pipeline.status()['status'])

# perform the inference
input_df = pd.DataFrame({
    "input_number": [1,2,3],
    "id": [20000000004093819,20012684296980773,481562342]
})

print("Input:")
print(input_df)

print("Result:")
print(pipeline.infer(input_df))

_= pipeline.undeploy()