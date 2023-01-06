import torch
import pickle
import wallaroo
import os
import numpy as np
import json
import requests
import time
import imutils
import sys
 
from CVDemoUtils import CVDemo

wl = wallaroo.Client()

ws = wl.list_workspaces()
for w in ws:
    if w.name() == 'computer-vision':
        wl.set_current_workspace(w)

model_name = 'mobilenet'
onnx_model_path = 'models/mobilenet.pt.onnx'
mobilenet_model = wl.upload_model(model_name, onnx_model_path)

deployment_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(1).memory("8Gi").build()

pipeline_name = 'camera-3-pp'
pipeline = wl.build_pipeline(pipeline_name) \
            .add_model_step(mobilenet_model) \
            .deploy(deployment_config = deployment_config)
time.sleep(5) # needed to allow the pipeline to settle in.
url = pipeline._deployment._url()
pipeline.undeploy()


