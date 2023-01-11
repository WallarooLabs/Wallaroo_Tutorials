import torch
import pickle
import wallaroo

import os
import numpy as np
import json
import requests
import time
import imutils

from cv_store.cv_demo_utils import CVDemo
from cv_store.store import Store

wl = wallaroo.Client()
ws = wl.list_workspaces()
for w in ws:
    if w.name() == 'computer-vision':
        wl.set_current_workspace(w)
        
        
model_name = 'mobilenet'
onnx_model_path = 'models/mobilenet.pt.onnx'
mobilenet_model = wl.upload_model(model_name, onnx_model_path)

deployment_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(2).memory("12Gi").build()

pipeline_name = 'mobilenet-pp'
pipeline = wl.build_pipeline(pipeline_name) \
            .add_model_step(mobilenet_model) \
            .deploy(deployment_config = deployment_config)

time.sleep(5) # needed to allow the pipeline to settle in.
pipeline_url = pipeline._deployment._url()
print(pipeline_url)

cvDemo = CVDemo()

store = Store("Just Walk Out")

camera = {
    'name' : 'grocery',
    'src-loc' : 'videos/amazon-fresh-go.mp4',
    'dest-loc' : 'videos/grocery-amazon-fresh-go-inferenced.mp4',
    'fps' : 15,
    'width' :  640,
    'height' : 480,
    'endpoint-url' : pipeline_url,
    'inference' : 'WALLAROO_SDK', # "ONNX" or "WALLAROO_API" or "WALLAROO_SDK"
    'pipeline_name' : pipeline_name,
    'pipeline' : pipeline,
    'model_name' : model_name,
    'confidence-target' : 0.75,
    'color': CVDemo.AMBER
}
store.addCamera(camera)


enabled = True
#store.setObjectDetection(enabled)
#store.setAnomolyDetection(enabled)
#store.setEnableDriftDetection(enabled)
#store.setTrack(['person','hand bag'])
store.startCameras()

    