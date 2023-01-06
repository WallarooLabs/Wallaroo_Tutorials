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
import signal

from CVDemoUtils import CVDemo

wl = wallaroo.Client()
ws = wl.list_workspaces()
for w in ws:
    if w.name() == 'computer-vision':
        wl.set_current_workspace(w)

model_name = 'mobilenet'
onnx_model_path = 'models/mobilenet.pt.onnx'
mobilenet_model = wl.upload_model(model_name, onnx_model_path)

deployment_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(2).memory("8Gi").build()

pipeline_name = 'camera-2-pp'
pipeline = wl.build_pipeline(pipeline_name) \
            .add_model_step(mobilenet_model) \
            .deploy(deployment_config = deployment_config)


time.sleep(5) # needed to allow the pipeline to settle in.
url = pipeline._deployment._url()
print(url)

cvDemo = CVDemo()

# The size the image will be resized to
width = 640
height = 480

input_video = "videos/amazon-fresh-go.mp4"
#input_video = "videos/camera2.mp4"
output_video = "videos/amazon-fresh-go-inferenced.mp4"
save_frames_path = "images/output"
#input_video = "videos/ww2-warbirds-in-formation.mp4"
#output_video = "videos/ww2-warbirds-in-formation-inferenced.mp4"

config = {
    'input-video' : input_video, # source video
    'output-video' : output_video, # show the input video with the inferenced results drawn on each frame
    'save-frames-path' : save_frames_path, # show the input video with the inferenced results drawn on each frame
    'fps' : 15, # Frames per second
    'endpoint-url' : url, # the pipelines rest api endpoint url
    'width' : width, # the width of the url
    'height' : height, # the height of the url
    #'max-frames' : 400, # the # of frames to capture in the output video
    'skip-frames' : 225, # the # of frames to capture in the output video
    'confidence-target' : 0.75, # only display bounding boxes with confidence > provided #
    'color':CVDemo.CYAN, # color to draw bounding boxes and the text in the statistics
    'inference' : 'WALLAROO_SDK', # "ONNX" or "WALLAROO_API" or "WALLAROO_SDK"
    'onnx_model_path' : onnx_model_path,
    'model_name' : model_name,
    'pipeline' : pipeline, # provide this when using inference WALLAROO_SDK 
    'pipeline_name' : pipeline_name,
    'skip-frames-list' : [ (440,460), (1400,1430)]
#    'record-start-frame' : 225, # the # of frames to capture in the output video
#    'record-end-frame' : 275, # the # of frames to capture in the output video  
}
cvDemo.DEBUG = False
cvDemo.detectAndClassifyObjectsInVideo(config)
print("We are done.")


def receiveSignal(signalNumber, frame):
    print('Received:', signalNumber)
    return

