This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-model-cookbooks/computer-vision).

## Step 01: Detecting Objects Using mobilenet

The following tutorial demonstrates how to use a trained mobilenet model deployed in Wallaroo to detect objects.  This process will use the following steps:

1. Create a Wallaroo workspace and pipeline.
1. Upload a trained mobilenet ML model and add it as a pipeline step.
1. Deploy the pipeline.
1. Perform an inference on a sample image.
1. Draw the detected objects, their bounding boxes, their classifications, and the confidence of the classifications on the provided image.
1. Review our results.

## Steps

### Import Libraries

The first step will be to import our libraries.  Please check with **Step 00: Introduction and Setup** and verify that the necessary libraries and applications are added to your environment.

```python
import torch
import pickle
import wallaroo
from wallaroo.object import EntityNotFoundError
from wallaroo.framework import Framework

import numpy as np
import json
import requests
import time
import pandas as pd
from CVDemoUtils import CVDemo

# used to display dataframe information without truncating
from IPython.display import display
import pandas as pd
pd.set_option('display.max_colwidth', None)

# used for unique connection names

import string
import random
suffix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))
suffix=''
```

### Connect to the Wallaroo Instance

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
# Login through local service

wl = wallaroo.Client()
```

### Set Variables

The following variables and methods are used later to create or connect to an existing workspace, pipeline, and model.

```python
workspace_name = f'mobilenetworkspacetest{suffix}'
pipeline_name = f'mobilenetpipeline{suffix}'
model_name = f'mobilenet{suffix}'
model_file_name = 'models/mobilenet.pt.onnx'
```

```python
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
```

### Create Workspace

The workspace will be created or connected to, and set as the default workspace for this session.  Once that is done, then all models and pipelines will be set in that workspace.

```python
workspace = get_workspace(workspace_name)
wl.set_current_workspace(workspace)
wl.get_current_workspace()
```

    {'name': 'mobilenetworkspacetest', 'id': 8, 'archived': False, 'created_by': '1b26fc10-d0f9-4f92-a1a2-ab342b3d069e', 'created_at': '2023-10-30T18:15:15.756292+00:00', 'models': [], 'pipelines': []}

### Create Pipeline and Upload Model

We will now create or connect to an existing pipeline as named in the variables above.

```python
pipeline = get_pipeline(pipeline_name)
mobilenet_model = wl.upload_model(model_name, model_file_name, framework=Framework.ONNX).configure(batch_config="single", tensor_fields=["tensor"])
```

### Deploy Pipeline

With the model uploaded, we can add it is as a step in the pipeline, then deploy it.  Once deployed, resources from the Wallaroo instance will be reserved and the pipeline will be ready to use the model to perform inference requests. 

```python
pipeline.add_model_step(mobilenet_model)

pipeline.deploy()
```

<table><tr><th>name</th> <td>mobilenetpipeline</td></tr><tr><th>created</th> <td>2023-10-30 18:15:17.969988+00:00</td></tr><tr><th>last_updated</th> <td>2023-10-30 18:15:25.186553+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>7a1d84f9-13f7-405a-a4d0-5ec7de59f03e, b236b83b-1755-41d9-a2ae-804c040b9038</td></tr><tr><th>steps</th> <td>mobilenet</td></tr><tr><th>published</th> <td>False</td></tr></table>

### Prepare input image

Next we will load a sample image and resize it to the width and height required for the object detector.  Once complete, it the image will be converted to a numpy ndim array and added to a dictionary.

```python

# The size the image will be resized to
width = 640
height = 480

# Only objects that have a confidence > confidence_target will be displayed on the image
cvDemo = CVDemo()

imagePath = 'data/images/input/example/dairy_bottles.png'

# The image width and height needs to be set to what the model was trained for.  In this case 640x480.
tensor, resizedImage = cvDemo.loadImageAndResize(imagePath, width, height)

# get npArray from the tensorFloat
npArray = tensor.cpu().numpy()

#creates a dictionary with the wallaroo "tensor" key and the numpy ndim array representing image as the value.

dictData = {"tensor":[npArray]}
dataframedata = pd.DataFrame(dictData)

```

### Run Inference

With that done, we can have the model detect the objects on the image by running an inference through the pipeline, and storing the results for the next step.

```python
startTime = time.time()
# pass the dataframe in 
#infResults = pipeline.infer(dataframedata, dataset=["*", "metadata.elapsed"])
infResults = pipeline.infer_from_file('./data/dairy_bottles.df.json', dataset=["*", "metadata.elapsed"])

endTime = time.time()
```

### Draw the Inference Results

With our inference results, we can take them and use the Wallaroo CVDemo class and draw them onto the original image.  The bounding boxes and the confidence value will only be drawn on images where the model returned a 90% confidence rate in the object's identity.

```python
df = pd.DataFrame(columns=['classification','confidence','x','y','width','height'])
pd.options.mode.chained_assignment = None  # default='warn'
pd.options.display.float_format = '{:.2%}'.format

# Points to where all the inference results are
boxList = infResults.loc[0]["out.boxes"]

# # reshape this to an array of bounding box coordinates converted to ints
boxA = np.array(boxList)
boxes = boxA.reshape(-1, 4)
boxes = boxes.astype(int)

df[['x', 'y','width','height']] = pd.DataFrame(boxes)

classes = infResults.loc[0]["out.classes"]
confidences = infResults.loc[0]["out.confidences"]

infResults = {
    'model_name' : model_name,
    'pipeline_name' : pipeline_name,
    'width': width,
    'height': height,
    'image' : resizedImage,
    'boxes' : boxes,
    'classes' : classes,
    'confidences' : confidences,
    'confidence-target' : 0.90,
    'inference-time': (endTime-startTime),
    'onnx-time' : int(infResults.loc[0]["metadata.elapsed"][1]) / 1e+9,                
    'color':(255,0,0)
}

image = cvDemo.drawAndDisplayDetectedObjectsWithClassification(infResults)
```

    
{{<figure src="/images/2023.4.0/wallaroo-tutorials/computer-vision/01_computer_vision_tutorial_mobilenet-reference_files/01_computer_vision_tutorial_mobilenet-reference_19_0.png" width="800" label="png">}}
    

### Extract the Inference Information

To show what is going on in the background, we'll extract the inference results create a dataframe with columns representing the classification, confidence, and bounding boxes of the objects identified.

```python
idx = 0 
for idx in range(0,len(classes)):
    df['classification'][idx] = cvDemo.CLASSES[classes[idx]] # Classes contains the 80 different COCO classificaitons
    df['confidence'][idx] = confidences[idx]
df
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>classification</th>
      <th>confidence</th>
      <th>x</th>
      <th>y</th>
      <th>width</th>
      <th>height</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>bottle</td>
      <td>98.65%</td>
      <td>0</td>
      <td>210</td>
      <td>85</td>
      <td>479</td>
    </tr>
    <tr>
      <th>1</th>
      <td>bottle</td>
      <td>90.12%</td>
      <td>72</td>
      <td>197</td>
      <td>151</td>
      <td>468</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bottle</td>
      <td>60.78%</td>
      <td>211</td>
      <td>184</td>
      <td>277</td>
      <td>420</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bottle</td>
      <td>59.22%</td>
      <td>143</td>
      <td>203</td>
      <td>216</td>
      <td>448</td>
    </tr>
    <tr>
      <th>4</th>
      <td>refrigerator</td>
      <td>53.73%</td>
      <td>13</td>
      <td>41</td>
      <td>640</td>
      <td>480</td>
    </tr>
    <tr>
      <th>5</th>
      <td>bottle</td>
      <td>45.13%</td>
      <td>106</td>
      <td>206</td>
      <td>159</td>
      <td>463</td>
    </tr>
    <tr>
      <th>6</th>
      <td>bottle</td>
      <td>43.73%</td>
      <td>278</td>
      <td>1</td>
      <td>321</td>
      <td>93</td>
    </tr>
    <tr>
      <th>7</th>
      <td>bottle</td>
      <td>43.09%</td>
      <td>462</td>
      <td>104</td>
      <td>510</td>
      <td>224</td>
    </tr>
    <tr>
      <th>8</th>
      <td>bottle</td>
      <td>40.85%</td>
      <td>310</td>
      <td>1</td>
      <td>352</td>
      <td>94</td>
    </tr>
    <tr>
      <th>9</th>
      <td>bottle</td>
      <td>39.19%</td>
      <td>528</td>
      <td>268</td>
      <td>636</td>
      <td>475</td>
    </tr>
    <tr>
      <th>10</th>
      <td>bottle</td>
      <td>35.76%</td>
      <td>220</td>
      <td>0</td>
      <td>258</td>
      <td>90</td>
    </tr>
    <tr>
      <th>11</th>
      <td>bottle</td>
      <td>31.81%</td>
      <td>552</td>
      <td>96</td>
      <td>600</td>
      <td>233</td>
    </tr>
    <tr>
      <th>12</th>
      <td>bottle</td>
      <td>26.45%</td>
      <td>349</td>
      <td>0</td>
      <td>404</td>
      <td>98</td>
    </tr>
    <tr>
      <th>13</th>
      <td>bottle</td>
      <td>23.06%</td>
      <td>450</td>
      <td>264</td>
      <td>619</td>
      <td>472</td>
    </tr>
    <tr>
      <th>14</th>
      <td>bottle</td>
      <td>20.48%</td>
      <td>261</td>
      <td>193</td>
      <td>307</td>
      <td>408</td>
    </tr>
    <tr>
      <th>15</th>
      <td>bottle</td>
      <td>17.46%</td>
      <td>509</td>
      <td>101</td>
      <td>544</td>
      <td>235</td>
    </tr>
    <tr>
      <th>16</th>
      <td>bottle</td>
      <td>17.31%</td>
      <td>592</td>
      <td>100</td>
      <td>633</td>
      <td>239</td>
    </tr>
    <tr>
      <th>17</th>
      <td>bottle</td>
      <td>16.00%</td>
      <td>475</td>
      <td>297</td>
      <td>551</td>
      <td>468</td>
    </tr>
    <tr>
      <th>18</th>
      <td>bottle</td>
      <td>14.91%</td>
      <td>368</td>
      <td>163</td>
      <td>423</td>
      <td>362</td>
    </tr>
    <tr>
      <th>19</th>
      <td>book</td>
      <td>13.66%</td>
      <td>120</td>
      <td>0</td>
      <td>175</td>
      <td>81</td>
    </tr>
    <tr>
      <th>20</th>
      <td>book</td>
      <td>13.32%</td>
      <td>72</td>
      <td>0</td>
      <td>143</td>
      <td>85</td>
    </tr>
    <tr>
      <th>21</th>
      <td>bottle</td>
      <td>12.22%</td>
      <td>271</td>
      <td>200</td>
      <td>305</td>
      <td>274</td>
    </tr>
    <tr>
      <th>22</th>
      <td>book</td>
      <td>12.13%</td>
      <td>161</td>
      <td>0</td>
      <td>213</td>
      <td>85</td>
    </tr>
    <tr>
      <th>23</th>
      <td>bottle</td>
      <td>11.96%</td>
      <td>162</td>
      <td>0</td>
      <td>214</td>
      <td>83</td>
    </tr>
    <tr>
      <th>24</th>
      <td>bottle</td>
      <td>11.53%</td>
      <td>310</td>
      <td>190</td>
      <td>367</td>
      <td>397</td>
    </tr>
    <tr>
      <th>25</th>
      <td>bottle</td>
      <td>9.62%</td>
      <td>396</td>
      <td>166</td>
      <td>441</td>
      <td>360</td>
    </tr>
    <tr>
      <th>26</th>
      <td>cake</td>
      <td>8.65%</td>
      <td>439</td>
      <td>256</td>
      <td>640</td>
      <td>473</td>
    </tr>
    <tr>
      <th>27</th>
      <td>bottle</td>
      <td>7.84%</td>
      <td>544</td>
      <td>375</td>
      <td>636</td>
      <td>472</td>
    </tr>
    <tr>
      <th>28</th>
      <td>vase</td>
      <td>7.23%</td>
      <td>272</td>
      <td>2</td>
      <td>306</td>
      <td>96</td>
    </tr>
    <tr>
      <th>29</th>
      <td>bottle</td>
      <td>6.28%</td>
      <td>453</td>
      <td>303</td>
      <td>524</td>
      <td>463</td>
    </tr>
    <tr>
      <th>30</th>
      <td>bottle</td>
      <td>5.28%</td>
      <td>609</td>
      <td>94</td>
      <td>635</td>
      <td>211</td>
    </tr>
  </tbody>
</table>

### Undeploy the Pipeline

With the inference complete, we can undeploy the pipeline and return the resources back to the Wallaroo instance.

```python
pipeline.undeploy()
```

<table><tr><th>name</th> <td>mobilenetpipeline</td></tr><tr><th>created</th> <td>2023-10-30 18:15:17.969988+00:00</td></tr><tr><th>last_updated</th> <td>2023-10-30 18:15:25.186553+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>7a1d84f9-13f7-405a-a4d0-5ec7de59f03e, b236b83b-1755-41d9-a2ae-804c040b9038</td></tr><tr><th>steps</th> <td>mobilenet</td></tr><tr><th>published</th> <td>False</td></tr></table>

