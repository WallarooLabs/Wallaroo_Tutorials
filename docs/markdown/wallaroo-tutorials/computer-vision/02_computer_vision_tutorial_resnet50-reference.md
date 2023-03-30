This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-model-cookbooks/computer-vision).

## Step 02: Detecting Objects Using resnet50

The following tutorial demonstrates how to use a trained mobilenet model deployed in Wallaroo to detect objects.  This process will use the following steps:

1. Create a Wallaroo workspace and pipeline.
1. Upload a trained resnet50 ML model and add it as a pipeline step.
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
import os
import numpy as np
import json
import requests
import time
import pandas as pd
from CVDemoUtils import CVDemo
```

### Arrow Support

As of the 2023.1 release, Wallaroo provides support for DataFrame and Arrow for inference inputs.  This tutorial allows users to adjust their experience based on whether they have enabled Arrow support in their Wallaroo instance or not.

If Arrow support has been enabled, `arrowEnabled=True`. If disabled or you're not sure, set it to `arrowEnabled=False`

The examples below will be shown in an arrow enabled environment.


```python
import os
# Only set the below to make the OS environment ARROW_ENABLED to TRUE.  Otherwise, leave as is.
os.environ["ARROW_ENABLED"]="True"

if "ARROW_ENABLED" not in os.environ or os.environ["ARROW_ENABLED"].casefold() == "False".casefold():
    arrowEnabled = False
else:
    arrowEnabled = True
print(arrowEnabled)
```

### Connect to Wallaroo

Now we connect to the Wallaroo instance.  If you are connecting from a remote connection, set the `wallarooPrefix` and `wallarooSuffix` and use them to connect.  If the connection is from within the Wallaroo instance cluster, then just `wl = wallaroo.Client()` can be used.


```python
# Login through local service

wl = wallaroo.Client()

# SSO login through keycloak

wallarooPrefix = "YOUR PREFIX"
wallarooSuffix = "YOUR SUFFIX"

wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}.api.{wallarooSuffix}", 
                auth_endpoint=f"https://{wallarooPrefix}.keycloak.{wallarooSuffix}", 
                auth_type="sso")
```

### Set Variables

The following variables and methods are used later to create or connect to an existing workspace, pipeline, and model.  This example has both the resnet model, and a post process script.


```python
workspace_name = 'resnetworkspacetest'
pipeline_name = 'resnetnetpipelinetest'
model_name = 'resnet50'
model_file_name = 'models/frcnn-resnet.pt.onnx'
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
        pipeline = wl.pipelines_by_name(pipeline_name)[0]
    except EntityNotFoundError:
        pipeline = wl.build_pipeline(pipeline_name)
    return pipeline
```

### Create Workspace

The workspace will be created or connected to, and set as the default workspace for this session.  Once that is done, then all models and pipelines will be set in that workspace.


```python
workspace = get_workspace(workspace_name)
wl.set_current_workspace(workspace)
wl.get_current_workspace()
```

### Create Pipeline and Upload Model

We will now create or connect to an existing pipeline as named in the variables above.


```python
pipeline = get_pipeline(pipeline_name)

resnet_model = wl.upload_model(model_name, model_file_name)
```

### Deploy Pipeline

With the model uploaded, we can add it is as a step in the pipeline, then deploy it.  Once deployed, resources from the Wallaroo instance will be reserved and the pipeline will be ready to use the model to perform inference requests. 


```python
pipeline.add_model_step(resnet_model)

pipeline.deploy()
```




<table><tr><th>name</th> <td>resnetnetpipelinetest</td></tr><tr><th>created</th> <td>2023-03-02 19:35:32.620147+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-02 19:36:04.824928+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>3d79d2d8-369e-4781-b796-d8c0c7144736, 9e290e9a-a735-499e-b5d2-e9e6c4d7fe14</td></tr><tr><th>steps</th> <td>resnet50</td></tr></table>



## Test the pipeline by running inference on a sample image

### Prepare input image

Next we will load a sample image and resize it to the width and height required for the object detector.

We will convert the image to a numpy ndim array and add it do a dictionary


```python
#The size the image will be resized to
width = 640
height = 480

cvDemo = CVDemo()

imagePath = 'data/images/input/example/dairy_bottles.png'

# The image width and height needs to be set to what the model was trained for.  In this case 640x480.
tensor, resizedImage = cvDemo.loadImageAndResize(imagePath, width, height)

# get npArray from the tensorFloat
npArray = tensor.cpu().numpy()

#creates a dictionary with the wallaroo "tensor" key and the numpy ndim array representing image as the value.
dictData = {"tensor": npArray.tolist()}
```

### Run Inference

With that done, we can have the model detect the objects on the image by running an inference through the pipeline, and storing the results for the next step.

**IMPORTANT NOTE**:  If necessary, add `timeout=60` to the `infer` method if more time is needed to upload the data file for the inference request.


```python
startTime = time.time()
infResults = pipeline.infer(dictData)
endTime = time.time()

if arrowEnabled is True:
    results = infResults[0]
else:
    results = infResults[0].raw
```

### Draw the Inference Results

With our inference results, we can take them and use the Wallaroo CVDemo class and draw them onto the original image.  The bounding boxes and the confidence value will only be drawn on images where the model returned a 50% confidence rate in the object's identity.


```python
df = pd.DataFrame(columns=['classification','confidence','x','y','width','height'])
pd.options.mode.chained_assignment = None  # default='warn'
pd.options.display.float_format = '{:.2%}'.format

# Points to where all the inference results are
outputs = results['outputs']

boxes = outputs[0]

# reshape this to an array of bounding box coordinates converted to ints
boxList = boxes['Float']['data']
boxA = np.array(boxList)
boxes = boxA.reshape(-1, 4)
boxes = boxes.astype(int)

df[['x', 'y','width','height']] = pd.DataFrame(boxes)

classes = outputs[1]['Int64']['data']
confidences = outputs[2]['Float']['data']

infResults = {
    'model_name' : model_name,
    'pipeline_name' : pipeline_name,
    'width': width,
    'height': height,
    'image' : resizedImage,
    'boxes' : boxes,
    'classes' : classes,
    'confidences' : confidences,
    'confidence-target' : 0.50,
    'inference-time': (endTime-startTime),
    'onnx-time' : int(results['elapsed']) / 1e+9,                
    'color':(255,0,0)
}

cvDemo.drawAndDisplayDetectedObjectsWithClassification(infResults)
```


    
![png](02_computer_vision_tutorial_resnet50-reference_files/02_computer_vision_tutorial_resnet50-reference_22_0.png)
    


### Extract the Inference Information

To show what is going on in the background, we'll extract the inference results create a dataframe with columns representing the classification, confidence, and bounding boxes of the objects identified.


```python
idx = 0 
for idx in range(0,len(classes)):
    cocoClasses = cvDemo.getCocoClasses()
    df['classification'][idx] = cocoClasses[classes[idx]] # Classes contains the 80 different COCO classificaitons
    df['confidence'][idx] = confidences[idx]
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <td>99.65%</td>
      <td>2</td>
      <td>193</td>
      <td>76</td>
      <td>475</td>
    </tr>
    <tr>
      <th>1</th>
      <td>bottle</td>
      <td>98.83%</td>
      <td>610</td>
      <td>98</td>
      <td>639</td>
      <td>232</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bottle</td>
      <td>97.00%</td>
      <td>544</td>
      <td>98</td>
      <td>581</td>
      <td>230</td>
    </tr>
    <tr>
      <th>3</th>
      <td>bottle</td>
      <td>96.96%</td>
      <td>454</td>
      <td>113</td>
      <td>484</td>
      <td>210</td>
    </tr>
    <tr>
      <th>4</th>
      <td>bottle</td>
      <td>96.48%</td>
      <td>502</td>
      <td>331</td>
      <td>551</td>
      <td>476</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>bottle</td>
      <td>5.72%</td>
      <td>556</td>
      <td>287</td>
      <td>580</td>
      <td>322</td>
    </tr>
    <tr>
      <th>96</th>
      <td>refrigerator</td>
      <td>5.66%</td>
      <td>80</td>
      <td>161</td>
      <td>638</td>
      <td>480</td>
    </tr>
    <tr>
      <th>97</th>
      <td>bottle</td>
      <td>5.60%</td>
      <td>455</td>
      <td>334</td>
      <td>480</td>
      <td>349</td>
    </tr>
    <tr>
      <th>98</th>
      <td>bottle</td>
      <td>5.46%</td>
      <td>613</td>
      <td>267</td>
      <td>635</td>
      <td>375</td>
    </tr>
    <tr>
      <th>99</th>
      <td>bottle</td>
      <td>5.37%</td>
      <td>345</td>
      <td>2</td>
      <td>395</td>
      <td>99</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 6 columns</p>
</div>



### Undeploy the Pipeline

With the inference complete, we can undeploy the pipeline and return the resources back to the Wallaroo instance.


```python
pipeline.undeploy()    
```




<table><tr><th>name</th> <td>resnetnetpipelinetest</td></tr><tr><th>created</th> <td>2023-03-02 19:35:32.620147+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-02 19:36:04.824928+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>3d79d2d8-369e-4781-b796-d8c0c7144736, 9e290e9a-a735-499e-b5d2-e9e6c4d7fe14</td></tr><tr><th>steps</th> <td>resnet50</td></tr></table>


