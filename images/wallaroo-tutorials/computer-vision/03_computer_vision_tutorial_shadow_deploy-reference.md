This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-model-cookbooks/computer-vision).

## Step 03: Detecting Objects Using Shadow Deploy

The following tutorial demonstrates how to use two trained models, one based on the resnet50, the other on mobilenet, deployed in Wallaroo to detect objects.  This builds on the previous tutorials in this series, Step 01: Detecting Objects Using mobilenet" and  "Step 02: Detecting Objects Using resnet50".

For this tutorial, the Wallaroo feature [Shadow Deploy](https://docs.wallaroo.ai/wallaroo-tutorials/wallaroo-tutorials-testing/wallaroo-shadow-deployment-tutorial/) will be used to submit inference requests to both models at once.  The mobilnet object detector is the control and the faster-rcnn object detector is the challenger.  The results between the two will be compared for their confidence, and that confidence will be used to draw bounding boxes around identified objects.

This process will use the following steps:

1. Create a Wallaroo workspace and pipeline.
1. Upload a trained resnet50 ML model and trained mobilenet model and add them as a shadow deployed step with the mobilenet as the control model.
1. Deploy the pipeline.
1. Perform an inference on a sample image.
1. Based on the 
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

```

### Connect to the Wallaroo Instance

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
# Login through local service

wl = wallaroo.Client()

wl = wallaroo.Client()

wallarooPrefix = "doc-test."
wallarooSuffix = "wallarooexample.ai"

wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}api.{wallarooSuffix}", 
                    auth_endpoint=f"https://{wallarooPrefix}keycloak.{wallarooSuffix}", 
                    auth_type="sso")
```

### Set Variables

The following variables and methods are used later to create or connect to an existing workspace, pipeline, and model.  This example has both the resnet model, and a post process script.

```python
workspace_name = f'shadowimageworkspacetest{suffix}'
pipeline_name = f'shadowimagepipelinetest{suffix}'
control_model_name = f'mobilenet{suffix}'
control_model_file_name = 'models/mobilenet.pt.onnx'
challenger_model_name = f'resnet50{suffix}'
challenger_model_file_name = 'models/frcnn-resnet.pt.onnx'
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

    {'name': 'shadowimageworkspacetestkpld', 'id': 10, 'archived': False, 'created_by': '6236ad2a-7eb8-4bbc-a8c9-39ce92767bad', 'created_at': '2023-10-24T17:05:03.040658+00:00', 'models': [], 'pipelines': []}

### Create Pipeline and Upload Model

We will now create or connect to an existing pipeline as named in the variables above, then upload each of the models.

```python
pipeline = get_pipeline(pipeline_name)
```

```python
control =  wl.upload_model(control_model_name, control_model_file_name, framework=Framework.ONNX).configure(batch_config="single", tensor_fields=["tensor"])
```

```python
challenger = wl.upload_model(challenger_model_name, challenger_model_file_name, framework=Framework.ONNX).configure(batch_config="single", tensor_fields=["tensor"])
```

### Shadow Deploy Pipeline

For this step, rather than deploying each model into a separate step, both will be deployed into a single step as a Shadow Deploy step.  This will take the inference input data and process it through both pipelines at the same time.  The inference results for the control will be stored in it's `['outputs']` array, while the results for the challenger are stored the `['shadow_data']` array.

```python
pipeline.add_shadow_deploy(control, [challenger])

```

<table><tr><th>name</th> <td>shadowimagepipelinetestkpld</td></tr><tr><th>created</th> <td>2023-10-24 17:05:04.926467+00:00</td></tr><tr><th>last_updated</th> <td>2023-10-24 17:05:04.926467+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>2f6bee8b-bf31-41cf-b7d2-d4912bfdcca8</td></tr><tr><th>steps</th> <td></td></tr><tr><th>published</th> <td>False</td></tr></table>

```python
pipeline.deploy()
```

<table><tr><th>name</th> <td>shadowimagepipelinetestkpld</td></tr><tr><th>created</th> <td>2023-10-24 17:05:04.926467+00:00</td></tr><tr><th>last_updated</th> <td>2023-10-24 17:05:22.613731+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>433aa539-cb64-4733-85d3-68f6f769dd36, 2f6bee8b-bf31-41cf-b7d2-d4912bfdcca8</td></tr><tr><th>steps</th> <td>mobilenetkpld</td></tr><tr><th>published</th> <td>False</td></tr></table>

```python
pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.3.35',
       'name': 'engine-5888d5b5d6-2dvlb',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'shadowimagepipelinetestkpld',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'mobilenetkpld',
          'version': 'be97c0ef-afb1-4835-9f94-ec6534fa9c07',
          'sha': '9044c970ee061cc47e0c77e20b05e884be37f2a20aa9c0c3ce1993dbd486a830',
          'status': 'Running'},
         {'name': 'resnet50kpld',
          'version': 'f0967b5f-4b17-4dbd-b1d6-49b3339f041b',
          'sha': '43326e50af639105c81372346fb9ddf453fea0fe46648b2053c375360d9c1647',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.4.58',
       'name': 'engine-lb-584f54c899-hl4lx',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}

### Prepare input image

Next we will load a sample image and resize it to the width and height required for the object detector.

We will convert the image to a numpy ndim array and add it do a dictionary

```python

imagePath = 'data/images/input/example/store-front.png'

# The image width and height needs to be set to what the model was trained for.  In this case 640x480.
cvDemo = CVDemo()

# The size the image will be resized to meet the input requirements of the object detector
width = 640
height = 480
tensor, controlImage = cvDemo.loadImageAndResize(imagePath, width, height)
challengerImage = controlImage.copy()

# get npArray from the tensorFloat
npArray = tensor.cpu().numpy()

#creates a dictionary with the wallaroo "tensor" key and the numpy ndim array representing image as the value.
# dictData = {"tensor": npArray.tolist()}

dictData = {"tensor":[npArray]}
dataframedata = pd.DataFrame(dictData)
```

### Run Inference using Shadow Deployment

Now lets have the model detect the objects on the image by running inference and extracting the results 

```python
startTime = time.time()
infResults = pipeline.infer_from_file('./data/dairy_bottles.df.json', dataset=["*", "metadata.elapsed"])
#infResults = pipeline.infer(dataframedata, dataset=["*", "metadata.elapsed"])
endTime = time.time()
```

### Extract Control Inference Results

First we'll extract the inference result data for the control model and map it onto the image.

```python
df = pd.DataFrame(columns=['classification','confidence','x','y','width','height'])
pd.options.mode.chained_assignment = None  # default='warn'
pd.options.display.float_format = '{:.2%}'.format

# Points to where all the inference results are
# boxList = infResults[0]["out.output"]
boxList = infResults.loc[0]["out.boxes"]

# # reshape this to an array of bounding box coordinates converted to ints
boxA = np.array(boxList)
controlBoxes = boxA.reshape(-1, 4)
controlBoxes = controlBoxes.astype(int)

df[['x', 'y','width','height']] = pd.DataFrame(controlBoxes)

controlClasses = infResults.loc[0]["out.classes"]
controlConfidences = infResults.loc[0]["out.confidences"]

results = {
    'model_name' : control.name(),
    'pipeline_name' : pipeline.name(),
    'width': width,
    'height': height,
    'image' : controlImage,
    'boxes' : controlBoxes,
    'classes' : controlClasses,
    'confidences' : controlConfidences,
    'confidence-target' : 0.9,
    'color':CVDemo.RED, # color to draw bounding boxes and the text in the statistics
    'inference-time': (endTime-startTime),
    'onnx-time' : 0,                
}
cvDemo.drawAndDisplayDetectedObjectsWithClassification(results)
```

    
{{<figure src="/images/2023.4.0/wallaroo-tutorials/computer-vision/03_computer_vision_tutorial_shadow_deploy-reference_files/03_computer_vision_tutorial_shadow_deploy-reference_23_0.png" width="800" label="png">}}
    

### Display the Control Results

Here we will use the Wallaroo CVDemo helper class to draw the control model results on the image.

The full results will be displayed in a dataframe with columns representing the classification, confidence, and bounding boxes of the objects identified.

Once extracted from the results we will want to reshape the flattened array into an array with 4 elements (x,y,width,height).

```python
idx = 0 
cocoClasses = cvDemo.getCocoClasses()
for idx in range(0,len(controlClasses)):
    df['classification'][idx] = cocoClasses[controlClasses[idx]] # Classes contains the 80 different COCO classificaitons
    df['confidence'][idx] = controlConfidences[idx]
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

### Display the Challenger Results

Here we will use the Wallaroo CVDemo helper class to draw the challenger model results on the input image.

```python
challengerDf = pd.DataFrame(columns=['classification','confidence','x','y','width','height'])
pd.options.mode.chained_assignment = None  # default='warn'
pd.options.display.float_format = '{:.2%}'.format

# Points to where all the inference results are
boxList = infResults.loc[0][f"out_{challenger_model_name}.boxes"]

# outputs = results['outputs']
# boxes = outputs[0]

# # reshape this to an array of bounding box coordinates converted to ints
# boxList = boxes['Float']['data']
boxA = np.array(boxList)
challengerBoxes = boxA.reshape(-1, 4)
challengerBoxes = challengerBoxes.astype(int)

challengerDf[['x', 'y','width','height']] = pd.DataFrame(challengerBoxes)

challengerClasses = infResults.loc[0][f"out_{challenger_model_name}.classes"]
challengerConfidences = infResults.loc[0][f"out_{challenger_model_name}.confidences"]

results = {
    'model_name' : challenger.name(),
    'pipeline_name' : pipeline.name(),
    'width': width,
    'height': height,
    'image' : challengerImage,
    'boxes' : challengerBoxes,
    'classes' : challengerClasses,
    'confidences' : challengerConfidences,
    'confidence-target' : 0.9,
    'color':CVDemo.RED, # color to draw bounding boxes and the text in the statistics
    'inference-time': (endTime-startTime),
    'onnx-time' : 0,                
}
cvDemo.drawAndDisplayDetectedObjectsWithClassification(results)
```

    
{{<figure src="/images/2023.4.0/wallaroo-tutorials/computer-vision/03_computer_vision_tutorial_shadow_deploy-reference_files/03_computer_vision_tutorial_shadow_deploy-reference_27_0.png" width="800" label="png">}}
    

### Display Challenger Results

The inference results for the objects detected by the challenger model will be displayed including the confidence values.  Once extracted from the results we will want to reshape the flattened array into an array with 4 elements (x,y,width,height).

```python
idx = 0 
for idx in range(0,len(challengerClasses)):
    challengerDf['classification'][idx] = cvDemo.CLASSES[challengerClasses[idx]] # Classes contains the 80 different COCO classificaitons
    challengerDf['confidence'][idx] = challengerConfidences[idx]
challengerDf
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

```python
pipeline.undeploy()
```

<table><tr><th>name</th> <td>shadowimagepipelinetestkpld</td></tr><tr><th>created</th> <td>2023-10-24 17:05:04.926467+00:00</td></tr><tr><th>last_updated</th> <td>2023-10-24 17:05:22.613731+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>433aa539-cb64-4733-85d3-68f6f769dd36, 2f6bee8b-bf31-41cf-b7d2-d4912bfdcca8</td></tr><tr><th>steps</th> <td>mobilenetkpld</td></tr><tr><th>published</th> <td>False</td></tr></table>

### Conclusion

Notice the difference in the control confidence and the challenger confidence.  <b>Clearly we can see in this example the challenger resnet50 model is performing better than the control mobilenet model</b>.  This is likely due to the fact that frcnn resnet50 model is a 2 stage object detector vs the frcnn mobilenet is a single stage detector.

This completes using Wallaroo's shadow deployment feature to compare different computer vision models.

