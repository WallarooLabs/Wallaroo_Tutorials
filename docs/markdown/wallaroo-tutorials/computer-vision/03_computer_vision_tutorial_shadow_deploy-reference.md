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

```

### Connect to the Wallaroo Instance

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  If logging in externally, update the `wallarooPrefix` and `wallarooSuffix` variables with the proper DNS information.  For more information on Wallaroo DNS settings, see the [Wallaroo DNS Integration Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-dns-guide/).

```python
# Login through local service

wl = wallaroo.Client()
```

### Set Variables

The following variables and methods are used later to create or connect to an existing workspace, pipeline, and model.  This example has both the resnet model, and a post process script.

```python
workspace_name = 'shadowimageworkspacetest'
pipeline_name = 'shadowimagepipelinetest'
control_model_name = 'mobilenet'
control_model_file_name = 'models/mobilenet.pt.onnx'
challenger_model_name = 'resnet50'
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

    {'name': 'shadowimageworkspacetest', 'id': 28, 'archived': False, 'created_by': '028c8b48-c39b-4578-9110-0b5bdd3824da', 'created_at': '2023-05-17T22:07:31.167061+00:00', 'models': [], 'pipelines': []}

### Create Pipeline and Upload Model

We will now create or connect to an existing pipeline as named in the variables above, then upload each of the models.

```python
pipeline = get_pipeline(pipeline_name)
```

```python
control =  wl.upload_model(control_model_name, control_model_file_name, framework=Framework.ONNX).configure(batch_config="single")
```

```python
challenger = wl.upload_model(challenger_model_name, challenger_model_file_name, framework=Framework.ONNX).configure(batch_config="single")
```

### Shadow Deploy Pipeline

For this step, rather than deploying each model into a separate step, both will be deployed into a single step as a Shadow Deploy step.  This will take the inference input data and process it through both pipelines at the same time.  The inference results for the control will be stored in it's `['outputs']` array, while the results for the challenger are stored the `['shadow_data']` array.

```python
pipeline.add_shadow_deploy(control, [challenger])

```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>shadowimagepipelinetest</td></tr><tr><th>created</th> <td>2023-05-17 22:07:33.073463+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 22:07:33.073463+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>b41971db-5919-47a1-b2b7-7b7dfb2722d1</td></tr><tr><th>steps</th> <td></td></tr></table>
{{</table>}}

```python
pipeline.deploy()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>shadowimagepipelinetest</td></tr><tr><th>created</th> <td>2023-05-17 22:07:33.073463+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 22:08:33.662212+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>10450273-e76b-4513-a9ba-2467f9d7e149, b41971db-5919-47a1-b2b7-7b7dfb2722d1</td></tr><tr><th>steps</th> <td>mobilenet</td></tr></table>
{{</table>}}

```python
pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.2.170',
       'name': 'engine-74cc5d9f69-87tw4',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'shadowimagepipelinetest',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'mobilenet',
          'version': '4ebf53dd-8ffd-43ec-a61a-925c7e624b3a',
          'sha': 'f4c7009e53b679f5e44d70d9612e8dc365565cec88c25b5efa11b903b6b7bdc6',
          'status': 'Running'},
         {'name': 'resnet50',
          'version': '11c73a48-12d7-4af9-adc8-83361913d1cd',
          'sha': 'ee606dc9776a1029420b3adf59b6d29395c89d1d9460d75045a1f2f152d288e7',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.4.189',
       'name': 'engine-lb-584f54c899-dl69m',
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
infResults = pipeline.infer(dataframedata, dataset=["*", "metadata.elapsed"])
endTime = time.time()

display(infResults)
```

{{<table "table table-striped table-bordered" >}}
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out.2518</th>
      <th>out.2519</th>
      <th>out.output</th>
      <th>check_failures</th>
      <th>out_resnet50.3069</th>
      <th>out_resnet50.3070</th>
      <th>metadata.elapsed</th>
      <th>out_resnet50.output</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-05-17 22:08:48.947</td>
      <td>[0.9981781, 0.9542837, 0.8133338, 0.7237682, 0.581608, 0.47485322, 0.45198253, 0.4417335, 0.41891232, 0.40041047, 0.3921795, 0.3894252, 0.38495764, 0.37889498, 0.37044412, 0.3611252, 0.34545907, 0.3236507, 0.32191104, 0.30279082, 0.2650459, 0.23091778, 0.22657326, 0.22226617, 0.21884085, 0.21133144, 0.2015091, 0.19676292, 0.16859648, 0.1654867, 0.16219954, 0.1617982, 0.15132228, 0.10616817, 0.10007797, 0.09096525, 0.07947229, 0.07812073, 0.07175662, 0.06376821, 0.05691604]</td>
      <td>[3, 1, 28, 1, 28, 1, 1, 1, 1, 1, 64, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 64, 1, 1, 28, 28, 28, 28, 64, 28, 1]</td>
      <td>[278.87997, 335.21338, 494.20258, 471.65662, 32.198643, 303.0701, 66.17404, 365.9008, 117.497025, 256.20038, 209.67964, 322.92923, 183.82143, 310.25345, 203.3382, 367.92252, 213.59555, 273.99362, 298.52182, 309.41812, 155.87799, 307.32886, 180.0752, 365.3083, 263.84793, 315.9717, 303.77182, 422.7569, 8.386993, 304.63635, 36.567783, 361.8276, 608.29767, 330.57816, 628.42004, 375.5664, 557.5376, 330.72534, 582.2022, 395.2367, 241.44873, 193.24695, 315.8684, 292.1694, 547.05035, 329.7998, 573.16016, 397.19586, 615.91016, 331.1884, 634.11835, 372.09625, 553.2413, 321.90045, 576.21533, 374.4632, 147.58018, 304.73825, 170.9217, 366.58972, 515.0556, 322.937, 537.2063, 369.97394, 562.9472, 317.56598, 586.0538, 373.93054, 531.0953, 329.1975, 557.37616, 399.72592, 239.39209, 306.0158, 279.9354, 428.13208, 320.09695, 308.91736, 343.2187, 359.8366, 289.79147, 311.329, 310.10736, 380.05817, 371.34875, 307.95453, 394.90942, 337.69662, 295.30124, 300.42285, 340.61044, 373.4314, 1.7988257, 306.31195, 25.045603, 362.0882, 484.06668, 319.38544, 506.41733, 349.07382, ...]</td>
      <td>0</td>
      <td>[0.9990959, 0.9977418, 0.98879504, 0.9701535, 0.96819454, 0.958195, 0.9546498, 0.946965, 0.9401788, 0.9331664, 0.9245778, 0.92429465, 0.91924965, 0.9183872, 0.9068349, 0.8885886, 0.87426704, 0.8689579, 0.84102124, 0.8373341, 0.81157273, 0.80619556, 0.8004843, 0.79157895, 0.78372836, 0.754134, 0.7434203, 0.7210191, 0.71316636, 0.67353, 0.63615346, 0.6056083, 0.6000987, 0.5209629, 0.4587731, 0.44062626, 0.43137816, 0.43110648, 0.37025282, 0.36244312, 0.3458938, 0.341292, 0.29589733, 0.2944959, 0.26346523, 0.25633255, 0.2560418, 0.2553403, 0.24849786, 0.24383722, 0.19141474, 0.1790541, 0.17692076, 0.1701936, 0.16823667, 0.15538117, 0.14698146, 0.14135557, 0.13959822, 0.12418925, 0.123692915, 0.11508697, 0.114661165, 0.11444577, 0.11180547, 0.110967696, 0.108572185, 0.10580029, 0.10429848, 0.10333027, 0.09813049, 0.09095569, 0.08549411, 0.082057886, 0.07993688, 0.07378994, 0.06921979, 0.06908618, 0.06904188, 0.06126681, 0.060538013, 0.056145, 0.056017727, 0.055409227, 0.05519892, 0.05143937]</td>
      <td>[3, 1, 1, 3, 64, 1, 1, 1, 64, 1, 1, 28, 1, 1, 28, 1, 1, 1, 64, 1, 1, 28, 28, 1, 1, 28, 1, 1, 1, 1, 1, 1, 64, 1, 1, 28, 38, 3, 1, 1, 3, 1, 1, 64, 64, 64, 1, 14, 1, 64, 64, 64, 1, 1, 10, 72, 1, 1, 1, 1, 15, 1, 1, 64, 28, 27, 38, 64, 1, 1, 28, 31, 64, 64, 1, 1, 10, 1, 1, 1, 1, 1, 28, 1, 1, 1]</td>
      <td>[199232195, 26401]</td>
      <td>[274.28467, 332.3049, 496.415, 472.32785, 536.98975, 320.65692, 563.32153, 409.92398, 31.552103, 305.90497, 69.52175, 370.6107, 617.39325, 335.5266, 639.6564, 424.14944, 141.5756, 337.2296, 164.77254, 365.9992, 255.14586, 325.16907, 282.7709, 408.918, 506.71094, 310.3565, 528.6236, 345.83084, 182.70331, 317.82678, 205.7623, 376.4557, 1.6649555, 341.64255, 30.45084, 365.90402, 485.5722, 311.378, 505.83875, 345.55002, 289.61246, 309.67303, 311.3928, 383.75415, 477.92105, 276.51703, 566.71936, 312.27686, 324.223, 317.6657, 345.42676, 356.97507, 432.70374, 313.29636, 444.52927, 333.27655, 295.98373, 271.67447, 365.38544, 311.71338, 380.65524, 310.17752, 394.5112, 331.632, 310.8661, 321.49667, 329.5546, 368.74023, 469.553, 313.3934, 481.97534, 344.46222, 2.0225596, 305.42657, 36.529423, 367.66278, 162.96211, 314.6095, 185.57626, 363.08746, 589.2129, 321.07242, 602.71967, 368.34796, 209.69058, 266.0393, 291.9802, 309.825, 355.17767, 276.79196, 406.32568, 318.34106, 566.6238, 322.12747, 576.736, 344.0608, 394.41116, 315.93823, 408.83145, 331.8107, ...]</td>
    </tr>
  </tbody>
</table>
{{</table>}}

### Extract Control Inference Results

First we'll extract the inference result data for the control model and map it onto the image.

```python
df = pd.DataFrame(columns=['classification','confidence','x','y','width','height'])
pd.options.mode.chained_assignment = None  # default='warn'
pd.options.display.float_format = '{:.2%}'.format

# Points to where all the inference results are
# boxList = infResults[0]["out.output"]
boxList = infResults.loc[0]["out.output"]

# # reshape this to an array of bounding box coordinates converted to ints
boxA = np.array(boxList)
controlBoxes = boxA.reshape(-1, 4)
controlBoxes = controlBoxes.astype(int)

df[['x', 'y','width','height']] = pd.DataFrame(controlBoxes)

controlClasses = infResults.loc[0]["out.2519"]
controlConfidences = infResults.loc[0]["out.2518"]

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

    
![png](03_computer_vision_tutorial_shadow_deploy-reference_files/03_computer_vision_tutorial_shadow_deploy-reference_23_0.png)
    

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

{{<table "table table-striped table-bordered" >}}
<table>
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
      <td>car</td>
      <td>99.82%</td>
      <td>278</td>
      <td>335</td>
      <td>494</td>
      <td>471</td>
    </tr>
    <tr>
      <th>1</th>
      <td>person</td>
      <td>95.43%</td>
      <td>32</td>
      <td>303</td>
      <td>66</td>
      <td>365</td>
    </tr>
    <tr>
      <th>2</th>
      <td>umbrella</td>
      <td>81.33%</td>
      <td>117</td>
      <td>256</td>
      <td>209</td>
      <td>322</td>
    </tr>
    <tr>
      <th>3</th>
      <td>person</td>
      <td>72.38%</td>
      <td>183</td>
      <td>310</td>
      <td>203</td>
      <td>367</td>
    </tr>
    <tr>
      <th>4</th>
      <td>umbrella</td>
      <td>58.16%</td>
      <td>213</td>
      <td>273</td>
      <td>298</td>
      <td>309</td>
    </tr>
    <tr>
      <th>5</th>
      <td>person</td>
      <td>47.49%</td>
      <td>155</td>
      <td>307</td>
      <td>180</td>
      <td>365</td>
    </tr>
    <tr>
      <th>6</th>
      <td>person</td>
      <td>45.20%</td>
      <td>263</td>
      <td>315</td>
      <td>303</td>
      <td>422</td>
    </tr>
    <tr>
      <th>7</th>
      <td>person</td>
      <td>44.17%</td>
      <td>8</td>
      <td>304</td>
      <td>36</td>
      <td>361</td>
    </tr>
    <tr>
      <th>8</th>
      <td>person</td>
      <td>41.89%</td>
      <td>608</td>
      <td>330</td>
      <td>628</td>
      <td>375</td>
    </tr>
    <tr>
      <th>9</th>
      <td>person</td>
      <td>40.04%</td>
      <td>557</td>
      <td>330</td>
      <td>582</td>
      <td>395</td>
    </tr>
    <tr>
      <th>10</th>
      <td>potted plant</td>
      <td>39.22%</td>
      <td>241</td>
      <td>193</td>
      <td>315</td>
      <td>292</td>
    </tr>
    <tr>
      <th>11</th>
      <td>person</td>
      <td>38.94%</td>
      <td>547</td>
      <td>329</td>
      <td>573</td>
      <td>397</td>
    </tr>
    <tr>
      <th>12</th>
      <td>person</td>
      <td>38.50%</td>
      <td>615</td>
      <td>331</td>
      <td>634</td>
      <td>372</td>
    </tr>
    <tr>
      <th>13</th>
      <td>person</td>
      <td>37.89%</td>
      <td>553</td>
      <td>321</td>
      <td>576</td>
      <td>374</td>
    </tr>
    <tr>
      <th>14</th>
      <td>person</td>
      <td>37.04%</td>
      <td>147</td>
      <td>304</td>
      <td>170</td>
      <td>366</td>
    </tr>
    <tr>
      <th>15</th>
      <td>person</td>
      <td>36.11%</td>
      <td>515</td>
      <td>322</td>
      <td>537</td>
      <td>369</td>
    </tr>
    <tr>
      <th>16</th>
      <td>person</td>
      <td>34.55%</td>
      <td>562</td>
      <td>317</td>
      <td>586</td>
      <td>373</td>
    </tr>
    <tr>
      <th>17</th>
      <td>person</td>
      <td>32.37%</td>
      <td>531</td>
      <td>329</td>
      <td>557</td>
      <td>399</td>
    </tr>
    <tr>
      <th>18</th>
      <td>person</td>
      <td>32.19%</td>
      <td>239</td>
      <td>306</td>
      <td>279</td>
      <td>428</td>
    </tr>
    <tr>
      <th>19</th>
      <td>person</td>
      <td>30.28%</td>
      <td>320</td>
      <td>308</td>
      <td>343</td>
      <td>359</td>
    </tr>
    <tr>
      <th>20</th>
      <td>person</td>
      <td>26.50%</td>
      <td>289</td>
      <td>311</td>
      <td>310</td>
      <td>380</td>
    </tr>
    <tr>
      <th>21</th>
      <td>person</td>
      <td>23.09%</td>
      <td>371</td>
      <td>307</td>
      <td>394</td>
      <td>337</td>
    </tr>
    <tr>
      <th>22</th>
      <td>person</td>
      <td>22.66%</td>
      <td>295</td>
      <td>300</td>
      <td>340</td>
      <td>373</td>
    </tr>
    <tr>
      <th>23</th>
      <td>person</td>
      <td>22.23%</td>
      <td>1</td>
      <td>306</td>
      <td>25</td>
      <td>362</td>
    </tr>
    <tr>
      <th>24</th>
      <td>person</td>
      <td>21.88%</td>
      <td>484</td>
      <td>319</td>
      <td>506</td>
      <td>349</td>
    </tr>
    <tr>
      <th>25</th>
      <td>person</td>
      <td>21.13%</td>
      <td>272</td>
      <td>327</td>
      <td>297</td>
      <td>405</td>
    </tr>
    <tr>
      <th>26</th>
      <td>person</td>
      <td>20.15%</td>
      <td>136</td>
      <td>304</td>
      <td>160</td>
      <td>363</td>
    </tr>
    <tr>
      <th>27</th>
      <td>person</td>
      <td>19.68%</td>
      <td>520</td>
      <td>338</td>
      <td>543</td>
      <td>392</td>
    </tr>
    <tr>
      <th>28</th>
      <td>person</td>
      <td>16.86%</td>
      <td>478</td>
      <td>317</td>
      <td>498</td>
      <td>348</td>
    </tr>
    <tr>
      <th>29</th>
      <td>person</td>
      <td>16.55%</td>
      <td>365</td>
      <td>319</td>
      <td>391</td>
      <td>344</td>
    </tr>
    <tr>
      <th>30</th>
      <td>person</td>
      <td>16.22%</td>
      <td>621</td>
      <td>339</td>
      <td>639</td>
      <td>403</td>
    </tr>
    <tr>
      <th>31</th>
      <td>potted plant</td>
      <td>16.18%</td>
      <td>0</td>
      <td>361</td>
      <td>215</td>
      <td>470</td>
    </tr>
    <tr>
      <th>32</th>
      <td>person</td>
      <td>15.13%</td>
      <td>279</td>
      <td>313</td>
      <td>300</td>
      <td>387</td>
    </tr>
    <tr>
      <th>33</th>
      <td>person</td>
      <td>10.62%</td>
      <td>428</td>
      <td>312</td>
      <td>444</td>
      <td>337</td>
    </tr>
    <tr>
      <th>34</th>
      <td>umbrella</td>
      <td>10.01%</td>
      <td>215</td>
      <td>252</td>
      <td>313</td>
      <td>315</td>
    </tr>
    <tr>
      <th>35</th>
      <td>umbrella</td>
      <td>9.10%</td>
      <td>295</td>
      <td>294</td>
      <td>346</td>
      <td>357</td>
    </tr>
    <tr>
      <th>36</th>
      <td>umbrella</td>
      <td>7.95%</td>
      <td>358</td>
      <td>293</td>
      <td>402</td>
      <td>319</td>
    </tr>
    <tr>
      <th>37</th>
      <td>umbrella</td>
      <td>7.81%</td>
      <td>319</td>
      <td>307</td>
      <td>344</td>
      <td>356</td>
    </tr>
    <tr>
      <th>38</th>
      <td>potted plant</td>
      <td>7.18%</td>
      <td>166</td>
      <td>331</td>
      <td>221</td>
      <td>439</td>
    </tr>
    <tr>
      <th>39</th>
      <td>umbrella</td>
      <td>6.38%</td>
      <td>129</td>
      <td>264</td>
      <td>200</td>
      <td>360</td>
    </tr>
    <tr>
      <th>40</th>
      <td>person</td>
      <td>5.69%</td>
      <td>428</td>
      <td>318</td>
      <td>450</td>
      <td>343</td>
    </tr>
  </tbody>
</table>
{{</table>}}

### Display the Challenger Results

Here we will use the Wallaroo CVDemo helper class to draw the challenger model results on the input image.

```python
challengerDf = pd.DataFrame(columns=['classification','confidence','x','y','width','height'])
pd.options.mode.chained_assignment = None  # default='warn'
pd.options.display.float_format = '{:.2%}'.format

# Points to where all the inference results are
boxList = infResults.loc[0]["out_resnet50.output"]

# outputs = results['outputs']
# boxes = outputs[0]

# # reshape this to an array of bounding box coordinates converted to ints
# boxList = boxes['Float']['data']
boxA = np.array(boxList)
challengerBoxes = boxA.reshape(-1, 4)
challengerBoxes = challengerBoxes.astype(int)

challengerDf[['x', 'y','width','height']] = pd.DataFrame(challengerBoxes)

challengerClasses = infResults.loc[0]["out_resnet50.3070"]
challengerConfidences = infResults.loc[0]["out_resnet50.3069"]

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

    
![png](03_computer_vision_tutorial_shadow_deploy-reference_files/03_computer_vision_tutorial_shadow_deploy-reference_27_0.png)
    

### Display Challenger Results

The inference results for the objects detected by the challenger model will be displayed including the confidence values.  Once extracted from the results we will want to reshape the flattened array into an array with 4 elements (x,y,width,height).

```python
idx = 0 
for idx in range(0,len(challengerClasses)):
    challengerDf['classification'][idx] = cvDemo.CLASSES[challengerClasses[idx]] # Classes contains the 80 different COCO classificaitons
    challengerDf['confidence'][idx] = challengerConfidences[idx]
challengerDf
```

{{<table "table table-striped table-bordered" >}}
<table>
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
      <td>car</td>
      <td>99.91%</td>
      <td>274</td>
      <td>332</td>
      <td>496</td>
      <td>472</td>
    </tr>
    <tr>
      <th>1</th>
      <td>person</td>
      <td>99.77%</td>
      <td>536</td>
      <td>320</td>
      <td>563</td>
      <td>409</td>
    </tr>
    <tr>
      <th>2</th>
      <td>person</td>
      <td>98.88%</td>
      <td>31</td>
      <td>305</td>
      <td>69</td>
      <td>370</td>
    </tr>
    <tr>
      <th>3</th>
      <td>car</td>
      <td>97.02%</td>
      <td>617</td>
      <td>335</td>
      <td>639</td>
      <td>424</td>
    </tr>
    <tr>
      <th>4</th>
      <td>potted plant</td>
      <td>96.82%</td>
      <td>141</td>
      <td>337</td>
      <td>164</td>
      <td>365</td>
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
      <th>81</th>
      <td>person</td>
      <td>5.61%</td>
      <td>312</td>
      <td>316</td>
      <td>341</td>
      <td>371</td>
    </tr>
    <tr>
      <th>82</th>
      <td>umbrella</td>
      <td>5.60%</td>
      <td>328</td>
      <td>275</td>
      <td>418</td>
      <td>337</td>
    </tr>
    <tr>
      <th>83</th>
      <td>person</td>
      <td>5.54%</td>
      <td>416</td>
      <td>320</td>
      <td>425</td>
      <td>331</td>
    </tr>
    <tr>
      <th>84</th>
      <td>person</td>
      <td>5.52%</td>
      <td>406</td>
      <td>317</td>
      <td>419</td>
      <td>331</td>
    </tr>
    <tr>
      <th>85</th>
      <td>person</td>
      <td>5.14%</td>
      <td>277</td>
      <td>308</td>
      <td>292</td>
      <td>390</td>
    </tr>
  </tbody>
</table>
{{</table>}}
<p>86 rows × 6 columns</p>

```python
pipeline.undeploy()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>shadowimagepipelinetest</td></tr><tr><th>created</th> <td>2023-05-17 22:07:33.073463+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 22:08:33.662212+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>10450273-e76b-4513-a9ba-2467f9d7e149, b41971db-5919-47a1-b2b7-7b7dfb2722d1</td></tr><tr><th>steps</th> <td>mobilenet</td></tr></table>
{{</table>}}

### Conclusion

Notice the difference in the control confidence and the challenger confidence.  <b>Clearly we can see in this example the challenger resnet50 model is performing better than the control mobilenet model</b>.  This is likely due to the fact that frcnn resnet50 model is a 2 stage object detector vs the frcnn mobilenet is a single stage detector.

This completes using Wallaroo's shadow deployment feature to compare different computer vision models.

