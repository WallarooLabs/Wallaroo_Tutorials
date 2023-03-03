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
import os
import numpy as np
import json
import requests
import time
import pandas as pd
from CVDemoUtils import CVDemo

```

### Connect to Wallaroo

Now we connect to the Wallaroo instance.  If you are connecting from a remote connection, set the `wallarooPrefix` and `wallarooSuffix` and use them to connect.  If the connection is from within the Wallaroo instance cluster, then just `wl = wallaroo.Client()` can be used.


```python
# Login through local service

# wl = wallaroo.Client()

# SSO login through keycloak

wallarooPrefix = "YOUR PREFIX"
wallarooSuffix = "YOUR SUFFIX"

wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}.api.{wallarooSuffix}", 
                auth_endpoint=f"https://{wallarooPrefix}.keycloak.{wallarooSuffix}", 
                auth_type="sso")
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

We will now create or connect to an existing pipeline as named in the variables above, then upload each of the models.


```python
pipeline = get_pipeline(pipeline_name)
```


```python
control =  wl.upload_model(control_model_name, control_model_file_name)
```


```python
challenger = wl.upload_model(challenger_model_name, challenger_model_file_name)
```

### Shadow Deploy Pipeline

For this step, rather than deploying each model into a separate step, both will be deployed into a single step as a Shadow Deploy step.  This will take the inference input data and process it through both pipelines at the same time.  The inference results for the control will be stored in it's `['outputs']` array, while the results for the challenger are stored the `['shadow_data']` array.


```python
pipeline.add_shadow_deploy(control, [challenger])

```




<table><tr><th>name</th> <td>shadowimagepipelinetest</td></tr><tr><th>created</th> <td>2023-03-02 19:37:25.349488+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-02 19:37:25.349488+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>474cfb6d-51fc-4e9c-923e-4ca553e73ccd</td></tr><tr><th>steps</th> <td></td></tr></table>




```python
pipeline.deploy()
```




<table><tr><th>name</th> <td>shadowimagepipelinetest</td></tr><tr><th>created</th> <td>2023-03-02 19:37:25.349488+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-02 19:38:07.386270+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>5b41a12c-f643-47ec-8b9f-842e658fd45c, 474cfb6d-51fc-4e9c-923e-4ca553e73ccd</td></tr><tr><th>steps</th> <td>mobilenet</td></tr></table>




```python
pipeline.status()
```




    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.13.25',
       'name': 'engine-6775774cb8-ngrz7',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'shadowimagepipelinetest',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'resnet50',
          'version': 'e30e6def-5e32-40d2-bb9f-11896cc36bd9',
          'sha': 'ee606dc9776a1029420b3adf59b6d29395c89d1d9460d75045a1f2f152d288e7',
          'status': 'Running'},
         {'name': 'mobilenet',
          'version': '483465ed-5f41-488e-8539-66a0b028662b',
          'sha': 'f4c7009e53b679f5e44d70d9612e8dc365565cec88c25b5efa11b903b6b7bdc6',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.12.52',
       'name': 'engine-lb-ddd995646-qrj6m',
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
dictData = {"tensor": npArray.tolist()}

```

### Run Inference using Shadow Deployment

Now lets have the model detect the objects on the image by running inference and extracting the results 


```python
startTime = time.time()
infResults = pipeline.infer(dictData, timeout=60)
endTime = time.time()

if arrowEnabled is True:
    results = infResults[0]
else:
    results = infResults[0].raw
```

### Extract Control Inference Results

First we'll extract the inference result data for the control model and map it onto the image.


```python
df = pd.DataFrame(columns=['classification','confidence','x','y','width','height'])
pd.options.mode.chained_assignment = None  # default='warn'
pd.options.display.float_format = '{:.2%}'.format

# Points to where all the inference results are
outputs = results['outputs']
shadow_data = results['shadow_data']

controlBoxes = outputs[0]

# reshape this to an array of bounding box coordinates converted to ints
boxList = controlBoxes['Float']['data']
boxA = np.array(boxList)
controlBoxes = boxA.reshape(-1, 4)
controlBoxes = controlBoxes.astype(int)

df[['x', 'y','width','height']] = pd.DataFrame(controlBoxes)

controlClasses = outputs[1]['Int64']['data']
controlConfidences = outputs[2]['Float']['data']

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


    
![png](03_computer_vision_tutorial_shadow_deploy-reference_files/03_computer_vision_tutorial_shadow_deploy-reference_25_0.png)
    


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
</div>



### Display the Challenger Results

Here we will use the Wallaroo CVDemo helper class to draw the challenger model results on the input image.


```python
challengerBoxes = shadow_data['resnet50'][0]
# reshape this to an array of bounding box coordinates converted to ints
boxList = challengerBoxes['Float']['data']
boxA = np.array(boxList)
challengerBoxes = boxA.reshape(-1, 4)
challengerBoxes = challengerBoxes.astype(int)

challengerDf = pd.DataFrame(columns=['classification','confidence','x','y','width','height'])
pd.options.mode.chained_assignment = None  # default='warn'
pd.options.display.float_format = '{:.2%}'.format

challengerDf[['x', 'y','width','height']] = pd.DataFrame(challengerBoxes)
#pd.options.display.float_format = '{:.2%}'.format
challengerClasses = shadow_data['resnet50'][1]['Int64']['data']
challengerConfidences = shadow_data['resnet50'][2]['Float']['data']

blue = (255, 0, 0)
results = {
    'model_name' : challenger.name(),
    'pipeline_name' : pipeline.name(),
    'width': width,
    'height': height,
    'image' : challengerImage,
    'boxes' : challengerBoxes,
    'classes' : challengerClasses,
    'confidences' : challengerConfidences,
    'confidence-target' : 0.90,
    'color':CVDemo.BLUE, # color to draw bounding boxes and the text in the statistics
    'inference-time': (endTime-startTime),
    'onnx-time' : 0,                
}
cvDemo.drawAndDisplayDetectedObjectsWithClassification(results)
```


    
![png](03_computer_vision_tutorial_shadow_deploy-reference_files/03_computer_vision_tutorial_shadow_deploy-reference_29_0.png)
    



### Display Challenger Results

The inference results for the objects detected by the challenger model will be displayed including the confidence values.  Once extracted from the results we will want to reshape the flattened array into an array with 4 elements (x,y,width,height).


```python
idx = 0 
for idx in range(0,len(challengerClasses)):
    challengerDf['classification'][idx] = cvDemo.CLASSES[challengerClasses[idx]] # Classes contains the 80 different COCO classificaitons
    challengerDf['confidence'][idx] = challengerConfidences[idx]
challengerDf
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
<p>86 rows × 6 columns</p>
</div>




```python
pipeline.undeploy()
```




<table><tr><th>name</th> <td>shadowimagepipelinetest</td></tr><tr><th>created</th> <td>2023-03-02 19:37:25.349488+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-02 19:38:07.386270+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>5b41a12c-f643-47ec-8b9f-842e658fd45c, 474cfb6d-51fc-4e9c-923e-4ca553e73ccd</td></tr><tr><th>steps</th> <td>mobilenet</td></tr></table>



### Conclusion

Notice the difference in the control confidence and the challenger confidence.  <b>Clearly we can see in this example the challenger resnet50 model is performing better than the control mobilenet model</b>.  This is likely due to the fact that frcnn resnet50 model is a 2 stage object detector vs the frcnn mobilenet is a single stage detector.

This completes using Wallaroo's shadow deployment feature to compare different computer vision models.

