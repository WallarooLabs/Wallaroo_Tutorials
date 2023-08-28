This tutorial can be found on the [Wallaroo Tutorials Github Repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/blob/main/wallaroo-model-cookbooks/computer-vision-mitochondria-imaging).

## Image Detection for Health Care Computer Vision Tutorial Part 01: Mitochondria Detection

The following tutorial demonstrates how to use Wallaroo to detect mitochondria from high resolution images.  For this example we will be using a high resolution 1536x2048 image that is broken down into "patches" of 256x256 images that can be quickly analyzed.

Mitochondria are known as the "powerhouse" of the cell, and having a healthy amount of mitochondria indicates that a patient has enough energy to live a healthy life, or may have underling issues that a doctor can check for.

Scanning high resolution images of patient cells can be used to count how many mitochondria a patient has, but the process is laborious.  The following ML Model is trained to examine an image of cells, then detect which structures are mitochondria.  This is used to speed up the process of testing patients and determining next steps.

## Tutorial Goals

This tutorial will perform the following:

1. Upload and deploy the `mitochondria_epochs_15.onnx` model to a Wallaroo pipeline.
1. Randomly select from from a selection of 256x256 images that were originally part of a larger 1536x2048 image.
1. Convert the images into a numpy array inserted into a pandas DataFrame.
1. Submit the DataFrame to the Wallaroo pipeline and use the results to create a mask image of where the model detects mitochondria.
1. Compare the original image against a map of "ground truth" and the model's mask image.
1. Undeploy the pipeline and return the resources back to the Wallaroo instance.

## Prerequisites

Complete the steps from Mitochondria Detection Computer Vision Tutorial Part 00: Prerequisites.

## Mitochondria Computer Vision Detection Steps

### Import Libraries

The first step is to import the necessary libraries.  Included with this tutorial are the following custom modules:

* `tiff_utils`: Organizes the tiff images to perform random image selections and other tasks.

Note that tensorflow may return warnings depending on the environment.

```python
import json
import IPython.display as display
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from lib.TiffImageUtils import TiffUtils
import tifffile as tiff

import pandas as pd

import wallaroo
from wallaroo.object import EntityNotFoundError

import numpy as np
from matplotlib import pyplot as plt
import cv2
from keras.utils import normalize

tiff_utils = TiffUtils()

# used for unique connection names

import string
import random

suffix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))

# ignoring warnings for demonstration
import warnings
warnings.filterwarnings('ignore')
```

### Open a Connection to Wallaroo

The next step is connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  If logging in externally, update the `wallarooPrefix` and `wallarooSuffix` variables with the proper DNS information.  For more information on Wallaroo DNS settings, see the [Wallaroo DNS Integration Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-dns-guide/).

```python
wl = wallaroo.Client()
```

### Create Workspace and Pipeline

We will create a workspace to manage our pipeline and models.  The following variables will set the name of our sample workspace then set it as the current workspace.

Workspace, pipeline, and model names should be unique to each Wallaroo instance, so we'll add in a randomly generated suffix so multiple people can run this tutorial in a Wallaroo instance without affecting each other.

```python
workspace_name = f'biolabsworkspace{suffix}'
pipeline_name = f'biolabspipeline{suffix}'
model_name = f'biolabsmodel{suffix}'
model_file_name = 'models/mitochondria_epochs_15.onnx'
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

```python
workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)

pipeline = get_pipeline(pipeline_name)
pipeline
```

<table><tr><th>name</th> <td>biolabspipelinebspy</td></tr><tr><th>created</th> <td>2023-07-14 15:28:32.639523+00:00</td></tr><tr><th>last_updated</th> <td>2023-07-14 15:28:32.639523+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>c70dbdfe-e380-41b0-9da6-97bbfae90554</td></tr><tr><th>steps</th> <td></td></tr></table>

### Upload the Models

Now we will:

1. Upload our model.
1. Apply it as a step in our pipeline.
1. Create a pipeline deployment with enough memory to perform the inferences.
1. Deploy the pipeline.

```python
deployment_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(1).memory("2Gi").build()

model = wl.upload_model(model_name, model_file_name, framework=wallaroo.framework.Framework.ONNX)

pipeline = wl.build_pipeline(pipeline_name) \
            .add_model_step(model) \
            .deploy(deployment_config = deployment_config)
```

### Retrieve Image and Convert to Data

The next step is to process the image into a numpy array that the model is trained to detect from.

We start by retrieving all the patch images from a recorded time series tiff recorded on one of our microscopes.

```python
sample_mitochondria_patches_path = "./patches/ms-01-atl-3-22-23_9-50"

patches = tiff_utils.get_all_patches(sample_mitochondria_patches_path)
```

Randomly we will retrieve a 256x256 patch image and use it to do our semantic segmentation prediction.

We'll then convert it into a numpy array and insert into a DataFrame for a single inference.

The following helper function `loadImageAndConvertTiff` is used to convert the image into a numpy, then insert that into the DataFrame.  This allows a later command to take the randomly grabbed image perform the process on other images.

```python
def loadImageAndConvertTiff(imagePath, width, height):
    img = cv2.imread(imagePath, 0)
    imgNorm = np.expand_dims(normalize(np.array(img), axis=1),2)
    imgNorm=imgNorm[:,:,0][:,:,None]
    imgNorm=np.expand_dims(imgNorm, 0)
    
    resizedImage = None
    #creates a dictionary with the wallaroo "tensor" key and the numpy ndim array representing image as the value.
    dictData = {"tensor":[imgNorm]}
    dataframedata = pd.DataFrame(dictData)
    # display(dataframedata)
    return dataframedata, resizedImage
```

```python
def run_semantic_segmentation_inference(pipeline, input_tiff_image, width, height, threshold):
    
    tensor, resizedImage = loadImageAndConvertTiff(input_tiff_image, width, height)
    # print(tensor)

    # #
    # # run inference on the 256x256 patch image get the predicted mitochandria mask
    # #
    output = pipeline.infer(tensor)
    # print(output)

    # # Obtain the flattened predicted mitochandria mask result
    list1d = output.loc[0]["out.conv2d_37"]
    np1d = np.array(list1d)
    
    # # unflatten it
    predicted_mask = np1d.reshape(1,width,height,1)
    
    # # perform the element-wise comaprison operation using the threshold provided
    predicted_mask = (predicted_mask[0,:,:,0] > threshold).astype(np.uint8)
    
    # return predicted_mask
    return predicted_mask
```

### Infer and Display Results

We will now perform our inferences and display the results.  This results in a predicted mask showing us where the mitochondria cells are located.

1. The first image is the input image.
1. The 2nd image is the ground truth. The mask was created by a human who identified the mitochondria cells in the input image
1. The 3rd image is the predicted mask after running inference on the Wallaroo pipeline.

We'll perform this 10 times to show how quickly the inferences can be submitted.

```python
for x in range(10):     
    # get a sample 256x256 mitochondria image
    random_patch = tiff_utils.get_random_patch_sample(patches)

    # build the path to the image
    patch_image_path = sample_mitochondria_patches_path + "/images/" + random_patch['patch_image_file']

    # run inference in order to get the predicted 256x256 mask
    predicted_mask = run_semantic_segmentation_inference(pipeline, patch_image_path, 256,256, 0.2)

    # # plot the results
    test_image = random_patch['patch_image'][:,:,0]
    test_image_title = f"Testing Image - {random_patch['index']}"

    ground_truth_image = random_patch['patch_mask'][:,:,0]
    ground_truth_image_title = "Ground Truth Mask"

    predicted_mask_title = 'Predicted Mask'

    tiff_utils.plot_test_results(test_image, test_image_title, \
                            ground_truth_image, ground_truth_image_title, \
                            predicted_mask, predicted_mask_title)
```

    
{{<figure src="/images/current/wallaroo-tutorials/wallaroo-tutorials-model-cookbooks/computer-vision-mitochondria/01_computer-vision-mitochondria-imaging-example_files/01_computer-vision-mitochondria-imaging-example_18_0.png" width="800" label="png">}}
    

    
{{<figure src="/images/current/wallaroo-tutorials/wallaroo-tutorials-model-cookbooks/computer-vision-mitochondria/01_computer-vision-mitochondria-imaging-example_files/01_computer-vision-mitochondria-imaging-example_18_1.png" width="800" label="png">}}
    

    
{{<figure src="/images/current/wallaroo-tutorials/wallaroo-tutorials-model-cookbooks/computer-vision-mitochondria/01_computer-vision-mitochondria-imaging-example_files/01_computer-vision-mitochondria-imaging-example_18_2.png" width="800" label="png">}}
    

    
{{<figure src="/images/current/wallaroo-tutorials/wallaroo-tutorials-model-cookbooks/computer-vision-mitochondria/01_computer-vision-mitochondria-imaging-example_files/01_computer-vision-mitochondria-imaging-example_18_3.png" width="800" label="png">}}
    

    
{{<figure src="/images/current/wallaroo-tutorials/wallaroo-tutorials-model-cookbooks/computer-vision-mitochondria/01_computer-vision-mitochondria-imaging-example_files/01_computer-vision-mitochondria-imaging-example_18_4.png" width="800" label="png">}}
    

    
{{<figure src="/images/current/wallaroo-tutorials/wallaroo-tutorials-model-cookbooks/computer-vision-mitochondria/01_computer-vision-mitochondria-imaging-example_files/01_computer-vision-mitochondria-imaging-example_18_5.png" width="800" label="png">}}
    

    
{{<figure src="/images/current/wallaroo-tutorials/wallaroo-tutorials-model-cookbooks/computer-vision-mitochondria/01_computer-vision-mitochondria-imaging-example_files/01_computer-vision-mitochondria-imaging-example_18_6.png" width="800" label="png">}}
    

    
{{<figure src="/images/current/wallaroo-tutorials/wallaroo-tutorials-model-cookbooks/computer-vision-mitochondria/01_computer-vision-mitochondria-imaging-example_files/01_computer-vision-mitochondria-imaging-example_18_7.png" width="800" label="png">}}
    

    
{{<figure src="/images/current/wallaroo-tutorials/wallaroo-tutorials-model-cookbooks/computer-vision-mitochondria/01_computer-vision-mitochondria-imaging-example_files/01_computer-vision-mitochondria-imaging-example_18_8.png" width="800" label="png">}}
    

    
{{<figure src="/images/current/wallaroo-tutorials/wallaroo-tutorials-model-cookbooks/computer-vision-mitochondria/01_computer-vision-mitochondria-imaging-example_files/01_computer-vision-mitochondria-imaging-example_18_9.png" width="800" label="png">}}
    

### Complete Tutorial

With the demonstration complete, the pipeline is undeployed and the resources returned back to the Wallaroo instance.

```python
pipeline.undeploy()
```

<table><tr><th>name</th> <td>biolabspipelinebspy</td></tr><tr><th>created</th> <td>2023-07-14 15:28:32.639523+00:00</td></tr><tr><th>last_updated</th> <td>2023-07-14 15:28:38.163950+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>0460ef47-3de3-43b2-8f62-16e76be8ce93, ef41bc7b-8213-4dd4-a1b9-54ac1253c652, c70dbdfe-e380-41b0-9da6-97bbfae90554</td></tr><tr><th>steps</th> <td>biolabsmodelbspy</td></tr></table>

