The following tutorial is available on the [Wallaroo Github Repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/blob/20231004-wallaroo-inference-server/wallaroo-model-cookbooks/computer-vision-yolov8).

## Computer Vision Yolov8n Deployment in Wallaroo

The [Yolov8](https://github.com/ultralytics/ultralytics) computer vision model is used for fast recognition of objects in images.  This tutorial demonstrates how to deploy a Yolov8n pre-trained model into a Wallaroo Ops server and perform inferences on it.

For this tutorial, the helper module `CVDemoUtils` and `WallarooUtils` are used to transform a sample image into a pandas DataFrame.  This DataFrame is then submitted to the Yolov8n model deployed in Wallaroo.

This demonstration follows these steps:

* Upload the Yolo8 model to Wallaroo
* Add the Yolo8 model as a Wallaroo pipeline step
* Deploy the Wallaroo pipeline and allocate cluster resources to the pipeline
* Perform sample inferences
* Undeploy and return the resources 

## References

* [Wallaroo Workspaces](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-workspace/):  Workspaces are environments were users upload models, create pipelines and other artifacts.  The workspace should be considered the fundamental area where work is done.  Workspaces are shared with other users to give them access to the same models, pipelines, etc.
* [Wallaroo Model Upload and Registration](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/): ML Models are uploaded to Wallaroo through the SDK or the MLOps API to a **workspace**.  ML models include default runtimes (ONNX, Python Step, and TensorFlow) that are run directly through the Wallaroo engine, and containerized runtimes (Hugging Face, PyTorch, etc) that are run through in a container through the Wallaroo engine.
* [Wallaroo Pipelines](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/): Pipelines are used to deploy models for inferencing.  Each model is a **pipeline step** in a pipelines, where the inputs of the previous step are fed into the next.  Pipeline steps can be ML models, Python scripts, or Arbitrary Python (these contain necessary models and artifacts for running a model).

## Steps

### Load Libraries

The first step is loading the required libraries including the [Wallaroo Python module](https://pypi.org/project/wallaroo/).

```python
# Import Wallaroo Python SDK
import wallaroo
from wallaroo.object import EntityNotFoundError
from wallaroo.framework import Framework
from CVDemoUtils import CVDemo
from WallarooUtils import Util
cvDemo = CVDemo()
util = Util()

# used to display DataFrame information without truncating
from IPython.display import display
import pandas as pd
pd.set_option('display.max_colwidth', None)

```

### Connect to the Wallaroo Instance through the User Interface

The next step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
wl = wallaroo.Client()
```

### Create a New Workspace

We'll use the SDK below to create our workspace , assign as our **current workspace**, then display all of the workspaces we have at the moment.  We'll also set up variables for our models and pipelines down the road, so we have one spot to change names to whatever fits your organization's standards best.

To allow this tutorial to be run by multiple users in the same Wallaroo instance, a random 4 character prefix will be added to the workspace, pipeline, and model.  Feel free to set `suffix=''` if this is not required.

```python
# import string
# import random

# # make a random 4 character suffix to verify uniqueness in tutorials
# suffix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))

suffix = ''

model_name = 'yolov8n'
model_filename = 'models/yolov8n.onnx'
pipeline_name = 'yolo8demonstration'
workspace_name = f'yolo8-demonstration{suffix}'

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
```

### Upload the Model

When a model is uploaded to a Wallaroo cluster, it is optimized and packaged to make it ready to run as part of a pipeline. In many times, the Wallaroo Server can natively run a model without any Python overhead. In other cases, such as a Python script, a custom Python environment will be automatically generated. This is comparable to the process of "containerizing" a model by adding a small HTTP server and other wrapping around it.

Our pretrained model is in ONNX format, which is specified in the `framework` parameter.  For this model, the tensor fields are set to `images` to match the input parameters, and the batch configuration is set to `single` - only one record will be submitted at a time.

```python
# Upload Retrained Yolo8 Model 
yolov8_model = (wl.upload_model(model_name, 
                               model_filename, 
                               framework=Framework.ONNX)
                               .configure(tensor_fields=['images'],
                                          batch_config="single"
                                          )
                )
```

### Pipeline Deployment Configuration

For our pipeline we set the deployment configuration to only use 1 cpu and 1 GiB of RAM.

```python
deployment_config = wallaroo.DeploymentConfigBuilder() \
                    .replica_count(1) \
                    .cpus(1) \
                    .memory("1Gi") \
                    .build()
```

### Build and Deploy the Pipeline

Now we build our pipeline and set our Yolo8 model as a pipeline step, then deploy the pipeline using the deployment configuration above.

```python
pipeline = wl.build_pipeline(pipeline_name) \
            .add_model_step(yolov8_model)        
```

```python
pipeline.deploy(deployment_config=deployment_config)
```

<table><tr><th>name</th> <td>yolo8demonstration</td></tr><tr><th>created</th> <td>2023-10-19 15:33:26.144685+00:00</td></tr><tr><th>last_updated</th> <td>2023-10-19 15:33:27.154726+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>39b02242-de8b-46cd-849e-8a896226a84a, bed60f2a-ddd6-4a48-a9fd-debe6b1e1bca</td></tr><tr><th>steps</th> <td>yolov8n</td></tr><tr><th>published</th> <td>False</td></tr></table>

### Convert Image to DataFrame

The sample image `dogbike.png` was converted to a DataFrame using the `cvDemo` helper modules.  The converted DataFrame is stored as `./data/dogbike.df.json` to save time.

The code sample below demonstrates how to use this module to convert the sample image to a DataFrame.

```python
# convert the image to a tensor

width, height = 640, 640
tensor1, resizedImage1 = cvDemo.loadImageAndResize('dogbike.png', width, height)
tensor1.flatten()

# add the tensor to a DataFrame and save the DataFrame in pandas record format
df = util.convert_data(tensor1,'images')
df.to_json("dogbike.df.json", orient = 'records')
```

```python
# convert the image to a tensor

width, height = 640, 640
tensor1, resizedImage1 = cvDemo.loadImageAndResize('./data/dogbike.png', width, height)

tensor1.flatten()

# add the tensor to a DataFrame and save the DataFrame in pandas record format
df = util.convert_data(tensor1,'images')
df.to_json("dogbike.df.json", orient = 'records')
```

### Inference Request

We submit the DataFrame to the pipeline using `wallaroo.pipeline.infer_from_file`, and store the results in the variable `inf1`.

```python
inf1 = pipeline.infer_from_file('./data/dogbike.df.json')
```

### Display Bounding Boxes

Using our helper method `cvDemo` we'll identify the objects detected in the photo and their bounding boxes.  Only objects with a confidence threshold of 50% or more are shown.

```python
inf1.loc[:, ['out.output0']]
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out.output0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[17.09787, 16.459343, 17.259743, 19.960602, 43.600235, 59.986958, 62.826073, 68.24793, 77.43261, 80.82158, 89.44183, 96.168915, 99.22421, 112.584045, 126.75803, 131.9707, 137.1645, 141.93822, 146.29594, 152.00876, 155.94037, 165.20976, 175.27249, 184.05307, 193.66891, 201.51189, 215.04979, 223.80424, 227.24472, 234.19638, 244.9743, 248.5781, 252.42526, 264.95795, 278.48563, 285.758, 293.1897, 300.48227, 305.47742, 314.46085, 319.89404, 324.83658, 335.99536, 345.1116, 350.31964, 352.41107, 365.44934, 381.30008, 391.52316, 399.29163, 405.78503, 411.33804, 415.93207, 421.6868, 431.67108, 439.9069, 447.71542, 459.38522, 474.13187, 479.32642, 484.49884, 493.5153, 501.29932, 507.7967, 514.26044, 523.1473, 531.3479, 542.5094, 555.619, 557.7229, 564.6408, 571.5525, 572.8373, 587.95703, 604.2997, 609.452, 616.31714, 623.5797, 624.13153, 634.47266, 16.970057, 16.788723, 17.441803, 17.900642, 36.188023, 57.277973, 61.664352, 62.556896, 63.43486, 79.50621, 83.844, 95.983765, 106.166, 115.368454, 123.09253, 124.5821, 128.65866, 139.16113, 142.02315, 143.69855, ...]</td>
    </tr>
  </tbody>
</table>

```python
confidence_thres = 0.50
iou_thres = 0.25

cvDemo.drawYolo8Boxes(inf1, resizedImage1, width, height, confidence_thres, iou_thres, draw=True)
```

      Score: 86.47% | Class: Dog | Bounding Box: [108, 250, 149, 356]
      Score: 81.13% | Class: Bicycle | Bounding Box: [97, 149, 375, 323]
      Score: 63.16% | Class: Car | Bounding Box: [390, 85, 186, 108]

    
{{<figure src="/images/2023.4.0/wallaroo-tutorials/computer-vision/yolov8/computer-vision-yolov8-demonstration-reference_files/computer-vision-yolov8-demonstration-reference_21_1.png" width="800" label="png">}}
    

    array([[[ 34,  34,  34],
            [ 35,  35,  35],
            [ 33,  33,  33],
            ...,
            [ 33,  33,  33],
            [ 33,  33,  33],
            [ 35,  35,  35]],
    
           [[ 33,  33,  33],
            [ 34,  34,  34],
            [ 34,  34,  34],
            ...,
            [ 34,  34,  34],
            [ 33,  33,  33],
            [ 34,  34,  34]],
    
           [[ 53,  54,  48],
            [ 54,  55,  49],
            [ 54,  55,  49],
            ...,
            [153, 178, 111],
            [151, 183, 108],
            [159, 176,  99]],
    
           ...,
    
           [[159, 167, 178],
            [159, 165, 177],
            [158, 163, 175],
            ...,
            [126, 127, 121],
            [127, 125, 120],
            [128, 120, 117]],
    
           [[160, 168, 179],
            [156, 162, 174],
            [152, 157, 169],
            ...,
            [126, 127, 121],
            [129, 127, 122],
            [127, 118, 116]],
    
           [[155, 163, 174],
            [155, 162, 174],
            [152, 158, 170],
            ...,
            [127, 127, 121],
            [130, 126, 122],
            [128, 119, 116]]], dtype=uint8)

### Inference Through Pipeline API

Another method of performing an inference using the pipeline's deployment url.

Performing an inference through an API requires the following:

* The authentication token to authorize the connection to the pipeline.
* The pipeline's inference URL.
* Inference data to sent to the pipeline - in JSON, DataFrame records format, or Apache Arrow.

Full details are available through the [Wallaroo API Connection Guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-connection-guide/) on how retrieve an authorization token and perform inferences through the pipeline's API.

For this demonstration we'll submit the pandas record, request a pandas record as the return, and set the authorization header.  The results will be stored in the file `curl_response.df`.

```python
deploy_url = pipeline._deployment._url()

headers = wl.auth.auth_header()

headers['Content-Type']='application/json; format=pandas-records'
headers['Accept']='application/json; format=pandas-records'
```

```python
!curl -X POST {deploy_url} \
    -H "Authorization:{headers['Authorization']}" \
    -H "Content-Type:application/json; format=pandas-records" \
    -H "Accept:application/json; format=pandas-records" \
    --data @./data/dogbike.df.json > curl_response.df
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100 38.0M  100 22.9M  100 15.0M  5624k  3701k  0:00:04  0:00:04 --:--:-- 9334k

```python
pipeline.undeploy()
```

<table><tr><th>name</th> <td>yolo8demonstration</td></tr><tr><th>created</th> <td>2023-10-11 14:37:32.252497+00:00</td></tr><tr><th>last_updated</th> <td>2023-10-11 14:59:03.213137+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>2a3933c4-52db-40c6-b80f-9031664fd08a, 95bbabf1-1f15-4e4b-9e67-f7730c2b2cbd, 6c672144-ed4f-4505-97eb-a5b1763af847, 7149e0bc-089b-4d57-9a0b-5d4f4a9a4097, 329e394b-5105-4dc3-b0ff-5411623fc139, 7acaea4e-6ae3-426b-9f97-5e3dcc39c48e, a8b2c009-e7b5-4b96-81b9-40447797a05f, 09952a45-2401-4ebd-8e85-c678365b64a7, d870a558-10ef-448e-b00d-068c10c7e82b, fa531e16-1706-43c4-98d9-e0dd6355fe6f, 4c0b535e-b39b-40f4-82a7-34965b2f7c2a, 3507964d-382f-4e1c-84c7-64c5e27f819c, 9971f8dd-a17b-4d6a-ab72-d786d4990fab, b92a035f-903c-4039-8303-8ceb979a53c2</td></tr><tr><th>steps</th> <td>yolov8n</td></tr><tr><th>published</th> <td>False</td></tr></table>

