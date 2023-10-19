The following tutorial is available on the [Wallaroo Github Repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/blob/20231011-2023.4.0-testing/wallaroo-inference-server-tutorials/wallaroo-inference-server-cv-yolov8).

## Wallaroo Inference Server:  Hugging Face Summarizer

This notebook is used in conjunction with the [Wallaroo Inference Server Free Edition](https://docs.wallaroo.ai/wallaroo-inferencing-server/) for Hugging Face Summarizer.  This provides a free license for performing inferences through the [Computer Vision YoloV8n]([Yolov8](https://github.com/ultralytics/ultralytics) ) model.

### Prerequisites

* A deployed Wallaroo Inference Server Free Edition with one of the following options:
  * **Wallaroo.AI Yolov8 Inference Server- x64**
* Access via port 8080 to the Wallaroo Inference Server Free Edition.

## Computer Vision Yolo8 Model Schemas

### Inputs

The Resnet Model takes the following inputs.

| Field | Type | Description |
|---|---|---|
| `tensor` | Float | Tensor in the shape (n, 3, 480, 640) float.  This is the normalized pixel values of the 640x480 color image.

### Outputs

| Field | Type | Description |
|---|---|---|
| `output0` | Variable length *List[Float]* | A flattened numpy array of detected objects.  When reshaped into a `(1, 84, 8400)` returns where the bounding boxes for each detected object are elements `[0:3]` representing (x_coordinate, y_coordinate, width, height), the classes and scores are in elements `[4:]`. |

## Wallaroo Inference Server API Endpoints

The following HTTPS API endpoints are available for Wallaroo Inference Server.

### Pipelines Endpoint

* Endpoint: HTTPS GET `/pipelines`
* Returns:
  * List of `pipelines` with the following fields.
    * **id** (*String*): The name of the pipeline.
    * **status** (*String*): The pipeline status.  `Running` indicates the pipeline is available for inferences.

#### Pipeline Endpoint Example

The following demonstrates using `curl` to retrieve the Pipelines endpoint.  Replace the HOSTNAME with the address of your Wallaroo Inference Server.

```python
!curl HOSTNAME:8080/pipelines
```

    {"pipelines":[{"id":"yolo-v8","status":"Running"}]}

### Models Endpoint

* Endpoint: GET `/models`
* Returns:
  * List of `models` with the following fields.
    * **name** (*String*):  The name of the model.
    * **sha** (*String*):  The `sha` hash of the model.
    * **status** (*String*):  The model status.  `Running` indicates the models is available for inferences.
    * **version** (*String*): The model version in UUID format.

#### Models Endpoint Example

The following demonstrates using `curl` to retrieve the Models endpoint.  Replace the HOSTNAME with the address of your Wallaroo Inference Server.

```python
!curl HOSTNAME:8080/models
```

    {"models":[{"name":"yolo-v8","sha":"3ed5cd199e0e6e419bd3d474cf74f2e378aacbf586e40f24d1f8c89c2c476a08","status":"Running","version":"af82c216-4590-41ad-8579-48b7eccc7144"}]}

### Inference Endpoint

The following endpoints are available from the Wallaroo Server for Computer Vision Yolov8n deployment.

* Endpoint: HTTPS POST `/pipelines/hf-summarizer-standard`
* Headers:
  * `Content-Type: application/vnd.apache.arrow.file`: For Apache Arrow tables.
  * `Content-Type: application/json; format=pandas-records`: For pandas DataFrame in record format.
* Input Parameters:   The images **must** be in 640x640 format converted to a float tensor.DataFrame in `application/json; format=pandas-records` **OR** Apache Arrow table in `application/vnd.apache.arrow.file` with the shape `(n, 3, 640, 640)` then flattened, with the tensor values in the field `images`.

The following code is used to create a DataFrame from a 640x640 image.

  ```python
  import cv2
  import torch
  import numpy as np
  import pandas as pd
  
  # load the image from disk, convert to BGR, resize to specified width, height, convert the image back to RGB
  # convert the image to a float tensor and returns it.  Also return the original resized image for drawing bounding boxes in BGR
  def imageResize(image, 640, 640):
      #self.print("Image Mode:"+image.mode)
      im_pillow = np.array(image)
      image = cv2.cvtColor(im_pillow, cv2.COLOR_BGR2RGB) #scott
      image = cv2.flip(im_pillow, 1)
      image = cv2.flip(image, 1)
      #image = cv2.imread(im_pillow)
      #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      #image = cv2.cvtColor(im_pillow, cv2.COLOR_GRAY2BGR)
      self.debug("Resizing to w:"+str(width) + " height:"+str(height))
      image = cv2.resize(image, (width, height))
      
      # convert the image from BGR to RGB channel ordering and change the
      # image from channels last to channels first ordering
      #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      image = image.transpose((2, 0, 1))

      # add the batch dimension, scale the raw pixel intensities to the
      # range [0, 1], and convert the image to a floating point tensor
      image = np.expand_dims(image, axis=0)
      image = image / 255.0
      tensor = torch.FloatTensor(image)
      tensor.flatten()

      npArray = tensor.cpu().numpy()
      dictData = {"images":[npArray]}
      dataframedata = pd.DataFrame(dictData)
  ```

* Returns:
  * Headers
    * `Content-Type: application/json; format=pandas-records`: pandas DataFrame in record format.
  * Data
    * **time** (*Integer*): The time since UNIX epoch.
    * **in**:  The original input.
      * **images**:  The flattened tensor values for the original image.
    * **out**: The outputs of the inference result separated by data type.
      * **output0**: The float outputs for the inference.  This list is flattened, and when reshaped into `(1,84,8400)` with each **row** correlating to a detected object.  The elements break down as follows:
        * [0:3]: The bounding box with the positions left, top, width, height.
        * [4:]:  The classes and scores of the detected object.

        For more details for breaking down the Yolo8n inference results into objects, see the `CVDemoUtils.py` module with the [Computer Vision Yolov8n Deployment in Wallaroo](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-model-cookbooks/computer-vision-yolov8)

    * **check_failures** (*List[Integer]*): Whether any validation checks were triggered.  For more information, see [Wallaroo SDK Essentials Guide: Pipeline Management: Anomaly Testing]({{<ref "wallaroo-sdk-essentials-pipeline#anomaly-testing">}}).
    * **metadata**: Additional data for the inference.
      * **last_model**:  The model used for the inference.
        * **model_name** (*String*): The name of the model used.
        * **model_sha** (*String*): The sha of the model used.
      * **pipeline_version** (*String*): The pipeline version in UUID format.
      * **elapsed** (*List[Integer]*): A list of time in nanoseconds for:
        * [0] The time to serialize the input.
        * [1...n] How long each step took.
      * **dropped** (*List*): Any dropped input tables.

### Inference Endpoint Example

The Wallaroo Inference Server accepts pandas DataFrame or Apache Arrow tables as inference inputs.  The sample file `./data/dogbike.df.json` was converted from the file `./data/dogbike.png` as an example using the helper module `CVDemoUtils` and `WallarooUtils` are used to transform a sample image into a pandas DataFrame.  This DataFrame is then submitted to the Yolov8n model deployed in Wallaroo.

The following code segment demonstrates converting the image to a DataFrame.

```python
from CVDemoUtils import CVDemo
from WallarooUtils import Util
cvDemo = CVDemo()
util = Util()

width, height = 640, 640
tensor1, resizedImage1 = cvDemo.loadImageAndResize('./data/dogbike.png', width, height)
tensor1.flatten()

# add the tensor to a DataFrame and save the DataFrame in pandas record format
df = util.convert_data(tensor1,'images')
df.to_json("dogbike.df.json", orient = 'records')
```

The following code segment demonstrates performing an inference through the Wallaroo Inference Server with the Yolov8n model deployed.  Replace `HOSTNAME`  with the hostname or IP address of your Wallaroo Inference Server instance.

```python
!curl -X POST HOSTNAME:8080/pipelines/yolo-v8 \
    -H "Content-Type: application/json; format=pandas-records" \
    -d @./data/dogbike.df.json > edge-results.df.json
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100 28.7M  100 13.6M  100 15.0M  12.7M  14.1M  0:00:03  0:00:01  0:00:02 18.9M01  0:00:01 --:--:-- 27.0M

