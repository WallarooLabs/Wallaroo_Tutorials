# computer-vision

Contains notebooks that demonstrate how to improve your computer vision models using the Wallaroo platform.

## Getting Started


There are 2 object detectors featured.

mobilenet - is a single stage object detector that performs fast inferennces.  Typically good at identifying objects at a distance as well.

resnet50 - is a dual stage object detector that is slower at inferencing, but is able to detect objects that are closer to each other.

The playbooks below show you how to use wallaroo to compare these models in a simulated production environment to determine the pros and cons of each object detector.


## The notebooks

Simply execute playbooks 1-4 to learn how to integrate your computer vision models into a wallaroo cluster.

playbook-1-pipeline-mobilenet.ipynb - Shows you how easy it is to deploy a pipeline that uses the mobilenet object detector into a wallaroo cluster.  The notebook runs inference on a sampple image.  Detects the bounding boxes, the COCO classifications identified, and the classificaiton confidence percentage.  It then draws the results on top of the input image and display them.

playbook-1-pipeline-resnet50.ipynb - Shows you how easy it is to deploy a pipeline that uses resnet50 object detector into a wallaroo cluster.  The notebook runs inference on a sampple image.  Detects the bounding boxes, the COCO classifications identified, and the classificaiton confidence percentage.  It then draws the results on top of the input image and display them.

playbook-2-dahdow-deployment.ipynb - Shows you how to deploy both the mobilenet and the resnet50 models into a shadaw deployment pipeline.  It then compares the statistics of the control and challenger in a data frame and draws the inference results on the sample input images so that the data scientist gets a full picture of how these 2 models are stacking up against each other


playbook-3-pipeline-anomoly-mobilenet.ipynb - Is all about finding the opportunities to improve our models.  In this notebook we build a custom anomoly object dector that identifies inferences that have a classification percentage below 75%.   The results a displayed in a. dataframe and drawn on the input image to give the data scientist the full picture. This helps the data scientist easily identify the objects that are fallling below the companies mandated classification percentage.  So when they retrain the model what objects identified would demonstrate a measurable improvement.

playbook-3-pipeline-anomoly-resnet50.ipynb - Lets perform the anomoly detection on the resnet50 model as well.

playbook-4-pipeline-drift - Is all about identifying drift in our object detectors

## Directory Structure explained

data - contains the sample imput images and videos used by the object detectors to run inference.  The output directory contains the inference results drawn on the sample image or frames in a mp4 video.

examples - contains additional examples of how to use wallaroo in different domains/verticals

models - contains the object detectors used in these notebooks

object_detectors - contains notebooks that show how to download the mobilenet and resnet50 object detectors in pytorch, run inference using these object detectors, and draw the bounding box for the identified objects, their COCO classifications, and percentage confidence in those classifications.  It then save those models to a pickle file for future use.

In addition there are notebooks that will load the pytorch models from the pickle file and convert the models to onnx.  We then run inference using these object detectors, and draw the bounding box for the identified objects, their COCO classifications, and percentage confidence in those classifications to ensure we are getting the same results.

Lastly, we save the ONNX models to onnx file format so they can be uploaded to the wallaroo cluster for the playbook notebooks.

video - contains notebooks that demonstrate how to use wallaroo to improve object detection in video streams that use computer vision

CVDemoUtils.py - is a wallaroo python class that performs all the drawing of inferenced results on the input images and video frames.