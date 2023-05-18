### Models for Wallaroo Computer Vision Tutorials

In order for the wallaroo tutorial notebooks to run properly, the videos directory must contain these models in the models directory.

<<<<<<< HEAD
=======
To download the Wallaroo Computer Vision tutorials

git clone https://github.com/WallarooLabs/csa_demo.git

cd computer-vision/models

>>>>>>> 07b717a (pipeline logs and other updates/)
Use the following cmd in a terminal

gcloud storage cp gs://wallaroo-model-zoo/open-source/computer-vision/models/* .

# Directory contents

coco_classes.pickle - contain the 80 COCO classifications used by resnet50 and mobilenet object detectors.  

frcnn-resent.pt - PyTorch resnet50 model

frcnn-resnet.pt.onnx - PyTorch resnet50 model converted to onyx

mobilenet.pt - PyTorch mobilenet model

mobilenet.pt.onnx - PyTorch mobilenet model converted to onyx
