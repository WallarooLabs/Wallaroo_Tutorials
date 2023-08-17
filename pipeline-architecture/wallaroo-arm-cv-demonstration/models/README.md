### Models for Wallaroo Computer Vision Tutorials

In order for the wallaroo tutorial notebooks to run properly, the videos directory must contain these models in the models directory.  This is available from the Wallaroo Google Public Data and is available through the following `gcloud` command:

gcloud storage cp "gs://wallaroo-public-data/cv-demo-models/*" .

# Directory contents

coco_classes.pickle - contain the 80 COCO classifications used by resnet50 and mobilenet object detectors.  

frcnn-resent.pt - PyTorch resnet50 model

frcnn-resnet.pt.onnx - PyTorch resnet50 model converted to onyx

mobilenet.pt - PyTorch mobilenet model

mobilenet.pt.onnx - PyTorch mobilenet model converted to onyx
