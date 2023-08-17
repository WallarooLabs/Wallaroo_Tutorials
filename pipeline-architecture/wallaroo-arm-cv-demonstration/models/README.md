### Models for Wallaroo Computer Vision Tutorials

In order for the wallaroo tutorial notebooks to run properly, the videos directory must contain these models in the models directory.

1. Go to https://github.com/WallarooLabs/Wallaroo_Tutorials/releases.
1. Select the most recent release.
1. Download the file `computer_vision.zip`.

This contains the entire tutorial, plus the model files.

# Directory contents

coco_classes.pickle - contain the 80 COCO classifications used by resnet50 and mobilenet object detectors.  

frcnn-resent.pt - PyTorch resnet50 model

frcnn-resnet.pt.onnx - PyTorch resnet50 model converted to onyx

mobilenet.pt - PyTorch mobilenet model

mobilenet.pt.onnx - PyTorch mobilenet model converted to onyx
