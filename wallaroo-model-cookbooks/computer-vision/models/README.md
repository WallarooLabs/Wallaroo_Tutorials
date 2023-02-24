### Models for Wallaroo Computer Vision Tutorials

In order for the wallaroo tutorial notebooks to run properly, the videos directory must contain these models in the models directory.  Use either [Git Large File Storage (LFS)](https://git-lfs.com) or the [Computer Vision direct download](https://github.com/WallarooLabs/Wallaroo_Tutorials/releases/download/1.27-2022.4-cv4/computer-vision.zip)

# Directory contents

coco_classes.pickle - contain the 80 COCO classifications used by resnet50 and mobilenet object detectors.  

frcnn-resent.pt - PyTorch resnet50 model

frcnn-resnet.pt.onnx - PyTorch resnet50 model converted to onyx

mobilenet.pt - PyTorch mobilenet model

mobilenet.pt.onnx - PyTorch mobilenet model converted to onyx
