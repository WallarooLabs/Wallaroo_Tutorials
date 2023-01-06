### Models for Wallaroo Computer Vision Tutorials

In order for the wallaroo tutorial notebooks to run properly, the videos directory must contain these models in the models directory.

The following applications are required:

* Python 3.9
* [google-crc32c](https://pypi.org/project/google-crc32c/) Python library.
* [gcloud](https://cloud.google.com/sdk/docs/install)

1. Assuming gcloud is installed, install `googlecrc` with the following command:

    ```bash
    pip install google-crc32c --upgrade --target /opt/google-cloud-sdk/lib/third_party
    ```

1. Download the models with the following command:

    ```bash
    gcloud storage cp 'gs://wallaroo-model-zoo/open-source/computer-vision/models/*' .
    ```



Use the following cmd in a terminal

gcloud storage cp 'gs://wallaroo-model-zoo/open-source/computer-vision/models/*' .

# Directory contents

coco_classes.pickle - contain the 80 COCO classifications used by resnet50 and mobilenet object detectors.  

frcnn-resent.pt - PyTorch resnet50 model

frcnn-resnet.pt.onnx - PyTorch resnet50 model converted to onyx

mobilenet.pt - PyTorch mobilenet model

mobilenet.pt.onnx - PyTorch mobilenet model converted to onyx
