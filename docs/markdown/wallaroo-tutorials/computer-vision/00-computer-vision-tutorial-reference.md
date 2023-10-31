This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-model-cookbooks/computer-vision).

## Step 00: Introduction and Setup

This tutorial demonstrates how to use the Wallaroo to detect objects in images through the following models:

* **rnn mobilenet**: A single stage object detector that performs fast inferences.  Mobilenet is typically good at identifying objects at a distance.
* **resnet50**:  A dual stage object detector with slower inferencing but but is able to detect objects that are closer to each other.

This tutorial series will demonstrate the following:

* How to deploy a Wallaroo pipeline with trained rnn mobilenet model and perform sample inferences to detect objects in pictures, then display those objects.
* How to deploy a Wallaroo pipeline with a trained resnet50 model and perform sample inferences to detect objects in pictures, then display those objects.
* Use the Wallaroo feature shadow deploy to have both models perform inferences, then select the inference result with the higher confidence and show the objects detected.

This tutorial assumes that users have installed the [Wallaroo SDK](https://pypi.org/project/wallaroo/) or are running these tutorials from within their Wallaroo instance's JupyterHub service.

This demonstration should be run within a Wallaroo JupyterHub instance for best results.

## Prerequisites

The included OpenCV class is included in this demonstration as `CVDemoUtils.py`, and requires the following dependencies:

* ffmpeg
* libsm
* libxext

### Internal JupyterHub Service

To install these dependencies in the Wallaroo JupyterHub service, use the following commands from a terminal shell via the following procedure:

1. Launch the JupyterHub Service within the Wallaroo install.
1. Select **File->New->Terminal**.
1. Enter the following:

    ```bash
    sudo apt-get update
    ```

    ```bash
    sudo apt-get install ffmpeg libsm6 libxext6  -y
    ```

### External SDK Users

For users using the Wallaroo SDK to connect with a remote Wallaroo instance, the following commands will install the required dependancies:

For Linux users, this can be installed with:

```bash
sudo apt-get update
sudo apt-get install ffmpeg libsm6 libxext6  -y
```

MacOS users can prepare their environments using a package manager such as [Brew](https://brew.sh/) with the following:

```bash
brew install ffmpeg libsm libxext
```

### Libraries and Dependencies

1. This repository may use large file sizes for the models.  Use the [Wallaroo Tutorials Releases](https://github.com/WallarooLabs/Wallaroo_Tutorials/releases) to download a .zip file of the most recent computer vision tutorial that includes the models.
1. Import the following Python libraries into your environment:
    1. [torch](https://pypi.org/project/torch/)
    1. [wallaroo](https://pypi.org/project/wallaroo/)
    1. [torchvision](https://pypi.org/project/torchvision/)
    1. [opencv-python](https://pypi.org/project/opencv-python/)
    1. [onnx](https://pypi.org/project/onnx/)
    1. [onnxruntime](https://pypi.org/project/onnxruntime/)
    1. [imutils](https://pypi.org/project/imutils/)
    1. [pytz](https://pypi.org/project/pytz/)
    1. [ipywidgets](https://pypi.org/project/ipywidgets/)

These can be installed by running the command below in the Wallaroo JupyterHub service.  Note the use of `pip install torch --no-cache-dir` for low memory environments.

```python
!pip install torchvision
!pip install torch --no-cache-dir
!pip install opencv-python
!pip install onnx
!pip install onnxruntime
!pip install imutils
!pip install pytz
!pip install ipywidgets
```

    Requirement already satisfied: torchvision in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0/lib/python3.8/site-packages (0.14.1)
    Requirement already satisfied: typing-extensions in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0/lib/python3.8/site-packages (from torchvision) (4.5.0)
    Requirement already satisfied: numpy in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0/lib/python3.8/site-packages (from torchvision) (1.22.3)
    Requirement already satisfied: requests in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0/lib/python3.8/site-packages (from torchvision) (2.25.1)
    Requirement already satisfied: torch in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0/lib/python3.8/site-packages (from torchvision) (1.13.1)
    Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0/lib/python3.8/site-packages (from torchvision) (10.1.0)
    Requirement already satisfied: chardet<5,>=3.0.2 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0/lib/python3.8/site-packages (from requests->torchvision) (4.0.0)
    Requirement already satisfied: idna<3,>=2.5 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0/lib/python3.8/site-packages (from requests->torchvision) (2.10)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0/lib/python3.8/site-packages (from requests->torchvision) (1.26.18)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0/lib/python3.8/site-packages (from requests->torchvision) (2023.7.22)
    Requirement already satisfied: torch in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0/lib/python3.8/site-packages (1.13.1)
    Requirement already satisfied: typing-extensions in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0/lib/python3.8/site-packages (from torch) (4.5.0)
    Requirement already satisfied: opencv-python in /Users/johnhansarick/.local/lib/python3.8/site-packages (4.8.1.78)
    Requirement already satisfied: numpy>=1.21.0 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0/lib/python3.8/site-packages (from opencv-python) (1.22.3)
    Requirement already satisfied: onnx in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0/lib/python3.8/site-packages (1.15.0)
    Requirement already satisfied: numpy in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0/lib/python3.8/site-packages (from onnx) (1.22.3)
    Requirement already satisfied: protobuf>=3.20.2 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0/lib/python3.8/site-packages (from onnx) (4.24.4)
    Collecting onnxruntime
      Using cached onnxruntime-1.16.1-cp38-cp38-macosx_11_0_arm64.whl.metadata (4.1 kB)
    Collecting coloredlogs (from onnxruntime)
      Using cached coloredlogs-15.0.1-py2.py3-none-any.whl (46 kB)
    Requirement already satisfied: flatbuffers in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0/lib/python3.8/site-packages (from onnxruntime) (1.12)
    Requirement already satisfied: numpy>=1.21.6 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0/lib/python3.8/site-packages (from onnxruntime) (1.22.3)
    Requirement already satisfied: packaging in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0/lib/python3.8/site-packages (from onnxruntime) (23.1)
    Requirement already satisfied: protobuf in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0/lib/python3.8/site-packages (from onnxruntime) (4.24.4)
    Collecting sympy (from onnxruntime)
      Using cached sympy-1.12-py3-none-any.whl (5.7 MB)
    Collecting humanfriendly>=9.1 (from coloredlogs->onnxruntime)
      Using cached humanfriendly-10.0-py2.py3-none-any.whl (86 kB)
    Collecting mpmath>=0.19 (from sympy->onnxruntime)
      Using cached mpmath-1.3.0-py3-none-any.whl (536 kB)
    Using cached onnxruntime-1.16.1-cp38-cp38-macosx_11_0_arm64.whl (6.1 MB)
    Installing collected packages: mpmath, sympy, humanfriendly, coloredlogs, onnxruntime
    Successfully installed coloredlogs-15.0.1 humanfriendly-10.0 mpmath-1.3.0 onnxruntime-1.16.1 sympy-1.12
    Collecting imutils
      Using cached imutils-0.5.4-py3-none-any.whl
    Installing collected packages: imutils
    Successfully installed imutils-0.5.4
    Requirement already satisfied: pytz in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0/lib/python3.8/site-packages (2023.3.post1)
    Collecting ipywidgets
      Using cached ipywidgets-8.1.1-py3-none-any.whl.metadata (2.4 kB)
    Collecting comm>=0.1.3 (from ipywidgets)
      Using cached comm-0.1.4-py3-none-any.whl.metadata (4.2 kB)
    Requirement already satisfied: ipython>=6.1.0 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0/lib/python3.8/site-packages (from ipywidgets) (7.24.1)
    Requirement already satisfied: traitlets>=4.3.1 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0/lib/python3.8/site-packages (from ipywidgets) (5.12.0)
    Collecting widgetsnbextension~=4.0.9 (from ipywidgets)
      Using cached widgetsnbextension-4.0.9-py3-none-any.whl.metadata (1.6 kB)
    Collecting jupyterlab-widgets~=3.0.9 (from ipywidgets)
      Using cached jupyterlab_widgets-3.0.9-py3-none-any.whl.metadata (4.1 kB)
    Requirement already satisfied: setuptools>=18.5 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets) (68.0.0)
    Requirement already satisfied: jedi>=0.16 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets) (0.18.1)
    Requirement already satisfied: decorator in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets) (5.1.1)
    Requirement already satisfied: pickleshare in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets) (0.7.5)
    Requirement already satisfied: prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets) (3.0.36)
    Requirement already satisfied: pygments in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets) (2.15.1)
    Requirement already satisfied: backcall in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets) (0.2.0)
    Requirement already satisfied: matplotlib-inline in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets) (0.1.6)
    Requirement already satisfied: pexpect>4.3 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets) (4.8.0)
    Requirement already satisfied: appnope in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets) (0.1.2)
    Requirement already satisfied: parso<0.9.0,>=0.8.0 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0/lib/python3.8/site-packages (from jedi>=0.16->ipython>=6.1.0->ipywidgets) (0.8.3)
    Requirement already satisfied: ptyprocess>=0.5 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0/lib/python3.8/site-packages (from pexpect>4.3->ipython>=6.1.0->ipywidgets) (0.7.0)
    Requirement already satisfied: wcwidth in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0/lib/python3.8/site-packages (from prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0->ipython>=6.1.0->ipywidgets) (0.2.5)
    Using cached ipywidgets-8.1.1-py3-none-any.whl (139 kB)
    Using cached comm-0.1.4-py3-none-any.whl (6.6 kB)
    Using cached jupyterlab_widgets-3.0.9-py3-none-any.whl (214 kB)
    Using cached widgetsnbextension-4.0.9-py3-none-any.whl (2.3 MB)
    Installing collected packages: widgetsnbextension, jupyterlab-widgets, comm, ipywidgets
      Attempting uninstall: comm
        Found existing installation: comm 0.1.2
        Uninstalling comm-0.1.2:
          Successfully uninstalled comm-0.1.2
    Successfully installed comm-0.1.4 ipywidgets-8.1.1 jupyterlab-widgets-3.0.9 widgetsnbextension-4.0.9

The rest of the tutorials will rely on these libraries and applications, so finish their installation before running the tutorials in this series.

## Models for Wallaroo Computer Vision Tutorials

In order for the wallaroo tutorial notebooks to run properly, the videos directory must contain these models in the models directory.

To download the Wallaroo Computer Vision models, use the following link:

https://storage.googleapis.com/wallaroo-public-data/cv-demo-models/cv-retail-models.zip

Unzip the contents into the directory `models`.

### Directory contents

* coco_classes.pickle - contain the 80 COCO classifications used by resnet50 and mobilenet object detectors.  
* frcnn-resent.pt - PyTorch resnet50 model
* frcnn-resnet.pt.onnx - PyTorch resnet50 model converted to onnx
* mobilenet.pt - PyTorch mobilenet model
* mobilenet.pt.onnx - PyTorch mobilenet model converted to onnx

