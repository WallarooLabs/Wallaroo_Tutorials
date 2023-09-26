The following tutorial is available on the [Wallaroo Github Repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/pipeline-edge-publish/edge-cv-healthcare-images).

## Image Detection for Health Care Computer Vision Tutorial Part 00: Prerequisites

The following tutorial demonstrates how to use Wallaroo to detect mitochondria from high resolution images.  For this example we will be using a high resolution 1536x2048 image that is broken down into "patches" of 256x256 images that can be quickly analyzed.

Mitochondria are known as the "powerhouse" of the cell, and having a healthy amount of mitochondria indicates that a patient has enough energy to live a healthy life, or may have underlying issues that a doctor can check for.

Scanning high resolution images of patient cells can be used to count how many mitochondria a patient has, but the process is laborious.  The following ML Model is trained to examine an image of cells, then detect which structures are mitochondria.  This is used to speed up the process of testing patients and determining next steps.

## Prerequisites

The included TiffImagesUtils class is included in this demonstration as `CVDemoUtils.py`, and requires the following dependencies:

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

### Libraries and Dependencies

1. This repository may use large file sizes for the models.  If necessary, install [Git Large File Storage (LFS)](https://git-lfs.com) or use the [Wallaroo Tutorials Releases](https://github.com/WallarooLabs/Wallaroo_Tutorials/releases) to download a .zip file of the most recent computer vision tutorial that includes the models.
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
!pip install torchvision==0.15.2
!pip install torch==2.0.1 --no-cache-dir
!pip install opencv-python==4.7.0.72
!pip install onnx==1.12.0
!pip install onnxruntime==1.15.0
!pip install imutils==0.5.4
!pip install pytz
!pip install ipywidgets==8.0.6
!pip install patchify==0.2.3
!pip install tifffile==2023.4.12
!pip install piexif==1.1.3
```

    Requirement already satisfied: torchvision==0.15.2 in /opt/conda/lib/python3.9/site-packages (0.15.2)
    Requirement already satisfied: requests in /opt/conda/lib/python3.9/site-packages (from torchvision==0.15.2) (2.25.1)
    Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /opt/conda/lib/python3.9/site-packages (from torchvision==0.15.2) (9.2.0)
    Requirement already satisfied: torch==2.0.1 in /opt/conda/lib/python3.9/site-packages (from torchvision==0.15.2) (2.0.1)
    Requirement already satisfied: numpy in /opt/conda/lib/python3.9/site-packages (from torchvision==0.15.2) (1.22.3)
    Requirement already satisfied: nvidia-cuda-nvrtc-cu11==11.7.99 in /opt/conda/lib/python3.9/site-packages (from torch==2.0.1->torchvision==0.15.2) (11.7.99)
    Requirement already satisfied: filelock in /opt/conda/lib/python3.9/site-packages (from torch==2.0.1->torchvision==0.15.2) (3.12.0)
    Requirement already satisfied: jinja2 in /opt/conda/lib/python3.9/site-packages (from torch==2.0.1->torchvision==0.15.2) (3.1.2)
    Requirement already satisfied: nvidia-cufft-cu11==10.9.0.58 in /opt/conda/lib/python3.9/site-packages (from torch==2.0.1->torchvision==0.15.2) (10.9.0.58)
    Requirement already satisfied: nvidia-cudnn-cu11==8.5.0.96 in /opt/conda/lib/python3.9/site-packages (from torch==2.0.1->torchvision==0.15.2) (8.5.0.96)
    Requirement already satisfied: nvidia-cublas-cu11==11.10.3.66 in /opt/conda/lib/python3.9/site-packages (from torch==2.0.1->torchvision==0.15.2) (11.10.3.66)
    Requirement already satisfied: nvidia-cusparse-cu11==11.7.4.91 in /opt/conda/lib/python3.9/site-packages (from torch==2.0.1->torchvision==0.15.2) (11.7.4.91)
    Requirement already satisfied: networkx in /opt/conda/lib/python3.9/site-packages (from torch==2.0.1->torchvision==0.15.2) (3.1)
    Requirement already satisfied: nvidia-nccl-cu11==2.14.3 in /opt/conda/lib/python3.9/site-packages (from torch==2.0.1->torchvision==0.15.2) (2.14.3)
    Requirement already satisfied: nvidia-curand-cu11==10.2.10.91 in /opt/conda/lib/python3.9/site-packages (from torch==2.0.1->torchvision==0.15.2) (10.2.10.91)
    Requirement already satisfied: triton==2.0.0 in /opt/conda/lib/python3.9/site-packages (from torch==2.0.1->torchvision==0.15.2) (2.0.0)
    Requirement already satisfied: nvidia-nvtx-cu11==11.7.91 in /opt/conda/lib/python3.9/site-packages (from torch==2.0.1->torchvision==0.15.2) (11.7.91)
    Requirement already satisfied: typing-extensions in /opt/conda/lib/python3.9/site-packages (from torch==2.0.1->torchvision==0.15.2) (4.3.0)
    Requirement already satisfied: sympy in /opt/conda/lib/python3.9/site-packages (from torch==2.0.1->torchvision==0.15.2) (1.12)
    Requirement already satisfied: nvidia-cusolver-cu11==11.4.0.1 in /opt/conda/lib/python3.9/site-packages (from torch==2.0.1->torchvision==0.15.2) (11.4.0.1)
    Requirement already satisfied: nvidia-cuda-cupti-cu11==11.7.101 in /opt/conda/lib/python3.9/site-packages (from torch==2.0.1->torchvision==0.15.2) (11.7.101)
    Requirement already satisfied: nvidia-cuda-runtime-cu11==11.7.99 in /opt/conda/lib/python3.9/site-packages (from torch==2.0.1->torchvision==0.15.2) (11.7.99)
    Requirement already satisfied: wheel in /opt/conda/lib/python3.9/site-packages (from nvidia-cublas-cu11==11.10.3.66->torch==2.0.1->torchvision==0.15.2) (0.37.1)
    Requirement already satisfied: setuptools in /opt/conda/lib/python3.9/site-packages (from nvidia-cublas-cu11==11.10.3.66->torch==2.0.1->torchvision==0.15.2) (62.3.2)
    Requirement already satisfied: cmake in /opt/conda/lib/python3.9/site-packages (from triton==2.0.0->torch==2.0.1->torchvision==0.15.2) (3.26.4)
    Requirement already satisfied: lit in /opt/conda/lib/python3.9/site-packages (from triton==2.0.0->torch==2.0.1->torchvision==0.15.2) (16.0.5.post0)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /opt/conda/lib/python3.9/site-packages (from requests->torchvision==0.15.2) (1.26.9)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.9/site-packages (from requests->torchvision==0.15.2) (2022.5.18.1)
    Requirement already satisfied: idna<3,>=2.5 in /opt/conda/lib/python3.9/site-packages (from requests->torchvision==0.15.2) (2.10)
    Requirement already satisfied: chardet<5,>=3.0.2 in /opt/conda/lib/python3.9/site-packages (from requests->torchvision==0.15.2) (4.0.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /opt/conda/lib/python3.9/site-packages (from jinja2->torch==2.0.1->torchvision==0.15.2) (2.1.1)
    Requirement already satisfied: mpmath>=0.19 in /opt/conda/lib/python3.9/site-packages (from sympy->torch==2.0.1->torchvision==0.15.2) (1.3.0)
    Requirement already satisfied: torch==2.0.1 in /opt/conda/lib/python3.9/site-packages (2.0.1)
    Requirement already satisfied: jinja2 in /opt/conda/lib/python3.9/site-packages (from torch==2.0.1) (3.1.2)
    Requirement already satisfied: nvidia-cusparse-cu11==11.7.4.91 in /opt/conda/lib/python3.9/site-packages (from torch==2.0.1) (11.7.4.91)
    Requirement already satisfied: networkx in /opt/conda/lib/python3.9/site-packages (from torch==2.0.1) (3.1)
    Requirement already satisfied: nvidia-cuda-nvrtc-cu11==11.7.99 in /opt/conda/lib/python3.9/site-packages (from torch==2.0.1) (11.7.99)
    Requirement already satisfied: nvidia-curand-cu11==10.2.10.91 in /opt/conda/lib/python3.9/site-packages (from torch==2.0.1) (10.2.10.91)
    Requirement already satisfied: triton==2.0.0 in /opt/conda/lib/python3.9/site-packages (from torch==2.0.1) (2.0.0)
    Requirement already satisfied: nvidia-cuda-runtime-cu11==11.7.99 in /opt/conda/lib/python3.9/site-packages (from torch==2.0.1) (11.7.99)
    Requirement already satisfied: nvidia-cufft-cu11==10.9.0.58 in /opt/conda/lib/python3.9/site-packages (from torch==2.0.1) (10.9.0.58)
    Requirement already satisfied: nvidia-nccl-cu11==2.14.3 in /opt/conda/lib/python3.9/site-packages (from torch==2.0.1) (2.14.3)
    Requirement already satisfied: filelock in /opt/conda/lib/python3.9/site-packages (from torch==2.0.1) (3.12.0)
    Requirement already satisfied: nvidia-cuda-cupti-cu11==11.7.101 in /opt/conda/lib/python3.9/site-packages (from torch==2.0.1) (11.7.101)
    Requirement already satisfied: nvidia-cublas-cu11==11.10.3.66 in /opt/conda/lib/python3.9/site-packages (from torch==2.0.1) (11.10.3.66)
    Requirement already satisfied: sympy in /opt/conda/lib/python3.9/site-packages (from torch==2.0.1) (1.12)
    Requirement already satisfied: nvidia-cudnn-cu11==8.5.0.96 in /opt/conda/lib/python3.9/site-packages (from torch==2.0.1) (8.5.0.96)
    Requirement already satisfied: nvidia-cusolver-cu11==11.4.0.1 in /opt/conda/lib/python3.9/site-packages (from torch==2.0.1) (11.4.0.1)
    Requirement already satisfied: nvidia-nvtx-cu11==11.7.91 in /opt/conda/lib/python3.9/site-packages (from torch==2.0.1) (11.7.91)
    Requirement already satisfied: typing-extensions in /opt/conda/lib/python3.9/site-packages (from torch==2.0.1) (4.3.0)
    Requirement already satisfied: setuptools in /opt/conda/lib/python3.9/site-packages (from nvidia-cublas-cu11==11.10.3.66->torch==2.0.1) (62.3.2)
    Requirement already satisfied: wheel in /opt/conda/lib/python3.9/site-packages (from nvidia-cublas-cu11==11.10.3.66->torch==2.0.1) (0.37.1)
    Requirement already satisfied: cmake in /opt/conda/lib/python3.9/site-packages (from triton==2.0.0->torch==2.0.1) (3.26.4)
    Requirement already satisfied: lit in /opt/conda/lib/python3.9/site-packages (from triton==2.0.0->torch==2.0.1) (16.0.5.post0)
    Requirement already satisfied: MarkupSafe>=2.0 in /opt/conda/lib/python3.9/site-packages (from jinja2->torch==2.0.1) (2.1.1)
    Requirement already satisfied: mpmath>=0.19 in /opt/conda/lib/python3.9/site-packages (from sympy->torch==2.0.1) (1.3.0)
    Requirement already satisfied: opencv-python==4.7.0.72 in /opt/conda/lib/python3.9/site-packages (4.7.0.72)
    Requirement already satisfied: numpy>=1.19.3 in /opt/conda/lib/python3.9/site-packages (from opencv-python==4.7.0.72) (1.22.3)
    Requirement already satisfied: onnx==1.12.0 in /opt/conda/lib/python3.9/site-packages (1.12.0)
    Requirement already satisfied: typing-extensions>=3.6.2.1 in /opt/conda/lib/python3.9/site-packages (from onnx==1.12.0) (4.3.0)
    Requirement already satisfied: protobuf<=3.20.1,>=3.12.2 in /opt/conda/lib/python3.9/site-packages (from onnx==1.12.0) (3.19.5)
    Requirement already satisfied: numpy>=1.16.6 in /opt/conda/lib/python3.9/site-packages (from onnx==1.12.0) (1.22.3)
    Requirement already satisfied: onnxruntime==1.15.0 in /opt/conda/lib/python3.9/site-packages (1.15.0)
    Requirement already satisfied: packaging in /opt/conda/lib/python3.9/site-packages (from onnxruntime==1.15.0) (21.3)
    Requirement already satisfied: flatbuffers in /opt/conda/lib/python3.9/site-packages (from onnxruntime==1.15.0) (1.12)
    Requirement already satisfied: protobuf in /opt/conda/lib/python3.9/site-packages (from onnxruntime==1.15.0) (3.19.5)
    Requirement already satisfied: numpy>=1.21.6 in /opt/conda/lib/python3.9/site-packages (from onnxruntime==1.15.0) (1.22.3)
    Requirement already satisfied: coloredlogs in /opt/conda/lib/python3.9/site-packages (from onnxruntime==1.15.0) (15.0.1)
    Requirement already satisfied: sympy in /opt/conda/lib/python3.9/site-packages (from onnxruntime==1.15.0) (1.12)
    Requirement already satisfied: humanfriendly>=9.1 in /opt/conda/lib/python3.9/site-packages (from coloredlogs->onnxruntime==1.15.0) (10.0)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /opt/conda/lib/python3.9/site-packages (from packaging->onnxruntime==1.15.0) (3.0.9)
    Requirement already satisfied: mpmath>=0.19 in /opt/conda/lib/python3.9/site-packages (from sympy->onnxruntime==1.15.0) (1.3.0)
    Requirement already satisfied: imutils==0.5.4 in /opt/conda/lib/python3.9/site-packages (0.5.4)
    Requirement already satisfied: pytz in /opt/conda/lib/python3.9/site-packages (2022.1)
    Requirement already satisfied: ipywidgets==8.0.6 in /opt/conda/lib/python3.9/site-packages (8.0.6)
    Requirement already satisfied: jupyterlab-widgets~=3.0.7 in /opt/conda/lib/python3.9/site-packages (from ipywidgets==8.0.6) (3.0.7)
    Requirement already satisfied: ipython>=6.1.0 in /opt/conda/lib/python3.9/site-packages (from ipywidgets==8.0.6) (7.24.1)
    Requirement already satisfied: ipykernel>=4.5.1 in /opt/conda/lib/python3.9/site-packages (from ipywidgets==8.0.6) (6.13.0)
    Requirement already satisfied: traitlets>=4.3.1 in /opt/conda/lib/python3.9/site-packages (from ipywidgets==8.0.6) (5.2.1.post0)
    Requirement already satisfied: widgetsnbextension~=4.0.7 in /opt/conda/lib/python3.9/site-packages (from ipywidgets==8.0.6) (4.0.7)
    Requirement already satisfied: psutil in /opt/conda/lib/python3.9/site-packages (from ipykernel>=4.5.1->ipywidgets==8.0.6) (5.9.1)
    Requirement already satisfied: matplotlib-inline>=0.1 in /opt/conda/lib/python3.9/site-packages (from ipykernel>=4.5.1->ipywidgets==8.0.6) (0.1.3)
    Requirement already satisfied: debugpy>=1.0 in /opt/conda/lib/python3.9/site-packages (from ipykernel>=4.5.1->ipywidgets==8.0.6) (1.6.0)
    Requirement already satisfied: nest-asyncio in /opt/conda/lib/python3.9/site-packages (from ipykernel>=4.5.1->ipywidgets==8.0.6) (1.5.5)
    Requirement already satisfied: packaging in /opt/conda/lib/python3.9/site-packages (from ipykernel>=4.5.1->ipywidgets==8.0.6) (21.3)
    Requirement already satisfied: tornado>=6.1 in /opt/conda/lib/python3.9/site-packages (from ipykernel>=4.5.1->ipywidgets==8.0.6) (6.1)
    Requirement already satisfied: jupyter-client>=6.1.12 in /opt/conda/lib/python3.9/site-packages (from ipykernel>=4.5.1->ipywidgets==8.0.6) (7.3.1)
    Requirement already satisfied: pexpect>4.3 in /opt/conda/lib/python3.9/site-packages (from ipython>=6.1.0->ipywidgets==8.0.6) (4.8.0)
    Requirement already satisfied: backcall in /opt/conda/lib/python3.9/site-packages (from ipython>=6.1.0->ipywidgets==8.0.6) (0.2.0)
    Requirement already satisfied: jedi>=0.16 in /opt/conda/lib/python3.9/site-packages (from ipython>=6.1.0->ipywidgets==8.0.6) (0.18.1)
    Requirement already satisfied: pickleshare in /opt/conda/lib/python3.9/site-packages (from ipython>=6.1.0->ipywidgets==8.0.6) (0.7.5)
    Requirement already satisfied: decorator in /opt/conda/lib/python3.9/site-packages (from ipython>=6.1.0->ipywidgets==8.0.6) (5.1.1)
    Requirement already satisfied: prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0 in /opt/conda/lib/python3.9/site-packages (from ipython>=6.1.0->ipywidgets==8.0.6) (3.0.29)
    Requirement already satisfied: setuptools>=18.5 in /opt/conda/lib/python3.9/site-packages (from ipython>=6.1.0->ipywidgets==8.0.6) (62.3.2)
    Requirement already satisfied: pygments in /opt/conda/lib/python3.9/site-packages (from ipython>=6.1.0->ipywidgets==8.0.6) (2.12.0)
    Requirement already satisfied: parso<0.9.0,>=0.8.0 in /opt/conda/lib/python3.9/site-packages (from jedi>=0.16->ipython>=6.1.0->ipywidgets==8.0.6) (0.8.3)
    Requirement already satisfied: entrypoints in /opt/conda/lib/python3.9/site-packages (from jupyter-client>=6.1.12->ipykernel>=4.5.1->ipywidgets==8.0.6) (0.4)
    Requirement already satisfied: python-dateutil>=2.8.2 in /opt/conda/lib/python3.9/site-packages (from jupyter-client>=6.1.12->ipykernel>=4.5.1->ipywidgets==8.0.6) (2.8.2)
    Requirement already satisfied: jupyter-core>=4.9.2 in /opt/conda/lib/python3.9/site-packages (from jupyter-client>=6.1.12->ipykernel>=4.5.1->ipywidgets==8.0.6) (4.10.0)
    Requirement already satisfied: pyzmq>=22.3 in /opt/conda/lib/python3.9/site-packages (from jupyter-client>=6.1.12->ipykernel>=4.5.1->ipywidgets==8.0.6) (23.0.0)
    Requirement already satisfied: ptyprocess>=0.5 in /opt/conda/lib/python3.9/site-packages (from pexpect>4.3->ipython>=6.1.0->ipywidgets==8.0.6) (0.7.0)
    Requirement already satisfied: wcwidth in /opt/conda/lib/python3.9/site-packages (from prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0->ipython>=6.1.0->ipywidgets==8.0.6) (0.2.5)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /opt/conda/lib/python3.9/site-packages (from packaging->ipykernel>=4.5.1->ipywidgets==8.0.6) (3.0.9)
    Requirement already satisfied: six>=1.5 in /opt/conda/lib/python3.9/site-packages (from python-dateutil>=2.8.2->jupyter-client>=6.1.12->ipykernel>=4.5.1->ipywidgets==8.0.6) (1.16.0)
    Requirement already satisfied: patchify==0.2.3 in /opt/conda/lib/python3.9/site-packages (0.2.3)
    Requirement already satisfied: numpy<2,>=1 in /opt/conda/lib/python3.9/site-packages (from patchify==0.2.3) (1.22.3)
    Requirement already satisfied: tifffile==2023.4.12 in /opt/conda/lib/python3.9/site-packages (2023.4.12)
    Requirement already satisfied: numpy in /opt/conda/lib/python3.9/site-packages (from tifffile==2023.4.12) (1.22.3)
    Requirement already satisfied: piexif==1.1.3 in /opt/conda/lib/python3.9/site-packages (1.1.3)

### Wallaroo SDK

The Wallaroo SDK is provided by default with the Wallaroo instance's JupyterHub service.  To install the Wallaroo SDK manually, it is provided from the [Python Package Index](https://pypi.org/project/wallaroo/) and is installed with `pip`.  Verify that the same version of the Wallaroo SDK is the same version as the Wallaroo instance.  For example for Wallaroo release 2023.2, the SDK install command is:

```python
pip install wallaroo==2023.3.0
```

See the [Wallaroo SDK Install Guides](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-install-guides/) for full details.
