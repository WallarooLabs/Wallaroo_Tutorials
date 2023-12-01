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

    Collecting torchvision==0.15.2
      Downloading torchvision-0.15.2-cp38-cp38-macosx_11_0_arm64.whl (1.4 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.4/1.4 MB[0m [31m13.0 MB/s[0m eta [36m0:00:00[0m [36m0:00:01[0m
    [?25hRequirement already satisfied: numpy in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from torchvision==0.15.2) (1.22.3)
    Requirement already satisfied: requests in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from torchvision==0.15.2) (2.25.1)
    Collecting torch==2.0.1 (from torchvision==0.15.2)
      Downloading torch-2.0.1-cp38-none-macosx_11_0_arm64.whl (55.8 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m55.8/55.8 MB[0m [31m21.0 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hRequirement already satisfied: pillow!=8.3.*,>=5.3.0 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from torchvision==0.15.2) (10.1.0)
    Collecting filelock (from torch==2.0.1->torchvision==0.15.2)
      Using cached filelock-3.13.1-py3-none-any.whl.metadata (2.8 kB)
    Requirement already satisfied: typing-extensions in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from torch==2.0.1->torchvision==0.15.2) (4.7.1)
    Collecting sympy (from torch==2.0.1->torchvision==0.15.2)
      Using cached sympy-1.12-py3-none-any.whl (5.7 MB)
    Collecting networkx (from torch==2.0.1->torchvision==0.15.2)
      Using cached networkx-3.1-py3-none-any.whl (2.1 MB)
    Collecting jinja2 (from torch==2.0.1->torchvision==0.15.2)
      Using cached Jinja2-3.1.2-py3-none-any.whl (133 kB)
    Requirement already satisfied: chardet<5,>=3.0.2 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from requests->torchvision==0.15.2) (4.0.0)
    Requirement already satisfied: idna<3,>=2.5 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from requests->torchvision==0.15.2) (2.10)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from requests->torchvision==0.15.2) (1.26.18)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from requests->torchvision==0.15.2) (2023.11.17)
    Collecting MarkupSafe>=2.0 (from jinja2->torch==2.0.1->torchvision==0.15.2)
      Using cached MarkupSafe-2.1.3-cp38-cp38-macosx_10_9_universal2.whl.metadata (3.0 kB)
    Collecting mpmath>=0.19 (from sympy->torch==2.0.1->torchvision==0.15.2)
      Using cached mpmath-1.3.0-py3-none-any.whl (536 kB)
    Using cached filelock-3.13.1-py3-none-any.whl (11 kB)
    Using cached MarkupSafe-2.1.3-cp38-cp38-macosx_10_9_universal2.whl (17 kB)
    Installing collected packages: mpmath, sympy, networkx, MarkupSafe, filelock, jinja2, torch, torchvision
    Successfully installed MarkupSafe-2.1.3 filelock-3.13.1 jinja2-3.1.2 mpmath-1.3.0 networkx-3.1 sympy-1.12 torch-2.0.1 torchvision-0.15.2
    Requirement already satisfied: torch==2.0.1 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (2.0.1)
    Requirement already satisfied: filelock in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from torch==2.0.1) (3.13.1)
    Requirement already satisfied: typing-extensions in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from torch==2.0.1) (4.7.1)
    Requirement already satisfied: sympy in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from torch==2.0.1) (1.12)
    Requirement already satisfied: networkx in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from torch==2.0.1) (3.1)
    Requirement already satisfied: jinja2 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from torch==2.0.1) (3.1.2)
    Requirement already satisfied: MarkupSafe>=2.0 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from jinja2->torch==2.0.1) (2.1.3)
    Requirement already satisfied: mpmath>=0.19 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from sympy->torch==2.0.1) (1.3.0)
    Collecting opencv-python==4.7.0.72
      Downloading opencv_python-4.7.0.72-cp37-abi3-macosx_11_0_arm64.whl (32.6 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m32.6/32.6 MB[0m [31m70.8 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hRequirement already satisfied: numpy>=1.21.0 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from opencv-python==4.7.0.72) (1.22.3)
    Installing collected packages: opencv-python
      Attempting uninstall: opencv-python
        Found existing installation: opencv-python 4.8.1.78
        Uninstalling opencv-python-4.8.1.78:
          Successfully uninstalled opencv-python-4.8.1.78
    Successfully installed opencv-python-4.7.0.72
    Collecting onnx==1.12.0
      Using cached onnx-1.12.0.tar.gz (10.1 MB)
      Preparing metadata (setup.py) ... [?25ldone
    [?25hRequirement already satisfied: numpy>=1.16.6 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from onnx==1.12.0) (1.22.3)
    Collecting protobuf<=3.20.1,>=3.12.2 (from onnx==1.12.0)
      Using cached protobuf-3.20.1-py2.py3-none-any.whl (162 kB)
    Requirement already satisfied: typing-extensions>=3.6.2.1 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from onnx==1.12.0) (4.7.1)
    Building wheels for collected packages: onnx
      Building wheel for onnx (setup.py) ... [?25lerror
      [1;31merror[0m: [1msubprocess-exited-with-error[0m
      
      [31m×[0m [32mpython setup.py bdist_wheel[0m did not run successfully.
      [31m│[0m exit code: [1;36m1[0m
      [31m╰─>[0m [31m[675 lines of output][0m
      [31m   [0m fatal: not a git repository (or any of the parent directories): .git
      [31m   [0m /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages/setuptools/__init__.py:84: _DeprecatedInstaller: setuptools.installer and fetch_build_eggs are deprecated.
      [31m   [0m !!
      [31m   [0m 
      [31m   [0m         ********************************************************************************
      [31m   [0m         Requirements should be satisfied by a PEP 517 installer.
      [31m   [0m         If you are using pip, you can try `pip install --use-pep517`.
      [31m   [0m         ********************************************************************************
      [31m   [0m 
      [31m   [0m !!
      [31m   [0m   dist.fetch_build_eggs(dist.setup_requires)
      [31m   [0m running bdist_wheel
      [31m   [0m running build
      [31m   [0m running build_py
      [31m   [0m running create_version
      [31m   [0m running cmake_build
      [31m   [0m Using cmake args: ['/opt/homebrew/bin/cmake', '-DPYTHON_INCLUDE_DIR=/opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/include/python3.8', '-DPYTHON_EXECUTABLE=/opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/bin/python', '-DBUILD_ONNX_PYTHON=ON', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', '-DONNX_NAMESPACE=onnx', '-DPY_EXT_SUFFIX=.cpython-38-darwin.so', '-DCMAKE_BUILD_TYPE=Release', '-DONNX_ML=1', '/private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f']
      [31m   [0m [0mCMake Deprecation Warning at CMakeLists.txt:2 (cmake_minimum_required):
      [31m   [0m   Compatibility with CMake < 3.5 will be removed from a future version of
      [31m   [0m   CMake.
      [31m   [0m 
      [31m   [0m   Update the VERSION argument <min> value or use a ...<max> suffix to tell
      [31m   [0m   CMake that the project does not need compatibility with older versions.
      [31m   [0m 
      [31m   [0m [0m
      [31m   [0m -- The C compiler identification is AppleClang 15.0.0.15000040
      [31m   [0m -- The CXX compiler identification is AppleClang 15.0.0.15000040
      [31m   [0m -- Detecting C compiler ABI info
      [31m   [0m -- Detecting C compiler ABI info - done
      [31m   [0m -- Check for working C compiler: /Library/Developer/CommandLineTools/usr/bin/cc - skipped
      [31m   [0m -- Detecting C compile features
      [31m   [0m -- Detecting C compile features - done
      [31m   [0m -- Detecting CXX compiler ABI info
      [31m   [0m -- Detecting CXX compiler ABI info - done
      [31m   [0m -- Check for working CXX compiler: /Library/Developer/CommandLineTools/usr/bin/c++ - skipped
      [31m   [0m -- Detecting CXX compile features
      [31m   [0m -- Detecting CXX compile features - done
      [31m   [0m [33mCMake Warning (dev) at CMakeLists.txt:117 (find_package):
      [31m   [0m   Policy CMP0148 is not set: The FindPythonInterp and FindPythonLibs modules
      [31m   [0m   are removed.  Run "cmake --help-policy CMP0148" for policy details.  Use
      [31m   [0m   the cmake_policy command to set the policy and suppress this warning.
      [31m   [0m 
      [31m   [0m This warning is for project developers.  Use -Wno-dev to suppress it.
      [31m   [0m [0m
      [31m   [0m -- Found PythonInterp: /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/bin/python (found version "3.8.18")
      [31m   [0m [33mCMake Warning (dev) at CMakeLists.txt:118 (find_package):
      [31m   [0m   Policy CMP0148 is not set: The FindPythonInterp and FindPythonLibs modules
      [31m   [0m   are removed.  Run "cmake --help-policy CMP0148" for policy details.  Use
      [31m   [0m   the cmake_policy command to set the policy and suppress this warning.
      [31m   [0m 
      [31m   [0m This warning is for project developers.  Use -Wno-dev to suppress it.
      [31m   [0m [0m
      [31m   [0m -- Found PythonLibs: /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/libpython3.8.dylib (found version "3.8.18")
      [31m   [0m -- Found Protobuf: /opt/homebrew/lib/libprotobuf.dylib (found version "4.25.1")
      [31m   [0m [0mGenerated: /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-ml.proto[0m
      [31m   [0m [0mGenerated: /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-operators-ml.proto[0m
      [31m   [0m [0mGenerated: /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-data.proto[0m
      [31m   [0m -- Could NOT find pybind11 (missing: pybind11_DIR)
      [31m   [0m [0mCMake Deprecation Warning at third_party/pybind11/CMakeLists.txt:8 (cmake_minimum_required):
      [31m   [0m   Compatibility with CMake < 3.5 will be removed from a future version of
      [31m   [0m   CMake.
      [31m   [0m 
      [31m   [0m   Update the VERSION argument <min> value or use a ...<max> suffix to tell
      [31m   [0m   CMake that the project does not need compatibility with older versions.
      [31m   [0m 
      [31m   [0m [0m
      [31m   [0m -- pybind11 v2.9.1
      [31m   [0m [33mCMake Warning (dev) at third_party/pybind11/tools/FindPythonLibsNew.cmake:98 (find_package):
      [31m   [0m   Policy CMP0148 is not set: The FindPythonInterp and FindPythonLibs modules
      [31m   [0m   are removed.  Run "cmake --help-policy CMP0148" for policy details.  Use
      [31m   [0m   the cmake_policy command to set the policy and suppress this warning.
      [31m   [0m 
      [31m   [0m Call Stack (most recent call first):
      [31m   [0m   third_party/pybind11/tools/pybind11Tools.cmake:50 (find_package)
      [31m   [0m   third_party/pybind11/tools/pybind11Common.cmake:206 (include)
      [31m   [0m   third_party/pybind11/CMakeLists.txt:200 (include)
      [31m   [0m This warning is for project developers.  Use -Wno-dev to suppress it.
      [31m   [0m [0m
      [31m   [0m -- Found PythonLibs: /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/libpython3.8.dylib
      [31m   [0m -- Performing Test HAS_FLTO
      [31m   [0m -- Performing Test HAS_FLTO - Success
      [31m   [0m -- Performing Test HAS_FLTO_THIN
      [31m   [0m -- Performing Test HAS_FLTO_THIN - Success
      [31m   [0m --
      [31m   [0m -- ******** Summary ********
      [31m   [0m --   CMake version             : 3.27.8
      [31m   [0m --   CMake command             : /opt/homebrew/Cellar/cmake/3.27.8/bin/cmake
      [31m   [0m --   System                    : Darwin
      [31m   [0m --   C++ compiler              : /Library/Developer/CommandLineTools/usr/bin/c++
      [31m   [0m --   C++ compiler version      : 15.0.0.15000040
      [31m   [0m --   CXX flags                 :  -Wnon-virtual-dtor
      [31m   [0m --   Build type                : Release
      [31m   [0m --   Compile definitions       : __STDC_FORMAT_MACROS
      [31m   [0m --   CMAKE_PREFIX_PATH         :
      [31m   [0m --   CMAKE_INSTALL_PREFIX      : /usr/local
      [31m   [0m --   CMAKE_MODULE_PATH         :
      [31m   [0m --
      [31m   [0m --   ONNX version              : 1.12.0
      [31m   [0m --   ONNX NAMESPACE            : onnx
      [31m   [0m --   ONNX_USE_LITE_PROTO       : OFF
      [31m   [0m --   USE_PROTOBUF_SHARED_LIBS  : OFF
      [31m   [0m --   Protobuf_USE_STATIC_LIBS  : ON
      [31m   [0m --   ONNX_DISABLE_EXCEPTIONS   : OFF
      [31m   [0m --   ONNX_WERROR               : OFF
      [31m   [0m --   ONNX_BUILD_TESTS          : OFF
      [31m   [0m --   ONNX_BUILD_BENCHMARKS     : OFF
      [31m   [0m --   ONNXIFI_DUMMY_BACKEND     : OFF
      [31m   [0m --   ONNXIFI_ENABLE_EXT        : OFF
      [31m   [0m --
      [31m   [0m --   Protobuf compiler         : /opt/homebrew/bin/protoc
      [31m   [0m --   Protobuf includes         : /opt/homebrew/include
      [31m   [0m --   Protobuf libraries        : /opt/homebrew/lib/libprotobuf.dylib
      [31m   [0m --   BUILD_ONNX_PYTHON         : ON
      [31m   [0m --     Python version        :
      [31m   [0m --     Python executable     : /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/bin/python
      [31m   [0m --     Python includes       : /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/include/python3.8
      [31m   [0m -- Configuring done (4.3s)
      [31m   [0m -- Generating done (0.0s)
      [31m   [0m -- Build files have been written to: /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build
      [31m   [0m [  1%] [32mBuilding C object CMakeFiles/onnxifi_loader.dir/onnx/onnxifi_loader.c.o[0m
      [31m   [0m [  2%] [34m[1mRunning gen_proto.py on onnx/onnx.in.proto[0m
      [31m   [0m [  4%] [32mBuilding C object CMakeFiles/onnxifi_dummy.dir/onnx/onnxifi_dummy.c.o[0m
      [31m   [0m Processing /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/onnx/onnx.in.proto
      [31m   [0m Writing /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-ml.proto
      [31m   [0m Writing /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-ml.proto3
      [31m   [0m generating /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx_pb.py
      [31m   [0m [  5%] [34m[1mRunning C++ protocol buffer compiler on /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-ml.proto[0m
      [31m   [0m [  8%] [32m[1mLinking C shared library libonnxifi_dummy.dylib[0m
      [31m   [0m [  8%] [32m[1mLinking C static library libonnxifi_loader.a[0m
      [31m   [0m [  8%] Built target onnxifi_dummy
      [31m   [0m [  8%] Built target onnxifi_loader
      [31m   [0m [  9%] [32mBuilding C object CMakeFiles/onnxifi_wrapper.dir/onnx/onnxifi_wrapper.c.o[0m
      [31m   [0m Writing mypy to onnx/onnx_ml_pb2.pyi
      [31m   [0m [ 11%] [32m[1mLinking C shared module libonnxifi.dylib[0m
      [31m   [0m [ 11%] Built target gen_onnx_proto
      [31m   [0m [ 12%] [34m[1mRunning gen_proto.py on onnx/onnx-data.in.proto[0m
      [31m   [0m [ 14%] [34m[1mRunning gen_proto.py on onnx/onnx-operators.in.proto[0m
      [31m   [0m [ 14%] Built target onnxifi_wrapper
      [31m   [0m Processing /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/onnx/onnx-operators.in.proto
      [31m   [0m Processing /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/onnx/onnx-data.in.proto
      [31m   [0m Writing /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-operators-ml.proto
      [31m   [0m Writing /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-data.proto
      [31m   [0m Writing /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-operators-ml.proto3
      [31m   [0m Writing /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-data.proto3
      [31m   [0m generating /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx_operators_pb.py
      [31m   [0m generating /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx_data_pb.py
      [31m   [0m [ 15%] [34m[1mRunning C++ protocol buffer compiler on /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-operators-ml.proto[0m
      [31m   [0m [ 16%] [34m[1mRunning C++ protocol buffer compiler on /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-data.proto[0m
      [31m   [0m Writing mypy to onnx/onnx_operators_ml_pb2.pyi
      [31m   [0m Writing mypy to onnx/onnx_data_pb2.pyi
      [31m   [0m [ 16%] Built target gen_onnx_operators_proto
      [31m   [0m [ 16%] Built target gen_onnx_data_proto
      [31m   [0m [ 19%] [32mBuilding CXX object CMakeFiles/onnx_proto.dir/onnx/onnx-operators-ml.pb.cc.o[0m
      [31m   [0m [ 19%] [32mBuilding CXX object CMakeFiles/onnx_proto.dir/onnx/onnx-ml.pb.cc.o[0m
      [31m   [0m [ 21%] [32mBuilding CXX object CMakeFiles/onnx_proto.dir/onnx/onnx-data.pb.cc.o[0m
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-operators-ml.pb.cc:4:
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-operators-ml.pb.h:13:
      [31m   [0m In file included from /opt/homebrew/include/google/protobuf/port_def.inc:33:
      [31m   [0m In file included from /opt/homebrew/include/absl/base/attributes.h:37:
      [31m   [0m In file included from /opt/homebrew/include/absl/base/config.h:86:
      [31m   [0m /opt/homebrew/include/absl/base/policy_checks.h:79:2: error: "C++ versions less than C++14 are not supported."
      [31m   [0m #error "C++ versions less than C++14 are not supported."
      [31m   [0m  ^
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-data.pb.cc:4:
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-data.pb.h:13:
      [31m   [0m In file included from /opt/homebrew/include/google/protobuf/port_def.inc:33:
      [31m   [0m In file included from /opt/homebrew/include/absl/base/attributes.h:37:
      [31m   [0m In file included from /opt/homebrew/include/absl/base/config.h:86:
      [31m   [0m /opt/homebrew/include/absl/base/policy_checks.h:79:2: error: "C++ versions less than C++14 are not supported."
      [31m   [0m #error "C++ versions less than C++14 are not supported."
      [31m   [0m  ^
      [31m   [0m In file included from In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-data.pb.cc/private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-operators-ml.pb.cc::44:
      [31m   [0m :
      [31m   [0m In file included from In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-data.pb.h/private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-operators-ml.pb.h::1313:
      [31m   [0m :
      [31m   [0m /opt/homebrew/include/google/protobuf/port_def.inc/opt/homebrew/include/google/protobuf/port_def.inc::159159::11::  error: error: static assertion failed due to requirement '201103L >= 201402L': Protobuf only supports C++14 and newer.static assertion failed due to requirement '201103L >= 201402L': Protobuf only supports C++14 and newer.
      [31m   [0m static_assert(PROTOBUF_CPLUSPLUS_MIN(201402L), "Protobuf only supports C++14 and newer.");
      [31m   [0m ^             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      [31m   [0m 
      [31m   [0m static_assert(PROTOBUF_CPLUSPLUS_MIN(201402L), "Protobuf only supports C++14 and newer.");
      [31m   [0m ^             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-ml.pb.cc:4:
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-ml.pb.h:13:
      [31m   [0m In file included from /opt/homebrew/include/google/protobuf/port_def.inc:33:
      [31m   [0m In file included from /opt/homebrew/include/absl/base/attributes.h:37:
      [31m   [0m In file included from /opt/homebrew/include/absl/base/config.h:86:
      [31m   [0m /opt/homebrew/include/absl/base/policy_checks.h:79:2: error: "C++ versions less than C++14 are not supported."
      [31m   [0m #error "C++ versions less than C++14 are not supported."
      [31m   [0m  ^
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-ml.pb.cc:4:
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-ml.pb.h:13:
      [31m   [0m /opt/homebrew/include/google/protobuf/port_def.inc:159:1: error: static assertion failed due to requirement '201103L >= 201402L': Protobuf only supports C++14 and newer.
      [31m   [0m static_assert(PROTOBUF_CPLUSPLUS_MIN(201402L), "Protobuf only supports C++14 and newer.");
      [31m   [0m ^             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      [31m   [0m In file included from In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-operators-ml.pb.cc/private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-ml.pb.cc:4:
      [31m   [0m In file included from :4:
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-operators-ml.pb.h:26:
      [31m   [0m In file included from /opt/homebrew/include/google/protobuf/io/coded_stream.h:107:
      [31m   [0m In file included from /opt/homebrew/include/google/protobuf/stubs/common.h:22:
      [31m   [0m In file included from /opt/homebrew/include/google/protobuf/stubs/port.h:22:
      [31m   [0m /opt/homebrew/include/google/protobuf/port_def.inc:159:1: error: static assertion failed due to requirement '201103L >= 201402L': Protobuf only supports C++14 and newer.
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-data.pb.cc:4:
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-data.pb.h:26:
      [31m   [0m In file included from static_assert(PROTOBUF_CPLUSPLUS_MIN(201402L), "Protobuf only supports C++14 and newer.");
      [31m   [0m ^             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      [31m   [0m /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-ml.pb.h:26:
      [31m   [0m In file included from /opt/homebrew/include/google/protobuf/io/coded_stream.h:107:
      [31m   [0m In file included from /opt/homebrew/include/google/protobuf/stubs/common.h:22:
      [31m   [0m In file included from /opt/homebrew/include/google/protobuf/stubs/port.h:22:
      [31m   [0m /opt/homebrew/include/google/protobuf/port_def.inc:159:1: error: static assertion failed due to requirement '201103L >= 201402L': Protobuf only supports C++14 and newer.
      [31m   [0m /opt/homebrew/include/google/protobuf/io/coded_stream.h:107:
      [31m   [0m In file included from /opt/homebrew/include/google/protobuf/stubs/common.h:22:
      [31m   [0m In file included from /opt/homebrew/include/google/protobuf/stubs/port.h:22:
      [31m   [0m /opt/homebrew/include/google/protobuf/port_def.inc:159:1: error: static assertion failed due to requirement '201103L >= 201402L': Protobuf only supports C++14 and newer.
      [31m   [0m static_assert(PROTOBUF_CPLUSPLUS_MIN(201402L), "Protobuf only supports C++14 and newer.");
      [31m   [0m ^             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      [31m   [0m static_assert(PROTOBUF_CPLUSPLUS_MIN(201402L), "Protobuf only supports C++14 and newer.");
      [31m   [0m ^             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      [31m   [0m In file included from In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-operators-ml.pb.cc:4:
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-operators-ml.pb.h:/private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-data.pb.cc:4:
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-data.pb.h:26:
      [31m   [0m In file included from /opt/homebrew/include/google/protobuf/io/coded_stream.h:107:
      [31m   [0m In file included from /opt/homebrew/include/google/protobuf/stubs/common.h:34:
      [31m   [0m /opt/homebrew/include/google/protobuf/port_def.inc:159:1: error: static assertion failed due to requirement '201103L >= 201402L': Protobuf only supports C++14 and newer.
      [31m   [0m static_assert(PROTOBUF_CPLUSPLUS_MIN(201402L), "Protobuf only supports C++14 and newer.");
      [31m   [0m ^             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-ml.pb.cc:4:
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-ml.pb.h:26:
      [31m   [0m In file included from /opt/homebrew/include/google/protobuf/io/coded_stream.h:107:
      [31m   [0m In file included from /opt/homebrew/include/google/protobuf/stubs/common.h:34:
      [31m   [0m /opt/homebrew/include/google/protobuf/port_def.inc:159:1: error: static assertion failed due to requirement '201103L >= 201402L': Protobuf only supports C++14 and newer.
      [31m   [0m static_assert(PROTOBUF_CPLUSPLUS_MIN(201402L), "Protobuf only supports C++14 and newer.");
      [31m   [0m ^             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      [31m   [0m 26:
      [31m   [0m In file included from /opt/homebrew/include/google/protobuf/io/coded_stream.h:107:
      [31m   [0m In file included from /opt/homebrew/include/google/protobuf/stubs/common.h:34:
      [31m   [0m /opt/homebrew/include/google/protobuf/port_def.inc:159:1: error: static assertion failed due to requirement '201103L >= 201402L': Protobuf only supports C++14 and newer.
      [31m   [0m static_assert(PROTOBUF_CPLUSPLUS_MIN(201402L), "Protobuf only supports C++14 and newer.");
      [31m   [0m ^             ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-data.pb.cc:4:
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-data.pb.h:26:
      [31m   [0m In file included from /opt/homebrew/include/google/protobuf/io/coded_stream.h:In file included from 109/private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-ml.pb.cc:4:
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-ml.pb.h:26:
      [31m   [0m In file included from /opt/homebrew/include/google/protobuf/io/coded_stream.h:109:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/absl_check.h:38:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/check_impl.h:19:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/check_op.h:37:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/strip.h:24:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/log_message.h:41:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/log_entry.h:35:
      [31m   [0m /opt/homebrew/include/absl/time/time.h:284:11: error: cannot assign to non-static data member within const member function 'operator='
      [31m   [0m       hi_ = static_cast<uint32_t>(unsigned_value >> 32);
      [31m   [0m       ~~~ ^
      [31m   [0m :
      [31m   [0m In file included from /opt/homebrew/include/absl/log/absl_check.h:38:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/check_impl.h:19:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/check_op.h:37:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/strip.h:24:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/log_message.h:41:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/log_entry.h:35:
      [31m   [0m /opt/homebrew/include/absl/time/time.h:284:11: error: cannot assign to non-static data member within const member function 'operator='
      [31m   [0m       hi_ = static_cast<uint32_t>(unsigned_value >> 32);
      [31m   [0m       ~~~ ^
      [31m   [0m /opt/homebrew/include/absl/time/time.h:278:22: note: member function 'absl::Duration::HiRep::operator=' is declared const here
      [31m   [0m /opt/homebrew/include/absl/time/time.h:278:    constexpr HiRep& operator=(const int64_t value) {
      [31m   [0m     ~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      [31m   [0m 22: note: member function 'absl::Duration::HiRep::operator=' is declared const here
      [31m   [0m /opt/homebrew/include/absl/time/time.h:285:11:     constexpr HiRep& operator=(const int64_t value) {error: cannot assign to non-static data member within const member function 'operator='
      [31m   [0m       lo_ = static_cast<uint32_t>(unsigned_value);
      [31m   [0m       ~~~ ^
      [31m   [0m /opt/homebrew/include/absl/time/time.h:278
      [31m   [0m :22: note: member function 'absl::Duration::HiRep::operator=' is declared const here
      [31m   [0m     constexpr HiRep& operator=(const int64_t value) {    ~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      [31m   [0m 
      [31m   [0m     ~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      [31m   [0m /opt/homebrew/include/absl/time/time.h:285:11: error: cannot assign to non-static data member within const member function 'operator='
      [31m   [0m /opt/homebrew/include/absl/time/time.h:286:14:       lo_ = static_cast<uint32_t>(unsigned_value);error: binding reference of type 'HiRep' to value of type 'const absl::Duration::HiRep' drops 'const' qualifier
      [31m   [0m       return *this;
      [31m   [0m              ^~~~~
      [31m   [0m       ~~~ ^
      [31m   [0m 
      [31m   [0m /opt/homebrew/include/absl/time/time.h:278:22: note: member function 'absl::Duration::HiRep::operator=' is declared const here
      [31m   [0m     constexpr HiRep& operator=(const int64_t value) {
      [31m   [0m     ~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      [31m   [0m /opt/homebrew/include/absl/time/time.h:286:14: error: binding reference of type 'HiRep' to value of type 'const absl::Duration::HiRep' drops 'const' qualifier
      [31m   [0m       return *this;
      [31m   [0m              ^~~~~
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-operators-ml.pb.cc:4:
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-operators-ml.pb.h:26:
      [31m   [0m In file included from /opt/homebrew/include/google/protobuf/io/coded_stream.h:109:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/absl_check.h:38:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/check_impl.h:19:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/check_op.h:37:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/strip.h:24:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/log_message.h:41:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/log_entry.h:35:
      [31m   [0m /opt/homebrew/include/absl/time/time.h:284:11: error: cannot assign to non-static data member within const member function 'operator='
      [31m   [0m       hi_ = static_cast<uint32_t>(unsigned_value >> 32);
      [31m   [0m       ~~~ ^
      [31m   [0m /opt/homebrew/include/absl/time/time.h:278:22: note: member function 'absl::Duration::HiRep::operator=' is declared const here
      [31m   [0m     constexpr HiRep& operator=(const int64_t value) {
      [31m   [0m     ~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      [31m   [0m /opt/homebrew/include/absl/time/time.h:285:11: error: cannot assign to non-static data member within const member function 'operator='
      [31m   [0m       lo_ = static_cast<uint32_t>(unsigned_value);
      [31m   [0m       ~~~ ^
      [31m   [0m /opt/homebrew/include/absl/time/time.h:278:22: note: member function 'absl::Duration::HiRep::operator=' is declared const here
      [31m   [0m     constexpr HiRep& operator=(const int64_t value) {
      [31m   [0m     ~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      [31m   [0m /opt/homebrew/include/absl/time/time.h:286:14: error: binding reference of type 'HiRep' to value of type 'const absl::Duration::HiRep' drops 'const' qualifier
      [31m   [0m       return *this;
      [31m   [0m              ^~~~~
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-ml.pb.cc:4:
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-ml.pb.h:26:
      [31m   [0m In file included from /opt/homebrew/include/google/protobuf/io/coded_stream.h:109:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/absl_check.h:38:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/check_impl.h:19:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/check_op.h:37:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/strip.h:24:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/log_message.h:41:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/log_entry.h:36:
      [31m   [0m In file included from /opt/homebrew/include/absl/types/span.h:69:
      [31m   [0m /opt/homebrew/include/absl/types/internal/span.h:119:21: error: no template named 'remove_const_t' in namespace 'std'; did you mean simply 'remove_const_t'?
      [31m   [0m   using Container = std::remove_const_t<T>;
      [31m   [0m                     ^~~~~
      [31m   [0m /opt/homebrew/include/absl/meta/type_traits.h:241:1: note: 'remove_const_t' declared here
      [31m   [0m using remove_const_t = typename std::remove_const<T>::type;
      [31m   [0m ^
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-data.pb.cc:4:
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-data.pb.h:26:
      [31m   [0m In file included from /opt/homebrew/include/google/protobuf/io/coded_stream.h:109:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/absl_check.h:38:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/check_impl.h:19:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/check_op.h:37:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/strip.h:24:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/log_message.h:41:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/log_entry.h:36:
      [31m   [0m In file included from /opt/homebrew/include/absl/types/span.h:69:
      [31m   [0m /opt/homebrew/include/absl/types/internal/span.h:119:21: error: no template named 'remove_const_t' in namespace 'std'; did you mean simply 'remove_const_t'?
      [31m   [0m   using Container = std::remove_const_t<T>;
      [31m   [0m                     ^~~~~
      [31m   [0m /opt/homebrew/include/absl/meta/type_traits.h:241:1: note: 'remove_const_t' declared here
      [31m   [0m using remove_const_t = typename std::remove_const<T>::type;
      [31m   [0m ^
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-ml.pb.cc:4:
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-ml.pb.h:26:
      [31m   [0m In file included from /opt/homebrew/include/google/protobuf/io/coded_stream.h:109:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/absl_check.h:38:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/check_impl.h:19:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/check_op.h:37:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/strip.h:24:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/log_message.h:41:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/log_entry.h:36:
      [31m   [0m In file included from /opt/homebrew/include/absl/types/span.h:69:
      [31m   [0m /opt/homebrew/include/absl/types/internal/span.h:130:24: error: no template named 'enable_if_t' in namespace 'std'; did you mean simply 'enable_if_t'?
      [31m   [0m using EnableIfIsView = std::enable_if_t<IsView<T>::value, int>;
      [31m   [0m                        ^~~~~
      [31m   [0m /opt/homebrew/include/absl/meta/type_traits.h:307:1: note: 'enable_if_t' declared here
      [31m   [0m using enable_if_t = typename std::enable_if<B, T>::type;
      [31m   [0m ^
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-data.pb.cc:4:
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-data.pb.h:26:
      [31m   [0m In file included from /opt/homebrew/include/google/protobuf/io/coded_stream.h:109:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/absl_check.h:38:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/check_impl.h:19:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/check_op.h:37:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/strip.h:24:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/log_message.h:41:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/log_entry.h:36:
      [31m   [0m In file included from /opt/homebrew/include/absl/types/span.h:69:
      [31m   [0m /opt/homebrew/include/absl/types/internal/span.h:130:24: error: no template named 'enable_if_t' in namespace 'std'; did you mean simply 'enable_if_t'?
      [31m   [0m using EnableIfIsView = std::enable_if_t<IsView<T>::value, int>;
      [31m   [0m                        ^~~~~
      [31m   [0m /opt/homebrew/include/absl/meta/type_traits.h:307:1: note: 'enable_if_t' declared here
      [31m   [0m using enable_if_t = typename std::enable_if<B, T>::type;
      [31m   [0m ^
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-ml.pb.cc:4:
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-ml.pb.h:26:
      [31m   [0m In file included from /opt/homebrew/include/google/protobuf/io/coded_stream.h:109:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/absl_check.h:38:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/check_impl.h:19:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/check_op.h:37:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/strip.h:24:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/log_message.h:41:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/log_entry.h:36:
      [31m   [0m In file included from /opt/homebrew/include/absl/types/span.h:69:
      [31m   [0m /opt/homebrew/include/absl/types/internal/span.h:133:27: error: no template named 'enable_if_t' in namespace 'std'; did you mean simply 'enable_if_t'?
      [31m   [0m using EnableIfNotIsView = std::enable_if_t<!IsView<T>::value, int>;
      [31m   [0m                           ^~~~~
      [31m   [0m /opt/homebrew/include/absl/meta/type_traits.h:307:1: note: 'enable_if_t' declared here
      [31m   [0m using enable_if_t = typename std::enable_if<B, T>::type;
      [31m   [0m ^
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-data.pb.cc:4:
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-data.pb.h:26:
      [31m   [0m In file included from /opt/homebrew/include/google/protobuf/io/coded_stream.h:109:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/absl_check.h:38:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/check_impl.h:19:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/check_op.h:37:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/strip.h:24:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/log_message.h:41:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/log_entry.h:36:
      [31m   [0m In file included from /opt/homebrew/include/absl/types/span.h:69:
      [31m   [0m /opt/homebrew/include/absl/types/internal/span.h:133:27: error: no template named 'enable_if_t' in namespace 'std'; did you mean simply 'enable_if_t'?
      [31m   [0m using EnableIfNotIsView = std::enable_if_t<!IsView<T>::value, int>;
      [31m   [0m                           ^~~~~
      [31m   [0m /opt/homebrew/include/absl/meta/type_traits.h:307:1: note: 'enable_if_t' declared here
      [31m   [0m using enable_if_t = typename std::enable_if<B, T>::type;
      [31m   [0m ^
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-operators-ml.pb.cc:4:
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-operators-ml.pb.h:26:
      [31m   [0m In file included from /opt/homebrew/include/google/protobuf/io/coded_stream.h:109:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/absl_check.h:38:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/check_impl.h:19:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/check_op.h:37:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/strip.h:24:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/log_message.h:41:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/log_entry.h:36:
      [31m   [0m In file included from /opt/homebrew/include/absl/types/span.h:69:
      [31m   [0m /opt/homebrew/include/absl/types/internal/span.h:119:21: error: no template named 'remove_const_t' in namespace 'std'; did you mean simply 'remove_const_t'?
      [31m   [0m   using Container = std::remove_const_t<T>;
      [31m   [0m                     ^~~~~
      [31m   [0m /opt/homebrew/include/absl/meta/type_traits.h:241:1: note: 'remove_const_t' declared here
      [31m   [0m using remove_const_t = typename std::remove_const<T>::type;
      [31m   [0m ^
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-ml.pb.cc:4:
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-ml.pb.h:26:
      [31m   [0m In file included from /opt/homebrew/include/google/protobuf/io/coded_stream.h:109:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/absl_check.h:38:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/check_impl.h:19:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/check_op.h:37:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/strip.h:24:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/log_message.h:43:
      [31m   [0m /opt/homebrew/include/absl/strings/internal/has_absl_stringify.h:46:8: error: no template named 'enable_if_t' in namespace 'std'; did you mean simply 'enable_if_t'?
      [31m   [0m     T, std::enable_if_t<std::is_void<decltype(AbslStringify(
      [31m   [0m        ^~~~~
      [31m   [0m /opt/homebrew/include/absl/meta/type_traits.h:307:1: note: 'enable_if_t' declared here
      [31m   [0m using enable_if_t = typename std::enable_if<B, T>::type;
      [31m   [0m ^
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-data.pb.cc:4:
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-data.pb.h:26:
      [31m   [0m In file included from /opt/homebrew/include/google/protobuf/io/coded_stream.h:109:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/absl_check.h:38:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/check_impl.h:19:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/check_op.h:37:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/strip.h:24:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/log_message.h:43:
      [31m   [0m /opt/homebrew/include/absl/strings/internal/has_absl_stringify.h:46:8: error: no template named 'enable_if_t' in namespace 'std'; did you mean simply 'enable_if_t'?
      [31m   [0m     T, std::enable_if_t<std::is_void<decltype(AbslStringify(
      [31m   [0m        ^~~~~
      [31m   [0m /opt/homebrew/include/absl/meta/type_traits.h:307:1: note: 'enable_if_t' declared here
      [31m   [0m using enable_if_t = typename std::enable_if<B, T>::type;
      [31m   [0m ^
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-operators-ml.pb.cc:4:
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-operators-ml.pb.h:26:
      [31m   [0m In file included from /opt/homebrew/include/google/protobuf/io/coded_stream.h:109:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/absl_check.h:38:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/check_impl.h:19:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/check_op.h:37:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/strip.h:24:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/log_message.h:41:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/log_entry.h:36:
      [31m   [0m In file included from /opt/homebrew/include/absl/types/span.h:69:
      [31m   [0m /opt/homebrew/include/absl/types/internal/span.h:130:24: error: no template named 'enable_if_t' in namespace 'std'; did you mean simply 'enable_if_t'?
      [31m   [0m using EnableIfIsView = std::enable_if_t<IsView<T>::value, int>;
      [31m   [0m                        ^~~~~
      [31m   [0m /opt/homebrew/include/absl/meta/type_traits.h:307:1: note: 'enable_if_t' declared here
      [31m   [0m using enable_if_t = typename std::enable_if<B, T>::type;
      [31m   [0m ^
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-operators-ml.pb.cc:4:
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-operators-ml.pb.h:26:
      [31m   [0m In file included from /opt/homebrew/include/google/protobuf/io/coded_stream.h:109:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/absl_check.h:38:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/check_impl.h:19:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/check_op.h:37:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/strip.h:24:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/log_message.h:41:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/log_entry.h:36:
      [31m   [0m In file included from /opt/homebrew/include/absl/types/span.h:69:
      [31m   [0m /opt/homebrew/include/absl/types/internal/span.h:133:27: error: no template named 'enable_if_t' in namespace 'std'; did you mean simply 'enable_if_t'?
      [31m   [0m using EnableIfNotIsView = std::enable_if_t<!IsView<T>::value, int>;
      [31m   [0m                           ^~~~~
      [31m   [0m /opt/homebrew/include/absl/meta/type_traits.h:307:1: note: 'enable_if_t' declared here
      [31m   [0m using enable_if_t = typename std::enable_if<B, T>::type;
      [31m   [0m ^
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-operators-ml.pb.cc:4:
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-operators-ml.pb.h:26:
      [31m   [0m In file included from /opt/homebrew/include/google/protobuf/io/coded_stream.h:109:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/absl_check.h:38:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/check_impl.h:19:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/check_op.h:37:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/strip.h:24:
      [31m   [0m In file included from /opt/homebrew/include/absl/log/internal/log_message.h:43:
      [31m   [0m /opt/homebrew/include/absl/strings/internal/has_absl_stringify.h:46:8: error: no template named 'enable_if_t' in namespace 'std'; did you mean simply 'enable_if_t'?
      [31m   [0m     T, std::enable_if_t<std::is_void<decltype(AbslStringify(
      [31m   [0m        ^~~~~
      [31m   [0m /opt/homebrew/include/absl/meta/type_traits.h:307:1: note: 'enable_if_t' declared here
      [31m   [0m using enable_if_t = typename std::enable_if<B, T>::type;
      [31m   [0m ^
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-data.pb.cc:4:
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-data.pb.h:26:
      [31m   [0m In file included from /opt/homebrew/include/google/protobuf/io/coded_stream.h:111:
      [31m   [0m In file included from /opt/homebrew/include/absl/strings/cord.h:78:
      [31m   [0m In file included from /opt/homebrew/include/absl/container/inlined_vector.h:53:
      [31m   [0m In file included from /opt/homebrew/include/absl/container/internal/inlined_vector.h:30:
      [31m   [0m In file included from /opt/homebrew/include/absl/container/internal/compressed_tuple.h:40:
      [31m   [0m /opt/homebrew/include/absl/utility/utility.h:164:12: error: no member named 'in_place_t' in namespace 'std'
      [31m   [0m using std::in_place_t;
      [31m   [0m       ~~~~~^
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-ml.pb.cc:4:
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-ml.pb.h:26:
      [31m   [0m In file included from /opt/homebrew/include/google/protobuf/io/coded_stream.h:111:
      [31m   [0m In file included from /opt/homebrew/include/absl/strings/cord.h:78:
      [31m   [0m In file included from /opt/homebrew/include/absl/container/inlined_vector.h:53:
      [31m   [0m In file included from /opt/homebrew/include/absl/container/internal/inlined_vector.h:30:
      [31m   [0m In file included from /opt/homebrew/include/absl/container/internal/compressed_tuple.h:40:
      [31m   [0m /opt/homebrew/include/absl/utility/utility.h:164:12: error: no member named 'in_place_t' in namespace 'std'
      [31m   [0m using std::in_place_t;
      [31m   [0m       ~~~~~^
      [31m   [0m /opt/homebrew/include/absl/utility/utility.h:165:12: error: no member named 'in_place' in namespace 'std'
      [31m   [0m using std::in_place;
      [31m   [0m       ~~~~~^
      [31m   [0m /opt/homebrew/include/absl/utility/utility.h:165:12: error: no member named 'in_place' in namespace 'std'
      [31m   [0m using std::in_place;
      [31m   [0m       ~~~~~^
      [31m   [0m /opt/homebrew/include/absl/utility/utility.h:181:12: error: no member named 'in_place_type' in namespace 'std'
      [31m   [0m using std::in_place_type;
      [31m   [0m       ~~~~~^
      [31m   [0m /opt/homebrew/include/absl/utility/utility.h:181:12: error: no member named 'in_place_type' in namespace 'std'
      [31m   [0m using std::in_place_type;
      [31m   [0m       ~~~~~^
      [31m   [0m /opt/homebrew/include/absl/utility/utility.h:182:12: error: no member named 'in_place_type_t' in namespace 'std'
      [31m   [0m using std::in_place_type_t;
      [31m   [0m       ~~~~~^
      [31m   [0m /opt/homebrew/include/absl/utility/utility.h:182:12: error: no member named 'in_place_type_t' in namespace 'std'
      [31m   [0m using std::in_place_type_t;
      [31m   [0m       ~~~~~^
      [31m   [0m /opt/homebrew/include/absl/utility/utility.h:198:12: error: no member named 'in_place_index' in namespace 'std'
      [31m   [0m using std::in_place_index;
      [31m   [0m       ~~~~~^
      [31m   [0m /opt/homebrew/include/absl/utility/utility.h:198:12: error: no member named 'in_place_index' in namespace 'std'
      [31m   [0m using std::in_place_index;
      [31m   [0m       ~~~~~^
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-operators-ml.pb.cc:4:
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-operators-ml.pb.h:26:
      [31m   [0m In file included from /opt/homebrew/include/google/protobuf/io/coded_stream.h:111:
      [31m   [0m In file included from /opt/homebrew/include/absl/strings/cord.h:78:
      [31m   [0m In file included from /opt/homebrew/include/absl/container/inlined_vector.h:53:
      [31m   [0m In file included from /opt/homebrew/include/absl/container/internal/inlined_vector.h:30:
      [31m   [0m In file included from /opt/homebrew/include/absl/container/internal/compressed_tuple.h:40:
      [31m   [0m /opt/homebrew/include/absl/utility/utility.h:164:12: error: no member named 'in_place_t' in namespace 'std'
      [31m   [0m using std::in_place_t;
      [31m   [0m       ~~~~~^
      [31m   [0m /opt/homebrew/include/absl/utility/utility.h:165:12: error: no member named 'in_place' in namespace 'std'
      [31m   [0m using std::in_place;
      [31m   [0m       ~~~~~^
      [31m   [0m /opt/homebrew/include/absl/utility/utility.h:199:12: error: no member named 'in_place_index_t' in namespace 'std'
      [31m   [0m using std::in_place_index_t;
      [31m   [0m       ~~~~~^
      [31m   [0m /opt/homebrew/include/absl/utility/utility.h:199:12: error: no member named 'in_place_index_t' in namespace 'std'
      [31m   [0m using std::in_place_index_t;
      [31m   [0m       ~~~~~^
      [31m   [0m /opt/homebrew/include/absl/utility/utility.h:181:12: error: no member named 'in_place_type' in namespace 'std'
      [31m   [0m using std::in_place_type;
      [31m   [0m       ~~~~~^
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-data.pb.cc:4:
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-data.pb.h:26:
      [31m   [0m In file included from /opt/homebrew/include/google/protobuf/io/coded_stream.h:111:
      [31m   [0m In file included from /opt/homebrew/include/absl/strings/cord.h:78:
      [31m   [0m In file included from /opt/homebrew/include/absl/container/inlined_vector.h:53:
      [31m   [0m In file included from /opt/homebrew/include/absl/container/internal/inlined_vector.h:30:
      [31m   [0m /opt/homebrew/include/absl/container/internal/compressed_tuple.h:77:16: error: no member named 'is_final' in namespace 'std'
      [31m   [0m          !std::is_final<T>::value &&
      [31m   [0m           ~~~~~^
      [31m   [0m /opt/homebrew/include/absl/container/internal/compressed_tuple.h:77:25: error: 'T' does not refer to a value
      [31m   [0m          !std::is_final<T>::value &&
      [31m   [0m                         ^
      [31m   [0m /opt/homebrew/include/absl/container/internal/compressed_tuple.h:74:20: note: declared here
      [31m   [0m template <typename T>
      [31m   [0m                    ^
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-ml.pb.cc:4:
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-ml.pb.h:26:
      [31m   [0m In file included from /opt/homebrew/include/google/protobuf/io/coded_stream.h:111:
      [31m   [0m In file included from /opt/homebrew/include/absl/strings/cord.h:78:
      [31m   [0m In file included from /opt/homebrew/include/absl/container/inlined_vector.h:53:
      [31m   [0m In file included from /opt/homebrew/include/absl/container/internal/inlined_vector.h:30:
      [31m   [0m /opt/homebrew/include/absl/container/internal/compressed_tuple.h:77:16: error: no member named 'is_final' in namespace 'std'
      [31m   [0m          !std::is_final<T>::value &&
      [31m   [0m           ~~~~~^
      [31m   [0m /opt/homebrew/include/absl/container/internal/compressed_tuple.h:77:25: error: 'T' does not refer to a value
      [31m   [0m          !std::is_final<T>::value &&
      [31m   [0m                         ^
      [31m   [0m /opt/homebrew/include/absl/container/internal/compressed_tuple.h:74:20: note: declared here
      [31m   [0m template <typename T>
      [31m   [0m                    ^
      [31m   [0m /opt/homebrew/include/absl/utility/utility.h:182:12: error: no member named 'in_place_type_t' in namespace 'std'
      [31m   [0m using std::in_place_type_t;
      [31m   [0m       ~~~~~^
      [31m   [0m fatal error: too many errors emitted, stopping now [-ferror-limit=]
      [31m   [0m fatal error: too many errors emitted, stopping now [-ferror-limit=]
      [31m   [0m /opt/homebrew/include/absl/utility/utility.h:198:12: error: no member named 'in_place_index' in namespace 'std'
      [31m   [0m using std::in_place_index;
      [31m   [0m       ~~~~~^
      [31m   [0m /opt/homebrew/include/absl/utility/utility.h:199:12: error: no member named 'in_place_index_t' in namespace 'std'
      [31m   [0m using std::in_place_index_t;
      [31m   [0m       ~~~~~^
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-operators-ml.pb.cc:4:
      [31m   [0m In file included from /private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/.setuptools-cmake-build/onnx/onnx-operators-ml.pb.h:26:
      [31m   [0m In file included from /opt/homebrew/include/google/protobuf/io/coded_stream.h:111:
      [31m   [0m In file included from /opt/homebrew/include/absl/strings/cord.h:78:
      [31m   [0m In file included from /opt/homebrew/include/absl/container/inlined_vector.h:53:
      [31m   [0m In file included from /opt/homebrew/include/absl/container/internal/inlined_vector.h:30:
      [31m   [0m /opt/homebrew/include/absl/container/internal/compressed_tuple.h:77:16: error: no member named 'is_final' in namespace 'std'
      [31m   [0m          !std::is_final<T>::value &&
      [31m   [0m           ~~~~~^
      [31m   [0m /opt/homebrew/include/absl/container/internal/compressed_tuple.h:77:25: error: 'T' does not refer to a value
      [31m   [0m          !std::is_final<T>::value &&
      [31m   [0m                         ^
      [31m   [0m /opt/homebrew/include/absl/container/internal/compressed_tuple.h:74:20: note: declared here
      [31m   [0m template <typename T>
      [31m   [0m                    ^
      [31m   [0m fatal error: too many errors emitted, stopping now [-ferror-limit=]
      [31m   [0m 20 errors generated.
      [31m   [0m 20 errors generated.
      [31m   [0m gmake[2]: *** [CMakeFiles/onnx_proto.dir/build.make:123: CMakeFiles/onnx_proto.dir/onnx/onnx-operators-ml.pb.cc.o] Error 1
      [31m   [0m gmake[2]: *** Waiting for unfinished jobs....
      [31m   [0m gmake[2]: *** [CMakeFiles/onnx_proto.dir/build.make:137: CMakeFiles/onnx_proto.dir/onnx/onnx-data.pb.cc.o] Error 1
      [31m   [0m 20 errors generated.
      [31m   [0m gmake[2]: *** [CMakeFiles/onnx_proto.dir/build.make:109: CMakeFiles/onnx_proto.dir/onnx/onnx-ml.pb.cc.o] Error 1
      [31m   [0m gmake[1]: *** [CMakeFiles/Makefile2:193: CMakeFiles/onnx_proto.dir/all] Error 2
      [31m   [0m gmake: *** [Makefile:136: all] Error 2
      [31m   [0m Traceback (most recent call last):
      [31m   [0m   File "<string>", line 2, in <module>
      [31m   [0m   File "<pip-setuptools-caller>", line 34, in <module>
      [31m   [0m   File "/private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/setup.py", line 332, in <module>
      [31m   [0m     setuptools.setup(
      [31m   [0m   File "/opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages/setuptools/__init__.py", line 107, in setup
      [31m   [0m     return distutils.core.setup(**attrs)
      [31m   [0m   File "/opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages/setuptools/_distutils/core.py", line 185, in setup
      [31m   [0m     return run_commands(dist)
      [31m   [0m   File "/opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages/setuptools/_distutils/core.py", line 201, in run_commands
      [31m   [0m     dist.run_commands()
      [31m   [0m   File "/opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages/setuptools/_distutils/dist.py", line 969, in run_commands
      [31m   [0m     self.run_command(cmd)
      [31m   [0m   File "/opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages/setuptools/dist.py", line 1234, in run_command
      [31m   [0m     super().run_command(command)
      [31m   [0m   File "/opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages/setuptools/_distutils/dist.py", line 988, in run_command
      [31m   [0m     cmd_obj.run()
      [31m   [0m   File "/opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages/wheel/bdist_wheel.py", line 364, in run
      [31m   [0m     self.run_command("build")
      [31m   [0m   File "/opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages/setuptools/_distutils/cmd.py", line 318, in run_command
      [31m   [0m     self.distribution.run_command(command)
      [31m   [0m   File "/opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages/setuptools/dist.py", line 1234, in run_command
      [31m   [0m     super().run_command(command)
      [31m   [0m   File "/opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages/setuptools/_distutils/dist.py", line 988, in run_command
      [31m   [0m     cmd_obj.run()
      [31m   [0m   File "/opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages/setuptools/_distutils/command/build.py", line 131, in run
      [31m   [0m     self.run_command(cmd_name)
      [31m   [0m   File "/opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages/setuptools/_distutils/cmd.py", line 318, in run_command
      [31m   [0m     self.distribution.run_command(command)
      [31m   [0m   File "/opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages/setuptools/dist.py", line 1234, in run_command
      [31m   [0m     super().run_command(command)
      [31m   [0m   File "/opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages/setuptools/_distutils/dist.py", line 988, in run_command
      [31m   [0m     cmd_obj.run()
      [31m   [0m   File "/private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/setup.py", line 223, in run
      [31m   [0m     self.run_command("cmake_build")
      [31m   [0m   File "/opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages/setuptools/_distutils/cmd.py", line 318, in run_command
      [31m   [0m     self.distribution.run_command(command)
      [31m   [0m   File "/opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages/setuptools/dist.py", line 1234, in run_command
      [31m   [0m     super().run_command(command)
      [31m   [0m   File "/opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages/setuptools/_distutils/dist.py", line 988, in run_command
      [31m   [0m     cmd_obj.run()
      [31m   [0m   File "/private/var/folders/rs/yt_dh9xn6y39_h0_jth1mjb40000gq/T/pip-install-adm97gad/onnx_c4081e76da95458e90e4de0d9dacd74f/setup.py", line 217, in run
      [31m   [0m     subprocess.check_call(build_args)
      [31m   [0m   File "/opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/subprocess.py", line 364, in check_call
      [31m   [0m     raise CalledProcessError(retcode, cmd)
      [31m   [0m subprocess.CalledProcessError: Command '['/opt/homebrew/bin/cmake', '--build', '.', '--', '-j', '8']' returned non-zero exit status 2.
      [31m   [0m [31m[end of output][0m
      
      [1;35mnote[0m: This error originates from a subprocess, and is likely not a problem with pip.
    [31m  ERROR: Failed building wheel for onnx[0m[31m
    [0m[?25h  Running setup.py clean for onnx
    Failed to build onnx
    [31mERROR: Could not build wheels for onnx, which is required to install pyproject.toml-based projects[0m[31m
    [0mCollecting onnxruntime==1.15.0
      Downloading onnxruntime-1.15.0-cp38-cp38-macosx_11_0_arm64.whl.metadata (3.9 kB)
    Collecting coloredlogs (from onnxruntime==1.15.0)
      Using cached coloredlogs-15.0.1-py2.py3-none-any.whl (46 kB)
    Collecting flatbuffers (from onnxruntime==1.15.0)
      Using cached flatbuffers-23.5.26-py2.py3-none-any.whl.metadata (850 bytes)
    Requirement already satisfied: numpy>=1.21.6 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from onnxruntime==1.15.0) (1.22.3)
    Requirement already satisfied: packaging in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from onnxruntime==1.15.0) (23.1)
    Requirement already satisfied: protobuf in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from onnxruntime==1.15.0) (4.25.1)
    Requirement already satisfied: sympy in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from onnxruntime==1.15.0) (1.12)
    Collecting humanfriendly>=9.1 (from coloredlogs->onnxruntime==1.15.0)
      Using cached humanfriendly-10.0-py2.py3-none-any.whl (86 kB)
    Requirement already satisfied: mpmath>=0.19 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from sympy->onnxruntime==1.15.0) (1.3.0)
    Downloading onnxruntime-1.15.0-cp38-cp38-macosx_11_0_arm64.whl (6.1 MB)
    [2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m6.1/6.1 MB[0m [31m13.8 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hUsing cached flatbuffers-23.5.26-py2.py3-none-any.whl (26 kB)
    Installing collected packages: flatbuffers, humanfriendly, coloredlogs, onnxruntime
    Successfully installed coloredlogs-15.0.1 flatbuffers-23.5.26 humanfriendly-10.0 onnxruntime-1.15.0
    Collecting imutils==0.5.4
      Using cached imutils-0.5.4-py3-none-any.whl
    Installing collected packages: imutils
    Successfully installed imutils-0.5.4
    Requirement already satisfied: pytz in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (2023.3.post1)
    Collecting ipywidgets==8.0.6
      Downloading ipywidgets-8.0.6-py3-none-any.whl (138 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m138.3/138.3 kB[0m [31m2.8 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hRequirement already satisfied: ipykernel>=4.5.1 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from ipywidgets==8.0.6) (6.25.0)
    Requirement already satisfied: ipython>=6.1.0 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from ipywidgets==8.0.6) (7.24.1)
    Requirement already satisfied: traitlets>=4.3.1 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from ipywidgets==8.0.6) (5.13.0)
    Collecting widgetsnbextension~=4.0.7 (from ipywidgets==8.0.6)
      Using cached widgetsnbextension-4.0.9-py3-none-any.whl.metadata (1.6 kB)
    Collecting jupyterlab-widgets~=3.0.7 (from ipywidgets==8.0.6)
      Using cached jupyterlab_widgets-3.0.9-py3-none-any.whl.metadata (4.1 kB)
    Requirement already satisfied: appnope in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from ipykernel>=4.5.1->ipywidgets==8.0.6) (0.1.2)
    Requirement already satisfied: comm>=0.1.1 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from ipykernel>=4.5.1->ipywidgets==8.0.6) (0.1.2)
    Requirement already satisfied: debugpy>=1.6.5 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from ipykernel>=4.5.1->ipywidgets==8.0.6) (1.6.7)
    Requirement already satisfied: jupyter-client>=6.1.12 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from ipykernel>=4.5.1->ipywidgets==8.0.6) (8.6.0)
    Requirement already satisfied: jupyter-core!=5.0.*,>=4.12 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from ipykernel>=4.5.1->ipywidgets==8.0.6) (5.5.0)
    Requirement already satisfied: matplotlib-inline>=0.1 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from ipykernel>=4.5.1->ipywidgets==8.0.6) (0.1.6)
    Requirement already satisfied: nest-asyncio in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from ipykernel>=4.5.1->ipywidgets==8.0.6) (1.5.6)
    Requirement already satisfied: packaging in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from ipykernel>=4.5.1->ipywidgets==8.0.6) (23.1)
    Requirement already satisfied: psutil in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from ipykernel>=4.5.1->ipywidgets==8.0.6) (5.9.0)
    Requirement already satisfied: pyzmq>=20 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from ipykernel>=4.5.1->ipywidgets==8.0.6) (25.1.0)
    Requirement already satisfied: tornado>=6.1 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from ipykernel>=4.5.1->ipywidgets==8.0.6) (6.3.3)
    Requirement already satisfied: setuptools>=18.5 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets==8.0.6) (68.0.0)
    Requirement already satisfied: jedi>=0.16 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets==8.0.6) (0.18.1)
    Requirement already satisfied: decorator in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets==8.0.6) (5.1.1)
    Requirement already satisfied: pickleshare in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets==8.0.6) (0.7.5)
    Requirement already satisfied: prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets==8.0.6) (3.0.36)
    Requirement already satisfied: pygments in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets==8.0.6) (2.15.1)
    Requirement already satisfied: backcall in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets==8.0.6) (0.2.0)
    Requirement already satisfied: pexpect>4.3 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets==8.0.6) (4.8.0)
    Requirement already satisfied: parso<0.9.0,>=0.8.0 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from jedi>=0.16->ipython>=6.1.0->ipywidgets==8.0.6) (0.8.3)
    Requirement already satisfied: importlib-metadata>=4.8.3 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from jupyter-client>=6.1.12->ipykernel>=4.5.1->ipywidgets==8.0.6) (6.0.0)
    Requirement already satisfied: python-dateutil>=2.8.2 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from jupyter-client>=6.1.12->ipykernel>=4.5.1->ipywidgets==8.0.6) (2.8.2)
    Requirement already satisfied: platformdirs>=2.5 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from jupyter-core!=5.0.*,>=4.12->ipykernel>=4.5.1->ipywidgets==8.0.6) (3.10.0)
    Requirement already satisfied: ptyprocess>=0.5 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from pexpect>4.3->ipython>=6.1.0->ipywidgets==8.0.6) (0.7.0)
    Requirement already satisfied: wcwidth in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from prompt-toolkit!=3.0.0,!=3.0.1,<3.1.0,>=2.0.0->ipython>=6.1.0->ipywidgets==8.0.6) (0.2.10)
    Requirement already satisfied: zipp>=0.5 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from importlib-metadata>=4.8.3->jupyter-client>=6.1.12->ipykernel>=4.5.1->ipywidgets==8.0.6) (3.11.0)
    Requirement already satisfied: six>=1.5 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from python-dateutil>=2.8.2->jupyter-client>=6.1.12->ipykernel>=4.5.1->ipywidgets==8.0.6) (1.16.0)
    Using cached jupyterlab_widgets-3.0.9-py3-none-any.whl (214 kB)
    Using cached widgetsnbextension-4.0.9-py3-none-any.whl (2.3 MB)
    Installing collected packages: widgetsnbextension, jupyterlab-widgets, ipywidgets
    Successfully installed ipywidgets-8.0.6 jupyterlab-widgets-3.0.9 widgetsnbextension-4.0.9
    Collecting patchify==0.2.3
      Downloading patchify-0.2.3-py3-none-any.whl (6.6 kB)
    Requirement already satisfied: numpy<2,>=1 in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from patchify==0.2.3) (1.22.3)
    Installing collected packages: patchify
    Successfully installed patchify-0.2.3
    Collecting tifffile==2023.4.12
      Downloading tifffile-2023.4.12-py3-none-any.whl (219 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m219.4/219.4 kB[0m [31m3.8 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hRequirement already satisfied: numpy in /opt/homebrew/anaconda3/envs/wallaroosdk.2023.4.0-test/lib/python3.8/site-packages (from tifffile==2023.4.12) (1.22.3)
    Installing collected packages: tifffile
    Successfully installed tifffile-2023.4.12
    Collecting piexif==1.1.3
      Downloading piexif-1.1.3-py2.py3-none-any.whl (20 kB)
    Installing collected packages: piexif
    Successfully installed piexif-1.1.3

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

