#!/usr/bin/env python

"""
Converts the following Python notebooks into the same format used for the Wallaroo Documentation site.

This uses the jupyter nbconvert command.  For now this will always assume we're exporting to markdown:

    jupyter nbconvert {file} --to markdown --output {output}

"""

import os
import nbformat
from traitlets.config import Config
import re
import shutil
import glob
#import argparse

c = Config()

c.NbConvertApp.export_format = "markdown"

docs_directory = "docs/markdown"

fileList = [
    # ## wallaroo 101
    # {
    #     "inputFile": "wallaroo-101/Wallaroo-101.ipynb",
    #     "outputDir": "/reference/wallaroo-101",
    #     "outputFile": "wallaroo-101-reference.md"
    # },
    # ## Development
    # ### MLOps API
    # #### MLOps API Workspace Management
    # {
    #     "inputFile": "development/mlops_api/Wallaroo-MLOps-Tutorial-Workspace-Management.ipynb",
    #     "outputDir": "/reference/wallaroo-developer-guides/wallaroo-api-guides",
    #     "outputFile": "Wallaroo-MLOps-Tutorial-Workspace-Management-reference.md"
    # },
    # #### MLOps API User Management
    # {
    #     "inputFile": "development/mlops_api/Wallaroo-MLOps-Tutorial-User-Management.ipynb",
    #     "outputDir": "/reference/wallaroo-developer-guides/wallaroo-api-guides",
    #     "outputFile": "Wallaroo-MLOps-Tutorial-User-Management-reference.md"
    # },
    # #### MLOps API Pipeline Management
    # {
    #     "inputFile": "development/mlops_api/Wallaroo-MLOps-Tutorial-Pipeline-Management.ipynb",
    #     "outputDir": "/reference/wallaroo-developer-guides/wallaroo-api-guides",
    #     "outputFile": "Wallaroo-MLOps-Tutorial-Pipeline-Management-reference.md"
    # },
    # #### MLOps API Model Management
    # {
    #     "inputFile": "development/mlops_api/Wallaroo-MLOps-Tutorial-Model-Management.ipynb",
    #     "outputDir": "/reference/wallaroo-developer-guides/wallaroo-api-guides",
    #     "outputFile": "Wallaroo-MLOps-Tutorial-Model-Management-reference.md"
    # },
    # #### MLOps API Assays
    # {
    #     "inputFile": "development/mlops_api/Wallaroo-MLOps-Tutorial-Assay-Management-Plus.ipynb",
    #     "outputDir": "/reference/wallaroo-developer-guides/wallaroo-api-guides",
    #     "outputFile": "Wallaroo-MLOps-Tutorial-Assay-Management-Plus-reference.md"
    # },
    # ### SDK Install Guildes
    # #### SDK Standard Install
    # {
    #     "inputFile": "development/sdk-install-guides/standard-install/install-wallaroo-sdk-standard-guide.ipynb",
    #     "outputDir": "/reference/wallaroo-developer-guides/wallaroo-sdk-guides",
    #     "outputFile": "install-wallaroo-sdk-standard-guide-reference.md"
    # },
    # #### SDK AWS Sagemaker Install
    # {
    #     "inputFile": "development/sdk-install-guides/aws-sagemaker-install/install-wallaroo-aws-sagemaker-guide.ipynb",
    #     "outputDir": "/reference/wallaroo-developer-guides/wallaroo-sdk-guides",
    #     "outputFile": "install-wallaroo-aws-sagemaker-guide-reference.md"
    # },
    # #### SDK Azure ML Workspace Install
    # {
    #     "inputFile": "development/sdk-install-guides/azure-ml-sdk-install/install-wallaroo-sdk-azureml-guide.ipynb",
    #     "outputDir": "/reference/wallaroo-developer-guides/wallaroo-sdk-guides",
    #     "outputFile": "install-wallaroo-sdk-azureml-guide-reference.md"
    # },
    # #### SDK Azure Databricks Install
    # {
    #     "inputFile": "development/sdk-install-guides/databricks-azure-sdk-install/install-wallaroo-sdk-databricks-azure-guide.ipynb",
    #     "outputDir": "/reference/wallaroo-developer-guides/wallaroo-sdk-guides",
    #     "outputFile": "install-wallaroo-sdk-azureml-guide-reference.md"
    # },
    # #### SDK Google Vertex Install
    # {
    #     "inputFile": "development/sdk-install-guides/google-vertex-sdk-install/install-wallaroo-sdk-google-vertex-guide.ipynb",
    #     "outputDir": "/reference/wallaroo-developer-guides/wallaroo-sdk-guides",
    #     "outputFile": "install-wallaroo-sdk-google-vertex-guide-reference.md"
    # },
    # ## wallaroo free
    # ### computer vision frcnn
    # {
    #     "inputFile": "wallaroo-free-tutorials/wallaroo-inference-server-tutorials/wallaroo-inference-server-cv-frcnn/wallaroo-inference-server-cv-frcnn.ipynb",
    #     "outputDir": "/reference/wallaroo-free-tutorials/wallaroo-inference-server-tutorials",
    #     "outputFile": "wallaroo-inference-server-cv-frcnn-reference.md"
    # },
    # ### computer vision resnet
    # {
    #     "inputFile": "wallaroo-free-tutorials/wallaroo-inference-server-tutorials/wallaroo-inference-server-cv-resnet/wallaroo-inference-server-cv-resnet.ipynb",
    #     "outputDir": "/reference/wallaroo-free-tutorials/wallaroo-inference-server-tutorials",
    #     "outputFile": "wallaroo-inference-server-cv-resnet-reference.md"
    # },
    # ### computer vision unet
    # {
    #     "inputFile": "wallaroo-free-tutorials/wallaroo-inference-server-tutorials/wallaroo-inference-server-cv-unet/wallaroo-inference-server-cv-unet.ipynb",
    #     "outputDir": "/reference/wallaroo-free-tutorials/wallaroo-inference-server-tutorials",
    #     "outputFile": "wallaroo-inference-server-cv-unet-reference.md"
    # },
    # ### computer vision yolov8
    # {
    #     "inputFile": "wallaroo-free-tutorials/wallaroo-inference-server-tutorials/wallaroo-inference-server-cv-yolov8/wallaroo-inference-server-cv-yolov8.ipynb",
    #     "outputDir": "/reference/wallaroo-free-tutorials/wallaroo-inference-server-tutorials",
    #     "outputFile": "wallaroo-inference-server-cv-yolov8-reference.md"
    # },
    # ### hf summarizer
    # {
    #     "inputFile": "wallaroo-free-tutorials/wallaroo-inference-server-tutorials/wallaroo-inference-server-hf-summarizer/wallaroo-inference-server-hf-summarization.ipynb",
    #     "outputDir": "/reference/wallaroo-free-tutorials/wallaroo-inference-server-tutorials",
    #     "outputFile": "wallaroo-inference-server-hf-summarization-reference.md"
    # },
    # ### llama v2
    # {
    #     "inputFile": "wallaroo-free-tutorials/wallaroo-inference-server-tutorials/wallaroo-inference-server-llama2/wallaroo-inference-server-llama2.ipynb",
    #     "outputDir": "/reference//wallaroo-free-tutorials/wallaroo-inference-server-tutorials",
    #     "outputFile": "wallaroo-inference-server-llama2-reference.md"
    # },
    # ## LLMs
    # ### Autoscale Triggers with Llamacpp
    # {
    #     "inputFile": "wallaroo-llms/autoscale_triggers_llamacpp/autoscale_triggers_llamacpp_sdk.ipynb",
    #     "outputDir": "/reference/wallaroo-tutorials/wallaroo-llms",
    #     "outputFile": "autoscale_triggers_llamacpp_sdk-reference.md"
    # },
    # ### Dynamic Batching with Llama 3 8B quantized with llama-cpp and dynamic batching on CPUs Tutorial
    # {
    #     "inputFile": "wallaroo-llms/dynamic_batching_tutorial_llamacpp/llamacpp-sdk-dynamic-batching-tutorial.ipynb",
    #     "outputDir": "/reference//wallaroo-tutorials/wallaroo-llms",
    #     "outputFile": "llamacpp-sdk-dynamic-batching-tutorial-reference.md"
    # },
    # ## Dynamic Batching with Llama 3 8B Instruct LLM Tutorial
    # {
    #     "inputFile": "wallaroo-llms/dynamic_batching_tutorial_vllm/llama3-8b-vllm-dynamic-batching-benchmarks.ipynb",
    #     "outputDir": "/reference//wallaroo-tutorials/wallaroo-llms",
    #     "outputFile": "llama3-8b-vllm-dynamic-batching-benchmarks-reference.md"
    # },
    # ## IBM Granite 8B Code Instruct
    # {
    #     "inputFile": "wallaroo-llms/ibm-granite-llms/deployment_ibm_granite_8b_code_instruct.ipynb",
    #     "outputDir": "/reference/wallaroo-tutorials/wallaroo-llms",
    #     "outputFile": "deployment_ibm_granite_8b_code_instruct-reference.md"
    # },
    # ## Model Operations
    # ### Model Deploy
    # #### Model Deploy by Framework
    # ##### BYOP VGG16
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/BYOP/arbitrary-python-upload-tutorials/00_wallaroo-upload-arbitrary-python-vgg16-model-generation.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/BYOP",
    #     "outputFile": "00_wallaroo-upload-arbitrary-python-vgg16-model-generation-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/BYOP/arbitrary-python-upload-tutorials/01_wallaroo-upload-arbitrary-python-vgg16-model-deployment.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/BYOP",
    #     "outputFile": "01_wallaroo-upload-arbitrary-python-vgg16-model-deployment-reference.md"
    # },
    # ##### Hugging Face Clip Vit
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/hugging-face/hf-clip-vit-base/clip-vit-hugging-face.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/hugging-face",
    #     "outputFile": "clip-vit-hugging-face-reference.md"
    # },
    # ##### Hugging Face Upload
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/hugging-face/hugging-face-upload-tutorials/wallaroo-api-upload-hf-zero_shot_classification.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/hugging-face",
    #     "outputFile": "wallaroo-api-upload-hf-zero_shot_classification-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/hugging-face/hugging-face-upload-tutorials/wallaroo-sdk-upload-hf-zero_shot_classification.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/hugging-face",
    #     "outputFile": "wallaroo-sdk-upload-hf-zero_shot_classification-reference.md"
    # },
    # ##### Keras
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/keras/wallaroo-upload-keras_sequential_model_single_io.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/keras",
    #     "outputFile": "wallaroo-upload-keras_sequential_model_single_io-reference.md"
    # },
    # ##### MLFlow
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/mlflow/wallaroo-mlflow-tutorial.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/mlflow",
    #     "outputFile": "wallaroo-mlflow-tutorial-reference.md"
    # },
    # ##### Model Registry Service
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/model-registry-service/Wallaroo-model-registry-demonstration.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/model-registry-service",
    #     "outputFile": "Wallaroo-model-registry-demonstration-reference.md"
    # },
    # ##### ONNX
    # ##### Demand Curve
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/onnx/demand_curve/demandcurve_demo.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/onnx",
    #     "outputFile": "demand_curve/demandcurve_demo-reference.md"
    # },
    # ##### IMDB
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/onnx/imdb/imdb_sample.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/onnx",
    #     "outputFile": "imdb_sample-reference.md"
    # },
    # ##### Multi Input Demo
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/onnx/onnx-multi-input-demo/onnx-multi-io.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/onnx",
    #     "outputFile": "onnx-multi-io-reference.md"
    # },
    # ##### Python Models
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/python-models/python-step-dataframe-output-logging-example-sdk.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/python-models",
    #     "outputFile": "python-step-dataframe-output-logging-example-sdk-reference.md"
    # },
    # ##### Pytorch
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/pytorch/wallaroo-upload-pytorch-multi-input-output.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/pytorch",
    #     "outputFile": "wallaroo-upload-pytorch-multi-input-output-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/pytorch/wallaroo-upload-pytorch-single-input-output.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/pytorch",
    #     "outputFile": "wallaroo-upload-pytorch-single-input-output-reference.md"
    # },
    # ##### Sklearn
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/sklearn/wallaroo-upload-sklearn-clustering-kmeans.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/sklearn",
    #     "outputFile": "wallaroo-upload-sklearn-clustering-kmeans-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/sklearn/wallaroo-upload-sklearn-clustering-svm-pca.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/sklearn",
    #     "outputFile": "wallaroo-upload-sklearn-clustering-svm-pca-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/sklearn/wallaroo-upload-sklearn-clustering-svm.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/sklearn",
    #     "outputFile": "wallaroo-upload-sklearn-clustering-svm-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/sklearn/wallaroo-upload-sklearn-linear-regression.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/sklearn",
    #     "outputFile": "wallaroo-upload-sklearn-linear-regression-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/sklearn/wallaroo-upload-sklearn-logistic-regression.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/sklearn",
    #     "outputFile": "wallaroo-upload-sklearn-logistic-regression-reference.md"
    # },
    # ##### Tensorflow
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/tensorflow/wallaroo-upload-tensorflow.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/tensorflow",
    #     "outputFile": "wallaroo-upload-tensorflow-reference.md"
    # },
    # ##### XGboost
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/xgboost/wallaroo-sdk-upload-booster-binary-classification-conversion.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/xgboost",
    #     "outputFile": "wallaroo-sdk-upload-booster-binary-classification-conversion-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/xgboost/wallaroo-sdk-upload-booster-multi-classification-softmax-conversion.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/xgboost",
    #     "outputFile": "wallaroo-sdk-upload-booster-multi-classification-softmax-conversion-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/xgboost/wallaroo-sdk-upload-booster-multi-classification-softprob-conversion.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/xgboost",
    #     "outputFile": "wallaroo-sdk-upload-booster-multi-classification-softprob-conversion-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/xgboost/wallaroo-sdk-upload-booster-regression-conversion.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/xgboost",
    #     "outputFile": "wallaroo-sdk-upload-booster-regression-conversion-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/xgboost/wallaroo-sdk-upload-booster-rf-classification-conversion.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/xgboost",
    #     "outputFile": "wallaroo-sdk-upload-booster-rf-classification-conversion-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/xgboost/wallaroo-sdk-upload-booster-rf-regression-conversion.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/xgboost",
    #     "outputFile": "wallaroo-sdk-upload-booster-rf-regression-conversion-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/xgboost/wallaroo-sdk-upload-xbg-classification.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/xgboost",
    #     "outputFile": "wallaroo-sdk-upload-xbg-classification-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/xgboost/wallaroo-sdk-upload-xbg-regressor.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/xgboost",
    #     "outputFile": "wallaroo-sdk-upload-xbg-regressor-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/xgboost/wallaroo-sdk-upload-xbg-rf-classification.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/xgboost",
    #     "outputFile": "wallaroo-sdk-upload-xbg-rf-classification-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/xgboost/wallaroo-sdk-upload-xbg-rf-regressor.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-framework/xgboost",
    #     "outputFile": "wallaroo-sdk-upload-xbg-rf-regressor-reference.md"
    # },
    # #### Model Deploy by Use Case
    # ##### Automatic Speech Detection
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-use-case/automatic-speech-detection/wallaroo-whisper_demo.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-use-case",
    #     "outputFile": "wallaroo-whisper_demo-reference.md"
    # },
    # ##### Computer Vision
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-use-case/computer-vision/00_computer_vision_tutorial_intro.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-use-case/computer-vision",
    #     "outputFile": "00_computer_vision_tutorial_intro-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-use-case/computer-vision/01_computer_vision_tutorial_mobilenet.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-use-case/computer-vision",
    #     "outputFile": "01_computer_vision_tutorial_mobilenet-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-use-case/computer-vision/02_computer_vision_tutorial_resnet50.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-use-case/computer-vision",
    #     "outputFile": "02_computer_vision_tutorial_resnet50-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-use-case/computer-vision/03_computer_vision_tutorial_shadow_deploy.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-use-case/computer-vision",
    #     "outputFile": "03_computer_vision_tutorial_shadow_deploy-reference.md"
    # },
    # ##### Computer Vision Healthcare Imaging
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-use-case/computer-vision-mitochondria-imaging/00_computer-vision-mitochondria-imaging-example.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-use-case/computer-vision-mitochondria-imaging",
    #     "outputFile": "00_computer-vision-mitochondria-imaging-example-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-use-case/computer-vision-mitochondria-imaging/01_computer-vision-mitochondria-imaging-example.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-use-case/computer-vision-mitochondria-imaging",
    #     "outputFile": "01_computer-vision-mitochondria-imaging-example-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-use-case/computer-vision-mitochondria-imaging/02_computer-vision-mitochondria-imaging-example.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-use-case/computer-vision-mitochondria-imaging",
    #     "outputFile": "02_computer-vision-mitochondria-imaging-example-reference.md"
    # },
    # ##### Computer Vision Yolo8
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-use-case/computer-vision-yolov8/computer-vision-yolov8-demonstration.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-use-case",
    #     "outputFile": "computer-vision-yolov8-demonstration-reference.md"
    # },
    # ##### Notebooks in Production
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-use-case/notebooks_in_prod/00_notebooks_in_prod_introduction.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-use-case/notebooks_in_prod",
    #     "outputFile": "00_notebooks_in_prod_introduction-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-use-case/notebooks_in_prod/01_notebooks_in_prod_explore_and_train.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-use-case/notebooks_in_prod",
    #     "outputFile": "01_notebooks_in_prod_explore_and_train-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-use-case/notebooks_in_prod/02_notebooks_in_prod_automated_training_process.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-use-case/notebooks_in_prod",
    #     "outputFile": "02_notebooks_in_prod_automated_training_process-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-use-case/notebooks_in_prod/03_notebooks_in_prod_deploy_model_python.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-use-case/notebooks_in_prod",
    #     "outputFile": "03_notebooks_in_prod_deploy_model_python-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-use-case/notebooks_in_prod/04_notebooks_in_prod_regular_batch_inferences.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-use-case/notebooks_in_prod",
    #     "outputFile": "04_notebooks_in_prod_regular_batch_inferences-reference.md"
    # },
    # ##### Multiple Replicas Tutorial
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-use-case/pipeline_multiple_replicas_forecast_tutorial/00_multiple_replicas_forecast.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-use-case/pipeline_multiple_replicas_forecast_tutorial",
    #     "outputFile": "00_multiple_replicas_forecast-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-use-case/pipeline_multiple_replicas_forecast_tutorial/01_multiple_replicas_forecast.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-use-case/pipeline_multiple_replicas_forecast_tutorial",
    #     "outputFile": "01_multiple_replicas_forecast-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-use-case/pipeline_multiple_replicas_forecast_tutorial/02_multiple_replicas_forecast.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-deploy/packaging-and-deployment-by-use-case/pipeline_multiple_replicas_forecast_tutorial",
    #     "outputFile": "02_multiple_replicas_forecast-reference.md"
    # },
    # ### Model Inference
    # #### Async Infer
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-inference/aloha_async_infer/aloha_async_infer_tutorial.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-inference",
    #     "outputFile": "aloha_async_infer_tutorial-reference.md"
    # },
    # #### Parallel Infer
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-inference/parallel-inferences-sdk-aloha-tutorial/wallaroo-parallel-infer-sdk-with-aloha.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-inference",
    #     "outputFile": "wallaroo-parallel-infer-sdk-with-aloha-reference.md"
    # },
    # #### Inference Results aka Pipeline Logs
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-inference/pipeline-log-tutorial/pipeline_log_tutorial.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-inference",
    #     "outputFile": "pipeline_log_tutorial-reference.md"
    # },
    # #### Inference Endpoints
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-inference/wallaroo-model-endpoints/wallaroo-model-endpoints-sdk.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-inference/wallaroo-model-endpoints",
    #     "outputFile": "wallaroo-model-endpoints-sdk-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-inference/wallaroo-model-endpoints/wallaroo-model-endpoints-api.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-inference/wallaroo-model-endpoints",
    #     "outputFile": "wallaroo-model-endpoints-api-reference.md"
    # },
    # ### Model Management
    # #### AB Testing
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-management/abtesting/wallaroo-abtesting-tutorial.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-management",
    #     "outputFile": "wallaroo-abtesting-tutorial-reference.md"
    # },
    # #### Model Hot Swap
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-management/model_hot_swap/wallaroo_hot_swap_tutorial.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-management",
    #     "outputFile": "wallaroo_hot_swap_tutorial-reference.md"
    # },
    # #### Shadow Deploy
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-management/shadow_deploy/shadow_deployment_tutorial.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-management",
    #     "outputFile": "shadow_deployment_tutorial-reference.md"
    # },
    # #### Tag Management
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-management/wallaroo-tag-management/wallaroo-tags-guide.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-management",
    #     "outputFile": "wallaroo-tags-guide-reference.md"
    # },
    # ### Model Observability
    # ##### Houseprice Saga
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/model-management/wallaroo-tag-management/wallaroo-tags-guide.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/model-management",
    #     "outputFile": "wallaroo-tags-guide-reference.md"
    # },


]

def format(outputdir, document_file):
    # Take the markdown file, remove the extra spaces
    document = open(f'{docs_directory}{outputdir}/{document_file}', "r").read()
    result = re.sub
    
    # fix tables for publication
    # document = re.sub(r'<table.*?>', r'{{<table "table table-striped table-bordered" >}}\n<table>', document)
    # document = re.sub('</table>', r'</table>\n{{</table>}}', document)
    # remove any div table sections
    document = re.sub('<div.*?>', '', document)
    document = re.sub(r'<style.*?>.*?</style>', '', document, flags=re.S)
    document = re.sub('</div>', '', document)

    # remove non-public domains
    document = re.sub('wallaroocommunity.ninja', 'wallarooexample.ai', document)

    # fix image directories
    # ](01_notebooks_in_prod_explore_and_train-reference_files
    # image_replace = f'![png]({outputdir}'
    document = re.sub('!\[png\]\(', f'![png](/images/2024.4{outputdir}/', document)
    document = re.sub('\(./images', '(/images/2024.4', document)
    # move them all to Docsy figures
    document = re.sub(r'!\[(.*?)\]\((.*?)\)', r'{{<figure src="\2" width="800" label="\1">}}', document)

    # move the assay image for UI
    document = re.sub('"images/housepricesaga-sample-assay.png"', '"/images/housepricesaga-sample-assay.png"', document)

    # remove gib
    document = re.sub('gib.bhojraj@wallaroo.ai', 
                      'sample.user@wallaroo.ai', 
                      document)
    # fix github link for final release
    # document = re.sub('https://github.com/WallarooLabs/Wallaroo_Tutorials/blob/wallaroo2024.1_tutorials/', 
    #                   'https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/', 
    #                   document)
    
     # obfuscate databricks url
    document = re.sub('https://adb-5939996465837398.18.azuredatabricks.net', 
                      'https://sample.registry.service.azuredatabricks.net', 
                      document)
    
    # obfuscate gcp registry URL url
    document = re.sub('us-central1-docker.pkg.dev/wallaroo-dev-253816', 
                      'sample.registry.example.com', 
                      document)

    # remove edge bundle
    document = re.sub("'EDGE_BUNDLE': '.*?'", 
                      "'EDGE_BUNDLE': 'abcde'", 
                      document)
    
    # fix pyarrow outputs
    # document = re.sub(r"pyarrow.Table\ntime: timestamp[ms]")

   # document = re.sub('![png](', 'bob', document)

    # strip the excess newlines - match any pattern of newline plus another one or more empty newlines
    document = re.sub(r'\n[\n]+', r'\n\n', document)

    # remove the whitespace before a table
    document = re.sub(r"^ +<", r"<", document, flags = re.MULTILINE)
    #document.strip() # - test this for whitespace before and after

    # save the file for publishing
    newdocument = open(f'{docs_directory}{outputdir}/{document_file}', "w")
    newdocument.write(document)
    newdocument.close()

def move_images(image_directory):
    source_directory = f"{docs_directory}{image_directory}"
    target_directory = f"./images{image_directory}"
    # check the current directory for reference files
    # reference_directories = os.listdir(image_directory)
    print(source_directory)
    reference_directories = [ name for name in os.listdir(source_directory) if os.path.isdir(os.path.join(source_directory, name)) ]
    # copy only the directories to their image location
    for reference in reference_directories:
        print(f"mkdir -p {target_directory} && cp -rf ./{source_directory}/{reference} {target_directory}")
        # print(f"To: {target_directory}/{reference}")
        os.system(f"mkdir -p {target_directory} &&  cp -rf ./{source_directory}/{reference} {target_directory}")

def main():
    for currentFile in fileList:
        convert_cmd = f'''
            jupyter nbconvert \
                 --to markdown \
                 --output-dir {docs_directory}{currentFile["outputDir"]} \
                 --output {currentFile["outputFile"]} {currentFile["inputFile"]} \
                 --TemplateExporter.extra_template_basedirs=scripts/nbconvert/templates
        '''
        print(convert_cmd)
        os.system(convert_cmd)
        # format(f'{docs_directory}{currentFile["outputDir"]}/{currentFile["outputFile"]}')
        format(currentFile["outputDir"],currentFile["outputFile"])
        move_images(currentFile["outputDir"])
    # get rid of any extra markdown files
    os.system("find ./images -name '*.md' -type f -delete")
    os.system("find ./docs/markdown -name '*.png' -type f -delete")

if __name__ == '__main__':
    main()