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
    #     "inputFile": "development/mlops-api/Wallaroo-MLOps-Tutorial-Workspace-Management.ipynb",
    #     "outputDir": "/reference/wallaroo-developer-guides/wallaroo-api-guides",
    #     "outputFile": "Wallaroo-MLOps-Tutorial-Workspace-Management-reference.md"
    # },
    # #### MLOps API User Management
    # {
    #     "inputFile": "development/mlops-api/Wallaroo-MLOps-Tutorial-User-Management.ipynb",
    #     "outputDir": "/reference/wallaroo-developer-guides/wallaroo-api-guides",
    #     "outputFile": "Wallaroo-MLOps-Tutorial-User-Management-reference.md"
    # },
    # #### MLOps API Pipeline Management
    # {
    #     "inputFile": "development/mlops-api/Wallaroo-MLOps-Tutorial-Pipeline-Management.ipynb",
    #     "outputDir": "/reference/wallaroo-developer-guides/wallaroo-api-guides",
    #     "outputFile": "Wallaroo-MLOps-Tutorial-Pipeline-Management-reference.md"
    # },
    # #### MLOps API Model Management
    # {
    #     "inputFile": "development/mlops-api/Wallaroo-MLOps-Tutorial-Model-Management.ipynb",
    #     "outputDir": "/reference/wallaroo-developer-guides/wallaroo-api-guides",
    #     "outputFile": "Wallaroo-MLOps-Tutorial-Model-Management-reference.md"
    # },
    # #### MLOps API Assays
    # {
    #     "inputFile": "development/mlops-api/Wallaroo-MLOps-Tutorial-Assay-Management-Plus.ipynb",
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
    #     "outputFile": "install-wallaroo-sdk-databricks-azure-guide-reference.md"
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
    # ## Model Operations
    # ### Model Deploy
    # #### Model Deploy by Framework
    # ##### BYOP
    # ###### BYOP VGG16
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-framework/BYOP/arbitrary-python-upload-tutorials/00-wallaroo-upload-arbitrary-python-vgg16-model-generation.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-framework/BYOP",
    #     "outputFile": "00-wallaroo-upload-arbitrary-python-vgg16-model-generation-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-framework/BYOP/arbitrary-python-upload-tutorials/01-wallaroo-upload-arbitrary-python-vgg16-model-deployment.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-framework/BYOP",
    #     "outputFile": "01-wallaroo-upload-arbitrary-python-vgg16-model-deployment-reference.md"
    # },
    # ###### BYOP CV
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-framework/BYOP/wallaroo-model-upload-deploy-byop-cv-tutorial/wallaroo-model-upload-deploy-byop-cv-tutorial.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-framework/BYOP",
    #     "outputFile": "wallaroo-model-upload-deploy-byop-cv-tutorial-reference.md"
    # },
    # ##### Hugging Face Clip Vit
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-framework/hugging-face/hf-clip-vit-base/clip-vit-hugging-face.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-framework/hugging-face",
    #     "outputFile": "clip-vit-hugging-face-reference.md"
    # },
    # ##### Hugging Face Upload
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-framework/hugging-face/hugging-face-upload-tutorials/wallaroo-api-upload-hf-zero-shot-classification.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-framework/hugging-face",
    #     "outputFile": "wallaroo-api-upload-hf-zero-shot-classification-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-framework/hugging-face/hugging-face-upload-tutorials/wallaroo-sdk-upload-hf-zero-shot-classification.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-framework/hugging-face",
    #     "outputFile": "wallaroo-sdk-upload-hf-zero-shot-classification-reference.md"
    # },
    # ##### Keras
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-framework/keras/wallaroo-upload-keras-sequential-model-single-io.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-framework/keras",
    #     "outputFile": "wallaroo-upload-keras-sequential-model-single-io-reference.md"
    # },
    # ##### MLFlow
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-framework/mlflow/wallaroo-mlflow-tutorial.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-framework/mlflow",
    #     "outputFile": "wallaroo-mlflow-tutorial-reference.md"
    # },
    # ##### Model Registry Service
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-framework/model-registry-service/Wallaroo-model-registry-demonstration.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-framework/model-registry-service",
    #     "outputFile": "Wallaroo-model-registry-demonstration-reference.md"
    # },
    # ##### ONNX
    # ##### Demand Curve
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-framework/onnx/demand-curve/demandcurve-demo.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-framework/onnx",
    #     "outputFile": "demandcurve-demo-reference.md"
    # },
    # ##### IMDB
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-framework/onnx/imdb/imdb-sample.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-framework/onnx",
    #     "outputFile": "imdb-sample-reference.md"
    # },
    # ##### Multi Input Demo
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-framework/onnx/onnx-multi-input-demo/onnx-multi-io.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-framework/onnx",
    #     "outputFile": "onnx-multi-io-reference.md"
    # },
    # ##### Python Models
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-framework/python-models/python-step-dataframe-output-logging-example-sdk.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-framework/python-models",
    #     "outputFile": "python-step-dataframe-output-logging-example-sdk-reference.md"
    # },
    # ##### Pytorch
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-framework/pytorch/wallaroo-upload-pytorch-multi-input-output.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-framework/pytorch",
    #     "outputFile": "wallaroo-upload-pytorch-multi-input-output-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-framework/pytorch/wallaroo-upload-pytorch-single-input-output.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-framework/pytorch",
    #     "outputFile": "wallaroo-upload-pytorch-single-input-output-reference.md"
    # },
    # ##### Sklearn
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-framework/sklearn/wallaroo-upload-sklearn-clustering-kmeans.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-framework/sklearn",
    #     "outputFile": "wallaroo-upload-sklearn-clustering-kmeans-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-framework/sklearn/wallaroo-upload-sklearn-clustering-svm-pca.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-framework/sklearn",
    #     "outputFile": "wallaroo-upload-sklearn-clustering-svm-pca-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-framework/sklearn/wallaroo-upload-sklearn-clustering-svm.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-framework/sklearn",
    #     "outputFile": "wallaroo-upload-sklearn-clustering-svm-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-framework/sklearn/wallaroo-upload-sklearn-linear-regression.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-framework/sklearn",
    #     "outputFile": "wallaroo-upload-sklearn-linear-regression-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-framework/sklearn/wallaroo-upload-sklearn-logistic-regression.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-framework/sklearn",
    #     "outputFile": "wallaroo-upload-sklearn-logistic-regression-reference.md"
    # },
    # ##### Tensorflow
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-framework/tensorflow/wallaroo-upload-tensorflow.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-framework/tensorflow",
    #     "outputFile": "wallaroo-upload-tensorflow-reference.md"
    # },
    # ##### XGboost
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-framework/xgboost/wallaroo-sdk-upload-booster-binary-classification-conversion.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-framework/xgboost",
    #     "outputFile": "wallaroo-sdk-upload-booster-binary-classification-conversion-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-framework/xgboost/wallaroo-sdk-upload-booster-multi-classification-softmax-conversion.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-framework/xgboost",
    #     "outputFile": "wallaroo-sdk-upload-booster-multi-classification-softmax-conversion-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-framework/xgboost/wallaroo-sdk-upload-booster-multi-classification-softprob-conversion.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-framework/xgboost",
    #     "outputFile": "wallaroo-sdk-upload-booster-multi-classification-softprob-conversion-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-framework/xgboost/wallaroo-sdk-upload-booster-regression-conversion.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-framework/xgboost",
    #     "outputFile": "wallaroo-sdk-upload-booster-regression-conversion-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-framework/xgboost/wallaroo-sdk-upload-booster-rf-classification-conversion.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-framework/xgboost",
    #     "outputFile": "wallaroo-sdk-upload-booster-rf-classification-conversion-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-framework/xgboost/wallaroo-sdk-upload-booster-rf-regression-conversion.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-framework/xgboost",
    #     "outputFile": "wallaroo-sdk-upload-booster-rf-regression-conversion-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-framework/xgboost/wallaroo-sdk-upload-xbg-classification.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-framework/xgboost",
    #     "outputFile": "wallaroo-sdk-upload-xbg-classification-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-framework/xgboost/wallaroo-sdk-upload-xbg-regressor.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-framework/xgboost",
    #     "outputFile": "wallaroo-sdk-upload-xbg-regressor-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-framework/xgboost/wallaroo-sdk-upload-xbg-rf-classification.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-framework/xgboost",
    #     "outputFile": "wallaroo-sdk-upload-xbg-rf-classification-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-framework/xgboost/wallaroo-sdk-upload-xbg-rf-regressor.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-framework/xgboost",
    #     "outputFile": "wallaroo-sdk-upload-xbg-rf-regressor-reference.md"
    # },
    # # #### Model Deploy by Use Case
    # # ##### Automatic Speech Detection
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-use-case/automatic-speech-detection/wallaroo-whisper-demo.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-use-case",
    #     "outputFile": "wallaroo-whisper-demo-reference.md"
    # },
    # # ##### Sentiment Analysis
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-use-case/sentiment-analysis-hugging-face/sentiment-analysis-hugging-face-tutorial.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-use-case",
    #     "outputFile": "sentiment-analysis-hugging-face-tutorial-reference.md"
    # },
    # # ##### Computer Vision
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-use-case/computer-vision/00-computer-vision-tutorial-intro.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-use-case/computer-vision",
    #     "outputFile": "00-computer-vision-tutorial-intro-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-use-case/computer-vision/01-computer-vision-tutorial-mobilenet.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-use-case/computer-vision",
    #     "outputFile": "01-computer-vision-tutorial-mobilenet-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-use-case/computer-vision/02-computer-vision-tutorial-resnet50.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-use-case/computer-vision",
    #     "outputFile": "02-computer-vision-tutorial-resnet50-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-use-case/computer-vision/03-computer-vision-tutorial-shadow-deploy.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-use-case/computer-vision",
    #     "outputFile": "03-computer-vision-tutorial-shadow-deploy-reference.md"
    # },
    # # ##### Computer Vision Healthcare Imaging
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-use-case/computer-vision-mitochondria-imaging/00-computer-vision-mitochondria-imaging-example.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-use-case/computer-vision-mitochondria-imaging",
    #     "outputFile": "00-computer-vision-mitochondria-imaging-example-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-use-case/computer-vision-mitochondria-imaging/01-computer-vision-mitochondria-imaging-example.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-use-case/computer-vision-mitochondria-imaging",
    #     "outputFile": "01-computer-vision-mitochondria-imaging-example-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-use-case/computer-vision-mitochondria-imaging/02-computer-vision-mitochondria-imaging-example.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-use-case/computer-vision-mitochondria-imaging",
    #     "outputFile": "02-computer-vision-mitochondria-imaging-example-reference.md"
    # },
    # # ##### Computer Vision Yolo8
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-use-case/computer-vision-yolov8/computer-vision-yolov8-demonstration.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-use-case",
    #     "outputFile": "computer-vision-yolov8-demonstration-reference.md"
    # },
    # # ##### Notebooks in Production
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-use-case/notebooks-in-prod/00-notebooks-in-prod-introduction.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-use-case/notebooks-in-prod",
    #     "outputFile": "00-notebooks-in-prod-introduction-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-use-case/notebooks-in-prod/01-notebooks-in-prod-explore-and-train.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-use-case/notebooks-in-prod",
    #     "outputFile": "01-notebooks-in-prod-explore-and-train-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-use-case/notebooks-in-prod/02-notebooks-in-prod-automated-training-process.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-use-case/notebooks-in-prod",
    #     "outputFile": "02-notebooks-in-prod-automated-training-process-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-use-case/notebooks-in-prod/03-notebooks-in-prod-deploy-model-python.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-use-case/notebooks-in-prod",
    #     "outputFile": "03-notebooks-in-prod-deploy-model-python-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-use-case/notebooks-in-prod/04-notebooks-in-prod-regular-batch-inferences.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-use-case/notebooks-in-prod",
    #     "outputFile": "04-notebooks-in-prod-regular-batch-inferences-reference.md"
    # },
    # # ##### Multiple Replicas Tutorial
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-use-case/pipeline-multiple-replicas-forecast-tutorial/00-multiple-replicas-forecast.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-use-case/pipeline-multiple-replicas-forecast-tutorial",
    #     "outputFile": "00-multiple-replicas-forecast-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-use-case/pipeline-multiple-replicas-forecast-tutorial/01-multiple-replicas-forecast.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-use-case/pipeline-multiple-replicas-forecast-tutorial",
    #     "outputFile": "01-multiple-replicas-forecast-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/deploy/by-use-case/pipeline-multiple-replicas-forecast-tutorial/02-multiple-replicas-forecast.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/deploy/by-use-case/pipeline-multiple-replicas-forecast-tutorial",
    #     "outputFile": "02-multiple-replicas-forecast-reference.md"
    # },
    # # ### Model Inference
    # # #### Async Infer
    {
        "inputFile": "wallaroo-model-operations-tutorials/infer/async-infer/async-infer-tutorial.ipynb",
        "outputDir": "/reference/wallaroo-model-operations-tutorials/infer",
        "outputFile": "async-infer-tutorial-reference.md"
    },
    # #### Parallel Infer
    {
        "inputFile": "wallaroo-model-operations-tutorials/infer/parallel-infer-tutorial/wallaroo-parallel-infer-tutorial.ipynb",
        "outputDir": "/reference/wallaroo-model-operations-tutorials/infer",
        "outputFile": "wallaroo-parallel-infer-tutorial-reference.md"
    },
    # #### Inference Results aka Pipeline Logs
    {
        "inputFile": "wallaroo-model-operations-tutorials/infer/inference-log-tutorial/inference-log-tutorial.ipynb",
        "outputDir": "/reference/wallaroo-model-operations-tutorials/infer",
        "outputFile": "inference-log-tutorial-reference.md"
    },
    # #### Inference Endpoints
    {
        "inputFile": "wallaroo-model-operations-tutorials/infer/infer/infer-sdk.ipynb",
        "outputDir": "/reference/wallaroo-model-operations-tutorials/infer/infer",
        "outputFile": "infer-sdk-reference.md"
    },
    {
        "inputFile": "wallaroo-model-operations-tutorials/infer/infer/infer-api.ipynb",
        "outputDir": "/reference/wallaroo-model-operations-tutorials/infer/infer",
        "outputFile": "infer-api-reference.md"
    },
    # ### Model Management
    # #### AB Testing
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/management/abtesting/wallaroo-abtesting-tutorial.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/management",
    #     "outputFile": "wallaroo-abtesting-tutorial-reference.md"
    # },
    # #### Model Hot Swap
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/management/inline-model-update/wallaroo-inline-model-update-tutorial.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/management",
    #     "outputFile": "wallaroo-inline-model-update-tutorial-reference.md"
    # },
    # #### Shadow Deploy
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/management/shadow-deploy/shadow-deployment-tutorial.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/management",
    #     "outputFile": "shadow-deployment-tutorial-reference.md"
    # },
    # #### Tag Management
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/management/wallaroo-tag-management/wallaroo-tags-guide.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/management",
    #     "outputFile": "wallaroo-tags-guide-reference.md"
    # },
    # ### Model Observability
    # ##### Anomaly Detection
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/observability/anomaly-detection-tutorial/anomaly-detection-tutorial.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/observability",
    #     "outputFile": "anomaly-detection-tutorial-reference.md"
    # },
    # ##### Model Drift aka Assays
    {
        "inputFile": "wallaroo-model-operations-tutorials/observability/model-drift-detection-with-assays/model-drift-detection-with-assays.ipynb",
        "outputDir": "/reference/wallaroo-model-operations-tutorials/observability",
        "outputFile": "model-drift-detection-with-assays-reference.md"
    },
    # ### Model Automation
    # #### Automation and Connections Tutorial
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/automation/automation-and-connections-tutorial/data-connectors-and-orchestrators-simple-tutorial.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/automation",
    #     "outputFile": "data-connectors-and-orchestrators-simple-tutorial-reference.md"
    # },
    # #### MLOps API Connections and Automations with Google BigQuery
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/automation/connection-api-bigquery-tutorial/connection-api-bigquery-tutorial.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/automation",
    #     "outputFile": "connection-api-bigquery-tutorial-reference.md"
    # },
    # ##### MLOps API Connections
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/automation/orchestration-api-simple-tutorial/data-orchestrators-api-simple-tutorial.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/automation",
    #     "outputFile": "data-orchestrators-api-simple-tutorial-reference.md"
    # },
    # ##### Orchestrations and Connection with BigQuery
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/automation/orchestration-sdk-bigquery-houseprice-tutorial/orchestration-sdk-bigquery-houseprice-tutorial.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/automation",
    #     "outputFile": "orchestration-sdk-bigquery-houseprice-tutorial-reference.md"
    # },
    # ##### Orchestrations and Connections Comprehensive Tutorial
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/automation/orchestration-sdk-comprehensive-tutorial/data-connectors-and-orchestrators-comprehensive-tutorial.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/automation",
    #     "outputFile": "data-connectors-and-orchestrators-comprehensive-tutorial-reference.md"
    # },
    # ##### Multiple Pipeline Deployment with Orchestrations
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/automation/orchestration-sdk-multiple-pipelines-tutorials/orchestration-sdk-multiple-pipelines-tutorial.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/automation",
    #     "outputFile": "orchestration-sdk-multiple-pipelines-tutorial-reference.md"
    # },
    # ##### Orchestrations and Connections Run Continuously
    # {
    #     "inputFile": "wallaroo-model-operations-tutorials/automation/orchestration-sdk-run-continuously-tutorial/orchestration-sdk-run-continuously-tutorial.ipynb",
    #     "outputDir": "/reference/wallaroo-model-operations-tutorials/automation",
    #     "outputFile": "orchestration-sdk-run-continuously-tutorial-reference.md"
    # },
    # ## LLMs
    # ### LLM Deploy
    # #### IBM Granite 8B Code Instruct Large Language Model (LLM) with GPU
    # {
    #     "inputFile": "wallaroo-llms/llm-deploy/ibm-granite-llms/deployment-ibm-granite-8b-code-instruct.ipynb",
    #     "outputDir": "/reference/wallaroo-llms/llm-deploy",
    #     "outputFile": "deployment-ibm-granite-8b-code-instruct-reference.md"
    # },
    # #### Llamacpp Deploy on IBM Power10 Tutorial
    # {
    #     "inputFile": "wallaroo-llms/llm-deploy/power10-deploy-llamacpp/llamacpp-sdk-power.ipynb",
    #     "outputDir": "/reference/wallaroo-llms/llm-deploy",
    #     "outputFile": "llamacpp-sdk-power-reference.md"
    # },
    # #### Inference Endpoint Tutorials
    # {
    #     "inputFile": "wallaroo-llms/llm-deploy/llm-managed-inference-endpoint/llm-managed-inference-endpoint-llama-vertex/managed-inference-endpoint-vertex.ipynb",
    #     "outputDir": "/reference/wallaroo-llms/llm-deploy/llm-managed-inference-endpoint",
    #     "outputFile": "managed-inference-endpoint-vertex-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-llms/llm-deploy/llm-managed-inference-endpoint/llm-managed-inference-endpoint-openai/managed-inference-endpoint-openai.ipynb",
    #     "outputDir": "/reference/wallaroo-llms/llm-deploy/llm-managed-inference-endpoint",
    #     "outputFile": "managed-inference-endpoint-openai-reference.md"
    # },
    # ### LLM Monitoring
    # #### LLM Harmful Language Listener Tutorial
    # {
    #     "inputFile": "wallaroo-llms/llm-monitoring/llamacpp-with-safeguards/llamacpp-sdk-with-safeguards.ipynb",
    #     "outputDir": "/reference/wallaroo-llms/llm-monitoring",
    #     "outputFile": "llamacpp-sdk-with-safeguards-reference.md"
    # },
    # #### LLM Validation Listener Example
    # {
    #     "inputFile": "wallaroo-llms/llm-monitoring/llm-in-line-monitoring/summary-quality-revised.ipynb",
    #     "outputDir": "/reference/wallaroo-llms/llm-monitoring",
    #     "outputFile": "summary-quality-revised-reference.md"
    # },
    # #### LLM Listener Monitoring with Llama V3 Instruct
    # {
    #     "inputFile": "wallaroo-llms/llm-monitoring/llm-listener-monitoring/llm-monitoring-orchestration-setup.ipynb",
    #     "outputDir": "/reference/wallaroo-llms/llm-monitoring",
    #     "outputFile": "llm-monitoring-orchestration-setup-reference.md"
    # },
    # ### RAG LLMs
    # #### RAG LLMs: Automated Vector Database Enrichment in Wallaroo
    # {
    #     "inputFile": "wallaroo-llms/rag-llms/vector-database-embedding-with-ml-orchestrations/Batch-Embedding-Computation.ipynb",
    #     "outputDir": "/reference/wallaroo-llms/rag-llms",
    #     "outputFile": "Batch-Embedding-Computation-reference.md"
    # },
    # #### RAG LLMs: Inference in Wallaroo
    # {
    #     "inputFile": "wallaroo-llms/rag-llms/vector-database-embedding-with-ml-orchestrations/RAG-LLM-Inferencing.ipynb",
    #     "outputDir": "/reference/wallaroo-llms/rag-llms",
    #     "outputFile": "RAG-LLM-Inferencing-reference.md"
    # },
    # ### Performance Optimizations
    # #### RAG LLMs: Inference in Wallaroo
    # {
    #     "inputFile": "wallaroo-llms/llm-performance-optimizations/autoscale-triggers-llamacpp/autoscale-triggers-llamacpp-sdk.ipynb",
    #     "outputDir": "/reference/wallaroo-llms/llm-performance-optimizations",
    #     "outputFile": "autoscale-triggers-llamacpp-sdk-reference.md"
    # },
    # #### Dynamic Batching with Llama 3 8B with Llama.cpp CPUs Tutorial
    # {
    #     "inputFile": "wallaroo-llms/llm-performance-optimizations/dynamic-batching-tutorial-llamacpp/llamacpp-sdk-dynamic-batching-tutorial.ipynb",
    #     "outputDir": "/reference/wallaroo-llms/llm-performance-optimizations",
    #     "outputFile": "llamacpp-sdk-dynamic-batching-tutorial-reference.md"
    # },
    # #### Dynamic Batching with Llama 3 8B Instruct vLLM Tutorial
    # {
    #     "inputFile": "wallaroo-llms/llm-performance-optimizations/dynamic-batching-tutorial-vllm/llama3-8b-vllm-dynamic-batching-benchmarks.ipynb",
    #     "outputDir": "/reference/wallaroo-llms/llm-performance-optimizations",
    #     "outputFile": "llama3-8b-vllm-dynamic-batching-benchmarks-reference.md"
    # },
    # #### Llama 3 8B Instruct Inference with vLLM
    # {
    #     "inputFile": "wallaroo-llms/llm-performance-optimizations/llama3-8b-instruct-vllm/deployment-llama3-8b-instruct-vllm.ipynb",
    #     "outputDir": "/reference/wallaroo-llms/llm-performance-optimizations",
    #     "outputFile": "deployment-llama3-8b-instruct-vllm-reference.md"
    # },
    # ## Run Anywhere
    # ### Infer
    # #### Publish
    # ##### U-Net for Brain Segmentation Publish in Wallaroo
    # {
    #     "inputFile": "wallaroo-run-anywhere/inference/publish/edge-unet-brain-segmentation-publish/unet-run-anywhere-publish.ipynb",
    #     "outputDir": "/reference/wallaroo-run-anywhere/inference/publish",
    #     "outputFile": "unet-run-anywhere-publish-reference.md"
    # },
    # # ##### Computer Vision Yolov8n Edge Publish in Wallaroo
    # {
    #     "inputFile": "wallaroo-run-anywhere/inference/publish/edge-computer-vision-yolov8-publish/edge-computer-vision-yolov8-publish.ipynb",
    #     "outputDir": "/reference/wallaroo-run-anywhere/inference/publish",
    #     "outputFile": "edge-computer-vision-yolov8-publish-reference.md"
    # },
    # # #### Deploy
    # # ##### U-Net for Brain Segmentation Deploy and Inference in Wallaroo @JOHN rename to infer
    # {
    #     "inputFile": "wallaroo-run-anywhere/inference/deploy/edge-unet-brain-segmentation-deploy/unet-run-anywhere-deploy.ipynb",
    #     "outputDir": "/reference/wallaroo-run-anywhere/inference/deploy",
    #     "outputFile": "unet-run-anywhere-deploy-reference.md"
    # },
    # ##### Computer Vision Yolov8n Edge Deployment in Wallaroo
    # {
    #     "inputFile": "wallaroo-run-anywhere/inference/deploy/edge-computer-vision-yolov8-deploy/edge-computer-vision-yolov8-deploy.ipynb",
    #     "outputDir": "/reference/wallaroo-run-anywhere/inference/deploy",
    #     "outputFile": "edge-computer-vision-yolov8-deploy-reference.md"
    # },
    # ### Inference on ARM
    # #### Custom Model ARM Deployment Tutorial
    # {
    #     "inputFile": "wallaroo-run-anywhere/inference/arm/wallaroo-arm-byop-vgg16/wallaroo-arm-arbitrary-python-vgg16-model-deployment.ipynb",
    #     "outputDir": "/reference/wallaroo-run-anywhere/inference/arm",
    #     "outputFile": "wallaroo-arm-arbitrary-python-vgg16-model-deployment-reference.md"
    # },
    # #### Classification Cybersecurity with Arm Architecture
    # {
    #     "inputFile": "wallaroo-run-anywhere/inference/arm/wallaroo-arm-classification-cybersecurity/arm-classification-cybersecurity.ipynb",
    #     "outputDir": "/reference/wallaroo-run-anywhere/inference/arm",
    #     "outputFile": "arm-classification-cybersecurity-reference.md"
    # },
    # #### Classification Financial Services with Arm Architecture
    # {
    #     "inputFile": "wallaroo-run-anywhere/inference/arm/wallaroo-arm-classification-finserv/arm-classification-finserv.ipynb",
    #     "outputDir": "/reference/wallaroo-run-anywhere/inference/arm",
    #     "outputFile": "arm-classification-finserv-reference.md"
    # },
    # #### Computer Vision Yolov8n ARM Deployment in Wallaroo
    # {
    #     "inputFile": "wallaroo-run-anywhere/inference/arm/wallaroo-arm-computer-vision-yolov8/wallaroo-arm-cv-yolov8-demonstration.ipynb",
    #     "outputDir": "/reference/wallaroo-run-anywhere/inference/arm",
    #     "outputFile": "wallaroo-arm-cv-yolov8-demonstration-reference.md"
    # },
    # #### Computer Vision Yolov8n ARM Deployment in Wallaroo
    # {
    #     "inputFile": "wallaroo-run-anywhere/inference/arm/wallaroo-arm-cv-arrow/arm-computer-vision-demonstration.ipynb",
    #     "outputDir": "/reference/wallaroo-run-anywhere/inference/arm",
    #     "outputFile": "arm-computer-vision-demonstration-reference.md"
    # },
    # #### LLM Summarization ARM Edge Deployment
    # {
    #     "inputFile": "wallaroo-run-anywhere/inference/arm/wallaroo-arm-llm-summarization/wallaroo-arm-llm-summarization-demonstration.ipynb",
    #     "outputDir": "/reference/wallaroo-run-anywhere/inference/arm",
    #     "outputFile": "wallaroo-arm-llm-summarization-demonstration-reference.md"
    # },
    # ### GPU
    # #### Large Language Model with GPU Pipeline Deployment in Wallaroo Tutorial
    # {
    #     "inputFile": "wallaroo-run-anywhere/inference-on-any-architecture/cuda/gpu-deployment/wallaroo-llm-with-gpu-demonstration.ipynb",
    #     "outputDir": "/reference/wallaroo-run-anywhere/inference/gpu",
    #     "outputFile": "wallaroo-llm-with-gpu-demonstration-reference.md"
    # },
    # #### LLM Summarization GPU Edge Deployment on Cuda
    # {
    #     "inputFile": "wallaroo-run-anywhere/inference-on-any-architecture/cuda/wallaroo-gpu-llm-summarization/wallaroo-gpu-llm-summarization-demonstration-accelerator.ipynb",
    #     "outputDir": "/reference/wallaroo-run-anywhere/inference/gpu",
    #     "outputFile": "wallaroo-gpu-llm-summarization-demonstration-accelerator-reference.md"
    # },
    # #### LLM Summarization GPU Edge Deployment Tutorial
    # {
    #     "inputFile": "wallaroo-run-anywhere/inference-on-any-architecture/cuda/wallaroo-gpu-llm-summarization/wallaroo-gpu-llm-summarization-demonstration.ipynb",
    #     "outputDir": "/reference/wallaroo-run-anywhere/inference/gpu",
    #     "outputFile": "wallaroo-gpu-llm-summarization-demonstration-reference.md"
    # },
    # ### Model Management
    # #### In-Line Model Updates at the Edge Tutorial
    # {
    #     "inputFile": "wallaroo-run-anywhere/model-management/inline-edge-model-replacements-tutorial/inline-edge-model-replacements-tutorial.ipynb",
    #     "outputDir": "/reference/wallaroo-run-anywhere/management",
    #     "outputFile": "inline-edge-model-replacements-tutorial-reference.md"
    # },
    # ### Observability
    # #### Wallaroo Edge Observability with Assays Tutorial
    # {
    #     "inputFile": "wallaroo-run-anywhere/observability/edge-observability-assays/edge-observability-assays.ipynb",
    #     "outputDir": "/reference/wallaroo-run-anywhere/observability",
    #     "outputFile": "edge-observability-assays-reference.md"
    # },
    # #### Model Drift Detection for Edge Deployments Tutorial
    # {
    #     "inputFile": "wallaroo-run-anywhere/observability/edge-observability-assays-complete/01_drift-detection-for-edge-deployments-tutorial-examples.ipynb",
    #     "outputDir": "/reference/wallaroo-run-anywhere/observability",
    #     "outputFile": "01_drift-detection-for-edge-deployments-tutorial-examples-reference.md"
    # },
    # #### Classification Financial Services Edge Deployment Demonstration
    # {
    #     "inputFile": "wallaroo-run-anywhere/observability/edge-observability-classification-finserv/edge-observabilty-classification-finserv-deployment.ipynb",
    #     "outputDir": "/reference/wallaroo-run-anywhere/observability",
    #     "outputFile": "edge-observabilty-classification-finserv-deployment-reference.md"
    # },
    # #### Edge Deployment and Observability via the Wallaroo MLOps API
    # {
    #     "inputFile": "wallaroo-run-anywhere/observability/edge-observability-classification-finserv-api/edge-observability-classification-finserv-deployment-via-api.ipynb",
    #     "outputDir": "/reference/wallaroo-run-anywhere/observability",
    #     "outputFile": "edge-observability-classification-finserv-deployment-via-api-reference.md"
    # },
    # #### Airgapped Edge Observability with No/Low Connection Tutorial
    # {
    #     "inputFile": "wallaroo-run-anywhere/observability/edge-observability-low-no-connection/edge-observability-low-no-connection-tutorial.ipynb",
    #     "outputDir": "/reference/wallaroo-run-anywhere/observability",
    #     "outputFile": "edge-observability-low-no-connection-tutorial-reference.md"
    # },
    # ### Inference on Any Hardware
    # #### ARM
    # ##### Run Anywhere for ARM Architecture Tutorial: Custom Inference Computer Vision with Resnet50
    # {
    #     "inputFile": "wallaroo-run-anywhere/inference-on-any-architecture/arm/edge-arm-publish-cv-resnet-model/wallaroo-run-anywhere-model-architecture-publish-cv-resnet-model.ipynb",
    #     "outputDir": "/reference/wallaroo-run-anywhere/inference-on-any-architecture/arm",
    #     "outputFile": "wallaroo-run-anywhere-model-architecture-publish-cv-resnet-model-reference.md"
    # },
    # ##### Run Anywhere for ARM Architecture Tutorial: Hugging Face Summarization Model
    # {
    #     "inputFile": "wallaroo-run-anywhere/inference-on-any-architecture/arm/edge-arm-publish-hf-summarization-model/wallaroo-run-anywhere-model-architecture-publish-hf-summarization.ipynb",
    #     "outputDir": "/reference/wallaroo-run-anywhere/inference-on-any-architecture/arm",
    #     "outputFile": "wallaroo-run-anywhere-model-architecture-publish-hf-summarization-reference.md"
    # },
    # ##### Run Anywhere for ARM Architecture Tutorial: House Price Predictor Model
    # {
    #     "inputFile": "wallaroo-run-anywhere/inference-on-any-architecture/arm/edge-arm-publish-linear-regression-houseprice-model/wallaroo-run-anywhere-model-architecture-linear-regression-houseprice-tutorial.ipynb",
    #     "outputDir": "/reference/wallaroo-run-anywhere/inference-on-any-architecture/arm",
    #     "outputFile": "wallaroo-run-anywhere-model-architecture-linear-regression-houseprice-tutorial-reference.md"
    # },
    # #### Jetson
    # ##### Run Anywhere With Jetson Acceleration Tutorial: Aloha Model
    # {
    #     "inputFile": "wallaroo-run-anywhere/inference-on-any-architecture/jetson/run-anywhere-acceleration-aloha.ipynb",
    #     "outputDir": "/reference/wallaroo-run-anywhere/inference-on-any-architecture/jetson",
    #     "outputFile": "run-anywhere-acceleration-aloha-reference.md"
    # },
    # #### power10
    # ##### Run Anywhere:  Deploy and Publish Computer Vision Model Resnet50 for IBM Power10
    # {
    #     "inputFile": "wallaroo-run-anywhere/inference-on-any-architecture/power10/computer-vision-resnet-power10/run-anywhere-power10-computer-vision-resnet50-benchmarking.ipynb",
    #     "outputDir": "/reference/wallaroo-run-anywhere/inference-on-any-architecture/power10/computer-vision-resnet-power10",
    #     "outputFile": "run-anywhere-power10-computer-vision-resnet50-benchmarking-reference.md"
    # },
    # ##### Run Anywhere:  Deploy and Publish Computer Vision Model Resnet50 with Post Processing for IBM Power10
    # {
    #     "inputFile": "wallaroo-run-anywhere/inference-on-any-architecture/power10/computer-vision-resnet-power10/run-anywhere-power10-computer-vision-resnet50.ipynb",
    #     "outputDir": "/reference/wallaroo-run-anywhere/inference-on-any-architecture/power10/computer-vision-resnet-power10",
    #     "outputFile": "run-anywhere-power10-computer-vision-resnet50-reference.md"
    # },
    # #### x86 - @JOHN rename to infer
    # ##### Classification Cybersecurity Services Edge Deployment Tutorial
    # {
    #     "inputFile": "wallaroo-run-anywhere/inference-on-any-architecture/x86/edge-classification-cybersecurity/edge-classification-cybersecurity-deployment.ipynb",
    #     "outputDir": "/reference/wallaroo-run-anywhere/inference-on-any-architecture/x86",
    #     "outputFile": "edge-classification-cybersecurity-deployment-reference.md"
    # },
    # ##### Classification Financial Services Edge Deployment Tutorial
    # {
    #     "inputFile": "wallaroo-run-anywhere/inference-on-any-architecture/x86/edge-classification-finserv/edge-classification-finserv-deployment.ipynb",
    #     "outputDir": "/reference/wallaroo-run-anywhere/inference-on-any-architecture/x86",
    #     "outputFile": "edge-classification-finserv-deployment-reference.md"
    # },
    # ##### Classification Financial Services Edge Deployment Demonstration via API
    # {
    #     "inputFile": "wallaroo-run-anywhere/inference-on-any-architecture/x86/edge-classification-finserv-api/edge-classification-finserv-deployment-via-api.ipynb",
    #     "outputDir": "/reference/wallaroo-run-anywhere/inference-on-any-architecture/x86",
    #     "outputFile": "edge-classification-finserv-deployment-via-api-reference.md"
    # },
    # ##### Image Detection for Health Care Computer Vision Tutorial Part 00: Prerequisites
    # {
    #     "inputFile": "wallaroo-run-anywhere/inference-on-any-architecture/x86/edge-cv-healthcare-images/00_computer-vision-mitochondria-imaging-edge-deployment-example.ipynb",
    #     "outputDir": "/reference/wallaroo-run-anywhere/inference-on-any-architecture/x86/edge-cv-healthcare-images",
    #     "outputFile": "00_computer-vision-mitochondria-imaging-edge-deployment-example-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-run-anywhere/inference-on-any-architecture/x86/edge-cv-healthcare-images/01_computer-vision-mitochondria-imaging-edge-deployment-example.ipynb",
    #     "outputDir": "/reference/wallaroo-run-anywhere/inference-on-any-architecture/x86/edge-cv-healthcare-images",
    #     "outputFile": "01_computer-vision-mitochondria-imaging-edge-deployment-example-reference.md"
    # },
    # ##### Summarization Text Edge Deployment Demonstration
    # {
    #     "inputFile": "wallaroo-run-anywhere/inference-on-any-architecture/x86/edge-llm-summarization/edge-hf-summarization.ipynb",
    #     "outputDir": "/reference/wallaroo-run-anywhere/inference-on-any-architecture/x86",
    #     "outputFile": "edge-hf-summarization-reference.md"
    # },
    # ##### Computer Vision for Object Detection for Edge Deployments in Retail
    # {
    #     "inputFile": "wallaroo-run-anywhere/inference-on-any-architecture/x86/edge-observability-cv/cv-retail-edge-observability.ipynb",
    #     "outputDir": "/reference/wallaroo-run-anywhere/inference-on-any-architecture/x86",
    #     "outputFile": "cv-retail-edge-observability-reference.md"
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
    # ](01_notebooks-in-prod_explore_and_train-reference_files
    # image_replace = f'![png]({outputdir}'
    document = re.sub('!\[png\]\(', f'![png](/images/2025.1{outputdir}/', document)
    document = re.sub('\(./images', '(/images/2025.1', document)
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