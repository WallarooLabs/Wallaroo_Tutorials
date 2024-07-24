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
    ### wallaroo 101
    # {
    #     "inputFile": "wallaroo-101/Wallaroo-101.ipynb",
    #     "outputDir": "/wallaroo-101",
    #     "outputFile": "wallaroo-101-reference.md"
    # },
    # # ## deploy and serve
    # ### parallel infer with aloha
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/parallel-inferences-sdk-aloha-tutorial/wallaroo-parallel-infer-sdk-with-aloha.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve",
    #     "outputFile": "parallel-infer-with-aloha-reference.md"
    # },

    # ## onnx multi io
    {
        "inputFile": "wallaroo-model-deploy-and-serve/onnx-multi-input-demo/test_autoconv_onnx_multi_io.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve",
        "outputFile": "test_autoconv_onnx_multi_io-reference.md"
    },
    # ## model registry
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/mlflow-registries-upload-tutorials/Wallaroo-model-registry-demonstration.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/model-registry",
    #     "outputFile": "Wallaroo-model-registry-demonstration-reference.md"
    # },
    # ## keras
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/keras-upload-tutorials/wallaroo-upload-keras_sequential_model_single_io.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/keras",
    #     "outputFile": "wallaroo-upload-keras_sequential_model_single_io-reference.md"
    # },
    # ## hugging face
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/hugging-face-upload-tutorials/wallaroo-api-upload-hf-zero_shot_classification.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/hugging-face",
    #     "outputFile": "wallaroo-api-upload-hf-zero_shot_classification.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/hugging-face-upload-tutorials/wallaroo-sdk-upload-hf-zero_shot_classification.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/hugging-face",
    #     "outputFile": "wallaroo-sdk-upload-hf-zero_shot_classification.md"
    # },
    # ## computer vision
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/computer-vision/00_computer_vision_tutorial_intro.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/computer-vision",
    #     "outputFile": "00-computer-vision-tutorial-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/computer-vision/01_computer_vision_tutorial_mobilenet.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/computer-vision",
    #     "outputFile": "01_computer_vision_tutorial_mobilenet-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/computer-vision/02_computer_vision_tutorial_resnet50.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/computer-vision",
    #     "outputFile": "02_computer_vision_tutorial_resnet50-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/computer-vision/03_computer_vision_tutorial_shadow_deploy.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/computer-vision",
    #     "outputFile": "03_computer_vision_tutorial_shadow_deploy-reference.md"
    # },
    # ## BYOP
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/arbitrary-python-upload-tutorials/00_wallaroo-upload-arbitrary-python-vgg16-model-generation.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/arbitrary-python",
    #     "outputFile": "00_wallaroo-upload-arbitrary-python-vgg16-model-generation-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/arbitrary-python-upload-tutorials/01_wallaroo-upload-arbitrary-python-vgg16-model-deployment.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/arbitrary-python",
    #     "outputFile": "01_wallaroo-upload-arbitrary-python-vgg16-model-deployment-reference.md"
    # },
    # ## Python steps
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/python-upload-tutorials/python-step-dataframe-output-logging-example-sdk.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve",
    #     "outputFile": "python-step-dataframe-output-logging-example-sdk-reference.md"
    # },
    # ## notebooks in prod
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/notebooks_in_prod/00_notebooks_in_prod_introduction.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/notebook_in_prod",
    #     "outputFile": "_index.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/notebooks_in_prod/01_notebooks_in_prod_explore_and_train.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/notebook_in_prod",
    #     "outputFile": "01_notebooks_in_prod_explore_and_train-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/notebooks_in_prod/02_notebooks_in_prod_automated_training_process.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/notebook_in_prod",
    #     "outputFile": "02_notebooks_in_prod_automated_training_process-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/notebooks_in_prod/03_notebooks_in_prod_deploy_model_python.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/notebook_in_prod",
    #     "outputFile": "03_notebooks_in_prod_deploy_model-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/notebooks_in_prod/04_notebooks_in_prod_regular_batch_inferences.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/notebook_in_prod",
    #     "outputFile": "04_notebooks_in_prod_regular_batch_inferences-reference.md"
    # },
    # ## pytorch
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/pytorch-upload-tutorials/wallaroo-upload-pytorch-multi-input-output.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/pytorch",
    #     "outputFile": "wallaroo-upload-pytorch-multi-input-output-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/pytorch-upload-tutorials/wallaroo-upload-pytorch-single-input-output.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/pytorch",
    #     "outputFile": "wallaroo-upload-pytorch-single-input-output-reference.md"
    # },
    # ## tensorflow
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/tensorflow-upload-tutorials/wallaroo-upload-tensorflow.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/tensorflow",
    #     "outputFile": "wallaroo-upload-tensorflow-reference.md"
    # },
    # ## onnx multi io
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/onnx-multi-input-demo/test_autoconv_onnx_multi_io.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve",
    #     "outputFile": "test_autoconv_onnx_multi_io-reference.md"
    # },
    # ## model registry
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/mlflow-registries-upload-tutorials/Wallaroo-model-registry-demonstration.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/model-registry",
    #     "outputFile": "Wallaroo-model-registry-demonstration-reference.md"
    # },
    # ## keras
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/keras-upload-tutorials/wallaroo-upload-keras_sequential_model_single_io.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/keras",
    #     "outputFile": "wallaroo-upload-keras_sequential_model_single_io-reference.md"
    # },
    # ## hugging face
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/hugging-face-upload-tutorials/wallaroo-api-upload-hf-zero_shot_classification.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/hugging-face",
    #     "outputFile": "wallaroo-api-upload-hf-zero_shot_classification.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/hugging-face-upload-tutorials/wallaroo-sdk-upload-hf-zero_shot_classification.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/hugging-face",
    #     "outputFile": "wallaroo-sdk-upload-hf-zero_shot_classification.md"
    # },
    # ## computer vision
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/computer-vision/00_computer_vision_tutorial_intro.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/computer-vision",
    #     "outputFile": "00-computer-vision-tutorial-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/computer-vision/01_computer_vision_tutorial_mobilenet.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/computer-vision",
    #     "outputFile": "01_computer_vision_tutorial_mobilenet-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/computer-vision/02_computer_vision_tutorial_resnet50.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/computer-vision",
    #     "outputFile": "02_computer_vision_tutorial_resnet50-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/computer-vision/03_computer_vision_tutorial_shadow_deploy.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/computer-vision",
    #     "outputFile": "03_computer_vision_tutorial_shadow_deploy-reference.md"
    # },
    # ## BYOP
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/arbitrary-python-upload-tutorials/00_wallaroo-upload-arbitrary-python-vgg16-model-generation.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/arbitrary-python",
    #     "outputFile": "00_wallaroo-upload-arbitrary-python-vgg16-model-generation-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/arbitrary-python-upload-tutorials/01_wallaroo-upload-arbitrary-python-vgg16-model-deployment.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/arbitrary-python",
    #     "outputFile": "01_wallaroo-upload-arbitrary-python-vgg16-model-deployment-reference.md"
    # },
    # ## Python steps
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/python-upload-tutorials/python-step-dataframe-output-logging-example-sdk.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve",
    #     "outputFile": "python-step-dataframe-output-logging-example-sdk-reference.md"
    # },
    # ## notebooks in prod
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/notebooks_in_prod/00_notebooks_in_prod_introduction.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/notebook_in_prod",
    #     "outputFile": "_index.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/notebooks_in_prod/01_notebooks_in_prod_explore_and_train.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/notebook_in_prod",
    #     "outputFile": "01_notebooks_in_prod_explore_and_train-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/notebooks_in_prod/02_notebooks_in_prod_automated_training_process.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/notebook_in_prod",
    #     "outputFile": "02_notebooks_in_prod_automated_training_process-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/notebooks_in_prod/03_notebooks_in_prod_deploy_model_python.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/notebook_in_prod",
    #     "outputFile": "03_notebooks_in_prod_deploy_model-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/notebooks_in_prod/04_notebooks_in_prod_regular_batch_inferences.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/notebook_in_prod",
    #     "outputFile": "04_notebooks_in_prod_regular_batch_inferences-reference.md"
    # },
    # ## pytorch
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/pytorch-upload-tutorials/wallaroo-upload-pytorch-multi-input-output.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/pytorch",
    #     "outputFile": "wallaroo-upload-pytorch-multi-input-output-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/pytorch-upload-tutorials/wallaroo-upload-pytorch-single-input-output.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/pytorch",
    #     "outputFile": "wallaroo-upload-pytorch-single-input-output-reference.md"
    # },
    # ## tensorflow
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/tensorflow-upload-tutorials/wallaroo-upload-tensorflow.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/tensorflow",
    #     "outputFile": "wallaroo-upload-tensorflow-reference.md"
    # },

    # ## xgboost
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/xgboost-upload-tutorials/wallaroo-sdk-upload-booster-binary-classification-conversion.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/xgboost-upload-tutorials",
    #     "outputFile": "wallaroo-sdk-upload-booster-binary-classification-conversion-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/xgboost-upload-tutorials/wallaroo-sdk-upload-booster-multi-classification-softmax-conversion.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/xgboost-upload-tutorials",
    #     "outputFile": "wallaroo-sdk-upload-booster-multi-classification-softmax-conversion-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/xgboost-upload-tutorials/wallaroo-sdk-upload-booster-multi-classification-softprob-conversion.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/xgboost-upload-tutorials",
    #     "outputFile": "wallaroo-sdk-upload-booster-multi-classification-softprob-conversion-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/xgboost-upload-tutorials/wallaroo-sdk-upload-booster-regression-conversion.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/xgboost-upload-tutorials",
    #     "outputFile": "wallaroo-sdk-upload-booster-regression-conversion-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/xgboost-upload-tutorials/wallaroo-sdk-upload-booster-rf-classification-conversion.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/xgboost-upload-tutorials",
    #     "outputFile": "wallaroo-sdk-upload-booster-rf-classification-conversion-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/xgboost-upload-tutorials/wallaroo-sdk-upload-booster-rf-regression-conversion.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/xgboost-upload-tutorials",
    #     "outputFile": "wallaroo-sdk-upload-booster-rf-regression-conversion-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/xgboost-upload-tutorials/wallaroo-sdk-upload-xbg-classification.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/xgboost-upload-tutorials",
    #     "outputFile": "wallaroo-sdk-upload-xbg-classification-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/xgboost-upload-tutorials/wallaroo-sdk-upload-xbg-regressor.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/xgboost-upload-tutorials",
    #     "outputFile": "wallaroo-sdk-upload-xbg-regressor-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/xgboost-upload-tutorials/wallaroo-sdk-upload-xbg-rf-classification.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/xgboost-upload-tutorials",
    #     "outputFile": "wallaroo-sdk-upload-xbg-rf-classification-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/xgboost-upload-tutorials/wallaroo-sdk-upload-xbg-rf-regressor.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/xgboost-upload-tutorials",
    #     "outputFile": "wallaroo-sdk-upload-xbg-rf-regressor-reference.md"
    # },
    # ## sklearn
    {
        "inputFile": "wallaroo-model-deploy-and-serve/sklearn-upload-tutorials/wallaroo-upload-sklearn-clustering-kmeans.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/sklearn",
        "outputFile": "wallaroo-upload-sklearn-clustering-kmeans.md"
    },
    {
        "inputFile": "wallaroo-model-deploy-and-serve/sklearn-upload-tutorials/wallaroo-upload-sklearn-clustering-svm-pca.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/sklearn",
        "outputFile": "wallaroo-upload-sklearn-clustering-svm-pca.md"
    },
    {
        "inputFile": "wallaroo-model-deploy-and-serve/sklearn-upload-tutorials/wallaroo-upload-sklearn-clustering-svm.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/sklearn",
        "outputFile": "wallaroo-upload-sklearn-clustering-svm.md"
    },
    {
        "inputFile": "wallaroo-model-deploy-and-serve/sklearn-upload-tutorials/wallaroo-upload-sklearn-linear-regression.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/sklearn",
        "outputFile": "wallaroo-upload-sklearn-linear-regression.md"
    },
    {
        "inputFile": "wallaroo-model-deploy-and-serve/sklearn-upload-tutorials/wallaroo-upload-sklearn-logistic-regression.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/sklearn",
        "outputFile": "wallaroo-upload-sklearn-logistic-regression.md"
    },
    # ### multiple replicas forecast updates
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/pipeline_multiple_replicas_forecast_tutorial/00_multiple_replicas_forecast.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/statsmodel/",
    #     "outputFile": "00_multiple_replicas_forecast-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/pipeline_multiple_replicas_forecast_tutorial/01_multiple_replicas_forecast.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/statsmodel/",
    #     "outputFile": "01_multiple_replicas_forecast-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/pipeline_multiple_replicas_forecast_tutorial/02_multiple_replicas_forecast.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/statsmodel/",
    #     "outputFile": "02_multiple_replicas_forecast-reference.md"
    # },
    # ## aloha main
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/aloha/aloha_demo.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/",
    #     "outputFile": "aloha_demo.ipynb-reference.md"
    # },
    # ## computer vision mitochondria
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/computer-vision-mitochondria-imaging/00_computer-vision-mitochondria-imaging-example.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/cv-mitochondria-imaging",
    #     "outputFile": "00_computer-vision-mitochondria-imaging-example-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/computer-vision-mitochondria-imaging/01_computer-vision-mitochondria-imaging-example.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/cv-mitochondria-imaging",
    #     "outputFile": "01_computer-vision-mitochondria-imaging-example-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/computer-vision-mitochondria-imaging/02_computer-vision-mitochondria-imaging-example.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/cv-mitochondria-imaging",
    #     "outputFile": "02_computer-vision-mitochondria-imaging-example-reference.md"
    # },
    # ## cv yolov8
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/computer-vision-yolov8/computer-vision-yolov8-demonstration.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve",
    #     "outputFile": "computer-vision-yolov8-demonstration-reference.md"
    # },
    # ## demand curve
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/demand_curve/demandcurve_demo.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve",
    #     "outputFile": "demandcurve_demo-reference.md"
    # },
    # ## hf CLIP ViT-B/32 Transformer
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/hf-clip-vit-base/clip-vit-hugging-face.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve",
    #     "outputFile": "clip-vit-hugging-face-reference.md"
    # },
    # ## whisper demo
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/hf-whisper/wallaroo-whisper_demo.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve",
    #     "outputFile": "wallaroo-whisper_demo-reference.md"
    # },
    # ## imdb
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/imdb/imdb_sample.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve",
    #     "outputFile": "imdb_sample-reference.md"
    # },
    # ## mlflow
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/mlflow-tutorial/wallaroo-mlflow-tutorial.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve",
    #     "outputFile": "wallaroo-mlflow-tutorial-reference.md"
    # },
    # ## computer vision BYOP
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/wallaroo-model-upload-deploy-byop-cv-tutorial/wallaroo-model-upload-deploy-byop-cv-tutorial.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve",
    #     "outputFile": "wallaroo-model-upload-deploy-byop-cv-tutorial-reference.md"
    # },
    # ## tag management
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/wallaroo-tag-management/wallaroo-tags-guide.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve",
    #     "outputFile": "wallaroo-tags-guide-reference.md"
    # },
    # ## endpoints sdk and api
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/wallaroo-model-endpoints/wallaroo-model-endpoints-api.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/model-endpoints",
    #     "outputFile": "wallaroo-model-endpoints-api-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/wallaroo-model-endpoints/wallaroo-model-endpoints-sdk.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/model-endpoints",
    #     "outputFile": "wallaroo-model-endpoints-sdk-reference.md"
    # },
    # ## automate
    # ### connection api bigquery
    # {
    #     "inputFile": "wallaroo-automate/connection_api_bigquery_tutorial/connection_api_bigquery_tutorial.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-automate",
    #     "outputFile": "connection_api_bigquery_tutorial-reference.md"
    # },
    # ### connection simple tutorial
    # {
    #     "inputFile": "wallaroo-automate/orchestration_api_simple_tutorial/data_orchestrators_api_simple_tutorial.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-automate",
    #     "outputFile": "data_orchestrators_api_simple_tutorial-reference.md"
    # },
    # ### orchestration sdk bigquery houseprice
    # {
    #     "inputFile": "wallaroo-automate/orchestration_sdk_bigquery_houseprice_tutorial/orchestration_sdk_bigquery_houseprice_tutorial.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-automate",
    #     "outputFile": "orchestration_sdk_bigquery_houseprice_tutorial-reference.md"
    # },
    # ### orchestration sdk comprehensive 
    # {
    #     "inputFile": "wallaroo-automate/orchestration_sdk_comprehensive_tutorial/data_connectors_and_orchestrators_comprehensive_tutorial.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-automate",
    #     "outputFile": "data_connectors_and_orchestrators_comprehensive_tutorial-reference.md"
    # },
    # ### orchestration sdk simple 
    # {
    #     "inputFile": "wallaroo-automate/orchestration_sdk_simple_tutorial/data_connectors_and_orchestrators_simple_tutorial.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-automate",
    #     "outputFile": "data_connectors_and_orchestrators_simple_tutorial-reference.md"
    # },
    # ## wallaroo free
    # ### computer vision frcnn
    # {
    #     "inputFile": "wallaroo-free-tutorials/wallaroo-inference-server-tutorials/wallaroo-inference-server-cv-frcnn/wallaroo-inference-server-cv-frcnn.ipynb",
    #     "outputDir": "/wallaroo-free-tutorials/wallaroo-inference-server-tutorials",
    #     "outputFile": "wallaroo-inference-server-cv-frcnn-reference.md"
    # },
    # ### computer vision resnet
    # {
    #     "inputFile": "wallaroo-free-tutorials/wallaroo-inference-server-tutorials/wallaroo-inference-server-cv-resnet/wallaroo-inference-server-cv-resnet.ipynb",
    #     "outputDir": "/wallaroo-free-tutorials/wallaroo-inference-server-tutorials",
    #     "outputFile": "wallaroo-inference-server-cv-resnet-reference.md"
    # },
    # ### computer vision unet
    # {
    #     "inputFile": "wallaroo-free-tutorials/wallaroo-inference-server-tutorials/wallaroo-inference-server-cv-unet/wallaroo-inference-server-cv-unet.ipynb",
    #     "outputDir": "/wallaroo-free-tutorials/wallaroo-inference-server-tutorials",
    #     "outputFile": "wallaroo-inference-server-cv-unet-reference.md"
    # },
    # ### computer vision yolov8
    # {
    #     "inputFile": "wallaroo-free-tutorials/wallaroo-inference-server-tutorials/wallaroo-inference-server-cv-yolov8/wallaroo-inference-server-cv-yolov8.ipynb",
    #     "outputDir": "/wallaroo-free-tutorials/wallaroo-inference-server-tutorials",
    #     "outputFile": "wallaroo-inference-server-cv-yolov8-reference.md"
    # },
    # ### hf summarizer
    # {
    #     "inputFile": "wallaroo-free-tutorials/wallaroo-inference-server-tutorials/wallaroo-inference-server-hf-summarizer/wallaroo-inference-server-hf-summarization.ipynb",
    #     "outputDir": "/wallaroo-free-tutorials/wallaroo-inference-server-tutorials",
    #     "outputFile": "wallaroo-inference-server-hf-summarization-reference.md"
    # },
    # ### llama v2
    # {
    #     "inputFile": "wallaroo-free-tutorials/wallaroo-inference-server-tutorials/wallaroo-inference-server-llama2/wallaroo-inference-server-llama2.ipynb",
    #     "outputDir": "/wallaroo-free-tutorials/wallaroo-inference-server-tutorials",
    #     "outputFile": "wallaroo-inference-server-llama2-reference.md"
    # },
    # ## observe tutorials
    # ### House Price Testing Life Cycle Comprehensive
    # {
    #     "inputFile": "wallaroo-observe/houseprice-saga/house-price-model-saga-comprehensive.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-observe",
    #     "outputFile": "house-price-model-saga-comprehensive-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-observe/houseprice-saga/house-price-model-saga-prep.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-observe",
    #     "outputFile": "house-price-model-saga-prep-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-observe/houseprice-saga/house-price-model-saga-short.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-observe",
    #     "outputFile": "house-price-model-saga-short-reference.md"
    # },
    # ## Anomaly Detection with CCFraud
    # {
    #     "inputFile": "wallaroo-observe/model-observability-anomaly-detection-ccfraud-sdk-tutorial/model-observability-anomaly-detection-ccfraud-sdk-tutorial.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-observe",
    #     "outputFile": "model-observability-anomaly-detection-ccfraud-sdk-tutorial-reference.md"
    # },
    # ## Anomaly Detection with House Price Prediction
    # {
    #     "inputFile": "wallaroo-observe/model-observability-anomaly-detection-houseprice-sdk-tutorial/model-observability-anomaly-detection-house-price-sdk-tutorial.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-observe",
    #     "outputFile": "model-observability-anomaly-detection-house-price-sdk-tutorial-reference.md"
    # },
    # ### Pipeline Log Tutorial SDK
    # {
    #     "inputFile": "wallaroo-observe/pipeline-log-tutorial/pipeline_log_tutorial.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-observe",
    #     "outputFile": "pipeline_log_tutorial-reference.md"
    # },
    # ### Pipeline Log Tutorial API
    # {
    #     "inputFile": "wallaroo-observe/pipeline_api_log_tutorial/pipeline_api_log_tutorial.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-observe",
    #     "outputFile": "pipeline_api_log_tutorial-reference.md"
    # },
    # ### Computer Vision Pipeline API Log
    # {
    #     "inputFile": "wallaroo-observe/pipeline_api_log_tutorial_cv/pipeline_api_log_tutorial_computer_vision.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-observe",
    #     "outputFile": "pipeline_api_log_tutorial_computer_vision-reference.md"
    # },
    # ### model observability assays single 
    # {
    #     "inputFile": "wallaroo-observe/wallaro-model-observability-assays/wallaroo_model_observability_assays.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-observe",
    #     "outputFile": "wallaroo_model_observability_assays-reference.md"
    # },
    # ## optimize
    # ### ab testing
    # {
    #     "inputFile": "wallaroo-optimize/abtesting/wallaroo-abtesting-tutorial.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-optimize",
    #     "outputFile": "wallaroo-abtesting-tutorial-reference.md"
    # },
    # ### hot swap
    # {
    #     "inputFile": "wallaroo-optimize/model_hot_swap/wallaroo_hot_swap_tutorial.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-optimize",
    #     "outputFile": "wallaroo_hot_swap_tutorial-reference.md"
    # },
    # ### shadow deploy
    # {
    #     "inputFile": "wallaroo-optimize/shadow_deploy/shadow_deployment_tutorial.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-optimize",
    #     "outputFile": "shadow_deployment_tutorial-reference.md"
    # },
    # ## run anywhere
    # ### computer vision unet
    # {
    #     "inputFile": "wallaroo-run-anywhere/pipeline-edge-publish/edge-unet-brain-segmentation-demonstration/unet-run-anywhere-demonstration.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-run-anywhere",
    #     "outputFile": "unet-run-anywhere-demonstration-reference.md"
    # },
    # ### Computer Vision for Object Detection for Edge Deployments for Retail
    # {
    #     "inputFile": "wallaroo-run-anywhere/pipeline-edge-publish/edge-observability-cv/cv-retail-edge-observability.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-run-anywhere",
    #     "outputFile": "cv-retail-edge-observability-reference.md"
    # },
    # ### Classification Financial Services Edge Deployment
    # {
    #     "inputFile": "wallaroo-run-anywhere/pipeline-edge-publish/edge-observability-classification-finserv/edge-observabilty-classification-finserv-deployment.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-run-anywhere",
    #     "outputFile": "edge-observabilty-classification-finserv-deployment-reference.md"
    # },
    # ### Classification Financial Services Edge Deployment Demonstration via API
    # {
    #     "inputFile": "wallaroo-run-anywhere/pipeline-edge-publish/edge-observability-classification-finserv-api/edge-observability-classification-finserv-deployment-via-api.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-run-anywhere",
    #     "outputFile": "edge-observability-classification-finserv-deployment-via-api-reference.md"
    # },
    # ### Wallaroo Edge Observability with Assays
    # {
    #     "inputFile": "wallaroo-run-anywhere/pipeline-edge-publish/edge-observability-assays/edge-observability-assays.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-run-anywhere",
    #     "outputFile": "edge-observability-assays-reference.md"
    # },
    # ### Summarization Text Edge Deployment
    # {
    #     "inputFile": "wallaroo-run-anywhere/pipeline-edge-publish/edge-llm-summarization/edge-hf-summarization.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-run-anywhere",
    #     "outputFile": "edge-hf-summarization-reference.md"
    # },
    # ### Image Detection for Health Care Computer Vision aka mitochrondria edge
    # {
    #     "inputFile": "wallaroo-run-anywhere/pipeline-edge-publish/edge-cv-healthcare-images/00_computer-vision-mitochondria-imaging-edge-deployment-example.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-run-anywhere",
    #     "outputFile": "00_computer-vision-mitochondria-imaging-edge-deployment-example-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-run-anywhere/pipeline-edge-publish/edge-cv-healthcare-images/01_computer-vision-mitochondria-imaging-edge-deployment-example.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-run-anywhere",
    #     "outputFile": "01_computer-vision-mitochondria-imaging-edge-deployment-example-reference.md"
    # },
    # ### Computer Vision Yolov8n Edge Deployment
    # {
    #     "inputFile": "wallaroo-run-anywhere/pipeline-edge-publish/edge-computer-vision-yolov8/edge-computer-vision-yolov8.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-run-anywhere",
    #     "outputFile": "edge-computer-vision-yolov8-reference.md"
    # },
    # ### Classification Financial Services Edge Deployment
    # {
    #     "inputFile": "wallaroo-run-anywhere/pipeline-edge-publish/edge-classification-finserv/edge-classification-finserv-deployment.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-run-anywhere",
    #     "outputFile": "edge-classification-finserv-deployment-reference.md"
    # },
    # ### Classification Financial Services Edge Deployment Demonstration via API
    # {
    #     "inputFile": "wallaroo-run-anywhere/pipeline-edge-publish/edge-classification-finserv-api/edge-classification-finserv-deployment-via-api.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-run-anywhere",
    #     "outputFile": "edge-classification-finserv-deployment-via-api-reference.md"
    # },
    # ### Classification Financial Services Edge 
    # {
    #     "inputFile": "wallaroo-run-anywhere/pipeline-edge-publish/edge-classification-cybersecurity/edge-classification-cybersecurity-deployment.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-run-anywhere",
    #     "outputFile": "edge-classification-cybersecurity-deployment-reference.md"
    # },
    # ### Arbitrary Python Edge Deploy
    # {
    #     "inputFile": "wallaroo-run-anywhere/pipeline-edge-publish/edge-arbitrary-python/edge-arbitrary-python-demonstration.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-run-anywhere",
    #     "outputFile": "edge-arbitrary-python-demonstration-reference.md"
    # },
    # ### LLM Summarization GPU Edge Deployment
    # {
    #     "inputFile": "wallaroo-run-anywhere/pipeline-architecture/wallaroo-gpu-llm-summarization/wallaroo-gpu-llm-summarization-demonstration.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-run-anywhere",
    #     "outputFile": "wallaroo-gpu-llm-summarization-demonstration-reference.md"
    # },
    # ### LLM Summarization ARM Edge Deployment
    # {
    #     "inputFile": "wallaroo-run-anywhere/pipeline-architecture/wallaroo-arm-llm-summarization/wallaroo-arm-llm-summarization-demonstration.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-run-anywhere",
    #     "outputFile": "wallaroo-arm-llm-summarization-demonstration-reference.md"
    # },
    # ### Computer Vision ARM Edge Deployment Tutorial
    # {
    #     "inputFile": "wallaroo-run-anywhere/pipeline-architecture/wallaroo-arm-cv-arrow/arm-computer-vision-demonstration.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-run-anywhere",
    #     "outputFile": "arm-computer-vision-demonstration-reference.md"
    # },
    # ### Computer Vision Yolov8n ARM Deployment
    # {
    #     "inputFile": "wallaroo-run-anywhere/pipeline-architecture/wallaroo-arm-computer-vision-yolov8/wallaroo-arm-cv-yolov8-demonstration.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-run-anywhere",
    #     "outputFile": "wallaroo-arm-cv-yolov8-demonstration-reference.md"
    # },
    # ### Classification Financial Services with Arm Architecture
    # {
    #     "inputFile": "wallaroo-run-anywhere/pipeline-architecture/wallaroo-arm-classification-finserv/arm-classification-finserv.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-run-anywhere",
    #     "outputFile": "arm-classification-finserv-reference.md"
    # },
    # ### Classification Cybersecurity with Arm Architecture
    # {
    #     "inputFile": "wallaroo-run-anywhere/pipeline-architecture/wallaroo-arm-classification-cybersecurity/arm-classification-cybersecurity.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-run-anywhere",
    #     "outputFile": "arm-classification-cybersecurity-reference.md"
    # },
    # ### Arbitrary Python ARM Deployment Tutorial
    # {
    #     "inputFile": "wallaroo-run-anywhere/pipeline-architecture/wallaroo-arm-byop-vgg16/wallaroo-arm-arbitrary-python-vgg16-model-deployment.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-run-anywhere",
    #     "outputFile": "wallaroo-arm-arbitrary-python-vgg16-model-deployment-reference.md"
    # },
    # ### Run Anywhere With Acceleration Tutorial: Aloha Model
    # {
    #     "inputFile": "wallaroo-run-anywhere/run-anywhere-acceleration-aloha/run-anywhere-acceleration-aloha.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-run-anywhere",
    #     "outputFile": "run-anywhere-acceleration-aloha-reference.md"
    # },
    # ### In-Line Model Updates at the Edge Tutorial
    # {
    #     "inputFile": "wallaroo-run-anywhere/in-line-edge-model-replacements-tutorial/in-line-edge-model-replacements-tutorial.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-run-anywhere",
    #     "outputFile": "in-line-edge-model-replacements-tutorial-reference.md"
    # },
    # ### Large Language Model with GPU Pipeline Deployment
    # {
    #     "inputFile": "wallaroo-run-anywhere/gpu-deployment/wallaroo-llm-with-gpu-demonstration.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-run-anywhere",
    #     "outputFile": "wallaroo-llm-with-gpu-demonstration-reference.md"
    # },
    # ### Edge Observability with No/Low Connection
    # {
    #     "inputFile": "wallaroo-run-anywhere/edge-observability-low-no-connection/edge-observability-low-no-connection-tutorial.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-run-anywhere",
    #     "outputFile": "edge-observability-low-no-connection-tutorial-reference.md"
    # },
    # ### Model Drift Detection for Edge Deployments
    # {
    #     "inputFile": "wallaroo-run-anywhere/edge-observability-assays/00_drift-detection-for-edge-deployments-tutorial-prep.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-run-anywhere",
    #     "outputFile": "00_drift-detection-for-edge-deployments-tutorial-prep-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-run-anywhere/edge-observability-assays/01_drift-detection-for-edge-deployments-tutorial-examples.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-run-anywhere",
    #     "outputFile": "01_drift-detection-for-edge-deployments-tutorial-examples-reference.md"
    # },
    # ### Run Anywhere for ARM Architecture Tutorial: House Price Predictor Model
    # {
    #     "inputFile": "wallaroo-run-anywhere/edge-architecture-publish-linear-regression-houseprice-model/wallaroo-run-anywhere-model-architecture-linear-regression-houseprice-tutorial.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-run-anywhere",
    #     "outputFile": "wallaroo-run-anywhere-model-architecture-linear-regression-houseprice-tutorial-reference.md"
    # },
    # ### Run Anywhere for ARM Architecture Tutorial: House Price Predictor Model
    # {
    #     "inputFile": "wallaroo-run-anywhere/edge-architecture-publish-linear-regression-houseprice-model/wallaroo-run-anywhere-model-architecture-linear-regression-houseprice-tutorial.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-run-anywhere",
    #     "outputFile": "wallaroo-run-anywhere-model-architecture-linear-regression-houseprice-tutorial-reference.md"
    # },
    # ### Run Anywhere for ARM Architecture Tutorial: Hugging Face Summarization
    # {
    #     "inputFile": "wallaroo-run-anywhere/edge-architecture-publish-hf-summarization-model/wallaroo-run-anywhere-model-architecture-publish-hf-summarization.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-run-anywhere",
    #     "outputFile": "wallaroo-run-anywhere-model-architecture-publish-hf-summarization-reference.md"
    # },
    # ### Run Anywhere for ARM Architecture Tutorial: Custom Inference Computer Vision with Resnet50
    # {
    #     "inputFile": "wallaroo-run-anywhere/edge-architecture-publish-cv-resnet-model/wallaroo-run-anywhere-model-architecture-publish-cv-resnet-model.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-run-anywhere",
    #     "outputFile": "wallaroo-run-anywhere-model-architecture-publish-cv-resnet-model-reference.md"
    # },
    # ### Computer Vision for Object Detection for Edge Deployments
    # {
    #     "inputFile": "wallaroo-run-anywhere/pipeline-edge-publish/edge-observability-cv/cv-retail-edge-observability.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-run-anywhere",
    #     "outputFile": "cv-retail-edge-observability-reference.md"
    # },
    # ## tools
    # ### Wallaroo JSON Inference Data to DataFrame and Arrow Tutorials
    # {
    #     "inputFile": "tools/convert_wallaroo_data_to_pandas_arrow/convert_wallaroo_inference_data.ipynb",
    #     "outputDir": "/wallaroo-tutorials/tools",
    #     "outputFile": "convert_wallaroo_inference_data-reference.md"
    # },
    # ## LLMs
    # ### in-line monitoring
    # {
    #     "inputFile": "wallaroo-llms/llm-monitoring/llm-in-line-monitoring/summary_quality_revised.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-llms",
    #     "outputFile": "summary_quality_revised-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-llms/llm-monitoring/llm-listener-monitoring/llm-monitoring-orchestration-setup.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-llms",
    #     "outputFile": "llm-monitoring-orchestration-setup.md"
    # },
    ### RAG LLM Orchestration
    # {
    #     "inputFile": "wallaroo-llms/vector-database-embedding-with-ml-orchestrations/Batch_Embedding_Computation.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-llms",
    #     "outputFile": "Batch_Embedding_Computation-reference.md"
    # },
    ### Development
    # {
    #     "inputFile": "development/sdk-install-guides/standard-install/install-wallaroo-sdk-standard-guide.ipynb",
    #     "outputDir": "/wallaroo-sdk-guides",
    #     "outputFile": "install-wallaroo-sdk-standard-guide-reference.md"
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
    document = re.sub('!\[png\]\(', f'![png](/images/2024.2{outputdir}/', document)
    document = re.sub('\(./images', '(/images/2024.2', document)
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
        convert_cmd = f'jupyter nbconvert --to markdown --output-dir {docs_directory}{currentFile["outputDir"]} --output {currentFile["outputFile"]} {currentFile["inputFile"]}'
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