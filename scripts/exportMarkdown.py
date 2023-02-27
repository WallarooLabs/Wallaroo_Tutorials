#!/usr/bin/env python

"""
Converts the following Python notebooks into the same format used for the Wallaroo Documentation site.

This uses the jupyter nbconvert command.  For now this will always assume we're exporting to markdown:

    jupyter nbconvert {file} --to markdown --output {output}

"""

import os
import nbformat
from traitlets.config import Config
#import argparse

c = Config()

c.NbConvertApp.export_format = "markdown"

fileList = [
    {
        "inputFile": "wallaroo-model-cookbooks/aloha/aloha_demo.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials",
        "outputFile": "wallaroo-quick-start-aloha-reference.md"
    },
    {
        "inputFile": "wallaroo-model-cookbooks/demand_curve/demandcurve_demo.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials",
        "outputFile": "wallaroo-quick-start-demandcurve-reference.md"
    },
    {
        "inputFile": "wallaroo-model-cookbooks/imdb/imdb_sample.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials",
        "outputFile": "wallaroo-quick-start-imdb-reference.md"
    },
    {
        "inputFile": "wallaroo-features/model_insights/model-insights.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials/wallaroo-tutorial-features",
        "outputFile": "wallaroo-model-insights-reference.md"
    },
    {
        "inputFile": "wallaroo-testing-tutorials/shadow_deploy/shadow_deployment_tutorial.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials",
        "outputFile": "wallaroo-shadow-deployment-tutorial-reference.md"
    },
    {
        "inputFile": "wallaroo-101/Wallaroo-101.ipynb",
        "outputDir": "docs/markdown/wallaroo-101/",
        "outputFile": "wallaroo-101-reference.md"
    },
    {
        "inputFile": "model_conversion/autoconversion-tutorial/auto-convert-tutorial.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials/conversion-tutorials",
        "outputFile": "sklearn-auto-conversion-reference.md"
    },
    {
        "inputFile": "model_conversion/keras-to-onnx/autoconvert-keras-tutorial.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials/conversion-tutorials",
        "outputFile": "keras-auto-conversion-reference.md"
    },
    {
        "inputFile": "model_conversion/pytorch-to-onnx/pytorch-to-onnx.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials/conversion-tutorials",
        "outputFile": "pytorch-to-onnx-reference.md"
    },
    {
        "inputFile": "model_conversion/sklearn-classification-to-onnx/convert-sklearn-classification-to-onnx.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials/conversion-tutorials",
        "outputFile": "sklearn-logistic-to-onnx-reference.md"
    },
    {
        "inputFile": "model_conversion/sklearn-regression-to-onnx/convert-sklearn-regression-to-onnx.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials/conversion-tutorials",
        "outputFile": "sklearn-regression-to-onnx-reference.md"
    },
    {
        "inputFile": "model_conversion/statsmodels/convert-statsmodel-tutorial.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials/conversion-tutorials",
        "outputFile": "statsmodel-conversion-reference.md"
    },
    {
        "inputFile": "model_conversion/xgboost-autoconversion/xgboost-autoconversion-classification-tutorial.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials/conversion-tutorials",
        "outputFile": "xgboost-autoconversion-classification-tutorial-reference.md"
    },
    {
        "inputFile": "model_conversion/xgboost-autoconversion/xgboost-autoconversion-regression-tutorial.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials/conversion-tutorials",
        "outputFile": "xgboost-autoconversion-regression-tutorial-reference.md"
    },
    {
        "inputFile": "wallaroo-features/simulated_edge/simulated_edge.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials",
        "outputFile": "wallaroo-simulated_edge-reference.md"
    },
    {
        "inputFile": "notebooks_in_prod/00_notebooks_in_prod_Introduction.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials/notebook_in_prod",
        "outputFile": "_index.md"
    },
    {
        "inputFile": "notebooks_in_prod/01_notebooks_in_prod_explore_and_train.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials/notebook_in_prod",
        "outputFile": "01_notebooks_in_prod_explore_and_train-reference.md"
    },
    {
        "inputFile": "notebooks_in_prod/02_notebooks_in_prod_automated_training_process.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials/notebook_in_prod",
        "outputFile": "02_notebooks_in_prod_automated_training_process-reference.md"
    },
    {
        "inputFile": "notebooks_in_prod/03_notebooks_in_prod_deploy_model.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials/notebook_in_prod",
        "outputFile": "03_notebooks_in_prod_deploy_model-reference.md"
    },
    {
        "inputFile": "notebooks_in_prod/04_notebooks_in_prod_regular_batch_inferences.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials/notebook_in_prod",
        "outputFile": "04_notebooks_in_prod_regular_batch_inferences-reference.md"
    },
    {
        "inputFile": "wallaroo-testing-tutorials/abtesting/wallaroo-abtesting-tutorial.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials",
        "outputFile": "wallaroo-abtesting-tutorial-reference.md"
    },
    {
        "inputFile": "wallaroo-testing-tutorials/anomaly_detection/wallaroo-anomaly-detection.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials",
        "outputFile": "wallaroo-anomaly-detection-reference.md"
    },
    {
        "inputFile": "model_conversion/xgboost-to-onnx/xgboost-to-onnx.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials/conversion-tutorials",
        "outputFile": "xgboost-to-onnx-reference.md"
    },
    {
        "inputFile": "wallaroo-features/wallaroo-model-endpoints/wallaroo-model-endpoints-api-tutorial.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials/wallaroo-model-endpoints",
        "outputFile": "wallaroo-model-endpoints-api-tutorial-reference.md"
    },
    {
        "inputFile": "wallaroo-features/wallaroo-model-endpoints/wallaroo-model-endpoints-setup.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials/wallaroo-model-endpoints",
        "outputFile": "wallaroo-model-endpoints-setup-reference.md"
    },
    {
        "inputFile": "development/sdk-install-guides/google-vertex-sdk-install/install-wallaroo-sdk-google-vertex-guide.ipynb",
        "outputDir": "docs/markdown/wallaroo-developer-guides/wallaroo-sdk-guides/",
        "outputFile": "install-wallaroo-sdk-google-vertex-guide-reference.md"
    },
    {
        "inputFile": "development/sdk-install-guides/standard-install/install-wallaroo-sdk-standard-guide.ipynb",
        "outputDir": "docs/markdown/wallaroo-developer-guides/wallaroo-sdk-guides/",
        "outputFile": "install-wallaroo-sdk-standard-guide-reference.md"
    },
    {
        "inputFile": "wallaroo-features/wallaroo-tag-management/wallaroo-tags-guide.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials/wallaroo-tutorial-features/",
        "outputFile": "wallaroo-tags-guide-reference.md"
    },
    {
        "inputFile": "development/mlops_api/Wallaroo-MLOps-Tutorial.ipynb",
        "outputDir": "docs/markdown/wallaroo-developer-guides/wallaroo-api-guides/",
        "outputFile": "wallaroo-mlops-tutorial-reference.md"
    },
    {
        "inputFile": "development/sdk-install-guides/azure-ml-sdk-install/install-wallaroo-sdk-azureml-guide.ipynb",
        "outputDir": "docs/markdown/wallaroo-developer-guides/wallaroo-sdk-guides/",
        "outputFile": "install-wallaroo-sdk-azureml-guide-reference.md"
    },
    {
        "inputFile": "development/sdk-install-guides/aws-sagemaker-install/install-wallaroo-aws-sagemaker-guide.ipynb",
        "outputDir": "docs/markdown/wallaroo-developer-guides/wallaroo-sdk-guides/",
        "outputFile": "install-wallaroo-sdk-aws-sagemaker-guide-reference.md"
    },
    {
        "inputFile": "development/sdk-install-guides/databricks-azure-sdk-install/install-wallaroo-sdk-databricks-azure-guide.ipynb",
        "outputDir": "docs/markdown/wallaroo-developer-guides/wallaroo-sdk-guides/",
        "outputFile": "install-wallaroo-sdk-databricks-azure-guide-reference.md"
    },
    {
        "inputFile": "wallaroo-model-cookbooks/mlflow-tutorial/wallaroo-mlflow-tutorial.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials",
        "outputFile": "wallaroo-mlflow-tutorial-reference.md"
    },
    {
        "inputFile": "wallaroo-features/model_hot_swap/wallaroo_hot_swap_tutorial.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials/wallaroo-tutorial-features",
        "outputFile": "wallaroo-hot-swap-models-reference.md"
    },
    {
        "inputFile": "wallaroo-model-cookbooks/computer-vision/00_computer_vision_tutorial_intro.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials/computer-vision",
        "outputFile": "00-computer-vision-tutorial-reference.md"
    },
    {
        "inputFile": "wallaroo-model-cookbooks/computer-vision/01_computer_vision_tutorial_mobilenet.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials/computer-vision",
        "outputFile": "01_computer_vision_tutorial_mobilenet-reference.md"
    },
    {
        "inputFile": "wallaroo-model-cookbooks/computer-vision/02_computer_vision_tutorial_resnet50.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials/computer-vision",
        "outputFile": "02_computer_vision_tutorial_resnet50-reference.md"
    },
    {
        "inputFile": "wallaroo-model-cookbooks/computer-vision/03_computer_vision_tutorial_shadow_deploy.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials/computer-vision",
        "outputFile": "03_computer_vision_tutorial_shadow_deploy-reference.md"
    }
]

def main():
    for currentFile in fileList:
        convert_cmd = f'jupyter nbconvert --config ./config/exportconfig.py --to markdown --output-dir {currentFile["outputDir"]} --output {currentFile["outputFile"]} {currentFile["inputFile"]}'
        #print(convert_cmd)
        os.system(convert_cmd)

if __name__ == '__main__':
    main()