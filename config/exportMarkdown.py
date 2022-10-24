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
        "inputFile": "aloha/aloha_demo.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials",
        "outputFile": "wallaroo-quick-start-aloha.md"
    },
    {
        "inputFile": "demand_curve/demandcurve_demo.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials",
        "outputFile": "wallaroo-quick-start-demandcurve.md"
    },
    {
        "inputFile": "imdb/imdb_sample.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials",
        "outputFile": "wallaroo-quick-start-imdb.md"
    },
    {
        "inputFile": "model_insights/model-insights.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials/wallaroo-tutorial-features",
        "outputFile": "wallaroo-model-insights.md"
    },
    {
        "inputFile": "shadow_deploy/shadow_deployment_tutorial.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials",
        "outputFile": "wallaroo-shadow-deployment-tutorial.md"
    },
    {
        "inputFile": "wallaroo-101/Wallaroo-101.ipynb",
        "outputDir": "docs/markdown/wallaroo-101",
        "outputFile": "_index.md"
    },
    {
        "inputFile": "model_conversion/autoconversion-tutorial/auto-convert-tutorial.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials/conversion-tutorials",
        "outputFile": "sklearn-auto-conversion.md"
    },
    {
        "inputFile": "model_conversion/keras-to-onnx/autoconvert-keras-tutorial.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials/conversion-tutorials",
        "outputFile": "keras-auto-conversion.md"
    },
    {
        "inputFile": "model_conversion/pytorch-to-onnx/pytorch-to-onnx.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials/conversion-tutorials",
        "outputFile": "pytorch-to-onnx.md"
    },
    {
        "inputFile": "model_conversion/sklearn-classification-to-onnx/convert-sklearn-classification-to-onnx.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials/conversion-tutorials",
        "outputFile": "sklearn-logistic-to-onnx.md"
    },
    {
        "inputFile": "model_conversion/sklearn-regression-to-onnx/convert-sklearn-regression-to-onnx.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials/conversion-tutorials",
        "outputFile": "sklearn-regression-to-onnx.md"
    },
    {
        "inputFile": "model_conversion/statsmodels/convert-statsmodel-tutorial.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials/conversion-tutorials",
        "outputFile": "statsmodel-conversion.md"
    },
    {
        "inputFile": "model_conversion/xgboost-autoconversion/xgboost-autoconversion-classification-tutorial.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials/conversion-tutorials",
        "outputFile": "xgboost-autoconversion-classification-tutorial.md"
    },
    {
        "inputFile": "model_conversion/xgboost-autoconversion/xgboost-autoconversion-regression-tutorial.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials/conversion-tutorials",
        "outputFile": "xgboost-autoconversion-regression-tutorial.md"
    },
    {
        "inputFile": "simulated_edge/simulated_edge.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials",
        "outputFile": "wallaroo-simulated_edge.md"
    },
    {
        "inputFile": "notebooks_in_prod/00_notebooks_in_prod_Introduction.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials/notebook_in_prod",
        "outputFile": "_index.md"
    },
    {
        "inputFile": "notebooks_in_prod/01_notebooks_in_prod_explore_and_train.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials/notebook_in_prod",
        "outputFile": "01_notebooks_in_prod_explore_and_train.md"
    },
    {
        "inputFile": "notebooks_in_prod/02_notebooks_in_prod_automated_training_process.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials/notebook_in_prod",
        "outputFile": "02_notebooks_in_prod_automated_training_process.md"
    },
    {
        "inputFile": "notebooks_in_prod/03_notebooks_in_prod_deploy_model.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials/notebook_in_prod",
        "outputFile": "03_notebooks_in_prod_deploy_model.md"
    },
    {
        "inputFile": "notebooks_in_prod/04_notebooks_in_prod_regular_batch_inferences.ipynb",
        "outputDir": "docs/markdown/wallaroo-tutorials/notebook_in_prod",
        "outputFile": "04_notebooks_in_prod_regular_batch_inferences.md"
    }
]

def main():
    for currentFile in fileList:
        convert_cmd = f'jupyter nbconvert --config ./config/exportconfig.py --to markdown --output-dir {currentFile["outputDir"]} --output {currentFile["outputFile"]} {currentFile["inputFile"]}'
        # print(convert_cmd)
        os.system(convert_cmd)

if __name__ == '__main__':
    main()