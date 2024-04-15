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
    # @TODO: UNPUBLISHED
    # {
    #     "inputFile": "wallaroo-features/parallel-inference-aloha-tutorial/parallel-infer-with-aloha.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorial-features",
    #     "outputFile": "parallel-infer-with-aloha-reference.md"
    # },
    # wallaroo 101
    {
        "inputFile": "wallaroo-101/Wallaroo-101.ipynb",
        "outputDir": "/wallaroo-101/",
        "outputFile": "wallaroo-101-reference.md"
    },
        ## deploy and serve
    ## model registry
    {
        "inputFile": "wallaroo-model-deploy-and-serve/mlflow-registries-upload-tutorials/Wallaroo-model-registry-demonstration.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/model-registry",
        "outputFile": "Wallaroo-model-registry-demonstration-reference.md"
    },
    ## keras
    {
        "inputFile": "wallaroo-model-deploy-and-serve/keras-upload-tutorials/wallaroo-upload-keras_sequential_model_single_io.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/keras",
        "outputFile": "wallaroo-upload-keras_sequential_model_single_io-reference.md"
    },
    ## hugging face
    {
        "inputFile": "wallaroo-model-deploy-and-serve/hugging-face-upload-tutorials/wallaroo-api-upload-hf-zero_shot_classification.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/hugging-face",
        "outputFile": "wallaroo-api-upload-hf-zero_shot_classification.md"
    },
    {
        "inputFile": "wallaroo-model-deploy-and-serve/hugging-face-upload-tutorials/wallaroo-sdk-upload-hf-zero_shot_classification.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/hugging-face",
        "outputFile": "wallaroo-sdk-upload-hf-zero_shot_classification.md"
    },
    ## computer vision
    {
        "inputFile": "wallaroo-model-deploy-and-serve/computer-vision/00_computer_vision_tutorial_intro.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/computer-vision",
        "outputFile": "00-computer-vision-tutorial-reference.md"
    },
    {
        "inputFile": "wallaroo-model-deploy-and-serve/computer-vision/01_computer_vision_tutorial_mobilenet.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/computer-vision",
        "outputFile": "01_computer_vision_tutorial_mobilenet-reference.md"
    },
    {
        "inputFile": "wallaroo-model-deploy-and-serve/computer-vision/02_computer_vision_tutorial_resnet50.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/computer-vision",
        "outputFile": "02_computer_vision_tutorial_resnet50-reference.md"
    },
    {
        "inputFile": "wallaroo-model-deploy-and-serve/computer-vision/03_computer_vision_tutorial_shadow_deploy.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/computer-vision",
        "outputFile": "03_computer_vision_tutorial_shadow_deploy-reference.md"
    },
    ## BYOP
    {
        "inputFile": "wallaroo-model-deploy-and-serve/arbitrary-python-upload-tutorials/00_wallaroo-upload-arbitrary-python-vgg16-model-generation.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/arbitrary-python",
        "outputFile": "00_wallaroo-upload-arbitrary-python-vgg16-model-generation-reference.md"
    },
    {
        "inputFile": "wallaroo-model-deploy-and-serve/arbitrary-python-upload-tutorials/01_wallaroo-upload-arbitrary-python-vgg16-model-deployment.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/arbitrary-python",
        "outputFile": "01_wallaroo-upload-arbitrary-python-vgg16-model-deployment-reference.md"
    },
    ## Python steps
    {
        "inputFile": "wallaroo-model-deploy-and-serve/python-upload-tutorials/python-step-dataframe-output-logging-example-sdk.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve",
        "outputFile": "python-step-dataframe-output-logging-example-sdk-reference.md"
    },
    ## notebooks in prod
    {
        "inputFile": "wallaroo-model-deploy-and-serve/notebooks_in_prod/00_notebooks_in_prod_introduction.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/notebook_in_prod",
        "outputFile": "_index.md"
    },
    {
        "inputFile": "wallaroo-model-deploy-and-serve/notebooks_in_prod/01_notebooks_in_prod_explore_and_train.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/notebook_in_prod",
        "outputFile": "01_notebooks_in_prod_explore_and_train-reference.md"
    },
    {
        "inputFile": "wallaroo-model-deploy-and-serve/notebooks_in_prod/02_notebooks_in_prod_automated_training_process.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/notebook_in_prod",
        "outputFile": "02_notebooks_in_prod_automated_training_process-reference.md"
    },
    {
        "inputFile": "wallaroo-model-deploy-and-serve/notebooks_in_prod/03_notebooks_in_prod_deploy_model_python.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/notebook_in_prod",
        "outputFile": "03_notebooks_in_prod_deploy_model-reference.md"
    },
    {
        "inputFile": "wallaroo-model-deploy-and-serve/notebooks_in_prod/04_notebooks_in_prod_regular_batch_inferences.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/notebook_in_prod",
        "outputFile": "04_notebooks_in_prod_regular_batch_inferences-reference.md"
    },
    ## pytorch
    {
        "inputFile": "wallaroo-model-deploy-and-serve/pytorch-upload-tutorials/wallaroo-upload-pytorch-multi-input-output.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/pytorch",
        "outputFile": "wallaroo-upload-pytorch-multi-input-output-reference.md"
    },
    {
        "inputFile": "wallaroo-model-deploy-and-serve/pytorch-upload-tutorials/wallaroo-upload-pytorch-single-input-output.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/pytorch",
        "outputFile": "wallaroo-upload-pytorch-single-input-output-reference.md"
    },
    ## tensorflow
    {
        "inputFile": "wallaroo-model-deploy-and-serve/tensorflow-upload-tutorials/wallaroo-upload-tensorflow.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/tensorflow",
        "outputFile": "wallaroo-upload-tensorflow-reference.md"
    },

    ## xgboost
    {
        "inputFile": "wallaroo-model-deploy-and-serve/xgboost-upload-tutorials/wallaroo-sdk-upload-booster-binary-classification-conversion.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/xgboost-upload-tutorials",
        "outputFile": "wallaroo-sdk-upload-booster-binary-classification-conversion-reference.md"
    },
    {
        "inputFile": "wallaroo-model-deploy-and-serve/xgboost-upload-tutorials/wallaroo-sdk-upload-booster-multi-classification-softmax-conversion.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/xgboost-upload-tutorials",
        "outputFile": "wallaroo-sdk-upload-booster-multi-classification-softmax-conversion-reference.md"
    },
    {
        "inputFile": "wallaroo-model-deploy-and-serve/xgboost-upload-tutorials/wallaroo-sdk-upload-booster-multi-classification-softprob-conversion.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/xgboost-upload-tutorials",
        "outputFile": "wallaroo-sdk-upload-booster-multi-classification-softprob-conversion-reference.md"
    },
    {
        "inputFile": "wallaroo-model-deploy-and-serve/xgboost-upload-tutorials/wallaroo-sdk-upload-booster-regression-conversion.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/xgboost-upload-tutorials",
        "outputFile": "wallaroo-sdk-upload-booster-regression-conversion-reference.md"
    },
    {
        "inputFile": "wallaroo-model-deploy-and-serve/xgboost-upload-tutorials/wallaroo-sdk-upload-booster-rf-classification-conversion.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/xgboost-upload-tutorials",
        "outputFile": "wallaroo-sdk-upload-booster-rf-classification-conversion-reference.md"
    },
    {
        "inputFile": "wallaroo-model-deploy-and-serve/xgboost-upload-tutorials/wallaroo-sdk-upload-booster-rf-regression-conversion.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/xgboost-upload-tutorials",
        "outputFile": "wallaroo-sdk-upload-booster-rf-regression-conversion-reference.md"
    },
    {
        "inputFile": "wallaroo-model-deploy-and-serve/xgboost-upload-tutorials/wallaroo-sdk-upload-xbg-classification.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/xgboost-upload-tutorials",
        "outputFile": "wallaroo-sdk-upload-xbg-classification-reference.md"
    },
    {
        "inputFile": "wallaroo-model-deploy-and-serve/xgboost-upload-tutorials/wallaroo-sdk-upload-xbg-regressor.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/xgboost-upload-tutorials",
        "outputFile": "wallaroo-sdk-upload-xbg-regressor-reference.md"
    },
    {
        "inputFile": "wallaroo-model-deploy-and-serve/xgboost-upload-tutorials/wallaroo-sdk-upload-xbg-rf-classification.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/xgboost-upload-tutorials",
        "outputFile": "wallaroo-sdk-upload-xbg-rf-classification-reference.md"
    },
    {
        "inputFile": "wallaroo-model-deploy-and-serve/xgboost-upload-tutorials/wallaroo-sdk-upload-xbg-rf-regressor.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/xgboost-upload-tutorials",
        "outputFile": "wallaroo-sdk-upload-xbg-rf-regressor-reference.md"
    },
    ## sklearn
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
    ### multiple replicas forecast updates
    {
        "inputFile": "wallaroo-model-deploy-and-serve/pipeline_multiple_replicas_forecast_tutorial/00_multiple_replicas_forecast.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/statsmodel/",
        "outputFile": "00_multiple_replicas_forecast-reference.md"
    },
    {
        "inputFile": "wallaroo-model-deploy-and-serve/pipeline_multiple_replicas_forecast_tutorial/01_multiple_replicas_forecast.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/statsmodel/",
        "outputFile": "01_multiple_replicas_forecast-reference.md"
    },
    {
        "inputFile": "wallaroo-model-deploy-and-serve/pipeline_multiple_replicas_forecast_tutorial/02_multiple_replicas_forecast.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/statsmodel/",
        "outputFile": "02_multiple_replicas_forecast-reference.md"
    },
    ## aloha main
    {
        "inputFile": "wallaroo-model-deploy-and-serve/aloha/aloha_demo.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/",
        "outputFile": "aloha_demo.ipynb-reference.md"
    },
    ## computer vision mitochondria
    {
        "inputFile": "wallaroo-model-deploy-and-serve/computer-vision-mitochondria-imaging/00_computer-vision-mitochondria-imaging-example.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/cv-mitochondria-imaging",
        "outputFile": "00_computer-vision-mitochondria-imaging-example-reference.md"
    },
    {
        "inputFile": "wallaroo-model-deploy-and-serve/computer-vision-mitochondria-imaging/01_computer-vision-mitochondria-imaging-example.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/cv-mitochondria-imaging",
        "outputFile": "01_computer-vision-mitochondria-imaging-example-reference.md"
    },
    {
        "inputFile": "wallaroo-model-deploy-and-serve/computer-vision-mitochondria-imaging/02_computer-vision-mitochondria-imaging-example.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve/cv-mitochondria-imaging",
        "outputFile": "02_computer-vision-mitochondria-imaging-example-reference.md"
    },
    ## cv yolov8
    {
        "inputFile": "wallaroo-model-deploy-and-serve/computer-vision-yolov8/computer-vision-yolov8-demonstration.ipynb",
        "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve",
        "outputFile": "computer-vision-yolov8-demonstration-reference.md"
    },
    ## demand curve
    ## observe tutorials
    


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
    document = re.sub('!\[png\]\(', f'![png](/images/2024.1{outputdir}/', document)
    document = re.sub('\(./images', '(/images/2024.1', document)
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

    # remove edge bundle
    # obfuscate databricks url
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
        print(f"cp -rf ./{source_directory}/{reference} {target_directory}")
        # print(f"To: {target_directory}/{reference}")
        os.system(f"cp -rf ./{source_directory}/{reference} {target_directory}")

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

if __name__ == '__main__':
    main()