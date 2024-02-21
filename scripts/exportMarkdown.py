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
    # wallaroo-observe-tutorials
    # {
    #     "inputFile": "wallaroo-observe-tutorials/model-observability-anomaly-detection-ccfraud-sdk-tutorial/model-observability-anomaly-detection-ccfraud-sdk-tutorial.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-observability",
    #     "outputFile": "model-observability-anomaly-detection-ccfraud-sdk-tutorial-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-observe-tutorials/model-observability-anomaly-detection-houseprice-sdk-tutorial/model-observability-anomaly-detection-house-price-sdk-tutorial.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-observability",
    #     "outputFile": "model-observability-anomaly-detection-house-price-sdk-tutorial-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-observe-tutorials/pipeline-log-tutorial/pipeline_log_tutorial.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-observability",
    #     "outputFile": "pipeline_log_tutorial-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-observe-tutorials/wallaro-model-observability-assays/wallaroo_model_observability_assays.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-tutorials-observability",
    #     "outputFile": "wallaroo_model_observability_assays-reference.md"
    # },
    # # wallaroo-model-upload-tutorials
    # {
    #     "inputFile": "wallaroo-model-upload-tutorials/aloha-upload-deploy-tutorial/wallaroo-upload-aloha-tutorial.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-upload-tutorials",
    #     "outputFile": "wallaroo-upload-aloha-tutorial-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-upload-tutorials/arbitrary-python-upload-tutorials/01_wallaroo-upload-arbitrary-python-vgg16-model-deployment.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-upload-tutorials",
    #     "outputFile": "01_wallaroo-upload-arbitrary-python-vgg16-model-deployment-reference.md"
    # },
    # {
    #     "inputFile": "wallaroo-model-upload-tutorials/arbitrary-python-upload-tutorials/00_wallaroo-upload-arbitrary-python-vgg16-model-generation.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-upload-tutorials",
    #     "outputFile": "00_wallaroo-upload-arbitrary-python-vgg16-model-generation-reference.md"
    # },
    # ## deploy and serve tutorials
    # {
    #     "inputFile": "wallaroo-model-deploy-and-serve/unet-brain-segmentation-demonstration/unet_demonstration.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-model-deploy-and-serve",
    #     "outputFile": "unet_demonstration-reference.md"
    # },
    # # wallaroo-run-anywhere-tutorials
    # {
    #     "inputFile": "wallaroo-run-anywhere-tutorials/edge-unet-brain-segmentation-demonstration/unet-run-anywhere-demonstration.ipynb",
    #     "outputDir": "/wallaroo-tutorials/wallaroo-run-anywhere-tutorials",
    #     "outputFile": "unet-run-anywhere-demonstration-reference.md"
    # },
    # wallaroo free tutorials
    # wallaroo inference server tutorials
    {
        "inputFile": "wallaroo-free-tutorials/wallaroo-inference-server-tutorials/wallaroo-inference-server-cv-unet/wallaroo-inference-server-cv-unet.ipynb",
        "outputDir": "/wallaroo-free-tutorials/wallaroo-inference-server-tutorials",
        "outputFile": "wallaroo-inference-server-cv-unet-reference.md"
    },
    {
        "inputFile": "wallaroo-free-tutorials/wallaroo-inference-server-tutorials/wallaroo-inference-server-cv-frcnn/wallaroo-inference-server-cv-frcnn.ipynb",
        "outputDir": "/wallaroo-free-tutorials/wallaroo-inference-server-tutorials",
        "outputFile": "wallaroo-inference-server-cv-frcnn-reference.md"
    },
    {
        "inputFile": "wallaroo-free-tutorials/wallaroo-inference-server-tutorials/wallaroo-inference-server-cv-resnet/wallaroo-inference-server-cv-resnet.ipynb",
        "outputDir": "/wallaroo-free-tutorials/wallaroo-inference-server-tutorials",
        "outputFile": "wallaroo-inference-server-cv-resnet-reference.md"
    },
    {
        "inputFile": "wallaroo-free-tutorials/wallaroo-inference-server-tutorials/wallaroo-inference-server-cv-yolov8/wallaroo-inference-server-cv-yolov8.ipynb",
        "outputDir": "/wallaroo-free-tutorials/wallaroo-inference-server-tutorials",
        "outputFile": "wallaroo-inference-server-cv-yolov8-reference.md"
    },
    {
        "inputFile": "wallaroo-free-tutorials/wallaroo-inference-server-tutorials/wallaroo-inference-server-hf-summarizer/wallaroo-inference-server-hf-summarization.ipynb",
        "outputDir": "/wallaroo-free-tutorials/wallaroo-inference-server-tutorials",
        "outputFile": "wallaroo-inference-server-hf-summarization-reference.md"
    },
    {
        "inputFile": "wallaroo-free-tutorials/wallaroo-inference-server-tutorials/wallaroo-inference-server-llama2/wallaroo-inference-server-llama2.ipynb",
        "outputDir": "/wallaroo-free-tutorials/wallaroo-inference-server-tutorials",
        "outputFile": "wallaroo-inference-server-llama2-reference.md"
    },



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
    document = re.sub('!\[png\]\(', f'![png](/images/2023.4.1{outputdir}/', document)
    document = re.sub('\(./images', '(/images/2023.4.1', document)
    # move them all to Docsy figures
    document = re.sub(r'!\[(.*?)\]\((.*?)\)', r'{{<figure src="\2" width="800" label="\1">}}', document)

    # remove gib
    document = re.sub('gib.bhojraj@wallaroo.ai	', 
                      'sample.user@wallaroo.ai', 
                      document)
    # fix github link for final release
    document = re.sub('https://github.com/WallarooLabs/Wallaroo_Tutorials/blob/20240208_anomaly_detection/', 
                      'https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/', 
                      document)
    
     # obfuscate databricks url
    document = re.sub('https://adb-5939996465837398.18.azuredatabricks.net', 
                      'https://sample.registry.service.azuredatabricks.net', 
                      document)

    # change images to root
    document = re.sub('"images/', 
                      '"/images/', 
                      document)

    # remove edge bundle
    # obfuscate databricks url
    document = re.sub("'EDGE_BUNDLE': '.*?'", 
                      "'EDGE_BUNDLE': 'abcde'", 
                      document)
   # document = re.sub('![png](', 'bob', document)

    # strip the excess newlines - match any pattern of newline plus another one or more empty newlines
    document = re.sub(r'\n[\n]+', r'\n\n', document)

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