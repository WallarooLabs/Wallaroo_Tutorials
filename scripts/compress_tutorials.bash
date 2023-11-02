#!/opt/homebrew/bin/bash

# This relies on Bash 4 and above
# Meant to be run from the root of this project folder

declare -A tutorials

tutorials=(
    ["aws-sagemaker-install-guide"]="development/sdk-install-guides/aws-sagemaker-install"
    ["azure-ml-sdk-install-guide"]="development/sdk-install-guides/azure-ml-sdk-install"
    ["databricks-azure-sdk-install-guide"]="development/sdk-install-guides/databricks-azure-sdk-install"
    ["google-vertex-sdk-install"]="development/sdk-install-guides/google-vertex-sdk-install"
    ["wallaroo-sdk-standard-install-guide"]="development/sdk-install-guides/standard-install"
    ["wallaroo-mlops-api-guide"]="development/mlops_api"
    ["BYOP-arbitary-python-model-upload-tutorials"]="model_uploads/arbitrary-python-upload-tutorials"
    ["hugging-face-upload-tutorials"]="model_uploads/hugging-face-upload-tutorials"
    ["keras-upload-tutorials"]="model_uploads/keras-upload-tutorials"
    ["mlflow-registries-upload-tutorials"]"model_uploads/mlflow-registries-upload-tutorials"
    ["python-step-upload-tutorials"]="model_uploads/python-upload-tutorials]"
    ["pytorch-upload-tutorials"]="model_uploads/pytorch-upload-tutorials"
    ["sklearn-upload-tutorials"]="model_uploads/sklearn-upload-tutorials"
    ["tensorflow-upload-tutorials"]="model_uploads/tensorflow-upload-tutorials"
    ["xgboost-upload-tutorials"]="model_uploads/xgboost-upload-tutorials"
    ["notebooks-in-production-guides"]="notebooks_in_prod"
    ["wallaroo-arm-byop-vgg16"]="pipeline-architecture/wallaroo-arm-byop-vgg16"
    ["wallaroo-arm-classification-cybersecurity"]="pipeline-architecture/wallaroo-arm-classification-cybersecurity"
    ["wallaroo-arm-classification-finserv"]="pipeline-architecture/wallaroo-arm-classification-finserv"
    ["wallaroo-arm-computer-vision-yolov8"]="pipeline-architecture/wallaroo-arm-computer-vision-yolov8"
    ["wallaroo-arm-cv-arrow"]="pipeline-architecture/wallaroo-arm-cv-arrow"
    ["wallaroo-arm-llm-summarization"]="pipeline-architecture/wallaroo-arm-llm-summarization"
    ["edge-arbitrary-python"]="pipeline-edge-publish/edge-arbitrary-python"
    ["edge-classification-cybersecurity"]="pipeline-edge-publish/edge-classification-cybersecurity"
    ["edge-classification-finserv"]="pipeline-edge-publish/edge-classification-finserv"
    ["edge-classification-finserv-api"]="pipeline-edge-publish/edge-classification-finserv-api"
    ["edge-computer-vision-yolov8"]="pipeline-edge-publish/edge-computer-vision-yolov8"
    ["edge-computer-vision"]="pipeline-edge-publish/edge-cv"
    ["edge-cv-healthcare-images"]="pipeline-edge-publish/edge-cv-healthcare-images"
    ["edge-llm-summarization"]="pipeline-edge-publish/edge-llm-summarization"
    ["edge-observability-assays"]="pipeline-edge-publish/edge-observability-assays"
    ["edge-observability-classification-finserv"]="pipeline-edge-publish/edge-observability-classification-finserv"
    ["edge-observability-classification-finserv-api"]="pipeline-edge-publish/edge-observability-classification-finserv-api"
    ["wallaroo-101"]="wallaroo-101"
    ["assay-model-insights"]="wallaroo-features/assay-model-insights"
    ["gpu-deployment"]="wallaroo-features/gpu-deployment"
    ["model_hot_swap"]="wallaroo-features/model_hot_swap"
    ["parallel-inference-aloha-tutorial"]="wallaroo-features/parallel-inference-aloha-tutorial"
    ["pipeline_api_log_tutorial"]="wallaroo-features/pipeline_api_log_tutorial"
    ["pipeline_log_tutorial"]="wallaroo-features/pipeline_log_tutorial"
    ["pipeline_multiple_replicas_forecast_tutorial"]="wallaroo-features/pipeline_multiple_replicas_forecast_tutorial"
    ["wallaroo-model-endpoints-guide"]="wallaroo-features/wallaroo-model-endpoints"
    ["wallaroo-tag-management-guide"]="wallaroo-features/wallaroo-tag-management"
    ["wallaroo-inference-server-cv-frcnn"]="wallaroo-inference-server-tutorials/wallaroo-inference-server-cv-frcnn"
    ["wallaroo-inference-server-cv-resnet"]="wallaroo-inference-server-tutorials/wallaroo-inference-server-cv-resnet"
    ["wallaroo-inference-server-cv-yolov8"]="wallaroo-inference-server-tutorials/wallaroo-inference-server-cv-yolov8"
    ["wallaroo-inference-server-hf-summarizer"]="wallaroo-inference-server-tutorials/wallaroo-inference-server-hf-summarizer"
    ["wallaroo-inference-server-llama2"]="wallaroo-inference-server-tutorials/wallaroo-inference-server-llama2"
    ["wallaroo-model-cookbook-aloha"]="wallaroo-model-cookbooks/aloha"
    ["wallaroo-model-cookbooks-computer-vision"]="wallaroo-model-cookbooks/computer-vision"
    ["wallaroo-model-cookbooks-computer-vision-mitochondria-imaging"]="wallaroo-model-cookbooks/computer-vision-mitochondria-imaging"
    ["wallaroo-model-cookbooks-computer-vision-yolov8"]="wallaroo-model-cookbooks/computer-vision-yolov8"
    ["wallaroo-model-cookbooks-demand_curve"]="wallaroo-model-cookbooks/demand_curve"
    ["wallaroo-model-cookbooks-imdb"]="wallaroo-model-cookbooks/imdb"
    ["wallaroo-model-cookbooks-mlflow-tutorial"]="wallaroo-model-cookbooks/mlflow-tutorial"
    ["wallaroo-testing-tutorials-abtesting"]="wallaroo-testing-tutorials/abtesting"
    ["wallaroo-testing-tutorials-anomaly_detection"]="wallaroo-testing-tutorials/anomaly_detection"
    ["wallaroo-testing-tutorials-houseprice-saga]="wallaroo-testing-tutorials/houseprice-saga"
    ["wallaroo-testing-tutorials-shadow_deploy]="wallaroo-testing-tutorials/shadow_deploy"
    ["connection_api_bigquery_tutorial"]="workload-orchestrations/connection_api_bigquery_tutorial"
    ["orchestration_api_simple_tutorial"]="workload-orchestrations/orchestration_api_simple_tutorial"
    ["orchestration_sdk_bigquery_houseprice_tutorial"]="workload-orchestrations/orchestration_sdk_bigquery_houseprice_tutorial"
    ["orchestration_sdk_bigquery_statsmodel_tutorial"]="workload-orchestrations/orchestration_sdk_bigquery_statsmodel_tutorial"
    ["orchestration_sdk_comprehensive_tutorial"]="workload-orchestrations/orchestration_sdk_comprehensive_tutorial"
    ["orchestration_sdk_simple_tutorial"]="workload-orchestrations/orchestration_sdk_simple_tutorial"
    )

currentDirectory=$PWD

# for zip in "${!tutorials[@]}"; 
#     do (cd ${tutorials[$zip]}/..;zip -r $currentDirectory/compress_tutorials/$zip.zip $zip);
# done

for zip in ${!tutorials[@]}; do
    #echo cd ${tutorials[${zip}]}/..;
    zip -r ./compress_tutorials/${zip}.zip ${tutorials[${zip}]};
done