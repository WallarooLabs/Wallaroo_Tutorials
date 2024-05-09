#!/opt/homebrew/bin/bash

# This relies on Bash 4 and above
# Meant to be run from the root of this project folder

declare -A tutorials

tutorials=(
    [aloha]="wallaroo-model-deploy-and-serve/aloha"
    [wallaroo-model-endpoints]="wallaroo-model-deploy-and-serve/wallaroo-model-endpoints"
    [wallaroo-tag-management]="wallaroo-model-deploy-and-serve/wallaroo-tag-management"
    [wallaroo-model-upload-deploy-byop-cv-tutorial]="wallaroo-model-deploy-and-serve/wallaroo-model-upload-deploy-byop-cv-tutorial"
    [pipeline_multiple_replicas_forecast_tutorial]="wallaroo-model-deploy-and-serve/pipeline_multiple_replicas_forecast_tutorial"
    [onnx-multi-input-demo]="wallaroo-model-deploy-and-serve/onnx-multi-input-demo"
    [mlflow-tutorial]="wallaroo-model-deploy-and-serve/mlflow-tutorial"
    [imdb]="wallaroo-model-deploy-and-serve/imdb"
    [hf-whisper]="wallaroo-model-deploy-and-serve/hf-whisper"
    [hf-clip-vit-base]="wallaroo-model-deploy-and-serve/hf-clip-vit-base"
    [demand_curve]="wallaroo-model-deploy-and-serve/demand_curve"
    [computer-vision-yolov8]="wallaroo-model-deploy-and-serve/computer-vision-yolov8"
    [computer-vision-mitochondria-imaging]="wallaroo-model-deploy-and-serve/computer-vision-mitochondria-imaging"
    [wallaroo-arm-arbitrary-python-vgg16-model-deployment]="pipeline-architecture/wallaroo-arm-byop-vgg16"
    [arm-classification-cybersecurity]="pipeline-architecture/wallaroo-arm-classification-cybersecurity"
    [arm-classification-finserv]="pipeline-architecture/wallaroo-arm-classification-finserv"
    [wallaroo-arm-computer-vision-yolov8]="pipeline-architecture/wallaroo-arm-computer-vision-yolov8"
    [wallaroo-arm-cv-arrow]="pipeline-architecture/wallaroo-arm-cv-arrow"
    [wallaroo-arm-llm-summarization]="pipeline-architecture/wallaroo-arm-llm-summarization"
    [wallaroo-gpu-llm-summarization]="pipeline-architecture/wallaroo-gpu-llm-summarization"
    [edge-arbitrary-python]="pipeline-edge-publish/edge-arbitrary-python"
    [edge-classification-cybersecurity]="pipeline-edge-publish/edge-classification-cybersecurity"
    [edge-classification-finserv]="pipeline-edge-publish/edge-classification-finserv"
    [edge-classification-finserv-api]="pipeline-edge-publish/edge-classification-finserv-api"
    [edge-computer-vision-yolov8]="pipeline-edge-publish/edge-computer-vision-yolov8"
    [edge-cv]="pipeline-edge-publish/edge-cv"
    [edge-cv-demo]="pipeline-edge-publish/edge-cv-demo"
    [edge-cv-healthcare-images]="pipeline-edge-publish/edge-cv-healthcare-images"
    [edge-llm-summarization]="pipeline-edge-publish/edge-llm-summarization"
    [edge-observability-assays]="pipeline-edge-publish/edge-observability-assays"
    [edge-observability-classification-finserv]="pipeline-edge-publish/edge-observability-classification-finserv"
    [edge-observability-classification-finserv-api]="pipeline-edge-publish/edge-observability-classification-finserv-api"
    [edge-observability-cv]="pipeline-edge-publish/edge-observability-cv"
    [edge-unet-brain-segmentation-demonstration]="pipeline-edge-publish/edge-unet-brain-segmentation-demonstration"
    [convert_wallaroo_data_to_pandas_arrow]="tools/convert_wallaroo_data_to_pandas_arrow"
    [wallaroo-101]="wallaroo-101"
    [gpu-deployment]="wallaroo-features/gpu-deployment"
    [model_hot_swap]="wallaroo-features/model_hot_swap"
    [onnx-multi-input-demo]="wallaroo-features/onnx-multi-input-demo"
    [pipeline_api_log_tutorial]="wallaroo-features/pipeline_api_log_tutorial"
    [pipeline_api_log_tutorial_cv]="wallaroo-features/pipeline_api_log_tutorial_cv"
    [pipeline_log_tutorial]="wallaroo-features/pipeline_log_tutorial"
    [wallaroo-model-endpoints]="wallaroo-features/wallaroo-model-endpoints"
    [wallaroo-tag-management]="wallaroo-features/wallaroo-tag-management"
    [wallaroo-inference-server-cv-frcnn]="wallaroo-free-tutorials/wallaroo-inference-server-tutorials/wallaroo-inference-server-cv-frcnn"
    [wallaroo-inference-server-hf-summarizer]="wallaroo-free-tutorials/wallaroo-inference-server-tutorials/wallaroo-inference-server-hf-summarizer"
    [arbitrary-python-upload-tutorials]="wallaroo-model-deploy-and-serve/arbitrary-python-upload-tutorials"
    [computer-vision]="wallaroo-model-deploy-and-serve/computer-vision"
    [hugging-face-upload-tutorials]="wallaroo-model-deploy-and-serve/hugging-face-upload-tutorials"
    [keras-upload-tutorials]="wallaroo-model-deploy-and-serve/keras-upload-tutorials"
    [mlflow-registries-upload-tutorials]="wallaroo-model-deploy-and-serve/mlflow-registries-upload-tutorials"
    [notebooks_in_prod]="wallaroo-model-deploy-and-serve/notebooks_in_prod"
    [parallel-inferences-sdk-aloha-tutorial]="wallaroo-model-deploy-and-serve/parallel-inferences-sdk-aloha-tutorial"
    [python-upload-tutorials]="wallaroo-model-deploy-and-serve/python-upload-tutorials"
    [pytorch-upload-tutorials]="wallaroo-model-deploy-and-serve/pytorch-upload-tutorials"
    [tensorflow-upload-tutorials]="wallaroo-model-deploy-and-serve/tensorflow-upload-tutorials"
    [xgboost-upload-tutorials]="wallaroo-model-deploy-and-serve/xgboost-upload-tutorials"
    [model-observability-anomaly-detection-ccfraud-sdk-tutorial]="wallaroo-observe-tutorials/model-observability-anomaly-detection-ccfraud-sdk-tutorial"
    [model-observability-anomaly-detection-houseprice-sdk-tutorial]="wallaroo-observe-tutorials/model-observability-anomaly-detection-houseprice-sdk-tutorial"
    [pipeline-log-tutorial]="wallaroo-observe-tutorials/pipeline-log-tutorial"
    [wallaro-model-observability-assays]="wallaroo-observe-tutorials/wallaro-model-observability-assays"
    [edge-architecture-publish-cv-resnet-model]="wallaroo-run-anywhere/edge-architecture-publish-cv-resnet-model"
    [edge-architecture-publish-linear-regression-houseprice-model]="wallaroo-run-anywhere/edge-architecture-publish-linear-regression-houseprice-model"
    [edge-observability-assays]="wallaroo-run-anywhere/edge-observability-assays"
    [edge-observability-low-no-connection]="wallaroo-run-anywhere/edge-observability-low-no-connection"
    [in-line-edge-model-replacements-tutorial]="wallaroo-run-anywhere/in-line-edge-model-replacements-tutorial"
    [run-anywhere-acceleration-aloha]="wallaroo-run-anywhere/run-anywhere-acceleration-aloha"
    [abtesting]="wallaroo-testing-tutorials/abtesting"
    [anomaly_detection]="wallaroo-testing-tutorials/anomaly_detection"
    [houseprice-saga]="wallaroo-testing-tutorials/houseprice-saga"
    [shadow_deploy]="wallaroo-testing-tutorials/shadow_deploy"
    [wallaro-assay-builder-tutorial]="wallaroo-testing-tutorials/wallaro-assay-builder-tutorial"
    [wallaro-model_observability_assays]="wallaroo-testing-tutorials/wallaro-model_observability_assays"
    [connection_api_bigquery_tutorial]="workload-orchestrations/connection_api_bigquery_tutorial"
    [orchestration_api_simple_tutorial]="workload-orchestrations/orchestration_api_simple_tutorial"
    [orchestration_sdk_bigquery_houseprice_tutorial]="workload-orchestrations/orchestration_sdk_bigquery_houseprice_tutorial"
    [orchestration_sdk_bigquery_statsmodel_tutorial]="workload-orchestrations/orchestration_sdk_bigquery_statsmodel_tutorial"
    [orchestration_sdk_comprehensive_tutorial]="workload-orchestrations/orchestration_sdk_comprehensive_tutorial"
    [orchestration_sdk_simple_tutorial]="workload-orchestrations/orchestration_sdk_simple_tutorial"


    )

currentDirectory=$PWD

# for zip in "${!tutorials[@]}"; 
#     do (cd ${tutorials[$zip]}/..;zip -r $currentDirectory/compress_tutorials/$zip.zip $zip);
# done

for zip in ${!tutorials[@]}; do
    #echo cd ${tutorials[${zip}]}/..;
    zip -r ./compress_tutorials/${zip}.zip ${tutorials[${zip}]};
done