#!/opt/homebrew/bin/bash

# This relies on Bash 4 and above
# Meant to be run from the root of this project folder

declare -A tutorials

tutorials=( 
    ["standard-install"]="development/sdk-install-guides/standard-install" 
    ["google-vertex-sdk-install"]="development/sdk-install-guides/google-vertex-sdk-install"
    ["databricks-azure-sdk-install"]="development/sdk-install-guides/databricks-azure-sdk-install"
    ["azure-ml-sdk-install"]="development/sdk-install-guides/azure-ml-sdk-install"
    ["aws-sagemaker-install"]="development/sdk-install-guides/aws-sagemaker-install"
    ["wallaroo-101"]="wallaroo-101"
    ["model_hot_swap"]="wallaroo-features/model_hot_swap"
    ["simulated_edge"]="wallaroo-features/simulated_edge"
    ["wallaroo-model-endpoints"]="wallaroo-features/wallaroo-model-endpoints"
    ["wallaroo-tag-management"]="wallaroo-features/wallaroo-tag-management"
    ["aloha"]="wallaroo-model-cookbooks/aloha"
    ["demand_curve"]="wallaroo-model-cookbooks/demand_curve"
    ["imdb"]="wallaroo-model-cookbooks/imdb"
    ["abtesting"]="wallaroo-testing-tutorials/abtesting"
    ["autoconversion-tutorial"]="model_conversion/autoconversion-tutorial"
    ["keras-to-onnx"]="model_conversion/keras-to-onnx"
    ["pytorch-to-onnx"]="model_conversion/pytorch-to-onnx"
    ["sklearn-classification-to-onnx"]="model_conversion/sklearn-classification-to-onnx"
    ["sklearn-regression-to-onnx"]="model_conversion/sklearn-regression-to-onnx"
    ["statsmodels"]="model_conversion/statsmodels"
    ["xgboost-autoconversion"]="model_conversion/xgboost-autoconversion"
    ["notebooks_in_prod"]="notebooks_in_prod"
    ["anomaly_detection"]="wallaroo-testing-tutorials/anomaly_detection"
    ["shadow_deploy"]="wallaroo-testing-tutorials/shadow_deploy"
    ["assays_model_insights"]="wallaroo-features/model_insights"
    ["mlflow-tutorial"]="wallaroo-model-cookbooks/mlflow-tutorial"
    ["connection_api_bigquery_tutorial"]="workload-orchestrations/connection_api_bigquery_tutorial"
    ["orchestration_api_simple_tutorial"]="workload-orchestrations/orchestration_api_simple_tutorial"
    ["orchestration_sdk_bigquery_houseprice_tutorial"]="workload-orchestrations/orchestration_sdk_bigquery_houseprice_tutorial"
    ["orchestration_sdk_bigquery_statsmodel_tutorial"]="workload-orchestrations/orchestration_sdk_bigquery_statsmodel_tutorial"
    ["orchestration_sdk_comprehensive_tutorial"]="workload-orchestrations/orchestration_sdk_comprehensive_tutorial"
    ["orchestration_sdk_simple_tutorial"]="workload-orchestrations/orchestration_sdk_simple_tutorial"
    ["pipeline_log_tutorial"]=wallaroo-features/pipeline_log_tutorial
    ["pipeline_api_log_tutorial"]=wallaroo-features/pipeline_api_log_tutorial
    ["computer-vision"]=wallaroo-model-cookbooks/computer-vision
    ["houseprice-saga"]=wallaroo-testing-tutorials/houseprice-saga
    )

currentDirectory=$PWD

for zip in "${!tutorials[@]}"; 
    do (cd ${tutorials[$zip]}/..;zip -r $currentDirectory/compress_tutorials/$zip.zip $zip);
done

