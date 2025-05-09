# %% [markdown]
# 
# This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/blob/wallaroo2025.1_tutorials/wallaroo-model-operations-tutorials/infer/infer).
# 
# ## Wallaroo SDK Infer Tutorial
# 
# Wallaroo provides the ability to perform inferences through deployed pipelines via the Wallaroo SDK and the Wallaroo MLOps API.  This tutorial demonstrates performing inferences using the Wallaroo SDK.
# 
# This tutorial provides the following:
# 
# * `ccfraud.onnx`:  A pre-trained credit card fraud detection model.
# * `data/cc_data_1k.arrow`, `data/cc_data_10k.arrow`: Sample testing data in Apache Arrow format with 1,000 and 10,000 records respectively.
# * `infer-sdk.py`: A code-only version of this tutorial as a Python script.
# 
# This tutorial and sample data comes from the Machine Learning Group's demonstration on [Credit Card Fraud detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
# 
# ### Prerequisites
# 
# The following is required for this tutorial:
# 
# * A [deployed Wallaroo instance](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-install-guides/) with [Model Endpoints Enabled](https://docs.wallaroo.ai/wallaroo-platform-operations/wallaroo-platform-operations-install/wallaroo-platform-operations-install-configure/wallaroo-platform-operations-configure/wallaroo-model-endpoints-guide/)
# * The following Python libraries:
#   * [`pandas`](https://pypi.org/project/pandas/)
#   * [`pyarrow`](https://pypi.org/project/pyarrow/)
#   * [`wallaroo`](https://pypi.org/project/wallaroo/) (Installed in the Wallaroo JupyterHub service by default).
# 
# ### Tutorial Goals
# 
# This demonstration provides a quick tutorial on performing inferences using the Wallaroo SDK using the Pipeline `infer` and `infer_from_file` methods.  This following steps will be performed:
# 
# * Connect to a Wallaroo instance using environmental variables.  This bypasses the browser link confirmation for a seamless login.  For more information, see the [Wallaroo SDK Essentials Guide:  Client Connection](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).
# * Create a workspace for our models and pipelines.
# * Upload the `ccfraud` model.
# * Create a pipeline and add the `ccfraud` model as a pipeline step.
# * Run a sample inference through SDK Pipeline `infer` method.
# * Run a batch inference through SDK Pipeline `infer_from_file` method.
# * Run a DataFrame and Arrow based inference through the pipeline Inference URL.

# %% [markdown]
# ## Open a Connection to Wallaroo
# 
# The first step is to connect to Wallaroo through the Wallaroo client.

# %%
import wallaroo
from wallaroo.object import EntityNotFoundError
import pandas as pd
import os
import pyarrow as pa

# used to display dataframe information without truncating
from IPython.display import display
pd.set_option('display.max_colwidth', None)

import requests

# %%
wl = wallaroo.Client()

# %% [markdown]
# ## Create the Workspace
# 
# We will create a workspace to work in and call it the `sdkinferenceexampleworkspace`, then set it as current workspace environment.  We'll also create our pipeline in advance as `sdkinferenceexamplepipeline`.
# 
# The model to be uploaded and used for inference will be labeled as `ccfraud`.

# %%
workspace_name = f'sdkinferenceexampleworkspace'
pipeline_name = f'sdkinferenceexamplepipeline'
model_name = f'ccfraud'
model_file_name = './ccfraud.onnx'

# %%
workspace = wl.get_workspace(name=workspace_name, create_if_not_exist=True)

wl.set_current_workspace(workspace)

# %% [markdown]
# ## Build Pipeline
# 
# In a production environment, the pipeline would already be set up with the model and pipeline steps.  We would then select it and use it to perform our inferences.
# 
# For this example we will create the pipeline and add the `ccfraud` model as a pipeline step and deploy it.  Deploying a pipeline allocates resources from the Kubernetes cluster hosting the Wallaroo instance and prepares it for performing inferences.
# 
# If this process was already completed, it can be commented out and skipped for the next step [Select Pipeline](#select-pipeline).
# 
# Then we will list the pipelines and select the one we will be using for the inference demonstrations.

# %%
# Create or select the current pipeline

ccfraudpipeline = wl.build_pipeline(pipeline_name)

# Add ccfraud model as the pipeline step

ccfraud_model = (wl.upload_model(model_name, 
                                 model_file_name, 
                                 framework=wallaroo.framework.Framework.ONNX)
                                 .configure(tensor_fields=["tensor"])
                )

ccfraudpipeline.add_model_step(ccfraud_model).deploy()

# %% [markdown]
# ## Select Pipeline
# 
# This step assumes that the pipeline is prepared with `ccfraud` as the current step.  The method `pipelines_by_name(name)` returns an array of pipelines with names matching the `pipeline_name` field.  This example assumes only one pipeline is assigned the name `sdkinferenceexamplepipeline`.

# %%
# List the pipelines by name in the current workspace - just the first several to save space.

display(wl.list_pipelines()[:5])

# Set the `pipeline` variable to our sample pipeline.

pipeline = wl.pipelines_by_name(pipeline_name)[0]
display(pipeline)

# %% [markdown]
# ## Inferences via SDK
# 
# Once a pipeline has been deployed, an inference can be run.  This will submit data to the pipeline, where it is processed through each of the pipeline's steps with the output of the previous step providing the input for the next step.  The final step will then output the result of all of the pipeline's steps.
# 
# * Inputs are either sent one of the following:
#   * [pandas.DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html).  The return value will be a pandas.DataFrame.
#   * [Apache Arrow](https://arrow.apache.org/).  The return value will be an Apache Arrow table.
#   * [Custom JSON](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-inferences/#inferenceresult-object).  The return value will be a Wallaroo InferenceResult object.
# 
# Inferences are performed through the Wallaroo SDK via the Pipeline `infer` and `infer_from_file` methods.
# 
# ### infer Method
# 
# Now that the pipeline is deployed we'll perform an inference using the Pipeline `infer` method, and submit a pandas DataFrame as our input data.  This will return a pandas DataFrame as the inference output.
# 
# For more information, see the [Wallaroo SDK Essentials Guide: Inferencing: Run Inference through Local Variable](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-inferences/#run-inference-through-local-variable).

# %%
smoke_test = pd.DataFrame.from_records([
    {
        "tensor":[
            1.0678324729,
            0.2177810266,
            -1.7115145262,
            0.682285721,
            1.0138553067,
            -0.4335000013,
            0.7395859437,
            -0.2882839595,
            -0.447262688,
            0.5146124988,
            0.3791316964,
            0.5190619748,
            -0.4904593222,
            1.1656456469,
            -0.9776307444,
            -0.6322198963,
            -0.6891477694,
            0.1783317857,
            0.1397992467,
            -0.3554220649,
            0.4394217877,
            1.4588397512,
            -0.3886829615,
            0.4353492889,
            1.7420053483,
            -0.4434654615,
            -0.1515747891,
            -0.2668451725,
            -1.4549617756
        ]
    }
])
result = pipeline.infer(smoke_test)
display(result)

# %% [markdown]
# ### infer_from_file Method
# 
# This example uses the Pipeline method `infer_from_file` to submit 10,000 records as a batch using an Apache Arrow table.  The method will return an Apache Arrow table.  For more information, see the [Wallaroo SDK Essentials Guide: Inferencing: Run Inference From A File](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-inferences/#inference-from-file)
# 
# The results will be converted into a [pandas.DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html).  The results will be filtered by transactions likely to be credit card fraud.

# %%
result = pipeline.infer_from_file('./data/cc_data_10k.arrow')

display(result)

# %%
# use pyarrow to convert results to a pandas DataFrame and display only the results with > 0.75

list = [0.75]

outputs =  result.to_pandas()
# display(outputs)
filter = [elt[0] > 0.75 for elt in outputs['out.dense_1']]
outputs = outputs.loc[filter]
display(outputs)

# %% [markdown]
# ## Inferences via HTTP POST
# 
# Each pipeline has its own Inference URL that allows HTTP/S POST submissions of inference requests.  Full details are available from the [Inferencing via the Wallaroo MLOps API](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-api-essential-guide/wallaroo-mlops-api-essential-guide-inferences/).
# 
# This example will demonstrate performing inferences with a DataFrame input and an Apache Arrow input.

# %% [markdown]
# ### Request JWT Token
# 
# There are two ways to retrieve the JWT token used to authenticate to the Wallaroo MLOps API.
# 
# * [Wallaroo SDK](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-api-essential-guide/#through-the-wallaroo-sdk).  This method requires a Wallaroo based user.
# * [API Clent Secret](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-api-essential-guide/#through-keycloak).  This is the recommended method as it is user independent.  It allows any valid user to make an inference request.
# 
# This tutorial will use the Wallaroo SDK method Wallaroo Client `wl.auth.auth_header()` method, extracting the Authentication header from the response.
# 
# Reference:  [MLOps API Retrieve Token Through Wallaroo SDK](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-api-essential-guide/#through-the-wallaroo-sdk)

# %%
headers = wl.auth.auth_header()
display(headers)

# %% [markdown]
# ### Retrieve the Pipeline Inference URL
# 
# The Pipeline Inference URL is retrieved via the Wallaroo SDK with the Pipeline `._deployment._url()` method.
# 
# * **IMPORTANT NOTE**:  The `_deployment._url()` method will return an **internal** URL when using Python commands from within the Wallaroo instance - for example, the Wallaroo JupyterHub service.  When connecting via an external connection, `_deployment._url()` returns an **external** URL.
#   * External URL connections requires [the authentication be included in the HTTP request](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/), and [Model Endpoints](https://docs.wallaroo.ai/wallaroo-platform-operations/wallaroo-platform-operations-install/wallaroo-platform-operations-install-configure/wallaroo-platform-operations-configure/wallaroo-model-endpoints-guide/) are enabled in the Wallaroo configuration options.

# %%
deploy_url = pipeline._deployment._url()
print(deploy_url)

# %% [markdown]
# ### HTTP Inference with DataFrame Input
# 
# The following example performs a HTTP Inference request with a DataFrame input.  The request will be made with first a Python `requests` method, then using `curl`.

# %%
# get authorization header
headers = wl.auth.auth_header()

## Inference through external URL using dataframe

# retrieve the json data to submit
data = pd.DataFrame.from_records([
    {
        "tensor":[
            1.0678324729,
            0.2177810266,
            -1.7115145262,
            0.682285721,
            1.0138553067,
            -0.4335000013,
            0.7395859437,
            -0.2882839595,
            -0.447262688,
            0.5146124988,
            0.3791316964,
            0.5190619748,
            -0.4904593222,
            1.1656456469,
            -0.9776307444,
            -0.6322198963,
            -0.6891477694,
            0.1783317857,
            0.1397992467,
            -0.3554220649,
            0.4394217877,
            1.4588397512,
            -0.3886829615,
            0.4353492889,
            1.7420053483,
            -0.4434654615,
            -0.1515747891,
            -0.2668451725,
            -1.4549617756
        ]
    }
])


# set the content type for pandas records
headers['Content-Type']= 'application/json; format=pandas-records'

# set accept as pandas-records
headers['Accept']='application/json; format=pandas-records'

# submit the request via POST, import as pandas DataFrame
response = pd.DataFrame.from_records(
                requests.post(
                    deploy_url, 
                    data=data.to_json(orient="records"), 
                    headers=headers)
                .json()
            )
display(response.loc[:,["time", "out"]])

# %%
!curl -X POST {deploy_url} -H "Authorization: {headers['Authorization']}" -H "Content-Type:{headers['Content-Type']}" -H "Accept:{headers['Accept']}" --data '{data.to_json(orient="records")}'

# %% [markdown]
# ### HTTP Inference with Arrow Input
# 
# The following example performs a HTTP Inference request with an Apache Arrow input.  The request will be made with first a Python `requests` method, then using `curl`.
# 
# Only the first 5 rows will be displayed for space purposes.

# %%
# get authorization header
headers = wl.auth.auth_header()

# Submit arrow file
dataFile="./data/cc_data_10k.arrow"

data = open(dataFile,'rb').read()

# set the content type for Arrow table
headers['Content-Type']= "application/vnd.apache.arrow.file"

# set accept as Apache Arrow
headers['Accept']="application/vnd.apache.arrow.file"

response = requests.post(
                    deploy_url, 
                    headers=headers, 
                    data=data, 
                    verify=True
                )

# Arrow table is retrieved 
with pa.ipc.open_file(response.content) as reader:
    arrow_table = reader.read_all()

# convert to Polars DataFrame and display the first 5 rows
display(arrow_table.to_pandas().head(5).loc[:,["time", "out"]])

# %%
!curl -X POST {deploy_url} -H "Authorization: {headers['Authorization']}" -H "Content-Type:{headers['Content-Type']}" -H "Accept:{headers['Accept']}" --data-binary @{dataFile} > curl_response.arrow

# %% [markdown]
# ## Undeploy Pipeline
# 
# When finished with our tests, we will undeploy the pipeline so we have the Kubernetes resources back for other tasks.

# %%
pipeline.undeploy()


