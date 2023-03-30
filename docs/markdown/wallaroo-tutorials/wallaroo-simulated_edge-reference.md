This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-features/simulated_edge).

# Simulated Edge Demo

This notebook will explore "Edge ML", meaning deploying a model intended to be run on "the edge". What is "the edge"?  This is typically defined as a resource (CPU, memory, and/or bandwidth) constrained environment or where a combination of latency requirements and bandwidth available requires the models to run locally.

Wallaroo provides two key capabilities when it comes to deploying models to edge devices:

1. Since the same engine is used in both environments, the model behavior can often be simulated accurately using Wallaroo in a data center for testing prior to deployment.
2. Wallaroo makes edge deployments "observable" so the same tools used to monitor model performance can be used in both kinds of deployments. 

This notebook closely parallels the [Aloha tutorial](https://docs.wallaroo.ai/wallaroo-tutorials/wallaroo-quick-start-aloha/).  The primary difference is instead of provide ample resources to a pipeline to allow high-throughput operation we will specify a resource budget matching what is expected in the final deployment. Then we can apply the expected load to the model and observe how it behaves given the available resources.

This example uses the open source [Aloha CNN LSTM model](https://www.researchgate.net/publication/348920204_Using_Auxiliary_Inputs_in_Deep_Learning_Models_for_Detecting_DGA-based_Domain_Names) for classifying Domain names as being either legitimate or being used for nefarious purposes such as malware distribution. This could be deployed on a network router to detect suspicious domains in real-time. Of course, it is important to monitor the behavior of the model across all of the deployments so we can see if the detect rate starts to drift over time.

Note that this example is not intended for production use and is meant of an example of running Wallaroo in a restrained environment.  The environment is based on the [Wallaroo AWS EC2 Setup guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-install-guides/YOUR SUFFIX-install-guides/wallaroo-setup-environment-community/wallaroo-aws-vm-community-setup/).

Full details on how to configure a deployment through the SDK, see the [Wallaroo SDK guides](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/).

For our example, we will perform the following:

* Create a workspace for our work.
* Upload the Aloha model.
* Define a resource budget for our inference pipeline.
* Create a pipeline that can ingest our submitted data, submit it to the model, and export the results
* Run a sample inference through our pipeline by loading a file
* Run a batch inference through our pipeline's URL and store the results in a file and find that the original memory
  allocation is too small.
* Redeploy the pipeline with a larger memory budget and attempt sending the same batch of requests through again.

All sample data and models are available through the [Wallaroo Quick Start Guide Samples repository](https://github.com/WallarooLabs/quickstartguide_samples).

## Prerequisites

* A deployed Wallaroo instance
* The following Python libraries installed:
  * `os`
  * `string`
  * `random`
  * [`wallaroo`](https://pypi.org/project/wallaroo/): The Wallaroo SDK. Included with the Wallaroo JupyterHub service by default.
  * [`pandas`](https://pypi.org/project/pandas/): Pandas, mainly used for Pandas DataFrame
  * [`pyarrow`](https://pypi.org/project/pyarrow/): PyArrow for Apache Arrow support

## Open a Connection to Wallaroo

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  If logging in externally, update the `wallarooPrefix` and `wallarooSuffix` variables with the proper DNS information.  For more information on Wallaroo DNS settings, see the [Wallaroo DNS Integration Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-dns-guide/).


```python
import wallaroo
from wallaroo.object import EntityNotFoundError

# to display dataframe tables
from IPython.display import display
# used to display dataframe information without truncating
import pandas as pd
pd.set_option('display.max_colwidth', None)
import pyarrow as pa

import os
# Used for the Wallaroo SDK version 2023.1
os.environ["ARROW_ENABLED"]="True"
```


```python
# Login through local Wallaroo instance

wl = wallaroo.Client()

# SSO login through keycloak

# wallarooPrefix = "YOUR PREFIX"
# wallarooSuffix = "YOUR PREFIX"

# wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}.api.{wallarooSuffix}", 
#                     auth_endpoint=f"https://{wallarooPrefix}.keycloak.{wallarooSuffix}", 
#                     auth_type="sso")
```

    Please log into the following URL in a web browser:
    
    	https://wallaroo.keycloak.example.com/auth/realms/master/device?user_code=WQBJ-LJAV
    
    Login successful!


## Useful variables

The following variables and methods are used to create a workspace, the pipeline in the example workspace and upload models into it.

To allow this tutorial to be run multiple times or by multiple users in the same Wallaroo instance, a random 4 character prefix will be added to the workspace, pipeline, and model.


```python
import string
import random

# make a random 4 character prefix
prefix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))

pipeline_name = f'{prefix}edgepipelineexample'
workspace_name = f'{prefix}edgeworkspaceexample'
model_name = f'{prefix}alohamodel'
model_file_name = './alohacnnlstm.zip'
```


```python
def get_workspace(name):
    workspace = None
    for ws in wl.list_workspaces():
        if ws.name() == name:
            workspace= ws
    if(workspace == None):
        workspace = wl.create_workspace(name)
    return workspace

def get_pipeline(name):
    try:
        pipeline = wl.pipelines_by_name(pipeline_name)[0]
    except EntityNotFoundError:
        pipeline = wl.build_pipeline(pipeline_name)
    return pipeline
```

### Create or Set the Workspace

Create the workspace and set it as our default workspace.  If a workspace by the same name already exists, then that workspace will be used.


```python
workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)
workspace
```




    {'name': 'pqrdedgeworkspaceexample', 'id': 19, 'archived': False, 'created_by': '56d97480-bb64-4575-acb6-f93d05652e86', 'created_at': '2023-03-28T19:49:30.615031+00:00', 'models': [], 'pipelines': []}



# Upload the Models

Now we will upload our models.  Note that for this example we are applying the model from a .ZIP file.  The Aloha model is a [protobuf](https://developers.google.com/protocol-buffers) file that has been defined for evaluating web pages, and we will configure it to use data in the `tensorflow` format.


```python
model = wl.upload_model(model_name, model_file_name).configure("tensorflow")
```

# Define the resource budget
The DeploymentConfig object specifies the resources to allocate for a model pipeline. In this case, we're going to set a very small budget, one that is too small for this model and then expand it based on testing. To start with, we'll use 1 CPU and 250 MB of RAM.


```python
deployment_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(1).memory("250Mi").build()
```

## Deploy a model
Now that we have a model that we want to use we will create a deployment for it using the resource limits defined above. 

We will tell the deployment we are using a tensorflow model and give the deployment name and the configuration we want for the deployment.

To do this, we'll create our pipeline that can ingest the data, pass the data to our Aloha model, and give us a final output.  We'll call our pipeline `edgepipeline`, then deploy it so it's ready to receive data.  The deployment process usually takes about 45 seconds.

* **Note**:  If you receive an error that the pipeline could not be deployed because there are not enough resources, undeploy any other pipelines and deploy this one again.  This command can quickly undeploy all pipelines to regain resources.  We recommend **not** running this command in a production environment since it will cancel any running pipelines:

```python
for p in wl.list_pipelines(): p.undeploy()
```


```python
pipeline = get_pipeline(pipeline_name)
pipeline.add_model_step(model)
```




<table><tr><th>name</th> <td>pqrdedgepipelineexample</td></tr><tr><th>created</th> <td>2023-03-28 19:49:41.000671+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-28 20:00:37.785345+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>5654e857-ba60-4fb6-8927-5a003d31b74b, f221f159-187c-408b-8104-11805544429e, de8c734f-06cd-48ca-968a-6250181d97f7, 5dde4f15-6982-47a8-9cf9-a269b1fb7cb4</td></tr><tr><th>steps</th> <td>pqrdalohamodel</td></tr></table>




```python
pipeline.deploy(deployment_config=deployment_config)
```




<table><tr><th>name</th> <td>pqrdedgepipelineexample</td></tr><tr><th>created</th> <td>2023-03-28 19:49:41.000671+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-28 21:01:58.133579+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>bb7854a2-9f6e-48ae-b796-12ff3448c1ba, 28a2346b-fac9-43bc-b428-31f35fe9b959, 5413f63b-302a-4676-9d35-388754f3b0bd, b199203a-655e-4791-bae5-b747c833b721, 5232dd8e-e322-4a02-817f-bd44ed4dccd2, 0ac298bd-b55c-4c84-9f4a-4092ecb4b584, 297bd048-dd96-4162-83c3-cf7a1754d61c, 52cd469f-47b5-48fd-b625-3e5b10e0b106, 7bd47ce9-c6ec-4cbc-84ba-ac5fe14ce552, 6f6b8136-0484-4c2e-bff0-b58da7e8374f, 8d856b1b-2eba-4c75-89f3-79fdee22a916, 91e8e81e-3b3d-45d5-8173-77856d12b550, a0bf21c0-10e6-4646-8728-b2ac7aab1e15, 5654e857-ba60-4fb6-8927-5a003d31b74b, f221f159-187c-408b-8104-11805544429e, de8c734f-06cd-48ca-968a-6250181d97f7, 5dde4f15-6982-47a8-9cf9-a269b1fb7cb4</td></tr><tr><th>steps</th> <td>pqrdalohamodel</td></tr></table>




```python
# pipeline.undeploy()
```




<table><tr><th>name</th> <td>pqrdedgepipelineexample</td></tr><tr><th>created</th> <td>2023-03-28 19:49:41.000671+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-28 20:59:58.347692+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>28a2346b-fac9-43bc-b428-31f35fe9b959, 5413f63b-302a-4676-9d35-388754f3b0bd, b199203a-655e-4791-bae5-b747c833b721, 5232dd8e-e322-4a02-817f-bd44ed4dccd2, 0ac298bd-b55c-4c84-9f4a-4092ecb4b584, 297bd048-dd96-4162-83c3-cf7a1754d61c, 52cd469f-47b5-48fd-b625-3e5b10e0b106, 7bd47ce9-c6ec-4cbc-84ba-ac5fe14ce552, 6f6b8136-0484-4c2e-bff0-b58da7e8374f, 8d856b1b-2eba-4c75-89f3-79fdee22a916, 91e8e81e-3b3d-45d5-8173-77856d12b550, a0bf21c0-10e6-4646-8728-b2ac7aab1e15, 5654e857-ba60-4fb6-8927-5a003d31b74b, f221f159-187c-408b-8104-11805544429e, de8c734f-06cd-48ca-968a-6250181d97f7, 5dde4f15-6982-47a8-9cf9-a269b1fb7cb4</td></tr><tr><th>steps</th> <td>pqrdalohamodel</td></tr></table>



We can verify that the pipeline is running and list what models are associated with it.


```python
pipeline.status()
```




    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.3.33',
       'name': 'engine-58fd5b5fd9-jpzdq',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'pqrdedgepipelineexample',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'pqrdalohamodel',
          'version': '8794837b-2802-4faa-877a-71f0333d469a',
          'sha': 'd71d9ffc61aaac58c2b1ed70a2db13d1416fb9d3f5b891e5e4e2e97180fe22f8',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.1.58',
       'name': 'engine-lb-ddd995646-7xk74',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}



## Inferences

### Infer 1 row

Now that the pipeline is deployed and our model is in place, we'll perform a smoke test to verify the pipeline is up and running properly.  We'll use the `infer_from_file` command to load a single encoded URL into the inference engine and print the results back out.

The result should tell us that the tokenized URL is legitimate (0) or fraud (1).  This sample data should return close to 1.


```python
smoke_test = pd.DataFrame.from_records(
    [
    {
        "text_input":[
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            28,
            16,
            32,
            23,
            29,
            32,
            30,
            19,
            26,
            17
        ]
    }
]
)

result = pipeline.infer(smoke_test)
display(result.loc[:, ["time","out.main"]])
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out.main</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-03-28 21:02:37.475</td>
      <td>[0.997564]</td>
    </tr>
  </tbody>
</table>
</div>


* **IMPORTANT NOTE**:  The `_deployment._url()` method will return an **internal** URL when using Python commands from within the Wallaroo instance - for example, the Wallaroo JupyterHub service.  When connecting via an external connection, `_deployment._url()` returns an **external** URL.  External URL connections requires [the authentication be included in the HTTP request](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/), and that [Model Endpoints Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-model-endpoints-guide/) external endpoints are enabled in the Wallaroo configuration options.


```python
inference_url = pipeline._deployment._url()
display(inference_url)
connection =wl.mlops().__dict__
token = connection['token']
```


    'https://wallaroo.api.example.com/v1/api/pipelines/infer/pqrdedgepipelineexample-29'



```python
dataFile="./data/data_1k.arrow"
contentType="application/vnd.apache.arrow.file"
```


```python
!curl -X POST {inference_url} -H "Authorization: Bearer {token}" -H "Content-Type:{contentType}" --data-binary @{dataFile} > curl_response.df
```

# Redeploy with a little larger budget 
If you look in the file curl_response.txt, you will see that the inference failed:
> upstream connect error or disconnect/reset before headers. reset reason: connection termination

Even though a single inference passed, submitted a larger batch of work did not. If this is an expected usage case for
this model, we need to add more memory. Let's do that now.

The following DeploymentConfig is the same as the original, but increases the memory from 300MB to 600MB. This sort
of budget would be available on some network routers.


```python
pipeline.undeploy()
deployment_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(1).memory("600Mi").build()
pipeline.deploy(deployment_config=deployment_config)
```




<table><tr><th>name</th> <td>pqrdedgepipelineexample</td></tr><tr><th>created</th> <td>2023-03-28 19:49:41.000671+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-28 21:03:41.204901+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>30b0c272-3de3-46e9-936a-45556d34db3a, bb7854a2-9f6e-48ae-b796-12ff3448c1ba, 28a2346b-fac9-43bc-b428-31f35fe9b959, 5413f63b-302a-4676-9d35-388754f3b0bd, b199203a-655e-4791-bae5-b747c833b721, 5232dd8e-e322-4a02-817f-bd44ed4dccd2, 0ac298bd-b55c-4c84-9f4a-4092ecb4b584, 297bd048-dd96-4162-83c3-cf7a1754d61c, 52cd469f-47b5-48fd-b625-3e5b10e0b106, 7bd47ce9-c6ec-4cbc-84ba-ac5fe14ce552, 6f6b8136-0484-4c2e-bff0-b58da7e8374f, 8d856b1b-2eba-4c75-89f3-79fdee22a916, 91e8e81e-3b3d-45d5-8173-77856d12b550, a0bf21c0-10e6-4646-8728-b2ac7aab1e15, 5654e857-ba60-4fb6-8927-5a003d31b74b, f221f159-187c-408b-8104-11805544429e, de8c734f-06cd-48ca-968a-6250181d97f7, 5dde4f15-6982-47a8-9cf9-a269b1fb7cb4</td></tr><tr><th>steps</th> <td>pqrdalohamodel</td></tr></table>



# Re-run inference
Running the same curl command again should now produce a curl_response.txt file containing the expected results.


```python
connection =wl.mlops().__dict__
token = connection['token']
print(f'curl -X POST {inference_url} -H "Authorization: Bearer {token}" -H "Content-Type:{contentType}" --data-binary @{dataFile} > curl_response.df')
```

    curl -X POST https://wallaroo.api.example.com/v1/api/pipelines/infer/pqrdedgepipelineexample-29 -H "Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJCRFdIZ3Q0WmxRdEIxVDNTTkJ2RjlkYkU3RmxkSWdXRENwb041UkJLeTlrIn0.eyJleHAiOjE2ODAwMzc1NTIsImlhdCI6MTY4MDAzNzQ5MiwiYXV0aF90aW1lIjoxNjgwMDEyNzg2LCJqdGkiOiIxMjYzMGYzNy1mZjA0LTRkMmQtYTM1Mi0wZGMzMTA0NWRiY2IiLCJpc3MiOiJodHRwczovL2RvYy10ZXN0LmtleWNsb2FrLndhbGxhcm9vY29tbXVuaXR5Lm5pbmphL2F1dGgvcmVhbG1zL21hc3RlciIsImF1ZCI6WyJtYXN0ZXItcmVhbG0iLCJhY2NvdW50Il0sInN1YiI6IjU2ZDk3NDgwLWJiNjQtNDU3NS1hY2I2LWY5M2QwNTY1MmU4NiIsInR5cCI6IkJlYXJlciIsImF6cCI6InNkay1jbGllbnQiLCJzZXNzaW9uX3N0YXRlIjoiYWNhNzFhZmEtODA3NC00MjE3LWE5ZWItNTk4YzVkMzI5Yjg1IiwiYWNyIjoiMCIsInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJkZWZhdWx0LXJvbGVzLW1hc3RlciIsIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iXX0sInJlc291cmNlX2FjY2VzcyI6eyJtYXN0ZXItcmVhbG0iOnsicm9sZXMiOlsibWFuYWdlLXVzZXJzIiwidmlldy11c2VycyIsInF1ZXJ5LWdyb3VwcyIsInF1ZXJ5LXVzZXJzIl19LCJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6InByb2ZpbGUgZW1haWwiLCJzaWQiOiJhY2E3MWFmYS04MDc0LTQyMTctYTllYi01OThjNWQzMjliODUiLCJlbWFpbF92ZXJpZmllZCI6ZmFsc2UsImh0dHBzOi8vaGFzdXJhLmlvL2p3dC9jbGFpbXMiOnsieC1oYXN1cmEtdXNlci1pZCI6IjU2ZDk3NDgwLWJiNjQtNDU3NS1hY2I2LWY5M2QwNTY1MmU4NiIsIngtaGFzdXJhLWRlZmF1bHQtcm9sZSI6InVzZXIiLCJ4LWhhc3VyYS1hbGxvd2VkLXJvbGVzIjpbInVzZXIiXSwieC1oYXN1cmEtdXNlci1ncm91cHMiOiJ7fSJ9LCJuYW1lIjoiSm9obiBIYW5zYXJpY2siLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJqb2huLmh1bW1lbEB3YWxsYXJvby5haSIsImdpdmVuX25hbWUiOiJKb2huIiwiZmFtaWx5X25hbWUiOiJIYW5zYXJpY2siLCJlbWFpbCI6ImpvaG4uaHVtbWVsQHdhbGxhcm9vLmFpIn0.m1Ge_NdHJqKbm8ymg4bJcxXLjeaPGDfDVg2bFzrypekKSLzbINlNvp617D5nFg5NZL2_sOl6lLRXC_iN1_De9GmKOj9o-VibAn0UfqIZNIeXK9L3JsHLpzuY1CDgOG84rplqHHFWv_hqduHoRbsy2P9mnCLxG-5scAt653EoTGHP94O7bWa8vzh0T04tq1f9bVcL9o6rptIPRgoGqPCmCqyAjtrB876rqyekmd0wVemFLpAk9P0_o5_3Bt5iiEUXH877a5fwYbWyI4DaoilVsWw-ESNI2GKPOMfXhN5bAmQxs2SXHfOo4Vpo6HZepSkdfDyNqivGJc9DamGKop0HsQ" -H "Content-Type:application/vnd.apache.arrow.file" --data-binary @./data/data_1k.arrow > curl_response.df



```python
!curl -X POST {inference_url} -H "Authorization: Bearer {token}" -H "Content-Type:{contentType}" --data-binary @{dataFile} > curl_response.df
```

It is important to note that increasing the memory was necessary to run a batch of 1,000 inferences at once. If this is not a design
use case for your system, running with the smaller memory budget may be acceptable. Wallaroo allows you to easily test difference
loading patterns to get a sense for what resources are required with sufficient buffer to allow for robust operation of your system
while not over-provisioning scarce resources.

## Undeploy Pipeline

When finished with our tests, we will undeploy the pipeline so we have the Kubernetes resources back for other tasks.  Note that if the deployment variable is unchanged aloha_pipeline.deploy() will restart the inference engine in the same configuration as before.


```python
pipeline.undeploy()
```




<table><tr><th>name</th> <td>pqrdedgepipelineexample</td></tr><tr><th>created</th> <td>2023-03-28 19:49:41.000671+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-28 21:03:41.204901+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>30b0c272-3de3-46e9-936a-45556d34db3a, bb7854a2-9f6e-48ae-b796-12ff3448c1ba, 28a2346b-fac9-43bc-b428-31f35fe9b959, 5413f63b-302a-4676-9d35-388754f3b0bd, b199203a-655e-4791-bae5-b747c833b721, 5232dd8e-e322-4a02-817f-bd44ed4dccd2, 0ac298bd-b55c-4c84-9f4a-4092ecb4b584, 297bd048-dd96-4162-83c3-cf7a1754d61c, 52cd469f-47b5-48fd-b625-3e5b10e0b106, 7bd47ce9-c6ec-4cbc-84ba-ac5fe14ce552, 6f6b8136-0484-4c2e-bff0-b58da7e8374f, 8d856b1b-2eba-4c75-89f3-79fdee22a916, 91e8e81e-3b3d-45d5-8173-77856d12b550, a0bf21c0-10e6-4646-8728-b2ac7aab1e15, 5654e857-ba60-4fb6-8927-5a003d31b74b, f221f159-187c-408b-8104-11805544429e, de8c734f-06cd-48ca-968a-6250181d97f7, 5dde4f15-6982-47a8-9cf9-a269b1fb7cb4</td></tr><tr><th>steps</th> <td>pqrdalohamodel</td></tr></table>

