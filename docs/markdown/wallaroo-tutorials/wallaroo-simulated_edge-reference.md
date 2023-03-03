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

## Open a Connection to Wallaroo

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  If logging in externally, update the `wallarooPrefix` and `wallarooSuffix` variables with the proper DNS information.  For more information on Wallaroo DNS settings, see the [Wallaroo DNS Integration Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-dns-guide/).


```python
import wallaroo
from wallaroo.object import EntityNotFoundError
import pandas as pd
import json
from IPython.display import display

# used to display dataframe information without truncating
pd.set_option('display.max_colwidth', None)
```


```python
# Client connection from local Wallaroo instance

wl = wallaroo.Client()

# SSO login through keycloak

# wallarooPrefix = "YOUR PREFIX"
# wallarooSuffix = "YOUR SUFFIX"

# wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}.api.{wallarooSuffix}", 
#                     auth_endpoint=f"https://{wallarooPrefix}.keycloak.{wallarooSuffix}", 
#                     auth_type="sso")
```

### Arrow Support

As of the 2023.1 release, Wallaroo provides support for dataframe and Arrow for inference inputs.  This tutorial allows users to adjust their experience based on whether they have enabled Arrow support in their Wallaroo instance or not.

If Arrow support has been enabled, `arrowEnabled=True`. If disabled or you're not sure, set it to `arrowEnabled=False`

The examples below will be shown in an arrow enabled environment.


```python
import os
# Only set the below to make the OS environment ARROW_ENABLED to TRUE.  Otherwise, leave as is.
# os.environ["ARROW_ENABLED"]="True"

if "ARROW_ENABLED" not in os.environ or os.environ["ARROW_ENABLED"].casefold() == "False".casefold():
    arrowEnabled = False
else:
    arrowEnabled = True
print(arrowEnabled)
```

    True


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




    {'name': 'wjtxedgeworkspaceexample', 'id': 86, 'archived': False, 'created_by': '138bd7e6-4dc8-4dc1-a760-c9e721ef3c37', 'created_at': '2023-02-27T17:46:51.007989+00:00', 'models': [], 'pipelines': []}



# Upload the Models

Now we will upload our models.  Note that for this example we are applying the model from a .ZIP file.  The Aloha model is a [protobuf](https://developers.google.com/protocol-buffers) file that has been defined for evaluating web pages, and we will configure it to use data in the `tensorflow` format.


```python
model = wl.upload_model(model_name, model_file_name).configure("tensorflow")
```

# Define the resource budget
The DeploymentConfig object specifies the resources to allocate for a model pipeline. In this case, we're going to set a very small budget, one that is too small for this model and then expand it based on testing. To start with, we'll use 1 CPU and 150 MB of RAM.


```python
deployment_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(1).memory("150Mi").build()
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
pipeline.deploy(deployment_config=deployment_config)
```




<table><tr><th>name</th> <td>wjtxedgepipelineexample</td></tr><tr><th>created</th> <td>2023-02-27 17:46:53.711612+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-27 17:46:54.419854+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>e33295e1-fa57-4709-a5ee-23cdd2b66131, 1fc9d0e7-6783-4050-9b86-6f0276fb8745</td></tr><tr><th>steps</th> <td>wjtxalohamodel</td></tr></table>



We can verify that the pipeline is running and list what models are associated with it.


```python
pipeline.status()
```




    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.48.0.146',
       'name': 'engine-77958c5d4-5zj9x',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'wjtxedgepipelineexample',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'wjtxalohamodel',
          'version': 'f1e3cbc4-b58a-4c3c-95fb-8e46803115b6',
          'sha': 'd71d9ffc61aaac58c2b1ed70a2db13d1416fb9d3f5b891e5e4e2e97180fe22f8',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.48.0.148',
       'name': 'engine-lb-74b4969486-227j2',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}



## Inferences

### Infer 1 row

Now that the pipeline is deployed and our model is in place, we'll perform a smoke test to verify the pipeline is up and running properly.  We'll use the `infer_from_file` command to load a single encoded URL into the inference engine and print the results back out.

The result should tell us that the tokenized URL is legitimate (0) or fraud (1).  This sample data should return close to 0.


```python
if arrowEnabled is True:
    result = pipeline.infer_from_file('./data/data_1.df.json')
else:
    result = pipeline.infer_from_file("./data/data_1.json")
display(result)
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
      <th>in.text_input</th>
      <th>out.qakbot</th>
      <th>out.banjori</th>
      <th>out.dircrypt</th>
      <th>out.corebot</th>
      <th>out.suppobox</th>
      <th>out.cryptolocker</th>
      <th>out.simda</th>
      <th>out.ramnit</th>
      <th>out.main</th>
      <th>out.matsnu</th>
      <th>out.kraken</th>
      <th>out.ramdo</th>
      <th>out.gozi</th>
      <th>out.locky</th>
      <th>out.pykspa</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-02-27 17:47:15.407</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 16, 32, 23, 29, 32, 30, 19, 26, 17]</td>
      <td>[0.016155045]</td>
      <td>[0.0015195842]</td>
      <td>[4.7591206e-05]</td>
      <td>[0.98291475]</td>
      <td>[1.3889844e-27]</td>
      <td>[0.012099553]</td>
      <td>[1.7933435e-26]</td>
      <td>[0.0009985747]</td>
      <td>[0.997564]</td>
      <td>[0.010341614]</td>
      <td>[0.00031977228]</td>
      <td>[0.0062362333]</td>
      <td>[2.0289332e-05]</td>
      <td>[0.011029261]</td>
      <td>[0.008038961]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


* **IMPORTANT NOTE**:  The `_deployment._url()` method will return an **internal** URL when using Python commands from within the Wallaroo instance - for example, the Wallaroo JupyterHub service.  When connecting via an external connection, `_deployment._url()` returns an **external** URL.  External URL connections requires [the authentication be included in the HTTP request](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/), and that [Model Endpoints Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-model-endpoints-guide/) external endpoints are enabled in the Wallaroo configuration options.


```python
inference_url = pipeline._deployment._url()
print(inference_url)
connection =wl.mlops().__dict__
token = connection['token']
```

    https://sparkly-apple-3026.api.wallaroo.community/v1/api/pipelines/infer/wjtxedgepipelineexample-106



```python
if arrowEnabled is True:
    dataFile="./data/data_1k.df.json"
    contentType="application/json; format=pandas-records"
else:
    dataFile="./data/data_1k.json"
    contentType="application/json"
```


```python
!curl -X POST {inference_url} -H "Authorization: Bearer {token}" -H "Content-Type:{contentType}" --data @{dataFile} > curl_response.txt
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  735k  100    95  100  735k     52   408k  0:00:01  0:00:01 --:--:--  409k


# Redeploy with a little larger budget 
If you look in the file curl_response.txt, you will see that the inference failed:
> upstream connect error or disconnect/reset before headers. reset reason: connection termination

Even though a single inference passed, submitted a larger batch of work did not. If this is an expected usage case for
this model, we need to add more memory. Let's do that now.

The following DeploymentConfig is the same as the original, but increases the memory from 150MB to 300MB. This sort
of budget would be available on some network routers.


```python
pipeline.undeploy()
deployment_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(1).memory("300Mi").build()
pipeline.deploy(deployment_config=deployment_config)
```




<table><tr><th>name</th> <td>wjtxedgepipelineexample</td></tr><tr><th>created</th> <td>2023-02-27 17:46:53.711612+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-27 17:47:57.057785+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>2ebda792-0d70-459b-b1cf-0348abcd6a06, e33295e1-fa57-4709-a5ee-23cdd2b66131, 1fc9d0e7-6783-4050-9b86-6f0276fb8745</td></tr><tr><th>steps</th> <td>wjtxalohamodel</td></tr></table>



# Re-run inference
Running the same curl command again should now produce a curl_response.txt file containing the expected results.


```python
!curl -X POST {inference_url} -H "Authorization: Bearer {token}" -H "Content-Type:{contentType}" --data @{dataFile} > curl_response.txt
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100 1402k  100  666k  100  735k   172k   190k  0:00:03  0:00:03 --:--:--  362k


It is important to note that increasing the memory was necessary to run a batch of 1,000 inferences at once. If this is not a design
use case for your system, running with the smaller memory budget may be acceptable. Wallaroo allows you to easily test difference
loading patterns to get a sense for what resources are required with sufficient buffer to allow for robust operation of your system
while not over-provisioning scarce resources.

## Undeploy Pipeline

When finished with our tests, we will undeploy the pipeline so we have the Kubernetes resources back for other tasks.  Note that if the deployment variable is unchanged aloha_pipeline.deploy() will restart the inference engine in the same configuration as before.


```python
pipeline.undeploy()

```




<table><tr><th>name</th> <td>wjtxedgepipelineexample</td></tr><tr><th>created</th> <td>2023-02-27 17:46:53.711612+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-27 17:47:57.057785+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>2ebda792-0d70-459b-b1cf-0348abcd6a06, e33295e1-fa57-4709-a5ee-23cdd2b66131, 1fc9d0e7-6783-4050-9b86-6f0276fb8745</td></tr><tr><th>steps</th> <td>wjtxalohamodel</td></tr></table>


