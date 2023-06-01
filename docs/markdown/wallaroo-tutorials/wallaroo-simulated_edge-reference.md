This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-features/simulated_edge).

# Simulated Edge Demo

This notebook will explore "Edge ML", meaning deploying a model intended to be run on "the edge". What is "the edge"?  This is typically defined as a resource (CPU, memory, and/or bandwidth) constrained environment or where a combination of latency requirements and bandwidth available requires the models to run locally.

Wallaroo provides two key capabilities when it comes to deploying models to edge devices:

1. Since the same engine is used in both environments, the model behavior can often be simulated accurately using Wallaroo in a data center for testing prior to deployment.
2. Wallaroo makes edge deployments "observable" so the same tools used to monitor model performance can be used in both kinds of deployments. 

This notebook closely parallels the [Aloha tutorial](https://docs.wallaroo.ai/wallaroo-tutorials/wallaroo-quick-start-aloha/).  The primary difference is instead of provide ample resources to a pipeline to allow high-throughput operation we will specify a resource budget matching what is expected in the final deployment. Then we can apply the expected load to the model and observe how it behaves given the available resources.

This example uses the open source [Aloha CNN LSTM model](https://www.researchgate.net/publication/348920204_Using_Auxiliary_Inputs_in_Deep_Learning_Models_for_Detecting_DGA-based_Domain_Names) for classifying Domain names as being either legitimate or being used for nefarious purposes such as malware distribution. This could be deployed on a network router to detect suspicious domains in real-time. Of course, it is important to monitor the behavior of the model across all of the deployments so we can see if the detect rate starts to drift over time.

Note that this example is not intended for production use and is meant of an example of running Wallaroo in a restrained environment.  The environment is based on the [Wallaroo AWS EC2 Setup guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-install-guides/wallaroo-community-install-guides/wallaroo-setup-environment-community/wallaroo-aws-vm-community-setup/).

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

```python
import wallaroo
from wallaroo.object import EntityNotFoundError

# to display dataframe tables
from IPython.display import display
# used to display dataframe information without truncating
import pandas as pd
pd.set_option('display.max_colwidth', None)
import pyarrow as pa
```

### Connect to the Wallaroo Instance

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
# Login through local Wallaroo instance

wl = wallaroo.Client()
```

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
        pipeline = wl.pipelines_by_name(name)[0]
    except EntityNotFoundError:
        pipeline = wl.build_pipeline(name)
    return pipeline
```

### Create or Set the Workspace

Create the workspace and set it as our default workspace.  If a workspace by the same name already exists, then that workspace will be used.

```python
workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)
workspace
```

    {'name': 'gobtedgeworkspaceexample', 'id': 21, 'archived': False, 'created_by': '028c8b48-c39b-4578-9110-0b5bdd3824da', 'created_at': '2023-05-17T21:50:10.05059+00:00', 'models': [], 'pipelines': []}

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

<table><tr><th>name</th> <td>gobtedgepipelineexample</td></tr><tr><th>created</th> <td>2023-05-17 21:50:13.166628+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 21:50:13.166628+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>9efda57b-c18b-4ebb-9681-33647e7d7e66</td></tr><tr><th>steps</th> <td></td></tr></table>

```python
pipeline.deploy(deployment_config=deployment_config)
```

<table><tr><th>name</th> <td>gobtedgepipelineexample</td></tr><tr><th>created</th> <td>2023-05-17 21:50:13.166628+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 21:50:14.868118+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>5cf788a6-50ff-471f-a3ee-4bfdc24def34, 9efda57b-c18b-4ebb-9681-33647e7d7e66</td></tr><tr><th>steps</th> <td>gobtalohamodel</td></tr></table>

We can verify that the pipeline is running and list what models are associated with it.

```python
pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.3.150',
       'name': 'engine-7c78c78bb8-lrhb9',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'gobtedgepipelineexample',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'gobtalohamodel',
          'version': '969b91cb-1cef-49c5-9292-36af48e494b5',
          'sha': 'd71d9ffc61aaac58c2b1ed70a2db13d1416fb9d3f5b891e5e4e2e97180fe22f8',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.4.182',
       'name': 'engine-lb-584f54c899-757hh',
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
      <td>2023-05-17 21:50:26.790</td>
      <td>[0.997564]</td>
    </tr>
  </tbody>
</table>

* **IMPORTANT NOTE**:  The `_deployment._url()` method will return an **internal** URL when using Python commands from within the Wallaroo instance - for example, the Wallaroo JupyterHub service.  When connecting via an external connection, `_deployment._url()` returns an **external** URL.  External URL connections requires [the authentication be included in the HTTP request](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/), and that [Model Endpoints Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-model-endpoints-guide/) external endpoints are enabled in the Wallaroo configuration options.

```python
inference_url = pipeline._deployment._url()
display(inference_url)
connection =wl.mlops().__dict__
token = connection['token']
```

    'https://doc-test.api.wallarooexample.ai/v1/api/pipelines/infer/gobtedgepipelineexample-23/gobtedgepipelineexample'

```python
dataFile="./data/data_1k.arrow"
contentType="application/vnd.apache.arrow.file"
```

```python
!curl -X POST {inference_url} -H "Authorization: Bearer {token}" -H "Content-Type:{contentType}" --data-binary @{dataFile} > curl_response.df
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  196k  100    95  100  196k     90   187k  0:00:01  0:00:01 --:--:--  190k

# Redeploy with a little larger budget 

If you look in the file curl_response.df, you will see that the inference failed:
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

<table><tr><th>name</th> <td>gobtedgepipelineexample</td></tr><tr><th>created</th> <td>2023-05-17 21:50:13.166628+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 21:51:06.928374+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>dc0238e7-f3e3-4579-9a63-24902cb3e3bd, 5cf788a6-50ff-471f-a3ee-4bfdc24def34, 9efda57b-c18b-4ebb-9681-33647e7d7e66</td></tr><tr><th>steps</th> <td>gobtalohamodel</td></tr></table>

# Re-run inference
Running the same curl command again should now produce a curl_response.txt file containing the expected results.

```python
connection =wl.mlops().__dict__
token = connection['token']
print(f'curl -X POST {inference_url} -H "Authorization: Bearer {token}" -H "Content-Type:{contentType}" --data-binary @{dataFile} > curl_response.df')
```

    curl -X POST https://doc-test.api.wallarooexample.ai/v1/api/pipelines/infer/gobtedgepipelineexample-23/gobtedgepipelineexample -H "Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJDYkFqN19QY0xCWTFkWmJiUDZ6Q3BsbkNBYTd6US0tRHlyNy0yLXlQb25nIn0.eyJleHAiOjE2ODQzNjAzMTksImlhdCI6MTY4NDM2MDI1OSwiYXV0aF90aW1lIjoxNjg0MzU1OTU5LCJqdGkiOiI1ZjU4NTQ2Yy1lOTVlLTQ5YjktODgyYS0zYWMxMzgxYzdkODYiLCJpc3MiOiJodHRwczovL2RvYy10ZXN0LmtleWNsb2FrLndhbGxhcm9vY29tbXVuaXR5Lm5pbmphL2F1dGgvcmVhbG1zL21hc3RlciIsImF1ZCI6WyJtYXN0ZXItcmVhbG0iLCJhY2NvdW50Il0sInN1YiI6IjAyOGM4YjQ4LWMzOWItNDU3OC05MTEwLTBiNWJkZDM4MjRkYSIsInR5cCI6IkJlYXJlciIsImF6cCI6InNkay1jbGllbnQiLCJzZXNzaW9uX3N0YXRlIjoiMGJlODJjN2ItNzg1My00ZjVkLWJiNWEtOTlkYjUwYjhiNDVmIiwiYWNyIjoiMCIsInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJkZWZhdWx0LXJvbGVzLW1hc3RlciIsIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iXX0sInJlc291cmNlX2FjY2VzcyI6eyJtYXN0ZXItcmVhbG0iOnsicm9sZXMiOlsibWFuYWdlLXVzZXJzIiwidmlldy11c2VycyIsInF1ZXJ5LWdyb3VwcyIsInF1ZXJ5LXVzZXJzIl19LCJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6InByb2ZpbGUgZW1haWwiLCJzaWQiOiIwYmU4MmM3Yi03ODUzLTRmNWQtYmI1YS05OWRiNTBiOGI0NWYiLCJlbWFpbF92ZXJpZmllZCI6ZmFsc2UsImh0dHBzOi8vaGFzdXJhLmlvL2p3dC9jbGFpbXMiOnsieC1oYXN1cmEtdXNlci1pZCI6IjAyOGM4YjQ4LWMzOWItNDU3OC05MTEwLTBiNWJkZDM4MjRkYSIsIngtaGFzdXJhLWRlZmF1bHQtcm9sZSI6InVzZXIiLCJ4LWhhc3VyYS1hbGxvd2VkLXJvbGVzIjpbInVzZXIiXSwieC1oYXN1cmEtdXNlci1ncm91cHMiOiJ7fSJ9LCJuYW1lIjoiSm9obiBIYW5zYXJpY2siLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJqb2huLmh1bW1lbEB3YWxsYXJvby5haSIsImdpdmVuX25hbWUiOiJKb2huIiwiZmFtaWx5X25hbWUiOiJIYW5zYXJpY2siLCJlbWFpbCI6ImpvaG4uaHVtbWVsQHdhbGxhcm9vLmFpIn0.j86GU-Zi07DvuMnOi1iz8G7ySEL_GeC0A-ol0oI1-X_OdncCpuYBcJWBnf6w66xWkl3oi3-1eHWFcQkPG7W-pNaYW00oYR2o5vBd18_iHWeMTSOeW6ooooseDeGzmk88j9Z02C517fFjHPG1WB_EB1L12cB0PzBOWjoQu9o2tXpSDx8zjP0A-AQZWx5_itrOrMcSwffq3KNgzIscrVjSY4rcin_c5bdZkTvrKeW8uG9wHGyVN_BSVyceTeXqD21oDUmIvnYVDZyx9gmDytWtp43ahX_qHaV7chWOfnaTcd4e4_mAotcLP_PjfptushhanhSfWty1z1b5xv0ut3SxUQ" -H "Content-Type:application/vnd.apache.arrow.file" --data-binary @./data/data_1k.arrow > curl_response.df

```python
!curl -X POST {inference_url} -H "Authorization: Bearer {token}" -H "Content-Type:{contentType}" --data-binary @{dataFile} > curl_response.df
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100 1045k  100  849k  100  196k   411k  97565  0:00:02  0:00:02 --:--:--  512k

It is important to note that increasing the memory was necessary to run a batch of 1,000 inferences at once. If this is not a design
use case for your system, running with the smaller memory budget may be acceptable. Wallaroo allows you to easily test difference
loading patterns to get a sense for what resources are required with sufficient buffer to allow for robust operation of your system
while not over-provisioning scarce resources.

## Undeploy Pipeline

When finished with our tests, we will undeploy the pipeline so we have the Kubernetes resources back for other tasks.  Note that if the deployment variable is unchanged aloha_pipeline.deploy() will restart the inference engine in the same configuration as before.

```python
pipeline.undeploy()
```

<table><tr><th>name</th> <td>gobtedgepipelineexample</td></tr><tr><th>created</th> <td>2023-05-17 21:50:13.166628+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 21:51:06.928374+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>dc0238e7-f3e3-4579-9a63-24902cb3e3bd, 5cf788a6-50ff-471f-a3ee-4bfdc24def34, 9efda57b-c18b-4ebb-9681-33647e7d7e66</td></tr><tr><th>steps</th> <td>gobtalohamodel</td></tr></table>

