This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-model-cookbooks/aloha).

## Aloha Demo

In this notebook we will walk through a simple pipeline deployment to inference on a model. For this example we will be using an open source model that uses an [Aloha CNN LSTM model](https://www.researchgate.net/publication/348920204_Using_Auxiliary_Inputs_in_Deep_Learning_Models_for_Detecting_DGA-based_Domain_Names) for classifying Domain names as being either legitimate or being used for nefarious purposes such as malware distribution.  

For our example, we will perform the following:

* Create a workspace for our work.
* Upload the Aloha model.
* Create a pipeline that can ingest our submitted data, submit it to the model, and export the results
* Run a sample inference through our pipeline by loading a file
* Run a sample inference through our pipeline's URL and store the results in a file.

All sample data and models are available through the [Wallaroo Quick Start Guide Samples repository](https://github.com/WallarooLabs/quickstartguide_samples).

## Open a Connection to Wallaroo

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

```python
import wallaroo
from wallaroo.object import EntityNotFoundError

# used to display dataframe information without truncating
import pandas as pd
pd.set_option('display.max_colwidth', None)

# to display dataframe tables
from IPython.display import display
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

if "ARROW_ENABLED" not in os.environ or os.environ["ARROW_ENABLED"] == "False":
    arrowEnabled = False
else:
    arrowEnabled = True
print(arrowEnabled)
```

    True

## Create the Workspace

We will create a workspace to work in and call it the "alohaworkspace", then set it as current workspace environment.  We'll also create our pipeline in advance as `alohapipeline`.  The model name and the model file will be specified for use in later steps.

To allow this tutorial to be run multiple times or by multiple users in the same Wallaroo instance, a random 4 character prefix will be added to the workspace, pipeline, and model.

```python
import string
import random

# make a random 4 character prefix
prefix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))
workspace_name = f'{prefix}alohaworkspace'
pipeline_name = f'{prefix}alohapipeline'
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

```python
wl.list_workspaces()
```

<table>
    <tr>
        <th>Name</th>
        <th>Created At</th>
        <th>Users</th>
        <th>Models</th>
        <th>Pipelines</th>
    </tr>

<tr >
    <td>john.hummel@wallaroo.ai - Default Workspace</td>
    <td>2023-02-27 17:18:34</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>0</td>
    <td>0</td>
</tr>

<tr >
    <td>jnhcccfraudworkspace</td>
    <td>2023-02-27 17:19:26</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>beticcfraudworkspace</td>
    <td>2023-02-27 17:23:26</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>uuzmhotswapworkspace</td>
    <td>2023-02-27 17:28:00</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>2</td>
    <td>1</td>
</tr>

<tr >
    <td>ggwzhotswapworkspace</td>
    <td>2023-02-27 17:33:52</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>2</td>
    <td>1</td>
</tr>

<tr >
    <td>azwsedgeworkspaceexample</td>
    <td>2023-02-27 17:37:33</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>mlbaedgeworkspaceexample</td>
    <td>2023-02-27 17:40:11</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>urldemoworkspace</td>
    <td>2023-02-27 17:55:11</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>efxvtagtestworkspace</td>
    <td>2023-02-27 18:02:16</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

</table>
{{</table>}}

```python
workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)

aloha_pipeline = get_pipeline(pipeline_name)
aloha_pipeline
```

<table><tr><th>name</th> <td>uxelalohapipeline</td></tr><tr><th>created</th> <td>2023-02-27 18:07:39.988279+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-27 18:07:39.988279+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>c7a45891-9797-4522-8ef9-fbd153bfa6c4</td></tr><tr><th>steps</th> <td></td></tr></table>
{{</table>}}

We can verify the workspace is created the current default workspace with the `get_current_workspace()` command.

```python
wl.get_current_workspace()
```

    {'name': 'uxelalohaworkspace', 'id': 16, 'archived': False, 'created_by': '435da905-31e2-4e74-b423-45c38edb5889', 'created_at': '2023-02-27T18:07:39.250191+00:00', 'models': [], 'pipelines': [{'name': 'uxelalohapipeline', 'create_time': datetime.datetime(2023, 2, 27, 18, 7, 39, 988279, tzinfo=tzutc()), 'definition': '[]'}]}

# Upload the Models

Now we will upload our models.  Note that for this example we are applying the model from a .ZIP file.  The Aloha model is a [protobuf](https://developers.google.com/protocol-buffers) file that has been defined for evaluating web pages, and we will configure it to use data in the `tensorflow` format.

```python
model = wl.upload_model(model_name, model_file_name).configure("tensorflow")
```

## Deploy a model

Now that we have a model that we want to use we will create a deployment for it. 

We will tell the deployment we are using a tensorflow model and give the deployment name and the configuration we want for the deployment.

To do this, we'll create our pipeline that can ingest the data, pass the data to our Aloha model, and give us a final output.  We'll call our pipeline `aloha-test-demo`, then deploy it so it's ready to receive data.  The deployment process usually takes about 45 seconds.

* **Note**:  If you receive an error that the pipeline could not be deployed because there are not enough resources, undeploy any other pipelines and deploy this one again.  This command can quickly undeploy all pipelines to regain resources.  We recommend **not** running this command in a production environment since it will cancel any running pipelines:

```python
for p in wl.list_pipelines(): p.undeploy()
```

```python
aloha_pipeline.add_model_step(model)
```

<table><tr><th>name</th> <td>uxelalohapipeline</td></tr><tr><th>created</th> <td>2023-02-27 18:07:39.988279+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-27 18:07:39.988279+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>c7a45891-9797-4522-8ef9-fbd153bfa6c4</td></tr><tr><th>steps</th> <td></td></tr></table>
{{</table>}}

```python
aloha_pipeline.deploy()
```

<table><tr><th>name</th> <td>uxelalohapipeline</td></tr><tr><th>created</th> <td>2023-02-27 18:07:39.988279+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-27 18:07:45.684639+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>178f6fd4-ee38-4931-adb1-6a4b48f6a473, c7a45891-9797-4522-8ef9-fbd153bfa6c4</td></tr><tr><th>steps</th> <td>uxelalohamodel</td></tr></table>
{{</table>}}

We can verify that the pipeline is running and list what models are associated with it.

```python
aloha_pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.0.41',
       'name': 'engine-74fb5d899d-q885n',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'uxelalohapipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'uxelalohamodel',
          'version': '3f354fb2-4220-4873-9418-6debc06ada57',
          'sha': 'd71d9ffc61aaac58c2b1ed70a2db13d1416fb9d3f5b891e5e4e2e97180fe22f8',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.2.16',
       'name': 'engine-lb-ddd995646-j5nr2',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}

## Interferences

### Infer 1 row

Now that the pipeline is deployed and our Aloha model is in place, we'll perform a smoke test to verify the pipeline is up and running properly.  We'll use the `infer_from_file` command to load a single encoded URL into the inference engine and print the results back out.

The result should tell us that the tokenized URL is legitimate (0) or fraud (1).  This sample data should return close to 0.

```python
if arrowEnabled is True:
    result = aloha_pipeline.infer_from_file('./data/data_1.df.json')
else:
    result = aloha_pipeline.infer_from_file("./data/data_1.json")
display(result)
```

{{<table "table table-bordered">}}
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.text_input</th>
      <th>out.matsnu</th>
      <th>out.simda</th>
      <th>out.locky</th>
      <th>out.cryptolocker</th>
      <th>out.qakbot</th>
      <th>out.main</th>
      <th>out.corebot</th>
      <th>out.pykspa</th>
      <th>out.banjori</th>
      <th>out.gozi</th>
      <th>out.dircrypt</th>
      <th>out.ramdo</th>
      <th>out.kraken</th>
      <th>out.ramnit</th>
      <th>out.suppobox</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-02-27 18:07:58.248</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 16, 32, 23, 29, 32, 30, 19, 26, 17]</td>
      <td>[0.010341615]</td>
      <td>[1.793378e-26]</td>
      <td>[0.011029283]</td>
      <td>[0.012099565]</td>
      <td>[0.016155062]</td>
      <td>[0.997564]</td>
      <td>[0.9829148]</td>
      <td>[0.008038961]</td>
      <td>[0.0015195857]</td>
      <td>[2.0289372e-05]</td>
      <td>[4.7591297e-05]</td>
      <td>[0.006236233]</td>
      <td>[0.0003197726]</td>
      <td>[0.0009985751]</td>
      <td>[1.3889951e-27]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}


### Batch Inference

Now that our smoke test is successful, let's really give it some data.  We have two inference files we can use:

* `data_1k.json`:  Contains 10,000 inferences
* `data_25k.json`: Contains 25,000 inferences

We'll pipe the `data_25k.json` file through the `aloha_pipeline` deployment URL, and place the results in a file named `response.txt`.  We'll also display the time this takes.  Note that for larger batches of 50,000 inferences or more can be difficult to view in Jupyter Hub because of its size.

* **IMPORTANT NOTE**:  The `_deployment._url()` method will return an **internal** URL when using Python commands from within the Wallaroo instance - for example, the Wallaroo JupyterHub service.  When connecting via an external connection, `_deployment._url()` returns an **external** URL.  External URL connections requires [the authentication be included in the HTTP request](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/), and that [Model Endpoints Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-model-endpoints-guide/) external endpoints are enabled in the Wallaroo configuration options.

```python
inference_url = aloha_pipeline._deployment._url()
inference_url
```

    'https://doc-test.api.wallaroocommunity.ninja/v1/api/pipelines/infer/uxelalohapipeline-13'

```python
connection =wl.mlops().__dict__
token = connection['token']
token
```

    'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJrTzQ2VjhoQWZDZTBjWU1ETkZobEZWS25HSC1HZy1xc1JkSlhwTTNQYjBJIn0.eyJleHAiOjE2Nzc1MjEyOTMsImlhdCI6MTY3NzUyMTIzMywiYXV0aF90aW1lIjoxNjc3NTE4MzEyLCJqdGkiOiI4YTNjODgwOC04MDY5LTQxOTItYTIyYi1iYzZlNWQ3ZGQ5YjUiLCJpc3MiOiJodHRwczovL2RvYy10ZXN0LmtleWNsb2FrLndhbGxhcm9vY29tbXVuaXR5Lm5pbmphL2F1dGgvcmVhbG1zL21hc3RlciIsImF1ZCI6WyJtYXN0ZXItcmVhbG0iLCJhY2NvdW50Il0sInN1YiI6IjQzNWRhOTA1LTMxZTItNGU3NC1iNDIzLTQ1YzM4ZWRiNTg4OSIsInR5cCI6IkJlYXJlciIsImF6cCI6InNkay1jbGllbnQiLCJzZXNzaW9uX3N0YXRlIjoiYWNlMWEzMGQtNjZiYy00NGQ5LWJkMGEtYzYyMzc0NzhmZGFhIiwiYWNyIjoiMCIsInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJkZWZhdWx0LXJvbGVzLW1hc3RlciIsIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iXX0sInJlc291cmNlX2FjY2VzcyI6eyJtYXN0ZXItcmVhbG0iOnsicm9sZXMiOlsibWFuYWdlLXVzZXJzIiwidmlldy11c2VycyIsInF1ZXJ5LWdyb3VwcyIsInF1ZXJ5LXVzZXJzIl19LCJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6InByb2ZpbGUgZW1haWwiLCJzaWQiOiJhY2UxYTMwZC02NmJjLTQ0ZDktYmQwYS1jNjIzNzQ3OGZkYWEiLCJlbWFpbF92ZXJpZmllZCI6ZmFsc2UsImh0dHBzOi8vaGFzdXJhLmlvL2p3dC9jbGFpbXMiOnsieC1oYXN1cmEtdXNlci1pZCI6IjQzNWRhOTA1LTMxZTItNGU3NC1iNDIzLTQ1YzM4ZWRiNTg4OSIsIngtaGFzdXJhLWRlZmF1bHQtcm9sZSI6InVzZXIiLCJ4LWhhc3VyYS1hbGxvd2VkLXJvbGVzIjpbInVzZXIiXSwieC1oYXN1cmEtdXNlci1ncm91cHMiOiJ7fSJ9LCJuYW1lIjoiSm9obiBIYW5zYXJpY2siLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJqb2huLmh1bW1lbEB3YWxsYXJvby5haSIsImdpdmVuX25hbWUiOiJKb2huIiwiZmFtaWx5X25hbWUiOiJIYW5zYXJpY2siLCJlbWFpbCI6ImpvaG4uaHVtbWVsQHdhbGxhcm9vLmFpIn0.I3cIyxGf9z-N8ZnuiscyWuT-mV80LJOqXYzARL2EM0JRY_lxqfMr_PO_toOgJCfXDnDI5dTqaltes7kBmrTbyynIAKTDvY566RaUJGR2u0l3wjFm6ImJy6Eu78ck7q0bCLKOJkDNSqBPwDJv9b71uW816GRTYdGYUpmtUULiLYH8y3RBGf3odhIuGeWaTwa3PxG1Affq7rNqGG5LWYHvoRoN4-4eAtu3L5jdfC2wmc2MRh3MNK-UPMx3Fiz3r4GoTiSpsNyH6vmLFNUq1Rd-dKTDWu8UNtOihB-tOqJkmcMr2YINgh5PMKtKEpXMIa2kTNvRpNOtfsik0ZagUMxA9g'

```python
if arrowEnabled is True:
    dataFile="./data/data_25k.df.json"
    contentType="application/json; format=pandas-records"
else:
    dataFile="./data/data_25k.json"
    contentType="application/json"
```

```python
!curl -X POST {inference_url} -H "Authorization: Bearer {token}" -H "Content-Type:{contentType}" --data @{dataFile} > curl_response.txt
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100 34.4M  100 16.3M  100 18.0M  1161k  1278k  0:00:14  0:00:14 --:--:-- 4077k

## Undeploy Pipeline

When finished with our tests, we will undeploy the pipeline so we have the Kubernetes resources back for other tasks.  Note that if the deployment variable is unchanged aloha_pipeline.deploy() will restart the inference engine in the same configuration as before.

```python
aloha_pipeline.undeploy()
```

<table><tr><th>name</th> <td>uxelalohapipeline</td></tr><tr><th>created</th> <td>2023-02-27 18:07:39.988279+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-27 18:07:45.684639+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>178f6fd4-ee38-4931-adb1-6a4b48f6a473, c7a45891-9797-4522-8ef9-fbd153bfa6c4</td></tr><tr><th>steps</th> <td>uxelalohamodel</td></tr></table>
{{</table>}}

