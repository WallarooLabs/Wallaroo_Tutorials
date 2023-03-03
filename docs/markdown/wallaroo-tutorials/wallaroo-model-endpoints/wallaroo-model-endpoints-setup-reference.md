This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-features/wallaroo-model-endpoints).

## Internal Pipeline Inference URL Tutorial

Wallaroo provides the ability to perform inferences through deployed pipelines via both internal and external inference URLs.  These inference URLs allow inferences to be performed by submitting data to the internal or external URL with the inference results returned in the same format as the [InferenceResult Object](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/#run-inference-through-a-pipeline).

**Internal URLs** are available only through the internal Kubernetes environment hosting the Wallaroo instance as demonstrated in this tutorial.
**External URLs** are available outside of the Kubernetes environment, such as the public internet.  These are demonstrated in the External Pipeline Deployment URL Tutorial.

The following tutorial shows how to set up an environment and demonstrates how to use the Internal Deployment URL.  This example provides the following:

* `alohacnnlstm.zip`:  Aloha model used as part of the [Aloha Quick Tutorial](https://docs.wallaroo.ai/wallaroo-tutorials/wallaroo-quick-start-aloha/).
* For Arrow enabled instances:
  * `data_1.df.json`, `data_1k.df.json` and `data_25k.df.json`:  Sample data used for testing inferences with the sample model.
* For Arrow distabled instances:
  * `data_1.json`, `data_1k.json` and `data_25k.json`:  Sample data used for testing inferences with the sample model.

For our example, we will perform the following:

* Create a workspace for our work.
* Upload the Aloha model.
* Create a pipeline that can ingest our submitted data, submit it to the model, and export the results.
* Run a sample inference through our pipeline via the SDK to demonstrate the inference is accurate.
* Run a sample inference through our pipeline's Internal URL and store the results in a file.

All sample data and models are available through the [Wallaroo Quick Start Guide Samples repository](https://github.com/WallarooLabs/quickstartguide_samples).

## Open a Connection to Wallaroo

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.


```python
import wallaroo
from wallaroo.object import EntityNotFoundError
import pandas as pd

# used to display dataframe information without truncating
from IPython.display import display
pd.set_option('display.max_colwidth', None)
```


```python
# Client connection from local Wallaroo instance

wl = wallaroo.Client()

# SSO login through keycloak

# wallarooPrefix = "YOUR PREFIX"
# wallarooSuffix = "YOUR SUFFIX"

# wallarooPrefix = "doc-test"
# wallarooSuffix = "wallaroocommunity.ninja"

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


## Create the Workspace

We will create a workspace to work in and call it the `urldemoworkspace`, then set it as current workspace environment.  We'll also create our pipeline in advance as `urldemopipeline`.

The model to be uploaded and used for inference will be labeled as `urldemomodel`.  Modify these to your organizations requirements.

Once complete, the workspace will be created or, if already existing, set to the current workspace to host the pipelines and models.


```python
workspace_name = 'urldemoworkspace'
pipeline_name = 'urldemopipeline'
model_name = 'urldemomodel'
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
workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)

pipeline = get_pipeline(pipeline_name)
pipeline
```




<table><tr><th>name</th> <td>urldemopipeline</td></tr><tr><th>created</th> <td>2023-02-27 17:55:12.813456+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-27 17:55:12.813456+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>54158104-c71d-4980-a6a3-25564c909b44</td></tr><tr><th>steps</th> <td></td></tr></table>



We can verify the workspace is created the current default workspace with the `get_current_workspace()` command.


```python
wl.get_current_workspace()
```




    {'name': 'urldemoworkspace', 'id': 14, 'archived': False, 'created_by': '435da905-31e2-4e74-b423-45c38edb5889', 'created_at': '2023-02-27T17:55:11.802586+00:00', 'models': [], 'pipelines': [{'name': 'urldemopipeline', 'create_time': datetime.datetime(2023, 2, 27, 17, 55, 12, 813456, tzinfo=tzutc()), 'definition': '[]'}]}



# Upload the Models

Now we will upload our models.  Note that for this example we are applying the model from a .ZIP file.  The Aloha model is a [protobuf](https://developers.google.com/protocol-buffers) file that has been defined for evaluating web pages, and we will configure it to use data in the `tensorflow` format.


```python
model = wl.upload_model(model_name, model_file_name).configure("tensorflow")
```

## Deploy The Pipeline
Now that we have a model that we want to use we will create a deployment for it. 

We will tell the deployment we are using a tensorflow model and give the deployment name and the configuration we want for the deployment.


```python
pipeline.add_model_step(model)
pipeline.deploy()
```




<table><tr><th>name</th> <td>urldemopipeline</td></tr><tr><th>created</th> <td>2023-02-27 17:55:12.813456+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-27 17:56:25.368424+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>930fe54d-9503-4768-8bf9-499f72272098, 54158104-c71d-4980-a6a3-25564c909b44</td></tr><tr><th>steps</th> <td>urldemomodel</td></tr></table>



We can verify that the pipeline is running and list what models are associated with it.


```python
pipeline.status()
```




    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.0.37',
       'name': 'engine-85c895dbbf-tfq4r',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'urldemopipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'urldemomodel',
          'version': 'a4a80d9f-dcfe-419d-becd-dbab31b65904',
          'sha': 'd71d9ffc61aaac58c2b1ed70a2db13d1416fb9d3f5b891e5e4e2e97180fe22f8',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.1.12',
       'name': 'engine-lb-ddd995646-p4nb2',
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
      <th>out.gozi</th>
      <th>out.cryptolocker</th>
      <th>out.pykspa</th>
      <th>out.kraken</th>
      <th>out.locky</th>
      <th>out.corebot</th>
      <th>out.ramdo</th>
      <th>out.suppobox</th>
      <th>out.simda</th>
      <th>out.matsnu</th>
      <th>out.banjori</th>
      <th>out.main</th>
      <th>out.ramnit</th>
      <th>out.dircrypt</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-02-27 17:57:33.165</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 16, 32, 23, 29, 32, 30, 19, 26, 17]</td>
      <td>[0.016155062]</td>
      <td>[2.0289372e-05]</td>
      <td>[0.012099565]</td>
      <td>[0.008038961]</td>
      <td>[0.0003197726]</td>
      <td>[0.011029283]</td>
      <td>[0.9829148]</td>
      <td>[0.006236233]</td>
      <td>[1.3889951e-27]</td>
      <td>[1.793378e-26]</td>
      <td>[0.010341615]</td>
      <td>[0.0015195857]</td>
      <td>[0.997564]</td>
      <td>[0.0009985751]</td>
      <td>[4.7591297e-05]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


### Batch Inference

Now that our smoke test is successful, we will retrieve the Internal Deployment URL and perform an inference by submitting our data through a `curl` command as detailed below.

* **IMPORTANT NOTE**:  The `_deployment._url()` method will return an **internal** URL when using Python commands from within the Wallaroo instance - for example, the Wallaroo JupyterHub service.  When connecting via an external connection, `_deployment._url()` returns an **external** URL.  External URL connections requires [the authentication be included in the HTTP request](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/), and that [Model Endpoints Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-model-endpoints-guide/) external endpoints are enabled in the Wallaroo configuration options.


```python
inference_url = pipeline._deployment._url()
print(inference_url)
connection =wl.mlops().__dict__
token = connection['token']
print(token)
```

    https://doc-test.api.wallaroocommunity.ninja/v1/api/pipelines/infer/urldemopipeline-11
    eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJrTzQ2VjhoQWZDZTBjWU1ETkZobEZWS25HSC1HZy1xc1JkSlhwTTNQYjBJIn0.eyJleHAiOjE2Nzc1MjA3MDIsImlhdCI6MTY3NzUyMDY0MiwiYXV0aF90aW1lIjoxNjc3NTE4MzEyLCJqdGkiOiI1NTdkZDAxYi1jNTVkLTQ0MDQtYTI5ZC01MmRlOWU0MTc2NTciLCJpc3MiOiJodHRwczovL2RvYy10ZXN0LmtleWNsb2FrLndhbGxhcm9vY29tbXVuaXR5Lm5pbmphL2F1dGgvcmVhbG1zL21hc3RlciIsImF1ZCI6WyJtYXN0ZXItcmVhbG0iLCJhY2NvdW50Il0sInN1YiI6IjQzNWRhOTA1LTMxZTItNGU3NC1iNDIzLTQ1YzM4ZWRiNTg4OSIsInR5cCI6IkJlYXJlciIsImF6cCI6InNkay1jbGllbnQiLCJzZXNzaW9uX3N0YXRlIjoiYWNlMWEzMGQtNjZiYy00NGQ5LWJkMGEtYzYyMzc0NzhmZGFhIiwiYWNyIjoiMCIsInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJkZWZhdWx0LXJvbGVzLW1hc3RlciIsIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iXX0sInJlc291cmNlX2FjY2VzcyI6eyJtYXN0ZXItcmVhbG0iOnsicm9sZXMiOlsibWFuYWdlLXVzZXJzIiwidmlldy11c2VycyIsInF1ZXJ5LWdyb3VwcyIsInF1ZXJ5LXVzZXJzIl19LCJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6InByb2ZpbGUgZW1haWwiLCJzaWQiOiJhY2UxYTMwZC02NmJjLTQ0ZDktYmQwYS1jNjIzNzQ3OGZkYWEiLCJlbWFpbF92ZXJpZmllZCI6ZmFsc2UsImh0dHBzOi8vaGFzdXJhLmlvL2p3dC9jbGFpbXMiOnsieC1oYXN1cmEtdXNlci1pZCI6IjQzNWRhOTA1LTMxZTItNGU3NC1iNDIzLTQ1YzM4ZWRiNTg4OSIsIngtaGFzdXJhLWRlZmF1bHQtcm9sZSI6InVzZXIiLCJ4LWhhc3VyYS1hbGxvd2VkLXJvbGVzIjpbInVzZXIiXSwieC1oYXN1cmEtdXNlci1ncm91cHMiOiJ7fSJ9LCJuYW1lIjoiSm9obiBIYW5zYXJpY2siLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJqb2huLmh1bW1lbEB3YWxsYXJvby5haSIsImdpdmVuX25hbWUiOiJKb2huIiwiZmFtaWx5X25hbWUiOiJIYW5zYXJpY2siLCJlbWFpbCI6ImpvaG4uaHVtbWVsQHdhbGxhcm9vLmFpIn0.QTTxK6rE-SZIBR7z7hN1aIsGSsPYmBmuI-KxsmwzATjrCtOLO3ObE5YtBye3ITXkG4NAVN3c2llTSzrYLDeBMcsz17_T8UdSpwOVsDeTko-muhzQkcMnUrXGLsJDiOofS3ZT-_S66-IrCfGUD2D1Gj7ufbAnMipyTuE69L1QBEdoszcRfTR-epCqniayB3s6SkhBSgjmgvJcmMSIHxj3zg0siZAjQoxM6_E5GO_o__91p7FiADa0FH3xCmT9iOMM1NcF7FheBNX7xCXBBWekiy9bpB0BQISvMi1IcVCGeMZnTyO1o9ZgFbV5MG-SoKFyZrYUmhBf-JoRjecv1FYgIg



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
    100 34.3M  100 16.3M  100 18.0M   895k   988k  0:00:18  0:00:18 --:--:--  538k


## Undeploy Pipeline

When finished with our tests, we will undeploy the pipeline so we have the Kubernetes resources back for other tasks.

**IMPORTANT NOTE**:  For the External Pipeline Deployment URL Tutorial, this pipeline will have to be deployed to make the External Deployment URL available.


```python
pipeline.undeploy()
```




<table><tr><th>name</th> <td>urldemopipeline</td></tr><tr><th>created</th> <td>2023-02-27 17:55:12.813456+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-27 17:56:25.368424+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>930fe54d-9503-4768-8bf9-499f72272098, 54158104-c71d-4980-a6a3-25564c909b44</td></tr><tr><th>steps</th> <td>urldemomodel</td></tr></table>


