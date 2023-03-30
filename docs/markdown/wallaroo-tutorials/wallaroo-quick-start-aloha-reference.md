This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-model-cookbooks/aloha).

## Aloha Demo

In this notebook we will walk through a simple pipeline deployment to inference on a model. For this example we will be using an open source model that uses an [Aloha CNN LSTM model](https://www.researchgate.net/publication/348920204_Using_Auxiliary_Inputs_in_Deep_Learning_Models_for_Detecting_DGA-based_Domain_Names) for classifying Domain names as being either legitimate or being used for nefarious purposes such as malware distribution.  

## Prerequisites

* An installed Wallaroo instance.
* The following Python libraries installed:
  * `os`
  * [`wallaroo`](https://pypi.org/project/wallaroo/): The Wallaroo SDK. Included with the Wallaroo JupyterHub service by default.
  * [`pandas`](https://pypi.org/project/pandas/): Pandas, mainly used for Pandas DataFrame
  * [`pyarrow`](https://pypi.org/project/pyarrow/): PyArrow for Apache Arrow support
  * [`polars`](https://pypi.org/project/polars/): Polars for DataFrame with native Apache Arrow support

## Tutorial Goals

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

# to display dataframe tables
from IPython.display import display
# used to display dataframe information without truncating
import pandas as pd
pd.set_option('display.max_colwidth', None)
import pyarrow as pa

import polars

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
    <td>2023-03-27 21:18:30</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>0</td>
    <td>0</td>
</tr>


<tr >
    <td>cjeuccfraudworkspace</td>
    <td>2023-03-27 21:19:03</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>


<tr >
    <td>abtesting</td>
    <td>2023-03-27 21:23:15</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>2</td>
    <td>1</td>
</tr>


<tr >
    <td>housepricing</td>
    <td>2023-03-27 21:37:22</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>3</td>
    <td>1</td>
</tr>


<tr >
    <td>rnwkalohaworkspace</td>
    <td>2023-03-27 21:56:35</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>


<tr >
    <td>bikedayevalworkspace</td>
    <td>2023-03-27 22:05:03</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>


<tr >
    <td>demandcurveworkspace</td>
    <td>2023-03-27 22:07:56</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>3</td>
    <td>1</td>
</tr>

</table>





```python
workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)

aloha_pipeline = get_pipeline(pipeline_name)
aloha_pipeline
```




<table><tr><th>name</th> <td>zxukalohapipeline</td></tr><tr><th>created</th> <td>2023-03-27 22:12:14.559376+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-27 22:12:14.559376+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>a7cb3cd7-affd-48bf-88fd-543b24fcad43</td></tr><tr><th>steps</th> <td></td></tr></table>



We can verify the workspace is created the current default workspace with the `get_current_workspace()` command.


```python
wl.get_current_workspace()
```




    {'name': 'zxukalohaworkspace', 'id': 11, 'archived': False, 'created_by': '56d97480-bb64-4575-acb6-f93d05652e86', 'created_at': '2023-03-27T22:12:13.863035+00:00', 'models': [], 'pipelines': [{'name': 'zxukalohapipeline', 'create_time': datetime.datetime(2023, 3, 27, 22, 12, 14, 559376, tzinfo=tzutc()), 'definition': '[]'}]}



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




<table><tr><th>name</th> <td>zxukalohapipeline</td></tr><tr><th>created</th> <td>2023-03-27 22:12:14.559376+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-27 22:12:14.559376+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>a7cb3cd7-affd-48bf-88fd-543b24fcad43</td></tr><tr><th>steps</th> <td></td></tr></table>




```python
aloha_pipeline.deploy()
```




<table><tr><th>name</th> <td>zxukalohapipeline</td></tr><tr><th>created</th> <td>2023-03-27 22:12:14.559376+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-27 22:12:19.798012+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>7e4e0f5a-e4cd-4c1c-8dd0-c4e51b17ec2b, a7cb3cd7-affd-48bf-88fd-543b24fcad43</td></tr><tr><th>steps</th> <td>zxukalohamodel</td></tr></table>



We can verify that the pipeline is running and list what models are associated with it.


```python
aloha_pipeline.status()
```




    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.2.25',
       'name': 'engine-8564f999cf-mhlhd',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'zxukalohapipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'zxukalohamodel',
          'version': 'b9f31fc6-d77e-48a9-b33d-1ccbdb8d6654',
          'sha': 'd71d9ffc61aaac58c2b1ed70a2db13d1416fb9d3f5b891e5e4e2e97180fe22f8',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.1.31',
       'name': 'engine-lb-ddd995646-z52jl',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}



## Interferences

### Infer 1 row

Now that the pipeline is deployed and our Aloha model is in place, we'll perform a smoke test to verify the pipeline is up and running properly.  We'll use the `infer_from_file` command to load a single encoded URL into the inference engine and print the results back out.

The result should tell us that the tokenized URL is legitimate (0) or fraud (1).  This sample data should return close to 1 in `out.main`.


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

result = aloha_pipeline.infer(high_fraud)
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
      <td>2023-03-27 22:13:40.234</td>
      <td>[0.997564]</td>
    </tr>
  </tbody>
</table>
</div>


### Infer From File

This time, we'll give it a bigger set of data to infer.  `./data/data_1k.arrow` is an Apache Arrow table with 1,000 records in it.  Once submitted, we'll turn the result into a DataFrame and display the first five results.


```python
result = aloha_pipeline.infer_from_file('./data/data_1k.arrow')
display(result)
```


    pyarrow.Table
    time: timestamp[ms]
    in.text_input: list<item: float> not null
      child 0, item: float
    out.banjori: list<inner: float not null> not null
      child 0, inner: float not null
    out.corebot: list<inner: float not null> not null
      child 0, inner: float not null
    out.cryptolocker: list<inner: float not null> not null
      child 0, inner: float not null
    out.dircrypt: list<inner: float not null> not null
      child 0, inner: float not null
    out.gozi: list<inner: float not null> not null
      child 0, inner: float not null
    out.kraken: list<inner: float not null> not null
      child 0, inner: float not null
    out.locky: list<inner: float not null> not null
      child 0, inner: float not null
    out.main: list<inner: float not null> not null
      child 0, inner: float not null
    out.matsnu: list<inner: float not null> not null
      child 0, inner: float not null
    out.pykspa: list<inner: float not null> not null
      child 0, inner: float not null
    out.qakbot: list<inner: float not null> not null
      child 0, inner: float not null
    out.ramdo: list<inner: float not null> not null
      child 0, inner: float not null
    out.ramnit: list<inner: float not null> not null
      child 0, inner: float not null
    out.simda: list<inner: float not null> not null
      child 0, inner: float not null
    out.suppobox: list<inner: float not null> not null
      child 0, inner: float not null
    check_failures: int8
    ----
    time: [[2023-03-27 22:13:54.719,2023-03-27 22:13:54.719,2023-03-27 22:13:54.719,2023-03-27 22:13:54.719,2023-03-27 22:13:54.719,...,2023-03-27 22:13:54.719,2023-03-27 22:13:54.719,2023-03-27 22:13:54.719,2023-03-27 22:13:54.719,2023-03-27 22:13:54.719]]
    in.text_input: [[[0,0,0,0,0,...,32,30,19,26,17],[0,0,0,0,0,...,29,12,36,31,12],...,[0,0,0,0,0,...,35,16,35,27,16],[0,0,0,0,0,...,24,29,14,36,13]]]
    out.banjori: [[[0.0015195871],[0.00002837503],...,[0.000005631568],[1.3068676e-12]]]
    out.corebot: [[[0.9829148],[0.0000127531175],...,[0.000003364268],[1.1029446e-9]]]
    out.cryptolocker: [[[0.012099565],[0.025435211],...,[0.13612257],[0.014839977]]]
    out.dircrypt: [[[0.000047591344],[6.150943e-10],...,[5.6732376e-11],[2.2757407e-8]]]
    out.gozi: [[[0.000020289392],[2.321783e-10],...,[2.7730579e-8],[8.438471e-15]]]
    out.kraken: [[[0.0003197726],[0.051351093],...,[0.0025221596],[0.30495796]]]
    out.locky: [[[0.011029272],[0.022038758],...,[0.05455696],[0.11627983]]]
    out.main: [[[0.997564],[0.9885122],...,[0.9998954],[0.99999803]]]
    ...



```python
outputs =  result.to_pandas()
display(outputs.loc[:5, ["time","out.main"]])
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
      <td>2023-03-27 22:13:54.719</td>
      <td>[0.997564]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-03-27 22:13:54.719</td>
      <td>[0.9885122]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-03-27 22:13:54.719</td>
      <td>[0.9993358]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-03-27 22:13:54.719</td>
      <td>[0.99999857]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-03-27 22:13:54.719</td>
      <td>[0.9984837]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2023-03-27 22:13:54.719</td>
      <td>[0.99999994]</td>
    </tr>
  </tbody>
</table>
</div>


### Batch Inference

Now that our smoke test is successful, let's really give it some data.  We have two inference files we can use:

* `data_1k.arrow`:  Contains 10,000 inferences
* `data_25k.arrow`: Contains 25,000 inferences

When Apache Arrow tables are submitted to a Wallaroo Pipeline, the inference is processed natively as an Arrow table, and the results are returned as an Arrow table.  This allows for faster data processing than with JSON files or DataFrame objects.

We'll pipe the `data_25k.arrow` file through the `aloha_pipeline` deployment URL, and place the results in a file named `response.arrow`.  We'll also display the time this takes.  Note that for larger batches of 50,000 inferences or more can be difficult to view in Jupyter Hub because of its size, so we'll only display the first five rows.

* **IMPORTANT NOTE**:  The `_deployment._url()` method will return an **internal** URL when using Python commands from within the Wallaroo instance - for example, the Wallaroo JupyterHub service.  When connecting via an external connection, `_deployment._url()` returns an **external** URL.  External URL connections requires [the authentication be included in the HTTP request](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/), and that [Model Endpoints Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-model-endpoints-guide/) external endpoints are enabled in the Wallaroo configuration options.


```python
inference_url = aloha_pipeline._deployment._url()
inference_url
```




    'https://wallaroo.api.example.com/v1/api/pipelines/infer/zxukalohapipeline-8'




```python
connection =wl.mlops().__dict__
token = connection['token']
token
```




    'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJCRFdIZ3Q0WmxRdEIxVDNTTkJ2RjlkYkU3RmxkSWdXRENwb041UkJLeTlrIn0.eyJleHAiOjE2Nzk5NTU2NDgsImlhdCI6MTY3OTk1NTU4OCwiYXV0aF90aW1lIjoxNjc5OTUxOTA3LCJqdGkiOiJiMTJiZWFjMi0yZGQwLTRkMjUtYTI5YS04MDJkOWI3M2IxZDAiLCJpc3MiOiJodHRwczovL2RvYy10ZXN0LmtleWNsb2FrLndhbGxhcm9vY29tbXVuaXR5Lm5pbmphL2F1dGgvcmVhbG1zL21hc3RlciIsImF1ZCI6WyJtYXN0ZXItcmVhbG0iLCJhY2NvdW50Il0sInN1YiI6IjU2ZDk3NDgwLWJiNjQtNDU3NS1hY2I2LWY5M2QwNTY1MmU4NiIsInR5cCI6IkJlYXJlciIsImF6cCI6InNkay1jbGllbnQiLCJzZXNzaW9uX3N0YXRlIjoiNjUzZDUwMmYtNjI4MS00YmE1LTk5NzQtOTlhZDlkY2Y1OWVhIiwiYWNyIjoiMCIsInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJkZWZhdWx0LXJvbGVzLW1hc3RlciIsIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iXX0sInJlc291cmNlX2FjY2VzcyI6eyJtYXN0ZXItcmVhbG0iOnsicm9sZXMiOlsibWFuYWdlLXVzZXJzIiwidmlldy11c2VycyIsInF1ZXJ5LWdyb3VwcyIsInF1ZXJ5LXVzZXJzIl19LCJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6InByb2ZpbGUgZW1haWwiLCJzaWQiOiI2NTNkNTAyZi02MjgxLTRiYTUtOTk3NC05OWFkOWRjZjU5ZWEiLCJlbWFpbF92ZXJpZmllZCI6ZmFsc2UsImh0dHBzOi8vaGFzdXJhLmlvL2p3dC9jbGFpbXMiOnsieC1oYXN1cmEtdXNlci1pZCI6IjU2ZDk3NDgwLWJiNjQtNDU3NS1hY2I2LWY5M2QwNTY1MmU4NiIsIngtaGFzdXJhLWRlZmF1bHQtcm9sZSI6InVzZXIiLCJ4LWhhc3VyYS1hbGxvd2VkLXJvbGVzIjpbInVzZXIiXSwieC1oYXN1cmEtdXNlci1ncm91cHMiOiJ7fSJ9LCJuYW1lIjoiSm9obiBIYW5zYXJpY2siLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJqb2huLmh1bW1lbEB3YWxsYXJvby5haSIsImdpdmVuX25hbWUiOiJKb2huIiwiZmFtaWx5X25hbWUiOiJIYW5zYXJpY2siLCJlbWFpbCI6ImpvaG4uaHVtbWVsQHdhbGxhcm9vLmFpIn0.mYj-HESqbqkD3SAsxj97zw9B9UrBmImJh6VkQDK-q141nr7gvNPc8gd1caGKLCMzYTKFLRtlPyRazdulp7ib8qLl3C02Jb2sONJJB56B2sJFwYVlzZ-84ze9Mc3w14be-7vWLg5t8vSvj3N8Aan9HZne5pPPM3FLfuSnD-JhfQUdhPnEzfxKO48ayA1Aaydlwebw8pUpSJ8RAVp6h5DnOKa57s4rKrzzcacjMHQszYi0Qua6rHz8NF0xwojv9cdfSdztzq4JNT16Eqe1_8lO6Ec11pZDFQAUmJy2l3naj6M6CD6W7cMm7AjSwUDE0g8gidSOo3ajztwoR1WKiJJl5w'




```python
dataFile="./data/data_25k.arrow"
contentType="application/vnd.apache.arrow.file"
```


```python
!curl -X POST {inference_url} -H "Authorization: Bearer {token}" -H "Content-Type:{contentType}" --data-binary @{dataFile} > curl_response.df
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100 23.5M  100 18.7M  100 4874k  2646k   671k  0:00:07  0:00:07 --:--:-- 3999k0 4874k  1944k   769k  0:00:09  0:00:06  0:00:03 2606k



```python
cc_data_from_file =  pd.read_json('./curl_response.df', orient="records")
display(cc_data_from_file.head(5))
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
      <th>in</th>
      <th>out</th>
      <th>check_failures</th>
      <th>metadata</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1679955641115</td>
      <td>{'text_input': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 28.0, 16.0, 32.0, 23.0, 29.0, 32.0, 30.0, 19.0, 26.0, 17.0]}</td>
      <td>{'banjori': [0.0015195871], 'corebot': [0.9829148], 'cryptolocker': [0.012099565000000001], 'dircrypt': [4.7591344e-05], 'gozi': [2.0289392e-05], 'kraken': [0.0003197726], 'locky': [0.011029272000000001], 'main': [0.997564], 'matsnu': [0.010341625], 'pykspa': [0.008038965], 'qakbot': [0.016155062], 'ramdo': [0.006236233000000001], 'ramnit': [0.0009985756], 'simda': [1.793378e-26], 'suppobox': [1.3889898e-27]}</td>
      <td>[]</td>
      <td>{'last_model': '{"model_name":"zxukalohamodel","model_sha":"d71d9ffc61aaac58c2b1ed70a2db13d1416fb9d3f5b891e5e4e2e97180fe22f8"}'}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1679955641115</td>
      <td>{'text_input': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 30.0, 20.0, 19.0, 27.0, 14.0, 17.0, 24.0, 13.0, 23.0, 20.0, 18.0, 35.0, 18.0, 22.0, 23.0]}</td>
      <td>{'banjori': [7.447225e-18], 'corebot': [6.7359245e-08], 'cryptolocker': [0.17081991], 'dircrypt': [1.3220147000000001e-09], 'gozi': [1.2758853e-24], 'kraken': [0.22559536], 'locky': [0.34209844], 'main': [0.99999994], 'matsnu': [0.30801848], 'pykspa': [0.18282163], 'qakbot': [3.8022553999999996e-11], 'ramdo': [0.20622534], 'ramnit': [0.15215826], 'simda': [1.17020745e-30], 'suppobox': [3.1514464999999997e-38]}</td>
      <td>[]</td>
      <td>{'last_model': '{"model_name":"zxukalohamodel","model_sha":"d71d9ffc61aaac58c2b1ed70a2db13d1416fb9d3f5b891e5e4e2e97180fe22f8"}'}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1679955641115</td>
      <td>{'text_input': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 25.0, 33.0, 25.0, 36.0, 25.0, 31.0, 14.0, 32.0, 36.0, 25.0, 12.0, 35.0, 34.0, 30.0, 28.0, 27.0, 24.0, 29.0, 27.0]}</td>
      <td>{'banjori': [2.8599304999999997e-21], 'corebot': [9.302004999999999e-08], 'cryptolocker': [0.04445295], 'dircrypt': [6.1637580000000004e-09], 'gozi': [8.34974e-23], 'kraken': [0.48234479999999996], 'locky': [0.2633289], 'main': [1.0], 'matsnu': [0.29800323], 'pykspa': [0.22361766], 'qakbot': [1.5238920999999999e-06], 'ramdo': [0.3282038], 'ramnit': [0.029332466], 'simda': [1.1995533000000001e-31], 'suppobox': [0.0]}</td>
      <td>[]</td>
      <td>{'last_model': '{"model_name":"zxukalohamodel","model_sha":"d71d9ffc61aaac58c2b1ed70a2db13d1416fb9d3f5b891e5e4e2e97180fe22f8"}'}</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1679955641115</td>
      <td>{'text_input': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 17.0, 23.0, 22.0, 15.0, 12.0, 35.0, 34.0, 36.0, 12.0, 18.0, 24.0, 34.0, 32.0, 36.0, 12.0, 14.0, 16.0, 27.0, 22.0, 23.0]}</td>
      <td>{'banjori': [2.1386805e-15], 'corebot': [3.8817485e-10], 'cryptolocker': [0.045599725], 'dircrypt': [1.9090386e-07], 'gozi': [1.3139924000000002e-25], 'kraken': [0.59542614], 'locky': [0.17374131], 'main': [0.9999996999999999], 'matsnu': [0.2315157], 'pykspa': [0.17591687], 'qakbot': [1.087611e-09], 'ramdo': [0.21832284000000002], 'ramnit': [0.012869288000000001], 'simda': [6.158882e-28], 'suppobox': [1.438591e-35]}</td>
      <td>[]</td>
      <td>{'last_model': '{"model_name":"zxukalohamodel","model_sha":"d71d9ffc61aaac58c2b1ed70a2db13d1416fb9d3f5b891e5e4e2e97180fe22f8"}'}</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1679955641115</td>
      <td>{'text_input': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 32.0, 13.0, 14.0, 12.0, 33.0, 16.0, 23.0, 15.0, 22.0, 30.0, 28.0, 26.0, 12.0, 16.0, 32.0, 37.0, 29.0, 22.0, 28.0, 22.0, 16.0, 27.0, 32.0]}</td>
      <td>{'banjori': [9.453381e-15], 'corebot': [7.091152e-10], 'cryptolocker': [0.049815107000000004], 'dircrypt': [5.2914135e-09], 'gozi': [7.4132087e-19], 'kraken': [1.5504637e-13], 'locky': [1.079181e-15], 'main': [0.9999988999999999], 'matsnu': [1.5003076000000002e-15], 'pykspa': [0.33075709999999997], 'qakbot': [2.6258948e-07], 'ramdo': [0.50362796], 'ramnit': [0.020393757000000002], 'simda': [0.0], 'suppobox': [0.0]}</td>
      <td>[]</td>
      <td>{'last_model': '{"model_name":"zxukalohamodel","model_sha":"d71d9ffc61aaac58c2b1ed70a2db13d1416fb9d3f5b891e5e4e2e97180fe22f8"}'}</td>
    </tr>
  </tbody>
</table>
</div>


## Undeploy Pipeline

When finished with our tests, we will undeploy the pipeline so we have the Kubernetes resources back for other tasks.  Note that if the deployment variable is unchanged aloha_pipeline.deploy() will restart the inference engine in the same configuration as before.


```python
aloha_pipeline.undeploy()
```
