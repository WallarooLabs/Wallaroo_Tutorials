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

### Connect to the Wallaroo Instance

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
```

```python
# Login through local Wallaroo instance

wl = wallaroo.Client()

wallarooPrefix = "doc-test"
wallarooSuffix = "wallarooexample.ai"

wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}.api.{wallarooSuffix}", 
                    auth_endpoint=f"https://{wallarooPrefix}.keycloak.{wallarooSuffix}", 
                    auth_type="sso")
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
        pipeline = wl.pipelines_by_name(name)[0]
    except EntityNotFoundError:
        pipeline = wl.build_pipeline(name)
    return pipeline
```

```python
wl.list_workspaces()
```

{{<table "table table-striped table-bordered" >}}
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
    <td>2023-05-17 20:36:36</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>0</td>
    <td>0</td>
</tr>

<tr >
    <td>housepricedrift</td>
    <td>2023-05-17 20:41:50</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>sdkquickworkspace</td>
    <td>2023-05-17 20:43:36</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>apiworkspaces</td>
    <td>2023-05-17 20:50:36</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>2</td>
    <td>3</td>
</tr>

<tr >
    <td>azuremlsdkworkspace</td>
    <td>2023-05-17 21:01:45</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>databricksazuresdkworkspace</td>
    <td>2023-05-17 21:02:53</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>gcpsdkworkspace</td>
    <td>2023-05-17 21:03:43</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>testautoconversion</td>
    <td>2023-05-17 21:11:40</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>2</td>
    <td>0</td>
</tr>

<tr >
    <td>externalkerasautoconvertworkspace</td>
    <td>2023-05-17 21:13:26</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>isolettest</td>
    <td>2023-05-17 21:17:28</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>statsmodelworkspace</td>
    <td>2023-05-17 21:19:51</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>xgboost-classification-autoconvert-workspace</td>
    <td>2023-05-17 21:21:19</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>xgboost-regression-autoconvert-workspace</td>
    <td>2023-05-17 21:21:55</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>housepricing</td>
    <td>2023-05-17 21:26:51</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>3</td>
    <td>1</td>
</tr>

<tr >
    <td>qjjoccfraudworkspace</td>
    <td>2023-05-17 21:31:31</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>btffhotswapworkspace</td>
    <td>2023-05-17 21:37:15</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>3</td>
    <td>1</td>
</tr>

<tr >
    <td>logworkspace</td>
    <td>2023-05-17 21:41:02</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>3</td>
    <td>1</td>
</tr>

<tr >
    <td>gobtedgeworkspaceexample</td>
    <td>2023-05-17 21:50:10</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>fvqusdkinferenceexampleworkspace</td>
    <td>2023-05-17 21:53:12</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>osysapiinferenceexampleworkspace</td>
    <td>2023-05-17 21:54:54</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>rehqtagtestworkspace</td>
    <td>2023-05-17 21:56:18</td>
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

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>nzssalohapipeline</td></tr><tr><th>created</th> <td>2023-05-17 21:58:51.973070+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 21:58:51.973070+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>9d7ad02e-f87f-4d05-ac2a-64445c5b2e11</td></tr><tr><th>steps</th> <td></td></tr></table>
{{</table>}}

We can verify the workspace is created the current default workspace with the `get_current_workspace()` command.

```python
wl.get_current_workspace()
```

    {'name': 'nzssalohaworkspace', 'id': 25, 'archived': False, 'created_by': '028c8b48-c39b-4578-9110-0b5bdd3824da', 'created_at': '2023-05-17T21:58:51.38289+00:00', 'models': [], 'pipelines': [{'name': 'nzssalohapipeline', 'create_time': datetime.datetime(2023, 5, 17, 21, 58, 51, 973070, tzinfo=tzutc()), 'definition': '[]'}]}

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

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>nzssalohapipeline</td></tr><tr><th>created</th> <td>2023-05-17 21:58:51.973070+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 21:58:51.973070+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>9d7ad02e-f87f-4d05-ac2a-64445c5b2e11</td></tr><tr><th>steps</th> <td></td></tr></table>
{{</table>}}

```python
aloha_pipeline.deploy()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>nzssalohapipeline</td></tr><tr><th>created</th> <td>2023-05-17 21:58:51.973070+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 21:58:57.421158+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>a948c68f-6830-44db-b10d-82b25064a154, 9d7ad02e-f87f-4d05-ac2a-64445c5b2e11</td></tr><tr><th>steps</th> <td>nzssalohamodel</td></tr></table>
{{</table>}}

We can verify that the pipeline is running and list what models are associated with it.

```python
aloha_pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.3.153',
       'name': 'engine-76c5f5fdd4-vzvgx',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'nzssalohapipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'nzssalohamodel',
          'version': 'b42b3e0b-f1bb-45bc-b0e8-e8ca5053da95',
          'sha': 'd71d9ffc61aaac58c2b1ed70a2db13d1416fb9d3f5b891e5e4e2e97180fe22f8',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.4.186',
       'name': 'engine-lb-584f54c899-nsjx7',
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

result = aloha_pipeline.infer(smoke_test)
display(result)
display(result.loc[:, ["time","out.main"]])
```

{{<table "table table-striped table-bordered" >}}
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.text_input</th>
      <th>out.banjori</th>
      <th>out.corebot</th>
      <th>out.cryptolocker</th>
      <th>out.dircrypt</th>
      <th>out.gozi</th>
      <th>out.kraken</th>
      <th>out.locky</th>
      <th>out.main</th>
      <th>out.matsnu</th>
      <th>out.pykspa</th>
      <th>out.qakbot</th>
      <th>out.ramdo</th>
      <th>out.ramnit</th>
      <th>out.simda</th>
      <th>out.suppobox</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-05-17 21:59:09.225</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 16, 32, 23, 29, 32, 30, 19, 26, 17]</td>
      <td>[0.0015195842]</td>
      <td>[0.98291475]</td>
      <td>[0.012099549]</td>
      <td>[4.759116e-05]</td>
      <td>[2.028935e-05]</td>
      <td>[0.00031977228]</td>
      <td>[0.011029261]</td>
      <td>[0.997564]</td>
      <td>[0.010341614]</td>
      <td>[0.008038961]</td>
      <td>[0.016155055]</td>
      <td>[0.0062362333]</td>
      <td>[0.0009985747]</td>
      <td>[1.7933368e-26]</td>
      <td>[1.3889844e-27]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}

{{<table "table table-striped table-bordered" >}}
<table>
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
      <td>2023-05-17 21:59:09.225</td>
      <td>[0.997564]</td>
    </tr>
  </tbody>
</table>
{{</table>}}

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
    time: [[2023-05-17 21:59:10.075,2023-05-17 21:59:10.075,2023-05-17 21:59:10.075,2023-05-17 21:59:10.075,2023-05-17 21:59:10.075,...,2023-05-17 21:59:10.075,2023-05-17 21:59:10.075,2023-05-17 21:59:10.075,2023-05-17 21:59:10.075,2023-05-17 21:59:10.075]]
    in.text_input: [[[0,0,0,0,0,...,32,30,19,26,17],[0,0,0,0,0,...,29,12,36,31,12],...,[0,0,0,0,0,...,35,16,35,27,16],[0,0,0,0,0,...,24,29,14,36,13]]]
    out.banjori: [[[0.0015195821],[0.00002837503],...,[0.0000056315566],[1.3068675e-12]]]
    out.corebot: [[[0.98291475],[0.000012753118],...,[0.0000033642746],[1.1029468e-9]]]
    out.cryptolocker: [[[0.012099549],[0.025435215],...,[0.13612257],[0.014839977]]]
    out.dircrypt: [[[0.000047591115],[6.150966e-10],...,[5.6732154e-11],[2.275736e-8]]]
    out.gozi: [[[0.000020289428],[2.3217829e-10],...,[2.7730737e-8],[8.438438e-15]]]
    out.kraken: [[[0.00031977257],[0.051351104],...,[0.0025221605],[0.30495808]]]
    out.locky: [[[0.011029262],[0.022038758],...,[0.05455697],[0.116279885]]]
    out.main: [[[0.997564],[0.9885122],...,[0.9998954],[0.99999803]]]
    ...

```python
outputs =  result.to_pandas()
display(outputs.loc[:5, ["time","out.main"]])
```

{{<table "table table-striped table-bordered" >}}
<table>
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
      <td>2023-05-17 21:59:10.075</td>
      <td>[0.997564]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-05-17 21:59:10.075</td>
      <td>[0.9885122]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-05-17 21:59:10.075</td>
      <td>[0.9993358]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-05-17 21:59:10.075</td>
      <td>[0.99999857]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-05-17 21:59:10.075</td>
      <td>[0.9984837]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2023-05-17 21:59:10.075</td>
      <td>[1.0]</td>
    </tr>
  </tbody>
</table>
{{</table>}}

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

    'https://doc-test.api.wallarooexample.ai/v1/api/pipelines/infer/nzssalohapipeline-27/nzssalohapipeline'

```python
connection =wl.mlops().__dict__
token = connection['token']
token
```

    'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJDYkFqN19QY0xCWTFkWmJiUDZ6Q3BsbkNBYTd6US0tRHlyNy0yLXlQb25nIn0.eyJleHAiOjE2ODQzNjA3OTcsImlhdCI6MTY4NDM2MDczNywiYXV0aF90aW1lIjoxNjg0MzU1OTU5LCJqdGkiOiIzNTBiODdjMC1mNmVhLTQwM2UtOTE2NC0xYmIyNDQ3NDIyNmUiLCJpc3MiOiJodHRwczovL2RvYy10ZXN0LmtleWNsb2FrLndhbGxhcm9vY29tbXVuaXR5Lm5pbmphL2F1dGgvcmVhbG1zL21hc3RlciIsImF1ZCI6WyJtYXN0ZXItcmVhbG0iLCJhY2NvdW50Il0sInN1YiI6IjAyOGM4YjQ4LWMzOWItNDU3OC05MTEwLTBiNWJkZDM4MjRkYSIsInR5cCI6IkJlYXJlciIsImF6cCI6InNkay1jbGllbnQiLCJzZXNzaW9uX3N0YXRlIjoiMGJlODJjN2ItNzg1My00ZjVkLWJiNWEtOTlkYjUwYjhiNDVmIiwiYWNyIjoiMCIsInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJkZWZhdWx0LXJvbGVzLW1hc3RlciIsIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iXX0sInJlc291cmNlX2FjY2VzcyI6eyJtYXN0ZXItcmVhbG0iOnsicm9sZXMiOlsibWFuYWdlLXVzZXJzIiwidmlldy11c2VycyIsInF1ZXJ5LWdyb3VwcyIsInF1ZXJ5LXVzZXJzIl19LCJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6InByb2ZpbGUgZW1haWwiLCJzaWQiOiIwYmU4MmM3Yi03ODUzLTRmNWQtYmI1YS05OWRiNTBiOGI0NWYiLCJlbWFpbF92ZXJpZmllZCI6ZmFsc2UsImh0dHBzOi8vaGFzdXJhLmlvL2p3dC9jbGFpbXMiOnsieC1oYXN1cmEtdXNlci1pZCI6IjAyOGM4YjQ4LWMzOWItNDU3OC05MTEwLTBiNWJkZDM4MjRkYSIsIngtaGFzdXJhLWRlZmF1bHQtcm9sZSI6InVzZXIiLCJ4LWhhc3VyYS1hbGxvd2VkLXJvbGVzIjpbInVzZXIiXSwieC1oYXN1cmEtdXNlci1ncm91cHMiOiJ7fSJ9LCJuYW1lIjoiSm9obiBIYW5zYXJpY2siLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJqb2huLmh1bW1lbEB3YWxsYXJvby5haSIsImdpdmVuX25hbWUiOiJKb2huIiwiZmFtaWx5X25hbWUiOiJIYW5zYXJpY2siLCJlbWFpbCI6ImpvaG4uaHVtbWVsQHdhbGxhcm9vLmFpIn0.EeUraMJjtC6UA1UzAM4yrbeoJFqHbBY42BZdzuPG9kzPX5skbRLetwAbYinVS0WZiZ-tBPH_RK5kLHjt-yw1ID3Xa_-u8mdfek3q67OydXXbr-a_dYABoLhulxfG4Eg8lo6sxE-r5BISJiMHiHFGmBK41xpr0ztZxMW9FXQtJkH5gBwE4eQtjDV7trkGIErPC73FO6Sp8WS6ywDG3SfYuTLIm9XlPjNxxjqglugcDjAVBw0TeJGNdAk2XhYfmIDUHCQfBtjWGy4P8OmJ_dfiibYl0E9pxQ0g4GsRfV9_-y45f8d-7OO-NxtrT0Z48pOMHrsBx8r2VWmqYVc_8SIxRg'

```python
dataFile="./data/data_25k.arrow"
contentType="application/vnd.apache.arrow.file"
```

```python
!curl -X POST {inference_url} -H "Authorization: Bearer {token}" -H "Content-Type:{contentType}" --data-binary @{dataFile} > curl_response.df
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100 25.6M  100 20.8M  100 4874k  2219k   506k  0:00:09  0:00:09 --:--:-- 4971k-  0:00:10  450k

```python
cc_data_from_file =  pd.read_json('./curl_response.df', orient="records")
display(cc_data_from_file.head(5))
```

{{<table "table table-striped table-bordered" >}}
<table>
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
      <td>1684360752527</td>
      <td>{'text_input': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 28.0, 16.0, 32.0, 23.0, 29.0, 32.0, 30.0, 19.0, 26.0, 17.0]}</td>
      <td>{'banjori': [0.0015195821], 'corebot': [0.9829147500000001], 'cryptolocker': [0.012099549000000001], 'dircrypt': [4.7591115e-05], 'gozi': [2.0289428e-05], 'kraken': [0.00031977256999999996], 'locky': [0.011029262000000001], 'main': [0.997564], 'matsnu': [0.010341609], 'pykspa': [0.008038961], 'qakbot': [0.016155055], 'ramdo': [0.00623623], 'ramnit': [0.0009985747000000001], 'simda': [1.7933434e-26], 'suppobox': [1.388995e-27]}</td>
      <td>[]</td>
      <td>{'last_model': '{"model_name":"nzssalohamodel","model_sha":"d71d9ffc61aaac58c2b1ed70a2db13d1416fb9d3f5b891e5e4e2e97180fe22f8"}', 'pipeline_version': 'a948c68f-6830-44db-b10d-82b25064a154', 'elapsed': [2482631, 4294967295]}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1684360752527</td>
      <td>{'text_input': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 30.0, 20.0, 19.0, 27.0, 14.0, 17.0, 24.0, 13.0, 23.0, 20.0, 18.0, 35.0, 18.0, 22.0, 23.0]}</td>
      <td>{'banjori': [7.447196e-18], 'corebot': [6.7359245e-08], 'cryptolocker': [0.1708199], 'dircrypt': [1.3220122000000002e-09], 'gozi': [1.2758705999999999e-24], 'kraken': [0.22559543], 'locky': [0.34209849999999997], 'main': [0.99999994], 'matsnu': [0.3080186], 'pykspa': [0.1828217], 'qakbot': [3.8022549999999994e-11], 'ramdo': [0.2062254], 'ramnit': [0.15215826], 'simda': [1.1701982e-30], 'suppobox': [3.1514454e-38]}</td>
      <td>[]</td>
      <td>{'last_model': '{"model_name":"nzssalohamodel","model_sha":"d71d9ffc61aaac58c2b1ed70a2db13d1416fb9d3f5b891e5e4e2e97180fe22f8"}', 'pipeline_version': 'a948c68f-6830-44db-b10d-82b25064a154', 'elapsed': [2482631, 4294967295]}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1684360752527</td>
      <td>{'text_input': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 25.0, 33.0, 25.0, 36.0, 25.0, 31.0, 14.0, 32.0, 36.0, 25.0, 12.0, 35.0, 34.0, 30.0, 28.0, 27.0, 24.0, 29.0, 27.0]}</td>
      <td>{'banjori': [2.8598648999999997e-21], 'corebot': [9.302004000000001e-08], 'cryptolocker': [0.04445298], 'dircrypt': [6.1637580000000004e-09], 'gozi': [8.3496755e-23], 'kraken': [0.48234479999999996], 'locky': [0.26332903], 'main': [1.0], 'matsnu': [0.29800338], 'pykspa': [0.22361776], 'qakbot': [1.5238920999999999e-06], 'ramdo': [0.32820392], 'ramnit': [0.029332489000000003], 'simda': [1.1995622e-31], 'suppobox': [0.0]}</td>
      <td>[]</td>
      <td>{'last_model': '{"model_name":"nzssalohamodel","model_sha":"d71d9ffc61aaac58c2b1ed70a2db13d1416fb9d3f5b891e5e4e2e97180fe22f8"}', 'pipeline_version': 'a948c68f-6830-44db-b10d-82b25064a154', 'elapsed': [2482631, 4294967295]}</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1684360752527</td>
      <td>{'text_input': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 17.0, 23.0, 22.0, 15.0, 12.0, 35.0, 34.0, 36.0, 12.0, 18.0, 24.0, 34.0, 32.0, 36.0, 12.0, 14.0, 16.0, 27.0, 22.0, 23.0]}</td>
      <td>{'banjori': [2.1387213e-15], 'corebot': [3.8817485e-10], 'cryptolocker': [0.045599736], 'dircrypt': [1.9090386e-07], 'gozi': [1.3140123e-25], 'kraken': [0.59542626], 'locky': [0.17374137], 'main': [0.9999996999999999], 'matsnu': [0.23151578], 'pykspa': [0.17591679999999998], 'qakbot': [1.0876152e-09], 'ramdo': [0.21832279999999998], 'ramnit': [0.0128692705], 'simda': [6.1588803e-28], 'suppobox': [1.4386237e-35]}</td>
      <td>[]</td>
      <td>{'last_model': '{"model_name":"nzssalohamodel","model_sha":"d71d9ffc61aaac58c2b1ed70a2db13d1416fb9d3f5b891e5e4e2e97180fe22f8"}', 'pipeline_version': 'a948c68f-6830-44db-b10d-82b25064a154', 'elapsed': [2482631, 4294967295]}</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1684360752527</td>
      <td>{'text_input': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 32.0, 13.0, 14.0, 12.0, 33.0, 16.0, 23.0, 15.0, 22.0, 30.0, 28.0, 26.0, 12.0, 16.0, 32.0, 37.0, 29.0, 22.0, 28.0, 22.0, 16.0, 27.0, 32.0]}</td>
      <td>{'banjori': [9.453342500000001e-15], 'corebot': [7.091151e-10], 'cryptolocker': [0.049815163], 'dircrypt': [5.2914135e-09], 'gozi': [7.4132087e-19], 'kraken': [1.5504574999999998e-13], 'locky': [1.079181e-15], 'main': [0.9999988999999999], 'matsnu': [1.5003075e-15], 'pykspa': [0.33075705], 'qakbot': [2.6258850000000004e-07], 'ramdo': [0.5036279], 'ramnit': [0.020393765], 'simda': [0.0], 'suppobox': [2.3292326e-38]}</td>
      <td>[]</td>
      <td>{'last_model': '{"model_name":"nzssalohamodel","model_sha":"d71d9ffc61aaac58c2b1ed70a2db13d1416fb9d3f5b891e5e4e2e97180fe22f8"}', 'pipeline_version': 'a948c68f-6830-44db-b10d-82b25064a154', 'elapsed': [2482631, 4294967295]}</td>
    </tr>
  </tbody>
</table>
{{</table>}}

## Undeploy Pipeline

When finished with our tests, we will undeploy the pipeline so we have the Kubernetes resources back for other tasks.  Note that if the deployment variable is unchanged aloha_pipeline.deploy() will restart the inference engine in the same configuration as before.

```python
aloha_pipeline.undeploy()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>nzssalohapipeline</td></tr><tr><th>created</th> <td>2023-05-17 21:58:51.973070+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 21:58:57.421158+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>a948c68f-6830-44db-b10d-82b25064a154, 9d7ad02e-f87f-4d05-ac2a-64445c5b2e11</td></tr><tr><th>steps</th> <td>nzssalohamodel</td></tr></table>
{{</table>}}

