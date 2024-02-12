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

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).e/wallaroo-sdk-essentials-client/).

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
```

## Create the Workspace

We will create a workspace to work in and call it the "alohaworkspace", then set it as current workspace environment.  We'll also create our pipeline in advance as `alohapipeline`.  The model name and the model file will be specified for use in later steps.

Workspace names are unique across the Wallaroo instance.  The helper function below creates a new workspace if one with the same name does not exist, or retrieves an existing one.  If a workspace with the same name exists and the current SDK user is not granted access, an error is returned.

Verify the workspace name is either not created or is available for your use in your Wallaroo instance before proceeding.

```python
workspace_name = f'alohaworkspace'
pipeline_name = f'alohapipeline'
model_name = f'alohamodel'
model_file_name = './alohacnnlstm.zip'
```

```python
def get_workspace(name, client):
    workspace = None
    for ws in client.list_workspaces():
        if ws.name() == name:
            workspace= ws
    if(workspace == None):
        workspace = client.create_workspace(name)
    return workspace
```

```python
wl.list_workspaces()[0:5]
```

    [{'name': 'john.hummel@wallaroo.ai - Default Workspace', 'id': 1, 'archived': False, 'created_by': '784e4c99-ee08-4aab-9eaa-0d8ad8e1af53', 'created_at': '2024-02-12T16:51:56.978044+00:00', 'models': [], 'pipelines': []}]

```python
workspace = get_workspace(workspace_name, wl)

wl.set_current_workspace(workspace)

aloha_pipeline = wl.build_pipeline(pipeline_name)
aloha_pipeline
```

<table><tr><th>name</th> <td>alohapipeline</td></tr><tr><th>created</th> <td>2024-02-12 17:36:10.798258+00:00</td></tr><tr><th>last_updated</th> <td>2024-02-12 17:36:10.798258+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>accel</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>777f8b2d-c253-4674-876f-b6fefe3c3392</td></tr><tr><th>steps</th> <td></td></tr><tr><th>published</th> <td>False</td></tr></table>

We can verify the workspace is created the current default workspace with the `get_current_workspace()` command.

```python
wl.get_current_workspace()
```

    {'name': 'alohaworkspaceudjo', 'id': 13, 'archived': False, 'created_by': 'db364f8c-b866-4865-96b7-0b65662cb384', 'created_at': '2023-08-28T17:10:04.351972+00:00', 'models': [], 'pipelines': [{'name': 'alohapipelineudjo', 'create_time': datetime.datetime(2023, 8, 28, 17, 10, 5, 43499, tzinfo=tzutc()), 'definition': '[]'}]}

# Upload the Models

Now we will upload our models.  Note that for this example we are applying the model from a .ZIP file.  The Aloha model is a [protobuf](https://developers.google.com/protocol-buffers) file that has been defined for evaluating web pages, and we will configure it to use data in the `tensorflow` format.

```python
from wallaroo.framework import Framework

model = wl.upload_model(model_name, 
                        model_file_name,
                        framework=Framework.TENSORFLOW
                        )
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

<table><tr><th>name</th> <td>alohapipeline</td></tr><tr><th>created</th> <td>2024-02-12 17:36:10.798258+00:00</td></tr><tr><th>last_updated</th> <td>2024-02-12 17:36:10.798258+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>accel</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>777f8b2d-c253-4674-876f-b6fefe3c3392</td></tr><tr><th>steps</th> <td></td></tr><tr><th>published</th> <td>False</td></tr></table>

```python
deploy_config = wallaroo.deployment_config.DeploymentConfigBuilder() \
    .cpus(4)\
    .build()

aloha_pipeline.deploy(deployment_config=deploy_config)
```

<table><tr><th>name</th> <td>alohapipeline</td></tr><tr><th>created</th> <td>2024-02-12 17:36:10.798258+00:00</td></tr><tr><th>last_updated</th> <td>2024-02-12 17:46:04.189394+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>accel</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>a8d32a49-2367-4bec-8e22-901d71a019cc, 189b140d-e595-43d0-8f9c-c14ea8cd955f, 00945700-f7a1-45cd-aa29-1e0cbaa71e98, 777f8b2d-c253-4674-876f-b6fefe3c3392</td></tr><tr><th>steps</th> <td>alohamodel</td></tr><tr><th>published</th> <td>False</td></tr></table>

We can verify that the pipeline is running and list what models are associated with it.

```python
aloha_pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.100.1.127',
       'name': 'engine-6c8869d9c4-gw65s',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'alohapipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'alohamodel',
          'version': '0fd2f2f2-7be5-4042-a9bf-ef018a713919',
          'sha': 'd71d9ffc61aaac58c2b1ed70a2db13d1416fb9d3f5b891e5e4e2e97180fe22f8',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.100.0.121',
       'name': 'engine-lb-dcd9c8cd7-2b2wv',
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
      <td>2024-02-12 17:38:05.379</td>
      <td>[0.997564]</td>
    </tr>
  </tbody>
</table>

### Infer From File

This time, we'll give it a bigger set of data to infer.  `./data/data_1k.arrow` is an Apache Arrow table with 1,000 records in it.  Once submitted, we'll turn the result into a DataFrame and display the first five results.

```python
result = aloha_pipeline.infer_from_file('./data/data_1k.arrow')
display(result.to_pandas().loc[:, ["time","out.main"]])
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
      <td>2024-02-12 17:38:07.173</td>
      <td>[0.997564]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-02-12 17:38:07.173</td>
      <td>[0.9885122]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-02-12 17:38:07.173</td>
      <td>[0.9993358]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-02-12 17:38:07.173</td>
      <td>[0.99999857]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-02-12 17:38:07.173</td>
      <td>[0.9984837]</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>2024-02-12 17:38:07.173</td>
      <td>[0.9999754]</td>
    </tr>
    <tr>
      <th>996</th>
      <td>2024-02-12 17:38:07.173</td>
      <td>[0.9999727]</td>
    </tr>
    <tr>
      <th>997</th>
      <td>2024-02-12 17:38:07.173</td>
      <td>[0.66066873]</td>
    </tr>
    <tr>
      <th>998</th>
      <td>2024-02-12 17:38:07.173</td>
      <td>[0.9998954]</td>
    </tr>
    <tr>
      <th>999</th>
      <td>2024-02-12 17:38:07.173</td>
      <td>[0.99999803]</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 2 columns</p>

```python
outputs =  result.to_pandas()
display(outputs.loc[:5, ["time","out.main"]])
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
      <td>2024-02-12 17:38:07.173</td>
      <td>[0.997564]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-02-12 17:38:07.173</td>
      <td>[0.9885122]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-02-12 17:38:07.173</td>
      <td>[0.9993358]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-02-12 17:38:07.173</td>
      <td>[0.99999857]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-02-12 17:38:07.173</td>
      <td>[0.9984837]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2024-02-12 17:38:07.173</td>
      <td>[1.0]</td>
    </tr>
  </tbody>
</table>

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

    'https://doc-test.api.wallarooexample.ai/v1/api/pipelines/infer/alohapipeline-1/alohapipeline'

```python
headers = wl.auth.auth_header()
headers['Content-Type']="application/vnd.apache.arrow.file"
```

```python
dataFile="./data/data_25k.arrow"
```

```python
!curl -X POST {inference_url} -H "Authorization: {wl.auth.auth_header()['Authorization']}" -H "Content-Type:{headers['Content-Type']}" --data-binary @{dataFile} > curl_response.df
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100 25.9M  100 21.1M  100 4874k  1918k   431k  0:00:11  0:00:11 --:--:-- 5344k06k  0:00:03  0:00:03 --:--:-- 1506k

```python
cc_data_from_file =  pd.read_json('./curl_response.df', orient="records")
display(cc_data_from_file.head(5).loc[:5, ["time","out"]])
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1707760059710</td>
      <td>{'banjori': [0.0015195814], 'corebot': [0.9829147500000001], 'cryptolocker': [0.012099549000000001], 'dircrypt': [4.7591115e-05], 'gozi': [2.0289312e-05], 'kraken': [0.00031977256999999996], 'locky': [0.011029262000000001], 'main': [0.997564], 'matsnu': [0.010341609], 'pykspa': [0.008038961], 'qakbot': [0.016155047000000002], 'ramdo': [0.00623623], 'ramnit': [0.0009985747000000001], 'simda': [1.7933434e-26], 'suppobox': [1.388995e-27]}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1707760059710</td>
      <td>{'banjori': [7.447168e-18], 'corebot': [6.7359245e-08], 'cryptolocker': [0.17081994], 'dircrypt': [1.3220122000000002e-09], 'gozi': [1.2758656999999998e-24], 'kraken': [0.22559547], 'locky': [0.34209849999999997], 'main': [0.99999994], 'matsnu': [0.3080186], 'pykspa': [0.1828217], 'qakbot': [3.8022407999999996e-11], 'ramdo': [0.2062254], 'ramnit': [0.15215829], 'simda': [1.1701982e-30], 'suppobox': [3.1513975e-38]}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1707760059710</td>
      <td>{'banjori': [2.8598757999999993e-21], 'corebot': [9.302004000000001e-08], 'cryptolocker': [0.044453003000000005], 'dircrypt': [6.163770000000001e-09], 'gozi': [8.3496755e-23], 'kraken': [0.48234475], 'locky': [0.26332903], 'main': [1.0], 'matsnu': [0.29800338], 'pykspa': [0.22361776], 'qakbot': [1.5238921e-06], 'ramdo': [0.32820404000000003], 'ramnit': [0.029332504000000002], 'simda': [1.1995622e-31], 'suppobox': [0.0]}</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1707760059710</td>
      <td>{'banjori': [2.1387130000000002e-15], 'corebot': [3.881756e-10], 'cryptolocker': [0.045599725], 'dircrypt': [1.9090367999999998e-07], 'gozi': [1.3140173e-25], 'kraken': [0.5954264], 'locky': [0.17374137], 'main': [0.9999996999999999], 'matsnu': [0.23151578], 'pykspa': [0.17591679999999998], 'qakbot': [1.0876152e-09], 'ramdo': [0.21832279999999998], 'ramnit': [0.012869265000000001], 'simda': [6.1588803e-28], 'suppobox': [1.4386127e-35]}</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1707760059710</td>
      <td>{'banjori': [9.453379e-15], 'corebot': [7.091178e-10], 'cryptolocker': [0.049815144000000006], 'dircrypt': [5.2914135e-09], 'gozi': [7.4132087e-19], 'kraken': [1.5504633e-13], 'locky': [1.0791851e-15], 'main': [0.9999988999999999], 'matsnu': [1.5003075e-15], 'pykspa': [0.33075709999999997], 'qakbot': [2.62589e-07], 'ramdo': [0.5036279], 'ramnit': [0.020393759], 'simda': [0.0], 'suppobox': [2.3292147999999998e-38]}</td>
    </tr>
  </tbody>
</table>

## Undeploy Pipeline

When finished with our tests, we will undeploy the pipeline so we have the Kubernetes resources back for other tasks.  Note that if the deployment variable is unchanged aloha_pipeline.deploy() will restart the inference engine in the same configuration as before.

```python
aloha_pipeline.undeploy()
```

<table><tr><th>name</th> <td>alohapipeline</td></tr><tr><th>created</th> <td>2024-02-12 17:36:10.798258+00:00</td></tr><tr><th>last_updated</th> <td>2024-02-12 17:46:04.189394+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>accel</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>a8d32a49-2367-4bec-8e22-901d71a019cc, 189b140d-e595-43d0-8f9c-c14ea8cd955f, 00945700-f7a1-45cd-aa29-1e0cbaa71e98, 777f8b2d-c253-4674-876f-b6fefe3c3392</td></tr><tr><th>steps</th> <td>alohamodel</td></tr><tr><th>published</th> <td>False</td></tr></table>

