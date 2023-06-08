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

To allow this tutorial to be run multiple times or by multiple users in the same Wallaroo instance, a random 4 character prefix will be added to the workspace, pipeline, and model.

```python
import string
import random

# make a random 4 character suffix to verify uniqueness in tutorials
suffix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))
workspace_name = f'alohaworkspace{suffix}'
pipeline_name = f'alohapipeline{suffix}'
model_name = f'alohamodel{suffix}'
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
wl.list_workspaces()[0:5]
```

    [{'name': 'john.hummel@wallaroo.ai - Default Workspace', 'id': 1, 'archived': False, 'created_by': '028c8b48-c39b-4578-9110-0b5bdd3824da', 'created_at': '2023-05-17T20:36:36.312003+00:00', 'models': [], 'pipelines': []},
     {'name': 'housepricedrift', 'id': 5, 'archived': False, 'created_by': '028c8b48-c39b-4578-9110-0b5bdd3824da', 'created_at': '2023-05-17T20:41:50.351766+00:00', 'models': [{'name': 'housepricemodel', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 5, 17, 20, 41, 50, 697557, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 5, 17, 20, 41, 50, 697557, tzinfo=tzutc())}], 'pipelines': [{'name': 'housepricepipe', 'create_time': datetime.datetime(2023, 5, 17, 20, 41, 50, 504206, tzinfo=tzutc()), 'definition': '[]'}]},
     {'name': 'sdkquickworkspace', 'id': 6, 'archived': False, 'created_by': '028c8b48-c39b-4578-9110-0b5bdd3824da', 'created_at': '2023-05-17T20:43:36.727099+00:00', 'models': [{'name': 'sdkquickmodel', 'versions': 2, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 5, 17, 20, 45, 44, 687223, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 5, 17, 20, 43, 42, 734108, tzinfo=tzutc())}], 'pipelines': [{'name': 'sdkquickpipeline', 'create_time': datetime.datetime(2023, 5, 17, 20, 43, 38, 111213, tzinfo=tzutc()), 'definition': '[]'}]},
     {'name': 'apiworkspaces', 'id': 7, 'archived': False, 'created_by': '028c8b48-c39b-4578-9110-0b5bdd3824da', 'created_at': '2023-05-17T20:50:36.298217+00:00', 'models': [{'name': 'apimodel', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 5, 17, 20, 51, 51, 92416, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 5, 17, 20, 51, 51, 92416, tzinfo=tzutc())}, {'name': 'apiteststreammodel', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 5, 17, 20, 51, 53, 77997, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 5, 17, 20, 51, 53, 77997, tzinfo=tzutc())}], 'pipelines': [{'name': 'pipelinenomodel', 'create_time': datetime.datetime(2023, 5, 17, 20, 52, 4, 815142, tzinfo=tzutc()), 'definition': '[]'}, {'name': 'pipelinemodels', 'create_time': datetime.datetime(2023, 5, 17, 20, 52, 6, 768968, tzinfo=tzutc()), 'definition': '[]'}, {'name': 'copiedmodelpipeline', 'create_time': datetime.datetime(2023, 5, 17, 20, 54, 1, 568303, tzinfo=tzutc()), 'definition': '[]'}]},
     {'name': 'azuremlsdkworkspace', 'id': 8, 'archived': False, 'created_by': '028c8b48-c39b-4578-9110-0b5bdd3824da', 'created_at': '2023-05-17T21:01:45.647036+00:00', 'models': [{'name': 'azuremlsdkmodel', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 5, 17, 21, 1, 50, 122294, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 5, 17, 21, 1, 50, 122294, tzinfo=tzutc())}], 'pipelines': [{'name': 'azuremlsdkpipeline', 'create_time': datetime.datetime(2023, 5, 17, 21, 1, 46, 655024, tzinfo=tzutc()), 'definition': '[]'}]}]

```python
workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)

aloha_pipeline = get_pipeline(pipeline_name)
aloha_pipeline
```

<table><tr><th>name</th> <td>alohapipelineulmp</td></tr><tr><th>created</th> <td>2023-05-19 18:11:07.415859+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-19 18:11:07.415859+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>610ed3a7-d443-40c8-8fd8-cc19e720b336</td></tr><tr><th>steps</th> <td></td></tr></table>

We can verify the workspace is created the current default workspace with the `get_current_workspace()` command.

```python
wl.get_current_workspace()
```

    {'name': 'alohaworkspaceulmp', 'id': 43, 'archived': False, 'created_by': '028c8b48-c39b-4578-9110-0b5bdd3824da', 'created_at': '2023-05-19T18:11:06.792668+00:00', 'models': [], 'pipelines': [{'name': 'alohapipelineulmp', 'create_time': datetime.datetime(2023, 5, 19, 18, 11, 7, 415859, tzinfo=tzutc()), 'definition': '[]'}]}

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

<table><tr><th>name</th> <td>alohapipelineulmp</td></tr><tr><th>created</th> <td>2023-05-19 18:11:07.415859+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-19 18:11:07.415859+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>610ed3a7-d443-40c8-8fd8-cc19e720b336</td></tr><tr><th>steps</th> <td></td></tr></table>

```python
aloha_pipeline.deploy()
```

<table><tr><th>name</th> <td>alohapipelineulmp</td></tr><tr><th>created</th> <td>2023-05-19 18:11:07.415859+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-19 18:11:12.238317+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>45e8b2f0-619d-41c3-b812-661b92d261f1, 610ed3a7-d443-40c8-8fd8-cc19e720b336</td></tr><tr><th>steps</th> <td>alohamodelulmp</td></tr></table>

We can verify that the pipeline is running and list what models are associated with it.

```python
aloha_pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.0.152',
       'name': 'engine-5487c945f5-9xjx2',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'alohapipelineulmp',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'alohamodelulmp',
          'version': '75883efa-f99e-4e67-a099-52097f24184c',
          'sha': 'd71d9ffc61aaac58c2b1ed70a2db13d1416fb9d3f5b891e5e4e2e97180fe22f8',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.4.220',
       'name': 'engine-lb-584f54c899-2x4g5',
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
      <td>2023-05-19 18:11:24.158</td>
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
      <td>2023-05-19 18:12:46.427</td>
      <td>[0.997564]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-05-19 18:12:46.427</td>
      <td>[0.9885122]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-05-19 18:12:46.427</td>
      <td>[0.9993358]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-05-19 18:12:46.427</td>
      <td>[0.99999857]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-05-19 18:12:46.427</td>
      <td>[0.9984837]</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>2023-05-19 18:12:46.427</td>
      <td>[0.9999754]</td>
    </tr>
    <tr>
      <th>996</th>
      <td>2023-05-19 18:12:46.427</td>
      <td>[0.9999727]</td>
    </tr>
    <tr>
      <th>997</th>
      <td>2023-05-19 18:12:46.427</td>
      <td>[0.66066873]</td>
    </tr>
    <tr>
      <th>998</th>
      <td>2023-05-19 18:12:46.427</td>
      <td>[0.9998954]</td>
    </tr>
    <tr>
      <th>999</th>
      <td>2023-05-19 18:12:46.427</td>
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
      <td>2023-05-19 18:11:25.064</td>
      <td>[0.997564]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-05-19 18:11:25.064</td>
      <td>[0.9885122]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-05-19 18:11:25.064</td>
      <td>[0.9993358]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-05-19 18:11:25.064</td>
      <td>[0.99999857]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-05-19 18:11:25.064</td>
      <td>[0.9984837]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2023-05-19 18:11:25.064</td>
      <td>[0.99999994]</td>
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

    'https://doc-test.api.wallarooexample.ai/v1/api/pipelines/infer/alohapipelineulmp-72/alohapipelineulmp'

```python
connection =wl.mlops().__dict__
token = connection['token']
```

```python
dataFile="./data/data_25k.arrow"
contentType="application/vnd.apache.arrow.file"
```

```python
!curl -X POST {inference_url} -H "Authorization: Bearer {token}" -H "Content-Type:{contentType}" --data-binary @{dataFile} > curl_response.df
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100 25.6M  100 20.8M  100 4874k  2156k   574k  0:00:09  0:00:08  0:00:01 4398kk  2377k   542k  0:00:08  0:00:08 --:--:-- 5266k

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
      <td>1684519887796</td>
      <td>{'banjori': [0.0015195871], 'corebot': [0.9829148], 'cryptolocker': [0.012099565000000001], 'dircrypt': [4.7591344e-05], 'gozi': [2.0289392e-05], 'kraken': [0.0003197726], 'locky': [0.011029272000000001], 'main': [0.997564], 'matsnu': [0.010341625], 'pykspa': [0.008038965], 'qakbot': [0.016155062], 'ramdo': [0.006236233000000001], 'ramnit': [0.0009985756], 'simda': [1.793378e-26], 'suppobox': [1.3889898e-27]}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1684519887796</td>
      <td>{'banjori': [7.447225e-18], 'corebot': [6.7359245e-08], 'cryptolocker': [0.17081991], 'dircrypt': [1.3220147000000001e-09], 'gozi': [1.2758853e-24], 'kraken': [0.22559536], 'locky': [0.34209844], 'main': [0.99999994], 'matsnu': [0.30801848], 'pykspa': [0.18282163], 'qakbot': [3.8022553999999996e-11], 'ramdo': [0.20622534], 'ramnit': [0.15215826], 'simda': [1.17020745e-30], 'suppobox': [3.1514464999999997e-38]}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1684519887796</td>
      <td>{'banjori': [2.8599304999999997e-21], 'corebot': [9.302004999999999e-08], 'cryptolocker': [0.04445295], 'dircrypt': [6.1637580000000004e-09], 'gozi': [8.34974e-23], 'kraken': [0.48234479999999996], 'locky': [0.2633289], 'main': [1.0], 'matsnu': [0.29800323], 'pykspa': [0.22361766], 'qakbot': [1.5238920999999999e-06], 'ramdo': [0.3282038], 'ramnit': [0.029332466], 'simda': [1.1995533000000001e-31], 'suppobox': [0.0]}</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1684519887796</td>
      <td>{'banjori': [2.1386805e-15], 'corebot': [3.8817485e-10], 'cryptolocker': [0.045599725], 'dircrypt': [1.9090386e-07], 'gozi': [1.3139924000000002e-25], 'kraken': [0.59542614], 'locky': [0.17374131], 'main': [0.9999996999999999], 'matsnu': [0.2315157], 'pykspa': [0.17591687], 'qakbot': [1.087611e-09], 'ramdo': [0.21832284000000002], 'ramnit': [0.012869288000000001], 'simda': [6.158882e-28], 'suppobox': [1.438591e-35]}</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1684519887796</td>
      <td>{'banjori': [9.453381e-15], 'corebot': [7.091152e-10], 'cryptolocker': [0.049815107000000004], 'dircrypt': [5.2914135e-09], 'gozi': [7.4132087e-19], 'kraken': [1.5504637e-13], 'locky': [1.079181e-15], 'main': [0.9999988999999999], 'matsnu': [1.5003076000000002e-15], 'pykspa': [0.33075709999999997], 'qakbot': [2.6258948e-07], 'ramdo': [0.50362796], 'ramnit': [0.020393757000000002], 'simda': [0.0], 'suppobox': [0.0]}</td>
    </tr>
  </tbody>
</table>

## Undeploy Pipeline

When finished with our tests, we will undeploy the pipeline so we have the Kubernetes resources back for other tasks.  Note that if the deployment variable is unchanged aloha_pipeline.deploy() will restart the inference engine in the same configuration as before.

```python
aloha_pipeline.undeploy()
```

<table><tr><th>name</th> <td>alohapipelineejcz</td></tr><tr><th>created</th> <td>2023-05-19 15:33:49.843485+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-19 15:33:55.224991+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>65d9561f-ca24-4d91-8639-846850d0e1e0, a0fed191-6265-435b-9257-bc59847a96cf</td></tr><tr><th>steps</th> <td>alohamodelejcz</td></tr></table>

