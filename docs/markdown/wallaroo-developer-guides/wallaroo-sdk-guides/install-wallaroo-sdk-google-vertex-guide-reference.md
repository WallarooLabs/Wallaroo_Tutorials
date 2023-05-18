This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/development/sdk-install-guides/google-vertex-sdk-install).

## Installing the Wallaroo SDK into Google Vertex Workbench

Organizations that use Google Vertex for model training and development can deploy models to Wallaroo through the [Wallaroo SDK](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/).  The following guide is created to assist users with installing the Wallaroo SDK, setting up authentication through Google Cloud Platform (GCP), and making a standard connection to a Wallaroo instance through Google Workbench.

These instructions are based on the on the [Wallaroo SSO for Google Cloud Platform](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-sso-authentication/wallaroo-sso-gcp/) and the [Connect to Wallaroo](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/) guides.

This tutorial provides the following:

* `aloha-cnn-lstm.zip`: A pre-trained open source model that uses an [Aloha CNN LSTM model](https://www.researchgate.net/publication/348920204_Using_Auxiliary_Inputs_in_Deep_Learning_Models_for_Detecting_DGA-based_Domain_Names) for classifying Domain names as being either legitimate or being used for nefarious purposes such as malware distribution.
* Test Data Files:
  * `data_1k.arrow`: 1,000 records
  * `data_25k.arrow`: 25,000 records

To use the Wallaroo SDK within Google Workbench, a virtual environment will be used.  This will set the necessary libraries and specific Python version required.

## Prerequisites

The following is required for this tutorial:

* A Wallaroo instance version 2023.1 or later.
* Python 3.8.6 or later installed locally
* [Conda](https://docs.conda.io/en/latest/):  Used for managing python virtual environments.
* The following Python libraries installed:
  * `os`
  * [`wallaroo`](https://pypi.org/project/wallaroo/): The Wallaroo SDK. Included with the Wallaroo JupyterHub service by default.
  * [`pandas`](https://pypi.org/project/pandas/): Pandas, mainly used for Pandas DataFrame
  * [`pyarrow`](https://pypi.org/project/pyarrow/): PyArrow for Apache Arrow support
  * [`polars`](https://pypi.org/project/polars/): Polars for DataFrame with native Apache Arrow support

## General Steps

For our example, we will perform the following:

* Wallaroo SDK Install
  * Set up a Python virtual environment through `conda` with the libraries that enable the virtual environment for use in a Jupyter Hub environment.
  * Install the Wallaroo SDK.
* Wallaroo SDK from remote JupyterHub Demonstration (Optional):  The following steps are an optional exercise to demonstrate using the Wallaroo SDK from a remote connection.  The entire tutorial can be found on the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/sdk-install-guides/google-vertex-sdk-install).
  * Connect to a remote Wallaroo instance.
  * Create a workspace for our work.
  * Upload the Aloha model.
  * Create a pipeline that can ingest our submitted data, submit it to the model, and export the results
  * Run a sample inference through our pipeline by loading a file
  * Retrieve the external deployment URL.  This sample Wallaroo instance has been configured to create external inference URLs for pipelines.  For more information, see the [External Inference URL Guide](https://docs.wallaroo.aiwallaroo-operations-guide/wallaroo-configuration/wallaroo-model-endpoints-guide/).
  * Run a sample inference through our pipeline's external URL and store the results in a file.  This assumes that the External Inference URLs have been enabled for the target Wallaroo instance.
  * Undeploy the pipeline and return resources back to the Wallaroo instance's Kubernetes environment.

## Install Wallaroo SDK

### Set Up Virtual Python Environment

To set up the virtual environment in Google Workbench for using the Wallaroo SDK with Google Workbench:

1. Start a separate terminal by selecting **File->New->Terminal**.
1. Create the Python virtual environment with `conda`.  Replace `wallaroosdk` with the name of the virtual environment as required by your organization.  Note that Python 3.8.6 and above is specified as a requirement for Python libraries used with the Wallaroo SDK.  The following will install the latest version of Python 3.8, which as of this time is 3.8.15.

    ```bash
    conda create -n wallaroosdk python=3.8
    ```

1. Activate the new environment.

    ```bash
    conda activate wallaroosdk
    ```

1. Install the `ipykernel` library.  This allows the JupyterHub notebooks to access the Python virtual environment as a kernel.

    ```bash
    conda install ipykernel
    ```
    
1. Install the new virtual environment as a python kernel.

    ```bash
    ipython kernel install --user --name=wallaroosdk
    ```
    
1. Install the Wallaroo SDK.  This process may take several minutes while the other required Python libraries are added to the virtual environment.

    ```bash
    pip install wallaroo==2023.2.0
    ```

Once the conda virtual environment has been installed, it can either be selected as a new Jupyter Notebook kernel, or the Notebook's kernel can be set to an existing Jupyter notebook.

![](./images/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-install-guides/wallaroo-select-kernel.png)

To use a new Notebook:

1. From the main menu, select **File->New-Notebook**.
1. From the Kernel selection dropbox, select the new virtual environment - in this case, **wallaroosdk**.

To update an existing Notebook to use the new virtual environment as a kernel:

1. From the main menu, select **Kernel->Change Kernel**.
1. Select the new kernel.

## Sample Wallaroo Connection

With the Wallaroo Python SDK installed, remote commands and inferences can be performed through the following steps.

### Open a Connection to Wallaroo

The first step is to connect to Wallaroo through the Wallaroo client.

This is accomplished using the `wallaroo.Client(api_endpoint, auth_endpoint, auth_type command)` command that connects to the Wallaroo instance services.

The `Client` method takes the following parameters:

* **api_endpoint** (*String*): The URL to the Wallaroo instance API service.
* **auth_endpoint** (*String*): The URL to the Wallaroo instance Keycloak service.
* **auth_type command** (*String*): The authorization type.  In this case, `SSO`.

The URLs are based on the Wallaroo Prefix and Wallaroo Suffix for the Wallaroo instance.  For more information, see the [DNS Integration Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-dns-guide/).  In the example below, replace "YOUR PREFIX" and "YOUR SUFFIX" with the Wallaroo Prefix and Suffix, respectively.

Once run, the `wallaroo.Client` command provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Depending on the configuration of the Wallaroo instance, the user will either be presented with a login request to the Wallaroo instance or be authenticated through a broker such as Google, Github, etc.  To use the broker, select it from the list under the username/password login forms.  For more information on Wallaroo authentication configurations, see the [Wallaroo Authentication Configuration Guides](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-sso-authentication/).

![Wallaroo Login](./images/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-install-guides/wallaroo-sdk-login.png)

Once authenticated, the user will verify adding the device the user is establishing the connection from.  Once both steps are complete, then the connection is granted.

![Device Registration](./images/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-install-guides/wallaroo-device-access.png)

The connection is stored in the variable `wl` for use in all other Wallaroo calls.

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
# SSO login through keycloak

wallarooPrefix = "YOUR PREFIX"
wallarooSuffix = "YOUR PREFIX"

wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}.api.{wallarooSuffix}", 
                    auth_endpoint=f"https://{wallarooPrefix}.keycloak.{wallarooSuffix}", 
                    auth_type="sso")
```

### Create the Workspace

We will create a workspace to work in and call it the `gcpsdkworkspace`, then set it as current workspace environment.  We'll also create our pipeline in advance as `gcpsdkpipeline`.

* **IMPORTANT NOTE**:  For this example, the Aloha model is stored in the file `alohacnnlstm.zip`.  When using tensor based models, the zip file **must** match the name of the tensor directory.  For example, if the tensor directory is `alohacnnlstm`, then the .zip file must be named `alohacnnlstm.zip`.

```python
workspace_name = 'gcpsdkworkspace'
pipeline_name = 'gcpsdkpipeline'
model_name = 'gcpsdkmodel'
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
workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)

pipeline = get_pipeline(pipeline_name)
pipeline
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>gcpsdkpipeline</td></tr><tr><th>created</th> <td>2023-05-17 21:03:44.485720+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 21:03:44.485720+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>7c043d3c-c894-4ae9-9ec1-c35518130b90</td></tr><tr><th>steps</th> <td></td></tr></table>
{{</table>}}

We can verify the workspace is created the current default workspace with the `get_current_workspace()` command.

```python
wl.get_current_workspace()
```

    {'name': 'gcpsdkworkspace', 'id': 10, 'archived': False, 'created_by': '028c8b48-c39b-4578-9110-0b5bdd3824da', 'created_at': '2023-05-17T21:03:43.54944+00:00', 'models': [], 'pipelines': [{'name': 'gcpsdkpipeline', 'create_time': datetime.datetime(2023, 5, 17, 21, 3, 44, 485720, tzinfo=tzutc()), 'definition': '[]'}]}

### Upload the Models

Now we will upload our model.  Note that for this example we are applying the model from a .ZIP file.  The Aloha model is a [protobuf](https://developers.google.com/protocol-buffers) file that has been defined for evaluating web pages, and we will configure it to use data in the `tensorflow` format.

```python
model = wl.upload_model(model_name, model_file_name).configure("tensorflow")
```

### Deploy a Model

Now that we have a model that we want to use we will create a deployment for it. 

We will tell the deployment we are using a tensorflow model and give the deployment name and the configuration we want for the deployment.

To do this, we'll create our pipeline that can ingest the data, pass the data to our Aloha model, and give us a final output.  We'll call our pipeline `externalsdkpipeline`, then deploy it so it's ready to receive data.  The deployment process usually takes about 45 seconds.

```python
pipeline.add_model_step(model)
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>gcpsdkpipeline</td></tr><tr><th>created</th> <td>2023-05-17 21:03:44.485720+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 21:03:44.485720+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>7c043d3c-c894-4ae9-9ec1-c35518130b90</td></tr><tr><th>steps</th> <td></td></tr></table>
{{</table>}}

```python
pipeline.deploy()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>gcpsdkpipeline</td></tr><tr><th>created</th> <td>2023-05-17 21:03:44.485720+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 21:03:49.137632+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>6398cafc-50c4-49e3-9499-6025b7808245, 7c043d3c-c894-4ae9-9ec1-c35518130b90</td></tr><tr><th>steps</th> <td>gcpsdkmodel</td></tr></table>
{{</table>}}

We can verify that the pipeline is running and list what models are associated with it.

```python
pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.2.157',
       'name': 'engine-7694d96677-f4jfk',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'gcpsdkpipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'gcpsdkmodel',
          'version': 'aff60f1f-b036-47d7-920c-1819be75d734',
          'sha': 'd71d9ffc61aaac58c2b1ed70a2db13d1416fb9d3f5b891e5e4e2e97180fe22f8',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.4.169',
       'name': 'engine-lb-584f54c899-fc8lh',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}

## Interferences

### Infer 1 row

Now that the pipeline is deployed and our Aloha model is in place, we'll perform a smoke test to verify the pipeline is up and running properly.  We'll use the `infer_from_file` command to load a single encoded URL into the inference engine and print the results back out.

The result should tell us that the tokenized URL is legitimate (0) or fraud (1).  This sample data should return close to 1.

```python
## Demonstrate via straight infer

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
      <td>2023-05-17 21:04:10.047</td>
      <td>[0.997564]</td>
    </tr>
  </tbody>
</table>
{{</table>}}

## Infer 1,000 Rows

We can also infer an entire batch as one request either with the Pipeline `infer` method with multiple rows, or loaded from a file using the Pipeline `infer_from_file` method.  For this example, we will run a batch on 1,000 records using the file `data_1k.arrow`.  This is an Apache Arrow table, which gives the added benefit of speed and lower file size as a binary file rather than a text JSON file.

We'll infer the 1,000 records, then convert it to a DataFrame and display the first 5 to save space in our Jupyter Notebook.

```python
result = pipeline.infer_from_file('./data/data_1k.arrow')

outputs = result.to_pandas()
display(outputs.head(5).loc[:, ["time","out.main"]])
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
      <td>2023-05-17 21:04:10.886</td>
      <td>[0.997564]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-05-17 21:04:10.886</td>
      <td>[0.9885122]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-05-17 21:04:10.886</td>
      <td>[0.9993358]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-05-17 21:04:10.886</td>
      <td>[0.99999857]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-05-17 21:04:10.886</td>
      <td>[0.9984837]</td>
    </tr>
  </tbody>
</table>
{{</table>}}

### Batch Inference

Now that our smoke test is successful, let's really give it some data.  We have two inference files we can use:

* `data-1k.arrow`:  Contains 10,000 inferences
* `data-25k.arrow`: Contains 25,000 inferences

These inference inputs are Apache Arrow tables, which Wallaroo can ingest natively.  These are binary files, and are faster to transmit because of their smaller size compared to JSON.

We'll pipe the `data-25k.arrow` file through the `pipeline` deployment URL, and place the results in a file named `response.df`.  Note that for larger batches of 1,000 inferences or more can be difficult to view in Juypter Hub because of its size, so we'll only display the first 5 results of the inference.

When retrieving the pipeline inference URL through an external SDK connection, the External Inference URL will be returned.  This URL will function provided that the **Enable external URL inference endpoints** is enabled.  For more information, see the [Wallaroo Model Endpoints Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-model-endpoints-guide/).

```python
inference_url = pipeline._deployment._url()
inference_url
```

The API connection details can be retrieved through the Wallaroo client `mlops()` command.  This will display the connection URL, bearer token, and other information.  The bearer token is available for one hour before it expires.

For this example, the API connection details will be retrieved, then used to submit an inference request through the external inference URL retrieved earlier.

```python
connection =wl.mlops().__dict__
token = connection['token']
token
```

    'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJDYkFqN19QY0xCWTFkWmJiUDZ6Q3BsbkNBYTd6US0tRHlyNy0yLXlQb25nIn0.eyJleHAiOjE2ODQzNTc0NjYsImlhdCI6MTY4NDM1NzQwNiwiYXV0aF90aW1lIjoxNjg0MzU1OTU5LCJqdGkiOiIwYTA1Y2EyYS0xNzVhLTQ0MTctYTJhYi03MmFlYmE5YTA5NDMiLCJpc3MiOiJodHRwczovL2RvYy10ZXN0LmtleWNsb2FrLndhbGxhcm9vY29tbXVuaXR5Lm5pbmphL2F1dGgvcmVhbG1zL21hc3RlciIsImF1ZCI6WyJtYXN0ZXItcmVhbG0iLCJhY2NvdW50Il0sInN1YiI6IjAyOGM4YjQ4LWMzOWItNDU3OC05MTEwLTBiNWJkZDM4MjRkYSIsInR5cCI6IkJlYXJlciIsImF6cCI6InNkay1jbGllbnQiLCJzZXNzaW9uX3N0YXRlIjoiMGJlODJjN2ItNzg1My00ZjVkLWJiNWEtOTlkYjUwYjhiNDVmIiwiYWNyIjoiMCIsInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJkZWZhdWx0LXJvbGVzLW1hc3RlciIsIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iXX0sInJlc291cmNlX2FjY2VzcyI6eyJtYXN0ZXItcmVhbG0iOnsicm9sZXMiOlsibWFuYWdlLXVzZXJzIiwidmlldy11c2VycyIsInF1ZXJ5LWdyb3VwcyIsInF1ZXJ5LXVzZXJzIl19LCJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6InByb2ZpbGUgZW1haWwiLCJzaWQiOiIwYmU4MmM3Yi03ODUzLTRmNWQtYmI1YS05OWRiNTBiOGI0NWYiLCJlbWFpbF92ZXJpZmllZCI6ZmFsc2UsImh0dHBzOi8vaGFzdXJhLmlvL2p3dC9jbGFpbXMiOnsieC1oYXN1cmEtdXNlci1pZCI6IjAyOGM4YjQ4LWMzOWItNDU3OC05MTEwLTBiNWJkZDM4MjRkYSIsIngtaGFzdXJhLWRlZmF1bHQtcm9sZSI6InVzZXIiLCJ4LWhhc3VyYS1hbGxvd2VkLXJvbGVzIjpbInVzZXIiXSwieC1oYXN1cmEtdXNlci1ncm91cHMiOiJ7fSJ9LCJuYW1lIjoiSm9obiBIYW5zYXJpY2siLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJqb2huLmh1bW1lbEB3YWxsYXJvby5haSIsImdpdmVuX25hbWUiOiJKb2huIiwiZmFtaWx5X25hbWUiOiJIYW5zYXJpY2siLCJlbWFpbCI6ImpvaG4uaHVtbWVsQHdhbGxhcm9vLmFpIn0.oGodsdxjFu-XP8pzmET-MAUfrH-wNS-7y6WbT-OsQMyt_0xK5ilEHWD6YCtfREKbjmXd9U-a9LFSV3hIySO2truJeXQi6uQS3UTbvcoMdCdOMcnx9mYvxpGyiLw444AGHlbfKvqw4KLxaDi9pDWsyZFZkB8Ha1MLyvbaJvzWFWTsR2d12BptL1wdXFBiXtPfbywlKuUlpa4vDleGIAoZ3oywRdJ_wPsg5X2rCgr79BhwNufTXeKIAOjL_cZfuOZkASf8MzueT7aYGO3CMeWcFDkRcek3Svi58-7CTyTfYn3-0aQ0a73NjoNCb_Jta-cfFoTmCgD5G6h6SgftOXNh-Q'

```python
dataFile="./data/data_25k.arrow"
contentType="application/vnd.apache.arrow.file"
```

```python
!curl -X POST {inference_url} -H "Authorization: Bearer {token}" -H "Content-Type:{contentType}" --data-binary @{dataFile} > curl_response.df
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100 25.5M  100 20.8M  100 4874k  2151k   492k  0:00:09  0:00:09 --:--:-- 4494k

```python
cc_data_from_file =  pd.read_json('./curl_response.df', orient="records")
display(cc_data_from_file.head(5).loc[:, ["time","out"]])
```

{{<table "table table-striped table-bordered" >}}
<table>
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
      <td>1684357453186</td>
      <td>{'banjori': [0.0015195871], 'corebot': [0.9829148], 'cryptolocker': [0.012099565000000001], 'dircrypt': [4.7591344e-05], 'gozi': [2.0289392e-05], 'kraken': [0.0003197726], 'locky': [0.011029272000000001], 'main': [0.997564], 'matsnu': [0.010341625], 'pykspa': [0.008038965], 'qakbot': [0.016155062], 'ramdo': [0.006236233000000001], 'ramnit': [0.0009985756], 'simda': [1.793378e-26], 'suppobox': [1.3889898e-27]}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1684357453186</td>
      <td>{'banjori': [7.447225e-18], 'corebot': [6.7359245e-08], 'cryptolocker': [0.17081991], 'dircrypt': [1.3220147000000001e-09], 'gozi': [1.2758853e-24], 'kraken': [0.22559536], 'locky': [0.34209844], 'main': [0.99999994], 'matsnu': [0.30801848], 'pykspa': [0.18282163], 'qakbot': [3.8022553999999996e-11], 'ramdo': [0.20622534], 'ramnit': [0.15215826], 'simda': [1.17020745e-30], 'suppobox': [3.1514464999999997e-38]}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1684357453186</td>
      <td>{'banjori': [2.8599304999999997e-21], 'corebot': [9.302004999999999e-08], 'cryptolocker': [0.04445295], 'dircrypt': [6.1637580000000004e-09], 'gozi': [8.34974e-23], 'kraken': [0.48234479999999996], 'locky': [0.2633289], 'main': [1.0], 'matsnu': [0.29800323], 'pykspa': [0.22361766], 'qakbot': [1.5238920999999999e-06], 'ramdo': [0.3282038], 'ramnit': [0.029332466], 'simda': [1.1995533000000001e-31], 'suppobox': [0.0]}</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1684357453186</td>
      <td>{'banjori': [2.1386805e-15], 'corebot': [3.8817485e-10], 'cryptolocker': [0.045599725], 'dircrypt': [1.9090386e-07], 'gozi': [1.3139924000000002e-25], 'kraken': [0.59542614], 'locky': [0.17374131], 'main': [0.9999996999999999], 'matsnu': [0.2315157], 'pykspa': [0.17591687], 'qakbot': [1.087611e-09], 'ramdo': [0.21832284000000002], 'ramnit': [0.012869288000000001], 'simda': [6.158882e-28], 'suppobox': [1.438591e-35]}</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1684357453186</td>
      <td>{'banjori': [9.453381e-15], 'corebot': [7.091152e-10], 'cryptolocker': [0.049815107000000004], 'dircrypt': [5.2914135e-09], 'gozi': [7.4132087e-19], 'kraken': [1.5504637e-13], 'locky': [1.079181e-15], 'main': [0.9999988999999999], 'matsnu': [1.5003076000000002e-15], 'pykspa': [0.33075709999999997], 'qakbot': [2.6258948e-07], 'ramdo': [0.50362796], 'ramnit': [0.020393757000000002], 'simda': [0.0], 'suppobox': [0.0]}</td>
    </tr>
  </tbody>
</table>
{{</table>}}

## Undeploy Pipeline

When finished with our tests, we will undeploy the pipeline so we have the Kubernetes resources back for other tasks.  Note that if the deployment variable is unchanged pipeline.deploy() will restart the inference engine in the same configuration as before.

```python
pipeline.undeploy()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>gcpsdkpipeline</td></tr><tr><th>created</th> <td>2023-05-17 21:03:44.485720+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 21:03:49.137632+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>6398cafc-50c4-49e3-9499-6025b7808245, 7c043d3c-c894-4ae9-9ec1-c35518130b90</td></tr><tr><th>steps</th> <td>gcpsdkmodel</td></tr></table>
{{</table>}}

