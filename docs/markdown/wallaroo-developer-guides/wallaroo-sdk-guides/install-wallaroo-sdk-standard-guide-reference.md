This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/development/sdk-install-guides/standard-install).

## Installing the Wallaroo SDK

Organizations that develop machine learning models can deploy models to Wallaroo from their local systems to a Wallaroo instance through the [Wallaroo SDK](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/).  The following guide is created to assist users with installing the Wallaroo SDK and making a standard connection to a Wallaroo instance.

These instructions are based on the on the [Connect to Wallaroo](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/) guides.

This tutorial provides the following:

* `aloha-cnn-lstm.zip`: A pre-trained open source model that uses an [Aloha CNN LSTM model](https://www.researchgate.net/publication/348920204_Using_Auxiliary_Inputs_in_Deep_Learning_Models_for_Detecting_DGA-based_Domain_Names) for classifying Domain names as being either legitimate or being used for nefarious purposes such as malware distribution.
* Test Data Files:
  * `data_1k.arrow`: 1,000 records
  * `data_25k.arrow`: 25,000 records

For this example, a virtual python environment will be used.  This will set the necessary libraries and specific Python version required.

## Prerequisites

The following is required for this tutorial:

* A Wallaroo instance version 2023.1 or later.
* Python 3.8.6 or later installed locally.
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
* Wallaroo SDK from remote JupyterHub Demonstration (Optional):  The following steps are an optional exercise to demonstrate using the Wallaroo SDK from a remote connection.  The entire tutorial can be found on the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/development/sdk-install-guides/standard-install).
  * Connect to a remote Wallaroo instance.
  * Create a workspace for our work.
  * Upload the Aloha model.
  * Create a pipeline that can ingest our submitted data, submit it to the model, and export the results
  * Run a sample inference through our pipeline by loading a file
  * Retrieve the external deployment URL.  This sample Wallaroo instance has been configured to create external inference URLs for pipelines.  For more information, see the [External Inference URL Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-model-endpoints-guide/).
  * Run a sample inference through our pipeline's external URL and store the results in a file.  This assumes that the External Inference URLs have been enabled for the target Wallaroo instance.
  * Undeploy the pipeline and return resources back to the Wallaroo instance's Kubernetes environment.

## Install Wallaroo SDK

### Set Up Virtual Python Environment

To set up the Python virtual environment for use of the Wallaroo SDK:

1. From a terminal shell, create the Python virtual environment with `conda`.  Replace `wallaroosdk` with the name of the virtual environment as required by your organization.  Note that Python 3.8.6 and above is specified as a requirement for Python libraries used with the Wallaroo SDK.  The following will install the latest version of Python 3.8.

    ```bash
    conda create -n wallaroosdk python=3.8
    ```

1. Activate the new environment.

    ```bash
    conda activate wallaroosdk
    ```

1. (Optional) For organizations who want to use the Wallaroo SDk from within Jupyter and similar environments:
    1. Install the `ipykernel` library.  This allows the JupyterHub notebooks to access the Python virtual environment as a kernel, and it required for the second part of this tutorial.

        ```bash
        conda install ipykernel
        ```
    
    1. Install the new virtual environment as a python kernel.

        ```bash
        ipython kernel install --user --name=wallaroosdk
        ```
    
1. Install the Wallaroo SDK.  This process may take several minutes while the other required Python libraries are added to the virtual environment.

    * **IMPORTANT NOTE**:  The version of the Wallaroo SDK should match the Wallaroo instance.  For example, this example connects to a Wallaroo Enterprise version `2023.1` instance, so the SDK version should be `wallaroo==2023.1.0`.

    ```bash
    pip install wallaroo==2023.2.1
    ```

For organizations who will be using the Wallaroo SDK with Jupyter or similar services, the conda virtual environment has been installed, it can either be selected as a new Jupyter Notebook kernel, or the Notebook's kernel can be set to an existing Jupyter notebook.

{{<figure src="/images/current/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-install-guides/wallaroo-select-kernel.png" width="800" label="">}}

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

{{<figure src="/images/current/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-install-guides/wallaroo-sdk-login.png" width="800" label="Wallaroo Login">}}

Once authenticated, the user will verify adding the device the user is establishing the connection from.  Once both steps are complete, then the connection is granted.

{{<figure src="/images/current/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-install-guides/wallaroo-device-access.png" width="800" label="Device Registration">}}

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
wallaroo.__version__
```

    '2023.2.0rc3'

### Connect to Wallaroo

For this example, a connection through the Wallaroo SDK is used.  For more information, see the [Wallaroo SDK Essentials Guide:  Client Connection](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

For `wallarooPrefix = "YOUR PREFIX."` and `wallarooSuffix = "YOUR SUFFIX"`, enter the prefix and suffix for your Wallaroo instance DNS name.  If the prefix instance is blank, then it can be `wallarooPrefix = ""`.  **Note that the prefix includes the `.` for proper formatting.**  For example, if the prefix is empty and the suffix is `wallaroo.example.com`, then the settings would be:

```python
wallarooPrefix = ""
wallarooSuffix = "wallaroo.example.com"
```

If the prefix is `sales.` and the suffix `example.com`, then the settings would be:

```python
wallarooPrefix = "sales."
wallarooSuffix = "wallaroo.example.com"
```

```python
# SSO login through keycloak

wallarooPrefix = "YOUR PREFIX."
wallarooSuffix = "YOUR SUFFIX"

wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}api.{wallarooSuffix}", 
                    auth_endpoint=f"https://{wallarooPrefix}keycloak.{wallarooSuffix}", 
                    auth_type="sso")
```

## Wallaroo Remote SDK Examples

The following examples can be used by an organization to test using the Wallaroo SDK from a remote location from their Wallaroo instance.  These examples show how to create workspaces, deploy pipelines, and perform inferences through the SDK and API.

### Create the Workspace

We will create a workspace to work in and call it the `sdkworkspace`, then set it as current workspace environment.  We'll also create our pipeline in advance as `sdkpipeline`.

* **IMPORTANT NOTE**:  For this example, the Aloha model is stored in the file `alohacnnlstm.zip`.  When using tensor based models, the zip file **must** match the name of the tensor directory.  For example, if the tensor directory is `alohacnnlstm`, then the .zip file must be named `alohacnnlstm.zip`.

```python
workspace_name = 'sdkquickworkspace'
pipeline_name = 'sdkquickpipeline'
model_name = 'sdkquickmodel'
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

<table><tr><th>name</th> <td>sdkquickpipeline</td></tr><tr><th>created</th> <td>2023-05-17 20:43:38.111213+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 20:43:38.111213+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>d72c468a-a0e2-4189-aa7a-4e27127a2f2b</td></tr><tr><th>steps</th> <td></td></tr></table>

We can verify the workspace is created the current default workspace with the `get_current_workspace()` command.

```python
wl.get_current_workspace()
```

    {'name': 'sdkquickworkspace', 'id': 6, 'archived': False, 'created_by': '028c8b48-c39b-4578-9110-0b5bdd3824da', 'created_at': '2023-05-17T20:43:36.727099+00:00', 'models': [], 'pipelines': [{'name': 'sdkquickpipeline', 'create_time': datetime.datetime(2023, 5, 17, 20, 43, 38, 111213, tzinfo=tzutc()), 'definition': '[]'}]}

### Upload the Models

Now we will upload our model.  Note that for this example we are applying the model from a .ZIP file.  The Aloha model is a [protobuf](https://developers.google.com/protocol-buffers) file that has been defined for evaluating web pages, and we will configure it to use data in the `tensorflow` format.

```python
from wallaroo.framework import Framework
model = wl.upload_model(model_name, model_file_name, framework=Framework.TENSORFLOW).configure("tensorflow")
```

### Deploy a Model

Now that we have a model that we want to use we will create a deployment for it. 

We will tell the deployment we are using a tensorflow model and give the deployment name and the configuration we want for the deployment.

To do this, we'll create our pipeline that can ingest the data, pass the data to our Aloha model, and give us a final output.  We'll call our pipeline `externalsdkpipeline`, then deploy it so it's ready to receive data.  The deployment process usually takes about 45 seconds.

```python
pipeline.add_model_step(model)
```

<table><tr><th>name</th> <td>sdkquickpipeline</td></tr><tr><th>created</th> <td>2023-05-17 20:43:38.111213+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 20:43:38.111213+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>d72c468a-a0e2-4189-aa7a-4e27127a2f2b</td></tr><tr><th>steps</th> <td></td></tr></table>

```python
pipeline
```

<table><tr><th>name</th> <td>sdkquickpipeline</td></tr><tr><th>created</th> <td>2023-05-17 20:43:38.111213+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 20:43:38.111213+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>d72c468a-a0e2-4189-aa7a-4e27127a2f2b</td></tr><tr><th>steps</th> <td></td></tr></table>

```python
pipeline.deploy()
```

<table><tr><th>name</th> <td>sdkquickpipeline</td></tr><tr><th>created</th> <td>2023-05-17 20:43:38.111213+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 20:43:46.036128+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>2bd5c838-f7cc-4f48-91ea-28a9ce0f7ed8, d72c468a-a0e2-4189-aa7a-4e27127a2f2b</td></tr><tr><th>steps</th> <td>sdkquickmodel</td></tr></table>

We can verify that the pipeline is running and list what models are associated with it.

```python
pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.3.134',
       'name': 'engine-6c9b85fd76-rjdxp',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'sdkquickpipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'sdkquickmodel',
          'version': 'f404ffe0-33ca-44cf-8a19-731e088c3262',
          'sha': 'd71d9ffc61aaac58c2b1ed70a2db13d1416fb9d3f5b891e5e4e2e97180fe22f8',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.4.164',
       'name': 'engine-lb-584f54c899-wh22c',
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
      <td>2023-05-17 20:44:04.622</td>
      <td>[0.997564]</td>
    </tr>
  </tbody>
</table>

## Infer 1,000 Rows

We can also infer an entire batch as one request either with the Pipeline `infer` method with multiple rows, or loaded from a file using the Pipeline `infer_from_file` method.  For this example, we will run a batch on 1,000 records using the file `data_1k.arrow`.  This is an Apache Arrow table, which gives the added benefit of speed and lower file size as a binary file rather than a text JSON file.

We'll infer the 1,000 records, then convert it to a DataFrame and display the first 5 to save space in our Jupyter Notebook.

```python
result = pipeline.infer_from_file('./data/data_1k.arrow')

outputs = result.to_pandas()
display(outputs.head(5).loc[:, ["time","out.main"]])
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
      <td>2023-05-17 20:44:05.905</td>
      <td>[0.997564]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-05-17 20:44:05.905</td>
      <td>[0.9885122]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-05-17 20:44:05.905</td>
      <td>[0.9993358]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-05-17 20:44:05.905</td>
      <td>[0.99999857]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-05-17 20:44:05.905</td>
      <td>[0.9984837]</td>
    </tr>
  </tbody>
</table>

### Batch Inference

Now that our smoke test is successful, let's really give it some data.  We have two inference files we can use:

* `data_1k.arrow`:  Contains 10,000 inferences
* `data_25k.arrow`: Contains 25,000 inferences

We'll pipe the `data-25k.json` file through the `pipeline` deployment URL, and place the results in a file named `response.txt`.  We'll also display the time this takes.  Note that for larger batches of 50,000 inferences or more can be difficult to view in Juypter Hub because of its size.

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

    'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJDYkFqN19QY0xCWTFkWmJiUDZ6Q3BsbkNBYTd6US0tRHlyNy0yLXlQb25nIn0.eyJleHAiOjE2ODQzNTYyNzUsImlhdCI6MTY4NDM1NjIxNSwiYXV0aF90aW1lIjoxNjg0MzU1OTU5LCJqdGkiOiI1ZjRiNDQ1Zi1hNmI1LTQ2NzgtODZhMS05MWRjMjMxMjJiODIiLCJpc3MiOiJodHRwczovL2RvYy10ZXN0LmtleWNsb2FrLndhbGxhcm9vY29tbXVuaXR5Lm5pbmphL2F1dGgvcmVhbG1zL21hc3RlciIsImF1ZCI6WyJtYXN0ZXItcmVhbG0iLCJhY2NvdW50Il0sInN1YiI6IjAyOGM4YjQ4LWMzOWItNDU3OC05MTEwLTBiNWJkZDM4MjRkYSIsInR5cCI6IkJlYXJlciIsImF6cCI6InNkay1jbGllbnQiLCJzZXNzaW9uX3N0YXRlIjoiMGJlODJjN2ItNzg1My00ZjVkLWJiNWEtOTlkYjUwYjhiNDVmIiwiYWNyIjoiMCIsInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJkZWZhdWx0LXJvbGVzLW1hc3RlciIsIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iXX0sInJlc291cmNlX2FjY2VzcyI6eyJtYXN0ZXItcmVhbG0iOnsicm9sZXMiOlsibWFuYWdlLXVzZXJzIiwidmlldy11c2VycyIsInF1ZXJ5LWdyb3VwcyIsInF1ZXJ5LXVzZXJzIl19LCJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6InByb2ZpbGUgZW1haWwiLCJzaWQiOiIwYmU4MmM3Yi03ODUzLTRmNWQtYmI1YS05OWRiNTBiOGI0NWYiLCJlbWFpbF92ZXJpZmllZCI6ZmFsc2UsImh0dHBzOi8vaGFzdXJhLmlvL2p3dC9jbGFpbXMiOnsieC1oYXN1cmEtdXNlci1pZCI6IjAyOGM4YjQ4LWMzOWItNDU3OC05MTEwLTBiNWJkZDM4MjRkYSIsIngtaGFzdXJhLWRlZmF1bHQtcm9sZSI6InVzZXIiLCJ4LWhhc3VyYS1hbGxvd2VkLXJvbGVzIjpbInVzZXIiXSwieC1oYXN1cmEtdXNlci1ncm91cHMiOiJ7fSJ9LCJuYW1lIjoiSm9obiBIYW5zYXJpY2siLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJqb2huLmh1bW1lbEB3YWxsYXJvby5haSIsImdpdmVuX25hbWUiOiJKb2huIiwiZmFtaWx5X25hbWUiOiJIYW5zYXJpY2siLCJlbWFpbCI6ImpvaG4uaHVtbWVsQHdhbGxhcm9vLmFpIn0.MS3R4uCav5m3UUQwxrKIXWLgw86DOoWnGJ9TSLAzbn-AVUliDNzBCt2LwWRgOmiUAlIrhFd2l8MCfHg4Ye8M7ucM099OH1VTmaXKOxBVc3Vlbr3tZQD9QcdDkonxe8DttrnOzMVxYKwci7YAiwEEz9eLmLvgxP6FSSl1rMMdxnTdAU9VFEDMUq00p3pu7hFGzsR_4XhGXOyJep-vhNvzP0eh1w3OVh11VjxaFs8a-2jXqYwBByg4oh6jArJ6Nhjcobcv0bqW2h5Q9stL_hT2ZNPd0F5bycc7UQRtDQ0Mt1JbNvm2G6pz3n3vo1nQJQ5y08fnQph6bvNT6LEpIOAEZA'

```python
dataFile="./data/data_25k.arrow"
contentType="application/vnd.apache.arrow.file"
```

```python
!curl -X POST {inference_url} -H "Authorization: Bearer {token}" -H "Content-Type:{contentType}" --data-binary @{dataFile} > curl_response.df
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100 25.6M  100 20.8M  100 4874k   771k   176k  0:00:27  0:00:27 --:--:-- 1161k0:00:35  0:00:19  0:00:16 1079k

```python
cc_data_from_file =  pd.read_json('./curl_response.df', orient="records")
display(cc_data_from_file.head(5).loc[:, ["time","out"]])
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
      <td>1684356249123</td>
      <td>{'banjori': [0.0015195821], 'corebot': [0.9829147500000001], 'cryptolocker': [0.012099549000000001], 'dircrypt': [4.7591115e-05], 'gozi': [2.0289428e-05], 'kraken': [0.00031977256999999996], 'locky': [0.011029262000000001], 'main': [0.997564], 'matsnu': [0.010341609], 'pykspa': [0.008038961], 'qakbot': [0.016155055], 'ramdo': [0.00623623], 'ramnit': [0.0009985747000000001], 'simda': [1.7933434e-26], 'suppobox': [1.388995e-27]}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1684356249123</td>
      <td>{'banjori': [7.447196e-18], 'corebot': [6.7359245e-08], 'cryptolocker': [0.1708199], 'dircrypt': [1.3220122000000002e-09], 'gozi': [1.2758705999999999e-24], 'kraken': [0.22559543], 'locky': [0.34209849999999997], 'main': [0.99999994], 'matsnu': [0.3080186], 'pykspa': [0.1828217], 'qakbot': [3.8022549999999994e-11], 'ramdo': [0.2062254], 'ramnit': [0.15215826], 'simda': [1.1701982e-30], 'suppobox': [3.1514454e-38]}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1684356249123</td>
      <td>{'banjori': [2.8598648999999997e-21], 'corebot': [9.302004000000001e-08], 'cryptolocker': [0.04445298], 'dircrypt': [6.1637580000000004e-09], 'gozi': [8.3496755e-23], 'kraken': [0.48234479999999996], 'locky': [0.26332903], 'main': [1.0], 'matsnu': [0.29800338], 'pykspa': [0.22361776], 'qakbot': [1.5238920999999999e-06], 'ramdo': [0.32820392], 'ramnit': [0.029332489000000003], 'simda': [1.1995622e-31], 'suppobox': [0.0]}</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1684356249123</td>
      <td>{'banjori': [2.1387213e-15], 'corebot': [3.8817485e-10], 'cryptolocker': [0.045599736], 'dircrypt': [1.9090386e-07], 'gozi': [1.3140123e-25], 'kraken': [0.59542626], 'locky': [0.17374137], 'main': [0.9999996999999999], 'matsnu': [0.23151578], 'pykspa': [0.17591679999999998], 'qakbot': [1.0876152e-09], 'ramdo': [0.21832279999999998], 'ramnit': [0.0128692705], 'simda': [6.1588803e-28], 'suppobox': [1.4386237e-35]}</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1684356249123</td>
      <td>{'banjori': [9.453342500000001e-15], 'corebot': [7.091151e-10], 'cryptolocker': [0.049815163], 'dircrypt': [5.2914135e-09], 'gozi': [7.4132087e-19], 'kraken': [1.5504574999999998e-13], 'locky': [1.079181e-15], 'main': [0.9999988999999999], 'matsnu': [1.5003075e-15], 'pykspa': [0.33075705], 'qakbot': [2.6258850000000004e-07], 'ramdo': [0.5036279], 'ramnit': [0.020393765], 'simda': [0.0], 'suppobox': [2.3292326e-38]}</td>
    </tr>
  </tbody>
</table>

## Undeploy Pipeline

When finished with our tests, we will undeploy the pipeline so we have the Kubernetes resources back for other tasks.  Note that if the deployment variable is unchanged pipeline.deploy() will restart the inference engine in the same configuration as before.

```python
pipeline.undeploy()
```

<table><tr><th>name</th> <td>sdkquickpipeline</td></tr><tr><th>created</th> <td>2023-05-17 20:43:38.111213+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 20:43:46.036128+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>2bd5c838-f7cc-4f48-91ea-28a9ce0f7ed8, d72c468a-a0e2-4189-aa7a-4e27127a2f2b</td></tr><tr><th>steps</th> <td>sdkquickmodel</td></tr></table>

