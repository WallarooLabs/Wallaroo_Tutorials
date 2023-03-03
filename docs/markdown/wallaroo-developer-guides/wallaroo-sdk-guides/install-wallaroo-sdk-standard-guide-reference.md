This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/development/sdk-install-guides/standard-install).

## Installing the Wallaroo SDK

Organizations that develop machine learning models can deploy models to Wallaroo from their local systems to a Wallaroo instance through the [Wallaroo SDK](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/).  The following guide is created to assist users with installing the Wallaroo SDK and making a standard connection to a Wallaroo instance.

These instructions are based on the on the [Connect to Wallaroo](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/) guides.

This tutorial provides the following:

* `aloha-cnn-lstm.zip`: A pre-trained open source model that uses an [Aloha CNN LSTM model](https://www.researchgate.net/publication/348920204_Using_Auxiliary_Inputs_in_Deep_Learning_Models_for_Detecting_DGA-based_Domain_Names) for classifying Domain names as being either legitimate or being used for nefarious purposes such as malware distribution.
* Test Data Files:
  * `data-1.json`: 1 record
  * `data-1k.json`: 1,000 records
  * `data-25k.json`: 25,000 records

For this example, a virtual python environment will be used.  This will set the necessary libraries and specific Python version required.

## Prerequisites

The following is required for this tutorial:

* A Wallaroo instance version 2023.1 or later.
* Python 3.8.6 or later installed locally.
* [Conda](https://docs.conda.io/en/latest/):  Used for managing python virtual environments.

## General Steps

For our example, we will perform the following:

* Wallaroo SDK Install
  * Set up a Python virtual environment through `conda` with the libraries that enable the virtual environment for use in a Jupyter Hub environment.
  * Install the Wallaroo SDK.
* Wallaroo SDK from remote JupyterHub Demonstration (Optional):  The following steps are an optional exercise to demonstrate using the Wallaroo SDK from a remote connection.  The entire tutorial can be found on the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/sdk-install-guides/standard-install).
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

    ```bash
    pip install wallaroo==2022.4
    ```

For organizations who will be using the Wallaroo SDK with Jupyter or similar services, the conda virtual environment has been installed, it can either be selected as a new Jupyter Notebook kernel, or the Notebook's kernel can be set to an existing Jupyter notebook.

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

# used to display dataframe information without truncating
from IPython.display import display
import pandas as pd
pd.set_option('display.max_colwidth', None)
```


```python
wallaroo.__version__
```




    '2023.1.0rc1'




```python
# Client connection from local Wallaroo instance

# wl = wallaroo.Client()

# SSO login through keycloak

wallarooPrefix = "YOUR PREFIX"
wallarooSuffix = "YOUR SUFFIX"

wallarooPrefix = "sparkly-apple-3026"
wallarooSuffix = "wallaroo.community"

wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}.api.{wallarooSuffix}", 
                    auth_endpoint=f"https://{wallarooPrefix}.keycloak.{wallarooSuffix}", 
                    auth_type="sso")
```

### Arrow Support

As of the 2023.1 release, Wallaroo provides support for dataframe and Arrow for inference inputs.  This tutorial allows users to adjust their experience based on whether they have enabled Arrow support in their Wallaroo instance or not.

If Arrow support has been enabled, `arrowEnabled=True`. If disabled or you're not sure, set it to `arrowEnabled=False`

The examples below will be shown in an arrow enabled environment.


```python
import os
# Only set the below to make the OS environment ARROW_ENABLED to TRUE.  Otherwise, leave as is.
os.environ["ARROW_ENABLED"]="True"

if "ARROW_ENABLED" not in os.environ or os.environ["ARROW_ENABLED"].casefold() == "False".casefold():
    arrowEnabled = False
else:
    arrowEnabled = True
print(arrowEnabled)
```

    True


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




<table><tr><th>name</th> <td>sdkquickpipeline</td></tr><tr><th>created</th> <td>2023-02-22 21:25:42.312061+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-22 21:47:51.453099+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>19c972e4-41fa-42ea-be84-604e54b4b5ba, 37acbb17-8a8c-4169-8286-662e9cba3245, 6cd41955-6456-4401-84c6-72eb0b6b550b, f00085b6-a7d6-4297-92b7-9f52139e18ab, 109e38ce-908e-4821-ae14-a210e81f4def</td></tr><tr><th>steps</th> <td>sdkquickmodel</td></tr></table>



We can verify the workspace is created the current default workspace with the `get_current_workspace()` command.


```python
wl.get_current_workspace()
```




    {'name': 'sdkquickworkspace', 'id': 66, 'archived': False, 'created_by': '138bd7e6-4dc8-4dc1-a760-c9e721ef3c37', 'created_at': '2023-02-22T21:25:41.546584+00:00', 'models': [{'name': 'sdkquickmodel', 'versions': 2, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 2, 22, 21, 46, 28, 298510, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 2, 22, 21, 25, 44, 989355, tzinfo=tzutc())}], 'pipelines': [{'name': 'sdkquickpipeline', 'create_time': datetime.datetime(2023, 2, 22, 21, 25, 42, 312061, tzinfo=tzutc()), 'definition': '[]'}]}



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




<table><tr><th>name</th> <td>sdkquickpipeline</td></tr><tr><th>created</th> <td>2023-02-22 21:25:42.312061+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-22 21:47:51.453099+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>19c972e4-41fa-42ea-be84-604e54b4b5ba, 37acbb17-8a8c-4169-8286-662e9cba3245, 6cd41955-6456-4401-84c6-72eb0b6b550b, f00085b6-a7d6-4297-92b7-9f52139e18ab, 109e38ce-908e-4821-ae14-a210e81f4def</td></tr><tr><th>steps</th> <td>sdkquickmodel</td></tr></table>




```python
pipeline
```




<table><tr><th>name</th> <td>sdkquickpipeline</td></tr><tr><th>created</th> <td>2023-02-22 21:25:42.312061+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-22 21:47:51.453099+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>19c972e4-41fa-42ea-be84-604e54b4b5ba, 37acbb17-8a8c-4169-8286-662e9cba3245, 6cd41955-6456-4401-84c6-72eb0b6b550b, f00085b6-a7d6-4297-92b7-9f52139e18ab, 109e38ce-908e-4821-ae14-a210e81f4def</td></tr><tr><th>steps</th> <td>sdkquickmodel</td></tr></table>




```python
pipeline.deploy()
```




<table><tr><th>name</th> <td>sdkquickpipeline</td></tr><tr><th>created</th> <td>2023-02-22 21:25:42.312061+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-03 22:41:07.127677+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>a85e4f21-8afd-4d02-8daf-fcaf404ec9ba, 19c972e4-41fa-42ea-be84-604e54b4b5ba, 37acbb17-8a8c-4169-8286-662e9cba3245, 6cd41955-6456-4401-84c6-72eb0b6b550b, f00085b6-a7d6-4297-92b7-9f52139e18ab, 109e38ce-908e-4821-ae14-a210e81f4def</td></tr><tr><th>steps</th> <td>sdkquickmodel</td></tr></table>



We can verify that the pipeline is running and list what models are associated with it.


```python
pipeline.status()
```




    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.48.4.213',
       'name': 'engine-6d8d987474-z9xxg',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'sdkquickpipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'sdkquickmodel',
          'version': '9fc7f3d3-f337-471f-a60f-ad4c4b953fb9',
          'sha': 'd71d9ffc61aaac58c2b1ed70a2db13d1416fb9d3f5b891e5e4e2e97180fe22f8',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.48.4.212',
       'name': 'engine-lb-86bc6bd77b-bkt54',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}



## Interferences

### Infer 1 row

Now that the pipeline is deployed and our Aloha model is in place, we'll perform a smoke test to verify the pipeline is up and running properly.  We'll use the `infer_from_file` command to load a single encoded URL into the inference engine and print the results back out.

The result should tell us that the tokenized URL is legitimate (0) or fraud (1).  This sample data should return close to 0.


```python
## Demonstrate via straight infer
import json
if arrowEnabled is True:
    data = pd.read_json("./data/data_1.df.json")
    result = pipeline.infer(data)
    display(result.loc[:,["time","in.text_input","out.main", "check_failures"]])
else:
    data = json.load(open("./data/data_1.json"))
    result = pipeline.infer(data)
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
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.text_input</th>
      <th>out.main</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-03-03 22:42:20.661</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 16, 32, 23, 29, 32, 30, 19, 26, 17]</td>
      <td>[0.997564]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Infer from file
if arrowEnabled is True:
    result = pipeline.infer_from_file('./data/data_1.df.json')
    display(result.loc[:,["time","in.text_input","out.main", "check_failures"]])
else:
    result = pipeline.infer_from_file("./data/data_1.json")
    display(result[0].data())
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
      <th>out.main</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-03-03 22:42:35.237</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 16, 32, 23, 29, 32, 30, 19, 26, 17]</td>
      <td>[0.997564]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


### Batch Inference

Now that our smoke test is successful, let's really give it some data.  We have two inference files we can use:

* `data-1k.json`:  Contains 10,000 inferences
* `data-25k.json`: Contains 25,000 inferences

We'll pipe the `data-25k.json` file through the `pipeline` deployment URL, and place the results in a file named `response.txt`.  We'll also display the time this takes.  Note that for larger batches of 50,000 inferences or more can be difficult to view in Juypter Hub because of its size.

When retrieving the pipeline inference URL through an external SDK connection, the External Inference URL will be returned.  This URL will function provided that the **Enable external URL inference endpoints** is enabled.  For more information, see the [Wallaroo Model Endpoints Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-model-endpoints-guide/).


```python
external_url = pipeline._deployment._url()
external_url
```




    'https://sparkly-apple-3026.api.wallaroo.community/v1/api/pipelines/infer/sdkquickpipeline-47'



The API connection details can be retrieved through the Wallaroo client `mlops()` command.  This will display the connection URL, bearer token, and other information.  The bearer token is available for one hour before it expires.

For this example, the API connection details will be retrieved, then used to submit an inference request through the external inference URL retrieved earlier.


```python
connection =wl.mlops().__dict__
token = connection['token']
token
```




    'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJjTGZaYmhVQWl0a210Z0VLV0l1NnczTWlXYmUzWjc3cHdqVjJ2QWM2WUdZIn0.eyJleHAiOjE2NzcxMDI1NDQsImlhdCI6MTY3NzEwMjQ4NCwiYXV0aF90aW1lIjoxNjc3MTAxMTM3LCJqdGkiOiI4ZWRmOTE3MS1kMzg0LTQ5MGItYjUyZi01MmI2NWViMjNjZTUiLCJpc3MiOiJodHRwczovL3NwYXJrbHktYXBwbGUtMzAyNi5rZXljbG9hay53YWxsYXJvby5jb21tdW5pdHkvYXV0aC9yZWFsbXMvbWFzdGVyIiwiYXVkIjpbIm1hc3Rlci1yZWFsbSIsImFjY291bnQiXSwic3ViIjoiMTM4YmQ3ZTYtNGRjOC00ZGMxLWE3NjAtYzllNzIxZWYzYzM3IiwidHlwIjoiQmVhcmVyIiwiYXpwIjoic2RrLWNsaWVudCIsInNlc3Npb25fc3RhdGUiOiI1MGY3NWI3MC1kOWFmLTRmYTYtODVkZC1lMzU0MmVmYjE2N2IiLCJhY3IiOiIwIiwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbImNyZWF0ZS1yZWFsbSIsImRlZmF1bHQtcm9sZXMtbWFzdGVyIiwib2ZmbGluZV9hY2Nlc3MiLCJhZG1pbiIsInVtYV9hdXRob3JpemF0aW9uIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsibWFzdGVyLXJlYWxtIjp7InJvbGVzIjpbInZpZXctcmVhbG0iLCJ2aWV3LWlkZW50aXR5LXByb3ZpZGVycyIsIm1hbmFnZS1pZGVudGl0eS1wcm92aWRlcnMiLCJpbXBlcnNvbmF0aW9uIiwiY3JlYXRlLWNsaWVudCIsIm1hbmFnZS11c2VycyIsInF1ZXJ5LXJlYWxtcyIsInZpZXctYXV0aG9yaXphdGlvbiIsInF1ZXJ5LWNsaWVudHMiLCJxdWVyeS11c2VycyIsIm1hbmFnZS1ldmVudHMiLCJtYW5hZ2UtcmVhbG0iLCJ2aWV3LWV2ZW50cyIsInZpZXctdXNlcnMiLCJ2aWV3LWNsaWVudHMiLCJtYW5hZ2UtYXV0aG9yaXphdGlvbiIsIm1hbmFnZS1jbGllbnRzIiwicXVlcnktZ3JvdXBzIl19LCJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6ImVtYWlsIHByb2ZpbGUiLCJzaWQiOiI1MGY3NWI3MC1kOWFmLTRmYTYtODVkZC1lMzU0MmVmYjE2N2IiLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiaHR0cHM6Ly9oYXN1cmEuaW8vand0L2NsYWltcyI6eyJ4LWhhc3VyYS11c2VyLWlkIjoiMTM4YmQ3ZTYtNGRjOC00ZGMxLWE3NjAtYzllNzIxZWYzYzM3IiwieC1oYXN1cmEtZGVmYXVsdC1yb2xlIjoidXNlciIsIngtaGFzdXJhLWFsbG93ZWQtcm9sZXMiOlsidXNlciJdLCJ4LWhhc3VyYS11c2VyLWdyb3VwcyI6Int9In0sInByZWZlcnJlZF91c2VybmFtZSI6ImpvaG4uaGFuc2FyaWNrQHdhbGxhcm9vLmFpIiwiZW1haWwiOiJqb2huLmhhbnNhcmlja0B3YWxsYXJvby5haSJ9.ErXJsmadl3w3YleT29dZoo6TNoC8QOxMHVWHTyV9FVsUhIegpQlwTmVjHITtPu5aoX9EXW-lFOtz3-gduozU31sxwtPQG4DbWy62akxA2H11EepXg70AqzMZvFxeSp5blPI6p6miNxPTGsjLds6vlgZwd48IgDOg5RBSgS6uWCvMaj4AsvsUFuSxKuNkf_WDK10J_x8dq3osBGAytCUTwiF0ybPObVDBtg9UySv-PHQyUsJXLZl7DLsBOJ9fPy-Te4AHsKBjOg2UYL6KGzduQ_BIwDMb8obG2ILc2s2_rjnfZ_4sImp3A7DppohwjRNl18-caJhOSnVrYU0en_LDHw'




```python
if arrowEnabled is True:
    dataFile="./data/data_25k.df.json"
    contentType="application/json; format=pandas-records"
else:
    dataFile="./data/data_25k.json"
    contentType="application/json"
```


```python
!curl -X POST {external_url} -H "Content-Type:{contentType}" -H "Authorization: Bearer {token}" --data @{dataFile} > curl_response.txt
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100 34.3M  100 16.3M  100 18.0M  1115k  1229k  0:00:15  0:00:15 --:--:-- 4153k


## Undeploy Pipeline

When finished with our tests, we will undeploy the pipeline so we have the Kubernetes resources back for other tasks.  Note that if the deployment variable is unchanged pipeline.deploy() will restart the inference engine in the same configuration as before.


```python
pipeline.undeploy()
```


```python

```
