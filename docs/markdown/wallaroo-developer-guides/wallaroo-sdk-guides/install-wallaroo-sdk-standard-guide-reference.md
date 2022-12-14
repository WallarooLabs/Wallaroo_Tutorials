This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/sdk-install-guides/standard-install).

## Installing the Wallaroo SDK

Organizations that develop machine learning models can deploy models to Wallaroo from their local systems to a Wallaroo instance through the [Wallaroo SDK](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/).  The following guide is created to assist users with installing the Wallaroo SDK and making a standard connection to a Wallaroo instance.

These instructions are based on the on the [Connect to Wallaroo](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/#connect-to-wallaroo) guides.

This tutorial provides the following:

* `aloha-cnn-lstm.zip`: A pre-trained open source model that uses an [Aloha CNN LSTM model](https://www.researchgate.net/publication/348920204_Using_Auxiliary_Inputs_in_Deep_Learning_Models_for_Detecting_DGA-based_Domain_Names) for classifying Domain names as being either legitimate or being used for nefarious purposes such as malware distribution.
* `data-1.json`, `data-1k.json`, `data-25k.json`: Data files with 1, 1,000, and 25,000 records for testing.

For this example, a virtual python environment will be used.  This will set the necessary libraries and specific Python version required.

## Prerequisites

The following is required for this tutorial:

* A Wallaroo instance version 2022.4 or later.
* Python 3.8.6 or later installed locally
* [Conda](https://docs.conda.io/en/latest/):  Used for managing python virtual environments.

## General Steps

For our example, we will perform the following:

* Wallaroo SDK Install
  * Set up a Python virtual environment through `conda` with the libraries that enable the virtual environment for use in a Jupyter Hub environment.
  * Install the Wallaroo Python SDK.
  * Connect to a remote Wallaroo instance.  This instance is configured to use the standard Keycloak service.
* Wallaroo SDK from remote JupyterHub Demonstration (Optional)
  * The following steps are used to demonstrate using the Wallaroo SDK in a external from the Wallaroo instance JupyterHub or Jupyter Notebook environment.  The entire tutorial can be found on the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/sdk-install-guides/standard-install).
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

1. From a terminal shell, create the Python virtual environment with `conda`.  Replace `wallaroosdk` with the name of the virtual environment as required by your organization.  Note that Python 3.8.6 and above is specified as a requirement for Python libraries used with the Wallaroo SDK.  The following will install the latest version of Python 3.8, which as of this time is 3.8.15.

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
    pip install wallaroo==2022.4.0rc3
    ```

For organizations who will be using the Wallaroo SDK with Jupyter or similar services, the conda virtual environment has been installed, it can either be selected as a new Jupyter Notebook kernel, or the Notebook's kernel can be set to an existing Jupyter notebook.

![](/images/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-install-guides/wallaroo-select-kernel.png)

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

This is accomplished using the `wallaroo.Client(api_endpoint, auth_endpoint, auth_type command)` command that connects to the Wallaroo instance services.  For more information on the DNS names of Wallaroo services, see the [DNS Integration Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-dns-guide/).

The `Client` method takes the following parameters:

* **api_endpoint** (*String*): The URL to the Wallaroo instance API service.
* **auth_endpoint** (*String*): The URL to the Wallaroo instance Keycloak service.
* **auth_type command** (*String*): The authorization type.  In this case, `SSO`.

Once run, the `wallaroo.Client` command provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Depending on the configuration of the Wallaroo instance, the user will either be presented with a login request to the Wallaroo instance or be authenticated through a broker such as Google, Github, etc.  To use the broker, select it from the list under the username/password login forms.  For more information on Wallaroo authentication configurations, see the [Wallaroo Authentication Configuration Guides](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-sso-authentication/).

![Wallaroo Login](/images/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-install-guides/wallaroo-sdk-login.png)

Once authenticated, the user will verify adding the device the user is establishing the connection from.  Once both steps are complete, then the connection is granted.

![Device Registration](/images/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-install-guides/wallaroo-device-access.png)

The connection is stored in the variable `wl` for use in all other Wallaroo calls.

```python
import wallaroo
from wallaroo.object import EntityNotFoundError
```

```python
wallaroo.__version__
```

    '2022.4.0rc3'

```python
# SSO login through keycloak

wallarooPrefix = "YOURPREFIX"
wallarooSuffix = "YOURSUFFIX"

wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}.api.{wallarooSuffix}", 
                    auth_endpoint=f"https://{wallarooPrefix}.keycloak.{wallarooSuffix}", 
                    auth_type="sso")
```

## Wallaroo Remote SDK Examples

The following examples can be used by an organization to test using the Wallaroo SDK from a remote location from their Wallaroo instance.  These examples show how to create workspaces, deploy pipelines, and perform inferences through the SDK and API.

### Create the Workspace

We will create a workspace to work in and call it the `sdkworkspace`, then set it as current workspace environment.  We'll also create our pipeline in advance as `sdkpipeline`.

```python
workspace_name = 'sdkworkspace'
pipeline_name = 'sdkpipeline'
model_name = 'sdkmodel'
model_file_name = './aloha-cnn-lstm.zip'
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

<table><tr><th>name</th> <td>sdkpipeline</td></tr><tr><th>created</th> <td>2022-12-12 22:53:13.761344+00:00</td></tr><tr><th>last_updated</th> <td>2022-12-14 15:48:18.202824+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>751d6910-14c2-47d2-bd33-debf39bb475b, 85cb2cc5-39a2-4e68-b5bf-e3cceb270df2, 862b5c66-98a6-4dee-9c92-4c82d7ce49a6, 7a373546-27b8-4541-bc96-33a30e14200c, 929a93db-1478-400a-8e21-0ecfd8090faf, 682dc9af-c3b7-401d-a2bd-d8511dfa3bcc</td></tr><tr><th>steps</th> <td>sdkmodel</td></tr></table>

We can verify the workspace is created the current default workspace with the `get_current_workspace()` command.

```python
wl.get_current_workspace()
```

    {'name': 'sdkworkspace', 'id': 3, 'archived': False, 'created_by': 'f1f32bdf-9bd9-4595-a531-aca5778ceaf0', 'created_at': '2022-12-12T22:53:12.691709+00:00', 'models': [{'name': 'sdkmodel', 'versions': 5, 'owner_id': '""', 'last_update_time': datetime.datetime(2022, 12, 14, 15, 48, 16, 779883, tzinfo=tzutc()), 'created_at': datetime.datetime(2022, 12, 12, 22, 53, 17, 576900, tzinfo=tzutc())}], 'pipelines': [{'name': 'sdkpipeline', 'create_time': datetime.datetime(2022, 12, 12, 22, 53, 13, 761344, tzinfo=tzutc()), 'definition': '[]'}]}

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

<table><tr><th>name</th> <td>sdkpipeline</td></tr><tr><th>created</th> <td>2022-12-12 22:53:13.761344+00:00</td></tr><tr><th>last_updated</th> <td>2022-12-14 15:48:18.202824+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>751d6910-14c2-47d2-bd33-debf39bb475b, 85cb2cc5-39a2-4e68-b5bf-e3cceb270df2, 862b5c66-98a6-4dee-9c92-4c82d7ce49a6, 7a373546-27b8-4541-bc96-33a30e14200c, 929a93db-1478-400a-8e21-0ecfd8090faf, 682dc9af-c3b7-401d-a2bd-d8511dfa3bcc</td></tr><tr><th>steps</th> <td>sdkmodel</td></tr></table>

```python
pipeline.deploy()
```

<table><tr><th>name</th> <td>sdkpipeline</td></tr><tr><th>created</th> <td>2022-12-12 22:53:13.761344+00:00</td></tr><tr><th>last_updated</th> <td>2022-12-14 15:54:43.308811+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>ef60d525-e959-4b4f-acb8-e7f1c9540668, 751d6910-14c2-47d2-bd33-debf39bb475b, 85cb2cc5-39a2-4e68-b5bf-e3cceb270df2, 862b5c66-98a6-4dee-9c92-4c82d7ce49a6, 7a373546-27b8-4541-bc96-33a30e14200c, 929a93db-1478-400a-8e21-0ecfd8090faf, 682dc9af-c3b7-401d-a2bd-d8511dfa3bcc</td></tr><tr><th>steps</th> <td>sdkmodel</td></tr></table>

We can verify that the pipeline is running and list what models are associated with it.

```python
pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.5.8',
       'name': 'engine-67bb596df5-7qcrs',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'sdkpipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'sdkmodel',
          'version': '7f8d1a2e-4308-4a6d-961c-a14623c5e2bf',
          'sha': '7c89707252ce389980d5348c37885d6d72af4c20cd303422e2de7e66dd7ff184',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.2.8',
       'name': 'engine-lb-c6485cfd5-wbxfc',
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

file = open('./data-1.json')

data = json.load(file)

result = pipeline.infer(data)
print(result)
```

    [InferenceResult({'check_failures': [],
     'elapsed': 228355973,
     'model_name': 'sdkmodel',
     'model_version': '7f8d1a2e-4308-4a6d-961c-a14623c5e2bf',
     'original_data': {'text_input': [[0,
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
                                       17]]},
     'outputs': [{'Float': {'data': [0.0015195857267826796],
                            'dim': [1, 1],
                            'dtype': 'Float',
                            'v': 1}},
                 {'Float': {'data': [0.9829148054122925],
                            'dim': [1, 1],
                            'dtype': 'Float',
                            'v': 1}},
                 {'Float': {'data': [0.012099565006792545],
                            'dim': [1, 1],
                            'dtype': 'Float',
                            'v': 1}},
                 {'Float': {'data': [4.759129660669714e-05],
                            'dim': [1, 1],
                            'dtype': 'Float',
                            'v': 1}},
                 {'Float': {'data': [2.0289371605031192e-05],
                            'dim': [1, 1],
                            'dtype': 'Float',
                            'v': 1}},
                 {'Float': {'data': [0.0003197726036887616],
                            'dim': [1, 1],
                            'dtype': 'Float',
                            'v': 1}},
                 {'Float': {'data': [0.011029282584786415],
                            'dim': [1, 1],
                            'dtype': 'Float',
                            'v': 1}},
                 {'Float': {'data': [0.9975640177726746],
                            'dim': [1, 1],
                            'dtype': 'Float',
                            'v': 1}},
                 {'Float': {'data': [0.010341615416109562],
                            'dim': [1, 1],
                            'dtype': 'Float',
                            'v': 1}},
                 {'Float': {'data': [0.008038961328566074],
                            'dim': [1, 1],
                            'dtype': 'Float',
                            'v': 1}},
                 {'Float': {'data': [0.0161550622433424],
                            'dim': [1, 1],
                            'dtype': 'Float',
                            'v': 1}},
                 {'Float': {'data': [0.006236232817173004],
                            'dim': [1, 1],
                            'dtype': 'Float',
                            'v': 1}},
                 {'Float': {'data': [0.0009985751239582896],
                            'dim': [1, 1],
                            'dtype': 'Float',
                            'v': 1}},
                 {'Float': {'data': [1.7933780470763777e-26],
                            'dim': [1, 1],
                            'dtype': 'Float',
                            'v': 1}},
                 {'Float': {'data': [1.3889951203666574e-27],
                            'dim': [1, 1],
                            'dtype': 'Float',
                            'v': 1}}],
     'pipeline_name': 'sdkpipeline',
     'shadow_data': {},
     'time': 1671033309872})]

```python
# Demonstrate from infer_from_file
result = pipeline.infer_from_file("./data-1.json")
```

```python
result[0].data()
```

    [array([[0.00151959]]),
     array([[0.98291481]]),
     array([[0.01209957]]),
     array([[4.75912966e-05]]),
     array([[2.02893716e-05]]),
     array([[0.00031977]]),
     array([[0.01102928]]),
     array([[0.99756402]]),
     array([[0.01034162]]),
     array([[0.00803896]]),
     array([[0.01615506]]),
     array([[0.00623623]]),
     array([[0.00099858]]),
     array([[1.79337805e-26]]),
     array([[1.38899512e-27]])]

### Batch Inference

Now that our smoke test is successful, let's really give it some data.  We have two inference files we can use:

* `data-1k.json`:  Contains 1,0000 inferences
* `data-25k.json`: Contains 25,000 inferences

We'll pipe the `data-25k.json` file through the `pipeline` deployment URL, and place the results in a file named `response.txt`.  We'll also display the time this takes.  Note that for larger batches of 50,000 inferences or more can be difficult to view in Juypter Hub because of its size.

When retrieving the pipeline inference URL through an external SDK connection, the External Inference URL will be returned.  This URL will function provided that the **Enable external URL inference endpoints** is enabled.  For more information, see the [Wallaroo Model Endpoints Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-model-endpoints-guide/).

```python
external_url = pipeline._deployment._url()
external_url
```

    'https://magical-bear-3782.api.wallaroo.community/v1/api/pipelines/infer/sdkpipeline-3'

The API connection details can be retrieved through the Wallaroo client `mlops()` command.  This will display the connection URL, bearer token, and other information.  The bearer token is available for one hour before it expires.

For this example, the API connection details will be retrieved, then used to submit an inference request through the external inference URL retrieved earlier.

```python
connection =wl.mlops().__dict__
token = connection['token']
token
```

    'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJnTHBSY1B6QkhjQ1k1RTFHTVZoTlQtelI0VDY2YUM0QWh2eXVORmpVOTBjIn0.eyJleHAiOjE2NzEwMzMzMzUsImlhdCI6MTY3MTAzMzI3NSwiYXV0aF90aW1lIjoxNjcxMDMyODgyLCJqdGkiOiJiNDk3YmM3Yy1kMTc5LTRhYWQtODdmZC0yZGJiYTBlZDI4ZDYiLCJpc3MiOiJodHRwczovL21hZ2ljYWwtYmVhci0zNzgyLmtleWNsb2FrLndhbGxhcm9vLmNvbW11bml0eS9hdXRoL3JlYWxtcy9tYXN0ZXIiLCJhdWQiOlsibWFzdGVyLXJlYWxtIiwiYWNjb3VudCJdLCJzdWIiOiJmMWYzMmJkZi05YmQ5LTQ1OTUtYTUzMS1hY2E1Nzc4Y2VhZjAiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJzZGstY2xpZW50Iiwic2Vzc2lvbl9zdGF0ZSI6IjYzYzNiZjYwLTNmNjMtNDBjNC05NmI1LWNiYTk4ZjZhOGNmNyIsImFjciI6IjEiLCJyZWFsbV9hY2Nlc3MiOnsicm9sZXMiOlsiY3JlYXRlLXJlYWxtIiwiZGVmYXVsdC1yb2xlcy1tYXN0ZXIiLCJvZmZsaW5lX2FjY2VzcyIsImFkbWluIiwidW1hX2F1dGhvcml6YXRpb24iXX0sInJlc291cmNlX2FjY2VzcyI6eyJtYXN0ZXItcmVhbG0iOnsicm9sZXMiOlsidmlldy1pZGVudGl0eS1wcm92aWRlcnMiLCJ2aWV3LXJlYWxtIiwibWFuYWdlLWlkZW50aXR5LXByb3ZpZGVycyIsImltcGVyc29uYXRpb24iLCJjcmVhdGUtY2xpZW50IiwibWFuYWdlLXVzZXJzIiwicXVlcnktcmVhbG1zIiwidmlldy1hdXRob3JpemF0aW9uIiwicXVlcnktY2xpZW50cyIsInF1ZXJ5LXVzZXJzIiwibWFuYWdlLWV2ZW50cyIsIm1hbmFnZS1yZWFsbSIsInZpZXctZXZlbnRzIiwidmlldy11c2VycyIsInZpZXctY2xpZW50cyIsIm1hbmFnZS1hdXRob3JpemF0aW9uIiwibWFuYWdlLWNsaWVudHMiLCJxdWVyeS1ncm91cHMiXX0sImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJtYW5hZ2UtYWNjb3VudC1saW5rcyIsInZpZXctcHJvZmlsZSJdfX0sInNjb3BlIjoiZW1haWwgcHJvZmlsZSIsInNpZCI6IjYzYzNiZjYwLTNmNjMtNDBjNC05NmI1LWNiYTk4ZjZhOGNmNyIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJodHRwczovL2hhc3VyYS5pby9qd3QvY2xhaW1zIjp7IngtaGFzdXJhLXVzZXItaWQiOiJmMWYzMmJkZi05YmQ5LTQ1OTUtYTUzMS1hY2E1Nzc4Y2VhZjAiLCJ4LWhhc3VyYS1kZWZhdWx0LXJvbGUiOiJ1c2VyIiwieC1oYXN1cmEtYWxsb3dlZC1yb2xlcyI6WyJ1c2VyIl0sIngtaGFzdXJhLXVzZXItZ3JvdXBzIjoie30ifSwicHJlZmVycmVkX3VzZXJuYW1lIjoiam9obi5oYW5zYXJpY2tAd2FsbGFyb28uYWkiLCJlbWFpbCI6ImpvaG4uaGFuc2FyaWNrQHdhbGxhcm9vLmFpIn0.EEt9UK1jxvO1DYg_hiy1ne4s9iK8mJtqbVfE7MPQfMRYhzXqDU4gFpP3Nwzlo0iW9fSLDiCxPg303Rz-l4it3oPFu5SaS1S8pQpqvtMAJqy8V_CNPp5H5ggQFYm4Z50aAPdPzOOOkVQOZUhupRsEeUERvK1-eFqtG1bb-IUV6DpQO_XaRVcQbIVubFi48C0_im5Tb3i4WFCNA_1pRrEBKFbZLWgzSCu8fglBQ27mODqfmRQVbTeXLjxsQX5O8meErSfibEGmsJKQytGCJ3NYdnXfal3YhWEqp6A4dG0tkoRW1eD-aKBpsHf9nKKzxcSsjeXDQF6iQAONCGmC40oqHQ'

```python
!curl -X POST {external_url} -H "Content-Type:application/json" -H "Authorization: Bearer {token}" --data @data-25k.json > curl_response.txt
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100 13.0M  100 10.1M  100 2886k  1375k   380k  0:00:07  0:00:07 --:--:-- 3084k

## Undeploy Pipeline

When finished with our tests, we will undeploy the pipeline so we have the Kubernetes resources back for other tasks.  Note that if the deployment variable is unchanged pipeline.deploy() will restart the inference engine in the same configuration as before.

```python
pipeline.undeploy()
```

<table><tr><th>name</th> <td>sdkpipeline</td></tr><tr><th>created</th> <td>2022-12-12 22:53:13.761344+00:00</td></tr><tr><th>last_updated</th> <td>2022-12-14 15:54:43.308811+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>ef60d525-e959-4b4f-acb8-e7f1c9540668, 751d6910-14c2-47d2-bd33-debf39bb475b, 85cb2cc5-39a2-4e68-b5bf-e3cceb270df2, 862b5c66-98a6-4dee-9c92-4c82d7ce49a6, 7a373546-27b8-4541-bc96-33a30e14200c, 929a93db-1478-400a-8e21-0ecfd8090faf, 682dc9af-c3b7-401d-a2bd-d8511dfa3bcc</td></tr><tr><th>steps</th> <td>sdkmodel</td></tr></table>

