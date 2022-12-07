This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/sdk-install-guides/standard-install).

## Installing the Wallaroo SDK

Organizations that develop machine learning models can deploy models to Wallaroo from their local systems to a Wallaroo instance through the [Wallaroo SDK](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/).  The following guide is created to assist users with installing the Wallaroo SDK and making a standard connection to a Wallaroo instance.

These instructions are based on the on the [Connect to Wallaroo](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/#connect-to-wallaroo) guides.

This tutorial provides the following:

* `libraries/wallaroo-0.35.0-py3-none-any.whl`: Wallaroo Python wheel version 0.35.0.  This file is a placeholder until the Wallaroo public library is available.
* `aloha-cnn-lstm.zip`: A pre-trained open source model that uses an [Aloha CNN LSTM model](https://www.researchgate.net/publication/348920204_Using_Auxiliary_Inputs_in_Deep_Learning_Models_for_Detecting_DGA-based_Domain_Names) for classifying Domain names as being either legitimate or being used for nefarious purposes such as malware distribution.
* `data-1.json`, `data-1k.json`, `data-25k.json`: Data files with 1, 1,000, and 25,000 records for testing.

For this example, a virtual python environment will be used.  This will set the necessary libraries and specific Python version required.

## Prerequisites

The following is required for this tutorial:

* A Wallaroo instance version 2022.4 or later.
* Python 3.8 or later installed locally
* [Conda](https://docs.conda.io/en/latest/):  Used for managing python virtual environments.

## General Steps

For our example, we will perform the following:

* Wallaroo SDK Install
  * Set up a Python virtual environment through `conda` with the libraries that enable the virtual environment for use in a Jupyter Hub environment.
  * Install the Wallaroo Python wheel file.
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

1. Download the Wallaroo SDK Python wheel file `wallaroo-0.35.0-py-non-any.whl` to the system where the virtual environment will be established.
1. From a terminal shell, create the Python virtual environment with `conda`.  Replace `wallaroosdk` with the name of the virtual environment as required by your organization.  Note that Python 3.8 is specified as a requirement for Python libraries used with the Wallaroo SDK.

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
    
1. Install the Wallaroo SDK from the uploaded wheel file.  This process may take several minutes while the other required Python libraries are added to the virtual environment.

    ```bash
    pip install --user ./libraries/wallaroo-0.35.0-py3-none-any.whl
    ```

For organizations who will be using the Wallaroo SDK with Jupyter or similar services, the conda virtual environment has been installed, it can either be selected as a new Jupyter Notebook kernel, or the Notebook's kernel can be set to an existing Jupyter notebook.

![](/images/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-install-guides/wallaroo-select-kernel.png)

To use a new Notebook:

1. From the main menu, select **File->New-Notebook**.
1. From the Kernel selection dropbox, select the new virtual environment - in this case, **wallaroosdk**.

To update an existing Notebook to use the new virtual environment as a kernel:

1. From the main menu, select **Kernel->Change Kernel**.

### 

## Sample Wallaroo Connection

With the Wallaroo Python SDK installed, remote commands and inferences can be performed through the following steps.

### Open a Connection to Wallaroo

The first step is to connect to Wallaroo through the Wallaroo client.

This is accomplished using the `wallaroo.Client(api_endpoint, auth_endpoint, auth_type command)` command that connects to the Wallaroo instance services.  For more information on the DNS names of Wallaroo services, see the [DNS Integration Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-dns-guide/).

The `Client` method takes the following parameters:

* **api_endpoint** (*String*): The URL to the Wallaroo instance API service.
* **auth_endpoint** (*String*): The URL to the Wallaroo instance Keycloak service.
* **auth_type command** (*String*): The authorization type.  In this case, `SSO`.

Once run, the `wallaroo.Client` command provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

```python
import wallaroo
from wallaroo.object import EntityNotFoundError
```

```python
# SSO login through keycloak

wl = wallaroo.Client(api_endpoint="https://magical-bear-3782.api.wallaroo.community", 
                    auth_endpoint="https://magical-bear-3782.keycloak.wallaroo.community", 
                    auth_type="sso")
```

    Please log into the following URL in a web browser:
    
    	https://magical-bear-3782.keycloak.wallaroo.community/auth/realms/master/device?user_code=GGRF-SRFX
    
    Login successful!

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

<table><tr><th>name</th> <td>sdkpipeline</td></tr><tr><th>created</th> <td>2022-12-07 15:18:15.704015+00:00</td></tr><tr><th>last_updated</th> <td>2022-12-07 15:18:15.704015+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>6736057c-e9e8-48d4-a657-158d10af1273</td></tr><tr><th>steps</th> <td></td></tr></table>

We can verify the workspace is created the current default workspace with the `get_current_workspace()` command.

```python
wl.get_current_workspace()
```

    {'name': 'sdkworkspace', 'id': 12, 'archived': False, 'created_by': '0bbf2f62-a4f1-4fe5-aad8-ec1cb7485939', 'created_at': '2022-12-07T15:18:14.569577+00:00', 'models': [], 'pipelines': [{'name': 'sdkpipeline', 'create_time': datetime.datetime(2022, 12, 7, 15, 18, 15, 704015, tzinfo=tzutc()), 'definition': '[]'}]}

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

<table><tr><th>name</th> <td>sdkpipeline</td></tr><tr><th>created</th> <td>2022-12-07 15:18:15.704015+00:00</td></tr><tr><th>last_updated</th> <td>2022-12-07 15:18:15.704015+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>6736057c-e9e8-48d4-a657-158d10af1273</td></tr><tr><th>steps</th> <td></td></tr></table>

```python
pipeline.deploy()
```

<table><tr><th>name</th> <td>sdkpipeline</td></tr><tr><th>created</th> <td>2022-12-07 15:18:15.704015+00:00</td></tr><tr><th>last_updated</th> <td>2022-12-07 15:18:20.882795+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>af2526e0-1410-47bf-8475-9a2fa157212d, 6736057c-e9e8-48d4-a657-158d10af1273</td></tr><tr><th>steps</th> <td>sdkmodel</td></tr></table>

We can verify that the pipeline is running and list what models are associated with it.

```python
pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.3.50',
       'name': 'engine-7766467cdb-jj448',
       'status': 'Running',
       'reason': None,
       'details': ['containers with unready status: [engine]',
        'containers with unready status: [engine]'],
       'pipeline_statuses': {'pipelines': [{'id': 'sdkpipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'sdkmodel',
          'version': 'fd1832a7-723d-4318-a0d7-c677b1956206',
          'sha': '7c89707252ce389980d5348c37885d6d72af4c20cd303422e2de7e66dd7ff184',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.2.6',
       'name': 'engine-lb-c6485cfd5-h75v5',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}

## Interferences

### Infer 1 row

Now that the pipeline is deployed and our Aloha model is in place, we'll perform a smoke test to verify the pipeline is up and running properly.  We'll use the `infer_from_file` command to load a single encoded URL into the inference engine and print the results back out.

The result should tell us that the tokenized URL is legitimate (0) or fraud (1).  This sample data should return close to 0.

```python
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

When retrieving the pipeline inference URL through an external SDK connection, the External Inference URL will be returned.  This URL will function provided that the **Enable external URL inference endpoints** is enabled.  For more information, see the [Wallaroo Documentation Site](https://docs.wallaroo.ai).

```python
external_url = pipeline._deployment._url()
external_url
```

    'https://magical-bear-3782.api.wallaroo.community/v1/api/pipelines/infer/sdkpipeline-14'

The API connection details can be retrieved through the Wallaroo client `mlops()` command.  This will display the connection URL, bearer token, and other information.  The bearer token is available for one hour before it expires.

For this example, the API connection details will be retrieved, then used to submit an inference request through the external inference URL retrieved earlier.

```python
connection =wl.mlops().__dict__
token = connection['token']
token
```

    'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJyb3dHSmdNdnlCODFyRFBzQURxc3RIM0hIbFdZdmdhMnluUmtGXzllSWhjIn0.eyJleHAiOjE2NzA0MjY0MDgsImlhdCI6MTY3MDQyNjM0OCwiYXV0aF90aW1lIjoxNjcwNDI2MjkyLCJqdGkiOiJiZDVkYTU0NS03MDMwLTRiZTktODRhOS00MDgyN2JjNjJhNTMiLCJpc3MiOiJodHRwczovL21hZ2ljYWwtYmVhci0zNzgyLmtleWNsb2FrLndhbGxhcm9vLmNvbW11bml0eS9hdXRoL3JlYWxtcy9tYXN0ZXIiLCJhdWQiOlsibWFzdGVyLXJlYWxtIiwiYWNjb3VudCJdLCJzdWIiOiIwYmJmMmY2Mi1hNGYxLTRmZTUtYWFkOC1lYzFjYjc0ODU5MzkiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJzZGstY2xpZW50Iiwic2Vzc2lvbl9zdGF0ZSI6ImU0MTJkMDA0LTYyODMtNDZhNC05NDExLTYzMTFkOGI0ZDczZCIsImFjciI6IjEiLCJyZWFsbV9hY2Nlc3MiOnsicm9sZXMiOlsiY3JlYXRlLXJlYWxtIiwiZGVmYXVsdC1yb2xlcy1tYXN0ZXIiLCJvZmZsaW5lX2FjY2VzcyIsImFkbWluIiwidW1hX2F1dGhvcml6YXRpb24iXX0sInJlc291cmNlX2FjY2VzcyI6eyJtYXN0ZXItcmVhbG0iOnsicm9sZXMiOlsidmlldy1yZWFsbSIsInZpZXctaWRlbnRpdHktcHJvdmlkZXJzIiwibWFuYWdlLWlkZW50aXR5LXByb3ZpZGVycyIsImltcGVyc29uYXRpb24iLCJjcmVhdGUtY2xpZW50IiwibWFuYWdlLXVzZXJzIiwicXVlcnktcmVhbG1zIiwidmlldy1hdXRob3JpemF0aW9uIiwicXVlcnktY2xpZW50cyIsInF1ZXJ5LXVzZXJzIiwibWFuYWdlLWV2ZW50cyIsIm1hbmFnZS1yZWFsbSIsInZpZXctZXZlbnRzIiwidmlldy11c2VycyIsInZpZXctY2xpZW50cyIsIm1hbmFnZS1hdXRob3JpemF0aW9uIiwibWFuYWdlLWNsaWVudHMiLCJxdWVyeS1ncm91cHMiXX0sImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJtYW5hZ2UtYWNjb3VudC1saW5rcyIsInZpZXctcHJvZmlsZSJdfX0sInNjb3BlIjoicHJvZmlsZSBlbWFpbCIsInNpZCI6ImU0MTJkMDA0LTYyODMtNDZhNC05NDExLTYzMTFkOGI0ZDczZCIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJodHRwczovL2hhc3VyYS5pby9qd3QvY2xhaW1zIjp7IngtaGFzdXJhLXVzZXItaWQiOiIwYmJmMmY2Mi1hNGYxLTRmZTUtYWFkOC1lYzFjYjc0ODU5MzkiLCJ4LWhhc3VyYS1kZWZhdWx0LXJvbGUiOiJ1c2VyIiwieC1oYXN1cmEtYWxsb3dlZC1yb2xlcyI6WyJ1c2VyIl0sIngtaGFzdXJhLXVzZXItZ3JvdXBzIjoie30ifSwicHJlZmVycmVkX3VzZXJuYW1lIjoiam9obi5oYW5zYXJpY2tAd2FsbGFyb28uYWkiLCJlbWFpbCI6ImpvaG4uaGFuc2FyaWNrQHdhbGxhcm9vLmFpIn0.gC4yOFYeUURmfFCTh88-TLDlm_ns38__lJUH7jP9SiZJafLAq6p2R5Fj-j-G62D9Ne2qiwwQoniOdLWrc9wkkA8JKMq890p-0Vq89VXsXQd-Ut9QdL7wIXNKMJjDshWNNJpcMRPLbXTnQytVXlNYLTSQPLWnqMtfQYYtq862FNVU7MfazApZGFsP8Z0LlHWyPpepR-q2o-_sBQd2qj35O7swT4lyvTTwVNqDgp8eJL2ZXqWk5vH4hXQjRinctiGj4W6WgS5-O78Gs-VtFVqYwMhY9GfjU9HN8R_IwIRlx3rc3hd9YEjARZzhwdlc-h9DCH3HKSaZQg28Us6-kFHbsQ'

```python
!curl -X POST {external_url} -H "Content-Type:application/json" -H "Authorization: Bearer {token}" --data @data-25k.json > curl_response.txt
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100 13.0M  100 10.1M  100 2886k  1409k   389k  0:00:07  0:00:07 --:--:-- 2478k-  0:00:07  365k

## Undeploy Pipeline

When finished with our tests, we will undeploy the pipeline so we have the Kubernetes resources back for other tasks.  Note that if the deployment variable is unchanged pipeline.deploy() will restart the inference engine in the same configuration as before.

```python
pipeline.undeploy()
```

<table><tr><th>name</th> <td>sdkpipeline</td></tr><tr><th>created</th> <td>2022-12-07 15:18:15.704015+00:00</td></tr><tr><th>last_updated</th> <td>2022-12-07 15:18:20.882795+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>af2526e0-1410-47bf-8475-9a2fa157212d, 6736057c-e9e8-48d4-a657-158d10af1273</td></tr><tr><th>steps</th> <td>sdkmodel</td></tr></table>

```python

```
