This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/development/sdk-install-guides/azure-ml-sdk-install).

## Installing the Wallaroo SDK into Azure ML Workspace

Organizations that use Azure ML for model training and development can deploy models to Wallaroo through the [Wallaroo SDK](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/).  The following guide is created to assist users with installing the Wallaroo SDK, setting up authentication through Azure ML, and making a standard connection to a Wallaroo instance through Azure ML Workspace.

These instructions are based on the on the [Wallaroo SSO for Microsoft Azure](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-sso-authentication/wallaroo-sso-azure/) and the [Connect to Wallaroo](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/) guides.

This tutorial provides the following:

* `aloha-cnn-lstm.zip`: A pre-trained open source model that uses an [Aloha CNN LSTM model](https://www.researchgate.net/publication/348920204_Using_Auxiliary_Inputs_in_Deep_Learning_Models_for_Detecting_DGA-based_Domain_Names) for classifying Domain names as being either legitimate or being used for nefarious purposes such as malware distribution.
* Test Data Files:
  * `data-1.json`: 1 record
  * `data-1k.json`: 1,000 records
  * `data-25k.json`: 25,000 records

To use the Wallaroo SDK within Azure ML Workspace, a virtual environment will be used.  This will set the necessary libraries and specific Python version required.

## Prerequisites

The following is required for this tutorial:

* A Wallaroo instance version 2022.4 or later.
* Python 3.8.6 or later installed locally
* [Conda](https://docs.conda.io/en/latest/):  Used for managing python virtual environments.  This is automatically included in Azure ML Workspace.
* An Azure ML workspace is created with a compute configured.

## General Steps

For our example, we will perform the following:

* Wallaroo SDK Install
  * Set up a Python virtual environment through `conda` with the libraries that enable the virtual environment for use in a Jupyter Hub environment.
  * Install the Wallaroo SDK.
* Wallaroo SDK from remote JupyterHub Demonstration (Optional):  The following steps are an optional exercise to demonstrate using the Wallaroo SDK from a remote connection.  The entire tutorial can be found on the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/sdk-install-guides/azure-ml-sdk-install)).
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

To set up the virtual environment in Azure ML for using the Wallaroo SDK with Azure ML Workspace:

1. Select **Notebooks**.
1. Create a new folder where the Jupyter Notebooks for Wallaroo will be installed.
1. From this repository, upload `sdk-install-guides/azure-ml-sdk-install.zip`, or upload the entire folder `sdk-install-guides/azure-ml-sdk-install`.  This tutorial will assume the .zip file was uploaded.
1. Select **Open Terminal**.  Navigate to the target directory.
1. Run `unzip azure-ml-sdk-install.zip` to unzip the directory, then cd into it with `cd azure-ml-sdk-install`.
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
    pip install wallaroo
    ```


Once the conda virtual environment has been installed, it can either be selected as a new Jupyter Notebook kernel, or the Notebook's kernel can be set to an existing Jupyter notebook.  If a notebook is existing, close it then reopen to select the new Wallaroo SDK environment.

![](./images/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-install-guides/azureml-sdk-guide/azureml-select-kernel.png)

To use a new Notebook:

1. From the left navigation panel, select **+->Notebook**.
1. From the Kernel selection dropbox on the upper right side, select the new virtual environment - in this case, **wallaroosdk**.

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

![Wallaroo Login](./images/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-install-guides/azureml-sdk-guide/azure-initial-login.png)

Once authenticated, the user will verify adding the device the user is establishing the connection from.  Once both steps are complete, then the connection is granted.

![Device Registration](./images//wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-install-guides/wallaroo-device-access.png)

The connection is stored in the variable `wl` for use in all other Wallaroo calls.


```python
import wallaroo
from wallaroo.object import EntityNotFoundError
```


```python
# SSO login through keycloak

wallarooPrefix = "YOURPREFIX"
wallarooSuffix = "YOURSUFFIX"

wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}.api.{wallarooSuffix}", 
                    auth_endpoint=f"https://{wallarooPrefix}.keycloak.{wallarooSuffix}", 
                    auth_type="sso")
```

### Create the Workspace

We will create a workspace to work in and call it the `azuremlsdkworkspace`, then set it as current workspace environment.  We'll also create our pipeline in advance as `azuremlsdkpipeline`.

* **IMPORTANT NOTE**:  For this example, the Aloha model is stored in the file `alohacnnlstm.zip`.  When using tensor based models, the zip file **must** match the name of the tensor directory.  For example, if the tensor directory is `alohacnnlstm`, then the .zip file must be named `alohacnnlstm.zip`.


```python
workspace_name = 'azuremlsdkworkspace'
pipeline_name = 'azuremlsdkpipeline'
model_name = 'azuremlsdkmodel'
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




<table><tr><th>name</th> <td>gcpsdkpipeline</td></tr><tr><th>created</th> <td>2022-12-06 21:35:51.201925+00:00</td></tr><tr><th>last_updated</th> <td>2022-12-06 21:35:51.201925+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>90045b0b-1978-48bb-9f37-05c0c5d8bf22</td></tr><tr><th>steps</th> <td></td></tr></table>



We can verify the workspace is created the current default workspace with the `get_current_workspace()` command.


```python
wl.get_current_workspace()
```




    {'name': 'gcpsdkworkspace', 'id': 10, 'archived': False, 'created_by': '0bbf2f62-a4f1-4fe5-aad8-ec1cb7485939', 'created_at': '2022-12-06T21:35:50.34358+00:00', 'models': [], 'pipelines': [{'name': 'gcpsdkpipeline', 'create_time': datetime.datetime(2022, 12, 6, 21, 35, 51, 201925, tzinfo=tzutc()), 'definition': '[]'}]}



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




<table><tr><th>name</th> <td>gcpsdkpipeline</td></tr><tr><th>created</th> <td>2022-12-06 21:35:51.201925+00:00</td></tr><tr><th>last_updated</th> <td>2022-12-06 21:35:51.201925+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>90045b0b-1978-48bb-9f37-05c0c5d8bf22</td></tr><tr><th>steps</th> <td></td></tr></table>




```python
pipeline.deploy()
```




<table><tr><th>name</th> <td>gcpsdkpipeline</td></tr><tr><th>created</th> <td>2022-12-06 21:35:51.201925+00:00</td></tr><tr><th>last_updated</th> <td>2022-12-06 21:35:55.428652+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>269179a8-79e4-4c58-b9c3-d05436ad7be3, 90045b0b-1978-48bb-9f37-05c0c5d8bf22</td></tr><tr><th>steps</th> <td>gcpsdkmodel</td></tr></table>



We can verify that the pipeline is running and list what models are associated with it.


```python
pipeline.status()
```




    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.1.174',
       'name': 'engine-7888f44c8b-r2gpr',
       'status': 'Running',
       'reason': None,
       'details': ['containers with unready status: [engine]',
        'containers with unready status: [engine]'],
       'pipeline_statuses': {'pipelines': [{'id': 'gcpsdkpipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'gcpsdkmodel',
          'version': 'c468d323-257b-4717-bbd8-8539a8746496',
          'sha': '7c89707252ce389980d5348c37885d6d72af4c20cd303422e2de7e66dd7ff184',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.1.173',
       'name': 'engine-lb-c6485cfd5-kqsn6',
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

When retrieving the pipeline inference URL through an external SDK connection, the External Inference URL will be returned.  This URL will function provided that the **Enable external URL inference endpoints** is enabled.  For more information, see the [Wallaroo Model Endpoints Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-model-endpoints-guide/).


```python
external_url = pipeline._deployment._url()
external_url
```




    'https://YOUR PREFIX.api.example.wallaroo.ai/v1/api/pipelines/infer/gcpsdkpipeline-13'



The API connection details can be retrieved through the Wallaroo client `mlops()` command.  This will display the connection URL, bearer token, and other information.  The bearer token is available for one hour before it expires.

For this example, the API connection details will be retrieved, then used to submit an inference request through the external inference URL retrieved earlier.


```python
connection =wl.mlops().__dict__
token = connection['token']
token
```




    'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJyb3dHSmdNdnlCODFyRFBzQURxc3RIM0hIbFdZdmdhMnluUmtGXzllSWhjIn0.eyJleHAiOjE2NzAzNjI2NjMsImlhdCI6MTY3MDM2MjYwMywiYXV0aF90aW1lIjoxNjcwMzYyNTQ1LCJqdGkiOiI5NDk5M2Y2Ni0yMjk2LTRiMTItOTYwMi1iOWEyM2UxY2RhZGIiLCJpc3MiOiJodHRwczovL21hZ2ljYWwtYmVhci0zNzgyLmtleWNsb2FrLndhbGxhcm9vLmNvbW11bml0eS9hdXRoL3JlYWxtcy9tYXN0ZXIiLCJhdWQiOlsibWFzdGVyLXJlYWxtIiwiYWNjb3VudCJdLCJzdWIiOiIwYmJmMmY2Mi1hNGYxLTRmZTUtYWFkOC1lYzFjYjc0ODU5MzkiLCJ0eXAiOiJCZWFyZXIiLCJhenAiOiJzZGstY2xpZW50Iiwic2Vzc2lvbl9zdGF0ZSI6ImQyYjlkMzFjLWU3ZmMtNDI4OS1hOThjLTI2ZTMwMDBiMzVkMiIsImFjciI6IjEiLCJyZWFsbV9hY2Nlc3MiOnsicm9sZXMiOlsiY3JlYXRlLXJlYWxtIiwiZGVmYXVsdC1yb2xlcy1tYXN0ZXIiLCJvZmZsaW5lX2FjY2VzcyIsImFkbWluIiwidW1hX2F1dGhvcml6YXRpb24iXX0sInJlc291cmNlX2FjY2VzcyI6eyJtYXN0ZXItcmVhbG0iOnsicm9sZXMiOlsidmlldy1yZWFsbSIsInZpZXctaWRlbnRpdHktcHJvdmlkZXJzIiwibWFuYWdlLWlkZW50aXR5LXByb3ZpZGVycyIsImltcGVyc29uYXRpb24iLCJjcmVhdGUtY2xpZW50IiwibWFuYWdlLXVzZXJzIiwicXVlcnktcmVhbG1zIiwidmlldy1hdXRob3JpemF0aW9uIiwicXVlcnktY2xpZW50cyIsInF1ZXJ5LXVzZXJzIiwibWFuYWdlLWV2ZW50cyIsIm1hbmFnZS1yZWFsbSIsInZpZXctZXZlbnRzIiwidmlldy11c2VycyIsInZpZXctY2xpZW50cyIsIm1hbmFnZS1hdXRob3JpemF0aW9uIiwibWFuYWdlLWNsaWVudHMiLCJxdWVyeS1ncm91cHMiXX0sImFjY291bnQiOnsicm9sZXMiOlsibWFuYWdlLWFjY291bnQiLCJtYW5hZ2UtYWNjb3VudC1saW5rcyIsInZpZXctcHJvZmlsZSJdfX0sInNjb3BlIjoicHJvZmlsZSBlbWFpbCIsInNpZCI6ImQyYjlkMzFjLWU3ZmMtNDI4OS1hOThjLTI2ZTMwMDBiMzVkMiIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJodHRwczovL2hhc3VyYS5pby9qd3QvY2xhaW1zIjp7IngtaGFzdXJhLXVzZXItaWQiOiIwYmJmMmY2Mi1hNGYxLTRmZTUtYWFkOC1lYzFjYjc0ODU5MzkiLCJ4LWhhc3VyYS1kZWZhdWx0LXJvbGUiOiJ1c2VyIiwieC1oYXN1cmEtYWxsb3dlZC1yb2xlcyI6WyJ1c2VyIl0sIngtaGFzdXJhLXVzZXItZ3JvdXBzIjoie30ifSwicHJlZmVycmVkX3VzZXJuYW1lIjoiam9obi5oYW5zYXJpY2tAd2FsbGFyb28uYWkiLCJlbWFpbCI6ImpvaG4uaGFuc2FyaWNrQHdhbGxhcm9vLmFpIn0.Gnig3PdpMFGSrQ2J4Tj3Nqbk2UOfBCH4MEw2i6p5pLkQ51F8FM7Dq-VOGoNYAXZn2OXw_bKh0Ae60IqglB0PSFTlksVzb1uSGKOPgcZNkI0fTMK99YW71UctMDk9MYrN09bT2GhGQ7FV-tJNqemYSXB3eMIaTkah6AMUfJIYYvf6J2OqXyNJqc6Hwf0-44FGso_N0WXF6GM-ww72ampVjc10Mad30kYzQX508U9RuZXd3uvOrRQHreOcPPmjso1yDbUx8gqLeov_uq3dg5hUY55v2oVBdtXT60-ZBIQP8uETNetv6529Nm52uwKNT7DdjXk85kbJBK8oV6etyfKRDw'




```python
!curl -X POST {external_url} -H "Content-Type:application/json" -H "Authorization: Bearer {token}" --data @data-25k.json > curl_response.txt
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100 13.0M  100 10.1M  100 2886k  2322k   642k  0:00:04  0:00:04 --:--:-- 2965k


## Undeploy Pipeline

When finished with our tests, we will undeploy the pipeline so we have the Kubernetes resources back for other tasks.  Note that if the deployment variable is unchanged pipeline.deploy() will restart the inference engine in the same configuration as before.


```python
pipeline.undeploy()
```


```python

```
