This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/development/sdk-install-guides/databricks-azure-sdk-install).

## Installing the Wallaroo SDK into  Workspace

Organizations that use Azure Databricks for model training and development can deploy models to Wallaroo through the [Wallaroo SDK](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/).  The following guide is created to assist users with installing the Wallaroo SDK, setting up authentication through Azure Databricks, and making a standard connection to a Wallaroo instance through Azure Databricks Workspace.

These instructions are based on the on the [Wallaroo SSO for Microsoft Azure](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-sso-authentication/wallaroo-sso-azure/) and the [Connect to Wallaroo](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/) guides.

This tutorial provides the following:

* `aloha-cnn-lstm.zip`: A pre-trained open source model that uses an [Aloha CNN LSTM model](https://www.researchgate.net/publication/348920204_Using_Auxiliary_Inputs_in_Deep_Learning_Models_for_Detecting_DGA-based_Domain_Names) for classifying Domain names as being either legitimate or being used for nefarious purposes such as malware distribution.
* `data-1.json`, `data-1k.json`, `data-25k.json`: Data files with 1, 1,000, and 25,000 records for testing.

To use the Wallaroo SDK within Azure Databricks Workspace, a virtual environment will be used.  This will set the necessary libraries and specific Python version required.

## Prerequisites

The following is required for this tutorial:

* A Wallaroo instance version 2023.1 or later.
* An Azure Databricks workspace with a cluster

## General Steps

For our example, we will perform the following:

* Wallaroo SDK Install
  * Install the Wallaroo SDK into the Azure Databricks cluster.
  * Install the Wallaroo Python SDK.
  * Connect to a remote Wallaroo instance.  This instance is configured to use the standard Keycloak service.
* Wallaroo SDK from Azure Databricks Workspace (Optional)
  * The following steps are used to demonstrate using the Wallaroo SDK in an Azure Databricks Workspace environment.  The entire tutorial can be found on the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/sdk-install-guides/azure-ml-sdk-install).
    * Create a workspace for our work.
    * Upload the CCFraud model.
    * Create a pipeline that can ingest our submitted data, submit it to the model, and export the results
    * Run a sample inference through our pipeline by loading a file
    * Undeploy the pipeline and return resources back to the Wallaroo instance's Kubernetes environment.

## Install Wallaroo SDK

### Add Wallaroo SDK to Cluster

To install the Wallaroo SDK in a Azure Databricks environment:

1. From the Azure Databricks dashboard, select **Computer**, then the cluster to use.
1. Select **Libraries**.
1. Select **Install new**.
1. Select **PyPI**.  In the **Package** field, enter the current version of the [Wallaroo SDK](https://pypi.org/project/wallaroo/).  It is recommended to specify the version, which as of this writing is `wallaroo==2022.4.0`.
1. Select **Install**.

Once the **Status** shows **Installed**, it will be available in Azure Databricks notebooks and other tools that use the cluster.

Once the Wallaroo SDK is installed, it can be used in a new or imported workbook.

To use a new Notebook:

1. From the left navigation panel, select **+ New**.
1. Select **Notebook**.

To upload an existing notebook:

1. From the left navigation panel, select **Workspace**, the workspace to use, then the dropdown icon and select **Import**.
1. Select the Jupyter Notebook to import.

## Sample Wallaroo Connection

With the Wallaroo Python SDK installed, remote commands and inferences can be performed through the following steps.

### Open a Connection to Wallaroo

The first step is to connect to Wallaroo through the Wallaroo client.

This is accomplished using the `wallaroo.Client(api_endpoint, auth_endpoint, auth_type command)` command that connects to the Wallaroo instance services.

The `Client` method takes the following parameters:

* **api_endpoint** (*String*): The URL to the Wallaroo instance API service.
* **auth_endpoint** (*String*): The URL to the Wallaroo instance Keycloak service.
* **auth_type command** (*String*): The authorization type.  In this case, `SSO`.

The URLs are based on the Wallaroo Prefix and Wallaroo Suffix for the Wallaroo instance.  For more information, see the [DNS Integration Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-dns-guide/).  In the example below, replace "YOUR PREFIX" and "YOUR SUFFIX" with the Wallaroo Prefix and Suffix, respectively.  In the example below, replace "YOUR PREFIX" and "YOUR SUFFIX" with the Wallaroo Prefix and Suffix, respectively.

Once run, the `wallaroo.Client` command provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.

Depending on the configuration of the Wallaroo instance, the user will either be presented with a login request to the Wallaroo instance or be authenticated through a broker such as Google, Github, etc.  To use the broker, select it from the list under the username/password login forms.  For more information on Wallaroo authentication configurations, see the [Wallaroo Authentication Configuration Guides](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-sso-authentication/).

![Wallaroo Login](./images/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-install-guides/databricks-azure-sdk-guide/azure-initial-login.png)

Once authenticated, the user will verify adding the device the user is establishing the connection from.  Once both steps are complete, then the connection is granted.

![Device Registration](./images/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-install-guides/wallaroo-device-access.png)

The connection is stored in the variable `wl` for use in all other Wallaroo calls.


```python
import wallaroo
from wallaroo.object import EntityNotFoundError
```


```python
# SSO login through keycloak

wallarooPrefix = "YOUR PREFIX"
wallarooSuffix = "YOUR SUFFIX"

wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}.api.{wallarooSuffix}", 
                    auth_endpoint=f"https://{wallarooPrefix}.keycloak.{wallarooSuffix}", 
                    auth_type="sso")
```

### Create the Workspace

We will create a workspace to work in and call it the `databricksazuresdkworkspace`, then set it as current workspace environment.  We'll also create our pipeline in advance as `azuremlsdkpipeline`.

* **IMPORTANT NOTE**:  For this example, the Aloha model is stored in the file `alohacnnlstm.zip`.  When using tensor based models, the zip file **must** match the name of the tensor directory.  For example, if the tensor directory is `alohacnnlstm`, then the .zip file must be named `alohacnnlstm.zip`.

In the example below, replace `/dbfs/FileStore/YOUR PATH/alohacnnlstm.zip` with the full `dbfs` path to the uploaded `alohacnnlstm.zip` file.


```python
workspace_name = 'databricksazuresdkworkspace'
pipeline_name = 'databricksazuresdkpipeline'
model_name = 'alohamodel'
model_file_name = '/dbfs/FileStore/YOUR PATH/alohacnnlstm.zip'
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


<table><tr><th>name</th> <td>databricksazuresdkpipeline</td></tr><tr><th>created</th> <td>2023-01-13 17:28:30.723301+00:00</td></tr><tr><th>last_updated</th> <td>2023-01-13 17:28:30.723301+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>6042c356-5b6c-42ea-8024-b4bc313b101d</td></tr><tr><th>steps</th> <td></td></tr></table>


We can verify the workspace is created the current default workspace with the `get_current_workspace()` command.


```python
wl.get_current_workspace()
```


    Out[6]: {'name': 'databricksazuresdkworkspace', 'id': 20, 'archived': False, 'created_by': '59e7c5e0-64da-424f-8b5e-de53348f3347', 'created_at': '2023-01-13T17:28:30.374032+00:00', 'models': [], 'pipelines': [{'name': 'databricksazuresdkpipeline', 'create_time': datetime.datetime(2023, 1, 13, 17, 28, 30, 723301, tzinfo=tzutc()), 'definition': '[]'}]}


### Upload the Models

Now we will upload our model.

**IMPORTANT NOTE**:  Use the local file path format such as `/dbfs/FileStore/shared_uploads/YOURWORKSPACE/file` format rather than the `dbfs:` format.


```python
model = wl.upload_model(model_name, model_file_name).configure("tensorflow")
```

### Deploy a Model

Now that we have a model that we want to use we will create a deployment for it. 

To do this, we'll create our pipeline that can ingest the data, pass the data to our CCFraud model, and give us a final output.  We'll call our pipeline `databricksazuresdkpipeline`, then deploy it so it's ready to receive data.  The deployment process usually takes about 45 seconds.


```python
pipeline.add_model_step(model)
```


<table><tr><th>name</th> <td>databricksazuresdkpipeline</td></tr><tr><th>created</th> <td>2023-01-13 17:28:30.723301+00:00</td></tr><tr><th>last_updated</th> <td>2023-01-13 17:28:30.723301+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>6042c356-5b6c-42ea-8024-b4bc313b101d</td></tr><tr><th>steps</th> <td></td></tr></table>



```python
pipeline.deploy()
```


<table><tr><th>name</th> <td>databricksazuresdkpipeline</td></tr><tr><th>created</th> <td>2023-01-13 17:28:30.723301+00:00</td></tr><tr><th>last_updated</th> <td>2023-01-13 17:35:01.706224+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>dd9801b9-aeb1-4801-beb8-f0b993e30b81, 6042c356-5b6c-42ea-8024-b4bc313b101d</td></tr><tr><th>steps</th> <td>alohamodel</td></tr></table>


We can verify that the pipeline is running and list what models are associated with it.


```python
pipeline.status()
```


    Out[16]: {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.2.58',
       'name': 'engine-76ff897498-87pnm',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'databricksazuresdkpipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'alohamodel',
          'version': '025a778b-7718-4c1f-9ec6-d51135ce8896',
          'sha': 'd71d9ffc61aaac58c2b1ed70a2db13d1416fb9d3f5b891e5e4e2e97180fe22f8',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.3.11',
       'name': 'engine-lb-55dcdff64c-mf6c4',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}


## Interferences

### Infer 1 row

Now that the pipeline is deployed and our CCfraud model is in place, we'll perform a smoke test to verify the pipeline is up and running properly.  We'll use the `infer_from_file` command to load a single transaction and determine if it is flagged for fraud.  If it returns correctly, a small valud should be returned indicating a low likelihood that the transaction was fraudulent.

In the example below, replace `/dbfs/FileStore/YOUR PATH/data_1.json` with the full DBFS path to the uploaded sample data file.


```python
result = pipeline.infer_from_file("/dbfs/FileStore/YOUR PATH/data_1.json")

```


```python
result[0].data()
```


    Out[18]: [array([[0.00151959]]),
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

Now that our smoke test is successful, let's really give it some data.  We'll use the `cc_data_1k.json` file that contains 1,000 inferences to be performed.

In the example below, replace `/dbfs/FileStore/YOUR PATH/data_1k.json` with the full DBFS path to the uploaded sample data file.


```python
result = pipeline.infer_from_file("/dbfs/FileStore/YOUR PATH/data_1k.json")
result
```

## Undeploy Pipeline

When finished with our tests, we will undeploy the pipeline so we have the Kubernetes resources back for other tasks.  Note that if the deployment variable is unchanged pipeline.deploy() will restart the inference engine in the same configuration as before.


```python
pipeline.undeploy()
```


<table><tr><th>name</th> <td>databricksazuresdkpipeline</td></tr><tr><th>created</th> <td>2023-01-13 17:28:30.723301+00:00</td></tr><tr><th>last_updated</th> <td>2023-01-13 17:35:01.706224+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>dd9801b9-aeb1-4801-beb8-f0b993e30b81, 6042c356-5b6c-42ea-8024-b4bc313b101d</td></tr><tr><th>steps</th> <td>alohamodel</td></tr></table>



```python

```
