This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/development/sdk-install-guides/databricks-azure-sdk-install).

## Installing the Wallaroo SDK into Databricks Azure Workspace

Organizations that use Databricks Azure for model training and development can deploy models to Wallaroo through the [Wallaroo SDK](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/).  The following guide is created to assist users with installing the Wallaroo SDK, setting up authentication through Databricks Azure, and making a standard connection to a Wallaroo instance through Databricks Azure Workspace.

These instructions are based on the on the [Wallaroo SSO for Microsoft Azure](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-sso-authentication/wallaroo-sso-azure/) and the [Connect to Wallaroo](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/) guides.

This tutorial provides the following:

* `ccfraud.onnx`: A pre-trained CCFraud model that detects if a credit card transaction is likely to be fraudulent of not.
* `high_fraud.json`: A test file that results in the CCFraud indicating a high likelihood that the credit card transaction was fraudulent.
* `smoke_test.json`: A test file that results in the CCFraud indicating a high likelihood that the credit card transaction was **not** fraudulent.
* `cc_data_1k.json` and `cc_data_10k.json`: Sample input files with 1,000 and 10,000 inputs for inferences.

## Prerequisites

The following is required for this tutorial:

* A Wallaroo instance version 2022.4 or later.
* An Azure Databricks workspace with a cluster

## General Steps

For our example, we will perform the following:

* Wallaroo SDK Install
  * Install the Wallaroo SDK into the Databricks Azure cluster.
  * Connect to a remote Wallaroo instance.  This instance is configured to use the standard Keycloak service.
* Wallaroo SDK Demonstration from Databricks Azure Workspace (Optional)
  * The following steps are used to demonstrate using the Wallaroo SDK in an Databricks Azure Workspace environment.  The entire tutorial can be found on the [Wallaroo Tutorials repository]([Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/sdk-install-guides/databricks-azure-sdk-install).
    * Create a workspace for our work.
    * Upload the CCFraud model.
    * Create a pipeline that can ingest our submitted data, submit it to the model, and export the results
    * Run a sample inference through our pipeline by loading a file
    * Undeploy the pipeline and return resources back to the Wallaroo instance's Kubernetes environment.

## Install Wallaroo SDK

### Add Wallaroo SDK to Cluster

To install the Wallaroo SDK in a Databricks Azure environment:

1. From the Databricks Azure dashboard, select **Computer**, then the cluster to use.
1. Select **Libraries**.
1. Select **Install new**.
1. Select **PyPI**.  In the **Package** field, enter the current version of the [Wallaroo SDK](https://pypi.org/project/wallaroo/).  It is recommended to specify the version, which as of this writing is `wallaroo==2022.4.0`.
1. Select **Install**.

![](./images/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-install-guides/azureml-sdk-guide/wallaroo-sdk-databricks-azure-install-wallaroo-sdk.png)

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

This is accomplished using the `wallaroo.Client(api_endpoint, auth_endpoint, auth_type command)` command that connects to the Wallaroo instance services.  For more information on the DNS names of Wallaroo services, see the [DNS Integration Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-dns-guide/).

The `Client` method takes the following parameters:

* **api_endpoint** (*String*): The URL to the Wallaroo instance API service.
* **auth_endpoint** (*String*): The URL to the Wallaroo instance Keycloak service.
* **auth_type command** (*String*): The authorization type.  In this case, `SSO`.

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

wallarooPrefix = "YOUR PREFIX"
wallarooSuffix = "YOUR SUFFIX"

wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}.api.{wallarooSuffix}", 
                    auth_endpoint=f"https://{wallarooPrefix}.keycloak.{wallarooSuffix}", 
                    auth_type="sso")
```

### Create the Workspace

We will create a workspace to work in and call it the `databricksazuresdkworkspace`, then set it as current workspace environment.  We'll also create our pipeline in advance as `azuremlsdkpipeline`.  Replace `YOURPATH` with the path fo your files.


```python
workspace_name = 'databricksazuresdkworkspace'
pipeline_name = 'databricksazuresdkpipeline'
model_name = 'databricksazuresdkmodel'
model_file_name = '/dbfs/FileStore/YOURPATH/ccfraud.onnx'
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


<table><tr><th>name</th> <td>databricksazuresdkpipeline</td></tr><tr><th>created</th> <td>2023-01-10 17:55:39.358777+00:00</td></tr><tr><th>last_updated</th> <td>2023-01-10 17:55:53.661965+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>ba1e0ef9-46e1-494f-9d51-c0fa59e8cacb, fae0c1c8-75d9-46a9-a3a6-0ffba26f5530</td></tr><tr><th>steps</th> <td>databricksazuresdkmodel</td></tr></table>


We can verify the workspace is created the current default workspace with the `get_current_workspace()` command.


```python
wl.get_current_workspace()
```


    Out[23]: {'name': 'databricksazuresdkworkspace', 'id': 12, 'archived': False, 'created_by': 'ebe1b913-dedc-4eb7-810c-ab4882925c9f', 'created_at': '2023-01-10T17:55:39.284757+00:00', 'models': [{'name': 'databricksazuresdkmodel', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 1, 10, 17, 55, 46, 576382, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 1, 10, 17, 55, 46, 576382, tzinfo=tzutc())}], 'pipelines': [{'name': 'databricksazuresdkpipeline', 'create_time': datetime.datetime(2023, 1, 10, 17, 55, 39, 358777, tzinfo=tzutc()), 'definition': '[]'}]}


### Upload the Models

Now we will upload our model.

**IMPORTANT NOTE**:  Use the local file path format such as `/dbfs/FileStore/shared_uploads/YOURWORKSPACE/file` format rather than the `dbfs:` format.


```python
model = wl.upload_model(model_name, model_file_name).configure()
```

### Deploy a Model

Now that we have a model that we want to use we will create a deployment for it. 

To do this, we'll create our pipeline that can ingest the data, pass the data to our CCFraud model, and give us a final output.  We'll call our pipeline `databricksazuresdkpipeline`, then deploy it so it's ready to receive data.  The deployment process usually takes about 45 seconds.


```python
pipeline.add_model_step(model)
```


<table><tr><th>name</th> <td>databricksazuresdkpipeline</td></tr><tr><th>created</th> <td>2023-01-10 17:55:39.358777+00:00</td></tr><tr><th>last_updated</th> <td>2023-01-10 17:55:53.661965+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>ba1e0ef9-46e1-494f-9d51-c0fa59e8cacb, fae0c1c8-75d9-46a9-a3a6-0ffba26f5530</td></tr><tr><th>steps</th> <td>databricksazuresdkmodel</td></tr></table>



```python
pipeline.deploy()
```


<table><tr><th>name</th> <td>databricksazuresdkpipeline</td></tr><tr><th>created</th> <td>2023-01-10 17:55:39.358777+00:00</td></tr><tr><th>last_updated</th> <td>2023-01-10 18:09:52.777311+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>fb0bf252-c52c-4b9b-8ce3-85597bb60b38, ba1e0ef9-46e1-494f-9d51-c0fa59e8cacb, fae0c1c8-75d9-46a9-a3a6-0ffba26f5530</td></tr><tr><th>steps</th> <td>databricksazuresdkmodel</td></tr></table>


We can verify that the pipeline is running and list what models are associated with it.


```python
pipeline.status()
```


    Out[27]: {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.3.9',
       'name': 'engine-5ccd6fcf4-gngcc',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'databricksazuresdkpipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'databricksazuresdkmodel',
          'version': 'ef46e72c-ba78-437c-883c-737d4e9908a5',
          'sha': 'bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.3.10',
       'name': 'engine-lb-55dcdff64c-4ggbr',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}


## Interferences

### Infer 1 row

Now that the pipeline is deployed and our CCfraud model is in place, we'll perform a smoke test to verify the pipeline is up and running properly.  We'll use the `infer_from_file` command to load a single transaction and determine if it is flagged for fraud.  If it returns correctly, a small valud should be returned indicating a low likelihood that the transaction was fraudulent.


```python
result = pipeline.infer_from_file("/dbfs/FileStore/YOURPATH/smoke_test.json")

```


```python
result[0].data()
```


    Out[29]: [array([[0.00149742]])]


### Batch Inference

Now that our smoke test is successful, let's really give it some data.  We'll use the `cc_data_1k.json` file that contains 1,000 inferences to be performed.


```python
result = pipeline.infer_from_file("/dbfs/FileStore/YOURPATH/cc_data_1k.json")
result
```

## Undeploy Pipeline

When finished with our tests, we will undeploy the pipeline so we have the Kubernetes resources back for other tasks.  Note that if the deployment variable is unchanged pipeline.deploy() will restart the inference engine in the same configuration as before.


```python
pipeline.undeploy()
```


<table><tr><th>name</th> <td>databricksazuresdkpipeline</td></tr><tr><th>created</th> <td>2023-01-10 17:55:39.358777+00:00</td></tr><tr><th>last_updated</th> <td>2023-01-10 18:09:52.777311+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>fb0bf252-c52c-4b9b-8ce3-85597bb60b38, ba1e0ef9-46e1-494f-9d51-c0fa59e8cacb, fae0c1c8-75d9-46a9-a3a6-0ffba26f5530</td></tr><tr><th>steps</th> <td>databricksazuresdkmodel</td></tr></table>

