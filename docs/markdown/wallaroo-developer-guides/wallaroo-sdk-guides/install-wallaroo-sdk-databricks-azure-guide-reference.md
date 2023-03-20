This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/development/sdk-install-guides/databricks-azure-sdk-install).

## Installing the Wallaroo SDK into  Workspace

Organizations that use Azure Databricks for model training and development can deploy models to Wallaroo through the [Wallaroo SDK](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/).  The following guide is created to assist users with installing the Wallaroo SDK, setting up authentication through Azure Databricks, and making a standard connection to a Wallaroo instance through Azure Databricks Workspace.

These instructions are based on the on the [Wallaroo SSO for Microsoft Azure](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-sso-authentication/wallaroo-sso-azure/) and the [Connect to Wallaroo](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/) guides.

This tutorial provides the following:

* `ccfraud.onnx`:  A pretrained model from the Machine Learning Group's demonstration on [Credit Card Fraud detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
* Sample inference test data:
  * `ccfraud_high_fraud.json`: Test input file that returns a high likelihood of credit card fraud.
  * `ccfraud_smoke_test.json`: Test input file that returns a low likelihood of credit card fraud.
  * `cc_data_1k.json`:  Sample input file with 1,000 records.
  * `cc_data_10k.json`:  Sample input file with 10,000 records.

To use the Wallaroo SDK within Azure Databricks Workspace, a virtual environment will be used.  This will set the necessary libraries and specific Python version required.

## Prerequisites

The following is required for this tutorial:

* A Wallaroo instance version 2023.1 or later with [External Inference URls enabled](https://docs.wallaroo.ai/wallaroo-tutorials/wallaroo-tutorial-features/wallaroo-model-endpoints/wallaroo-external-inference-tutorial/).
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
1. Select **PyPI**.  In the **Package** field, enter the current version of the [Wallaroo SDK](https://pypi.org/project/wallaroo/).  It is recommended to specify the version, which as of this writing is `wallaroo==2023.1.0`.

    * **IMPORTANT NOTE**:  The version of the Wallaroo SDK should match the Wallaroo instance.  For example, this example connects to a Wallaroo Enterprise version `2023.1` instance, so the SDK version should be `wallaroo==2023.1.0`.

1. Select **Install**.

Once the **Status** shows **Installed**, it will be available in Azure Databricks notebooks and other tools that use the cluster.

### Add Tutorial Files

The following instructions can be used to upload this tutorial and it's files into Databricks.  Depending on how your Azure Databricks is configured and your organizations standards, there are multiple ways of uploading files to your Azure Databricks environment.  The following example is used for the tutorial and makes it easy to reference data files from within this Notebook.  Adjust based on your requirements.

* **IMPORTANT NOTE**:  Importing a repo from a Git repository may not convert the included Jupyter Notebooks into the Databricks format.  This method 

1. From the Azure Databricks dashboard, select **Repos**.
1. Select where to place the repo, then select **Add Repo**.
 
    ![](./images/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-install-guides/databricks-azure-sdk-guide/azure-databricks-select-add-repo.png)

1. Set the following:
 
    ![](./images/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-install-guides/databricks-azure-sdk-guide/azure-databricks-add-repo-settings.png)

    1. **Create repo by cloning a Git repository**:  Uncheck
    1. **Repository name**:  Set any name based on the Databricks standard (no spaces, etc).
    1. Select **Create Repo**.
1. Select the new tutorial, then from the repo menu dropdown, select **Import**.
 
    ![](./images/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-install-guides/databricks-azure-sdk-guide/azure-databricks-repo-select-import.png)

1. Select the files to upload.  For this example, the following files are uploaded:

    ![](./images/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-install-guides/databricks-azure-sdk-guide/azure-databricks-repo-import-files.png)

    1. `ccfraud.onnx`:  A pretrained model from the Machine Learning Group's demonstration on [Credit Card Fraud detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
    1. Sample inference test data:
        1. `ccfraud_high_fraud.json`: Test input file that returns a high likelihood of credit card fraud.
        1. `ccfraud_smoke_test.json`: Test input file that returns a low likelihood of credit card fraud.
        1. `cc_data_1k.json`:  Sample input file with 1,000 records.
        1. `cc_data_10k.json`:  Sample input file with 10,000 records.
    1. `install-wallaroo-sdk-databricks-azure-guide.ipynb`: This notebook.
1. Select **Import**.

The Jupyter Notebook can be opened from this new Azure Databricks repository, and relative files it references will be accessible with the exceptions listed below.

Zip files added via the method above are automatically decompressed, so can not be used as model files.  For example, tensor based models such as the [Wallaroo Aloha Demo](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-model-cookbooks/aloha).  Zip files can be uploaded using DBFS and used through the following process:

To upload model files to Azure Databricks using DBFS:

1. From the Azure Databricks dashboard, select **Data**.
1. Select **Add->Add data**.

    ![](./images/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-install-guides/databricks-azure-sdk-guide/azure_databricks_select_add_data.png)
 
1. Select **DBFS**.

    ![](./images/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-install-guides/databricks-azure-sdk-guide/azure_databricks_select_dbfs.png)

1. Select **Upload File** and enter the following:
 
    ![](./images/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-install-guides/databricks-azure-sdk-guide/azure_databricks_upload_files.png) 

    1. **DBFS Target Directory** (*Optional*): Optional step:  Set the directory where the files will be uploaded.
1. Select the files to upload.  Note that each file will be given a location and they can be access with `/dbfs/PATH`.  For example, the file `alohacnnlstm.zip` uploaded to the directory `aloha` would be referenced with `/dbfs/FileStore/tables/aloha/alohacnnlstm.zip

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

Replace `YOUR PREFIX` and `YOUR SUFFIX` with the DNS prefix and suffix for the Wallaroo instance.  For more information, see the [DNS Integration Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-dns-guide/).


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

We will create a workspace to work in and call it the `databricksazuresdkworkspace`, then set it as current workspace environment.  We'll also create our pipeline in advance as `databricksazuresdkpipeline`.

* **IMPORTANT NOTE**:  For this example, the CCFraud model is stored in the file `ccfraud.onnx` and is referenced from a relative link.  For platforms such as Databricks, the files may need to be in a universal file format.  For those, the example file location below may be:

`model_file_name = '/dbfs/FileStore/tables/aloha/alohacnnlstm.zip`

Adjust file names and locations based on your requirements.


```python
workspace_name = 'databricksazuresdkworkspace'
pipeline_name = 'databricksazuresdkpipeline'
model_name = 'ccfraudmodel'
model_file_name = './ccfraud.onnx'
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


<table><tr><th>name</th> <td>databricksazuresdkpipeline</td></tr><tr><th>created</th> <td>2023-02-07 15:55:40.574745+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-07 15:55:40.574745+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>ca6d6ea6-2e45-4795-9253-5c40e8483dc9</td></tr><tr><th>steps</th> <td></td></tr></table>


We can verify the workspace is created the current default workspace with the `get_current_workspace()` command.


```python
wl.get_current_workspace()
```


    Out[6]: {'name': 'databricksazuresdkworkspace', 'id': 8, 'archived': False, 'created_by': '3547815c-b48d-4e69-bfbd-fff9d525c5d7', 'created_at': '2023-02-07T15:55:39.548497+00:00', 'models': [], 'pipelines': [{'name': 'databricksazuresdkpipeline', 'create_time': datetime.datetime(2023, 2, 7, 15, 55, 40, 574745, tzinfo=tzutc()), 'definition': '[]'}]}


### Upload the Models

Now we will upload our model.

**IMPORTANT NOTE**:  If using DBFS, use the file path format such as `/dbfs/FileStore/shared_uploads/YOURWORKSPACE/file` format rather than the `dbfs:` format.


```python
model = wl.upload_model(model_name, model_file_name).configure()
model
```


    Out[15]: {'name': 'ccfraudmodel', 'version': 'ccb488dd-36ed-4aaf-99cf-9a16bd3654db', 'file_name': 'ccfraud.onnx', 'image_path': None, 'last_update_time': datetime.datetime(2023, 2, 7, 16, 1, 0, 303545, tzinfo=tzutc())}


### Deploy a Model

Now that we have a model that we want to use we will create a deployment for it. 

To do this, we'll create our pipeline that can ingest the data, pass the data to our CCFraud model, and give us a final output.  We'll call our pipeline `databricksazuresdkpipeline`, then deploy it so it's ready to receive data.  The deployment process usually takes about 45 seconds.


```python
pipeline.add_model_step(model)
```


<table><tr><th>name</th> <td>databricksazuresdkpipeline</td></tr><tr><th>created</th> <td>2023-02-07 15:55:40.574745+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-07 15:57:24.803281+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>971f9db7-1b73-4e72-8cdb-cfa2d5a9ddd7, 6c1028c4-3ca7-47b0-b3a6-834d12b57fc9, ca6d6ea6-2e45-4795-9253-5c40e8483dc9</td></tr><tr><th>steps</th> <td>ccfraudmodel</td></tr></table>



```python
pipeline.deploy()
```


<table><tr><th>name</th> <td>databricksazuresdkpipeline</td></tr><tr><th>created</th> <td>2023-02-07 15:55:40.574745+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-07 16:04:35.891487+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>091eed0f-8984-4753-9316-0fbbf68bb398, bb701da9-440b-4ce6-8b92-36446347e85c, 971f9db7-1b73-4e72-8cdb-cfa2d5a9ddd7, 6c1028c4-3ca7-47b0-b3a6-834d12b57fc9, ca6d6ea6-2e45-4795-9253-5c40e8483dc9</td></tr><tr><th>steps</th> <td>ccfraudmodel</td></tr></table>


We can verify that the pipeline is running and list what models are associated with it.


```python
pipeline.status()
```


    Out[23]: {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.2.34',
       'name': 'engine-754b5c457d-5c4pc',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'databricksazuresdkpipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'ccfraudmodel',
          'version': 'ccb488dd-36ed-4aaf-99cf-9a16bd3654db',
          'sha': 'bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.0.29',
       'name': 'engine-lb-74b4969486-mslkt',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}


## Interferences

### Infer 1 row

Now that the pipeline is deployed and our CCfraud model is in place, we'll perform a smoke test to verify the pipeline is up and running properly.  We'll use the `infer_from_file` command to load a single transaction and determine if it is flagged for fraud.  If it returns correctly, a small value should be returned indicating a low likelihood that the transaction was fraudulent.


```python
result = pipeline.infer_from_file("./ccfraud_smoke_test.json")

```


```python
result[0].data()
```


    Out[25]: [array([[0.00149742]])]


### Batch Inference

Now that our smoke test is successful, let's really give it some data.  We'll use the `cc_data_1k.json` file that contains 1,000 inferences to be performed.


```python
result = pipeline.infer_from_file("./cc_data_1k.json")
result
```


    Out[26]: [InferenceResult({'check_failures': [],
      'elapsed': 245003,
      'model_name': 'ccfraudmodel',
      'model_version': 'ccb488dd-36ed-4aaf-99cf-9a16bd3654db',
      'original_data': {'tensor': [[-1.060329750089797,
                                    2.354496709462385,
                                    -3.563878832646437,
                                    5.138734892618555,
                                    -1.23084570186641,
                                    -0.7687824607744093,
                                    -3.588122810891446,
                                    1.888083766259287,
                                    -3.2789674273886593,
                                    -3.956325455353324,
                                    4.099343911805088,
                                    -5.653917639476211,
                                    -0.8775733373342495,
                                    -9.131571191990632,
                                    -0.6093537872620682,
                                    -3.748027677256424,
                                    -5.030912501659983,
                                    -0.8748149525506821,
                                    1.9870535692026476,
                                    0.7005485718467245,
                                    0.9204422758154284,
                                    -0.10414918089758483,
                                    0.3229564351284999,
                                    -0.7418141656910608,
                                    0.03841201586730117,
                                    1.099343914614657,
                                    1.2603409755785089,
                                    -0.14662447391576958,
                                    -1.446321243938815],
                                   [-1.060329750089797,
                                    2.354496709462385,
                                    -3.563878832646437,
                                    5.138734892618555,
                                    -1.23084570186641,
                                    -0.7687824607744093,
                                    -3.588122810891446,
                                    1.888083766259287,
                                    -3.2789674273886593,
                                    -3.956325455353324,
                                    4.099343911805088,
                                    -5.653917639476211,
                                    -0.8775733373342495,
                                    -9.131571191990632,
                                    -0.6093537872620682,
                                    -3.748027677256424,
                                    -5.030912501659983,
                                    -0.8748149525506821,
                                    1.9870535692026476,
                                    0.7005485718467245,
                                    0.9204422758154284,
                                    -0.10414918089758483,
                                    0.3229564351284999,
                                    -0.7418141656910608,
                                    0.03841201586730117,
                                    1.099343914614657,
                                    1.2603409755785089,
                                    -0.14662447391576958,
                                    -1.446321243938815],
                                   [-1.060329750089797,
                                    2.354496709462385,
                                    -3.563878832646437,
                                    5.138734892618555,
                                    -1.23084570186641,
                                    -0.7687824607744093,
                                    -3.588122810891446,
                                    1.888083766259287,
                                    -3.2789674273886593,
                                    -3.956325455353324,
                                    4.099343911805088,
                                    -5.653917639476211,
                                    -0.8775733373342495,
                                    -9.131571191990632,
                                    -0.6093537872620682,
                                    -3.748027677256424,
                                    -5.030912501659983,
                                    -0.8748149525506821,
                                    1.9870535692026476,
                                    0.7005485718467245,
                                    0.9204422758154284,
                                    -0.10414918089758483,
                                    0.3229564351284999,
                                    -0.7418141656910608,
                                    0.03841201586730117,
                                    1.099343914614657,
                                    1.2603409755785089,
                                    -0.14662447391576958,
                                    -1.446321243938815],
                                   [-1.060329750089797,
                                    2.354496709462385,
                                    -3.563878832646437,
                                    5.138734892618555,
                                    -1.23084570186641,
                                    -0.7687824607744093,
                                    -3.588122810891446,
                                    1.888083766259287,
                                    -3.2789674273886593,
                                    -3.956325455353324,
                                    4.099343911805088,
                                    -5.653917639476211,
                                    -0.8775733373342495,
                                    -9.131571191990632,
                                    -0.6093537872620682,
                                    -3.748027677256424,
                                    -5.030912501659983,
                                    -0.8748149525506821,
                                    1.9870535692026476,
                                    0.7005485718467245,
                                    0.9204422758154284,
                                    -0.10414918089758483,
                                    0.3229564351284999,
                                    -0.7418141656910608,
                                    0.03841201586730117,
                                    1.099343914614657,
                                    1.2603409755785089,
                                    -0.14662447391576958,
                                    -1.446321243938815],
                                   [0.5817662107606553,
                                    0.0978815509566172,
                                    0.1546819423995403,
                                    0.475410194903404,
                                    -0.1978862305998003,
                                    -0.45043448542395703,
                                    0.016654044671806197,
                                    -0.025607055099995037,
                                    0.09205616023555586,
                                    -0.27839171528517387,
                                    0.059329944112281194,
                                    -0.019658541640589822,
                                    -0.4225083156745137,
                                    -0.12175388766841427,
                                    1.547309489412488,
                                    0.23916228635697,
                                    0.35539748808055915,
                                    -0.7685165300981693,
                                    -0.7000849354838512,
                                    -0.11900432852127547,
                                    -0.3450517133266211,
                                    -1.1065114107709193,
                                    0.2523411195349452,
                                    0.02094418256934876,
                                    0.2199267436399366,
                                    0.2540689265485751,
                                    -0.04502250942505252,
                                    0.1086773897916229,
                                    0.2547179311087416],
                                   [-0.7621273681123774,
                                    0.8854701414345362,
                                    0.5235808652087769,
                                    -0.8139743550578189,
                                    0.3793240543966917,
                                    0.15606533645358955,
                                    0.545129966459155,
                                    0.07859272424715734,
                                    0.41439685426159006,
                                    0.49140523482948895,
                                    0.07743910220902032,
                                    1.050105025966046,
                                    0.9901440216912372,
                                    -0.614248313100663,
                                    -1.5260740653027238,
                                    0.2053324702711796,
                                    -1.0185637854071916,
                                    0.04909869191405787,
                                    0.6964184879033418,
                                    0.5948331721915132,
                                    -0.3934362921711871,
                                    -0.5922492660097428,
                                    -0.3953093077108832,
                                    -1.331042702500481,
                                    0.6287441286760012,
                                    0.8665525995997287,
                                    0.7974673604471482,
                                    1.1174342262023085,
                                    -0.6700716550561031],
                                   [-0.2836830106617754,
                                    0.2281341607542476,
                                    1.0358808684971377,
                                    1.031141364744695,
                                    0.6485053916657638,
                                    0.6993338999916012,
                                    0.1827667194489511,
                                    0.09897462120147606,
                                    -0.573448773318372,
                                    0.5928927597000144,
                                    0.3085637362189933,
                                    0.15338699178269907,
                                    -0.3628347922840285,
                                    -0.28650544988763965,
                                    -1.138044653648458,
                                    -0.22071176013852775,
                                    -0.12060339309501608,
                                    -0.23252469358947547,
                                    0.8675179232286943,
                                    -0.00813230344349814,
                                    -0.015330414985472576,
                                    0.41692378222119375,
                                    -0.42490253139063966,
                                    -0.983445197690985,
                                    -1.117590357786289,
                                    2.107670188520057,
                                    -0.33619500725255724,
                                    -0.3469573431212065,
                                    0.019307669007054214],
                                   [1.037963634604398,
                                    -0.15298730197183308,
                                    -1.0912561861755297,
                                    -0.003333982808610693,
                                    0.48042818357577816,
                                    0.11207084748490805,
                                    0.023315770873913674,
                                    0.0009213037997834434,
                                    0.4021730182105383,
                                    0.2120753711962651,
                                    -0.14628042225168944,
                                    0.44244770274013223,
                                    -0.4641602116945049,
                                    0.49842564302053766,
                                    -0.8230270969280085,
                                    0.3168388183929484,
                                    -0.905044097738204,
                                    0.07103650391675659,
                                    1.1111388586922986,
                                    -0.2157914053975094,
                                    -0.37375912900543384,
                                    -1.033007534671374,
                                    0.31447209128965764,
                                    -0.5109243112374892,
                                    -0.16859104983418324,
                                    0.5918324405536384,
                                    -0.22317928245806465,
                                    -0.22871533772536015,
                                    -0.0868944761624121],
                                   [0.15172836621737265,
                                    0.6589966337195882,
                                    -0.33237136470392026,
                                    0.7285871978728441,
                                    0.6430271572675802,
                                    -0.036105130607259,
                                    0.22015305036081068,
                                    -1.4928731939082054,
                                    -0.5895806486715522,
                                    0.22272511026018857,
                                    0.4443729713208923,
                                    0.8411555815062762,
                                    -0.24129130201177532,
                                    0.8986828750053317,
                                    -0.9866307095643508,
                                    -0.891930176747572,
                                    -0.08788759139559761,
                                    0.11633324608127409,
                                    1.1469566645633804,
                                    -0.5417470436307007,
                                    2.232136056300802,
                                    -0.16792713816415766,
                                    -0.8071223667464775,
                                    -0.6379226787209245,
                                    1.9121889390871136,
                                    -0.5565545168737087,
                                    0.6528273963811771,
                                    0.8163897965987713,
                                    -0.22816150171105992],
                                   [-0.16831002464168482,
                                    0.7070470316726095,
                                    0.18752349479594543,
                                    -0.3885406952480356,
                                    0.8190382136893654,
                                    -0.2264929889455448,
                                    0.920446915383558,
                                    -0.1362740973549585,
                                    -0.3336344399134833,
                                    -0.31742816858138206,
                                    1.190347893355806,
                                    0.17742920974706458,
                                    -0.5681631428570322,
                                    -0.8907063934925815,
                                    -0.5603225648833638,
                                    0.08978317373468075,
                                    0.41875259056737263,
                                    0.34062690461012146,
                                    0.7358794384123696,
                                    0.2162316926274178,
                                    -0.4090832914654094,
                                    -0.873608946074589,
                                    -0.11287065093605424,
                                    1.0027861773717552,
                                    -0.940491615382638,
                                    0.34471446407049355,
                                    0.09082338670023896,
                                    0.03385948858451272,
                                    -1.5295522680268],
                                   [0.6066235673660867,
                                    0.06318393046103796,
                                    -0.08029619730834595,
                                    0.6955262344665573,
                                    -0.1775255858536255,
                                    -0.37571582613170335,
                                    -0.10034783809984708,
                                    -0.002020697400621504,
                                    0.6859442462445478,
                                    -0.6582840559236135,
                                    -0.9995187665924608,
                                    -0.5340094457850662,
                                    -1.1303344301902345,
                                    -1.4048093394603511,
                                    -0.09533161186902651,
                                    0.34286507076318934,
                                    1.137627771131194,
                                    0.42483092016552,
                                    0.23163849625535257,
                                    -0.11453707463184153,
                                    -0.30158635696358,
                                    -0.6731341245200443,
                                    -0.2723217481414279,
                                    -0.392522783076639,
                                    1.1115261431276475,
                                    0.9205381913240704,
                                    -0.028059000408212655,
                                    0.13116439016892018,
                                    0.2152022580020345],
                                   [0.6022605285983497,
                                    0.03354188522587924,
                                    0.07384927695250888,
                                    0.18511785364463623,
                                    -0.305485553894443,
                                    -0.7940218336809065,
                                    0.16549419059256967,
                                    -0.13036002461367513,
                                    -0.18841586940040084,
                                    0.06659757810555761,
                                    1.4810974231280167,
                                    0.6472122044773744,
                                    -0.6703196483832992,
                                    0.7565686747307261,
                                    0.2134058731218033,
                                    0.15054757512303818,
                                    -0.4312378588876496,
                                    -0.01829519245300039,
                                    0.2851995511280944,
                                    -0.10765090263665966,
                                    0.006824282636551462,
                                    -0.10765890483072864,
                                    -0.0788026490786185,
                                    0.9475328124756416,
                                    0.8413388083261754,
                                    1.1769860739049118,
                                    -0.20262122059889132,
                                    -0.0006311264993808188,
                                    0.18515595494858325],
                                   [-1.2004162236340663,
                                    -0.02934247149289781,
                                    0.6002673902810236,
                                    -1.0581165763998934,
                                    0.8618826503029525,
                                    0.9173564431626324,
                                    0.07531515110044265,
                                    0.22061892248030848,
                                    1.218873509137122,
                                    -0.3886523829726902,
                                    -0.6095125829994053,
                                    0.19650432666838064,
                                    -0.2661495951765694,
                                    -0.6379133677491714,
                                    0.48339834201800247,
                                    -0.4985531206523148,
                                    -0.30642432885045834,
                                    -1.452449679301684,
                                    -3.114069963143443,
                                    -1.0750208205893026,
                                    0.33412238420877444,
                                    1.5687760942001978,
                                    -0.520167136432032,
                                    -0.5317761577207334,
                                    -0.383294946943516,
                                    -0.9846864506812528,
                                    -2.8976275684335313,
                                    -0.5073512289684565,
                                    -0.3252693380620513],
                                   [-2.842735703124214,
                                    2.8260142810969406,
                                    -1.595334491992825,
                                    -0.2991672885943705,
                                    -1.5495220405376615,
                                    1.6401772163256094,
                                    -3.282195184902111,
                                    0.4863028450594385,
                                    0.35768012762513235,
                                    0.32223721627031443,
                                    0.2710846268609854,
                                    2.0025589607976957,
                                    -1.2151492104208754,
                                    2.2835338743639055,
                                    -0.4148465662141878,
                                    -0.6786230740923882,
                                    2.6106103197644366,
                                    -1.6468850705007771,
                                    -1.608012375610504,
                                    0.99954742225582,
                                    -0.4665201752903192,
                                    0.8316050291541919,
                                    1.2855678736315532,
                                    -2.3561879047775687,
                                    0.21079384022245212,
                                    1.1256706306463826,
                                    1.1538189945359587,
                                    1.0332061029880848,
                                    -1.4715524921303922],
                                   [0.37025168630329713,
                                    -0.559565196966135,
                                    0.6757255982084903,
                                    0.6920586122163292,
                                    -1.097595083688384,
                                    -0.3864326068450856,
                                    -0.24648264110773657,
                                    -0.030465499323007045,
                                    0.6888634287660991,
                                    -0.25688240219544295,
                                    -0.31262126741537455,
                                    0.4177154048654652,
                                    0.08658272648758861,
                                    -0.23586536095779972,
                                    0.7111355283012393,
                                    0.2937196011516156,
                                    -0.21506657714816485,
                                    -0.03799982400637199,
                                    -0.5299304856390635,
                                    0.4921897195724529,
                                    0.3506791137140131,
                                    0.4934112481922948,
                                    -0.36453056821705304,
                                    1.3046786490119904,
                                    0.38837918082008666,
                                    1.0901335639291738,
                                    -0.0981875890062536,
                                    0.22211438851412624,
                                    1.2586007559041816],
                                   [1.0740600994534184,
                                    -0.5004444122961789,
                                    -0.6665104458726077,
                                    -0.513795450608903,
                                    -0.22180251404665496,
                                    0.17340111491593263,
                                    -0.6004987142090168,
                                    0.010577019331547527,
                                    -0.4816997216919937,
                                    0.8861570874243399,
                                    0.18529769780712504,
                                    0.8741913982235725,
                                    1.1965272846208048,
                                    -0.10993396869909432,
                                    -0.6057668522157109,
                                    -1.1254346438176983,
                                    -0.7830480707969095,
                                    1.9148497747436344,
                                    -0.3837603969088797,
                                    -0.6387752587185815,
                                    -0.4853295482654116,
                                    -0.5961116065703739,
                                    0.4123371083341403,
                                    0.17603697938449023,
                                    -0.5173803145442223,
                                    1.1181808796610917,
                                    -0.0934318754864336,
                                    -0.1756922307137465,
                                    -0.2551430327198796],
                                   [-0.3389501953944846,
                                    0.4600633972547716,
                                    1.5422134202684443,
                                    0.026738992407616496,
                                    0.11589317308681447,
                                    0.5045446890411369,
                                    0.05163626851385762,
                                    0.26452863620
    
    *** WARNING: max output size exceeded, skipping output. ***
    
                             0.0008339285850524902,
                                      0.0006878077983856201,
                                      0.001112222671508789,
                                      0.0005952417850494385,
                                      0.0003427863121032715,
                                      0.0006614029407501221,
                                      0.001322627067565918,
                                      0.0005146563053131104,
                                      4.824995994567871e-05,
                                      0.00046452879905700684,
                                      0.0003368556499481201,
                                      0.0012190043926239014,
                                      0.00046455860137939453,
                                      0.0009738504886627197,
                                      0.00035002827644348145,
                                      0.00039589405059814453,
                                      0.000307619571685791,
                                      0.0005711615085601807,
                                      0.0005376338958740234,
                                      0.0001920461654663086,
                                      0.0009895861148834229,
                                      0.0007052123546600342,
                                      0.0005137920379638672,
                                      0.00035962462425231934,
                                      0.0007860660552978516,
                                      0.000491708517074585,
                                      7.635354995727539e-05,
                                      0.00026789307594299316,
                                      0.0019146502017974854,
                                      0.0006752610206604004,
                                      0.0008069276809692383,
                                      0.0004373788833618164,
                                      0.0007348060607910156,
                                      0.00010257959365844727,
                                      0.0003650486469268799,
                                      0.001430898904800415,
                                      0.0011163949966430664,
                                      0.0005064606666564941,
                                      0.0006780624389648438,
                                      0.0007084012031555176,
                                      0.0005066394805908203,
                                      0.0005592107772827148,
                                      0.0007954835891723633,
                                      0.000926285982131958,
                                      0.0006126761436462402,
                                      0.0003502964973449707,
                                      0.000958859920501709,
                                      0.0002881288528442383,
                                      0.00016897916793823242,
                                      0.0006831586360931396,
                                      0.0003865659236907959,
                                      0.00016203522682189941,
                                      0.0008713304996490479,
                                      0.0004932284355163574,
                                      0.0004909336566925049,
                                      0.00022536516189575195,
                                      0.0009913146495819092,
                                      0.0002721548080444336,
                                      8.744001388549805e-05,
                                      0.0006993114948272705,
                                      0.0010588765144348145,
                                      0.0009733438491821289,
                                      0.0006800591945648193,
                                      0.0002625584602355957,
                                      0.0006255805492401123,
                                      0.00024187564849853516,
                                      0.0002522468566894531,
                                      0.0008753836154937744,
                                      0.0002613067626953125,
                                      0.0005331039428710938,
                                      0.0002490878105163574,
                                      0.0001704394817352295,
                                      0.00031509995460510254,
                                      0.0015914440155029297,
                                      0.00025537610054016113,
                                      8.07344913482666e-05,
                                      0.0008647739887237549,
                                      0.0004987716674804688,
                                      0.001710742712020874,
                                      0.0013418197631835938,
                                      0.00037536025047302246,
                                      0.0003878176212310791,
                                      0.0005452334880828857,
                                      0.0007519721984863281,
                                      0.0008081197738647461,
                                      0.000502467155456543,
                                      0.0003039240837097168,
                                      0.0005827546119689941,
                                      0.0006529092788696289,
                                      0.0010212063789367676,
                                      0.00034746527671813965,
                                      0.0008154213428497314,
                                      0.00038063526153564453,
                                      0.0005306899547576904,
                                      0.00025406479835510254,
                                      0.00018146634101867676,
                                      0.0013905465602874756,
                                      0.0006494820117950439,
                                      0.0006037354469299316,
                                      0.0014120042324066162,
                                      0.00041112303733825684,
                                      0.00040650367736816406,
                                      0.0005333423614501953,
                                      0.0007215738296508789,
                                      0.0001367330551147461,
                                      0.0003502070903778076,
                                      0.0009997785091400146,
                                      0.0008716285228729248,
                                      0.0005594789981842041,
                                      0.000410228967666626,
                                      0.0001429915428161621,
                                      0.0003579556941986084,
                                      0.0011880695819854736,
                                      0.0003827214241027832,
                                      0.0012142062187194824,
                                      0.0005961358547210693,
                                      0.000471651554107666,
                                      0.0006967782974243164,
                                      0.00037926435470581055,
                                      0.0003273487091064453,
                                      0.0016745328903198242,
                                      0.0003102719783782959,
                                      0.0010521411895751953,
                                      3.841519355773926e-05,
                                      0.0004825592041015625,
                                      0.0009035468101501465,
                                      0.0009154081344604492,
                                      0.0009016096591949463,
                                      0.0011216700077056885,
                                      0.0002802610397338867,
                                      0.0007374584674835205,
                                      0.0005075931549072266,
                                      0.0006051957607269287,
                                      0.0005790889263153076,
                                      0.00032085180282592773,
                                      0.00042501091957092285,
                                      0.0007457137107849121,
                                      0.0006720125675201416,
                                      0.0003052949905395508,
                                      0.0006992816925048828,
                                      0.0003927946090698242,
                                      0.00024440884590148926,
                                      0.0001997053623199463,
                                      0.0002860724925994873,
                                      0.000585019588470459,
                                      0.00021448731422424316,
                                      0.000881195068359375,
                                      0.0004405081272125244,
                                      0.0008642077445983887,
                                      0.0005924403667449951,
                                      0.0007340312004089355,
                                      0.0004509389400482178,
                                      0.0008679628372192383,
                                      0.00037926435470581055,
                                      0.0008240938186645508,
                                      0.0007452666759490967,
                                      0.00033849477767944336,
                                      0.0011382997035980225,
                                      0.0003623068332672119,
                                      0.0002282559871673584,
                                      0.0005411803722381592,
                                      0.001323312520980835,
                                      0.0009799599647521973,
                                      0.0008512735366821289,
                                      0.0007756352424621582,
                                      0.0003809928894042969,
                                      0.00017562508583068848,
                                      0.0005088448524475098,
                                      0.00014969706535339355,
                                      9.685754776000977e-05,
                                      0.0016102492809295654,
                                      0.0003826320171356201,
                                      0.0013871490955352783,
                                      0.00020483136177062988,
                                      0.0011193156242370605,
                                      0.0008026957511901855,
                                      0.00047454237937927246,
                                      0.0005080103874206543,
                                      0.0012269020080566406,
                                      0.00022527575492858887,
                                      0.00020378828048706055,
                                      0.0004162788391113281,
                                      0.0008330047130584717,
                                      2.4050474166870117e-05,
                                      0.0006586611270904541,
                                      0.000383526086807251,
                                      0.00040608644485473633,
                                      0.00040709972381591797,
                                      0.00020489096641540527,
                                      0.0006171464920043945,
                                      0.0012582242488861084,
                                      0.0004496574401855469,
                                      0.0005507469177246094,
                                      0.0008178949356079102,
                                      0.001517951488494873,
                                      0.00017982721328735352,
                                      0.000568687915802002,
                                      0.001766800880432129,
                                      0.0002658367156982422,
                                      0.000822216272354126,
                                      0.0004229545593261719,
                                      0.00025528669357299805,
                                      0.0004892349243164062,
                                      0.000771939754486084,
                                      0.0010519325733184814,
                                      0.0010221898555755615,
                                      9.08970832824707e-05,
                                      0.0008391737937927246,
                                      0.00022780895233154297,
                                      0.0007468760013580322,
                                      0.0007697641849517822,
                                      0.0019667446613311768,
                                      0.0012534558773040771,
                                      0.0001010596752166748,
                                      0.0005205869674682617,
                                      0.0002041459083557129,
                                      0.0006001889705657959,
                                      0.0009807944297790527,
                                      0.000767141580581665,
                                      0.00038120150566101074,
                                      0.0002471506595611572,
                                      0.00038233399391174316,
                                      0.00037872791290283203,
                                      0.0007638931274414062,
                                      0.00029391050338745117,
                                      0.0008871853351593018,
                                      0.0004890561103820801,
                                      0.0015825629234313965,
                                      0.0005756914615631104,
                                      0.0003350973129272461,
                                      0.00026857852935791016,
                                      0.0010086894035339355,
                                      0.00048220157623291016,
                                      0.00024381279945373535,
                                      6.434321403503418e-05,
                                      2.0682811737060547e-05,
                                      0.0003471970558166504,
                                      0.00022557377815246582,
                                      0.0002627372741699219,
                                      0.0003419220447540283,
                                      0.000281602144241333,
                                      0.0012967884540557861,
                                      0.0011523962020874023,
                                      0.0004177987575531006,
                                      0.0010204315185546875,
                                      0.0010258853435516357,
                                      0.0011347532272338867,
                                      0.00038436055183410645,
                                      0.0009618997573852539,
                                      0.00035199522972106934,
                                      0.000282973051071167,
                                      0.00024309754371643066,
                                      0.0001265406608581543,
                                      6.946921348571777e-05,
                                      0.00015616416931152344,
                                      0.0014993548393249512,
                                      0.0006575882434844971,
                                      0.0003606081008911133,
                                      0.0023556947708129883,
                                      0.0007058978080749512,
                                      0.0014238357543945312,
                                      0.0007699429988861084,
                                      0.0008679032325744629,
                                      0.00018492341041564941,
                                      0.0007839500904083252,
                                      0.0009354352951049805,
                                      0.00027167797088623047,
                                      0.0009218454360961914,
                                      0.00035691261291503906,
                                      0.0005003809928894043,
                                      0.0004172325134277344,
                                      0.0011021196842193604,
                                      0.0010276734828948975,
                                      0.0006104707717895508,
                                      0.00045561790466308594,
                                      0.0006892085075378418,
                                      0.0004885494709014893,
                                      0.0004724562168121338,
                                      0.001522064208984375,
                                      0.0005326271057128906,
                                      0.00010651350021362305,
                                      0.0002598762512207031,
                                      0.0013784170150756836,
                                      0.0004596710205078125,
                                      0.0003192126750946045,
                                      0.0009370148181915283,
                                      0.0006310641765594482,
                                      0.0005830228328704834,
                                      0.00036329030990600586,
                                      0.0009173750877380371,
                                      0.0006718039512634277,
                                      3.796815872192383e-05,
                                      0.00077781081199646,
                                      0.00033274292945861816,
                                      0.0001729726791381836,
                                      0.0008949339389801025,
                                      0.00026357173919677734,
                                      0.000757366418838501,
                                      0.0007928907871246338,
                                      0.0012267529964447021,
                                      0.0013829469680786133,
                                      0.0005187392234802246,
                                      0.0003561079502105713,
                                      0.000646054744720459,
                                      0.001015990972518921,
                                      0.0015155971050262451,
                                      0.0002993941307067871,
                                      0.00013318657875061035,
                                      0.0008256733417510986,
                                      0.0005404055118560791,
                                      0.0003667771816253662,
                                      0.0005891323089599609,
                                      0.0007394552230834961,
                                      0.0010330379009246826,
                                      0.0007327795028686523,
                                      0.0001760423183441162,
                                      0.0001805126667022705,
                                      0.0011722445487976074,
                                      0.00023120641708374023,
                                      0.00046622753143310547,
                                      0.0005017220973968506,
                                      0.00037470459938049316,
                                      0.0007470846176147461,
                                      0.00034102797508239746,
                                      0.0018736720085144043,
                                      0.0007473528385162354,
                                      0.0008576810359954834,
                                      0.0012683570384979248,
                                      0.0005511641502380371,
                                      0.0008003413677215576,
                                      0.0002823770046234131,
                                      0.0006742775440216064,
                                      0.0006029307842254639,
                                      0.00045618414878845215,
                                      0.00017344951629638672,
                                      0.0012264251708984375,
                                      0.001613914966583252,
                                      0.0009235143661499023,
                                      0.00029850006103515625,
                                      0.0003133118152618408,
                                      0.00010135769844055176,
                                      0.0004534423351287842,
                                      0.00031444430351257324,
                                      0.0007798373699188232,
                                      0.00038126111030578613,
                                      0.00026619434356689453,
                                      0.000617682933807373,
                                      0.0006511211395263672,
                                      0.0008475780487060547,
                                      1.519918441772461e-06,
                                      0.0002251267433166504,
                                      0.0002655982971191406,
                                      0.0005814731121063232,
                                      0.001587003469467163,
                                      0.00012886524200439453,
                                      0.000906139612197876,
                                      0.0006060898303985596,
                                      0.0004534125328063965,
                                      0.0005573630332946777,
                                      0.001411139965057373,
                                      0.0007226467132568359,
                                      0.0007477104663848877,
                                      0.0007035136222839355,
                                      0.00022074580192565918,
                                      0.0014317333698272705,
                                      0.0018418431282043457,
                                      0.00010865926742553711,
                                      0.0008140206336975098,
                                      0.0005422532558441162,
                                      0.00045371055603027344,
                                      0.0006635785102844238,
                                      0.0006209909915924072,
                                      0.0005052685737609863,
                                      0.0005816519260406494,
                                      0.9873101711273193,
                                      0.0006915628910064697,
                                      0.0007537007331848145,
                                      0.00029602646827697754,
                                      0.00020524859428405762,
                                      0.0011404454708099365,
                                      0.0007368624210357666,
                                      0.0002035200595855713,
                                      0.00048407912254333496,
                                      0.00041028857231140137,
                                      3.3676624298095703e-06,
                                      0.0004755854606628418,
                                      0.000834733247756958,
                                      0.0003497898578643799,
                                      0.0012320280075073242,
                                      0.0005603432655334473,
                                      0.0003822147846221924,
                                      0.0009741783142089844,
                                      0.0003153085708618164,
                                      0.0008485913276672363,
                                      0.0035923421382904053,
                                      0.00045371055603027344,
                                      0.0012863576412200928,
                                      0.000866323709487915,
                                      7.393956184387207e-05,
                                      0.0012035071849822998,
                                      0.00018787384033203125,
                                      0.00031045079231262207,
                                      0.0004418790340423584,
                                      0.0001100003719329834,
                                      0.0006164610385894775,
                                      2.7120113372802734e-06,
                                      0.0007382631301879883,
                                      0.00021120905876159668,
                                      0.00043717026710510254,
                                      0.0018209218978881836,
                                      0.00035813450813293457,
                                      0.00024771690368652344,
                                      0.0005538463592529297,
                                      0.0003204941749572754,
                                      0.0013484358787536621,
                                      0.0010192394256591797,
                                      0.0020678043365478516,
                                      0.00020268559455871582,
                                      0.00033402442932128906,
                                      0.00022429227828979492,
                                      0.00023245811462402344,
                                      0.00013360381126403809,
                                      0.0005823671817779541,
                                      0.0003317594528198242,
                                      0.0003043711185455322,
                                      0.0013128221035003662,
                                      0.0008148550987243652,
                                      0.0005481243133544922,
                                      0.0001258552074432373,
                                      0.00011596083641052246,
                                      0.0002785325050354004,
                                      0.00110703706741333,
                                      0.0008533000946044922,
                                      0.001249849796295166],
                             'dim': [1001, 1],
                             'dtype': 'Float',
                             'v': 1}}],
      'pipeline_name': 'databricksazuresdkpipeline',
      'shadow_data': {},
      'time': 1675785912268})]


## Undeploy Pipeline

When finished with our tests, we will undeploy the pipeline so we have the Kubernetes resources back for other tasks.  Note that if the deployment variable is unchanged pipeline.deploy() will restart the inference engine in the same configuration as before.


```python
pipeline.undeploy()
```
