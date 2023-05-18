The Wallaroo 101 tutorial can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-101).

## Introduction

Welcome to the Wallaroo, the fastest, easiest, and most efficient production ready machine learning system.

This tutorial is created to help you get started with Wallaroo right away.  We'll start with a brief explanation of how Wallaroo works, then provide the credit card fraud detection model so you can see it working.

This guide assumes that you've installed Wallaroo in your cloud Kubernetes cluster.  This can be either:

* Amazon Web Services (AWS)
* Microsoft Azure
* Google Cloud Platform

For instructions on setting up your cloud Kubernetes environment, check out the [Wallaroo Environment Setup Guides](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-setup-environment/) for your particular cloud provider.

### How to Use This Notebook

It is recommended that you run this notebook command at a time so you can see the results and make any changes you need based on your own environment.

## Prerequisites

* An installed Wallaroo instance.
* The following Python libraries installed:
  * `os`
  * [`wallaroo`](https://pypi.org/project/wallaroo/): The Wallaroo SDK. Included with the Wallaroo JupyterHub service by default.
  * [`pandas`](https://pypi.org/project/pandas/): Pandas, mainly used for Pandas DataFrame
  * [`pyarrow`](https://pypi.org/project/pyarrow/): PyArrow for Apache Arrow support
  * [`polars`](https://pypi.org/project/polars/): Polars for DataFrame with native Apache Arrow support

## SDK Introduction

The Wallaroo SDK lets you quickly get your models working with your data and getting results.  The typical flow follows these steps:

* **Connect**:  Connect to your Wallaroo Instance.
* **Create or Connect to a Workspace**:  Create a new workspace that will contain your models and pipelines, or connect to an existing one.
* **Upload or Use Existing Models**:  Upload your models to your workspace, or use ones that have already been uploaded.
* **Create or Use Existing Pipelines**:  Create or use an existing pipeline.  This is where you'll set the **steps** that will ingest your data, submit it through each successive model, then return a result.
* **Deploy Your Pipeline**:  Deploying a pipeline allocates resources from your Kubernetes environment for your models.
* **Run an Inference**:  This is where it all comes together.  Submit data through your pipeline either as a file or to your pipeline's deployment url, and get results.
* **Undeploy Your Pipeline**:  This returns the Kubernetes resources your pipeline used back to the Kubernetes environment.

For a more detailed rundown of the Wallaroo SDK, see the [Wallaroo SDK Essentials Guide](https://docs.wallaroo.ai/wallaroo-sdk/wallaroo-sdk-essentials-guide/).

### Introduction to Workspaces

A Wallaroo **Workspace** allows you to manage a set of models and pipelines.  You can assign users to a workspace as either an **owner** or **collaborator**.

When working within the Wallaroo SDK, the first thing you'll do after connecting is either create a workspace or set an existing workspace your **current workspace**.  From that point on, all models uploaded and pipelines created or used will be in the context of the current workspace.

### Introduction to Models

A Wallaroo **model** is a trained Machine Learning model that is uploaded to your current workspace.  These are the engines that take in data, run it through whatever process they have been trained for, and return a result.

Models don't work in a vacuum - they are allocated to a pipeline as detailed in the next step.

### Introduction to Pipelines

A Wallaroo **pipeline** is where the real work occurs.  A pipeline contains a series of **steps** - sequential sets of models which take in the data from the preceding step, process it through the model, then return a result.  Some models can be simple, such as the `cc_fraud` example listed below where the pipeline has only one step:

* Step 0: Take in data
* Step 1: Submit data to the model `ccfraudModel`.
* Step Final:  Return a result

Some models can be more complex with a whole series of models - and those results can be submitted to still other pipeline.  You can make pipelines as simple or complex as long as it meets your needs.

Once a step is created you can add additional steps, remove a step, or swap one out until everything is running perfectly.

**Note**: The Community Edition of Wallaroo limits users to two active pipelines, with a maximum of five steps per pipeline.

With all of that introduction out of the way, let's proceed to our Credit Card Detection Model.

This example will demonstrate how to use Wallaroo to detect credit card fraud through a trained model and sample data.  By the end of this example, you'll be able to:

* Start the Wallaroo client.
* Create a workspace.
* Upload the credit card fraud detection model to the workspace.
* Create a new pipeline and set it to our credit card fraud detection model.
* Run a smoke test to verify the pipeline and model is working properly.
* Perform a bulk inference and display the results.
* Undeploy the pipeline to get back the resources from our Kubernetes cluster.

This example and sample data comes from the Machine Learning Group's demonstration on [Credit Card Fraud detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Open a Connection to Wallaroo

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  If logging in externally, update the `wallarooPrefix` and `wallarooSuffix` variables with the proper DNS information.  For more information on Wallaroo DNS settings, see the [Wallaroo DNS Integration Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-dns-guide/).

```python
import wallaroo
from wallaroo.object import EntityNotFoundError

import pyarrow as pa

# used to display dataframe information without truncating
from IPython.display import display
import pandas as pd
pd.set_option('display.max_colwidth', None)
```

```python
wallaroo.__version__
```

    '2023.2.0rc3'

```python
# Login through local Wallaroo instance

wl = wallaroo.Client()
```

```python
wl.list_workspaces()
```

{{<table "table table-striped table-bordered" >}}
<table>
    <tr>
        <th>Name</th>
        <th>Created At</th>
        <th>Users</th>
        <th>Models</th>
        <th>Pipelines</th>
    </tr>

<tr >
    <td>john.hummel@wallaroo.ai - Default Workspace</td>
    <td>2023-05-17 20:36:36</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>0</td>
    <td>0</td>
</tr>

<tr >
    <td>housepricedrift</td>
    <td>2023-05-17 20:41:50</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>sdkquickworkspace</td>
    <td>2023-05-17 20:43:36</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>apiworkspaces</td>
    <td>2023-05-17 20:50:36</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>2</td>
    <td>3</td>
</tr>

<tr >
    <td>azuremlsdkworkspace</td>
    <td>2023-05-17 21:01:45</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>databricksazuresdkworkspace</td>
    <td>2023-05-17 21:02:53</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>gcpsdkworkspace</td>
    <td>2023-05-17 21:03:43</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>testautoconversion</td>
    <td>2023-05-17 21:11:40</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>2</td>
    <td>0</td>
</tr>

<tr >
    <td>externalkerasautoconvertworkspace</td>
    <td>2023-05-17 21:13:26</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>isolettest</td>
    <td>2023-05-17 21:17:28</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>statsmodelworkspace</td>
    <td>2023-05-17 21:19:51</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>xgboost-classification-autoconvert-workspace</td>
    <td>2023-05-17 21:21:19</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>xgboost-regression-autoconvert-workspace</td>
    <td>2023-05-17 21:21:55</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>housepricing</td>
    <td>2023-05-17 21:26:51</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>3</td>
    <td>1</td>
</tr>

</table>
{{</table>}}

## Create a New Workspace

Next we're going to create a new workspace called `ccfraudworkspace` for our model, then set it as our current workspace context.  We'll be doing this through the SDK, but here's an example of doing it through the Wallaroo dashboard.

The method we'll introduce below will either **create** a new workspace if it doesn't exist, or **select** an existing workspace.  So if you create the workspace `ccfraudworkspace` then you're covered either way.

The first part is to return to your Wallaroo Dashboard.  In the top navigation panel next to your user name there's a drop down with your workspaces.  In this example it just has "My Workspace".  Select **View Workspaces**.

![Select View Workspaces](./images/wallaroo-101/wallaroo-dashboard-select-view-workspaces.png)

From here, enter the name of our new workspace as `ccfraud-workspace`.  If it already exists, you can skip this step.

* **IMPORTANT NOTE**:  Workspaces do not have forced unique names.  It is highly recommended to use an existing workspace when possible, or establish a naming convention for your workspaces to keep their names unique to remove confusion with teams.

![Create ccfraud-workspace](./images/wallaroo-101/wallaroo-dashboard-create-workspace-ccfraud.png)

Once complete, you'll be able to select the workspace from the drop down list in your dashboard.

![ccfraud-workspace exists](./images/wallaroo-101/wallaroo-dashboard-ccfraud-workspace-exists.png)

Just for the sake of this tutorial, we'll use the SDK below to create our workspace , assign as our **current workspace**, then display all of the workspaces we have at the moment.  We'll also set up for our models and pipelines down the road, so we have one spot to change names to whatever fits your organization's standards best.

To allow this tutorial to be run multiple times or by multiple users in the same Wallaroo instance, a random 4 character prefix will be added to the workspace, pipeline, and model.

When we create our new workspace, we'll save it in the Python variable `workspace` so we can refer to it as needed.

```python
wl.list_pipelines()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th><th>created</th><th>last_updated</th><th>deployed</th><th>tags</th><th>versions</th><th>steps</th></tr><tr><td>housing-pipe</td><td>2023-17-May 21:26:56</td><td>2023-17-May 21:29:05</td><td>False</td><td></td><td>34e75a0c-01bd-4ca2-a6e8-ebdd25473aab, b7dbd380-e48c-487c-8f23-398a2ba558c3, 5ea6f182-5764-4377-9f83-d363e349ef32</td><td>preprocess</td></tr><tr><td>xgboost-regression-autoconvert-pipeline</td><td>2023-17-May 21:21:56</td><td>2023-17-May 21:21:59</td><td>False</td><td></td><td>f5337089-2756-469a-871a-1cb9e3416847, 324433ae-db9a-4d43-9563-ff76df59953d</td><td>xgb-regression-model</td></tr><tr><td>xgboost-classification-autoconvert-pipeline</td><td>2023-17-May 21:21:19</td><td>2023-17-May 21:21:22</td><td>False</td><td></td><td>5f7bb0cc-f60d-4cee-8425-c5e85331ae2f, bbe4dce4-f62a-4f4f-a45c-aebbfce23304</td><td>xgb-class-model</td></tr><tr><td>statsmodelpipeline</td><td>2023-17-May 21:19:52</td><td>2023-17-May 21:19:55</td><td>False</td><td></td><td>4af264e3-f427-4b02-b5ad-4f6690b0ee06, 5456dd2a-3167-4b3c-ad3a-85544292a230</td><td>bikedaymodel</td></tr><tr><td>isoletpipeline</td><td>2023-17-May 21:17:33</td><td>2023-17-May 21:17:44</td><td>False</td><td></td><td>c129b33c-cefc-4873-ad2c-d186fe2b8228, 145b768e-79f2-44fd-ab6b-14d675501b83</td><td>isolettest</td></tr><tr><td>externalkerasautoconvertpipeline</td><td>2023-17-May 21:13:27</td><td>2023-17-May 21:13:30</td><td>False</td><td></td><td>7be0dd01-ef82-4335-b60d-6f1cd5287e5b, 3948e0dc-d591-4ff5-a48f-b8d17195a806</td><td>externalsimple-sentiment-model</td></tr><tr><td>gcpsdkpipeline</td><td>2023-17-May 21:03:44</td><td>2023-17-May 21:03:49</td><td>False</td><td></td><td>6398cafc-50c4-49e3-9499-6025b7808245, 7c043d3c-c894-4ae9-9ec1-c35518130b90</td><td>gcpsdkmodel</td></tr><tr><td>databricksazuresdkpipeline</td><td>2023-17-May 21:02:55</td><td>2023-17-May 21:02:59</td><td>False</td><td></td><td>f125dc67-f690-4011-986a-8f6a9a23c48a, 8c4a15b4-2ef0-4da1-8e2d-38088fde8c56</td><td>ccfraudmodel</td></tr><tr><td>azuremlsdkpipeline</td><td>2023-17-May 21:01:46</td><td>2023-17-May 21:01:51</td><td>False</td><td></td><td>28a7a5aa-5359-4320-842b-bad84258f7e4, e011272d-c22c-4b2d-ab9f-b17c60099434</td><td>azuremlsdkmodel</td></tr><tr><td>copiedmodelpipeline</td><td>2023-17-May 20:54:01</td><td>2023-17-May 20:54:01</td><td>(unknown)</td><td></td><td>bcf5994f-1729-4036-a910-00b662946801</td><td></td></tr><tr><td>pipelinemodels</td><td>2023-17-May 20:52:06</td><td>2023-17-May 20:52:06</td><td>False</td><td></td><td>55f45c16-591e-4a16-8082-3ab6d843b484</td><td>apimodel</td></tr><tr><td>pipelinenomodel</td><td>2023-17-May 20:52:04</td><td>2023-17-May 20:52:04</td><td>(unknown)</td><td></td><td>a6dd2cee-58d6-4d24-9e25-f531dbbb95ad</td><td></td></tr><tr><td>sdkquickpipeline</td><td>2023-17-May 20:43:38</td><td>2023-17-May 20:46:02</td><td>False</td><td></td><td>961c909d-f5ae-472a-b8ae-1e6a00fbc36e, bf7c2146-ed14-430b-bf96-1e8b1047eb2e, 2bd5c838-f7cc-4f48-91ea-28a9ce0f7ed8, d72c468a-a0e2-4189-aa7a-4e27127a2f2b</td><td>sdkquickmodel</td></tr><tr><td>housepricepipe</td><td>2023-17-May 20:41:50</td><td>2023-17-May 20:41:50</td><td>False</td><td></td><td>4d9dfb3b-c9ae-402a-96fc-20ae0a2b2279, fc68f5f2-7bbf-435e-b434-e0c89c28c6a9</td><td>housepricemodel</td></tr></table>
{{</table>}}

```python
import string
import random

# make a random 4 character prefix
prefix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))
workspace_name = f'{prefix}ccfraudworkspace'
workspace_name = "qjjoccfraudworkspace"
pipeline_name = f'{prefix}ccfraudpipeline'
pipeline_name = "qjjoccfraudpipeline"
model_name = f'{prefix}ccfraudmodel'
model_name = "qjjoccfraudmodel"
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
        pipeline = wl.pipelines_by_name(name)[0]
    except EntityNotFoundError:
        pipeline = wl.build_pipeline(name)
    return pipeline
```

```python
workspace = get_workspace(workspace_name)

#wl.set_current_workspace(workspace)
```

```python
wl.list_workspaces()
```

{{<table "table table-striped table-bordered" >}}
<table>
    <tr>
        <th>Name</th>
        <th>Created At</th>
        <th>Users</th>
        <th>Models</th>
        <th>Pipelines</th>
    </tr>

<tr >
    <td>john.hummel@wallaroo.ai - Default Workspace</td>
    <td>2023-05-17 20:36:36</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>0</td>
    <td>0</td>
</tr>

<tr >
    <td>housepricedrift</td>
    <td>2023-05-17 20:41:50</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>sdkquickworkspace</td>
    <td>2023-05-17 20:43:36</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>apiworkspaces</td>
    <td>2023-05-17 20:50:36</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>2</td>
    <td>3</td>
</tr>

<tr >
    <td>azuremlsdkworkspace</td>
    <td>2023-05-17 21:01:45</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>databricksazuresdkworkspace</td>
    <td>2023-05-17 21:02:53</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>gcpsdkworkspace</td>
    <td>2023-05-17 21:03:43</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>testautoconversion</td>
    <td>2023-05-17 21:11:40</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>2</td>
    <td>0</td>
</tr>

<tr >
    <td>externalkerasautoconvertworkspace</td>
    <td>2023-05-17 21:13:26</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>isolettest</td>
    <td>2023-05-17 21:17:28</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>statsmodelworkspace</td>
    <td>2023-05-17 21:19:51</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>xgboost-classification-autoconvert-workspace</td>
    <td>2023-05-17 21:21:19</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>xgboost-regression-autoconvert-workspace</td>
    <td>2023-05-17 21:21:55</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>housepricing</td>
    <td>2023-05-17 21:26:51</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>3</td>
    <td>1</td>
</tr>

<tr >
    <td>qjjoccfraudworkspace</td>
    <td>2023-05-17 21:31:31</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>0</td>
    <td>0</td>
</tr>

</table>
{{</table>}}

Just to make sure, let's list our current workspace.  If everything is going right, it will show us we're in the `ccfraud-workspace` with the appropriate prefix.

```python
wl.set_current_workspace(workspace)
wl.get_current_workspace()
```

    {'name': 'qjjoccfraudworkspace', 'id': 18, 'archived': False, 'created_by': '028c8b48-c39b-4578-9110-0b5bdd3824da', 'created_at': '2023-05-17T21:31:31.410613+00:00', 'models': [], 'pipelines': []}

## Upload a model

Our workspace is created.  Let's upload our credit card fraud model to it.  This is the file name `ccfraud.onnx`, and we'll upload it as `ccfraudmodel`.  The credit card fraud model is trained to detect credit card fraud based on a 0 to 1 model:  The closer to 0 the less likely the transactions indicate fraud, while the closer to 1 the more likely the transactions indicate fraud.

Since we're already in our default workspace `ccfraudworkspace`, it'll be uploaded right to there.  Once that's done uploading, we'll list out all of the models currently deployed so we can see it included.

```python
ccfraud_model = wl.upload_model(model_name, model_file_name).configure()
```

We can verify that our model was uploaded by listing the models uploaded to our Wallaroo instance with the `list_models()` command.  Note that since we uploaded this model before, we now have different versions of it we can use for our testing.

```python
wl.list_models()
```

{{<table "table table-striped table-bordered" >}}
<table>
  <tr>
    <th>Name</th>
    <th># of Versions</th>
    <th>Owner ID</th>
    <th>Last Updated</th>
    <th>Created At</th>
  </tr>

  <tr>
    <td>qjjoccfraudmodel</td>
    <td>1</td>
    <td>""</td>
    <td>2023-05-17 21:32:04.686252+00:00</td>
    <td>2023-05-17 21:32:04.686252+00:00</td>
  </tr>

</table>
{{</table>}}

## Create a Pipeline

With our model uploaded, time to create our pipeline and deploy it so it can accept data and process it through our `ccfraudmodel`.  We'll call our pipeline `ccfraudpipeline`.

* **NOTE**:  Pipeline names must be unique.  If two pipelines are assigned the same name, the new pipeline is created as a new **version** of the pipeline.

```python
ccfraud_pipeline = get_pipeline(pipeline_name)
```

Now our pipeline is set.  Let's add a single **step** to it - in this case, our `ccfraud_model` that we uploaded to our workspace.

```python
ccfraud_pipeline.add_model_step(ccfraud_model)
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>qjjoccfraudpipeline</td></tr><tr><th>created</th> <td>2023-05-17 21:32:06.660305+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 21:32:06.660305+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>c0f8551d-cefe-49c8-8701-c2a307c0ad99</td></tr><tr><th>steps</th> <td></td></tr></table>
{{</table>}}

And now we can deploy our pipeline and assign resources to it.  This typically takes about 45 seconds once the command is issued.

```python
ccfraud_pipeline.deploy()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>qjjoccfraudpipeline</td></tr><tr><th>created</th> <td>2023-05-17 21:32:06.660305+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 21:32:08.181067+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>89b634d6-f538-4ac6-98a2-fbb9883fdeb6, c0f8551d-cefe-49c8-8701-c2a307c0ad99</td></tr><tr><th>steps</th> <td>qjjoccfraudmodel</td></tr></table>
{{</table>}}

We can see our new pipeline with the `status()` command.

```python
ccfraud_pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.3.145',
       'name': 'engine-6b5bd7dfcb-bk77c',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'qjjoccfraudpipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'qjjoccfraudmodel',
          'version': 'fb39df83-17f2-48c6-97ce-51583d39ca6e',
          'sha': 'bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.4.177',
       'name': 'engine-lb-584f54c899-z4zsq',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}

## Running Interfences

With our pipeline deployed, let's run a smoke test to make sure it's working right.  We'll run an inference through our pipeline from the variable `smoke_test` and see the results.  This should give us a result near 0 - not likely a fraudulent activity.

Wallaroo accepts the following inputs for inferences:

* [Apache Arrow tables](https://arrow.apache.org/) (Default):  Wallaroo highly encourages organizations to use Apache Arrow as their default inference input method for speed and accuracy.  This requires the Arrow table schema matches what the model expects.
* [Pandas DataFrame](https://pandas.pydata.org/): DataFrame inputs are highly useful for data scientists to recognize what data is being input into the pipeline before finalizing to Arrow format.

This first two examples will use a pandas DataFrame record to display a sample input.  The rest of the examples will use Arrow tables.

```python
smoke_test = pd.DataFrame.from_records([
    {
        "tensor":[
            1.0678324729,
            0.2177810266,
            -1.7115145262,
            0.682285721,
            1.0138553067,
            -0.4335000013,
            0.7395859437,
            -0.2882839595,
            -0.447262688,
            0.5146124988,
            0.3791316964,
            0.5190619748,
            -0.4904593222,
            1.1656456469,
            -0.9776307444,
            -0.6322198963,
            -0.6891477694,
            0.1783317857,
            0.1397992467,
            -0.3554220649,
            0.4394217877,
            1.4588397512,
            -0.3886829615,
            0.4353492889,
            1.7420053483,
            -0.4434654615,
            -0.1515747891,
            -0.2668451725,
            -1.4549617756
        ]
    }
])
result = ccfraud_pipeline.infer(smoke_test)
display(result)
```

{{<table "table table-striped table-bordered" >}}
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.dense_1</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-05-17 21:32:25.181</td>
      <td>[1.0678324729, 0.2177810266, -1.7115145262, 0.682285721, 1.0138553067, -0.4335000013, 0.7395859437, -0.2882839595, -0.447262688, 0.5146124988, 0.3791316964, 0.5190619748, -0.4904593222, 1.1656456469, -0.9776307444, -0.6322198963, -0.6891477694, 0.1783317857, 0.1397992467, -0.3554220649, 0.4394217877, 1.4588397512, -0.3886829615, 0.4353492889, 1.7420053483, -0.4434654615, -0.1515747891, -0.2668451725, -1.4549617756]</td>
      <td>[0.0014974177]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}

Looks good!  Time to run the real test on some real data.  Run another inference this time from the file `high_fraud.json` and let's see the results.  This should give us an output that indicates a high level of fraud - well over 90%.

```python
high_fraud = pd.DataFrame.from_records([
    {
        "tensor":[
            1.0678324729,
            18.1555563975,
            -1.6589551058,
            5.2111788045,
            2.3452470645,
            10.4670835778,
            5.0925820522,
            12.8295153637,
            4.9536770468,
            2.3934736228,
            23.912131818,
            1.759956831,
            0.8561037518,
            1.1656456469,
            0.5395988814,
            0.7784221343,
            6.7580610727,
            3.9274118477,
            12.4621782767,
            12.3075382165,
            13.7879519066,
            1.4588397512,
            3.6818346868,
            1.753914366,
            8.4843550037,
            14.6454097667,
            26.8523774363,
            2.7165292377,
            3.0611957069
        ]
    }
])

result = ccfraud_pipeline.infer(high_fraud)
display(result)
```

{{<table "table table-striped table-bordered" >}}
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.dense_1</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-05-17 21:32:25.595</td>
      <td>[1.0678324729, 18.1555563975, -1.6589551058, 5.2111788045, 2.3452470645, 10.4670835778, 5.0925820522, 12.8295153637, 4.9536770468, 2.3934736228, 23.912131818, 1.759956831, 0.8561037518, 1.1656456469, 0.5395988814, 0.7784221343, 6.7580610727, 3.9274118477, 12.4621782767, 12.3075382165, 13.7879519066, 1.4588397512, 3.6818346868, 1.753914366, 8.4843550037, 14.6454097667, 26.8523774363, 2.7165292377, 3.0611957069]</td>
      <td>[0.981199]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}

Now that we've tested our pipeline, let's run it with something larger.  We have two batch files - `cc_data_1k.arrow` that contains 1,000 credit card records to test for fraud.  The other is `cc_data_10k.arrow` which has 10,000 credit card records to test.

First let's run a batch result for `cc_data_10k.arrow` and see the results.  

With the inference result we'll output just the cases likely to be fraud.

```python
result = ccfraud_pipeline.infer_from_file('./data/cc_data_10k.arrow')

display(result)

# using pandas conversion, display only the results with > 0.75

list = [0.75]

outputs =  result.to_pandas()
# display(outputs)
filter = [elt[0] > 0.75 for elt in outputs['out.dense_1']]
outputs = outputs.loc[filter]
display(outputs)
```

    pyarrow.Table
    time: timestamp[ms]
    in.tensor: list<item: float> not null
      child 0, item: float
    out.dense_1: list<inner: float not null> not null
      child 0, inner: float not null
    check_failures: int8
    ----
    time: [[2023-05-17 21:32:26.542,2023-05-17 21:32:26.542,2023-05-17 21:32:26.542,2023-05-17 21:32:26.542,2023-05-17 21:32:26.542,...,2023-05-17 21:32:26.542,2023-05-17 21:32:26.542,2023-05-17 21:32:26.542,2023-05-17 21:32:26.542,2023-05-17 21:32:26.542]]
    in.tensor: [[[-1.0603298,2.3544967,-3.5638788,5.138735,-1.2308457,...,0.038412016,1.0993439,1.2603409,-0.14662448,-1.4463212],[-1.0603298,2.3544967,-3.5638788,5.138735,-1.2308457,...,0.038412016,1.0993439,1.2603409,-0.14662448,-1.4463212],...,[-2.1694233,-3.1647356,1.2038506,-0.2649221,0.0899006,...,1.8174038,-0.19327773,0.94089776,0.825025,1.6242892],[-0.12405868,0.73698884,1.0311689,0.59917533,0.11831961,...,-0.36567155,-0.87004745,0.41288367,0.49470216,-0.6710689]]]
    out.dense_1: [[[0.99300325],[0.99300325],...,[0.00024175644],[0.0010648072]]]
    check_failures: [[0,0,0,0,0,...,0,0,0,0,0]]

{{<table "table table-striped table-bordered" >}}
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.dense_1</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-05-17 21:32:26.542</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>[0.99300325]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-05-17 21:32:26.542</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>[0.99300325]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-05-17 21:32:26.542</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>[0.99300325]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-05-17 21:32:26.542</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>[0.99300325]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>161</th>
      <td>2023-05-17 21:32:26.542</td>
      <td>[-9.716793, 9.174981, -14.450761, 8.653825, -11.039951, 0.6602411, -22.825525, -9.919395, -8.064324, -16.737926, 4.852197, -12.563343, -1.0762653, -7.524591, -3.2938414, -9.62102, -15.6501045, -7.089741, 1.7687134, 5.044906, -11.365625, 4.5987034, 4.4777045, 0.31702697, -2.2731977, 0.07944675, -10.052058, -2.024108, -1.0611985]</td>
      <td>[1.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>941</th>
      <td>2023-05-17 21:32:26.542</td>
      <td>[-0.50492376, 1.9348029, -3.4217603, 2.2165704, -0.6545315, -1.9004827, -1.6786858, 0.5380051, -2.7229102, -5.265194, 3.504164, -5.4661765, 0.68954825, -8.725291, 2.0267954, -5.4717045, -4.9123807, -1.6131229, 3.8021576, 1.3881834, 1.0676425, 0.28200775, -0.30759808, -0.48498034, 0.9507336, 1.5118006, 1.6385275, 1.072455, 0.7959132]</td>
      <td>[0.9873102]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1445</th>
      <td>2023-05-17 21:32:26.542</td>
      <td>[-7.615594, 4.659706, -12.057331, 7.975307, -5.1068773, -1.6116138, -12.146941, -0.5952333, -6.4605103, -12.535655, 10.017626, -14.839381, 0.34900802, -14.953928, -0.3901092, -9.342014, -14.285043, -5.758632, 0.7512068, 1.4632998, -3.3777077, 0.9950705, -0.5855211, -1.6528498, 1.9089833, 1.6860862, 5.5044003, -3.703297, -1.4715525]</td>
      <td>[1.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2092</th>
      <td>2023-05-17 21:32:26.542</td>
      <td>[-14.115489, 9.905631, -18.67885, 4.602589, -15.404288, -3.7169847, -15.887272, 15.616176, -3.2883947, -7.0224414, 4.086536, -5.7809114, 1.2251061, -5.4301147, -0.14021407, -6.0200763, -12.957546, -5.545689, 0.86074656, 2.2463796, 2.492611, -2.9649208, -2.265674, 0.27490455, 3.9263225, -0.43438172, 3.1642237, 1.2085277, 0.8223642]</td>
      <td>[0.99999]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2220</th>
      <td>2023-05-17 21:32:26.542</td>
      <td>[-0.1098309, 2.5842443, -3.5887418, 4.63558, 1.1825614, -1.2139517, -0.7632139, 0.6071841, -3.7244265, -3.501917, 4.3637576, -4.612757, -0.44275254, -10.346612, 0.66243565, -0.33048683, 1.5961986, 2.5439718, 0.8787973, 0.7406088, 0.34268215, -0.68495077, -0.48357907, -1.9404846, -0.059520483, 1.1553137, 0.9918434, 0.7067319, -1.6016251]</td>
      <td>[0.91080534]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4135</th>
      <td>2023-05-17 21:32:26.542</td>
      <td>[-0.547029, 2.2944348, -4.149202, 2.8648357, -0.31232587, -1.5427867, -2.1489344, 0.9471863, -2.663241, -4.2572775, 2.1116028, -6.2264414, -1.1307784, -6.9296007, 1.0049651, -5.876498, -5.6855297, -1.5800936, 3.567338, 0.5962099, 1.6361043, 1.8584082, -0.08202618, 0.46620172, -2.234368, -0.18116793, 1.744976, 2.1414309, -1.6081295]</td>
      <td>[0.98877275]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4236</th>
      <td>2023-05-17 21:32:26.542</td>
      <td>[-3.135635, -1.483817, -3.0833669, 1.6626456, -0.59695035, -0.30199608, -3.316563, 1.869609, -1.8006078, -4.5662026, 2.8778172, -4.0887237, -0.43401834, -3.5816982, 0.45171788, -5.725131, -8.982029, -4.0279546, 0.89264476, 0.24721873, 1.8289508, 1.6895254, -2.5555577, -2.4714024, -0.4500012, 0.23333028, 2.2119386, -2.041805, 1.1568314]</td>
      <td>[0.95601666]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5658</th>
      <td>2023-05-17 21:32:26.542</td>
      <td>[-5.4078765, 3.9039962, -8.98522, 5.128742, -7.373224, -2.946234, -11.033238, 5.914019, -5.669241, -12.041053, 6.950792, -12.488795, 1.2236942, -14.178565, 1.6514667, -12.47019, -22.350504, -8.928755, 4.54775, -0.11478994, 3.130207, -0.70128506, -0.40275285, 0.7511918, -0.1856308, 0.92282087, 0.146656, -1.3761806, 0.42997098]</td>
      <td>[1.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6768</th>
      <td>2023-05-17 21:32:26.542</td>
      <td>[-16.900557, 11.7940855, -21.349983, 4.746453, -17.54182, -3.415758, -19.897173, 13.8569145, -3.570626, -7.388376, 3.0761156, -4.0583425, 1.2901028, -2.7997534, -0.4298746, -4.777225, -11.371295, -5.2725616, 0.0964799, 4.2148075, -0.8343371, -2.3663573, -1.6571938, 0.2110055, 4.438088, -0.49057993, 2.342008, 1.4479793, -1.4715525]</td>
      <td>[0.9999745]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6780</th>
      <td>2023-05-17 21:32:26.542</td>
      <td>[-0.74893713, 1.3893062, -3.7477517, 2.4144504, -0.11061429, -1.0737498, -3.1504633, 1.2081385, -1.332872, -4.604276, 4.438548, -7.687688, 1.1683422, -5.3296027, -0.19838685, -5.294243, -5.4928794, -1.3254275, 4.387228, 0.68643385, 0.87228596, -0.1154091, -0.8364338, -0.61202216, 0.10518055, 2.2618086, 1.1435078, -0.32623357, -1.6081295]</td>
      <td>[0.9852645]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7133</th>
      <td>2023-05-17 21:32:26.542</td>
      <td>[-7.5131927, 6.507386, -12.439463, 5.7453, -9.513038, -1.4236209, -17.402607, -3.0903268, -5.378041, -15.169325, 5.7585907, -13.448207, -0.45244268, -8.495097, -2.2323692, -11.429063, -19.578058, -8.367617, 1.8869618, 2.1813896, -4.799091, 2.4388566, 2.9503248, 0.6293566, -2.6906652, -2.1116931, -6.4196434, -1.4523355, -1.4715525]</td>
      <td>[1.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7566</th>
      <td>2023-05-17 21:32:26.542</td>
      <td>[-2.1804514, 1.0243497, -4.3890443, 3.4924, -3.7609894, 0.023624033, -2.7677023, 1.1786921, -2.9450424, -6.8823, 6.1294384, -9.564066, -1.6273017, -10.940607, 0.3062539, -8.854589, -15.382658, -5.419305, 3.2210033, -0.7381137, 0.9632334, 0.6612066, 2.1337948, -0.90536207, 0.7498649, -0.019404415, 5.5950212, 0.26602694, 1.7534728]</td>
      <td>[0.9999705]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7911</th>
      <td>2023-05-17 21:32:26.542</td>
      <td>[-1.594454, 1.8545462, -2.6311765, 2.759316, -2.6988854, -0.08155677, -3.8566258, -0.04912437, -1.9640644, -4.2058415, 3.391933, -6.471933, -0.9877536, -6.188904, 1.2249585, -8.652863, -11.170872, -6.134417, 2.5400054, -0.29327056, 3.591464, 0.3057127, -0.052313827, 0.06196331, -0.82863224, -0.2595842, 1.0207018, 0.019899422, 1.0935433]</td>
      <td>[0.9980203]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8921</th>
      <td>2023-05-17 21:32:26.542</td>
      <td>[-0.21756083, 1.786712, -3.4240367, 2.7769134, -1.420116, -2.1018193, -3.4615245, 0.7367844, -2.3844852, -6.3140697, 4.382665, -8.348951, -1.6409378, -10.611383, 1.1813216, -6.251184, -10.577264, -3.5184007, 0.7997489, 0.97915924, 1.081642, -0.7852368, -0.4761941, -0.10635195, 2.066527, -0.4103488, 2.8288178, 1.9340333, -1.4715525]</td>
      <td>[0.99950194]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9244</th>
      <td>2023-05-17 21:32:26.542</td>
      <td>[-3.314442, 2.4431305, -6.1724143, 3.6737356, -3.81542, -1.5950849, -4.8292923, 2.9850774, -4.22416, -7.5519834, 6.1932964, -8.59886, 0.25443414, -11.834097, -0.39583337, -6.015362, -13.532762, -4.226845, 1.1153877, 0.17989528, 1.3166595, -0.64433384, 0.2305495, -0.5776498, 0.7609739, 2.2197483, 4.01189, -1.2347667, 1.2847253]</td>
      <td>[0.9999876]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10176</th>
      <td>2023-05-17 21:32:26.542</td>
      <td>[-5.0815525, 3.9294617, -8.4077635, 6.373701, -7.391173, -2.1574461, -10.345097, 5.5896044, -6.3736906, -11.330594, 6.618754, -12.93748, 1.1884484, -13.9628935, 1.0340953, -12.278127, -23.333889, -8.886669, 3.5720036, -0.3243157, 3.4229393, 0.493529, 0.08469851, 0.791218, 0.30968663, 0.6811129, 0.39306796, -1.5204874, 0.9061435]</td>
      <td>[1.0]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}

We can view the inputs either through the `in.tensor` column from our DataFrame for Arrow enabled environments, or with the InferenceResult object through the `input_data()` for non-Arrow enabled environments.  We'll display just the first row in either case.

## Batch Deployment through a Pipeline Deployment URL

This next step requires some manual use.  We're going to have `ccfraud_pipeline` display its deployment url - this allows us to submit data through a HTTP interface and get the results back.

First we'll request the url with the `_deployment._url()` method.

* **IMPORTANT NOTE**:  The `_deployment._url()` method will return an **internal** URL when using Python commands from within the Wallaroo instance - for example, the Wallaroo JupyterHub service.  When connecting via an external connection, `_deployment._url()` returns an **external** URL.
  * External URL connections requires [the authentication be included in the HTTP request](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/), and [Model Endpoints](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-model-endpoints-guide/) are enabled in the Wallaroo configuration options.

The API connection details can be retrieved through the Wallaroo client `auth.auth_header()` command.  This will display the connection URL, bearer token, and other information.  The bearer token is available for one hour before it expires.

For this example, the API connection details will be retrieved, then used to submit an inference request through the external inference URL retrieved earlier.

The `deploy_url` variable will be used to access the pipeline inference URL, Content-Type and Accept parameters will set the submitted values first as a DataFrame record, then the second will use the Python `requests` library to submit the inference as an Apache Arrow file, and the received values as a Pandas DataFrame.

```python

deploy_url = ccfraud_pipeline._deployment._url()

headers = wl.auth.auth_header()
# dataFile="./data/cc_data_1k.arrow"
dataFile="./data/cc_data_10k.df.json"
# set Content-Type type
# headers['Content-Type']='application/vnd.apache.arrow.file'
headers['Content-Type']='application/json; format=pandas-records'
headers['Accept']='application/json; format=pandas-records'
# contentType="application/vnd.apache.arrow.file"

# !curl -X POST {deploy_url} -H "Authorization:{headers['Authorization']}" -H "Content-Type:{headers['Content-Type']}" -H "Accept:{headers['Accept']}" --data @{dataFile} > curl_response.df
```

```python
!curl -X POST {deploy_url} -H "Authorization:{headers['Authorization']}" -H "Content-Type:{headers['Content-Type']}" -H "Accept:{headers['Accept']}" --data @{dataFile} > curl_response.df
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100 14.5M  100 7196k  100 7743k  1858k  1999k  0:00:03  0:00:03 --:--:-- 3861k

```python
cc_data_from_file =  pd.read_json('./curl_response.df', orient="records")
display(cc_data_from_file.head(5))
```

{{<table "table table-striped table-bordered" >}}
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in</th>
      <th>out</th>
      <th>check_failures</th>
      <th>metadata</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1684359150001</td>
      <td>{'tensor': [-1.0603297501, 2.3544967095000002, -3.5638788326, 5.1387348926, -1.2308457019, -0.7687824608, -3.5881228109, 1.8880837663, -3.2789674274, -3.9563254554, 4.0993439118, -5.6539176395, -0.8775733373, -9.131571192, -0.6093537873, -3.7480276773, -5.0309125017, -0.8748149526000001, 1.9870535692, 0.7005485718000001, 0.9204422758, -0.1041491809, 0.3229564351, -0.7418141657, 0.0384120159, 1.0993439146, 1.2603409756000001, -0.1466244739, -1.4463212439]}</td>
      <td>{'dense_1': [0.99300325]}</td>
      <td>[]</td>
      <td>{'last_model': '{"model_name":"qjjoccfraudmodel","model_sha":"bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507"}', 'pipeline_version': '89b634d6-f538-4ac6-98a2-fbb9883fdeb6', 'elapsed': [48043619, 3723948]}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1684359150001</td>
      <td>{'tensor': [-1.0603297501, 2.3544967095000002, -3.5638788326, 5.1387348926, -1.2308457019, -0.7687824608, -3.5881228109, 1.8880837663, -3.2789674274, -3.9563254554, 4.0993439118, -5.6539176395, -0.8775733373, -9.131571192, -0.6093537873, -3.7480276773, -5.0309125017, -0.8748149526000001, 1.9870535692, 0.7005485718000001, 0.9204422758, -0.1041491809, 0.3229564351, -0.7418141657, 0.0384120159, 1.0993439146, 1.2603409756000001, -0.1466244739, -1.4463212439]}</td>
      <td>{'dense_1': [0.99300325]}</td>
      <td>[]</td>
      <td>{'last_model': '{"model_name":"qjjoccfraudmodel","model_sha":"bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507"}', 'pipeline_version': '89b634d6-f538-4ac6-98a2-fbb9883fdeb6', 'elapsed': [48043619, 3723948]}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1684359150001</td>
      <td>{'tensor': [-1.0603297501, 2.3544967095000002, -3.5638788326, 5.1387348926, -1.2308457019, -0.7687824608, -3.5881228109, 1.8880837663, -3.2789674274, -3.9563254554, 4.0993439118, -5.6539176395, -0.8775733373, -9.131571192, -0.6093537873, -3.7480276773, -5.0309125017, -0.8748149526000001, 1.9870535692, 0.7005485718000001, 0.9204422758, -0.1041491809, 0.3229564351, -0.7418141657, 0.0384120159, 1.0993439146, 1.2603409756000001, -0.1466244739, -1.4463212439]}</td>
      <td>{'dense_1': [0.99300325]}</td>
      <td>[]</td>
      <td>{'last_model': '{"model_name":"qjjoccfraudmodel","model_sha":"bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507"}', 'pipeline_version': '89b634d6-f538-4ac6-98a2-fbb9883fdeb6', 'elapsed': [48043619, 3723948]}</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1684359150001</td>
      <td>{'tensor': [-1.0603297501, 2.3544967095000002, -3.5638788326, 5.1387348926, -1.2308457019, -0.7687824608, -3.5881228109, 1.8880837663, -3.2789674274, -3.9563254554, 4.0993439118, -5.6539176395, -0.8775733373, -9.131571192, -0.6093537873, -3.7480276773, -5.0309125017, -0.8748149526000001, 1.9870535692, 0.7005485718000001, 0.9204422758, -0.1041491809, 0.3229564351, -0.7418141657, 0.0384120159, 1.0993439146, 1.2603409756000001, -0.1466244739, -1.4463212439]}</td>
      <td>{'dense_1': [0.99300325]}</td>
      <td>[]</td>
      <td>{'last_model': '{"model_name":"qjjoccfraudmodel","model_sha":"bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507"}', 'pipeline_version': '89b634d6-f538-4ac6-98a2-fbb9883fdeb6', 'elapsed': [48043619, 3723948]}</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1684359150001</td>
      <td>{'tensor': [0.5817662108, 0.09788155100000001, 0.1546819424, 0.4754101949, -0.19788623060000002, -0.45043448540000003, 0.016654044700000002, -0.0256070551, 0.0920561602, -0.2783917153, 0.059329944100000004, -0.0196585416, -0.4225083157, -0.12175388770000001, 1.5473094894, 0.2391622864, 0.3553974881, -0.7685165301, -0.7000849355000001, -0.1190043285, -0.3450517133, -1.1065114108, 0.2523411195, 0.0209441826, 0.2199267436, 0.2540689265, -0.0450225094, 0.10867738980000001, 0.2547179311]}</td>
      <td>{'dense_1': [0.0010916889]}</td>
      <td>[]</td>
      <td>{'last_model': '{"model_name":"qjjoccfraudmodel","model_sha":"bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507"}', 'pipeline_version': '89b634d6-f538-4ac6-98a2-fbb9883fdeb6', 'elapsed': [48043619, 3723948]}</td>
    </tr>
  </tbody>
</table>
{{</table>}}

```python
import requests

# Again with Arrow and requests

# Retrieve the token
headers = wl.auth.auth_header()

# set Content-Type type
headers['Content-Type']='application/vnd.apache.arrow.file'

# set accept as pandas-records
headers['Accept']="application/vnd.apache.arrow.file"

# Submit arrow file
dataFile="./data/cc_data_10k.arrow"

data = open(dataFile,'rb').read()

response = requests.post(
                    deploy_url, 
                    headers=headers, 
                    data=data, 
                    verify=True
                )

# Arrow table is retrieved 
with pa.ipc.open_file(response.content) as reader:
    arrow_table = reader.read_all()

# convert to Polars DataFrame and display the first 5 rows
display(arrow_table.to_pandas().head(5).loc[:,["time", "out"]])
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
      <td>1684359221866</td>
      <td>{'dense_1': [0.99300325]}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1684359221866</td>
      <td>{'dense_1': [0.99300325]}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1684359221866</td>
      <td>{'dense_1': [0.99300325]}</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1684359221866</td>
      <td>{'dense_1': [0.99300325]}</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1684359221866</td>
      <td>{'dense_1': [0.0010916889]}</td>
    </tr>
  </tbody>
</table>
{{</table>}}

With the arrow file returned, we'll show the first five rows for comparison.

With our work in the pipeline done, we'll undeploy it to get back our resources from the Kubernetes cluster.  If we keep the same settings we can redeploy the pipeline with the same configuration in the future.

```python
ccfraud_pipeline.logs()
```

    Warning: There are more logs available. Please set a larger limit or request a file using export_logs.

{{<table "table table-striped table-bordered" >}}
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.dense_1</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-05-17 21:33:41.866</td>
      <td>[-0.12405868, 0.73698884, 1.0311689, 0.59917533, 0.11831961, -0.47327134, 0.67207795, -0.16058543, -0.77808416, -0.2683532, -0.22364272, 0.64049035, 1.5038015, 0.004768592, 1.0683576, -0.3107394, -0.3427847, -0.078522496, 0.5455177, 0.1297957, 0.10491481, 0.4318976, -0.25777233, 0.701442, -0.36567155, -0.87004745, 0.41288367, 0.49470216, -0.6710689]</td>
      <td>[0.0010648072]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-05-17 21:33:41.866</td>
      <td>[-2.1694233, -3.1647356, 1.2038506, -0.2649221, 0.0899006, -0.18928011, -0.3495527, 0.17693162, 0.70965016, 0.19469678, -1.4288249, -1.5722991, -1.8213876, -1.2968907, -0.95062584, 0.8917047, 0.751387, -1.5475872, 0.3787144, 0.7525444, -0.03103788, 0.5858543, 2.6022773, -0.5311059, 1.8174038, -0.19327773, 0.94089776, 0.825025, 1.6242892]</td>
      <td>[0.00024175644]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-05-17 21:33:41.866</td>
      <td>[-0.24798988, 0.40499672, 0.49408177, -0.37252557, -0.412961, -0.38151076, -0.042232547, 0.293104, -2.1088455, 0.49807099, 0.8488427, -0.9078823, -0.89734304, 0.8601335, 0.6696, 0.48890257, 1.0623674, -0.5090369, 3.616667, 0.29580164, 0.43410888, 0.8741068, -0.6503351, 0.034938015, 0.96057385, 0.43238926, -0.1975937, -0.04551184, -0.12277038]</td>
      <td>[0.00150159]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-05-17 21:33:41.866</td>
      <td>[-0.2260837, 0.12802614, -0.8732004, -2.089788, 1.8722432, 2.05272, 0.09746246, 0.49692878, -1.0799059, 0.80176145, -0.26333216, -1.0224636, -0.056668393, -0.060779527, -0.20089072, 1.2798951, -0.55411917, -1.270442, 0.41811302, 0.2239133, 0.3109173, 1.0051724, 0.07736663, 1.7022146, -0.93862903, -0.99493766, -0.68271357, -0.71875495, -1.4715525]</td>
      <td>[0.00037947297]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-05-17 21:33:41.866</td>
      <td>[-0.90164274, -0.50116056, 1.2045985, 0.4078851, 0.2981652, -0.26469636, 0.4460249, 0.16928293, -0.15559517, -0.7641287, -0.8956279, -0.6098771, -0.87228906, 0.158441, 0.7461226, 0.43037805, -0.7037308, 0.7927367, -1.111509, 0.83980113, 0.6249728, 0.7301589, 0.5632024, 1.7966471, 1.5083653, -1.0206859, -0.11091206, 0.37982145, 1.2463697]</td>
      <td>[0.0001988411]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>2023-05-17 21:33:41.866</td>
      <td>[-0.1093998, -0.031678658, 0.9885652, -0.6860291, -0.6046543, 0.6653355, -0.6293921, -1.1772763, 1.4608942, -1.1322296, -1.9279546, 1.4049336, 1.2282782, -1.4884002, -3.115575, 0.41227773, -0.47484678, -0.9897973, -1.1200552, -0.66070575, 1.6864017, -1.4189101, -0.70692146, -0.5732528, 1.981664, 1.7516811, 0.28014255, 0.30193287, 0.80388844]</td>
      <td>[0.00020942092]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>96</th>
      <td>2023-05-17 21:33:41.866</td>
      <td>[0.44973943, -0.35288164, 0.5224735, 0.910402, -0.72067416, -0.050270014, -0.3414758, 0.10386056, 0.6459101, -0.019469967, -1.1449441, -0.7871593, -1.2093763, 0.26724637, 1.7972686, 1.0460316, -0.9273358, 0.91270906, -0.69702125, 0.19257616, 0.23213731, 0.08892933, -0.34931126, -0.31643763, 0.583289, -0.6749049, 0.0472516, 0.20378655, 1.0983771]</td>
      <td>[0.00031492114]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>97</th>
      <td>2023-05-17 21:33:41.866</td>
      <td>[0.82174337, -0.50793207, -1.358988, 0.37136176, 0.19260807, -0.60984015, 0.63114387, -0.31723085, 0.34576818, 0.015056487, -0.9967559, -0.037889328, -0.68294096, 0.6202497, -0.32679954, -0.6409717, -0.0055463314, -0.8609782, 0.119142644, 0.3092495, 0.1808159, -0.019580727, -0.20877448, 1.1818781, 0.2868476, 1.1510639, -0.37393016, -0.094152406, 1.2670404]</td>
      <td>[0.00081187487]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>98</th>
      <td>2023-05-17 21:33:41.866</td>
      <td>[1.0252348, 0.37717652, -1.4182774, 0.7057443, 0.36312255, -1.3660663, 0.17341496, -0.3454704, 1.7643102, -1.3501495, 0.9257918, -3.374893, 0.3617876, -0.8583969, 0.5060873, 0.8873245, 2.925866, 2.0265691, -1.1160102, -0.36432365, -0.0936597, 0.25772303, -0.02305712, -0.45073295, 0.37329674, -0.2838264, -0.0411118, 0.006249274, -1.4715525]</td>
      <td>[0.001860708]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>99</th>
      <td>2023-05-17 21:33:41.866</td>
      <td>[-0.36498702, 0.11005125, 0.7734325, 1.0163404, -0.38190573, 0.41608095, 1.4093872, -0.12511922, -0.14253987, -0.093657725, -0.6349157, -0.41843006, -0.91369456, -0.0038188277, 0.3744724, -1.3620936, 0.6263981, -0.57914644, 0.82675296, 0.9850866, 0.08680151, 0.28205827, 0.7979858, 0.065717764, -0.052254554, -0.53277296, 0.40100586, 0.0075293095, 1.3380127]</td>
      <td>[0.00064843893]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}
<p>100 rows × 4 columns</p>

```python
ccfraud_pipeline.undeploy()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>qjjoccfraudpipeline</td></tr><tr><th>created</th> <td>2023-05-17 21:32:06.660305+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 21:32:08.181067+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>89b634d6-f538-4ac6-98a2-fbb9883fdeb6, c0f8551d-cefe-49c8-8701-c2a307c0ad99</td></tr><tr><th>steps</th> <td>qjjoccfraudmodel</td></tr></table>
{{</table>}}

And there we have it!  Feel free to use this as a template for other models, inferences and pipelines that you want to deploy with Wallaroo!
