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
import json

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
# Login through local Wallaroo instance

wl = wallaroo.Client()

# SSO login through keycloak

# wallarooPrefix = "YOUR PREFIX"
# wallarooSuffix = "YOUR PREFIX"

# wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}.api.{wallarooSuffix}", 
#                     auth_endpoint=f"https://{wallarooPrefix}.keycloak.{wallarooSuffix}", 
#                     auth_type="sso")
```

### Arrow Support

As of the 2023.1 release, Wallaroo provides support for DataFrame and Arrow for inference inputs.  This tutorial allows users to adjust their experience based on whether they have enabled Arrow support in their Wallaroo instance or not.

If Arrow support has been enabled, `arrowEnabled=True`. If disabled or you're not sure, set it to `arrowEnabled=False`

The examples below will be shown in an arrow enabled environment.


```python
import os
# Only set the below to make the OS environment ARROW_ENABLED to TRUE.  Otherwise, leave as is.
# os.environ["ARROW_ENABLED"]="True"

if "ARROW_ENABLED" not in os.environ or os.environ["ARROW_ENABLED"].casefold() == "False".casefold():
    arrowEnabled = False
else:
    arrowEnabled = True
print(arrowEnabled)
```

    True


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
import string
import random

# make a random 4 character prefix
prefix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))
workspace_name = f'{prefix}ccfraudworkspace'
pipeline_name = f'{prefix}ccfraudpipeline'
model_name = f'{prefix}ccfraudmodel'
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
```




    {'name': 'uupfccfraudworkspace', 'id': 114, 'archived': False, 'created_by': '138bd7e6-4dc8-4dc1-a760-c9e721ef3c37', 'created_at': '2023-03-01T21:59:42.738409+00:00', 'models': [], 'pipelines': []}




```python
wl.list_workspaces()
```





<table>
    <tr>
        <th>Name</th>
        <th>Created At</th>
        <th>Users</th>
        <th>Models</th>
        <th>Pipelines</th>
    </tr>

<tr >
    <td>john.hansarick@wallaroo.ai - Default Workspace</td>
    <td>2023-02-17 20:36:12</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>2</td>
    <td>1</td>
</tr>


<tr >
    <td>testautoconversion</td>
    <td>2023-02-21 17:02:22</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>2</td>
    <td>0</td>
</tr>


<tr >
    <td>kerasautoconvertworkspace</td>
    <td>2023-02-21 18:09:28</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>


<tr >
    <td>externalkerasautoconvertworkspace</td>
    <td>2023-02-21 18:16:14</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>


<tr >
    <td>ccfraudcomparisondemo</td>
    <td>2023-02-21 18:31:10</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>3</td>
    <td>1</td>
</tr>


<tr >
    <td>isolettest</td>
    <td>2023-02-21 21:24:33</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>


<tr >
    <td>bikedayevalworkspace</td>
    <td>2023-02-22 16:42:58</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>


<tr >
    <td>xgboost-classification-autoconvert-workspace</td>
    <td>2023-02-22 17:28:52</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>


<tr >
    <td>xgboost-regression-autoconvert-workspace</td>
    <td>2023-02-22 17:36:30</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>


<tr >
    <td>housepricing</td>
    <td>2023-02-22 18:28:40</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>3</td>
    <td>1</td>
</tr>


<tr >
    <td>sdkquickworkspace</td>
    <td>2023-02-22 21:25:41</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>


<tr >
    <td>jchdemandcurveworkspace</td>
    <td>2023-02-22 22:23:21</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>3</td>
    <td>1</td>
</tr>


<tr >
    <td>jchdemandcurveworkspace2</td>
    <td>2023-02-22 22:33:41</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>3</td>
    <td>1</td>
</tr>


<tr >
    <td>demandcurveworkspace</td>
    <td>2023-02-23 15:14:32</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>3</td>
    <td>1</td>
</tr>


<tr >
    <td>yqecccfraudworkspace</td>
    <td>2023-02-23 16:00:59</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>


<tr >
    <td>mobilenetworkspace2</td>
    <td>2023-02-23 18:03:12</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>


<tr >
    <td>mobilenetworkspacetest</td>
    <td>2023-02-23 18:12:45</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>


<tr >
    <td>mlflowstatsmodelworkspace</td>
    <td>2023-02-23 23:14:12</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>2</td>
    <td>1</td>
</tr>


<tr >
    <td>statsmodelworkspace</td>
    <td>2023-02-24 17:17:13</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>2</td>
    <td>1</td>
</tr>


<tr >
    <td>wjtxedgeworkspaceexample</td>
    <td>2023-02-27 17:46:51</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>


<tr >
    <td>xtwjccfraudworkspace</td>
    <td>2023-02-28 19:14:51</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>


<tr >
    <td>ccfraudcomparisondemo2a</td>
    <td>2023-02-28 20:01:40</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>3</td>
    <td>1</td>
</tr>


<tr >
    <td>ccfraudcomparisondemo3</td>
    <td>2023-02-28 20:07:27</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>3</td>
    <td>1</td>
</tr>


<tr >
    <td>ccfraudcomparisondemo4</td>
    <td>2023-02-28 20:21:21</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>3</td>
    <td>1</td>
</tr>


<tr >
    <td>ccfraudcomparisondemo5</td>
    <td>2023-02-28 20:23:24</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>3</td>
    <td>1</td>
</tr>


<tr >
    <td>ccfraudcomparisondemo6</td>
    <td>2023-02-28 20:25:57</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>3</td>
    <td>1</td>
</tr>


<tr >
    <td>anomalyexampletest</td>
    <td>2023-02-28 20:37:58</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>


<tr >
    <td>uupfccfraudworkspace</td>
    <td>2023-03-01 21:59:42</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>0</td>
    <td>0</td>
</tr>

</table>




Just to make sure, let's list our current workspace.  If everything is going right, it will show us we're in the `ccfraud-workspace` with the appropriate prefix.


```python
wl.set_current_workspace(workspace)
wl.get_current_workspace()
```




    {'name': 'uupfccfraudworkspace', 'id': 114, 'archived': False, 'created_by': '138bd7e6-4dc8-4dc1-a760-c9e721ef3c37', 'created_at': '2023-03-01T21:59:42.738409+00:00', 'models': [], 'pipelines': []}



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





<table>
  <tr>
    <th>Name</th>
    <th># of Versions</th>
    <th>Owner ID</th>
    <th>Last Updated</th>
    <th>Created At</th>
  </tr>

  <tr>
    <td>uupfccfraudmodel</td>
    <td>1</td>
    <td>""</td>
    <td>2023-03-01 22:00:36.294590+00:00</td>
    <td>2023-03-01 22:00:36.294590+00:00</td>
  </tr>

</table>




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




<table><tr><th>name</th> <td>uupfccfraudpipeline</td></tr><tr><th>created</th> <td>2023-03-01 22:00:37.919002+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-01 22:00:37.919002+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>b78f3d0a-ea52-47a6-959d-d4aa0b60c7b1</td></tr><tr><th>steps</th> <td></td></tr></table>




```python
for d in wl.list_deployments(): d.undeploy()
```

And now we can deploy our pipeline and assign resources to it.  This typically takes about 45 seconds once the command is issued.


```python
ccfraud_pipeline.deploy()
```




<table><tr><th>name</th> <td>uupfccfraudpipeline</td></tr><tr><th>created</th> <td>2023-03-01 22:00:37.919002+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-01 22:04:51.394606+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>6b63ccf6-870a-4299-8213-ec3b4538ce0d, 70349464-78ca-4699-ae31-1b6322e44232, b78f3d0a-ea52-47a6-959d-d4aa0b60c7b1</td></tr><tr><th>steps</th> <td>uupfccfraudmodel</td></tr></table>



We can see our new pipeline with the `status()` command.


```python
ccfraud_pipeline.status()
```




    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.48.1.9',
       'name': 'engine-774f566bcb-czqzn',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'uupfccfraudpipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'uupfccfraudmodel',
          'version': '8e68f30d-3fbe-44f6-8468-4d3e54c93e36',
          'sha': 'bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.48.1.10',
       'name': 'engine-lb-86bc6bd77b-bl7s9',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}



## Running Interfences

With our pipeline deployed, let's run a smoke test to make sure it's working right.  We'll run an inference through our pipeline from the file `smoke_test.json` and see the results.  This should give us a result near 0 - not likely a fraudulent activity.


```python
if arrowEnabled is True:
    result = ccfraud_pipeline.infer_from_file('./data/smoke_test.df.json')
else:
    result = ccfraud_pipeline.infer_from_file('./data/smoke_test.json')
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
<table border="1" class="dataframe">
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
      <td>2023-03-01 22:05:19.842</td>
      <td>[1.0678324729, 0.2177810266, -1.7115145262, 0.682285721, 1.0138553067, -0.4335000013, 0.7395859437, -0.2882839595, -0.447262688, 0.5146124988, 0.3791316964, 0.5190619748, -0.4904593222, 1.1656456469, -0.9776307444, -0.6322198963, -0.6891477694, 0.1783317857, 0.1397992467, -0.3554220649, 0.4394217877, 1.4588397512, -0.3886829615, 0.4353492889, 1.7420053483, -0.4434654615, -0.1515747891, -0.2668451725, -1.4549617756]</td>
      <td>[0.0014974177]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


Looks good!  Time to run the real test on some real data.  Run another inference this time from the file `high_fraud.json` and let's see the results.  This should give us an output that indicates a high level of fraud - well over 90%.


```python
if arrowEnabled is True:
    result = ccfraud_pipeline.infer_from_file('./data/high_fraud.df.json')
else:
    result = ccfraud_pipeline.infer_from_file('./data/high_fraud.json')
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
<table border="1" class="dataframe">
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
      <td>2023-03-01 22:05:20.986</td>
      <td>[1.0678324729, 18.1555563975, -1.6589551058, 5.2111788045, 2.3452470645, 10.4670835778, 5.0925820522, 12.8295153637, 4.9536770468, 2.3934736228, 23.912131818, 1.759956831, 0.8561037518, 1.1656456469, 0.5395988814, 0.7784221343, 6.7580610727, 3.9274118477, 12.4621782767, 12.3075382165, 13.7879519066, 1.4588397512, 3.6818346868, 1.753914366, 8.4843550037, 14.6454097667, 26.8523774363, 2.7165292377, 3.0611957069]</td>
      <td>[0.981199]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


Now that we've tested our pipeline, let's run it with something larger.  We have two batch files - `cc_data_1k.json` that contains 1,000 credit card records to test for fraud.  The other is `cc_data_10k.json` which has 10,000 credit card records to test.

First let's run a batch result for `cc_data_1k.json` and see the results.  

For Arrow enabled instances of Wallaroo, inference results are returned as a [pandas.DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) object.

For non-Arrow enabled instances of Wallaroo, inferences are returned as the [InferenceResult object](https://docs.wallaroo.ai/wallaroo-sdk/wallaroo-sdk-essentials-guide/#inferenceresult-object).

With the inference result we'll output just the cases likely to be fraud.  For Arrow enabled Wallaroo instances, we'll parse the dataframe returned as the inference result.  For non-Arrow Wallaroo instances, we'll use the Inference Result object's `data()` method.


```python
outputs = None
if arrowEnabled is True:
    outputs = ccfraud_pipeline.infer_from_file('./data/cc_data_10k.df.json')
    filter = [elt[0] > 0.75 for elt in outputs['out.dense_1']]
    result = outputs.loc[filter, ['in.tensor','out.dense_1']]
    display(result['out.dense_1'])
else:
    outputs = ccfraud_pipeline.infer_from_file('./data/cc_data_10k.json')
    sequence = outputs[0].data()
    result = filter(lambda x: x > 0.75, sequence[0])
    display(sequence[0])
```


    0        [0.99300325]
    1        [0.99300325]
    2        [0.99300325]
    3        [0.99300325]
    161             [1.0]
    941       [0.9873102]
    1445            [1.0]
    2092        [0.99999]
    2220     [0.91080534]
    4135     [0.98877275]
    4236     [0.95601666]
    5658            [1.0]
    6768      [0.9999745]
    6780      [0.9852645]
    7133            [1.0]
    7566      [0.9999705]
    7911      [0.9980203]
    8921     [0.99950194]
    9244      [0.9999876]
    10176           [1.0]
    Name: out.dense_1, dtype: object


We can view the inputs either through the `in.tensor` column from our DataFrame for Arrow enabled environments, or with the InferenceResult object through the `input_data()` for non-Arrow enabled environments.  We'll display just the first row in either case.


```python
if arrowEnabled is True:
    display(result['in.tensor'][0])
else:
    display(outputs[0].input_data()["tensor"][0])
```


    [-1.0603297501,
     2.3544967095,
     -3.5638788326,
     5.1387348926,
     -1.2308457019,
     -0.7687824608,
     -3.5881228109,
     1.8880837663,
     -3.2789674274,
     -3.9563254554,
     4.0993439118,
     -5.6539176395,
     -0.8775733373,
     -9.131571192,
     -0.6093537873,
     -3.7480276773,
     -5.0309125017,
     -0.8748149526,
     1.9870535692,
     0.7005485718,
     0.9204422758,
     -0.1041491809,
     0.3229564351,
     -0.7418141657,
     0.0384120159,
     1.0993439146,
     1.2603409756,
     -0.1466244739,
     -1.4463212439]


## Batch Deployment through a Pipeline Deployment URL

This next step requires some manual use.  We're going to have `ccfraud_pipeline` display its deployment url - this allows us to submit data through a HTTP interface and get the results back.

First we'll request the url with the `_deployment._url()` method.

* **IMPORTANT NOTE**:  The `_deployment._url()` method will return an **internal** URL when using Python commands from within the Wallaroo instance - for example, the Wallaroo JupyterHub service.  When connecting via an external connection, `_deployment._url()` returns an **external** URL.  External URL connections requires [the authentication be included in the HTTP request](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/), and that [Model Endpoints Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-model-endpoints-guide/) external endpoints are enabled in the Wallaroo configuration options.


```python
deploy_url = ccfraud_pipeline._deployment._url()
print(deploy_url)
```

    https://sparkly-apple-3026.api.wallaroo.community/v1/api/pipelines/infer/uupfccfraudpipeline-166


The API connection details can be retrieved through the Wallaroo client `mlops()` command.  This will display the connection URL, bearer token, and other information.  The bearer token is available for one hour before it expires.

For this example, the API connection details will be retrieved, then used to submit an inference request through the external inference URL retrieved earlier.


```python
connection =wl.mlops().__dict__
token = connection['token']
token
```




    'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJjTGZaYmhVQWl0a210Z0VLV0l1NnczTWlXYmUzWjc3cHdqVjJ2QWM2WUdZIn0.eyJleHAiOjE2Nzc3MDg0MTksImlhdCI6MTY3NzcwODM1OSwiYXV0aF90aW1lIjoxNjc3NzA3NDIzLCJqdGkiOiJhZWY2YTFjYy02NDgxLTRkODEtYjg2Yi0yMGU1ZTY4MTcwZTAiLCJpc3MiOiJodHRwczovL3NwYXJrbHktYXBwbGUtMzAyNi5rZXljbG9hay53YWxsYXJvby5jb21tdW5pdHkvYXV0aC9yZWFsbXMvbWFzdGVyIiwiYXVkIjpbIm1hc3Rlci1yZWFsbSIsImFjY291bnQiXSwic3ViIjoiMTM4YmQ3ZTYtNGRjOC00ZGMxLWE3NjAtYzllNzIxZWYzYzM3IiwidHlwIjoiQmVhcmVyIiwiYXpwIjoic2RrLWNsaWVudCIsInNlc3Npb25fc3RhdGUiOiIyYWIzZDY0ZC01NmJiLTQ1MDgtOGU4ZS1lNmExMzJlZDEwYWYiLCJhY3IiOiIwIiwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbImNyZWF0ZS1yZWFsbSIsImRlZmF1bHQtcm9sZXMtbWFzdGVyIiwib2ZmbGluZV9hY2Nlc3MiLCJhZG1pbiIsInVtYV9hdXRob3JpemF0aW9uIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsibWFzdGVyLXJlYWxtIjp7InJvbGVzIjpbInZpZXctcmVhbG0iLCJ2aWV3LWlkZW50aXR5LXByb3ZpZGVycyIsIm1hbmFnZS1pZGVudGl0eS1wcm92aWRlcnMiLCJpbXBlcnNvbmF0aW9uIiwiY3JlYXRlLWNsaWVudCIsIm1hbmFnZS11c2VycyIsInF1ZXJ5LXJlYWxtcyIsInZpZXctYXV0aG9yaXphdGlvbiIsInF1ZXJ5LWNsaWVudHMiLCJxdWVyeS11c2VycyIsIm1hbmFnZS1ldmVudHMiLCJtYW5hZ2UtcmVhbG0iLCJ2aWV3LWV2ZW50cyIsInZpZXctdXNlcnMiLCJ2aWV3LWNsaWVudHMiLCJtYW5hZ2UtYXV0aG9yaXphdGlvbiIsIm1hbmFnZS1jbGllbnRzIiwicXVlcnktZ3JvdXBzIl19LCJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6ImVtYWlsIHByb2ZpbGUiLCJzaWQiOiIyYWIzZDY0ZC01NmJiLTQ1MDgtOGU4ZS1lNmExMzJlZDEwYWYiLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiaHR0cHM6Ly9oYXN1cmEuaW8vand0L2NsYWltcyI6eyJ4LWhhc3VyYS11c2VyLWlkIjoiMTM4YmQ3ZTYtNGRjOC00ZGMxLWE3NjAtYzllNzIxZWYzYzM3IiwieC1oYXN1cmEtZGVmYXVsdC1yb2xlIjoidXNlciIsIngtaGFzdXJhLWFsbG93ZWQtcm9sZXMiOlsidXNlciJdLCJ4LWhhc3VyYS11c2VyLWdyb3VwcyI6Int9In0sInByZWZlcnJlZF91c2VybmFtZSI6ImpvaG4uaGFuc2FyaWNrQHdhbGxhcm9vLmFpIiwiZW1haWwiOiJqb2huLmhhbnNhcmlja0B3YWxsYXJvby5haSJ9.gTzaCHqmV_LsYIROOr2bxu3s9eDAhyz1ZVjGhXm21iWAe-1YMHgMUoC0zHtZKRt3fGaPswkj3LpC9RC2KaiYFc94KXSbyVKVq9--M5CVYQ2xqXrIY02fOsogMuqGtRusHQISRvx6dLYjJJdw6B_DwQL8vTC0C82VeBdWz-mIpi_gqQ2T2UKlNB-kBnjtY5BtlzBugF4_DKi5o3ePBoZZzT4EuionnDDJEF-L-5Pg5ioWXYyH7ianb-FcTLJqESAhgs9PlCWhtwGnJQVJmv5BdREkGvtRmDWzlbVYKX4tl_Giag_zO0U58fqD9AdiMFTndiryn1zxKyf1hxoS-3fXxQ'



The `deploy_url` variable will be used to access the pipeline inference URL, and the `token` variable used to authenticate for this batch inference process.


```python
if arrowEnabled is True:
    dataFile="./data/cc_data_10k.df.json"
    contentType="application/json; format=pandas-records"
else:
    dataFile="./data/cc_data_10k.json"
    contentType="application/json"
```


```python
!curl -X POST {deploy_url} -H "Authorization: Bearer {token}" -H "Content-Type:{contentType}" --data @{dataFile} > curl_response.txt
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100 13.7M  100 6325k  100 7743k  2560k  3133k  0:00:02  0:00:02 --:--:-- 5702k


With our work in the pipeline done, we'll undeploy it to get back our resources from the Kubernetes cluster.  If we keep the same settings we can redeploy the pipeline with the same configuration in the future.


```python
ccfraud_pipeline.undeploy()
```




<table><tr><th>name</th> <td>uupfccfraudpipeline</td></tr><tr><th>created</th> <td>2023-03-01 22:00:37.919002+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-01 22:04:51.394606+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>6b63ccf6-870a-4299-8213-ec3b4538ce0d, 70349464-78ca-4699-ae31-1b6322e44232, b78f3d0a-ea52-47a6-959d-d4aa0b60c7b1</td></tr><tr><th>steps</th> <td>uupfccfraudmodel</td></tr></table>



And there we have it!  Feel free to use this as a template for other models, inferences and pipelines that you want to deploy with Wallaroo!
