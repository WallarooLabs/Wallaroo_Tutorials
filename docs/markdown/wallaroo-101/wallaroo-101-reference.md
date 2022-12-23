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

```python
import wallaroo
```

```python
# Login through local Wallaroo instance

# wl = wallaroo.Client()

# SSO login through keycloak

wallarooPrefix = "YOUR PREFIX"
wallarooSuffix = "YOUR SUFFIX"

wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}.api.{wallarooSuffix}", 
                    auth_endpoint=f"https://{wallarooPrefix}.keycloak.{wallarooSuffix}", 
                    auth_type="sso")
```

## Create a New Workspace

Next we're going to create a new workspace called `ccfraudworkspace` for our model, then set it as our current workspace context.  We'll be doing this through the SDK, but here's an example of doing it through the Wallaroo dashboard.

The method we'll introduce below will either **create** a new workspace if it doesn't exist, or **select** an existing workspace.  So if you create the workspace `ccfraudworkspace` then you're covered either way.

The first part is to return to your Wallaroo Dashboard.  In the top navigation panel next to your user name there's a drop down with your workspaces.  In this example it just has "My Workspace".  Select **View Workspaces**.

![Select View Workspaces](/images/wallaroo-101/wallaroo-dashboard-select-view-workspaces.png)

From here, enter the name of our new workspace as `ccfraud-workspace`.  If it already exists, you can skip this step.

* **IMPORTANT NOTE**:  Workspaces do not have forced unique names.  It is highly recommended to use an existing workspace when possible, or establish a naming convention for your workspaces to keep their names unique to remove confusion with teams.

![Create ccfraud-workspace](/images/wallaroo-101/wallaroo-dashboard-create-workspace-ccfraud.png)

Once complete, you'll be able to select the workspace from the drop down list in your dashboard.

![ccfraud-workspace exists](/images/wallaroo-101/wallaroo-dashboard-ccfraud-workspace-exists.png)

Just for the sake of this tutorial, we'll use the SDK below to create our workspace , assign as our **current workspace**, then display all of the workspaces we have at the moment.  We'll also set up for our models and pipelines down the road, so we have one spot to change names to whatever fits your organization's standards best.

When we create our new workspace, we'll save it in the Python variable `workspace` so we can refer to it as needed.

```python
workspace_name = 'ccfraudworkspace'
pipeline_name = 'ccfraudpipeline'
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
```

    {'name': 'ccfraudworkspace', 'id': 17, 'archived': False, 'created_by': 'f1f32bdf-9bd9-4595-a531-aca5778ceaf0', 'created_at': '2022-12-15T15:40:13.340199+00:00', 'models': [], 'pipelines': []}

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
    <td>2022-12-12 22:13:38</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>0</td>
    <td>0</td>
</tr>

<tr >
    <td>alohaworkspace</td>
    <td>2022-12-12 22:48:05</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>sdkworkspace</td>
    <td>2022-12-12 22:53:12</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>housepricedrift</td>
    <td>2022-12-13 16:32:11</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>alohaworkspace-regression</td>
    <td>2022-12-14 20:03:18</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>abtestworkspace</td>
    <td>2022-12-14 21:04:26</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>2</td>
    <td>1</td>
</tr>

<tr >
    <td>anomalyexample</td>
    <td>2022-12-14 22:03:34</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>demandcurveworkspace</td>
    <td>2022-12-14 22:06:43</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>3</td>
    <td>1</td>
</tr>

<tr >
    <td>imdbworkspace</td>
    <td>2022-12-14 22:09:41</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>2</td>
    <td>1</td>
</tr>

<tr >
    <td>testautoconversion</td>
    <td>2022-12-14 22:12:36</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>1</td>
    <td>0</td>
</tr>

<tr >
    <td>keras-autoconvert-workspace</td>
    <td>2022-12-14 22:20:58</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>bikedayevalworkspace</td>
    <td>2022-12-14 22:29:09</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>

<tr >
    <td>xgboost-classification-autoconvert-workspace</td>
    <td>2022-12-15 15:24:08</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>0</td>
    <td>1</td>
</tr>

<tr >
    <td>xgboost-regression-autoconvert-workspace</td>
    <td>2022-12-15 15:26:33</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>0</td>
    <td>1</td>
</tr>

<tr >
    <td>housepricing2</td>
    <td>2022-12-15 15:36:34</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>0</td>
    <td>0</td>
</tr>

<tr >
    <td>ccfraud-comparison-demo</td>
    <td>2022-12-15 15:37:35</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>3</td>
    <td>1</td>
</tr>

<tr >
    <td>ccfraudworkspace</td>
    <td>2022-12-15 15:40:13</td>
    <td>['john.hansarick@wallaroo.ai']</td>
    <td>0</td>
    <td>0</td>
</tr>

</table>

Just to make sure, let's list our current workspace.  If everything is going right, it will show us we're in the `ccfraud-workspace`.

```python
wl.set_current_workspace(workspace)
wl.get_current_workspace()
```

    {'name': 'ccfraudworkspace', 'id': 17, 'archived': False, 'created_by': 'f1f32bdf-9bd9-4595-a531-aca5778ceaf0', 'created_at': '2022-12-15T15:40:13.340199+00:00', 'models': [], 'pipelines': []}

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
    <td>ccfraudmodel</td>
    <td>1</td>
    <td>""</td>
    <td>2022-12-15 15:40:51.058061+00:00</td>
    <td>2022-12-15 15:40:51.058061+00:00</td>
  </tr>

</table>

## Create a Pipeline

With our model uploaded, time to create our pipeline and deploy it so it can accept data and process it through our `ccfraudmodel`.  We'll call our pipeline `ccfraudpipeline`.

* **NOTE**:  Pipeline names must be unique.  If two pipelines are assigned the same name, the new pipeline is created as a new **version** of the pipeline.

```python
ccfraud_pipeline = wl.build_pipeline(pipeline_name)
```

Now our pipeline is set.  Let's add a single **step** to it - in this case, our `ccfraud_model` that we uploaded to our workspace.

```python
ccfraud_pipeline.add_model_step(ccfraud_model)
```

<table><tr><th>name</th> <td>ccfraudpipeline</td></tr><tr><th>created</th> <td>2022-12-15 15:40:52.827438+00:00</td></tr><tr><th>last_updated</th> <td>2022-12-15 15:40:52.827438+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>c6ca1538-4a87-48e7-9187-c4d80d1b2a9c</td></tr><tr><th>steps</th> <td></td></tr></table>

And now we can deploy our pipeline and assign resources to it.  This typically takes about 45 seconds once the command is issued.

```python
ccfraud_pipeline.deploy()
```

<table><tr><th>name</th> <td>ccfraudpipeline</td></tr><tr><th>created</th> <td>2022-12-15 15:40:52.827438+00:00</td></tr><tr><th>last_updated</th> <td>2022-12-15 15:40:54.457218+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>a923f5e0-23d5-49eb-8908-db85406387f1, c6ca1538-4a87-48e7-9187-c4d80d1b2a9c</td></tr><tr><th>steps</th> <td>ccfraudmodel</td></tr></table>

We can see our new pipeline with the `status()` command.

```python
ccfraud_pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.2.37',
       'name': 'engine-57f588684c-hx2w2',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'ccfraudpipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'ccfraudmodel',
          'version': 'f0bf324c-3640-40b6-9db6-cd669c55000f',
          'sha': 'bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.2.36',
       'name': 'engine-lb-c6485cfd5-6w9zg',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}

## Running Interfences

With our pipeline deployed, let's run a smoke test to make sure it's working right.  We'll run an inference through our pipeline from the file `smoke_test.json` and see the results.  This should give us a result near 0 - not likely a fraudulent activity.

```python
ccfraud_pipeline.infer_from_file('./smoke_test.json')
```

    [InferenceResult({'check_failures': [],
      'elapsed': 143102,
      'model_name': 'ccfraudmodel',
      'model_version': 'f0bf324c-3640-40b6-9db6-cd669c55000f',
      'original_data': {'tensor': [[1.0678324729342086,
                                    0.21778102664937624,
                                    -1.7115145261843976,
                                    0.6822857209662413,
                                    1.0138553066742804,
                                    -0.43350000129006655,
                                    0.7395859436561657,
                                    -0.28828395953577357,
                                    -0.44726268795990787,
                                    0.5146124987725894,
                                    0.3791316964287545,
                                    0.5190619748123175,
                                    -0.4904593221655364,
                                    1.1656456468728569,
                                    -0.9776307444180006,
                                    -0.6322198962519854,
                                    -0.6891477694494687,
                                    0.17833178574255615,
                                    0.1397992467197424,
                                    -0.35542206494183326,
                                    0.4394217876939808,
                                    1.4588397511627804,
                                    -0.3886829614721505,
                                    0.4353492889350186,
                                    1.7420053483337177,
                                    -0.4434654615252943,
                                    -0.15157478906219238,
                                    -0.26684517248765616,
                                    -1.454961775612449]]},
      'outputs': [{'Float': {'data': [0.001497417688369751],
                             'dim': [1, 1],
                             'dtype': 'Float',
                             'v': 1}}],
      'pipeline_name': 'ccfraudpipeline',
      'shadow_data': {},
      'time': 1671118939781})]

Looks good!  Time to run the real test on some real data.  Run another inference this time from the file `high_fraud.json` and let's see the results.  This should give us an output that indicates a high level of fraud - well over 90%.

```python
ccfraud_pipeline.infer_from_file('./high_fraud.json')
```

    [InferenceResult({'check_failures': [],
      'elapsed': 186303,
      'model_name': 'ccfraudmodel',
      'model_version': 'f0bf324c-3640-40b6-9db6-cd669c55000f',
      'original_data': {'tensor': [[1.0678324729342086,
                                    18.155556397512136,
                                    -1.658955105843852,
                                    5.2111788045436445,
                                    2.345247064454334,
                                    10.467083577773014,
                                    5.0925820522419745,
                                    12.82951536371218,
                                    4.953677046849403,
                                    2.3934736228338225,
                                    23.912131817957253,
                                    1.7599568310350209,
                                    0.8561037518143335,
                                    1.1656456468728569,
                                    0.5395988813934498,
                                    0.7784221343010385,
                                    6.75806107274245,
                                    3.927411847659908,
                                    12.462178276650056,
                                    12.307538216518656,
                                    13.787951906620115,
                                    1.4588397511627804,
                                    3.681834686805714,
                                    1.753914366037974,
                                    8.484355003656184,
                                    14.6454097666836,
                                    26.852377436250144,
                                    2.716529237720336,
                                    3.061195706890285]]},
      'outputs': [{'Float': {'data': [0.9811990261077881],
                             'dim': [1, 1],
                             'dtype': 'Float',
                             'v': 1}}],
      'pipeline_name': 'ccfraudpipeline',
      'shadow_data': {},
      'time': 1671118941160})]

Now that we've tested our pipeline, let's run it with something larger.  We have two batch files - `cc_data_1k.json` that contains 1,000 credit card records to test for fraud.  The other is `cc_data_10k.json` which has 10,000 credit card records to test.

First let's run a batch result for `cc_data_1k.json` and see the results.  Inferences are returned as the [InferenceResult object](https://docs.wallaroo.ai/wallaroo-sdk/wallaroo-sdk-essentials-guide/#inferenceresult-object).  We'll retrieve the InferenceResult object and store it into a variable.

```python
output = ccfraud_pipeline.infer_from_file('./cc_data_10k.json')
```

Now we can isolate just the output with the `.data()` method, then isolate it down to just the results likely to be fraud.

```python
sequence = output[0].data()
result = filter(lambda x: x > 0.75, sequence[0])
print(list(result))
```

    [array([0.99300325]), array([0.99300325]), array([0.99300325]), array([0.99300325]), array([1.]), array([0.98731017]), array([1.]), array([0.99998999]), array([0.91080534]), array([0.98877275]), array([0.95601666]), array([1.]), array([0.99997449]), array([0.98526448]), array([1.]), array([0.9999705]), array([0.99802029]), array([0.99950194]), array([0.9999876]), array([1.])]

We can also retrieve the inputs from our InferenceResult object through the `input_data()` method as follows - in this case, just the first record.

```python
output[0].input_data()["tensor"][0]
```

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
     -1.446321243938815]

## Batch Deployment through a Pipeline Deployment URL

This next step requires some manual use.  We're going to have `ccfraud_pipeline` display its deployment url - this allows us to submit data through a HTTP interface and get the results back.

First we'll request the url with the `_deployment._url()` method:

```python
deploy_url = ccfraud_pipeline._deployment._url()
```

The API connection details can be retrieved through the Wallaroo client `mlops()` command.  This will display the connection URL, bearer token, and other information.  The bearer token is available for one hour before it expires.

For this example, the API connection details will be retrieved, then used to submit an inference request through the external inference URL retrieved earlier.

```python
connection =wl.mlops().__dict__
token = connection['token']
token
```

Copy and paste the results above into the curl command, replacing the {YOUR URL HERE} with your deploy url for `ccfraud_pipeline`, and uncomment it.

```python
!curl -X POST {deploy_url} -H "Content-Type:application/json" -H "Authorization: Bearer {token}" -H "Content-Type:application/json" --data @cc_data_10k.json > curl_response.txt
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
      0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0curl: (6) Could not resolve host: deploy_url

With our work in the pipeline done, we'll undeploy it to get back our resources from the Kubernetes cluster.  If we keep the same settings we can redeploy the pipeline with the same configuration in the future.

```python
ccfraud_pipeline.undeploy()
```

<table><tr><th>name</th> <td>ccfraudpipeline</td></tr><tr><th>created</th> <td>2022-12-15 15:40:52.827438+00:00</td></tr><tr><th>last_updated</th> <td>2022-12-15 15:40:54.457218+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>a923f5e0-23d5-49eb-8908-db85406387f1, c6ca1538-4a87-48e7-9187-c4d80d1b2a9c</td></tr><tr><th>steps</th> <td>ccfraudmodel</td></tr></table>

And there we have it!  Feel free to use this as a template for other models, inferences and pipelines that you want to deploy with Wallaroo!
