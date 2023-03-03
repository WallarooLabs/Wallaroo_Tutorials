This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-features/model_hot_swap).

## Model Hot Swap Tutorial

One of the biggest challenges facing organizations once they have a model trained is deploying the model:  Getting all of the resources together, MLOps configured and systems prepared to allow inferences to run.

The next biggest challenge?  Replacing the model while keeping the existing production systems running.

This tutorial demonstrates how Wallaroo model hot swap can update a pipeline step with a new model with one command.  This lets organizations keep their production systems running while changing a ML model, with the change taking only milliseconds, and any inference requests in that time are processed after the hot swap is completed.

This example and sample data comes from the Machine Learning Group's demonstration on [Credit Card Fraud detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

This tutorial provides the following:

* `ccfraud.onnx`: A pre-trained ML model used to detect potential credit card fraud.
* `xgboost_ccfraud.onnx`: A pre-trained ML model used to detect potential credit card fraud originally converted from an XGBoost model.  This will be used to swap with the `ccfraud.onnx`.
* `smoke_test.json`: A data file used to verify that the model will return a low possibility of credit card fraud.
* `high_fraud.json`: A data file used to verify that the model will return a high possibility of credit card fraud.
* Sample inference data files: Data files used for inference examples with the following number of records:
  * `cc_data_5.json`: 5 records.
  * `cc_data_1k.json`: 1,000 records.
  * `cc_data_10k.json`: 10,000 records.
  * `cc_data_40k.json`: Over 40,000 records.

## Reference

For more information about Wallaroo and related features, see the [Wallaroo Documentation Site](https://docs.wallaroo.ai).

## Steps

The following steps demonstrate the following:

* Connect to a Wallaroo instance.
* Create a workspace and pipeline.
* Upload both models to the workspace.
* Deploy the pipe with the `ccfraud.onnx` model as a pipeline step.
* Perform sample inferences.
* Hot swap and replace the existing model with the `xgboost_ccfraud.onnx` while keeping the pipeline deployed.
* Conduct additional inferences to demonstrate the model hot swap was successful.
* Undeploy the pipeline and return the resources back to the Wallaroo instance.

### Load the Libraries

Load the Python libraries used to connect and interact with the Wallaroo instance.

```python
import wallaroo
from wallaroo.object import EntityNotFoundError

import pandas as pd

# used to display dataframe information without truncating
pd.set_option('display.max_colwidth', None)  

```

### Arrow Support

As of the 2023.1 release, Wallaroo provides support for dataframe and Arrow for inference inputs.  This tutorial allows users to adjust their experience based on whether they have enabled Arrow support in their Wallaroo instance or not.

If Arrow support has been enabled, `arrowEnabled=True`. If disabled or you're not sure, set it to `arrowEnabled=False`

The examples below will be shown in an arrow enabled environment.

```python
import os
# Only set the below to make the OS environment ARROW_ENABLED to TRUE.  Otherwise, leave as is.
# os.environ["ARROW_ENABLED"]="True"

if "ARROW_ENABLED" not in os.environ or os.environ["ARROW_ENABLED"] == "False":
    arrowEnabled = False
else:
    arrowEnabled = True
print(arrowEnabled)
```

    True

### Open a Connection to Wallaroo

The first step is to connect to Wallaroo through the Wallaroo client.

This is accomplished using the `wallaroo.Client(api_endpoint, auth_endpoint, auth_type command)` command that connects to the Wallaroo instance services.

The `Client` method takes the following parameters:

* **api_endpoint** (*String*): The URL to the Wallaroo instance API service.
* **auth_endpoint** (*String*): The URL to the Wallaroo instance Keycloak service.
* **auth_type command** (*String*): The authorization type.  In this case, `SSO`.

The URLs are based on the Wallaroo Prefix and Wallaroo Suffix for the Wallaroo instance.  For more information, see the [DNS Integration Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-dns-guide/).  In the example below, replace "YOUR PREFIX" and "YOUR SUFFIX" with the Wallaroo Prefix and Suffix, respectively.

If connecting from within the Wallaroo instance's JupyterHub service, then only `wl = wallaroo.Client()` is required.

Once run, the `wallaroo.Client` command provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Depending on the configuration of the Wallaroo instance, the user will either be presented with a login request to the Wallaroo instance or be authenticated through a broker such as Google, Github, etc.  To use the broker, select it from the list under the username/password login forms.  For more information on Wallaroo authentication configurations, see the [Wallaroo Authentication Configuration Guides](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-sso-authentication/).

```python
# Internal Login

wl = wallaroo.Client()

# Remote Login

# wallarooPrefix = "YOUR PREFIX"
# wallarooSuffix = "YOUR SUFFIX"

# wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}.api.{wallarooSuffix}", 
#                     auth_endpoint=f"https://{wallarooPrefix}.keycloak.{wallarooSuffix}", 
#                     auth_type="sso")
```

### Set the Variables

The following variables are used in the later steps for creating the workspace, pipeline, and uploading the models.  Modify them according to your organization's requirements.

Just for the sake of this tutorial, we'll use the SDK below to create our workspace , assign as our **current workspace**, then display all of the workspaces we have at the moment.  We'll also set up for our models and pipelines down the road, so we have one spot to change names to whatever fits your organization's standards best.

To allow this tutorial to be run multiple times or by multiple users in the same Wallaroo instance, a random 4 character prefix will be added to the workspace, pipeline, and model.

```python
import string
import random

# make a random 4 character prefix
prefix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))

workspace_name = f'{prefix}hotswapworkspace'
pipeline_name = f'{prefix}hotswappipeline'
original_model_name = f'{prefix}ccfraudoriginal'
original_model_file_name = './ccfraud.onnx'
replacement_model_name = f'{prefix}ccfraudreplacement'
replacement_model_file_name = './xgboost_ccfraud.onnx'
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

### Create the Workspace

We will create a workspace based on the variable names set above, and set the new workspace as the `current` workspace.  This workspace is where new pipelines will be created in and store uploaded models for this session.

Once set, the pipeline will be created.

```python
workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)

pipeline = get_pipeline(pipeline_name)
pipeline
```

<table><tr><th>name</th> <td>ggwzhotswappipeline</td></tr><tr><th>created</th> <td>2023-02-27 17:33:53.541871+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-27 17:33:53.541871+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>ad943ff6-1a38-4304-a243-6958ba118df2</td></tr><tr><th>steps</th> <td></td></tr></table>
{{</table>}}

### Upload Models

We can now upload both of the models.  In a later step, only one model will be added as a [pipeline step](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipeline/#add-a-step-to-a-pipeline), where the pipeline will submit inference requests to the pipeline.

```python
original_model = wl.upload_model(original_model_name , original_model_file_name)
replacement_model = wl.upload_model(replacement_model_name , replacement_model_file_name)
```

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
    <td>ggwzccfraudreplacement</td>
    <td>1</td>
    <td>""</td>
    <td>2023-02-27 17:33:55.531013+00:00</td>
    <td>2023-02-27 17:33:55.531013+00:00</td>
  </tr>

  <tr>
    <td>ggwzccfraudoriginal</td>
    <td>1</td>
    <td>""</td>
    <td>2023-02-27 17:33:55.147884+00:00</td>
    <td>2023-02-27 17:33:55.147884+00:00</td>
  </tr>

</table>
{{</table>}}

### Add Model to Pipeline Step

With the models uploaded, we will add the original model as a pipeline step, then deploy the pipeline so it is available for performing inferences.

```python
pipeline.add_model_step(original_model)
pipeline
```

<table><tr><th>name</th> <td>ggwzhotswappipeline</td></tr><tr><th>created</th> <td>2023-02-27 17:33:53.541871+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-27 17:33:53.541871+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>ad943ff6-1a38-4304-a243-6958ba118df2</td></tr><tr><th>steps</th> <td></td></tr></table>
{{</table>}}

```python
pipeline.deploy()
```

<table><tr><th>name</th> <td>ggwzhotswappipeline</td></tr><tr><th>created</th> <td>2023-02-27 17:33:53.541871+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-27 17:33:58.263239+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>3078dffa-4e10-41ef-85bc-e7a0de5afa82, ad943ff6-1a38-4304-a243-6958ba118df2</td></tr><tr><th>steps</th> <td>ggwzccfraudoriginal</td></tr></table>
{{</table>}}

```python
pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.0.30',
       'name': 'engine-7bb46d756-2v7j7',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'ggwzhotswappipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'ggwzccfraudoriginal',
          'version': '55fe0137-2e0b-4c7d-9cf2-6521b526339e',
          'sha': 'bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.0.29',
       'name': 'engine-lb-ddd995646-wjg4k',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}

### Verify the Model

The pipeline is deployed with our model.  The following will verify that the model is operating correctly.  The `high_fraud.json` file contains data that the model should process as a high likelihood of being a fraudulent transaction.

```python
if arrowEnabled is True:
    result = pipeline.infer_from_file('./data/smoke_test.df.json')
else:
    result = pipeline.infer_from_file('./data/smoke_test.json')
display(result)
```

{{<table "table table-bordered">}}
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out.dense_1</th>
      <th>check_failures</th>
      <th>metadata.last_model</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1677519249926</td>
      <td>[0.0014974177]</td>
      <td>[]</td>
      <td>{"model_name":"ggwzccfraudoriginal","model_sha":"bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507"}</td>
    </tr>
  </tbody>
</table>
{{</table>}}


### Replace the Model

The pipeline is currently deployed and is able to handle inferences.  The model will now be replaced without having to undeploy the pipeline.  This is done using the pipeline method [`replace_with_model_step(index, model)`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/pipeline/#Pipeline.replace_with_model_step).  Steps start at `0`, so the method called below will replace step 0 in our pipeline with the replacement model.

As an exercise, this deployment can be performed while inferences are actively being submitted to the pipeline to show how quickly the swap takes place.

```python
pipeline.replace_with_model_step(0, replacement_model).deploy()
```

<table><tr><th>name</th> <td>ggwzhotswappipeline</td></tr><tr><th>created</th> <td>2023-02-27 17:33:53.541871+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-27 17:34:11.049248+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>a620354f-291e-4a98-b5f7-9d8bf165b1df, 3078dffa-4e10-41ef-85bc-e7a0de5afa82, ad943ff6-1a38-4304-a243-6958ba118df2</td></tr><tr><th>steps</th> <td>ggwzccfraudoriginal</td></tr></table>
{{</table>}}

```python
# Display the pipeline
pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.0.30',
       'name': 'engine-7bb46d756-2v7j7',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'ggwzhotswappipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'ggwzccfraudoriginal',
          'version': '55fe0137-2e0b-4c7d-9cf2-6521b526339e',
          'sha': 'bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.0.29',
       'name': 'engine-lb-ddd995646-wjg4k',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}

### Verify the Swap

To verify the swap, we'll submit a set of inferences to the pipeline using the new model.  We'll display just the first 5 rows for space reasons.

```python
if arrowEnabled is True:
    result = pipeline.infer_from_file('./data/cc_data_1k.df.json')
    display(result.loc[0:4,:])
else:
    result = pipeline.infer_from_file('./data/cc_data_1k.json')
    display(result)
```

{{<table "table table-bordered">}}
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out.dense_1</th>
      <th>check_failures</th>
      <th>metadata.last_model</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1677519255319</td>
      <td>[0.99300325]</td>
      <td>[]</td>
      <td>{"model_name":"ggwzccfraudreplacement","model_sha":"bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507"}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1677519255319</td>
      <td>[0.99300325]</td>
      <td>[]</td>
      <td>{"model_name":"ggwzccfraudreplacement","model_sha":"bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507"}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1677519255319</td>
      <td>[0.99300325]</td>
      <td>[]</td>
      <td>{"model_name":"ggwzccfraudreplacement","model_sha":"bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507"}</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1677519255319</td>
      <td>[0.99300325]</td>
      <td>[]</td>
      <td>{"model_name":"ggwzccfraudreplacement","model_sha":"bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507"}</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1677519255319</td>
      <td>[0.0010916889]</td>
      <td>[]</td>
      <td>{"model_name":"ggwzccfraudreplacement","model_sha":"bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507"}</td>
    </tr>
  </tbody>
</table>
{{</table>}}


### Undeploy the Pipeline

With the tutorial complete, the pipeline is undeployed to return the resources back to the Wallaroo instance.

```python
pipeline.undeploy()
```

<table><tr><th>name</th> <td>ggwzhotswappipeline</td></tr><tr><th>created</th> <td>2023-02-27 17:33:53.541871+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-27 17:34:11.049248+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>a620354f-291e-4a98-b5f7-9d8bf165b1df, 3078dffa-4e10-41ef-85bc-e7a0de5afa82, ad943ff6-1a38-4304-a243-6958ba118df2</td></tr><tr><th>steps</th> <td>ggwzccfraudoriginal</td></tr></table>
{{</table>}}

