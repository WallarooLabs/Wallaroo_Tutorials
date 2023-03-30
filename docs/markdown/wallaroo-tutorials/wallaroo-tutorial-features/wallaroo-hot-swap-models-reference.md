This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-features/model_hot_swap).

## Model Hot Swap Tutorial

One of the biggest challenges facing organizations once they have a model trained is deploying the model:  Getting all of the resources together, MLOps configured and systems prepared to allow inferences to run.

The next biggest challenge?  Replacing the model while keeping the existing production systems running.

This tutorial demonstrates how Wallaroo model hot swap can update a pipeline step with a new model with one command.  This lets organizations keep their production systems running while changing a ML model, with the change taking only milliseconds, and any inference requests in that time are processed after the hot swap is completed.

This example and sample data comes from the Machine Learning Group's demonstration on [Credit Card Fraud detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

This tutorial provides the following:

* Models:
  * `rf_model.onnx`: The champion model that has been used in this environment for some time.
  * `xgb_model.onnx` and `gbr_model.onnx`: Rival models that we will swap out from the champion model.
* Data:
  * xtest-1.df.json and xtest-1k.df.json:  DataFrame JSON inference inputs with 1 input and 1,000 inputs.
  * xtest-1.arrow and xtest-1k.arrow:  Apache Arrow inference inputs with 1 input and 1,000 inputs.

## Reference

For more information about Wallaroo and related features, see the [Wallaroo Documentation Site](https://docs.wallaroo.ai).

## Prerequisites

* A deployed Wallaroo instance
* The following Python libraries installed:
  * `os`
  * [`wallaroo`](https://pypi.org/project/wallaroo/): The Wallaroo SDK. Included with the Wallaroo JupyterHub service by default.
  * [`pandas`](https://pypi.org/project/pandas/): Pandas, mainly used for Pandas DataFrame
  * [`pyarrow`](https://pypi.org/project/pyarrow/): PyArrow for Apache Arrow support
  * [`polars`](https://pypi.org/project/polars/): Polars for DataFrame with native Apache Arrow support

## Steps

The following steps demonstrate the following:

* Connect to a Wallaroo instance.
* Create a workspace and pipeline.
* Upload both models to the workspace.
* Deploy the pipe with the `rf_model.onnx` model as a pipeline step.
* Perform sample inferences.
* Hot swap and replace the existing model with the `xgb_model.onnx` while keeping the pipeline deployed.
* Conduct additional inferences to demonstrate the model hot swap was successful.
* Hot swap again with gbr_model.onnx, and perform more sample inferences.
* Undeploy the pipeline and return the resources back to the Wallaroo instance.

### Load the Libraries

Load the Python libraries used to connect and interact with the Wallaroo instance.


```python
import wallaroo
from wallaroo.object import EntityNotFoundError

# to display dataframe tables
from IPython.display import display
# used to display dataframe information without truncating
import pandas as pd
pd.set_option('display.max_colwidth', None)
import pyarrow as pa

import os
# Used for the Wallaroo SDK version 2023.1
os.environ["ARROW_ENABLED"]="True"
```

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
# Login through local Wallaroo instance

wl = wallaroo.Client()

# SSO login through keycloak

# wallarooPrefix = "YOUR PREFIX"
# wallarooSuffix = "YOUR PREFIX"

# wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}.api.{wallarooSuffix}", 
#                     auth_endpoint=f"https://{wallarooPrefix}.keycloak.{wallarooSuffix}", 
#                     auth_type="sso")
```

    Please log into the following URL in a web browser:
    
    	https://wallaroo.keycloak.example.com/auth/realms/master/device?user_code=XENO-VUGB
    
    Login successful!


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
original_model_name = f'{prefix}housingmodelcontrol'
original_model_file_name = './models/rf_model.onnx'
replacement_model_name01 = f'{prefix}gbrhousingchallenger'
replacement_model_file_name01 = './models/gbr_model.onnx'
replacement_model_name02 = f'{prefix}xgbhousingchallenger'
replacement_model_file_name02 = './models/xgb_model.onnx'

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




<table><tr><th>name</th> <td>axnhhotswappipeline</td></tr><tr><th>created</th> <td>2023-03-28 16:50:30.027322+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-28 16:50:30.027322+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>ad6f8f3b-300f-4b3e-8ed5-9412f2150703</td></tr><tr><th>steps</th> <td></td></tr></table>



### Upload Models

We can now upload both of the models.  In a later step, only one model will be added as a [pipeline step](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipeline/#add-a-step-to-a-pipeline), where the pipeline will submit inference requests to the pipeline.


```python
original_model = wl.upload_model(original_model_name , original_model_file_name)
replacement_model01 = wl.upload_model(replacement_model_name01 , replacement_model_file_name01)
replacement_model02 = wl.upload_model(replacement_model_name02 , replacement_model_file_name02)
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
    <td>axnhxgbhousingchallenger</td>
    <td>1</td>
    <td>""</td>
    <td>2023-03-28 16:50:33.364914+00:00</td>
    <td>2023-03-28 16:50:33.364914+00:00</td>
  </tr>

  <tr>
    <td>axnhgbrhousingchallenger</td>
    <td>1</td>
    <td>""</td>
    <td>2023-03-28 16:50:32.740417+00:00</td>
    <td>2023-03-28 16:50:32.740417+00:00</td>
  </tr>

  <tr>
    <td>axnhhousingmodelcontrol</td>
    <td>1</td>
    <td>""</td>
    <td>2023-03-28 16:50:31.994506+00:00</td>
    <td>2023-03-28 16:50:31.994506+00:00</td>
  </tr>

</table>




### Add Model to Pipeline Step

With the models uploaded, we will add the original model as a pipeline step, then deploy the pipeline so it is available for performing inferences.


```python
pipeline.add_model_step(original_model)
pipeline
```




<table><tr><th>name</th> <td>axnhhotswappipeline</td></tr><tr><th>created</th> <td>2023-03-28 16:50:30.027322+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-28 16:50:30.027322+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>ad6f8f3b-300f-4b3e-8ed5-9412f2150703</td></tr><tr><th>steps</th> <td></td></tr></table>




```python
pipeline.deploy()
```




<table><tr><th>name</th> <td>axnhhotswappipeline</td></tr><tr><th>created</th> <td>2023-03-28 16:50:30.027322+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-28 16:50:36.572557+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>52dd546d-3ba5-4232-af19-06e91ad83978, ad6f8f3b-300f-4b3e-8ed5-9412f2150703</td></tr><tr><th>steps</th> <td>axnhhousingmodelcontrol</td></tr></table>




```python
pipeline.status()
```




    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.3.16',
       'name': 'engine-77864944c-8vxsn',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'axnhhotswappipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'axnhhousingmodelcontrol',
          'version': '83972a44-db4c-408d-a0db-9f33024face6',
          'sha': 'e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.1.24',
       'name': 'engine-lb-ddd995646-vnjhl',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}



### Verify the Model

The pipeline is deployed with our model.  The following will verify that the model is operating correctly.  The `high_fraud.json` file contains data that the model should process as a high likelihood of being a fraudulent transaction.


```python
normal_input = pd.DataFrame.from_records({"tensor": [[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]]})
result = pipeline.infer(normal_input)
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
      <th>out.variable</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-03-28 16:50:49.458</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.7]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
large_house_input = pd.DataFrame.from_records({'tensor': [[4.0, 3.0, 3710.0, 20000.0, 2.0, 0.0, 2.0, 5.0, 10.0, 2760.0, 950.0, 47.6696, -122.261, 3970.0, 20000.0, 79.0, 0.0, 0.0]]})
large_house_result = pipeline.infer(large_house_input)
display(large_house_result)
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
      <th>out.variable</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-03-28 16:50:49.872</td>
      <td>[4.0, 3.0, 3710.0, 20000.0, 2.0, 0.0, 2.0, 5.0, 10.0, 2760.0, 950.0, 47.6696, -122.261, 3970.0, 20000.0, 79.0, 0.0, 0.0]</td>
      <td>[1514079.4]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


### Replace the Model

The pipeline is currently deployed and is able to handle inferences.  The model will now be replaced without having to undeploy the pipeline.  This is done using the pipeline method [`replace_with_model_step(index, model)`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/pipeline/#Pipeline.replace_with_model_step).  Steps start at `0`, so the method called below will replace step 0 in our pipeline with the replacement model.

As an exercise, this deployment can be performed while inferences are actively being submitted to the pipeline to show how quickly the swap takes place.


```python
pipeline.replace_with_model_step(0, replacement_model01).deploy()
```




<table><tr><th>name</th> <td>axnhhotswappipeline</td></tr><tr><th>created</th> <td>2023-03-28 16:50:30.027322+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-28 16:50:50.842313+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>2852ada2-38f1-4cdf-9a11-599dc3e8b1a2, 52dd546d-3ba5-4232-af19-06e91ad83978, ad6f8f3b-300f-4b3e-8ed5-9412f2150703</td></tr><tr><th>steps</th> <td>axnhhousingmodelcontrol</td></tr></table>



### Verify the Swap

To verify the swap, we'll submit the same inferences and display the result.  Note that `out.variable` has a different output than with the original model.


```python
normal_input = pd.DataFrame.from_records({"tensor": [[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]]})
result02 = pipeline.infer(normal_input)
display(result02)
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
      <th>out.variable</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-03-28 16:50:54.340</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[704901.9]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
large_house_input = pd.DataFrame.from_records({'tensor': [[4.0, 3.0, 3710.0, 20000.0, 2.0, 0.0, 2.0, 5.0, 10.0, 2760.0, 950.0, 47.6696, -122.261, 3970.0, 20000.0, 79.0, 0.0, 0.0]]})
large_house_result02 = pipeline.infer(large_house_input)
display(large_house_result02)
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
      <th>out.variable</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-03-28 16:51:06.805</td>
      <td>[4.0, 3.0, 3710.0, 20000.0, 2.0, 0.0, 2.0, 5.0, 10.0, 2760.0, 950.0, 47.6696, -122.261, 3970.0, 20000.0, 79.0, 0.0, 0.0]</td>
      <td>[1981238.0]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


### Replace the Model Again

Let's do one more hot swap, this time with our `replacement_model02`, then get some test inferences.


```python
pipeline.replace_with_model_step(0, replacement_model02).deploy()
```




<table><tr><th>name</th> <td>axnhhotswappipeline</td></tr><tr><th>created</th> <td>2023-03-28 16:50:30.027322+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-28 16:51:10.272698+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>253bd50e-1b3a-4a41-942f-e54c900e40c2, 2852ada2-38f1-4cdf-9a11-599dc3e8b1a2, 52dd546d-3ba5-4232-af19-06e91ad83978, ad6f8f3b-300f-4b3e-8ed5-9412f2150703</td></tr><tr><th>steps</th> <td>axnhhousingmodelcontrol</td></tr></table>




```python
normal_input = pd.DataFrame.from_records({"tensor": [[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]]})
result03 = pipeline.infer(normal_input)
display(result03)
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
      <th>out.variable</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-03-28 16:51:13.939</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[659806.0]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
large_house_input = pd.DataFrame.from_records({'tensor': [[4.0, 3.0, 3710.0, 20000.0, 2.0, 0.0, 2.0, 5.0, 10.0, 2760.0, 950.0, 47.6696, -122.261, 3970.0, 20000.0, 79.0, 0.0, 0.0]]})
large_house_result03 = pipeline.infer(large_house_input)
display(large_house_result03)
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
      <th>out.variable</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-03-28 16:51:14.335</td>
      <td>[4.0, 3.0, 3710.0, 20000.0, 2.0, 0.0, 2.0, 5.0, 10.0, 2760.0, 950.0, 47.6696, -122.261, 3970.0, 20000.0, 79.0, 0.0, 0.0]</td>
      <td>[2176827.0]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


### Compare Outputs

We'll display the outputs of our inferences through the different models for comparison.


```python
display([original_model_name, result.loc[0, "out.variable"]])
display([replacement_model_name01, result02.loc[0, "out.variable"]])
display([replacement_model_name02, result03.loc[0, "out.variable"]])
```


    ['axnhhousingmodelcontrol', [718013.7]]



    ['axnhgbrhousingchallenger', [704901.9]]



    ['axnhxgbhousingchallenger', [659806.0]]



```python
display([original_model_name, large_house_result.loc[0, "out.variable"]])
display([replacement_model_name01, large_house_result02.loc[0, "out.variable"]])
display([replacement_model_name02, large_house_result03.loc[0, "out.variable"]])
```


    ['axnhhousingmodelcontrol', [1981238.0]]



    ['axnhgbrhousingchallenger', [1981238.0]]



    ['axnhxgbhousingchallenger', [2176827.0]]


### Undeploy the Pipeline

With the tutorial complete, the pipeline is undeployed to return the resources back to the Wallaroo instance.


```python
pipeline.undeploy()
```




<table><tr><th>name</th> <td>axnhhotswappipeline</td></tr><tr><th>created</th> <td>2023-03-28 16:50:30.027322+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-28 16:51:10.272698+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>253bd50e-1b3a-4a41-942f-e54c900e40c2, 2852ada2-38f1-4cdf-9a11-599dc3e8b1a2, 52dd546d-3ba5-4232-af19-06e91ad83978, ad6f8f3b-300f-4b3e-8ed5-9412f2150703</td></tr><tr><th>steps</th> <td>axnhhousingmodelcontrol</td></tr></table>


