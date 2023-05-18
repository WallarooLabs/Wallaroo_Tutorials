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
```

### Connect to the Wallaroo Instance

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  If logging in externally, update the `wallarooPrefix` and `wallarooSuffix` variables with the proper DNS information.  For more information on Wallaroo DNS settings, see the [Wallaroo DNS Integration Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-dns-guide/).

```python
# Login through local Wallaroo instance

wl = wallaroo.Client()
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
        pipeline = wl.pipelines_by_name(name)[0]
    except EntityNotFoundError:
        pipeline = wl.build_pipeline(name)
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

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>btffhotswappipeline</td></tr><tr><th>created</th> <td>2023-05-17 21:37:16.033792+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 21:37:16.033792+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>3c5f951b-e815-4bc7-93bf-84de3d46718d</td></tr><tr><th>steps</th> <td></td></tr></table>
{{</table>}}

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
    <td>btffxgbhousingchallenger</td>
    <td>1</td>
    <td>""</td>
    <td>2023-05-17 21:37:19.054417+00:00</td>
    <td>2023-05-17 21:37:19.054417+00:00</td>
  </tr>

  <tr>
    <td>btffgbrhousingchallenger</td>
    <td>1</td>
    <td>""</td>
    <td>2023-05-17 21:37:18.489996+00:00</td>
    <td>2023-05-17 21:37:18.489996+00:00</td>
  </tr>

  <tr>
    <td>btffhousingmodelcontrol</td>
    <td>1</td>
    <td>""</td>
    <td>2023-05-17 21:37:17.825726+00:00</td>
    <td>2023-05-17 21:37:17.825726+00:00</td>
  </tr>

</table>
{{</table>}}

### Add Model to Pipeline Step

With the models uploaded, we will add the original model as a pipeline step, then deploy the pipeline so it is available for performing inferences.

```python
pipeline.add_model_step(original_model)
pipeline
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>btffhotswappipeline</td></tr><tr><th>created</th> <td>2023-05-17 21:37:16.033792+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 21:37:16.033792+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>3c5f951b-e815-4bc7-93bf-84de3d46718d</td></tr><tr><th>steps</th> <td></td></tr></table>
{{</table>}}

```python
pipeline.deploy()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>btffhotswappipeline</td></tr><tr><th>created</th> <td>2023-05-17 21:37:16.033792+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 21:37:21.972663+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>e4b8d7ca-00fa-4e31-8671-3d0a3bf4c16e, 3c5f951b-e815-4bc7-93bf-84de3d46718d</td></tr><tr><th>steps</th> <td>btffhousingmodelcontrol</td></tr></table>
{{</table>}}

```python
pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.3.146',
       'name': 'engine-67df6b7596-6jltl',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'btffhotswappipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'btffhousingmodelcontrol',
          'version': 'e2f17763-fff5-4180-9b6b-f513102173da',
          'sha': 'e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.4.178',
       'name': 'engine-lb-584f54c899-kgd8k',
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

{{<table "table table-striped table-bordered" >}}
<table>
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
      <td>2023-05-17 21:37:33.700</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.7]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}

```python
large_house_input = pd.DataFrame.from_records({'tensor': [[4.0, 3.0, 3710.0, 20000.0, 2.0, 0.0, 2.0, 5.0, 10.0, 2760.0, 950.0, 47.6696, -122.261, 3970.0, 20000.0, 79.0, 0.0, 0.0]]})
large_house_result = pipeline.infer(large_house_input)
display(large_house_result)
```

{{<table "table table-striped table-bordered" >}}
<table>
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
      <td>2023-05-17 21:37:34.107</td>
      <td>[4.0, 3.0, 3710.0, 20000.0, 2.0, 0.0, 2.0, 5.0, 10.0, 2760.0, 950.0, 47.6696, -122.261, 3970.0, 20000.0, 79.0, 0.0, 0.0]</td>
      <td>[1514079.4]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}

### Replace the Model

The pipeline is currently deployed and is able to handle inferences.  The model will now be replaced without having to undeploy the pipeline.  This is done using the pipeline method [`replace_with_model_step(index, model)`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/pipeline/#Pipeline.replace_with_model_step).  Steps start at `0`, so the method called below will replace step 0 in our pipeline with the replacement model.

As an exercise, this deployment can be performed while inferences are actively being submitted to the pipeline to show how quickly the swap takes place.

```python
pipeline.replace_with_model_step(0, replacement_model01).deploy()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>btffhotswappipeline</td></tr><tr><th>created</th> <td>2023-05-17 21:37:16.033792+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 21:37:35.040199+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>4fc11650-1003-43c2-bd3a-96b9cdacbb6d, e4b8d7ca-00fa-4e31-8671-3d0a3bf4c16e, 3c5f951b-e815-4bc7-93bf-84de3d46718d</td></tr><tr><th>steps</th> <td>btffhousingmodelcontrol</td></tr></table>
{{</table>}}

### Verify the Swap

To verify the swap, we'll submit the same inferences and display the result.  Note that `out.variable` has a different output than with the original model.

```python
normal_input = pd.DataFrame.from_records({"tensor": [[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]]})
result02 = pipeline.infer(normal_input)
display(result02)
```

{{<table "table table-striped table-bordered" >}}
<table>
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
      <td>2023-05-17 21:37:38.375</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[704901.9]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}

```python
large_house_input = pd.DataFrame.from_records({'tensor': [[4.0, 3.0, 3710.0, 20000.0, 2.0, 0.0, 2.0, 5.0, 10.0, 2760.0, 950.0, 47.6696, -122.261, 3970.0, 20000.0, 79.0, 0.0, 0.0]]})
large_house_result02 = pipeline.infer(large_house_input)
display(large_house_result02)
```

{{<table "table table-striped table-bordered" >}}
<table>
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
      <td>2023-05-17 21:37:38.820</td>
      <td>[4.0, 3.0, 3710.0, 20000.0, 2.0, 0.0, 2.0, 5.0, 10.0, 2760.0, 950.0, 47.6696, -122.261, 3970.0, 20000.0, 79.0, 0.0, 0.0]</td>
      <td>[1981238.0]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}

### Replace the Model Again

Let's do one more hot swap, this time with our `replacement_model02`, then get some test inferences.

```python
pipeline.replace_with_model_step(0, replacement_model02).deploy()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>btffhotswappipeline</td></tr><tr><th>created</th> <td>2023-05-17 21:37:16.033792+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 21:37:39.738867+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>438796a3-e320-4a51-9e64-35eb32d57b49, 4fc11650-1003-43c2-bd3a-96b9cdacbb6d, e4b8d7ca-00fa-4e31-8671-3d0a3bf4c16e, 3c5f951b-e815-4bc7-93bf-84de3d46718d</td></tr><tr><th>steps</th> <td>btffhousingmodelcontrol</td></tr></table>
{{</table>}}

```python
normal_input = pd.DataFrame.from_records({"tensor": [[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]]})
result03 = pipeline.infer(normal_input)
display(result03)
```

{{<table "table table-striped table-bordered" >}}
<table>
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
      <td>2023-05-17 21:37:43.235</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[659806.0]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}

```python
large_house_input = pd.DataFrame.from_records({'tensor': [[4.0, 3.0, 3710.0, 20000.0, 2.0, 0.0, 2.0, 5.0, 10.0, 2760.0, 950.0, 47.6696, -122.261, 3970.0, 20000.0, 79.0, 0.0, 0.0]]})
large_house_result03 = pipeline.infer(large_house_input)
display(large_house_result03)
```

{{<table "table table-striped table-bordered" >}}
<table>
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
      <td>2023-05-17 21:37:43.613</td>
      <td>[4.0, 3.0, 3710.0, 20000.0, 2.0, 0.0, 2.0, 5.0, 10.0, 2760.0, 950.0, 47.6696, -122.261, 3970.0, 20000.0, 79.0, 0.0, 0.0]</td>
      <td>[2176827.0]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}

### Compare Outputs

We'll display the outputs of our inferences through the different models for comparison.

```python
display([original_model_name, result.loc[0, "out.variable"]])
display([replacement_model_name01, result02.loc[0, "out.variable"]])
display([replacement_model_name02, result03.loc[0, "out.variable"]])
```

    ['btffhousingmodelcontrol', [718013.7]]

    ['btffgbrhousingchallenger', [704901.9]]

    ['btffxgbhousingchallenger', [659806.0]]

```python
display([original_model_name, large_house_result.loc[0, "out.variable"]])
display([replacement_model_name01, large_house_result02.loc[0, "out.variable"]])
display([replacement_model_name02, large_house_result03.loc[0, "out.variable"]])
```

    ['btffhousingmodelcontrol', [1514079.4]]

    ['btffgbrhousingchallenger', [1981238.0]]

    ['btffxgbhousingchallenger', [2176827.0]]

### Undeploy the Pipeline

With the tutorial complete, the pipeline is undeployed to return the resources back to the Wallaroo instance.

```python
pipeline.undeploy()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>btffhotswappipeline</td></tr><tr><th>created</th> <td>2023-05-17 21:37:16.033792+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 21:37:39.738867+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>438796a3-e320-4a51-9e64-35eb32d57b49, 4fc11650-1003-43c2-bd3a-96b9cdacbb6d, e4b8d7ca-00fa-4e31-8671-3d0a3bf4c16e, 3c5f951b-e815-4bc7-93bf-84de3d46718d</td></tr><tr><th>steps</th> <td>btffhousingmodelcontrol</td></tr></table>
{{</table>}}

