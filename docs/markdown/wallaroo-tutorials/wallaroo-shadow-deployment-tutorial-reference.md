This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-testing-tutorials/shadow_deploy).

## Shadow Deployment Tutorial

Wallaroo provides a method of testing the same data against two different models or sets of models at the same time through **shadow deployments** otherwise known as **parallel deployments**.  This allows data to be submitted to a pipeline with inferences running on two different sets of models.  Typically this is performed on a model that is known to provide accurate results - the **champion** - and a model that is being tested to see if it provides more accurate or faster responses depending on the criteria known as the **challengers**.  Multiple challengers can be tested against a single champion.

As described in the Wallaroo blog post [The What, Why, and How of Model A/B Testing](https://www.wallaroo.ai/blog/the-what-why-and-how-of-a/b-testing):

> In data science, A/B tests can also be used to choose between two models in production, by measuring which model performs better in the real world. In this formulation, the control is often an existing model that is currently in production, sometimes called the champion. The treatment is a new model being considered to replace the old one. This new model is sometimes called the challenger....

> Keep in mind that in machine learning, the terms experiments and trials also often refer to the process of finding a training configuration that works best for the problem at hand (this is sometimes called hyperparameter optimization).

When a shadow deployment is created, only the inference from the champion is returned in the InferenceResult Object `data`, while the result data for the shadow deployments is stored in the InferenceResult Object `shadow_data`.

The following tutorial will demonstrate how:

* Upload champion and challenger models into a Wallaroo instance.
* Create a shadow deployment in a Wallaroo pipeline.
* Perform an inference through a pipeline with a shadow deployment.
* View the `data` and `shadow_data` results from the InferenceResult Object.
* View the pipeline logs and pipeline shadow logs.

This tutorial provides the following:

* `dev_smoke_test.json`:  Sample test data used for the inference testing.
* `models/keras_ccfraud.onnx`:  The champion model.
* `models/modelA.onnx`: A challenger model.
* `models/xgboost_ccfraud.onnx`: A challenger model.

All models are similar to the ones used for the Wallaroo-101 example included in the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials).

## Prerequisites

* A deployed Wallaroo instance
* The following Python libraries installed:
  * `os`
  * `json`
  * [`wallaroo`](https://pypi.org/project/wallaroo/): The Wallaroo SDK. Included with the Wallaroo JupyterHub service by default.
  * [`pandas`](https://pypi.org/project/pandas/): Pandas, mainly used for Pandas DataFrame

## Steps

### Import libraries

The first step is to import the libraries required.

```python
import wallaroo
from wallaroo.object import EntityNotFoundError

# used to display dataframe information without truncating
from IPython.display import display
import pandas as pd
pd.set_option('display.max_colwidth', None)
```

### Connect to the Wallaroo Instance

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
# Login through local Wallaroo instance

wl = wallaroo.Client()

```

### Set Variables

The following variables are used to create or use existing workspaces, pipelines, and upload the models.  Adjust them based on your Wallaroo instance and organization requirements.

```python
import string
import random

suffix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))

workspace_name = f'ccfraudcomparisondemo{suffix}'
pipeline_name = f'ccshadow{suffix}'
champion_model_name = f'ccfraud-lstm{suffix}'
champion_model_file = 'models/keras_ccfraud.onnx'
shadow_model_01_name = f'ccfraud-xgb{suffix}'
shadow_model_01_file = 'models/xgboost_ccfraud.onnx'
shadow_model_02_name = f'ccfraud-rf{suffix}'
shadow_model_02_file = 'models/modelA.onnx'
```

### Workspace and Pipeline

The following creates or connects to an existing workspace based on the variable `workspace_name`, and creates or connects to a pipeline based on the variable `pipeline_name`.

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

wl.set_current_workspace(workspace)

pipeline = get_pipeline(pipeline_name)
pipeline

```

<table><tr><th>name</th> <td>ccshadoweonn</td></tr><tr><th>created</th> <td>2023-05-19 15:13:48.963815+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-19 15:13:48.963815+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>08f4c75f-3e61-48d6-ac76-38e6dddcfaf6</td></tr><tr><th>steps</th> <td></td></tr></table>

### Load the Models

The models will be uploaded into the current workspace based on the variable names set earlier and listed as the `champion`, `model2` and `model3`.

```python
champion = wl.upload_model(champion_model_name, champion_model_file).configure()
model2 = wl.upload_model(shadow_model_01_name, shadow_model_01_file).configure()
model3 = wl.upload_model(shadow_model_02_name, shadow_model_02_file).configure()
```

### Create Shadow Deployment

A shadow deployment is created using the `add_shadow_deploy(champion, challengers[])` method where:

* `champion`: The model that will be primarily used for inferences run through the pipeline.  Inference results will be returned through the Inference Object's `data` element.
* `challengers[]`: An array of models that will be used for inferences iteratively.  Inference results will be returned through the Inference Object's `shadow_data` element.

```python
pipeline.add_shadow_deploy(champion, [model2, model3])
pipeline.deploy()
```

<table><tr><th>name</th> <td>ccshadoweonn</td></tr><tr><th>created</th> <td>2023-05-19 15:13:48.963815+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-19 15:13:53.060392+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>c6c28139-dd9d-4fdf-b7fc-d391bae58bc8, 08f4c75f-3e61-48d6-ac76-38e6dddcfaf6</td></tr><tr><th>steps</th> <td>ccfraud-lstmeonn</td></tr></table>

### Run Test Inference

Using the data from `sample_data_file`, a test inference will be made.

For Arrow enabled Wallaroo instances the model outputs are listed by column.  The output data is set by the term `out`, followed by the name of the model.  For the default model, this is `out.dense_1`, while the shadow deployed models are in the format `out_{model name}.variable`, where `{model name}` is the name of the shadow deployed model.

For Arrow disabled environments, the output is from the Wallaroo InferenceResult object.### Run Test Inference

Using the data from `sample_data_file`, a test inference will be made.  As mentioned earlier, the inference results from the `champion` model will be available in the returned InferenceResult Object's `data` element, while inference results from each of the `challenger` models will be in the returned InferenceResult Object's `shadow_data` element.

```python
sample_data_file = './smoke_test.df.json'
response = pipeline.infer_from_file(sample_data_file)
display(response)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.dense_1</th>
      <th>check_failures</th>
      <th>out_ccfraud-rfeonn.variable</th>
      <th>out_ccfraud-xgbeonn.variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-05-19 15:14:08.898</td>
      <td>[1.0678324729, 0.2177810266, -1.7115145262, 0.682285721, 1.0138553067, -0.4335000013, 0.7395859437, -0.2882839595, -0.447262688, 0.5146124988, 0.3791316964, 0.5190619748, -0.4904593222, 1.1656456469, -0.9776307444, -0.6322198963, -0.6891477694, 0.1783317857, 0.1397992467, -0.3554220649, 0.4394217877, 1.4588397512, -0.3886829615, 0.4353492889, 1.7420053483, -0.4434654615, -0.1515747891, -0.2668451725, -1.4549617756]</td>
      <td>[0.0014974177]</td>
      <td>0</td>
      <td>[1.0]</td>
      <td>[0.0005066991]</td>
    </tr>
  </tbody>
</table>

### View Pipeline Logs

With the inferences complete, we can retrieve the log data from the pipeline with the pipeline `logs` method.  Note that for **each** inference request, the logs return **one entry per model**.  For this example, for one inference request three log entries will be created.

```python
pipeline.logs()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.dense_1</th>
      <th>check_failures</th>
      <th>out_ccfraud-rfeonn.variable</th>
      <th>out_ccfraud-xgbeonn.variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-05-19 15:14:08.898</td>
      <td>[1.0678324729, 0.2177810266, -1.7115145262, 0.682285721, 1.0138553067, -0.4335000013, 0.7395859437, -0.2882839595, -0.447262688, 0.5146124988, 0.3791316964, 0.5190619748, -0.4904593222, 1.1656456469, -0.9776307444, -0.6322198963, -0.6891477694, 0.1783317857, 0.1397992467, -0.3554220649, 0.4394217877, 1.4588397512, -0.3886829615, 0.4353492889, 1.7420053483, -0.4434654615, -0.1515747891, -0.2668451725, -1.4549617756]</td>
      <td>[0.0014974177]</td>
      <td>0</td>
      <td>[1.0]</td>
      <td>[0.0005066991]</td>
    </tr>
  </tbody>
</table>

### View Logs Per Model

Another way of displaying the logs would be to specify the model.

For Arrow enabled Wallaroo instances the model outputs are listed by column.  The output data is set by the term `out`, followed by the name of the model.  For the default model, this is `out.dense_1`, while the shadow deployed models are in the format `out_{model name}.variable`, where `{model name}` is the name of the shadow deployed model.

For arrow disabled Wallaroo instances, to view the inputs and results for the shadow deployed models, use the pipeline `logs_shadow_deploy()` method.  The results will be grouped by the inputs.

```python
logs = pipeline.logs()
display(logs)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.dense_1</th>
      <th>check_failures</th>
      <th>out_ccfraud-rfeonn.variable</th>
      <th>out_ccfraud-xgbeonn.variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-05-19 15:14:08.898</td>
      <td>[1.0678324729, 0.2177810266, -1.7115145262, 0.682285721, 1.0138553067, -0.4335000013, 0.7395859437, -0.2882839595, -0.447262688, 0.5146124988, 0.3791316964, 0.5190619748, -0.4904593222, 1.1656456469, -0.9776307444, -0.6322198963, -0.6891477694, 0.1783317857, 0.1397992467, -0.3554220649, 0.4394217877, 1.4588397512, -0.3886829615, 0.4353492889, 1.7420053483, -0.4434654615, -0.1515747891, -0.2668451725, -1.4549617756]</td>
      <td>[0.0014974177]</td>
      <td>0</td>
      <td>[1.0]</td>
      <td>[0.0005066991]</td>
    </tr>
  </tbody>
</table>

### Undeploy the Pipeline

With the tutorial complete, we undeploy the pipeline and return the resources back to the system.

```python
pipeline.undeploy()
```

<table><tr><th>name</th> <td>ccshadoweonn</td></tr><tr><th>created</th> <td>2023-05-19 15:13:48.963815+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-19 15:13:53.060392+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>c6c28139-dd9d-4fdf-b7fc-d391bae58bc8, 08f4c75f-3e61-48d6-ac76-38e6dddcfaf6</td></tr><tr><th>steps</th> <td>ccfraud-lstmeonn</td></tr></table>

