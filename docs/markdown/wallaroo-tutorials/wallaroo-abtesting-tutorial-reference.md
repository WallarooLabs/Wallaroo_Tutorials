This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-testing-tutorials/abtesting).

## A/B Testing

A/B testing is a method that provides the ability to test out ML models for performance, accuracy or other useful benchmarks.  A/B testing is contrasted with the Wallaroo Shadow Deployment feature.  In both cases, two sets of models are added to a pipeline step:

* Control or Champion model:  The model currently used for inferences.
* Challenger model(s): One or more models that are to be compared to the champion model.

The two feature are different in this way:

| Feature | Description |
|---|---|
| A/B Testing | A subset of inferences are submitted to either the champion ML model or a challenger ML model. |
| Shadow Deploy | All inferences are submitted to the champion model and one or more challenger models. |

So to repeat:  A/B testing submits *some* of the inference requests to the champion model, some to the challenger model with one set of outputs, while shadow testing submits *all* of the inference requests to champion and shadow models, and has separate outputs.

This tutorial demonstrate how to conduct A/B testing in Wallaroo.  For this example we will be using an open source model that uses an [Aloha CNN LSTM model](https://www.researchgate.net/publication/348920204_Using_Auxiliary_Inputs_in_Deep_Learning_Models_for_Detecting_DGA-based_Domain_Names) for classifying Domain names as being either legitimate or being used for nefarious purposes such as malware distribution.  

For our example, we will perform the following:

* Create a workspace for our work.
* Upload the Aloha model and a challenger model.
* Create a pipeline that can ingest our submitted data with the champion model and the challenger model set into a A/B step.
* Run a series of sample inferences to display inferences that are run through the champion model versus the challenger model, then determine which is more efficient.

All sample data and models are available through the [Wallaroo Quick Start Guide Samples repository](https://github.com/WallarooLabs/quickstartguide_samples).

## Prerequisites

* A deployed Wallaroo instance
* The following Python libraries installed:
  * `os`
  * `json`
  * [`wallaroo`](https://pypi.org/project/wallaroo/): The Wallaroo SDK. Included with the Wallaroo JupyterHub service by default.
  * [`pandas`](https://pypi.org/project/pandas/): Pandas, mainly used for Pandas DataFrame

## Steps

### Import libraries

Here we will import the libraries needed for this notebook.

```python
import wallaroo
from wallaroo.object import EntityNotFoundError
import os
import pandas as pd
import json
from IPython.display import display

# used to display dataframe information without truncating
from IPython.display import display
pd.set_option('display.max_colwidth', None)
```

```python
wallaroo.__version__
```

    '2023.2.0rc3'

### Connect to the Wallaroo Instance

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  If logging in externally, update the `wallarooPrefix` and `wallarooSuffix` variables with the proper DNS information.  For more information on Wallaroo DNS settings, see the [Wallaroo DNS Integration Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-dns-guide/).

```python
# Login through local Wallaroo instance

wl = wallaroo.Client()
```

### Create Workspace

We will create a workspace to manage our pipeline and models.  The following variables will set the name of our sample workspace then set it as the current workspace for all other commands.

To allow this tutorial to be run multiple times or by multiple users in the same Wallaroo instance, a random 4 character prefix will be added to the workspace, pipeline, and model.

```python
workspace_name = 'abtesting'
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
```

```python
workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)
```

    {'name': 'abtesting', 'id': 33, 'archived': False, 'created_by': '028c8b48-c39b-4578-9110-0b5bdd3824da', 'created_at': '2023-05-18T13:55:21.887136+00:00', 'models': [], 'pipelines': []}

### Set Up the Champion and Challenger Models

Now we upload the Champion and Challenger models to our workspace.  We will use two models:

1. `aloha-cnn-lstm` model.
2. `aloha-cnn-lstm-new` (a retrained version)

### Set the Champion Model

We upload our champion model, labeled as `control`.

```python
control =  wl.upload_model("aloha-control",   'models/aloha-cnn-lstm.zip').configure('tensorflow')
```

### Set the Challenger Model

Now we upload the Challenger model, labeled as `challenger`.

```python
challenger = wl.upload_model("aloha-challenger",   'models/aloha-cnn-lstm-new.zip').configure('tensorflow')
```

### Define The Pipeline

Here we will configure a pipeline with two models and set the control model with a random split chance of receiving 2/3 of the data.  Because this is a random split, it is possible for one model or the other to receive more inferences than a strict 2:1 ratio, but the more inferences are run, the more likely it is for the proper ratio split.

```python
pipeline = (wl.build_pipeline("randomsplitpipeline-demo")
            .add_random_split([(2, control), (1, challenger)], "session_id"))
```

### Deploy the pipeline

Now we deploy the pipeline so we can run our inference through it.

```python
experiment_pipeline = pipeline.deploy()
```

```python
experiment_pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.3.161',
       'name': 'engine-66cbb56b67-4j46k',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'randomsplitpipeline-demo',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'aloha-control',
          'version': '7e5d3218-f7ad-4f08-9984-e1a459f6bc1c',
          'sha': 'fd998cd5e4964bbbb4f8d29d245a8ac67df81b62be767afbceb96a03d1a01520',
          'status': 'Running'},
         {'name': 'aloha-challenger',
          'version': 'dcdd8ef9-e30a-4785-ac91-06bc396487ec',
          'sha': '223d26869d24976942f53ccb40b432e8b7c39f9ffcf1f719f3929d7595bceaf3',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.4.194',
       'name': 'engine-lb-584f54c899-ks6s8',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}

# Run a single inference
Now we have our deployment set up let's run a single inference. In the results we will be able to see the inference results as well as which model the inference went to under model_id.  We'll run the inference request 5 times, with the odds are that the challenger model being run at least once.

```python
results = []
# use dataframe JSON files
for x in range(5):
    result = experiment_pipeline.infer_from_file("data/data-1.df.json")
    display(result.loc[:,["out._model_split", "out.main"]])    
```

{{<table "table table-striped table-bordered" >}}
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out._model_split</th>
      <th>out.main</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{"name":"aloha-control","version":"7e5d3218-f7ad-4f08-9984-e1a459f6bc1c","sha":"fd998cd5e4964bbbb4f8d29d245a8ac67df81b62be767afbceb96a03d1a01520"}]</td>
      <td>[0.997564]</td>
    </tr>
  </tbody>
</table>
{{</table>}}

{{<table "table table-striped table-bordered" >}}
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out._model_split</th>
      <th>out.main</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{"name":"aloha-control","version":"7e5d3218-f7ad-4f08-9984-e1a459f6bc1c","sha":"fd998cd5e4964bbbb4f8d29d245a8ac67df81b62be767afbceb96a03d1a01520"}]</td>
      <td>[0.997564]</td>
    </tr>
  </tbody>
</table>
{{</table>}}

{{<table "table table-striped table-bordered" >}}
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out._model_split</th>
      <th>out.main</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{"name":"aloha-challenger","version":"dcdd8ef9-e30a-4785-ac91-06bc396487ec","sha":"223d26869d24976942f53ccb40b432e8b7c39f9ffcf1f719f3929d7595bceaf3"}]</td>
      <td>[0.997564]</td>
    </tr>
  </tbody>
</table>
{{</table>}}

{{<table "table table-striped table-bordered" >}}
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out._model_split</th>
      <th>out.main</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{"name":"aloha-control","version":"7e5d3218-f7ad-4f08-9984-e1a459f6bc1c","sha":"fd998cd5e4964bbbb4f8d29d245a8ac67df81b62be767afbceb96a03d1a01520"}]</td>
      <td>[0.997564]</td>
    </tr>
  </tbody>
</table>
{{</table>}}

{{<table "table table-striped table-bordered" >}}
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out._model_split</th>
      <th>out.main</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{"name":"aloha-challenger","version":"dcdd8ef9-e30a-4785-ac91-06bc396487ec","sha":"223d26869d24976942f53ccb40b432e8b7c39f9ffcf1f719f3929d7595bceaf3"}]</td>
      <td>[0.997564]</td>
    </tr>
  </tbody>
</table>
{{</table>}}

### Run Inference Batch

We will submit 1000 rows of test data through the pipeline, then loop through the responses and display which model each inference was performed in.  The results between the control and challenger should be approximately 2:1.

```python
responses = []
test_data = pd.read_json('data/data-1k.df.json')
# For each row, submit that row as a separate dataframe
# Add the results to the responses array
for index, row in test_data.head(1000).iterrows():
    responses.append(experiment_pipeline.infer(row.to_frame('text_input').reset_index()))

#now get our responses for each row
l = [json.loads(r.loc[0]["out._model_split"][0])["name"] for r in responses]
df = pd.DataFrame({'model': l})
display(df.model.value_counts())
```

    aloha-control       666
    aloha-challenger    334
    Name: model, dtype: int64

### Test Challenger

Now we have run a large amount of data we can compare the results.

For this experiment we are looking for a significant change in the fraction of inferences that predicted a probability of the seventh category being high than 0.5 so we can determine whether our challenger model is more "successful" than the champion model at identifying category 7.

```python
control_count = 0
challenger_count = 0

control_success = 0
challenger_success = 0

for r in responses:
    if json.loads(r.loc[0]["out._model_split"][0])["name"] == "aloha-control":
        control_count += 1
        if(r.loc[0]["out.main"][0] > .5):
            control_success += 1
    else:
        challenger_count += 1
        if(r.loc[0]["out.main"][0] > .5):
            challenger_success += 1

print("control class 7 prediction rate: " + str(control_success/control_count))
print("challenger class 7 prediction rate: " + str(challenger_success/challenger_count))
```

    control class 7 prediction rate: 0.972972972972973
    challenger class 7 prediction rate: 0.9850299401197605

### Logs

Logs can be viewed with the Pipeline method `logs()`.  For this example, only the first 5 logs will be shown.  For Arrow enabled environments, the model type can be found in the column `out._model_split`.

```python
logs = experiment_pipeline.logs(limit=5)
display(logs.loc[:,['time', 'out._model_split', 'out.main']])
```

    Warning: There are more logs available. Please set a larger limit or request a file using export_logs.

{{<table "table table-striped table-bordered" >}}
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out._model_split</th>
      <th>out.main</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-05-18 14:02:08.525</td>
      <td>[{"name":"aloha-challenger","version":"dcdd8ef9-e30a-4785-ac91-06bc396487ec","sha":"223d26869d24976942f53ccb40b432e8b7c39f9ffcf1f719f3929d7595bceaf3"}]</td>
      <td>[0.99999803]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-05-18 14:02:08.141</td>
      <td>[{"name":"aloha-control","version":"7e5d3218-f7ad-4f08-9984-e1a459f6bc1c","sha":"fd998cd5e4964bbbb4f8d29d245a8ac67df81b62be767afbceb96a03d1a01520"}]</td>
      <td>[0.9998954]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-05-18 14:02:07.758</td>
      <td>[{"name":"aloha-control","version":"7e5d3218-f7ad-4f08-9984-e1a459f6bc1c","sha":"fd998cd5e4964bbbb4f8d29d245a8ac67df81b62be767afbceb96a03d1a01520"}]</td>
      <td>[0.66066873]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-05-18 14:02:07.374</td>
      <td>[{"name":"aloha-control","version":"7e5d3218-f7ad-4f08-9984-e1a459f6bc1c","sha":"fd998cd5e4964bbbb4f8d29d245a8ac67df81b62be767afbceb96a03d1a01520"}]</td>
      <td>[0.9999727]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-05-18 14:02:07.021</td>
      <td>[{"name":"aloha-challenger","version":"dcdd8ef9-e30a-4785-ac91-06bc396487ec","sha":"223d26869d24976942f53ccb40b432e8b7c39f9ffcf1f719f3929d7595bceaf3"}]</td>
      <td>[0.9999754]</td>
    </tr>
  </tbody>
</table>
{{</table>}}

### Undeploy Pipeline

With the testing complete, we undeploy the pipeline to return the resources back to the environment.

```python
experiment_pipeline.undeploy()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>randomsplitpipeline-demo</td></tr><tr><th>created</th> <td>2023-05-18 13:55:25.914690+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-18 13:55:27.144796+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>6350d3ee-8b11-4eac-a8f5-e32659ea0dd2, 170fb233-5b26-492a-ba86-e2ee72129d16</td></tr><tr><th>steps</th> <td>aloha-control</td></tr></table>
{{</table>}}

