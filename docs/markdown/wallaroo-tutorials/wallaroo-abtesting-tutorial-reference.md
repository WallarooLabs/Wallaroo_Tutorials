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

import os
# Used for Wallaroo SDK 2023.1
os.environ["ARROW_ENABLED"]="True"

# used to display dataframe information without truncating
from IPython.display import display
pd.set_option('display.max_colwidth', None)
```


```python
wallaroo.__version__
```




    '2023.1.0'



### Connect to the Wallaroo Instance

This command will be used to set up a connection to the Wallaroo cluster and allow creating and use of Wallaroo inference engines.


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




    {'name': 'abtesting', 'id': 6, 'archived': False, 'created_by': '56d97480-bb64-4575-acb6-f93d05652e86', 'created_at': '2023-03-27T21:23:15.213957+00:00', 'models': [], 'pipelines': []}



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
     'engines': [{'ip': '10.244.3.22',
       'name': 'engine-74478bbd79-q2bpj',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'randomsplitpipeline-demo',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'aloha-control',
          'version': 'a84a7111-8a0e-47fd-9fbc-e4f2b81d3cc2',
          'sha': 'fd998cd5e4964bbbb4f8d29d245a8ac67df81b62be767afbceb96a03d1a01520',
          'status': 'Running'},
         {'name': 'aloha-challenger',
          'version': 'ff920bbc-68e8-4d46-a837-cb6419a8f4aa',
          'sha': '223d26869d24976942f53ccb40b432e8b7c39f9ffcf1f719f3929d7595bceaf3',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.0.172',
       'name': 'engine-lb-ddd995646-np695',
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
      <th>out._model_split</th>
      <th>out.main</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{"name":"aloha-control","version":"a84a7111-8a0e-47fd-9fbc-e4f2b81d3cc2","sha":"fd998cd5e4964bbbb4f8d29d245a8ac67df81b62be767afbceb96a03d1a01520"}]</td>
      <td>[0.997564]</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>out._model_split</th>
      <th>out.main</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{"name":"aloha-control","version":"a84a7111-8a0e-47fd-9fbc-e4f2b81d3cc2","sha":"fd998cd5e4964bbbb4f8d29d245a8ac67df81b62be767afbceb96a03d1a01520"}]</td>
      <td>[0.997564]</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>out._model_split</th>
      <th>out.main</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{"name":"aloha-control","version":"a84a7111-8a0e-47fd-9fbc-e4f2b81d3cc2","sha":"fd998cd5e4964bbbb4f8d29d245a8ac67df81b62be767afbceb96a03d1a01520"}]</td>
      <td>[0.997564]</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>out._model_split</th>
      <th>out.main</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{"name":"aloha-challenger","version":"ff920bbc-68e8-4d46-a837-cb6419a8f4aa","sha":"223d26869d24976942f53ccb40b432e8b7c39f9ffcf1f719f3929d7595bceaf3"}]</td>
      <td>[0.997564]</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>out._model_split</th>
      <th>out.main</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{"name":"aloha-challenger","version":"ff920bbc-68e8-4d46-a837-cb6419a8f4aa","sha":"223d26869d24976942f53ccb40b432e8b7c39f9ffcf1f719f3929d7595bceaf3"}]</td>
      <td>[0.997564]</td>
    </tr>
  </tbody>
</table>
</div>


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


    aloha-control       677
    aloha-challenger    323
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

    control class 7 prediction rate: 0.9807976366322009
    challenger class 7 prediction rate: 0.9690402476780186


### Logs

Logs can be viewed with the Pipeline method `logs()`.  For this example, only the first 5 logs will be shown.  For Arrow enabled environments, the model type can be found in the column `out._model_split`.


```python
logs = experiment_pipeline.logs(limit=5)
display(logs.loc[:,['time', 'out._model_split', 'out.main']])
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
      <th>out._model_split</th>
      <th>out.main</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-03-27 21:34:45.939</td>
      <td>[{"name":"aloha-control","version":"a84a7111-8a0e-47fd-9fbc-e4f2b81d3cc2","sha":"fd998cd5e4964bbbb4f8d29d245a8ac67df81b62be767afbceb96a03d1a01520"}]</td>
      <td>[0.9999754]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-03-27 21:34:46.362</td>
      <td>[{"name":"aloha-challenger","version":"ff920bbc-68e8-4d46-a837-cb6419a8f4aa","sha":"223d26869d24976942f53ccb40b432e8b7c39f9ffcf1f719f3929d7595bceaf3"}]</td>
      <td>[0.9999727]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-03-27 21:34:46.779</td>
      <td>[{"name":"aloha-challenger","version":"ff920bbc-68e8-4d46-a837-cb6419a8f4aa","sha":"223d26869d24976942f53ccb40b432e8b7c39f9ffcf1f719f3929d7595bceaf3"}]</td>
      <td>[0.66066873]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-03-27 21:34:47.222</td>
      <td>[{"name":"aloha-control","version":"a84a7111-8a0e-47fd-9fbc-e4f2b81d3cc2","sha":"fd998cd5e4964bbbb4f8d29d245a8ac67df81b62be767afbceb96a03d1a01520"}]</td>
      <td>[0.9998954]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-03-27 21:34:47.642</td>
      <td>[{"name":"aloha-challenger","version":"ff920bbc-68e8-4d46-a837-cb6419a8f4aa","sha":"223d26869d24976942f53ccb40b432e8b7c39f9ffcf1f719f3929d7595bceaf3"}]</td>
      <td>[0.99999803]</td>
    </tr>
  </tbody>
</table>
</div>


### Undeploy Pipeline

With the testing complete, we undeploy the pipeline to return the resources back to the environment.


```python
experiment_pipeline.undeploy()
```




<table><tr><th>name</th> <td>randomsplitpipeline-demo</td></tr><tr><th>created</th> <td>2023-03-27 21:23:19.527680+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-27 21:23:21.497076+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>8b418929-be2d-4f62-8ea5-b8f294df37b8, 60210f85-df35-433b-ba7c-d9231ebf83c9</td></tr><tr><th>steps</th> <td>aloha-control</td></tr></table>




```python

```
