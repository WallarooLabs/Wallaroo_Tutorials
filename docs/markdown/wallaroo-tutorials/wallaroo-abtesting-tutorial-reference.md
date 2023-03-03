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

    '2023.1.0+ca09d8a5'

### Arrow Support

As of the 2023.1 release, Wallaroo provides support for dataframe and Arrow for inference inputs.  This tutorial allows users to adjust their experience based on whether they have enabled Arrow support in their Wallaroo instance or not.

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

### Connect to the Wallaroo Instance

This command will be used to set up a connection to the Wallaroo cluster and allow creating and use of Wallaroo inference engines.

```python
# Client connection from local Wallaroo instance

wl = wallaroo.Client()

# SSO login through keycloak

# wallarooPrefix = "YOUR PREFIX"
# wallarooSuffix = "YOUR SUFFIX"

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

    {'name': 'abtesting', 'id': 137, 'archived': False, 'created_by': '138bd7e6-4dc8-4dc1-a760-c9e721ef3c37', 'created_at': '2023-03-03T18:58:38.546346+00:00', 'models': [{'name': 'aloha-control', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 3, 3, 18, 58, 39, 117617, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 3, 3, 18, 58, 39, 117617, tzinfo=tzutc())}, {'name': 'aloha-challenger', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 3, 3, 18, 58, 39, 431462, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 3, 3, 18, 58, 39, 431462, tzinfo=tzutc())}], 'pipelines': [{'name': 'randomsplitpipeline-demo', 'create_time': datetime.datetime(2023, 3, 3, 18, 58, 39, 517955, tzinfo=tzutc()), 'definition': '[]'}]}

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

    Waiting for deployment - this will take up to 45s .................................. ok

```python
experiment_pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.48.4.199',
       'name': 'engine-5df76d6869-t6nhk',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'randomsplitpipeline-demo',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'aloha-challenger',
          'version': '3acd3835-be72-42c4-bcae-84368f416998',
          'sha': '223d26869d24976942f53ccb40b432e8b7c39f9ffcf1f719f3929d7595bceaf3',
          'status': 'Running'},
         {'name': 'aloha-control',
          'version': '89389786-0c17-4214-938c-aa22dd28359f',
          'sha': 'fd998cd5e4964bbbb4f8d29d245a8ac67df81b62be767afbceb96a03d1a01520',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.48.4.198',
       'name': 'engine-lb-86bc6bd77b-j7pqs',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}

# Run a single inference
Now we have our deployment set up let's run a single inference. In the results we will be able to see the inference results as well as which model the inference went to under model_id.  We'll run the inference request 5 times, with the odds are that the challenger model being run at least once.

```python
results = []
if arrowEnabled is True:
    # use dataframe JSON files
    for x in range(5):
        result = experiment_pipeline.infer_from_file("data/data-1.df.json")
        display(result.loc[:,["out._model_split", "out.main"]])    
else:
    # use Wallaroo JSON files
    results.append(experiment_pipeline.infer_from_file("data/data-1.json"))
    results.append(experiment_pipeline.infer_from_file("data/data-1.json"))
    results.append(experiment_pipeline.infer_from_file("data/data-1.json"))
    results.append(experiment_pipeline.infer_from_file("data/data-1.json"))
    results.append(experiment_pipeline.infer_from_file("data/data-1.json"))
    for result in results:
        print(result[0].model())
        print(result[0].data())
```

{{<table "table table-bordered">}}
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
      <td>[{"name":"aloha-control","version":"89389786-0c17-4214-938c-aa22dd28359f","sha":"fd998cd5e4964bbbb4f8d29d245a8ac67df81b62be767afbceb96a03d1a01520"}]</td>
      <td>[0.997564]</td>
    </tr>
  </tbody>
</table>
{{</table>}}


{{<table "table table-bordered">}}
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
      <td>[{"name":"aloha-control","version":"89389786-0c17-4214-938c-aa22dd28359f","sha":"fd998cd5e4964bbbb4f8d29d245a8ac67df81b62be767afbceb96a03d1a01520"}]</td>
      <td>[0.997564]</td>
    </tr>
  </tbody>
</table>
{{</table>}}


{{<table "table table-bordered">}}
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
      <td>[{"name":"aloha-control","version":"89389786-0c17-4214-938c-aa22dd28359f","sha":"fd998cd5e4964bbbb4f8d29d245a8ac67df81b62be767afbceb96a03d1a01520"}]</td>
      <td>[0.997564]</td>
    </tr>
  </tbody>
</table>
{{</table>}}


{{<table "table table-bordered">}}
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
      <td>[{"name":"aloha-challenger","version":"3acd3835-be72-42c4-bcae-84368f416998","sha":"223d26869d24976942f53ccb40b432e8b7c39f9ffcf1f719f3929d7595bceaf3"}]</td>
      <td>[0.997564]</td>
    </tr>
  </tbody>
</table>
{{</table>}}


{{<table "table table-bordered">}}
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
      <td>[{"name":"aloha-control","version":"89389786-0c17-4214-938c-aa22dd28359f","sha":"fd998cd5e4964bbbb4f8d29d245a8ac67df81b62be767afbceb96a03d1a01520"}]</td>
      <td>[0.997564]</td>
    </tr>
  </tbody>
</table>
{{</table>}}


### Run Inference Batch

We will submit 1000 rows of test data through the pipeline, then loop through the responses and display which model each inference was performed in.  The results between the control and challenger should be approximately 2:1.

```python
responses = []
if arrowEnabled is True:
    # responses = pd.DataFrame()
    #Read in the test data as one dataframe
    test_data = pd.read_json('data/data-1k.df.json')
    # For each row, submit that row as a separate dataframe
    # Add the results to the responses array
    for index, row in test_data.head(1000).iterrows():
        responses.append(experiment_pipeline.infer(row.to_frame('text_input').reset_index()))
        # display(responses)
    #now get our responses for each row
    # each r is a dataframe, then get the result from out.split into json and get the model name
    # for r in responses:
    #     display(r.loc[0]["out.split"])
    l = [json.loads(r.loc[0]["out._model_split"][0])["name"] for r in responses]
    df = pd.DataFrame({'model': l})
    display(df.model.value_counts())
else:
    l = []
    responses =[]
    from data import test_data
    for nth in range(1000):
        responses.extend(experiment_pipeline.infer(test_data.data[nth]))
    l = [r.raw['model_name'] for r in responses]
    df = pd.DataFrame({'model': l})
    display(df.model.value_counts())
```

    aloha-control       656
    aloha-challenger    344
    Name: model, dtype: int64

### Test Challenger

Now we have run a large amount of data we can compare the results.

For this experiment we are looking for a significant change in the fraction of inferences that predicted a probability of the seventh category being high than 0.5 so we can determine whether our challenger model is more "successful" than the champion model at identifying category 7.

```python
control_count = 0
challenger_count = 0

control_success = 0
challenger_success = 0

if arrowEnabled is True:
    # do nothing
    for r in responses:
        if json.loads(r.loc[0]["out._model_split"][0])["name"] == "aloha-control":
            control_count += 1
            if(r.loc[0]["out.main"][0] > .5):
                control_success += 1
        else:
            challenger_count += 1
            if(r.loc[0]["out.main"][0] > .5):
               challenger_success += 1
else:
    for r in responses:
        if r.raw['model_name'] == "aloha-control":
            control_count += 1
            if(r.raw['outputs'][7]['Float']['data'][0] > .5):
                control_success += 1
        else:
            challenger_count +=1
            if(r.raw['outputs'][7]['Float']['data'][0] > .5):
                challenger_success += 1

print("control class 7 prediction rate: " + str(control_success/control_count))
print("challenger class 7 prediction rate: " + str(challenger_success/challenger_count))
```

    control class 7 prediction rate: 0.9725609756097561
    challenger class 7 prediction rate: 0.9854651162790697

### Logs

Logs can be viewed with the Pipeline method `logs()`.  For this example, only the first 5 logs will be shown.  For Arrow enabled environments, the model type can be found in the column `out._model_split`.

```python
logs = experiment_pipeline.logs(limit=5)

if arrowEnabled is True:
    display(logs.loc[:,['time', 'out._model_split', 'out.main']])
else:
    display(logs)
```

{{<table "table table-bordered">}}
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
      <td>2023-03-03 19:08:35.653</td>
      <td>[{"name":"aloha-control","version":"89389786-0c17-4214-938c-aa22dd28359f","sha":"fd998cd5e4964bbbb4f8d29d245a8ac67df81b62be767afbceb96a03d1a01520"}]</td>
      <td>[0.9999754]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-03-03 19:08:35.702</td>
      <td>[{"name":"aloha-challenger","version":"3acd3835-be72-42c4-bcae-84368f416998","sha":"223d26869d24976942f53ccb40b432e8b7c39f9ffcf1f719f3929d7595bceaf3"}]</td>
      <td>[0.9999727]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-03-03 19:08:35.753</td>
      <td>[{"name":"aloha-challenger","version":"3acd3835-be72-42c4-bcae-84368f416998","sha":"223d26869d24976942f53ccb40b432e8b7c39f9ffcf1f719f3929d7595bceaf3"}]</td>
      <td>[0.6606688]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-03-03 19:08:35.799</td>
      <td>[{"name":"aloha-control","version":"89389786-0c17-4214-938c-aa22dd28359f","sha":"fd998cd5e4964bbbb4f8d29d245a8ac67df81b62be767afbceb96a03d1a01520"}]</td>
      <td>[0.9998954]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-03-03 19:08:35.846</td>
      <td>[{"name":"aloha-control","version":"89389786-0c17-4214-938c-aa22dd28359f","sha":"fd998cd5e4964bbbb4f8d29d245a8ac67df81b62be767afbceb96a03d1a01520"}]</td>
      <td>[0.99999803]</td>
    </tr>
  </tbody>
</table>
{{</table>}}


### Undeploy Pipeline

With the testing complete, we undeploy the pipeline to return the resources back to the environment.

```python
experiment_pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s .................................... ok

<table><tr><th>name</th> <td>randomsplitpipeline-demo</td></tr><tr><th>created</th> <td>2023-03-03 18:58:39.517955+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-03 19:05:41.670973+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>afa1b6ff-7684-4e58-a513-79295024a442, ff71a8fc-947a-42bd-9a97-b4317ace6a9c, 694eb936-6d94-48e3-887f-d83d9b526d84, a6beb0ac-36c8-4fff-9d45-303790cbf7fe, 66161c1f-df31-4ca1-b4ae-c7f5fb3e73bb, ddca1949-bc85-4b12-9ec8-497b9630244d</td></tr><tr><th>steps</th> <td>aloha-control</td></tr></table>
{{</table>}}

