This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/model_conversion/statsmodels).

## Introduction

Organizations can deploy a Machine Learning (ML) model based on the [statsmodels](https://www.statsmodels.org/stable/index.html) directly into Wallaroo through the following process.  This conversion process transforms the model into an open format that can be run across different frameworks at compiled C-language speeds.

This example provides the following:

* `train-statsmodel.ipynb`: A sample Jupyter Notebook that trains a sample model.  The model predicts how many bikes will be rented on each of the next 7 days, based on the previous 7 days' bike rentals, temperature, and wind speed.  Additional files to support this example are:
  * `day.csv`: Data used to train the sample `statsmodel` example.
  * `infer.py`: The inference script that is part of the `statsmodel`.
* `convert-statsmodel-tutorial.ipynb`: A sample Jupyter Notebook that demonstrates how to upload, convert, and deploy the `statsmodel` example into a Wallaroo instance.    Additional files to support this example are:
  * `bike_day_model.pkl`: A `statsmodel` ML model trained from the `train-statsmodel.ipynb` Notebook.

    **IMPORTANT NOTE:** The `statsmodel` ML model is composed of two parts that are contained in the .pkl file:

    * The pickled Python runtime expects a dictionary with two keys: `model` and `script`:

      * `model`—the pickled model, which will be automatically loaded into the python runtime with the name 'model'
      * `script`—the text of the python script to be run, in a format similar to the existing python script steps (i.e. defining a wallaroo_json method which operates on the data).  In this cae, the file `infer.py` is the script used.

  * `bike_day_eval.json`: Evaluation data used to test the model's performance.

## Steps

The following steps will perform the following:

1. Upload the `statsmodel` ML model `bike_day_model.pkl` into a Wallaroo.
2. Deploy the model into a pipeline.
3. Run a test inference.
4. Undeploy the pipeline.

### Import Libraries

The first step is to import the libraries that we will need.

```python
import json
import os
import datetime

import wallaroo
from wallaroo.object import EntityNotFoundError
```

### Initialize connection

Start a connect to the Wallaroo instance and save the connection into the variable `wl`.

```python
wl = wallaroo.Client()
```

### Set Configurations

The following will set the workspace, model name, and pipeline that will be used for this example.  If the workspace or pipeline already exist, then they will assigned for use in this example.  If they do not exist, they will be created based on the names listed below.

```python
workspace_name = 'bikedayevalworkspace'
pipeline_name = 'bikedayevalpipeline'
model_name = 'bikedaymodel'
model_file_name = 'bike_day_model.pkl'
```

## Set the Workspace and Pipeline

This sample code will create or use the existing workspace `bike-day-workspace` as the current workspace.

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

workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)

pipeline = get_pipeline(pipeline_name)
pipeline
```

<table><tr><th>name</th> <td>bike-day-evel-pipeline</td></tr><tr><th>created</th> <td>2022-07-05 19:09:22.895067+00:00</td></tr><tr><th>last_updated</th> <td>2022-07-05 19:11:16.553505+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>steps</th> <td>bike-day-model</td></tr></table>

### Upload Pickled Package Statsmodel Model

Upload the statsmodel stored into the pickled package `bike_day_model.pkl`.  See the Notebook `train-statsmodel.ipynb` for more details on creating this package.

Note that this package is being specified as a `python` configuration.

```python
file_name = "bike_day_model.pkl"

bike_day_model = wl.upload_model(model_name, model_file_name).configure(runtime="python")
```

### Deploy the Pipeline

We will now add the uploaded model as a step for the pipeline, then deploy it.

```python
pipeline.add_model_step(bike_day_model)
```

<table><tr><th>name</th> <td>bike-day-evel-pipeline</td></tr><tr><th>created</th> <td>2022-07-05 19:09:22.895067+00:00</td></tr><tr><th>last_updated</th> <td>2022-07-05 19:11:16.553505+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>steps</th> <td>bike-day-model</td></tr></table>

```python
pipeline.deploy()
```

    Waiting for deployment - this will take up to 45s ................. ok

<table><tr><th>name</th> <td>bike-day-evel-pipeline</td></tr><tr><th>created</th> <td>2022-07-05 19:09:22.895067+00:00</td></tr><tr><th>last_updated</th> <td>2022-07-05 20:10:27.589019+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>steps</th> <td>bike-day-model</td></tr></table>

```python
pipeline.status()
```

    {'status': 'Running',
     'details': None,
     'engines': [{'ip': '10.164.3.4',
       'name': 'engine-5f75f487c6-9d456',
       'status': 'Running',
       'reason': None,
       'pipeline_statuses': {'pipelines': [{'id': 'bike-day-evel-pipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'bike-day-model',
          'version': 'ff154938-4e49-468e-ac6a-4ee37d62a724',
          'sha': 'ba1fc2a6e8b876684f2fd11534ee6212f840f02cbaefaa48615016cb9e90b30c',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.164.5.61',
       'name': 'engine-lb-85846c64f8-khznn',
       'status': 'Running',
       'reason': None}]}

### Run Inference

Perform an inference from the evaluation data JSON file `bike_day_eval.json`.

```python
pipeline.infer_from_file('bike_day_eval.json')
```

    Waiting for inference response - this will take up to 45s .. ok

    [InferenceResult({'check_failures': [],
      'elapsed': 5369777,
      'model_name': 'bike-day-model',
      'model_version': 'ff154938-4e49-468e-ac6a-4ee37d62a724',
      'original_data': {'holiday': {'0': 0,
                                    '1': 0,
                                    '2': 0,
                                    '3': 0,
                                    '4': 0,
                                    '5': 0,
                                    '6': 0},
                        'temp': {'0': 0.317391,
                                 '1': 0.365217,
                                 '2': 0.415,
                                 '3': 0.54,
                                 '4': 0.4725,
                                 '5': 0.3325,
                                 '6': 0.430435},
                        'windspeed': {'0': 0.184309,
                                      '1': 0.203117,
                                      '2': 0.209579,
                                      '3': 0.231017,
                                      '4': 0.368167,
                                      '5': 0.207721,
                                      '6': 0.288783},
                        'workingday': {'0': 1,
                                       '1': 1,
                                       '2': 1,
                                       '3': 1,
                                       '4': 0,
                                       '5': 0,
                                       '6': 1}},
      'outputs': [{'Json': {'data': [{'forecast': [1882.3784554842296,
                                                   2130.607915715519,
                                                   2340.8400538168335,
                                                   2895.754978556798,
                                                   2163.65751556893,
                                                   1509.1792126536425,
                                                   2431.1838923984033]}],
                            'dim': [1],
                            'v': 1}}],
      'pipeline_name': 'bike-day-evel-pipeline',
      'time': 1657051854529})]

### Undeploy the Pipeline

Undeploy the pipeline and return the resources back to the Wallaroo instance.

```python
pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s ................................ ok

<table><tr><th>name</th> <td>bike-day-evel-pipeline</td></tr><tr><th>created</th> <td>2022-07-05 19:09:22.895067+00:00</td></tr><tr><th>last_updated</th> <td>2022-07-05 20:10:27.589019+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>steps</th> <td>bike-day-model</td></tr></table>

```python

```
