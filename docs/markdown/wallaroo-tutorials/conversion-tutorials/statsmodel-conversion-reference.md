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

# used to display dataframe information without truncating
from IPython.display import display
import pandas as pd
pd.set_option('display.max_colwidth', None)
```

### Initialize connection

Start a connect to the Wallaroo instance and save the connection into the variable `wl`.

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

<table><tr><th>name</th> <td>bikedayevalpipeline</td></tr><tr><th>created</th> <td>2023-02-27 20:15:07.848652+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-27 20:15:07.848652+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>14a140e1-640b-4cf5-9d45-dd27fe00ad80</td></tr><tr><th>steps</th> <td></td></tr></table>
{{</table>}}

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

<table><tr><th>name</th> <td>bikedayevalpipeline</td></tr><tr><th>created</th> <td>2023-02-27 20:15:07.848652+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-27 20:15:07.848652+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>14a140e1-640b-4cf5-9d45-dd27fe00ad80</td></tr><tr><th>steps</th> <td></td></tr></table>
{{</table>}}

```python
pipeline.deploy()
```

<table><tr><th>name</th> <td>bikedayevalpipeline</td></tr><tr><th>created</th> <td>2023-02-27 20:15:07.848652+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-27 20:16:18.874015+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>00e53810-6278-463b-b1c7-6a63f25fd1ef, 91b1dd51-2adb-4003-ab68-91a8415210c1, 14a140e1-640b-4cf5-9d45-dd27fe00ad80</td></tr><tr><th>steps</th> <td>bikedaymodel</td></tr></table>
{{</table>}}

```python
pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.0.51',
       'name': 'engine-6d5cc888b-v7mhj',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'bikedayevalpipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'bikedaymodel',
          'version': 'd14c4f84-4238-49cd-9a63-ff96d4d28b24',
          'sha': '1bb486598732259efdd131b45d471e165b594d5443928ea9e50fa4c7e0b1b718',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.1.13',
       'name': 'engine-lb-ddd995646-49mdq',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}

### Run Inference

Perform an inference from the evaluation data JSON file `bike_day_eval.json`.

```python
if arrowEnabled is True:
    results = pipeline.infer_from_file('bike_day_eval.json', data_format="custom-json")
else:
    results = pipeline.infer_from_file('bike_day_eval.json')
display(results)
```

    [{'forecast': [1882.378455403016,
       2130.6079157429585,
       2340.840053800859,
       2895.754978555364,
       2163.6575155637433,
       1509.1792126514365,
       2431.183892393437]}]

### Undeploy the Pipeline

Undeploy the pipeline and return the resources back to the Wallaroo instance.

```python
pipeline.undeploy()
```

<table><tr><th>name</th> <td>bikedayevalpipeline</td></tr><tr><th>created</th> <td>2023-02-27 20:15:07.848652+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-27 20:15:11.565861+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>91b1dd51-2adb-4003-ab68-91a8415210c1, 14a140e1-640b-4cf5-9d45-dd27fe00ad80</td></tr><tr><th>steps</th> <td>bikedaymodel</td></tr></table>
{{</table>}}

