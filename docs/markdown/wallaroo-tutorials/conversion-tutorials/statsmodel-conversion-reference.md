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

### Connect to the Wallaroo Instance

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  If logging in externally, update the `wallarooPrefix` and `wallarooSuffix` variables with the proper DNS information.  For more information on Wallaroo DNS settings, see the [Wallaroo DNS Integration Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-dns-guide/).

```python
# Login through local Wallaroo instance

wl = wallaroo.Client()
```

### Set Configurations

The following will set the workspace, model name, and pipeline that will be used for this example.  If the workspace or pipeline already exist, then they will assigned for use in this example.  If they do not exist, they will be created based on the names listed below.

```python
workspace_name = 'statsmodelworkspace'
pipeline_name = 'statsmodelpipeline'
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

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>statsmodelpipeline</td></tr><tr><th>created</th> <td>2023-05-17 21:19:52.898178+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 21:19:52.898178+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>5456dd2a-3167-4b3c-ad3a-85544292a230</td></tr><tr><th>steps</th> <td></td></tr></table>
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

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>statsmodelpipeline</td></tr><tr><th>created</th> <td>2023-05-17 21:19:52.898178+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 21:19:52.898178+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>5456dd2a-3167-4b3c-ad3a-85544292a230</td></tr><tr><th>steps</th> <td></td></tr></table>
{{</table>}}

```python
pipeline.deploy()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>statsmodelpipeline</td></tr><tr><th>created</th> <td>2023-05-17 21:19:52.898178+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 21:19:55.996411+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>4af264e3-f427-4b02-b5ad-4f6690b0ee06, 5456dd2a-3167-4b3c-ad3a-85544292a230</td></tr><tr><th>steps</th> <td>bikedaymodel</td></tr></table>
{{</table>}}

```python
pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.3.141',
       'name': 'engine-c77f759f7-f7fxd',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'statsmodelpipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'bikedaymodel',
          'version': '66bf61d5-d144-4f77-82f1-58dabf2bbc33',
          'sha': '09b50a8e6a5cff566598dae6fb94f5d7d35c94e278373251cd8b1fd9a000c0a7',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.4.172',
       'name': 'engine-lb-584f54c899-67jbk',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}

### Run Inference

Perform an inference from the evaluation data JSON file `bike_day_eval.json`.

```python
results = pipeline.infer_from_file('bike_day_eval.json', data_format="custom-json")

display(results)
```

    [{'forecast': [1882.3784555157672,
       2130.607915701861,
       2340.84005381799,
       2895.754978552066,
       2163.657515565616,
       1509.1792126509536,
       2431.1838923957016]}]

### Undeploy the Pipeline

Undeploy the pipeline and return the resources back to the Wallaroo instance.

```python
pipeline.undeploy()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>statsmodelpipeline</td></tr><tr><th>created</th> <td>2023-05-17 21:19:52.898178+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 21:19:55.996411+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>4af264e3-f427-4b02-b5ad-4f6690b0ee06, 5456dd2a-3167-4b3c-ad3a-85544292a230</td></tr><tr><th>steps</th> <td>bikedaymodel</td></tr></table>
{{</table>}}

