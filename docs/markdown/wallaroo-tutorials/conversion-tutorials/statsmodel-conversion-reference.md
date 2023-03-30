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

### Prerequisites

Before uploading and running an inference with a MLFlow model in Wallaroo the following will be required:

* An installed Wallaroo instance.
* The following Python libraries installed:
  * `os`
  * [`wallaroo`](https://pypi.org/project/wallaroo/): The Wallaroo SDK. Included with the Wallaroo JupyterHub service by default.
  

## Steps

The following steps will perform the following:
    
1. Upload the `statsmodel` ML model `bike_day_model.pkl` into a Wallaroo.
2. Deploy the model into a pipeline.
3. Run a test inference.
4. Undeploy the pipeline.

### Import Libraries

The first step is to import the libraries that we will need.


```python
import wallaroo
from wallaroo.object import EntityNotFoundError

import os
# Only set the below to make the OS environment ARROW_ENABLED to TRUE.  Otherwise, leave as is.
os.environ["ARROW_ENABLED"]="True"
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




<table><tr><th>name</th> <td>bikedayevalpipeline</td></tr><tr><th>created</th> <td>2023-03-27 22:05:04.176546+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-27 22:05:07.911187+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>27aac448-908f-45fb-aafa-f473652f9eb2, 450c7027-ffd7-4155-a27e-760e0f0c95f1</td></tr><tr><th>steps</th> <td>bikedaymodel</td></tr></table>



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




<table><tr><th>name</th> <td>bikedayevalpipeline</td></tr><tr><th>created</th> <td>2023-03-27 22:05:04.176546+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-27 22:05:07.911187+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>27aac448-908f-45fb-aafa-f473652f9eb2, 450c7027-ffd7-4155-a27e-760e0f0c95f1</td></tr><tr><th>steps</th> <td>bikedaymodel</td></tr></table>




```python
pipeline.deploy()
```




<table><tr><th>name</th> <td>bikedayevalpipeline</td></tr><tr><th>created</th> <td>2023-03-27 22:05:04.176546+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-29 15:57:45.551386+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>3903bff8-e4e2-4215-88fc-3a19eb074f37, 27aac448-908f-45fb-aafa-f473652f9eb2, 450c7027-ffd7-4155-a27e-760e0f0c95f1</td></tr><tr><th>steps</th> <td>bikedaymodel</td></tr></table>




```python
pipeline.status()
```




    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.0.13',
       'name': 'engine-779758b49f-rgskg',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'bikedayevalpipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'bikedaymodel',
          'version': 'b500a872-1cb3-4780-89ba-5523f1c5d465',
          'sha': '06bf64d8e190431925e1ecde5bfa110b92b738b0ce73cd866cbcdb5190d26915',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.3.9',
       'name': 'engine-lb-ddd995646-4twrz',
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




<table><tr><th>name</th> <td>bikedayevalpipeline</td></tr><tr><th>created</th> <td>2023-03-27 22:05:04.176546+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-29 15:57:45.551386+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>3903bff8-e4e2-4215-88fc-3a19eb074f37, 27aac448-908f-45fb-aafa-f473652f9eb2, 450c7027-ffd7-4155-a27e-760e0f0c95f1</td></tr><tr><th>steps</th> <td>bikedaymodel</td></tr></table>


