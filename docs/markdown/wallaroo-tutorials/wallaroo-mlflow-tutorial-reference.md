This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-model-cookbooks/mlflow-tutorial).

## MLFlow Inference with Wallaroo Tutorial 

Wallaroo users can register their trained [MLFlow ML Models](https://www.mlflow.org/docs/latest/models.html) from a containerized model container registry into their Wallaroo instance and perform inferences with it through a Wallaroo pipeline.

As of this time, Wallaroo only supports **MLFlow 1.3.0** containerized models.  For information on how to containerize an MLFlow model, see the [MLFlow Documentation](https://mlflow.org/docs/latest/projects.html).

This tutorial assumes that you have a Wallaroo instance, and have either your own containerized model or use the one from the reference and are running this Notebook from the Wallaroo Jupyter Hub service.

See the [Wallaroo Private Containerized Model Container Registry Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-private-model-registry/) for details on how to configure a Wallaroo instance with a private model registry.

## MLFlow Data Formats

When using containerized MLFlow models with Wallaroo, the inputs and outputs must be named.  For example, the following output:

```json
[-12.045839810372835]
```

Would need to be wrapped with the data values named:

```json
[{"prediction": -12.045839810372835}]
```

A short sample code for wrapping data may be:

```python
output_df = pd.DataFrame(prediction, columns=["prediction"])
return output_df
```

### MLFlow Models and Wallaroo

MLFlow models are composed of two parts:  the model, and the flavors.  When submitting a MLFlow model to Wallaroo, both aspects must be part of the ML Model included in the container.  For full information about MLFlow model structure, see the [MLFlow Documentation](https://www.mlflow.org/docs/latest/index.html).

Wallaroo registers the models from container registries.  Organizations will either have to make their containers available in a public or through a private Containerized Model Container Registry service.  For examples on setting up a private container registry service, see the [Docker Documentation "Deploy a registry server"](https://docs.docker.com/registry/deploying/).  For more details on setting up a container registry in a cloud environment, see the related documentation for your preferred cloud provider:
  * [Google Cloud Platform Container Registry](https://cloud.google.com/container-registry)
  * [Amazon Web Services Elastic Container Registry](https://docs.aws.amazon.com/AmazonECR/latest/userguide/what-is-ecr.html)
  *  [Microsoft Azure Container Registry](https://azure.microsoft.com/en-us/free/container-registry/)

For this example, we will be using the MLFlow containers that was registered in a GitHub container registry service in MLFlow Creation Tutorial Part 03: Container Registration.  The address of those containers are:

* postprocess: ghcr.io/johnhansarickwallaroo/mlflowtests/mlflow-postprocess-example .  Used for format data after the statsmodel inferences.
* statsmodel: ghcr.io/johnhansarickwallaroo/mlflowtests/mlflow-statsmodels-example . The statsmodel generated in MLFlow Creation Tutorial Part 01: Model Creation.

### Prerequisites

Before uploading and running an inference with a MLFlow model in Wallaroo the following will be required:

* **MLFlow Input Schema**:  The input schema with the fields and data types for each MLFlow model type uploaded to Wallaroo.  In the examples below, the data types are imported using the `pyarrow` library.
* A Wallaroo instance version 2023.1 or later.

**IMPORTANT NOTE**:  Wallaroo supports MLFlow 1.3.0.  Please ensure the MLFlow models used in Wallaroo meet this specification.

## MLFlow Inference Steps

To register a containerized MLFlow ML Model into Wallaroo, use the following general step:

* Import Libraries
* Connect to Wallaroo
* Set MLFlow Input Schemas
* Register MLFlow Model
* Create Pipeline and Add Model Steps
* Run Inference

### Import Libraries

We start by importing the libraries we will need to connect to Wallaroo and use our MLFlow models. This includes the `wallaroo` libraries, `pyarrow` for data types, and the `json` library for handling JSON data.


```python
import uuid
import json

import numpy as np

import wallaroo
from wallaroo.object import EntityNotFoundError

import pyarrow as pa
```

### Connect to Wallaroo

Connect to Wallaroo and store the connection in the variable `wl`.

The folowing methods are used to create the workspace and pipeline for this tutorial.  A workspace is created and set as the current workspace that will contain the registered models and pipelines.


```python
# Login through local service

# wl = wallaroo.Client()

# SSO login through keycloak

wallarooPrefix = "YOUR PREFIX"
wallarooSuffix = "YOUR SUFFIX"

wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}.api.{wallarooSuffix}", 
                auth_endpoint=f"https://{wallarooPrefix}.keycloak.{wallarooSuffix}", 
                auth_type="sso")
```

### Arrow Support

As of the 2023.1 release, Wallaroo provides support for DataFrame and Arrow for inference inputs.  This tutorial allows users to adjust their experience based on whether they have enabled Arrow support in their Wallaroo instance or not.

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


```python
prefix = 'mlflow'
workspace_name= f"{prefix}statsmodelworkspace"
pipeline_name = f"{prefix}statsmodelpipeline"

mlflowworkspace = get_workspace(workspace_name)
wl.set_current_workspace(mlflowworkspace)


pipeline = get_pipeline(pipeline_name)
```

### Set MLFlow Input Schemas

Set the MLFlow input schemas through the `pyarrow` library.  In the examples below, the input schemas for both the MLFlow model `statsmodels-test` and the `statsmodels-test-postprocess` model.


```python
sm_input_schema = pa.schema([
  pa.field('temp', pa.float32()),
  pa.field('holiday', pa.uint8()),
  pa.field('workingday', pa.uint8()),
  pa.field('windspeed', pa.float32())
])

pp_input_schema = pa.schema([
    pa.field('predicted_mean', pa.float32())
])
```

### Register MLFlow Model

Use the `register_model_image` method to register the Docker container containing the MLFlow models.


```python
statsmodelUrl = "ghcr.io/wallaroolabs/wallaroo_tutorials/mlflow-statsmodels-example:2022.4"
postprocessUrl = "ghcr.io/wallaroolabs/wallaroo_tutorials/mlflow-postprocess-example:2022.4"

sm_model = wl.register_model_image(
    name=f"{prefix}statmodels",
    image=f"{statsmodelUrl}"
).configure("mlflow", input_schema=sm_input_schema, output_schema=pp_input_schema)
pp_model = wl.register_model_image(
    name=f"{prefix}postprocess",
    image=f"{postprocessUrl}"
).configure("mlflow", input_schema=pp_input_schema, output_schema=pp_input_schema)
```

### Create Pipeline and Add Model Steps

With the models registered, we can add the MLFlow models as steps in the pipeline.  Once ready, we will deploy the pipeline so it is available for submitting data for running inferences.


```python
pipeline.add_model_step(sm_model)
pipeline.add_model_step(pp_model)
```




<table><tr><th>name</th> <td>mlflowstatsmodelpipeline</td></tr><tr><th>created</th> <td>2023-03-02 18:06:43.227930+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-02 18:06:43.227930+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>7e0bdce0-20ce-4b46-ae24-e1ac53a1c6d2</td></tr><tr><th>steps</th> <td></td></tr></table>




```python
pipeline.deploy()
```




<table><tr><th>name</th> <td>mlflowstatsmodelpipeline</td></tr><tr><th>created</th> <td>2023-03-02 18:06:43.227930+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-02 18:06:48.462230+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>79d6eba9-5eb6-4d70-9676-dc2db997a7ea, 7e0bdce0-20ce-4b46-ae24-e1ac53a1c6d2</td></tr><tr><th>steps</th> <td>mlflowstatmodels</td></tr></table>




```python
pipeline.status()
```




    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.15.72',
       'name': 'engine-6cf5554547-rvswq',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'mlflowstatsmodelpipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'mlflowstatmodels',
          'version': 'aa17282c-0d17-4a95-bc0a-741ffdbd35c8',
          'sha': '6029a5ba3dd2f7aee588bc285ed97bb7cabb302c190c9b9337f0985614f1ed93',
          'status': 'Running'},
         {'name': 'mlflowpostprocess',
          'version': 'c2f63298-db6f-43c4-91f7-f2f22d9aced0',
          'sha': '3dd892e3bba894455bc6c7031b4dc9ce79e70330c65a1de2689dca00cdec59df',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.16.252',
       'name': 'engine-lb-ddd995646-mlh2q',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': [{'ip': '10.244.16.251',
       'name': 'engine-sidekick-mlflowstatmodels-6-65d85f97fd-kcs4h',
       'status': 'Running',
       'reason': None,
       'details': [],
       'statuses': '\n'},
      {'ip': '10.244.15.71',
       'name': 'engine-sidekick-mlflowpostprocess-7-649797d44b-gwhjl',
       'status': 'Running',
       'reason': None,
       'details': [],
       'statuses': '\n'}]}



### Run Inference

Once the pipeline is running, we can submit our data to the pipeline and return our results.  Once finished, we will undeploy the pipeline to return the resources back to the cluster.


```python
results = pipeline.infer_from_file('./resources/bike_day_eval_engine.json')
results
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
      <th>model_name</th>
      <th>model_version</th>
      <th>pipeline_name</th>
      <th>outputs</th>
      <th>elapsed</th>
      <th>time</th>
      <th>original_data</th>
      <th>check_failures</th>
      <th>shadow_data</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>mlflowpostprocess</td>
      <td>c2f63298-db6f-43c4-91f7-f2f22d9aced0</td>
      <td>mlflowstatsmodelpipeline</td>
      <td>[{'Float': {'v': 1, 'dim': [7, 1], 'data': [0....</td>
      <td>200</td>
      <td>2023-03-02 18:21:26.057</td>
      <td>None</td>
      <td>0</td>
      <td>{}</td>
    </tr>
  </tbody>
</table>
</div>




```python
if arrowEnabled is True:
    display(results.loc[0,'outputs'][0]['Float']['data'])
else:
    display(results[0].data()[0])
```


    [0.2819833755493164,
     0.6588467955589294,
     0.5723681449890137,
     0.6198734641075134,
     -1.2178009748458862,
     -1.8491559028625488,
     0.9338850378990173]



```python
pipeline.undeploy()
```




<table><tr><th>name</th> <td>mlflowstatsmodelpipeline</td></tr><tr><th>created</th> <td>2023-03-02 18:06:43.227930+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-02 18:06:48.462230+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>79d6eba9-5eb6-4d70-9676-dc2db997a7ea, 7e0bdce0-20ce-4b46-ae24-e1ac53a1c6d2</td></tr><tr><th>steps</th> <td>mlflowstatmodels</td></tr></table>


