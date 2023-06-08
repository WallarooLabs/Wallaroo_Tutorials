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
* An installed Wallaroo instance.
* The following Python libraries installed:
  * `os`
  * [`wallaroo`](https://pypi.org/project/wallaroo/): The Wallaroo SDK. Included with the Wallaroo JupyterHub service by default.

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
import wallaroo
from wallaroo.object import EntityNotFoundError
import pyarrow as pa
import pandas as pd
```

### Connect to the Wallaroo Instance

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
# Login through local Wallaroo instance

wl = wallaroo.Client()
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
statsmodelUrl = "ghcr.io/wallaroolabs/wallaroo_tutorials/mlflow-statsmodels-example:2023.1"
postprocessUrl = "ghcr.io/wallaroolabs/wallaroo_tutorials/mlflow-postprocess-example:2023.1"

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

<table><tr><th>name</th> <td>mlflowstatsmodelpipeline</td></tr><tr><th>created</th> <td>2023-05-18 13:36:53.441420+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-18 13:37:53.233639+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>0aef343e-a950-4d79-bba0-7a83de83c781, 80805d0f-a6f2-49b2-b896-d154bab9c84b, f6d11021-374d-4f69-9772-23ec5fa1086b</td></tr><tr><th>steps</th> <td>mlflowstatmodels</td></tr></table>

```python
# from wallaroo.deployment_config import DeploymentConfigBuilder

# deployment_config = DeploymentConfigBuilder() \
#     .cpus(0.25).memory('1Gi') \
#     .sidekick_env(sm_model, {"GUNICORN_CMD_ARGS": "--timeout=180 --workers=1"}) \
#     .sidekick_env(pp_model, {"GUNICORN_CMD_ARGS": "--timeout=180 --workers=1"}) \
#     .build()

# pipeline.deploy(deployment_config=deployment_config)

pipeline.deploy()
```

<table><tr><th>name</th> <td>mlflowstatsmodelpipeline</td></tr><tr><th>created</th> <td>2023-05-18 13:36:53.441420+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-18 13:38:23.110383+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>0eb9e238-8dd1-4273-97d9-af234a14e4c6, 0aef343e-a950-4d79-bba0-7a83de83c781, 80805d0f-a6f2-49b2-b896-d154bab9c84b, f6d11021-374d-4f69-9772-23ec5fa1086b</td></tr><tr><th>steps</th> <td>mlflowstatmodels</td></tr></table>

```python
pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.3.159',
       'name': 'engine-d8f5f987-fbm2v',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'mlflowstatsmodelpipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'mlflowpostprocess',
          'version': 'e1c18451-63b7-49ab-8d81-d29132730f70',
          'sha': '825ebae48014d297134930028ab0e823bc0d9551334b9a4402c87a714e8156b2',
          'status': 'Running'},
         {'name': 'mlflowstatmodels',
          'version': 'd624894e-3558-402e-bdb4-6d872ce60a61',
          'sha': '3afd13d9c5070679e284050cd099e84aa2e5cb7c08a788b21d6cb2397615d018',
          'status': 'Running'}]}},
      {'ip': '10.244.3.160',
       'name': 'engine-777db7b76d-djjr5',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'mlflowstatsmodelpipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'mlflowpostprocess',
          'version': 'e1c18451-63b7-49ab-8d81-d29132730f70',
          'sha': '825ebae48014d297134930028ab0e823bc0d9551334b9a4402c87a714e8156b2',
          'status': 'Running'},
         {'name': 'mlflowstatmodels',
          'version': 'd624894e-3558-402e-bdb4-6d872ce60a61',
          'sha': '3afd13d9c5070679e284050cd099e84aa2e5cb7c08a788b21d6cb2397615d018',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.4.193',
       'name': 'engine-lb-584f54c899-lrpqn',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': [{'ip': '10.244.0.126',
       'name': 'engine-sidekick-mlflowstatmodels-42-56896b494f-qwth9',
       'status': 'Running',
       'reason': None,
       'details': [],
       'statuses': '\n'},
      {'ip': '10.244.2.173',
       'name': 'engine-sidekick-mlflowpostprocess-43-6dd5c857b7-qjgbm',
       'status': 'Running',
       'reason': None,
       'details': [],
       'statuses': '\n'}]}

### Run Inference

Once the pipeline is running, we can submit our data to the pipeline and return our results.  Once finished, we will undeploy the pipeline to return the resources back to the cluster.

```python
# Initial container run may need extra time to finish deploying - adding 90 second timeout to compensate
results = pipeline.infer_from_file('./resources/bike_day_eval_engine.df.json', timeout=90)
display(results.loc[:,["out.predicted_mean"]])
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out.predicted_mean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.281983</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.658847</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.572368</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.619873</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.217801</td>
    </tr>
    <tr>
      <th>5</th>
      <td>-1.849156</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.933885</td>
    </tr>
  </tbody>
</table>

```python
pipeline.undeploy()
```

<table><tr><th>name</th> <td>mlflowstatsmodelpipeline</td></tr><tr><th>created</th> <td>2023-05-18 13:36:53.441420+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-18 13:38:23.110383+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>0eb9e238-8dd1-4273-97d9-af234a14e4c6, 0aef343e-a950-4d79-bba0-7a83de83c781, 80805d0f-a6f2-49b2-b896-d154bab9c84b, f6d11021-374d-4f69-9772-23ec5fa1086b</td></tr><tr><th>steps</th> <td>mlflowstatmodels</td></tr></table>

