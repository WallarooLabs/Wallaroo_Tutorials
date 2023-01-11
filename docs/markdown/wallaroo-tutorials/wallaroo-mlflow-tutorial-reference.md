## MLFlow Inference with Wallaroo Tutorial 

Wallaroo users can register their trained [MLFlow ML Models](https://www.mlflow.org/docs/latest/models.html) from a containerized model container registry into their Wallaroo instance and perform inferences with it through a Wallaroo pipeline.

As of this time, Wallaroo only supports MLFlow 1.3.0 containerized models.  For information on how to containerize an MLFlow model, see the [MLFlow Documentation](https://mlflow.org/docs/latest/projects.html).

This tutorial assumes that you have a Wallaroo instance, and have either your own containerized model or use the one from the reference and are running this Notebook from the Wallaroo Jupyter Hub service.

## MLFlow Data Formats

When using containerized MLFLow models with Wallaroo, the inputs and outputs must be named.  For example, the following output:

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

MLFlow models are composed of two parts:  the model, and the flavors.  When submitting a MLFlow model to Wallaroo, both aspects must be part of the ML Model included in the container.  For full information about MLFlow model structure, see the [MLFLow Documentation](https://www.mlflow.org/docs/latest/index.html).

Wallaroo registers the models from container registries.  Organizations will either have to make their containers available in a public or through a private Containerized Model Container Registry service.  For examples on setting up a private container registry service, see the [Docker Documentation "Deploy a registry server"](https://docs.docker.com/registry/deploying/).  For more details on setting up a container registry in a cloud environment, see the related documentation for your preferred cloud provider:
  * [Google Cloud Platform Container Registry](https://cloud.google.com/container-registry)
  * [Amazon Web Services Elastic Container Registry](https://docs.aws.amazon.com/AmazonECR/latest/userguide/what-is-ecr.html)
  *  [Microsoft Azure Container Registry](https://azure.microsoft.com/en-us/free/container-registry/)

For this example, we will be using the MLFlow containers that was registered in a GitHub container registry service in MLFLow Creation Tutorial Part 03: Container Registration.  The address of those containers are:

* postprocess: ghcr.io/johnhansarickwallaroo/mlflowtests/mlflow-postprocess-example .  Used for format data after the statsmodel inferences.
* statsmodel: ghcr.io/johnhansarickwallaroo/mlflowtests/mlflow-statsmodels-example . The statsmodel generated in MLFLow Creation Tutorial Part 01: Model Creation.

### Prerequisites

Before uploading and running an inference with a MLFlow model in Wallaroo the following will be required:

* **MLFlow Input Schema**:  The input schema with the fields and data types for each MLFLow model type uploaded to Wallaroo.  In the examples below, the data types are imported using the `pyarrow` library.
* A Wallaroo instance version 2022.4 or later.

**IMPORTANT NOTE**:  Wallaroo supports MLFlow 1.3.0.  Please ensure the MLFlow models used in Wallaroo meet this specification.

## Private Containerized Model Container Registry Steps

The following steps are used to register a Private Containerized Model Container Registry in a Wallaroo instance.  For organizations that only intend to use a public model registry (such as DockerHub or the GitHub example below), these steps can be skipped.

### Prerequisites

Before starting, the following must be available:

* A private model registry that is accessible from the Wallaroo instance.
* `kubectl` and either `kots` or `helm` depending on the install method, and a connection to the Kubernetes cluster where the Wallaroo instance was installed.
* The username, password, and email (OPTIONAL) address of the model registry user used to authenticate to the private model registry.
  * If using a service such as GitHub, then the token used in "MLFLow Creation Tutorial Part 03: Container Registration" can be used.

### Configure Via Kots

If Wallaroo was installed via `kots`, use the following procedure to add the private model registry information.

1. Launch the Wallaroo Administrative Dashboard through a terminal linked to the Kubernetes cluster.  Replace the namespace with the one used in your installation.
    
    ```bash
    kubectl kots admin-console --namespace wallaroo
    ```
1. Launch the dashboard, by default at http://localhost:8800.
1. From the admin dashboard, select **Config -> Private Model Container Registry**.
1. Enable **Provide private container registry credentials for model images**.
1. Provide the following:
    1. **Registry URL**: The URL of the Containerized Model Container Registry.  Typically in the format `host:port`.  In this example, the registry for GitHub is used.  **NOTE**:  When setting the URL for the Containerized Model Container Registry, only the actual service address is needed.  For example:  with the full URL of the model as `ghcr.io/wallaroolabs/wallaroo_tutorials/mlflow-statsmodels-example:2022.4`, the URL would be `ghcr.io/wallaroolabs`.
    1. **email**: The email address of the user authenticating to the  registry service.
    1. **username**:  The username of the user authentication to the  registry service.
    1. **password**:  The password of the user authentication to the  registry service.  In the GitHub example from "MLFLow Creation Tutorial Part 03: Container Registration", this would be the token.
    
        ![](./images/wallaroo-configuration/wallaroo-private-model-registry/kots-private-registry.png)
     
1. Scroll down and select **Save config**.
1. Deploy the new version.

Once complete, the Wallaroo instance will be able to authenticate to the Containerized Model Container Registry and retrieve the images as defined in the example "MLFlow Inference with Wallaroo Tutorial".

### Configure via Helm

1. During either the installation process or updates, set the following in the `local-values.yaml` file:
    1. `privateModelRegistry`:
      1. `enabled`: true
      1. `secretName`: `model-registry-secret`
      1. `registry`: The URL of the private registry.
      1. `email`: The email address of the user authenticating to the registry service.
      1. `username`:  The username of the user authentication to the registry service.
      1. `password`:  The password of the user authentication to the registry service.  In the GitHub example from "MLFLow Creation Tutorial Part 03: Container Registration", this would be the token.
      
        For example:

        ```yml

        # Other settings - DNS entries, etc.

        # The private registry settings
        privateModelRegistry:
          enabled: true
          secretName: model-registry-secret
          registry: "YOUR REGISTRY URL:YOUR REGISTRY PORT"
          email: "YOUR EMAIL ADDRESS"
          username: "YOUR USERNAME"
          password: "Your Password here"
        ```

1. Install or update the Wallaroo instance via Helm as per the [Wallaroo Helm Install instructions](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-install-guides/wallaroo-enterprise-install-guides/wallaroo-enterprise-install/wallaroo-enterprise-install-helm/wallaroo-helm-install-standard/).

Once complete, the Wallaroo instance will be able to authenticate to the registry service and retrieve the images as defined in the example "MLFlow Inference with Wallaroo Tutorial".

## MLFLow Inference Steps

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

import wallaroo
from wallaroo.object import EntityNotFoundError

import pyarrow as pa
```

### Connect to Wallaroo

Connect to Wallaroo and store the connection in the variable `wl`.

The folowing methods are used to create the workspace and pipeline for this tutorial.  A workspace is created and set as the current workspace that will contain the registered models and pipelines.


```python
# SSO login through keycloak

wallarooPrefix = "YOUR PREFIX"
wallarooSuffix = "YOUR SUFFIX"


wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}.api.{wallarooSuffix}", 
                auth_endpoint=f"https://{wallarooPrefix}.keycloak.{wallarooSuffix}", 
                auth_type="sso")
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
        pipeline = wl.pipelines_by_name(pipeline_name)[0]
    except EntityNotFoundError:
        pipeline = wl.build_pipeline(pipeline_name)
    return pipeline
```


```python
prefix = 'statsmodels-test'
workspace_name= "statsmodelworkspace"
pipeline_name = "statsmodelpipeline"

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
    name=f"{prefix}-statmodels",
    image=f"{statsmodelUrl}"
).configure("mlflow", input_schema=sm_input_schema, output_schema=pp_input_schema)
pp_model = wl.register_model_image(
    name=f"{prefix}-postprocess",
    image=f"{postprocessUrl}"
).configure("mlflow", input_schema=pp_input_schema, output_schema=pp_input_schema)
```

### Create Pipeline and Add Model Steps

With the models registered, we can add the MLFlow models as steps in the pipeline.  Once ready, we will deploy the pipeline so it is available for submitting data for running inferences.


```python
pipeline.add_model_step(sm_model)
pipeline.add_model_step(pp_model)
```




<table><tr><th>name</th> <td>statsmodelpipeline</td></tr><tr><th>created</th> <td>2023-01-09 19:11:40.149983+00:00</td></tr><tr><th>last_updated</th> <td>2023-01-09 20:08:04.722635+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>facae641-8eef-421c-b8e7-71a34959e1b6, 950bd9a6-130d-491f-9328-8e627e9b95f0, ba1e7414-a383-4679-90e2-681102bad9df, 3edbdab3-f7d2-427b-9d27-78c6815931a8, 164955dd-e9af-4128-8e04-10be1c26eda3, c92cdeb9-f54a-4ae4-b990-13d00ceca3a5, d8bd9a57-5f05-46b0-be43-4fc15c137f08, 406c6969-c3f4-4dc6-bb26-b77bd25d3d6b, 7838cb49-4c51-446a-a48e-5eeb37299b9e</td></tr><tr><th>steps</th> <td>statsmodels-test-statmodels</td></tr></table>




```python
pipeline.deploy()
```




<table><tr><th>name</th> <td>statsmodelpipeline</td></tr><tr><th>created</th> <td>2023-01-09 19:11:40.149983+00:00</td></tr><tr><th>last_updated</th> <td>2023-01-09 20:11:55.754558+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>50bf648c-082b-4a13-a4a6-162884d14802, facae641-8eef-421c-b8e7-71a34959e1b6, 950bd9a6-130d-491f-9328-8e627e9b95f0, ba1e7414-a383-4679-90e2-681102bad9df, 3edbdab3-f7d2-427b-9d27-78c6815931a8, 164955dd-e9af-4128-8e04-10be1c26eda3, c92cdeb9-f54a-4ae4-b990-13d00ceca3a5, d8bd9a57-5f05-46b0-be43-4fc15c137f08, 406c6969-c3f4-4dc6-bb26-b77bd25d3d6b, 7838cb49-4c51-446a-a48e-5eeb37299b9e</td></tr><tr><th>steps</th> <td>statsmodels-test-statmodels</td></tr></table>



### Run Inference

Once the pipeline is running, we can submit our data to the pipeline and return our results.  Once finished, we will undeploy the pipeline to return the resources back to the cluster.


```python
pipeline.status()
```




    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.2.90',
       'name': 'engine-7bfd4c684f-k7lr4',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'statsmodelpipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'statsmodels-test-postprocess',
          'version': 'cb654179-e9ee-4ec9-972f-0fc625c13e32',
          'sha': '3dd892e3bba894455bc6c7031b4dc9ce79e70330c65a1de2689dca00cdec59df',
          'status': 'Running'},
         {'name': 'statsmodels-test-statmodels',
          'version': '01d5b5a1-39f5-4097-a023-9bd8da5e01dd',
          'sha': '6029a5ba3dd2f7aee588bc285ed97bb7cabb302c190c9b9337f0985614f1ed93',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.0.45',
       'name': 'engine-lb-55dcdff64c-5nn69',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': [{'ip': '10.244.2.89',
       'name': 'engine-sidekick-statsmodels-test-postprocess-27-778b6bbcb6jhkcr',
       'status': 'Running',
       'reason': None,
       'details': [],
       'statuses': '\n'},
      {'ip': '10.244.0.46',
       'name': 'engine-sidekick-statsmodels-test-statmodels-26-74d565568-8ttbd',
       'status': 'Running',
       'reason': None,
       'details': [],
       'statuses': '\n'}]}




```python
results = pipeline.infer_from_file('./resources/bike_day_eval_engine.json')
results
```




    [InferenceResult({'check_failures': [],
      'elapsed': 300,
      'model_name': 'statsmodels-test-postprocess',
      'model_version': 'cb654179-e9ee-4ec9-972f-0fc625c13e32',
      'original_data': {'holiday': [0, 0, 0, 0, 0, 0, 0],
                        'temp': [0.317391,
                                 0.365217,
                                 0.415,
                                 0.54,
                                 0.4725,
                                 0.3325,
                                 0.430435],
                        'windspeed': [0.184309,
                                      0.203117,
                                      0.209579,
                                      0.231017,
                                      0.368167,
                                      0.207721,
                                      0.288783],
                        'workingday': [1, 1, 1, 1, 0, 0, 1]},
      'outputs': [{'Float': {'data': [-0.7701932787895203,
                                      -0.15543800592422485,
                                      0.36521396040916443,
                                      1.739493727684021,
                                      -0.07358897477388382,
                                      -1.6944431066513062,
                                      0.5889557003974915],
                             'dim': [7, 1],
                             'dtype': 'Float',
                             'v': 1}}],
      'pipeline_name': 'statsmodelpipeline',
      'shadow_data': {},
      'time': 1673295133657})]




```python
assert results[0].data()[0].shape == (7, 1)
```


```python
pipeline.undeploy()
```




<table><tr><th>name</th> <td>statsmodelpipeline</td></tr><tr><th>created</th> <td>2023-01-09 19:11:40.149983+00:00</td></tr><tr><th>last_updated</th> <td>2023-01-09 20:11:55.754558+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>50bf648c-082b-4a13-a4a6-162884d14802, facae641-8eef-421c-b8e7-71a34959e1b6, 950bd9a6-130d-491f-9328-8e627e9b95f0, ba1e7414-a383-4679-90e2-681102bad9df, 3edbdab3-f7d2-427b-9d27-78c6815931a8, 164955dd-e9af-4128-8e04-10be1c26eda3, c92cdeb9-f54a-4ae4-b990-13d00ceca3a5, d8bd9a57-5f05-46b0-be43-4fc15c137f08, 406c6969-c3f4-4dc6-bb26-b77bd25d3d6b, 7838cb49-4c51-446a-a48e-5eeb37299b9e</td></tr><tr><th>steps</th> <td>statsmodels-test-statmodels</td></tr></table>




```python

```
