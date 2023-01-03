This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-features/wallaroo-model-endpoints).

## Internal Pipeline Inference URL Tutorial

Wallaroo provides the ability to perform inferences through deployed pipelines via both internal and external inference URLs.  These inference URLs allow inferences to be performed by submitting data to the internal or external URL with the inference results returned in the same format as the [InferenceResult Object](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/#run-inference-through-a-pipeline).

**Internal URLs** are available only through the internal Kubernetes environment hosting the Wallaroo instance as demonstrated in this tutorial.
**External URLs** are available outside of the Kubernetes environment, such as the public internet.  These are demonstrated in the External Pipeline Deployment URL Tutorial.

The following tutorial shows how to set up an environment and demonstrates how to use the Internal Deployment URL.  This example provides the following:

* `aloha-cnn-lstm.zip`:  Aloha model used as part of the [Aloha Quick Tutorial](https://docs.wallaroo.ai/wallaroo-tutorials/wallaroo-quick-start-aloha/).
* `data-1.json`, `data-1k.json` and `data-25k.json`:  Sample data used for testing inferences with the sample model.

For our example, we will perform the following:

* Create a workspace for our work.
* Upload the Aloha model.
* Create a pipeline that can ingest our submitted data, submit it to the model, and export the results.
* Run a sample inference through our pipeline via the SDK to demonstrate the inference is accurate.
* Run a sample inference through our pipeline's Internal URL and store the results in a file.

All sample data and models are available through the [Wallaroo Quick Start Guide Samples repository](https://github.com/WallarooLabs/quickstartguide_samples).

## Open a Connection to Wallaroo

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

```python
import wallaroo
from wallaroo.object import EntityNotFoundError
```

```python
# Client connection from local Wallaroo instance

# wl = wallaroo.Client()

# SSO login through keycloak

wallarooPrefix = "YOUR PREFIX"
wallarooSuffix = "YOUR SUFFIX"

wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}.api.{wallarooSuffix}", 
                    auth_endpoint=f"https://{wallarooPrefix}.keycloak.{wallarooSuffix}", 
                    auth_type="sso")
```

## Create the Workspace

We will create a workspace to work in and call it the `urldemoworkspace`, then set it as current workspace environment.  We'll also create our pipeline in advance as `urldemopipeline`.

The model to be uploaded and used for inference will be labeled as `urldemomodel`.  Modify these to your organizations requirements.

Once complete, the workspace will be created or, if already existing, set to the current workspace to host the pipelines and models.

```python
workspace_name = 'urldemoworkspace'
pipeline_name = 'urldemopipeline'
model_name = 'urldemomodel'
model_file_name = './aloha-cnn-lstm.zip'
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
workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)

pipeline = get_pipeline(pipeline_name)
pipeline
```

<table><tr><th>name</th> <td>urldemopipeline</td></tr><tr><th>created</th> <td>2022-11-30 15:19:57.505044+00:00</td></tr><tr><th>last_updated</th> <td>2022-11-30 15:19:57.505044+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>3575355c-452e-4668-bd48-46821307cf65</td></tr><tr><th>steps</th> <td></td></tr></table>

We can verify the workspace is created the current default workspace with the `get_current_workspace()` command.

```python
wl.get_current_workspace()
```

    {'name': 'urldemoworkspace', 'id': 16, 'archived': False, 'created_by': '74a4109a-9798-4d7c-98be-62d9380c9606', 'created_at': '2022-11-30T15:19:57.293347+00:00', 'models': [], 'pipelines': [{'name': 'urldemopipeline', 'create_time': datetime.datetime(2022, 11, 30, 15, 19, 57, 505044, tzinfo=tzutc()), 'definition': '[]'}]}

# Upload the Models

Now we will upload our models.  Note that for this example we are applying the model from a .ZIP file.  The Aloha model is a [protobuf](https://developers.google.com/protocol-buffers) file that has been defined for evaluating web pages, and we will configure it to use data in the `tensorflow` format.

```python
model = wl.upload_model(model_name, model_file_name).configure("tensorflow")
```

## Deploy The Pipeline
Now that we have a model that we want to use we will create a deployment for it. 

We will tell the deployment we are using a tensorflow model and give the deployment name and the configuration we want for the deployment.

```python
pipeline.add_model_step(model)
pipeline.deploy()
```

    Waiting for deployment - this will take up to 45s ........ ok

<table><tr><th>name</th> <td>urldemopipeline</td></tr><tr><th>created</th> <td>2022-11-30 15:19:57.505044+00:00</td></tr><tr><th>last_updated</th> <td>2022-11-30 15:19:58.319115+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>2f1169ba-e325-4b01-aeda-663ad34305aa, 3575355c-452e-4668-bd48-46821307cf65</td></tr><tr><th>steps</th> <td>urldemomodel</td></tr></table>

We can verify that the pipeline is running and list what models are associated with it.

```python
pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.48.0.60',
       'name': 'engine-7dc4fc8cf8-vq9m9',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'urldemopipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'urldemomodel',
          'version': '47299047-a3e0-4637-9b87-cc243f4552f3',
          'sha': '7c89707252ce389980d5348c37885d6d72af4c20cd303422e2de7e66dd7ff184',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.48.1.33',
       'name': 'engine-lb-8d5c8cb76-sbrf5',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}

## Interferences

### Infer 1 row

Now that the pipeline is deployed and our Aloha model is in place, we'll perform a smoke test to verify the pipeline is up and running properly.  We'll use the `infer_from_file` command to load a single encoded URL into the inference engine and print the results back out.

The result should tell us that the tokenized URL is legitimate (0) or fraud (1).  This sample data should return close to 0.

```python
pipeline.infer_from_file("data-1.json")
```

    [InferenceResult({'check_failures': [],
      'elapsed': 290668215,
      'model_name': 'urldemomodel',
      'model_version': '47299047-a3e0-4637-9b87-cc243f4552f3',
      'original_data': {'text_input': [[0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        28,
                                        16,
                                        32,
                                        23,
                                        29,
                                        32,
                                        30,
                                        19,
                                        26,
                                        17]]},
      'outputs': [{'Float': {'data': [0.0015195842133834958],
                             'dim': [1, 1],
                             'dtype': 'Float',
                             'v': 1}},
                  {'Float': {'data': [0.9829147458076477],
                             'dim': [1, 1],
                             'dtype': 'Float',
                             'v': 1}},
                  {'Float': {'data': [0.012099552899599075],
                             'dim': [1, 1],
                             'dtype': 'Float',
                             'v': 1}},
                  {'Float': {'data': [4.7591205657226965e-05],
                             'dim': [1, 1],
                             'dtype': 'Float',
                             'v': 1}},
                  {'Float': {'data': [2.0289331587264314e-05],
                             'dim': [1, 1],
                             'dtype': 'Float',
                             'v': 1}},
                  {'Float': {'data': [0.00031977228354662657],
                             'dim': [1, 1],
                             'dtype': 'Float',
                             'v': 1}},
                  {'Float': {'data': [0.011029261164367199],
                             'dim': [1, 1],
                             'dtype': 'Float',
                             'v': 1}},
                  {'Float': {'data': [0.9975640177726746],
                             'dim': [1, 1],
                             'dtype': 'Float',
                             'v': 1}},
                  {'Float': {'data': [0.010341613553464413],
                             'dim': [1, 1],
                             'dtype': 'Float',
                             'v': 1}},
                  {'Float': {'data': [0.008038961328566074],
                             'dim': [1, 1],
                             'dtype': 'Float',
                             'v': 1}},
                  {'Float': {'data': [0.016155045479536057],
                             'dim': [1, 1],
                             'dtype': 'Float',
                             'v': 1}},
                  {'Float': {'data': [0.0062362332828342915],
                             'dim': [1, 1],
                             'dtype': 'Float',
                             'v': 1}},
                  {'Float': {'data': [0.0009985746582970023],
                             'dim': [1, 1],
                             'dtype': 'Float',
                             'v': 1}},
                  {'Float': {'data': [1.7933435344117743e-26],
                             'dim': [1, 1],
                             'dtype': 'Float',
                             'v': 1}},
                  {'Float': {'data': [1.388984431455466e-27],
                             'dim': [1, 1],
                             'dtype': 'Float',
                             'v': 1}}],
      'pipeline_name': 'urldemopipeline',
      'shadow_data': {},
      'time': 1669821622803})]

### Batch Inference

Now that our smoke test is successful, we will retrieve the Internal Deployment URL and perform an inference by submitting our data through a `curl` command as detailed below.

```python
internal_url = pipeline._deployment._url()
internal_url
```

    'http://engine-lb.urldemopipeline-5:29502/pipelines/urldemopipeline'

```python
!curl -X POST {internal_url} -H "Content-Type:application/json" --data @data-1.json > curl_response.txt
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  1392  100  1264  100   128  35111   3555 --:--:-- --:--:-- --:--:-- 38666

## Undeploy Pipeline

When finished with our tests, we will undeploy the pipeline so we have the Kubernetes resources back for other tasks.

**IMPORTANT NOTE**:  For the External Pipeline Deployment URL Tutorial, this pipeline will have to be deployed to make the External Deployment URL available.

```python
pipeline.undeploy()
```

    Please log into the following URL in a web browser:
    
    	https://YOUR PREFIX.keycloak.example.wallaroo.ai/auth/realms/master/device?user_code=VUSN-BRDZ
    
    Login successful!
    Waiting for undeployment - this will take up to 45s .................................... ok

<table><tr><th>name</th> <td>urldemopipeline</td></tr><tr><th>created</th> <td>2022-11-30 15:19:57.505044+00:00</td></tr><tr><th>last_updated</th> <td>2022-11-30 15:19:58.319115+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>2f1169ba-e325-4b01-aeda-663ad34305aa, 3575355c-452e-4668-bd48-46821307cf65</td></tr><tr><th>steps</th> <td>urldemomodel</td></tr></table>

```python

```
