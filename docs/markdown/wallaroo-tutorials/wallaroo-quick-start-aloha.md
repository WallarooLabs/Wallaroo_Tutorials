This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/aloha).

## Aloha Demo

In this notebook we will walk through a simple pipeline deployment to inference on a model. For this example we will be using an open source model that uses an [Aloha CNN LSTM model](https://www.researchgate.net/publication/348920204_Using_Auxiliary_Inputs_in_Deep_Learning_Models_for_Detecting_DGA-based_Domain_Names) for classifiying Domain names as being either legitimate or being used for nefarious purposes such as malware distribution.  

For our example, we will perform the following:

* Create a workspace for our work.
* Upload the Aloha model.
* Create a pipeline that can ingest our submitted data, submit it to the model, and export the results
* Run a sample inference through our pipeline by loading a file
* Run a sample inference through our pipeline's URL and store the results in a file.

All sample data and models are available through the [Wallaroo Quick Start Guide Samples repository](https://github.com/WallarooLabs/quickstartguide_samples).

## Open a Connection to Wallaroo

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

```python
import wallaroo
from wallaroo.object import EntityNotFoundError
```

```python
wl = wallaroo.Client()
```

    Please log into the following URL in a web browser:
    
    	https://yellow-platypus-9801.keycloak.wallaroo.community/auth/realms/master/device?user_code=NACK-RXGV
    
    Login successful!

## Create the Workspace

We will create a workspace to work in and call it the "alohaworkspace", then set it as current workspace environment.  We'll also create our pipeline in advance as `alohapipeline`.

```python
workspace_name = 'alohaworkspace'
pipeline_name = 'alohapipeline'
model_name = 'alohamodel'
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

aloha_pipeline = get_pipeline(pipeline_name)
aloha_pipeline
```

We can verify the workspace is created the current default workspace with the `get_current_workspace()` command.

```python
wl.get_current_workspace()
```



    {'name': 'aloha-workspace', 'id': 6, 'archived': False, 'created_by': '7dbb3754-4c14-4730-8b77-33caeea7a2a0', 'created_at': '2022-03-29T16:14:08.85824+00:00', 'models': [], 'pipelines': []}


# Upload the Models

Now we will upload our models.  Note that for this example we are applying the model from a .ZIP file.  The Aloha model is a [protobuf](https://developers.google.com/protocol-buffers) file that has been defined for evaluating web pages, and we will configure it to use data in the `tensorflow` format.

```python
model = wl.upload_model(model_name, model_file_name).configure("tensorflow")
```

## Deploy a model
Now that we have a model that we want to use we will create a deployment for it. 

We will tell the deployment we are using a tensorflow model and give the deployment name and the configuration we want for the deployment.

To do this, we'll create our pipeline that can ingest the data, pass the data to our Aloha model, and give us a final output.  We'll call our pipeline `aloha-test-demo`, then deploy it so it's ready to receive data.  The deployment process usually takes about 45 seconds.

* **Note**:  If you receive an error that the pipeline could not be deployed because there are not enough resources, undeploy any other pipelines and deploy this one again.  This command can quickly undeploy all pipelines to regain resources.  We recommend **not** running this command in a production environment since it will cancel any running pipelines:

```python
for p in wl.list_pipelines(): p.undeploy()
```

```python
aloha_pipeline.add_model_step(model)
aloha_pipeline.deploy()
```

    Waiting for deployment - this will take up to 45s ....... ok



    {'name': 'aloha-test-demo', 'create_time': datetime.datetime(2022, 3, 29, 16, 15, 31, 638290, tzinfo=tzutc()), 'definition': "[{'ModelInference': {'models': [{'name': 'aloha-2', 'version': '496e6860-a658-4d35-8b55-0f8cc6ad6fde', 'sha': 'fd998cd5e4964bbbb4f8d29d245a8ac67df81b62be767afbceb96a03d1a01520'}]}}]"}


We can verify that the pipeline is running and list what models are associated with it.

```python
aloha_pipeline.status()
```



    {'status': 'Running',
     'details': None,
     'engines': [{'ip': '10.12.1.236',
       'name': 'engine-864d86d898-k26hv',
       'status': 'Running',
       'reason': None,
       'pipeline_statuses': {'pipelines': [{'id': 'aloha-test-demo',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'aloha-2',
          'version': '496e6860-a658-4d35-8b55-0f8cc6ad6fde',
          'sha': 'fd998cd5e4964bbbb4f8d29d245a8ac67df81b62be767afbceb96a03d1a01520',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.12.1.235',
       'name': 'engine-lb-85846c64f8-dcj4f',
       'status': 'Running',
       'reason': None}]}


## Interferences

### Infer 1 row

Now that the pipeline is deployed and our Aloha model is in place, we'll perform a smoke test to verify the pipeline is up and running properly.  We'll use the `infer_from_file` command to load a single encoded URL into the inference engine and print the results back out.

The result should tell us that the tokenized URL is legitimate (0) or fraud (1).  This sample data should return close to 0.

```python
aloha_pipeline.infer_from_file("data-1.json")
```



    [InferenceResult({'check_failures': [],
      'elapsed': 631348351,
      'model_name': 'aloha-2',
      'model_version': '496e6860-a658-4d35-8b55-0f8cc6ad6fde',
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
      'outputs': [{'Float': {'data': [0.001519620418548584], 'dim': [1, 1], 'v': 1}},
                  {'Float': {'data': [0.9829147458076477], 'dim': [1, 1], 'v': 1}},
                  {'Float': {'data': [0.012099534273147583], 'dim': [1, 1], 'v': 1}},
                  {'Float': {'data': [4.7593468480044976e-05],
                             'dim': [1, 1],
                             'v': 1}},
                  {'Float': {'data': [2.0289742678869516e-05],
                             'dim': [1, 1],
                             'v': 1}},
                  {'Float': {'data': [0.0003197789192199707],
                             'dim': [1, 1],
                             'v': 1}},
                  {'Float': {'data': [0.011029303073883057], 'dim': [1, 1], 'v': 1}},
                  {'Float': {'data': [0.9975639581680298], 'dim': [1, 1], 'v': 1}},
                  {'Float': {'data': [0.010341644287109375], 'dim': [1, 1], 'v': 1}},
                  {'Float': {'data': [0.008038878440856934], 'dim': [1, 1], 'v': 1}},
                  {'Float': {'data': [0.016155093908309937], 'dim': [1, 1], 'v': 1}},
                  {'Float': {'data': [0.006236225366592407], 'dim': [1, 1], 'v': 1}},
                  {'Float': {'data': [0.0009985864162445068],
                             'dim': [1, 1],
                             'v': 1}},
                  {'Float': {'data': [1.7933435344117743e-26],
                             'dim': [1, 1],
                             'v': 1}},
                  {'Float': {'data': [1.388984431455466e-27],
                             'dim': [1, 1],
                             'v': 1}}],
      'pipeline_name': 'aloha-test-demo',
      'time': 1648570552486})]


### Batch Inference

Now that our smoke test is successfully, let's really give it some data.  We have two inference files we can use:

* `data-1k.json`:  Contains 1,0000 inferences
* `data-25k.json`: Contains 25,000 inferences

We'll pipe the `data-25k.json` file through the `aloha_pipeline` deployment URL, and place the results in a file named `response.txt`.  We'll also display the time this takes.  Note that for larger batches of 50,000 inferences or more can be difficult to view in Juypter Hub because of its size.

When running this example, replace the URL from the `_deployment._url()` command into the `curl` command below.

```python
aloha_pipeline._deployment._url()
```



    'http://engine-lb.aloha-test-demo-5:29502/pipelines/aloha-test-demo'



```python
!curl -X POST http://engine-lb.aloha-test-demo-5:29502/pipelines/aloha-test-demo -H "Content-Type:application/json" --data @data-25k.json > curl_response.txt
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100 12.9M  100 10.1M  100 2886k   539k   149k  0:00:19  0:00:19 --:--:-- 2570k

## Undeploy Pipeline

When finished with our tests, we will undeploy the pipeline so we have the Kubernetes resources back for other tasks.  Note that if the deployment variable is unchanged aloha_pipeline.deploy() will restart the inference engine in the same configuration as before.

```python
aloha_pipeline.undeploy()
```



    {'name': 'aloha-test-demo', 'create_time': datetime.datetime(2022, 3, 29, 16, 15, 31, 638290, tzinfo=tzutc()), 'definition': "[{'ModelInference': {'models': [{'name': 'aloha-2', 'version': '496e6860-a658-4d35-8b55-0f8cc6ad6fde', 'sha': 'fd998cd5e4964bbbb4f8d29d245a8ac67df81b62be767afbceb96a03d1a01520'}]}}]"}

