This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/simulated_edge).

# Simulated Edge Demo

This notebook will explore "Edge ML", meaning deploying a model intended to be run on "the edge". What is "the edge"?  This is typically defined as a resource (CPU, memory, and/or bandwidth) constrained environment or where a combination of latency requirements and bandwidth available requires the models to run locally.

Wallaroo provides two key capabilities when it comes to deploying models to edge devices:

1. Since the same engine is used in both environments, the model behavior can often be simulated accurately using Wallaroo in a data center for testing prior to deployment.
2. Wallaroo makes edge deployments "observable" so the same tools used to monitor model performance can be used in both kinds of deployments. 

This notebook closely parallels the [Aloha tutorial](https://docs.wallaroo.ai/wallaroo-tutorials/wallaroo-quick-start-aloha/).  The primary difference is instead of provide ample resources to a pipeline to allow high-throughput operation we will specify a resource budget matching what is expected in the final deployment. Then we can apply the expected load to the model and observe how it behaves given the available resources.

This example uses the open source [Aloha CNN LSTM model](https://www.researchgate.net/publication/348920204_Using_Auxiliary_Inputs_in_Deep_Learning_Models_for_Detecting_DGA-based_Domain_Names) for classifiying Domain names as being either legitimate or being used for nefarious purposes such as malware distribution. This could be deployed on a network router to detect suspicious domains in real-time. Of course, it is important to monitor the behavior of the model across all of the deployments so we can see if the detect rate starts to drift over time.

Note that this example is not intended for production use and is meant of an example of running Wallaroo in a restrained environment.  The environment is based on the [Wallaroo AWS EC2 Setup guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-install-guides/wallaroo-community-install-guides/wallaroo-setup-environment-community/wallaroo-aws-vm-community-setup/).

Full details on how to configure a deployment through the SDK, see the [Wallaroo SDK guides](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/).

For our example, we will perform the following:

* Create a workspace for our work.
* Upload the Aloha model.
* Define a resource budget for our inference pipeline.
* Create a pipeline that can ingest our submitted data, submit it to the model, and export the results
* Run a sample inference through our pipeline by loading a file
* Run a batch inference through our pipeline's URL and store the results in a file and find that the original memory
  allocation is too small.
* Redeploy the pipeline with a larger memory budget and attempt sending the same batch of requests through again.

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
    
    	https://beautiful-couch-3504.keycloak.wallaroo.community/auth/realms/master/device?user_code=OEHR-BULW
    
    Login successful!

## Useful variables

The following variables and methods are used to create a workspace, the pipeline in the example workspace and upload models into it.

```python
pipeline_name = 'edgepipelineexample'
workspace_name = 'edgeworkspaceexample'
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

### Create or Set the Workspace

Create the workspace and set it as our default workspace.  If a workspace by the same name already exists, then that workspace will be used.

```python
workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)
workspace
```

    {'name': 'edgeworkspaceexample', 'id': 2, 'archived': False, 'created_by': 'ac217b38-6f50-46fd-9c04-f790ffc5cb0e', 'created_at': '2022-10-13T17:10:35.150766+00:00', 'models': [], 'pipelines': []}

# Upload the Models

Now we will upload our models.  Note that for this example we are applying the model from a .ZIP file.  The Aloha model is a [protobuf](https://developers.google.com/protocol-buffers) file that has been defined for evaluating web pages, and we will configure it to use data in the `tensorflow` format.

```python
model = wl.upload_model(model_name, model_file_name).configure("tensorflow")
```

# Define the resource budget
The DeploymentConfig object specifies the resources to allocate for a model pipeline. In this case, we're going to set a very small budget, one that is too small for this model and then expand it based on testing. To start with, we'll use 1 CPU and 150 MB of RAM.

```python
deployment_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(1).memory("150Mi").build()
```

## Deploy a model
Now that we have a model that we want to use we will create a deployment for it using the resource limits defined above. 

We will tell the deployment we are using a tensorflow model and give the deployment name and the configuration we want for the deployment.

To do this, we'll create our pipeline that can ingest the data, pass the data to our Aloha model, and give us a final output.  We'll call our pipeline `edgepipeline`, then deploy it so it's ready to receive data.  The deployment process usually takes about 45 seconds.

* **Note**:  If you receive an error that the pipeline could not be deployed because there are not enough resources, undeploy any other pipelines and deploy this one again.  This command can quickly undeploy all pipelines to regain resources.  We recommend **not** running this command in a production environment since it will cancel any running pipelines:

```python
for p in wl.list_pipelines(): p.undeploy()
```

```python
pipeline = wl.build_pipeline(pipeline_name)
pipeline.add_model_step(model)
pipeline.deploy(deployment_config=deployment_config)
```

    Waiting for deployment - this will take up to 45s ........................... ok

<table><tr><th>name</th> <td>edgepipelineexample</td></tr><tr><th>created</th> <td>2022-10-13 17:10:54.680327+00:00</td></tr><tr><th>last_updated</th> <td>2022-10-13 17:10:54.745255+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>steps</th> <td>alohamodel</td></tr></table>

We can verify that the pipeline is running and list what models are associated with it.

```python
pipeline.status()
```

    {'status': 'Running',
     'details': None,
     'engines': [{'ip': '10.32.0.54',
       'name': 'engine-644f699c7f-nmllg',
       'status': 'Running',
       'reason': None,
       'pipeline_statuses': {'pipelines': [{'id': 'edgepipelineexample',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'alohamodel',
          'version': '6b56bd8c-563a-4be8-8175-efb771dace44',
          'sha': '7c89707252ce389980d5348c37885d6d72af4c20cd303422e2de7e66dd7ff184',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.32.0.55',
       'name': 'engine-lb-67c854cc86-spddl',
       'status': 'Running',
       'reason': None}]}

## Interferences

### Infer 1 row

Now that the pipeline is deployed and our model is in place, we'll perform a smoke test to verify the pipeline is up and running properly.  We'll use the `infer_from_file` command to load a single encoded URL into the inference engine and print the results back out.

The result should tell us that the tokenized URL is legitimate (0) or fraud (1).  This sample data should return close to 0.

```python
pipeline.infer_from_file("data-1.json")
```

    Waiting for inference response - this will take up to 45s ......... ok

    [InferenceResult({'check_failures': [],
      'elapsed': 348896807,
      'model_name': 'alohamodel',
      'model_version': '6b56bd8c-563a-4be8-8175-efb771dace44',
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
      'outputs': [{'Float': {'data': [0.001519531011581421], 'dim': [1, 1], 'v': 1}},
                  {'Float': {'data': [0.9829148054122925], 'dim': [1, 1], 'v': 1}},
                  {'Float': {'data': [0.012099534273147583], 'dim': [1, 1], 'v': 1}},
                  {'Float': {'data': [4.7593468480044976e-05],
                             'dim': [1, 1],
                             'v': 1}},
                  {'Float': {'data': [2.0289722669986077e-05],
                             'dim': [1, 1],
                             'v': 1}},
                  {'Float': {'data': [0.000319749116897583], 'dim': [1, 1], 'v': 1}},
                  {'Float': {'data': [0.011029303073883057], 'dim': [1, 1], 'v': 1}},
                  {'Float': {'data': [0.9975640773773193], 'dim': [1, 1], 'v': 1}},
                  {'Float': {'data': [0.010341644287109375], 'dim': [1, 1], 'v': 1}},
                  {'Float': {'data': [0.008038878440856934], 'dim': [1, 1], 'v': 1}},
                  {'Float': {'data': [0.01615503430366516], 'dim': [1, 1], 'v': 1}},
                  {'Float': {'data': [0.006236225366592407], 'dim': [1, 1], 'v': 1}},
                  {'Float': {'data': [0.0009985864162445068],
                             'dim': [1, 1],
                             'v': 1}},
                  {'Float': {'data': [1.7933435344117743e-26],
                             'dim': [1, 1],
                             'v': 1}},
                  {'Float': {'data': [1.3889950240701602e-27],
                             'dim': [1, 1],
                             'v': 1}}],
      'pipeline_name': 'edgepipelineexample',
      'shadow_data': {},
      'time': 1665681096577})]

```python
pipeline._deployment._url()
```

    'http://engine-lb.edgepipelineexample-1:29502/pipelines/edgepipelineexample'

```python
!curl -X POST http://engine-lb.edgepipelineexample-1:29502/pipelines/edgepipelineexample -H "Content-Type:application/json" --data @data-1k.json > curl_response.txt
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  111k  100    95  100  111k    250   293k --:--:-- --:--:-- --:--:--  293k

# Redeploy with a little larger budget 
If you look in the file curl_response.txt, you will see that the inference failed:
> upstream connect error or disconnect/reset before headers. reset reason: connection termination

Even though a single inference passed, submitted a larger batch of work did not. If this is an expected usage case for
this model, we need to add more memory. Let's do that now.

The following DeploymentConfig is the same as the original, but increases the memory from 150MB to 300MB. This sort
of budget would be available on some network routers.

```python
pipeline.undeploy()
deployment_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(1).memory("300Mi").build()
pipeline = wl.build_pipeline(pipeline_name)
pipeline.add_model_step(model)
pipeline.deploy(deployment_config=deployment_config)
```

     ok
    Waiting for deployment - this will take up to 45s ..... ok

<table><tr><th>name</th> <td>edgepipelineexample</td></tr><tr><th>created</th> <td>2022-10-13 17:10:54.680327+00:00</td></tr><tr><th>last_updated</th> <td>2022-10-13 17:13:53.075759+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>steps</th> <td>alohamodel</td></tr></table>

# Re-run inference
Running the same curl command again should now produce a curl_response.txt file containing the expected results.

```python
!curl -X POST http://engine-lb.edgepipelineexample-1:29502/pipelines/edgepipelineexample -H "Content-Type:application/json" --data @data-1k.json > curl_response.txt
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100  524k  100  413k  100  111k   176k  48857  0:00:02  0:00:02 --:--:--  224k

It is important to note that increasing the memory was necessary to run a batch of 1,000 inferences at once. If this is not a design
use case for your system, running with the smaller memory budget may be acceptable. Wallaroo allows you to easily test difference
loading patterns to get a sense for what resources are required with sufficient buffer to allow for robust operation of your system
while not over-provisioning scarce resources.

## Undeploy Pipeline

When finished with our tests, we will undeploy the pipeline so we have the Kubernetes resources back for other tasks.  Note that if the deployment variable is unchanged aloha_pipeline.deploy() will restart the inference engine in the same configuration as before.

```python
pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s .. ok

<table><tr><th>name</th> <td>edgepipelineexample</td></tr><tr><th>created</th> <td>2022-10-13 17:10:54.680327+00:00</td></tr><tr><th>last_updated</th> <td>2022-10-13 17:13:53.075759+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>steps</th> <td>alohamodel</td></tr></table>

```python

```
