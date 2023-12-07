This workshop is available from the [Wallaroo Workshop GitHub Repository](https://github.com/WallarooLabs/Workshops/blob/main/Edge-Deployment).

## Large Language Model Summerization Deployment in Wallaroo

The [Yolov8](https://github.com/ultralytics/ultralytics) computer vision model is used for fast recognition of objects in images.  This tutorial demonstrates how to:

* Deploy a Hugging Face LLM Summarization pre-trained model into a Wallaroo Ops server and perform inferences on it.
* Publish the pipeline to the OCI registry configured in the Wallaroo Ops server.
* Add an edge location to the Wallaroo pipeline publish.
* Deploy the pipeline as a Wallaroo Server on an edge device through Docker, and display the inference logs submitted to the Wallaroo Ops server.

Wallaroo Ops Center provides the ability to publish Wallaroo pipelines to an Open Continer Initative (OCI) compliant registry, then deploy those pipelines on edge devices as Docker container or Kubernetes pods.  See [Wallaroo SDK Essentials Guide: Pipeline Edge Publication](https://docs.wallaroo.ai/20230300/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline-publication/) for full details.

This demonstration will focus on deployment to the edge.  The sample model is available at the following URL.  **This model should be downloaded and placed into the `./models/llm-summarization` folder before beginning this demonstration.**

[model-auto-conversion_hugging-face_complex-pipelines_hf-summarisation-bart-large-samsun.zip (1.4 GB)](https://storage.googleapis.com/wallaroo-public-data/llm-models/model-auto-conversion_hugging-face_complex-pipelines_hf-summarisation-bart-large-samsun.zip)

## References

* [Wallaroo Workspaces](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-workspace/):  Workspaces are environments were users upload models, create pipelines and other artifacts.  The workspace should be considered the fundamental area where work is done.  Workspaces are shared with other users to give them access to the same models, pipelines, etc.
* [Wallaroo Model Upload and Registration](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/): ML Models are uploaded to Wallaroo through the SDK or the MLOps API to a **workspace**.  ML models include default runtimes (ONNX, Python Step, and TensorFlow) that are run directly through the Wallaroo engine, and containerized runtimes (Hugging Face, PyTorch, etc) that are run through in a container through the Wallaroo engine.
* [Wallaroo Pipelines](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/): Pipelines are used to deploy models for inferencing.  Each model is a **pipeline step** in a pipelines, where the inputs of the previous step are fed into the next.  Pipeline steps can be ML models, Python scripts, or Arbitrary Python (these contain necessary models and artifacts for running a model).
* [Wallaroo SDK Essentials Guide: Pipeline Edge Publication](https://docs.wallaroo.ai/20230300/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline-publication/): Details on publishing a Wallaroo pipeline to an OCI Registry and deploying it as a Wallaroo Server instance.

## Data Scientist Steps

The following details the steps a Data Scientist performs in uploading and verifying the model in a Wallaroo Ops server.

### Load Libraries

The first step is loading the required libraries including the [Wallaroo Python module](https://pypi.org/project/wallaroo/).

```python
# Import Wallaroo Python SDK
import wallaroo
from wallaroo.object import EntityNotFoundError
from wallaroo.framework import Framework

# used to display DataFrame information without truncating
from IPython.display import display
import pandas as pd
pd.set_option('display.max_colwidth', None)

import pyarrow as pa

```

### Connect to the Wallaroo Instance through the User Interface

The next step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
wl = wallaroo.Client()
```

### Create a New Workspace

We'll use the SDK below to create our workspace , assign as our **current workspace**, then display all of the workspaces we have at the moment.  We'll also set up variables for our models and pipelines down the road, so we have one spot to change names to whatever fits your organization's standards best.

To allow this tutorial to be run by multiple users in the same Wallaroo instance, update suffix with your first and last name.  For example:

```python
suffix = 'lazel-geth'
```

#### Create a New Workspace Exercise

Set the model name, file name, pipeline name, and workspace name.

Sample code:

```python
suffix = ''

model_name = 'llm-summarization'
model_filename = './models/llm-summarization/model-auto-conversion_hugging-face_complex-pipelines_hf-summarisation-bart-large-samsun.zip'
pipeline_name = 'llm-edge-summarization'
workspace_name = f'llm-edge-summarization{suffix}'
```

```python
suffix = ''

model_name = 'llm-summarization'
model_filename = './models/llm-summarization/model-auto-conversion_hugging-face_complex-pipelines_hf-summarisation-bart-large-samsun.zip'
pipeline_name = 'llm-edge-summarization'
workspace_name = f'llm-edge-summarization{suffix}'

```

### Set the Current Workspace

Set the current workspace where the models are uploaded to and pipelines created.

* References
  * [Wallaroo SDK Essentials Guide: Workspace Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-workspace/)

#### Set the Current Workspace Exercise

Setting the workspace is performed with the `wallaroo.client.set_current_workspace(workspace)` method.

Sample code:

```python
workspace = get_workspace(workspace_name, client)
wl.set_current_workspace(workspace)
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

workspace = get_workspace(workspace_name)
wl.set_current_workspace(workspace)
```

    {'name': 'llm-edge-summarization', 'id': 8, 'archived': False, 'created_by': 'a6e82da8-817d-4cca-bb62-5dbacd38ca22', 'created_at': '2023-12-05T20:43:07.604805+00:00', 'models': [{'name': 'llm-summarization', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 12, 5, 20, 44, 13, 14974, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 12, 5, 20, 44, 13, 14974, tzinfo=tzutc())}], 'pipelines': [{'name': 'llm-edge-summarization', 'create_time': datetime.datetime(2023, 12, 5, 20, 48, 36, 535526, tzinfo=tzutc()), 'definition': '[]'}]}

### Upload the Model

When a model is uploaded to a Wallaroo cluster, it is optimized and packaged to make it ready to run as part of a pipeline. In many times, the Wallaroo Server can natively run a model without any Python overhead. In other cases, such as a Python script, a custom Python environment will be automatically generated. This is comparable to the process of "containerizing" a model by adding a small HTTP server and other wrapping around it.

Our pretrained model is in the Hugging Face framework, which is specified in the `framework` parameter.  Hugging Face is a non-native runtime, which requires that the input and output schemas as specified during the model upload process.

* References
  * [Wallaroo SDK Essentials Guide: Model Uploads and Registrations: Hugging Face](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/wallaroo-sdk-model-upload-hugging-face/)

#### Upload the Model Exercise

The model name and file name were set in the variables above.  Use them to upload the model.

Sample code:

```python
input_schema = pa.schema([
    pa.field('inputs', pa.string()),
    pa.field('return_text', pa.bool_()),
    pa.field('return_tensors', pa.bool_()),
    pa.field('clean_up_tokenization_spaces', pa.bool_()),
    # pa.field('generate_kwargs', pa.map_(pa.string(), pa.null())), # dictionaries are not currently supported by the engine
])

output_schema = pa.schema([
    pa.field('summary_text', pa.string()),
])

model = wl.upload_model(model_name, 
                        model_filename, 
                        framework=wallaroo.framework.Framework.HUGGING_FACE_SUMMARIZATION, 
                        input_schema=input_schema, 
                        output_schema=output_schema
                        )
```

```python
# Upload Retrained LLM Summarization Model 

input_schema = pa.schema([
    pa.field('inputs', pa.string()),
    pa.field('return_text', pa.bool_()),
    pa.field('return_tensors', pa.bool_()),
    pa.field('clean_up_tokenization_spaces', pa.bool_()),
    # pa.field('generate_kwargs', pa.map_(pa.string(), pa.null())), # dictionaries are not currently supported by the engine
])

output_schema = pa.schema([
    pa.field('summary_text', pa.string()),
])

model = wl.upload_model(model_name, 
                        model_filename, 
                        framework=wallaroo.framework.Framework.HUGGING_FACE_SUMMARIZATION, 
                        input_schema=input_schema, 
                        output_schema=output_schema
                        )
```

    Waiting for model loading - this will take up to 10.0min.
    Model is pending loading to a container runtime..
    Model is attempting loading to a container runtime.............................................successful
    
    Ready

### Pipeline Deployment Configuration

For our pipeline we set the deployment configuration to set the resources the pipeline will be allocated from the Kubernetes cluster hosting the Wallaroo Ops instance. The Hugging Face model is deployed as a Containerized Runtime in Wallaroo, so the configuration specified the `sidekick` cpu and memory options.

* References
  * [Wallaroo SDK Essentials Guide: Pipeline Deployment Configuration](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline-deployment-config/)

#### Pipeline Deployment Configuration Exercise

Use the deployment configuration below.

```python
deployment_config = wallaroo.DeploymentConfigBuilder() \
    .cpus(0.25).memory('1Gi') \
    .sidekick_cpus(model, 4) \
    .sidekick_memory(model, "8Gi") \
    .build()
```

### Build and Deploy the Pipeline

Now we build our pipeline and set our Yolo8 model as a pipeline step, then deploy the pipeline using the deployment configuration above.

* References
  * [Wallaroo SDK Essentials Guide: Pipeline Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline/)

#### Build and Deploy the Pipeline Exercise

We'll do both commands in one step:

* Build the pipeline with `wallaroo.client.build_pipeline`.
* Set the model as a pipeline step with `wallaroo.pipeline.add_model_step(model)` method.

Sample code:

```python
pipeline = wl.build_pipeline(pipeline_name) \
            .add_model_step(model)    
```

```python
pipeline = wl.build_pipeline(pipeline_name) \
            .add_model_step(model)        
```

### Deploy the Pipeline

We deploy the pipeline with the `wallaroo.pipeline.deploy(deployment_config)` command, using the deployment configuration set up in previous steps.

* References
  * [Wallaroo SDK Essentials Guide: Pipeline Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline/)

#### Deploy the Pipeline Exercise

Deploy the pipeline.

Sample code:

```python
pipeline.deploy(deployment_config=deployment_config)
```

```python
pipeline.deploy(deployment_config=deployment_config)
```

<table><tr><th>name</th> <td>llm-edge-summarization</td></tr><tr><th>created</th> <td>2023-12-05 20:48:36.535526+00:00</td></tr><tr><th>last_updated</th> <td>2023-12-05 20:48:37.172602+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>040afe0c-9a17-4f0b-8cb2-68e8dc85d9a4, 6f1db20f-e1b5-47a5-a6b4-c6a31f9c1023</td></tr><tr><th>steps</th> <td>llm-summarization</td></tr><tr><th>published</th> <td>False</td></tr></table>

### Inference Request

We submit the DataFrame to the pipeline using `wallaroo.pipeline.infer` and display the results.  We'll use both the Wallaroo SDK and the MLOps API.

#### Inference Request Exercise

Perform an inference request.  We'll generate our sample dataframe, then use it for the inference.

Sample Code:

```python
# sample dataframe input

input_data = {
        "inputs": ["LinkedIn (/lɪŋktˈɪn/) is a business and employment-focused social media platform that works through websites and mobile apps. It launched on May 5, 2003. It is now owned by Microsoft. The platform is primarily used for professional networking and career development, and allows jobseekers to post their CVs and employers to post jobs. From 2015 most of the company's revenue came from selling access to information about its members to recruiters and sales professionals. Since December 2016, it has been a wholly owned subsidiary of Microsoft. As of March 2023, LinkedIn has more than 900 million registered members from over 200 countries and territories. LinkedIn allows members (both workers and employers) to create profiles and connect with each other in an online social network which may represent real-world professional relationships. Members can invite anyone (whether an existing member or not) to become a connection. LinkedIn can also be used to organize offline events, join groups, write articles, publish job postings, post photos and videos, and more"], # required
        "return_text": [True], # optional: using the defaults, similar to not passing this parameter
        "return_tensors": [False], # optional: using the defaults, similar to not passing this parameter
        "clean_up_tokenization_spaces": [False], # optional: using the defaults, similar to not passing this parameter
}
dataframe = pd.DataFrame(input_data)
dataframe

```

```python
# sample dataframe input

input_data = {
        "inputs": ["LinkedIn (/lɪŋktˈɪn/) is a business and employment-focused social media platform that works through websites and mobile apps. It launched on May 5, 2003. It is now owned by Microsoft. The platform is primarily used for professional networking and career development, and allows jobseekers to post their CVs and employers to post jobs. From 2015 most of the company's revenue came from selling access to information about its members to recruiters and sales professionals. Since December 2016, it has been a wholly owned subsidiary of Microsoft. As of March 2023, LinkedIn has more than 900 million registered members from over 200 countries and territories. LinkedIn allows members (both workers and employers) to create profiles and connect with each other in an online social network which may represent real-world professional relationships. Members can invite anyone (whether an existing member or not) to become a connection. LinkedIn can also be used to organize offline events, join groups, write articles, publish job postings, post photos and videos, and more"], # required
        "return_text": [True], # optional: using the defaults, similar to not passing this parameter
        "return_tensors": [False], # optional: using the defaults, similar to not passing this parameter
        "clean_up_tokenization_spaces": [False], # optional: using the defaults, similar to not passing this parameter
}
dataframe = pd.DataFrame(input_data)
dataframe
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>inputs</th>
      <th>return_text</th>
      <th>return_tensors</th>
      <th>clean_up_tokenization_spaces</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LinkedIn (/lɪŋktˈɪn/) is a business and employment-focused social media platform that works through websites and mobile apps. It launched on May 5, 2003. It is now owned by Microsoft. The platform is primarily used for professional networking and career development, and allows jobseekers to post their CVs and employers to post jobs. From 2015 most of the company's revenue came from selling access to information about its members to recruiters and sales professionals. Since December 2016, it has been a wholly owned subsidiary of Microsoft. As of March 2023, LinkedIn has more than 900 million registered members from over 200 countries and territories. LinkedIn allows members (both workers and employers) to create profiles and connect with each other in an online social network which may represent real-world professional relationships. Members can invite anyone (whether an existing member or not) to become a connection. LinkedIn can also be used to organize offline events, join groups, write articles, publish job postings, post photos and videos, and more</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>

Use the Pipeline Deployment URL for the inference request, and submit the sample DataFrame as our input.

Sample code:

```bash
!curl {pipeline._deployment._url()} \
    -H "Content-Type: application/json; format=pandas-records" \
    -H "Authorization: {wl.auth.auth_header()['Authorization']}" \
    -H "Accept:{headers['Accept']}" \
     --data @./data/llm-summarization/test_summarization.df.json
```

```python
# inference request here

!curl {pipeline._deployment._url()} \
    -H "Content-Type: application/json; format=pandas-records" \
    -H "Authorization: {wl.auth.auth_header()['Authorization']}" \
    -H "Accept:{headers['Accept']}" \
     --data @./data/llm-summarization/test_summarization.df.json
```

### Undeploy the Pipeline

With the testing complete, we undeploy the pipeline and return the resources back to the cluster.

#### Undeploy the Pipeline Exercise

Sample code:

```python
pipeline.undeploy()
```

```python
# undeploy the pipeline

pipeline.undeploy()
```

<table><tr><th>name</th> <td>llm-edge-summarization</td></tr><tr><th>created</th> <td>2023-12-05 20:48:36.535526+00:00</td></tr><tr><th>last_updated</th> <td>2023-12-05 20:48:37.172602+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>040afe0c-9a17-4f0b-8cb2-68e8dc85d9a4, 6f1db20f-e1b5-47a5-a6b4-c6a31f9c1023</td></tr><tr><th>steps</th> <td>llm-summarization</td></tr><tr><th>published</th> <td>False</td></tr></table>

### Publish the Pipeline for Edge Deployment

It worked! For a demo, we'll take working once as "tested". So now that we've tested our pipeline, we are ready to publish it for edge deployment.

Publishing it means assembling all of the configuration files and model assets and pushing them to an Open Container Initiative (OCI) repository set in the Wallaroo instance as the Edge Registry service.  DevOps engineers then retrieve that image and deploy it through Docker, Kubernetes, or similar deployments.

See [Edge Deployment Registry Guide](https://staging.docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-edge-deployment/) for details on adding an OCI Registry Service to Wallaroo as the Edge Deployment Registry.

This is done through the SDK command `wallaroo.pipeline.publish(deployment_config)` which has the following parameters and returns.

#### Publish a Pipeline Parameters

The `publish` method takes the following parameters.  The containerized pipeline will be pushed to the Edge registry service with the model, pipeline configurations, and other artifacts needed to deploy the pipeline.

| Parameter | Type | Description |
|---|---|---|
| `deployment_config` | `wallaroo.deployment_config.DeploymentConfig` (*Optional*) | Sets the pipeline deployment configuration.  For example:    For more information on pipeline deployment configuration, see the [Wallaroo SDK Essentials Guide: Pipeline Deployment Configuration](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline-deployment-config/).

#### Publish a Pipeline Returns

| Field | Type | Description |
|---|---|---|
| id | integer | Numerical Wallaroo id of the published pipeline. |
| pipeline version id | integer | Numerical Wallaroo id of the pipeline version published. |
| status | string | The status of the pipeline publication.  Values include:  <ul><li>PendingPublish: The pipeline publication is about to be uploaded or is in the process of being uploaded.</li><li>Published:  The pipeline is published and ready for use.</li></ul> |
| Engine URL | string | The URL of the published pipeline engine in the edge registry. |
| Pipeline URL | string | The URL of the published pipeline in the edge registry. |
| Helm Chart URL | string | The URL of the helm chart for the published pipeline in the edge registry. |
| Helm Chart Reference | string | The help chart reference. |
| Helm Chart Version | string | The version of the Helm Chart of the published pipeline.  This is also used as the Docker tag. |
| Engine Config | `wallaroo.deployment_config.DeploymentConfig` | The pipeline configuration included with the published pipeline. |
| Created At | DateTime | When the published pipeline was created. |
| Updated At | DateTime | When the published pipeline was updated. |

#### Publish Exercise

We will now publish the pipeline to our Edge Deployment Registry with the `pipeline.publish(deployment_config)` command.  `deployment_config` is an optional field that specifies the pipeline deployment.  This can be overridden by the DevOps engineer during deployment.

Save the publish to a variable for later use.

Sample code:

```python
pub = pipeline.publish(deployment_config)
pub
```

```python
pub = pipeline.publish(deployment_config)
pub
```

    Waiting for pipeline publish... It may take up to 600 sec.
    Pipeline is Publishing................Published.

<table>
    <tr><td>ID</td><td>1</td></tr>
    <tr><td>Pipeline Version</td><td>1c8d0586-f3ff-453c-82a9-14602830f97f</td></tr>
    <tr><td>Status</td><td>Published</td></tr>
    <tr><td>Engine URL</td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/engines/proxy/wallaroo/ghcr.io/wallaroolabs/standalone-mini:v2023.4.0-4103'>ghcr.io/wallaroolabs/doc-samples/engines/proxy/wallaroo/ghcr.io/wallaroolabs/standalone-mini:v2023.4.0-4103</a></td></tr>
    <tr><td>Pipeline URL</td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/pipelines/llm-edge-summarization:1c8d0586-f3ff-453c-82a9-14602830f97f'>ghcr.io/wallaroolabs/doc-samples/pipelines/llm-edge-summarization:1c8d0586-f3ff-453c-82a9-14602830f97f</a></td></tr>
    <tr><td>Helm Chart URL</td><td>oci://<a href='https://ghcr.io/wallaroolabs/doc-samples/charts/llm-edge-summarization'>ghcr.io/wallaroolabs/doc-samples/charts/llm-edge-summarization</a></td></tr>
    <tr><td>Helm Chart Reference</td><td>ghcr.io/wallaroolabs/doc-samples/charts@sha256:0a0c5b2df650b33fe4457a6eb7b2701caea19d0c81f97c0fe6345b088baaac41</td></tr>
    <tr><td>Helm Chart Version</td><td>0.0.1-1c8d0586-f3ff-453c-82a9-14602830f97f</td></tr>
    <tr><td>Engine Config</td><td>{'engine': {'resources': {'limits': {'cpu': 1.0, 'memory': '512Mi'}, 'requests': {'cpu': 1.0, 'memory': '512Mi'}}}, 'engineAux': {'images': {}}, 'enginelb': {'resources': {'limits': {'cpu': 1.0, 'memory': '512Mi'}, 'requests': {'cpu': 1.0, 'memory': '512Mi'}}}}</td></tr>
    <tr><td>User Images</td><td>[]</td></tr>
    <tr><td>Created By</td><td>john.hummel@wallaroo.ai</td></tr>
    <tr><td>Created At</td><td>2023-12-05 20:50:16.153199+00:00</td></tr>
    <tr><td>Updated At</td><td>2023-12-05 20:50:16.153199+00:00</td></tr>
    <tr><td>Docker Run Variables</td><td>{}</td></tr>
</table>

### List Published Pipeline

The method `wallaroo.client.list_pipelines()` shows a list of all pipelines in the Wallaroo instance, and includes the `published` field that indicates whether the pipeline was published to the registry (`True`), or has not yet been published (`False`).

#### List Published Pipeline Exercise

List the pipelines and verify which ones are published or not.

Sample code:

```python
wl.list_pipelines()
```

```python
wl.list_pipelines()
```

<table><tr><th>name</th><th>created</th><th>last_updated</th><th>deployed</th><th>arch</th><th>tags</th><th>versions</th><th>steps</th><th>published</th></tr><tr><td>llm-edge-summarization</td><td>2023-05-Dec 20:48:36</td><td>2023-05-Dec 20:50:14</td><td>False</td><td>None</td><td></td><td>1c8d0586-f3ff-453c-82a9-14602830f97f, 040afe0c-9a17-4f0b-8cb2-68e8dc85d9a4, 6f1db20f-e1b5-47a5-a6b4-c6a31f9c1023</td><td>llm-summarization</td><td>True</td></tr></table>

### List Publishes from a Pipeline

All publishes created from a pipeline are displayed with the `wallaroo.pipeline.publishes` method.  The `pipeline_version_id` is used to know what version of the pipeline was used in that specific publish.  This allows for pipelines to be updated over time, and newer versions to be sent and tracked to the Edge Deployment Registry service.

#### List Publishes Parameters

N/A

#### List Publishes Returns

A List of the following fields:

| Field | Type | Description |
|---|---|---|
| id | integer | Numerical Wallaroo id of the published pipeline. |
| pipeline_version_id | integer | Numerical Wallaroo id of the pipeline version published. |
| engine_url | string | The URL of the published pipeline engine in the edge registry. |
| pipeline_url | string | The URL of the published pipeline in the edge registry. |
| created_by | string | The email address of the user that published the pipeline.
| Created At | DateTime | When the published pipeline was created. |
| Updated At | DateTime | When the published pipeline was updated. |

#### List Publishes from a Pipeline Exercise

List the publishes from a pipeline.

Sample code:

```python
pipeline.publishes()
```

```python
pipeline.publishes()
```

<table><tr><th>id</th><th>pipeline_version_name</th><th>engine_url</th><th>pipeline_url</th><th>created_by</th><th>created_at</th><th>updated_at</th></tr><tr><td>1</td><td>1c8d0586-f3ff-453c-82a9-14602830f97f</td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/engines/proxy/wallaroo/ghcr.io/wallaroolabs/standalone-mini:v2023.4.0-4103'>ghcr.io/wallaroolabs/doc-samples/engines/proxy/wallaroo/ghcr.io/wallaroolabs/standalone-mini:v2023.4.0-4103</a></td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/pipelines/llm-edge-summarization:1c8d0586-f3ff-453c-82a9-14602830f97f'>ghcr.io/wallaroolabs/doc-samples/pipelines/llm-edge-summarization:1c8d0586-f3ff-453c-82a9-14602830f97f</a></td><td>john.hummel@wallaroo.ai</td><td>2023-05-Dec 20:50:16</td><td>2023-05-Dec 20:50:16</td></tr></table>

### Add Edge Location

With the pipeline publish created, we can add an Edge Location.  This allows the edge deployment to upload its inference results back to the Wallaroo Ops location, which are then added to the pipeline the publish originated from.  These are added to the pipeline logs `partition` metadata.

First we'll retrieve the pipeline logs for our current pipeline, and show the current pipeline logs metadata.

For the edge name, set it to `firstname-lastname-edge-yolo`.

* References
  * [Wallaroo SDK Essentials Guide: Pipeline Log Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline-logs/)
  * [Wallaroo SDK Essentials Guide: Pipeline Edge Publication](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline-publication/#add-edge)

#### Add Edge Location Exercise

Display the log information with the `metadata.partition`, then add the edge location to the publish.  Note that edge names **must** be unique, so add your first and last name to the list.

Sample code:

```python
logs = pipeline.logs(dataset=['time', 'out.output0', 'metadata'])
display(logs.loc[:, ['time', 'metadata.partition']])

first_last_name = '-Gale-Karlach'

edge_name = f'hf-summarizer-edge-demo{first_last_name}'

edge_publish = pub.add_edge(edge_name)
display(edge_publish)
```

```python
logs = pipeline.logs(dataset=['time', 'out.summary_text', 'metadata'])
display(logs.loc[:, ['time', 'out.summary_text', 'metadata.partition']])
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out.summary_text</th>
      <th>metadata.partition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-12-05 20:49:24.105</td>
      <td>LinkedIn is a business and employment-focused social media platform that works through websites and mobile apps. It launched on May 5, 2003. LinkedIn allows members (both workers and employers) to create profiles and connect with each other in an online social network which may represent real-world professional relationships.</td>
      <td>engine-66f69d685-4d4s2</td>
    </tr>
  </tbody>
</table>

```python
edge_name = 'edge-summarization-demo-arm'

edge_publish = pub.add_edge(edge_name)
display(edge_publish)
```

<table>
    <tr><td>ID</td><td>1</td></tr>
    <tr><td>Pipeline Version</td><td>1c8d0586-f3ff-453c-82a9-14602830f97f</td></tr>
    <tr><td>Status</td><td>Published</td></tr>
    <tr><td>Engine URL</td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/engines/proxy/wallaroo/ghcr.io/wallaroolabs/standalone-mini:v2023.4.0-4103'>ghcr.io/wallaroolabs/doc-samples/engines/proxy/wallaroo/ghcr.io/wallaroolabs/standalone-mini:v2023.4.0-4103</a></td></tr>
    <tr><td>Pipeline URL</td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/pipelines/llm-edge-summarization:1c8d0586-f3ff-453c-82a9-14602830f97f'>ghcr.io/wallaroolabs/doc-samples/pipelines/llm-edge-summarization:1c8d0586-f3ff-453c-82a9-14602830f97f</a></td></tr>
    <tr><td>Helm Chart URL</td><td>oci://<a href='https://ghcr.io/wallaroolabs/doc-samples/charts/llm-edge-summarization'>ghcr.io/wallaroolabs/doc-samples/charts/llm-edge-summarization</a></td></tr>
    <tr><td>Helm Chart Reference</td><td>ghcr.io/wallaroolabs/doc-samples/charts@sha256:0a0c5b2df650b33fe4457a6eb7b2701caea19d0c81f97c0fe6345b088baaac41</td></tr>
    <tr><td>Helm Chart Version</td><td>0.0.1-1c8d0586-f3ff-453c-82a9-14602830f97f</td></tr>
    <tr><td>Engine Config</td><td>{'engine': {'resources': {'limits': {'cpu': 1.0, 'memory': '512Mi'}, 'requests': {'cpu': 1.0, 'memory': '512Mi'}}}, 'engineAux': {'images': {}}, 'enginelb': {'resources': {'limits': {'cpu': 1.0, 'memory': '512Mi'}, 'requests': {'cpu': 1.0, 'memory': '512Mi'}}}}</td></tr>
    <tr><td>User Images</td><td>[]</td></tr>
    <tr><td>Created By</td><td>john.hummel@wallaroo.ai</td></tr>
    <tr><td>Created At</td><td>2023-12-05 20:50:16.153199+00:00</td></tr>
    <tr><td>Updated At</td><td>2023-12-05 20:50:16.153199+00:00</td></tr>
    <tr><td>Docker Run Variables</td><td>{'EDGE_BUNDLE': 'abcde'}</td></tr>
</table>

## DevOps - Pipeline Edge Deployment

Once a pipeline is deployed to the Edge Registry service, it can be deployed in environments such as Docker, Kubernetes, or similar container running services by a DevOps engineer.

### Docker Deployment

First, the DevOps engineer must authenticate to the same OCI Registry service used for the Wallaroo Edge Deployment registry.

For more details, check with the documentation on your artifact service.  The following are provided for the three major cloud services:

* [Set up authentication for Docker](https://cloud.google.com/artifact-registry/docs/docker/authentication)
* [Authenticate with an Azure container registry](https://learn.microsoft.com/en-us/azure/container-registry/container-registry-authentication?tabs=azure-cli)
* [Authenticating Amazon ECR Repositories for Docker CLI with Credential Helper](https://aws.amazon.com/blogs/compute/authenticating-amazon-ecr-repositories-for-docker-cli-with-credential-helper/)

For the deployment, the engine URL is specified with the following environmental variables:

* `DEBUG` (true|false): Whether to include debug output.
* `OCI_REGISTRY`: The URL of the registry service.
* `CONFIG_CPUS`: The number of CPUs to use.
* `OCI_USERNAME`: The edge registry username.
* `OCI_PASSWORD`:  The edge registry password or token.
* `PIPELINE_URL`: The published pipeline URL.

#### Docker Deployment Example

Using our sample environment, here's sample deployment using Docker with a computer vision ML model, the same used in the [Wallaroo Use Case Tutorials Computer Vision: Retail](https://docs.wallaroo.ai/wallaroo-use-case-tutorials/wallaroo-use-case-computer-vision/use-case-computer-vision-retail/) tutorials.

Note the use of the `-v ./data:/persist` option.  This will store the one time authentication token stored in the `EDGE_BUNDLE`

```bash
mkdir ./data

docker run -p 8080:8080 \
    -v ./data:/persist \
    -e DEBUG=true -e OCI_REGISTRY={your registry server} \
    -e EDGE_BUNDLE={edge_publish.docker_run_variables['EDGE_BUNDLE']} \
    -e CONFIG_CPUS=4 \
    -e OCI_USERNAME=oauth2accesstoken \
    -e OCI_PASSWORD={registry token here} \
    -e PIPELINE_URL={your registry server}/pipelines/yolo8demonstration:bf70eaf7-8c11-4b46-b751-916a43b1a555 \
    {your registry server}/engine:v2023.3.0-main-3707
```

### Docker Compose Deployment

For users who prefer to use `docker compose`, the following sample `compose.yaml` file is used to launch the Wallaroo Edge pipeline.  This is the same used in the [Wallaroo Use Case Tutorials Computer Vision: Retail](https://docs.wallaroo.ai/wallaroo-use-case-tutorials/wallaroo-use-case-computer-vision/use-case-computer-vision-retail/) tutorials.

The `volumes` settings allows for persistent volumes to store the session information.  Without it, the one-time authentication token included in the `EDGE_BUNDLE` settings would have to be regenerated.

```yml
services:
  engine:
    image: {Your Engine URL}
    volumes:
      - ./data:/persist
    ports:
      - 8080:8080
    environment:
      EDGE_BUNDLE: abcdefg
      PIPELINE_URL: {Your Pipeline URL}
      OCI_REGISTRY: {Your Edge Registry URL}
      OCI_USERNAME:  {Your Registry Username}
      OCI_PASSWORD: {Your Token or Password}
      CONFIG_CPUS: 4
```

For example:

```yml
services:
  engine:
    image: sample-registry.com/engine:v2023.3.0-main-3707
    ports:
      - 8080:8080
    environment:
      PIPELINE_URL: sample-registry.com/pipelines/yolo8demonstration:bf70eaf7-8c11-4b46-b751-916a43b1a555
      OCI_REGISTRY: sample-registry.com
      OCI_USERNAME:  _json_key_base64
      OCI_PASSWORD: abc123
      CONFIG_CPUS: 4
```

#### Docker Compose Deployment Example

The deployment and undeployment is then just a simple `docker compose up` and `docker compose down`.  The following shows an example of deploying the Wallaroo edge pipeline using `docker compose`.

```bash
docker compose up
[+] Running 1/1
 ✔ Container yolo8demonstration-engine-1  Recreated                                                                                                                                                                 0.5s
Attaching to yolo8demonstration-engine-1
yolo8demonstration-engine-1  | Wallaroo Engine - Standalone mode
yolo8demonstration-engine-1  | Login Succeeded
yolo8demonstration-engine-1  | Fetching manifest and config for pipeline: sample-registry.com/pipelines/yolo8demonstration:bf70eaf7-8c11-4b46-b751-916a43b1a555
yolo8demonstration-engine-1  | Fetching model layers
yolo8demonstration-engine-1  | digest: sha256:c6c8869645962e7711132a7e17aced2ac0f60dcdc2c7faa79b2de73847a87984
yolo8demonstration-engine-1  |   filename: c6c8869645962e7711132a7e17aced2ac0f60dcdc2c7faa79b2de73847a87984
yolo8demonstration-engine-1  |   name: yolov8n
yolo8demonstration-engine-1  |   type: model
yolo8demonstration-engine-1  |   runtime: onnx
yolo8demonstration-engine-1  |   version: 693e19b5-0dc7-4afb-9922-e3f7feefe66d
yolo8demonstration-engine-1  |
yolo8demonstration-engine-1  | Fetched
yolo8demonstration-engine-1  | Starting engine
yolo8demonstration-engine-1  | Looking for preexisting `yaml` files in //modelconfigs
yolo8demonstration-engine-1  | Looking for preexisting `yaml` files in //pipelines
```

### Helm Deployment

Published pipelines can be deployed through the use of helm charts.

Helm deployments take up to two steps - the first step is in retrieving the required `values.yaml` and making updates to override.

Kubernetes provides persistent volume support, so no settings are required.

1. Pull the helm charts from the published pipeline.  The two fields are the Helm Chart URL and the Helm Chart version to specify the OCI .    This typically takes the format of:

  ```bash
  helm pull oci://{published.helm_chart_url} --version {published.helm_chart_version}
  ```

1. Extract the `tgz` file and copy the `values.yaml` and copy the values used to edit engine allocations, etc.  The following are **required** for the deployment to run:

  ```yml
  ociRegistry:
    registry: {your registry service}
    username:  {registry username here}
    password: {registry token here}
  ```

  Store this into another file, suc as `local-values.yaml`.

1. Create the namespace to deploy the pipeline to.  For example, the namespace `wallaroo-edge-pipeline` would be:

  ```bash
  kubectl create -n wallaroo-edge-pipeline
  ```

1. Deploy the `helm` installation with `helm install` through one of the following options:
    1. Specify the `tgz` file that was downloaded and the local values file.  For example:

        ```bash
        helm install --namespace {namespace} --values {local values file} {helm install name} {tgz path}
        ```

    1. Specify the expended directory from the downloaded `tgz` file.

        ```bash
        helm install --namespace {namespace} --values {local values file} {helm install name} {helm directory path}
        ```

    1. Specify the Helm Pipeline Helm Chart and the Pipeline Helm Version.

        ```bash
        helm install --namespace {namespace} --values {local values file} {helm install name} oci://{published.helm_chart_url} --version {published.helm_chart_version}
        ```

1. Once deployed, the DevOps engineer will have to forward the appropriate ports to the `svc/engine-svc` service in the specific pipeline.  For example, using `kubectl port-forward` to the namespace `ccfraud` that would be:

    ```bash
    kubectl port-forward svc/engine-svc -n ccfraud01 8080 --address 0.0.0.0`
    ```

### Docker Deployment Code Generation Exercise

The following code segment generates a `docker run` template based on the previously published pipeline.  Replace the `$REGISTRYURL`, `$REGISTRYUSERNAME`, and `$REGISTRYPASSWORD` to match the OCI Registry being used.

```python
docker_deploy = f'''
docker run -p 8080:8080 \\
    -v ./data:/persist \\
    -e DEBUG=true \\
    -e OCI_REGISTRY=$REGISTRYURL \\
    -e EDGE_BUNDLE={edge_publish.docker_run_variables['EDGE_BUNDLE']} \\
    -e CONFIG_CPUS=1 \\
    -e OCI_USERNAME=$REGISTRYUSERNAME \\
    -e OCI_PASSWORD=$REGISTRYPASSWORD \\
    -e PIPELINE_URL={edge_publish.pipeline_url} \\
    {edge_publish.engine_url}
'''

print(docker_deploy)
```

    
    docker run -p 8080:8080 \
        -v ./data:/persist \
        -e DEBUG=true \
        -e OCI_REGISTRY=$REGISTRYURL \
        -e EDGE_BUNDLE=ZXhwb3J0IEJVTkRMRV9WRVJTSU9OPTEKZXhwb3J0IEVER0VfTkFNRT1lZGdlLXN1bW1hcml6YXRpb24tZGVtby1hcm0KZXhwb3J0IEpPSU5fVE9LRU49MTlkMGE4NzAtNTgyYy00YWExLWE5YjAtMWQxMTQ4MWI4MWRjCmV4cG9ydCBPUFNDRU5URVJfSE9TVD1kb2MtdGVzdC5lZGdlLndhbGxhcm9vY29tbXVuaXR5Lm5pbmphCmV4cG9ydCBQSVBFTElORV9VUkw9Z2hjci5pby93YWxsYXJvb2xhYnMvZG9jLXNhbXBsZXMvcGlwZWxpbmVzL2xsbS1lZGdlLXN1bW1hcml6YXRpb246MWM4ZDA1ODYtZjNmZi00NTNjLTgyYTktMTQ2MDI4MzBmOTdmCmV4cG9ydCBXT1JLU1BBQ0VfSUQ9OA== \
        -e CONFIG_CPUS=1 \
        -e OCI_USERNAME=$REGISTRYUSERNAME \
        -e OCI_PASSWORD=$REGISTRYPASSWORD \
        -e PIPELINE_URL=ghcr.io/wallaroolabs/doc-samples/pipelines/llm-edge-summarization:1c8d0586-f3ff-453c-82a9-14602830f97f \
        ghcr.io/wallaroolabs/doc-samples/engines/proxy/wallaroo/ghcr.io/wallaroolabs/standalone-mini:v2023.4.0-4103
    

## Edge Deployed Pipeline API Endpoints

Once deployed, we can check the pipelines and models available.  We'll use a `curl` command, but any HTTP based request will work the same way.

The endpoint `/pipelines` returns:

* **id** (*String*):  The name of the pipeline.
* **status** (*String*):  The status as either `Running`, or `Error` if there are any issues.

```bash
curl localhost:8080/pipelines
{"pipelines":[{"id":"yolo8demonstration","status":"Running"}]}
```

The following example uses the host `localhost`.  Replace with your own host name of your Edge deployed pipeline.

```python
!curl workshop-hf-summarizer-demo.eastus.cloudapp.azure.com:8080/pipelines
```

    {"pipelines":[{"id":"hf-summarizer-standard","status":"Running"}]}

The endpoint `/models` returns a List of models with the following fields:

* **name** (*String*): The model name.
* **sha** (*String*): The sha hash value of the ML model.
* **status** (*String*):  The status of either Running or Error if there are any issues.
* **version** (*String*):  The model version.  This matches the version designation used by Wallaroo to track model versions in UUID format.

```bash
{"models":[{"name":"yolov8n","sha":"3ed5cd199e0e6e419bd3d474cf74f2e378aacbf586e40f24d1f8c89c2c476a08","status":"Running","version":"7af40d06-d18f-4b3f-9dd3-0a15248f01c8"}]}
```

The following example uses the host `localhost`.  Replace with your own host name of your Edge deployed pipeline.

```python
!curl workshop-hf-summarizer-demo.eastus.cloudapp.azure.com:8080/models
```

    {"models":[{"name":"hf-summarizer-standard","sha":"ee71d066a83708e7ca4a3c07caf33fdc528bb000039b6ca2ef77fa2428dc6268","status":"Running","version":"7dbae7b4-20d0-40f7-a3f5-eeabdd77f418"}]}

### Edge Inference Endpoint

The inference endpoint takes the following pattern:

* `/pipelines/{pipeline-name}`:  The `pipeline-name` is the same as returned from the [`/pipelines`](#list-pipelines) endpoint as `id`.

Wallaroo inference endpoint URLs accept the following data inputs through the `Content-Type` header:

* `Content-Type: application/vnd.apache.arrow.file`: For Apache Arrow tables.
* `Content-Type: application/json; format=pandas-records`: For pandas DataFrame in record format.

Once deployed, we can perform an inference through the deployment URL.

The endpoint returns `Content-Type: application/json; format=pandas-records` by default with the following fields:

* **check_failures** (*List[Integer]*): Whether any validation checks were triggered.  For more information, see [Wallaroo SDK Essentials Guide: Pipeline Management: Anomaly Testing](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline/#anomaly-testing).
* **elapsed** (*List[Integer]*): A list of time in nanoseconds for:
  * [0] The time to serialize the input.
  * [1...n] How long each step took.
* **model_name** (*String*): The name of the model used.
* **model_version** (*String*): The version of the model in UUID format.
* **original_data**: The original input data.  Returns `null` if the input may be too long for a proper return.
* **outputs** (*List*): The outputs of the inference result separated by data type, where each data type includes:
  * **data**: The returned values.
  * **dim** (*List[Integer]*): The dimension shape returned.
  * **v** (*Integer*): The vector shape of the data.
* **pipeline_name**  (*String*): The name of the pipeline.
* **shadow_data**: Any shadow deployed data inferences in the same format as **outputs**.
* **time** (*Integer*): The time since UNIX epoch.
  
Once deployed, we can perform an inference through the deployment URL.  We'll assume we're running the inference request through the localhost and submitting the local file `./data/dogbike.df.json`.  Note that our inference endpoint is `pipelines/yolo8demonstration` - the same as our pipeline name.

The following example demonstrates sending an inference request to the edge deployed pipeline and storing the results in a pandas DataFrame in record format.  The results can then be exported to other processes to render the detected images or other use cases.

```python
!curl workshop-hf-summarizer-demo.eastus.cloudapp.azure.com:8080/pipelines/hf-summarizer-standard \
    -H "Content-Type: application/json; format=pandas-records" \
    --data @./data/llm-summarization/test_summarization.df.json
```

    [{"check_failures":[],"elapsed":[310666674,4294967295],"model_name":"hf-summarizer-standard","model_version":"7dbae7b4-20d0-40f7-a3f5-eeabdd77f418","original_data":null,"outputs":[{"String":{"data":["LinkedIn is a business and employment-focused social media platform that works through websites and mobile apps. It launched on May 5, 2003. LinkedIn allows members (both workers and employers) to create profiles and connect with each other in an online social network which may represent real-world professional relationships."],"dim":[1,1],"v":1}}],"pipeline_name":"hf-summarizer-standard","shadow_data":{},"time":1701815846451}]
