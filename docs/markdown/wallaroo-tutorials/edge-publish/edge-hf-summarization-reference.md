The following tutorial is available on the [Wallaroo Github Repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/pipeline-edge-publish/edge-llm-summarization).

## Summarization Text Edge Deployment Demonstration

This notebook will walk through building a summarization text pipeline in Wallaroo, deploying it to the local cluster for testing, and then publishing it for edge deployment.

This demonstration will focus on deployment to the edge.  The sample model is available at the following URL.  This model should be downloaded and placed into the `./models` folder before beginning this demonstration.

[model-auto-conversion_hugging-face_complex-pipelines_hf-summarisation-bart-large-samsun.zip (1.4 GB)](https://storage.googleapis.com/wallaroo-public-data/llm-models/model-auto-conversion_hugging-face_complex-pipelines_hf-summarisation-bart-large-samsun.zip)

This demonstration will perform the following:

1. As a Data Scientist:
    1. Upload a computer vision model to Wallaroo, deploy it in a Wallaroo pipeline, then perform a sample inference.
    1. Publish the pipeline to an Open Container Initiative (OCI) Registry service.  This is configured in the Wallaroo instance.  See [Edge Deployment Registry Guide](https://staging.docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-edge-deployment/) for details on adding an OCI Registry Service to Wallaroo as the Edge Deployment Registry.
    1. View the pipeline publish details.
1. As a DevOps Engineer:
    1. Deploy the published pipeline into an edge instance.  This example will use Docker.
    1. Perform a sample inference into the deployed pipeline with the same data used in the data scientist example.

## Data Scientist Pipeline Publish Steps

### Load Libraries

The first step is to import the libraries used in this notebook.

```python
import wallaroo
from wallaroo.object import EntityNotFoundError

import pyarrow as pa
import pandas as pd

# used to display dataframe information without truncating
from IPython.display import display
pd.set_option('display.max_colwidth', None)
```

### Connect to the Wallaroo Instance through the User Interface

The next step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
wl = wallaroo.Client()

wallarooPrefix = "doc-test."
wallarooSuffix = "wallarooexample.ai"

wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}api.{wallarooSuffix}", 
                    auth_endpoint=f"https://{wallarooPrefix}keycloak.{wallarooSuffix}", 
                    auth_type="sso")
```

### Create a New Workspace

We'll use the SDK below to create our workspace , assign as our **current workspace**, then display all of the workspaces we have at the moment.  We'll also set up variables for our models and pipelines down the road, so we have one spot to change names to whatever fits your organization's standards best.

To allow this tutorial to be run by multiple users in the same Wallaroo instance, a random 4 character prefix will be added to the workspace, pipeline, and model.  Feel free to set `suffix=''` if this is not required.

```python
import string
import random

# make a random 4 character prefix
suffix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))

suffix=''

workspace_name = f'edge-hf-summarization{suffix}'
pipeline_name = 'edge-hf-summarization'
model_name = 'hf-summarization'
model_file_name = './models/model-auto-conversion_hugging-face_complex-pipelines_hf-summarisation-bart-large-samsun.zip'
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
workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)
```

    {'name': 'edge-hf-summarization', 'id': 8, 'archived': False, 'created_by': 'db364f8c-b866-4865-96b7-0b65662cb384', 'created_at': '2023-08-25T18:43:02.41099+00:00', 'models': [], 'pipelines': []}

### Configure PyArrow Schema

This is required for non-native runtimes for models deployed to Wallaroo.

You can find more info on the available inputs under [TextSummarizationInputs](https://github.com/WallarooLabs/platform/blob/main/conductor/model-auto-conversion/flavors/hugging-face/src/io/pipeline_inputs/text_summarization_inputs.py#L14) or under the [official source code](https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/pipelines/text2text_generation.py#L241) from `🤗 Hugging Face`.

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
```

### Upload the Model

When a model is uploaded to a Wallaroo cluster, it is optimized and packaged to make it ready to run as part of a pipeline. In many times, the Wallaroo Server can natively run a model without any Python overhead. In other cases, such as a Python script, a custom Python environment will be automatically generated. This is comparable to the process of "containerizing" a model by adding a small HTTP server and other wrapping around it.

Our pretrained model is in HuggingFace format, which is specified in the `framework` parameter.  The input and output schemas are included as part of the model upload.  For more information, see [Wallaroo SDK Essentials Guide: Model Uploads and Registrations: Hugging Face](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/wallaroo-sdk-model-upload-hugging-face/).

```python
model = wl.upload_model(model_name, 
                        model_file_name, 
                        framework=wallaroo.framework.Framework.HUGGING_FACE_SUMMARIZATION, 
                        input_schema=input_schema, 
                        output_schema=output_schema
                        )
model
```

    Waiting for model loading - this will take up to 10.0min.
    Model is pending loading to a container runtime..
    Model is attempting loading to a container runtime...................successful
    
    Ready

<table>
        <tr>
          <td>Name</td>
          <td>hf-summarization</td>
        </tr>
        <tr>
          <td>Version</td>
          <td>88025146-7eae-4299-a8e2-f30cf6bef103</td>
        </tr>
        <tr>
          <td>File Name</td>
          <td>model-auto-conversion_hugging-face_complex-pipelines_hf-summarisation-bart-large-samsun.zip</td>
        </tr>
        <tr>
          <td>SHA</td>
          <td>ee71d066a83708e7ca4a3c07caf33fdc528bb000039b6ca2ef77fa2428dc6268</td>
        </tr>
        <tr>
          <td>Status</td>
          <td>ready</td>
        </tr>
        <tr>
          <td>Image Path</td>
          <td>proxy.replicated.com/proxy/wallaroo/ghcr.io/wallaroolabs/mlflow-deploy:v2023.3.0-main-3731</td>
        </tr>
        <tr>
          <td>Updated At</td>
          <td>2023-25-Aug 18:51:29</td>
        </tr>
      </table>

### Reserve Pipeline Resources

Before deploying an inference engine we need to tell wallaroo what resources it will need.
To do this we will use the wallaroo DeploymentConfigBuilder() and fill in the options listed below to determine what the properties of our inference engine will be.

We will be testing this deployment for an edge scenario, so the resource specifications are kept small -- what's the minimum needed to meet the expected load on the planned hardware.

- cpus - 8 => allow the engine to use 8 CPU cores when running the neural net
- memory - 8Gi => each inference engine will have 8 GB of memory, which is plenty for processing a single image at a time.

```python
deployment_config = wallaroo.DeploymentConfigBuilder() \
    .cpus(0.25).memory('1Gi') \
    .sidekick_cpus(model, 4) \
    .sidekick_memory(model, "8Gi") \
    .build()
```

### Simulated Edge Deployment

We will now deploy our pipeline into the current Kubernetes environment using the specified resource constraints. This is a "simulated edge" deploy in that we try to mimic the edge hardware as closely as possible.

```python
pipeline = wl.build_pipeline(pipeline_name)
pipeline.add_model_step(model)

pipeline.deploy(deployment_config=deployment_config)
```

<table><tr><th>name</th> <td>edge-hf-summarization</td></tr><tr><th>created</th> <td>2023-08-25 18:53:19.465175+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-25 18:53:20.194327+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>ad77bf95-36c9-4669-9fca-69163f5de601, e0ad32af-21e4-42bd-9eb1-273532cf4f15</td></tr><tr><th>steps</th> <td>hf-summarization</td></tr><tr><th>published</th> <td>False</td></tr></table>

```python
pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.3.90',
       'name': 'engine-56f478fd4d-kngpb',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'edge-hf-summarization',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'hf-summarization',
          'version': '88025146-7eae-4299-a8e2-f30cf6bef103',
          'sha': 'ee71d066a83708e7ca4a3c07caf33fdc528bb000039b6ca2ef77fa2428dc6268',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.4.105',
       'name': 'engine-lb-584f54c899-9qmw9',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': [{'ip': '10.244.3.89',
       'name': 'engine-sidekick-hf-summarization-5-b6bc5994f-7wpdn',
       'status': 'Running',
       'reason': None,
       'details': [],
       'statuses': '\n'}]}

### Run Sample Inference

A single inference using sample input data is prepared below.  We'll run through it to verify the pipeline inference is working.

```python
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

```python
# Adjust timeout as needed, started liberally with a 10 min timeout
out = pipeline.infer(dataframe, timeout=600)
out
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.clean_up_tokenization_spaces</th>
      <th>in.inputs</th>
      <th>in.return_tensors</th>
      <th>in.return_text</th>
      <th>out.summary_text</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-08-25 18:54:01.406</td>
      <td>False</td>
      <td>LinkedIn (/lɪŋktˈɪn/) is a business and employment-focused social media platform that works through websites and mobile apps. It launched on May 5, 2003. It is now owned by Microsoft. The platform is primarily used for professional networking and career development, and allows jobseekers to post their CVs and employers to post jobs. From 2015 most of the company's revenue came from selling access to information about its members to recruiters and sales professionals. Since December 2016, it has been a wholly owned subsidiary of Microsoft. As of March 2023, LinkedIn has more than 900 million registered members from over 200 countries and territories. LinkedIn allows members (both workers and employers) to create profiles and connect with each other in an online social network which may represent real-world professional relationships. Members can invite anyone (whether an existing member or not) to become a connection. LinkedIn can also be used to organize offline events, join groups, write articles, publish job postings, post photos and videos, and more</td>
      <td>False</td>
      <td>True</td>
      <td>LinkedIn is a business and employment-focused social media platform that works through websites and mobile apps. It launched on May 5, 2003. LinkedIn allows members (both workers and employers) to create profiles and connect with each other in an online social network which may represent real-world professional relationships.</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

```python
out["out.summary_text"][0]
```

    'LinkedIn is a business and employment-focused social media platform that works through websites and mobile apps. It launched on May 5, 2003. LinkedIn allows members (both workers and employers) to create profiles and connect with each other in an online social network which may represent real-world professional relationships.'

### Undeploy the Pipeline

Just to clear up resources, we'll undeploy the pipeline.

```python
pipeline.undeploy()
```

<table><tr><th>name</th> <td>edge-hf-summarization</td></tr><tr><th>created</th> <td>2023-08-25 18:53:19.465175+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-25 18:53:20.194327+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>ad77bf95-36c9-4669-9fca-69163f5de601, e0ad32af-21e4-42bd-9eb1-273532cf4f15</td></tr><tr><th>steps</th> <td>hf-summarization</td></tr><tr><th>published</th> <td>False</td></tr></table>

### Publish the Pipeline for Edge Deployment

It worked! For a demo, we'll take working once as "tested". So now that we've tested our pipeline, we are ready to publish it for edge deployment.

Publishing it means assembling all of the configuration files and model assets and pushing them to an Open Container Initiative (OCI) repository set in the Wallaroo instance as the Edge Registry service.  DevOps engineers then retrieve that image and deploy it through Docker, Kubernetes, or similar deployments.

See [Edge Deployment Registry Guide](https://staging.docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-edge-deployment/) for details on adding an OCI Registry Service to Wallaroo as the Edge Deployment Registry.

This is done through the SDK command `wallaroo.pipeline.publish(deployment_config)` which has the following parameters and returns.

#### Publish a Pipeline Parameters

The `publish` method takes the following parameters.  The containerized pipeline will be pushed to the Edge registry service with the model, pipeline configurations, and other artifacts needed to deploy the pipeline.

| Parameter | Type | Description |
|---|---|---|
| `deployment_config` | `wallaroo.deployment_config.DeploymentConfig` (*Optional*) | Sets the pipeline deployment configuration.  For example:    For more information on pipeline deployment configuration, see the [Wallaroo SDK Essentials Guide: Pipeline Deployment Configuration]({{<ref "wallaroo-sdk-essentials-pipeline-deployment-config">}}).

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

### Publish Example

We will now publish the pipeline to our Edge Deployment Registry with the `pipeline.publish(deployment_config)` command.  `deployment_config` is an optional field that specifies the pipeline deployment.  This can be overridden by the DevOps engineer during deployment.

```python
## This may still show an error status despite but if both containers show running it should be good to go
pipeline.publish(deployment_config)
```

    Waiting for pipeline publish... It may take up to 600 sec.
    Pipeline is Publishing.........Published.

<table>
    <tr><td>ID</td><td>5</td></tr>
    <tr><td>Pipeline Version</td><td>af77957c-6af6-4332-aeeb-a4d9e1a22963</td></tr>
    <tr><td>Status</td><td>Published</td></tr>
    <tr><td>Engine URL</td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/engine:v2023.3.0-main-3731'>ghcr.io/wallaroolabs/doc-samples/engine:v2023.3.0-main-3731</a></td></tr>
    <tr><td>Pipeline URL</td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/pipelines/edge-hf-summarization:af77957c-6af6-4332-aeeb-a4d9e1a22963'>ghcr.io/wallaroolabs/doc-samples/pipelines/edge-hf-summarization:af77957c-6af6-4332-aeeb-a4d9e1a22963</a></td></tr>
    <tr><td>Helm Chart URL</td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/charts/edge-hf-summarization'>ghcr.io/wallaroolabs/doc-samples/charts/edge-hf-summarization</a></td></tr>
    <tr><td>Helm Chart Reference</td><td>ghcr.io/wallaroolabs/doc-samples/charts@sha256:f65885c83790bc47f6fc989d55d152fdbfc61df67f64a28857dc10c53aa799ab</td></tr>
    <tr><td>Helm Chart Version</td><td>0.0.1-af77957c-6af6-4332-aeeb-a4d9e1a22963</td></tr>
    <tr><td>Engine Config</td><td>{'engine': {'resources': {'limits': {'cpu': 0.25, 'memory': '1Gi'}, 'requests': {'cpu': 0.25, 'memory': '1Gi'}}}, 'engineAux': {'images': {'hf-summarization-5': {'resources': {'limits': {'cpu': 4.0, 'memory': '8Gi'}, 'requests': {'cpu': 4.0, 'memory': '8Gi'}}}}}, 'enginelb': {}}</td></tr>
    <tr><td>Created By</td><td>john.hummel@wallaroo.ai</td></tr>
    <tr><td>Created At</td><td>2023-08-25 18:54:50.065020+00:00</td></tr>
    <tr><td>Updated At</td><td>2023-08-25 18:54:50.065020+00:00</td></tr>
</table>

### List Published Pipeline

The method `wallaroo.client.list_pipelines()` shows a list of all pipelines in the Wallaroo instance, and includes the `published` field that indicates whether the pipeline was published to the registry (`True`), or has not yet been published (`False`).

```python
wl.list_pipelines()
```

<table><tr><th>name</th><th>created</th><th>last_updated</th><th>deployed</th><th>tags</th><th>versions</th><th>steps</th><th>published</th></tr><tr><td>edge-hf-summarization</td><td>2023-25-Aug 18:53:19</td><td>2023-25-Aug 18:54:49</td><td>False</td><td></td><td>af77957c-6af6-4332-aeeb-a4d9e1a22963, ad77bf95-36c9-4669-9fca-69163f5de601, e0ad32af-21e4-42bd-9eb1-273532cf4f15</td><td>hf-summarization</td><td>True</td></tr><tr><td>hf-summarization-pipeline-edge</td><td>2023-25-Aug 15:52:02</td><td>2023-25-Aug 16:24:04</td><td>False</td><td></td><td>c8d94cce-b237-4d03-bef4-eca89d8d5c88, c7a067bc-997b-47c2-89c7-29ddd507cf7d, c1164da4-e044-49d3-a079-2c6c6a8cdc3f, 28176ea4-5717-4c60-b9c0-91a695bfb78d, 2d55d49d-45d6-4d88-9c6b-a3225a2ba565, 55760fa6-3919-4790-93a2-121be29d1962</td><td>hf-summarization-demoyns2</td><td>True</td></tr><tr><td>houseprice-estimator</td><td>2023-24-Aug 21:16:00</td><td>2023-24-Aug 21:22:22</td><td>False</td><td></td><td>72dbd1e6-2852-4790-885b-9c24865e5126, acd2c0fb-107f-49d6-9041-77ffb8c6979a</td><td>house-price-prime</td><td>False</td></tr><tr><td>edge-pipeline</td><td>2023-24-Aug 16:57:29</td><td>2023-24-Aug 17:05:01</td><td>True</td><td></td><td>710aad65-1437-487b-b6db-0f732b5d2d73, 44c71e77-d8fa-4015-aeee-1cdbccfeb0ef, 7c4383d2-b79d-4179-91a1-b592803e1373, d0a55f2b-0938-45a0-ae58-7d78b9b590d6</td><td>ccfraud</td><td>True</td></tr></table>

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

```python
pipeline.publishes()
```

<table><tr><th>id</th><th>pipeline_version_name</th><th>engine_url</th><th>pipeline_url</th><th>created_by</th><th>created_at</th><th>updated_at</th></tr><tr><td>5</td><td>af77957c-6af6-4332-aeeb-a4d9e1a22963</td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/engine:v2023.3.0-main-3731'>ghcr.io/wallaroolabs/doc-samples/engine:v2023.3.0-main-3731</a></td><td><a href='https://ghcr.io/wallaroolabs/doc-samples/pipelines/edge-hf-summarization:af77957c-6af6-4332-aeeb-a4d9e1a22963'>ghcr.io/wallaroolabs/doc-samples/pipelines/edge-hf-summarization:af77957c-6af6-4332-aeeb-a4d9e1a22963</a></td><td>john.hummel@wallaroo.ai</td><td>2023-25-Aug 18:54:50</td><td>2023-25-Aug 18:54:50</td></tr></table>

## DevOps Deployment

We now have our pipeline published to our Edge Registry service.  We can deploy this in a x86 environment running Docker that is logged into the same registry service that we deployed to.

For more details, check with the documentation on your artifact service.  The following are provided for the three major cloud services:

* [Google: Set up authentication for Docker](https://cloud.google.com/artifact-registry/docs/docker/authentication)
* [Azure: Authenticate with an Azure container registry](https://learn.microsoft.com/en-us/azure/container-registry/container-registry-authentication?tabs=azure-cli)
* [AWS: Authenticating Amazon ECR Repositories for Docker CLI with Credential Helper](https://aws.amazon.com/blogs/compute/authenticating-amazon-ecr-repositories-for-docker-cli-with-credential-helper/)

Using our sample environment, here's sample deployment using Docker.

```bash
docker run -p 8080:8080 \
    -e DEBUG=true -e OCI_REGISTRY={your registry server} \
    -e CONFIG_CPUS=4 \
    -e OCI_USERNAME={registry username here} \
    -e OCI_PASSWORD={registry token here} \
    -e PIPELINE_URL={your registry server}/pipelines/edge-hf-summarization:af77957c-6af6-4332-aeeb-a4d9e1a22963 \
    {your registry server}/engine:v2023.3.0-main-3707
```

Once deployed, we can check the pipelines and models available.  We'll use a `curl` command, but any HTTP based request will work the same way.

The endpoint `/pipelines` returns:

* **id** (*String*):  The name of the pipeline.
* **status** (*String*):  The status as either `Running`, or `Error` if there are any issues.

```bash
curl localhost:8080/pipelines
{"pipelines":[{"id":"edge-cv-retail","status":"Running"}]}
```

The endpoint `/models` returns a List of models with the following fields:

* **name** (*String*): The model name.
* **sha** (*String*): The sha hash value of the ML model.
* **status** (*String*):  The status of either Running or Error if there are any issues.
* **version** (*String*):  The model version.  This matches the version designation used by Wallaroo to track model versions in UUID format.

```bash
curl localhost:8080/models
{"models":[{"name":"resnet-50","sha":"c6c8869645962e7711132a7e17aced2ac0f60dcdc2c7faa79b2de73847a87984","status":"Running","version":"693e19b5-0dc7-4afb-9922-e3f7feefe66d"}]}
```

```python
!curl testboy.local:8080/pipelines
```

    {"pipelines":[{"id":"edge-hf-summarization","status":"Running"}]}

```python
!curl -X POST http://testboy.local:8080/pipelines/edge-hf-summarization -H "Content-Type: application/json; format=pandas-records" -d @./data/test_summarization.df.json
```

    [{"check_failures":[],"elapsed":[257863,2939529889],"model_name":"hf-summarization","model_version":"88025146-7eae-4299-a8e2-f30cf6bef103","original_data":null,"outputs":[{"String":{"data":["LinkedIn is a business and employment-focused social media platform that works through websites and mobile apps. It launched on May 5, 2003. LinkedIn allows members (both workers and employers) to create profiles and connect with each other in an online social network which may represent real-world professional relationships."],"dim":[1,1],"v":1}}],"pipeline_name":"edge-hf-summarization","shadow_data":{},"time":1692990188043}]

```python
import json
import requests

# set the content type and accept headers
headers = {
    'Content-Type': 'application/json; format=pandas-records'
}

# Submit arrow file
dataFile="./data/test_summarization.df.json"

data = json.load(open(dataFile))

host = 'http://testboy.local:8080'

deployurl = f'{host}/pipelines/edge-hf-summarization'

response = requests.post(
                    deployurl, 
                    headers=headers, 
                    json=data, 
                    verify=True
                )

# display(response)
display(response.json()[0]['outputs'][0]['String']['data'][0])

```

    'LinkedIn is a business and employment-focused social media platform that works through websites and mobile apps. It launched on May 5, 2003. LinkedIn allows members (both workers and employers) to create profiles and connect with each other in an online social network which may represent real-world professional relationships.'

