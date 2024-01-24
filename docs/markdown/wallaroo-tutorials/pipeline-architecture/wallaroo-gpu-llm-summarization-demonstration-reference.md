This tutorial is available on the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/blob/main/pipeline-architecture/wallaroo-gpu-llm-summarization).

This tutorial demonstrates how to use the Wallaroo combined with GPU processors to perform inferences with pre-trained computer vision ML models.  This demonstration assumes that:

* A Wallaroo version 2023.3 or above instance is installed.
* A nodepools with GPUs part of the Kubernetes cluster.  See [Create GPU Nodepools for Kubernetes Clusters](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-install-guides/wallaroo-install-configurations/wallaroo-gpu-nodepools/) for more detials.
* The model [`hf-summarization-bart-large-samsun.zip` (1.4 G)](https://storage.googleapis.com/wallaroo-public-data/llm-models/model-auto-conversion_hugging-face_complex-pipelines_hf-summarisation-bart-large-samsun.zip) has been downloaded to the `./models` folder.

### Tutorial Goals

For our example, we will perform the following:

* Create a workspace for our work.
* Upload the model.
* Create a pipeline and specify the gpus in the pipeline deployment.
* Perform a sample inference.

## Steps

### Import Libraries

The first step will be to import our libraries.

```python
import json
import os

import wallaroo
from wallaroo.pipeline   import Pipeline
from wallaroo.deployment_config import DeploymentConfigBuilder
from wallaroo.framework import Framework

import pyarrow as pa
import numpy as np
import pandas as pd
```

### Connect to the Wallaroo Instance

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
wl = wallaroo.Client()
```

### Configure PyArrow Schema

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

### Upload Model

We will now create or connect to our pipeline and upload the model.

```python
model = wl.upload_model('hf-summarization-yns', 
                        'hf-summarisation-bart-large-samsun.zip', 
                        framework=Framework.HUGGING_FACE_SUMMARIZATION, 
                        input_schema=input_schema, 
                        output_schema=output_schema)
model
```

    Waiting for model loading - this will take up to 10.0min.
    Model is pending loading to a container runtime..
    Model is attempting loading to a container runtime.......................successful
    
    Ready

<table>
        <tr>
          <td>Name</td>
          <td>hf-summarization-yns</td>
        </tr>
        <tr>
          <td>Version</td>
          <td>2f708f1b-0ace-448b-b4ab-a337c962e6d9</td>
        </tr>
        <tr>
          <td>File Name</td>
          <td>hf-summarisation-bart-large-samsun.zip</td>
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
          <td>proxy.replicated.com/proxy/wallaroo/ghcr.io/wallaroolabs/mlflow-deploy:v2023.3.0-3798</td>
        </tr>
        <tr>
          <td>Updated At</td>
          <td>2023-08-Sep 13:17:33</td>
        </tr>
      </table>

### Deploy Pipeline

With the model uploaded, we can add it is as a step in the pipeline, then deploy it.  

For GPU deployment, the pipeline deployment configuration allocated the cpus, ram, gpus, and other settings for the pipeline.  For gpus,  both the number of GPUs and the nodepool containing the gpus must be specified.

For Wallaroo Native Runtime models (`onnx`, `tensorflow`), the method is `wallaroo.deployment_config.gpus(int)` to allocate the number of gpus to the pipeline.  This applies to all Wallaroo Native Runtime models in the pipeline.

For Wallaroo Containerized models (`hugging-face`, etc), the method is `wallaroo.deployment_config.sidekick_gpus(int)` to allocate the number of gpus to the model.

The deployment label is set with the `wallaroo.deployment_config.deployment_label(string)` method.

For more information on allocating resources to a Wallaroo pipeline for deployment, see [Wallaroo SDK Essentials Guide: Pipeline Deployment Configuration](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline-deployment-config/).

For this example, 1 gpu will be allocated to the pipeline from the nodepool with the deployment label `wallaroo.ai/accelerator: a100`.

```python
deployment_config = DeploymentConfigBuilder() \
    .cpus(1).memory('1Gi') \
    .sidekick_gpus(model, 1) \
    .sidekick_cpus(model,4) \
    .sidekick_memory(model, '8Gi') \
    .deployment_label('wallaroo.ai/accelerator: a100') \
    .build()
```

```python
pipeline_name = "hf-summarization-pipeline"
```

```python
pipeline = wl.build_pipeline(pipeline_name)
pipeline.add_model_step(model)

pipeline.deploy(deployment_config=deployment_config)
```

### Run inference

We can now run a sample inference using the `wallaroo.pipeline.infer` method and display the results.

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
      <td>LinkedIn (/lɪŋktˈɪn/) is a business and employ...</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>

```python
out = pipeline.infer(dataframe)
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
      <td>2023-09-08 13:18:58.557</td>
      <td>False</td>
      <td>LinkedIn (/lɪŋktˈɪn/) is a business and employ...</td>
      <td>False</td>
      <td>True</td>
      <td>LinkedIn is a business and employment-focused ...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

```python
out["out.summary_text"][0]
```

    'LinkedIn is a business and employment-focused social media platform that works through websites and mobile apps. It launched on May 5, 2003. LinkedIn allows members (both workers and employers) to create profiles and connect with each other in an online social network which may represent real-world professional relationships.'

### Model Inferencing with Pipeline Deployment Endpoint

The other option is to use the pipeline's inference endpoint.

```python
!curl -X POST {pipeline.url()} \
    -H "Content-Type: application/json; format=pandas-records" \
        -d @./data/test_summarization.json
```

    [{"time":1694179270999,"in":{"clean_up_tokenization_spaces":[false],"inputs":["LinkedIn (/lɪŋktˈɪn/) is a business and employment-focused social media platform that works through websites and mobile apps. It launched on May 5, 2003. It is now owned by Microsoft. The platform is primarily used for professional networking and career development, and allows jobseekers to post their CVs and employers to post jobs. From 2015 most of the company's revenue came from selling access to information about its members to recruiters and sales professionals. Since December 2016, it has been a wholly owned subsidiary of Microsoft. As of March 2023, LinkedIn has more than 900 million registered members from over 200 countries and territories. LinkedIn allows members (both workers and employers) to create profiles and connect with each other in an online social network which may represent real-world professional relationships. Members can invite anyone (whether an existing member or not) to become a connection. LinkedIn can also be used to organize offline events, join groups, write articles, publish job postings, post photos and videos, and more"],"return_tensors":[false],"return_text":[true]},"out":{"summary_text":"LinkedIn is a business and employment-focused social media platform that works through websites and mobile apps. It launched on May 5, 2003. LinkedIn allows members (both workers and employers) to create profiles and connect with each other in an online social network which may represent real-world professional relationships."},"check_failures":[],"metadata":{"last_model":"{\"model_name\":\"hf-summarization-yns\",\"model_sha\":\"ee71d066a83708e7ca4a3c07caf33fdc528bb000039b6ca2ef77fa2428dc6268\"}","pipeline_version":"4aeb608f-166b-4b59-bb10-c06f9e49df23","elapsed":[41800,4294967295],"dropped":[]}}]

### Undeploy the Pipeline

With the demonstration complete, we can undeploy the pipeline and return the resources back to the Wallaroo instance.

```python
pipeline.undeploy()
```
