This tutorial is available on the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/pipeline-architecture/wallaroo-arm-llm-summarization).

This tutorial demonstrates how to use the Wallaroo combined with ARM processors to perform inferences with pre-trained computer vision ML models.  This demonstration assumes that:

* A Wallaroo version 2023.3 or above instance is installed.
* A nodepools with ARM architecture virtual machines are part of the Kubernetes cluster.  For example, Azure supports Ampere® Altra® Arm-based processor included with the following virtual machines:
  * [Dpsv5 and Dpdsv5-series](https://learn.microsoft.com/en-us/azure/virtual-machines/dpsv5-dpdsv5-series)
  * [Epsv5 and Epdsv5-series](https://learn.microsoft.com/en-us/azure/virtual-machines/epsv5-epdsv5-series)
* The model [`hf-summarization-bart-large-samsun.zip` (1.4 G)](https://storage.googleapis.com/wallaroo-public-data/llm-models/model-auto-conversion_hugging-face_complex-pipelines_hf-summarisation-bart-large-samsun.zip) has been downloaded to the `./models` folder.

### Tutorial Goals

For our example, we will perform the following:

* Create a workspace for our work.
* Upload the the resnet computer vision model model.
* Create a pipeline using the default architecture that can ingest our submitted data, submit it to the model, and export the results while tracking how long the inference took.
* Redeploy the same pipeline on the ARM architecture, then perform the same inference on the same data and model and track how long the inference took.
* Compare the inference timing through the default architecture versus the ARM architecture.

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

We will now create or connect to our pipeline and upload the model.  We will set the architecture of the model to ARM for its deployment.

```python
from wallaroo.engine_config import Architecture
model = wl.upload_model('hf-summarization-yns', 
                        'hf-summarisation-bart-large-samsun.zip', 
                        framework=Framework.HUGGING_FACE_SUMMARIZATION, 
                        input_schema=input_schema, 
                        output_schema=output_schema, 
                        arch=Architecture.ARM)
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

With the model uploaded, we can add it is as a step in the pipeline, then deploy it.  The model has already been set as `arm` so the pipeline will use that architecture.

```python
deployment_config = DeploymentConfigBuilder() \
    .cpus(0.25).memory('1Gi') \
    .sidekick_cpus(model, 4) \
    .sidekick_memory(model, "8Gi") \
    .build()
```

```python
pipeline_name = "hf-summarization-pipeline-arm"
```

```python
pipeline = wl.build_pipeline(pipeline_name)
pipeline.add_model_step(model)

pipeline.deploy(deployment_config=deployment_config)
pipeline.status()
```

    Waiting for deployment - this will take up to 90s .................................. ok

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.2.53',
       'name': 'engine-8494968846-jdj28',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'hf-summarization-pipeline-arm',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'hf-summarization-yns',
          'version': '2f708f1b-0ace-448b-b4ab-a337c962e6d9',
          'sha': 'ee71d066a83708e7ca4a3c07caf33fdc528bb000039b6ca2ef77fa2428dc6268',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.3.34',
       'name': 'engine-lb-584f54c899-2zkz6',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': [{'ip': '10.244.2.52',
       'name': 'engine-sidekick-hf-summarization-yns-6-6555bb7d74-27ncn',
       'status': 'Running',
       'reason': None,
       'details': [],
       'statuses': '\n'}]}

```python
pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.2.53',
       'name': 'engine-8494968846-jdj28',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'hf-summarization-pipeline-arm',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'hf-summarization-yns',
          'version': '2f708f1b-0ace-448b-b4ab-a337c962e6d9',
          'sha': 'ee71d066a83708e7ca4a3c07caf33fdc528bb000039b6ca2ef77fa2428dc6268',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.3.34',
       'name': 'engine-lb-584f54c899-2zkz6',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': [{'ip': '10.244.2.52',
       'name': 'engine-sidekick-hf-summarization-yns-6-6555bb7d74-27ncn',
       'status': 'Running',
       'reason': None,
       'details': [],
       'statuses': '\n'}]}

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
pipeline.url()
```

    'http://engine-lb.hf-summarization-pipeline-arm-3:29502/pipelines/hf-summarization-pipeline-arm'

```python
!curl -X POST http://engine-lb.hf-summarization-pipeline-arm-3:29502/pipelines/hf-summarization-pipeline-arm \
    -H "Content-Type: application/json; format=pandas-records" \
        -d @./data/test_summarization.json
```

    [{"time":1694179270999,"in":{"clean_up_tokenization_spaces":[false],"inputs":["LinkedIn (/lɪŋktˈɪn/) is a business and employment-focused social media platform that works through websites and mobile apps. It launched on May 5, 2003. It is now owned by Microsoft. The platform is primarily used for professional networking and career development, and allows jobseekers to post their CVs and employers to post jobs. From 2015 most of the company's revenue came from selling access to information about its members to recruiters and sales professionals. Since December 2016, it has been a wholly owned subsidiary of Microsoft. As of March 2023, LinkedIn has more than 900 million registered members from over 200 countries and territories. LinkedIn allows members (both workers and employers) to create profiles and connect with each other in an online social network which may represent real-world professional relationships. Members can invite anyone (whether an existing member or not) to become a connection. LinkedIn can also be used to organize offline events, join groups, write articles, publish job postings, post photos and videos, and more"],"return_tensors":[false],"return_text":[true]},"out":{"summary_text":"LinkedIn is a business and employment-focused social media platform that works through websites and mobile apps. It launched on May 5, 2003. LinkedIn allows members (both workers and employers) to create profiles and connect with each other in an online social network which may represent real-world professional relationships."},"check_failures":[],"metadata":{"last_model":"{\"model_name\":\"hf-summarization-yns\",\"model_sha\":\"ee71d066a83708e7ca4a3c07caf33fdc528bb000039b6ca2ef77fa2428dc6268\"}","pipeline_version":"4aeb608f-166b-4b59-bb10-c06f9e49df23","elapsed":[41800,4294967295],"dropped":[]}}]

### Undeploy the Pipeline

With the demonstration complete, we can undeploy the pipeline and return the resources back to the Wallaroo instance.

```python
pipeline.undeploy()
```
