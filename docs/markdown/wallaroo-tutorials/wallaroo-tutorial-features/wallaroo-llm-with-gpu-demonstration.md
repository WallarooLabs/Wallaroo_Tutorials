This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-features/gpu-deployment).

## Large Language Model with GPU Pipeline Deployment in Wallaroo Demonstration

Wallaroo supports the use of GPUs for model deployment and inferences.  This demonstration demonstrates using a Hugging Face Large Language Model (LLM) stored in a registry service that creates summaries of larger text strings.

### Tutorial Goals

For this demonstration, a cluster with GPU resources will be hosting the Wallaroo instance.

1. The containerized model `hf-bart-summarizer3` will be registered to a Wallaroo workspace.
1. The model will be added as a step to a Wallaroo pipeline.
1. When the pipeline is deployed, the deployment configuration will specify the allocation of a GPU to the pipeline.
1. A sample inference summarizing a set of text is used as an inference input, and the sample results and time period displayed.

### Prerequisites

The following is required for this tutorial:

* A Wallaroo Enterprise version 2023.2.1 or greater instance installed into a  GPU enabled Kubernetes cluster as described in the [Wallaroo Create GPU Nodepools Kubernetes Clusters guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-install-guides/wallaroo-install-configurations/wallaroo-gpu-nodepools/).
* The Wallaroo SDK version 2023.2.1 or greater.

### References

* [Wallaroo SDK Essentials Guide: Pipeline Deployment Configuration](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline-deployment-config/)
* [Wallaroo SDK Reference wallaroo.deployment_config](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/deployment_config/)

## Tutorial Steps

### Import Libraries

The first step is to import the libraries we'll be using.  These are included by default in the Wallaroo instance's JupyterHub service.

```python
import json
import os
import pickle

import wallaroo
from wallaroo.pipeline   import Pipeline
from wallaroo.deployment_config import DeploymentConfigBuilder
from wallaroo.framework import Framework

import pyarrow as pa
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
```

### Connect to the Wallaroo Instance through the User Interface

The next step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
wl = wallaroo.Client()
```

### Register MLFlow Model in Wallaroo

MLFlow Containerized Model require the input and output schemas be defined in Apache Arrow format.  Both the input and output schema is a string.

Once complete, the MLFlow containerized model is registered to the Wallaroo workspace.

```python
input_schema = pa.schema([
    pa.field('inputs', pa.string())
])
output_schema = pa.schema([
    pa.field('summary_text', pa.string()),
])

model = wl.register_model_image(
    name="hf-bart-summarizer3",
    image=f"sampleregistry.com/gpu-hf-summ-official2:1.30"
).configure("mlflow", input_schema=input_schema, output_schema=output_schema)
```

```python
model
```

<table>
        <tr>
          <td>Name</td>
          <td>hf-bart-summarizer3</td>
        </tr>
        <tr>
          <td>Version</td>
          <td>d511a20c-9612-4112-9368-2d79ae764dec</td>
        </tr>
        <tr>
          <td>File Name</td>
          <td>none</td>
        </tr>
        <tr>
          <td>SHA</td>
          <td>360dcd343a593e87639106757bad58a7d960899c915bbc9787e7601073bc1121</td>
        </tr>
        <tr>
          <td>Status</td>
          <td>ready</td>
        </tr>
        <tr>
          <td>Image Path</td>
          <td>proxy.replicated.com/proxy/wallaroo/ghcr.io/wallaroolabs/gpu-hf-summ-official2:1.30</td>
        </tr>
        <tr>
          <td>Updated At</td>
          <td>2023-11-Jul 19:23:57</td>
        </tr>
      </table>

### Pipeline Deployment With GPU

The registered model will be added to our sample pipeline as a pipeline step.  When the pipeline is deployed, a specific resource configuration is applied that allocated a GPU to our MLFlow containerized model.

MLFlow models are run in the Containerized Runtime in the pipeline.  As such, the `DeploymentConfigBuilder` method `.sidekick_gpus(model: wallaroo.model.Model, core_count: int)` is used to allocate 1 GPU to our model.

The pipeline is then deployed with our deployment configuration, and a GPU from the cluster is allocated for use by this model.

```python
pipeline_name = f"test-gpu7"
pipeline = wl.build_pipeline(pipeline_name)
pipeline.add_model_step(model)

deployment_config = DeploymentConfigBuilder() \
    .cpus(0.25).memory('1Gi').gpus(0) \
    .sidekick_gpus(model, 1) \
    .sidekick_env(model, {"GUNICORN_CMD_ARGS": "--timeout=180 --workers=1"}) \
    .image("proxy.replicated.com/proxy/wallaroo/ghcr.io/wallaroolabs/fitzroy-mini-cuda:v2023.3.0-josh-fitzroy-gpu-3374") \
    .build()
deployment_config
```

    {'engine': {'cpu': 0.25,
      'resources': {'limits': {'cpu': 0.25, 'memory': '1Gi', 'nvidia.com/gpu': 0},
       'requests': {'cpu': 0.25, 'memory': '1Gi', 'nvidia.com/gpu': 0}},
      'gpu': 0,
      'image': 'proxy.replicated.com/proxy/wallaroo/ghcr.io/wallaroolabs/fitzroy-mini-cuda:v2023.3.0-josh-fitzroy-gpu-3374'},
     'enginelb': {},
     'engineAux': {'images': {'hf-bart-summarizer3-28': {'resources': {'limits': {'nvidia.com/gpu': 1},
         'requests': {'nvidia.com/gpu': 1}},
        'env': [{'name': 'GUNICORN_CMD_ARGS',
          'value': '--timeout=180 --workers=1'}]}}},
     'node_selector': {}}

```python
pipeline.deploy(deployment_config=deployment_config)
pipeline.status()
```

    Waiting for deployment - this will take up to 90s ................ ok

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.38.26',
       'name': 'engine-7457c88db4-42ww6',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'test-gpu7',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'hf-bart-summarizer3',
          'version': 'd511a20c-9612-4112-9368-2d79ae764dec',
          'sha': '360dcd343a593e87639106757bad58a7d960899c915bbc9787e7601073bc1121',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.0.113',
       'name': 'engine-lb-584f54c899-ht5cd',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': [{'ip': '10.244.41.21',
       'name': 'engine-sidekick-hf-bart-summarizer3-28-f5f8d6567-zzh62',
       'status': 'Running',
       'reason': None,
       'details': [],
       'statuses': '\n'}]}

```python
pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.38.26',
       'name': 'engine-7457c88db4-42ww6',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'test-gpu7',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'hf-bart-summarizer3',
          'version': 'd511a20c-9612-4112-9368-2d79ae764dec',
          'sha': '360dcd343a593e87639106757bad58a7d960899c915bbc9787e7601073bc1121',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.0.113',
       'name': 'engine-lb-584f54c899-ht5cd',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': [{'ip': '10.244.41.21',
       'name': 'engine-sidekick-hf-bart-summarizer3-28-f5f8d6567-zzh62',
       'status': 'Running',
       'reason': None,
       'details': [],
       'statuses': '\n'}]}

### Sample Text Inference

A sample inference is performed 10 times using the definition of LinkedIn, and the time to completion displayed.  In this case, the total time to create a summary of the text multiple times is around 2 seconds per inference request.

```python
input_data = {
    "inputs": ["LinkedIn (/lɪŋktˈɪn/) is a business and employment-focused social media platform that works through websites and mobile apps. It launched on May 5, 2003. It is now owned by Microsoft. The platform is primarily used for professional networking and career development, and allows jobseekers to post their CVs and employers to post jobs. From 2015 most of the company's revenue came from selling access to information about its members to recruiters and sales professionals. Since December 2016, it has been a wholly owned subsidiary of Microsoft. As of March 2023, LinkedIn has more than 900 million registered members from over 200 countries and territories. LinkedIn allows members (both workers and employers) to create profiles and connect with each other in an online social network which may represent real-world professional relationships. Members can invite anyone (whether an existing member or not) to become a connection. LinkedIn can also be used to organize offline events, join groups, write articles, publish job postings, post photos and videos, and more."]
}
```

```python
dataframe = pd.DataFrame(input_data)
dataframe.to_json('test_data.json', orient='records')
```

```python
dataframe
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>inputs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LinkedIn (/lɪŋktˈɪn/) is a business and employ...</td>
    </tr>
  </tbody>
</table>

```python
import time
```

```python

start = time.time()

end = time.time()

end - start
```

    2.765655517578125e-05

```python
start = time.time()
elapsed_time = 0
for i in range(10):
    s = time.time()
    res = pipeline.infer_from_file('test_data.json', timeout=120)
    print(res)
    e = time.time()

    el = e-s
    print(el)
end = time.time()

elapsed_time += end - start
print('Execution time:', elapsed_time, 'seconds')
```

                         time                                          in.inputs  \
    0 2023-07-11 19:27:50.806  LinkedIn (/lɪŋktˈɪn/) is a business and employ...   
    
                                        out.summary_text  check_failures  
    0  LinkedIn is a business and employment-focused ...               0  
    2.616016387939453
                         time                                          in.inputs  \
    0 2023-07-11 19:27:53.421  LinkedIn (/lɪŋktˈɪn/) is a business and employ...   
    
                                        out.summary_text  check_failures  
    0  LinkedIn is a business and employment-focused ...               0  
    2.478372097015381
                         time                                          in.inputs  \
    0 2023-07-11 19:27:55.901  LinkedIn (/lɪŋktˈɪn/) is a business and employ...   
    
                                        out.summary_text  check_failures  
    0  LinkedIn is a business and employment-focused ...               0  
    2.453855514526367
                         time                                          in.inputs  \
    0 2023-07-11 19:27:58.365  LinkedIn (/lɪŋktˈɪn/) is a business and employ...   
    
                                        out.summary_text  check_failures  
    0  LinkedIn is a business and employment-focused ...               0  
    2.4600493907928467
                         time                                          in.inputs  \
    0 2023-07-11 19:28:00.819  LinkedIn (/lɪŋktˈɪn/) is a business and employ...   
    
                                        out.summary_text  check_failures  
    0  LinkedIn is a business and employment-focused ...               0  
    2.461345672607422
                         time                                          in.inputs  \
    0 2023-07-11 19:28:03.273  LinkedIn (/lɪŋktˈɪn/) is a business and employ...   
    
                                        out.summary_text  check_failures  
    0  LinkedIn is a business and employment-focused ...               0  
    2.4581406116485596
                         time                                          in.inputs  \
    0 2023-07-11 19:28:05.732  LinkedIn (/lɪŋktˈɪn/) is a business and employ...   
    
                                        out.summary_text  check_failures  
    0  LinkedIn is a business and employment-focused ...               0  
    2.4555394649505615
                         time                                          in.inputs  \
    0 2023-07-11 19:28:08.192  LinkedIn (/lɪŋktˈɪn/) is a business and employ...   
    
                                        out.summary_text  check_failures  
    0  LinkedIn is a business and employment-focused ...               0  
    2.4681003093719482
                         time                                          in.inputs  \
    0 2023-07-11 19:28:10.657  LinkedIn (/lɪŋktˈɪn/) is a business and employ...   
    
                                        out.summary_text  check_failures  
    0  LinkedIn is a business and employment-focused ...               0  
    2.4639062881469727
                         time                                          in.inputs  \
    0 2023-07-11 19:28:13.120  LinkedIn (/lɪŋktˈɪn/) is a business and employ...   
    
                                        out.summary_text  check_failures  
    0  LinkedIn is a business and employment-focused ...               0  
    2.4664926528930664
    Execution time: 24.782114267349243 seconds

```python
elapsed_time / 10
```

    2.4782114267349242

### Undeploy the Pipeline

With the inferences completed, the pipeline is undeployed.  This returns the resources back to the cluster for use by other pipeline.

```python
pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s ..............
