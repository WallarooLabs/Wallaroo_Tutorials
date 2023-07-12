## Scikit-Learn Logistic Regression

The following example will:

* Set the input and output schemas.
* Upload a SKLearn Logistic Regression model to Wallaroo.
* Deploy a pipeline with the uploaded SKLearn model as a pipeline step.
* Perform a test inference.
* Undeploy the pipeline.

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
from sklearn.linear_model import LogisticRegression

```

```python
wl = wallaroo.Client()

# wallarooPrefix = ""
# wallarooSuffix = "autoscale-uat-ee.wallaroo.dev"

# wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}api.{wallarooSuffix}", 
#                     auth_endpoint=f"https://{wallarooPrefix}keycloak.{wallarooSuffix}", 
#                     auth_type="sso")
```

    Please log into the following URL in a web browser:
    
    	https://keycloak.autoscale-uat-ee.wallaroo.dev/auth/realms/master/device?user_code=GWEK-NNCP
    
    Login successful!

```python
def get_workspace(name):
    workspace = None
    for ws in wl.list_workspaces():
        if ws.name() == name:
            workspace= ws
    if(workspace == None):
        workspace = wl.create_workspace(name)
    return workspace

prefix = "sklearn-kmeans"
```

```python
workspace = get_workspace(f"{prefix}-jch")
wl.set_current_workspace(workspace)
```

    {'name': 'sklearn-kmeans-jch', 'id': 34, 'archived': False, 'created_by': 'd9a72bd9-2a1c-44dd-989f-3c7c15130885', 'created_at': '2023-06-27T22:01:20.995281+00:00', 'models': [{'name': 'sklearn-kmeans', 'versions': 3, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 7, 5, 14, 53, 37, 366702, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 6, 27, 22, 1, 26, 539571, tzinfo=tzutc())}], 'pipelines': [{'name': 'sklearn-kmeans-pipeline', 'create_time': datetime.datetime(2023, 6, 27, 22, 1, 35, 606699, tzinfo=tzutc()), 'definition': '[]'}]}

## Data & Model Creation

```python
input_schema = pa.schema([
    pa.field('inputs', pa.list_(pa.float64(), list_size=4))
])

output_schema = pa.schema([
    pa.field('predictions', pa.int32())
])
```

## Upload model

```python
model = wl.upload_model(f"{prefix}", 
                        'models/model-auto-conversion_sklearn_kmeans.pkl', 
                        framework=Framework.SKLEARN, 
                        input_schema=input_schema, 
                        output_schema=output_schema)
model
```

    Waiting for model conversion... It may take up to 10.0min.
    Model is Pending conversion........Converting..........Ready.

    {'name': 'sklearn-kmeans', 'version': 'cb9e5d91-a01a-4dc3-bcee-3e4949959779', 'file_name': 'model-auto-conversion_sklearn_kmeans.pkl', 'image_path': 'proxy.replicated.com/proxy/wallaroo/ghcr.io/wallaroolabs/mlflow-deploy:v2023.3.0-main-3466', 'last_update_time': datetime.datetime(2023, 7, 5, 14, 56, 24, 691475, tzinfo=tzutc())}

## Configure model and pipeline

```python
deployment_config = DeploymentConfigBuilder() \
    .cpus(0.25).memory('1Gi') \
    .build()
```

```python
pipeline_name = f"{prefix}-pipeline"
pipeline = wl.build_pipeline(pipeline_name)
pipeline.add_model_step(model)

pipeline.deploy(deployment_config=deployment_config)
pipeline.status()
```

    Waiting for deployment - this will take up to 90s ........... ok

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.7.5',
       'name': 'engine-889d9c745-qk6lw',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'sklearn-kmeans-pipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'sklearn-kmeans',
          'version': 'cb9e5d91-a01a-4dc3-bcee-3e4949959779',
          'sha': 'b378a614854619dd573ec65b9b4ac73d0b397d50a048e733d96b68c5fdbec896',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.5.5',
       'name': 'engine-lb-584f54c899-brj9z',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': [{'ip': '10.244.0.206',
       'name': 'engine-sidekick-sklearn-kmeans-116-f6b696c89-bbn7h',
       'status': 'Running',
       'reason': None,
       'details': [],
       'statuses': '\n'}]}

## Inference

SKLearn models must have all of the data as one line to prevent columns from being read out of order when submitting in JSON.  The following will take in the data, convert the rows into a single `inputs` for the table, then perform the inference.  From the `output_schema` we have defined the output as `predictions` which will be displayed in our inference result output as `out.predictions`.

```python
data = pd.read_json('data/test-sklearn-kmeans.json')
data

# pipeline.infer_from_file('data/test-sklearn-kmeans.json')
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>

```python
dataframe = pd.DataFrame({"inputs": data[:2].values.tolist()})
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
      <td>[5.1, 3.5, 1.4, 0.2]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[4.9, 3.0, 1.4, 0.2]</td>
    </tr>
  </tbody>
</table>

```python
pipeline.infer(dataframe)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.inputs</th>
      <th>out.predictions</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-07-05 15:02:15.030</td>
      <td>[5.1, 3.5, 1.4, 0.2]</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-07-05 15:02:15.030</td>
      <td>[4.9, 3.0, 1.4, 0.2]</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

```python
pipeline.undeploy()
```
