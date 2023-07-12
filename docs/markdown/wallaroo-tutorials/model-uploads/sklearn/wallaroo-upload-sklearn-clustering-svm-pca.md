## Scikit-Learn Clustering KMeans Testing

The following example will:

* Upload and convert a 

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

wl = wallaroo.Client(auth_type="sso", interactive=True)
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
```

```python
workspace = get_workspace("sklearn-svm-pca-workspace")
wl.set_current_workspace(workspace)
```

    {'name': 'sklearn-svm-pca-workspace', 'id': 87, 'archived': False, 'created_by': 'd9a72bd9-2a1c-44dd-989f-3c7c15130885', 'created_at': '2023-07-05T15:09:56.931033+00:00', 'models': [], 'pipelines': []}

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
model = wl.upload_model('sklearn-svm-pca', 'models/model-auto-conversion_sklearn_svm_pca_pipeline.pkl', framework=Framework.SKLEARN, input_schema=input_schema, output_schema=output_schema)

model
```

    Waiting for model conversion... It may take up to 10.0min.
    Model is Pending conversion..Converting..Pending conversion..Converting.........Ready.

    {'name': 'sklearn-svm-pca', 'version': '1a0638bc-a5df-49f6-9b8c-34a2eeccf113', 'file_name': 'model-auto-conversion_sklearn_svm_pca_pipeline.pkl', 'image_path': 'proxy.replicated.com/proxy/wallaroo/ghcr.io/wallaroolabs/mlflow-deploy:v2023.3.0-main-3466', 'last_update_time': datetime.datetime(2023, 7, 5, 15, 11, 14, 52542, tzinfo=tzutc())}

## Configure model and pipeline

```python
deployment_config = DeploymentConfigBuilder() \
    .cpus(0.25).memory('1Gi') \
    .build()
```

```python
pipeline_name = f"sklearn-svm-pca-pipeline"
pipeline = wl.build_pipeline(pipeline_name)
pipeline.add_model_step(model)

pipeline.deploy(deployment_config=deployment_config)
pipeline.status()
```

    Waiting for deployment - this will take up to 90s ........... ok

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.12.5',
       'name': 'engine-5d69f9c58d-rwjhw',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'sklearn-svm-pca-pipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'sklearn-svm-pca',
          'version': '1a0638bc-a5df-49f6-9b8c-34a2eeccf113',
          'sha': '524b05d22f13fa4ce5feaf07b86710b447f0c80a02601be86ee5b6bc748fe7fd',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.15.5',
       'name': 'engine-lb-584f54c899-w94f6',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': [{'ip': '10.244.0.211',
       'name': 'engine-sidekick-sklearn-svm-pca-117-5bdbd47dd6-r7vdb',
       'status': 'Running',
       'reason': None,
       'details': [],
       'statuses': '\n'}]}

## Inference

```python
data = pd.read_json('data/test-sklearn-kmeans.json')
display(data)

# move the column values to a single array input
dataframe = pd.DataFrame({"inputs": data[:2].values.tolist()})
display(dataframe)
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
      <td>2023-07-05 15:11:29.776</td>
      <td>[5.1, 3.5, 1.4, 0.2]</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-07-05 15:11:29.776</td>
      <td>[4.9, 3.0, 1.4, 0.2]</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

```python
pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s ..................................... ok

<table><tr><th>name</th> <td>sklearn-svm-pca-pipeline</td></tr><tr><th>created</th> <td>2023-07-05 15:11:17.606017+00:00</td></tr><tr><th>last_updated</th> <td>2023-07-05 15:11:17.641020+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>a12c7fe4-e1ac-4783-a310-b51bd68d86d6, e69307a8-976d-477c-b810-96f60337b893</td></tr><tr><th>steps</th> <td>sklearn-svm-pca</td></tr></table>

```python

```
