## XGB Classification Upload Tutorial

The following example will:

* Set the input and output schemas.
* Upload a XGB Classification model to Wallaroo.
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
from xgboost import XGBClassifier
```

```python
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

prefix = "xgb-classification"
```

```python
workspace = get_workspace(f"{prefix}-jch")
wl.set_current_workspace(workspace)
```

    {'name': 'xgb-classification-jch', 'id': 90, 'archived': False, 'created_by': 'd9a72bd9-2a1c-44dd-989f-3c7c15130885', 'created_at': '2023-07-05T16:09:39.06093+00:00', 'models': [], 'pipelines': []}

## Data & Model Creation

```python
input_schema = pa.schema([
    pa.field('inputs', pa.list_(pa.float64(), list_size=4))
])

output_schema = pa.schema([
    pa.field('output', pa.float64())
])
```

## Upload model

```python
model = wl.upload_model(f"{prefix}", 'models/model-auto-conversion_xgboost_xgb_classification_iris.pkl', framework=Framework.XGBOOST, input_schema=input_schema, output_schema=output_schema)
model
```

    Waiting for model conversion... It may take up to 10.0min.
    Model is Pending conversion..Converting.Pending conversion..Converting........Ready.

    {'name': 'xgb-classification', 'version': '55501c46-51bf-45c2-8586-a256c74cd1a4', 'file_name': 'model-auto-conversion_xgboost_xgb_classification_iris.pkl', 'image_path': 'proxy.replicated.com/proxy/wallaroo/ghcr.io/wallaroolabs/mlflow-deploy:v2023.3.0-main-3466', 'last_update_time': datetime.datetime(2023, 7, 5, 16, 15, 1, 931747, tzinfo=tzutc())}

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
```

<table><tr><th>name</th> <td>xgb-classification-pipeline</td></tr><tr><th>created</th> <td>2023-07-05 16:10:49.754102+00:00</td></tr><tr><th>last_updated</th> <td>2023-07-05 16:15:04.538435+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>c412b259-90eb-4f58-a31e-c4854f51310f, 1626c864-b4bd-444a-acf4-47dcb664ddeb, 88d34ca1-7196-4a2c-84ae-c880cf9ab189</td></tr><tr><th>steps</th> <td>xgb-classification</td></tr></table>

```python
pipeline.deploy(deployment_config=deployment_config)
pipeline.status()
```

     ok

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.4.81',
       'name': 'engine-854b5f65f5-gxp87',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'xgb-classification-pipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'xgb-classification',
          'version': '55501c46-51bf-45c2-8586-a256c74cd1a4',
          'sha': '4a1844c460e8c8503207305fb807e3a28e788062588925021807c54ee80cc7f9',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.25.40',
       'name': 'engine-lb-584f54c899-nggwp',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': [{'ip': '10.244.4.80',
       'name': 'engine-sidekick-xgb-classification-124-bc674dcdb-8mxv8',
       'status': 'Running',
       'reason': None,
       'details': [],
       'statuses': '\n'}]}

## Inference

```python
data = pd.read_json('data/test-xgboost-classification-data.json')
display(data)

dataframe = pd.DataFrame({"inputs": data[:2].values.tolist()})
display(dataframe)

results = pipeline.infer(dataframe)
display(results)
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

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.inputs</th>
      <th>out.output</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-07-05 16:15:55.802</td>
      <td>[5.1, 3.5, 1.4, 0.2]</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-07-05 16:15:55.802</td>
      <td>[4.9, 3.0, 1.4, 0.2]</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

```python
pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s ..................................... ok

<table><tr><th>name</th> <td>xgb-classification-pipeline</td></tr><tr><th>created</th> <td>2023-07-05 16:10:49.754102+00:00</td></tr><tr><th>last_updated</th> <td>2023-07-05 16:15:38.120839+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>f00ef12f-06a1-4d1c-a0a5-b3e32905d8f3, 244cf4a8-f224-40e9-8ff6-98149f31e47b, 46a2a595-42e9-4bee-b2f5-6475a4a6a7b4, c412b259-90eb-4f58-a31e-c4854f51310f, 1626c864-b4bd-444a-acf4-47dcb664ddeb, 88d34ca1-7196-4a2c-84ae-c880cf9ab189</td></tr><tr><th>steps</th> <td>xgb-classification</td></tr></table>

