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

prefix = "xgb-rf-regressor"
```

```python
workspace = get_workspace(f"{prefix}-jch")
wl.set_current_workspace(workspace)
```

    {'name': 'xgb-rf-regressor-jch', 'id': 94, 'archived': False, 'created_by': 'd9a72bd9-2a1c-44dd-989f-3c7c15130885', 'created_at': '2023-07-05T16:40:41.858312+00:00', 'models': [], 'pipelines': []}

## Data Schema

```python
input_schema = pa.schema([
    pa.field('inputs', pa.list_(pa.float64(), list_size=10))
])

output_schema = pa.schema([
    pa.field('output', pa.float64())
])
```

## Upload model

```python
model = wl.upload_model(f"{prefix}", 'models/model-auto-conversion_xgboost_xgb_rf_regressor_diabetes.pkl', framework=Framework.XGBOOST, input_schema=input_schema, output_schema=output_schema)
model
```

    Waiting for model conversion... It may take up to 10.0min.
    Model is Pending conversion..Converting.Pending conversion..Converting.........Ready.

    {'name': 'xgb-rf-regressor', 'version': 'cdd35c48-e19d-41bf-b250-93cf294396eb', 'file_name': 'model-auto-conversion_xgboost_xgb_rf_regressor_diabetes.pkl', 'image_path': 'proxy.replicated.com/proxy/wallaroo/ghcr.io/wallaroolabs/mlflow-deploy:v2023.3.0-main-3466', 'last_update_time': datetime.datetime(2023, 7, 5, 16, 41, 49, 39162, tzinfo=tzutc())}

## Configure model and pipeline

```python
deployment_config = DeploymentConfigBuilder() \
    .cpus(0.25).memory('1Gi') \
    .build()
```

```python
pipeline_name = f"{prefix}-pipeline"
pipeline = wl.build_pipeline(pipeline_name)
pipeline.clear()
pipeline.add_model_step(model)
```

<table><tr><th>name</th> <td>xgb-rf-regressor-pipeline</td></tr><tr><th>created</th> <td>2023-07-05 16:41:52.513989+00:00</td></tr><tr><th>last_updated</th> <td>2023-07-05 16:41:52.513989+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>24cebf13-1732-4c0e-b0b1-b2e534a87fe6</td></tr><tr><th>steps</th> <td></td></tr></table>

```python
pipeline.deploy(deployment_config=deployment_config)
pipeline.status()
```

    Waiting for deployment - this will take up to 90s ............... ok

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.19.5',
       'name': 'engine-6dcc8595c8-czwmh',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'xgb-rf-regressor-pipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'xgb-rf-regressor',
          'version': 'cdd35c48-e19d-41bf-b250-93cf294396eb',
          'sha': '461341d78d54a9bfc8e4faa94be6037aef15217974ba59bad92d31ef48e6bd99',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.20.5',
       'name': 'engine-lb-584f54c899-7s6j2',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': [{'ip': '10.244.4.124',
       'name': 'engine-sidekick-xgb-rf-regressor-130-df5f7df47-6fw2d',
       'status': 'Running',
       'reason': None,
       'details': [],
       'statuses': '\n'}]}

## Inference

```python
data = pd.read_json('./data/test_xgb_rf-regressor.json')
display(data)

dataframe = pd.DataFrame({"inputs": data[:2].values.tolist()})
display(dataframe)

pipeline.infer(dataframe)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>bp</th>
      <th>s1</th>
      <th>s2</th>
      <th>s3</th>
      <th>s4</th>
      <th>s5</th>
      <th>s6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.038076</td>
      <td>0.050680</td>
      <td>0.061696</td>
      <td>0.021872</td>
      <td>-0.044223</td>
      <td>-0.034821</td>
      <td>-0.043401</td>
      <td>-0.002592</td>
      <td>0.019907</td>
      <td>-0.017646</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.001882</td>
      <td>-0.044642</td>
      <td>-0.051474</td>
      <td>-0.026328</td>
      <td>-0.008449</td>
      <td>-0.019163</td>
      <td>0.074412</td>
      <td>-0.039493</td>
      <td>-0.068332</td>
      <td>-0.092204</td>
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
      <td>[0.0380759064, 0.0506801187, 0.0616962065, 0.0...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[-0.0018820165, -0.0446416365, -0.051474061200...</td>
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
      <td>2023-07-05 16:43:10.778</td>
      <td>[0.0380759064, 0.0506801187, 0.0616962065, 0.0...</td>
      <td>166.618774</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-07-05 16:43:10.778</td>
      <td>[-0.0018820165, -0.0446416365, -0.0514740612, ...</td>
      <td>76.189583</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

```python
pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s .................................... ok

<table><tr><th>name</th> <td>xgb-rf-regressor-pipeline</td></tr><tr><th>created</th> <td>2023-07-05 16:41:52.513989+00:00</td></tr><tr><th>last_updated</th> <td>2023-07-05 16:41:52.590954+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>18ac4337-1a4f-434a-ac1c-d128482c4ea5, 24cebf13-1732-4c0e-b0b1-b2e534a87fe6</td></tr><tr><th>steps</th> <td>xgb-rf-regressor</td></tr></table>

```python

```
