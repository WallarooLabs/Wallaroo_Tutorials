## XGB Random Forest Classification Upload Tutorial

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

# wallarooPrefix = ""
# wallarooSuffix = "autoscale-uat-ee.wallaroo.dev"

# wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}api.{wallarooSuffix}", 
#                     auth_endpoint=f"https://{wallarooPrefix}keycloak.{wallarooSuffix}", 
#                     auth_type="sso")
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

prefix = "xgb-rf-classification"
```

```python
workspace = get_workspace(f"{prefix}-jch")
wl.set_current_workspace(workspace)
```

    {'name': 'xgb-rf-classification-jch', 'id': 93, 'archived': False, 'created_by': 'd9a72bd9-2a1c-44dd-989f-3c7c15130885', 'created_at': '2023-07-05T16:37:12.345096+00:00', 'models': [], 'pipelines': []}

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
model = wl.upload_model(f"{prefix}", 'models/model-auto-conversion_xgboost_xgb_rf_classification_iris.pkl', framework=Framework.XGBOOST, input_schema=input_schema, output_schema=output_schema)
model
```

    Waiting for model conversion... It may take up to 10.0min.
    Model is Pending conversion..Converting.Pending conversion..Converting..........Ready.

    {'name': 'xgb-rf-classification', 'version': '23a587f2-39cd-42a1-bfef-380a1c2cf2c2', 'file_name': 'model-auto-conversion_xgboost_xgb_rf_classification_iris.pkl', 'image_path': 'proxy.replicated.com/proxy/wallaroo/ghcr.io/wallaroolabs/mlflow-deploy:v2023.3.0-main-3466', 'last_update_time': datetime.datetime(2023, 7, 5, 16, 38, 22, 976104, tzinfo=tzutc())}

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

<table><tr><th>name</th> <td>xgb-rf-classification-pipeline</td></tr><tr><th>created</th> <td>2023-07-05 16:38:28.006387+00:00</td></tr><tr><th>last_updated</th> <td>2023-07-05 16:38:28.006387+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>f6aa5f94-ee96-44ab-a999-55f1a7762017</td></tr><tr><th>steps</th> <td></td></tr></table>

```python
pipeline.deploy(deployment_config=deployment_config)
pipeline.status()
```

    Waiting for deployment - this will take up to 90s ................ ok

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.24.4',
       'name': 'engine-5b5cdd555d-8cgdv',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'xgb-rf-classification-pipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'xgb-rf-classification',
          'version': '23a587f2-39cd-42a1-bfef-380a1c2cf2c2',
          'sha': '2aeb56c084a279770abdd26d14caba949159698c1a5d260d2aafe73090e6cb03',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.21.4',
       'name': 'engine-lb-584f54c899-hvrrp',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': [{'ip': '10.244.4.114',
       'name': 'engine-sidekick-xgb-rf-classification-128-6cbd8867cf-dbw58',
       'status': 'Running',
       'reason': None,
       'details': [],
       'statuses': '\n'}]}

## Inference

```python
data = pd.read_json('./data/test-xgboost-rf-classification-data.json')
data

dataframe = pd.DataFrame({"inputs": data[:2].values.tolist()})
dataframe

pipeline.infer(dataframe)
```

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
      <td>2023-07-05 16:38:45.492</td>
      <td>[5.1, 3.5, 1.4, 0.2]</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-07-05 16:38:45.492</td>
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

<table><tr><th>name</th> <td>xgb-rf-classification-pipeline</td></tr><tr><th>created</th> <td>2023-07-05 16:38:28.006387+00:00</td></tr><tr><th>last_updated</th> <td>2023-07-05 16:38:28.088610+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>445f803a-7452-415a-9046-47dd1b3935b4, f6aa5f94-ee96-44ab-a999-55f1a7762017</td></tr><tr><th>steps</th> <td>xgb-rf-classification</td></tr></table>

