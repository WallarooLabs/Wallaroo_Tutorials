## Scikit-Learn Linear Regression

The following example will:

* Set the input and output schemas.
* Upload a SKLearn Linear Regression model to Wallaroo.
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

from sklearn.datasets import load_diabetes
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
workspace = get_workspace("sklearn-linear-regression-jch")
wl.set_current_workspace(workspace)
```

    {'name': 'sklearn-linear-regression-jch', 'id': 33, 'archived': False, 'created_by': 'd9a72bd9-2a1c-44dd-989f-3c7c15130885', 'created_at': '2023-06-27T21:52:58.179105+00:00', 'models': [{'name': 'sklearn-linear-regression', 'versions': 2, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 6, 27, 22, 16, 58, 273914, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 6, 27, 21, 52, 58, 380627, tzinfo=tzutc())}], 'pipelines': [{'name': 'sklearn-linear-regression-pipeline', 'create_time': datetime.datetime(2023, 6, 27, 22, 17, 53, 569729, tzinfo=tzutc()), 'definition': '[]'}]}

## Data & Model Creation

```python
input_schema = pa.schema([
    pa.field('inputs', pa.list_(pa.float64(), list_size=10))
])

output_schema = pa.schema([
    pa.field('predictions', pa.float64())
])
```

## Upload model

```python
model = wl.upload_model('sklearn-linear-regression', 'models/model-auto-conversion_sklearn_linreg_diabetes.pkl', framework=Framework.SKLEARN, input_schema=input_schema, output_schema=output_schema)
model
```

    Waiting for model conversion... It may take up to 10.0min.
    Model is Pending conversion..Converting..Pending conversion.Converting.......Ready.

    {'name': 'sklearn-linear-regression', 'version': '260810f1-ce30-4183-aa14-04f8595764b6', 'file_name': 'model-auto-conversion_sklearn_linreg_diabetes.pkl', 'image_path': 'proxy.replicated.com/proxy/wallaroo/ghcr.io/wallaroolabs/mlflow-deploy:v2023.3.0-main-3466', 'last_update_time': datetime.datetime(2023, 7, 5, 15, 43, 14, 693014, tzinfo=tzutc())}

## Configure model and pipeline

```python
deployment_config = DeploymentConfigBuilder() \
    .cpus(0.25).memory('1Gi') \
    .build()
```

```python
pipeline_name = f"sklearn-linear-regression-pipeline"
pipeline = wl.build_pipeline(pipeline_name)
pipeline.add_model_step(model)

pipeline.deploy(deployment_config=deployment_config)
pipeline.status()
```

    Waiting for deployment - this will take up to 90s ............. ok

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.12.9',
       'name': 'engine-58cc7f9b59-pqfdg',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'sklearn-linear-regression-pipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'sklearn-linear-regression',
          'version': '260810f1-ce30-4183-aa14-04f8595764b6',
          'sha': '6a9085e2d65bf0379934651d2272d3c6c4e020e36030933d85df3a8d15135a45',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.15.9',
       'name': 'engine-lb-584f54c899-xb2dw',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': [{'ip': '10.244.4.77',
       'name': 'engine-sidekick-sklearn-linear-regression-122-678d8dc46b-6p2fw',
       'status': 'Running',
       'reason': None,
       'details': [],
       'statuses': '\n'}]}

## Inference

```python
data = pd.read_json('data/test_linear_regression_data.json')
display(data)

# move the column values to a single array input
dataframe = pd.DataFrame({"inputs": data[:2].values.tolist()})
display(dataframe)
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
      <td>2023-07-05 15:43:33.065</td>
      <td>[0.0380759064, 0.0506801187, 0.0616962065, 0.0...</td>
      <td>206.116677</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-07-05 15:43:33.065</td>
      <td>[-0.0018820165, -0.0446416365, -0.0514740612, ...</td>
      <td>68.071033</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

```python
pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s ..................................... ok

<table><tr><th>name</th> <td>sklearn-linear-regression-pipeline</td></tr><tr><th>created</th> <td>2023-06-27 22:17:53.569729+00:00</td></tr><tr><th>last_updated</th> <td>2023-07-05 15:43:18.945313+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>ad523ea5-2a96-4a15-8c4d-f7b565fc65a2, cb0afa37-5ef8-4f76-a285-cb31cc8708ef, a9daa415-1359-4f30-a4b0-9f4760d2d714, ab4bd989-fcb9-4adb-aad8-ce1c27057363, 31dc5a29-d357-46e3-8a46-44d75a64efc9, 87286200-1b04-4e41-aa76-cee4b6fee308</td></tr><tr><th>steps</th> <td>sklearn-linear-regression</td></tr></table>

