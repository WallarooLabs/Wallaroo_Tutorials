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
workspace = get_workspace("sklearn-logistic-regression-jch")
wl.set_current_workspace(workspace)
```

    {'name': 'sklearn-logistic-regression-jch', 'id': 89, 'archived': False, 'created_by': 'd9a72bd9-2a1c-44dd-989f-3c7c15130885', 'created_at': '2023-07-05T15:36:30.535039+00:00', 'models': [], 'pipelines': []}

## Data & Model Creation

```python
input_schema = pa.schema([
    pa.field('inputs', pa.list_(pa.float64(), list_size=4))
])

output_schema = pa.schema([
    pa.field('predictions', pa.int32()),
    pa.field('probabilities', pa.list_(pa.float64(), list_size=3))
])
```

## Upload model

```python
model = wl.upload_model('sklearn-logistic-regression', 'models/logreg.pkl', framework=Framework.SKLEARN, input_schema=input_schema, output_schema=output_schema)
model
```

    Waiting for model conversion... It may take up to 10.0min.
    Model is Pending conversion..Converting..Pending conversion.Converting.......Ready.

    {'name': 'sklearn-logistic-regression', 'version': '092956e3-0e56-4349-956a-231ec9a0d83d', 'file_name': 'logreg.pkl', 'image_path': 'proxy.replicated.com/proxy/wallaroo/ghcr.io/wallaroolabs/mlflow-deploy:v2023.3.0-main-3466', 'last_update_time': datetime.datetime(2023, 7, 5, 15, 37, 28, 137902, tzinfo=tzutc())}

## Configure model and pipeline

```python
deployment_config = DeploymentConfigBuilder() \
    .cpus(0.25).memory('1Gi') \
    .build()
```

```python
pipeline_name = f"sklearn-logistic-regression-pipeline"
pipeline = wl.build_pipeline(pipeline_name)
pipeline.add_model_step(model)

pipeline.deploy(deployment_config=deployment_config)
pipeline.status()
```

    Waiting for deployment - this will take up to 90s ......... ok

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.12.7',
       'name': 'engine-dcf786657-wmssz',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'sklearn-logistic-regression-pipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'sklearn-logistic-regression',
          'version': '092956e3-0e56-4349-956a-231ec9a0d83d',
          'sha': '9302df6cc64a2c0d12daa257657f07f9db0bb2072bb3fb92396500b21358e0b9',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.15.7',
       'name': 'engine-lb-584f54c899-wqdbh',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': [{'ip': '10.244.4.75',
       'name': 'engine-sidekick-sklearn-logistic-regression-120-d75d6566-fq7qv',
       'status': 'Running',
       'reason': None,
       'details': [],
       'statuses': '\n'}]}

## Inference

```python
data = pd.read_json('data/test_logreg_data.json')
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
      <th>out.probabilities</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-07-05 15:38:02.322</td>
      <td>[5.1, 3.5, 1.4, 0.2]</td>
      <td>0</td>
      <td>[0.9815821465852236, 0.018417838912958125, 1.4...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-07-05 15:38:02.322</td>
      <td>[4.9, 3.0, 1.4, 0.2]</td>
      <td>0</td>
      <td>[0.9713374799347873, 0.028662489870060148, 3.0...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

```python
pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s ..................................... ok

<table><tr><th>name</th> <td>sklearn-logistic-regression-pipeline</td></tr><tr><th>created</th> <td>2023-07-05 15:37:31.123470+00:00</td></tr><tr><th>last_updated</th> <td>2023-07-05 15:37:31.152213+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>4d8a2c6a-53ca-4339-956b-8ec2d885b6c1, 498be539-3a34-40d0-8919-72f1129c328f</td></tr><tr><th>steps</th> <td>sklearn-logistic-regression</td></tr></table>

