## PyTortch Regression

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

```

    Please log into the following URL in a web browser:
    
    	https://keycloak.autoscale-uat-ee.wallaroo.dev/auth/realms/master/device?user_code=PYXH-ZCNX
    
    Login successful!

```python

wl = wallaroo.Client()
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

prefix = "pytorch-multi-input"
```

```python
workspace = get_workspace(f"{prefix}-jch")
wl.set_current_workspace(workspace)
```

    {'name': 'pytorch-single-input-jch', 'id': 47, 'archived': False, 'created_by': '0892876b-8b50-4541-bf29-0e570e590c01', 'created_at': '2023-06-23T02:07:54.59147+00:00', 'models': [], 'pipelines': []}

## Data & Model Creation

```python
mock_inference_data = [np.random.rand(10, 10), np.random.rand(10, 5)]
mock_dataframe = pd.DataFrame(
    {
        "input_1": mock_inference_data[0].tolist(),
        "input_2": mock_inference_data[1].tolist(),
    }
)
mock_json = mock_dataframe.to_json('data/pytorch-multi-io-data.json', orient="records")
```

```python
input_schema = pa.schema([
    pa.field('input_1', pa.list_(pa.float64(), list_size=10)),
    pa.field('input_2', pa.list_(pa.float64(), list_size=5))
])
output_schema = pa.schema([
    pa.field('output_1', pa.list_(pa.float64(), list_size=3)),
    pa.field('output_2', pa.list_(pa.float64(), list_size=2))
])
```

## Upload model

```python
model = wl.upload_model('pt-multi-io-model', 
                        "./models/model-auto-conversion_pytorch_multi_io_model.pt", 
                        framework=Framework.PYTORCH, 
                        input_schema=input_schema, 
                        output_schema=output_schema
                       )
model
```

    Waiting for model conversion... It may take up to 10.0min.
    Model is Pending conversion..Converting...........Ready.

    {'name': 'pt-single-io-model', 'version': '8f91dee1-79e0-449b-9a59-0e93ba4a1ba9', 'file_name': 'model-auto-conversion_pytorch_single_io_model.pt', 'image_path': 'proxy.replicated.com/proxy/wallaroo/ghcr.io/wallaroolabs/mlflow-deploy:v2023.3.0-main-3397', 'last_update_time': datetime.datetime(2023, 6, 23, 2, 8, 56, 669565, tzinfo=tzutc())}

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

    Waiting for deployment - this will take up to 90s ................... ok

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.6.32',
       'name': 'engine-699fc4898f-8rv9t',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'pytorch-single-input-pipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'pt-single-io-model',
          'version': '8f91dee1-79e0-449b-9a59-0e93ba4a1ba9',
          'sha': '23bdbafc51c3df7ac84e5f8b2833c592d7da2b27715a7da3e45bf732ea85b8bb',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.6.31',
       'name': 'engine-lb-584f54c899-jrkg9',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': [{'ip': '10.244.4.202',
       'name': 'engine-sidekick-pt-single-io-model-150-66f8c754bf-x69tx',
       'status': 'Running',
       'reason': None,
       'details': [],
       'statuses': '\n'}]}

## Inference

```python
pipeline.infer_from_file('./data/pytorch-multi-io-data.json')
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.input</th>
      <th>out.output</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-06-23 02:09:44.784</td>
      <td>[0.1810392682, 0.257671871, 0.9975622994, 0.13...</td>
      <td>[0.150135800242424]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-06-23 02:09:44.784</td>
      <td>[0.6790846298, 0.0180313244, 0.5641078074, 0.3...</td>
      <td>[-0.08427916467189789]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-06-23 02:09:44.784</td>
      <td>[0.5731166617, 0.5533275412, 0.4172515078, 0.3...</td>
      <td>[-0.01238897442817688]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-06-23 02:09:44.784</td>
      <td>[0.7686341548, 0.9895859489, 0.1525565068, 0.0...</td>
      <td>[-0.0985543504357338]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-06-23 02:09:44.784</td>
      <td>[0.916929814, 0.0148056624, 0.8978319544, 0.64...</td>
      <td>[-0.09987331926822662]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2023-06-23 02:09:44.784</td>
      <td>[0.2164508004, 0.3161185598, 0.4462939944, 0.2...</td>
      <td>[-0.07775364816188812]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2023-06-23 02:09:44.784</td>
      <td>[0.3901195238, 0.4068669654, 0.370721485, 0.99...</td>
      <td>[-0.15208657085895538]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2023-06-23 02:09:44.784</td>
      <td>[0.4070531149, 0.1575033787, 0.9212983823, 0.7...</td>
      <td>[-0.026245146989822388]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2023-06-23 02:09:44.784</td>
      <td>[0.775045768, 0.4525766175, 0.7998572455, 0.85...</td>
      <td>[-0.12369337677955627]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2023-06-23 02:09:44.784</td>
      <td>[0.7134285141, 0.4573823559, 0.3010524343, 0.4...</td>
      <td>[-0.03293849527835846]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

```python
pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s ...................................... ok

<table><tr><th>name</th> <td>pytorch-single-input-pipeline</td></tr><tr><th>created</th> <td>2023-06-23 02:09:00.149322+00:00</td></tr><tr><th>last_updated</th> <td>2023-06-23 02:09:00.181270+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>474221ef-b47b-41ae-9191-d156e0094ed5, 68164858-7608-4a79-a16b-d5ca3b7bfdfc</td></tr><tr><th>steps</th> <td>pt-single-io-model</td></tr></table>

```python

```
