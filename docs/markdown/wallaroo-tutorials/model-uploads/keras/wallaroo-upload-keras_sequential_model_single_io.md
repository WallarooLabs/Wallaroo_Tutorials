## Keras Sequential Model Single IO

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

import datetime

wl = wallaroo.Client(auth_type="sso", interactive=True)
```

    Please log into the following URL in a web browser:
    
    	https://keycloak.autoscale-uat-ee.wallaroo.dev/auth/realms/master/device?user_code=UJEE-FFVG
    
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

prefix = "keras-sequential-model-single-io"
```

```python
workspace = get_workspace(f"{prefix}-jch")
wl.set_current_workspace(workspace)
```

    {'name': 'keras-sequential-model-single-io-jch', 'id': 57, 'archived': False, 'created_by': '3cc9e92a-fa3c-4371-a7a7-487884df059e', 'created_at': '2023-06-20T13:54:12.895611+00:00', 'models': [], 'pipelines': []}

## Data & Model Creation

```python
input_schema = pa.schema([
    pa.field('input', pa.list_(pa.float64(), list_size=10))
])
output_schema = pa.schema([
    pa.field('output', pa.list_(pa.float64(), list_size=32))
])
```

## Upload model

```python
model_upload_start = datetime.datetime.now()

model = wl.upload_model(f"{prefix}", 
                        'models/model-auto-conversion_keras_single_io_keras_sequential_model.h5', 
                        framework=Framework.KERAS, 
                        input_schema=input_schema, 
                        output_schema=output_schema)
model_upload_end = datetime.datetime.now()
model
```

    Waiting for model conversion... It may take up to 10.0min.
    Model is Pending conversion........Converting...................Ready.

    {'name': 'keras-sequential-model-single-io', 'version': 'b4b4b490-fe21-4f3c-8464-d24c9a2c8049', 'file_name': 'model-auto-conversion_keras_single_io_keras_sequential_model.h5', 'image_path': 'proxy.replicated.com/proxy/wallaroo/ghcr.io/wallaroolabs/mlflow-deploy:v2023.3.0-main-3367', 'last_update_time': datetime.datetime(2023, 6, 20, 14, 8, 14, 884034, tzinfo=tzutc())}

```python
display(model_upload_end - model_upload_start)
```

    datetime.timedelta(seconds=135, microseconds=652975)

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

<table><tr><th>name</th> <td>keras-sequential-model-single-io-pipeline</td></tr><tr><th>created</th> <td>2023-06-20 13:56:53.710879+00:00</td></tr><tr><th>last_updated</th> <td>2023-06-20 14:09:33.632124+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>e5c2a9d0-9420-4fba-be63-581b37984c40, c425ae54-fe4a-4ec5-9b57-93ab4545d58b, 38ec1b19-372d-4ca1-87bc-fcc218d93701, 8d7093fa-6d25-4194-bfa0-4e3e62f79494, 648c6b38-2452-4177-b073-6c040a9c897f, 900ce28e-a2f6-4397-88ad-05a22ebee388</td></tr><tr><th>steps</th> <td>keras-sequential-model-single-io</td></tr></table>

```python
pipeline.deploy(deployment_config=deployment_config)
```

    Waiting for deployment - this will take up to 90s .......................... ok

<table><tr><th>name</th> <td>keras-sequential-model-single-io-pipeline</td></tr><tr><th>created</th> <td>2023-06-20 13:56:53.710879+00:00</td></tr><tr><th>last_updated</th> <td>2023-06-20 14:09:35.178169+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>77cd8df4-1794-4485-b9f4-9e3926db0422, e5c2a9d0-9420-4fba-be63-581b37984c40, c425ae54-fe4a-4ec5-9b57-93ab4545d58b, 38ec1b19-372d-4ca1-87bc-fcc218d93701, 8d7093fa-6d25-4194-bfa0-4e3e62f79494, 648c6b38-2452-4177-b073-6c040a9c897f, 900ce28e-a2f6-4397-88ad-05a22ebee388</td></tr><tr><th>steps</th> <td>keras-sequential-model-single-io</td></tr></table>

## Inference

```python
input_data = np.random.rand(10, 10)
mock_dataframe = pd.DataFrame({
    "input": input_data.tolist()
})
mock_dataframe
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>input</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[0.8948402037930523, 0.8236923767258912, 0.669...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[0.2889161093430169, 0.02527054036324672, 0.95...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[0.8060020370068443, 0.8821186682883927, 0.559...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[0.8883645451955529, 0.19204670320917383, 0.09...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[0.05892172677764429, 0.8634608360310956, 0.44...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>[0.4010007936827573, 0.7701734803672011, 0.033...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>[0.6251592472839257, 0.8087976037612621, 0.866...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>[0.14250389929667917, 0.18866998852295247, 0.8...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>[0.2247017624310551, 0.7870476471329036, 0.932...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>[0.07932631109734134, 0.9855873946298688, 0.99...</td>
    </tr>
  </tbody>
</table>

```python
pipeline.infer(mock_dataframe)
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
      <td>2023-06-20 14:17:18.475</td>
      <td>[0.8948402038, 0.8236923767, 0.6692281767, 0.6...</td>
      <td>[0.022978410124778748, 0.023622576147317886, 0...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-06-20 14:17:18.475</td>
      <td>[0.2889161093, 0.0252705404, 0.9563431751, 0.3...</td>
      <td>[0.018787803128361702, 0.03189399093389511, 0....</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-06-20 14:17:18.475</td>
      <td>[0.806002037, 0.8821186683, 0.5597011509, 0.29...</td>
      <td>[0.028999775648117065, 0.020044947043061256, 0...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-06-20 14:17:18.475</td>
      <td>[0.8883645452, 0.1920467032, 0.0982324665, 0.6...</td>
      <td>[0.024776356294751167, 0.02724037691950798, 0....</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-06-20 14:17:18.475</td>
      <td>[0.0589217268, 0.863460836, 0.4416552874, 0.70...</td>
      <td>[0.02560083381831646, 0.025948569178581238, 0....</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2023-06-20 14:17:18.475</td>
      <td>[0.4010007937, 0.7701734804, 0.0330423197, 0.7...</td>
      <td>[0.018146220594644547, 0.031117431819438934, 0...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2023-06-20 14:17:18.475</td>
      <td>[0.6251592473, 0.8087976038, 0.8664463129, 0.5...</td>
      <td>[0.02749345824122429, 0.03163997083902359, 0.0...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2023-06-20 14:17:18.475</td>
      <td>[0.1425038993, 0.1886699885, 0.8979911291, 0.2...</td>
      <td>[0.030713597312569618, 0.030822383239865303, 0...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2023-06-20 14:17:18.475</td>
      <td>[0.2247017624, 0.7870476471, 0.932017211, 0.66...</td>
      <td>[0.033133771270513535, 0.02262025512754917, 0....</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2023-06-20 14:17:18.475</td>
      <td>[0.0793263111, 0.9855873946, 0.9910993908, 0.0...</td>
      <td>[0.036864764988422394, 0.021373195573687553, 0...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

```python
pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s ...................................... ok

<table><tr><th>name</th> <td>keras-sequential-model-single-io-pipeline</td></tr><tr><th>created</th> <td>2023-06-20 13:56:53.710879+00:00</td></tr><tr><th>last_updated</th> <td>2023-06-20 14:09:35.178169+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>77cd8df4-1794-4485-b9f4-9e3926db0422, e5c2a9d0-9420-4fba-be63-581b37984c40, c425ae54-fe4a-4ec5-9b57-93ab4545d58b, 38ec1b19-372d-4ca1-87bc-fcc218d93701, 8d7093fa-6d25-4194-bfa0-4e3e62f79494, 648c6b38-2452-4177-b073-6c040a9c897f, 900ce28e-a2f6-4397-88ad-05a22ebee388</td></tr><tr><th>steps</th> <td>keras-sequential-model-single-io</td></tr></table>

```python

```
