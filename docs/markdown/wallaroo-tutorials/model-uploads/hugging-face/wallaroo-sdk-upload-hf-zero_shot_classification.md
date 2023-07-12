# Zero Shot Classification (text) Pipeline Example

## Imports

```python
import json
import os

import wallaroo
from wallaroo.pipeline   import Pipeline
from wallaroo.deployment_config import DeploymentConfigBuilder
from wallaroo.framework import Framework

import pyarrow as pa
import numpy as np
import pandas as pd
```

```python
wl = wallaroo.Client()
```

    Please log into the following URL in a web browser:
    
    	https://keycloak.autoscale-uat-ee.wallaroo.dev/auth/realms/master/device?user_code=VTLN-JKLY
    
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

prefix = "hf-zero-shot-classification"
```

## Setup up workspace

```python
workspace = get_workspace(f"{prefix}-jch")
wl.set_current_workspace(workspace)
```

    {'name': 'hf-zero-shot-classification-jch', 'id': 42, 'archived': False, 'created_by': '0892876b-8b50-4541-bf29-0e570e590c01', 'created_at': '2023-06-22T13:42:34.416552+00:00', 'models': [{'name': 'hf-zero-shot-classification-model', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 6, 22, 13, 44, 51, 233603, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 6, 22, 13, 44, 51, 233603, tzinfo=tzutc())}], 'pipelines': [{'name': 'hf-zero-shot-classification-pipeline', 'create_time': datetime.datetime(2023, 6, 22, 13, 46, 11, 623402, tzinfo=tzutc()), 'definition': '[]'}]}

### Configure PyArrow Schema

You can find more info on the available inputs under the [official source code](https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/pipelines/zero_shot_classification.py#L172) from `🤗 Hugging Face`.

```python
input_schema = pa.schema([
    pa.field('inputs', pa.string()), # required
    pa.field('candidate_labels', pa.list_(pa.string(), list_size=2)), # required
    pa.field('hypothesis_template', pa.string()), # optional
    pa.field('multi_label', pa.bool_()), # optional
])

output_schema = pa.schema([
    pa.field('sequence', pa.string()),
    pa.field('scores', pa.list_(pa.float64(), list_size=2)), # same as number of candidate labels, list_size can be skipped by may result in slightly worse performance
    pa.field('labels', pa.list_(pa.string(), list_size=2)), # same as number of candidate labels, list_size can be skipped by may result in slightly worse performance
])
```

### Upload Model

```python
model = wl.upload_model(f"{prefix}-model",
                        './models/model-auto-conversion_hugging-face_dummy-pipelines_zero-shot-classification-pipeline.zip', 
                        framework=Framework.HUGGING_FACE_ZERO_SHOT_CLASSIFICATION, 
                        input_schema=input_schema,
                        output_schema=output_schema)
model
```

    Waiting for model conversion... It may take up to 10.0min.
    Model is Pending conversion..Converting..............Ready.

    {'name': 'hf-zero-shot-classification-model', 'version': '6b930398-5887-4727-a26a-7fb29035c899', 'file_name': 'model-auto-conversion_hugging-face_dummy-pipelines_zero-shot-classification-pipeline.zip', 'image_path': 'proxy.replicated.com/proxy/wallaroo/ghcr.io/wallaroolabs/mlflow-deploy:v2023.3.0-main-3397', 'last_update_time': datetime.datetime(2023, 6, 22, 16, 56, 6, 951569, tzinfo=tzutc())}

## Deploy Pipeline

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

     ok

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.4.133',
       'name': 'engine-5c9889d5b4-dvd4d',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'hf-zero-shot-classification-pipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'hf-zero-shot-classification-model',
          'version': 'acb7fd8a-57aa-41cd-b49e-a3e39c111630',
          'sha': '3dcc14dd925489d4f0a3960e90a7ab5917ab685ce955beca8924aa7bb9a69398',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.4.132',
       'name': 'engine-lb-584f54c899-7sbdq',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': [{'ip': '10.244.4.131',
       'name': 'engine-sidekick-hf-zero-shot-classification-model-75-9b5dc6b7vx',
       'status': 'Running',
       'reason': None,
       'details': [],
       'statuses': '\n'}]}

## Run inference

```python
input_data = {
        "inputs": ["this is a test", "this is another test"], # required
        "candidate_labels": [["english", "german"], ["english", "german"]], # optional: using the defaults, similar to not passing this parameter
        "hypothesis_template": ["This example is {}.", "This example is {}."], # optional: using the defaults, similar to not passing this parameter
        "multi_label": [False, False], # optional: using the defaults, similar to not passing this parameter
}
dataframe = pd.DataFrame(input_data)
dataframe
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>inputs</th>
      <th>candidate_labels</th>
      <th>hypothesis_template</th>
      <th>multi_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>this is a test</td>
      <td>[english, german]</td>
      <td>This example is {}.</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>this is another test</td>
      <td>[english, german]</td>
      <td>This example is {}.</td>
      <td>False</td>
    </tr>
  </tbody>
</table>

```python
%time
pipeline.infer(dataframe)
```

    CPU times: user 2 µs, sys: 0 ns, total: 2 µs
    Wall time: 5.25 µs

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.candidate_labels</th>
      <th>in.hypothesis_template</th>
      <th>in.inputs</th>
      <th>in.multi_label</th>
      <th>out.labels</th>
      <th>out.scores</th>
      <th>out.sequence</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-06-22 16:56:11.396</td>
      <td>[english, german]</td>
      <td>This example is {}.</td>
      <td>this is a test</td>
      <td>False</td>
      <td>[english, german]</td>
      <td>[0.504054605960846, 0.49594545364379883]</td>
      <td>this is a test</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-06-22 16:56:11.396</td>
      <td>[english, german]</td>
      <td>This example is {}.</td>
      <td>this is another test</td>
      <td>False</td>
      <td>[english, german]</td>
      <td>[0.5037839412689209, 0.4962160289287567]</td>
      <td>this is another test</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

## Undeploy Pipelines

```python
pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s ....................................... ok

<table><tr><th>name</th> <td>hf-zero-shot-classification-pipeline</td></tr><tr><th>created</th> <td>2023-06-22 13:46:11.623402+00:00</td></tr><tr><th>last_updated</th> <td>2023-06-22 16:56:11.137324+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>55c7a8d9-cb6b-4018-b165-9728a84b6a62, 9951f37d-3620-4687-823a-3c3d0013d9c3, 2fee5e40-d356-432b-ba0a-ee79a5ed81f9, 6de8b8cc-a08d-4436-a169-2721d4137027</td></tr><tr><th>steps</th> <td>hf-zero-shot-classification-model</td></tr></table>

