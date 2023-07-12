# Wallaroo `upload_model` API Example

In order to call the `upload_model` endpoint, you need to have the following things ready:

- API Token
- Input/Output Arrow Schemas, encoded in base64
- `upload_model` usual arguments (i.e. model name, requirements, framework etc.)

### `upload_model` Example Call

```bash
curl --progress-bar -X POST \
  -H "Content-Type: multipart/form-data" \
  -H "Authorization: Bearer <token here>" \
  -F 'metadata={ "name": "gpt4all", "visibility": "private", "workspace_id": 6, "conversion": {"framework": "hugging-face-text-generation", "python_version": "3.8", "requirements": []}, "input_schema": <base64 input schema here>, "output_schema": <base64 output schema here>};type=application/json' \
  -F "file=@model-auto-conversion_hugging-face_LLM_gpt4all-groovy-hf-pipeline.zip;type=application/octet-stream" \
  https://autoscale-uat-ee.wallaroo.dev/v1/api/models/upload_and_convert | cat
```

### Generating Input/Output Encoded Schemas

```python
base64.b64encode(
                bytes(input_schema.serialize())
            ).decode("utf8")

base64.b64encode(
                bytes(output_schema.serialize())
            ).decode("utf8")
```

## Python Example

In order to show a concrete example of a request made in Python, we will show a HuggingFace Text Classification model uploaded with an API call.

### Imports

```python
import json
import os
import requests
import base64

import wallaroo
from wallaroo.pipeline   import Pipeline
from wallaroo.deployment_config import DeploymentConfigBuilder
from wallaroo.framework import Framework

import pyarrow as pa
import numpy as np
import pandas as pd
```

### Get framework

```python
# wl = wallaroo.Client(auth_type="sso", interactive=True)

wallarooPrefix = ""
wallarooSuffix = "autoscale-uat-ee.wallaroo.dev"

wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}api.{wallarooSuffix}", 
                    auth_endpoint=f"https://{wallarooPrefix}keycloak.{wallarooSuffix}", 
                    auth_type="sso")
```

    Please log into the following URL in a web browser:
    
    	https://keycloak.autoscale-uat-ee.wallaroo.dev/auth/realms/master/device?user_code=FNAR-NMFT
    
    Login successful!

```python
[e.value for e in Framework]
```

    ['onnx',
     'tensorflow',
     'python',
     'keras',
     'sklearn',
     'pytorch',
     'xgboost',
     'hugging-face-feature-extraction',
     'hugging-face-image-classification',
     'hugging-face-image-segmentation',
     'hugging-face-image-to-text',
     'hugging-face-object-detection',
     'hugging-face-question-answering',
     'hugging-face-stable-diffusion-text-2-img',
     'hugging-face-summarization',
     'hugging-face-text-classification',
     'hugging-face-translation',
     'hugging-face-zero-shot-classification',
     'hugging-face-zero-shot-image-classification',
     'hugging-face-zero-shot-object-detection',
     'hugging-face-sentiment-analysis',
     'hugging-face-text-generation',
     'custom']

### Configure PyArrow Schemas

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

```python
encoded_input_schema = base64.b64encode(
                bytes(input_schema.serialize())
            ).decode("utf8")
encoded_input_schema
```

    '/////0gBAAAQAAAAAAAKAAwABgAFAAgACgAAAAABBAAMAAAACAAIAAAABAAIAAAABAAAAAQAAADoAAAAbAAAADQAAAAEAAAAOP///wAAAQYQAAAAHAAAAAQAAAAAAAAACwAAAG11bHRpX2xhYmVsACz///9k////AAABBRAAAAAkAAAABAAAAAAAAAATAAAAaHlwb3RoZXNpc190ZW1wbGF0ZQBg////mP///wAAARAUAAAALAAAAAQAAAABAAAAKAAAABAAAABjYW5kaWRhdGVfbGFiZWxzAAAGAAgABAAGAAAAAgAAANj///8AAAEFEAAAABgAAAAEAAAAAAAAAAQAAABpdGVtAAAAAMj///8QABQACAAGAAcADAAAABAAEAAAAAAAAQUQAAAAHAAAAAQAAAAAAAAABgAAAGlucHV0cwAABAAEAAQAAAAAAAAA'

```python
encoded_output_schema = base64.b64encode(
                bytes(output_schema.serialize())
            ).decode("utf8")
encoded_output_schema
```

    '/////0ABAAAQAAAAAAAKAAwABgAFAAgACgAAAAABBAAMAAAACAAIAAAABAAIAAAABAAAAAMAAADcAAAAYAAAAAQAAABA////AAABEBQAAAAcAAAABAAAAAEAAAAYAAAABgAAAGxhYmVscwAApv///wIAAABw////AAABBRAAAAAYAAAABAAAAAAAAAAEAAAAaXRlbQAAAABc////mP///wAAARAUAAAAJAAAAAQAAAABAAAAIAAAAAYAAABzY29yZXMAAAAABgAIAAQABgAAAAIAAADQ////AAABAxAAAAAcAAAABAAAAAAAAAAEAAAAaXRlbQAABgAIAAYABgAAAAAAAgAQABQACAAGAAcADAAAABAAEAAAAAAAAQUQAAAAIAAAAAQAAAAAAAAACAAAAHNlcXVlbmNlAAAAAAQABAAEAAAAAAAAAA=='

### Build the request

```python
# API_TOKEN = "token"
```

```python
model_name = "zero-shot-classification-test"
workspace_id = 29
framework = "hugging-face-zero-shot-classification"
model_path = "./models/model-auto-conversion_hugging-face_dummy-pipelines_zero-shot-classification-pipeline.zip"
```

```python
metadata = {
    "name": model_name,
    "visibility": "private",
    "workspace_id": workspace_id,
    "conversion": {
        "framework": framework,
        "python_version": "3.8",
        "requirements": []
    },
    "input_schema": encoded_input_schema,
    "output_schema": encoded_output_schema,
}
```

```python
headers = wl.auth.auth_header()

files = {
    'metadata': (None, json.dumps(metadata), "application/json"),
    'file': (model_name, open(model_path,'rb'),'application/octet-stream')
}

response = requests.post('https://autoscale-uat-ee.wallaroo.dev/v1/api/models/upload_and_convert', 
                         headers=headers, 
                         files=files)
```

```python
print(response.json())
```

    {'insert_models': {'returning': [{'models': [{'id': 208}]}]}}

