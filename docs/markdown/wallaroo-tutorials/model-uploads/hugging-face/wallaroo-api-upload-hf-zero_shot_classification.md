The Wallaroo 101 tutorial can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/model_uploads/hugging-face-upload-tutorials).

## Wallaroo Model Upload via MLops API: Hugging Face Zero Shot Classification

The following tutorial demonstrates how to upload a Hugging Face Zero Shot model to a Wallaroo instance.

### Tutorial Goals

Demonstrate the following:

* Upload a Hugging Face Zero Shot Model to a Wallaroo instance.
* Create a pipeline and add the model as a pipeline step.
* Perform a sample inference.

### Prerequisites

* A Wallaroo version 2023.2.1 or above instance

### References

* [Wallaroo MLOps API Essentials Guide: Model Upload and Registrations](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-api-essential-guide/wallaroo-mlops-api-essentials-guide-model-uploads/)
* [Wallaroo API Connection Guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-connection-guide/)
* [DNS Integration Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-dns-guide/)

## Tutorial Steps

### Import Libraries

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

### Connect to Wallaroo

To perform the various Wallaroo MLOps API requests, we will use the Wallaroo SDK to generate the necessary tokens.  For details on other methods of requesting and using authentication tokens with the Wallaroo MLOps API, see the [Wallaroo API Connection Guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-connection-guide/).

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
wl = wallaroo.Client()
```

### Variables

The following variables will be set for the rest of the tutorial to set the following:

* Wallaroo Workspace
* Wallaroo Pipeline
* Wallaroo Model name and path
* Wallaroo Model Framework
* The DNS prefix and suffix for the Wallaroo instance.

To allow this tutorial to be run multiple times or by multiple users in the same Wallaroo instance, a random 4 character prefix will be added to the workspace, pipeline, and model.

Verify that the DNS prefix and suffix match the Wallaroo instance used for this tutorial.  See the [DNS Integration Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-dns-guide/) for more details.

```python
import string
import random

# make a random 4 character suffix to prevent overwriting other user's workspaces
suffix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))
workspace_name = f'hugging-face-zero-shot-api{suffix}'
pipeline_name = f'hugging-face-zero-shot'
model_name = f'zero-shot-classification'
model_file_name = "./models/model-auto-conversion_hugging-face_dummy-pipelines_zero-shot-classification-pipeline.zip"
framework = "hugging-face-zero-shot-classification"

wallarooPrefix = "YOUR PREFIX."
wallarooPrefix = "YOUR SUFFIX"

wallarooPrefix = "doc-test."
wallarooSuffix = "wallarooexample.ai"

APIURL=f"https://{wallarooPrefix}api.{wallarooSuffix}"
APIURL
```

    'https://doc-test.api.wallarooexample.ai'

### Create the Workspace

In a production environment, the Wallaroo workspace that contains the pipeline and models would be created and deployed.  We will quickly recreate those steps using the MLOps API.

Workspaces are created through the MLOps API with the `/v1/api/workspaces/create` command.  This requires the workspace name be provided, and that the workspace **not already exist** in the Wallaroo instance.

Reference: [MLOps API Create Workspace](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-api-essential-guide/wallaroo-mlops-api-essential-guide-workspaces/#create-workspace)

```python
# Retrieve the token
headers = wl.auth.auth_header()

# set Content-Type type
headers['Content-Type']='application/json'

# Create workspace
apiRequest = f"{APIURL}/v1/api/workspaces/create"

data = {
  "workspace_name": workspace_name
}

response = requests.post(apiRequest, json=data, headers=headers, verify=True).json()
display(response)
# Stored for future examples
workspaceId = response['workspace_id']
```

    {'workspace_id': 9}

### Upload the Model

* Endpoint:
  * `/v1/api/models/upload_and_convert`
* Headers:
  * **Content-Type**: `multipart/form-data`
* Parameters
  * **name** (*String* *Required*): The model name.
  * **visibility** (*String* *Required*): Either `public` or `private`.
  * **workspace_id** (*String* *Required*): The numerical ID of the workspace to upload the model to.
  * **conversion** (*String* *Required*):  The conversion parameters that include the following:
    * **framework** (*String* *Required*): The framework of the model being uploaded.  See the list of supported models for more details.
    * **python_version** (*String* *Required*):  The version of Python required for model.
    * **requirements**  (*String* *Required*):  Required libraries.  Can be `[]` if the requirements are default Wallaroo JupyterHub libraries.
    * **input_schema**  (*String* *Optional*): The input schema from the Apache Arrow `pyarrow.lib.Schema` format, encoded with `base64.b64encode`.  Only required for non-native runtime models.
    * **output_schema** (*String* *Optional*): The output schema from the Apache Arrow `pyarrow.lib.Schema` format, encoded with `base64.b64encode`.  Only required for non-native runtime models.

#### Set the Schemas

The input and output schemas will be defined according to the [Wallaroo Hugging Face schema requirements](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/wallaroo-sdk-model-upload-hugging-face/).  The inputs are then base64 encoded for attachment in the API request.

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
```

```python
encoded_output_schema = base64.b64encode(
                bytes(output_schema.serialize())
            ).decode("utf8")
```

### Build the Request

We will now build the request to include the required data.  We will be using the `workspaceId` returned when we created our workspace in a previous step, specifying the input and output schemas, and the framework.

```python
metadata = {
    "name": model_name,
    "visibility": "private",
    "workspace_id": workspaceId,
    "conversion": {
        "framework": framework,
        "python_version": "3.8",
        "requirements": []
    },
    "input_schema": encoded_input_schema,
    "output_schema": encoded_output_schema,
}
```

### Upload Model API Request

Now we will make our upload and convert request.  The model is is stored for the next set of steps.

```python
headers = wl.auth.auth_header()

files = {
    'metadata': (None, json.dumps(metadata), "application/json"),
    'file': (model_name, open(model_file_name,'rb'),'application/octet-stream')
}

response = requests.post(f'{APIURL}/v1/api/models/upload_and_convert', 
                         headers=headers, 
                         files=files)
print(response.json())
```

    {'insert_models': {'returning': [{'models': [{'id': 9}]}]}}

```python
# Get the model details

# Retrieve the token
headers = wl.auth.auth_header()

# set Content-Type type
headers['Content-Type']='application/json'

apiRequest = f"{APIURL}/v1/api/models/list_versions"

data = {
  "model_id": model_name,
  "models_pk_id" : modelId
}

status = None

while status != 'ready':
    response = requests.post(apiRequest, json=data, headers=headers, verify=True).json()
    # verify we have the right version
    display(model)
    model = next(model for model in response if model["id"] == modelId)
    display(model)
    status = model['status']
```

    {'sha': '3dcc14dd925489d4f0a3960e90a7ab5917ab685ce955beca8924aa7bb9a69398',
     'models_pk_id': 7,
     'model_version': '284a2e7c-679a-40cc-9121-2747b81a0228',
     'owner_id': '""',
     'model_id': 'zero-shot-classification',
     'id': 7,
     'file_name': 'zero-shot-classification',
     'image_path': 'proxy.replicated.com/proxy/wallaroo/ghcr.io/wallaroolabs/mlflow-deploy:v2023.3.0-main-3509',
     'status': 'ready'}

    {'sha': '3dcc14dd925489d4f0a3960e90a7ab5917ab685ce955beca8924aa7bb9a69398',
     'models_pk_id': 7,
     'model_version': '284a2e7c-679a-40cc-9121-2747b81a0228',
     'owner_id': '""',
     'model_id': 'zero-shot-classification',
     'id': 7,
     'file_name': 'zero-shot-classification',
     'image_path': 'proxy.replicated.com/proxy/wallaroo/ghcr.io/wallaroolabs/mlflow-deploy:v2023.3.0-main-3509',
     'status': 'ready'}

### Model Upload Complete

With that, the model upload is complete and can be deployed into a Wallaroo pipeline.
