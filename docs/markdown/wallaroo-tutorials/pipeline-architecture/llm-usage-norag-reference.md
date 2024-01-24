This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/blob/main/wallaroo-model-cookbooks/llamav2).

This tutorial demonstrates deploying the [LLamav2 Large Language Model (LLM)](https://ai.meta.com/llama/) model to Wallaroo and performing inferences through it.

This demonstrations takes the Llama V2 model and wraps it in a [Wallaroo BYOP (Bring Your Own Predict)](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/wallaroo-sdk-model-arbitrary-python/) framework.  This allows ML models outside of the standard [Wallaroo Native and Wallaroo Containerized Runtimes](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/) to be deployed in a Wallaroo pipeline.

## Prerequisites

* A Wallaroo Ops instance Version 2023.4 and above.
* A nodepool with at least 1 GPU.  See [Create GPU Nodepools for Kubernetes Clusters](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-install-guides/wallaroo-install-configurations/wallaroo-gpu-nodepools/) for instructions on setting up a nodepool with GPU virtual machines.
* The BYOP version of the LLamav2 model.  The total size of this model is 20 GB.  If needed, review the [Manage Minio Storage for Models Storage](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-minio-storage/) guide for instructions on increasing the model storage capacity in your Wallaroo Ops instance.  The model is available through the following link.  Store this model in the `./models` directory.
  * [https://storage.googleapis.com/wallaroo-public-data/llm-models/model-auto-conversion_BYOP_llama_byop_llamav2_new2.zip]https://storage.googleapis.com/wallaroo-public-data/llm-models/model-auto-conversion_BYOP_llama_byop_llamav2_new2.zip

## Tutorial Steps

This tutorial follows this process:

* Connect to the Wallaroo Ops instance.
* Create a workspace.
* Upload the model.
* Create a pipeline and deploy it.
* Perform a sample inference.

### Import Libraries

The first step will be to import our libraries.

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

from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain.vectorstores import FAISS

from transformers import pipeline
```

### Connect to the Wallaroo Instance

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
wl = wallaroo.Client(auth_type="sso", interactive=True)
```

### Create Workspace

We will create a workspace to manage our pipeline and models.  The following variables will set the name of our sample workspace then set it as the current workspace.

Workspace names must be unique.  The following helper function will either create a new workspace, or retrieve an existing one with the same name.  Verify that a pre-existing workspace has been shared with the targeted user.

Set the variables `workspace_name` to ensure a unique workspace name if required.

The workspace will then be set as the Current Workspace.  Model uploads and pipeline creation through the SDK are set in the current workspace.

* References
  * [Wallaroo SDK Essentials Guide: Workspace Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-workspace/)

```python
def getWorkspace(wl, ws_name):
    workspace = None
    for ws in wl.list_workspaces():
        if ws.name() == ws_name:
            workspace= ws
    if(workspace == None):
        workspace = wl.create_workspace(ws_name)
    return workspace
```

```python
workspace_name = "llama-models"
ws = getWorkspace(wl, workspace_name)
wl.set_current_workspace(ws)
```

    {'name': 'llama-models', 'id': 8, 'archived': False, 'created_by': 'e3c9f02f-988a-4097-8cc0-370fd3d629fa', 'created_at': '2024-01-16T02:45:53.745255+00:00', 'models': [{'name': 'llama-chat', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2024, 1, 16, 2, 56, 32, 639430, tzinfo=tzutc()), 'created_at': datetime.datetime(2024, 1, 16, 2, 56, 32, 639430, tzinfo=tzutc())}, {'name': 'llama-rag', 'versions': 2, 'owner_id': '""', 'last_update_time': datetime.datetime(2024, 1, 17, 0, 25, 7, 661947, tzinfo=tzutc()), 'created_at': datetime.datetime(2024, 1, 16, 13, 30, 8, 549780, tzinfo=tzutc())}], 'pipelines': [{'name': 'llamav2-pipe', 'create_time': datetime.datetime(2024, 1, 16, 3, 5, 59, 758313, tzinfo=tzutc()), 'definition': '[]'}, {'name': 'rag-llamav2-pipe', 'create_time': datetime.datetime(2024, 1, 16, 13, 42, 39, 879093, tzinfo=tzutc()), 'definition': '[]'}, {'name': 'llamav2-chat-1', 'create_time': datetime.datetime(2024, 1, 16, 14, 24, 40, 398877, tzinfo=tzutc()), 'definition': '[]'}]}

```python
workspace_id = workspace.id()
```

### Upload Model

The model is uploaded as a BYOP model, where the model, Python script and other artifacts are included in a .zip file.  This requires the input and output schemas for the model specified in Apache Arrow Schema format.

The following method will use the Wallaroo API to upload the model with its relevant input/output schemas.  Because of the size of this model, it may take anywhere from 15 to 45 minutes to upload, depending on the speed of your connection.

* References
  * [Wallaroo SDK Essentials Guide: Model Uploads and Registrations: Arbitrary Python](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/wallaroo-sdk-model-arbitrary-python/)

```python
# using curl to upload the file.

metadata = { 
    "name": "llama-chat",
    "visibility": "private",
    "workspace_id": workspace_id,
    "conversion": {
        "framework": "custom", 
        "python_version": "3.8", 
        "requirements": [], 
        "input_schema": "/////3AAAAAQAAAAAAAKAAwABgAFAAgACgAAAAABBAAMAAAACAAIAAAABAAIAAAABAAAAAEAAAAUAAAAEAAUAAgABgAHAAwAAAAQABAAAAAAAAEFEAAAABwAAAAEAAAAAAAAAAQAAAB0ZXh0AAAAAAQABAAEAAAA", 
        "output_schema": "/////7AAAAAQAAAAAAAKAAwABgAFAAgACgAAAAABBAAMAAAACAAIAAAABAAIAAAABAAAAAEAAAAEAAAAwP///wAAARAUAAAALAAAAAQAAAABAAAAOAAAAA4AAABnZW5lcmF0ZWRfdGV4dAAAAAAGAAgABAAGAAAAAQAAABAAFAAIAAYABwAMAAAAEAAQAAAAAAABBRAAAAAcAAAABAAAAAAAAAAEAAAAaXRlbQAAAAAEAAQABAAAAA=="
    }
}

# save metadata to a file
with open("./data/file_upload.json", "w") as outfile:
    json.dump(metadata, outfile)
```

```python
# only use once to upload the model

!curl {wl.api_endpoint}/v1/api/models/upload_and_convert \
  -H "Authorization: {wl.auth.auth_header()['Authorization']}" \
  -H "Content-Type: multipart/form-data" \
  -F "metadata=@./data/file_upload.json;type=application/json" \
  -F "file=@models/model-auto-conversion_BYOP_llama_byop_llamav2_new2.zip;type=application/octet-stream" \
  --progress-bar | cat
```

### Retrieve Model Version

We now retrieve the model version using the Wallaroo SDK.  This reference is used for the deployment steps.

```python
model = workspace.models()[-1].versions()[-1]
```

### Deploy Pipeline

Next we configure the hardware we want to use for deployment. If we plan on eventually deploying to edge, this is a good way to simulate edge hardware conditions.  The BYOP model is deployed as a Wallaroo Containerized Runtime, so the hardware allocation is performed through the `sidekick` options.

* References
  * [Wallaroo SDK Essentials Guide: Pipeline Deployment Configuration](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline-deployment-config/)

```python
deployment_config = DeploymentConfigBuilder() \
    .cpus(1).memory('1Gi') \
    .sidekick_gpus(model, 1) \
    .deployment_label('wallaroo.ai/gpu: a100') \
    .build()
```

```python
pipeline_name = "llamav2-chat-1"
llamav2_pipe = wl.build_pipeline(pipeline_name)
llamav2_pipe.add_model_step(model)

llamav2_pipe.deploy(deployment_config=deployment_config)
```

```python
llamav2_pipe.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.4.17',
       'name': 'engine-b44578ccb-mcvth',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'llamav2-chat-1',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'llama-chat',
          'version': '12c993cb-2b6c-4eaf-9cd7-099f4164d68c',
          'sha': '23c11e89fb3d3fe6e48f8817754a64e326d6d9ed9cd3cbdc0784cd48e684d4cc',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.0.203',
       'name': 'engine-lb-5df9b487cf-z62kz',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': [{'ip': '10.244.4.16',
       'name': 'engine-sidekick-llama-chat-1-87c574d49-f6d2q',
       'status': 'Running',
       'reason': None,
       'details': [],
       'statuses': '\n'}]}

### Testing Llama

We will now test our LlamaV2 model with a simple request, and display the generated text.

```python
input_df = pd.DataFrame({'text': ['Describe Virgin Australia']})
```

```python
out = llamav2_pipe.infer(input_df)
out["out.generated_text"][0]
```

    "Describe Virgin Australia's (VA) Fleet Structure\n\nVirgin Australia (VA) operates a diverse fleet of aircraft, with a mix of narrow-body, wide-body, and regional jets. Here is a brief overview of VA's current fleet structure:\n\n1. Narrow-body aircraft:\n\t* Airbus A320-200: 35 aircraft (used for short-haul flights within Australia and to nearby countries)\n\t* Airbus A321-200: 10 aircraft (used for long-haul flights within Australia and to nearby countries)\n2. Wide-body aircraft:\n\t* Boeing 777-300ER: 15 aircraft (used for long-haul flights to destinations in Asia, Europe, and the United States)\n\t* Airbus A330-200: 5 aircraft (used for long-haul flights to destinations in Asia and the Pacific)\n3. Regional jets:\n\t* Bombardier Q400: 10 aircraft (used for short-haul flights within Australia)\n\t* Fokker 100: 5 aircraft (used for short-haul flights within Australia)\n4. Future fleet plans:\n\t* Virgin Australia has announced plans to retire its Boeing 737-800 and 737-400 aircraft and replace them with new Airbus A320neos and A330neos.\n\t* The airline has also ordered 15 Airbus A220-300 aircraft, which will be delivered from 2020.\n\nVA's fleet structure is designed to meet the demand for domestic and international travel within Australia and to nearby countries. The airline's narrow-body aircraft are used for short-haul flights, while its wide-body aircraft are used for long-haul flights to more distant destinations. The regional jets are used for shorter flights within Australia."

### Undeploy Pipeline

With the demonstartion complete, we undeploy the pipeline and return the resources back to the cluster.

```python
pipeline.undeploy()
```
