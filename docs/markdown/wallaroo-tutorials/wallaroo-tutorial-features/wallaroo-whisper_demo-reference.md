This tutorial can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/blob/main/wallaroo-model-cookbooks/hf-whisper).

## ER Whisper Demo

The following tutorial demonstrates deploying the [openai/whisper-large-v2](https://huggingface.co/openai/whisper-large-v2) on a `Wallaroo` pipeline and performing  inferences on it using the [BYOP](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/wallaroo-sdk-model-arbitrary-python/) feature.

## Data Prepartions

For this example, the following Python libraries were used:

* [librosa](https://pypi.org/project/librosa/)
* [datasets](https://pypi.org/project/datasets/)

These can be installed with the following command:

```python
pip install librosa datasets --user
```

For these libraries, a sample of audio files was retrieved and converted using the following code.

```python
import librosa
from datasets import load_dataset

# load the sample dataset and retrieve the audio files
dataset = load_dataset("Narsil/asr_dummy")

# the following is used to play them
audio_1, sr_1 = librosa.load(dataset["test"][0]["file"])
audio_2, sr_2 = librosa.load(dataset["test"][1]["file"])

audio_files = [(audio_1, sr_1), (audio_2, sr_2)]

# convert the audio files to numpy values in a DataFrame
input_data = {
        "inputs": [audio_1, audio_2],
        "return_timestamps": ["word", "word"],
}
dataframe = pd.DataFrame(input_data)
```

The resulting pandas DataFrame can either be submitted directly to a deployed Wallaroo pipeline using `wallaroo.pipeline.infer`, or the DataFrame exported to a pandas Record file in pandas JSON format, and used for an inference request using `wallaroo.pipeline.infer_from_file`.

For this example, the audio files are pre-converted to a JSON pandas Record table file, and used for the inference result.  This removes the requirements to add additional Python libraries to a virtual environment or Wallaroo JupyterHub service.  The code above is provided as an example of converting the dataset audio into values for inference requests.

## Tutorial Steps

### Import Libraries

The first step is to import the libraries we'll be using.  These are included by default in the Wallaroo instance's JupyterHub service or are installed with the Wallaroo SDK.

* References
  * [Wallaroo SDK Guides](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/)

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

# ignoring warnings for demonstration
import warnings
warnings.filterwarnings('ignore')
```

```python
wallaroo.__version__
```

    '2023.4.0+5d935fefc'

## Open a Connection to Wallaroo

The next step is connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  If logging in externally, update the `wallarooPrefix` and `wallarooSuffix` variables with the proper DNS information.  For more information on Wallaroo DNS settings, see the [Wallaroo DNS Integration Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-dns-guide/).

For this tutorial, the `request_timeout` option is increased to allow the model conversion and pipeline deployment to proceed without any warning messages.

* References
  * [Wallaroo SDK Essentials Guide: Client Connection](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/)

```python
wl = wallaroo.Client(request_timeout=60000)
```

### Set Variables and Helper Functions

We'll set the name of our workspace, pipeline, models and files.  Workspace names must be unique across the Wallaroo workspace.  For this, we'll add in a randomly generated 4 characters to the workspace name to prevent collisions with other users' workspaces.  If running this tutorial, we recommend hard coding the workspace name so it will function in the same workspace each time it's run.

We'll set up some helper functions that will either use existing workspaces and pipelines, or create them if they do not already exist.

```python
def get_workspace(name, client):
    workspace = None
    for ws in wl.list_workspaces():
        if ws.name() == name:
            workspace= ws
    if(workspace == None):
        workspace = wl.create_workspace(name)
    return workspace
```

The names for our workspace, pipeline, model, and model files are set here to make updating this tutorial easier.  

* **IMPORTANT NOTE**:  Workspace names must be unique across the Wallaroo instance.  To verify unique names, the randomization code below is provided to allow the workspace name to be unique.  If this is not required, set `suffix` to `''`.

```python
import string
import random

# make a random 4 character suffix to prevent overwriting other user's workspaces
suffix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))

suffix=''

workspace_name = f'whisper-tiny-demo{suffix}'
pipeline_name = 'whisper-hf-byop'
model_name = 'whisper-byop'
model_file_name = './models/model-auto-conversion_hugging-face_complex-pipelines_asr-whisper-tiny.zip'

```

### Create Workspace and Pipeline

We will now create the Wallaroo workspace to store our model and set it as the current workspace.  Future commands will default to this workspace for pipeline creation, model uploads, etc.  We'll create our Wallaroo pipeline that is used to deploy our arbitrary Python model.

* References
  * [Wallaroo SDK Essentials Guide: Workspace Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-workspace/)

```python
workspace = get_workspace(workspace_name, wl)
wl.set_current_workspace(workspace)

pipeline = wl.build_pipeline(pipeline_name)

display(wl.get_current_workspace())
```

    {'name': 'whisper-tiny-demojch', 'id': 32, 'archived': False, 'created_by': '9aa81a1f-952f-435e-b77d-504dd0215914', 'created_at': '2023-12-19T15:51:28.027575+00:00', 'models': [], 'pipelines': [{'name': 'whisper-hf-byop', 'create_time': datetime.datetime(2023, 12, 19, 15, 51, 28, 50715, tzinfo=tzutc()), 'definition': '[]'}]}

## Configure & Upload Model

For this example, we will use the `openai/whisper-tiny` model for the `automatic-speech-recognition` pipeline task from the official `🤗 Hugging Face` [hub](https://huggingface.co/openai/whisper-tiny/tree/main).

To manually create an `automatic-speech-recognition` pipeline from the `🤗 Hugging Face` hub link above:

1. Download the original model from the the official `🤗 Hugging Face` [hub](https://huggingface.co/openai/whisper-tiny/tree/main).

```python
from transformers import pipeline

pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
pipe.save_pretrained("asr-whisper-tiny/")
```

As a last step, you can `zip` the folder containing all needed files as follows:

```bash
zip -r asr-whisper-tiny.zip asr-whisper-tiny/
```

### Configure PyArrow Schema

You can find more info on the available inputs for the `automatic-speech-recognition` pipeline under the [official source code](https://github.com/huggingface/transformers/blob/main/src/transformers/pipelines/automatic_speech_recognition.py#L294) from `🤗 Hugging Face`.

The input and output schemas are defined in Apache pyarrow Schema format.

The model is then uploaded with the `wallaroo.client.model_upload` method, where we define:

* The name to assign the model.
* The model file path.
* The input and output schemas.

The model is uploaded to the Wallaroo instance, where it is containerized to run with the Wallaroo Inference Engine.

* References
  * [Wallaroo SDK Essentials Guide: Model Uploads and Registrations: Arbitrary Python](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/wallaroo-sdk-model-arbitrary-python/)
  * [Wallaroo SDK Essentials Guide: Model Uploads and Registrations: Hugging Face](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/wallaroo-sdk-model-upload-hugging-face/)

```python
input_schema = pa.schema([
    pa.field('inputs', pa.list_(pa.float32())), # required: the audio stored in numpy arrays of shape (num_samples,) and data type `float32`
    pa.field('return_timestamps', pa.string()) # optional: return start & end times for each predicted chunk
]) 

output_schema = pa.schema([
    pa.field('text', pa.string()), # required: the output text corresponding to the audio input
    pa.field('chunks', pa.list_(pa.struct([('text', pa.string()), ('timestamp', pa.list_(pa.float32()))]))), # required (if `return_timestamps` is set), start & end times for each predicted chunk
])
```

```python
model = wl.upload_model(model_name, 
                        model_file_name, 
                        framework=Framework.HUGGING_FACE_AUTOMATIC_SPEECH_RECOGNITION, 
                        input_schema=input_schema, 
                        output_schema=output_schema)
model
```

    Waiting for model loading - this will take up to 10.0min.
    Model is pending loading to a container runtime..
    Model is attempting loading to a container runtime............................................successful
    
    Ready

<table>
        <tr>
          <td>Name</td>
          <td>whisper-byop</td>
        </tr>
        <tr>
          <td>Version</td>
          <td>6c36e84e-297c-4eec-9c38-9d7fb4e4e9db</td>
        </tr>
        <tr>
          <td>File Name</td>
          <td>model-auto-conversion_hugging-face_complex-pipelines_asr-whisper-tiny.zip</td>
        </tr>
        <tr>
          <td>SHA</td>
          <td>ddd57c9c8d3ed5417783ebb7101421aa1e79429365d20326155c9c02ae1e8a13</td>
        </tr>
        <tr>
          <td>Status</td>
          <td>ready</td>
        </tr>
        <tr>
          <td>Image Path</td>
          <td>proxy.replicated.com/proxy/wallaroo/ghcr.io/wallaroolabs/mlflow-deploy:v2023.4.0-4297</td>
        </tr>
        <tr>
          <td>Architecture</td>
          <td>None</td>
        </tr>
        <tr>
          <td>Updated At</td>
          <td>2023-19-Dec 15:55:27</td>
        </tr>
      </table>

### Deploy Pipeline

The model is deployed with the `wallaroo.pipeline.deploy(deployment_config)` command.  For the deployment configuration, we set the containerized aka `sidekick` memory to 8 GB to accommodate the size of the model, and CPUs to at least 4.  To optimize performance, a GPU could be assigned to the containerized model.

* References
  * [Wallaroo SDK Essentials Guide: Pipeline Deployment Configuration](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline-deployment-config/)
  * [Wallaroo SDK Essentials Guide: Pipeline Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline/)
  * [GPU Support](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline-deployment-config/#gpu-support)

```python
deployment_config = DeploymentConfigBuilder() \
    .cpus(0.25).memory('1Gi') \
    .sidekick_memory(model, '8Gi') \
    .sidekick_cpus(model, 4.0) \
    .build()
```

```python
pipeline = wl.build_pipeline(pipeline_name)
pipeline.add_model_step(model)

pipeline.deploy(deployment_config=deployment_config)
```

     ok

<table><tr><th>name</th> <td>whisper-hf-byop</td></tr><tr><th>created</th> <td>2023-12-19 15:51:28.050715+00:00</td></tr><tr><th>last_updated</th> <td>2023-12-19 16:03:12.950146+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>f03d5844-0a71-4791-bac9-b265680d0239, c9492252-e9c1-4e16-af96-4a9c9be2690a, 7579cda5-b36e-4b9b-a30e-1d544f03b9f8, 510b93fa-8389-4055-ae64-3e0e52016f88, 2e035d66-81d4-40f3-8e23-0b711953a176</td></tr><tr><th>steps</th> <td>whisper-byop</td></tr><tr><th>published</th> <td>False</td></tr></table>

After a couple of minutes we verify the pipeline deployment was successful.

```python
pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.9.7',
       'name': 'engine-65499bc68b-v2vbk',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'whisper-hf-byop',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'whisper-byop',
          'version': '6c36e84e-297c-4eec-9c38-9d7fb4e4e9db',
          'sha': 'ddd57c9c8d3ed5417783ebb7101421aa1e79429365d20326155c9c02ae1e8a13',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.2.112',
       'name': 'engine-lb-584f54c899-zlmqq',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': [{'ip': '10.244.9.6',
       'name': 'engine-sidekick-whisper-byop-16-6d5c4cd7f7-mfcpg',
       'status': 'Running',
       'reason': None,
       'details': [],
       'statuses': '\n'}]}

### Run inference on the example dataset

We perform a sample inference with the provided DataFrame, and display the results.

```python
%%time
result = pipeline.infer_from_file('./data/sound-examples.df.json', timeout=10000)
```

    CPU times: user 138 ms, sys: 12.4 ms, total: 150 ms
    Wall time: 6.02 s

```python
display(result)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.inputs</th>
      <th>in.return_timestamps</th>
      <th>out.chunks</th>
      <th>out.text</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-12-19 16:03:18.686</td>
      <td>[0.0003229662, 0.0003370901, 0.0002854846, 0.0...</td>
      <td>word</td>
      <td>[{'text': ' He', 'timestamp': [0.0, 1.08]}, {'...</td>
      <td>He hoped there would be Stu for dinner, turni...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-12-19 16:03:18.686</td>
      <td>[0.0010076478, 0.0012469155, 0.0008045971, 0.0...</td>
      <td>word</td>
      <td>[{'text': ' Stuff', 'timestamp': [29.78, 29.78...</td>
      <td>Stuff it into you. His belly calcled him.</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

### Evaluate results

Let's compare the results side by side with the audio inputs.

```python
for transcription in result['out.text'].values:
    print(f"Transcription: {transcription}\n")
```

    Transcription:  He hoped there would be Stu for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered, flour-fat and sauce.
    
    Transcription:  Stuff it into you. His belly calcled him.
    

### Undeploy Pipelines

With the demonstration complete, we undeploy the pipelines to return the resources back to the Wallaroo instance.

```python
pipeline.undeploy()
```

     ok

<table><tr><th>name</th> <td>whisper-hf-byop</td></tr><tr><th>created</th> <td>2023-12-19 15:51:28.050715+00:00</td></tr><tr><th>last_updated</th> <td>2023-12-19 16:03:12.950146+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>f03d5844-0a71-4791-bac9-b265680d0239, c9492252-e9c1-4e16-af96-4a9c9be2690a, 7579cda5-b36e-4b9b-a30e-1d544f03b9f8, 510b93fa-8389-4055-ae64-3e0e52016f88, 2e035d66-81d4-40f3-8e23-0b711953a176</td></tr><tr><th>steps</th> <td>whisper-byop</td></tr><tr><th>published</th> <td>False</td></tr></table>

