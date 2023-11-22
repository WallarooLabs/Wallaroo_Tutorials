This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/blob/main/wallaroo-features/onnx-multi-input-demo).

## ONNX Multiple Input Output Example

The following example demonstrates some of the data and input requirements when working with ONNX models in Wallaroo.  This example will:

* Upload an ONNX model trained to accept multiple inputs and return multiple outputs.
* Deploy the model, and show how to format the data for inference requests through a Wallaroo pipeline.

For more information on using ONNX models with Wallaroo, see [Wallaroo SDK Essentials Guide: Model Uploads and Registrations: ONNX](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/wallaroo-sdk-model-upload-onnx/).

## Steps

### Import Libraries

The first step is to import the libraries used for our demonstration - primarily the Wallaroo SDK, which is used to connect to the Wallaroo Ops instance, upload models, etc.

* References
  * [Wallaroo SDK Install Guides](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-install-guides/)

```python
import wallaroo
from wallaroo.deployment_config import DeploymentConfigBuilder
from wallaroo.framework import Framework
import pyarrow as pa
import numpy as np
import pandas as pd 
```

### Connect to the Wallaroo Instance

The next step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
# Login through local Wallaroo instance

wl = wallaroo.Client()
```

### Create Workspace

We will create a workspace to manage our pipeline and models.  The following variables will set the name of our sample workspace then set it as the current workspace.  If this tutorial has been run before, the helper function `get_workspace` will either create or connect to an existing workspace.

Workspace names must be unique; verify that no other workspaces have the same name when running this tutorial.  We then set the current workspace to our new workspace; all model uploads and other requests will use this 

```python
def get_workspace(workspace_name, wallaroo_client):
    workspace = None
    for ws in wallaroo_client.list_workspaces():
        if ws.name() == workspace_name:
            workspace= ws
    if(workspace == None):
        workspace = wallaroo_client.create_workspace(workspace_name)
    return workspace
```

```python
workspace_name = 'onnx-tutorial'

workspace = get_workspace(workspace_name, wl)

wl.set_current_workspace(workspace)
```

    {'name': 'onnx-tutorial', 'id': 9, 'archived': False, 'created_by': '12ea09d1-0f49-405e-bed1-27eb6d10fde4', 'created_at': '2023-11-22T16:24:47.786643+00:00', 'models': [], 'pipelines': []}

### Upload Model

The ONNX model `./models/multi_io.onnx` will be uploaded with the `wallaroo.client.upload_model` method.  This requires:

* The designated model name.
* The path for the file.
* The framework aka what kind of model it is based on the `wallaroo.framework.Framework` options.

If we wanted to overwrite the name of the input fields, we could use the `wallaroo.client.upload_model.configure(tensor_fields[field_names])` option.  This ONNX model takes the inputs `input_1` and `input_2`.

* References
  * [Wallaroo SDK Essentials Guide: Model Uploads and Registrations: ONNX](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/wallaroo-sdk-model-upload-onnx/)

```python
model = wl.upload_model('onnx-multi-io-model', 
                        "./models/multi_io.onnx", 
                        framework=Framework.ONNX)
model

```

<table>
        <tr>
          <td>Name</td>
          <td>onnx-multi-io-model</td>
        </tr>
        <tr>
          <td>Version</td>
          <td>7adb9245-53c2-43b4-95df-2c907bb88161</td>
        </tr>
        <tr>
          <td>File Name</td>
          <td>multi_io.onnx</td>
        </tr>
        <tr>
          <td>SHA</td>
          <td>bb3e51dfdaa6440359c2396033a84a4248656d0f81ba1f662751520b3f93de27</td>
        </tr>
        <tr>
          <td>Status</td>
          <td>ready</td>
        </tr>
        <tr>
          <td>Image Path</td>
          <td>None</td>
        </tr>
        <tr>
          <td>Architecture</td>
          <td>None</td>
        </tr>
        <tr>
          <td>Updated At</td>
          <td>2023-22-Nov 16:24:51</td>
        </tr>
      </table>

### Create the Pipeline and Add Steps

A new pipeline 'multi-io-example' is created with the `wallaroo.client.build_pipeline` method that creates a new Wallaroo pipeline within our current workspace.  We then add our `onnx-multi-io-model` as a pipeline step.

* References
  * [Wallaroo SDK Essentials Guide: Pipeline Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline/)

```python
pipeline_name = 'multi-io-example'

pipeline = wl.build_pipeline(pipeline_name)

# in case this pipeline was run before
pipeline.clear()
pipeline.add_model_step(model)
```

<table><tr><th>name</th> <td>multi-io-example</td></tr><tr><th>created</th> <td>2023-11-22 16:24:53.843958+00:00</td></tr><tr><th>last_updated</th> <td>2023-11-22 16:24:54.523098+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>73c1b57d-3227-471a-8e9b-4a8af62188dd, c8fb97d9-50cd-475d-8f36-1d2290e4c585</td></tr><tr><th>steps</th> <td>onnx-multi-io-model</td></tr><tr><th>published</th> <td>False</td></tr></table>

### Deploy Pipeline

With the model set, deploy the pipeline with a deployment configuration.  This sets the number of resources that the pipeline will be allocated from the Wallaroo Ops cluster and makes it available for inference requests.

* References
  * [Wallaroo SDK Essentials Guide: Pipeline Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline/)
  * [Wallaroo SDK Essentials Guide: Pipeline Deployment Configuration](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipelines/wallaroo-sdk-essentials-pipeline-deployment-config/)

```python
deployment_config = DeploymentConfigBuilder() \
    .cpus(0.25).memory('1Gi') \
    .build()

pipeline.deploy(deployment_config=deployment_config)
pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.3.143',
       'name': 'engine-857444867-nldj5',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'multi-io-example',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'onnx-multi-io-model',
          'version': '7adb9245-53c2-43b4-95df-2c907bb88161',
          'sha': 'bb3e51dfdaa6440359c2396033a84a4248656d0f81ba1f662751520b3f93de27',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.4.155',
       'name': 'engine-lb-584f54c899-h647p',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}

### Sample Inference

For our inference request, we will create a dummy DataFrame with the following fields:

* input_1: A list of randomly generated numbers.
* input_2: A list of randomly generated numbers.

10 rows will be created.

Inference requests for Wallaroo for ONNX models must meet the following criteria:

* Equal rows constraint:  The number of input rows and output rows must match.
* All inputs are tensors:  The inputs are tensor arrays with the same shape.
* Data Type Consistency:  Data types within each tensor are of the same type.

Note that each input meets these requirements:

* Each input is one row, and will correspond to a single output row.
* Each input is a tensor.  Field values are a list contained within their field.
* Each input is the same data type - for example, a list of floats.

For more details, see [Wallaroo SDK Essentials Guide: Model Uploads and Registrations: ONNX](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-model-uploads/wallaroo-sdk-model-upload-onnx/).

```python
np.random.seed(1)
mock_inference_data = [np.random.rand(10, 10), np.random.rand(10, 5)]
mock_dataframe = pd.DataFrame(
    {
        "input_1": mock_inference_data[0].tolist(),
        "input_2": mock_inference_data[1].tolist(),
    }
)

display(mock_dataframe)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>input_1</th>
      <th>input_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[0.417022004702574, 0.7203244934421581, 0.0001...</td>
      <td>[0.32664490177209615, 0.5270581022576093, 0.88...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[0.4191945144032948, 0.6852195003967595, 0.204...</td>
      <td>[0.6233601157918027, 0.015821242846556283, 0.9...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[0.8007445686755367, 0.9682615757193975, 0.313...</td>
      <td>[0.17234050834532855, 0.13713574962887776, 0.9...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[0.0983468338330501, 0.42110762500505217, 0.95...</td>
      <td>[0.7554630526024664, 0.7538761884612464, 0.923...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[0.9888610889064947, 0.7481656543798394, 0.280...</td>
      <td>[0.01988013383979559, 0.026210986877719278, 0....</td>
    </tr>
    <tr>
      <th>5</th>
      <td>[0.019366957870297075, 0.678835532939891, 0.21...</td>
      <td>[0.5388310643416528, 0.5528219786857659, 0.842...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>[0.10233442882782584, 0.4140559878195683, 0.69...</td>
      <td>[0.5857592714582879, 0.9695957483196745, 0.561...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>[0.9034019152878835, 0.13747470414623753, 0.13...</td>
      <td>[0.23297427384102043, 0.8071051956187791, 0.38...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>[0.8833060912058098, 0.6236722070556089, 0.750...</td>
      <td>[0.5562402339904189, 0.13645522566068502, 0.05...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>[0.11474597295337519, 0.9494892587070712, 0.44...</td>
      <td>[0.1074941291060929, 0.2257093386078547, 0.712...</td>
    </tr>
  </tbody>
</table>

We now perform an inference with our sample inference request with the `wallaroo.pipeline.infer` method.  The returning DataFrame displays the input variables as `in.{variable_name}`, and the output variables as `out.{variable_name}`.  Each inference output row corresponds with an input row.

* References
  * [Wallaroo SDK Essentials Guide: Inference Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-inferences/)

```python
results = pipeline.infer(mock_dataframe)
results
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.input_1</th>
      <th>in.input_2</th>
      <th>out.output_1</th>
      <th>out.output_2</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-11-22 16:27:10.632</td>
      <td>[0.4170220047, 0.7203244934, 0.0001143748, 0.3...</td>
      <td>[0.3266449018, 0.5270581023, 0.8859420993, 0.3...</td>
      <td>[-0.16188532, -0.2735075, -0.10427341]</td>
      <td>[-0.18745898, -0.035904408]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-11-22 16:27:10.632</td>
      <td>[0.4191945144, 0.6852195004, 0.2044522497, 0.8...</td>
      <td>[0.6233601158, 0.0158212428, 0.9294372337, 0.6...</td>
      <td>[-0.16437894, -0.24449202, -0.10489924]</td>
      <td>[-0.17241219, -0.09285815]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-11-22 16:27:10.632</td>
      <td>[0.8007445687, 0.9682615757, 0.3134241782, 0.6...</td>
      <td>[0.1723405083, 0.1371357496, 0.932595463, 0.69...</td>
      <td>[-0.1431846, -0.33338487, -0.1858185]</td>
      <td>[-0.25035447, -0.095617786]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-11-22 16:27:10.632</td>
      <td>[0.0983468338, 0.421107625, 0.9578895302, 0.53...</td>
      <td>[0.7554630526, 0.7538761885, 0.9230245355, 0.7...</td>
      <td>[-0.21010575, -0.38097042, -0.26413786]</td>
      <td>[-0.081432916, -0.12933002]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-11-22 16:27:10.632</td>
      <td>[0.9888610889, 0.7481656544, 0.2804439921, 0.7...</td>
      <td>[0.0198801338, 0.0262109869, 0.028306488, 0.24...</td>
      <td>[-0.29807547, -0.362104, -0.04459526]</td>
      <td>[-0.23403212, 0.019275911]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2023-11-22 16:27:10.632</td>
      <td>[0.0193669579, 0.6788355329, 0.211628116, 0.26...</td>
      <td>[0.5388310643, 0.5528219787, 0.8420308924, 0.1...</td>
      <td>[-0.14283556, -0.29290834, -0.1613777]</td>
      <td>[-0.20929304, -0.10064016]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2023-11-22 16:27:10.632</td>
      <td>[0.1023344288, 0.4140559878, 0.6944001577, 0.4...</td>
      <td>[0.5857592715, 0.9695957483, 0.5610302193, 0.0...</td>
      <td>[-0.2372348, -0.29803842, -0.17791237]</td>
      <td>[-0.20062584, -0.026013546]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2023-11-22 16:27:10.632</td>
      <td>[0.9034019153, 0.1374747041, 0.1392763473, 0.8...</td>
      <td>[0.2329742738, 0.8071051956, 0.3878606441, 0.8...</td>
      <td>[-0.27525327, -0.46431914, -0.2719731]</td>
      <td>[-0.17208403, -0.1618222]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2023-11-22 16:27:10.632</td>
      <td>[0.8833060912, 0.6236722071, 0.750942434, 0.34...</td>
      <td>[0.556240234, 0.1364552257, 0.0599176895, 0.12...</td>
      <td>[-0.3599869, -0.37006766, 0.05214046]</td>
      <td>[-0.26465484, 0.08243461]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2023-11-22 16:27:10.632</td>
      <td>[0.114745973, 0.9494892587, 0.4499121335, 0.57...</td>
      <td>[0.1074941291, 0.2257093386, 0.7129889804, 0.5...</td>
      <td>[-0.20812269, -0.3822521, -0.14788152]</td>
      <td>[-0.19157144, -0.12436578]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

### Undeploy the Pipeline

With the tutorial complete, we will undeploy the pipeline and return the resources back to the cluster.

```python
pipeline.undeploy()
```

<table><tr><th>name</th> <td>multi-io-example</td></tr><tr><th>created</th> <td>2023-11-22 16:24:53.843958+00:00</td></tr><tr><th>last_updated</th> <td>2023-11-22 16:26:59.272421+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>arch</th> <td>None</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>2487574d-133e-4af3-91f2-31c456dbe215, 73c1b57d-3227-471a-8e9b-4a8af62188dd, c8fb97d9-50cd-475d-8f36-1d2290e4c585</td></tr><tr><th>steps</th> <td>onnx-multi-io-model</td></tr><tr><th>published</th> <td>False</td></tr></table>

