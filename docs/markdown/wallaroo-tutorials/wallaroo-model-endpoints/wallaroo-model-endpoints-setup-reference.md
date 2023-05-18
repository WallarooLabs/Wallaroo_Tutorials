This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-features/wallaroo-model-endpoints).

## Wallaroo SDK Inference Tutorial

Wallaroo provides the ability to perform inferences through deployed pipelines via the Wallaroo SDK and the Wallaroo MLOps API.  This tutorial demonstrates performing inferences using the Wallaroo SDK.

This tutorial provides the following:

* `ccfraud.onnx`:  A pre-trained credit card fraud detection model.
* `data/cc_data_1k.arrow`, `data/cc_data_10k.arrow`: Sample testing data in Apache Arrow format with 1,000 and 10,000 records respectively.
* `wallaroo-model-endpoints-sdk.py`: A code-only version of this tutorial as a Python script.

This tutorial and sample data comes from the Machine Learning Group's demonstration on [Credit Card Fraud detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

### Prerequisites

The following is required for this tutorial:

* A [deployed Wallaroo instance](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-install-guides/) with [Model Endpoints Enabled](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-model-endpoints-guide/)
* The following Python libraries:
  * [`pandas`](https://pypi.org/project/pandas/)
  * [`polars`](https://pypi.org/project/polars/)
  * [`pyarrow`](https://pypi.org/project/pyarrow/)
  * [`wallaroo`](https://pypi.org/project/wallaroo/) (Installed in the Wallaroo JupyterHub service by default).

### Tutorial Goals

This demonstration provides a quick tutorial on performing inferences using the Wallaroo SDK using the Pipeline `infer` and `infer_from_file` methods.  This following steps will be performed:

* Connect to a Wallaroo instance using environmental variables.  This bypasses the browser link confirmation for a seamless login.  For more information, see the [Wallaroo SDK Essentials Guide:  Client Connection](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).
* Create a workspace for our models and pipelines.
* Upload the `ccfraud` model.
* Create a pipeline and add the `ccfraud` model as a pipeline step.
* Run a sample inference through SDK Pipeline `infer` method.
* Run a batch inference through SDK Pipeline `infer_from_file` method.
* Run a DataFrame and Arrow based inference through the pipeline Inference URL.

## Open a Connection to Wallaroo

The first step is to connect to Wallaroo through the Wallaroo client.  This example will store the user's credentials into the file `./creds.json` which contains the following:

```json
{
    "username": "{Connecting User's Username}", 
    "password": "{Connecting User's Password}", 
    "email": "{Connecting User's Email Address}"
}
```

Replace the `username`, `password`, and `email` fields with the user account connecting to the Wallaroo instance.  This allows a seamless connection to the Wallaroo instance and bypasses the standard browser based confirmation link.  For more information, see the [Wallaroo SDK Essentials Guide:  Client Connection](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

If running this example within the internal Wallaroo JupyterHub service, use the `wallaroo.Client(auth_type="user_password")` method. If connecting externally via the [Wallaroo SDK](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-install-guides/), use the following to specify the URL of the Wallaroo instance as defined in the [Wallaroo DNS Integration Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-dns-guide/), replacing `wallarooPrefix` and `wallarooSuffix` with your Wallaroo instance's DNS prefix and suffix.

```python
import wallaroo
from wallaroo.object import EntityNotFoundError
import pandas as pd

import pyarrow as pa

# used to display dataframe information without truncating
from IPython.display import display
pd.set_option('display.max_colwidth', None)

import requests

# Used to create unique workspace and pipeline names
import string
import random

# make a random 4 character prefix to prevent workspace and pipeline name clobbering
prefix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))
```

```python
# Retrieve the login credentials.
os.environ["WALLAROO_SDK_CREDENTIALS"] = './creds.json.example'

# Client connection from local Wallaroo instance

wl = wallaroo.Client(auth_type="user_password")

```

## Create the Workspace

We will create a workspace to work in and call it the `sdkinferenceexampleworkspace`, then set it as current workspace environment.  We'll also create our pipeline in advance as `sdkinferenceexamplepipeline`.

The model to be uploaded and used for inference will be labeled as `ccfraud`.

```python
workspace_name = f'{prefix}sdkinferenceexampleworkspace'
pipeline_name = f'{prefix}sdkinferenceexamplepipeline'
model_name = f'{prefix}ccfraud'
model_file_name = './ccfraud.onnx'
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

def get_pipeline(name):
    try:
        pipeline = wl.pipelines_by_name(name)[0]
    except EntityNotFoundError:
        pipeline = wl.build_pipeline(name)
    return pipeline
```

```python
workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)
```

    {'name': 'fvqusdkinferenceexampleworkspace', 'id': 22, 'archived': False, 'created_by': '028c8b48-c39b-4578-9110-0b5bdd3824da', 'created_at': '2023-05-17T21:53:12.482285+00:00', 'models': [], 'pipelines': []}

## Build Pipeline

In a production environment, the pipeline would already be set up with the model and pipeline steps.  We would then select it and use it to perform our inferences.

For this example we will create the pipeline and add the `ccfraud` model as a pipeline step and deploy it.  Deploying a pipeline allocates resources from the Kubernetes cluster hosting the Wallaroo instance and prepares it for performing inferences.

If this process was already completed, it can be commented out and skipped for the next step [Select Pipeline](#select-pipeline).

Then we will list the pipelines and select the one we will be using for the inference demonstrations.

```python
# Create or select the current pipeline

ccfraudpipeline = get_pipeline(pipeline_name)

# Add ccfraud model as the pipeline step

ccfraud_model = wl.upload_model(model_name, model_file_name).configure()

ccfraudpipeline.add_model_step(ccfraud_model).deploy()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>fvqusdkinferenceexamplepipeline</td></tr><tr><th>created</th> <td>2023-05-17 21:53:14.322069+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 21:53:15.826120+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>a987e13f-ffbe-4826-a6f5-9fd8de9f47fa, 0966d243-ce76-4132-aa69-0d287ae9a572</td></tr><tr><th>steps</th> <td>fvquccfraud</td></tr></table>
{{</table>}}

## Select Pipeline

This step assumes that the pipeline is prepared with `ccfraud` as the current step.  The method `pipelines_by_name(name)` returns an array of pipelines with names matching the `pipeline_name` field.  This example assumes only one pipeline is assigned the name `sdkinferenceexamplepipeline`.

```python
# List the pipelines by name in the current workspace - just the first several to save space.

display(wl.list_pipelines()[:5])

# Set the `pipeline` variable to our sample pipeline.

pipeline = wl.pipelines_by_name(pipeline_name)[0]
display(pipeline)
```

    [{'name': 'fvqusdkinferenceexamplepipeline', 'create_time': datetime.datetime(2023, 5, 17, 21, 53, 14, 322069, tzinfo=tzutc()), 'definition': '[]'},
     {'name': 'gobtedgepipelineexample', 'create_time': datetime.datetime(2023, 5, 17, 21, 50, 13, 166628, tzinfo=tzutc()), 'definition': '[]'},
     {'name': 'logpipeline', 'create_time': datetime.datetime(2023, 5, 17, 21, 41, 6, 308048, tzinfo=tzutc()), 'definition': '[]'},
     {'name': 'btffhotswappipeline', 'create_time': datetime.datetime(2023, 5, 17, 21, 37, 16, 33792, tzinfo=tzutc()), 'definition': '[]'},
     {'name': 'qjjoccfraudpipeline', 'create_time': datetime.datetime(2023, 5, 17, 21, 32, 6, 660305, tzinfo=tzutc()), 'definition': '[]'}]

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>fvqusdkinferenceexamplepipeline</td></tr><tr><th>created</th> <td>2023-05-17 21:53:14.322069+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 21:53:15.826120+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>a987e13f-ffbe-4826-a6f5-9fd8de9f47fa, 0966d243-ce76-4132-aa69-0d287ae9a572</td></tr><tr><th>steps</th> <td>fvquccfraud</td></tr></table>
{{</table>}}

## Interferences via SDK

Once a pipeline has been deployed, an inference can be run.  This will submit data to the pipeline, where it is processed through each of the pipeline's steps with the output of the previous step providing the input for the next step.  The final step will then output the result of all of the pipeline's steps.

* Inputs are either sent one of the following:
  * [pandas.DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html).  The return value will be a pandas.DataFrame.
  * [Apache Arrow](https://arrow.apache.org/).  The return value will be an Apache Arrow table.
  * [Custom JSON](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-inferences/#inferenceresult-object).  The return value will be a Wallaroo InferenceResult object.

Inferences are performed through the Wallaroo SDK via the Pipeline `infer` and `infer_from_file` methods.

### infer Method

Now that the pipeline is deployed we'll perform an inference using the Pipeline `infer` method, and submit a pandas DataFrame as our input data.  This will return a pandas DataFrame as the inference output.

For more information, see the [Wallaroo SDK Essentials Guide: Inferencing: Run Inference through Local Variable](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-inferences/wallaroo-sdk-inferences/#run-inference-through-local-variable).

```python
smoke_test = pd.DataFrame.from_records([
    {
        "tensor":[
            1.0678324729,
            0.2177810266,
            -1.7115145262,
            0.682285721,
            1.0138553067,
            -0.4335000013,
            0.7395859437,
            -0.2882839595,
            -0.447262688,
            0.5146124988,
            0.3791316964,
            0.5190619748,
            -0.4904593222,
            1.1656456469,
            -0.9776307444,
            -0.6322198963,
            -0.6891477694,
            0.1783317857,
            0.1397992467,
            -0.3554220649,
            0.4394217877,
            1.4588397512,
            -0.3886829615,
            0.4353492889,
            1.7420053483,
            -0.4434654615,
            -0.1515747891,
            -0.2668451725,
            -1.4549617756
        ]
    }
])
result = pipeline.infer(smoke_test)
display(result)
```

{{<table "table table-striped table-bordered" >}}
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.dense_1</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-05-17 21:53:39.595</td>
      <td>[1.0678324729, 0.2177810266, -1.7115145262, 0.682285721, 1.0138553067, -0.4335000013, 0.7395859437, -0.2882839595, -0.447262688, 0.5146124988, 0.3791316964, 0.5190619748, -0.4904593222, 1.1656456469, -0.9776307444, -0.6322198963, -0.6891477694, 0.1783317857, 0.1397992467, -0.3554220649, 0.4394217877, 1.4588397512, -0.3886829615, 0.4353492889, 1.7420053483, -0.4434654615, -0.1515747891, -0.2668451725, -1.4549617756]</td>
      <td>[0.0014974177]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}

### infer_from_file Method

This example uses the Pipeline method `infer_from_file` to submit 10,000 records as a batch using an Apache Arrow table.  The method will return an Apache Arrow table.  For more information, see the [Wallaroo SDK Essentials Guide: Inferencing: Run Inference From A File](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-inferences/wallaroo-sdk-inferences/#run-inference-from-a-file)

The results will be converted into a [pandas.DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html).  The results will be filtered by transactions likely to be credit card fraud.

```python
result = pipeline.infer_from_file('./data/cc_data_10k.arrow')

display(result)
```

    pyarrow.Table
    time: timestamp[ms]
    in.tensor: list<item: float> not null
      child 0, item: float
    out.dense_1: list<inner: float not null> not null
      child 0, inner: float not null
    check_failures: int8
    ----
    time: [[2023-05-17 21:53:40.465,2023-05-17 21:53:40.465,2023-05-17 21:53:40.465,2023-05-17 21:53:40.465,2023-05-17 21:53:40.465,...,2023-05-17 21:53:40.465,2023-05-17 21:53:40.465,2023-05-17 21:53:40.465,2023-05-17 21:53:40.465,2023-05-17 21:53:40.465]]
    in.tensor: [[[-1.0603298,2.3544967,-3.5638788,5.138735,-1.2308457,...,0.038412016,1.0993439,1.2603409,-0.14662448,-1.4463212],[-1.0603298,2.3544967,-3.5638788,5.138735,-1.2308457,...,0.038412016,1.0993439,1.2603409,-0.14662448,-1.4463212],...,[-2.1694233,-3.1647356,1.2038506,-0.2649221,0.0899006,...,1.8174038,-0.19327773,0.94089776,0.825025,1.6242892],[-0.12405868,0.73698884,1.0311689,0.59917533,0.11831961,...,-0.36567155,-0.87004745,0.41288367,0.49470216,-0.6710689]]]
    out.dense_1: [[[0.99300325],[0.99300325],...,[0.00024175644],[0.0010648072]]]
    check_failures: [[0,0,0,0,0,...,0,0,0,0,0]]

```python
# use pyarrow to convert results to a pandas DataFrame and display only the results with > 0.75

list = [0.75]

outputs =  result.to_pandas()
# display(outputs)
filter = [elt[0] > 0.75 for elt in outputs['out.dense_1']]
outputs = outputs.loc[filter]
display(outputs)
```

{{<table "table table-striped table-bordered" >}}
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.dense_1</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-05-17 21:53:40.465</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>[0.99300325]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-05-17 21:53:40.465</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>[0.99300325]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-05-17 21:53:40.465</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>[0.99300325]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-05-17 21:53:40.465</td>
      <td>[-1.0603298, 2.3544967, -3.5638788, 5.138735, -1.2308457, -0.76878244, -3.5881228, 1.8880838, -3.2789674, -3.9563255, 4.099344, -5.653918, -0.8775733, -9.131571, -0.6093538, -3.7480276, -5.0309124, -0.8748149, 1.9870535, 0.7005486, 0.9204423, -0.10414918, 0.32295644, -0.74181414, 0.038412016, 1.0993439, 1.2603409, -0.14662448, -1.4463212]</td>
      <td>[0.99300325]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>161</th>
      <td>2023-05-17 21:53:40.465</td>
      <td>[-9.716793, 9.174981, -14.450761, 8.653825, -11.039951, 0.6602411, -22.825525, -9.919395, -8.064324, -16.737926, 4.852197, -12.563343, -1.0762653, -7.524591, -3.2938414, -9.62102, -15.6501045, -7.089741, 1.7687134, 5.044906, -11.365625, 4.5987034, 4.4777045, 0.31702697, -2.2731977, 0.07944675, -10.052058, -2.024108, -1.0611985]</td>
      <td>[1.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>941</th>
      <td>2023-05-17 21:53:40.465</td>
      <td>[-0.50492376, 1.9348029, -3.4217603, 2.2165704, -0.6545315, -1.9004827, -1.6786858, 0.5380051, -2.7229102, -5.265194, 3.504164, -5.4661765, 0.68954825, -8.725291, 2.0267954, -5.4717045, -4.9123807, -1.6131229, 3.8021576, 1.3881834, 1.0676425, 0.28200775, -0.30759808, -0.48498034, 0.9507336, 1.5118006, 1.6385275, 1.072455, 0.7959132]</td>
      <td>[0.9873102]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1445</th>
      <td>2023-05-17 21:53:40.465</td>
      <td>[-7.615594, 4.659706, -12.057331, 7.975307, -5.1068773, -1.6116138, -12.146941, -0.5952333, -6.4605103, -12.535655, 10.017626, -14.839381, 0.34900802, -14.953928, -0.3901092, -9.342014, -14.285043, -5.758632, 0.7512068, 1.4632998, -3.3777077, 0.9950705, -0.5855211, -1.6528498, 1.9089833, 1.6860862, 5.5044003, -3.703297, -1.4715525]</td>
      <td>[1.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2092</th>
      <td>2023-05-17 21:53:40.465</td>
      <td>[-14.115489, 9.905631, -18.67885, 4.602589, -15.404288, -3.7169847, -15.887272, 15.616176, -3.2883947, -7.0224414, 4.086536, -5.7809114, 1.2251061, -5.4301147, -0.14021407, -6.0200763, -12.957546, -5.545689, 0.86074656, 2.2463796, 2.492611, -2.9649208, -2.265674, 0.27490455, 3.9263225, -0.43438172, 3.1642237, 1.2085277, 0.8223642]</td>
      <td>[0.99999]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2220</th>
      <td>2023-05-17 21:53:40.465</td>
      <td>[-0.1098309, 2.5842443, -3.5887418, 4.63558, 1.1825614, -1.2139517, -0.7632139, 0.6071841, -3.7244265, -3.501917, 4.3637576, -4.612757, -0.44275254, -10.346612, 0.66243565, -0.33048683, 1.5961986, 2.5439718, 0.8787973, 0.7406088, 0.34268215, -0.68495077, -0.48357907, -1.9404846, -0.059520483, 1.1553137, 0.9918434, 0.7067319, -1.6016251]</td>
      <td>[0.91080534]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4135</th>
      <td>2023-05-17 21:53:40.465</td>
      <td>[-0.547029, 2.2944348, -4.149202, 2.8648357, -0.31232587, -1.5427867, -2.1489344, 0.9471863, -2.663241, -4.2572775, 2.1116028, -6.2264414, -1.1307784, -6.9296007, 1.0049651, -5.876498, -5.6855297, -1.5800936, 3.567338, 0.5962099, 1.6361043, 1.8584082, -0.08202618, 0.46620172, -2.234368, -0.18116793, 1.744976, 2.1414309, -1.6081295]</td>
      <td>[0.98877275]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4236</th>
      <td>2023-05-17 21:53:40.465</td>
      <td>[-3.135635, -1.483817, -3.0833669, 1.6626456, -0.59695035, -0.30199608, -3.316563, 1.869609, -1.8006078, -4.5662026, 2.8778172, -4.0887237, -0.43401834, -3.5816982, 0.45171788, -5.725131, -8.982029, -4.0279546, 0.89264476, 0.24721873, 1.8289508, 1.6895254, -2.5555577, -2.4714024, -0.4500012, 0.23333028, 2.2119386, -2.041805, 1.1568314]</td>
      <td>[0.95601666]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5658</th>
      <td>2023-05-17 21:53:40.465</td>
      <td>[-5.4078765, 3.9039962, -8.98522, 5.128742, -7.373224, -2.946234, -11.033238, 5.914019, -5.669241, -12.041053, 6.950792, -12.488795, 1.2236942, -14.178565, 1.6514667, -12.47019, -22.350504, -8.928755, 4.54775, -0.11478994, 3.130207, -0.70128506, -0.40275285, 0.7511918, -0.1856308, 0.92282087, 0.146656, -1.3761806, 0.42997098]</td>
      <td>[1.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6768</th>
      <td>2023-05-17 21:53:40.465</td>
      <td>[-16.900557, 11.7940855, -21.349983, 4.746453, -17.54182, -3.415758, -19.897173, 13.8569145, -3.570626, -7.388376, 3.0761156, -4.0583425, 1.2901028, -2.7997534, -0.4298746, -4.777225, -11.371295, -5.2725616, 0.0964799, 4.2148075, -0.8343371, -2.3663573, -1.6571938, 0.2110055, 4.438088, -0.49057993, 2.342008, 1.4479793, -1.4715525]</td>
      <td>[0.9999745]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6780</th>
      <td>2023-05-17 21:53:40.465</td>
      <td>[-0.74893713, 1.3893062, -3.7477517, 2.4144504, -0.11061429, -1.0737498, -3.1504633, 1.2081385, -1.332872, -4.604276, 4.438548, -7.687688, 1.1683422, -5.3296027, -0.19838685, -5.294243, -5.4928794, -1.3254275, 4.387228, 0.68643385, 0.87228596, -0.1154091, -0.8364338, -0.61202216, 0.10518055, 2.2618086, 1.1435078, -0.32623357, -1.6081295]</td>
      <td>[0.9852645]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7133</th>
      <td>2023-05-17 21:53:40.465</td>
      <td>[-7.5131927, 6.507386, -12.439463, 5.7453, -9.513038, -1.4236209, -17.402607, -3.0903268, -5.378041, -15.169325, 5.7585907, -13.448207, -0.45244268, -8.495097, -2.2323692, -11.429063, -19.578058, -8.367617, 1.8869618, 2.1813896, -4.799091, 2.4388566, 2.9503248, 0.6293566, -2.6906652, -2.1116931, -6.4196434, -1.4523355, -1.4715525]</td>
      <td>[1.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7566</th>
      <td>2023-05-17 21:53:40.465</td>
      <td>[-2.1804514, 1.0243497, -4.3890443, 3.4924, -3.7609894, 0.023624033, -2.7677023, 1.1786921, -2.9450424, -6.8823, 6.1294384, -9.564066, -1.6273017, -10.940607, 0.3062539, -8.854589, -15.382658, -5.419305, 3.2210033, -0.7381137, 0.9632334, 0.6612066, 2.1337948, -0.90536207, 0.7498649, -0.019404415, 5.5950212, 0.26602694, 1.7534728]</td>
      <td>[0.9999705]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7911</th>
      <td>2023-05-17 21:53:40.465</td>
      <td>[-1.594454, 1.8545462, -2.6311765, 2.759316, -2.6988854, -0.08155677, -3.8566258, -0.04912437, -1.9640644, -4.2058415, 3.391933, -6.471933, -0.9877536, -6.188904, 1.2249585, -8.652863, -11.170872, -6.134417, 2.5400054, -0.29327056, 3.591464, 0.3057127, -0.052313827, 0.06196331, -0.82863224, -0.2595842, 1.0207018, 0.019899422, 1.0935433]</td>
      <td>[0.9980203]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8921</th>
      <td>2023-05-17 21:53:40.465</td>
      <td>[-0.21756083, 1.786712, -3.4240367, 2.7769134, -1.420116, -2.1018193, -3.4615245, 0.7367844, -2.3844852, -6.3140697, 4.382665, -8.348951, -1.6409378, -10.611383, 1.1813216, -6.251184, -10.577264, -3.5184007, 0.7997489, 0.97915924, 1.081642, -0.7852368, -0.4761941, -0.10635195, 2.066527, -0.4103488, 2.8288178, 1.9340333, -1.4715525]</td>
      <td>[0.99950194]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9244</th>
      <td>2023-05-17 21:53:40.465</td>
      <td>[-3.314442, 2.4431305, -6.1724143, 3.6737356, -3.81542, -1.5950849, -4.8292923, 2.9850774, -4.22416, -7.5519834, 6.1932964, -8.59886, 0.25443414, -11.834097, -0.39583337, -6.015362, -13.532762, -4.226845, 1.1153877, 0.17989528, 1.3166595, -0.64433384, 0.2305495, -0.5776498, 0.7609739, 2.2197483, 4.01189, -1.2347667, 1.2847253]</td>
      <td>[0.9999876]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10176</th>
      <td>2023-05-17 21:53:40.465</td>
      <td>[-5.0815525, 3.9294617, -8.4077635, 6.373701, -7.391173, -2.1574461, -10.345097, 5.5896044, -6.3736906, -11.330594, 6.618754, -12.93748, 1.1884484, -13.9628935, 1.0340953, -12.278127, -23.333889, -8.886669, 3.5720036, -0.3243157, 3.4229393, 0.493529, 0.08469851, 0.791218, 0.30968663, 0.6811129, 0.39306796, -1.5204874, 0.9061435]</td>
      <td>[1.0]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}

## Inferences via HTTP POST

Each pipeline has its own Inference URL that allows HTTP/S POST submissions of inference requests.  Full details are available from the [Inferencing via the Wallaroo MLOps API](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-inferences/wallaroo-api-inferences/).

This example will demonstrate performing inferences with a DataFrame input and an Apache Arrow input.

### Request JWT Token

There are two ways to retrieve the JWT token used to authenticate to the Wallaroo MLOps API.

* [Wallaroo SDK](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-api-essential-guide/#through-the-wallaroo-sdk).  This method requires a Wallaroo based user.
* [API Clent Secret](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-api-essential-guide/#through-keycloak).  This is the recommended method as it is user independent.  It allows any valid user to make an inference request.

This tutorial will use the Wallaroo SDK method Wallaroo Client `wl.auth.auth_header()` method, extracting the Authentication header from the response.

Reference:  [MLOps API Retrieve Token Through Wallaroo SDK](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/wallaroo-mlops-api-essential-guide/#through-the-wallaroo-sdk)

```python
headers = wl.auth.auth_header()
display(headers)
```

    {'Authorization': 'Bearer eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJDYkFqN19QY0xCWTFkWmJiUDZ6Q3BsbkNBYTd6US0tRHlyNy0yLXlQb25nIn0.eyJleHAiOjE2ODQzNjA0NTEsImlhdCI6MTY4NDM2MDM5MSwiYXV0aF90aW1lIjoxNjg0MzU1OTU5LCJqdGkiOiJkNGI0Mjc3ZC0yYTg4LTRmYTMtYTZjZS05OTc3YzgwZDBhMWMiLCJpc3MiOiJodHRwczovL2RvYy10ZXN0LmtleWNsb2FrLndhbGxhcm9vY29tbXVuaXR5Lm5pbmphL2F1dGgvcmVhbG1zL21hc3RlciIsImF1ZCI6WyJtYXN0ZXItcmVhbG0iLCJhY2NvdW50Il0sInN1YiI6IjAyOGM4YjQ4LWMzOWItNDU3OC05MTEwLTBiNWJkZDM4MjRkYSIsInR5cCI6IkJlYXJlciIsImF6cCI6InNkay1jbGllbnQiLCJzZXNzaW9uX3N0YXRlIjoiMGJlODJjN2ItNzg1My00ZjVkLWJiNWEtOTlkYjUwYjhiNDVmIiwiYWNyIjoiMCIsInJlYWxtX2FjY2VzcyI6eyJyb2xlcyI6WyJkZWZhdWx0LXJvbGVzLW1hc3RlciIsIm9mZmxpbmVfYWNjZXNzIiwidW1hX2F1dGhvcml6YXRpb24iXX0sInJlc291cmNlX2FjY2VzcyI6eyJtYXN0ZXItcmVhbG0iOnsicm9sZXMiOlsibWFuYWdlLXVzZXJzIiwidmlldy11c2VycyIsInF1ZXJ5LWdyb3VwcyIsInF1ZXJ5LXVzZXJzIl19LCJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6InByb2ZpbGUgZW1haWwiLCJzaWQiOiIwYmU4MmM3Yi03ODUzLTRmNWQtYmI1YS05OWRiNTBiOGI0NWYiLCJlbWFpbF92ZXJpZmllZCI6ZmFsc2UsImh0dHBzOi8vaGFzdXJhLmlvL2p3dC9jbGFpbXMiOnsieC1oYXN1cmEtdXNlci1pZCI6IjAyOGM4YjQ4LWMzOWItNDU3OC05MTEwLTBiNWJkZDM4MjRkYSIsIngtaGFzdXJhLWRlZmF1bHQtcm9sZSI6InVzZXIiLCJ4LWhhc3VyYS1hbGxvd2VkLXJvbGVzIjpbInVzZXIiXSwieC1oYXN1cmEtdXNlci1ncm91cHMiOiJ7fSJ9LCJuYW1lIjoiSm9obiBIYW5zYXJpY2siLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJqb2huLmh1bW1lbEB3YWxsYXJvby5haSIsImdpdmVuX25hbWUiOiJKb2huIiwiZmFtaWx5X25hbWUiOiJIYW5zYXJpY2siLCJlbWFpbCI6ImpvaG4uaHVtbWVsQHdhbGxhcm9vLmFpIn0.NA6esxU5tSVbu_JJIgLgX7-1ZN5WsWQkNcFy-6dW3uwV0wMAMUuYSgAk1wdsdi5Bp1UReXpIA4qjBKkFr5SductPGeloutLz1XeH2m4Ah8z6PyzjkxIBDarjrQo6GZlwxxxFx5e6UutwYpvFsnShemh4D44cmB6Og1xZFI_YqU6MwKbjrAsdy71QIcYsSpGqdZiFe1fYaFcumRXjxc_x068ZDvfb-1FIwNVVcGCe8U4hkVmCd5EIDWxp8L1XzDuItFxrit2XeiDrqacIF2DRI3ZF8jMdfr30r5TjLSydBVJ33RMxMI_I4-kqtZ43rnSZDNP1Dpd53Iwf5iEe-Zbsmg'}

### Retrieve the Pipeline Inference URL

The Pipeline Inference URL is retrieved via the Wallaroo SDK with the Pipeline `._deployment._url()` method.

* **IMPORTANT NOTE**:  The `_deployment._url()` method will return an **internal** URL when using Python commands from within the Wallaroo instance - for example, the Wallaroo JupyterHub service.  When connecting via an external connection, `_deployment._url()` returns an **external** URL.
  * External URL connections requires [the authentication be included in the HTTP request](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/), and [Model Endpoints](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-model-endpoints-guide/) are enabled in the Wallaroo configuration options.

```python
deploy_url = pipeline._deployment._url()
print(deploy_url)
```

    https://doc-test.api.wallarooexample.ai/v1/api/pipelines/infer/fvqusdkinferenceexamplepipeline-25/fvqusdkinferenceexamplepipeline

### HTTP Inference with DataFrame Input

The following example performs a HTTP Inference request with a DataFrame input.  The request will be made with first a Python `requests` method, then using `curl`.

```python
# get authorization header
headers = wl.auth.auth_header()

## Inference through external URL using dataframe

# retrieve the json data to submit
data = pd.DataFrame.from_records([
    {
        "tensor":[
            1.0678324729,
            0.2177810266,
            -1.7115145262,
            0.682285721,
            1.0138553067,
            -0.4335000013,
            0.7395859437,
            -0.2882839595,
            -0.447262688,
            0.5146124988,
            0.3791316964,
            0.5190619748,
            -0.4904593222,
            1.1656456469,
            -0.9776307444,
            -0.6322198963,
            -0.6891477694,
            0.1783317857,
            0.1397992467,
            -0.3554220649,
            0.4394217877,
            1.4588397512,
            -0.3886829615,
            0.4353492889,
            1.7420053483,
            -0.4434654615,
            -0.1515747891,
            -0.2668451725,
            -1.4549617756
        ]
    }
])

# set the content type for pandas records
headers['Content-Type']= 'application/json; format=pandas-records'

# set accept as pandas-records
headers['Accept']='application/json; format=pandas-records'

# submit the request via POST, import as pandas DataFrame
response = pd.DataFrame.from_records(
                requests.post(
                    deploy_url, 
                    data=data.to_json(orient="records"), 
                    headers=headers)
                .json()
            )
display(response.loc[:,["time", "out"]])
```

{{<table "table table-striped table-bordered" >}}
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1684360421564</td>
      <td>{'dense_1': [0.0014974177]}</td>
    </tr>
  </tbody>
</table>
{{</table>}}

```python
!curl -X POST {deploy_url} -H "Authorization: {headers['Authorization']}" -H "Content-Type:{headers['Content-Type']}" -H "Accept:{headers['Accept']}" --data '{data.to_json(orient="records")}'
```

    [{"time":1684360422009,"in":{"tensor":[1.0678324729,0.2177810266,-1.7115145262,0.682285721,1.0138553067,-0.4335000013,0.7395859437,-0.2882839595,-0.447262688,0.5146124988,0.3791316964,0.5190619748,-0.4904593222,1.1656456469,-0.9776307444,-0.6322198963,-0.6891477694,0.1783317857,0.1397992467,-0.3554220649,0.4394217877,1.4588397512,-0.3886829615,0.4353492889,1.7420053483,-0.4434654615,-0.1515747891,-0.2668451725,-1.4549617756]},"out":{"dense_1":[0.0014974177]},"check_failures":[],"metadata":{"last_model":"{\"model_name\":\"fvquccfraud\",\"model_sha\":\"bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507\"}","pipeline_version":"a987e13f-ffbe-4826-a6f5-9fd8de9f47fa","elapsed":[65601,286203]}}]

### HTTP Inference with Arrow Input

The following example performs a HTTP Inference request with an Apache Arrow input.  The request will be made with first a Python `requests` method, then using `curl`.

Only the first 5 rows will be displayed for space purposes.

```python
# get authorization header
headers = wl.auth.auth_header()

# Submit arrow file
dataFile="./data/cc_data_10k.arrow"

data = open(dataFile,'rb').read()

# set the content type for Arrow table
headers['Content-Type']= "application/vnd.apache.arrow.file"

# set accept as Apache Arrow
headers['Accept']="application/vnd.apache.arrow.file"

response = requests.post(
                    deploy_url, 
                    headers=headers, 
                    data=data, 
                    verify=True
                )

# Arrow table is retrieved 
with pa.ipc.open_file(response.content) as reader:
    arrow_table = reader.read_all()

# convert to Polars DataFrame and display the first 5 rows
display(arrow_table.to_pandas().head(5).loc[:,["time", "out"]])
```

{{<table "table table-striped table-bordered" >}}
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1684360470098</td>
      <td>{'dense_1': [0.99300325]}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1684360470098</td>
      <td>{'dense_1': [0.99300325]}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1684360470098</td>
      <td>{'dense_1': [0.99300325]}</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1684360470098</td>
      <td>{'dense_1': [0.99300325]}</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1684360470098</td>
      <td>{'dense_1': [0.0010916889]}</td>
    </tr>
  </tbody>
</table>
{{</table>}}

```python
!curl -X POST {deploy_url} -H "Authorization: {headers['Authorization']}" -H "Content-Type:{headers['Content-Type']}" -H "Accept:{headers['Accept']}" --data-binary @{dataFile} > curl_response.arrow
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100 4200k  100 3037k  100 1162k  1408k   539k  0:00:02  0:00:02 --:--:-- 1957kk  1290k   560k  0:00:02  0:00:02 --:--:-- 1859k

## Undeploy Pipeline

When finished with our tests, we will undeploy the pipeline so we have the Kubernetes resources back for other tasks.

```python
pipeline.undeploy()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>fvqusdkinferenceexamplepipeline</td></tr><tr><th>created</th> <td>2023-05-17 21:53:14.322069+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 21:53:15.826120+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>a987e13f-ffbe-4826-a6f5-9fd8de9f47fa, 0966d243-ce76-4132-aa69-0d287ae9a572</td></tr><tr><th>steps</th> <td>fvquccfraud</td></tr></table>
{{</table>}}

