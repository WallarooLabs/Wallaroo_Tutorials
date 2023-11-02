This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-features/parallel-inference-aloha-tutorial).

## Aloha Parallel Inference Demonstration

This tutorial will focus on the Pipeline method `parallel_infer`, which allows a List of data to be submitted to a Wallaroo instance for parallel inference requests.  This provides high speed increases in situations where data has to be broken up for size and memory needs, data is requested from multiple sources and submitted in a single request, or other use cases.

For this example we will be using an open source model that uses an [Aloha CNN LSTM model](https://www.researchgate.net/publication/348920204_Using_Auxiliary_Inputs_in_Deep_Learning_Models_for_Detecting_DGA-based_Domain_Names) for classifying Domain names as being either legitimate or being used for nefarious purposes such as malware distribution.  

## Tutorial Goals

* Create a workspace for our work.
* Upload the Aloha TensorFlow model.
* Create a pipeline that can ingest our submitted data, submit it to the model, and export the results.
* Run a sample inference through our pipeline by loading a file.
* Run a batch inference to show submitting a set of data to an inference request.
* Split a DataFrame into a List of 1,000 separate DataFrames to simulate separate inference requests.
* Submit the List of DataFrames sequentially and display how long this takes.
* Submit the same List of DataFrames with `parallel_infer` and compare how long it takes.

## Prerequisites

* A Wallaroo version 2023.2.1 and above instance.

## Reference

[Wallaroo SDK Essentials Guide: Inference Management](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-inferences/#parallel-inferences)

## Open a Connection to Wallaroo

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

```python
import wallaroo
import asyncio 
import datetime
from wallaroo.object import EntityNotFoundError
from wallaroo.framework import Framework

# used to display dataframe information without truncating
import pandas as pd
pd.set_option('display.max_colwidth', None)

# to display dataframe tables
from IPython.display import display

import warnings
warnings.filterwarnings('ignore')
```

```python
# Client connection from local Wallaroo instance

wl = wallaroo.Client()
```

## Create the Workspace

We will create a workspace to work in and call it the "alohaworkspace", then set it as current workspace environment.  We'll also create our pipeline in advance as `alohapipeline`.  The model name and the model file will be specified for use in later steps.

To allow this tutorial to be run multiple times or by multiple users in the same Wallaroo instance, a random 4 character prefix will be added to the workspace, pipeline, and model.

```python
import string
import random

# make a random 4 character prefix
prefix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))
workspace_name = f"{prefix}alohaworkspace"
pipeline_name = f"{prefix}alohapipeline"
model_name = f"{prefix}alohamodel"
model_file_name = './models/alohacnnlstm.zip'
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
        pipeline = wl.pipelines_by_name(pipeline_name)[0]
    except EntityNotFoundError:
        pipeline = wl.build_pipeline(pipeline_name)
    return pipeline
```

```python
workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)

aloha_pipeline = get_pipeline(pipeline_name)
aloha_pipeline
```

<table><tr><th>name</th> <td>ejatalohapipeline</td></tr><tr><th>created</th> <td>2023-08-28 16:59:39.670085+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-28 16:59:39.670085+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>71611219-8559-410b-9bae-5f55a2ae245e</td></tr><tr><th>steps</th> <td></td></tr><tr><th>published</th> <td>False</td></tr></table>

# Upload the Models

Now we will upload our models.  Note that for this example we are applying the model from a .ZIP file.  The Aloha model is a [protobuf](https://developers.google.com/protocol-buffers) file that has been defined for evaluating web pages, and we will configure it to use data in the `tensorflow` format.

```python
model = wl.upload_model(model_name, model_file_name, framework=wallaroo.framework.Framework.TENSORFLOW).configure("tensorflow")
```

## Deploy a model

Now that we have a model that we want to use we will create a deployment for it. 

We will tell the deployment we are using a tensorflow model and give the deployment name and the configuration we want for the deployment.

```python
aloha_pipeline.add_model_step(model)
```

<table><tr><th>name</th> <td>ejatalohapipeline</td></tr><tr><th>created</th> <td>2023-08-28 16:59:39.670085+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-28 16:59:39.670085+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>71611219-8559-410b-9bae-5f55a2ae245e</td></tr><tr><th>steps</th> <td></td></tr><tr><th>published</th> <td>False</td></tr></table>

```python
REPLICAS = 2
deployment_config = (wallaroo.DeploymentConfigBuilder()
    .replica_count(REPLICAS)
    .build())
```

```python
aloha_pipeline = aloha_pipeline.deploy(deployment_config =deployment_config)
```

We can verify that the pipeline is running and list what models are associated with it.

```python
aloha_pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.3.98',
       'name': 'engine-555d4dc96f-vkfsv',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'ejatalohapipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'ejatalohamodel',
          'version': '62f2e8aa-673c-4549-b792-f8e3feba9d7b',
          'sha': 'd71d9ffc61aaac58c2b1ed70a2db13d1416fb9d3f5b891e5e4e2e97180fe22f8',
          'status': 'Running'}]}},
      {'ip': '10.244.0.137',
       'name': 'engine-555d4dc96f-cktm6',
       'status': 'Running',
       'reason': None,
       'details': ['containers with unready status: [engine]',
        'containers with unready status: [engine]'],
       'pipeline_statuses': {'pipelines': [{'id': 'ejatalohapipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'ejatalohamodel',
          'version': '62f2e8aa-673c-4549-b792-f8e3feba9d7b',
          'sha': 'd71d9ffc61aaac58c2b1ed70a2db13d1416fb9d3f5b891e5e4e2e97180fe22f8',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.4.113',
       'name': 'engine-lb-584f54c899-frfgz',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}

## Interferences

### Infer 1 row

Now that the pipeline is deployed and our Aloha model is in place, we'll perform a smoke test to verify the pipeline is up and running properly.  We'll use the `infer_from_file` command to load a single encoded URL into the inference engine and print the results back out.

The result should tell us that the tokenized URL is legitimate (0) or fraud (1).  This sample data should return close to 0.

```python
result = aloha_pipeline.infer_from_file('./data/data_1.df.json')

display(result)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.text_input</th>
      <th>out.banjori</th>
      <th>out.corebot</th>
      <th>out.cryptolocker</th>
      <th>out.dircrypt</th>
      <th>out.gozi</th>
      <th>out.kraken</th>
      <th>out.locky</th>
      <th>out.main</th>
      <th>out.matsnu</th>
      <th>out.pykspa</th>
      <th>out.qakbot</th>
      <th>out.ramdo</th>
      <th>out.ramnit</th>
      <th>out.simda</th>
      <th>out.suppobox</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-08-28 17:00:01.886</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 16, 32, 23, 29, 32, 30, 19, 26, 17]</td>
      <td>[0.0015195842]</td>
      <td>[0.98291475]</td>
      <td>[0.012099549]</td>
      <td>[4.759116e-05]</td>
      <td>[2.028935e-05]</td>
      <td>[0.00031977228]</td>
      <td>[0.011029261]</td>
      <td>[0.997564]</td>
      <td>[0.010341614]</td>
      <td>[0.008038961]</td>
      <td>[0.016155055]</td>
      <td>[0.0062362333]</td>
      <td>[0.0009985747]</td>
      <td>[1.7933368e-26]</td>
      <td>[1.3889844e-27]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

### Batch Inference Example

Now we'll perform a batch inference.  We have the file `./data/data_25k.df.json`, which is a pandas DataFrame file with 25,000 records to analyze.  We'll provide it to the pipeline and perform a sample inference, and provide the first 20 rows.

```python
%time

test_data = pd.read_json("./data/data_25k.df.json")

batch_result = aloha_pipeline.infer(test_data.head(1000))
display(batch_result.head(20))
```

    CPU times: user 1 µs, sys: 0 ns, total: 1 µs
    Wall time: 6.91 µs

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.text_input</th>
      <th>out.banjori</th>
      <th>out.corebot</th>
      <th>out.cryptolocker</th>
      <th>out.dircrypt</th>
      <th>out.gozi</th>
      <th>out.kraken</th>
      <th>out.locky</th>
      <th>out.main</th>
      <th>out.matsnu</th>
      <th>out.pykspa</th>
      <th>out.qakbot</th>
      <th>out.ramdo</th>
      <th>out.ramnit</th>
      <th>out.simda</th>
      <th>out.suppobox</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-08-28 17:00:02.996</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 28, 16, 32, 23, 29, 32, 30, 19, 26, 17]</td>
      <td>[0.0015195821]</td>
      <td>[0.98291475]</td>
      <td>[0.012099549]</td>
      <td>[4.7591115e-05]</td>
      <td>[2.0289428e-05]</td>
      <td>[0.00031977257]</td>
      <td>[0.011029262]</td>
      <td>[0.997564]</td>
      <td>[0.010341609]</td>
      <td>[0.008038961]</td>
      <td>[0.016155055]</td>
      <td>[0.00623623]</td>
      <td>[0.0009985747]</td>
      <td>[1.7933434e-26]</td>
      <td>[1.388995e-27]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-08-28 17:00:02.996</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 20, 19, 27, 14, 17, 24, 13, 23, 20, 18, 35, 18, 22, 23]</td>
      <td>[7.447196e-18]</td>
      <td>[6.7359245e-08]</td>
      <td>[0.1708199]</td>
      <td>[1.3220122e-09]</td>
      <td>[1.2758706e-24]</td>
      <td>[0.22559543]</td>
      <td>[0.3420985]</td>
      <td>[0.99999994]</td>
      <td>[0.3080186]</td>
      <td>[0.1828217]</td>
      <td>[3.802255e-11]</td>
      <td>[0.2062254]</td>
      <td>[0.15215826]</td>
      <td>[1.1701982e-30]</td>
      <td>[3.1514454e-38]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-08-28 17:00:02.996</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 33, 25, 36, 25, 31, 14, 32, 36, 25, 12, 35, 34, 30, 28, 27, 24, 29, 27]</td>
      <td>[2.8598649e-21]</td>
      <td>[9.302004e-08]</td>
      <td>[0.04445298]</td>
      <td>[6.163758e-09]</td>
      <td>[8.3496755e-23]</td>
      <td>[0.4823448]</td>
      <td>[0.26332903]</td>
      <td>[1.0]</td>
      <td>[0.29800338]</td>
      <td>[0.22361776]</td>
      <td>[1.5238921e-06]</td>
      <td>[0.32820392]</td>
      <td>[0.029332489]</td>
      <td>[1.1995622e-31]</td>
      <td>[0.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-08-28 17:00:02.996</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 23, 22, 15, 12, 35, 34, 36, 12, 18, 24, 34, 32, 36, 12, 14, 16, 27, 22, 23]</td>
      <td>[2.1387213e-15]</td>
      <td>[3.8817485e-10]</td>
      <td>[0.045599736]</td>
      <td>[1.9090386e-07]</td>
      <td>[1.3140123e-25]</td>
      <td>[0.59542626]</td>
      <td>[0.17374137]</td>
      <td>[0.9999997]</td>
      <td>[0.23151578]</td>
      <td>[0.1759168]</td>
      <td>[1.0876152e-09]</td>
      <td>[0.2183228]</td>
      <td>[0.0128692705]</td>
      <td>[6.1588803e-28]</td>
      <td>[1.4386237e-35]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-08-28 17:00:02.996</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 13, 14, 12, 33, 16, 23, 15, 22, 30, 28, 26, 12, 16, 32, 37, 29, 22, 28, 22, 16, 27, 32]</td>
      <td>[9.4533425e-15]</td>
      <td>[7.091151e-10]</td>
      <td>[0.049815163]</td>
      <td>[5.2914135e-09]</td>
      <td>[7.4132087e-19]</td>
      <td>[1.5504575e-13]</td>
      <td>[1.079181e-15]</td>
      <td>[0.9999989]</td>
      <td>[1.5003075e-15]</td>
      <td>[0.33075705]</td>
      <td>[2.625885e-07]</td>
      <td>[0.5036279]</td>
      <td>[0.020393765]</td>
      <td>[0.0]</td>
      <td>[2.3292326e-38]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2023-08-28 17:00:02.996</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 29, 20, 33, 13, 36, 35, 30, 21, 29, 17, 26, 19, 25, 36, 14, 23, 16, 18, 15, 21, 18, 28, 35, 19]</td>
      <td>[1.7247285e-17]</td>
      <td>[8.1354194e-08]</td>
      <td>[0.013697116]</td>
      <td>[5.608618e-11]</td>
      <td>[1.4032912e-17]</td>
      <td>[0.49469122]</td>
      <td>[0.119788595]</td>
      <td>[0.99999994]</td>
      <td>[0.19000013]</td>
      <td>[0.105966926]</td>
      <td>[5.5244395e-06]</td>
      <td>[0.24210057]</td>
      <td>[0.006943502]</td>
      <td>[1.2804911e-34]</td>
      <td>[9.482465e-35]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2023-08-28 17:00:02.996</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 36, 14, 12, 23, 14, 13, 20, 20, 23, 27, 36, 29, 35, 19, 33, 22, 25, 26, 32, 21]</td>
      <td>[5.5500796e-18]</td>
      <td>[3.3608708e-07]</td>
      <td>[0.023452949]</td>
      <td>[1.1318812e-10]</td>
      <td>[1.0496877e-22]</td>
      <td>[0.23692918]</td>
      <td>[0.06445695]</td>
      <td>[0.99999183]</td>
      <td>[0.07306594]</td>
      <td>[0.06499429]</td>
      <td>[1.4302713e-08]</td>
      <td>[0.11925242]</td>
      <td>[0.0011031044]</td>
      <td>[1.520634e-32]</td>
      <td>[0.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2023-08-28 17:00:02.996</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36, 22, 28, 23, 20, 25, 21, 20, 16, 12, 33, 21, 14, 34, 34, 32, 19, 36, 17, 29, 26, 14, 29]</td>
      <td>[3.9222717e-18]</td>
      <td>[1.4074378e-10]</td>
      <td>[0.0109469]</td>
      <td>[8.202828e-11]</td>
      <td>[2.4549838e-24]</td>
      <td>[0.42107272]</td>
      <td>[0.071240015]</td>
      <td>[0.9982491]</td>
      <td>[0.118182994]</td>
      <td>[0.08340967]</td>
      <td>[1.9207924e-09]</td>
      <td>[0.16958168]</td>
      <td>[0.0005199056]</td>
      <td>[0.0]</td>
      <td>[0.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2023-08-28 17:00:02.996</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27, 30, 33, 29, 37, 24, 33, 16, 20, 24]</td>
      <td>[4.0574582e-11]</td>
      <td>[1.0878829e-09]</td>
      <td>[0.17916855]</td>
      <td>[1.7313038e-06]</td>
      <td>[8.697294e-18]</td>
      <td>[9.197087e-16]</td>
      <td>[3.8521368e-17]</td>
      <td>[0.9999977]</td>
      <td>[3.265452e-17]</td>
      <td>[0.32568428]</td>
      <td>[6.8342887e-09]</td>
      <td>[0.3700783]</td>
      <td>[0.44918337]</td>
      <td>[0.0]</td>
      <td>[2.0823871e-26]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2023-08-28 17:00:02.996</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 29, 19, 35, 31, 15, 14, 21, 26, 31, 34, 27, 22]</td>
      <td>[2.2576374e-09]</td>
      <td>[2.0812656e-09]</td>
      <td>[0.17788415]</td>
      <td>[1.1887505e-08]</td>
      <td>[1.0785658e-11]</td>
      <td>[0.04125281]</td>
      <td>[0.21430445]</td>
      <td>[0.9999988]</td>
      <td>[0.1785375]</td>
      <td>[0.13382338]</td>
      <td>[0.00011408964]</td>
      <td>[0.14033839]</td>
      <td>[0.011299975]</td>
      <td>[3.5758114e-24]</td>
      <td>[7.164692e-24]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2023-08-28 17:00:02.996</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 25, 19, 26, 30, 19, 29, 37, 20, 24, 23, 22, 20, 20, 12, 35, 29, 26, 16, 35, 36, 32, 23, 19]</td>
      <td>[7.892612e-12]</td>
      <td>[3.0390893e-07]</td>
      <td>[0.015696576]</td>
      <td>[5.4462755e-13]</td>
      <td>[1.2192627e-22]</td>
      <td>[2.9611054e-17]</td>
      <td>[2.630575e-20]</td>
      <td>[0.9999961]</td>
      <td>[6.984627e-20]</td>
      <td>[0.28895634]</td>
      <td>[1.8219469e-10]</td>
      <td>[0.5132747]</td>
      <td>[0.031628624]</td>
      <td>[0.0]</td>
      <td>[6.496084e-32]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2023-08-28 17:00:02.996</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 33, 34, 15, 23, 28, 34, 18, 33, 33]</td>
      <td>[2.656041e-16]</td>
      <td>[5.9408256e-09]</td>
      <td>[0.12814318]</td>
      <td>[3.334544e-08]</td>
      <td>[2.2118839e-18]</td>
      <td>[0.3078207]</td>
      <td>[0.2768143]</td>
      <td>[0.9999999]</td>
      <td>[0.2790456]</td>
      <td>[0.17737383]</td>
      <td>[7.047458e-08]</td>
      <td>[0.17205149]</td>
      <td>[0.20136173]</td>
      <td>[3.6788262e-29]</td>
      <td>[4.9193303e-33]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2023-08-28 17:00:02.996</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 28, 30, 13, 26, 28, 30, 33, 25, 30, 31, 30, 26, 34, 19, 18, 23, 18, 15]</td>
      <td>[1.9262605e-07]</td>
      <td>[0.00011627659]</td>
      <td>[0.015093412]</td>
      <td>[6.062207e-06]</td>
      <td>[2.7446008e-08]</td>
      <td>[0.1944088]</td>
      <td>[0.11690318]</td>
      <td>[0.9999991]</td>
      <td>[0.17412055]</td>
      <td>[0.064938694]</td>
      <td>[0.49536943]</td>
      <td>[0.08959365]</td>
      <td>[0.005527823]</td>
      <td>[2.4333354e-38]</td>
      <td>[1.3592967e-25]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2023-08-28 17:00:02.996</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 19, 17, 22, 25, 35, 29, 26, 15]</td>
      <td>[1.8286097e-05]</td>
      <td>[0.00021055616]</td>
      <td>[0.012560271]</td>
      <td>[1.669594e-12]</td>
      <td>[1.2260838e-07]</td>
      <td>[0.007982208]</td>
      <td>[0.01670425]</td>
      <td>[0.017594406]</td>
      <td>[0.017098008]</td>
      <td>[0.011611045]</td>
      <td>[0.00011716153]</td>
      <td>[0.009795011]</td>
      <td>[0.010660364]</td>
      <td>[3.1872973e-35]</td>
      <td>[6.0048404e-27]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2023-08-28 17:00:02.996</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36, 22, 17, 34, 27, 18, 18, 30]</td>
      <td>[3.6237115e-22]</td>
      <td>[1.0416453e-05]</td>
      <td>[0.33487734]</td>
      <td>[2.1746202e-06]</td>
      <td>[8.6172205e-23]</td>
      <td>[0.029006457]</td>
      <td>[0.2075723]</td>
      <td>[0.99999344]</td>
      <td>[0.13615957]</td>
      <td>[0.08263349]</td>
      <td>[2.8077036e-09]</td>
      <td>[0.05675183]</td>
      <td>[0.100090384]</td>
      <td>[1.0977557e-18]</td>
      <td>[1.6076543e-32]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2023-08-28 17:00:02.996</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 21, 19, 31, 31, 19, 36, 22, 12, 37, 18, 22, 31, 29]</td>
      <td>[2.6339812e-11]</td>
      <td>[3.014702e-10]</td>
      <td>[0.041572697]</td>
      <td>[2.9721878e-11]</td>
      <td>[4.1457936e-19]</td>
      <td>[2.8498805e-12]</td>
      <td>[1.091722e-13]</td>
      <td>[0.99999815]</td>
      <td>[1.532856e-13]</td>
      <td>[0.15687592]</td>
      <td>[6.4997073e-07]</td>
      <td>[0.27979007]</td>
      <td>[0.07243408]</td>
      <td>[6.264585e-28]</td>
      <td>[3.73621e-33]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2023-08-28 17:00:02.996</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21, 15, 26, 15, 14, 19, 32, 24, 33, 13, 20, 22, 32, 14, 25, 26, 35, 22, 12, 31, 23, 19, 31]</td>
      <td>[2.3916345e-11]</td>
      <td>[1.022118e-06]</td>
      <td>[0.0036410657]</td>
      <td>[3.0198125e-10]</td>
      <td>[6.5029504e-10]</td>
      <td>[0.01702933]</td>
      <td>[0.02470826]</td>
      <td>[0.99999654]</td>
      <td>[0.031047786]</td>
      <td>[0.029724386]</td>
      <td>[1.1598425e-05]</td>
      <td>[0.053846892]</td>
      <td>[6.4680105e-05]</td>
      <td>[1.9701085e-31]</td>
      <td>[8.561262e-37]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2023-08-28 17:00:02.996</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 31, 21, 21, 26, 33, 17, 16, 23, 28, 20, 19, 29, 25, 24, 30, 20, 35, 19, 36]</td>
      <td>[5.9892094e-14]</td>
      <td>[4.9572558e-05]</td>
      <td>[0.014003561]</td>
      <td>[6.212124e-13]</td>
      <td>[6.8363827e-18]</td>
      <td>[0.15793478]</td>
      <td>[0.040057223]</td>
      <td>[0.9999906]</td>
      <td>[0.057762194]</td>
      <td>[0.036209296]</td>
      <td>[1.1137859e-06]</td>
      <td>[0.05882591]</td>
      <td>[0.021252738]</td>
      <td>[2.8522333e-32]</td>
      <td>[2.9057893e-35]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2023-08-28 17:00:02.996</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 31, 14, 18, 29, 22, 19, 13, 23, 36, 16, 24, 17, 31, 35, 24, 26, 33, 37]</td>
      <td>[7.043643e-15]</td>
      <td>[3.8310843e-10]</td>
      <td>[0.010476029]</td>
      <td>[5.539169e-13]</td>
      <td>[4.2660397e-18]</td>
      <td>[1.8002028e-13]</td>
      <td>[3.1393036e-15]</td>
      <td>[0.99999946]</td>
      <td>[5.5198524e-15]</td>
      <td>[0.14957656]</td>
      <td>[3.944928e-07]</td>
      <td>[0.3118902]</td>
      <td>[0.0042013763]</td>
      <td>[0.0]</td>
      <td>[3.3585956e-34]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2023-08-28 17:00:02.996</td>
      <td>[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 22, 20, 20, 31, 30, 33, 32, 23, 24]</td>
      <td>[8.1578975e-05]</td>
      <td>[0.00566244]</td>
      <td>[0.25973403]</td>
      <td>[0.0003614567]</td>
      <td>[2.201271e-13]</td>
      <td>[0.022834523]</td>
      <td>[0.16723375]</td>
      <td>[0.9992838]</td>
      <td>[0.11602864]</td>
      <td>[0.066898234]</td>
      <td>[9.262361e-07]</td>
      <td>[0.035394162]</td>
      <td>[0.22199537]</td>
      <td>[1.975523e-20]</td>
      <td>[9.651789e-15]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

### Parallel Inference Example

This time, let's take the same file and split it into 1,000 **separate** DataFrames, which each indivual row as a single DataFrame.  This is toy data; we're just providing it as an example of how to submit a an inference request for parallel infer.

```python
test_data = pd.read_json("./data/data_25k.df.json")
test_list = []

for index, row in test_data.head(1000).iterrows():
    test_list.append(row.to_frame('text_input').reset_index())
```

Now we'll perform an inference with Parallel Infer through the pipeline.

The pipeline `parallel_infer(tensor_list, timeout, num_parallel, retries)` **asynchronous** method performs an inference as defined by the pipeline steps and takes the following arguments:

* **tensor_list** (*REQUIRED List*): The data submitted to the pipeline for inference as a List of the supported data types:
  * [pandas.DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html):  Data submitted as a pandas DataFrame are returned as a pandas DataFrame.  For models that output one column  based on the models outputs.
  * [Apache Arrow](https://arrow.apache.org/) (**Preferred**): Data submitted as an Apache Arrow are returned as an Apache Arrow.
* **timeout** (*OPTIONAL int*): A timeout in seconds before the inference throws an exception.  The default is 15 second per call to accommodate large, complex models.  Note that for a batch inference, this is **per list item** - with 10 inference requests, each would have a default timeout of 15 seconds.
* **num_parallel** (*OPTIONAL int*):  The number of parallel threads used for the submission.  **This should be no more than four times the number of pipeline replicas**.
* **retries** (*OPTIONAL int*):  The number of retries per inference request submitted.

`parallel_infer` is an asynchronous method that returns the Python callback list of tasks. Calling `parallel_infer` should be called with the `await` keyword to retrieve the callback results.

First we'll process the 1,000 rows serially and clock how long this takes.  This may take up to 3-10 minutes depending on the speed of the connection between the client and the Wallaroo instance.

```python
#
# Run the inference sequentially to establish a baseline
#
now = datetime.datetime.now()

results = []
for df in test_list:
    results.append(aloha_pipeline.infer(tensor=df, timeout=10))

total_sequential = datetime.datetime.now() - now

print(f"Elapsed = {total_sequential.total_seconds()} : {len(results)}")
```

    Elapsed = 484.785862 : 1000

Now we'll compare that to using the `parallel_infer` method.  The same data, but now submitted as multiple rows of the list of DataFrames at one time.

```python
timeout_secs=1200
now = datetime.datetime.now()
##########
parallel_results = await aloha_pipeline.parallel_infer(tensor_list=test_list, 
                                                       timeout=timeout_secs, 
                                                       num_parallel=2*REPLICAS, 
                                                       retries=3)
##########
total_parallel = datetime.datetime.now() - now
print(f"Elapsed_in_parallel = {total_parallel.total_seconds()} : {len(parallel_results)}")
```

    Elapsed_in_parallel = 26.939688 : 1000

```python
print(f"Comparison:\nTotal Time Sequentially: {total_sequential.total_seconds()}\nTotal Time Paralleled: {total_parallel.total_seconds()}")
```

    Comparison:
    Total Time Sequentially: 484.785862
    Total Time Paralleled: 26.939688

Depending on the connection and other requirements, the differences in time can be immense.  For a local connection, the time to process the List sequentially took 4 minutes - versus 13 seconds for the `parallel_infer` method.  This is an immense difference.

## Undeploy Pipeline

When finished with our tests, we will undeploy the pipeline so we have the Kubernetes resources back for other tasks.  Note that if the deployment variable is unchanged aloha_pipeline.deploy() will restart the inference engine in the same configuration as before.

```python
aloha_pipeline.undeploy()
```

<table><tr><th>name</th> <td>ejatalohapipeline</td></tr><tr><th>created</th> <td>2023-08-28 16:59:39.670085+00:00</td></tr><tr><th>last_updated</th> <td>2023-08-28 16:59:44.104892+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>7f4c15e6-f6e3-4145-bed7-02ab9018bd67, 71611219-8559-410b-9bae-5f55a2ae245e</td></tr><tr><th>steps</th> <td>ejatalohamodel</td></tr><tr><th>published</th> <td>False</td></tr></table>

