This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-testing-tutorials/anomaly_detection).

## House Price Testing Life Cycle Comprehensive Tutorial

This tutorial simulates using Wallaroo for testing a model for inference outliers, potential model drift, and methods to test competitive models against each other and deploy the final version to use.  This demonstrates using assays to detect model or data drift, then Wallaroo Shadow Deploy to compare different models to determine which one is most fit for an organization's needs.  These features allow organizations to monitor model performance and accuracy then swap out models as needed.

* **IMPORTANT NOTE**: This tutorial assumes that the House Price Model Life Cycle Preparation notebook was run before this notebook, and that the workspace, pipeline and models used are the same.  This is **critical** for the section on Assays below.  If the preparation notebook has not been run, skip the Assays section as there will be no historical data for the assays to function on.

This tutorial will demonstrate how to:

1. Select or create a workspace, pipeline and upload the champion model.
1. Add a pipeline step with the champion model, then deploy the pipeline and perform sample inferences.
1. Create an assay and set a baseline, then demonstrate inferences that trigger the assay alert threshold.
1. Swap out the pipeline step with the champion model with a shadow deploy step that compares the champion model against two competitors.
1. Evaluate the results of the champion versus competitor models.
1. Change the pipeline step from a shadow deploy step to an A/B testing step, and show the different results.
1. Change the A/B testing step back to standard pipeline step with the original control model, then demonstrate hot swapping the control model with a challenger model without undeploying the pipeline.
1. Undeploy the pipeline.

This tutorial provides the following:

* Models:
  * `models/rf_model.onnx`: The champion model that has been used in this environment for some time.
  * `models/xgb_model.onnx` and `models/gbr_model.onnx`: Rival models that will be tested against the champion.
* Data:
  * `data/xtest-1.df.json` and `data/xtest-1k.df.json`:  DataFrame JSON inference inputs with 1 input and 1,000 inputs.
  * `data/xtest-1k.arrow`:  Apache Arrow inference inputs with 1 input and 1,000 inputs.

## Prerequisites

* A deployed Wallaroo instance
* The following Python libraries installed:
  * [`wallaroo`](https://pypi.org/project/wallaroo/): The Wallaroo SDK. Included with the Wallaroo JupyterHub service by default.
  * [`pandas`](https://pypi.org/project/pandas/): Pandas, mainly used for Pandas DataFrame

## Initial Steps

### Import libraries

The first step is to import the libraries needed for this notebook.

```python
import wallaroo
from wallaroo.object import EntityNotFoundError

from IPython.display import display

# used to display DataFrame information without truncating
from IPython.display import display
import pandas as pd
pd.set_option('display.max_colwidth', None)

import datetime
import time

# used for unique connection names

import string
import random

suffix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))

import json
```

### Connect to the Wallaroo Instance

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
# Login through local Wallaroo instance

wl = wallaroo.Client()
```

### Create Workspace

We will create a workspace to manage our pipeline and models.  The following variables will set the name of our sample workspace then set it as the current workspace.

Workspace, pipeline, and model names should be unique to each user, so we'll add in a randomly generated suffix so multiple people can run this tutorial in a Wallaroo instance without effecting each other.

```python
workspace_name = f'housepricesagaworkspace'
main_pipeline_name = f'housepricesagapipeline'
model_name_control = f'housepricesagacontrol'
model_file_name_control = './models/rf_model.onnx'
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

def get_pipeline(name, workspace):
    pipelines = workspace.pipelines()
    pipe_filter = filter(lambda x: x.name() == name, pipelines)
    pipes = list(pipe_filter)
    # we can't have a pipe in the workspace with the same name, so it's always the first
    if pipes:
        pipeline = pipes[0]
    else:
        pipeline = wl.build_pipeline(name)
    return pipeline
```

```python
workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)
```

    {'name': 'housepricesagaworkspace', 'id': 16, 'archived': False, 'created_by': '01cd0b0d-0ffb-4a25-a463-82307aca3a61', 'created_at': '2023-06-26T15:05:01.05586+00:00', 'models': [{'name': 'housingchallenger01', 'versions': 4, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 6, 27, 16, 18, 30, 44352, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 6, 26, 16, 43, 10, 606597, tzinfo=tzutc())}, {'name': 'housingchallenger02', 'versions': 4, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 6, 27, 16, 18, 31, 342081, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 6, 26, 16, 43, 12, 954496, tzinfo=tzutc())}, {'name': 'housepricesagacontrol', 'versions': 3, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 6, 27, 17, 14, 45, 609782, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 6, 26, 15, 5, 2, 929977, tzinfo=tzutc())}], 'pipelines': [{'name': 'housepricesagapipeline', 'create_time': datetime.datetime(2023, 6, 26, 15, 5, 4, 424277, tzinfo=tzutc()), 'definition': '[]'}]}

### Upload The Champion Model

For our example, we will upload the champion model that has been trained to derive house prices from a variety of inputs.  The model file is `rf_model.onnx`, and is uploaded with the name `housingcontrol`.

```python
housing_model_control = wl.upload_model(model_name_control, model_file_name_control).configure()
```

## Standard Pipeline Steps

### Build the Pipeline

This pipeline is made to be an example of an existing situation where a model is deployed and being used for inferences in a production environment.  We'll call it `housepricepipeline`, set `housingcontrol` as a pipeline step, then run a few sample inferences.

This pipeline will be a simple one - just a single pipeline step.

```python
mainpipeline = get_pipeline(main_pipeline_name, workspace)

# clearing from previous runs and verifying it is undeployed
mainpipeline.clear()
mainpipeline.undeploy()
mainpipeline.add_model_step(housing_model_control).deploy()
```

<table><tr><th>name</th> <td>housepricesagapipeline</td></tr><tr><th>created</th> <td>2023-06-26 15:05:04.424277+00:00</td></tr><tr><th>last_updated</th> <td>2023-06-27 17:35:45.828428+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>9c9965dd-ad77-43c5-8cbe-ad5beeb1e67d, 29788ad8-e2d3-41c7-a79f-8a5942aee54b, 4041a2e1-5167-4d6a-b2c2-5d516788e904, 27571596-0afe-4ff3-864f-0588636ea4d7, 359d5d98-5d17-48e8-ac0c-5e6c8811ee25, 18a51d8b-ef00-4ae5-8106-7c2359ba0fb6, db66b362-0206-4fb2-bb44-24084bf04f75, 395f545f-ce73-4179-97d8-b392e2dad98b, bbd378e4-3fa2-4485-bdbd-17d9ce667980, a5509f97-29c5-4cce-81de-b930b2a6efb8, 02f5fb8d-f859-4e6c-bcdf-272d8c352202, a6b96cab-dc3f-4ddd-8fd0-ee4764faf4aa, a6fb4175-3409-440a-b93b-54c71d79b4b5, fdcc1d34-340a-499b-83ec-8cda4b897507, c927541d-36e9-4482-8c6e-7e6c00a9e6b7, 1937716e-9afa-4abd-9fe4-e390f73ce78e, f42f8f24-5623-4884-b5b6-737e2f8cae67, 6cf7fe20-d388-4098-9d7d-442f105bc558, dcb2d59b-af10-4c32-913c-d1daa7e24806, 69421415-b0b1-4a2a-9570-8ee998743780, b4bea5dd-5d42-4cec-a275-fa601ed29f03, 62b8a509-0633-4e74-9d47-dacea4d0dd56, 40b7985c-ddcb-40e7-9fff-dc4b9f818359, ad1f34ff-2f9f-4cc2-ba34-161df5934872, d5457fed-9238-4ac0-abe7-df5d734c09d0, c47f2b68-e8aa-4555-b76f-c7ba078ea17c, 287adf19-8dae-4478-b05e-80f01b26f13f, b0ceb741-c16a-4936-888e-4b5fbf9f06e6, 7821248a-6ecd-4786-b8d5-9bce47af2fba, 27a7ba4a-7a19-454b-be3e-229143285ed9, d5596c4e-1ee2-4a08-adde-bee3861f850a, a8571b4d-e27e-416b-b39e-f3a0902cf588, 18b0613c-20ba-4a4d-b039-d5c382eaba33, bc5c5038-0e09-40b8-93a1-b667dfa740b7, 6f2d6bc1-9c20-4ca7-980a-653466445180, e3d61e3f-86e4-4be3-a80c-42d477310b2b, f55f4f08-f8dd-4f56-940a-d77b757e658f, bad95bd1-d713-4e94-9228-a0796cb820fe, 2e60abd3-101d-4542-a672-402f81db8019, c29a8bb4-9c08-4546-aeb9-fc1ddd777cb1, 3bd2487e-2566-4cca-ac58-6b58a04c3e2f, c5adc9a7-0b09-47f4-8efd-1b1ab329c162, eed23a22-07f4-4520-9224-edaf557af7cd, f707f87e-3984-4397-84bb-ca232945f5cc, f1b25a3a-9634-4cf5-92de-971e6179ab64, e3758f4e-0f81-4ae6-a3b6-74e1ca3aa875, 06946d82-0b3d-453c-9629-54b13f2de16e, 899060e6-3b81-4b60-a51c-f346a230ff54, d037c8f1-f4b3-4d9e-bc36-60953873c514, 73f248a0-5832-4f3f-b458-f323379197b2, 0da65f0d-aff2-494e-90dd-a9bddb1c6ae8, 1fbe01b0-df5b-4ad8-b2e6-aa5d4d77c0d1, d61e9a26-c61f-471c-af12-c5a34d2fde5f, 32fe5243-860d-4444-af10-d0cde59870bd, 129ecfed-e90d-4b6f-8b2f-46eebd6c969c, 5af39d76-7b5c-48a0-b726-c0e42dbfbb21, 1fe52429-31a3-4c45-84bd-71f321904a6b, b0691409-cc58-4c88-a74a-a45dfd3a5cc1, 0e6ba1c4-2659-4f1f-9819-a310b5bf95e5, 0b03bc24-8211-40d2-82c4-e673868442f3, 7c617ee2-a6c2-4802-9294-2bc273378ff6, baaf3c19-ba5b-4bb6-ba84-caf868487a3f, 105a3f49-c28d-4b74-91cc-466e612499e3, 612a7ba3-6e1e-44ab-950a-c141fbc085a4, a79c8b7e-aeba-4bf8-a3e8-cb8fcaf3c52b, 50da7e9a-8b12-48a0-9b67-0d2ae17d4ff6, 02198c94-1696-4c4d-9990-dc0cb7933cb5, 8b629783-1806-4533-905f-8c769d55fb5d, 8f954ed9-3ca7-4462-974d-c3615e443ae3, e1b6e432-12b9-4c66-b506-a0eb1c6b8f1c, 5d002762-c8f8-4346-96ba-d713fb1d9a9e, c95f5e2d-9688-4415-835c-d7a2ced163bb, c46f49fe-1528-4f5f-ad20-cbeaac17912b, 61729940-d646-454f-a588-b99d77c21d0a, 1cd48e6e-26b7-4564-8784-0847d0357e9c, 22b64fc7-3e9c-47d9-a806-696a81aa3c2d, 5948e151-52f1-445d-ada3-5471b0b315d7, 7c1238e7-e225-4ff9-8c35-55f912e7e2c0, 893e4017-0e43-4fc3-83f1-dbdd81d69344, 52bc5dc5-eb1e-43f5-9ebe-b8088a89c85a, 791abff1-e914-4766-a65d-c33a83753679, df09da4a-6666-4bf8-ba80-0a0a95ab2774, 51dda3a1-d18c-43fa-84c2-013d6f409e7e, cbf4c9de-495d-42d4-8add-34f3c3c47e5f, 43ad44c5-7f1d-4a91-a800-ab636e63d7cf, ad79a8b4-043e-42d8-bb26-b9aea5918ea4, a33fbdec-95eb-4612-978c-6fc4c5f9b601, bc6907cb-9b4c-4560-b46e-453e145f4d75, f1d39372-4599-4d94-a6fb-770a39ff2b72, 203c4327-f3f8-4a75-a879-ba54e60d9304, 6c5c6f69-c6b5-40f9-a106-ac83e9da69d6, bfd96b0b-ea1a-4328-b0b7-609efbf354de, 93d1c3ca-a433-45cd-b5f6-6c00d76e8c0f, 8539bfb8-4c87-451b-b97e-428e2288b2b8, 80a966e2-eb9c-4817-b9d7-173ae7f7de6e, 90b6a253-f302-4c57-bb60-9145d1db5dc6, 28f91c04-68f7-452e-8325-8e62560fbce8, f7bedd4b-e747-427e-9931-0486ca088ba4, 41e71914-7592-49a8-b829-a083312164a0, aa20670c-12f3-471c-bf24-b8b98dbb7ca3, 3e1fd6d7-1b96-45d9-86bc-db7e108700c1, fa6b7721-f09e-4c58-a91b-e8491f3fc3b5, c953956e-e186-47f0-8850-fddb42cac72b, b2e9c65c-4776-44ee-8876-e4167a706ead, af6fc39c-45e7-4024-98f4-122ac14f87ca, cc6fa26e-dfb4-4772-9b10-289fbbaf9380, d8b541ab-0028-4e56-bcdd-11126627ed97, 4b464bec-982e-4a93-b939-cc2b87597259, c8d46e93-93b2-413a-b27f-0a434a428bf8, a89e2c92-dd57-4226-bc58-14a93075ac87, 1f1fdf43-d4d7-450a-a091-ccc8b21b8e64, 6789583a-c053-40c7-8454-4bf7e0c6e490, d6374228-fce9-4ab6-8db9-d5adc33d9137, 4227f0b3-197c-41b3-8c38-d220e0810187, c8b9e801-eaeb-4058-82e3-07480e229e02, 61a676c8-1982-4919-bbc4-ba40e0628b0a, 7acf71e8-b5b6-440a-aba2-b3bb201d192e, 872211ec-b33e-4a70-8ba3-dc1fd6f79e22, a11e3524-9c04-4a17-9eff-0870a7b920c2, 3be8928e-a81f-4475-8721-071109719c74, f824e785-a528-43e9-b70e-1af651037d59, 84b7d329-4020-45fb-b0d7-3b47308c608d, 1f18c3ae-fdd7-47d1-8376-d72bc57f318f, 8b680c24-9be3-42dc-b9a9-1b3f8e7bceb5</td></tr><tr><th>steps</th> <td>housepricesagacontrol</td></tr></table>

### Testing

We'll use two inferences as a quick sample test - one that has a house that should be determined around $700k, the other with a house determined to be around $1.5 million.  We'll also save the start and end periods for these events to for later log functionality.

```python
normal_input = pd.DataFrame.from_records({"tensor": [[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]]})
result = mainpipeline.infer(normal_input)
display(result)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.variable</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-06-27 17:36:36.387</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.7]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

```python
large_house_input = pd.DataFrame.from_records({'tensor': [[4.0, 3.0, 3710.0, 20000.0, 2.0, 0.0, 2.0, 5.0, 10.0, 2760.0, 950.0, 47.6696, -122.261, 3970.0, 20000.0, 79.0, 0.0, 0.0]]})
large_house_result = mainpipeline.infer(large_house_input)
display(large_house_result)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.variable</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-06-27 17:36:36.789</td>
      <td>[4.0, 3.0, 3710.0, 20000.0, 2.0, 0.0, 2.0, 5.0, 10.0, 2760.0, 950.0, 47.6696, -122.261, 3970.0, 20000.0, 79.0, 0.0, 0.0]</td>
      <td>[1514079.4]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

As one last sample, we'll run through roughly 1,000 inferences at once and show a few of the results.  For this example we'll use an Apache Arrow table, which has a smaller file size compared to uploading a pandas DataFrame JSON file.  The inference result is returned as an arrow table, which we'll convert into a pandas DataFrame to display the first 20 results.

```python
time.sleep(5)
control_model_start = datetime.datetime.now()
batch_inferences = mainpipeline.infer_from_file('./data/xtest-1k.arrow')

large_inference_result =  batch_inferences.to_pandas()
display(large_inference_result.head(20))
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.variable</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-06-27 17:36:42.415</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.75]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-06-27 17:36:42.415</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[615094.56]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-06-27 17:36:42.415</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
      <td>[448627.72]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-06-27 17:36:42.415</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[758714.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-06-27 17:36:42.415</td>
      <td>[3.0, 1.75, 2200.0, 11520.0, 1.0, 0.0, 0.0, 4.0, 7.0, 2200.0, 0.0, 47.7659, -122.341, 1690.0, 8038.0, 62.0, 0.0, 0.0]</td>
      <td>[513264.7]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2023-06-27 17:36:42.415</td>
      <td>[3.0, 2.0, 2140.0, 4923.0, 1.0, 0.0, 0.0, 4.0, 8.0, 1070.0, 1070.0, 47.6902, -122.339, 1470.0, 4923.0, 86.0, 0.0, 0.0]</td>
      <td>[668288.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2023-06-27 17:36:42.415</td>
      <td>[4.0, 3.5, 3590.0, 5334.0, 2.0, 0.0, 2.0, 3.0, 9.0, 3140.0, 450.0, 47.6763, -122.267, 2100.0, 6250.0, 9.0, 0.0, 0.0]</td>
      <td>[1004846.5]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2023-06-27 17:36:42.415</td>
      <td>[3.0, 2.0, 1280.0, 960.0, 2.0, 0.0, 0.0, 3.0, 9.0, 1040.0, 240.0, 47.602, -122.311, 1280.0, 1173.0, 0.0, 0.0, 0.0]</td>
      <td>[684577.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2023-06-27 17:36:42.415</td>
      <td>[4.0, 2.5, 2820.0, 15000.0, 2.0, 0.0, 0.0, 4.0, 9.0, 2820.0, 0.0, 47.7255, -122.101, 2440.0, 15000.0, 29.0, 0.0, 0.0]</td>
      <td>[727898.1]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2023-06-27 17:36:42.415</td>
      <td>[3.0, 2.25, 1790.0, 11393.0, 1.0, 0.0, 0.0, 3.0, 8.0, 1790.0, 0.0, 47.6297, -122.099, 2290.0, 11894.0, 36.0, 0.0, 0.0]</td>
      <td>[559631.1]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2023-06-27 17:36:42.415</td>
      <td>[3.0, 1.5, 1010.0, 7683.0, 1.5, 0.0, 0.0, 5.0, 7.0, 1010.0, 0.0, 47.72, -122.318, 1550.0, 7271.0, 61.0, 0.0, 0.0]</td>
      <td>[340764.53]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2023-06-27 17:36:42.415</td>
      <td>[3.0, 2.0, 1270.0, 1323.0, 3.0, 0.0, 0.0, 3.0, 8.0, 1270.0, 0.0, 47.6934, -122.342, 1330.0, 1323.0, 8.0, 0.0, 0.0]</td>
      <td>[442168.06]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2023-06-27 17:36:42.415</td>
      <td>[4.0, 1.75, 2070.0, 9120.0, 1.0, 0.0, 0.0, 4.0, 7.0, 1250.0, 820.0, 47.6045, -122.123, 1650.0, 8400.0, 57.0, 0.0, 0.0]</td>
      <td>[630865.6]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2023-06-27 17:36:42.415</td>
      <td>[4.0, 1.0, 1620.0, 4080.0, 1.5, 0.0, 0.0, 3.0, 7.0, 1620.0, 0.0, 47.6696, -122.324, 1760.0, 4080.0, 91.0, 0.0, 0.0]</td>
      <td>[559631.1]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2023-06-27 17:36:42.415</td>
      <td>[4.0, 3.25, 3990.0, 9786.0, 2.0, 0.0, 0.0, 3.0, 9.0, 3990.0, 0.0, 47.6784, -122.026, 3920.0, 8200.0, 10.0, 0.0, 0.0]</td>
      <td>[909441.1]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2023-06-27 17:36:42.415</td>
      <td>[4.0, 2.0, 1780.0, 19843.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1780.0, 0.0, 47.4414, -122.154, 2210.0, 13500.0, 52.0, 0.0, 0.0]</td>
      <td>[313096.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2023-06-27 17:36:42.415</td>
      <td>[4.0, 2.5, 2130.0, 6003.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2130.0, 0.0, 47.4518, -122.12, 1940.0, 4529.0, 11.0, 0.0, 0.0]</td>
      <td>[404040.8]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2023-06-27 17:36:42.415</td>
      <td>[3.0, 1.75, 1660.0, 10440.0, 1.0, 0.0, 0.0, 3.0, 7.0, 1040.0, 620.0, 47.4448, -121.77, 1240.0, 10380.0, 36.0, 0.0, 0.0]</td>
      <td>[292859.5]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2023-06-27 17:36:42.415</td>
      <td>[3.0, 2.5, 2110.0, 4118.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2110.0, 0.0, 47.3878, -122.153, 2110.0, 4044.0, 25.0, 0.0, 0.0]</td>
      <td>[338357.88]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2023-06-27 17:36:42.415</td>
      <td>[4.0, 2.25, 2200.0, 11250.0, 1.5, 0.0, 0.0, 5.0, 7.0, 1300.0, 900.0, 47.6845, -122.201, 2320.0, 10814.0, 94.0, 0.0, 0.0]</td>
      <td>[682284.6]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

### Graph of Prices

Here's a distribution plot of the inferences to view the values, with the X axis being the house price in millions, and the Y axis the number of houses fitting in a bin grouping.  The majority of houses are in the \$250,000 to \$500,000 range, with some outliers in the far end.

```python
import matplotlib.pyplot as plt
houseprices = pd.DataFrame({'sell_price': large_inference_result['out.variable'].apply(lambda x: x[0])})

houseprices.hist(column='sell_price', bins=75, grid=False, figsize=(12,8))
plt.axvline(x=0, color='gray', ls='--')
_ = plt.title('Distribution of predicted home sales price')
time.sleep(5)
control_model_end = datetime.datetime.now()
```

    
{{<figure src="/images/2023.2.1/wallaroo-tutorials/testing-tutorials/house-price-model-saga_files/house-price-model-saga_19_0.png" width="800" label="png">}}
    

### Pipeline Logs

Pipeline logs with standard pipeline steps are retrieved either with:

* Pipeline `logs` which returns either a pandas DataFrame or Apache Arrow table.
* Pipeline `export_logs` which saves the logs either a pandas DataFrame JSON file or Apache Arrow table.

For full details, see the Wallaroo Documentation Pipeline Log Management guide.

#### Pipeline Log Methods

The Pipeline `logs` method accepts the following parameters.

| **Parameter** | **Type** | **Description** |
|---|---|---|
| `limit` | **Int** (*Optional*) | Limits how many log records to display.  Defaults to `100`.  If there are more pipeline logs than are being displayed, the **Warning** message `Pipeline log record limit exceeded` will be displayed.  For example, if 100 log files were requested and there are a total of 1,000, the warning message will be displayed. |
| `start_datetimert` and `end_datetime` | **DateTime** (*Optional*) | Limits logs to all logs between the `start_datetime` and `end_datetime` DateTime parameters.  **Both parameters must be provided**. Submitting a `logs()` request with only `start_datetime` or `end_datetime` will generate an exception.<br />If `start_datetime` and `end_datetime` are provided as parameters, then the records are returned in **chronological** order, with the oldest record displayed first. |
| `arrow` | **Boolean** (*Optional*) | Defaults to **False**.  If `arrow` is set to `True`, then the logs are returned as an [Apache Arrow table](https://arrow.apache.org/).  If `arrow=False`, then the logs are returned as a pandas DataFrame. |

The following examples demonstrate displaying the logs, then displaying the logs between the `control_model_start` and `control_model_end` periods, then again retrieved as an Arrow table.

```python
# pipeline log retrieval - reverse chronological order

display(mainpipeline.logs())

# pipeline log retrieval between two dates - chronological order

display(mainpipeline.logs(start_datetime=control_model_start, end_datetime=control_model_end))

# pipeline log retrieval limited to the last 5 an an arrow table

display(mainpipeline.logs(arrow=True))
```

    Warning: There are more logs available. Please set a larger limit or request a file using export_logs.

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.variable</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-06-26 15:10:11.863</td>
      <td>[3.0, 2.5, 5403.0, 24069.0, 2.0, 1.0, 4.0, 4.0, 12.0, 5403.0, 0.0, 47.4169006348, -122.3479995728, 3980.0, 104374.0, 39.0, 0.0, 0.0]</td>
      <td>[1946437.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-06-26 15:10:11.863</td>
      <td>[4.0, 3.5, 4285.0, 9567.0, 2.0, 0.0, 1.0, 5.0, 10.0, 3485.0, 800.0, 47.6433982849, -122.408996582, 2960.0, 6902.0, 68.0, 0.0, 0.0]</td>
      <td>[1886959.4]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-06-26 15:10:11.863</td>
      <td>[4.0, 3.0, 4750.0, 21701.0, 1.5, 0.0, 0.0, 5.0, 11.0, 4750.0, 0.0, 47.645401001, -122.2180023193, 3120.0, 18551.0, 38.0, 0.0, 0.0]</td>
      <td>[2002393.5]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-06-26 15:10:11.863</td>
      <td>[3.0, 2.5, 5403.0, 24069.0, 2.0, 1.0, 4.0, 4.0, 12.0, 5403.0, 0.0, 47.4169006348, -122.3479995728, 3980.0, 104374.0, 39.0, 0.0, 0.0]</td>
      <td>[1946437.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-06-26 15:10:11.863</td>
      <td>[4.0, 4.5, 5770.0, 10050.0, 1.0, 0.0, 3.0, 5.0, 9.0, 3160.0, 2610.0, 47.6769981384, -122.2750015259, 2950.0, 6700.0, 65.0, 0.0, 0.0]</td>
      <td>[1689843.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>2023-06-26 15:10:11.863</td>
      <td>[4.0, 3.5, 4285.0, 9567.0, 2.0, 0.0, 1.0, 5.0, 10.0, 3485.0, 800.0, 47.6433982849, -122.408996582, 2960.0, 6902.0, 68.0, 0.0, 0.0]</td>
      <td>[1886959.4]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>96</th>
      <td>2023-06-26 15:10:11.863</td>
      <td>[4.0, 3.0, 3710.0, 20000.0, 2.0, 0.0, 2.0, 5.0, 10.0, 2760.0, 950.0, 47.6696014404, -122.2610015869, 3970.0, 20000.0, 79.0, 0.0, 0.0]</td>
      <td>[1514079.8]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>97</th>
      <td>2023-06-26 15:10:11.863</td>
      <td>[4.0, 3.5, 4285.0, 9567.0, 2.0, 0.0, 1.0, 5.0, 10.0, 3485.0, 800.0, 47.6433982849, -122.408996582, 2960.0, 6902.0, 68.0, 0.0, 0.0]</td>
      <td>[1886959.4]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>98</th>
      <td>2023-06-26 15:10:11.863</td>
      <td>[5.0, 4.25, 4860.0, 9453.0, 1.5, 0.0, 1.0, 5.0, 10.0, 3100.0, 1760.0, 47.6195983887, -122.2860031128, 3150.0, 8557.0, 109.0, 0.0, 0.0]</td>
      <td>[1910823.8]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>99</th>
      <td>2023-06-26 15:10:11.863</td>
      <td>[4.0, 4.5, 5770.0, 10050.0, 1.0, 0.0, 3.0, 5.0, 9.0, 3160.0, 2610.0, 47.6769981384, -122.2750015259, 2950.0, 6700.0, 65.0, 0.0, 0.0]</td>
      <td>[1689843.2]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 4 columns</p>

    Warning: Pipeline log size limit exceeded. Please request logs using export_logs

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.variable</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-06-27 17:36:42.415</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.75]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-06-27 17:36:42.415</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[615094.56]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-06-27 17:36:42.415</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
      <td>[448627.72]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-06-27 17:36:42.415</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[758714.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-06-27 17:36:42.415</td>
      <td>[3.0, 1.75, 2200.0, 11520.0, 1.0, 0.0, 0.0, 4.0, 7.0, 2200.0, 0.0, 47.7659, -122.341, 1690.0, 8038.0, 62.0, 0.0, 0.0]</td>
      <td>[513264.7]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>504</th>
      <td>2023-06-27 17:36:42.415</td>
      <td>[4.0, 2.5, 2800.0, 246114.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2800.0, 0.0, 47.6586, -121.962, 2750.0, 60351.0, 15.0, 0.0, 0.0]</td>
      <td>[765468.75]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>505</th>
      <td>2023-06-27 17:36:42.415</td>
      <td>[2.0, 1.0, 1120.0, 9912.0, 1.0, 0.0, 0.0, 4.0, 6.0, 1120.0, 0.0, 47.3735, -122.43, 1540.0, 9750.0, 34.0, 0.0, 0.0]</td>
      <td>[309800.75]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>506</th>
      <td>2023-06-27 17:36:42.415</td>
      <td>[3.0, 3.5, 2760.0, 4500.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2120.0, 640.0, 47.6529, -122.372, 1950.0, 6000.0, 10.0, 0.0, 0.0]</td>
      <td>[798188.94]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>507</th>
      <td>2023-06-27 17:36:42.415</td>
      <td>[4.0, 1.75, 2710.0, 11400.0, 1.0, 0.0, 0.0, 4.0, 9.0, 1430.0, 1280.0, 47.561, -122.153, 2640.0, 11000.0, 38.0, 0.0, 0.0]</td>
      <td>[772048.0]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>508</th>
      <td>2023-06-27 17:36:42.415</td>
      <td>[3.0, 2.5, 1700.0, 7496.0, 2.0, 0.0, 0.0, 3.0, 8.0, 1700.0, 0.0, 47.432, -122.189, 2280.0, 7496.0, 20.0, 0.0, 0.0]</td>
      <td>[310992.97]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>509 rows × 4 columns</p>

    Warning: There are more logs available. Please set a larger limit or request a file using export_logs.

    pyarrow.Table
    time: timestamp[ms]
    in.tensor: list<item: double> not null
      child 0, item: double
    out.variable: list<inner: float not null> not null
      child 0, inner: float not null
    check_failures: int8
    ----
    time: [[2023-06-26 15:10:11.863,2023-06-26 15:10:11.863,2023-06-26 15:10:11.863,2023-06-26 15:10:11.863,2023-06-26 15:10:11.863,...,2023-06-26 15:10:11.863,2023-06-26 15:10:11.863,2023-06-26 15:10:11.863,2023-06-26 15:10:11.863,2023-06-26 15:10:11.863]]
    in.tensor: [[[3,2.5,5403,24069,2,...,3980,104374,39,0,0],[4,3.5,4285,9567,2,...,2960,6902,68,0,0],...,[5,4.25,4860,9453,1.5,...,3150,8557,109,0,0],[4,4.5,5770,10050,1,...,2950,6700,65,0,0]]]
    out.variable: [[[1946437.2],[1886959.4],...,[1910823.8],[1689843.2]]]
    check_failures: [[0,0,0,0,0,...,0,0,0,0,0]]

## Anomaly Detection through Validations

Anomaly detection allows organizations to set validation parameters in a pipeline. A validation is added to a pipeline to test data based on an expression, and flag any inferences where the validation failed inference result and the pipeline logs.

Validations are added through the Pipeline `add_validation(name, validation)` command which uses the following parameters:

| Parameter | Type | Description |
|---|---|---|
| name | String (**Required**) | The name of the validation. |
| Validation | [Expression](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/checks/#Expression) (**Required**) | The validation test command in the format `model_name.outputs][field][index] {Operation} {Value}`. |

For this example, we want to detect the outputs of `housing_model_control` and validate that values are less than `1,500,000`.  Any outputs greater than that will trigger a `check_failure` which is shown in the output.

```python
## Add the validation to the pipeline

mainpipeline = mainpipeline.add_validation('price too high', housing_model_control.outputs[0][0] < 1500000.0)

mainpipeline.deploy()
```

<table><tr><th>name</th> <td>housepricesagapipeline</td></tr><tr><th>created</th> <td>2023-06-26 15:05:04.424277+00:00</td></tr><tr><th>last_updated</th> <td>2023-06-27 17:37:04.028058+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>97d6ca02-2edb-4b2e-9a6e-1d413e618c2a, 9c9965dd-ad77-43c5-8cbe-ad5beeb1e67d, 29788ad8-e2d3-41c7-a79f-8a5942aee54b, 4041a2e1-5167-4d6a-b2c2-5d516788e904, 27571596-0afe-4ff3-864f-0588636ea4d7, 359d5d98-5d17-48e8-ac0c-5e6c8811ee25, 18a51d8b-ef00-4ae5-8106-7c2359ba0fb6, db66b362-0206-4fb2-bb44-24084bf04f75, 395f545f-ce73-4179-97d8-b392e2dad98b, bbd378e4-3fa2-4485-bdbd-17d9ce667980, a5509f97-29c5-4cce-81de-b930b2a6efb8, 02f5fb8d-f859-4e6c-bcdf-272d8c352202, a6b96cab-dc3f-4ddd-8fd0-ee4764faf4aa, a6fb4175-3409-440a-b93b-54c71d79b4b5, fdcc1d34-340a-499b-83ec-8cda4b897507, c927541d-36e9-4482-8c6e-7e6c00a9e6b7, 1937716e-9afa-4abd-9fe4-e390f73ce78e, f42f8f24-5623-4884-b5b6-737e2f8cae67, 6cf7fe20-d388-4098-9d7d-442f105bc558, dcb2d59b-af10-4c32-913c-d1daa7e24806, 69421415-b0b1-4a2a-9570-8ee998743780, b4bea5dd-5d42-4cec-a275-fa601ed29f03, 62b8a509-0633-4e74-9d47-dacea4d0dd56, 40b7985c-ddcb-40e7-9fff-dc4b9f818359, ad1f34ff-2f9f-4cc2-ba34-161df5934872, d5457fed-9238-4ac0-abe7-df5d734c09d0, c47f2b68-e8aa-4555-b76f-c7ba078ea17c, 287adf19-8dae-4478-b05e-80f01b26f13f, b0ceb741-c16a-4936-888e-4b5fbf9f06e6, 7821248a-6ecd-4786-b8d5-9bce47af2fba, 27a7ba4a-7a19-454b-be3e-229143285ed9, d5596c4e-1ee2-4a08-adde-bee3861f850a, a8571b4d-e27e-416b-b39e-f3a0902cf588, 18b0613c-20ba-4a4d-b039-d5c382eaba33, bc5c5038-0e09-40b8-93a1-b667dfa740b7, 6f2d6bc1-9c20-4ca7-980a-653466445180, e3d61e3f-86e4-4be3-a80c-42d477310b2b, f55f4f08-f8dd-4f56-940a-d77b757e658f, bad95bd1-d713-4e94-9228-a0796cb820fe, 2e60abd3-101d-4542-a672-402f81db8019, c29a8bb4-9c08-4546-aeb9-fc1ddd777cb1, 3bd2487e-2566-4cca-ac58-6b58a04c3e2f, c5adc9a7-0b09-47f4-8efd-1b1ab329c162, eed23a22-07f4-4520-9224-edaf557af7cd, f707f87e-3984-4397-84bb-ca232945f5cc, f1b25a3a-9634-4cf5-92de-971e6179ab64, e3758f4e-0f81-4ae6-a3b6-74e1ca3aa875, 06946d82-0b3d-453c-9629-54b13f2de16e, 899060e6-3b81-4b60-a51c-f346a230ff54, d037c8f1-f4b3-4d9e-bc36-60953873c514, 73f248a0-5832-4f3f-b458-f323379197b2, 0da65f0d-aff2-494e-90dd-a9bddb1c6ae8, 1fbe01b0-df5b-4ad8-b2e6-aa5d4d77c0d1, d61e9a26-c61f-471c-af12-c5a34d2fde5f, 32fe5243-860d-4444-af10-d0cde59870bd, 129ecfed-e90d-4b6f-8b2f-46eebd6c969c, 5af39d76-7b5c-48a0-b726-c0e42dbfbb21, 1fe52429-31a3-4c45-84bd-71f321904a6b, b0691409-cc58-4c88-a74a-a45dfd3a5cc1, 0e6ba1c4-2659-4f1f-9819-a310b5bf95e5, 0b03bc24-8211-40d2-82c4-e673868442f3, 7c617ee2-a6c2-4802-9294-2bc273378ff6, baaf3c19-ba5b-4bb6-ba84-caf868487a3f, 105a3f49-c28d-4b74-91cc-466e612499e3, 612a7ba3-6e1e-44ab-950a-c141fbc085a4, a79c8b7e-aeba-4bf8-a3e8-cb8fcaf3c52b, 50da7e9a-8b12-48a0-9b67-0d2ae17d4ff6, 02198c94-1696-4c4d-9990-dc0cb7933cb5, 8b629783-1806-4533-905f-8c769d55fb5d, 8f954ed9-3ca7-4462-974d-c3615e443ae3, e1b6e432-12b9-4c66-b506-a0eb1c6b8f1c, 5d002762-c8f8-4346-96ba-d713fb1d9a9e, c95f5e2d-9688-4415-835c-d7a2ced163bb, c46f49fe-1528-4f5f-ad20-cbeaac17912b, 61729940-d646-454f-a588-b99d77c21d0a, 1cd48e6e-26b7-4564-8784-0847d0357e9c, 22b64fc7-3e9c-47d9-a806-696a81aa3c2d, 5948e151-52f1-445d-ada3-5471b0b315d7, 7c1238e7-e225-4ff9-8c35-55f912e7e2c0, 893e4017-0e43-4fc3-83f1-dbdd81d69344, 52bc5dc5-eb1e-43f5-9ebe-b8088a89c85a, 791abff1-e914-4766-a65d-c33a83753679, df09da4a-6666-4bf8-ba80-0a0a95ab2774, 51dda3a1-d18c-43fa-84c2-013d6f409e7e, cbf4c9de-495d-42d4-8add-34f3c3c47e5f, 43ad44c5-7f1d-4a91-a800-ab636e63d7cf, ad79a8b4-043e-42d8-bb26-b9aea5918ea4, a33fbdec-95eb-4612-978c-6fc4c5f9b601, bc6907cb-9b4c-4560-b46e-453e145f4d75, f1d39372-4599-4d94-a6fb-770a39ff2b72, 203c4327-f3f8-4a75-a879-ba54e60d9304, 6c5c6f69-c6b5-40f9-a106-ac83e9da69d6, bfd96b0b-ea1a-4328-b0b7-609efbf354de, 93d1c3ca-a433-45cd-b5f6-6c00d76e8c0f, 8539bfb8-4c87-451b-b97e-428e2288b2b8, 80a966e2-eb9c-4817-b9d7-173ae7f7de6e, 90b6a253-f302-4c57-bb60-9145d1db5dc6, 28f91c04-68f7-452e-8325-8e62560fbce8, f7bedd4b-e747-427e-9931-0486ca088ba4, 41e71914-7592-49a8-b829-a083312164a0, aa20670c-12f3-471c-bf24-b8b98dbb7ca3, 3e1fd6d7-1b96-45d9-86bc-db7e108700c1, fa6b7721-f09e-4c58-a91b-e8491f3fc3b5, c953956e-e186-47f0-8850-fddb42cac72b, b2e9c65c-4776-44ee-8876-e4167a706ead, af6fc39c-45e7-4024-98f4-122ac14f87ca, cc6fa26e-dfb4-4772-9b10-289fbbaf9380, d8b541ab-0028-4e56-bcdd-11126627ed97, 4b464bec-982e-4a93-b939-cc2b87597259, c8d46e93-93b2-413a-b27f-0a434a428bf8, a89e2c92-dd57-4226-bc58-14a93075ac87, 1f1fdf43-d4d7-450a-a091-ccc8b21b8e64, 6789583a-c053-40c7-8454-4bf7e0c6e490, d6374228-fce9-4ab6-8db9-d5adc33d9137, 4227f0b3-197c-41b3-8c38-d220e0810187, c8b9e801-eaeb-4058-82e3-07480e229e02, 61a676c8-1982-4919-bbc4-ba40e0628b0a, 7acf71e8-b5b6-440a-aba2-b3bb201d192e, 872211ec-b33e-4a70-8ba3-dc1fd6f79e22, a11e3524-9c04-4a17-9eff-0870a7b920c2, 3be8928e-a81f-4475-8721-071109719c74, f824e785-a528-43e9-b70e-1af651037d59, 84b7d329-4020-45fb-b0d7-3b47308c608d, 1f18c3ae-fdd7-47d1-8376-d72bc57f318f, 8b680c24-9be3-42dc-b9a9-1b3f8e7bceb5</td></tr><tr><th>steps</th> <td>housepricesagacontrol</td></tr></table>

### Validation Testing

Two validations will be tested:

* One that should return a house value lower than 1,500,000.  The validation will pass so `check_failure` will be 0.
* The other than should return a house value greater than 1,500,000.  The validation will fail, so `check_failure` will be 1.

```python
validation_start = datetime.datetime.now()

# Small value home

normal_input = pd.DataFrame.from_records({
        "tensor": [[
            3.0,
            2.25,
            1620.0,
            997.0,
            2.5,
            0.0,
            0.0,
            3.0,
            8.0,
            1540.0,
            80.0,
            47.5400009155,
            -122.0260009766,
            1620.0,
            1068.0,
            4.0,
            0.0,
            0.0
        ]]
    }
)

small_result = mainpipeline.infer(normal_input)

display(small_result.loc[:,["time", "out.variable", "check_failures"]])
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out.variable</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-06-27 17:37:44.824</td>
      <td>[544392.06]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

```python
# Big value home

big_input = pd.DataFrame.from_records({
        "tensor": [[
            4.0,
            4.5,
            5770.0,
            10050.0,
            1.0,
            0.0,
            3.0,
            5.0,
            9.0,
            3160.0,
            2610.0,
            47.6769981384,
            -122.2750015259,
            2950.0,
            6700.0,
            65.0,
            0.0,
            0.0
        ]]
    }
)

big_result = mainpipeline.infer(big_input)

display(big_result.loc[:,["time", "out.variable", "check_failures"]])
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out.variable</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-06-27 17:37:45.259</td>
      <td>[1689843.1]</td>
      <td>1</td>
    </tr>
  </tbody>
</table>

### Anomaly Results

We'll run through our previous batch, this time showing only those results outside of the validation, and a graph showing where the anomalies are against the other results.

```python
batch_inferences = mainpipeline.infer_from_file('./data/xtest-1k.arrow')

large_inference_result =  batch_inferences.to_pandas()
# Display only the anomalous results

display(large_inference_result[large_inference_result["check_failures"] > 0].loc[:,["time", "out.variable", "check_failures"]])
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out.variable</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>30</th>
      <td>2023-06-27 17:37:45.836</td>
      <td>[1514079.8]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>248</th>
      <td>2023-06-27 17:37:45.836</td>
      <td>[1967344.1]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>255</th>
      <td>2023-06-27 17:37:45.836</td>
      <td>[2002393.5]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>556</th>
      <td>2023-06-27 17:37:45.836</td>
      <td>[1886959.4]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>698</th>
      <td>2023-06-27 17:37:45.836</td>
      <td>[1689843.2]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>711</th>
      <td>2023-06-27 17:37:45.836</td>
      <td>[1946437.2]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>722</th>
      <td>2023-06-27 17:37:45.836</td>
      <td>[2005883.1]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>782</th>
      <td>2023-06-27 17:37:45.836</td>
      <td>[1910823.8]</td>
      <td>1</td>
    </tr>
    <tr>
      <th>965</th>
      <td>2023-06-27 17:37:45.836</td>
      <td>[2016006.0]</td>
      <td>1</td>
    </tr>
  </tbody>
</table>

```python
import matplotlib.pyplot as plt
houseprices = pd.DataFrame({'sell_price': large_inference_result['out.variable'].apply(lambda x: x[0])})

houseprices.hist(column='sell_price', bins=75, grid=False, figsize=(12,8))
plt.axvline(x=1500000, color='red', ls='--')
_ = plt.title('Distribution of predicted home sales price')
```

    
{{<figure src="/images/2023.2.1/wallaroo-tutorials/testing-tutorials/house-price-model-saga_files/house-price-model-saga_30_0.png" width="800" label="png">}}
    

## Assays

Wallaroo assays provide a method for detecting input or model drift.  These can be triggered either when unexpected input is provided for the inference, or when the model needs to be retrained from changing environment conditions.

Wallaroo assays can track either an input field and its index, or an output field and its index.  For full details, see the [Wallaroo Assays Management Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-pipeline-management/wallaroo-pipeline-assays/).

For this example, we will:

* Perform sample inferences based on lower priced houses.
* Create an assay with the baseline set off those lower priced houses.
* Generate inferences spread across all house values, plus specific set of high priced houses to trigger the assay alert.
* Run an interactive assay to show the detection of values outside the established baseline.

### Assay Generation

To start the demonstration, we'll create a baseline of values from houses with small estimated prices and set that as our baseline. Assays are typically run on a 24 hours interval based on a 24 hour window of data, but we'll bypass that by setting our baseline time even shorter.

```python
small_houses_inputs = pd.read_json('./data/smallinputs.df.json')
baseline_size = 500

# Where the baseline data will start
baseline_start = datetime.datetime.now()

# These inputs will be random samples of small priced houses.  Around 30,000 is a good number
small_houses = small_houses_inputs.sample(baseline_size, replace=True).reset_index(drop=True)

# Wait 30 seconds to set this data apart from the rest
time.sleep(30)
mainpipeline.infer(small_houses)

# Set the baseline end

baseline_end = datetime.datetime.now()

# Set the name of the assay
assay_name=f"small houses test {suffix}"

# Now build the assay, based on the start and end of our baseline time, 
# and tracking the output variable index 0
assay_builder = wl.build_assay(assay_name, mainpipeline, model_name_control, baseline_start, baseline_end
                               ).add_iopath("output variable 0")
```

```python
# Perform an interactive baseline run to set out baseline, then show the baseline statistics
baseline_run = assay_builder.build().interactive_baseline_run()
```

```python
baseline_run.baseline_stats()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Baseline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>375</td>
    </tr>
    <tr>
      <th>min</th>
      <td>236238.65625</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1364149.875</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>499089.995083</td>
    </tr>
    <tr>
      <th>median</th>
      <td>442856.3125</td>
    </tr>
    <tr>
      <th>std</th>
      <td>226524.067375</td>
    </tr>
    <tr>
      <th>start</th>
      <td>2023-06-27T17:37:46.241465Z</td>
    </tr>
    <tr>
      <th>end</th>
      <td>2023-06-27T17:38:17.181113Z</td>
    </tr>
  </tbody>
</table>

```python
display(assay_builder.baseline_dataframe().loc[:, ["time", "output_variable_0"]])
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>output_variable_0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1687887497042</td>
      <td>597475.75000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1687887497042</td>
      <td>700271.56250</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1687887497042</td>
      <td>444931.34375</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1687887497042</td>
      <td>236238.65625</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1687887497042</td>
      <td>675545.50000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>495</th>
      <td>1687887497042</td>
      <td>539867.12500</td>
    </tr>
    <tr>
      <th>496</th>
      <td>1687887497042</td>
      <td>441512.56250</td>
    </tr>
    <tr>
      <th>497</th>
      <td>1687887497042</td>
      <td>441512.56250</td>
    </tr>
    <tr>
      <th>498</th>
      <td>1687887497042</td>
      <td>827411.00000</td>
    </tr>
    <tr>
      <th>499</th>
      <td>1687887497042</td>
      <td>538316.37500</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 2 columns</p>

Now we'll perform some inferences with a spread of values, then a larger set with a set of larger house values to trigger our assay alert.

Because our assay windows are 1 minutes, we'll need to stagger our inference values to be set into the proper windows.  This will take about 4 minutes.

```python
# set the number of inferences to use
inference_size = 1000

# Get a spread of house values
regular_houses_inputs = pd.read_json('./data/xtest-1k.df.json', orient="records")

regular_houses = regular_houses_inputs.sample(inference_size, replace=True).reset_index(drop=True)

# And a spread of large house values

big_houses_inputs = pd.read_json('./data/biginputs.df.json', orient="records")
big_houses = big_houses_inputs.sample(inference_size, replace=True).reset_index(drop=True)

# Set the start for our assay window period.  Adjust date for the historical data used
assay_window_start = datetime.datetime.fromisoformat('2023-06-15T00:00:00+00:00')

# Run a set of regular house values, spread across 90 seconds
# Use to generate inferences now if historical data doesn't exit
for x in range(3):
    mainpipeline.infer(regular_houses)
    time.sleep(35)
    mainpipeline.infer(big_houses)
    time.sleep(35)

# End our assay window period
assay_window_end = datetime.datetime.fromisoformat('2023-06-30T00:00:00+00:00')
```

```python
# now set up our interactive assay based on the window set above.

assay_builder = assay_builder.add_run_until(assay_window_end)

# We don't have many records at the moment, so set the width to 1 minute so it'll slice each 
# one minute interval into a window to analyze
assay_builder.window_builder().add_width(minutes=1).add_interval(minutes=1)

# Build the assay and then do an interactive run rather than waiting for the next interval
assay_config = assay_builder.build()
assay_results = assay_config.interactive_run()
```

```python
# Show how many assay windows were analyzed, then show the chart
print(f"Generated {len(assay_results)} analyses")
assay_results.chart_scores()
```

    Generated 3 analyses

    
{{<figure src="/images/2023.2.1/wallaroo-tutorials/testing-tutorials/house-price-model-saga_files/house-price-model-saga_40_1.png" width="800" label="png">}}
    

```python
# Display the results as a DataFrame - we're mainly interested in the score and whether the 
# alert threshold was triggered
display(assay_results.to_dataframe().loc[:, ["score", "start", "alert_threshold", "status"]])
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>score</th>
      <th>start</th>
      <th>alert_threshold</th>
      <th>status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.677674</td>
      <td>2023-06-27T17:38:17.181113+00:00</td>
      <td>0.25</td>
      <td>Alert</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.677674</td>
      <td>2023-06-27T17:39:17.181113+00:00</td>
      <td>0.25</td>
      <td>Alert</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.013117</td>
      <td>2023-06-27T17:40:17.181113+00:00</td>
      <td>0.25</td>
      <td>Ok</td>
    </tr>
  </tbody>
</table>

```python
display(assay_name)
```

    'small houses test ztlv'

```python
assay_builder.upload()
```

    6

The assay is now visible through the Wallaroo UI by selecting the workspace, then the pipeline, then **Insights**.

{{<figure src="/images/2023.2.1/housepricesaga-sample-assay.png" width="800" label="Sample assay in the UI">}}

## Shadow Deploy

Let's assume that after analyzing the assay information we want to test two challenger models to our control.  We do that with the Shadow Deploy pipeline step.

In Shadow Deploy, the pipeline step is added with the `add_shadow_deploy` method, with the champion model listed first, then an array of challenger models after.  **All** inference data is fed to **all** models, with the champion results displayed in the `out.variable` column, and the shadow results in the format `out_{model name}.variable`.  For example, since we named our challenger models `housingchallenger01` and `housingchallenger02`, the columns `out_housingchallenger01.variable` and `out_housingchallenger02.variable` have the shadow deployed model results.

For this example, we will remove the previous pipeline step, then replace it with a shadow deploy step with `rf_model.onnx` as our champion, and models `xgb_model.onnx` and `gbr_model.onnx` as the challengers.  We'll deploy the pipeline and prepare it for sample inferences.

```python
# Upload the challenger models

model_name_challenger01 = 'housingchallenger01'
model_file_name_challenger01 = './models/xgb_model.onnx'

model_name_challenger02 = 'housingchallenger02'
model_file_name_challenger02 = './models/gbr_model.onnx'

housing_model_challenger01 = wl.upload_model(model_name_challenger01, model_file_name_challenger01).configure()
housing_model_challenger02 = wl.upload_model(model_name_challenger02, model_file_name_challenger02).configure()

```

```python
# Undeploy the pipeline
mainpipeline.clear()
# Add the new shadow deploy step with our challenger models
mainpipeline.add_shadow_deploy(housing_model_control, [housing_model_challenger01, housing_model_challenger02])

# Deploy the pipeline with the new shadow step
mainpipeline.deploy()
```

<table><tr><th>name</th> <td>housepricesagapipeline</td></tr><tr><th>created</th> <td>2023-06-26 15:05:04.424277+00:00</td></tr><tr><th>last_updated</th> <td>2023-06-27 17:42:00.622131+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>cbd3cc55-0344-437e-9916-4795d59bccc3, 97d6ca02-2edb-4b2e-9a6e-1d413e618c2a, 9c9965dd-ad77-43c5-8cbe-ad5beeb1e67d, 29788ad8-e2d3-41c7-a79f-8a5942aee54b, 4041a2e1-5167-4d6a-b2c2-5d516788e904, 27571596-0afe-4ff3-864f-0588636ea4d7, 359d5d98-5d17-48e8-ac0c-5e6c8811ee25, 18a51d8b-ef00-4ae5-8106-7c2359ba0fb6, db66b362-0206-4fb2-bb44-24084bf04f75, 395f545f-ce73-4179-97d8-b392e2dad98b, bbd378e4-3fa2-4485-bdbd-17d9ce667980, a5509f97-29c5-4cce-81de-b930b2a6efb8, 02f5fb8d-f859-4e6c-bcdf-272d8c352202, a6b96cab-dc3f-4ddd-8fd0-ee4764faf4aa, a6fb4175-3409-440a-b93b-54c71d79b4b5, fdcc1d34-340a-499b-83ec-8cda4b897507, c927541d-36e9-4482-8c6e-7e6c00a9e6b7, 1937716e-9afa-4abd-9fe4-e390f73ce78e, f42f8f24-5623-4884-b5b6-737e2f8cae67, 6cf7fe20-d388-4098-9d7d-442f105bc558, dcb2d59b-af10-4c32-913c-d1daa7e24806, 69421415-b0b1-4a2a-9570-8ee998743780, b4bea5dd-5d42-4cec-a275-fa601ed29f03, 62b8a509-0633-4e74-9d47-dacea4d0dd56, 40b7985c-ddcb-40e7-9fff-dc4b9f818359, ad1f34ff-2f9f-4cc2-ba34-161df5934872, d5457fed-9238-4ac0-abe7-df5d734c09d0, c47f2b68-e8aa-4555-b76f-c7ba078ea17c, 287adf19-8dae-4478-b05e-80f01b26f13f, b0ceb741-c16a-4936-888e-4b5fbf9f06e6, 7821248a-6ecd-4786-b8d5-9bce47af2fba, 27a7ba4a-7a19-454b-be3e-229143285ed9, d5596c4e-1ee2-4a08-adde-bee3861f850a, a8571b4d-e27e-416b-b39e-f3a0902cf588, 18b0613c-20ba-4a4d-b039-d5c382eaba33, bc5c5038-0e09-40b8-93a1-b667dfa740b7, 6f2d6bc1-9c20-4ca7-980a-653466445180, e3d61e3f-86e4-4be3-a80c-42d477310b2b, f55f4f08-f8dd-4f56-940a-d77b757e658f, bad95bd1-d713-4e94-9228-a0796cb820fe, 2e60abd3-101d-4542-a672-402f81db8019, c29a8bb4-9c08-4546-aeb9-fc1ddd777cb1, 3bd2487e-2566-4cca-ac58-6b58a04c3e2f, c5adc9a7-0b09-47f4-8efd-1b1ab329c162, eed23a22-07f4-4520-9224-edaf557af7cd, f707f87e-3984-4397-84bb-ca232945f5cc, f1b25a3a-9634-4cf5-92de-971e6179ab64, e3758f4e-0f81-4ae6-a3b6-74e1ca3aa875, 06946d82-0b3d-453c-9629-54b13f2de16e, 899060e6-3b81-4b60-a51c-f346a230ff54, d037c8f1-f4b3-4d9e-bc36-60953873c514, 73f248a0-5832-4f3f-b458-f323379197b2, 0da65f0d-aff2-494e-90dd-a9bddb1c6ae8, 1fbe01b0-df5b-4ad8-b2e6-aa5d4d77c0d1, d61e9a26-c61f-471c-af12-c5a34d2fde5f, 32fe5243-860d-4444-af10-d0cde59870bd, 129ecfed-e90d-4b6f-8b2f-46eebd6c969c, 5af39d76-7b5c-48a0-b726-c0e42dbfbb21, 1fe52429-31a3-4c45-84bd-71f321904a6b, b0691409-cc58-4c88-a74a-a45dfd3a5cc1, 0e6ba1c4-2659-4f1f-9819-a310b5bf95e5, 0b03bc24-8211-40d2-82c4-e673868442f3, 7c617ee2-a6c2-4802-9294-2bc273378ff6, baaf3c19-ba5b-4bb6-ba84-caf868487a3f, 105a3f49-c28d-4b74-91cc-466e612499e3, 612a7ba3-6e1e-44ab-950a-c141fbc085a4, a79c8b7e-aeba-4bf8-a3e8-cb8fcaf3c52b, 50da7e9a-8b12-48a0-9b67-0d2ae17d4ff6, 02198c94-1696-4c4d-9990-dc0cb7933cb5, 8b629783-1806-4533-905f-8c769d55fb5d, 8f954ed9-3ca7-4462-974d-c3615e443ae3, e1b6e432-12b9-4c66-b506-a0eb1c6b8f1c, 5d002762-c8f8-4346-96ba-d713fb1d9a9e, c95f5e2d-9688-4415-835c-d7a2ced163bb, c46f49fe-1528-4f5f-ad20-cbeaac17912b, 61729940-d646-454f-a588-b99d77c21d0a, 1cd48e6e-26b7-4564-8784-0847d0357e9c, 22b64fc7-3e9c-47d9-a806-696a81aa3c2d, 5948e151-52f1-445d-ada3-5471b0b315d7, 7c1238e7-e225-4ff9-8c35-55f912e7e2c0, 893e4017-0e43-4fc3-83f1-dbdd81d69344, 52bc5dc5-eb1e-43f5-9ebe-b8088a89c85a, 791abff1-e914-4766-a65d-c33a83753679, df09da4a-6666-4bf8-ba80-0a0a95ab2774, 51dda3a1-d18c-43fa-84c2-013d6f409e7e, cbf4c9de-495d-42d4-8add-34f3c3c47e5f, 43ad44c5-7f1d-4a91-a800-ab636e63d7cf, ad79a8b4-043e-42d8-bb26-b9aea5918ea4, a33fbdec-95eb-4612-978c-6fc4c5f9b601, bc6907cb-9b4c-4560-b46e-453e145f4d75, f1d39372-4599-4d94-a6fb-770a39ff2b72, 203c4327-f3f8-4a75-a879-ba54e60d9304, 6c5c6f69-c6b5-40f9-a106-ac83e9da69d6, bfd96b0b-ea1a-4328-b0b7-609efbf354de, 93d1c3ca-a433-45cd-b5f6-6c00d76e8c0f, 8539bfb8-4c87-451b-b97e-428e2288b2b8, 80a966e2-eb9c-4817-b9d7-173ae7f7de6e, 90b6a253-f302-4c57-bb60-9145d1db5dc6, 28f91c04-68f7-452e-8325-8e62560fbce8, f7bedd4b-e747-427e-9931-0486ca088ba4, 41e71914-7592-49a8-b829-a083312164a0, aa20670c-12f3-471c-bf24-b8b98dbb7ca3, 3e1fd6d7-1b96-45d9-86bc-db7e108700c1, fa6b7721-f09e-4c58-a91b-e8491f3fc3b5, c953956e-e186-47f0-8850-fddb42cac72b, b2e9c65c-4776-44ee-8876-e4167a706ead, af6fc39c-45e7-4024-98f4-122ac14f87ca, cc6fa26e-dfb4-4772-9b10-289fbbaf9380, d8b541ab-0028-4e56-bcdd-11126627ed97, 4b464bec-982e-4a93-b939-cc2b87597259, c8d46e93-93b2-413a-b27f-0a434a428bf8, a89e2c92-dd57-4226-bc58-14a93075ac87, 1f1fdf43-d4d7-450a-a091-ccc8b21b8e64, 6789583a-c053-40c7-8454-4bf7e0c6e490, d6374228-fce9-4ab6-8db9-d5adc33d9137, 4227f0b3-197c-41b3-8c38-d220e0810187, c8b9e801-eaeb-4058-82e3-07480e229e02, 61a676c8-1982-4919-bbc4-ba40e0628b0a, 7acf71e8-b5b6-440a-aba2-b3bb201d192e, 872211ec-b33e-4a70-8ba3-dc1fd6f79e22, a11e3524-9c04-4a17-9eff-0870a7b920c2, 3be8928e-a81f-4475-8721-071109719c74, f824e785-a528-43e9-b70e-1af651037d59, 84b7d329-4020-45fb-b0d7-3b47308c608d, 1f18c3ae-fdd7-47d1-8376-d72bc57f318f, 8b680c24-9be3-42dc-b9a9-1b3f8e7bceb5</td></tr><tr><th>steps</th> <td>housepricesagacontrol</td></tr></table>

### Shadow Deploy Sample Inference

We'll now use our same sample data for an inference to our shadow deployed pipeline, then display the first 20 results with just the comparative outputs.

```python
shadow_result = mainpipeline.infer_from_file('./data/xtest-1k.arrow')

shadow_outputs =  shadow_result.to_pandas()
display(shadow_outputs.loc[0:20,['out.variable','out_housingchallenger01.variable','out_housingchallenger02.variable']])
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out.variable</th>
      <th>out_housingchallenger01.variable</th>
      <th>out_housingchallenger02.variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[718013.75]</td>
      <td>[659806.0]</td>
      <td>[704901.9]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[615094.56]</td>
      <td>[732883.5]</td>
      <td>[695994.44]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[448627.72]</td>
      <td>[419508.84]</td>
      <td>[416164.8]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[758714.2]</td>
      <td>[634028.8]</td>
      <td>[655277.2]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[513264.7]</td>
      <td>[427209.44]</td>
      <td>[426854.66]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>[668288.0]</td>
      <td>[615501.9]</td>
      <td>[632556.1]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>[1004846.5]</td>
      <td>[1139732.5]</td>
      <td>[1100465.2]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>[684577.2]</td>
      <td>[498328.88]</td>
      <td>[528278.06]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>[727898.1]</td>
      <td>[722664.4]</td>
      <td>[659439.94]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>[559631.1]</td>
      <td>[525746.44]</td>
      <td>[534331.44]</td>
    </tr>
    <tr>
      <th>10</th>
      <td>[340764.53]</td>
      <td>[376337.1]</td>
      <td>[377187.2]</td>
    </tr>
    <tr>
      <th>11</th>
      <td>[442168.06]</td>
      <td>[382053.12]</td>
      <td>[403964.3]</td>
    </tr>
    <tr>
      <th>12</th>
      <td>[630865.6]</td>
      <td>[505608.97]</td>
      <td>[528991.3]</td>
    </tr>
    <tr>
      <th>13</th>
      <td>[559631.1]</td>
      <td>[603260.5]</td>
      <td>[612201.75]</td>
    </tr>
    <tr>
      <th>14</th>
      <td>[909441.1]</td>
      <td>[969585.4]</td>
      <td>[893874.7]</td>
    </tr>
    <tr>
      <th>15</th>
      <td>[313096.0]</td>
      <td>[313633.75]</td>
      <td>[318054.94]</td>
    </tr>
    <tr>
      <th>16</th>
      <td>[404040.8]</td>
      <td>[360413.56]</td>
      <td>[357816.75]</td>
    </tr>
    <tr>
      <th>17</th>
      <td>[292859.5]</td>
      <td>[316674.94]</td>
      <td>[294034.7]</td>
    </tr>
    <tr>
      <th>18</th>
      <td>[338357.88]</td>
      <td>[299907.44]</td>
      <td>[323254.3]</td>
    </tr>
    <tr>
      <th>19</th>
      <td>[682284.6]</td>
      <td>[811896.75]</td>
      <td>[770916.7]</td>
    </tr>
    <tr>
      <th>20</th>
      <td>[583765.94]</td>
      <td>[573618.5]</td>
      <td>[549141.4]</td>
    </tr>
  </tbody>
</table>

## A/B Testing

A/B Testing is another method of comparing and testing models.  Like shadow deploy, multiple models are compared against the champion or control models.  The difference is that instead of submitting the inference data to all models, then tracking the outputs of all of the models, the inference inputs are off of a ratio and other conditions.

For this example, we'll be using a 1:1:1 ratio with a random split between the champion model and the two challenger models.  Each time an inference request is made, there is a random equal chance of any one of them being selected.

When the inference results and log entries are displayed, they include the column `out._model_split` which displays:

| Field | Type | Description |
|---|---|---|
| `name` | String | The model name used for the inference.  |
| `version` | String| The version of the model. |
| `sha` | String | The sha hash of the model version. |

This is used to determine which model was used for the inference request.

```python
# remove the shadow deploy steps
mainpipeline.clear()

# Add the a/b test step to the pipeline
mainpipeline.add_random_split([(1, housing_model_control), (1, housing_model_challenger01), (1, housing_model_challenger02)], "session_id")

mainpipeline.deploy()

# Perform sample inferences of 20 rows and display the results
ab_date_start = datetime.datetime.now()
abtesting_inputs = pd.read_json('./data/xtest-1k.df.json')

df = pd.DataFrame(columns=["model", "value"])

for index, row in abtesting_inputs.sample(20).iterrows():
    result = mainpipeline.infer(row.to_frame('tensor').reset_index())
    value = result.loc[0]["out.variable"]
    model = json.loads(result.loc[0]["out._model_split"][0])['name']
    df = df.append({'model': model, 'value': value}, ignore_index=True)

display(df)
ab_date_end = datetime.datetime.now()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>housingchallenger02</td>
      <td>[431050.7]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>housingchallenger02</td>
      <td>[528651.56]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>housingchallenger01</td>
      <td>[411869.56]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>housingchallenger01</td>
      <td>[377321.88]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>housingchallenger02</td>
      <td>[510736.9]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>housingchallenger02</td>
      <td>[495967.72]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>housepricesagacontrol</td>
      <td>[559631.06]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>housepricesagacontrol</td>
      <td>[384215.0]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>housingchallenger01</td>
      <td>[403779.13]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>housingchallenger02</td>
      <td>[900959.4]</td>
    </tr>
    <tr>
      <th>10</th>
      <td>housepricesagacontrol</td>
      <td>[712309.9]</td>
    </tr>
    <tr>
      <th>11</th>
      <td>housingchallenger02</td>
      <td>[743433.2]</td>
    </tr>
    <tr>
      <th>12</th>
      <td>housingchallenger02</td>
      <td>[398355.63]</td>
    </tr>
    <tr>
      <th>13</th>
      <td>housingchallenger01</td>
      <td>[609696.94]</td>
    </tr>
    <tr>
      <th>14</th>
      <td>housingchallenger01</td>
      <td>[357229.3]</td>
    </tr>
    <tr>
      <th>15</th>
      <td>housingchallenger02</td>
      <td>[413190.1]</td>
    </tr>
    <tr>
      <th>16</th>
      <td>housepricesagacontrol</td>
      <td>[401263.88]</td>
    </tr>
    <tr>
      <th>17</th>
      <td>housingchallenger01</td>
      <td>[513281.28]</td>
    </tr>
    <tr>
      <th>18</th>
      <td>housingchallenger01</td>
      <td>[999334.1]</td>
    </tr>
    <tr>
      <th>19</th>
      <td>housingchallenger02</td>
      <td>[653251.8]</td>
    </tr>
  </tbody>
</table>

## Model Swap

Now that we've completed our testing, we can swap our deployed model in the original `housepricingpipeline` with one we feel works better.

We'll start by removing the A/B Testing pipeline step, then going back to the single pipeline step with the champion model and perform a test inference.

When going from a testing step such as A/B Testing or Shadow Deploy, it is best to undeploy the pipeline, change the steps, then deploy the pipeline.  In a production environment, there should be two pipelines:  One for production, the other for testing models.  Since this example uses one pipeline for simplicity, we will undeploy our main pipeline and reset it back to a one-step pipeline with the current champion model as our pipeline step.

Once done, we'll perform the hot swap with the model `gbr_model.onnx`, which was labeled `housing_model_challenger02` in a previous step.  We'll do an inference with the same data as used with the challenger model.  Note that previously, the inference through the original model returned `[718013.7]`.

```python
mainpipeline.undeploy()

# remove the shadow deploy steps
mainpipeline.clear()

mainpipeline.add_model_step(housing_model_control).deploy()

# Inference test
normal_input = pd.DataFrame.from_records({"tensor": [[4.0,
            2.25,
            2200.0,
            11250.0,
            1.5,
            0.0,
            0.0,
            5.0,
            7.0,
            1300.0,
            900.0,
            47.6845,
            -122.201,
            2320.0,
            10814.0,
            94.0,
            0.0,
            0.0]]})
controlresult = mainpipeline.infer(normal_input)
display(controlresult)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.variable</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-06-27 17:43:50.046</td>
      <td>[4.0, 2.25, 2200.0, 11250.0, 1.5, 0.0, 0.0, 5.0, 7.0, 1300.0, 900.0, 47.6845, -122.201, 2320.0, 10814.0, 94.0, 0.0, 0.0]</td>
      <td>[682284.56]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

Now we'll "hot swap" the control model.  We don't have to deploy the pipeline - we can just swap the model out in that pipeline step and continue with only a millisecond or two lost while the swap was performed.

```python
# Perform hot swap

mainpipeline.replace_with_model_step(0, housing_model_challenger02).deploy()

# inference after model swap
normal_input = pd.DataFrame.from_records({"tensor": [[4.0,
            2.25,
            2200.0,
            11250.0,
            1.5,
            0.0,
            0.0,
            5.0,
            7.0,
            1300.0,
            900.0,
            47.6845,
            -122.201,
            2320.0,
            10814.0,
            94.0,
            0.0,
            0.0]]})
challengerresult = mainpipeline.infer(normal_input)
display(challengerresult)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.variable</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-06-27 17:43:52.679</td>
      <td>[4.0, 2.25, 2200.0, 11250.0, 1.5, 0.0, 0.0, 5.0, 7.0, 1300.0, 900.0, 47.6845, -122.201, 2320.0, 10814.0, 94.0, 0.0, 0.0]</td>
      <td>[770916.6]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

```python
# Display the difference between the two

display(f'Original model output: {controlresult.loc[0]["out.variable"]}')
display(f'Hot swapped model  output: {challengerresult.loc[0]["out.variable"]}')
```

    'Original model output: [682284.56]'

    'Hot swapped model  output: [770916.6]'

### Undeploy Main Pipeline

With the examples and tutorial complete, we will undeploy the main pipeline and return the resources back to the Wallaroo instance.

```python
mainpipeline.undeploy()
```

<table><tr><th>name</th> <td>housepricesagapipeline</td></tr><tr><th>created</th> <td>2023-06-26 15:05:04.424277+00:00</td></tr><tr><th>last_updated</th> <td>2023-06-27 17:53:48.332947+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>48672bbd-5fc9-4162-a720-7d695720df50, 49a6b171-b228-4c88-a435-bb37bec49513, d69778e3-3af6-4246-a255-9cee3edef9b2, 3182053d-041c-4504-9245-5621fb4b2064, a27a3c8d-946f-41ff-9494-70a7f3c7b53e, cbd3cc55-0344-437e-9916-4795d59bccc3, 97d6ca02-2edb-4b2e-9a6e-1d413e618c2a, 9c9965dd-ad77-43c5-8cbe-ad5beeb1e67d, 29788ad8-e2d3-41c7-a79f-8a5942aee54b, 4041a2e1-5167-4d6a-b2c2-5d516788e904, 27571596-0afe-4ff3-864f-0588636ea4d7, 359d5d98-5d17-48e8-ac0c-5e6c8811ee25, 18a51d8b-ef00-4ae5-8106-7c2359ba0fb6, db66b362-0206-4fb2-bb44-24084bf04f75, 395f545f-ce73-4179-97d8-b392e2dad98b, bbd378e4-3fa2-4485-bdbd-17d9ce667980, a5509f97-29c5-4cce-81de-b930b2a6efb8, 02f5fb8d-f859-4e6c-bcdf-272d8c352202, a6b96cab-dc3f-4ddd-8fd0-ee4764faf4aa, a6fb4175-3409-440a-b93b-54c71d79b4b5, fdcc1d34-340a-499b-83ec-8cda4b897507, c927541d-36e9-4482-8c6e-7e6c00a9e6b7, 1937716e-9afa-4abd-9fe4-e390f73ce78e, f42f8f24-5623-4884-b5b6-737e2f8cae67, 6cf7fe20-d388-4098-9d7d-442f105bc558, dcb2d59b-af10-4c32-913c-d1daa7e24806, 69421415-b0b1-4a2a-9570-8ee998743780, b4bea5dd-5d42-4cec-a275-fa601ed29f03, 62b8a509-0633-4e74-9d47-dacea4d0dd56, 40b7985c-ddcb-40e7-9fff-dc4b9f818359, ad1f34ff-2f9f-4cc2-ba34-161df5934872, d5457fed-9238-4ac0-abe7-df5d734c09d0, c47f2b68-e8aa-4555-b76f-c7ba078ea17c, 287adf19-8dae-4478-b05e-80f01b26f13f, b0ceb741-c16a-4936-888e-4b5fbf9f06e6, 7821248a-6ecd-4786-b8d5-9bce47af2fba, 27a7ba4a-7a19-454b-be3e-229143285ed9, d5596c4e-1ee2-4a08-adde-bee3861f850a, a8571b4d-e27e-416b-b39e-f3a0902cf588, 18b0613c-20ba-4a4d-b039-d5c382eaba33, bc5c5038-0e09-40b8-93a1-b667dfa740b7, 6f2d6bc1-9c20-4ca7-980a-653466445180, e3d61e3f-86e4-4be3-a80c-42d477310b2b, f55f4f08-f8dd-4f56-940a-d77b757e658f, bad95bd1-d713-4e94-9228-a0796cb820fe, 2e60abd3-101d-4542-a672-402f81db8019, c29a8bb4-9c08-4546-aeb9-fc1ddd777cb1, 3bd2487e-2566-4cca-ac58-6b58a04c3e2f, c5adc9a7-0b09-47f4-8efd-1b1ab329c162, eed23a22-07f4-4520-9224-edaf557af7cd, f707f87e-3984-4397-84bb-ca232945f5cc, f1b25a3a-9634-4cf5-92de-971e6179ab64, e3758f4e-0f81-4ae6-a3b6-74e1ca3aa875, 06946d82-0b3d-453c-9629-54b13f2de16e, 899060e6-3b81-4b60-a51c-f346a230ff54, d037c8f1-f4b3-4d9e-bc36-60953873c514, 73f248a0-5832-4f3f-b458-f323379197b2, 0da65f0d-aff2-494e-90dd-a9bddb1c6ae8, 1fbe01b0-df5b-4ad8-b2e6-aa5d4d77c0d1, d61e9a26-c61f-471c-af12-c5a34d2fde5f, 32fe5243-860d-4444-af10-d0cde59870bd, 129ecfed-e90d-4b6f-8b2f-46eebd6c969c, 5af39d76-7b5c-48a0-b726-c0e42dbfbb21, 1fe52429-31a3-4c45-84bd-71f321904a6b, b0691409-cc58-4c88-a74a-a45dfd3a5cc1, 0e6ba1c4-2659-4f1f-9819-a310b5bf95e5, 0b03bc24-8211-40d2-82c4-e673868442f3, 7c617ee2-a6c2-4802-9294-2bc273378ff6, baaf3c19-ba5b-4bb6-ba84-caf868487a3f, 105a3f49-c28d-4b74-91cc-466e612499e3, 612a7ba3-6e1e-44ab-950a-c141fbc085a4, a79c8b7e-aeba-4bf8-a3e8-cb8fcaf3c52b, 50da7e9a-8b12-48a0-9b67-0d2ae17d4ff6, 02198c94-1696-4c4d-9990-dc0cb7933cb5, 8b629783-1806-4533-905f-8c769d55fb5d, 8f954ed9-3ca7-4462-974d-c3615e443ae3, e1b6e432-12b9-4c66-b506-a0eb1c6b8f1c, 5d002762-c8f8-4346-96ba-d713fb1d9a9e, c95f5e2d-9688-4415-835c-d7a2ced163bb, c46f49fe-1528-4f5f-ad20-cbeaac17912b, 61729940-d646-454f-a588-b99d77c21d0a, 1cd48e6e-26b7-4564-8784-0847d0357e9c, 22b64fc7-3e9c-47d9-a806-696a81aa3c2d, 5948e151-52f1-445d-ada3-5471b0b315d7, 7c1238e7-e225-4ff9-8c35-55f912e7e2c0, 893e4017-0e43-4fc3-83f1-dbdd81d69344, 52bc5dc5-eb1e-43f5-9ebe-b8088a89c85a, 791abff1-e914-4766-a65d-c33a83753679, df09da4a-6666-4bf8-ba80-0a0a95ab2774, 51dda3a1-d18c-43fa-84c2-013d6f409e7e, cbf4c9de-495d-42d4-8add-34f3c3c47e5f, 43ad44c5-7f1d-4a91-a800-ab636e63d7cf, ad79a8b4-043e-42d8-bb26-b9aea5918ea4, a33fbdec-95eb-4612-978c-6fc4c5f9b601, bc6907cb-9b4c-4560-b46e-453e145f4d75, f1d39372-4599-4d94-a6fb-770a39ff2b72, 203c4327-f3f8-4a75-a879-ba54e60d9304, 6c5c6f69-c6b5-40f9-a106-ac83e9da69d6, bfd96b0b-ea1a-4328-b0b7-609efbf354de, 93d1c3ca-a433-45cd-b5f6-6c00d76e8c0f, 8539bfb8-4c87-451b-b97e-428e2288b2b8, 80a966e2-eb9c-4817-b9d7-173ae7f7de6e, 90b6a253-f302-4c57-bb60-9145d1db5dc6, 28f91c04-68f7-452e-8325-8e62560fbce8, f7bedd4b-e747-427e-9931-0486ca088ba4, 41e71914-7592-49a8-b829-a083312164a0, aa20670c-12f3-471c-bf24-b8b98dbb7ca3, 3e1fd6d7-1b96-45d9-86bc-db7e108700c1, fa6b7721-f09e-4c58-a91b-e8491f3fc3b5, c953956e-e186-47f0-8850-fddb42cac72b, b2e9c65c-4776-44ee-8876-e4167a706ead, af6fc39c-45e7-4024-98f4-122ac14f87ca, cc6fa26e-dfb4-4772-9b10-289fbbaf9380, d8b541ab-0028-4e56-bcdd-11126627ed97, 4b464bec-982e-4a93-b939-cc2b87597259, c8d46e93-93b2-413a-b27f-0a434a428bf8, a89e2c92-dd57-4226-bc58-14a93075ac87, 1f1fdf43-d4d7-450a-a091-ccc8b21b8e64, 6789583a-c053-40c7-8454-4bf7e0c6e490, d6374228-fce9-4ab6-8db9-d5adc33d9137, 4227f0b3-197c-41b3-8c38-d220e0810187, c8b9e801-eaeb-4058-82e3-07480e229e02, 61a676c8-1982-4919-bbc4-ba40e0628b0a, 7acf71e8-b5b6-440a-aba2-b3bb201d192e, 872211ec-b33e-4a70-8ba3-dc1fd6f79e22, a11e3524-9c04-4a17-9eff-0870a7b920c2, 3be8928e-a81f-4475-8721-071109719c74, f824e785-a528-43e9-b70e-1af651037d59, 84b7d329-4020-45fb-b0d7-3b47308c608d, 1f18c3ae-fdd7-47d1-8376-d72bc57f318f, 8b680c24-9be3-42dc-b9a9-1b3f8e7bceb5</td></tr><tr><th>steps</th> <td>housepricesagacontrol</td></tr></table>

```python

```
