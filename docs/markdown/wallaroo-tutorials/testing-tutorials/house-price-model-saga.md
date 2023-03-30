This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-testing-tutorials/anomaly_detection).

## Anomaly Detection

Wallaroo provides multiple methods of analytical analysis to verify that the data received and generated during an inference is accurate.  This tutorial will demonstrate how to use anomaly detection to track the outputs from a sample model to verify that the model is outputting acceptable results.

Anomaly detection allows organizations to set validation parameters in a pipeline.  A **validation** is added to a pipeline to test data based on an expression, and flag any inferences where the validation failed to the [InferenceResult](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/#inferenceresult-object) object and the pipeline logs.

This tutorial will follow this process in setting up a validation to a pipeline and examining the results:

1. Create a workspace and upload the sample model.
1. Establish a pipeline and add the model as a step.
1. Add a validation to the pipeline.
1. Perform inferences and display anomalies through the `InferenceResult` object and the pipeline log files.

This tutorial provides the following:

* Models:
  * `rf_model.onnx`: The champion model that has been used in this environment for some time.
  * `xgb_model.onnx` and `models/gbr_model.onnx`: Rival models that will be tested against the champion.
* Data:
  * xtest-1.df.json and xtest-1k.df.json:  DataFrame JSON inference inputs with 1 input and 1,000 inputs.
  * xtest-1.arrow and xtest-1k.arrow:  Apache Arrow inference inputs with 1 input and 1,000 inputs.

## Prerequisites

* A deployed Wallaroo instance
* The following Python libraries installed:
  * `os`
  * `json`
  * [`wallaroo`](https://pypi.org/project/wallaroo/): The Wallaroo SDK. Included with the Wallaroo JupyterHub service by default.
  * [`pandas`](https://pypi.org/project/pandas/): Pandas, mainly used for Pandas DataFrame
  * [`polars`](https://pypi.org/project/polars/): Polars for DataFrame with native Apache Arrow support

## Steps

### Import libraries

The first step is to import the libraries needed for this notebook.


```python
import wallaroo
from wallaroo.object import EntityNotFoundError
import os
import json

from IPython.display import display

# used to display dataframe information without truncating
from IPython.display import display
import pandas as pd
pd.set_option('display.max_colwidth', None)

import polars

import os
# For Wallaroo SDK 2023.1
os.environ["ARROW_ENABLED"]="True"
```

### Connect to Wallaroo Instance

The following command will create a connection to the Wallaroo instance and store it in the variable `wl`.


```python
# Login through local Wallaroo instance

wl = wallaroo.Client()

# SSO login through keycloak

# wallarooPrefix = "YOUR PREFIX"
# wallarooSuffix = "YOUR PREFIX"

# wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}.api.{wallarooSuffix}", 
#                     auth_endpoint=f"https://{wallarooPrefix}.keycloak.{wallarooSuffix}", 
#                     auth_type="sso")
```

### Create Workspace

We will create a workspace to manage our pipeline and models.  The following variables will set the name of our sample workspace then set it as the current workspace.


```python
workspace_name = 'houseprice'
main_pipeline_name = 'housepricepipeline'
model_name_control = 'housingcontrol'
model_file_name_control = './models/rf_model.onnx'
model_name_challenger_01 = 'housingchallenger01'
model_name_challenger_02 = 'housingchallenger02'
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
```


```python
workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)
```




    {'name': 'houseprice', 'id': 24, 'archived': False, 'created_by': '56d97480-bb64-4575-acb6-f93d05652e86', 'created_at': '2023-03-29T17:26:54.314916+00:00', 'models': [{'name': 'housingcontrol', 'versions': 7, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 3, 29, 19, 42, 25, 629658, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 3, 29, 17, 26, 56, 389116, tzinfo=tzutc())}, {'name': 'housingchallenger01', 'versions': 5, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 3, 29, 19, 42, 46, 787792, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 3, 29, 17, 45, 39, 264722, tzinfo=tzutc())}, {'name': 'housingchallenger02', 'versions': 5, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 3, 29, 19, 42, 48, 922411, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 3, 29, 17, 45, 40, 666008, tzinfo=tzutc())}], 'pipelines': [{'name': 'housepriceshadowtesting', 'create_time': datetime.datetime(2023, 3, 29, 19, 31, 26, 274598, tzinfo=tzutc()), 'definition': '[]'}, {'name': 'housepricepipeline', 'create_time': datetime.datetime(2023, 3, 29, 17, 27, 50, 527879, tzinfo=tzutc()), 'definition': '[]'}, {'name': 'housepriceabtesting', 'create_time': datetime.datetime(2023, 3, 29, 17, 45, 41, 678930, tzinfo=tzutc()), 'definition': '[]'}]}



## Upload The Models

For our example, we will upload three models, all pre-trained to determine housing prices based on various variables.

The assumption is that we have a pipeline deployed that has been determining house prices for some time using 

* `rf_model.onnx`: The champion model that has been used in this environment for some time.
* `xgb_model.onnx` and `models/gbr_model.onnx`: Rival models that will be tested against the champion.

We will upload our control model into our workspace, then deploy a pipeline with the `rf_model.onnx` and perform some sample inferences.


```python
housing_model_control = wl.upload_model(model_name_control, model_file_name_control).configure()
```

### Build the Control Sample Pipeline

This pipeline is made to be an example of an existing situation where a model is deployed and being used for inferences in a production environment.  We'll call it `housepricepipeline`, set `rf_model.onnx` as a pipeline step as set in the variable declarations above, and run a few sample inferences.


```python
mainpipeline = wl.build_pipeline(main_pipeline_name).add_model_step(housing_model_control).deploy()
```

### Testing

We'll use two inferences as a quick sample test - one that has a house that should be determined around $700k, the other with a house determined to be around $1.5 million.


```python
normal_input = pd.DataFrame.from_records({"tensor": [[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]]})
result = mainpipeline.infer(normal_input)
display(result)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <td>2023-03-29 20:23:44.614</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.7]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
large_house_input = pd.DataFrame.from_records({'tensor': [[4.0, 3.0, 3710.0, 20000.0, 2.0, 0.0, 2.0, 5.0, 10.0, 2760.0, 950.0, 47.6696, -122.261, 3970.0, 20000.0, 79.0, 0.0, 0.0]]})
large_house_result = mainpipeline.infer(large_house_input)
display(large_house_result)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <td>2023-03-29 20:23:45.050</td>
      <td>[4.0, 3.0, 3710.0, 20000.0, 2.0, 0.0, 2.0, 5.0, 10.0, 2760.0, 950.0, 47.6696, -122.261, 3970.0, 20000.0, 79.0, 0.0, 0.0]</td>
      <td>[1514079.4]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


As one last sample, we'll run through roughly 1,000 inferences at once and show a few of the results.


```python
large_inference_result = mainpipeline.infer_from_file("./data/xtest-1k.df.json")
display(large_inference_result.head(5))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <td>2023-03-29 20:23:45.725</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.75]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-03-29 20:23:45.725</td>
      <td>[2.0, 2.5, 2170.0, 6361.0, 1.0, 0.0, 2.0, 3.0, 8.0, 2170.0, 0.0, 47.7109, -122.017, 2310.0, 7419.0, 6.0, 0.0, 0.0]</td>
      <td>[615094.56]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-03-29 20:23:45.725</td>
      <td>[3.0, 2.5, 1300.0, 812.0, 2.0, 0.0, 0.0, 3.0, 8.0, 880.0, 420.0, 47.5893, -122.317, 1300.0, 824.0, 6.0, 0.0, 0.0]</td>
      <td>[448627.72]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-03-29 20:23:45.725</td>
      <td>[4.0, 2.5, 2500.0, 8540.0, 2.0, 0.0, 0.0, 3.0, 9.0, 2500.0, 0.0, 47.5759, -121.994, 2560.0, 8475.0, 24.0, 0.0, 0.0]</td>
      <td>[758714.2]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-03-29 20:23:45.725</td>
      <td>[3.0, 1.75, 2200.0, 11520.0, 1.0, 0.0, 0.0, 4.0, 7.0, 2200.0, 0.0, 47.7659, -122.341, 1690.0, 8038.0, 62.0, 0.0, 0.0]</td>
      <td>[513264.7]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


## A/B Testing

Now that we have our main pipeline set, let's experiment with other models using two methods:  A/B Testing, and Shadow Deployment.  We'll use A/B testing first.

A/B Testing takes one champion or control model and pits it against one or more challenger models.  In this case, the inference data is split between the champion and control models based on a ratio we provide.  For our example, we will be using a random split, so there is a random weighted chance whether inference data is submitted to a champion or the challenger models.  Results are shown in the same `out.variable` as for the champion model, and we can determine which model received the input data based on the `out._model_split` column.

Shadow deploy works much the same way, only **all** inference data is submitted to **all** models equally, with only the results of the champion model displayed in the `out.variable` column.  We'll demonstrate that in a later example.

### Define The A/B Testing Pipeline

Here we will configure a pipeline with two models and set the control model with a random split chance of receiving 2/3 of the data.  Because this is a random split, it is possible for one model or the other to receive more inferences than a strict 2:1 ratio, but the more inferences are run, the more likely it is for the proper ratio split.

We'll upload our challenger models, then create the pipeline `housepriceabtesting` for our sample pipeline, then deploy it.


```python
ab_pipeline_name = 'housepriceabtesting'
model_file_name_challenger_01 = './models/xgb_model.onnx'
model_file_name_challenger_02 = './models/gbr_model.onnx'


housing_model_challenger01 = wl.upload_model(model_name_challenger_01, model_file_name_challenger_01).configure()
housing_model_challenger02 = wl.upload_model(model_name_challenger_02, model_file_name_challenger_02).configure()

abpipeline = (wl.build_pipeline(ab_pipeline_name)
            .add_random_split([(2, housing_model_control), (1, housing_model_challenger01 )], "session_id")).deploy()
```

### A/B Testing Single Inference

Now we have our deployment set up let's run a single inference. In the results we will be able to see the inference results as well as which model the inference went to under model_id.  We'll run the inference request 5 times, with the odds are that the challenger model being run at least once.


```python
results = []

for x in range(5):
    result = abpipeline.infer_from_file("data/xtest-1.df.json")
    display(result.loc[:,["out._model_split", "out.variable"]])
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out._model_split</th>
      <th>out.variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{"name":"housingcontrol","version":"e88fae0a-f6c1-4f61-825a-ba43a35e12a7","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}]</td>
      <td>[718013.7]</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out._model_split</th>
      <th>out.variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{"name":"housingchallenger01","version":"419f15fa-4256-48f5-9a2c-9d2393f5f787","sha":"31e92d6ccb27b041a324a7ac22cf95d9d6cc3aa7e8263a229f7c4aec4938657c"}]</td>
      <td>[659806.0]</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out._model_split</th>
      <th>out.variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{"name":"housingchallenger01","version":"419f15fa-4256-48f5-9a2c-9d2393f5f787","sha":"31e92d6ccb27b041a324a7ac22cf95d9d6cc3aa7e8263a229f7c4aec4938657c"}]</td>
      <td>[659806.0]</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out._model_split</th>
      <th>out.variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{"name":"housingcontrol","version":"e88fae0a-f6c1-4f61-825a-ba43a35e12a7","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}]</td>
      <td>[718013.7]</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>out._model_split</th>
      <th>out.variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{"name":"housingcontrol","version":"e88fae0a-f6c1-4f61-825a-ba43a35e12a7","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}]</td>
      <td>[718013.7]</td>
    </tr>
  </tbody>
</table>
</div>


### Run Inference Batch

We will submit 1000 rows of test data through the pipeline, then loop through the responses and display which model each inference was performed in.  The results between the control and challenger should be approximately 2:1.

The sample code will be shown below.  Since this splits a single inference request into 1,000 pieces and submits them serially, this can take a few minutes to run.  We'll just go to the results for our demonstration, but users can use the code below as a sample exercise.


```python
# for index, row in test_data.head(1000).iterrows():
#     responses = responses.append(abpipeline.infer(row.to_frame('tensor').reset_index(drop=True)))

# #now get our responses for each row
# # each r is a dataframe, then get the result from out.split into json and get the model name
# l = [json.loads(row['out._model_split'][0])['name'] for index, row in responses.iterrows()]
# df = pd.DataFrame({'model': l})
# display(df.model.value_counts())
```


```python
# load the inference result data from a/b testing
responses = pd.read_json('./data/abtestingresults.df.json', orient="records")
l = [json.loads(row['out._model_split'][0])['name'] for index, row in responses.iterrows()]
df = pd.DataFrame({'model': l})
display(df.model.value_counts())
```


    housingcontrol         702
    housingchallenger01    298
    Name: model, dtype: int64


### Compare A/B Testing Results

### Test Challenger

Now we have run a large amount of data we can compare the results.

For this experiment we are looking for a significant change average predicted price between the two models.


```python
control_count = df.model.value_counts()['housingcontrol']
challenger_count = df.model.value_counts()['housingchallenger01']

control_sum = 0
challenger_sum = 0

for index, row in responses.iterrows():
    if json.loads(row['out._model_split'][0])['name'] == 'housingcontrol':
        control_sum += row['out.variable'][0]
    else:
        challenger_sum += row['out.variable'][0]

print("control mean price prediction: " + str(control_sum/control_count))
print("challenger mean price prediction: " + str(challenger_sum/challenger_count))
```

    control mean price prediction: 541388.8181339029
    challenger mean price prediction: 539660.27885906


### View A/B Logs

Logs can be viewed with the Pipeline method `logs()`.  For this example, only the first 5 logs will be shown.  For Arrow enabled environments, the model type can be found in the column `out._model_split`.


```python
logs = abpipeline.logs().head(5)

display(logs)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out._model_split</th>
      <th>out.variable</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-03-29 20:10:42.532</td>
      <td>[3.0, 2.0, 2140.0, 4923.0, 1.0, 0.0, 0.0, 4.0, 8.0, 1070.0, 1070.0, 47.6902, -122.339, 1470.0, 4923.0, 86.0, 0.0, 0.0]</td>
      <td>[{"name":"housingchallenger01","version":"b2316d64-c6fb-4a5e-a607-ffd442dfdecb","sha":"31e92d6ccb27b041a324a7ac22cf95d9d6cc3aa7e8263a229f7c4aec4938657c"}]</td>
      <td>[615501.9]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-03-29 20:10:42.935</td>
      <td>[4.0, 3.5, 3590.0, 5334.0, 2.0, 0.0, 2.0, 3.0, 9.0, 3140.0, 450.0, 47.6763, -122.267, 2100.0, 6250.0, 9.0, 0.0, 0.0]</td>
      <td>[{"name":"housingcontrol","version":"87a75b3f-1cbc-46bb-9aca-066c42162bdc","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}]</td>
      <td>[1004846.56]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-03-29 20:10:43.343</td>
      <td>[3.0, 2.0, 1280.0, 960.0, 2.0, 0.0, 0.0, 3.0, 9.0, 1040.0, 240.0, 47.602, -122.311, 1280.0, 1173.0, 0.0, 0.0, 0.0]</td>
      <td>[{"name":"housingcontrol","version":"87a75b3f-1cbc-46bb-9aca-066c42162bdc","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}]</td>
      <td>[684577.25]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-03-29 20:10:43.794</td>
      <td>[4.0, 2.5, 2820.0, 15000.0, 2.0, 0.0, 0.0, 4.0, 9.0, 2820.0, 0.0, 47.7255, -122.101, 2440.0, 15000.0, 29.0, 0.0, 0.0]</td>
      <td>[{"name":"housingcontrol","version":"87a75b3f-1cbc-46bb-9aca-066c42162bdc","sha":"e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6"}]</td>
      <td>[727898.25]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-03-29 20:10:44.182</td>
      <td>[3.0, 2.25, 1790.0, 11393.0, 1.0, 0.0, 0.0, 3.0, 8.0, 1790.0, 0.0, 47.6297, -122.099, 2290.0, 11894.0, 36.0, 0.0, 0.0]</td>
      <td>[{"name":"housingchallenger01","version":"b2316d64-c6fb-4a5e-a607-ffd442dfdecb","sha":"31e92d6ccb27b041a324a7ac22cf95d9d6cc3aa7e8263a229f7c4aec4938657c"}]</td>
      <td>[525746.44]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


### Undeploy The A/B Pipeline

With the example complete, we undeploy the pipeline to return the resources back to the Wallaroo instance.


```python
abpipeline.undeploy()
```




<table><tr><th>name</th> <td>housepriceabtesting</td></tr><tr><th>created</th> <td>2023-03-29 17:45:41.678930+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-29 20:23:49.763417+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>c9caccac-10a0-4863-914b-265ce214f829, c58d8f85-e521-4062-8462-f11b238c06c6, a41c735b-e923-4551-a7ce-1b01d0985930, 8c07776c-cf2c-45ad-9255-2dee47f5894e, 0b7a8792-1d23-4e58-a62b-580f6be991ae, 7efe7092-01dc-4710-a69c-472c6e19fa32, 233e7f44-4b22-49de-8b4c-34980a050ab3, aeaf7cbc-9fae-4476-9497-5d666443d30e, f741c176-a929-4883-a5bf-8b8b29e0f929, 6506d5e6-fc33-4e55-913b-45250a74b601, a83d6964-95a2-4436-a0ec-93f1edcfffa3, 400c4272-9e85-467c-b564-da67637cc4e8, f7a7bc32-b7d6-4d4e-9b4e-222440c36477</td></tr><tr><th>steps</th> <td>housingcontrol</td></tr></table>



## Shadow Deploy

The other method for comparing models is Shadow Deploy.  In Shadow Deploy, the pipeline step is added with the `add_shadow_deploy` method, with the champion model listed first, then an array of challenger models after.  **All** inference data is fed to **all** models, with the champion results displayed in the `out.variable` column, and the shadow results in the format `out_{model name}.variable`.  For example, since we named our challenger models `housingchallenger01` and `housingchallenger02`, the columns `out_housingchallenger01.variable` and `out_housingchallenger02.variable` have the shadow deployed model results.

Here, we'll create a new pipeline called `housepriceshadowtesting`, then add `rf_model.onnx` as our champion, and models `xgb_model.onnx` and `gbr_model.onnx` as the challengers.  We'll deploy the pipeline and prepare it for sample inferences.


```python
shadow_pipeline_name = 'housepriceshadowtesting'
shadow_pipeline = wl.build_pipeline(shadow_pipeline_name).add_shadow_deploy(housing_model_control, [housing_model_challenger01, housing_model_challenger02]).deploy()
```

### Shadow Deploy Sample Inference

We'll now use our same sample data for an inference to our shadow deployed pipeline.


```python
shadow_result = shadow_pipeline.infer_from_file('./data/xtest-1.df.json')

display(shadow_result)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.variable</th>
      <th>check_failures</th>
      <th>out_housingchallenger01.variable</th>
      <th>out_housingchallenger02.variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-03-29 20:27:05.104</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.7]</td>
      <td>0</td>
      <td>[659806.0]</td>
      <td>[704901.9]</td>
    </tr>
  </tbody>
</table>
</div>


### Shadow Deploy Batch Inference

We can also perform batch inferences with shadow deployed pipelines.  Here we'll pass 1,000 inference requests at once, then display the results.


```python
shadow_results = shadow_pipeline.infer_from_file('./data/xtest-1k.arrow')

outputs =  shadow_results.to_pandas()

display(outputs.loc[:,['out.variable','out_housingchallenger01.variable','out_housingchallenger02.variable']])
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>995</th>
      <td>[827411.0]</td>
      <td>[743487.94]</td>
      <td>[787589.25]</td>
    </tr>
    <tr>
      <th>996</th>
      <td>[441960.38]</td>
      <td>[381577.16]</td>
      <td>[411258.3]</td>
    </tr>
    <tr>
      <th>997</th>
      <td>[1060847.5]</td>
      <td>[1520770.0]</td>
      <td>[1491293.8]</td>
    </tr>
    <tr>
      <th>998</th>
      <td>[706823.56]</td>
      <td>[663008.75]</td>
      <td>[594914.2]</td>
    </tr>
    <tr>
      <th>999</th>
      <td>[581003.0]</td>
      <td>[573391.1]</td>
      <td>[596933.5]</td>
    </tr>
  </tbody>
</table>
<p>1000 rows × 3 columns</p>
</div>


### Shadow Deploy Logs

Shadow deployed results are also displayed in the log files.  For Arrow enabled Wallaroo instances, it's just the pipeline `logs` method.  For Arrow disabled environments, the command `logs_shadow_deploy()` displays the shadow deployed model information.


```python
logs = shadow_pipeline.logs()
display(logs.loc[:,['out.variable','out_housingchallenger01.variable','out_housingchallenger02.variable']])
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <td>[937281.75]</td>
      <td>[779282.94]</td>
      <td>[1074469.5]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[450996.25]</td>
      <td>[431373.0]</td>
      <td>[433386.56]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[266405.6]</td>
      <td>[308694.3]</td>
      <td>[277806.5]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[727923.1]</td>
      <td>[612911.25]</td>
      <td>[521321.47]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[480151.5]</td>
      <td>[712661.25]</td>
      <td>[605705.8]</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>[236238.66]</td>
      <td>[158358.28]</td>
      <td>[180559.94]</td>
    </tr>
    <tr>
      <th>96</th>
      <td>[581003.0]</td>
      <td>[651141.75]</td>
      <td>[647863.06]</td>
    </tr>
    <tr>
      <th>97</th>
      <td>[446768.88]</td>
      <td>[451848.2]</td>
      <td>[397378.5]</td>
    </tr>
    <tr>
      <th>98</th>
      <td>[400628.5]</td>
      <td>[392271.75]</td>
      <td>[368913.88]</td>
    </tr>
    <tr>
      <th>99</th>
      <td>[546632.0]</td>
      <td>[510557.38]</td>
      <td>[509433.75]</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 3 columns</p>
</div>


### Undeploy Shadow Pipeline

We can now undeploy the shadow deployed pipeline to return the resources back to the Wallaroo instance.


```python
shadow_pipeline.undeploy()
```




<table><tr><th>name</th> <td>housepriceshadowtesting</td></tr><tr><th>created</th> <td>2023-03-29 19:31:26.274598+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-29 20:26:53.952065+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>31dcab61-21ac-4dc6-8de8-755caf994edf, cbe5c049-6ad8-443c-b905-9420cf1f930e, d46ddeac-9bd0-4020-b111-8320445b5e7d, dec7c584-ce2f-44ed-af37-9cb3785f4929</td></tr><tr><th>steps</th> <td>housingcontrol</td></tr></table>



## Model Swap

Now that we've completed our testing, we can swap our deployed model in the original `housepricingpipeline` with one we feel works better.  This is one with the pipeline `replace_with_model_step` method, where we specify the pipeline step and the model to replace it with.  This pipeline had only one step with the `rf_model.onnx` model, and we'll swap it out with the `gbr_model.onnx` model.

The model swap capability makes updating a pipeline with new models a quick production process.  

We'll do an inference with the current model, then swap out the old for the new, then another inference check.


```python
# inference before model swap

display(mainpipeline.status())
swapinference = mainpipeline.infer_from_file('./data/xtest-1.df.json')
display(swapinference)
```


    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.2.19',
       'name': 'engine-646bff55b5-nj9gj',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'housepricepipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'housingcontrol',
          'version': '87a75b3f-1cbc-46bb-9aca-066c42162bdc',
          'sha': 'e22a0831aafd9917f3cc87a15ed267797f80e2afa12ad7d8810ca58f173b8cc6',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.2.20',
       'name': 'engine-lb-ddd995646-p8b7j',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <td>2023-03-29 20:27:48.941</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[718013.7]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Swap the model with a new one, then redeploy the pipeline
mainpipeline.replace_with_model_step(0, housing_model_challenger01).deploy()
```




<table><tr><th>name</th> <td>housepricepipeline</td></tr><tr><th>created</th> <td>2023-03-29 17:27:50.527879+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-29 20:27:49.364409+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>00ad5db7-68f8-4c9a-b8d3-e11ecac82fe4, c65cbc00-e018-494f-83df-753fcf5fcd6b, 6140d071-f271-444c-a1b9-e2bbe8d00e92, 4859b778-52de-4c94-8230-4eaaf596ce32, 63c53886-9d38-44d0-a4df-a5ced0d2af98, 660af202-adff-4e23-966b-2b20e4138663, b0ffbd14-b58b-43f7-888d-37e0709cd90a, 6c9a282b-85fb-46e8-8e2e-3e804022b243, 2c1fe269-52d5-4288-9941-72cbbc3f8a45, 745e6ec4-9264-4f4b-8780-f9237cdd2569, 950430fd-2319-4e79-b3fe-986c50e219cf, 9a40b7d7-c65a-46fc-9b29-134fd65cf34b, 24de628e-c2b7-4a98-a81b-25c66d69f57a, d759b5b8-ee4d-408c-902a-022d6a773d0a, c3c67191-79b7-4690-a04c-0ee72eef2de5, 0245d76a-0221-496f-8e3f-9bff780a8dbe</td></tr><tr><th>steps</th> <td>housingcontrol</td></tr></table>




```python
# inference after model swap

display(mainpipeline.status())
swapinference = mainpipeline.infer_from_file('./data/xtest-1.df.json')
display(swapinference)
```


    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.2.19',
       'name': 'engine-646bff55b5-nj9gj',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'housepricepipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'housingchallenger01',
          'version': '419f15fa-4256-48f5-9a2c-9d2393f5f787',
          'sha': '31e92d6ccb27b041a324a7ac22cf95d9d6cc3aa7e8263a229f7c4aec4938657c',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.2.20',
       'name': 'engine-lb-ddd995646-p8b7j',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <td>2023-03-29 20:27:58.480</td>
      <td>[4.0, 2.5, 2900.0, 5505.0, 2.0, 0.0, 0.0, 3.0, 8.0, 2900.0, 0.0, 47.6063, -122.02, 2970.0, 5251.0, 12.0, 0.0, 0.0]</td>
      <td>[659806.0]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


### Undeploy Main Pipeline

With the examples and tutorial complete, we will undeploy the main pipeline and return the resources back to the Wallaroo instance.


```python
mainpipeline.undeploy()
```




<table><tr><th>name</th> <td>housepricepipeline</td></tr><tr><th>created</th> <td>2023-03-29 17:27:50.527879+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-29 20:27:49.364409+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>00ad5db7-68f8-4c9a-b8d3-e11ecac82fe4, c65cbc00-e018-494f-83df-753fcf5fcd6b, 6140d071-f271-444c-a1b9-e2bbe8d00e92, 4859b778-52de-4c94-8230-4eaaf596ce32, 63c53886-9d38-44d0-a4df-a5ced0d2af98, 660af202-adff-4e23-966b-2b20e4138663, b0ffbd14-b58b-43f7-888d-37e0709cd90a, 6c9a282b-85fb-46e8-8e2e-3e804022b243, 2c1fe269-52d5-4288-9941-72cbbc3f8a45, 745e6ec4-9264-4f4b-8780-f9237cdd2569, 950430fd-2319-4e79-b3fe-986c50e219cf, 9a40b7d7-c65a-46fc-9b29-134fd65cf34b, 24de628e-c2b7-4a98-a81b-25c66d69f57a, d759b5b8-ee4d-408c-902a-022d6a773d0a, c3c67191-79b7-4690-a04c-0ee72eef2de5, 0245d76a-0221-496f-8e3f-9bff780a8dbe</td></tr><tr><th>steps</th> <td>housingcontrol</td></tr></table>


