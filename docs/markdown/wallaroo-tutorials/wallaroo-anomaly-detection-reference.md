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

* Housing model: `./models/housing.zip` - a pretrained model used to determine standard home prices.
* Test Data:  `./data` - sample data.

## Prerequisites

* A deployed Wallaroo instance
* The following Python libraries installed:
  * `os`
  * `json`
  * [`wallaroo`](https://pypi.org/project/wallaroo/): The Wallaroo SDK. Included with the Wallaroo JupyterHub service by default.
  * [`pandas`](https://pypi.org/project/pandas/): Pandas, mainly used for Pandas DataFrame

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


```python
import os
# Only set the below to make the OS environment ARROW_ENABLED to TRUE.  Otherwise, leave as is.
# os.environ["ARROW_ENABLED"]="True"

if "ARROW_ENABLED" not in os.environ or os.environ["ARROW_ENABLED"].casefold() == "False".casefold():
    arrowEnabled = False
else:
    arrowEnabled = True
print(arrowEnabled)
```

    True


### Create Workspace

We will create a workspace to manage our pipeline and models.  The following variables will set the name of our sample workspace then set it as the current workspace.


```python
workspace_name = 'anomalytesting'

def get_workspace(name):
    workspace = None
    for ws in wl.list_workspaces():
        if ws.name() == name:
            workspace= ws
    if(workspace == None):
        workspace = wl.create_workspace(name)
    return workspace

workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)
```




    {'name': 'anomalyexamples', 'id': 139, 'archived': False, 'created_by': '138bd7e6-4dc8-4dc1-a760-c9e721ef3c37', 'created_at': '2023-03-03T19:11:35.338843+00:00', 'models': [{'name': 'anomaly-housing-model', 'versions': 1, 'owner_id': '""', 'last_update_time': datetime.datetime(2023, 3, 3, 19, 11, 35, 799799, tzinfo=tzutc()), 'created_at': datetime.datetime(2023, 3, 3, 19, 11, 35, 799799, tzinfo=tzutc())}], 'pipelines': [{'name': 'anomalyhousingpipeline', 'create_time': datetime.datetime(2023, 3, 3, 19, 11, 35, 879127, tzinfo=tzutc()), 'definition': '[]'}]}



## Upload The Model

The housing model will be uploaded for use in our pipeline.


```python
housing_model = wl.upload_model("anomaly-housing-model", "./models/housing.zip").configure("tensorflow")
```

### Build the Pipeline and Validation

The pipeline `anomaly-housing-pipeline` will be created and the `anomaly-housing-model` added as a step.  A validation will be created for outputs greater 100.0.  This is interpreted as houses with a value greater than $100 million with the `add_validation` method.  When houses greater than this value are detected, the `InferenceObject` will add it in the `check_failures` array with the message "price too high".

Once complete, the pipeline will be deployed and ready for inferences.



```python
p = wl.build_pipeline('anomalyhousing')
p = p.add_model_step(housing_model)

```


```python
p = p.add_validation('price too high', housing_model.outputs[0][0] < 100.0)
```


```python
pipeline = p.deploy()
```

    Waiting for deployment - this will take up to 45s ................ ok


### Testing

Two data points will be fed used for an inference.

The first, labeled `response_normal`, will not trigger an anomaly detection.  The other, labeled `response_trigger`, will trigger the anomaly detection, which will be shown in the InferenceResult `check_failures` array.  

Note that multiple validations can be created to allow for multiple anomalies detected.


```python
if arrowEnabled is True:
    test_input = pd.DataFrame.from_records({"dense_16_input":{"0":[0.02675675,0.0,0.02677953,0.0,0.0010046,0.00951931,0.14795322,0.0027145,0.03550877,0.98536841,0.02988655,0.04031725,0.04298041]}})
else:
    test_input = {"dense_16_input":[[0.02675675, 0.0, 0.02677953, 0.0, 0.0010046, 0.00951931, 0.14795322, 0.0027145,  0.03550877, 0.98536841, 0.02988655, 0.04031725, 0.04298041]]}

response_normal = pipeline.infer(test_input)
display(response_normal)
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
      <th>in.dense_16_input</th>
      <th>out.dense_19</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-03-03 19:19:44.569</td>
      <td>[0.02675675, 0.0, 0.02677953, 0.0, 0.0010046, 0.00951931, 0.14795322, 0.0027145, 0.03550877, 0.98536841, 0.02988655, 0.04031725, 0.04298041]</td>
      <td>[10.349835]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
if arrowEnabled is True:
    test_input = pd.DataFrame.from_records({"dense_16_input":{"0":[0.02675675,0.0,0.02677953,0.0,0.0010046,0.00951931,0.14795322,0.0027145,2,0.98536841,0.02988655,0.04031725,0.04298041]}})
else:
    test_input = {"dense_16_input":[[0.02675675, 0.0, 0.02677953, 0.0, 0.0010046, 0.00951931, 0.14795322, 0.0027145, 2, 0.98536841, 0.02988655, 0.04031725, 0.04298041]]}

response_trigger = pipeline.infer(test_input)
display(response_trigger)
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
      <th>in.dense_16_input</th>
      <th>out.dense_19</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-03-03 19:19:44.657</td>
      <td>[0.02675675, 0.0, 0.02677953, 0.0, 0.0010046, 0.00951931, 0.14795322, 0.0027145, 2.0, 0.98536841, 0.02988655, 0.04031725, 0.04298041]</td>
      <td>[350.4699]</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


### Multiple Tests

With the initial tests run, we can run the inferences against a larger set of data and identify anomalies that appear versus the expected results.  These will be displayed into a graph so we can see where the anomalies occur.  In this case with the house that came in at $350 million - outside of our validation range.

Note:  Because this is splitting one batch inference into 400 separate inferences for this example, it may take longer to run.


```python
if arrowEnabled is True:
    test_data = pd.read_json('./data/test_data_anomaly_df.json', orient="records")
    responses_anomaly = pd.DataFrame()
    # For the first 400 rows, submit that row as a separate DataFrame
    # Add the results to the responses_anomaly dataframe
    for index, row in test_data.head(400).iterrows():
        responses_anomaly = responses_anomaly.append(pipeline.infer(row.to_frame('dense_16_input').reset_index()))
else:
    responses_anomaly =[]
    from data import test_data_anomaly
    for nth in range(400):
        responses_anomaly.extend(pipeline.infer(test_data_anomaly.data[nth]))
```


```python
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
if arrowEnabled is True:
    houseprices = pd.DataFrame({'sell_price': responses_anomaly['out.dense_19'].apply(lambda x: x[0])})
else:
    houseprices = pd.DataFrame({'sell_price': [r.raw['outputs'][0]['Float']['data'][0] for  r in responses_anomaly]})

houseprices.hist(column='sell_price', bins=50, grid=False, figsize=(12,8))
plt.axvline(x=100, color='gray', ls='--')
_ = plt.title('Distribution of predicted home sales price')
```


    
![png](wallaroo-anomaly-detection-reference_files/wallaroo-anomaly-detection-reference_20_0.png)
    


### How To Check For Anomalies

There are two primary methods for detecting anomalies with Wallaroo:

* As demonstrated in the example above, from the `InferenceObject` `check_failures` array in the output of each inference to see if anything has happened.
* The other method is to view pipeline's logs and see what anomalies have been detected.

#### View Logs

Anomalies can be displayed through the pipeline `logs()` method.  The parameter `valid=False` will show any validations that were flagged as `False` - in this case, houses that were above 100 million in value.


```python
logs = pipeline.logs(valid=False)
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
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>


### Undeploy The Pipeline

With the example complete, we undeploy the pipeline to return the resources back to the Wallaroo instance.


```python
pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s ..................................... ok





<table><tr><th>name</th> <td>anomalyhousingpipeline</td></tr><tr><th>created</th> <td>2023-03-03 19:11:35.879127+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-03 19:19:27.462171+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>649283cf-6a4c-45b5-a6a7-a7c7dada5d84, 58c17376-838f-4121-91c4-4ff6dcb85728, f05819e7-8019-4f5c-ae07-6f74c02450d0, 09f7b6e3-009f-4e0f-b93a-9225975c8fbd</td></tr><tr><th>steps</th> <td>anomaly-housing-model</td></tr></table>


