This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-testing-tutorials/anomaly_detection).

## Anomaly Detection

Wallaroo provides multiple methods of analytical analysis to verify that the data received and generated during an inference is accurate.  This tutorial will demonstrate how to use anomaly detection to track the outputs from a sample model to verify that the model is outputting acceptable results.

Anomaly detection allows organizations to set validation parameters in a pipeline.  A **validation** is added to a pipeline to test data based on an expression, and flag any inferences where the validation failed inference result and the pipeline logs.

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

import datetime
```

### Connect to the Wallaroo Instance

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  If logging in externally, update the `wallarooPrefix` and `wallarooSuffix` variables with the proper DNS information.  For more information on Wallaroo DNS settings, see the [Wallaroo DNS Integration Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-dns-guide/).

```python
# Login through local Wallaroo instance

wl = wallaroo.Client()
```

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

    {'name': 'anomalytesting', 'id': 34, 'archived': False, 'created_by': '028c8b48-c39b-4578-9110-0b5bdd3824da', 'created_at': '2023-05-18T13:59:33.200997+00:00', 'models': [], 'pipelines': []}

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

### Testing

Two data points will be fed used for an inference.

The first, labeled `response_normal`, will not trigger an anomaly detection.  The other, labeled `response_trigger`, will trigger the anomaly detection, which will be shown in the InferenceResult `check_failures` array.  

Note that multiple validations can be created to allow for multiple anomalies detected.

```python
test_input = pd.DataFrame.from_records({"dense_16_input":{"0":[0.02675675,0.0,0.02677953,0.0,0.0010046,0.00951931,0.14795322,0.0027145,0.03550877,0.98536841,0.02988655,0.04031725,0.04298041]}})

response_normal = pipeline.infer(test_input)
display(response_normal)
```

{{<table "table table-striped table-bordered" >}}
<table>
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
      <td>2023-05-18 14:01:32.084</td>
      <td>[0.02675675, 0.0, 0.02677953, 0.0, 0.0010046, 0.00951931, 0.14795322, 0.0027145, 0.03550877, 0.98536841, 0.02988655, 0.04031725, 0.04298041]</td>
      <td>[10.349834]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}

```python
test_input = pd.DataFrame.from_records({"dense_16_input":{"0":[0.02675675,0.0,0.02677953,0.0,0.0010046,0.00951931,0.14795322,0.0027145,2,0.98536841,0.02988655,0.04031725,0.04298041]}})

response_trigger = pipeline.infer(test_input)
display(response_trigger)
```

{{<table "table table-striped table-bordered" >}}
<table>
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
      <td>2023-05-18 14:01:33.190</td>
      <td>[0.02675675, 0.0, 0.02677953, 0.0, 0.0010046, 0.00951931, 0.14795322, 0.0027145, 2.0, 0.98536841, 0.02988655, 0.04031725, 0.04298041]</td>
      <td>[350.46994]</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
{{</table>}}

### Multiple Tests

With the initial tests run, we can run the inferences against a larger set of data and identify anomalies that appear versus the expected results.  These will be displayed into a graph so we can see where the anomalies occur.  In this case with the house that came in at $350 million - outside of our validation range.

Note:  Because this is splitting one batch inference into 400 separate inferences for this example, it may take longer to run.

```python
validation_start = datetime.datetime.now()
test_data = pd.read_json('./data/test_data_anomaly_df.json', orient="records")
responses_anomaly = pd.DataFrame()
# For the first 400 rows, submit that row as a separate DataFrame
# Add the results to the responses_anomaly dataframe
for index, row in test_data.head(400).iterrows():
    responses_anomaly = responses_anomaly.append(pipeline.infer(row.to_frame('dense_16_input').reset_index()))
validation_end = datetime.datetime.now()
```

```python
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```

```python
houseprices = pd.DataFrame({'sell_price': responses_anomaly['out.dense_19'].apply(lambda x: x[0])})

houseprices.hist(column='sell_price', bins=50, grid=False, figsize=(12,8))
plt.axvline(x=100, color='gray', ls='--')
_ = plt.title('Distribution of predicted home sales price')
```

    
![png](wallaroo-anomaly-detection-reference_files/wallaroo-anomaly-detection-reference_19_0.png)
    

### How To Check For Anomalies

There are two primary methods for detecting anomalies with Wallaroo:

* As demonstrated in the example above, from the `InferenceObject` `check_failures` array in the output of each inference to see if anything has happened.
* The other method is to view pipeline's logs and see what anomalies have been detected.

#### View Logs

Anomalies can be displayed through the pipeline `logs()` method.  The parameter `valid=False` will show any validations that were flagged as `False` - in this case, houses that were above 100 million in value.

```python
logs = pipeline.logs(start_datetime=validation_start, end_datetime=validation_end)
display(logs[logs['check_failures'] > 0])
```

    Warning: Pipeline log size limit exceeded. Please request logs using export_logs

{{<table "table table-striped table-bordered" >}}
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.dense_16_input</th>
      <th>in.index</th>
      <th>out.dense_19</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>260</th>
      <td>2023-05-18 14:03:08.672</td>
      <td>[0.02675675, 0.0, 0.02677953, 0.0, 0.0010046, 0.00951931, 0.14795322, 0.0027145, 2.0, 0.98536841, 0.02988655, 0.04031725, 0.04298041]</td>
      <td>dense_16_input</td>
      <td>[350.46994]</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
{{</table>}}

### Undeploy The Pipeline

With the example complete, we undeploy the pipeline to return the resources back to the Wallaroo instance.

```python
pipeline.undeploy()
```

{{<table "table table-striped table-bordered" >}}
<table><tr><th>name</th> <td>anomalyhousing</td></tr><tr><th>created</th> <td>2023-05-18 13:59:35.467439+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-18 13:59:36.380839+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>1396543e-c7fe-4f7d-9653-600a7c3908e4, cde72224-21f3-486a-a817-05730e6adb7a</td></tr><tr><th>steps</th> <td>anomaly-housing-model</td></tr></table>
{{</table>}}

