# Anomaly Detection

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

This demonstration assumes that a Wallaroo instance has been installed.

## Steps

### Import libraries

The first step is to import the libraries needed for this notebook.

```python
import wallaroo
from wallaroo.object import EntityNotFoundError
import os
import json
```

### Connect to Wallaroo Instance

The following command will create a connection to the Wallaroo instance and store it in the variable `wl`.

```python
wl = wallaroo.Client()
```

### Create Workspace

We will create a workspace to manage our pipeline and models.  The following variables will set the name of our sample workspace then set it as the current workspace.

```python
workspace_name = 'anomalyexample'

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

    {'name': 'anomalyexample', 'id': 2, 'archived': False, 'created_by': '2438350f-e9ac-409d-9b64-10110cba4646', 'created_at': '2022-11-02T19:19:23.639545+00:00', 'models': [], 'pipelines': []}

## Upload The Model

The housing model will be uploaded for use in our pipeline.

```python
housing_model = wl.upload_model("anomaly-housing-model", "./models/housing.zip").configure("tensorflow")
```

### Build the Pipeline and Validation

The pipeline `anomaly-housing-pipeline` will be created and the `anomaly-housing-model` added as a step.  A validation will be created for outputs greater 100.0.  This is interpreted as houses with a value greater than $100 million with the `add_validation` method.  When houses greater than this value are detected, the `InferenceObject` will add it in the `check_failures` array with the message "price too high".

Once complete, the pipeline will be deployed and ready for inferences.

```python
p = wl.build_pipeline('anomaly-housing-pipeline')
p = p.add_model_step(housing_model)
p = p.add_validation('price too high', housing_model.outputs[0][0] < 100.0)
pipeline = p.deploy()

```

    Waiting for deployment - this will take up to 45s ....................... ok

### Testing

Two data points will be fed used for an inference.

The first, labeled `response_normal`, will not trigger an anomaly detection.  The other, labeled `response_trigger`, will trigger the anomaly detection, which will be shown in the InferenceResult `check_failures` array.  

Note that multiple validations can be created to allow for multiple anomalies detected.

```python
test_input = {"dense_16_input":[[0.02675675, 0.0, 0.02677953, 0.0, 0.0010046, 0.00951931, 0.14795322, 0.0027145,  0.03550877, 0.98536841, 0.02988655, 0.04031725, 0.04298041]]}
response_normal = pipeline.infer(test_input)
print("\n")
print(response_normal)
```

    
    
    [InferenceResult({'check_failures': [],
     'elapsed': 12112240,
     'model_name': 'anomaly-housing-model',
     'model_version': 'a3b1c29f-c827-4aad-817d-485de464d59b',
     'original_data': {'dense_16_input': [[0.02675675,
                                           0.0,
                                           0.02677953,
                                           0.0,
                                           0.0010046,
                                           0.00951931,
                                           0.14795322,
                                           0.0027145,
                                           0.03550877,
                                           0.98536841,
                                           0.02988655,
                                           0.04031725,
                                           0.04298041]]},
     'outputs': [{'Float': {'data': [10.349835395812988], 'dim': [1, 1], 'v': 1}}],
     'pipeline_name': 'anomaly-housing-pipeline',
     'shadow_data': {},
     'time': 1667416849684})]

```python
test_input = {"dense_16_input":[[0.02675675, 0.0, 0.02677953, 0.0, 0.0010046, 0.00951931, 0.14795322, 0.0027145,  2, 0.98536841, 0.02988655, 0.04031725, 0.04298041]]}
response_trigger = pipeline.infer(test_input)
print("\n")
print(response_trigger)
```

    
    
    [InferenceResult({'check_failures': [{'False': {'expr': 'anomaly-housing-model.outputs[0][0] < '
                                           '100'}}],
     'elapsed': 12196540,
     'model_name': 'anomaly-housing-model',
     'model_version': 'a3b1c29f-c827-4aad-817d-485de464d59b',
     'original_data': {'dense_16_input': [[0.02675675,
                                           0.0,
                                           0.02677953,
                                           0.0,
                                           0.0010046,
                                           0.00951931,
                                           0.14795322,
                                           0.0027145,
                                           2,
                                           0.98536841,
                                           0.02988655,
                                           0.04031725,
                                           0.04298041]]},
     'outputs': [{'Float': {'data': [350.46990966796875], 'dim': [1, 1], 'v': 1}}],
     'pipeline_name': 'anomaly-housing-pipeline',
     'shadow_data': {},
     'time': 1667416852255})]

### Multiple Tests

With the initial tests run, we can run the inferences against a larger set of data and identify anomalies that appear versus the expected results.  These will be displayed into a graph so we can see where the anomalies occur.  In this case with the house that came in at $350 million - outside of our validation range.

```python
from data import test_data_anomaly
responses_anomaly =[]
for nth in range(400):
    responses_anomaly.extend(pipeline.infer(test_data_anomaly.data[nth]))
```

```python
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```

```python
houseprices = [r.raw['outputs'][0]['Float']['data'][0] for  r in responses_anomaly]
hdf = pd.DataFrame({'sell_price': houseprices})
hdf.hist(column='sell_price', bins=50, grid=False, figsize=(12,8))
plt.axvline(x=100, color='gray', ls='--')
_ = plt.title('Distribution of predicted home sales price')
```

    
![png](wallaroo-anomaly-detection_files/wallaroo-anomaly-detection_17_0.png)
    

### How To Check For Anomalies

There are two primary methods for detecting anomalies with Wallaroo:

* As demonstrated in the example above, from the `InferenceObject` `check_failures` array in the output of each inference to see if anything has happened.
* The other method is to view pipeline's logs and see what anomalies have been detected.

#### View Logs

Anomalies can be displayed through the pipeline `logs()` method.  The parameter `valid=False` will show any validations that were flagged as `False` - in this case, houses that were above 100 million in value.

```python
pipeline.logs(valid=False)
```

<table>
    <tr>
        <th>Timestamp</th>
        <th>Output</th>
        <th>Input</th>
        <th>Anomalies</th>
    </tr>

<tr style="color: red;">
    <td>2022-02-Nov 19:20:52</td>
    <td>[array([[350.46990967]])]</td>
    <td>[[0.02675675, 0.0, 0.02677953, 0.0, 0.0010046, 0.00951931, 0.14795322, 0.0027145, 2, 0.98536841, 0.02988655, 0.04031725, 0.04298041]]</td>
    <td>1</td>
</tr>

<tr style="color: red;">
    <td>2022-02-Nov 19:20:57</td>
    <td>[array([[350.46990967]])]</td>
    <td>[[0.02675675, 0.0, 0.02677953, 0.0, 0.0010046, 0.00951931, 0.14795322, 0.0027145, 2, 0.98536841, 0.02988655, 0.04031725, 0.04298041]]</td>
    <td>1</td>
</tr>

</table>

### Undeploy The Pipeline

With the example complete, we undeploy the pipeline to return the resources back to the Wallaroo instance.

```python
pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s ................................... ok

<table><tr><th>name</th> <td>anomaly-housing-pipeline</td></tr><tr><th>created</th> <td>2022-11-02 19:19:23.977381+00:00</td></tr><tr><th>last_updated</th> <td>2022-11-02 19:19:24.039879+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>steps</th> <td>anomaly-housing-model</td></tr></table>

