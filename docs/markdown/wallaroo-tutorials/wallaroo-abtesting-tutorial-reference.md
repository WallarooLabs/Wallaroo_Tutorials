This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-testing-tutorials/abtesting).

## A/B Testing

A/B testing is a method that provides the ability to test out ML models for performance, accuracy or other useful benchmarks.  A/B testing is contrasted with the Wallaroo Shadow Deployment feature.  In both cases, two sets of models are added to a pipeline step:

* Control or Champion model:  The model currently used for inferences.
* Challenger model(s): One or more models that are to be compared to the champion model.

The two feature are different in this way:

| Feature | Description |
|---|---|
| A/B Testing | A subset of inferences are submitted to either the champion ML model or a challenger ML model, and the results of each are output as part of the `InferenceResult` `output` attribute. |
| Shadow Deploy | All inferences are submitted to the champion model and one or more challenger models.  The results for the champion model are output as part of the `InferenceResult` `output` attribute, while the results for the challenger models are output as part of the `InferenceResult` `shadow_data` attribute. |

So to repeat:  A/B testing submits *some* of the inference requests to the champion model, some to the challenger model with one set of outputs, while shadow testing submits *all* of the inference requests to champion and shadow models, and has separate outputs.

This tutorial demonstrate how to conduct A/B testing in Wallaroo.  For this example we will be using an open source model that uses an [Aloha CNN LSTM model](https://www.researchgate.net/publication/348920204_Using_Auxiliary_Inputs_in_Deep_Learning_Models_for_Detecting_DGA-based_Domain_Names) for classifying Domain names as being either legitimate or being used for nefarious purposes such as malware distribution.  

For our example, we will perform the following:

* Create a workspace for our work.
* Upload the Aloha model and a challenger model.
* Create a pipeline that can ingest our submitted data with the champion model and the challenger model set into a A/B step.
* Run a series of sample inferences to display inferences that are run through the champion model versus the challenger model, then determine which is more efficient.

All sample data and models are available through the [Wallaroo Quick Start Guide Samples repository](https://github.com/WallarooLabs/quickstartguide_samples).

## Steps

### Import libraries

Here we will import the libraries needed for this notebook.


```python
import wallaroo
from wallaroo.object import EntityNotFoundError
import os
import pandas as pd
```

### Connect to the Wallaroo Instance

This command will be used to set up a connection to the Wallaroo cluster and allow creating and use of Wallaroo inference engines.


```python
# SSO login through keycloak

wallarooPrefix = "YOUR PREFIX"
wallarooSuffix = "YOUR SUFFIX"


wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}.api.{wallarooSuffix}", 
                auth_endpoint=f"https://{wallarooPrefix}.keycloak.{wallarooSuffix}", 
                auth_type="sso")
```

### Create Workspace

We will create a workspace to manage our pipeline and models.  The following variables will set the name of our sample workspace then set it as the current workspace for all other commands.


```python
workspace_name = 'abtestworkspace'
pipeline_name = 'abtestpipeline'
model_name = 'alohamodel'
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

workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)
```




    {'name': 'abtestworkspace', 'id': 7, 'archived': False, 'created_by': '1439de81-4298-4091-95ce-a7b884ac8766', 'created_at': '2022-12-22T16:04:15.559188+00:00', 'models': [], 'pipelines': []}



### Set Up the Champion and Challenger Models

Now we upload the Champion and Challenger models to our workspace.  We will use two models:

1. `aloha-cnn-lstm` model.
2. `aloha-cnn-lstm-new` (a retrained version)

### Set the Champion Model

We upload our champion model, labeled as `control`.


```python
control =  wl.upload_model("aloha-control",   'models/aloha-cnn-lstm.zip').configure('tensorflow')
```

### Set the Challenger Model

Now we upload the Challenger model, labeled as `challenger`.


```python
challenger = wl.upload_model("aloha-challenger",   'models/aloha-cnn-lstm-new.zip').configure('tensorflow')
```

### Define The Pipeline

Here we will configure a pipeline with two models and set the control model with a random split chance of receiving 2/3 of the data.  Because this is a random split, it is possible for one model or the other to receive more inferences than a strict 2:1 ratio, but the more inferences are run, the more likely it is for the proper ratio split.


```python
pipeline = (wl.build_pipeline("randomsplitpipeline-demo")
            .add_random_split([(2, control), (1, challenger)], "session_id"))
```

### Deploy the pipeline

Now we deploy the pipeline so we can run our inference through it.


```python
experiment_pipeline = pipeline.deploy()
```

# Run a single inference
Now we have our deployment set up let's run a single inference. In the results we will be able to see the inference results as well as which model the inference went to under model_id.  We'll run the inference request 5 times, with the odds are that the challenger model being run at least once.


```python
results = []
results.append(experiment_pipeline.infer_from_file("data/data-1.json"))
results.append(experiment_pipeline.infer_from_file("data/data-1.json"))
results.append(experiment_pipeline.infer_from_file("data/data-1.json"))
results.append(experiment_pipeline.infer_from_file("data/data-1.json"))
results.append(experiment_pipeline.infer_from_file("data/data-1.json"))

for result in results:
    print(result[0].model())
    print(result[0].data())
```

    ('aloha-control', '296bf2c2-1d88-4951-898f-2a2d6955ce63')
    [array([[0.00151959]]), array([[0.98291481]]), array([[0.01209957]]), array([[4.75912966e-05]]), array([[2.02893716e-05]]), array([[0.00031977]]), array([[0.01102928]]), array([[0.99756402]]), array([[0.01034162]]), array([[0.00803896]]), array([[0.01615506]]), array([[0.00623623]]), array([[0.00099858]]), array([[1.79337805e-26]]), array([[1.38899512e-27]]), array([[7.2147857e-313]])]
    ('aloha-control', '296bf2c2-1d88-4951-898f-2a2d6955ce63')
    [array([[0.00151959]]), array([[0.98291481]]), array([[0.01209957]]), array([[4.75912966e-05]]), array([[2.02893716e-05]]), array([[0.00031977]]), array([[0.01102928]]), array([[0.99756402]]), array([[0.01034162]]), array([[0.00803896]]), array([[0.01615506]]), array([[0.00623623]]), array([[0.00099858]]), array([[1.79337805e-26]]), array([[1.38899512e-27]]), array([[7.2147857e-313]])]
    ('aloha-control', '296bf2c2-1d88-4951-898f-2a2d6955ce63')
    [array([[0.00151959]]), array([[0.98291481]]), array([[0.01209957]]), array([[4.75912966e-05]]), array([[2.02893716e-05]]), array([[0.00031977]]), array([[0.01102928]]), array([[0.99756402]]), array([[0.01034162]]), array([[0.00803896]]), array([[0.01615506]]), array([[0.00623623]]), array([[0.00099858]]), array([[1.79337805e-26]]), array([[1.38899512e-27]]), array([[7.2147857e-313]])]
    ('aloha-control', '296bf2c2-1d88-4951-898f-2a2d6955ce63')
    [array([[0.00151959]]), array([[0.98291481]]), array([[0.01209957]]), array([[4.75912966e-05]]), array([[2.02893716e-05]]), array([[0.00031977]]), array([[0.01102928]]), array([[0.99756402]]), array([[0.01034162]]), array([[0.00803896]]), array([[0.01615506]]), array([[0.00623623]]), array([[0.00099858]]), array([[1.79337805e-26]]), array([[1.38899512e-27]]), array([[7.2147857e-313]])]
    ('aloha-control', '296bf2c2-1d88-4951-898f-2a2d6955ce63')
    [array([[0.00151959]]), array([[0.98291481]]), array([[0.01209957]]), array([[4.75912966e-05]]), array([[2.02893716e-05]]), array([[0.00031977]]), array([[0.01102928]]), array([[0.99756402]]), array([[0.01034162]]), array([[0.00803896]]), array([[0.01615506]]), array([[0.00623623]]), array([[0.00099858]]), array([[1.79337805e-26]]), array([[1.38899512e-27]]), array([[7.2147857e-313]])]


### Run Inference Batch

We will submit 1000 rows of test data through the pipeline, then loop through the responses and display which model each inference was performed in.  The results between the control and challenger should be approximately 2:1.


```python
from data import test_data
responses =[]
for nth in range(1000):
    responses.extend(experiment_pipeline.infer(test_data.data[nth]))
    
```


```python
l = [r.raw['model_name'] for r in responses]
df = pd.DataFrame({'model': l})
df.model.value_counts()
```




    aloha-control       676
    aloha-challenger    324
    Name: model, dtype: int64



### Test Challenger

Now we have run a large amount of data we can compare the results.

For this experiment we are looking for a significant change in the fraction of inferences that predicted a probability of the seventh category being high than 0.5 so we can determine whether our challenger model is more "successful" than the champion model at identifying category 7.


```python
control_count = 0
challenger_count = 0

control_success = 0
challenger_success = 0

for r in responses:
    if r.raw['model_name'] == "aloha-control":
        control_count += 1
        if(r.raw['outputs'][7]['Float']['data'][0] > .5):
            control_success += 1
    else:
        challenger_count +=1
        if(r.raw['outputs'][7]['Float']['data'][0] > .5):
            challenger_success += 1
print("control class 7 prediction rate: " + str(control_success/control_count))
print("challenger class 7 prediction rate: " + str(challenger_success/challenger_count))
```

    control class 7 prediction rate: 0.9718934911242604
    challenger class 7 prediction rate: 0.9876543209876543


### Undeploy Pipeline

With the testing complete, we undeploy the pipeline to return the resources back to the environment.


```python
experiment_pipeline.undeploy()

```




<table><tr><th>name</th> <td>randomsplitpipeline-demo</td></tr><tr><th>created</th> <td>2022-12-22 16:04:21.505294+00:00</td></tr><tr><th>last_updated</th> <td>2022-12-22 16:04:24.500570+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>840aab1d-df9e-45a2-9b0d-b05edff7b547, ff986b38-26b1-4bfa-8dcd-49a2ef114b74</td></tr><tr><th>steps</th> <td>aloha-control</td></tr></table>




```python

```
