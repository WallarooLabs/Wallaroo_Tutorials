This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/shadow_deploy).

## Shadow Deployment Tutorial

Wallaroo provides a method of testing the same data against two different models or sets of models at the same time through **shadow deployments** otherwise known as **parallel deployments**.  This allows data to be submitted to a pipeline with inferences running on two different sets of models.  Typically this is performed on a model that is known to provide accurate results - the **champion** - and a model that is being tested to see if it provides more accurate or faster responses depending on the criteria known as the **challengers**.  Multiple challengers can be tested against a single champion.

As described in the Wallaroo blog post [The What, Why, and How of Model A/B Testing](https://www.wallaroo.ai/blog/the-what-why-and-how-of-a/b-testing):

> In data science, A/B tests can also be used to choose between two models in production, by measuring which model performs better in the real world. In this formulation, the control is often an existing model that is currently in production, sometimes called the champion. The treatment is a new model being considered to replace the old one. This new model is sometimes called the challenger....

> Keep in mind that in machine learning, the terms experiments and trials also often refer to the process of finding a training configuration that works best for the problem at hand (this is sometimes called hyperparameter optimization).

When a shadow deployment is created, only the inference from the champion is returned in the [InferenceResult Object](https://docs.wallaroo.ai/staging.documentation/wallaroo-sdk/wallaroo-sdk-essentials-guide/#inferenceresult-object) `data`, while the result data for the shadow deployments is stored in the [InferenceResult Object](https://docs.wallaroo.ai/staging.documentation/wallaroo-sdk/wallaroo-sdk-essentials-guide/#inferenceresult-object) `shadow_data`.

The following tutorial will demonstrate how:

* Upload champion and challenger models into a Wallaroo instance.
* Create a shadow deployment in a Wallaroo pipeline.
* Perform an inference through a pipeline with a shadow deployment.
* View the `data` and `shadow_data` results from the InferenceResult Object.
* View the pipeline logs and pipeline shadow logs.

This tutorial provides the following:

* `dev_smoke_test.json`:  Sample test data used for the inference testing.
* `models/keras_ccfraud.onnx`:  The champion model.
* `models/modelA.onnx`: A challenger model.
* `models/xgboost_ccfraud.onnx`: A challenger model.

All models are similar to the ones used for the Wallaroo-101 example included in the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials).

## Steps

### Import libraries

The first step is to import the libraries required.


```python
import wallaroo
from wallaroo.object import EntityNotFoundError
```

### Connect to Wallaroo

Connect to your Wallaroo instance and save the connection as the variable `wl`.

```python
wl = wallaroo.Client()
```

### Set Variables

The following variables are used to create or use existing workspaces, pipelines, and upload the models.  Adjust them based on your Wallaroo instance and organization requirements.

```python
workspace_name = 'ccfraud-comparison-demo'
pipeline_name = 'cc-shadow'
pipeline_name_multi = 'cc-shadow-multi'
champion_model_name = 'ccfraud-lstm'
champion_model_file = 'models/keras_ccfraud.onnx'
shadow_model_01_name = 'ccfraud-xgb'
shadow_model_01_file = 'models/xgboost_ccfraud.onnx'
shadow_model_02_name = 'ccfraud-rf'
shadow_model_02_file = 'models/modelA.onnx'
sample_data_file = './smoke_test.json'
```

### Workspace and Pipeline

The following creates or connects to an existing workspace based on the variable `workspace_name`, and creates or connects to a pipeline based on the variable `pipeline_name`.

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

pipeline = get_pipeline(pipeline_name)
pipeline

```



<table><tr><th>name</th> <td>cc-shadow</td></tr><tr><th>created</th> <td>2022-10-19 17:52:00.508852+00:00</td></tr><tr><th>last_updated</th> <td>2022-10-19 17:52:00.508852+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>steps</th> <td></td></tr></table>


### Load the Models

The models will be uploaded into the current workspace based on the variable names set earlier and listed as the `champion`, `model2` and `model3`.

```python
champion = wl.upload_model(champion_model_name, champion_model_file).configure()
model2 = wl.upload_model(shadow_model_01_name, shadow_model_01_file).configure()
model3 = wl.upload_model(shadow_model_02_name, shadow_model_02_file).configure()
```

### Create Shadow Deployment

A shadow deployment is created using the `add_shadow_deploy(champion, challengers[])` method where:

* `champion`: The model that will be primarily used for inferences run through the pipeline.  Inference results will be returned through the Inference Object's `data` element.
* `challengers[]`: An array of models that will be used for inferences iteratively.  Inference results will be returned through the Inference Object's `shadow_data` element.

```python
pipeline.add_shadow_deploy(champion, [model2, model3])
pipeline.deploy()
```

    Waiting for deployment - this will take up to 45s ..... ok



<table><tr><th>name</th> <td>cc-shadow</td></tr><tr><th>created</th> <td>2022-10-19 17:52:00.508852+00:00</td></tr><tr><th>last_updated</th> <td>2022-10-19 17:52:01.298216+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>steps</th> <td>ccfraud-lstm</td></tr></table>


### Run Test Inference

Using the data from `sample_data_file`, a test inference will be made.  As mentioned earlier, the inference results from the `champion` model will be available in the returned InferenceResult Object's `data` element, while inference results from each of the `challenger` models will be in the returned InferenceResult Object's `shadow_data` element.

```python
response = pipeline.infer_from_file(sample_data_file)
```

    Waiting for inference response - this will take up to 45s ........ ok


```python
response
```



    [InferenceResult({'check_failures': [],
      'elapsed': 226951,
      'model_name': 'ccfraud-lstm',
      'model_version': '1f7b2360-0d52-4000-b540-244fb33ad706',
      'original_data': {'tensor': [[1.0678324729342086,
                                    0.21778102664937624,
                                    -1.7115145261843976,
                                    0.6822857209662413,
                                    1.0138553066742804,
                                    -0.43350000129006655,
                                    0.7395859436561657,
                                    -0.28828395953577357,
                                    -0.44726268795990787,
                                    0.5146124987725894,
                                    0.3791316964287545,
                                    0.5190619748123175,
                                    -0.4904593221655364,
                                    1.1656456468728569,
                                    -0.9776307444180006,
                                    -0.6322198962519854,
                                    -0.6891477694494687,
                                    0.17833178574255615,
                                    0.1397992467197424,
                                    -0.35542206494183326,
                                    0.4394217876939808,
                                    1.4588397511627804,
                                    -0.3886829614721505,
                                    0.4353492889350186,
                                    1.7420053483337177,
                                    -0.4434654615252943,
                                    -0.15157478906219238,
                                    -0.26684517248765616,
                                    -1.454961775612449]]},
      'outputs': [{'Float': {'data': [0.001497417688369751],
                             'dim': [1, 1],
                             'v': 1}}],
      'pipeline_name': 'cc-shadow',
      'shadow_data': {'ccfraud-rf': [{'Float': {'data': [1.0],
                                                'dim': [1, 1],
                                                'v': 1}}],
                      'ccfraud-xgb': [{'Float': {'data': [0.0005066990852355957],
                                                 'dim': [1, 1],
                                                 'v': 1}}]},
      'time': 1666201934875})]


### View Pipeline Logs

With the inferences complete, we can retrieve the log data from the pipeline with the pipeline `logs` method.  Note that for **each** inference request, the logs return **one entry per model**.  For this example, for one inference request three log entries will be created.

```python
pipeline.logs()
```



<table>
    <tr>
        <th>Timestamp</th>
        <th>Output</th>
        <th>Input</th>
        <th>Anomalies</th>
    </tr>

<tr style="">
    <td>2022-19-Oct 17:52:14</td>
    <td>[array([[0.0005067]])]</td>
    <td>[[1.0678324729342086, 0.21778102664937624, -1.7115145261843976, 0.6822857209662413, 1.0138553066742804, -0.43350000129006655, 0.7395859436561657, -0.28828395953577357, -0.44726268795990787, 0.5146124987725894, 0.3791316964287545, 0.5190619748123175, -0.4904593221655364, 1.1656456468728569, -0.9776307444180006, -0.6322198962519854, -0.6891477694494687, 0.17833178574255615, 0.1397992467197424, -0.35542206494183326, 0.4394217876939808, 1.4588397511627804, -0.3886829614721505, 0.4353492889350186, 1.7420053483337177, -0.4434654615252943, -0.15157478906219238, -0.26684517248765616, -1.454961775612449]]</td>
    <td>0</td>
</tr>

<tr style="">
    <td>2022-19-Oct 17:52:14</td>
    <td>[array([[1.]])]</td>
    <td>[[1.0678324729342086, 0.21778102664937624, -1.7115145261843976, 0.6822857209662413, 1.0138553066742804, -0.43350000129006655, 0.7395859436561657, -0.28828395953577357, -0.44726268795990787, 0.5146124987725894, 0.3791316964287545, 0.5190619748123175, -0.4904593221655364, 1.1656456468728569, -0.9776307444180006, -0.6322198962519854, -0.6891477694494687, 0.17833178574255615, 0.1397992467197424, -0.35542206494183326, 0.4394217876939808, 1.4588397511627804, -0.3886829614721505, 0.4353492889350186, 1.7420053483337177, -0.4434654615252943, -0.15157478906219238, -0.26684517248765616, -1.454961775612449]]</td>
    <td>0</td>
</tr>

<tr style="">
    <td>2022-19-Oct 17:52:14</td>
    <td>[array([[0.00149742]])]</td>
    <td>[[1.0678324729342086, 0.21778102664937624, -1.7115145261843976, 0.6822857209662413, 1.0138553066742804, -0.43350000129006655, 0.7395859436561657, -0.28828395953577357, -0.44726268795990787, 0.5146124987725894, 0.3791316964287545, 0.5190619748123175, -0.4904593221655364, 1.1656456468728569, -0.9776307444180006, -0.6322198962519854, -0.6891477694494687, 0.17833178574255615, 0.1397992467197424, -0.35542206494183326, 0.4394217876939808, 1.4588397511627804, -0.3886829614721505, 0.4353492889350186, 1.7420053483337177, -0.4434654615252943, -0.15157478906219238, -0.26684517248765616, -1.454961775612449]]</td>
    <td>0</td>
</tr>

</table>



### View Logs Per Model

Another way of displaying the logs would be to specify my model.  The following code will display the log data based based on the model name and the inference output for that specific model.

```python
logs = pipeline.logs()

[(log.model_name, log.output) for log in logs]
```



    [('ccfraud-xgb', [array([[0.0005067]])]),
     ('ccfraud-rf', [array([[1.]])]),
     ('ccfraud-lstm', [array([[0.00149742]])])]


### View Shadow Deploy Pipeline Logs

To view the inputs and results for the shadow deployed models, use the pipeline `logs_shadow_deploy()` method.  The results will be grouped by the inputs.

```python
logs = pipeline.logs_shadow_deploy()
logs
```



                <h2>Shadow Deploy Logs</h2>
                <p>
                    <em>Logs from a shadow pipeline, grouped by their input.</em>
                </p>
                <table>
                    <tbody>

                    <tr><td colspan='6'>Log Entry 0</td></tr>
                    <tr><td colspan='6'></td></tr>
                    <tr>
			<td>
				<strong><em>Input</em></strong>
			</td>
                        <td colspan='6'>[[1.0678324729342086, 0.21778102664937624, -1.7115145261843976, 0.6822857209662413, 1.0138553066742804, -0.43350000129006655, 0.7395859436561657, -0.28828395953577357, -0.44726268795990787, 0.5146124987725894, 0.3791316964287545, 0.5190619748123175, -0.4904593221655364, 1.1656456468728569, -0.9776307444180006, -0.6322198962519854, -0.6891477694494687, 0.17833178574255615, 0.1397992467197424, -0.35542206494183326, 0.4394217876939808, 1.4588397511627804, -0.3886829614721505, 0.4353492889350186, 1.7420053483337177, -0.4434654615252943, -0.15157478906219238, -0.26684517248765616, -1.454961775612449]]</td>
                    </tr>

                    <tr>
                        <td>Model Type</td>
                        <td>
                            <strong>Model Name</strong>
                        </td>
                        <td>
                            <strong>Output</strong>
                        </td>
                        <td>
                            <strong>Timestamp</strong>
                        </td>
                        <td>
                            <strong>Model Version</strong>
                        </td>
                        <td>
                            <strong>Elapsed</strong>
                        </td>
                    </tr>
                    <tr>
                        <td><strong><em>Primary</em></strong></td>
                        <td>ccfraud-lstm</td>
                        <td>[array([[0.00149742]])]</td>
                        <td>2022-10-19T17:52:14.875000</td>
                        <td>1f7b2360-0d52-4000-b540-244fb33ad706</td>
                        <td>226951</td>
                    </tr>

                    <tr>
                        <td><strong><em>Challenger</em></strong></td>
                        <td>ccfraud-rf</td>
                        <td>[{'Float': {'v': 1, 'dim': [1, 1], 'data': [1.0]}}]</td>
                        <td colspan=3></td>
                    </tr>

                    <tr>
                        <td><strong><em>Challenger</em></strong></td>
                        <td>ccfraud-xgb</td>
                        <td>[{'Float': {'v': 1, 'dim': [1, 1], 'data': [0.0005066990852355957]}}]</td>
                        <td colspan=3></td>
                    </tr>

                    </tbody>
                <table>



### Undeploy the Pipeline

With the tutorial complete, we undeploy the pipeline and return the resources back to the system.

```python
pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s .................................... ok



<table><tr><th>name</th> <td>cc-shadow</td></tr><tr><th>created</th> <td>2022-10-19 17:52:00.508852+00:00</td></tr><tr><th>last_updated</th> <td>2022-10-19 17:52:01.298216+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>steps</th> <td>ccfraud-lstm</td></tr></table>

