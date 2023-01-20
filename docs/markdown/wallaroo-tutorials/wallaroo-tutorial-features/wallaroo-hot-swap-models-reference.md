## Model Hot Swap Tutorial

One of the biggest challenges facing organizations once they have a model trained is deploying the model:  Getting all of the resources together, MLOps configured and systems prepared to allow inferences to run.

The next biggest challenge?  Replacing the model while keeping the existing production systems running.

This tutorial demonstrates how Wallaroo model hot swap can update a pipeline step with a new model with one command.  This lets organizations keep their production systems running while changing a ML model, with the change taking only milliseconds.

This example and sample data comes from the Machine Learning Group's demonstration on [Credit Card Fraud detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

This tutorial provides the following:

* `ccfraud.onnx`: A pre-trained ML model used to detect potential credit card fraud.
* `xgboost_ccfraud.onnx`: A pre-trained ML model used to detect potential credit card fraud originally converted from an XGBoost model.  This will be used to swap with the `ccfraud.onnx`.
* `smoke_test.json`: A data file used to verify that the model will return a low possibility of credit card fraud.
* `high_fraud.json`: A data file used to verify that the model will return a high possibility of credit card fraud.
* Sample inference data files: Data files used for inference examples with the following number of records:
  * `cc_data_5.json`: 5 records.
  * `cc_data_1k.json`: 1,000 records.
  * `cc_data_10k.json`: 10,000 records.
  * `cc_data_40k.json`: Over 40,000 records.

## Reference

For more information about Wallaroo and related features, see the [Wallaroo Documentation Site](https://docs.wallaroo.ai).

## Steps

The following steps demonstrate the following:

* Connect to a Wallaroo instance.
* Create a workspace and pipeline.
* Upload both models to the workspace.
* Deploy the pipe with the `ccfraud.onnx` model as a pipeline step.
* Perform sample inferences.
* How swap and replace the existing model with the `xgboost_ccfraud.onnx` while keeping the pipeline deployed.
* Conduct additional inferences to demonstrate the model hot swap was successful.
* Undeploy the pipeline and return the resources back to the Wallaroo instance.

### Load the Libraries

Load the Python libraries used to connect and interact with the Wallaroo instance.


```python
import wallaroo
from wallaroo.object import EntityNotFoundError
```

### Open a Connection to Wallaroo

The first step is to connect to Wallaroo through the Wallaroo client.

This is accomplished using the `wallaroo.Client(api_endpoint, auth_endpoint, auth_type command)` command that connects to the Wallaroo instance services.

The `Client` method takes the following parameters:

* **api_endpoint** (*String*): The URL to the Wallaroo instance API service.
* **auth_endpoint** (*String*): The URL to the Wallaroo instance Keycloak service.
* **auth_type command** (*String*): The authorization type.  In this case, `SSO`.

The URLs are based on the Wallaroo Prefix and Wallaroo Suffix for the Wallaroo instance.  For more information, see the [DNS Integration Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-dns-guide/).  In the example below, replace "YOUR PREFIX" and "YOUR SUFFIX" with the Wallaroo Prefix and Suffix, respectively.

If connecting from within the Wallaroo instance's JupyterHub service, then only `wl = wallaroo.Client()` is required.

Once run, the `wallaroo.Client` command provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Depending on the configuration of the Wallaroo instance, the user will either be presented with a login request to the Wallaroo instance or be authenticated through a broker such as Google, Github, etc.  To use the broker, select it from the list under the username/password login forms.  For more information on Wallaroo authentication configurations, see the [Wallaroo Authentication Configuration Guides](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-sso-authentication/).


```python
# Remote Login

wallarooPrefix = "YOUR PREFIX"
wallarooSuffix = "YOUR SUFFIX"

wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}.api.{wallarooSuffix}", 
                    auth_endpoint=f"https://{wallarooPrefix}.keycloak.{wallarooSuffix}", 
                    auth_type="sso")

# Internal Login

# wl = wallaroo.Client()
```

### Set the Variables

The following variables are used in the later steps for creating the workspace, pipeline, and uploading the models.  Modify them according to your organization's requirements.


```python
workspace_name = 'hotswapworkspace'
pipeline_name = 'hotswappipeline'
original_model_name = 'ccfraudoriginal'
original_model_file_name = './ccfraud.onnx'
replacement_model_name = 'ccfraudreplacement'
replacement_model_file_name = './xgboost_ccfraud.onnx'
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

### Create the Workspace

We will create a workspace based on the variable names set above, and set the new workspace as the `current` workspace.  This workspace is where new pipelines will be created in and store uploaded models for this session.

Once set, the pipeline will be created.


```python
workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)

pipeline = get_pipeline(pipeline_name)
pipeline
```




<table><tr><th>name</th> <td>hotswappipeline</td></tr><tr><th>created</th> <td>2023-01-18 16:44:54.245465+00:00</td></tr><tr><th>last_updated</th> <td>2023-01-18 16:44:54.245465+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>9c993849-94d0-4f54-9fbe-9d62d2833cb3</td></tr><tr><th>steps</th> <td></td></tr></table>



### Upload Models

We can now upload both of the models.  In a later step, only one model will be added as a [pipeline step](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-pipeline/#add-a-step-to-a-pipeline), where the pipeline will submit inference requests to the pipeline.


```python
original_model = wl.upload_model(original_model_name , original_model_file_name)
replacement_model = wl.upload_model(replacement_model_name , replacement_model_file_name)
```

### Add Model to Pipeline Step

With the models uploaded, we will add the original model as a pipeline step, then deploy the pipeline so it is available for performing inferences.


```python
pipeline.add_model_step(original_model)
pipeline
```




<table><tr><th>name</th> <td>hotswappipeline</td></tr><tr><th>created</th> <td>2023-01-18 16:44:54.245465+00:00</td></tr><tr><th>last_updated</th> <td>2023-01-18 16:44:54.245465+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>9c993849-94d0-4f54-9fbe-9d62d2833cb3</td></tr><tr><th>steps</th> <td></td></tr></table>




```python
pipeline.deploy()
```




<table><tr><th>name</th> <td>hotswappipeline</td></tr><tr><th>created</th> <td>2023-01-18 16:44:54.245465+00:00</td></tr><tr><th>last_updated</th> <td>2023-01-18 16:55:46.047515+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>d0295b33-f1f9-420d-a1a6-c55485a13715, 9ccee697-98e5-4635-8db6-6ff8e577a47e, ab55afb4-2473-466b-b6eb-594f1a5727c3, 9c993849-94d0-4f54-9fbe-9d62d2833cb3</td></tr><tr><th>steps</th> <td>ccfraudoriginal</td></tr></table>




```python
pipeline.status()
```




    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.2.153',
       'name': 'engine-96486c95d-zfchr',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'hotswappipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'ccfraudreplacement',
          'version': '714efd19-5c83-42a8-aece-24b4ba530925',
          'sha': 'bc85ce596945f876256f41515c7501c399fd97ebcb9ab3dd41bf03f8937b4507',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.2.154',
       'name': 'engine-lb-55dcdff64c-9np9k',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}



### Verify the Model

The pipeline is deployed with our model.  The following will verify that the model is operating correctly.  The `high_fraud.json.json` file contains data that the model should process as a high likelihood of being a fraudulent transaction.


```python
pipeline.infer_from_file('./high_fraud.json')
```




    [InferenceResult({'check_failures': [],
      'elapsed': 123002,
      'model_name': 'ccfraudoriginal',
      'model_version': '3a03dc94-716e-46bb-84c8-91bc99ceb2c3',
      'original_data': {'tensor': [[1.0678324729342086,
                                    18.155556397512136,
                                    -1.658955105843852,
                                    5.2111788045436445,
                                    2.345247064454334,
                                    10.467083577773014,
                                    5.0925820522419745,
                                    12.82951536371218,
                                    4.953677046849403,
                                    2.3934736228338225,
                                    23.912131817957253,
                                    1.7599568310350209,
                                    0.8561037518143335,
                                    1.1656456468728569,
                                    0.5395988813934498,
                                    0.7784221343010385,
                                    6.75806107274245,
                                    3.927411847659908,
                                    12.462178276650056,
                                    12.307538216518656,
                                    13.787951906620115,
                                    1.4588397511627804,
                                    3.681834686805714,
                                    1.753914366037974,
                                    8.484355003656184,
                                    14.6454097666836,
                                    26.852377436250144,
                                    2.716529237720336,
                                    3.061195706890285]]},
      'outputs': [{'Float': {'data': [0.9811990261077881],
                             'dim': [1, 1],
                             'dtype': 'Float',
                             'v': 1}}],
      'pipeline_name': 'hotswappipeline',
      'shadow_data': {},
      'time': 1674060313971})]



### Replace the Model

The pipeline is currently deployed and is able to handle inferences.  The model will now be replaced without having to undeploy the pipeline.  This is done using the pipeline method [`replace_with_model_step(index, model)`](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-reference-guide/pipeline/#Pipeline.replace_with_model_step).  Steps start at `0`, so the method called below will replace step 0 in our pipeline with the replacement model.

As an exercise, this deployment can be performed while inferences are actively being submitted to the pipeline to show how quickly the swap takes place.


```python
pipeline.replace_with_model_step(0, replacement_model).deploy()
```




<table><tr><th>name</th> <td>hotswappipeline</td></tr><tr><th>created</th> <td>2023-01-18 16:44:54.245465+00:00</td></tr><tr><th>last_updated</th> <td>2023-01-18 16:47:12.577511+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>9ccee697-98e5-4635-8db6-6ff8e577a47e, ab55afb4-2473-466b-b6eb-594f1a5727c3, 9c993849-94d0-4f54-9fbe-9d62d2833cb3</td></tr><tr><th>steps</th> <td>ccfraudoriginal</td></tr></table>



### Verify the Swap

To verify the swap, we'll submit a set of inferences to the pipeline using the new model.


```python
pipeline.infer_from_file('./cc_data_5.json')
```




    [InferenceResult({'check_failures': [],
      'elapsed': 121101,
      'model_name': 'ccfraudreplacement',
      'model_version': '3a03dc94-716e-46bb-84c8-91bc99ceb2c3',
      'original_data': {'tensor': [[-1.060329750089797,
                                    2.354496709462385,
                                    -3.563878832646437,
                                    5.138734892618555,
                                    -1.23084570186641,
                                    -0.7687824607744093,
                                    -3.588122810891446,
                                    1.888083766259287,
                                    -3.2789674273886593,
                                    -3.956325455353324,
                                    4.099343911805088,
                                    -5.653917639476211,
                                    -0.8775733373342495,
                                    -9.131571191990632,
                                    -0.6093537872620682,
                                    -3.748027677256424,
                                    -5.030912501659983,
                                    -0.8748149525506821,
                                    1.9870535692026476,
                                    0.7005485718467245,
                                    0.9204422758154284,
                                    -0.10414918089758483,
                                    0.3229564351284999,
                                    -0.7418141656910608,
                                    0.03841201586730117,
                                    1.099343914614657,
                                    1.2603409755785089,
                                    -0.14662447391576958,
                                    -1.446321243938815],
                                   [-1.060329750089797,
                                    2.354496709462385,
                                    -3.563878832646437,
                                    5.138734892618555,
                                    -1.23084570186641,
                                    -0.7687824607744093,
                                    -3.588122810891446,
                                    1.888083766259287,
                                    -3.2789674273886593,
                                    -3.956325455353324,
                                    4.099343911805088,
                                    -5.653917639476211,
                                    -0.8775733373342495,
                                    -9.131571191990632,
                                    -0.6093537872620682,
                                    -3.748027677256424,
                                    -5.030912501659983,
                                    -0.8748149525506821,
                                    1.9870535692026476,
                                    0.7005485718467245,
                                    0.9204422758154284,
                                    -0.10414918089758483,
                                    0.3229564351284999,
                                    -0.7418141656910608,
                                    0.03841201586730117,
                                    1.099343914614657,
                                    1.2603409755785089,
                                    -0.14662447391576958,
                                    -1.446321243938815],
                                   [-1.060329750089797,
                                    2.354496709462385,
                                    -3.563878832646437,
                                    5.138734892618555,
                                    -1.23084570186641,
                                    -0.7687824607744093,
                                    -3.588122810891446,
                                    1.888083766259287,
                                    -3.2789674273886593,
                                    -3.956325455353324,
                                    4.099343911805088,
                                    -5.653917639476211,
                                    -0.8775733373342495,
                                    -9.131571191990632,
                                    -0.6093537872620682,
                                    -3.748027677256424,
                                    -5.030912501659983,
                                    -0.8748149525506821,
                                    1.9870535692026476,
                                    0.7005485718467245,
                                    0.9204422758154284,
                                    -0.10414918089758483,
                                    0.3229564351284999,
                                    -0.7418141656910608,
                                    0.03841201586730117,
                                    1.099343914614657,
                                    1.2603409755785089,
                                    -0.14662447391576958,
                                    -1.446321243938815],
                                   [-1.060329750089797,
                                    2.354496709462385,
                                    -3.563878832646437,
                                    5.138734892618555,
                                    -1.23084570186641,
                                    -0.7687824607744093,
                                    -3.588122810891446,
                                    1.888083766259287,
                                    -3.2789674273886593,
                                    -3.956325455353324,
                                    4.099343911805088,
                                    -5.653917639476211,
                                    -0.8775733373342495,
                                    -9.131571191990632,
                                    -0.6093537872620682,
                                    -3.748027677256424,
                                    -5.030912501659983,
                                    -0.8748149525506821,
                                    1.9870535692026476,
                                    0.7005485718467245,
                                    0.9204422758154284,
                                    -0.10414918089758483,
                                    0.3229564351284999,
                                    -0.7418141656910608,
                                    0.03841201586730117,
                                    1.099343914614657,
                                    1.2603409755785089,
                                    -0.14662447391576958,
                                    -1.446321243938815],
                                   [0.5817662107606553,
                                    0.0978815509566172,
                                    0.1546819423995403,
                                    0.475410194903404,
                                    -0.1978862305998003,
                                    -0.45043448542395703,
                                    0.016654044671806197,
                                    -0.025607055099995037,
                                    0.09205616023555586,
                                    -0.27839171528517387,
                                    0.059329944112281194,
                                    -0.019658541640589822,
                                    -0.4225083156745137,
                                    -0.12175388766841427,
                                    1.547309489412488,
                                    0.23916228635697,
                                    0.35539748808055915,
                                    -0.7685165300981693,
                                    -0.7000849354838512,
                                    -0.11900432852127547,
                                    -0.3450517133266211,
                                    -1.1065114107709193,
                                    0.2523411195349452,
                                    0.02094418256934876,
                                    0.2199267436399366,
                                    0.2540689265485751,
                                    -0.04502250942505252,
                                    0.1086773897916229,
                                    0.2547179311087416]]},
      'outputs': [{'Float': {'data': [0.993003249168396,
                                      0.993003249168396,
                                      0.993003249168396,
                                      0.993003249168396,
                                      0.001091688871383667],
                             'dim': [5, 1],
                             'dtype': 'Float',
                             'v': 1}}],
      'pipeline_name': 'hotswappipeline',
      'shadow_data': {},
      'time': 1674060447520})]



### Undeploy the Pipeline

With the tutorial complete, the pipeline is undeployed to return the resources back to the Wallaroo instance.


```python
pipeline.undeploy()
```




<table><tr><th>name</th> <td>hotswappipeline</td></tr><tr><th>created</th> <td>2023-01-18 16:44:54.245465+00:00</td></tr><tr><th>last_updated</th> <td>2023-01-18 16:47:12.577511+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>9ccee697-98e5-4635-8db6-6ff8e577a47e, ab55afb4-2473-466b-b6eb-594f1a5727c3, 9c993849-94d0-4f54-9fbe-9d62d2833cb3</td></tr><tr><th>steps</th> <td>ccfraudoriginal</td></tr></table>


