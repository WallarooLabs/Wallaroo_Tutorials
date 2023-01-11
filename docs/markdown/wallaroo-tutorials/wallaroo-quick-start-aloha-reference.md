This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-model-cookbooks/aloha).

## Aloha Demo

In this notebook we will walk through a simple pipeline deployment to inference on a model. For this example we will be using an open source model that uses an [Aloha CNN LSTM model](https://www.researchgate.net/publication/348920204_Using_Auxiliary_Inputs_in_Deep_Learning_Models_for_Detecting_DGA-based_Domain_Names) for classifying Domain names as being either legitimate or being used for nefarious purposes such as malware distribution.  

For our example, we will perform the following:

* Create a workspace for our work.
* Upload the Aloha model.
* Create a pipeline that can ingest our submitted data, submit it to the model, and export the results
* Run a sample inference through our pipeline by loading a file
* Run a sample inference through our pipeline's URL and store the results in a file.

All sample data and models are available through the [Wallaroo Quick Start Guide Samples repository](https://github.com/WallarooLabs/quickstartguide_samples).

## Open a Connection to Wallaroo

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.


```python
from platform import python_version

print(python_version())
```

    3.8.11



```python
import wallaroo
from wallaroo.object import EntityNotFoundError
```


```python
# Client connection from local Wallaroo instance

# wl = wallaroo.Client()

# SSO login through keycloak

wallarooPrefix = "YOUR PREFIX"
wallarooSuffix = "YOUR SUFFIX"

wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}.api.{wallarooSuffix}", 
                    auth_endpoint=f"https://{wallarooPrefix}.keycloak.{wallarooSuffix}", 
                    auth_type="sso")
```

## Create the Workspace

We will create a workspace to work in and call it the "alohaworkspace", then set it as current workspace environment.  We'll also create our pipeline in advance as `alohapipeline`.


```python
workspace_name = 'alohaworkspace'
pipeline_name = 'alohapipeline'
model_name = 'alohamodel'
model_file_name = './aloha-cnn-lstm.zip'
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




<table><tr><th>name</th> <td>alohapipeline</td></tr><tr><th>created</th> <td>2022-12-16 21:00:02.528654+00:00</td></tr><tr><th>last_updated</th> <td>2022-12-16 21:00:02.528654+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>5cdaa977-856c-4a7a-a14e-dae7228fb5d6</td></tr><tr><th>steps</th> <td></td></tr></table>



We can verify the workspace is created the current default workspace with the `get_current_workspace()` command.


```python
wl.get_current_workspace()
```




    {'name': 'alohaworkspace', 'id': 28, 'archived': False, 'created_by': '01a797f9-1357-4506-a4d2-8ab9c4681103', 'created_at': '2022-12-16T21:00:01.614796+00:00', 'models': [], 'pipelines': [{'name': 'alohapipeline', 'create_time': datetime.datetime(2022, 12, 16, 21, 0, 2, 528654, tzinfo=tzutc()), 'definition': '[]'}]}



# Upload the Models

Now we will upload our models.  Note that for this example we are applying the model from a .ZIP file.  The Aloha model is a [protobuf](https://developers.google.com/protocol-buffers) file that has been defined for evaluating web pages, and we will configure it to use data in the `tensorflow` format.


```python
model = wl.upload_model(model_name, model_file_name).configure("tensorflow")
```

## Deploy a model
Now that we have a model that we want to use we will create a deployment for it. 

We will tell the deployment we are using a tensorflow model and give the deployment name and the configuration we want for the deployment.

To do this, we'll create our pipeline that can ingest the data, pass the data to our Aloha model, and give us a final output.  We'll call our pipeline `aloha-test-demo`, then deploy it so it's ready to receive data.  The deployment process usually takes about 45 seconds.

* **Note**:  If you receive an error that the pipeline could not be deployed because there are not enough resources, undeploy any other pipelines and deploy this one again.  This command can quickly undeploy all pipelines to regain resources.  We recommend **not** running this command in a production environment since it will cancel any running pipelines:

```python
for p in wl.list_pipelines(): p.undeploy()
```


```python
aloha_pipeline.add_model_step(model)
```




<table><tr><th>name</th> <td>alohapipeline</td></tr><tr><th>created</th> <td>2022-12-16 21:00:02.528654+00:00</td></tr><tr><th>last_updated</th> <td>2022-12-16 21:00:02.528654+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>5cdaa977-856c-4a7a-a14e-dae7228fb5d6</td></tr><tr><th>steps</th> <td></td></tr></table>




```python
aloha_pipeline.deploy()
```




<table><tr><th>name</th> <td>alohapipeline</td></tr><tr><th>created</th> <td>2022-12-16 21:00:02.528654+00:00</td></tr><tr><th>last_updated</th> <td>2022-12-16 21:00:11.796841+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>f282c205-afe3-4a77-baf5-b379fe8a60d7, 5cdaa977-856c-4a7a-a14e-dae7228fb5d6</td></tr><tr><th>steps</th> <td>alohamodel</td></tr></table>



We can verify that the pipeline is running and list what models are associated with it.


```python
aloha_pipeline.status()
```




    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.4.2.6',
       'name': 'engine-7465877489-gf4g9',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'alohapipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'alohamodel',
          'version': '45224236-61f8-4628-a6c6-f96954863c89',
          'sha': '7c89707252ce389980d5348c37885d6d72af4c20cd303422e2de7e66dd7ff184',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.4.0.22',
       'name': 'engine-lb-7d6f4bfdd-qwdtx',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}



## Interferences

### Infer 1 row

Now that the pipeline is deployed and our Aloha model is in place, we'll perform a smoke test to verify the pipeline is up and running properly.  We'll use the `infer_from_file` command to load a single encoded URL into the inference engine and print the results back out.

The result should tell us that the tokenized URL is legitimate (0) or fraud (1).  This sample data should return close to 0.


```python
aloha_pipeline.infer_from_file("./data-1.json")
```




    [InferenceResult({'check_failures': [],
      'elapsed': 275299859,
      'model_name': 'alohamodel',
      'model_version': '45224236-61f8-4628-a6c6-f96954863c89',
      'original_data': {'text_input': [[0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        0,
                                        28,
                                        16,
                                        32,
                                        23,
                                        29,
                                        32,
                                        30,
                                        19,
                                        26,
                                        17]]},
      'outputs': [{'Float': {'data': [0.0015195842133834958],
                             'dim': [1, 1],
                             'dtype': 'Float',
                             'v': 1}},
                  {'Float': {'data': [0.9829147458076477],
                             'dim': [1, 1],
                             'dtype': 'Float',
                             'v': 1}},
                  {'Float': {'data': [0.012099552899599075],
                             'dim': [1, 1],
                             'dtype': 'Float',
                             'v': 1}},
                  {'Float': {'data': [4.7591205657226965e-05],
                             'dim': [1, 1],
                             'dtype': 'Float',
                             'v': 1}},
                  {'Float': {'data': [2.0289331587264314e-05],
                             'dim': [1, 1],
                             'dtype': 'Float',
                             'v': 1}},
                  {'Float': {'data': [0.00031977228354662657],
                             'dim': [1, 1],
                             'dtype': 'Float',
                             'v': 1}},
                  {'Float': {'data': [0.011029261164367199],
                             'dim': [1, 1],
                             'dtype': 'Float',
                             'v': 1}},
                  {'Float': {'data': [0.9975640177726746],
                             'dim': [1, 1],
                             'dtype': 'Float',
                             'v': 1}},
                  {'Float': {'data': [0.010341613553464413],
                             'dim': [1, 1],
                             'dtype': 'Float',
                             'v': 1}},
                  {'Float': {'data': [0.008038961328566074],
                             'dim': [1, 1],
                             'dtype': 'Float',
                             'v': 1}},
                  {'Float': {'data': [0.016155045479536057],
                             'dim': [1, 1],
                             'dtype': 'Float',
                             'v': 1}},
                  {'Float': {'data': [0.0062362332828342915],
                             'dim': [1, 1],
                             'dtype': 'Float',
                             'v': 1}},
                  {'Float': {'data': [0.0009985746582970023],
                             'dim': [1, 1],
                             'dtype': 'Float',
                             'v': 1}},
                  {'Float': {'data': [1.7933435344117743e-26],
                             'dim': [1, 1],
                             'dtype': 'Float',
                             'v': 1}},
                  {'Float': {'data': [1.388984431455466e-27],
                             'dim': [1, 1],
                             'dtype': 'Float',
                             'v': 1}}],
      'pipeline_name': 'alohapipeline',
      'shadow_data': {},
      'time': 1671224441072})]



### Batch Inference

Now that our smoke test is successful, let's really give it some data.  We have two inference files we can use:

* `data-1k.json`:  Contains 1,0000 inferences
* `data-25k.json`: Contains 25,000 inferences

We'll pipe the `data-25k.json` file through the `aloha_pipeline` deployment URL, and place the results in a file named `response.txt`.  We'll also display the time this takes.  Note that for larger batches of 50,000 inferences or more can be difficult to view in Jupyter Hub because of its size.

When running this example, replace the URL from the `_deployment._url()` command into the `curl` command below.

* **IMPORTANT NOTE**:  The `_deployment._url()` method will return an **internal** URL when using Python commands from within the Wallaroo instance - for example, the Wallaroo JupyterHub service.  When connecting via an external connection, `_deployment._url()` returns an **external** URL.  External URL connections requires [the authentication be included in the HTTP request](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/), and that [Model Endpoints Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-model-endpoints-guide/) external endpoints are enabled in the Wallaroo configuration options.

```python
inference_url = aloha_pipeline._deployment._url()
```


```python
connection =wl.mlops().__dict__
token = connection['token']
token

```

    'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJWalFITFhMMThub3BXNWVHM2hMOVJ5MDZ1SFVWMko1dHREUkVxSGtBT2VzIn0.eyJleHAiOjE2NzEyMjQ0NjAsImlhdCI6MTY3MTIyNDQwMCwiYXV0aF90aW1lIjoxNjcxMjIyMjM1LCJqdGkiOiJjMTZjZjU5ZS02ZDEzLTQ0OTMtYTc2NC0zZTY4ZjNiOWQ1ODciLCJpc3MiOiJodHRwczovL3NxdWlzaHktd2FsbGFyb28tNjE4Ny5rZXljbG9hay53YWxsYXJvby5kZXYvYXV0aC9yZWFsbXMvbWFzdGVyIiwiYXVkIjpbIm1hc3Rlci1yZWFsbSIsImFjY291bnQiXSwic3ViIjoiMDFhNzk3ZjktMTM1Ny00NTA2LWE0ZDItOGFiOWM0NjgxMTAzIiwidHlwIjoiQmVhcmVyIiwiYXpwIjoic2RrLWNsaWVudCIsInNlc3Npb25fc3RhdGUiOiJkODQ2NjQxZi00ZTExLTQ1YWItYThmZS01ZjI3ZGI2NzAyMjciLCJhY3IiOiIwIiwicmVhbG1fYWNjZXNzIjp7InJvbGVzIjpbImNyZWF0ZS1yZWFsbSIsImRlZmF1bHQtcm9sZXMtbWFzdGVyIiwib2ZmbGluZV9hY2Nlc3MiLCJhZG1pbiIsInVtYV9hdXRob3JpemF0aW9uIl19LCJyZXNvdXJjZV9hY2Nlc3MiOnsibWFzdGVyLXJlYWxtIjp7InJvbGVzIjpbInZpZXctaWRlbnRpdHktcHJvdmlkZXJzIiwidmlldy1yZWFsbSIsIm1hbmFnZS1pZGVudGl0eS1wcm92aWRlcnMiLCJpbXBlcnNvbmF0aW9uIiwiY3JlYXRlLWNsaWVudCIsIm1hbmFnZS11c2VycyIsInF1ZXJ5LXJlYWxtcyIsInZpZXctYXV0aG9yaXphdGlvbiIsInF1ZXJ5LWNsaWVudHMiLCJxdWVyeS11c2VycyIsIm1hbmFnZS1ldmVudHMiLCJtYW5hZ2UtcmVhbG0iLCJ2aWV3LWV2ZW50cyIsInZpZXctdXNlcnMiLCJ2aWV3LWNsaWVudHMiLCJtYW5hZ2UtYXV0aG9yaXphdGlvbiIsIm1hbmFnZS1jbGllbnRzIiwicXVlcnktZ3JvdXBzIl19LCJhY2NvdW50Ijp7InJvbGVzIjpbIm1hbmFnZS1hY2NvdW50IiwibWFuYWdlLWFjY291bnQtbGlua3MiLCJ2aWV3LXByb2ZpbGUiXX19LCJzY29wZSI6InByb2ZpbGUgZW1haWwiLCJzaWQiOiJkODQ2NjQxZi00ZTExLTQ1YWItYThmZS01ZjI3ZGI2NzAyMjciLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiaHR0cHM6Ly9oYXN1cmEuaW8vand0L2NsYWltcyI6eyJ4LWhhc3VyYS11c2VyLWlkIjoiMDFhNzk3ZjktMTM1Ny00NTA2LWE0ZDItOGFiOWM0NjgxMTAzIiwieC1oYXN1cmEtZGVmYXVsdC1yb2xlIjoidXNlciIsIngtaGFzdXJhLWFsbG93ZWQtcm9sZXMiOlsidXNlciJdLCJ4LWhhc3VyYS11c2VyLWdyb3VwcyI6Int9In0sInByZWZlcnJlZF91c2VybmFtZSI6ImpvaG4uaGFuc2FyaWNrQHdhbGxhcm9vLmFpIiwiZW1haWwiOiJqb2huLmhhbnNhcmlja0B3YWxsYXJvby5haSJ9.iToD4PosX9PWgyxr3Z_z3vyyI1sR-DinTBOwBGqxs_lV5LURRV6VlhnjkxZIfVzSABeWuvP7oCRKEwzOxk-IBP1GtDbBLB55fb8Mw4rlurUCD_KZ-940prnrnLDY29Vg2yRSZ2xZxO8Z6wRck6yu0NcoTJtF_VJlwtYzztKTh80RE_Sr9Ddy6PVaq8ElrT8h0OwAKh3dB9kiH5yh2RWHl3_VAubGBP4Ne2BEw5ZBPmj9gPjQ82BkA9lqaUlt5EeEMBgwEx39TWh3GjGBFmxzobdEiiBuhAZyUqIB9ffQEGrMs8Tz_r2EzdZATeUmkJ57l7zAysYnTJvvEHzVEs38AA'




```python
!curl -X POST {inference_url} -H "Content-Type:application/json" -H "Authorization: Bearer {token}" -H "Content-Type:application/json" --data @data-25k.json > curl_response.txt
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
      4 2886k    0     0    4  128k      0   709k  0:00:04 --:--:--  0:00:04  715k


## Undeploy Pipeline

When finished with our tests, we will undeploy the pipeline so we have the Kubernetes resources back for other tasks.  Note that if the deployment variable is unchanged aloha_pipeline.deploy() will restart the inference engine in the same configuration as before.


```python
aloha_pipeline.undeploy()
```




<table><tr><th>name</th> <td>alohapipeline</td></tr><tr><th>created</th> <td>2022-12-16 21:00:02.528654+00:00</td></tr><tr><th>last_updated</th> <td>2022-12-16 21:00:11.796841+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>f282c205-afe3-4a77-baf5-b379fe8a60d7, 5cdaa977-856c-4a7a-a14e-dae7228fb5d6</td></tr><tr><th>steps</th> <td>alohamodel</td></tr></table>




```python

```
