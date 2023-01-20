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

    3.8.15



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

We will create a workspace to work in and call it the "alohaworkspace", then set it as current workspace environment.  We'll also create our pipeline in advance as `alohapipeline`.  The model name and the model file will be specified for use in later steps.

* **IMPORTANT NOTE**:  For this example, the Aloha model is stored in the file `alohacnnlstm.zip`.  When using tensor based models, the zip file **must** match the name of the tensor directory.  For example, if the tensor directory is `alohacnnlstm`, then the .zip file must be named `alohacnnlstm.zip`.


```python
workspace_name = 'alohaworkspace'
pipeline_name = 'alohapipeline'
model_name = 'alohamodel'
model_file_name = './alohacnnlstm.zip'
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
wl.list_workspaces()
```





<table>
    <tr>
        <th>Name</th>
        <th>Created At</th>
        <th>Users</th>
        <th>Models</th>
        <th>Pipelines</th>
    </tr>

<tr >
    <td>john.hummel@wallaroo.ai - Default Workspace</td>
    <td>2023-01-11 19:08:48</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>3</td>
    <td>2</td>
</tr>


<tr >
    <td>mobilenetpipeline</td>
    <td>2023-01-12 15:39:21</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>


<tr >
    <td>mobilenetworkspace</td>
    <td>2023-01-12 17:05:11</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>1</td>
    <td>1</td>
</tr>


<tr >
    <td>resnetworkspace</td>
    <td>2023-01-12 17:09:52</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>2</td>
    <td>2</td>
</tr>


<tr >
    <td>shadowimageworkspace</td>
    <td>2023-01-13 14:55:26</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>2</td>
    <td>1</td>
</tr>


<tr >
    <td>Test</td>
    <td>2023-01-13 17:14:44</td>
    <td>['john.hummel@wallaroo.ai']</td>
    <td>0</td>
    <td>1</td>
</tr>

</table>





```python
workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)

aloha_pipeline = get_pipeline(pipeline_name)
aloha_pipeline
```


    ---------------------------------------------------------------------------

    Exception                                 Traceback (most recent call last)

    /var/folders/jf/_cj0q9d51s365wksymljdz4h0000gn/T/ipykernel_34338/2302154216.py in <module>
    ----> 1 workspace = get_workspace(workspace_name)
          2 
          3 wl.set_current_workspace(workspace)
          4 
          5 aloha_pipeline = get_pipeline(pipeline_name)


    /var/folders/jf/_cj0q9d51s365wksymljdz4h0000gn/T/ipykernel_34338/1020935072.py in get_workspace(name)
          5             workspace= ws
          6     if(workspace == None):
    ----> 7         workspace = wl.create_workspace(name)
          8     return workspace
          9 


    /opt/homebrew/anaconda3/envs/wallaroosdk/lib/python3.8/site-packages/wallaroo/client.py in create_workspace(self, workspace_name)
       1337         assert workspace_name is not None
       1338         require_dns_compliance(workspace_name)
    -> 1339         return Workspace._create_workspace(client=self, name=workspace_name)
       1340 
       1341     def list_workspaces(self) -> List[Workspace]:


    /opt/homebrew/anaconda3/envs/wallaroosdk/lib/python3.8/site-packages/wallaroo/workspace.py in _create_workspace(client, name)
        116 
        117         if not isinstance(res, WorkspacesCreateResponse200):
    --> 118             raise Exception("Failed to create workspace.")
        119 
        120         if res is None:


    Exception: Failed to create workspace.


We can verify the workspace is created the current default workspace with the `get_current_workspace()` command.


```python
wl.get_current_workspace()
```

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


```python
aloha_pipeline.deploy()
```

We can verify that the pipeline is running and list what models are associated with it.


```python
aloha_pipeline.status()
```

## Interferences

### Infer 1 row

Now that the pipeline is deployed and our Aloha model is in place, we'll perform a smoke test to verify the pipeline is up and running properly.  We'll use the `infer_from_file` command to load a single encoded URL into the inference engine and print the results back out.

The result should tell us that the tokenized URL is legitimate (0) or fraud (1).  This sample data should return close to 0.


```python
aloha_pipeline.infer_from_file("./data-1.json")
```

### Batch Inference

Now that our smoke test is successful, let's really give it some data.  We have two inference files we can use:

* `data-1k.json`:  Contains 10,000 inferences
* `data-25k.json`: Contains 25,000 inferences

We'll pipe the `data-25k.json` file through the `aloha_pipeline` deployment URL, and place the results in a file named `response.txt`.  We'll also display the time this takes.  Note that for larger batches of 50,000 inferences or more can be difficult to view in Jupyter Hub because of its size.

* **IMPORTANT NOTE**:  The `_deployment._url()` method will return an **internal** URL when using Python commands from within the Wallaroo instance - for example, the Wallaroo JupyterHub service.  When connecting via an external connection, `_deployment._url()` returns an **external** URL.  External URL connections requires [the authentication be included in the HTTP request](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-api-guide/), and that [Model Endpoints Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-model-endpoints-guide/) external endpoints are enabled in the Wallaroo configuration options.


```python
inference_url = aloha_pipeline._deployment._url()
```


```python
connection =wl.mlops().__dict__
token = connection['token']
token

```


```python
!curl -X POST {inference_url} -H "Authorization: Bearer {token}" -H "Content-Type:application/json" --data @data-25k.json > curl_response.txt
```

## Undeploy Pipeline

When finished with our tests, we will undeploy the pipeline so we have the Kubernetes resources back for other tasks.  Note that if the deployment variable is unchanged aloha_pipeline.deploy() will restart the inference engine in the same configuration as before.


```python
aloha_pipeline.undeploy()
```


```python

```
