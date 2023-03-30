This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-model-cookbooks/demand_curve).

## Demand Curve Pipeline Tutorial

This worksheet demonstrates a Wallaroo pipeline with data preprocessing, a model, and data postprocessing.

The model is a "demand curve" that predicts the expected number of units of a product that will be sold to a customer as a function of unit price and facts about the customer. Such models can be used for price optimization or sales volume forecasting.  This is purely a "toy" demonstration, but is useful for detailing the process of working with models and pipelines.

Data preprocessing is required to create the features used by the model. Simple postprocessing prevents nonsensical estimates (e.g. negative units sold).

## Prerequisites

* An installed Wallaroo instance.
* The following Python libraries installed:
  * `os`
  * [`wallaroo`](https://pypi.org/project/wallaroo/): The Wallaroo SDK. Included with the Wallaroo JupyterHub service by default.

## Open a Connection to Wallaroo

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

The other libraries shown below are used for this example.


```python
import wallaroo
from wallaroo.object import EntityNotFoundError

import os
# For Wallaroo SDK 2023.1
os.environ["ARROW_ENABLED"]="True"
```


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

Now that the Wallaroo client has been initialized, we can create the workspace and call it `demandcurveworkspace`, then set it as our current workspace.  We'll also create our pipeline so it's ready when we add our models to it.

We'll set some variables and methods to create our workspace, pipelines and models.  Note that as of the July 2022 release of Wallaroo, workspace names must be unique.  Pipelines with the same name will be created as a new version when built.


```python
workspace_name = 'demandcurveworkspace'
pipeline_name = 'demandcurvepipeline'
model_name = 'demandcurvemodel'
model_file_name = './demand_curve_v1.onnx'
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

demandcurve_pipeline = get_pipeline(pipeline_name)
demandcurve_pipeline
```




<table><tr><th>name</th> <td>demandcurvepipeline</td></tr><tr><th>created</th> <td>2023-03-27 22:07:57.525519+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-27 22:07:57.525519+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>4a7a59bd-2565-4310-805d-38c825d86831</td></tr><tr><th>steps</th> <td></td></tr></table>



With our workspace established, we'll upload three models:

* `demand_curve_v1.onnx`: Our demand_curve model.  We'll store the upload configuration into `demand_curve_model`.
* `preprocess`:  Takes the data and prepares it for the demand curve model.  We'll store the upload configuration into `module_pre`.
* `postprocess`:  Takes the results from our demand curve model and prepares it for our display.  We'll store the upload configuration into `module_post`.

Note that the order we upload our models isn't important - we'll be establishing the actual process of moving data from one model to the next when we set up our pipeline.


```python
# upload to wallaroo
demand_curve_model = wl.upload_model(model_name, model_file_name).configure()
```


```python
# load the preprocess module
module_pre = wl.upload_model("preprocess", "./preprocess.py").configure('python')
```


```python
# load the postprocess module
module_post = wl.upload_model("postprocess", "./postprocess.py").configure('python')
```

With our models uploaded, we're going to create our own pipeline and give it three steps:

* First, start with the preprocess module we called `module_pre` to prepare the data.
* Second, we apply the data to our `demand_curve_model`.
* And finally, we prepare our data for output with the `module_post`.


```python
# now make a pipeline
demandcurve_pipeline.add_model_step(module_pre)
demandcurve_pipeline.add_model_step(demand_curve_model)
demandcurve_pipeline.add_model_step(module_post)
```




<table><tr><th>name</th> <td>demandcurvepipeline</td></tr><tr><th>created</th> <td>2023-03-27 22:07:57.525519+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-27 22:07:57.525519+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>4a7a59bd-2565-4310-805d-38c825d86831</td></tr><tr><th>steps</th> <td></td></tr></table>



And with that - let's deploy our model pipeline.  This usually takes about 45 seconds for the deployment to finish.


```python
demandcurve_pipeline.deploy()
```




<table><tr><th>name</th> <td>demandcurvepipeline</td></tr><tr><th>created</th> <td>2023-03-27 22:07:57.525519+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-27 22:08:03.079372+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>3014a26a-c277-447b-ace9-e2712e207008, 4a7a59bd-2565-4310-805d-38c825d86831</td></tr><tr><th>steps</th> <td>preprocess</td></tr></table>



We can check the status of our pipeline to make sure everything was set up correctly:


```python
demandcurve_pipeline.status()
```




    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.3.27',
       'name': 'engine-54f878976c-4pbm5',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'demandcurvepipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'postprocess',
          'version': '1967c453-622e-4db9-9498-0a264261e331',
          'sha': '35fbb219462ed5d80f103c920c06f09fd3950c2334cd4c124af5a5c7d2ecbd2f',
          'status': 'Running'},
         {'name': 'demandcurvemodel',
          'version': '926ab687-f9ec-4d31-9bf6-d1e9de189848',
          'sha': '2820b42c9e778ae259918315f25afc8685ecab9967bad0a3d241e6191b414a0d',
          'status': 'Running'},
         {'name': 'preprocess',
          'version': '73b63787-5d87-4229-8556-727224f37a16',
          'sha': '1d0090808e807ccb20422e77e59d4d38e3cc39fae5ce115032e68a855a5a62c0',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.0.174',
       'name': 'engine-lb-ddd995646-cbd5r',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}



Everything is ready.  Let's feed our pipeline some data.  We have some information prepared with the `daily_purchasses.csv` spreadsheet.  We'll start with just one row to make sure that everything is working correctly.


```python
# read in some purchase data
purchases = pandas.read_csv('daily_purchases.csv')

# start with a one-row data frame for testing
subsamp_raw = purchases.iloc[0:1,: ]
subsamp_raw

# create the input dictionary from the original one-line dataframe
input_dict = conversion.pandas_to_dict(subsamp_raw)
```


```python
result = demandcurve_pipeline.infer(input_dict)
```

We can see from the `prediction` field that the demand curve has a predicted slope of 6.68 from our sample data.  We can isolate that by specifying just the data output below.


```python
display(result[0]['prediction'])
```


    [6.68025518653071]


# Bulk Inference

The initial test went perfectly.  Now let's throw some more data into our pipeline.  We'll draw 10 random rows from our spreadsheet, perform an inference from that, and then display the results and the logs showing the pipeline's actions.


```python
# Let's do 10 rows at once (drawn randomly)
ix = numpy.random.choice(purchases.shape[0], size=10, replace=False)
output = demandcurve_pipeline.infer(conversion.pandas_to_dict(purchases.iloc[ix,: ]))
```


```python
display(output[0]['prediction'])
```


    [33.125323160373426,
     33.125323160373426,
     6.771545926800889,
     6.771545926800889,
     33.125323160373426,
     33.125323160373426,
     33.125323160373426,
     6.771545926800889,
     6.771545926800889,
     33.125323160373426]


## Undeploy the Pipeline

Once we've finished with our demand curve demo, we'll undeploy the pipeline and give the resources back to our Kubernetes cluster.


```python
# demandcurve_pipeline.undeploy()
```

Thank you for being a part of this demonstration.  If you have additional questions, please feel free to contact us at Wallaroo.