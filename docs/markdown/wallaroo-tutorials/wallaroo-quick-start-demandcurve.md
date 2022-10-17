This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/demand_curve).

## Demand Curve Pipeline Tutorial

This worksheet demonstrates a Wallaroo pipeline with data preprocessing, a model, and data postprocessing.

The model is a "demand curve" that predicts the expected number of units of a product that will be sold to a customer as a function of unit price and facts about the customer. Such models can be used for price optimization or sales volume forecasting.  This is purely a "toy" demonstration, but is useful for detailing the process of working with models and pipelines.

Data preprocessing is required to create the features used by the model. Simple postprocessing prevents nonsensical estimates (e.g. negative units sold).


## Open a Connection to Wallaroo

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

The other libraries shown below are used for this example.


```python
import json
import wallaroo
import pandas
import numpy
import conversion
from wallaroo.object import EntityNotFoundError
```


```python
wl = wallaroo.Client()
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

And with that - let's deploy our model pipeline.  This usually takes about 45 seconds for the deployment to finish.


```python
demandcurve_pipeline.deploy()
```

    Waiting for deployment - this will take up to 45s ........ ok





    {'name': 'demand-curve-pipeline', 'create_time': datetime.datetime(2022, 3, 28, 16, 28, 22, 313553, tzinfo=tzutc()), 'definition': "[{'ModelInference': {'models': [{'name': 'preprocess', 'version': 'b1f51290-ac47-4289-8a55-310507d52af5', 'sha': 'c328e2d5bf0adeb96f37687ab4da32cecf5f2cc789fa3a427ec0dbd2c3b8b663'}]}}, {'ModelInference': {'models': [{'name': 'demandcurve', 'version': '9cd1fcae-1fa1-4e12-8e67-d4a67f240a46', 'sha': '2820b42c9e778ae259918315f25afc8685ecab9967bad0a3d241e6191b414a0d'}]}}, {'ModelInference': {'models': [{'name': 'postprocess', 'version': '06e79dfe-623e-482e-95f6-bd6fa1b26264', 'sha': '4bd3109602e999a3a5013893cd2eff1a434fd9f06d6e3e681724232db6fdd40d'}]}}]"}



We can check the status of our pipeline to make sure everything was set up correctly:


```python
demandcurve_pipeline.status()
```




    {'status': 'Running',
     'details': None,
     'engines': [{'ip': '10.12.1.227',
       'name': 'engine-7cbf9b8d6d-xs64b',
       'status': 'Running',
       'reason': None,
       'pipeline_statuses': {'pipelines': [{'id': 'demand-curve-pipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'demandcurve',
          'version': '9cd1fcae-1fa1-4e12-8e67-d4a67f240a46',
          'sha': '2820b42c9e778ae259918315f25afc8685ecab9967bad0a3d241e6191b414a0d',
          'status': 'Running'},
         {'name': 'preprocess',
          'version': 'b1f51290-ac47-4289-8a55-310507d52af5',
          'sha': 'c328e2d5bf0adeb96f37687ab4da32cecf5f2cc789fa3a427ec0dbd2c3b8b663',
          'status': 'Running'},
         {'name': 'postprocess',
          'version': '06e79dfe-623e-482e-95f6-bd6fa1b26264',
          'sha': '4bd3109602e999a3a5013893cd2eff1a434fd9f06d6e3e681724232db6fdd40d',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.12.1.226',
       'name': 'engine-lb-85846c64f8-6l9rr',
       'status': 'Running',
       'reason': None}]}



Everything is ready.  Let's feed our pipeline some data.  We have some information prepared with the `daily_purchasses.csv` spreadsheet.  We'll start with just one row to make sure that everything is working correctly.


```python
# read in some purchase data
purchases = pandas.read_csv('daily_purchases.csv')

# start with a one-row data frame for testing
subsamp_raw = purchases.iloc[0:1,: ]
subsamp_raw

# create the input dictionary from the original one-line dataframe
input_dict = conversion.pandas_to_dict(subsamp_raw)

result = demandcurve_pipeline.infer(input_dict)
result
```

    Waiting for inference response - this will take up to 45s .. ok





    [InferenceResult({'check_failures': [],
      'elapsed': 479657,
      'model_name': 'postprocess',
      'model_version': '06e79dfe-623e-482e-95f6-bd6fa1b26264',
      'original_data': {'colnames': ['Date',
                                     'cust_known',
                                     'StockCode',
                                     'UnitPrice',
                                     'UnitsSold'],
                        'query': [['2010-12-01', False, '21928', 4.21, 1]]},
      'outputs': [{'Json': {'data': [{'original': {'outputs': [{'Double': {'data': [6.68025518653071],
                                                                           'dim': [1,
                                                                                   1],
                                                                           'v': 1}}]},
                                      'prediction': [6.68025518653071]}],
                            'dim': [1],
                            'v': 1}}],
      'pipeline_name': 'demand-curve-pipeline',
      'time': 1648484916799})]



We can see from the `prediction` field that the demand curve has a predicted slope of 6.68 from our sample data.  We can isolate that by specifying just the data output below.


```python
result[0].data()
```




    [array([6.68025519])]



# Bulk Inference

The initial test went perfectly.  Now let's throw some more data into our pipeline.  We'll draw 10 random rows from our spreadsheet, perform an inference from that, and then display the results and the logs showing the pipeline's actions.


```python
# Let's do 10 rows at once (drawn randomly)
ix = numpy.random.choice(purchases.shape[0], size=10, replace=False)
output = demandcurve_pipeline.infer(conversion.pandas_to_dict(purchases.iloc[ix,: ]))
```


```python
output[0].data()
```




    [array([33.12532316,  6.77154593,  6.77154593, 40.57067889, 40.57067889,
             6.77154593, 33.12532316,  6.77154593,  9.11087115, 40.57067889])]



## Undeploy the Pipeline

Once we've finished with our demand curve demo, we'll undeploy the pipeline and give the resources back to our Kubernetes cluster.


```python
demandcurve_pipeline.undeploy()
```

Thank you for being a part of this demonstration.  If you have additional questions, please feel free to contact us at Wallaroo.
