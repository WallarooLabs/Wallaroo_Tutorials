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

```python
import json
import wallaroo
from wallaroo.object import EntityNotFoundError
import pandas
import numpy
import conversion
from wallaroo.object import EntityNotFoundError
import pyarrow as pa

# used to display dataframe information without truncating
from IPython.display import display
import pandas as pd
pd.set_option('display.max_colwidth', None)

# ignoring warnings for demonstration
import warnings
warnings.filterwarnings('ignore')
```

### Connect to the Wallaroo Instance

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
# Login through local Wallaroo instance

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
        pipeline = wl.pipelines_by_name(name)[0]
    except EntityNotFoundError:
        pipeline = wl.build_pipeline(name)
    return pipeline
```

```python
workspace = get_workspace(workspace_name)

wl.set_current_workspace(workspace)

demandcurve_pipeline = get_pipeline(pipeline_name)
demandcurve_pipeline
```

<table><tr><th>name</th> <td>demandcurvepipeline</td></tr><tr><th>created</th> <td>2023-09-11 18:28:08.036841+00:00</td></tr><tr><th>last_updated</th> <td>2023-09-11 18:28:08.036841+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>6e7452b9-1464-488b-9a28-85ff765f18d6</td></tr><tr><th>steps</th> <td></td></tr><tr><th>published</th> <td>False</td></tr></table>

With our workspace established, we'll upload two models:

* `demand_curve_v1.onnx`: Our demand_curve model.  We'll store the upload configuration into `demand_curve_model`.
* `postprocess.py`: A Python step that will zero out any negative values and return the output variable as "prediction".

Note that the order we upload our models isn't important - we'll be establishing the actual process of moving data from one model to the next when we set up our pipeline.

```python
demand_curve_model = wl.upload_model(model_name, model_file_name, framework=wallaroo.framework.Framework.ONNX)

preprocess_input_schema = pa.schema([
    pa.field('Date', pa.string()),
    pa.field('cust_known', pa.bool_()),
    pa.field('StockCode', pa.int32()),
    pa.field('UnitPrice', pa.float32()),
    pa.field('UnitsSold', pa.int32())
])

preprocess_input_output = pa.schema([
    pa.field('tensor', pa.list_(pa.float64()))
])

preprocess_step = (wl.upload_model('curve-preprocess', 
                        './preprocess.py', 
                        framework=wallaroo.framework.Framework.PYTHON)
                        .configure(
                            'python', 
                            input_schema=preprocess_input_schema, 
                            output_schema=preprocess_input_output
                        )
        )

input_schema = pa.schema([
    pa.field('variable', pa.list_(pa.float64()))
])

output_schema = pa.schema([
    pa.field('prediction', pa.list_(pa.float64()))
])

step = (wl.upload_model('curve-postprocess', 
                        './postprocess.py', 
                        framework=wallaroo.framework.Framework.PYTHON)
                        .configure(
                            'python', 
                            input_schema=input_schema, 
                            output_schema=output_schema
                        )
        )
```

With our models uploaded, we're going to create our own pipeline and give it three steps:

* First, we apply the data to our `demand_curve_model`.
* And finally, we prepare our data for output with the `module_post`.

```python
# now make a pipeline
demandcurve_pipeline.undeploy()
demandcurve_pipeline.clear()
demandcurve_pipeline.add_model_step(preprocess_step)
demandcurve_pipeline.add_model_step(demand_curve_model)
demandcurve_pipeline.add_model_step(step)
```

<table><tr><th>name</th> <td>demandcurvepipeline</td></tr><tr><th>created</th> <td>2023-09-11 18:28:08.036841+00:00</td></tr><tr><th>last_updated</th> <td>2023-09-11 18:28:08.036841+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>6e7452b9-1464-488b-9a28-85ff765f18d6</td></tr><tr><th>steps</th> <td></td></tr><tr><th>published</th> <td>False</td></tr></table>

And with that - let's deploy our model pipeline.  This usually takes about 45 seconds for the deployment to finish.

```python
deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(1).memory("1Gi").build()
demandcurve_pipeline.deploy(deployment_config=deploy_config)
```

<table><tr><th>name</th> <td>demandcurvepipeline</td></tr><tr><th>created</th> <td>2023-09-11 18:28:08.036841+00:00</td></tr><tr><th>last_updated</th> <td>2023-09-11 18:28:14.101496+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>cf1e4357-c98c-490d-822f-8c252a43ff99, 6e7452b9-1464-488b-9a28-85ff765f18d6</td></tr><tr><th>steps</th> <td>curve-preprocess</td></tr><tr><th>published</th> <td>False</td></tr></table>

We can check the status of our pipeline to make sure everything was set up correctly:

```python
demandcurve_pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.3.164',
       'name': 'engine-68597cfb4-6fxxd',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'demandcurvepipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'curve-preprocess',
          'version': '15ea1ce0-2091-4f18-8759-266d113cfa86',
          'sha': '3b2b32fb6408ccee5a1886ef5cdb493692080ff6699f49de518b20d9a6f6a133',
          'status': 'Running'},
         {'name': 'demandcurvemodel',
          'version': 'c9a1b515-8302-49f2-a772-3a32a182bad3',
          'sha': '2820b42c9e778ae259918315f25afc8685ecab9967bad0a3d241e6191b414a0d',
          'status': 'Running'},
         {'name': 'curve-postprocess',
          'version': '76f7f617-8e29-4aa8-a06e-563228581d79',
          'sha': 'ecf1a555bb27bcf62bfa42922cf69db23e7188f456015fe8299f02867c3167b2',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.4.200',
       'name': 'engine-lb-584f54c899-lqm67',
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
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>cust_known</th>
      <th>StockCode</th>
      <th>UnitPrice</th>
      <th>UnitsSold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2010-12-01</td>
      <td>False</td>
      <td>21928</td>
      <td>4.21</td>
      <td>1</td>
    </tr>
  </tbody>
</table>

```python
result = demandcurve_pipeline.infer(subsamp_raw)
display(result)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.Date</th>
      <th>in.StockCode</th>
      <th>in.UnitPrice</th>
      <th>in.UnitsSold</th>
      <th>in.cust_known</th>
      <th>out.prediction</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-09-11 18:28:28.627</td>
      <td>2010-12-01</td>
      <td>21928</td>
      <td>4.21</td>
      <td>1</td>
      <td>False</td>
      <td>[6.68025518653071]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

We can see from the `out.prediction` field that the demand curve has a predicted slope of 6.68 from our sample data.  We can isolate that by specifying just the data output below.

```python
display(result.loc[0, ['out.prediction']][0])
```

    [6.68025518653071]

# Bulk Inference

The initial test went perfectly.  Now let's throw some more data into our pipeline.  We'll draw 10 random rows from our spreadsheet, perform an inference from that, and then display the results and the logs showing the pipeline's actions.

```python
ix = numpy.random.choice(purchases.shape[0], size=10, replace=False)
converted = conversion.pandas_to_dict(purchases.iloc[ix,: ])
test_df = pd.DataFrame(converted['query'], columns=converted['colnames'])
display(test_df)

output = demandcurve_pipeline.infer(test_df)
display(output)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>cust_known</th>
      <th>StockCode</th>
      <th>UnitPrice</th>
      <th>UnitsSold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-09-15</td>
      <td>False</td>
      <td>20713</td>
      <td>4.13</td>
      <td>17</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-11-30</td>
      <td>True</td>
      <td>85099C</td>
      <td>2.08</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-10-13</td>
      <td>False</td>
      <td>85099B</td>
      <td>4.13</td>
      <td>15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-08-11</td>
      <td>True</td>
      <td>22411</td>
      <td>2.08</td>
      <td>30</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-08-10</td>
      <td>False</td>
      <td>23200</td>
      <td>4.13</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2011-11-03</td>
      <td>False</td>
      <td>22385</td>
      <td>4.13</td>
      <td>3</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2011-09-19</td>
      <td>True</td>
      <td>85099B</td>
      <td>1.79</td>
      <td>400</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2011-09-18</td>
      <td>True</td>
      <td>20712</td>
      <td>2.08</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2011-05-09</td>
      <td>False</td>
      <td>23201</td>
      <td>4.13</td>
      <td>6</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2011-11-17</td>
      <td>False</td>
      <td>85099B</td>
      <td>4.13</td>
      <td>6</td>
    </tr>
  </tbody>
</table>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.Date</th>
      <th>in.StockCode</th>
      <th>in.UnitPrice</th>
      <th>in.UnitsSold</th>
      <th>in.cust_known</th>
      <th>out.prediction</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-09-11 18:28:29.105</td>
      <td>2011-09-15</td>
      <td>20713</td>
      <td>4.13</td>
      <td>17</td>
      <td>False</td>
      <td>[6.771545926800889]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-09-11 18:28:29.105</td>
      <td>2011-11-30</td>
      <td>85099C</td>
      <td>2.08</td>
      <td>12</td>
      <td>True</td>
      <td>[33.125323160373426]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-09-11 18:28:29.105</td>
      <td>2011-10-13</td>
      <td>85099B</td>
      <td>4.13</td>
      <td>15</td>
      <td>False</td>
      <td>[6.771545926800889]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-09-11 18:28:29.105</td>
      <td>2011-08-11</td>
      <td>22411</td>
      <td>2.08</td>
      <td>30</td>
      <td>True</td>
      <td>[33.125323160373426]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-09-11 18:28:29.105</td>
      <td>2011-08-10</td>
      <td>23200</td>
      <td>4.13</td>
      <td>4</td>
      <td>False</td>
      <td>[6.771545926800889]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2023-09-11 18:28:29.105</td>
      <td>2011-11-03</td>
      <td>22385</td>
      <td>4.13</td>
      <td>3</td>
      <td>False</td>
      <td>[6.771545926800889]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2023-09-11 18:28:29.105</td>
      <td>2011-09-19</td>
      <td>85099B</td>
      <td>1.79</td>
      <td>400</td>
      <td>True</td>
      <td>[49.73419363867448]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2023-09-11 18:28:29.105</td>
      <td>2011-09-18</td>
      <td>20712</td>
      <td>2.08</td>
      <td>2</td>
      <td>True</td>
      <td>[33.125323160373426]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2023-09-11 18:28:29.105</td>
      <td>2011-05-09</td>
      <td>23201</td>
      <td>4.13</td>
      <td>6</td>
      <td>False</td>
      <td>[6.771545926800889]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2023-09-11 18:28:29.105</td>
      <td>2011-11-17</td>
      <td>85099B</td>
      <td>4.13</td>
      <td>6</td>
      <td>False</td>
      <td>[6.771545926800889]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

## Undeploy the Pipeline

Once we've finished with our demand curve demo, we'll undeploy the pipeline and give the resources back to our Kubernetes cluster.

```python
demandcurve_pipeline.undeploy()
```

<table><tr><th>name</th> <td>demandcurvepipeline</td></tr><tr><th>created</th> <td>2023-09-11 18:28:08.036841+00:00</td></tr><tr><th>last_updated</th> <td>2023-09-11 18:28:14.101496+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>cf1e4357-c98c-490d-822f-8c252a43ff99, 6e7452b9-1464-488b-9a28-85ff765f18d6</td></tr><tr><th>steps</th> <td>curve-preprocess</td></tr><tr><th>published</th> <td>False</td></tr></table>

Thank you for being a part of this demonstration.  If you have additional questions, please feel free to contact us at Wallaroo.
