This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-model-cookbooks/IMDB).

## IMDB Sample

The following example demonstrates how to use Wallaroo with chained models.  In this example, we will be using information from the IMDB (Internet Movie DataBase) with a sentiment model to detect whether a given review is positive or negative.  Imagine using this to automatically scan Tweets regarding your product and finding either customers who need help or have nice things to say about your product.

Note that this example is considered a "toy" model - only the first 100 words in the review were tokenized, and the embedding is very small.

The following example is based on the [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/), and sample data can be downloaded from the [aclIMDB dataset](http://s3.amazonaws.com/text-datasets/aclImdb.zip ).

## Prerequisites

* An installed Wallaroo instance.
* The following Python libraries installed:
  * `os`
  * [`wallaroo`](https://pypi.org/project/wallaroo/): The Wallaroo SDK. Included with the Wallaroo JupyterHub service by default.
  * [`pandas`](https://pypi.org/project/pandas/): Pandas, mainly used for Pandas DataFrame
  * [`pyarrow`](https://pypi.org/project/pyarrow/): PyArrow for Apache Arrow support
  * [`polars`](https://pypi.org/project/polars/): Polars for DataFrame with native Apache Arrow support

## Open a Connection to Wallaroo

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  If logging in externally, update the `wallarooPrefix` and `wallarooSuffix` variables with the proper DNS information.  For more information on Wallaroo DNS settings, see the [Wallaroo DNS Integration Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-dns-guide/).


```python
import wallaroo
from wallaroo.object import EntityNotFoundError

# to display dataframe tables
from IPython.display import display
# used to display dataframe information without truncating
import pandas as pd
pd.set_option('display.max_colwidth', None)
import pyarrow as pa

import polars as pl

import os
# Used for the Wallaroo SDK version 2023.1
os.environ["ARROW_ENABLED"]="True"
```


```python
print(wallaroo.__version__)
```

    2023.1.0



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

    Please log into the following URL in a web browser:
    
    	https://wallaroo.keycloak.example.com/auth/realms/master/device?user_code=UHKQ-BZXQ
    
    Login successful!


To test this model, we will perform the following:

* Create a workspace for our models.
* Upload two models:
  * `embedder`: Takes pre-tokenized text documents (model input: 100 integers/datum; output 800 numbers/datum) and creates an embedding from them.
  * `sentiment`:  The second model classifies the resulting embeddings from 0 to 1, which 0 being an unfavorable review, 1 being a favorable review.
* Create a pipeline that will take incoming data and pass it to the embedder, which will pass the output to the sentiment model, and then export the final result.
* To test it, we will use information that has already been tokenized and submit it to our pipeline and gauge the results.

Just for the sake of this tutorial, we'll use the SDK below to create our workspace , assign as our **current workspace**, then display all of the workspaces we have at the moment.  We'll also set up for our models and pipelines down the road, so we have one spot to change names to whatever fits your organization's standards best.

To allow this tutorial to be run multiple times or by multiple users in the same Wallaroo instance, a random 4 character prefix will be added to the workspace, pipeline, and model.

When we create our new workspace, we'll save it in the Python variable `workspace` so we can refer to it as needed.

First we'll create a workspace for our environment, and call it `imdbworkspace`.  We'll also set up our pipeline so it's ready for our models.


```python
import string
import random

# make a random 4 character prefix
prefix= ''.join(random.choice(string.ascii_lowercase) for i in range(4))
workspace_name = f'{prefix}imdbworkspace'
pipeline_name = f'{prefix}imdbpipeline'
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

imdb_pipeline = get_pipeline(pipeline_name)
imdb_pipeline
```




<table><tr><th>name</th> <td>uqtvimdbpipeline</td></tr><tr><th>created</th> <td>2023-03-29 15:20:04.617760+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-29 15:20:04.617760+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>f700d6b9-96e8-4e58-8c67-eb579986122f</td></tr><tr><th>steps</th> <td></td></tr></table>



Just to make sure, let's list our current workspace.  If everything is going right, it will show us we're in the `imdb-workspace`.


```python
wl.get_current_workspace()
```




    {'name': 'uqtvimdbworkspace', 'id': 20, 'archived': False, 'created_by': '56d97480-bb64-4575-acb6-f93d05652e86', 'created_at': '2023-03-29T15:20:03.558918+00:00', 'models': [], 'pipelines': [{'name': 'uqtvimdbpipeline', 'create_time': datetime.datetime(2023, 3, 29, 15, 20, 4, 617760, tzinfo=tzutc()), 'definition': '[]'}]}



Now we'll upload our two models:

* `embedder.onnx`: This will be used to embed the tokenized documents for evaluation.
* `sentiment_model.onnx`: This will be used to analyze the review and determine if it is a positive or negative review.  The closer to 0, the more likely it is a negative review, while the closer to 1 the more likely it is to be a positive review.


```python
embedder = wl.upload_model(f'{prefix}embedder-o', './embedder.onnx').configure()
smodel = wl.upload_model(f'{prefix}smodel-o', './sentiment_model.onnx').configure(runtime="onnx", tensor_fields=["flatten_1"])
```

With our models uploaded, now we'll create our pipeline that will contain two steps:

* First, it runs the data through the embedder.
* Second, it applies it to our sentiment model.


```python
# now make a pipeline
imdb_pipeline.add_model_step(embedder)
imdb_pipeline.add_model_step(smodel)
```




<table><tr><th>name</th> <td>uqtvimdbpipeline</td></tr><tr><th>created</th> <td>2023-03-29 15:20:04.617760+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-29 15:20:04.617760+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>f700d6b9-96e8-4e58-8c67-eb579986122f</td></tr><tr><th>steps</th> <td></td></tr></table>



Now that we have our pipeline set up with the steps, we can deploy the pipeline.


```python
imdb_pipeline.deploy()
```




<table><tr><th>name</th> <td>uqtvimdbpipeline</td></tr><tr><th>created</th> <td>2023-03-29 15:20:04.617760+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-29 15:20:10.822034+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>3c0bd524-d31e-48af-8c9d-90bc440dbf47, f700d6b9-96e8-4e58-8c67-eb579986122f</td></tr><tr><th>steps</th> <td>uqtvembedder-o</td></tr></table>



We'll check the pipeline status to verify it's deployed and the models are ready.


```python
imdb_pipeline.status()
```




    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.0.9',
       'name': 'engine-f7bbc47b9-fnxml',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'uqtvimdbpipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'uqtvsmodel-o',
          'version': 'f3b6d621-ff05-4700-8ea7-4630e3c214aa',
          'sha': '3473ea8700fbf1a1a8bfb112554a0dde8aab36758030dcde94a9357a83fd5650',
          'status': 'Running'},
         {'name': 'uqtvembedder-o',
          'version': 'd985a2c6-d212-4ba2-98f4-ace99d45c0db',
          'sha': 'd083fd87fa84451904f71ab8b9adfa88580beb92ca77c046800f79780a20b7e4',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.0.8',
       'name': 'engine-lb-ddd995646-xvjnw',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}



To test this out, we'll start with a single piece of information from our data directory.


```python
singleton = pd.DataFrame.from_records(
    [
    {
        "tensor":[
            1607.0,
            2635.0,
            5749.0,
            199.0,
            49.0,
            351.0,
            16.0,
            2919.0,
            159.0,
            5092.0,
            2457.0,
            8.0,
            11.0,
            1252.0,
            507.0,
            42.0,
            287.0,
            316.0,
            15.0,
            65.0,
            136.0,
            2.0,
            133.0,
            16.0,
            4311.0,
            131.0,
            286.0,
            153.0,
            5.0,
            2826.0,
            175.0,
            54.0,
            548.0,
            48.0,
            1.0,
            17.0,
            9.0,
            183.0,
            1.0,
            111.0,
            15.0,
            1.0,
            17.0,
            284.0,
            982.0,
            18.0,
            28.0,
            211.0,
            1.0,
            1382.0,
            8.0,
            146.0,
            1.0,
            19.0,
            12.0,
            9.0,
            13.0,
            21.0,
            1898.0,
            122.0,
            14.0,
            70.0,
            14.0,
            9.0,
            97.0,
            25.0,
            74.0,
            1.0,
            189.0,
            12.0,
            9.0,
            6.0,
            31.0,
            3.0,
            244.0,
            2497.0,
            3659.0,
            2.0,
            665.0,
            2497.0,
            63.0,
            180.0,
            1.0,
            17.0,
            6.0,
            287.0,
            3.0,
            646.0,
            44.0,
            15.0,
            161.0,
            50.0,
            71.0,
            438.0,
            351.0,
            31.0,
            5749.0,
            2.0,
            0.0,
            0.0
        ]
    }
]
)
results = imdb_pipeline.infer(singleton)
display(results)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>in.tensor</th>
      <th>out.dense_1</th>
      <th>check_failures</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-03-29 15:20:52.242</td>
      <td>[1607.0, 2635.0, 5749.0, 199.0, 49.0, 351.0, 16.0, 2919.0, 159.0, 5092.0, 2457.0, 8.0, 11.0, 1252.0, 507.0, 42.0, 287.0, 316.0, 15.0, 65.0, 136.0, 2.0, 133.0, 16.0, 4311.0, 131.0, 286.0, 153.0, 5.0, 2826.0, 175.0, 54.0, 548.0, 48.0, 1.0, 17.0, 9.0, 183.0, 1.0, 111.0, 15.0, 1.0, 17.0, 284.0, 982.0, 18.0, 28.0, 211.0, 1.0, 1382.0, 8.0, 146.0, 1.0, 19.0, 12.0, 9.0, 13.0, 21.0, 1898.0, 122.0, 14.0, 70.0, 14.0, 9.0, 97.0, 25.0, 74.0, 1.0, 189.0, 12.0, 9.0, 6.0, 31.0, 3.0, 244.0, 2497.0, 3659.0, 2.0, 665.0, 2497.0, 63.0, 180.0, 1.0, 17.0, 6.0, 287.0, 3.0, 646.0, 44.0, 15.0, 161.0, 50.0, 71.0, 438.0, 351.0, 31.0, 5749.0, 2.0, 0.0, 0.0]</td>
      <td>[0.37142318]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


Since that works, let's load up all 50,000 rows and do a full inference on each of them via an Apache Arrow file.  Wallaroo pipeline inferences use Apache Arrow as their core data type, making this inference fast.  

We'll do a demonstration with both pandas DataFrame, and a `polars` DataFrame, then display the first 5 results in either case.


```python
results = imdb_pipeline.infer_from_file('./data/test_data_50K.arrow')
```


```python
# using pandas DataFrame

outputs = results.to_pandas()
display(outputs.loc[:5, ["time","out.dense_1"]])
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>out.dense_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-03-29 15:21:08.307</td>
      <td>[0.8980188]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-03-29 15:21:08.307</td>
      <td>[0.056596935]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-03-29 15:21:08.307</td>
      <td>[0.9260802]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-03-29 15:21:08.307</td>
      <td>[0.926919]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-03-29 15:21:08.307</td>
      <td>[0.6618577]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2023-03-29 15:21:08.307</td>
      <td>[0.48736304]</td>
    </tr>
  </tbody>
</table>
</div>



```python
# using polars DataFrame

outputs = pl.from_arrow(results)
display(outputs.head(5))
```


<div><style>
.dataframe > thead > tr > th,
.dataframe > tbody > tr > td {
  text-align: right;
}
</style>
<small>shape: (5, 4)</small><table border="1" class="dataframe"><thead><tr><th>time</th><th>in.tensor</th><th>out.dense_1</th><th>check_failures</th></tr><tr><td>datetime[ms]</td><td>list[f32]</td><td>list[f32]</td><td>i8</td></tr></thead><tbody><tr><td>2023-03-29 15:21:08.307</td><td>[11.0, 6.0, … 0.0]</td><td>[0.898019]</td><td>0</td></tr><tr><td>2023-03-29 15:21:08.307</td><td>[54.0, 548.0, … 20.0]</td><td>[0.056597]</td><td>0</td></tr><tr><td>2023-03-29 15:21:08.307</td><td>[1.0, 9259.0, … 1.0]</td><td>[0.92608]</td><td>0</td></tr><tr><td>2023-03-29 15:21:08.307</td><td>[10.0, 25.0, … 0.0]</td><td>[0.926919]</td><td>0</td></tr><tr><td>2023-03-29 15:21:08.307</td><td>[10.0, 37.0, … 0.0]</td><td>[0.661858]</td><td>0</td></tr></tbody></table></div>


## Undeploy

With our pipeline's work done, we'll undeploy it and give our Kubernetes environment back its resources.


```python
imdb_pipeline.undeploy()
```




<table><tr><th>name</th> <td>uqtvimdbpipeline</td></tr><tr><th>created</th> <td>2023-03-29 15:20:04.617760+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-29 15:20:10.822034+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>3c0bd524-d31e-48af-8c9d-90bc440dbf47, f700d6b9-96e8-4e58-8c67-eb579986122f</td></tr><tr><th>steps</th> <td>uqtvembedder-o</td></tr></table>



And there is our example. Please feel free to contact us at Wallaroo for if you have any questions.
