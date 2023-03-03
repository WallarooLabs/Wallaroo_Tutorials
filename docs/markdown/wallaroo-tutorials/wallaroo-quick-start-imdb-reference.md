This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/wallaroo-model-cookbooks/IMDB).

## IMDB Sample

The following example demonstrates how to use Wallaroo with chained models.  In this example, we will be using information from the IMDB (Internet Movie DataBase) with a sentiment model to detect whether a given review is positive or negative.  Imagine using this to automatically scan Tweets regarding your product and finding either customers who need help or have nice things to say about your product.

Note that this example is considered a "toy" model - only the first 100 words in the review were tokenized, and the embedding is very small.

The following example is based on the [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/), and sample data can be downloaded from the [aclIMDB dataset](http://s3.amazonaws.com/text-datasets/aclImdb.zip ).

## Open a Connection to Wallaroo

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  If logging in externally, update the `wallarooPrefix` and `wallarooSuffix` variables with the proper DNS information.  For more information on Wallaroo DNS settings, see the [Wallaroo DNS Integration Guide](https://docs.wallaroo.ai/wallaroo-operations-guide/wallaroo-configuration/wallaroo-dns-guide/).

```python
import wallaroo
from wallaroo.object import EntityNotFoundError
from IPython.display import display

# used to display dataframe information without truncating
import pandas as pd
pd.set_option('display.max_colwidth', None)
```

```python
print(wallaroo.__version__)
```

    2023.1.0rc1

### Arrow Support

As of the 2023.1 release, Wallaroo provides support for dataframe and Arrow for inference inputs.  This tutorial allows users to adjust their experience based on whether they have enabled Arrow support in their Wallaroo instance or not.

If Arrow support has been enabled, `arrowEnabled=True`. If disabled or you're not sure, set it to `arrowEnabled=False`

The examples below will be shown in an arrow enabled environment.

```python
import os
# Only set the below to make the OS environment ARROW_ENABLED to TRUE.  Otherwise, leave as is.
# os.environ["ARROW_ENABLED"]="True"

if "ARROW_ENABLED" not in os.environ or os.environ["ARROW_ENABLED"] == "False":
    arrowEnabled = False
else:
    arrowEnabled = True
print(arrowEnabled)
```

    True

```python
# Login through local Wallaroo instance

wl = wallaroo.Client()

# SSO login through keycloak

# wallarooPrefix = "YOUR PREFIX"
# wallarooSuffix = "YOUR SUFFIX"

# wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}.api.{wallarooSuffix}", 
#                     auth_endpoint=f"https://{wallarooPrefix}.keycloak.{wallarooSuffix}", 
#                     auth_type="sso")
```

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

<table><tr><th>name</th> <td>vzfximdbpipeline</td></tr><tr><th>created</th> <td>2023-02-27 18:54:24.344787+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-27 18:54:24.344787+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>1f3f3e66-56ec-401b-b550-5f06655f4770</td></tr><tr><th>steps</th> <td></td></tr></table>
{{</table>}}

Just to make sure, let's list our current workspace.  If everything is going right, it will show us we're in the `imdb-workspace`.

```python
wl.get_current_workspace()
```

    {'name': 'vzfximdbworkspace', 'id': 18, 'archived': False, 'created_by': '435da905-31e2-4e74-b423-45c38edb5889', 'created_at': '2023-02-27T18:54:23.351035+00:00', 'models': [], 'pipelines': [{'name': 'vzfximdbpipeline', 'create_time': datetime.datetime(2023, 2, 27, 18, 54, 24, 344787, tzinfo=tzutc()), 'definition': '[]'}]}

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

<table><tr><th>name</th> <td>vzfximdbpipeline</td></tr><tr><th>created</th> <td>2023-02-27 18:54:24.344787+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-27 18:54:24.344787+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>1f3f3e66-56ec-401b-b550-5f06655f4770</td></tr><tr><th>steps</th> <td></td></tr></table>
{{</table>}}

Now that we have our pipeline set up with the steps, we can deploy the pipeline.

```python
imdb_pipeline.deploy()
```

<table><tr><th>name</th> <td>vzfximdbpipeline</td></tr><tr><th>created</th> <td>2023-02-27 18:54:24.344787+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-27 18:54:30.220884+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>bc2644e2-d780-4515-ab6a-044f4e6fd27e, 1f3f3e66-56ec-401b-b550-5f06655f4770</td></tr><tr><th>steps</th> <td>vzfxembedder-o</td></tr></table>
{{</table>}}

We'll check the pipeline status to verify it's deployed and the models are ready.

```python
imdb_pipeline.status()
```

    {'status': 'Running',
     'details': [],
     'engines': [{'ip': '10.244.0.45',
       'name': 'engine-57d4bf6d5d-j77n8',
       'status': 'Running',
       'reason': None,
       'details': [],
       'pipeline_statuses': {'pipelines': [{'id': 'vzfximdbpipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'vzfxsmodel-o',
          'version': '69f78f5e-633d-49d8-b46e-d57563669e7b',
          'sha': '3473ea8700fbf1a1a8bfb112554a0dde8aab36758030dcde94a9357a83fd5650',
          'status': 'Running'},
         {'name': 'vzfxembedder-o',
          'version': 'b50749b8-af50-4404-9f95-ffd084ee9416',
          'sha': 'd083fd87fa84451904f71ab8b9adfa88580beb92ca77c046800f79780a20b7e4',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.2.18',
       'name': 'engine-lb-ddd995646-njpqc',
       'status': 'Running',
       'reason': None,
       'details': []}],
     'sidekicks': []}

```python
print(arrowEnabled)
```

    True

To test this out, we'll start with a single piece of information from our data directory.

```python
if arrowEnabled is True:
    # using Dataframe JSON
    results = results = imdb_pipeline.infer_from_file('./data/singleton.df.json')
    display(results)
else:
    # pre-arrow Wallaroo JSON
    results = imdb_pipeline.infer_from_file('./data/singleton.json')
    display(results[0].data())
```

{{<table "table table-bordered">}}
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
      <td>2023-02-27 18:54:47.498</td>
      <td>[1607.0, 2635.0, 5749.0, 199.0, 49.0, 351.0, 16.0, 2919.0, 159.0, 5092.0, 2457.0, 8.0, 11.0, 1252.0, 507.0, 42.0, 287.0, 316.0, 15.0, 65.0, 136.0, 2.0, 133.0, 16.0, 4311.0, 131.0, 286.0, 153.0, 5.0, 2826.0, 175.0, 54.0, 548.0, 48.0, 1.0, 17.0, 9.0, 183.0, 1.0, 111.0, 15.0, 1.0, 17.0, 284.0, 982.0, 18.0, 28.0, 211.0, 1.0, 1382.0, 8.0, 146.0, 1.0, 19.0, 12.0, 9.0, 13.0, 21.0, 1898.0, 122.0, 14.0, 70.0, 14.0, 9.0, 97.0, 25.0, 74.0, 1.0, 189.0, 12.0, 9.0, 6.0, 31.0, 3.0, 244.0, 2497.0, 3659.0, 2.0, 665.0, 2497.0, 63.0, 180.0, 1.0, 17.0, 6.0, 287.0, 3.0, 646.0, 44.0, 15.0, 161.0, 50.0, 71.0, 438.0, 351.0, 31.0, 5749.0, 2.0, 0.0, 0.0]</td>
      <td>[0.37142318]</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
{{</table>}}


Since that works, let's load up all 50 rows and do a full inference on each of them.  Note that Jupyter Hub has a size limitation, so for production systems the outputs should be piped out to a different output.

```python
if arrowEnabled is True:
    # using Dataframe JSON
    results = imdb_pipeline.infer_from_file('./data/test_data.df.json')
    # just display the first row for space
    display(results.loc[0,:])
else:
    # pre-arrow Wallaroo JSON
    results = imdb_pipeline.infer_from_file('./data/test_data.json')
    # just display the first row for space
    display(results[0].data())
```

    time                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                2023-02-27 18:54:48.032000
    in.tensor         [1607.0, 2635.0, 5749.0, 199.0, 49.0, 351.0, 16.0, 2919.0, 159.0, 5092.0, 2457.0, 8.0, 11.0, 1252.0, 507.0, 42.0, 287.0, 316.0, 15.0, 65.0, 136.0, 2.0, 133.0, 16.0, 4311.0, 131.0, 286.0, 153.0, 5.0, 2826.0, 175.0, 54.0, 548.0, 48.0, 1.0, 17.0, 9.0, 183.0, 1.0, 111.0, 15.0, 1.0, 17.0, 284.0, 982.0, 18.0, 28.0, 211.0, 1.0, 1382.0, 8.0, 146.0, 1.0, 19.0, 12.0, 9.0, 13.0, 21.0, 1898.0, 122.0, 14.0, 70.0, 14.0, 9.0, 97.0, 25.0, 74.0, 1.0, 189.0, 12.0, 9.0, 6.0, 31.0, 3.0, 244.0, 2497.0, 3659.0, 2.0, 665.0, 2497.0, 63.0, 180.0, 1.0, 17.0, 6.0, 287.0, 3.0, 646.0, 44.0, 15.0, 161.0, 50.0, 71.0, 438.0, 351.0, 31.0, 5749.0, 2.0, 0.0, 0.0]
    out.dense_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       [0.37142318]
    check_failures                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               0
    Name: 0, dtype: object

## Undeploy

With our pipeline's work done, we'll undeploy it and give our Kubernetes environment back its resources.

```python
imdb_pipeline.undeploy()
```

<table><tr><th>name</th> <td>vzfximdbpipeline</td></tr><tr><th>created</th> <td>2023-02-27 18:54:24.344787+00:00</td></tr><tr><th>last_updated</th> <td>2023-02-27 18:54:30.220884+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>bc2644e2-d780-4515-ab6a-044f4e6fd27e, 1f3f3e66-56ec-401b-b550-5f06655f4770</td></tr><tr><th>steps</th> <td>vzfxembedder-o</td></tr></table>
{{</table>}}

And there is our example. Please feel free to contact us at Wallaroo for if you have any questions.
