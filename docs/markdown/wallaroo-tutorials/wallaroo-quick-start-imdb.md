This tutorial and the assets can be downloaded as part of the [Wallaroo Tutorials repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/IMDB).

## IMDB Sample

The following example demonstrates how to use Wallaroo with chained models.  In this example, we will be using information from the IMDB (Internet Movie DataBase) with a sentiment model to detect whether a given review is positive or negative.  Imagine using this to automatically scan Tweets regarding your product and finding either customers who need help or have nice things to say about your product.

Note that this example is considered a "toy" model - only the first 100 words in the review were tokenized, and the embedding is very small.

The following example is based on the [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/), and sample data can be downloaded from the [aclIMDB dataset](http://s3.amazonaws.com/text-datasets/aclImdb.zip ).

## Open a Connection to Wallaroo

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.


```python
import wallaroo
from wallaroo.object import EntityNotFoundError
```


```python
wl = wallaroo.Client()
```

To test this model, we will perform the following:

* Create a workspace for our models.
* Upload two models:
  * `embedder`: Takes pre-tokenized text documents (model input: 100 integers/datum; output 800 numbers/datum) and creates an embedding from them.
  * `sentiment`:  The second model classifies the resulting embeddings from 0 to 1, which 0 being an unfavorable review, 1 being a favorable review.
* Create a pipeline that will take incoming data and pass it to the embedder, which will pass the output to the sentiment model, and then export the final result.
* To test it, we will use information that has already been tokenized and submit it to our pipeline and gauge the results.

First we'll create a workspace for our environment, and call it `imdbworkspace`.  We'll also set up our pipeline so it's ready for our models.


```python
workspace_name = 'imdbworkspace'
pipeline_name = 'imdbpipeline'
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




<table><tr><th>name</th> <td>imdbpipeline</td></tr><tr><th>created</th> <td>2022-08-11 18:53:25.037247+00:00</td></tr><tr><th>last_updated</th> <td>2022-08-11 18:53:25.037247+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>steps</th> <td></td></tr></table>



Just to make sure, let's list our current workspace.  If everything is going right, it will show us we're in the `imdb-workspace`.


```python
wl.get_current_workspace()
```




    {'name': 'imdbworkspace', 'id': 14, 'archived': False, 'created_by': '0df04cff-3d74-426b-8dc1-5a1a97709bbd', 'created_at': '2022-08-11T18:53:24.903642+00:00', 'models': [], 'pipelines': []}



Now we'll upload our two models:

* `embedder.onnx`: This will be used to embed the tokenized documents for evaluation.
* `sentiment_model.onnx`: This will be used to analyze the review and determine if it is a positive or negative review.  The closer to 0, the more likely it is a negative review, while the closer to 1 the more likely it is to be a positive review.


```python
embedder = wl.upload_model('embedder-o', './embedder.onnx').configure()
smodel = wl.upload_model('smodel-o', './sentiment_model.onnx').configure(runtime="onnx", tensor_fields=["flatten_1"])
```

With our models uploaded, now we'll create our pipeline that will contain two steps:

* First, it runs the data through the embedder.
* Second, it applies it to our sentiment model.


```python
# now make a pipeline
imdb_pipeline.add_model_step(embedder)
imdb_pipeline.add_model_step(smodel)
```




<table><tr><th>name</th> <td>imdbpipeline</td></tr><tr><th>created</th> <td>2022-08-11 18:53:25.037247+00:00</td></tr><tr><th>last_updated</th> <td>2022-08-11 18:53:25.037247+00:00</td></tr><tr><th>deployed</th> <td>(none)</td></tr><tr><th>tags</th> <td></td></tr><tr><th>steps</th> <td></td></tr></table>



Now that we have our pipeline set up with the steps, we can deploy the pipeline.


```python
imdb_pipeline.deploy()
```

    Waiting for deployment - this will take up to 45s ..... ok





<table><tr><th>name</th> <td>imdbpipeline</td></tr><tr><th>created</th> <td>2022-08-11 18:53:25.037247+00:00</td></tr><tr><th>last_updated</th> <td>2022-08-11 18:53:25.638368+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>steps</th> <td>embedder-o</td></tr></table>



We'll check the pipeline status to verify it's deployed and the models are ready.


```python
imdb_pipeline.status()
```




    {'status': 'Running',
     'details': None,
     'engines': [{'ip': '10.244.3.173',
       'name': 'engine-845f4c47dc-xslbs',
       'status': 'Running',
       'reason': None,
       'pipeline_statuses': {'pipelines': [{'id': 'imdbpipeline',
          'status': 'Running'}]},
       'model_statuses': {'models': [{'name': 'embedder-o',
          'version': '748331fa-4023-4a79-a35f-5bafa5d975ea',
          'sha': 'd083fd87fa84451904f71ab8b9adfa88580beb92ca77c046800f79780a20b7e4',
          'status': 'Running'},
         {'name': 'smodel-o',
          'version': 'db13ab90-fb1c-4ec6-9f7d-20dad5232d1e',
          'sha': '3473ea8700fbf1a1a8bfb112554a0dde8aab36758030dcde94a9357a83fd5650',
          'status': 'Running'}]}}],
     'engine_lbs': [{'ip': '10.244.2.25',
       'name': 'engine-lb-674d8cbb8d-v5dmz',
       'status': 'Running',
       'reason': None}]}



To test this out, we'll start with a single piece of information from our data directory.


```python
results = imdb_pipeline.infer_from_file('./data/singleton.json')

results[0].data()
```

    Waiting for inference response - this will take up to 45s ... ok





    [array([[0.37142318]])]



Since that works, let's load up all 50 rows and do a full inference on each of them.  Note that Jupyter Hub has a size limitation, so for production systems the outputs should be piped out to a different output.


```python
# for the victory lap, infer on all 50 rows
results = imdb_pipeline.infer_from_file('./data/test_data.json')
results[0].data()
```




    [array([[3.71423185e-01],
            [9.65576112e-01],
            [7.60161877e-02],
            [2.46452361e-01],
            [8.63283277e-02],
            [6.39613509e-01],
            [2.47336328e-02],
            [5.02990067e-01],
            [9.34223831e-01],
            [7.17751265e-01],
            [2.04768777e-03],
            [3.55861127e-01],
            [2.48722464e-01],
            [2.73299277e-01],
            [9.60162282e-03],
            [4.95020479e-01],
            [8.30442309e-02],
            [5.34835458e-02],
            [2.74230242e-02],
            [1.26478374e-02],
            [2.39091218e-02],
            [8.63728166e-01],
            [1.57089770e-01],
            [3.46490622e-01],
            [3.56459022e-01],
            [7.97988474e-02],
            [6.78595304e-02],
            [3.17764282e-03],
            [4.39540178e-01],
            [3.33117247e-02],
            [1.46508217e-04],
            [7.39861846e-01],
            [1.51472032e-01],
            [2.41219997e-04],
            [2.69098580e-02],
            [9.06612277e-01],
            [8.55922699e-04],
            [4.60651517e-03],
            [4.51257825e-02],
            [6.71328604e-02],
            [3.86106908e-01],
            [2.73625672e-01],
            [3.87400389e-01],
            [1.92073256e-01],
            [1.40319228e-01],
            [1.50666535e-02],
            [1.26731277e-01],
            [7.53879547e-03],
            [9.44640994e-01],
            [7.55301118e-03]])]



## Undeploy

With our pipeline's work done, we'll undeploy it and give our Kubernetes environment back its resources.


```python
imdb_pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s .................................... ok





<table><tr><th>name</th> <td>imdbpipeline</td></tr><tr><th>created</th> <td>2022-08-11 18:53:25.037247+00:00</td></tr><tr><th>last_updated</th> <td>2022-08-11 18:53:25.638368+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>steps</th> <td>embedder-o</td></tr></table>



And there is our example. Please feel free to contact us at Wallaroo for if you have any questions.
