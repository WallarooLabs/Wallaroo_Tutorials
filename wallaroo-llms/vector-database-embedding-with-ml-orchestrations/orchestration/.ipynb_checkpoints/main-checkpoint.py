import wallaroo
from wallaroo.pipeline   import Pipeline
from wallaroo.deployment_config import DeploymentConfigBuilder
from wallaroo.framework import Framework
from wallaroo.engine_config import Architecture

import pymongo

import numpy as np
import pandas as pd

wl = wallaroo.Client()

if wl.in_task():
    arguments = wl.task_args()

    # arguments is a key/value pair, set the workspace and pipeline name
    connection_name = arguments['connection_name']   
    pipeline_name = arguments['pipeline_name']
    workspace_name = arguments['workspace_name']
    write_db = arguments["write_db"]

# False:  We're not in a Task, so set the pipeline manually
else:
    connection_name = 'mongodb_atlas'
    pipeline_name = 'byop-bge-pipe-base-v2'
    workspace_name = 'embedding-computation'
    use_db = False

# Set workspace and get pipeline
workspace = wl.get_workspace(workspace_name)
_ = wl.set_current_workspace(workspace)

pipeline = wl.get_pipeline(pipeline_name)

# Get Connection
connect = wl.get_connection(name=connection_name)

client = pymongo.MongoClient(connect.details()["uri"])
db = client.sample_mflix
collection = db.movies

if write_db:
    for doc in collection.find({'plot':{"$exists": True}}):
        myquery = { 'plot': doc['plot']}
        
        data = pd.DataFrame({'text': doc['plot']})
        embedding = pipeline.infer(data)['out.embedding']
        update = { '$set': { 'plot_embedding_hf': embedding } }
        
        collection.updateOne(myquery, update)
        
else:
    # Query Vector DB
    texts = []
    for doc in collection.find({'plot':{"$exists": True}}).limit(10):
        texts.append(doc['plot'])

    # Use pipeline to infer on data    
    data = pd.DataFrame({'text': texts})
    result = pipeline.infer(data, timeout=10000)
    
    # Log results
    print(result)
