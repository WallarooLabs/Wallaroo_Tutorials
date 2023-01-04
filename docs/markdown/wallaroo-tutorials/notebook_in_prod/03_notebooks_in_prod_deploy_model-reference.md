For more details on this tutorial's setup and process, see `00_Introduction.ipynb`.

# Stage 3: Deploy the Model in Wallaroo
 
In this stage, we upload the trained model and the processing steps to Wallaroo, then set up and deploy the inference pipeline. 

Once deployed we can feed the newest batch of data to the pipeline, do the inferences and write the results to a results table.

For clarity in this demo, we have split the training/upload task into two notebooks:

* `02_automated_training_process.ipynb`: Train and pickle ML model.
* `03_deploy_model.ipynb`: Upload the model to Wallaroo and deploy into a pipeline.

Assuming no changes are made to the structure of the model, these two notebooks, or a script based on them, can then be scheduled to run on a regular basis, to refresh the model with more recent training data and update the inference pipeline.

This notebook is expected to run within the Wallaroo instance's Jupyter Hub service to provide access to all required Wallaroo libraries and functionality.

## Resources

The following resources are used as part of this tutorial:

* **data**
  * `data/seattle_housing_col_description.txt`: Describes the columns used as part data analysis.
  * `data/seattle_housing.csv`: Sample data of the Seattle, Washington housing market between 2014 and 2015.
* **code**
  * `postprocess.py`: Formats the data after inference by the model is complete.
  * `preprocess.py`: Formats the incoming data for the model.
  * `simdb.py`: A simulated database to demonstrate sending and receiving queries.
  * `wallaroo_client.py`: Additional methods used with the Wallaroo instance to create workspaces, etc.
* **models**
  * `housing_model_xgb.onnx`: Model created in Stage 2: Training Process Automation Setup.

## Steps

The process of uploading the model to Wallaroo follows these steps:

* [Connect to Wallaroo](#connect-to-wallaroo): Connect to the Wallaroo instance and set up the workspace.
* [Upload The Model](#upload-the-model): Upload the model and autoconvert for use in the Wallaroo engine.
* [Upload the Processing Modules](#upload-the-processing-modules): Upload the processing modules.
* [Create and Deploy the Pipeline](#create-and-deploy-the-pipeline): Create the pipeline with the model and processing modules as steps, then deploy it.
* [Test the Pipeline](#test-the-pipeline): Verify that the pipeline works with the sample data.

### Connect to Wallaroo

First we import the required libraries to connect to the Wallaroo instance, then connect to the Wallaroo instance.

```python
import json
import pickle
import pandas as pd
import numpy as np

import simdb # module for the purpose of this demo to simulate pulling data from a database

from wallaroo.ModelConversion import ConvertXGBoostArgs, ModelConversionSource, ModelConversionInputType
import wallaroo
from wallaroo.object import EntityNotFoundError
```

```python
# Login through local Wallaroo instance

# wl = wallaroo.Client()

# SSO login through keycloak

wallarooPrefix = "YOUR PREFIX"
wallarooSuffix = "YOUR SUFFIX"

wl = wallaroo.Client(api_endpoint=f"https://{wallarooPrefix}.api.{wallarooSuffix}", 
                    auth_endpoint=f"https://{wallarooPrefix}.keycloak.{wallarooSuffix}", 
                    auth_type="sso")
```

    Please log into the following URL in a web browser:
    
    	https://YOUR PREFIX.keycloak.example.wallaroo.ai/auth/realms/master/device?user_code=XRAD-UUUM
    
    Login successful!

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
workspace_name = 'housepricing2'
model_name = "housepricemodel"
model_file = "./housing_model_xgb.onnx"
pipeline_name = "housing-pipe"
```

The workspace `housepricing` will either be created or, if already existing, used and set to the current workspace.

```python
new_workspace = get_workspace(workspace_name)
new_workspace
```

    {'name': 'housepricing2', 'id': 50, 'archived': False, 'created_by': '6e87ec2b-ad7f-4b0f-b426-5100c26944ba', 'created_at': '2022-10-18T19:41:52.672007+00:00', 'models': [{'name': 'housepricemodel', 'version': 'b66c8053-e28b-4f19-94ff-82f718e12681', 'file_name': 'housing_model_xgb.onnx', 'image_path': None, 'last_update_time': datetime.datetime(2022, 10, 18, 20, 30, 13, 695855, tzinfo=tzutc())}, {'name': 'preprocess', 'version': '09fda370-b5d6-42ef-93cf-429d3d116df3', 'file_name': 'preprocess.py', 'image_path': None, 'last_update_time': datetime.datetime(2022, 10, 18, 20, 30, 17, 364846, tzinfo=tzutc())}, {'name': 'postprocess', 'version': 'ac030fe2-4e86-4cdc-917e-4b5ad7b72838', 'file_name': 'postprocess.py', 'image_path': None, 'last_update_time': datetime.datetime(2022, 10, 18, 20, 30, 17, 766274, tzinfo=tzutc())}], 'pipelines': [{'name': 'housing-pipe', 'create_time': datetime.datetime(2022, 10, 18, 20, 30, 19, 448654, tzinfo=tzutc()), 'definition': '[]'}]}

```python
_ = wl.set_current_workspace(new_workspace)
```

### Upload The Model

With the connection set and workspace prepared, upload the model created in `02_automated_training_process.ipynb` into the current workspace.

```python
hpmodel = wl.upload_model(model_name, model_file).configure()
```

## Upload the Processing Modules

Upload the `preprocess.py` and `postprocess.py` modules as models to be added to the pipeline.

```python
# load the preprocess module
module_pre = wl.upload_model("preprocess", "./preprocess.py").configure('python')
```

```python
# load the postprocess module
module_post = wl.upload_model("postprocess", "./postprocess.py").configure('python')
```

### Create and Deploy the Pipeline

Create the pipeline with the preprocess module, housing model, and postprocess module as pipeline steps, then deploy the newpipeline.

```python
pipeline = (wl.build_pipeline(pipeline_name)
              .add_model_step(module_pre)
              .add_model_step(hpmodel)
              .add_model_step(module_post)
              .deploy()
           )
pipeline
```

    Waiting for deployment - this will take up to 45s ......... ok

<table><tr><th>name</th> <td>housing-pipe</td></tr><tr><th>created</th> <td>2022-10-18 20:30:19.448654+00:00</td></tr><tr><th>last_updated</th> <td>2022-10-19 17:24:17.505573+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>steps</th> <td>preprocess</td></tr></table>

### Test the Pipeline

We will use a single query from the simulated `housing_price` table and infer.  When successful, we will undeploy the pipeline to restore the resources back to the Kubernetes environment.

```python
conn = simdb.simulate_db_connection()

# create the query
query = f"select * from {simdb.tablename} limit 1"
print(query)

# read in the data
singleton = pd.read_sql_query(query, conn)
conn.close()

singleton
```

    select * from house_listings limit 1

<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>date</th>
      <th>list_price</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>sqft_living</th>
      <th>sqft_lot</th>
      <th>floors</th>
      <th>waterfront</th>
      <th>view</th>
      <th>...</th>
      <th>sqft_above</th>
      <th>sqft_basement</th>
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>zipcode</th>
      <th>lat</th>
      <th>long</th>
      <th>sqft_living15</th>
      <th>sqft_lot15</th>
      <th>sale_price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7129300520</td>
      <td>2022-03-07</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.0</td>
      <td>1180</td>
      <td>5650</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1180</td>
      <td>0</td>
      <td>1955</td>
      <td>0</td>
      <td>98178</td>
      <td>47.5112</td>
      <td>-122.257</td>
      <td>1340</td>
      <td>5650</td>
      <td>221900.0</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 22 columns</p>

```python
result = pipeline.infer({'query': singleton.to_json()})
# just display the output
result[0].data()
```

    Waiting for inference response - this will take up to 45s .. ok

    [array([224852.])]

When finished, we undeploy the pipeline to return the resources back to the environment.

```python
pipeline.undeploy()
```

    Waiting for undeployment - this will take up to 45s ........................................... ok

<table><tr><th>name</th> <td>housing-pipe</td></tr><tr><th>created</th> <td>2022-10-18 20:30:19.448654+00:00</td></tr><tr><th>last_updated</th> <td>2022-10-19 17:24:17.505573+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>steps</th> <td>preprocess</td></tr></table>

With this stage complete, we can proceed to Stage 4: Regular Batch Inference.
