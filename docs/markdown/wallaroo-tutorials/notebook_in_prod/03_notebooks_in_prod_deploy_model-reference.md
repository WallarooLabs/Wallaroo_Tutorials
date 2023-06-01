The following tutorials are available from the [Wallaroo Tutorials Repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/notebooks_in_prod).

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

# used to display dataframe information without truncating
from IPython.display import display
import pandas as pd
pd.set_option('display.max_colwidth', None)
```

### Connect to the Wallaroo Instance

The first step is to connect to Wallaroo through the Wallaroo client.  The Python library is included in the Wallaroo install and available through the Jupyter Hub interface provided with your Wallaroo environment.

This is accomplished using the `wallaroo.Client()` command, which provides a URL to grant the SDK permission to your specific Wallaroo environment.  When displayed, enter the URL into a browser and confirm permissions.  Store the connection into a variable that can be referenced later.

If logging into the Wallaroo instance through the internal JupyterHub service, use `wl = wallaroo.Client()`.  For more information on Wallaroo Client settings, see the [Client Connection guide](https://docs.wallaroo.ai/wallaroo-developer-guides/wallaroo-sdk-guides/wallaroo-sdk-essentials-guide/wallaroo-sdk-essentials-client/).

```python
# Login through local Wallaroo instance

wl = wallaroo.Client()
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
workspace_name = 'housepricing'
model_name = "housepricemodel"
model_file = "./housing_model_xgb.onnx"
pipeline_name = "housing-pipe"
```

The workspace `housepricing` will either be created or, if already existing, used and set to the current workspace.

```python
new_workspace = get_workspace(workspace_name)
new_workspace
```

    {'name': 'housepricing', 'id': 17, 'archived': False, 'created_by': '028c8b48-c39b-4578-9110-0b5bdd3824da', 'created_at': '2023-05-17T21:26:51.743388+00:00', 'models': [], 'pipelines': []}

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
pipeline = get_pipeline(pipeline_name)

pipeline.clear()

pipeline.add_model_step(module_pre)
pipeline.add_model_step(hpmodel)
pipeline.add_model_step(module_post)

pipeline.deploy()
```

<table><tr><th>name</th> <td>housing-pipe</td></tr><tr><th>created</th> <td>2023-05-17 21:26:56.133264+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 21:26:57.316392+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>b7dbd380-e48c-487c-8f23-398a2ba558c3, 5ea6f182-5764-4377-9f83-d363e349ef32</td></tr><tr><th>steps</th> <td>preprocess</td></tr></table>

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

singleton.loc[:, ["id", "date", "list_price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot"]]
```

    select * from house_listings limit 1

<table border="1" class="dataframe">
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7129300520</td>
      <td>2022-10-05</td>
      <td>221900.0</td>
      <td>3</td>
      <td>1.0</td>
      <td>1180</td>
      <td>5650</td>
    </tr>
  </tbody>
</table>

```python
result = pipeline.infer({'query': singleton.to_json()})
display(result)
```

    [{'prediction': [224852.0]}]

When finished, we undeploy the pipeline to return the resources back to the environment.

```python
pipeline.undeploy()
```

<table><tr><th>name</th> <td>housing-pipe</td></tr><tr><th>created</th> <td>2023-05-17 21:26:56.133264+00:00</td></tr><tr><th>last_updated</th> <td>2023-05-17 21:26:57.316392+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>b7dbd380-e48c-487c-8f23-398a2ba558c3, 5ea6f182-5764-4377-9f83-d363e349ef32</td></tr><tr><th>steps</th> <td>preprocess</td></tr></table>

With this stage complete, we can proceed to Stage 4: Regular Batch Inference.
