For more details on this tutorial's setup and process, see `00_Introduction.ipynb`.

# Stage 4: Regular Batch Inference

In Stage 3: Deploy the Model in Wallaroo, the housing model created and tested in Stage 2: Training Process Automation Setup was uploaded to a Wallaroo instance and added to the pipeline `housing-pipe` in the workspace `housepricing`.  This pipeline can be deployed at any point and time and used with new inferences.

For the purposes of this demo, let's say that every month we find the newly entered and still-unsold houses and predict their sale price.

The predictions are entered into a staging table, for further inspection before being joined to the primary housing data table.

We show this as a notebook, but this can also be scripted and scheduled, using CRON or some other process.

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

This process will use the following steps:

* [Connect to Wallaroo](#connect-to-wallaroo): Connect to the Wallaroo instance and the `housepricing` workspace.
* [Deploy the Pipeline](#deploy-the-pipeline): Deploy the pipeline to prepare it to run inferences.
* [Read In New House Listings](#read-in-new-house-listings): Read in the previous month's house listings and submit them to the pipeline for inference.
* [Send Predictions to Results Staging Table](#send-predictions-to-results-staging-table): Add the inference results to the results staging table.

### Connect to Wallaroo

Connect to the Wallaroo instance and set the `housepricing` workspace as the current workspace.


```python
import json
import pickle
import wallaroo
import pandas as pd
import numpy as np

import simdb # module for the purpose of this demo to simulate pulling data from a database

from wallaroo_client import get_workspace

import os
# Only set the below to make the OS environment ARROW_ENABLED to TRUE.  Otherwise, leave as is.
os.environ["ARROW_ENABLED"]="True"

# used to display dataframe information without truncating
from IPython.display import display
pd.set_option('display.max_colwidth', None)
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
new_workspace = get_workspace("housepricing")
_ = wl.set_current_workspace(new_workspace)
```

### Deploy the Pipeline

Deploy the `housing-pipe` workspace established in Stage 3: Deploy the Model in Wallaroo (`03_deploy_model.ipynb`).


```python
pipeline = wl.pipelines_by_name("housing-pipe")[-1]
pipeline.deploy()
```




<table><tr><th>name</th> <td>housing-pipe</td></tr><tr><th>created</th> <td>2023-03-27 21:37:27.267496+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-27 21:54:12.015116+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>bac1ae9d-8d64-4cdd-b332-fe00c5afbc45, 63b1214b-2026-461e-aab4-b6798c371222, 46761a50-b058-4cb4-9f49-b64667059c7e</td></tr><tr><th>steps</th> <td>preprocess</td></tr></table>



### Read In New House Listings

From the data store, load the previous month's house listing and submit them to the deployed pipeline.


```python
conn = simdb.simulate_db_connection()

# create the query
query = f"select * from {simdb.tablename} where date > DATE(DATE(), '-1 month') AND sale_price is NULL"
print(query)

# read in the data
newbatch = pd.read_sql_query(query, conn)
newbatch.shape
```

    select * from house_listings where date > DATE(DATE(), '-1 month') AND sale_price is NULL





    (729, 22)




```python
query = {'query': newbatch.to_json()}
result = pipeline.infer(query)
predicted_prices = result[0]['prediction']
len(predicted_prices)
```




    729



### Send Predictions to Results Staging Table

Take the predicted prices based on the inference results so they can be joined into the `house_listings` table.

Once complete, undeploy the pipeline to return the resources back to the Kubernetes environment.


```python
result_table = pd.DataFrame({
    'id': newbatch['id'],
    'saleprice_estimate': predicted_prices,
})

result_table.to_sql('results_table', conn, index=False, if_exists='append')
```


```python
# Display the top of the table for confirmation
pd.read_sql_query("select * from results_table limit 5", conn)
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
      <th>id</th>
      <th>saleprice_estimate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1695900060</td>
      <td>500198.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1432900240</td>
      <td>270739.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7960900060</td>
      <td>3067263.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6378500125</td>
      <td>378917.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022069200</td>
      <td>496464.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
conn.close()
pipeline.undeploy()
```




<table><tr><th>name</th> <td>housing-pipe</td></tr><tr><th>created</th> <td>2023-03-27 21:37:27.267496+00:00</td></tr><tr><th>last_updated</th> <td>2023-03-27 21:54:12.015116+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>bac1ae9d-8d64-4cdd-b332-fe00c5afbc45, 63b1214b-2026-461e-aab4-b6798c371222, 46761a50-b058-4cb4-9f49-b64667059c7e</td></tr><tr><th>steps</th> <td>preprocess</td></tr></table>



From here, organizations can automate this process.  Other features could be used such as data analysis using Wallaroo assays, or other features such as shadow deployments to test champion and challenger models to find which models provide the best results.
