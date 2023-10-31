The following tutorials are available from the [Wallaroo Tutorials Repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/notebooks_in_prod).

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
import pyarrow as pa
import datetime

import simdb # module for the purpose of this demo to simulate pulling data from a database

from wallaroo_client import get_workspace

# used to display dataframe information without truncating
from IPython.display import display
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
        pipeline = wl.pipelines_by_name(name)[0]
    except EntityNotFoundError:
        pipeline = wl.build_pipeline(name)
    return pipeline
```

```python
workspace_name = 'housepricing'
model_name = "housepricemodel"
model_file = "./housing_model_xgb.onnx"
pipeline_name = "housing-pipe"
```

```python
new_workspace = get_workspace(workspace_name)
_ = wl.set_current_workspace(new_workspace)
```

### Deploy the Pipeline

Deploy the `housing-pipe` workspace established in Stage 3: Deploy the Model in Wallaroo (`03_deploy_model.ipynb`).

```python
pipeline = get_pipeline(pipeline_name)

deploy_config = wallaroo.DeploymentConfigBuilder().replica_count(1).cpus(0.5).memory("1Gi").build()
pipeline.deploy(deployment_config=deploy_config)
```

<table><tr><th>name</th> <td>housing-pipe</td></tr><tr><th>created</th> <td>2023-10-26 16:18:34.784680+00:00</td></tr><tr><th>last_updated</th> <td>2023-10-26 16:22:03.472269+00:00</td></tr><tr><th>deployed</th> <td>True</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>67da1170-b201-418e-bf03-59908e1c547b, f5c2a6d8-2d10-4d89-8755-fac05a99634b, 933ec7f5-2612-497c-a571-a4bfa113d967</td></tr><tr><th>steps</th> <td>preprocess</td></tr><tr><th>published</th> <td>False</td></tr></table>

### Read In New House Listings

From the data store, load the previous month's house listing, prepare it as a DataFrame, then submit it for inferencing.

```python
conn = simdb.simulate_db_connection()

# create the query
query = f"select * from {simdb.tablename} where date > DATE(DATE(), '-1 month') AND sale_price is NULL"
print(query)

# read in the data
# can't have null values - turn them into 0
newbatch = pd.read_sql_query(query, conn)
newbatch['sale_price'] = newbatch.sale_price.apply(lambda x: 0)
display(newbatch.shape)
display(newbatch.head(10).loc[:, ["id", "date", "list_price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot"]])
```

    select * from house_listings where date > DATE(DATE(), '-1 month') AND sale_price is NULL

    (964, 22)

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
      <td>9215400105</td>
      <td>2023-09-27</td>
      <td>450000.0</td>
      <td>3</td>
      <td>1.75</td>
      <td>1250</td>
      <td>5963</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1695900060</td>
      <td>2023-10-10</td>
      <td>535000.0</td>
      <td>4</td>
      <td>1.00</td>
      <td>1610</td>
      <td>2982</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9545240070</td>
      <td>2023-09-27</td>
      <td>660500.0</td>
      <td>4</td>
      <td>2.25</td>
      <td>2010</td>
      <td>9603</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1432900240</td>
      <td>2023-10-07</td>
      <td>205000.0</td>
      <td>3</td>
      <td>1.00</td>
      <td>1610</td>
      <td>8579</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1400300055</td>
      <td>2023-09-27</td>
      <td>425000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>5040</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7960900060</td>
      <td>2023-10-03</td>
      <td>2900000.0</td>
      <td>4</td>
      <td>3.25</td>
      <td>5050</td>
      <td>20100</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6378500125</td>
      <td>2023-09-30</td>
      <td>436000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>1040</td>
      <td>7538</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2022069200</td>
      <td>2023-10-04</td>
      <td>455000.0</td>
      <td>4</td>
      <td>2.50</td>
      <td>2210</td>
      <td>49375</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9412900055</td>
      <td>2023-10-04</td>
      <td>405000.0</td>
      <td>3</td>
      <td>1.75</td>
      <td>2390</td>
      <td>6000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>7424700045</td>
      <td>2023-10-12</td>
      <td>2050000.0</td>
      <td>5</td>
      <td>3.00</td>
      <td>3830</td>
      <td>8480</td>
    </tr>
  </tbody>
</table>

```python
# query = {'query': newbatch.to_json()}

result = pipeline.infer(newbatch)
# display(result)
predicted_prices = pd.DataFrame(result['out.variable'].apply(lambda x: x[0])).rename(columns={'out.variable':'prediction'})
display(predicted_prices[0:5])
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prediction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>508255.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>500198.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>539598.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>270739.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>346586.0</td>
    </tr>
  </tbody>
</table>

### Send Predictions to Results Staging Table

Take the predicted prices based on the inference results so they can be joined into the `house_listings` table.

Once complete, undeploy the pipeline to return the resources back to the Kubernetes environment.

```python
result_table = pd.DataFrame({
    'id': newbatch['id'],
    'saleprice_estimate': predicted_prices['prediction']
})

display(result_table)

result_table.to_sql('results_table', conn, index=False, if_exists='append')
```

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
      <td>9215400105</td>
      <td>508255.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1695900060</td>
      <td>500198.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9545240070</td>
      <td>539598.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1432900240</td>
      <td>270739.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1400300055</td>
      <td>346586.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>959</th>
      <td>3304300300</td>
      <td>577492.0</td>
    </tr>
    <tr>
      <th>960</th>
      <td>6453550090</td>
      <td>882930.0</td>
    </tr>
    <tr>
      <th>961</th>
      <td>1760650820</td>
      <td>271484.0</td>
    </tr>
    <tr>
      <th>962</th>
      <td>3345700207</td>
      <td>537434.0</td>
    </tr>
    <tr>
      <th>963</th>
      <td>7853420110</td>
      <td>634226.0</td>
    </tr>
  </tbody>
</table>
<p>964 rows × 2 columns</p>

```python
# Display the top of the table for confirmation
pd.read_sql_query("select * from results_table limit 5", conn)
```

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
      <td>9215400105</td>
      <td>508255.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1695900060</td>
      <td>500198.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9545240070</td>
      <td>539598.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1432900240</td>
      <td>270739.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1400300055</td>
      <td>346586.0</td>
    </tr>
  </tbody>
</table>

```python
conn.close()
pipeline.undeploy()
```

<table><tr><th>name</th> <td>housing-pipe</td></tr><tr><th>created</th> <td>2023-10-26 16:18:34.784680+00:00</td></tr><tr><th>last_updated</th> <td>2023-10-26 16:22:03.472269+00:00</td></tr><tr><th>deployed</th> <td>False</td></tr><tr><th>tags</th> <td></td></tr><tr><th>versions</th> <td>67da1170-b201-418e-bf03-59908e1c547b, f5c2a6d8-2d10-4d89-8755-fac05a99634b, 933ec7f5-2612-497c-a571-a4bfa113d967</td></tr><tr><th>steps</th> <td>preprocess</td></tr><tr><th>published</th> <td>False</td></tr></table>

From here, organizations can automate this process.  Other features could be used such as data analysis using Wallaroo assays, or other features such as shadow deployments to test champion and challenger models to find which models provide the best results.
