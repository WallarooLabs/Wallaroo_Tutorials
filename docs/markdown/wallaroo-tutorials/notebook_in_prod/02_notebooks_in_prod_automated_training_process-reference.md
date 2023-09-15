The following tutorials are available from the [Wallaroo Tutorials Repository](https://github.com/WallarooLabs/Wallaroo_Tutorials/tree/main/notebooks_in_prod).

# Stage 2: Training Process Automation Setup
 
Now that we have decided on the type and structure of the model from Stage 1: Data Exploration And Model Selection, this notebook modularizes the various steps of the process in a structure that is compatible with production and with Wallaroo.

We have pulled the preprocessing and postprocessing steps out of the training notebook into individual scripts that can also be used when the model is deployed.

Assuming no changes are made to the structure of the model, this notebook, or a script based on this notebook, can then be scheduled to run on a regular basis, to refresh the model with more recent training data. We'd expect to run this notebook in conjunction with the Stage 3 notebook, `03_deploy_model.ipynb`.  For clarity in this demo, we have split the training/upload task into two notebooks, `02_automated_training_process.ipynb` and `03_deploy_model.ipynb`.

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

## Steps

The following steps are part of this process:

* [Retrieve Training Data](#retrieve-training-data): Connect to the data store and retrieve the training data.
* [Data Transformations](#data-transformations): Evaluate the data and train the model.
* [Generate and Test the Model](#generate-and-test-the-model): Create the model and verify it against the sample test data.
* [Pickle The Model](#pickle-the-model): Prepare the model to be uploaded to Wallaroo.

### Retrieve Training Data

Note that this connection is simulated to demonstrate how data would be retrieved from an existing data store.  For training, we will use the data on all houses sold in this market with the last two years.

```python
import numpy as np
import pandas as pd

import sklearn

import xgboost as xgb

import seaborn
import matplotlib
import matplotlib.pyplot as plt

import pickle

import simdb # module for the purpose of this demo to simulate pulling data from a database

from preprocess import create_features  # our custom preprocessing
from postprocess import postprocess    # our custom postprocessing

matplotlib.rcParams["figure.figsize"] = (12,6)

# ignoring warnings for demonstration
import warnings
warnings.filterwarnings('ignore')
```

```python
conn = simdb.simulate_db_connection()
tablename = simdb.tablename

query = f"select * from {tablename} where date > DATE(DATE(), '-24 month') AND sale_price is not NULL"
print(query)
# read in the data
housing_data = pd.read_sql_query(query, conn)

conn.close()
housing_data.loc[:, ["id", "date", "list_price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot"]]
```

    select * from house_listings where date > DATE(DATE(), '-24 month') AND sale_price is not NULL

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
      <td>1.00</td>
      <td>1180</td>
      <td>5650</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6414100192</td>
      <td>2022-12-01</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5631500400</td>
      <td>2023-02-17</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2487200875</td>
      <td>2022-12-01</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1954400510</td>
      <td>2023-02-10</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20518</th>
      <td>263000018</td>
      <td>2022-05-13</td>
      <td>360000.0</td>
      <td>3</td>
      <td>2.50</td>
      <td>1530</td>
      <td>1131</td>
    </tr>
    <tr>
      <th>20519</th>
      <td>6600060120</td>
      <td>2023-02-15</td>
      <td>400000.0</td>
      <td>4</td>
      <td>2.50</td>
      <td>2310</td>
      <td>5813</td>
    </tr>
    <tr>
      <th>20520</th>
      <td>1523300141</td>
      <td>2022-06-15</td>
      <td>402101.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>1350</td>
    </tr>
    <tr>
      <th>20521</th>
      <td>291310100</td>
      <td>2023-01-08</td>
      <td>400000.0</td>
      <td>3</td>
      <td>2.50</td>
      <td>1600</td>
      <td>2388</td>
    </tr>
    <tr>
      <th>20522</th>
      <td>1523300157</td>
      <td>2022-10-07</td>
      <td>325000.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>1076</td>
    </tr>
  </tbody>
</table>
<p>20523 rows × 7 columns</p>

### Data transformations

To improve relative error performance, we will predict on `log10` of the sale price.

Predict on log10 price to try to improve relative error performance

```python
housing_data['logprice'] = np.log10(housing_data.list_price)
```

```python
# split data into training and test
outcome = 'logprice'

runif = np.random.default_rng(2206222).uniform(0, 1, housing_data.shape[0])
gp = np.where(runif < 0.2, 'test', 'training')

hd_train = housing_data.loc[gp=='training', :].reset_index(drop=True, inplace=False)
hd_test = housing_data.loc[gp=='test', :].reset_index(drop=True, inplace=False)

# split the training into training and val for xgboost
runif = np.random.default_rng(123).uniform(0, 1, hd_train.shape[0])
xgb_gp = np.where(runif < 0.2, 'val', 'train')
```

```python
# for xgboost
train_features = hd_train.loc[xgb_gp=='train', :].reset_index(drop=True, inplace=False)
train_features = np.array(create_features(train_features))
train_labels = np.array(hd_train.loc[xgb_gp=='train', outcome])

val_features = hd_train.loc[xgb_gp=='val', :].reset_index(drop=True, inplace=False)
val_features = np.array(create_features(val_features))
val_labels = np.array(hd_train.loc[xgb_gp=='val', outcome])

print(f'train_features: {train_features.shape}, train_labels: {len(train_labels)}')
print(f'val_features: {val_features.shape}, val_labels: {len(val_labels)}')

```

    train_features: (13129, 18), train_labels: 13129
    val_features: (3300, 18), val_labels: 3300

### Generate and Test the Model

Based on the experimentation and testing performed in **Stage 1: Data Exploration And Model Selection**, XGBoost was selected as the ML model and the variables for training were selected.  The model will be generated and tested against sample data.

```python

xgb_model = xgb.XGBRegressor(
    objective = 'reg:squarederror', 
    max_depth=5, 
    base_score = np.mean(hd_train[outcome])
    )

xgb_model.fit( 
    train_features,
    train_labels,
    eval_set=[(train_features, train_labels), (val_features, val_labels)],
    verbose=False,
    early_stopping_rounds=35
)

```

<pre>XGBRegressor(base_score=5.666446833601829, booster=&#x27;gbtree&#x27;, callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, gamma=0, gpu_id=-1, grow_policy=&#x27;depthwise&#x27;,
             importance_type=None, interaction_constraints=&#x27;&#x27;,
             learning_rate=0.300000012, max_bin=256, max_cat_to_onehot=4,
             max_delta_step=0, max_depth=5, max_leaves=0, min_child_weight=1,
             missing=nan, monotone_constraints=&#x27;()&#x27;, n_estimators=100, n_jobs=0,
             num_parallel_tree=1, predictor=&#x27;auto&#x27;, random_state=0, reg_alpha=0,
             reg_lambda=1, ...)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">XGBRegressor</label><pre>XGBRegressor(base_score=5.666446833601829, booster=&#x27;gbtree&#x27;, callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, gamma=0, gpu_id=-1, grow_policy=&#x27;depthwise&#x27;,
             importance_type=None, interaction_constraints=&#x27;&#x27;,
             learning_rate=0.300000012, max_bin=256, max_cat_to_onehot=4,
             max_delta_step=0, max_depth=5, max_leaves=0, min_child_weight=1,
             missing=nan, monotone_constraints=&#x27;()&#x27;, n_estimators=100, n_jobs=0,
             num_parallel_tree=1, predictor=&#x27;auto&#x27;, random_state=0, reg_alpha=0,
             reg_lambda=1, ...)</pre>

```python
print(xgb_model.best_score)
print(xgb_model.best_iteration)
print(xgb_model.best_ntree_limit)
```

    0.07793614689092423
    99
    100

```python
test_features = np.array(create_features(hd_test.copy()))
test_labels = np.array(hd_test.loc[:, outcome])

pframe = pd.DataFrame({
    'pred' : postprocess(xgb_model.predict(test_features)),
    'actual' : postprocess(test_labels)
})

ax = seaborn.scatterplot(
    data=pframe,
    x='pred',
    y='actual',
    alpha=0.2
)
matplotlib.pyplot.plot(pframe.pred, pframe.pred, color='DarkGreen')
matplotlib.pyplot.title("test")
plt.show()
```

    
{{<figure src="/images/2023.3.0/wallaroo-tutorials/notebook_in_prod/02_notebooks_in_prod_automated_training_process-reference_files/02_notebooks_in_prod_automated_training_process-reference_10_0.png" width="800" label="png">}}
    

```python
pframe['se'] = (pframe.pred - pframe.actual)**2

pframe['pct_err'] = 100*np.abs(pframe.pred - pframe.actual)/pframe.actual
pframe.describe()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pred</th>
      <th>actual</th>
      <th>se</th>
      <th>pct_err</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4.094000e+03</td>
      <td>4.094000e+03</td>
      <td>4.094000e+03</td>
      <td>4094.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.340824e+05</td>
      <td>5.396937e+05</td>
      <td>1.657722e+10</td>
      <td>12.857674</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.413714e+05</td>
      <td>3.761666e+05</td>
      <td>1.276017e+11</td>
      <td>13.512028</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.216140e+05</td>
      <td>8.200000e+04</td>
      <td>1.000000e+00</td>
      <td>0.000500</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.167628e+05</td>
      <td>3.200000e+05</td>
      <td>3.245312e+08</td>
      <td>4.252492</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.568700e+05</td>
      <td>4.500000e+05</td>
      <td>1.602001e+09</td>
      <td>9.101485</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.310372e+05</td>
      <td>6.355250e+05</td>
      <td>6.575385e+09</td>
      <td>17.041227</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.126706e+06</td>
      <td>7.700000e+06</td>
      <td>6.637466e+12</td>
      <td>252.097895</td>
    </tr>
  </tbody>
</table>

```python
rmse = np.sqrt(np.mean(pframe.se))
mape = np.mean(pframe.pct_err)

print(f'rmse = {rmse}, mape = {mape}')
```

    rmse = 128752.54982046234, mape = 12.857674005250548

### Convert the Model to Onnx

This step converts the model to onnx for easy import into Wallaroo.

```python
import onnx
from onnxmltools.convert import convert_xgboost

from skl2onnx.common.data_types import FloatTensorType, DoubleTensorType

import preprocess

# set the number of columns
ncols = len(preprocess._vars)

# derive the opset value

from onnx.defs import onnx_opset_version
from onnxconverter_common.onnx_ex import DEFAULT_OPSET_NUMBER
TARGET_OPSET = min(DEFAULT_OPSET_NUMBER, onnx_opset_version())
```

```python
# Convert the model to onnx

onnx_model_converted = convert_xgboost(xgb_model, 'tree-based classifier',
                             [('input', FloatTensorType([None, ncols]))],
                             target_opset=TARGET_OPSET)

# Save the model

onnx.save_model(onnx_model_converted, "housing_model_xgb.onnx")
```

With the model trained and ready, we can now go to Stage 3: Deploy the Model in Wallaroo.
