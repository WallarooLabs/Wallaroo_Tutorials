For more details on this tutorial's setup and process, see `00_Introduction.ipynb`.

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
```

```python
conn = simdb.simulate_db_connection()
tablename = simdb.tablename

query = f"select * from {tablename} where date > DATE(DATE(), '-24 month') AND sale_price is not NULL"
print(query)
# read in the data
housing_data = pd.read_sql_query(query, conn)

conn.close()
housing_data
```

    select * from house_listings where date > DATE(DATE(), '-24 month') AND sale_price is not NULL

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
      <td>1.00</td>
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
    <tr>
      <th>1</th>
      <td>6414100192</td>
      <td>2022-05-03</td>
      <td>538000.0</td>
      <td>3</td>
      <td>2.25</td>
      <td>2570</td>
      <td>7242</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2170</td>
      <td>400</td>
      <td>1951</td>
      <td>1991</td>
      <td>98125</td>
      <td>47.7210</td>
      <td>-122.319</td>
      <td>1690</td>
      <td>7639</td>
      <td>538000.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5631500400</td>
      <td>2022-07-20</td>
      <td>180000.0</td>
      <td>2</td>
      <td>1.00</td>
      <td>770</td>
      <td>10000</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>770</td>
      <td>0</td>
      <td>1933</td>
      <td>0</td>
      <td>98028</td>
      <td>47.7379</td>
      <td>-122.233</td>
      <td>2720</td>
      <td>8062</td>
      <td>180000.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2487200875</td>
      <td>2022-05-03</td>
      <td>604000.0</td>
      <td>4</td>
      <td>3.00</td>
      <td>1960</td>
      <td>5000</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1050</td>
      <td>910</td>
      <td>1965</td>
      <td>0</td>
      <td>98136</td>
      <td>47.5208</td>
      <td>-122.393</td>
      <td>1360</td>
      <td>5000</td>
      <td>604000.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1954400510</td>
      <td>2022-07-13</td>
      <td>510000.0</td>
      <td>3</td>
      <td>2.00</td>
      <td>1680</td>
      <td>8080</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1680</td>
      <td>0</td>
      <td>1987</td>
      <td>0</td>
      <td>98074</td>
      <td>47.6168</td>
      <td>-122.045</td>
      <td>1800</td>
      <td>7503</td>
      <td>510000.0</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
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
      <td>2021-10-13</td>
      <td>360000.0</td>
      <td>3</td>
      <td>2.50</td>
      <td>1530</td>
      <td>1131</td>
      <td>3.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1530</td>
      <td>0</td>
      <td>2009</td>
      <td>0</td>
      <td>98103</td>
      <td>47.6993</td>
      <td>-122.346</td>
      <td>1530</td>
      <td>1509</td>
      <td>360000.0</td>
    </tr>
    <tr>
      <th>20519</th>
      <td>6600060120</td>
      <td>2022-07-18</td>
      <td>400000.0</td>
      <td>4</td>
      <td>2.50</td>
      <td>2310</td>
      <td>5813</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2310</td>
      <td>0</td>
      <td>2014</td>
      <td>0</td>
      <td>98146</td>
      <td>47.5107</td>
      <td>-122.362</td>
      <td>1830</td>
      <td>7200</td>
      <td>400000.0</td>
    </tr>
    <tr>
      <th>20520</th>
      <td>1523300141</td>
      <td>2021-11-15</td>
      <td>402101.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>1350</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1020</td>
      <td>0</td>
      <td>2009</td>
      <td>0</td>
      <td>98144</td>
      <td>47.5944</td>
      <td>-122.299</td>
      <td>1020</td>
      <td>2007</td>
      <td>402101.0</td>
    </tr>
    <tr>
      <th>20521</th>
      <td>291310100</td>
      <td>2022-06-10</td>
      <td>400000.0</td>
      <td>3</td>
      <td>2.50</td>
      <td>1600</td>
      <td>2388</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1600</td>
      <td>0</td>
      <td>2004</td>
      <td>0</td>
      <td>98027</td>
      <td>47.5345</td>
      <td>-122.069</td>
      <td>1410</td>
      <td>1287</td>
      <td>400000.0</td>
    </tr>
    <tr>
      <th>20522</th>
      <td>1523300157</td>
      <td>2022-03-09</td>
      <td>325000.0</td>
      <td>2</td>
      <td>0.75</td>
      <td>1020</td>
      <td>1076</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1020</td>
      <td>0</td>
      <td>2008</td>
      <td>0</td>
      <td>98144</td>
      <td>47.5941</td>
      <td>-122.299</td>
      <td>1020</td>
      <td>1357</td>
      <td>325000.0</td>
    </tr>
  </tbody>
</table>
<p>20523 rows × 22 columns</p>

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

    
![png](/images/wallaroo-tutorials/notebooks_in_prod/02_notebooks_in_prod_automated_training_process-reference_files/02_notebooks_in_prod_automated_training_process-reference_10_0.png)
    

```python
pframe['se'] = (pframe.pred - pframe.actual)**2

pframe['pct_err'] = 100*np.abs(pframe.pred - pframe.actual)/pframe.actual
pframe.describe()
```

<table>
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
# pickle up the model
# with open('housing_model_xgb.pkl', 'wb') as f:
#    pickle.dump(xgb_model, f)
```

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
