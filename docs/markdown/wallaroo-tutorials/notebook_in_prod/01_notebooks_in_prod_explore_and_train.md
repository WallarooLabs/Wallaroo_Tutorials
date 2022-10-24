For more details on this tutorial's setup and process, see `00_Introduction.ipynb`.

# Stage 1: Data Exploration And Model Selection

When starting a project, the data scientist focuses on exploration and experimentation, rather than turning the process into an immediate production system.  This notebook presents a simplified view of this stage.

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
* [Model Testing](#model-testing): Evaluate different models and determine which is best suited for the problem.

### Import Libraries

First we'll import the libraries we'll be using to evaluate the data and test different models.

```python
import numpy as np
import pandas as pd

import sklearn
import sklearn.ensemble

import xgboost as xgb

import seaborn
import matplotlib
import matplotlib.pyplot as plt

import simdb # module for the purpose of this demo to simulate pulling data from a database

matplotlib.rcParams["figure.figsize"] = (12,6)
```

### Retrieve Training Data

For training, we will use the data on all houses sold in this market with the last two years.  As a reminder, this data pulled from a simulated database as an example of how to pull from an existing data store.

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
      <td>2022-03-06</td>
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
      <td>2022-05-02</td>
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
      <td>2022-07-19</td>
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
      <td>2022-05-02</td>
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
      <td>2022-07-12</td>
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
      <td>2021-10-12</td>
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
      <td>2022-07-17</td>
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
      <td>2021-11-14</td>
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
      <td>2022-06-09</td>
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
      <td>2022-03-08</td>
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
</div>


### Data transformations

To improve relative error performance, we will predict on `log10` of the sale price.

Predict on log10 price to try to improve relative error performance

```python
housing_data['logprice'] = np.log10(housing_data.sale_price)
```

From the data, we will create the following features to evaluate:

* `house_age`: How old the house is.
* `renovated`: Whether the house has been renovated or not.
* `yrs_since_reno`: If the house has been renovated, how long has it been.

```python
import datetime

thisyear = datetime.datetime.now().year

housing_data['house_age'] = thisyear - housing_data['yr_built']
housing_data['renovated'] =  np.where((housing_data['yr_renovated'] > 0), 1, 0) 
housing_data['yrs_since_reno'] =  np.where(housing_data['renovated'], housing_data['yr_renovated'] - housing_data['yr_built'], 0)

housing_data.loc[:, ['yr_built', 'yr_renovated', 'house_age', 'renovated', 'yrs_since_reno']]

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
      <th>yr_built</th>
      <th>yr_renovated</th>
      <th>house_age</th>
      <th>renovated</th>
      <th>yrs_since_reno</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1955</td>
      <td>0</td>
      <td>67</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1951</td>
      <td>1991</td>
      <td>71</td>
      <td>1</td>
      <td>40</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1933</td>
      <td>0</td>
      <td>89</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1965</td>
      <td>0</td>
      <td>57</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1987</td>
      <td>0</td>
      <td>35</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20518</th>
      <td>2009</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20519</th>
      <td>2014</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20520</th>
      <td>2009</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20521</th>
      <td>2004</td>
      <td>0</td>
      <td>18</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20522</th>
      <td>2008</td>
      <td>0</td>
      <td>14</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>20523 rows × 5 columns</p>
</div>


Now we pick variables and split training data into training and holdout (test).

```python
vars = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view',
'condition', 'grade', 'sqft_above', 'sqft_basement', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'house_age', 'renovated', 'yrs_since_reno']

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
# for xgboost, further split into train and val
train_features = np.array(hd_train.loc[xgb_gp=='train', vars])
train_labels = np.array(hd_train.loc[xgb_gp=='train', outcome])

val_features = np.array(hd_train.loc[xgb_gp=='val', vars])
val_labels = np.array(hd_train.loc[xgb_gp=='val', outcome])

```

#### Postprocessing

Since we are fitting a model to predict `log10` price, we need to convert predictions back into price units. We also want to round to the nearest dollar.

```python
def postprocess(log10price):
    return np.rint(np.power(10, log10price))
```

### Model testing

For the purposes of this demo, let's say that we require a mean absolute percent error (MAPE) of 15% or less, and the we want to try a few models to decide which model we want to use.

One could also hyperparameter tune at this stage; for brevity, we'll omit that in this demo.

#### XGBoost

First we will test out using a XGBoost model.

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

    /opt/conda/lib/python3.9/site-packages/xgboost/sklearn.py:793: UserWarning: `early_stopping_rounds` in `fit` method is deprecated for better compatibility with scikit-learn, use `early_stopping_rounds` in constructor or`set_params` instead.
      warnings.warn(



<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>XGBRegressor(base_score=5.666446833601829, booster=&#x27;gbtree&#x27;, callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, gamma=0, gpu_id=-1, grow_policy=&#x27;depthwise&#x27;,
             importance_type=None, interaction_constraints=&#x27;&#x27;,
             learning_rate=0.300000012, max_bin=256, max_cat_to_onehot=4,
             max_delta_step=0, max_depth=5, max_leaves=0, min_child_weight=1,
             missing=nan, monotone_constraints=&#x27;()&#x27;, n_estimators=100, n_jobs=0,
             num_parallel_tree=1, predictor=&#x27;auto&#x27;, random_state=0, reg_alpha=0,
             reg_lambda=1, ...)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">XGBRegressor</label><div class="sk-toggleable__content"><pre>XGBRegressor(base_score=5.666446833601829, booster=&#x27;gbtree&#x27;, callbacks=None,
             colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
             early_stopping_rounds=None, enable_categorical=False,
             eval_metric=None, gamma=0, gpu_id=-1, grow_policy=&#x27;depthwise&#x27;,
             importance_type=None, interaction_constraints=&#x27;&#x27;,
             learning_rate=0.300000012, max_bin=256, max_cat_to_onehot=4,
             max_delta_step=0, max_depth=5, max_leaves=0, min_child_weight=1,
             missing=nan, monotone_constraints=&#x27;()&#x27;, n_estimators=100, n_jobs=0,
             num_parallel_tree=1, predictor=&#x27;auto&#x27;, random_state=0, reg_alpha=0,
             reg_lambda=1, ...)</pre></div></div></div></div></div>



```python
print(xgb_model.best_score)
print(xgb_model.best_iteration)
print(xgb_model.best_ntree_limit)
```

    0.07793614689092423
    99
    100

#### XGBoost Evaluate on holdout

With the sample model created, we will test it against the holdout data.  Note that we are calling the `postprocess` function on the data.

```python
test_features = np.array(hd_test.loc[:, vars])
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

    
![png](/images/wallaroo-tutorials/notebook_in_prod/01_notebooks_in_prod_explore_and_train_files/01_notebooks_in_prod_explore_and_train_18_0.png)
    


```python
pframe['se'] = (pframe.pred - pframe.actual)**2

pframe['pct_err'] = 100*np.abs(pframe.pred - pframe.actual)/pframe.actual
pframe.describe()
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
</div>



```python
rmse = np.sqrt(np.mean(pframe.se))
mape = np.mean(pframe.pct_err)

print(f'rmse = {rmse}, mape = {mape}')
```

    rmse = 128752.54982046234, mape = 12.857674005250548

#### Random Forest

The next model to test is Random Forest.

```python
model_rf = sklearn.ensemble.RandomForestRegressor(n_estimators=100, max_depth=5, n_jobs=2, max_samples=0.8)

train_features = np.array(hd_train.loc[:, vars])
train_labels = np.array(hd_train.loc[:, outcome])

model_rf.fit(train_features, train_labels)
```



<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>RandomForestRegressor(max_depth=5, max_samples=0.8, n_jobs=2)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestRegressor</label><div class="sk-toggleable__content"><pre>RandomForestRegressor(max_depth=5, max_samples=0.8, n_jobs=2)</pre></div></div></div></div></div>


#### Random Forest Evaluate on holdout

With the Random Forest sample model created, now we can test it against the holdout data.

```python
pframe = pd.DataFrame({
    'pred' : postprocess(model_rf.predict(test_features)),
    'actual' : postprocess(test_labels)
})

ax = seaborn.scatterplot(
    data=pframe,
    x='pred',
    y='actual',
    alpha=0.2
)
matplotlib.pyplot.plot(pframe.pred, pframe.pred, color='DarkGreen')
matplotlib.pyplot.title("random forest")
plt.show()
```

    
![png](/images/wallaroo-tutorials/notebook_in_prod/01_notebooks_in_prod_explore_and_train_files/01_notebooks_in_prod_explore_and_train_24_0.png)
    


```python
pframe['se'] = (pframe.pred - pframe.actual)**2

pframe['pct_err'] = 100*np.abs(pframe.pred - pframe.actual)/pframe.actual
pframe.describe()
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
      <td>5.195770e+05</td>
      <td>5.396937e+05</td>
      <td>3.859051e+10</td>
      <td>18.102360</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.803551e+05</td>
      <td>3.761666e+05</td>
      <td>4.080385e+11</td>
      <td>17.602594</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2.041320e+05</td>
      <td>8.200000e+04</td>
      <td>1.960000e+02</td>
      <td>0.005426</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.315125e+05</td>
      <td>3.200000e+05</td>
      <td>6.582175e+08</td>
      <td>6.020459</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.638960e+05</td>
      <td>4.500000e+05</td>
      <td>3.325252e+09</td>
      <td>13.214037</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.864730e+05</td>
      <td>6.355250e+05</td>
      <td>1.328194e+10</td>
      <td>24.651880</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.874049e+06</td>
      <td>7.700000e+06</td>
      <td>2.328980e+13</td>
      <td>186.080000</td>
    </tr>
  </tbody>
</table>
</div>



```python
rmse = np.sqrt(np.mean(pframe.se))
mape = np.mean(pframe.pct_err)

print(f'rmse = {rmse}, mape = {mape}')
```

    rmse = 196444.67813511938, mape = 18.102359731513623

### Final Decision

At this stage, we decide to go with the **xgboost** model, with the variables/settings above.

With this stage complete, we can move on to Stage 2: Training Process Automation Setup.
