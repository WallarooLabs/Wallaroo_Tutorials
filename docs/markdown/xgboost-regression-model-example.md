## XBBoost Regression Model Generation
  
The following tutorial demonstrates how to generate a **XGBoost** Regression Model.  This is used as part of the `auto-convert-to-onnx-tutorial.ipynb` tutorial.


```python
import numpy as np
import pandas as pd

import sklearn
import sklearn.datasets

import xgboost as xgb

import pickle
import json
```

# Regression


```python
# create data
Ntrain = 1000
Nvalid = 1000
Neval = 5
N = Ntrain+Nvalid+Neval

NF = 25
Ninformative = 10

X, Y = sklearn.datasets.make_regression(n_samples=N, n_features=NF, n_informative=Ninformative)

row_use = np.array(['train']*Ntrain + ['validate']*Nvalid + ['eval']*Neval)


Xtrain = X[row_use=='train', :]
Ytrain = Y[row_use=='train']

Xvalid = X[row_use=='validate', :]
Yvalid = Y[row_use=='validate']

Xeval = X[row_use=='eval', :]
Yeval = Y[row_use=='eval']

print(Xtrain.shape)
print(Xvalid.shape)
print(Xeval.shape)
```

    (1000, 25)
    (1000, 25)
    (5, 25)



```python
# create and fit model
xgb_reg = xgb.XGBRegressor(nthread=2)
xgb_reg.fit(
    Xtrain,
    Ytrain,
    eval_set=[(Xtrain, Ytrain), (Xvalid, Yvalid)],
    verbose=False,
    early_stopping_rounds=20
)
```




    XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                 colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
                 importance_type='gain', interaction_constraints='',
                 learning_rate=0.300000012, max_delta_step=0, max_depth=6,
                 min_child_weight=1, missing=nan, monotone_constraints='()',
                 n_estimators=100, n_jobs=2, nthread=2, num_parallel_tree=1,
                 random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 subsample=1, tree_method='exact', validate_parameters=1,
                 verbosity=None)




```python
# predict locally
xgb_reg.predict(Xeval)
```




    array([ 154.1054  ,  160.67448 ,  -49.945793,  -98.16371 , -195.32375 ],
          dtype=float32)




```python
with open('xgb_reg.pickle', 'wb') as f:
    pickle.dump(xgb_reg, f)
```
