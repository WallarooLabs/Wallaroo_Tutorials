## Sklearn Linear Regression Model Generation
  
The following tutorial demonstrates how to generate a **sklearn** Linear Regression Model.  This is used as part of the `auto-convert-to-onnx-tutorial.ipynb` tutorial.


```python
import numpy as np
import pandas as pd

import sklearn
import sklearn.linear_model
import sklearn.datasets

import pickle
import json

import wallaroo

from wallaroo.ModelConversion import ConvertSKLearnArguments, ModelConversionSource, ModelConversionInputType
```

# Linear Regression


```python
# create data
N = 1000
Neval = 5
NF = 25
Ninformative = 10

X, Y = sklearn.datasets.make_regression(n_samples=N, n_features=NF, n_informative=Ninformative)

Ntrain = N - Neval
Xtrain = X[0:Ntrain, :]
Ytrain = Y[0:Ntrain]

Xeval = X[Ntrain:N, :]
Yeval = Y[Ntrain:N]

# create and fit model
lm = sklearn.linear_model.LinearRegression()
lm.fit(Xtrain, Ytrain)
```




    LinearRegression()




```python
# predict locally
lm.predict(Xeval)
```




    array([ 173.41230126,   55.88947637,  -94.05667871, -147.37732616,
           -137.24328133])




```python
with open('sklearn-linear-model.pickle', 'wb') as f:
    pickle.dump(lm, f)
```
