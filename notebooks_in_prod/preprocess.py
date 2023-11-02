import datetime
import pandas as pd
import numpy as np
import json

_vars = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view',
'condition', 'grade', 'sqft_above', 'sqft_basement', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'house_age', 'renovated', 'yrs_since_reno']

'''
Primary preprocessing procedure.
Assumes incoming data in frame with at least columns:
 ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view',
  'condition', 'grade', 'sqft_above', 'sqft_basement', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'yr_built', 'yr_renovated']
 
 Creates the features 'house_age', 'renovated', 'yrs_since_reno'
'''
def create_features(housing_data:pd.DataFrame):
    thisyear = datetime.datetime.now().year
    housing_data['house_age'] = thisyear - housing_data['yr_built']
    housing_data['renovated'] =  np.where((housing_data['yr_renovated'] > 0), 1, 0) 
    housing_data['yrs_since_reno'] =  np.where(housing_data['renovated'], housing_data['yr_renovated'] - housing_data['yr_built'], 0)
    return housing_data.loc[:, _vars]


# If the data is a json string, call this wrapper instead
# Expected input:
# a dictionary with fields 'colnames', 'data'

'''
This is the function that Wallaroo expects to call, to invoke the procedure above.
It expects input of the form (as a json string):
input_dict = {
    'query': pandasframe.to_json(orient='split') # though I don't think it matters which orientation you use
    }
    
returns output of the form :
output_dict = {
     'tensor': pandasframe.to_numpy().tolist()
}
We don't need to re-convert it to pandas because the model takes numpy arrays as input.
'''
def wallaroo_json(data:pd.DataFrame):
    df = create_features(data)
    
    # convert to the shape that the onnx model expects
    df = pd.DataFrame({
        'tensor': df.to_numpy().tolist()
    })
    
    # then turn that into "json"
    value = df.to_dict(orient="records")
    print(value)
    return value