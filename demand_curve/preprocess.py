import numpy
import pandas

import json

# add interaction terms for the model
def actual_preprocess(pdata):
    pd = pdata.copy()
    # convert boolean cust_known to 0/1
    pd.cust_known = numpy.where(pd.cust_known, 1, 0)
    # interact UnitPrice and cust_known
    pd['UnitPriceXcust_known'] = pd.UnitPrice * pd.cust_known
    return pd.loc[:, ['UnitPrice', 'cust_known', 'UnitPriceXcust_known']]


# If the data is a json string, call this wrapper instead
# Expected input:
# a dictionary with fields 'colnames', 'data'

# test that the code works here
def wallaroo_json(data):
    obj = json.loads(data)
    pdata = pandas.DataFrame(obj['query'],
                             columns=obj['colnames'])
    pprocessed = actual_preprocess(pdata)
    
   # return a dictionary, with the fields the model expect
    return {
       'tensor_fields': ['model_input'],
       'model_input': pprocessed.to_numpy().tolist()
    }