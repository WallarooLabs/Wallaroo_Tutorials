import numpy
import pandas

import json

# add interaction terms for the model
def actual_preprocess(pdata):
    df = pdata.copy()
    # convert boolean cust_known to 0/1
    df.cust_known = numpy.where(df.cust_known, 1, 0)
    # interact UnitPrice and cust_known
    df['UnitPriceXcust_known'] = df.UnitPrice * df.cust_known
    batch = df.loc[:, ['UnitPrice', 'cust_known', 'UnitPriceXcust_known']].to_numpy()
    return batch

# If the data is a json string, call this wrapper instead
# Expected input:
# a dictionary with fields 'colnames', 'data'

# test that the code works here
def wallaroo_json(data:pandas.DataFrame):
    # obj = json.loads(data)
    # pdata = pandas.DataFrame(obj['query'],
    #                          columns=obj['colnames'])
    pprocessed = actual_preprocess(data)

    # return pandas.DataFrame({pprocessed}, columns=['tensor']).to_dict(orient="records")
    


   # return a dictionary, with the fields the model expect
    return pandas.DataFrame({
        'tensor': pprocessed.tolist()
    }).to_dict(orient="records")
