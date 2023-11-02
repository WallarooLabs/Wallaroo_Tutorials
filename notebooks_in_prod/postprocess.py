import numpy as np
import json
import pandas

'''
Primary postprocessing procedure.
Convert log10 price back to price, and round to the nearest integer value.
'''
def actual_postprocess(log10price):

    # convert to log10
    x = np.rint(np.power(10, log10price[0]))
    log10price[0] = x
    return log10price

'''Postprocess used outside wallaroo
'''
def postprocess(log10price):
    return np.rint(np.power(10, log10price))

'''
This is the function that Wallaroo expects to call, to invoke the procedure above.
It expects input of the form (as a json string):
input_dict = {
    'outputs': list of model outputs
}
where each output is a dictionary of the form:
{
  'Float:' {
              'data': output of the upstream model (as a list of floats)
           }
}
returns output of the form :
output_dict = {
     'tensor': pandasframe.to_numpy().tolist()
}
'''
def wallaroo_json(data:pandas.DataFrame):
    # obj = json.loads(data)

    # outputs = np.array(obj['outputs'][0]['Float']['data'])
    
    
    # prediction = postprocess(outputs).tolist()
    # result = {
    #     'prediction': prediction
    # }
    # return(result)

    df = data.copy()

    return pandas.DataFrame(df.variable.apply(actual_postprocess)).to_dict(orient="records")