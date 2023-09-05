import numpy
import pandas

import json


def remove_zero(x):
    if x[0] < 0:
        x[0] = 0
        return x
    else:
        x[0] = x[0]
        return x

# Check for negative predictions, and zero them out
def actual_postprocess(data:pandas.DataFrame):

    # if variables[x] < 0, replace with [0]
    data.variable.apply(remove_zero)

    # rename 'variable' to 'prediction'
    prediction = data.rename(columns={'variable':'prediction'})
    return prediction
    # # turn the results of predictions into a vector
    # # sklearn predictions are already a vector, so this is idempotent

    # # check the variable column for 0 values - dataframe filter
    # rows = predictions.shape[0]
    # predictions = predictions.reshape((rows, ))
    
    # return numpy.where(predictions < 0, 0, predictions)

# Expected input:
# A Dictionary with fields 'outputs',
# which contains a list with an element that is a dictionary
# with the fields 'data' - native array of outputs
#                 'dim' - of data
#                 'v' 
def wallaroo_json(data:pandas.DataFrame):
    # obj = json.loads(data)

    # outputs = numpy.array(obj['outputs'][0]['Double']['data'])

    # outputs = numpy.array(data['variable'].to_list())

    # predictions = actual_postprocess(outputs).tolist()

    # columns = ['prediction']

    # df = pd.DataFrame({
    #     'prediction': predictions
    # })

    # df['prediction'] = df['prediction'].map(lambda x: [x])

    # return df.to_dict(orient="records")

    return actual_postprocess(data).to_dict(orient="records")

    # return [
    #     {
    #         "prediction": actual_postprocess(outputs)
    #     }
    # ]
