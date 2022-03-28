import numpy
import pandas

import json

# Check for negative predictions, and zero them out
def actual_postprocess(predictions):
    # turn the results of predictions into a vector
    # sklearn predictions are already a vector, so this is idempotent
    rows = predictions.shape[0]
    predictions = predictions.reshape((rows, ))
    
    return numpy.where(predictions < 0, 0, predictions)


# Expected input:
# A Dictionary with fields 'outputs',
# which contains a list with an element that is a dictionary
# with the fields 'data' - native array of outputs
#                 'dim' - of data
#                 'v' 
def wallaroo_json(data):
    obj = json.loads(data)

    outputs = numpy.array(obj['outputs'][0]['Double']['data'])
    
    
    prediction = actual_postprocess(outputs).tolist()
    result = {
        'original': obj,
        'prediction': prediction
    }
    return(result)

    