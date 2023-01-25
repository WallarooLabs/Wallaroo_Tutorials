import numpy
import math
import json
import pandas

def actual_postprocess(predictions):
    for i in range(len(predictions)):
        if predictions[i] < 0:
            predictions[i] = int(math.floor(predictions[i]))
        else:
            predictions[i] = int(round(predictions[i]))
    return predictions

# def wallaroo_json(data):
#     obj = json.loads(data)

#     outputs = numpy.array(obj['outputs'][0]['Double']['data'])
    
    
#     prediction = actual_postprocess(outputs).tolist()
#     result = {
#         'original': obj,
#         'prediction': prediction
#     }
#     return(result)

def wallaroo_json(data):
    obj = json.loads(data)

    outputs = numpy.array(obj['outputs'][0]['Double']['data'])

    prediction = actual_postprocess(outputs).tolist()
    result = {
        'outputs': obj['outputs'],
        'prediction': prediction
    }
    return(result)