import numpy
import pandas
import json


# Expected input:
# A Dictionary with fields 'outputs',
# which contains a list with an element that is a dictionary
# with the fields 'data' - native array of outputs
#                 'dim' - of data
#                 'v' 
def wallaroo_json(data):
    infResult = json.loads(data)
    
    # Extract the identified object classifications and confidences from the model's inference results
    #classes = infResult['outputs'][1]['Int64']['data']
    confidences = infResult['outputs'][2]['Float']['data']

    avgConf = numpy.mean(confidences)
 
    result = {
        'original': {},
        'prediction': [ avgConf ]
    }
    return(result)