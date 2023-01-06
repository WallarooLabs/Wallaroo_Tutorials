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
    classes = infResult['outputs'][1]['Int64']['data']
    confidences = infResult['outputs'][2]['Float']['data']

    # Extract the bounding box coordinates for the object classifications and reshape them into a 1x4 array
    boxList =  infResult['outputs'][0]['Float']['data']
    boxA = numpy.array(boxList)
    boxes = boxA.reshape(-1, 4)
    boxes = boxes.astype(int)
    
    # Find the anomalies and add them to the inference results for later processing
    anomolyClasses = []
    anomolyConfidences = []
    anomolyBoxes = []
    for idx in range(0,len(confidences)):
        if confidences[idx] < 0.75: 
            anomolyClasses.append(classes[idx])
            anomolyConfidences.append(confidences[idx])
            anomolyBoxes.append(boxes[idx].tolist())

    result = {
        'original': [],
        'anomaly-classes': anomolyClasses,
        'anomaly-confidences': anomolyConfidences,
        'anomaly-boxes': anomolyBoxes
    }
    return(result)

    