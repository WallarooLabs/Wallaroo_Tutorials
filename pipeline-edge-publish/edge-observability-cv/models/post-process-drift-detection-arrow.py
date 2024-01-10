import numpy
import pandas as pd
import statistics 
from statistics import mean
import json


# Expected input:
# A Dictionary with fields 'outputs',
# which contains a list with an element that is a dictionary
# with the fields 'data' - native array of outputs
#                 'dim' - of data
#                 'v' 
def wallaroo_json(data: pd.DataFrame):
    print("-- wallaroo_json -- ")
    # calculate precision
    
    confidences = data["confidences"][0]

    print(confidences)
    classes = data["classes"][0]
    print(classes)
    boxes = data["boxes"][0]
    print(boxes)
    
    # avg_conf = [data["confidences"].mean()]
    # Why the extra 
    avg_conf = numpy.array(confidences).mean()
    print(avg_conf)
    print("-- boxes --")
 
    # calculate intersection of union
    return [
             {'boxes' : boxes},
             {'classes' : classes},
             {'confidences' : confidences},
             {'avg_conf': [avg_conf] }
           ]
