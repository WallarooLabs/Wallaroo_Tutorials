from PIL import Image
import numpy as np
import pandas as pd
import pyarrow as pa
import cv2
import wallaroo
import torch
import pickle
from datetime import datetime
import time
import matplotlib.pyplot as plt

def getWorkspace(wl, ws_name):
    workspace = None
    for ws in wl.list_workspaces():
        if ws.name() == ws_name:
            workspace= ws
    if(workspace == None):
        workspace = wl.create_workspace(ws_name)
    return workspace

def get_pipeline(wl, name):
    try:
        pipeline = wl.pipelines_by_name(name)[0]
    except EntityNotFoundError:
        pipeline = wl.build_pipeline(name)
    return pipeline

def mapColors(color):
    colorDict = {
        "AMBER": (0, 191, 255),
        "RED": (0, 0, 255),
        "GREEN": (0, 255, 0),
        "BLUE": (255, 0, 0),
        "BLACK": (0, 0, 0),
        "WHITE": (255, 255, 255),
        "CYAN": (255, 255, 0),
        "MAGENTA": (255, 0, 255),
        "ORANGE": (0, 165, 255)
    }

    return colorDict[color]

def getIOSchemas():
    # Function to quickly define and return pyarrow schemas for Python post-processing module
    field_boxes = pa.field('boxes', pa.list_(pa.list_(pa.float64(), 4)))
    field_classes = pa.field('classes', pa.list_(pa.int32()))
    field_confidences =  pa.field('confidences', pa.list_(pa.float64()))

    input_schema = pa.schema([field_boxes, field_classes, field_confidences])

    output_schema = pa.schema([field_boxes, field_classes, field_confidences, pa.field('avg_conf', pa.list_(pa.float64()))])

    return input_schema, output_schema

async def runInferences(pipeline, images_list):        
    assay_start = datetime.now()
    parallel_results = await pipeline.parallel_infer(tensor_list=images_list, timeout=20, num_parallel=3, retries=2)
    assay_end = datetime.now()
    
    return assay_start, assay_end

def processImages(images):
    images_list = []
    for image in images:
        width, height = 640, 480
        dfImage, _ = loadImageAndConvertToDataframe(image, width, height)

        images_list.append(dfImage)

    return images_list

async def runBaselineInferences(pipeline):
    baseline_images = [
    "./data/images/input/example/dairy_bottles.png",
    "./data/images/input/example/dairy_products.png",
    "./data/images/input/example/product_cheeses.png"
    ]

    baseline_images_list = processImages(baseline_images)

    assay_start, assay_end = await runInferences(pipeline, baseline_images_list)
    
    return assay_start, assay_end

async def runBlurredInferences(pipeline):
    blurred_images = [
    "./data/images/input/example/blurred-dairy_bottles.png",
    "./data/images/input/example/blurred-dairy_products.png",
    "./data/images/input/example/blurred-product_cheeses.png"
    ]

    blurred_images_list = processImages(blurred_images)
        
    assay_start, assay_end = await runInferences(pipeline, blurred_images_list)
    
    return assay_start, assay_end

async def simulateDrift(pipeline):
    baseline_images = [
            "./data/images/input/example/dairy_bottles.png",
            "./data/images/input/example/dairy_products.png",
            "./data/images/input/example/product_cheeses.png"
    ]

    baseline_images_list = processImages(baseline_images)
    
    
    blurred_images = [
        "./data/images/input/example/blurred-dairy_bottles.png",
        "./data/images/input/example/blurred-dairy_products.png",
        "./data/images/input/example/blurred-product_cheeses.png"
    ]

    blurred_images_list = processImages(blurred_images)
        
    assay_start = datetime.now()

    parallel_results = await pipeline.parallel_infer(tensor_list=baseline_images_list, timeout=20, num_parallel=3, retries=2)
    parallel_results = await pipeline.parallel_infer(tensor_list=baseline_images_list, timeout=20, num_parallel=3, retries=2)
    time.sleep(60)
    parallel_results = await pipeline.parallel_infer(tensor_list=baseline_images_list, timeout=20, num_parallel=3, retries=2)
    parallel_results = await pipeline.parallel_infer(tensor_list=blurred_images_list, timeout=20, num_parallel=3, retries=2)
    time.sleep(60)
    parallel_results = await pipeline.parallel_infer(tensor_list=blurred_images_list, timeout=20, num_parallel=3, retries=2)
    parallel_results = await pipeline.parallel_infer(tensor_list=blurred_images_list, timeout=20, num_parallel=3, retries=2)
    time.sleep(60)

    assay_end = datetime.now()
    
    return assay_start, assay_end

def loadImageAndConvertToDataframe(imagePath, width, height):
    tensor, resizedImage = loadImageAndResize(imagePath, width, height)

    # get npArray from the tensorFloat
    npArray = tensor.cpu().numpy()

    #creates a dictionary with the wallaroo "tensor" key and the numpy ndim array representing image as the value.
    df = pd.DataFrame({"tensor":[npArray]})
    return df, resizedImage

def loadImageAndResize(imagePath, width, height):
    image = Image.open(imagePath)

    im_pillow = np.array(image)
    image = cv2.cvtColor(im_pillow, cv2.COLOR_RGB2BGR)

    image = cv2.resize(image, (width, height))
    resizedImage = image.copy()

    # convert the image from BGR to RGB channel ordering and change the
    # image from channels last to channels first ordering
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose((2, 0, 1))

    # add the batch dimension, scale the raw pixel intensities to the
    # range [0, 1], and convert the image to a floating point tensor
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    tensor = torch.FloatTensor(image)
    return tensor, resizedImage

def drawDetectedObjectsFromInference(results):
    image = drawDetectedObjectClassifications(results)
    
    frameStats = "Frame"
    statsRowHeight = 25
    rows = 2
    statsHeight = statsRowHeight*rows
    statsImage = np.zeros([statsHeight,results['width'],3],dtype=np.uint8)
    
    results['color'] = (255,255,255)
    statsImage = drawStatsDashboard("Wallaroo Computer Vision Statistics Dashboard", results)
    image = cv2.vconcat([statsImage,image])

    # show the output image
    pltImshow("Output", image)

    return

def drawShadowDetectedObjectsFromInference(results, challenger):
    image = drawShadowDetectedObjectClassifications(results, challenger)
    
    frameStats = "Frame"
    statsRowHeight = 25
    rows = 2
    statsHeight = statsRowHeight*rows
    statsImage = np.zeros([statsHeight,results['width'],3],dtype=np.uint8)
    
    results['color'] = (255,255,255)
    statsImage = drawStatsDashboard("Wallaroo Computer Vision Statistics Dashboard", results)
    image = cv2.vconcat([statsImage,image])

    # show the output image
    pltImshow("Output", image)

    return

def drawShadowDetectedObjectClassifications(results, challenger):
    infResults = results['inf-results']
    if isinstance(infResults, pd.DataFrame):
        boxList = infResults[f'out_{challenger}.boxes'].tolist()
        classes = infResults[f'out_{challenger}.classes'].tolist()[0]
        results['classes'] = classes
        confidences = infResults[f'out_{challenger}.confidences'].tolist()[0]
        results['confidences'] = confidences
    else:
        outputs = infResults['shadow_data']
        boxes = outputs[challenger][0]
    
        #used later in dashboard
        results['boxes'] = boxes
            
        classes = outputs[challenger][1]['Int64']['data']
        #used later in dashboard
        results['classes'] = classes
    
        confidences = outputs[challenger][2]['Float']['data']
        results['confidences'] = confidences
            
        # Reshape box coord inferences to array with 4 elements (x,y,w,h)
        boxList = boxes['Float']['data']
        
    boxA = np.array(boxList)
    boxes = boxA.reshape(-1, 4)
    boxes = boxes.astype(int)
    results['boxes'] = boxes
    
    cocoClassPath = results['classes_file']
    classes = results['classes']
    image = results['image']
    modelName = results['model_name']
    infTime = "{:.2f}".format(results['inference-time'])
    

    for i in range(0, len(boxes)):
        # extract the confidence (i.e., probability) associated with the
        # classification prediction
        confidence = confidences[i]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
            # display the prediction to our terminal

        if confidence > results['confidence-target']:
            idx = int(classes[i])
            cocoClasses = getCocoClasses(cocoClassPath)
            label = "{}: {:.2f}%".format(cocoClasses[idx], confidence * 100)
            # extract the index of the class label from the detections,
            # then compute the (x, y)-coordinates of the bounding box
            # for the object
            box = boxes[i]
            (startX, startY, endX, endY) = box

            color = results['color']
                # draw the bounding box and label on the image
            cv2.rectangle(image, (startX, startY), (endX, endY),
                color, 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image   


def drawDetectedObjectClassifications(results):
    infResults = results['inf-results']
    if isinstance(infResults, pd.DataFrame):
        boxList = infResults['out.boxes'].tolist()
        classes = infResults['out.classes'].tolist()[0]
        results['classes'] = classes
        confidences = infResults['out.confidences'].tolist()[0]
        results['confidences'] = confidences
    else:
        outputs = infResults['outputs']
        boxes = outputs[0]

        #used later in dashboard
        results['boxes'] = boxes

        classes = outputs[1]['Int64']['data']
        #used later in dashboard
        results['classes'] = classes

        confidences = outputs[2]['Float']['data']
        results['confidences'] = confidences


        # Reshape box coord inferences to array with 4 elements (x,y,w,h)
        boxList = boxes['Float']['data']

    boxA = np.array(boxList)
    boxes = boxA.reshape(-1, 4)
    boxes = boxes.astype(int)
    results['boxes'] = boxes

    image = results['image']
    modelName = results['model_name']
    cocoClassPath = results['classes_file']
    infTime = "{:.2f}".format(results['inference-time'])
        
    for i in range(0, len(boxes)):
        # extract the confidence (i.e., probability) associated with the
        # classification prediction
        confidence = confidences[i]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
            # display the prediction to our terminal


        if confidence > results['confidence-target']:
            idx = int(classes[i])
            cocoClasses = getCocoClasses(cocoClassPath)
            label = "{}: {:.2f}%".format(cocoClasses[idx], confidence * 100)
            # extract the index of the class label from the detections,
            # then compute the (x, y)-coordinates of the bounding box
            # for the object
            box = boxes[i]
            (startX, startY, endX, endY) = box

            color = mapColors(results['color'])
            cv2.rectangle(image, (startX, startY), (endX, endY),
                color, 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    return image


def getCocoClasses(classPath):
    classes = pickle.loads(open(classPath, "rb").read())

    return classes

def drawStatsDashboard(title, results):
    statsRowHeight = 25
    rows = 2
    
    if ('anomaly-count' in results):
        rows += 1
    
    statsHeight = statsRowHeight*rows
    statsImage = np.zeros([statsHeight,results['width'],3],dtype=np.uint8)
    
    results['color'] = (255,255,255)
    statsImage = drawStats("Wallaroo Computer Vision Statistics Dashboard",statsImage, results, 1)
    return statsImage

def drawStats(title, image, config, row):
    # clear image
    image.fill(0)

    avgScore = 0
    confidences = config['confidences']
    classes = config['classes']
    classCnt = len(classes)
    avgScore = 0.0
    if (len(confidences) > 0):
        if 'confidence-target' in config:
            target = config['confidence-target']
            arrayConfidences = np.array(confidences)
            arrayConfidences = arrayConfidences[arrayConfidences > target]
            avgScore = np.mean(arrayConfidences)*100
        else:
            avgScore = np.mean(confidences)*100
        
    boxes = config['boxes']
    modelName = config['model_name']
    pipelineName = config['pipeline_name']
    infTime = config['inference-time']
    onnxTime = config['onnx-time']
        
    rowHeight=20
    y = row*rowHeight
        
    # sample text and font
    #unicode_text = u"Hello World!"
    #font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 14, encoding="unic")

    classCnt = len(set(classes))
    msg = "M: "+modelName+ " P: "+pipelineName+ "  Inf: {:2.3f}".format(infTime) + "/{:2.3f}".format(onnxTime) + " Obj: " + str(len(boxes)) + "  Cls: " +str(classCnt) + " Conf: {:3.2f}%".format(avgScore)
    fontThickness = 1
    fontScale = 1
    fontColor = config['color']
    
    # Draw Title
    titleSize = cv2.getTextSize(title, cv2.FONT_HERSHEY_PLAIN, 1, 2)[0]
    # get coords based on boundary
    titleX = int((config['width'] - titleSize[0]) / 2)
    
    #textY = (img.shape[0] + textsize[1]) / 2
    rowHeight = 20
    row = y
    cv2.putText(image, title, (titleX, row), cv2.FONT_HERSHEY_PLAIN, fontScale, fontColor, fontThickness, lineType = cv2.LINE_AA)
    row += rowHeight
    cv2.putText(image, msg, (5, row), cv2.FONT_HERSHEY_PLAIN, fontScale, fontColor, fontThickness, lineType = cv2.LINE_AA)
    
    if ('anomaly-count' in config):
        row += rowHeight
        fontColor = mapColors("AMBER")
        msg = "Anomalies: "+str(config['anomaly-count'])
        cv2.putText(image, msg, (5, row), cv2.FONT_HERSHEY_PLAIN, fontScale, fontColor, fontThickness, lineType = cv2.LINE_AA)
    return image

def pltImshow(title, image):
    # convert the image frame BGR to RGB color space and display it
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12,8))
    plt.title(title)
    plt.grid(False)
    plt.imshow(image)
    plt.show()

    return


