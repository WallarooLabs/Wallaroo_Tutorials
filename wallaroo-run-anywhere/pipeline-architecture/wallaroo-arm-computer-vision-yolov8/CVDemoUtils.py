#import yolov5
#from yolov5.utils.general import non_max_suppression
import wallaroo
import matplotlib.pyplot as plt
from torchvision.models import detection
from torchvision.models.detection import fasterrcnn_resnet50_fpn, fasterrcnn_mobilenet_v3_large_320_fpn
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch 
import cv2
from PIL import Image
import os
import onnx
import onnxruntime as ort
import json
import time
import requests
import imutils
from base64 import b64encode
from IPython.display import display, HTML
from datetime import datetime
import pandas as pd
import tensorflow
import pytz
import pandas as pd
#from keras.utils import normalize
from tensorflow.keras.utils import normalize
import argparse

#
# Wallaroo CV Demo class provides some helper functions for rendering inferences results onto images
# Author: Will Berger
class CVDemo():
    newYorkTz = pytz.timezone("America/New_York")
    format = "%m-%d-%Y %H:%M:%S.%f"
    
    
    # BGR Representation for opencv
    AMBER = (0, 191, 255)
    RED = (0, 0, 255)
    #RED = (69, 69, 255)

    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    CYAN = (255, 255, 0)
    MAGENTA = (255, 0, 255)
    ORANGE = (0, 165, 255)
    
    ALIGN_CENTER= "center"
    ALIGN_LEFT= "left"
    ALIGN_RIGHT="right"
    
    COCO_CLASSES_PATH = "coco_classes.pickle"
    
    DASHBOARD_CELL_PADDING = 5.0
    #def __init__(self, classes, colors, device):
    #    self.CLASSES = classes
    #    self.COLORS = colors
    #    self.DEVICE = device
    #    self.DEBUG = True # used to provide verbose output

   
    
    def __init__(self):
        self.DEBUG = False # used to provide verbose output
        # set the device we will be using to run the model
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # The set of COCO classifications
        self.CLASSES = None
        
        # Unique colors for each identified COCO class
        self.COLORS = None

    def setCurrentWorkspace(self, wl, ws_name):
        found = False
        ws = wl.list_workspaces()
        for w in ws:
            if w.name() == ws_name:
                wl.set_current_workspace(w)
                found = True
                break
        if not found:
            workspace = wl.create_workspace(ws_name)
            wl.set_current_workspace(workspace)

        
    def getCocoClasses(self):
        if self.CLASSES == None:
            self.CLASSES = pickle.loads(open(self.COCO_CLASSES_PATH, "rb").read())
            self.COLORS = np.random.uniform(0, 1, size=(len(self.CLASSES), 3))

        return self.CLASSES
    
    def debug(self, value):
        if self.DEBUG == True:
            localTime = datetime.now()
            #localTime = datetime.now(CVDemo.newYorkTz)

            localTime = localTime.strftime(CVDemo.format)
            print(localTime + " " + str(value))
            
    def print(self, value):
        localTime = datetime.now()
        #localTime = datetime.now(CVDemo.newYorkTz)

        localTime = localTime.strftime(CVDemo.format)
        print(localTime + " " + str(value))

    def playVideo(self,path, width, height):
        mp4 = open(path,'rb').read()
        data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
        
        htmlStr = "<video width="+str(width)+ " height=" + str(height) +" control><source src=\"" + data_url + "\" type=\"video/mp4\"> </video>"
        #self.print(htmlStr)
        display(htmlStr)
     
    # displays the image with a title in jupyter notebook
    def pltImshow(self, title, image):
        # convert the image frame BGR to RGB color space and display it
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imgage = cv2.flip(image, 1)
        imgage = cv2.flip(image, 1)
        plt.figure(figsize=(16,12))
        plt.title(title)
        plt.grid(False)
        plt.imshow(image)
        plt.show()
        
    
    # attempts to load a pipeline
    def loadPipeline(self, name,wl):
        try:
            pipeline = wl.pipelines_by_name(name)[0]
        except EntityNotFoundError:
            print("Could not load the pipeline named ["+name+"]")
        return pipeline

    # load the image from disk, convert to BGR, resize to specified width, height, convert the image back to RGB
    # convert the image to a float tensor and returns it.  Also return the original resized image for drawing bounding boxes in BGR
    def loadImageAndResize(self, imagePath, width, height):
        return self.imageResize(Image.open(imagePath), width, height)
        
    
    # load the image from disk, convert to BGR, resize to specified width, height, convert the image back to RGB
    # convert the image to a float tensor and returns it.  Also return the original resized image for drawing bounding boxes in BGR
    def imageResize(self, image, width, height):
        #self.print("Image Mode:"+image.mode)
        im_pillow = np.array(image)
        image = cv2.cvtColor(im_pillow, cv2.COLOR_BGR2RGB) #scott
        image = cv2.flip(im_pillow, 1)
        image = cv2.flip(image, 1)
        #image = cv2.imread(im_pillow)
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #image = cv2.cvtColor(im_pillow, cv2.COLOR_GRAY2BGR)
        self.debug("Resizing to w:"+str(width) + " height:"+str(height))
        image = cv2.resize(image, (width, height))
        resizedImage = image.copy()

        # convert the image from BGR to RGB channel ordering and change the
        # image from channels last to channels first ordering
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))

        # add the batch dimension, scale the raw pixel intensities to the
        # range [0, 1], and convert the image to a floating point tensor
        image = np.expand_dims(image, axis=0)
        image = image / 255.0
        tensor = torch.FloatTensor(image)
        return tensor, resizedImage
    
   
    # load the image from disk, convert to BGR, resize to specified width, height, convert the image back to RGB
    # convert the image to a float tensor and returns it.  Also return the original resized image for drawing bounding boxes in BGR
    def loadImageAndResizeTiff(self, imagePath, width, height):
        return self.imageResizeTiff(Image.open(imagePath), width, height)
        
    
    # load the image from disk, convert to BGR, resize to specified width, height, convert the image back to RGB
    # convert the image to a float tensor and returns it.  Also return the original resized image for drawing bounding boxes in BGR
    def imageResizeTiff(self, image, width, height):
        

        #self.print("Image Mode:"+image.mode)
        im_pillow = np.array(image)
        image = cv2.cvtColor(im_pillow, cv2.COLOR_RGB2BGR)
        #image = cv2.imread('images/example_09-v2.jpg')
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        self.debug("Resizing to w:"+str(width) + " height:"+str(height))
        image = cv2.resize(image, (width, height))
        resizedImage = image.copy()

        # convert the image from BGR to RGB channel ordering and change the
        # image from channels last to channels first ordering
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
#        image = image.transpose((2, 0, 1))

        # add the batch dimension, scale the raw pixel intensities to the
        # range [0, 1], and convert the image to a floating point tensor
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=0)

        image = image.transpose((1, 2, 3, 0))
        image = image / 255.0
        tensor = torch.FloatTensor(image)
        print("tensor shape")
        print(tensor.shape)
        return tensor, resizedImage

    # converts the pytorch model to onnx by loading the given pytorch model using the sampleImage as the input
    # converting to onnx and saving the onnx model with the name pytorchModelPath + ".onnx"
    def loadPytorchAndConvertToOnnx(self, pytorchModelPath, sampleImagePath, width, height):
        model = torch.load(pytorchModelPath)

        device = 'cpu'
        model = model.to(device)

        tensor, resizedImage = self.loadImageAndResize(sampleImagePath, width, height)
        input_names = ["data"]
        output_names = ["output"]
        torch.onnx.export(model,
                          tensor,
                          pytorchModelPath+'.onnx',
                          input_names=input_names,
                          output_names=output_names,
                          opset_version=11,
                          )

    # loads the image and resizes it to width, height, runs inference on model to detect objects, bounding boxes, and classes
    # draws the bounding boxes, coco classificaiton, and confidence on the orig image
    def detectAndClassifyObjectsWithPytorchModel(self, imagePath, model, width, height, confidence_target):
        
        tensor, resizedImage = self.loadImageAndResize(imagePath,width,height)

        im_pillow = np.array(tensor)
        self.debug(tensor.shape)

        # send the input to the device and pass the it through the network to
        # get the detections and predictions
        tensor = tensor.to(self.DEVICE)
        detections = model(tensor)[0]

        # loop over the detections
        for i in range(0, len(detections["boxes"])):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections["scores"][i]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > confidence_target:
                # extract the index of the class label from the detections,
                # then compute the (x, y)-coordinates of the bounding box
                # for the object
                idx = int(detections["labels"][i])
                box = detections["boxes"][i].detach().cpu().numpy()
                (startX, startY, endX, endY) = box.astype("int")

                # display the prediction to our terminal
                cocoClasses = self.getCocoClasses()
                label = "{}: {:.2f}%".format(cocoClasses[idx], confidence * 100)
                self.debug("[INFO] {}".format(label))

                # draw the bounding box and label on the image
                cv2.rectangle(resizedImage, (startX, startY), (endX, endY),
                    self.COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(resizedImage, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS[idx], 2)

        # show the output image
        self.pltImshow("Output", resizedImage)
       
    
    
    def saveInputToFile(self,inputFilePath, npArray):
        # handles converting ndarray to lists
        class NumpyArrayEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        #creates a dictionary with the wallaroo "tensor" key and the numpy ndim array representing image as the value.
        dictData = {"tensor": npArray}
        jsonInput = json.dumps(dictData, cls=NumpyArrayEncoder)
        with open(inputFilePath, "w") as outfile:
            outfile.write(jsonInput)
            
    def saveDataToJsonFile(self,filePath, data):
        # handles converting ndarray to lists
        class NumpyArrayEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        #creates a dictionary with the wallaroo "tensor" key and the numpy ndim array representing image as the value.
        dictData = {"data": data}
        jsonData = json.dumps(dictData, cls=NumpyArrayEncoder)
        with open(filePath, "w") as outfile:
            outfile.write(jsonData)
            
    #def saveJsonToFile(self,filePath, data):
    #    with open(filePath, 'w') as f:
    #        json.dump(data, f)

    # loads the image and resizes it to width, height, runs inference on model to detect objects, bounding boxes, and classes
    # draws the bounding boxes, coco classificaiton, and confidence on the orig image
    def detectAndClassifyObjectsWithOnnxModel(self, imagePath, onnx_model_path, width, height, confidence_target):
     
        # load the image we want to categorize using the onnx model
        tensor, resizedImage = self.loadImageAndResize(imagePath,width,height)

        npArray = tensor.cpu().numpy()
        #self.saveInputToFile("onnx-input.json",npArray)

        onnx_session= ort.InferenceSession(onnx_model_path)
        model = onnx.load(onnx_model_path)
        onnx.checker.check_model(model)
        
        #input_shapes = [[d.dim_value for d in _input.type.tensor_type.shape.dim] for _input in model.graph.input]
        #self.print("input_shapes")
        #self.print(input_shapes)

        onnx_inputs= {onnx_session.get_inputs()[0].name: npArray}
        onnx_output = onnx_session.run(None, onnx_inputs)
        out_y = onnx_output[0]

        boxes = onnx_output[0]
        categories = onnx_output[1]
        scores = onnx_output[2]

        confidenceLevel = 0.25
        im_pillow = np.array(Image.open(imagePath))

        imageBgr = cv2.cvtColor(im_pillow, cv2.COLOR_RGB2BGR)
        #image = cv2.imread('images/example_09-v2.jpg')
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #orig = imageBgr.copy()

        for i in range(0, len(boxes)):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = scores[i]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > confidenceLevel:
                # extract the index of the class label from the detections,
                # then compute the (x, y)-coordinates of the bounding box
                # for the object
                idx = int(categories[i])
                box = boxes[i]
                (startX, startY, endX, endY) = box.astype("int")

                # display the prediction to our terminal
                cocoClasses = self.getCocoClasses()

                label = "{}: {:.2f}%".format(cocoClasses[idx], confidence * 100)
                self.debug("[INFO] {}".format(label))

                # draw the bounding box and label on the image
                cv2.rectangle(resizedImage, (startX, startY), (endX, endY),
                    self.COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(resizedImage, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS[idx], 2)
        # show the output image
        self.pltImshow("Output", resizedImage)
        return model
        
    # loads the image and resizes it to width, height, runs inference on model to detect objects, bounding boxes, and classes
    # draws the bounding boxes, coco classificaiton, and confidence on the orig image
    #def drawDetectedObjectsWithClassification(self, modelName, image, boxes, classes, confidences, confidenceLevel):
    def drawDetectedObjectsWithClassification(self, results):
        confidences = results['confidences']
        boxes = results['boxes']
        
        # Reshape box coord inferences to array with 4 elements (x,y,w,h)
        boxList = boxes
        boxA = np.array(boxList)
        boxes = boxA.reshape(-1, 4)
        boxes = boxes.astype(int)
        
        classes = results['classes']
        image = results['image']
        modelName = results['model_name']
        infTime = "{:.2f}".format(results['inference-time'])
        #self.print("drawDetectedObjectsWithClassification boxes"+str(boxes))
        self.debug("confidence-target="+str(results['confidence-target']))
        for i in range(0, len(boxes)):
            # extract the confidence (i.e., probability) associated with the
            # classification prediction
            confidence = confidences[i]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
              # display the prediction to our terminal
           

            if confidence > results['confidence-target']:
                idx = int(classes[i])
                cocoClasses = self.getCocoClasses()
                label = "{}: {:.2f}%".format(cocoClasses[idx], confidence * 100)
                self.debug("[INFO] {}".format(label))
                # extract the index of the class label from the detections,
                # then compute the (x, y)-coordinates of the bounding box
                # for the object
                box = boxes[i]
                (startX, startY, endX, endY) = box
                   
                color = results['color']
                self.debug("color="+str(color))
                 # draw the bounding box and label on the image
                cv2.rectangle(image, (startX, startY), (endX, endY),
                    color, 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(image, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if results['write-to-disk'] == True: #aaron
            serialname = results['serial']
            cv2.imwrite(f'output_images/{serialname}.jpg', image)

        return image
        
        
    
    
    # loads the image and resizes it to width, height, runs inference on model to detect objects, bounding boxes, and classes
    # draws the bounding boxes, coco classificaiton, and confidence on the orig image
    #def drawDetectedObjectsWithClassification(self, modelName, image, boxes, classes, confidences, confidenceLevel):
    def drawDetectedAnomaliesWithClassification(self, results):
        confidences = results['anomalyConfidences']
        boxes = results['anomalyBoxes']
       
        # Reshape box coord inferences to array with 4 elements (x,y,w,h)
        boxList = boxes
        boxA = np.array(boxList)
        boxes = boxA.reshape(-1, 4)
        boxes = boxes.astype(int)
        
        classes = results['anomalyClasses']
        image = results['image']
        modelName = results['model_name']
        infTime = "{:.2f}".format(results['inference-time'])
        #self.print("drawDetectedObjectsWithClassification boxes"+str(boxes))
        self.debug("confidence-target="+str(results['confidence-target']))
        for i in range(0, len(boxes)):
            # extract the confidence (i.e., probability) associated with the
            # classification prediction
            confidence = confidences[i]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
              # display the prediction to our terminal
           

            if confidence > results['confidence-target']:
                idx = int(classes[i])
                cocoClasses = self.getCocoClasses()
                label = "{}: {:.2f}%".format(cocoClasses[idx], confidence)
                self.debug("[INFO] {}".format(label))
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
        
    
    def drawAndDisplayDetectedObjectsWithClassification(self, results):
        image = self.drawDetectedObjectsWithClassification(results)
        
        frameStats = "Frame"
        statsRowHeight = 25
        rows = 2
        statsHeight = statsRowHeight*rows

        statsImage = np.zeros([statsHeight,results['width'],3],dtype=np.uint8)
        
        results['color'] = (255,255,255)
        statsImage = self.drawStatsDashboard("Wallaroo Computer Vision Statistics Dashboard", results)
        #image = cv2.vconcat([statsImage,image]) #scott - this is erroring on sending second image - not sure why


        # show the output image #aaron
        if results['show-image'] == True:
            self.pltImshow("Output", image)
            
   
    
    # converts the frame to json with key "tensor" with size with and height
    def convertFrameToJsonTensor(self, frame, width, height):

        # The image width and height needs to be set to what the model was trained for.  In this case 640x480.
        #tensor, resizedImage = self.imageFrameResize(frame, width, height)
        #tensor, resizedImage = self.loadImageAndResize(frame, width, height)

        # convert the image from BGR to RGB channel ordering and change the
        # image from channels last to channels first ordering
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))

        # add the batch dimension, scale the raw pixel intensities to the
        # range [0, 1], and convert the image to a floating point tensor
        image = np.expand_dims(image, axis=0)
        image = image / 255.0
        tensor = torch.FloatTensor(image)

        # get npArray from the tensorFloat
        npArray = tensor.cpu().numpy()
        self.debug("frame shape:"+str(npArray.shape))

        # handles converting ndarray to lists
        class NumpyArrayEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        #creates a dictionary with the wallaroo "tensor" key and the numpy ndim array representing image as the value.
        dictData = {"tensor": npArray}
        jsonInput = json.dumps(dictData, cls=NumpyArrayEncoder)
        # Writing to sample.json
        #with open("sample-input.json", "w") as outfile:
        #    outfile.write(jsonInput)
        return jsonInput
    
    # converts the frame to json with key "tensor" with size with and height
    def convertFrameToTensorDict(self, frame, width, height):

        # The image width and height needs to be set to what the model was trained for.  In this case 640x480.
        #tensor, resizedImage = self.imageFrameResize(frame, width, height)
        #tensor, resizedImage = self.loadImageAndResize(frame, width, height)

        # convert the image from BGR to RGB channel ordering and change the
        # image from channels last to channels first ordering
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))

        # add the batch dimension, scale the raw pixel intensities to the
        # range [0, 1], and convert the image to a floating point tensor
        image = np.expand_dims(image, axis=0)
        image = image / 255.0
        tensor = torch.FloatTensor(image)

        # get npArray from the tensorFloat
        npArray = tensor.cpu().numpy()
        self.debug("frame shape:"+str(npArray.shape))

        
        #creates a dictionary with the wallaroo "tensor" key and the numpy ndim array representing image as the value.
        dictData = {"tensor": npArray.tolist()}
        
        return dictData
    
    # converts the frame to json with key "tensor" with size with and height
    def convertFrameToTensor(self, frame):

        # The image width and height needs to be set to what the model was trained for.  In this case 640x480.
        #tensor, resizedImage = self.imageFrameResize(frame, width, height)
        #tensor, resizedImage = self.loadImageAndResize(frame, width, height)

        # convert the image from BGR to RGB channel ordering and change the
        # image from channels last to channels first ordering
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))

        # add the batch dimension, scale the raw pixel intensities to the
        # range [0, 1], and convert the image to a floating point tensor
        image = np.expand_dims(image, axis=0)
        image = image / 255.0
        tensor = torch.FloatTensor(image)

        # get npArray from the tensorFloat
        npArray = tensor.cpu().numpy()
        self.debug("frame shape:"+str(npArray.shape))
        
        return npArray
    
    def convertWallarooJsonToInferenceResultDict(self,jsonResult):
        infResult = {}
        
        infResult['model_name'] = jsonResult[0]['model_name']
        infResult['pipeline_name'] = jsonResult[0]['pipeline_name']

        infResult['boxes'] = jsonResult[0]['outputs'][0]['Float']['data']
        infResult['classes'] = jsonResult[0]['outputs'][1]['Int64']['data']
        infResult['confidences'] = jsonResult[0]['outputs'][2]['Float']['data']
        infResult['onnx-time'] =  int(jsonResult[0]['elapsed']) / 1e+9

        self.debug("infResult")
        self.debug(infResult)
        return infResult
    #
    #
    #
    def runInferenceOnFrameUsingApi(self, frame, pipelineEndPointUrl, width, height):
        jsonInput = self.convertFrameToJsonTensor(frame, width, height)
        # Run inference by calling the wallaroo url rest api endpoint feeding it the jsonInput
        jsonOutput = ""
        try:
            self.debug("Running wallaroo pipeline inference using image represented as json")
            self.debug("Url Endpoint:"+pipelineEndPointUrl)
            headers = {'Content-type': 'application/json', 'content-encoding':'gzip', 'Accept': 'text/plain'}
            startTime = time.time()
            response = requests.post(pipelineEndPointUrl, data=jsonInput, headers=headers)
            self.print("status_code="+str(response.status_code))
            jsonOutput = response.json()
            if (response.status_code != 200):
                self.print("printing response's jsonOutput")
                self.print(jsonOutput)
                return None
            
            if not isinstance(jsonOutput, list):
                 self.print("json response not a list")
                 self.print(jsonOutput)
                 return None
                
            outputs = jsonOutput[0]['outputs']
            if outputs is None:
                self.print("Could not extract inference results from jsonOutput")
                self.print(jsonOutput)
            
            # get rid of orginal_data
            jsonOutput[0].pop('original_data', None)

            #if 'outputs' not in jsonOutput[0]:
            #    self.print(jsonOutput)
            #    jsonOutput = None
            #with open("sample-output.json", "w") as outfile:
            #    outfile.write(jsonOutput)
            #jsonOutput[0].pop('original_data', None)
            
            endTime = time.time()
            jsonOutput[0]['inference-time'] = (endTime - startTime)
            
            infResult = self.convertWallarooJsonToInferenceResultDict(jsonOutput)

        except Exception as e:
            self.print("An Exception occurred:")
            self.print(e)
            self.print("jsonOutput")
            self.print(jsonOutput)
            self.print("Could not inference frame")
            infResult = None
        return infResult
    
    
    def box_label(self, image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        imageShape = np.asarray(image)
        lw = max(round(sum(imageShape.shape) / 2 * 0.003), 2)
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image,
                label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                lw / 4, #3
                txt_color,
                thickness=tf,
                lineType=cv2.LINE_AA)

    def drawYolo8Boxes(self,inference, resizedImage, width, height, confidence_thres, iou_thres, draw=False):
        import ultralytics
        from ultralytics.utils import ROOT, yaml_load
        from ultralytics.utils.checks import check_requirements, check_yaml
        find = 0
        pred = np.array(inference["out.output0"][0]).reshape(1, 84, 8400) 
        #outputs = np.transpose(np.squeeze(inference[0]))
        outputs = np.transpose(np.squeeze(pred[0]))
        # Get the number of rows in the outputs array
        rows = outputs.shape[0]
        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []
        # Calculate the scaling factors for the bounding box coordinates
        x_factor = width / width
        y_factor = height / height
        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]
            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)
            #print(f"max score: {max_score*1000}")
            # If the maximum score is above the confidence threshold
            if max_score >= confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)
                #print(f"classid:{class_id}")
                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])
        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_thres, iou_thres)
        #print(f"Top items found from confidence level={confidence_thres}: {indices}")
        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            classes = yaml_load(check_yaml('coco128.yaml'))['names']
            # Generate a color palette for the classes
            color_palette = np.random.uniform(0, 255, size=(len(classes), 3))
            # Extract the coordinates of the bounding box
            x1, y1, w, h = box
            # Retrieve the color for the class ID
            color = color_palette[class_id]
            # Draw the bounding box on the image
            cv2.rectangle(resizedImage, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
            # Create the label text with class name and score
            label = f'{classes[class_id]}: {score:.2f}'
            # Calculate the dimensions of the label text
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # Calculate the position of the label text
            label_x = x1
            label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
            # Draw a filled rectangle as the background for the label text
            cv2.rectangle(resizedImage, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED)
            # Draw the label text on the image
            cv2.putText(resizedImage, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            category = classes[class_id] #capitalize()
            print(f"  Score: {round(scores[i]*100,2)}% | Class: {category.capitalize()} | Bounding Box: {box}")
            find = 1
        # Return the modified input image
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #plt.title("Image with Bounding Boxes")
        
        if draw == True:
            plt.figure(figsize=(16,12))
            plt.grid(False)
            plt.imshow(resizedImage)
            plt.show()
            
        if find == 1:
            return resizedImage
        
    def drawYolo8BoxesShadowDeployment(self,inference, resizedImage, width, height, confidence_thres, iou_thres, draw=False):
        import ultralytics
        from ultralytics.utils import ROOT, yaml_load
        from ultralytics.utils.checks import check_requirements, check_yaml
        find = 0
        #pred = np.array(inference["out.output0"][0]).reshape(1, 84, 8400)
        pred = np.array(inference[0]).reshape(1, 84, 8400)
        #outputs = np.transpose(np.squeeze(inference[0]))
        outputs = np.transpose(np.squeeze(pred[0]))
        # Get the number of rows in the outputs array
        rows = outputs.shape[0]
        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []
        # Calculate the scaling factors for the bounding box coordinates
        x_factor = width / width
        y_factor = height / height
        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]
            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)
            #print(f"max score: {max_score*1000}")
            # If the maximum score is above the confidence threshold
            if max_score >= confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)
                #print(f"classid:{class_id}")
                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])
        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_thres, iou_thres)
        #print(f"Top items found from confidence level={confidence_thres}: {indices}")
        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            classes = yaml_load(check_yaml('coco128.yaml'))['names']
            # Generate a color palette for the classes
            color_palette = np.random.uniform(0, 255, size=(len(classes), 3))
            # Extract the coordinates of the bounding box
            x1, y1, w, h = box
            # Retrieve the color for the class ID
            color = color_palette[class_id]
            # Draw the bounding box on the image
            cv2.rectangle(resizedImage, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
            # Create the label text with class name and score
            label = f'{classes[class_id]}: {score:.2f}'
            # Calculate the dimensions of the label text
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # Calculate the position of the label text
            label_x = x1
            label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
            # Draw a filled rectangle as the background for the label text
            cv2.rectangle(resizedImage, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED)
            # Draw the label text on the image
            cv2.putText(resizedImage, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            category = classes[class_id] #capitalize()
            print(f"  Score: {round(scores[i]*100,2)}% | Class: {category.capitalize()} | Bounding Box: {box}")
            find = 1
        # Return the modified input image
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #plt.title("Image with Bounding Boxes")
        
        if draw == True:
            plt.figure(figsize=(16,12))
            plt.grid(False)
            plt.imshow(resizedImage)
            plt.show()
            
        if find == 1:
            print("Found Object")
            #return resizedImage
    
    
    def drawDetectedObjectsWithClassificationsFromInferenceYolo(self, results):
        total = 0
        colors = [(89, 161, 197),(67, 161, 255),(19, 222, 24),(186, 55, 2),(167, 146, 11),(190, 76, 98),(130, 172, 179),(115, 209, 128),(204, 79, 135),(136, 126, 185),(209, 213, 45),(44, 52, 10),(101, 158, 121),(179, 124, 12),(25, 33, 189),(45, 115, 11),(73, 197, 184),(62, 225, 221),(32, 46, 52),(20, 165, 16),(54, 15, 57),(12, 150, 9),(10, 46, 99),(94, 89, 46),(48, 37, 106),(42, 10, 96),(7, 164, 128),(98, 213, 120),(40, 5, 219),(54, 25, 150),(251, 74, 172),(0, 236, 196),(21, 104, 190),(226, 74, 232),(120, 67, 25),(191, 106, 197),(8, 15, 134),(21, 2, 1),(142, 63, 109),(133, 148, 146),(187, 77, 253),(155, 22, 122),(218, 130, 77),(164, 102, 79),(43, 152, 125),(185, 124, 151),(95, 159, 238),(128, 89, 85),(228, 6, 60),(6, 41, 210),(11, 1, 133),(30, 96, 58),(230, 136, 109),(126, 45, 174),(164, 63, 165),(32, 111, 29),(232, 40, 70),(55, 31, 198),(148, 211, 129),(10, 186, 211),(181, 201, 94),(55, 35, 92),(129, 140, 233),(70, 250, 116),(61, 209, 152),(216, 21, 138),(100, 0, 176),(3, 42, 70),(151, 13, 44),(216, 102, 88),(125, 216, 93),(171, 236, 47),(253, 127, 103),(205, 137, 244),(193, 137, 224),(36, 152, 214),(17, 50, 238),(154, 165, 67),(114, 129, 60),(119, 24, 48),(73, 8, 110)]
        # Get the number of rows in the outputs array
        from yolov5.utils.general import non_max_suppression
        infResults = results['inf-results']
        newShape = results['shape']
        pred = np.array(infResults["out.output0"][0]).reshape(newShape)
        pred = torch.tensor(pred).float().to("cpu")
        conf_thres = 0.25 
        iou_thres = 0.45 
        pred = non_max_suppression(pred, conf_thres, iou_thres)
        for i, det in enumerate(pred):
            if len(det):
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = xyxy
                    x1=int(x1.round().item())
                    x2=int(x2.round().item())
                    y1=int(y1.round().item())
                    y2=int(y2.round().item())
                    confidence = conf.item()
                    classification = int(cls.item())
                    colorLocation = colors[classification]
                    label = "{}: {:.2f}%".format(self.getCocoClasses()[classification], confidence * 100)
                    box = [x1,y1,x2,y2]
                    image = results['image']
                    series = results['series']
                    self.box_label(image, box, label, color=colorLocation, txt_color=(255,255,255))
                    classificationDisplay = "{}".format(self.getCocoClasses()[classification], "")
                    con = (round(confidence * 100))
                    print(f"Image #:{series} | Bounding Box: {x1, y1, x2, y2}, Confidence: {con}%, Classification: {classification} | ",classificationDisplay)
                    total+=1
        plt.figure(figsize=(16,12))
        plt.title("Wallaroo Output Image")
        plt.grid(False)
        plt.imshow(image)
        plt.show()
        print(f"Total: {total}")
    
    def drawDetectedObjectsWithClassificationsFromInferenceYoloLoop(self,results):
        dataFilename = results['dataFilename']    
        lines = "-----------------------------------------------------------------------------------------------------------------------------"
        colors = [(0, 0, 0),(67, 161, 255),(19, 222, 24),(186, 55, 2),(167, 146, 11),(190, 76, 98),(130, 172, 179),(115, 209, 128),(204, 79, 135),(136, 126, 185),(209, 213, 45),(44, 52, 10),(101, 158, 121),(179, 124, 12),(25, 33, 189),(45, 115, 11),(73, 197, 184),(62, 225, 221),(32, 46, 52),(20, 165, 16),(54, 15, 57),(12, 150, 9),(10, 46, 99),(94, 89, 46),(48, 37, 106),(42, 10, 96),(7, 164, 128),(98, 213, 120),(40, 5, 219),(54, 25, 150),(251, 74, 172),(0, 236, 196),(21, 104, 190),(226, 74, 232),(120, 67, 25),(191, 106, 197),(8, 15, 134),(21, 2, 1),(142, 63, 109),(133, 148, 146),(187, 77, 253),(155, 22, 122),(218, 130, 77),(164, 102, 79),(43, 152, 125),(185, 124, 151),(95, 159, 238),(128, 89, 85),(228, 6, 60),(6, 41, 210),(11, 1, 133),(30, 96, 58),(230, 136, 109),(126, 45, 174),(164, 63, 165),(32, 111, 29),(232, 40, 70),(55, 31, 198),(148, 211, 129),(10, 186, 211),(181, 201, 94),(55, 35, 92),(129, 140, 233),(70, 250, 116),(61, 209, 152),(216, 21, 138),(100, 0, 176),(3, 42, 70),(151, 13, 44),(216, 102, 88),(125, 216, 93),(171, 236, 47),(253, 127, 103),(205, 137, 244),(193, 137, 224),(36, 152, 214),(17, 50, 238),(154, 165, 67),(114, 129, 60),(119, 24, 48),(73, 8, 110)]
        newShape = results['shape']
        image = results['image']
        milsec = results['inference-time']
        if results['write-to-file'] == True:
            with open(dataFilename,'a') as f:
                f.write(lines)
                f.write("Filename: %s   Data Shape: %s    Inference Time Elapsed: %s Milliseconds \n" % (results['filename'], results['datashape'],str(round(milsec, 2))))
        print("Filename: ", results['filename'], "     Data Shape: ", results['datashape'],"    Inference Time Elapsed: ", str(round(milsec, 2)), " Milliseconds")

        predictionResults= results['inf-results']["out.output0"][0]
        pred = np.array(predictionResults).reshape(newShape)
        pred = torch.tensor(pred).float().to("cpu")
        conf_thres = 0.25 
        iou_thres = 0.45 
        pred = non_max_suppression(pred, conf_thres, iou_thres) 
        for i, det in enumerate(pred):
            if len(det):
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = xyxy
                    x1=int(x1.round().item())
                    x2=int(x2.round().item())
                    y1=int(y1.round().item())
                    y2=int(y2.round().item())
                    confidence = conf.item()
                    classification = int(cls.item())
                    colorLocation = colors[classification]
                    classificationDisplay = "{}".format(self.getCocoClasses()[classification], "")
                    label = "{}: {:.2f}%".format(self.getCocoClasses()[classification], confidence * 100)
                    box = [x1,y1,x2,y2]
                    self.box_label(image, box, label, color=colorLocation, txt_color=(255,255,255))
                    
                    con = (round(confidence * 100),3)
                    print(f"Bounding Box: {x1, y1, x2, y2}, Confidence: {con}, Classification: {classification} | ",classificationDisplay)
                    if results['write-to-file'] == True:
                        with open(dataFilename,'a') as f:
                            f.write("   Bounding Box: %s, %s, %s, %s | Confidence: %s | Classification: %s \n" % (x1, y1, x2, y2, con, classificationDisplay))
        
        plt.figure(figsize=(16,12))
        if results['write-to-file'] == True:
            cv2.imwrite(results['outbox']+"/i-"+ 'boxes-' + results['filename'], image)
        plt.close('all')
        return image
    
    
    def inferUsingOnnx(self, image, config):
        # load the image we want to categorize using the onnx model
        #tensor, resizedImage = self.loadImageAndResize(imagePath,width,height)
        infResult = {}

        tensor = self.convertFrameToTensor(image)
        self.saveInputToFile("onnx-input.json",tensor)
        self.debug(config)
        
        
        #input_shapes = [[d.dim_value for d in _input.type.tensor_type.shape.dim] for _input in model.graph.input]
        #self.print("input_shapes")
        #self.print(input_shapes)

        self.debug("Onnx Input Shape")
        self.debug(tensor.shape)
        
        #TODO we only need to do this once.  Fix this
        #onnx_session= ort.InferenceSession(config['onnx_model_path'])
        #model = onnx.load(config['onnx_model_path'])
        #onnx.checker.check_model(model)

        startTime = time.time()
        onnx_session = config['onnx-session']
        onnx_inputs= {onnx_session.get_inputs()[0].name: tensor}
        onnx_output = onnx_session.run(None, onnx_inputs)
        endTime = time.time()
        infResult['onnx-time'] = (endTime - startTime)

        self.debug("Onnx Inference Result")
        self.debug(onnx_output)
        
        out_y = onnx_output[0]

        infResult['boxes'] = onnx_output[0]
        infResult['classes'] = onnx_output[1]
        infResult['confidences'] = onnx_output[2]
        
        return infResult
     
    def extractAnomaliesFromInference(self,infResultDict, walInfResult):
        results = walInfResult[0].raw
        anomalyClasses = results['outputs'][0]['Json']['data'][0]['anomaly-classes']
        anomalyConfidences = results['outputs'][0]['Json']['data'][0]['anomaly-confidences']
        anomalyBoxes = results['outputs'][0]['Json']['data'][0]['anomaly-boxes']
             
        infResultDict['anomalyClasses'] = anomalyClasses
        infResultDict['anomalyConfidences'] = anomalyConfidences
        infResultDict['anomalyBoxes'] = anomalyBoxes


    def convertWallarooResultToInferenceResultDict(self,walInfResult):
        self.debug("Wallaroo InferenceResult data = ")
        data = walInfResult[0].data()
        self.debug(data)
        
        infResultDict = {}
        infResultDict['model_name'] = walInfResult[0].model()[0]
      
        #infResultDict['pipeline_name'] = walInfResult[0].model()

        infResultDict['boxes'] = data[0]
        
        #infResultDict['classes'] = data[1].astype(int)
        infResultDict['classes'] = data[1]

        infResultDict['confidences'] = data[2]
        infResultDict['onnx-time'] =  int(walInfResult[0].time_elapsed().microseconds) / 1e+6 # convert to seconds

        results = walInfResult[0].raw
        #shadow_results['original_data'] = None
        
        
        if 'shadow_data' in results:
            self.debug('Found shadow data')
            shadowInf = {}
            self.debug(results['shadow_data'])
            infResultDict['shadow_data'] = {}
            for model_name in results['shadow_data']:
                shadowInfResult = results['shadow_data'][model_name]
                
                self.debug(shadowInfResult)
                shadowInf = { 'boxes' : shadowInfResult[0]['Float']['data'],
                              'classes' : shadowInfResult[1]['Int64']['data'],
                              'confidences' : shadowInfResult[2]['Float']['data'] }
                infResultDict['shadow_data'][model_name] = shadowInf


        else:
             self.debug('no shadow_data')
           
        #
        #if ('= None  # We are removing the input image json.  Not needed

        #self.debug(infResultDict)
        return infResultDict
    
   
    #
    # Uses the wallaroo SDK to run inference.  Returns a dictionary with the inference result
    #
    def runInferenceOnFrameUsingSdk(self, frame, config):
        dictTensor = self.convertFrameToTensorDict(frame, config['width'], config['height'])
        # Run inference by calling the wallaroo url rest api endpoint feeding it the jsonInput
        inferResult = None
        #try:
        self.debug("call runInferenceOnFrameUsingSdk ")
        startTime = time.time()
        #jsonData = json.dumps(dictTensor)
        #with open("dictTensor.json", "w") as outfile:
        #    outfile.write(jsonData)
        walInferResult = config['pipeline'].infer(dictTensor)
        inferResult = self.convertWallarooResultToInferenceResultDict(walInferResult)
        if ('extract-anomalies' in config) and config['extract-anomalies'] == True:
            self.extractAnomaliesFromInference(inferResult,walInferResult)

        endTime = time.time()
        inferResult['inference-time'] = (endTime - startTime)

        return inferResult
            
    #
    # runs inference on the frame and returns a dictionary with the result
    # if config['inference'] = 'ONNX' it will use the ONNX runtime locally to run inference
    # if config['inference'] = 'WALLAROO_SDK' or 'WALLAROO_API' it will use the pipeline to run inference on Wallaroo
    #
    def runInferenceOnFrame(self, frame, config):
        
        dictTensor = self.convertFrameToTensorDict(frame, config['width'], config['height'])
        # Run inference by calling the wallaroo url rest api endpoint feeding it the jsonInput
        inferResult = None
        json = None
        #try:
        self.debug("Running wallaroo pipeline inference")
        startTime = time.time()
        #jsonData = json.dumps(dictTensor)
        #with open("dictTensor.json", "w") as outfile:
        #    outfile.write(jsonData)
        if ('skip-frames-list' in config):
            for tplRange in config['skip-frames-list']:
                if (config['frame-cnt'] >= tplRange[0] and
                   config['frame-cnt'] <= tplRange[1]):
                          self.print("Frame in skip range. [" + str(config['frame-cnt']) + "]  skipping")
                          return None
            

        if (config['inference'] == "ONNX"):
            infResult = self.inferUsingOnnx(frame,config)
        elif (config['inference'] == "WALLAROO_API"):
            #pipeline = config['pipeline']
            #inferResult = pipeline.infer(dictTensor)
            #self.print(inferResult[0].data())
            infResult = self.runInferenceOnFrameUsingApi(frame, config['endpoint-url'],  config['width'], config['height'])
        elif (config['inference'] == "WALLAROO_SDK"):
             infResult = self.runInferenceOnFrameUsingSdk(frame, config)
        endTime = time.time()

        infResult['pipeline_name'] = config['pipeline_name']
        infResult['inference-time'] = (endTime - startTime)
        #except Exception as e:
        #    self.print("Could not inference frame")
        #    print(e)
        #    infResult = None
        return infResult
        
  
        
       
    #
    # traverses the control and list of challengers to determine
    # which model had the best overeall average confidence score
    #
    def selectModelWithBestAverageConfidenceOld(self,jsonResult, minConf):
        bestConfidences = jsonResult[0]['outputs'][2]['Float']['data']
        #self.print("bestConfidences"+str(bestConfidences))
        npBestConf = np.array(bestConfidences)
        confFilter = npBestConf > minConf
        bestAvgConfidence = np.mean(npBestConf[confFilter])

        bestModel = jsonResult[0]['model_name']
        for challenger in jsonResult[0]['shadow_data']:
            #self.print("challenger"+str(jsonResult[0]['shadow_data'][challenger]))
            challengerConfidences = jsonResult[0]['shadow_data'][challenger][2]['Float']['data']
            
            npBestConf = np.array(challengerConfidences)
            confFilter = npBestConf > minConf
    
            chalAvgConf = np.mean(npBestConf[confFilter])
            #self.print(challenger+" avg conf:"+str(chalAvgConf))
            if (bestAvgConfidence < chalAvgConf):
                bestModel = challenger
            
        return bestModel
    
     #
    # traverses the control and list of challengers to determine
    # which model had the best overeall average confidence score
    #
    def selectModelWithBestAverageConfidence(self,infResult, minConf):
        #print("infResult")
        #print(infResult)
        bestConfidences = infResult['confidences']
        #self.print("bestConfidences"+str(bestConfidences))
        npBestConf = np.array(bestConfidences)
        confFilter = npBestConf > minConf
        bestAvgConfidence = np.mean(npBestConf[confFilter])
        self.debug("control avg conf:"+str(bestAvgConfidence))

        bestModel = infResult['model_name']
        for challenger in infResult['shadow_data']:
            self.debug("challenger:"+str(challenger))
            self.debug("challenger:"+str(infResult['shadow_data'][challenger]['confidences']))
            #self.print("challenger"+str(jsonResult[0]['shadow_data'][challenger]))
            #challengerConfidences = infResult['shadow_data'][challenger][2]['Float']['data']
            challengerConfidences = infResult['shadow_data'][challenger]['confidences']

            npBestConf = np.array(challengerConfidences)
            confFilter = npBestConf > minConf
    
            chalAvgConf = np.mean(npBestConf[confFilter])
            self.debug(challenger+" avg conf:"+str(chalAvgConf))
            if (bestAvgConfidence < chalAvgConf):
                bestModel = challenger
            
        return bestModel
    
    def buildConfigFromPipelineInfernece(self, jsonResult):
        config = {}
        model_name = jsonResult[0]['model_name']
        pipeline_name = jsonResult[0]['pipeline_name']

        boxes = jsonResult[0]['outputs'][0]['Float']['data']
        classes = jsonResult[0]['outputs'][1]['Int64']['data']
        confidences = jsonResult[0]['outputs'][2]['Float']['data']

        #self.print("Control Inference Results: "+str(end_time-start_time))
        #self.print("boxes")
        #self.print(boxes)
        # reshape this to an array of bounding box coordinates converted to ints
        boxList = boxes
        boxA = np.array(boxList)
        boxes = boxA.reshape(-1, 4)
        boxes = boxes.astype(int)

        config = {
            'model_name' : 'TBD',
            #'image' : controlImage,
            'boxes' : boxes,
            'classes' : classes,
            'confidences' : confidences,
            'onnx-time' : jsonResult[0]['elapsed'],
            'model_name' : jsonResult[0]['model_name'],
            'pipeline_name': jsonResult[0]['pipeline_name']
        }
        
        return config
    
    def buildConfigFromInferenceResult(self, infResult):
        config = {}
        model_name = infResult['model_name']
        pipeline_name = infResult['pipeline_name']

        boxes = infResult['boxes']
        classes = json.jsonResult[0]['outputs'][1]['Int64']['data']
        confidences = json.jsonResult[0]['outputs'][2]['Float']['data']

        #self.print("Control Inference Results: "+str(end_time-start_time))
        #self.print("boxes")
        #self.print(boxes)
        # reshape this to an array of bounding box coordinates converted to ints
        boxList = boxes
        boxA = np.array(boxList)
        boxes = boxA.reshape(-1, 4)
        boxes = boxes.astype(int)
        model_name
        config = {
            'model_name' : 'TBD',
            #'image' : controlImage,
            'boxes' : boxes,
            'classes' : classes,
            'confidences' : confidences,
            'onnx-time' : json.jsonResult[0]['elapsed'],
            'model_name' : json.jsonResult[0]['model_name'],
            'pipeline_name': json.jsonResult[0]['pipeline_name']
        }
        
        return config
    
    def buildConfigsFromShadowDeplInferneces(self, infResult):
        self.debug("calling buildConfigsFromShadowDeplInferneces")
        config = {}
        configList = []
       
        boxes = infResult['boxes']
        self.debug("boxes")
        self.debug(boxes)
        classes = infResult['classes']
        confidences = infResult['confidences']

        # reshape this to an array of bounding box coordinates converted to ints
        boxList = boxes
        boxA = np.array(boxList)
        boxes = boxA.reshape(-1, 4)
        boxes = boxes.astype(int)
        config = {
            'model_name' : infResult['model_name'],
            'pipeline_name' : infResult['pipeline_name'],
            #'image' : controlImage,
            'boxes' : boxes,
            'classes' : classes,
            'confidences' : confidences,
            'confidence-target' : 0.75,
            'onnx-time': infResult['onnx-time']
        }
        configList.append(config)
        for key in infResult['shadow_data']:
            challenger = infResult['shadow_data'][key]
            modelName = key
   
            challengerBoxes = challenger['boxes']
            challengerClasses = challenger['classes']
            challengerConfidences = challenger['confidences']
            
            boxList = challengerBoxes
            boxA = np.array(boxList)
            challengerBoxes = boxA.reshape(-1, 4)
            challengerBoxes = challengerBoxes.astype(int)

            config = {
                'model_name' : modelName,
                'pipeline_name' : infResult['pipeline_name'],

                #'image' : controlImage,
                'boxes' : challengerBoxes,
                'classes' : challengerClasses,
                'confidences' : challengerConfidences,
                'confidence-target' : 0.75,
                'onnx-time': infResult['onnx-time']
            }
            configList.append(config)
        
        return configList

    
    #
    #
    #
    def addMessageToDashboard(self, config, dashboardFrame):
        fontThickness = 1
        fontScale = 1.30
        fontColor = config['color']
        msgList = config['dashboard-message-list']
        
        (frameHeight, frameWidth, frameDepth) = dashboardFrame.shape
        
        rows = len(msgList)
        # Calculate Center Positioning
        (msgWidth, msgHeight), baseline = cv2.getTextSize(msgList[0], cv2.FONT_HERSHEY_PLAIN, fontScale, fontThickness)
       
        msgCenterX = int((frameWidth - msgWidth) / 2)
        msgY = int( (frameHeight/2) - ((msgHeight * rows) / 2) )
        
        msgY += msgHeight
        
        self.debug("msgY:"+str(msgY))
        self.debug("msgCenterX:"+str(msgCenterX))
        
        rowPadding = 8
        for msg in msgList:
            (msgWidth, msgHeight), baseline = cv2.getTextSize(msg, cv2.FONT_HERSHEY_PLAIN, fontScale, fontThickness)
            msgCenterX = int((frameWidth - msgWidth) / 2)

            cv2.putText(dashboardFrame, msg, (msgCenterX, msgY), cv2.FONT_HERSHEY_PLAIN, fontScale, fontColor, fontThickness, lineType = cv2.LINE_AA)
            msgY += msgHeight + rowPadding
                   
            
        
    #
    # Creates and returns an image with the title specified and the provided config['width']
    #
    def addTitleToDashboard(self, title, config, dashboardFrame):
        fontThickness = 1
        fontScale = 1.30
        #fontColor = config['color']
        fontColor = CVDemo.WHITE

        # create the row image
        rowHeight=20
        rows = 2
        statsHeight = rowHeight * rows
                
        # Calculate Center Positioning
        #titleSize = cv2.getTextSize(title, cv2.FONT_HERSHEY_PLAIN, 1, 2)[0]
        titleSize = cv2.getTextSize(title, cv2.FONT_HERSHEY_PLAIN, fontScale, fontThickness)[0]

        titleCenterX = int((config['width'] - titleSize[0]) / 2)
        
        y = statsHeight
        cv2.putText(dashboardFrame, title, (titleCenterX, y), cv2.FONT_HERSHEY_PLAIN, fontScale, fontColor, fontThickness, lineType = cv2.LINE_AA)
    
    def drawStatsDashboard(self, title, results):
        statsRowHeight = 25
        rows = 2
        
        if ('anomaly-count' in results):
            rows += 1
        
        statsHeight = statsRowHeight*rows
        statsImage = np.zeros([statsHeight,results['width'],3],dtype=np.uint8)
        
        results['color'] = (255,255,255)
        #self.drawStatsDashboardTitle(title, results)
        statsImage = self.drawStats("Wallaroo Computer Vision Statistics Dashboard",statsImage, results, 1)
        return statsImage

    #
    # Draw the list of columns into the dashboardImage
    #
    def addColumnTitlesToToDashboard(self, columnList, dashboardImage):
        rowHeight = 20
        y = 3 * rowHeight
                    
        colTitlesRow = ""
        for colName in columnList:
            colTitlesRow += colName
            
        fontThickness = 1
        fontScale = 1
        fontColor = CVDemo.WHITE

        cv2.putText(dashboardImage, colTitlesRow, (0, y), cv2.FONT_HERSHEY_PLAIN, fontScale, fontColor, fontThickness, lineType = cv2.LINE_AA)


    def calcColumnXCoord(self, value, colWidth, align, fontScale, fontThickness):
        (valueWidth, valueHeight), baseline = cv2.getTextSize(value, cv2.FONT_HERSHEY_PLAIN, fontScale, fontThickness)
        paddedValue = ""
        xPos = 0
        if (align == "center"):
            diff = colWidth - valueWidth
            if (diff > 0):
                xPos = int((colWidth - valueWidth) / 2)
            else:
                xPos = 0
        return xPos
            
            
    #
    # Adds a message below the statistics table in the Message Board
    #
    def addNotesToDashboard(self, dashboardFrame, frameCnt, config):
        if ('note-list' in config):
            notesList = config['note-list']
            for note in notesList:
                if (frameCnt >= note['start-frame'] and \
                    frameCnt < note['end-frame']):
                    self.displayNoteInDashboard(note, dashboardFrame, config)

    
    def displayNoteInDashboard(self,note, dashboardFrame, config):
        row = 6
        rowHeight=25
        y = row * rowHeight - 5
        height, width, depth = dashboardFrame.shape
        cv2.rectangle(dashboardFrame, (0, y-5), (width, y-rowHeight), CVDemo.BLACK, -1)

        fontThickness = 1
        fontScale = 1.1
        if 'color' in note:
            fontColor = note['color']
        else:
            fontColor = CVDemo.WHITE
        
        msgWidth = cv2.getTextSize(note['note'], cv2.FONT_HERSHEY_PLAIN, fontScale, fontThickness)[0]
        msgCenterX = int((width - msgWidth[0]) / 2)
        
        cv2.putText(dashboardFrame, note['note'], (msgCenterX, y-10), cv2.FONT_HERSHEY_PLAIN, fontScale, fontColor, fontThickness, lineType = cv2.LINE_AA)
        
            
    #
    # Creates a image row containing the inference results and appends it to the dashboardImage
    # TODO Refactor method to make it more generic
    #
    def addInferenceResultsToDashboard(self, results, columns, row, dashboardImage):
        avgScore = 0
        
        confidences = results['confidences']
        classes = results['classes']
        classCnt = len(classes)
        avgScore = 0.0
        if (len(confidences) > 0):
            if 'confidence-target' in results:
                target = results['confidence-target']
                self.debug("target:"+str(target))

                arrayConfidences = np.array(confidences)
                arrayConfidences = arrayConfidences[arrayConfidences > target]
                avgScore = np.mean(arrayConfidences)*100
            else:
                avgScore = np.mean(confidences)*100

        boxes = results['boxes']
        modelName = results['model_name']
        pipelineName = results['pipeline_name']
        infTime = results['inference-time']
        onnxTime = results['onnx-time']
        #onnxTime =  int(results['onnx-time']) / 1e+9
            
        self.debug("infTime="+str(infTime))
        self.debug("onnxTime="+str(onnxTime))
        rowHeight=20
        y = row * rowHeight + 5
            
        classCnt = len(set(classes))
    
        # clear the row
        height, width, depth = dashboardImage.shape
        self.debug("dashboard shape:"+str(dashboardImage.shape))

        cv2.rectangle(dashboardImage, (0, y), (width, y-rowHeight), CVDemo.BLACK, -1)
        self.debug("row rect width="+str(width))
        self.debug("y="+str(y))
        self.debug("rowHeight="+str(rowHeight))

        # Build the table rows
        #
       
        fontThickness = 1
        fontScale = 1
        if 'color' in results:
            fontColor = results['color']
        else:
            fontColor = CVDemo.WHITE

        #
        # display cells
        #
        x = 0
        (colWidth, colHeight), baseline = cv2.getTextSize(columns[0], cv2.FONT_HERSHEY_PLAIN, fontScale, fontThickness)
        #self.print("colWidth:"+str(colWidth))
        alignOffset = self.calcColumnXCoord(modelName, colWidth, CVDemo.ALIGN_CENTER, fontScale, fontThickness) # Model
        xColPos = x + alignOffset
        # draw the cell text
        cv2.putText(dashboardImage, modelName, (xColPos, y), cv2.FONT_HERSHEY_PLAIN, fontScale, fontColor, fontThickness, lineType = cv2.LINE_AA)
        x += colWidth
        #self.debug("xColPos:"+str(xColPos))
        #self.debug("y:"+str(y))

        (colWidth, colHeight), baseline = cv2.getTextSize(columns[1], cv2.FONT_HERSHEY_PLAIN, fontScale, fontThickness)
        alignOffset = self.calcColumnXCoord(pipelineName, colWidth, CVDemo.ALIGN_CENTER, fontScale, fontThickness) # Model
        xColPos = x + alignOffset
        # draw the cell text
        cv2.putText(dashboardImage, pipelineName, (xColPos, y), cv2.FONT_HERSHEY_PLAIN, fontScale, fontColor, fontThickness, lineType = cv2.LINE_AA)
        x += colWidth
    
        value = "{:.2f}".format(infTime) + "/{:.2f}".format(float(onnxTime))
        (colWidth, colHeight), baseline = cv2.getTextSize(columns[2], cv2.FONT_HERSHEY_PLAIN, fontScale, fontThickness)
        alignOffset = self.calcColumnXCoord(value, colWidth, CVDemo.ALIGN_CENTER, fontScale, fontThickness) # Model
        xColPos = x + alignOffset
        # draw the cell text
        cv2.putText(dashboardImage, value, (xColPos, y), cv2.FONT_HERSHEY_PLAIN, fontScale, fontColor, fontThickness, lineType = cv2.LINE_AA)
        x += colWidth

        value = str(len(boxes))
        (colWidth, colHeight), baseline = cv2.getTextSize(columns[3], cv2.FONT_HERSHEY_PLAIN, fontScale, fontThickness)
        alignOffset = self.calcColumnXCoord(value, colWidth, CVDemo.ALIGN_CENTER, fontScale, fontThickness) # Model
        xColPos = x + alignOffset
        # draw the cell text
        cv2.putText(dashboardImage, value, (xColPos, y), cv2.FONT_HERSHEY_PLAIN, fontScale, fontColor, fontThickness, lineType = cv2.LINE_AA)
        x += colWidth
 
        value = str(classCnt)
        (colWidth, colHeight), baseline = cv2.getTextSize(columns[4], cv2.FONT_HERSHEY_PLAIN, fontScale, fontThickness)
        alignOffset = self.calcColumnXCoord(value, colWidth, CVDemo.ALIGN_CENTER, fontScale, fontThickness) # Model
        xColPos = x + alignOffset
        # draw the cell text
        cv2.putText(dashboardImage, value, (xColPos, y), cv2.FONT_HERSHEY_PLAIN, fontScale, fontColor, fontThickness, lineType = cv2.LINE_AA)
        x += colWidth
        
        value = "{:.2f}%".format(avgScore)
        (colWidth, colHeight), baseline = cv2.getTextSize(columns[5], cv2.FONT_HERSHEY_PLAIN, fontScale, fontThickness)
        alignOffset = self.calcColumnXCoord(value, colWidth, CVDemo.ALIGN_CENTER, fontScale, fontThickness) # Model
        xColPos = x + alignOffset
        # draw the cell text
        cv2.putText(dashboardImage, value, (xColPos, y), cv2.FONT_HERSHEY_PLAIN, fontScale, fontColor, fontThickness, lineType = cv2.LINE_AA)
        x += colWidth
   
        # Need to clean this up. Re
        if 'wins' in results:
            value = str(results['wins'])
        else:
            value = "-"
        
        (colWidth, colHeight), baseline = cv2.getTextSize(columns[6], cv2.FONT_HERSHEY_PLAIN, fontScale, fontThickness)
        alignOffset = self.calcColumnXCoord(value, colWidth, CVDemo.ALIGN_CENTER, fontScale, fontThickness) # Model
        xColPos = x + alignOffset
               
        # draw the cell text - TODO Hack fix this
        if 'anomalyConfidences' in results:
            value = str(len(results['anomalyConfidences']))
            cv2.putText(dashboardImage, value, (xColPos, y), cv2.FONT_HERSHEY_PLAIN, fontScale, CVDemo.RED, fontThickness, lineType = cv2.LINE_AA)
        else:
            cv2.putText(dashboardImage, value, (xColPos, y), cv2.FONT_HERSHEY_PLAIN, fontScale, fontColor, fontThickness, lineType = cv2.LINE_AA)
        x += colWidth

        value = "-"
        (colWidth, colHeight), baseline = cv2.getTextSize(columns[7], cv2.FONT_HERSHEY_PLAIN, fontScale, fontThickness)
        alignOffset = self.calcColumnXCoord(value, colWidth, CVDemo.ALIGN_CENTER, fontScale, fontThickness) # Model
        xColPos = x + alignOffset
        # draw the cell text
        cv2.putText(dashboardImage, value, (xColPos, y), cv2.FONT_HERSHEY_PLAIN, fontScale, fontColor, fontThickness, lineType = cv2.LINE_AA)
        x += colWidth
        
        value = "-"
        (colWidth, colHeight), baseline = cv2.getTextSize(columns[8], cv2.FONT_HERSHEY_PLAIN, fontScale, fontThickness)
        alignOffset = self.calcColumnXCoord(value, colWidth, CVDemo.ALIGN_CENTER, fontScale, fontThickness) # Model
        xColPos = x + alignOffset
        # draw the cell text
        cv2.putText(dashboardImage, value, (xColPos, y), cv2.FONT_HERSHEY_PLAIN, fontScale, fontColor, fontThickness, lineType = cv2.LINE_AA)
        x += colWidth
    
    def drawStats(self,title, image, config, row):
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
        #cv2.rectangle(image,(0,y),(640,rowHeight*2),(255, 255, 255), -1)
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
            fontColor = CVDemo.AMBER
            msg = "Anomalies: "+str(config['anomaly-count'])
            cv2.putText(image, msg, (5, row), cv2.FONT_HERSHEY_PLAIN, fontScale, fontColor, fontThickness, lineType = cv2.LINE_AA)
        return image
        
    def drawShadowDeplStats(self, jsonResult, image):
        controlConfig = buildConfigFromShadowDeplInferneces(jsonResult,-1)
        self.drawStats("Wallaroo Shadow Deployment Statistics Dashboard", image,controlConfig,1)
        
        bestModelIdx = -1
        idx = 2
        for challenger in jsonResult[0]['shadow_data']:
            challengerConfig = buildConfigFromShadowDeplInferneces(challenger,0)
            self.drawStats("Wallaroo Shadow Deployment Statistics Dashboard", image,challengerConfig,idx)
            idx += 1
            
        return image
    
    def count_frames_manual(self,video):
        # initialize the total number of frames read
        total = 0
        # loop over the frames of the video
        while True:
            # grab the current frame
            (grabbed, frame) = video.read()

            # check to see if we have reached the end of the
            # video
            if not grabbed:
                break
            # increment the total number of frames read
            total += 1
        # return the total number of frames in the video file
        return total
        
    # Reads through each frame in the inVideo,
    # Resizes te frame for the mdoel
    # Runs inference
    # Draws inference results on a copy of the frame
    # Writes frame out to video in outVideoPath
    def detectAndClassifyObjectsInVideo(self, config):
        running = True
        newYorkTz = pytz.timezone("America/New_York")
        localTime = datetime.now(newYorkTz)
        localTime = localTime.strftime("%I:%M:%S %p")
        self.print("Start Time:"+localTime)

        inVideoPath = config['input-video']
        outVideoPath = config['output-video']
        fps = config['fps']
        width = config['width']
        height = config['height']
        pipelineEndPointUrl = config['endpoint-url']

        maxFrameCnt = 0
        maxSkipCnt = 0
        maxFrameCnt = 0
        if 'max-frames' in config:
            maxFrameCnt = config['max-frames']
        maxSkipCnt = 0
        if 'skip-frames' in config:
            maxSkipCnt = config['skip-frames']
        
        
        if (maxSkipCnt > 0):
            self.print("Skipping [" + str(maxSkipCnt) +"] frames")
        if (maxFrameCnt > 0):
            self.print("Capturing up to [" + str(maxFrameCnt) +"] frames")
            
        # if we are inferencing locally with ONNX, load the onnx session with the model and put it in the config for later inferencing
        if config['inference'] == 'ONNX':
            onnx_session= ort.InferenceSession(config['onnx_model_path'])
            model = onnx.load(config['onnx_model_path'])
            onnx.checker.check_model(model)
            config['onnx-session'] = onnx_session
            
        cap = cv2.VideoCapture(inVideoPath)
        self.print("Video Properties")
        self.print("   video input:"+inVideoPath)
        self.print("   video output:"+outVideoPath)

        self.print("   format:"+str(cap.get(cv2.CAP_PROP_FORMAT)))
        self.print("   fourcc:"+str(cap.get(cv2.CAP_PROP_FOURCC)))
        self.print("   mode:"+str(cap.get(cv2.CAP_PROP_MODE)))

        self.print("   buffer:"+str(cap.get(cv2.CAP_PROP_BUFFERSIZE)))
        self.print("   width:"+str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.print("   height:"+str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.print("   fps:"+str(cap.get(cv2.CAP_PROP_FPS)))
        #self.print("   frame count:"+str(self.count_frames_manual(cv2.VideoCapture(inVideoPath))))
   
        frameStats = "Frame"
        rowHeight = 25
        rows = 6
        #statsHeight = statsRowHeight*rows
        #statsFrame = np.zeros([statsHeight,width,3],dtype=np.uint8)
        #frame_size = (width,height+statsHeight)
        
        dashboardHeight = rowHeight * rows
        dashboardFrame = np.zeros([dashboardHeight, width, 3],dtype=np.uint8)
        frameSize = ( width, height + dashboardHeight)
        self.print("   frame size:"+str(frameSize))

        output = cv2.VideoWriter(outVideoPath, cv2.VideoWriter_fourcc(*'mp4v'), fps, frameSize)
        
        self.addTitleToDashboard( "Wallaroo Computer Vision Statistics Dashboard", config, dashboardFrame)

        columns = [
            '    Model   ',
            '   Pipeline   ',
            '      Inf     ',
            ' Obj ',
            ' Cls',
            '   Conf  ',
            ' Anom',
            ' Drift',
            '  PSI ']
        self.addColumnTitlesToToDashboard(columns, dashboardFrame)
        
        recordStartFrame = 0
        if 'record-start-frame' in config:
            recordStartFrame = config['record-start-frame']
        recordEndFrame = 0
        if 'record-end-frame' in config:
            recordEndFrame = config['record-end-frame']
        self.print("recordStartFrame:"+str(recordStartFrame))
        self.print("recordEndFrame:"+str(recordEndFrame))

        frameCnt = 1
        config['frame-cnt']=frameCnt
        skipCnt = 0
        row = 1
        try:
            while cap.isOpened():
                totalStartTime = startTime = time.time()
                ret, frame = cap.read()
                endTime = time.time()
                skipCnt += 1
                if (skipCnt < maxSkipCnt):
                    #self.debug("skipping frame:"+str(frameCnt))
                    frameCnt += 1
                    continue
                if (ret == True):
                    frameStats += ":"+str(frameCnt) +" Read: {:.4f}".format(endTime-startTime)
                   
                    # resize frame for width and height the objecet detector is expecting
                    frame = cv2.resize(frame, (width, height))

                    # run inference on the frame using width and height object detector is expecting
                    try:
                        infResult = self.runInferenceOnFrame(frame, config)
                    except Exception as e:
                        self.print(e)
                        self.print("could not read frame")
                        self.print("exiting at frame:"+str(frameCnt))
                        raise e
                        break
                        
                    if (infResult == None):
                        self.print("Could not inference frame:"+str(frameCnt))
                        self.print("Pressing on.  Try reading next frame")
                        frameCnt += 1
                        config['frame-cnt']=frameCnt
                        continue
                    
                    #infConfig = self.buildConfigFromPipelineInfernece(json)
                    infResult['image'] = frame
                    infResult['model_name'] = config['model_name']
                    infResult['pipeline_name'] = config['pipeline_name']

                    infResult['confidence-target'] = config['confidence-target']
                    infResult['color'] = config['color']
                    
                    frameStats += " Inf: {:.4f}".format(infResult['inference-time'])
                    
                    #This formula is elapsed wallaroo time
                    #onnxTime =  int(infResult['onnx-time']) / 1e+6
                    frameStats += " Onnx: {:.4f}".format(infResult['onnx-time'])
                                        
                    # Drawing the inference results and stats
                    startTime = time.time()
                    detObjFrame = self.drawDetectedObjectsWithClassification(infResult)
                    
                    self.addInferenceResultsToDashboard(infResult, columns, row+3, dashboardFrame)
                    self.addNotesToDashboard(dashboardFrame, frameCnt, config)
                    image = cv2.vconcat([dashboardFrame,detObjFrame])
                    
                    if recordStartFrame > 0:
                        if frameCnt > recordStartFrame:
                            if frameCnt < recordEndFrame:
                                self.debug("recording image:"+str(frameCnt))
                                output.write(image)
                            else:
                                break #exit
                        else:
                            self.debug("skipping frame:"+str(frameCnt))
                    else:
                        self.debug("writing image:"+str(frameCnt))
                        output.write(image)
                        
                    endTime = time.time()
                    frameStats += " Draw: {:.4f}".format(endTime-startTime)
                    frameStats += " Total: {:.4f}".format(endTime-totalStartTime)

                    self.print(frameStats)
                    frameStats = "Frame"
                else:
                    self.print("frame reading error:"+str(frameCnt))

                # option to exit early
                if (maxFrameCnt > 0 and frameCnt > maxFrameCnt):
                    self.print("Exiting early frameCnt > maxFrameCnt")
                    break
                    
                frameCnt += 1
                config['frame-cnt']=frameCnt

        except KeyboardInterrupt:
            running = False
            self.print("Exiting")
            
        cap.release()
        output.release()
        self.print("Finished writing video:"+outVideoPath)

        localTime = datetime.now(newYorkTz)
        localTime = localTime.strftime("%H:%M:%S")
        self.print("End Time:"+localTime)
        return running
    
    # Reads through each frame in the inVideo,
    # Resizes te frame for the mdoel
    # Runs inference
    # Draws inference results on a copy of the frame
    # Writes frame out to video in outVideoPath
    def detectAndClassifyObjectsInVideoUsingShadowDeployment(self, config):
        running = True
        newYorkTz = pytz.timezone("America/New_York")
        localTime = datetime.now(newYorkTz)
        localTime = localTime.strftime("%I:%M:%S %p")
        self.print("Start Time:"+localTime)

        inVideoPath = config['input-video']
        outVideoPath = config['output-video']
        fps = config['fps']
        width = config['width']
        height = config['height']
        pipeline = config['pipeline']
        config['pipeline_name'] = pipeline.name()
        pipelineEndPointUrl = pipeline._deployment._url()
        controlModel = config['control-model']
        challengerModelList = config['challenger-model-list']

        maxFrameCnt = 0
        maxSkipCnt = 0
        if 'max-frame' in config:
            maxFrameCnt = config['max-frames']
        if 'skip-frames' in config:
            maxSkipCnt = config['skip-frames']
        
        if (maxSkipCnt > 0):
            self.print("Skipping [" + str(maxSkipCnt) +"] frames")
        if (maxFrameCnt > 0):
            self.print("Captureing up to [" + str(maxFrameCnt) +"] frames")
            
        cap = cv2.VideoCapture(inVideoPath)
        self.print("Video Properties")
        self.print("   format:"+str(cap.get(cv2.CAP_PROP_FORMAT)))
        self.print("   fourcc:"+str(cap.get(cv2.CAP_PROP_FOURCC)))
        self.print("   mode:"+str(cap.get(cv2.CAP_PROP_MODE)))

        self.print("   buffer:"+str(cap.get(cv2.CAP_PROP_BUFFERSIZE)))
        self.print("   width:"+str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.print("   height:"+str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.print("   fps:"+str(cap.get(cv2.CAP_PROP_FPS)))
        self.print("   frame count:"+str(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
           
        frameStats = "Frame"
        modelColors = [ CVDemo.CYAN, CVDemo.ORANGE ]
        
         #
        # We need to pre-calculate the dashboard height because the video writer width and height are constant and may not change
        # The video dimensions are the sum of the width and height of the dashboard and the frame
        #
        rowHeight = 25 # in pixels

        # dashboard height = Title Row + Table Column Titles + Control Row + Challenger List Rows + 2 Message Row
        rows = 1 + 1 + 1 + len(challengerModelList) + 2
                         
        self.debug("Dashboard Rows:"+str(rows))
        dashboardHeight = rowHeight * rows
        dashboardFrame = np.zeros([dashboardHeight, width, 3],dtype=np.uint8)
        frameSize = ( width, height + dashboardHeight)
        self.print("   frame size:"+str(frameSize))
        output = cv2.VideoWriter(outVideoPath, cv2.VideoWriter_fourcc(*'mp4v'), fps, frameSize)

          # randomize the colors for each model.  TODO Need to improve this.
        modelColors = np.random.uniform(0, 255, size=(len(challengerModelList)+1, 3))
        modelColors = [ CVDemo.CYAN, CVDemo.ORANGE ]

        frameCnt = 1
        skipCnt = 0
        self.addTitleToDashboard( "Wallaroo Computer Vision Statistics Dashboard", config, dashboardFrame)
        columns = [
            '    Model   ',
            '   Pipeline   ',
            '      Inf     ',
            ' Obj ',
            ' Cls',
            '   Conf  ',
            ' Anom',
            ' Drift',
            '  PSI ']

        self.addColumnTitlesToToDashboard(columns, dashboardFrame)
        wins = [0] * (len(challengerModelList) + 1) # for control
        
        frameCnt = 1
        skipCnt = 0
        config['frame-cnt'] = frameCnt
        try:
            while cap.isOpened():
                totalStartTime = startTime = time.time()
                ret, frame = cap.read()
                endTime = time.time()

                skipCnt += 1
                if (skipCnt < maxSkipCnt):
                    #self.debug("skipping frame:"+str(frameCnt))
                    frameCnt += 1
                    config['frame-cnt']=frameCnt
                    continue
                if (ret == True):
                    frameStats += ":" + str(frameCnt) + " Read: {:.4f}".format(endTime - startTime)

                    # resize frame for width and height the objecet detector is expecting
                    frame = cv2.resize(frame, (width, height))

                    try:
                        infResult = self.runInferenceOnFrame(frame, config)
                    except Exception as e:
                        self.print(e)
                        self.print("could not read frame")
                        self.print("exiting at frame:"+str(frameCnt))
                        raise e
                        break
                        
                    if (infResult == None):
                        self.print("Could not inference frame:"+str(frameCnt))
                        self.print("Pressing on.  Try reading next frame")
                        frameCnt += 1
                        config['frame-cnt']=frameCnt
                        continue
                        
                    #bestModel = self.selectModelWithBestAverageConfidence(json, config['confidence-target'])
                             
                    configList = self.buildConfigsFromShadowDeplInferneces(infResult)
                    
                    # clear the stats frame
                    #statsFrame.fill(0)
                    startTime = time.time()
                    frameStats += " Inf: {:.4f}".format(infResult['inference-time'])
                    onnxTime = infResult['onnx-time']
                    frameStats += " Onnx: {:.4f}".format(onnxTime)
                    bestScore = 0
                    for cnt, infConfig in enumerate(configList):
                        infConfig['image'] = frame
                        infConfig['inference-time'] = infResult['inference-time']
                        infConfig['confidence-target'] = config['confidence-target']
                        infConfig['width'] = config['width']
                        infConfig['height'] = config['height']
                        
                        onnxTime = int(infConfig['onnx-time']) / 1e+9
                        infConfig['onnx-time'] = onnxTime
                        infConfig['color'] = modelColors[cnt]
                        
                        
                        # Drawing the inference results and stats
                        detObjFrame = self.drawDetectedObjectsWithClassification(infConfig)

                        #self.drawStats("Wallaroo Image Statistics Dashboard", statsFrame, infConfig, cnt+1)
                        self.addInferenceResultsToDashboard(infConfig, columns, cnt+4, dashboardFrame)
                    self.addNotesToDashboard(dashboardFrame, frameCnt, config)

                    frame = cv2.vconcat([dashboardFrame, detObjFrame])
                    if (frameCnt > config['record-start-frame']):
                        output.write(frame)
                    endTime = time.time()
                    frameStats += " Draw: {:.4f}".format(endTime - startTime)
                    frameStats += " Total: {:.4f}".format(endTime - totalStartTime)

                    self.print(frameStats)
                    frameStats = "Frame"
                else:
                    self.debug("frame reading error:"+str(cnt))
                if (frameCnt > config['record-end-frame']):
                    break
                # option to exit early
                if (maxFrameCnt > 0 and cnt > maxFrameCnt):
                    break
                frameCnt += 1
                config['frame-cnt']=frameCnt

        except KeyboardInterrupt:
            running = False;
            self.print("Exiting:" + str(running))
            
        cap.release()
        output.release()
        self.print("Finished writing video:"+outVideoPath)
        localTime = datetime.now(newYorkTz)
        localTime = localTime.strftime("%H:%M:%S")
        self.print("End Time:"+localTime)
        return running
        
    # Reads through each frame in the inVideo,
    # Resizes te frame for the mdoel
    # Runs inference
    # Draws inference results on a copy of the frame
    # Writes frame out to video in outVideoPath
    def useBestObjectDetectorInVideoUsingShadowDeployment(self, config):
        newYorkTz = pytz.timezone("America/New_York")
        localTime = datetime.now(newYorkTz)
        localTime = localTime.strftime("%I:%M:%S %p")
        self.print("Start Time:"+localTime)

        inVideoPath = config['input-video']
        outVideoPath = config['output-video']
        fps = config['fps']
        width = config['width']
        height = config['height']
        pipeline = config['pipeline']
        config['pipeline_name'] = pipeline.name()
        pipelineEndPointUrl = pipeline._deployment._url()
        controlModel = config['control-model']
        challengerModelList = config['challenger-model-list']
        
        maxFrameCnt = 0
        maxSkipCnt = 0
        if 'max-frames' in config:
            maxFrameCnt = config['max-frames']
        if 'skip-frames' in config:
            maxSkipCnt = config['skip-frames']
        
        if (maxSkipCnt > 0):
            self.print("Skipping [" + str(maxSkipCnt) +"] frames")
        if (maxFrameCnt > 0):
            self.print("Captureing up to [" + str(maxFrameCnt) +"] frames")
            
        cap = cv2.VideoCapture(inVideoPath)
        self.print("Video Properties")
        self.print("   format:"+str(cap.get(cv2.CAP_PROP_FORMAT)))
        self.print("   fourcc:"+str(cap.get(cv2.CAP_PROP_FOURCC)))
        self.print("   mode:"+str(cap.get(cv2.CAP_PROP_MODE)))

        self.print("   buffer:"+str(cap.get(cv2.CAP_PROP_BUFFERSIZE)))
        self.print("   width:"+str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.print("   height:"+str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.print("   fps:"+str(cap.get(cv2.CAP_PROP_FPS)))
        self.print("   frame count:"+str(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
           
        frameStats = "Frame"
        
        #
        # We need to pre-calculate the dashboard height because the video writer width and height are constant and may not change
        # The video dimensions are the sum of the width and height of the dashboard and the frame
        #
        rowHeight = 25 # in pixels

        # dashboard height = Title Row + Table Column Titles + Control Row + Challenger List Rows + Message Row
        rows = 1 + 1 + 1 + len(challengerModelList) + 2
                         
        self.debug("Dashboard Rows:"+str(rows))
        dashboardHeight = rowHeight * rows
        dashboardFrame = np.zeros([dashboardHeight, width, 3],dtype=np.uint8)
        frameSize = ( width, height + dashboardHeight)
        self.print("   frame size:"+str(frameSize))
       
        # initialize the video writer with the frame size that accounts for the dashboard.
        output = cv2.VideoWriter(outVideoPath, cv2.VideoWriter_fourcc(*'mp4v'), fps, frameSize)
        #output = cv2.VideoWriter(outVideoPath, cv2.VideoWriter_fourcc(*'XVID'), fps, frame_size)
        
        # randomize the colors for each model.  TODO Need to improve this.
        modelColors = np.random.uniform(0, 255, size=(len(challengerModelList)+1, 3))
        modelColors = [ CVDemo.CYAN, CVDemo.ORANGE ]

        frameCnt = 1
        skipCnt = 0
        self.addTitleToDashboard( "Wallaroo Computer Vision Statistics Dashboard", config, dashboardFrame)
        columns = [
            '    Model   ',
            '   Pipeline   ',
            '      Inf     ',
            ' Obj ',
            ' Cls',
            '   Conf  ',
            ' Wins',
            ' Drift',
            '  PSI ']

        self.addColumnTitlesToToDashboard(columns, dashboardFrame)
        wins = [0] * (len(challengerModelList) + 1) # for control
        try:
            while cap.isOpened():
                totalStartTime = startTime = time.time()
                ret, frame = cap.read()
                endTime = time.time()
                
                skipCnt += 1
                if (skipCnt < maxSkipCnt):
                    #self.debug("skipping frame:"+str(frameCnt))
                    frameCnt += 1
                    config['frame-cnt']=frameCnt
                    continue
                if (ret == True):
                    frameStats += ":" + str(frameCnt) + " Read: {:.4f}".format(endTime - startTime)

                    # resize frame for width and height the objecet detector is expecting
                    frame = cv2.resize(frame, (width, height))

                    try:
                        infResult = self.runInferenceOnFrame(frame, config)
                    except Exception as e:
                        self.print(e)
                        self.print("could not read frame")
                        self.print("exiting at frame:"+str(frameCnt))
                        raise e
                        break
                        
                    if (infResult == None):
                        self.print("Could not inference frame:"+str(frameCnt))
                        self.print("Pressing on.  Try reading next frame")
                        frameCnt += 1
                        config['frame-cnt']=frameCnt
                        continue
                    
                    if (infResult == None):
                        self.debug("Could not inference frame:"+str(frameCnt))
                        frameCnt += 1
                        config['frame-cnt']=frameCnt
                        break
                    bestModel = self.selectModelWithBestAverageConfidence(infResult, config['confidence-target'])
                    self.print("best model="+bestModel)
                    configList = self.buildConfigsFromShadowDeplInferneces(infResult)
                    
                    
                    # clear the stats frame
                    #statsFrame.fill(0)
                    startTime = time.time()
                    frameStats += " Inf: {:.4f}".format(infResult['inference-time'])
                    onnxTime = int(infResult['elapsed']) / 1e+9
                    frameStats += " Onnx: {:.4f}".format(onnxTime)
                    bestScore = 0
                    for cnt, infConfig in enumerate(configList):
                        infConfig['image'] = frame
                        infConfig['inference-time'] = infResult['inference-time']
                        infConfig['confidence-target'] = config['confidence-target']
                        infConfig['width'] = config['width']
                        infConfig['height'] = config['height']
                        
                        onnxTime = int(infConfig['onnx-time']) / 1e+9
                        infConfig['onnx-time'] = onnxTime
                        infConfig['color'] = modelColors[cnt]
                        # Drawing the inference results and stats
                       
                        if (infConfig['model_name'] == bestModel):
                            wins[cnt] += 1
                            detObjFrame = self.drawDetectedObjectsWithClassification(infConfig)
                        #else:
                        #    infConfig['color'] = (255,255,255)
                        infConfig['wins'] = wins[cnt]
                        #self.drawStats("Wallaroo Image Statistics Dashboard", statsFrame, infConfig, cnt+1)
                        self.addInferenceResultsToDashboard(infConfig, columns, cnt+4, dashboardFrame)
                        
                    self.addNotesToDashboard(dashboardFrame, frameCnt, config)
                    frame = cv2.vconcat([dashboardFrame, detObjFrame])
                    if (frameCnt > config['record-start-frame']):
                        output.write(frame)
                    endTime = time.time()
                    frameStats += " Draw: {:.4f}".format(endTime - startTime)
                    frameStats += " Total: {:.4f}".format(endTime - totalStartTime)

                    self.print(frameStats)
                    frameStats = "Frame"
                else:
                    self.debug("frame reading error:"+str(cnt))
                if (frameCnt > config['record-end-frame']):
                    break
                # option to exit early
                if (maxFrameCnt > 0 and cnt > maxFrameCnt):
                    break
                frameCnt += 1
        except KeyboardInterrupt:
            self.print("Exiting")
            
        cap.release()
        output.release()
        self.print("Finished writing video:"+outVideoPath)

        localTime = datetime.now(newYorkTz)
        localTime = localTime.strftime("%H:%M:%S")
        self.print("End Time:"+localTime)

  
    # Reads through each frame in the inVideo,
    # Resizes te frame for the mdoel
    # Runs inference
    # Draws inference results on a copy of the frame
    # Writes frame out to video in outVideoPath
    def simulateDriftWhileDetectingAndClassifyingObjectsInVideoUsingPipeline(self, config):
        newYorkTz = pytz.timezone("America/New_York")
        localTime = datetime.now(newYorkTz)
        localTime = localTime.strftime("%I:%M:%S %p")
        self.print("Start Time:"+localTime)

        inVideoPath = config['input-video']
        outVideoPath = config['output-video']
        fps = config['fps']
        width = config['width']
        height = config['height']
        pipelineEndPointUrl = config['endpoint-url']

        maxFrameCnt = 0
        maxSkipCnt = 0
        maxFrameCnt = config['max-frames']
        if 'max-frame' in config:
            maxFrameCnt = config['max-frames']
        maxSkipCnt = config['skip-frames']
        if 'skip-frames' in config:
            maxSkipCnt = config['skip-frames']
        
        
        if (maxSkipCnt > 0):
            self.print("Skipping [" + str(maxSkipCnt) +"] frames")
        if (maxFrameCnt > 0):
            self.print("Captureing up to [" + str(maxFrameCnt) +"] frames")
            
            
        blurFrameStart = 0
        if 'blur-frame-start' in config:
            blurFrameStart = config['blur-frame-start']
        blurFrameEnd = 0
        if 'blur-frame-end' in config:
            blurFrameEnd = config['blur-frame-end']

        if (blurFrameStart > 0):
            self.print("Blurring at frame [" + str(blurFrameStart) + "] to [" + str(blurFrameEnd) + "]")

        cap = cv2.VideoCapture(inVideoPath)
        self.print("Video Properties")
        self.print("   format:"+str(cap.get(cv2.CAP_PROP_FORMAT)))
        self.print("   fourcc:"+str(cap.get(cv2.CAP_PROP_FOURCC)))
        self.print("   mode:"+str(cap.get(cv2.CAP_PROP_MODE)))

        self.print("   buffer:"+str(cap.get(cv2.CAP_PROP_BUFFERSIZE)))
        self.print("   width:"+str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.print("   height:"+str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.print("   fps:"+str(cap.get(cv2.CAP_PROP_FPS)))
        self.print("   frame count:"+str(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
           
        frameStats = "Frame"
        statsRowHeight = 25
        rows = 2
        statsHeight = statsRowHeight*rows
        statsFrame = np.zeros([statsHeight,width,3],dtype=np.uint8)
        frameSize = (width,height+statsHeight)
        self.print("   frame size:"+str(frameSize))

        output = cv2.VideoWriter(outVideoPath, cv2.VideoWriter_fourcc(*'mp4v'), fps, frameSize)
        #output = cv2.VideoWriter(outVideoPath, cv2.VideoWriter_fourcc(*'XVID'), fps, frame_size)
        
        frameCnt = 1
        skipCnt = 0
             
        try:
            while cap.isOpened():
                totalStartTime = startTime = time.time()
                ret, frame = cap.read()
                endTime = time.time()
                frameCnt += 1
                frame = cv2.resize(frame, (width, height))
                
                skipCnt += 1
                if (skipCnt < maxSkipCnt):
                    #self.print("skipping frame:"+str(frameCnt))
                    continue
                if (ret == True):
                    frameStats += ":"+str(frameCnt) +" Read: {:.4f}".format(endTime-startTime)
                   
                    # resize frame for width and height the objecet detector is expecting
                    frame = cv2.resize(frame, (width, height))

                    if (blurFrameStart > 0 and blurFrameStart < frameCnt):
                        self.debug("bluring frame:" + str(frameCnt))
                        frame = cv2.blur(frame,(30,30))
                        if (blurFrameEnd < frameCnt):
                            blurFrameStart = 0
                            blurFrameEnd = 0
                    # run inference on the frame using width and height object detector is expecting
                    json = self.runInferenceOnFrameUsingApi(frame, pipelineEndPointUrl, width, height)
                    
                    #self.saveDataToJsonFile("json-result-2.json",json)
                        
                    if (json == None):
                        self.print("Could not inference frame:"+str(frameCnt))
                        frameCnt += 1
                        break
                    
                    infConfig = self.buildConfigFromPipelineInfernece(json)
                    infConfig['image'] = frame
                    infConfig['inference-time'] = json[0]['inference-time']
                    infConfig['confidence-target'] = config['confidence-target']
                    infConfig['color'] = config['color']
                    
                    frameStats += " Inf: {:.4f}".format(infConfig['inference-time'])
                    onnxTime =  int(infConfig['onnx-time']) / 1e+9
                    frameStats += " Onnx: {:.4f}".format(onnxTime)
                    infConfig['onnx-time'] = onnxTime
                    
                    
                    # Drawing the inference results and stats
                    startTime = time.time()
                    image = self.drawDetectedObjectsWithClassification(infConfig)
                    
                    self.drawStats("Wallaroo Image Statistics Dashboard", statsFrame,infConfig,1)
                    
                    #self.debug("writing frame:"+str(frameCnt))
                    #cv2.imwrite("images/output/frame-:"+str(frameCnt)+".jpg",image)
                    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
                    image = cv2.vconcat([statsFrame,image])
                    
                    output.write(image)
                    endTime = time.time()
                    frameStats += " Draw: {:.4f}".format(endTime-startTime)
                    frameStats += " Total: {:.4f}".format(endTime-totalStartTime)

                    self.print(frameStats)
                    frameStats = "Frame"
                else:
                    self.print("frame reading error:"+str(frameCnt))

                # option to exit early
                if (maxFrameCnt > 0 and frameCnt > maxFrameCnt):
                    break
                    
        except KeyboardInterrupt:
            self.print("Exiting")
            
        cap.release()
        output.release()
        self.print("Finished writing video:"+outVideoPath)

        localTime = datetime.now(newYorkTz)
        localTime = localTime.strftime("%H:%M:%S")
        self.print("End Time:"+localTime)
 
    # Reads through each frame in the inVideo,
    # Resizes te frame for the mdoel
    # Runs inference
    # Draws inference results on a copy of the frame
    # Writes frame out to video in outVideoPath
    def recordVideo(self, config):
        newYorkTz = pytz.timezone("America/New_York")
        localTime = datetime.now(newYorkTz)
        localTime = localTime.strftime("%I:%M:%S %p")
        self.print("Start Time:"+localTime)

        inVideoPath = config['input-video']
        outVideoPath = config['output-video']
        fps = config['fps']
        width = config['width']
        height = config['height']
        
        maxSkipCnt = 0
       
        if 'skip-frames' in config:
            maxSkipCnt = config['skip-frames']
        
        cap = cv2.VideoCapture(inVideoPath)

        self.print("Video Properties")
        self.print("   format:"+str(cap.get(cv2.CAP_PROP_FORMAT)))
        self.print("   fourcc:"+str(cap.get(cv2.CAP_PROP_FOURCC)))
        self.print("   mode:"+str(cap.get(cv2.CAP_PROP_MODE)))

        self.print("   buffer:"+str(cap.get(cv2.CAP_PROP_BUFFERSIZE)))
        self.print("   width:"+str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.print("   height:"+str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.print("   fps:"+str(cap.get(cv2.CAP_PROP_FPS)))
        self.print("   frame count:"+str(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
           
        frameStats = "Frame"
        
        #
        # We need to pre-calculate the dashboard height because the video writer width and height are constant and may not change
        # The video dimensions are the sum of the width and height of the dashboard and the frame
        #
        rowHeight = 25 # in pixels

        # dashboard height = Title Row + Table Column Titles + Control Row + Challenger List Rows + Message Row
        rows = 6
                         
        self.print("Dashboard Rows:"+str(rows))
        dashboardHeight = rowHeight * rows
        dashboardFrame = np.zeros([dashboardHeight, width, 3],dtype=np.uint8)
        frameSize = ( width, height + dashboardHeight)
        self.print("   frame size:"+str(frameSize))

        # initialize the video writer with the frame size that accounts for the dashboard.
        output = cv2.VideoWriter(outVideoPath, cv2.VideoWriter_fourcc(*'mp4v'), fps, frameSize)

        frameCnt = 1
        skipCnt = 0
        try:
            while cap.isOpened():
                totalStartTime = startTime = time.time()
                ret, frame = cap.read()
                endTime = time.time()
                              
                skipCnt += 1
                if (skipCnt < maxSkipCnt):
                    #self.debug("skipping frame:"+str(frameCnt))
                    frameCnt += 1
                    continue
                if (ret == True):
                    frameStats += ":" + str(frameCnt) + " Read: {:.4f}".format(endTime - startTime)

                    # resize frame for width and height the objecet detector is expecting
                    frame = cv2.resize(frame, (width, height))
                    
                    if (frameCnt > config['dashboard-start-frame'] and \
                        frameCnt < config['dashboard-end-frame']):
                        self.addMessageToDashboard(config ,dashboardFrame)
                    else:
                        dashboardFrame.fill(0)

                    # concatenate video frame and statistics dashboard
                    frame = cv2.vconcat([dashboardFrame, frame])
                    if (frameCnt > config['record-start-frame']):
                        output.write(frame)
                    
                    self.print(frameStats)
                    frameStats = "Frame"
                else:
                    self.debug("frame reading error:"+str(cnt))

                if (frameCnt > config['record-end-frame']):
                    break
                frameCnt += 1
        except KeyboardInterrupt:
            self.print("Exiting")
            
        cap.release()
        output.release()
        self.print("Finished writing video:"+outVideoPath)

        localTime = datetime.now(newYorkTz)
        localTime = localTime.strftime("%H:%M:%S")
        self.print("End Time:"+localTime)
        
        
    def stichVideosTogether(self, config):
        videoList = config['video-list']
        outVideoPath = config['output-video']
        fps = config['fps']
        width = config['width']
        height = config['height']

        rowHeight = 25 # in pixels
        rows = 4
        self.print("Dashboard Rows:"+str(rows))
        dashboardHeight = rowHeight * rows
        dashboardFrame = np.zeros([dashboardHeight, width, 3],dtype=np.uint8)
        #frameSize = ( width, height + dashboardHeight)
        frameSize = (640,630)
        self.print("frameSize="+str(frameSize))

        output = cv2.VideoWriter(outVideoPath, cv2.VideoWriter_fourcc(*'mp4v'), fps, frameSize)

        try:
            for video in videoList:
                cap = cv2.VideoCapture(video)
                frameCnt = 0
                while cap.isOpened():
                    self.print("video "+str(video) + " frame cnt:"+str(frameCnt))
                    ret, frame = cap.read()
                    if (ret == True):
                        self.print("   frameSize="+str(frame.shape))
                        output.write(frame)
                    else:
                        self.print("failed to read frame video "+str(video) + " frame cnt:"+str(frameCnt))
                        cap.release()
                        break;
                    frameCnt += 1
                cap.release()


        except KeyboardInterrupt:
            self.print("Exiting")
            
        output.release()
        
         # Reads through each frame in the inVideo,
    # Resizes te frame for the mdoel
    # Runs inference
    # Draws inference results on a copy of the frame
    # Writes frame out to video in outVideoPath
    def detectAndClassifyAnomaliesInVideo(self, config):
        newYorkTz = pytz.timezone("America/New_York")
        localTime = datetime.now(newYorkTz)
        localTime = localTime.strftime("%I:%M:%S %p")
        self.print("Start Time:"+localTime)

        inVideoPath = config['input-video']
        outVideoPath = config['output-video']
        fps = config['fps']
        width = config['width']
        height = config['height']

        # currently only support using sdk
        config['inference'] = 'WALLAROO_SDK'
        # turn on extracting anomalies from inference results
        config['extract-anomalies'] = True
        config['confidence-target'] = 0.0 # only display bounding boxes with confidence > provided #
        
        maxFrameCnt = 0
        maxSkipCnt = 0
        maxFrameCnt = 0
        if 'max-frame' in config:
            maxFrameCnt = config['max-frames']
        maxSkipCnt = 0
        if 'skip-frames' in config:
            maxSkipCnt = config['skip-frames']
        
        
        if (maxSkipCnt > 0):
            self.print("Skipping [" + str(maxSkipCnt) +"] frames")
        if (maxFrameCnt > 0):
            self.print("Captureing up to [" + str(maxFrameCnt) +"] frames")
            
        # if we are inferencing locally with ONNX, load the onnx session with the model and put it in the config for later inferencing
        if config['inference'] == 'ONNX':
            onnx_session= ort.InferenceSession(config['onnx_model_path'])
            model = onnx.load(config['onnx_model_path'])
            onnx.checker.check_model(model)
            config['onnx-session'] = onnx_session
            
        cap = cv2.VideoCapture(inVideoPath)
        self.print("Video Properties")
        self.print("   video input:"+inVideoPath)
        self.print("   video output:"+outVideoPath)

        self.print("   format:"+str(cap.get(cv2.CAP_PROP_FORMAT)))
        self.print("   fourcc:"+str(cap.get(cv2.CAP_PROP_FOURCC)))
        self.print("   mode:"+str(cap.get(cv2.CAP_PROP_MODE)))

        self.print("   buffer:"+str(cap.get(cv2.CAP_PROP_BUFFERSIZE)))
        self.print("   width:"+str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.print("   height:"+str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.print("   fps:"+str(cap.get(cv2.CAP_PROP_FPS)))
        #self.print("   frame count:"+str(self.count_frames_manual(cv2.VideoCapture(inVideoPath))))
   
        frameStats = "Frame"
        rowHeight = 25
        rows = 6
        #statsHeight = statsRowHeight*rows
        #statsFrame = np.zeros([statsHeight,width,3],dtype=np.uint8)
        #frame_size = (width,height+statsHeight)
        
        dashboardHeight = rowHeight * rows
        dashboardFrame = np.zeros([dashboardHeight, width, 3],dtype=np.uint8)
        frameSize = ( width, height + dashboardHeight)
        self.print("   frame size:"+str(frameSize))
        output = cv2.VideoWriter(outVideoPath, cv2.VideoWriter_fourcc(*'mp4v'), fps, frameSize)
        
        self.addTitleToDashboard( "Wallaroo Computer Vision Statistics Dashboard", config, dashboardFrame)

        columns = [
            '    Model   ',
            '   Pipeline   ',
            '      Inf     ',
            ' Obj ',
            ' Cls',
            '   Conf  ',
            ' Anom',
            ' Drift',
            '  PSI ']
        self.addColumnTitlesToToDashboard(columns, dashboardFrame)
        
        recordStartFrame = 0
        if 'record-start-frame' in config:
            recordStartFrame = config['record-start-frame']
        recordEndFrame = 0
        if 'record-end-frame' in config:
            recordEndFrame = config['record-end-frame']
        self.print("recordStartFrame:"+str(recordStartFrame))
        self.print("recordEndFrame:"+str(recordEndFrame))

        frameCnt = 1
        skipCnt = 0
        row = 1
        try:
            while cap.isOpened():
                totalStartTime = startTime = time.time()
                ret, frame = cap.read()
                endTime = time.time()
                skipCnt += 1
                if (skipCnt < maxSkipCnt):
                    #self.debug("skipping frame:"+str(frameCnt))
                    frameCnt += 1
                    continue
                if (ret == True):
                    frameStats += ":"+str(frameCnt) +" Read: {:.4f}".format(endTime-startTime)
                   
                    # resize frame for width and height the objecet detector is expecting
                    frame = cv2.resize(frame, (width, height))

                    # run inference on the frame using width and height object detector is expecting
                    infResult = self.runInferenceOnFrame(frame, config)
                              
                    if (infResult == None):
                        self.print("Could not inference frame:"+str(frameCnt))
                        frameCnt += 1
                        break
                    
                    #infConfig = self.buildConfigFromPipelineInfernece(json)
                    infResult['image'] = frame
                    infResult['model_name'] = config['model_name']
                    infResult['pipeline_name'] = config['pipeline_name']

                    infResult['confidence-target'] = config['confidence-target']
                    infResult['color'] = config['color']
                    
                    frameStats += " Inf: {:.4f}".format(infResult['inference-time'])
                    
                    #This formula is elapsed wallaroo time
                    #onnxTime =  int(infResult['onnx-time']) / 1e+9
                    frameStats += " Onnx: {:.4f}".format(infResult['onnx-time'])
                                        
                    # Drawing the inference results and stats
                    startTime = time.time()
                    infResult['classes'] = infResult['anomalyClasses']
                    infResult['confidences'] = infResult['anomalyConfidences']
                    infResult['boxes'] = infResult['anomalyBoxes']

                    detObjFrame = self.drawDetectedAnomaliesWithClassification(infResult)
                    
                    self.addInferenceResultsToDashboard(infResult, columns, row+3, dashboardFrame)
                    self.addNotesToDashboard(dashboardFrame, frameCnt, config)

                    image = cv2.vconcat([dashboardFrame,detObjFrame])
                    
                    if recordStartFrame > 0:
                        if frameCnt > recordStartFrame:
                            if frameCnt < recordEndFrame:
                                self.debug("recording image:"+str(frameCnt))
                                output.write(image)
                            else:
                                break #exit
                        else:
                            self.debug("skipping frame:"+str(frameCnt))
                    else:
                        self.debug("writing image:"+str(frameCnt))
                        output.write(image)
                        
                    endTime = time.time()
                    frameStats += " Draw: {:.4f}".format(endTime-startTime)
                    frameStats += " Total: {:.4f}".format(endTime-totalStartTime)

                    self.print(frameStats)
                    frameStats = "Frame"
                else:
                    self.print("frame reading error:"+str(frameCnt))

                # option to exit early
                if (maxFrameCnt > 0 and frameCnt > maxFrameCnt):
                    self.print("Exiting early frameCnt > maxFrameCnt")
                    break
                    
                frameCnt += 1
        except KeyboardInterrupt:
            self.print("Exiting")
            
        cap.release()
        output.release()
        self.print("Finished writing video:"+outVideoPath)

        localTime = datetime.now(newYorkTz)
        localTime = localTime.strftime("%H:%M:%S")
        self.print("End Time:"+localTime)

    def extractInferenceResultsIntoDataFrame(self, results, key="outputs"):
        df = pd.DataFrame(columns=['classification','confidence','x1','y1','x2','y2'])
        pd.options.mode.chained_assignment = None  # default='warn'
        pd.options.display.float_format = '{:.2%}'.format

        # Points to where all the inference results are
        outputs = results[key]
        boxes = outputs[0]

        # reshape this to an array of bounding box coordinates converted to ints
        boxList = boxes['Float']['data']
        boxA = np.array(boxList)
        boxes = boxA.reshape(-1, 4)
        boxes = boxes.astype(int)

        df[['x1', 'y1','x2','y2']] = pd.DataFrame(boxes)

        classes = outputs[1]['Int64']['data']
        confidences = outputs[2]['Float']['data']

        idx = 0 
        for idx in range(0,len(classes)):
            df['classification'][idx] = self.getCocoClasses()[classes[idx]] # Classes contains the 80 different COCO classificaitons
            df['confidence'][idx] = confidences[idx]
        return df
    
    def extractShadowInferenceResultsIntoDataFrame(self, results, challenger):
        df = pd.DataFrame(columns=['classification','confidence','x1','y1','x2','y2'])
        pd.options.mode.chained_assignment = None  # default='warn'
        pd.options.display.float_format = '{:.2%}'.format

        # Points to where all the inference results are
        outputs = results['shadow_data']
        boxes = outputs[challenger][0]

        # reshape this to an array of bounding box coordinates converted to ints
        boxList = boxes['Float']['data']
        boxA = np.array(boxList)
        boxes = boxA.reshape(-1, 4)
        boxes = boxes.astype(int)

        df[['x1', 'y1','x2','y2']] = pd.DataFrame(boxes)

        classes = outputs[challenger][1]['Int64']['data']
        confidences = outputs[challenger][2]['Float']['data']

        idx = 0 
        for idx in range(0,len(classes)):
            df['classification'][idx] = self.getCocoClasses()[classes[idx]] # Classes contains the 80 different COCO classificaitons
            df['confidence'][idx] = confidences[idx]
        return df
    
    def extractAnomalyInferenceResultsIntoDataFrame(self, anomolyClasses, anomolyConfidences, anomolyBoxes):
        anomolyDf = pd.DataFrame(columns=['classification','confidence','x1','y1','x2','y2'])
        pd.options.mode.chained_assignment = None  # default='warn'
        pd.options.display.float_format = '{:.2%}'.format

        anomolyDf[['x1', 'y1','x2','y2']] = pd.DataFrame(anomolyBoxes)

        #classes = outputs[1]['Int64']['data']
        #confidences = outputs[2]['Float']['data']

        idx = 0 
        cocoClasses = self.getCocoClasses()
        for idx in range(0,len(anomolyClasses)):
            anomolyDf['classification'][idx] = cocoClasses[anomolyClasses[idx]] # Classes contains the 80 different COCO classificaitons
            anomolyDf['confidence'][idx] = anomolyConfidences[idx]
        return anomolyDf
                                            
    
    # loads the image, resizes in to width, height, converts the result to an tensor.
    # Returns the ndim numpy array and the resized image
    def loadImageAndConvert(self,imagePath, width, height):
        # The image width and height needs to be set to what the model was trained for.  In this case 640x480.
        tensor, resizedImage = self.loadImageAndResize(imagePath, width, height)

        # get npArray from the tensorFloat
        npArray = tensor.cpu().numpy()

        #creates a dictionary with the wallaroo "tensor" key and the numpy ndim array representing image as the value.
        return {"tensor": npArray.tolist()}, resizedImage
    
    # loads the image, resizes in to width, height, converts the result to an tensor.
    # Returns the ndim numpy array and the resized image
    def loadImageAndConvertTiff(self,imagePath, width, height):
        img = cv2.imread(imagePath, 0)
        imgNorm = np.expand_dims(normalize(np.array(img), axis=1),2)
        imgNorm=imgNorm[:,:,0][:,:,None]
        imgNorm=np.expand_dims(imgNorm, 0)
        
        resizedImage = None
        #creates a dictionary with the wallaroo "tensor" key and the numpy ndim array representing image as the value.
        return {"tensor": imgNorm.tolist()}, resizedImage

    def drawDetectedObjectsFromInference(self, results):
        image = self.drawDetectedObjectsWithClassificationsFromInference(results)
        
        frameStats = "Frame"
        statsRowHeight = 25
        rows = 2
        statsHeight = statsRowHeight*rows
        statsImage = np.zeros([statsHeight,results['width'],3],dtype=np.uint8)
        
        results['color'] = (255,255,255)
        statsImage = self.drawStatsDashboard("Wallaroo Computer Vision Statistics Dashboard", results)
        image = cv2.vconcat([statsImage,image])

        # show the output image
        self.pltImshow("Output", image)
        
    def drawDetectedObjectsWithClassificationsFromInference(self, results):
        infResults = results['inf-results']
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
        
        classes = results['classes']
        image = results['image']
        modelName = results['model_name']
        infTime = "{:.2f}".format(results['inference-time'])
        #self.print("drawDetectedObjectsWithClassification boxes"+str(boxes))
        self.debug("confidence-target="+str(results['confidence-target']))
        
        for i in range(0, len(boxes)):
            # extract the confidence (i.e., probability) associated with the
            # classification prediction
            confidence = confidences[i]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
              # display the prediction to our terminal


            if confidence > results['confidence-target']:
                idx = int(classes[i])
                cocoClasses = self.getCocoClasses()
                label = "{}: {:.2f}%".format(cocoClasses[idx], confidence * 100)
                self.debug("[INFO] {}".format(label))
                # extract the index of the class label from the detections,
                # then compute the (x, y)-coordinates of the bounding box
                # for the object
                box = boxes[i]
                (startX, startY, endX, endY) = box

                color = results['color']
                self.debug("color="+str(color))
                 # draw the bounding box and label on the image
                cv2.rectangle(image, (startX, startY), (endX, endY),
                    color, 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(image, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        return image

    def drawShadowDetectedObjectsFromInference(self, results, challenger):
        image = self.drawShadowDetectedObjectsWithClassificationFromInference(results, challenger)
        
        frameStats = "Frame"
        statsRowHeight = 25
        rows = 2
        statsHeight = statsRowHeight*rows
        statsImage = np.zeros([statsHeight,results['width'],3],dtype=np.uint8)
        
        results['color'] = (255,255,255)
        statsImage = self.drawStatsDashboard("Wallaroo Computer Vision Statistics Dashboard", results)
        image = cv2.vconcat([statsImage,image])

        # show the output image
        self.pltImshow("Output", image)
        
    #
    # Draws the challenger inferenced results on the image provided.
    #
    def drawShadowDetectedObjectsWithClassificationFromInference(self, results, challenger):
        infResults = results['inf-results']
        
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
        
        classes = results['classes']
        image = results['image']
        modelName = results['model_name']
        infTime = "{:.2f}".format(results['inference-time'])
        #self.print("drawDetectedObjectsWithClassification boxes"+str(boxes))
        self.debug("confidence-target="+str(results['confidence-target']))
        

        for i in range(0, len(boxes)):
            # extract the confidence (i.e., probability) associated with the
            # classification prediction
            confidence = confidences[i]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
              # display the prediction to our terminal

            if confidence > results['confidence-target']:
                idx = int(classes[i])
                cocoClasses = self.getCocoClasses()
                label = "{}: {:.2f}%".format(cocoClasses[idx], confidence * 100)
                self.debug("[INFO] {}".format(label))
                # extract the index of the class label from the detections,
                # then compute the (x, y)-coordinates of the bounding box
                # for the object
                box = boxes[i]
                (startX, startY, endX, endY) = box

                color = results['color']
                self.debug("color="+str(color))
                 # draw the bounding box and label on the image
                cv2.rectangle(image, (startX, startY), (endX, endY),
                    color, 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(image, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return image   
    
    def get_workspace(self,name):
        wl = wallaroo.Client()
        workspace = None
        for ws in wl.list_workspaces():
            print(ws)
            if ws.name() == name:
                workspace= ws
        if(workspace == None):
            workspace = wl.create_workspace(name)
        return workspace