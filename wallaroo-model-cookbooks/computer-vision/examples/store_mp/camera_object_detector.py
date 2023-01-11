
# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
from time import sleep, perf_counter
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from threading import Thread
from datetime import datetime
from cv_store.file_video_stream import FileVideoStream
from cv_store.cv_demo_utils import CVDemo
from queue import Queue

class CameraObjectDetector():
      
    def __init__(self, config):      
        self.config = config
        self.cvDemo = CVDemo()
        self.cvDemo.DEBUG = True
        
        queueSize = 128
        self.infQ = Queue(maxsize=queueSize)
        drawQueueSize = 60
        self.drawQ = Queue(maxsize=drawQueueSize)
        
        self.fvs = FileVideoStream(config, "Camera ["+ config['name'] + "]")
        
        rowHeight = 25
        rows = 6
        #statsHeight = statsRowHeight*rows
        #statsFrame = np.zeros([statsHeight,width,3],dtype=np.uint8)
        #frame_size = (width,height+statsHeight)
        
    
        
    def print(self,value):
        localTime = datetime.now()
        #localTime = datetime.now(CVDemo.newYorkTz)

        localTime = localTime.strftime(CVDemo.format)
        print(localTime +" Camera [ "+self.config['name']+" ] " + str(value))

    def start(self):
        
        self.print("CameraObjectDetector Starting start()")
        
        
        # start a thread to read frames from the file video stream
        self.fvs.start()
        self.print("Camera warmup [ 10 seconds ] ")
        time.sleep(2)
        self.print("Camera warmup [ 10 seconds ] Done")

     
        # start a thread to manage the thread pool that is used to detecting objects in frames
        self.cameraThread = Thread(target=self.detectObjectsInFrames, args=())
        self.cameraThread.daemon = True
        self.cameraThread.start()

        # start a thread to draw the inferenced results
        #drawInfThread = Thread(target=self.drawDetectedObjectsInFrames, args=())
        #drawInfThread.daemon = False
        #drawInfThread.start()
        
        self.cameraThread.join()
        self.print("CameraObjectDetector Exiting start")
        
        
    # the method that is executed in each thread
    def runInferenceOnFrame(self, frame, config):
        self.print("runInferenceOnFrame config")
        self.print(config)

        #frameStats += ":"+str(frameCnt) +" Read: {:.4f}".format(endTime-startTime)

        # resize frame for width and height the objecet detector is expecting
        frame = cv2.resize(frame, (config['width'], config['height'])) 

        self.print("resize Complete")

        # run inference on the frame using width and height object detector is expecting
        try:
            infResult = self.cvDemo.runInferenceOnFrame(frame, config)
            print("infResult")
            print(infResult)
            self.print("runInferenceOnFrame Complete")

        except Exception as e:
            self.print(e)
            self.print("could not read frame")
            self.print("exiting at frame:"+str(frameCnt))
            raise e

        if (infResult == None):
            self.print("Could not inference frame:"+str(frameCnt))   
            self.print("Pressing on.  Try reading next frame")    
            frameCnt += 1
            config['frame-cnt']=frameCnt          

        #infConfig = self.buildConfigFromPipelineInfernece(json)
        infResult['image'] = frame
        infResult['model_name'] = config['model_name']
        infResult['pipeline_name'] = config['pipeline_name']

        infResult['confidence-target'] = config['confidence-target'] 
        infResult['color'] = config['color']

        #frameStats += " Inf: {:.4f}".format(infResult['inference-time'])

        #This formula is elapsed wallaroo time
        #onnxTime =  int(infResult['onnx-time']) / 1e+6                
        #frameStats += " Onnx: {:.4f}".format(infResult['onnx-time'])
        #self.print(frameStats)
        self.infQ.put(infResult)

    
    def drawDetectedObjectsInFrames(self):   
        self.print("Started drawDetectedObjectsInFrames")

        inVideoPath = self.config['src-loc']
        outVideoPath = self.config['dest-loc']
        fps = self.config['fps']
        width = self.config['width']
        height = self.config['height']
        pipelineEndPointUrl = self.config['endpoint-url']

        maxFrameCnt = 0
        maxSkipCnt = 0
        maxFrameCnt = 0
        if 'max-frame' in self.config:
            maxFrameCnt = self.config['max-frames']
        maxSkipCnt = 0
        if 'skip-frames' in self.config:
            maxSkipCnt = self.config['skip-frames']
        
        
        if (maxSkipCnt > 0):
            self.print("Skipping [" + str(maxSkipCnt) +"] frames")
        if (maxFrameCnt > 0):
            self.print("Capturing up to [" + str(maxFrameCnt) +"] frames")
            
        # if we are inferencing locally with ONNX, load the onnx session with the model and put it in the config for later inferencing
        if self.config['inference'] == 'ONNX':
            onnx_session= ort.InferenceSession(self.config['onnx_model_path'])
            model = onnx.load(self.config['onnx_model_path'])
            onnx.checker.check_model(model)      
            self.config['onnx-session'] = onnx_session
            
      
        #self.print("   frame count:"+str(self.count_frames_manual(cv2.VideoCapture(inVideoPath))))
        rowHeight = 25
        rows = 6
        frameStats = "Frame" 
        dashboardHeight = rowHeight * rows
        dashboardFrame = np.zeros([dashboardHeight,  self.config['width'], 3],dtype=np.uint8)
        frameSize = ( self.config['width'], self.config['height'] + dashboardHeight)
        self.print("   frame size:"+str(frameSize))
        
        self.cvDemo.addTitleToDashboard( "Wallaroo Computer Vision Statistics Dashboard", self.config, dashboardFrame)

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
        self.cvDemo.addColumnTitlesToToDashboard(columns, dashboardFrame)
        
        recordStartFrame = 0
        if 'record-start-frame' in self.config:
            recordStartFrame = self.config['record-start-frame']
        recordEndFrame = 0
        if 'record-end-frame' in self.config:
            recordEndFrame = self.config['record-end-frame']
        self.print("recordStartFrame:"+str(recordStartFrame))
        self.print("recordEndFrame:"+str(recordEndFrame))

        dashboardHeight = rowHeight * rows
        dashboardFrame = np.zeros([dashboardHeight,  config['width'], 3],dtype=np.uint8)
        frameSize = ( config['width'], config['height'] + dashboardHeight)
        self.print("   frame size:"+str(frameSize))
        # default to writing to mp4
        self.output = cv2.VideoWriter(config['dest-loc'], cv2.VideoWriter_fourcc(*'mp4v'), config['fps'], frameSize)

        frameCnt = 1
        self.config['frame-cnt']=frameCnt
        skipCnt = 0
        row = 1
        
        while not self.drawQ.empty():
            future = self.drawQ.get()
            self.print("future")
            self.print(future)

            with ThreadPoolExecutor(max_workers=1) as executor:
                futureResult = executor.as_completed(future)
                infResult = futureResult.result()
                self.print("infResult")
                self.print(infResult)
            
            # Drawing the inference results and stats
            startTime = time.time()                   
            detObjFrame = self.drawDetectedObjectsWithClassification(infResult)

            self.cvDemo.addInferenceResultsToDashboard(infResult, columns, row+3, dashboardFrame)
            self.cvDemo.addNotesToDashboard(dashboardFrame, frameCnt, self.config)
            image = cv2.vconcat([dashboardFrame,detObjFrame])

            if recordStartFrame > 0:
                if frameCnt > recordStartFrame:
                    if frameCnt < recordEndFrame:
                        self.debug("recording image:"+str(frameCnt))
                        ouself.outputtput.write(image)
                    else:
                        break #exit
                else:
                    self.debug("skipping frame:"+str(frameCnt))
            else:
                self.debug("writing image:"+str(frameCnt))
                self.output.write(image)

            endTime = time.time()
            frameStats += " Draw: {:.4f}".format(endTime-startTime)
            frameStats += " Total: {:.4f}".format(endTime-totalStartTime)

            self.print(frameStats)
            frameStats = "Frame"
        self.print("Finished drawDetectedObjectsInFrames")
        self.output.release()
            
        
    def detectObjectsInFrames(self):
        # keep looping infinitely
        self.print("Started detectObjectsInFrames")
        frameCnt = 0
        while self.fvs.more():
            frame = self.fvs.read()

            with ProcesssPoolExecutor(5) as infProcess:
                self.print("Inference submit"+str(frameCnt))
                future = infProcess.submit(self.runInferenceOnFrame,frame,self.config)
                self.print("drawQ.put"+str(frameCnt))
                #self.drawQ.put(future)  
                frameCnt += 1
            
            time.sleep(0.001)     
                
        self.print("Finished detectObjectsInFrames")

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def more(self):
        # return True if there are still frames in the queue
        return self.Q.qsize() > 0
    
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True