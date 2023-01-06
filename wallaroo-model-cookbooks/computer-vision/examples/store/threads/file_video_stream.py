
# import the necessary packages
from threading import Thread
import sys
import cv2
from queue import Queue
from cv_store.cv_demo_utils import CVDemo
from datetime import datetime

class FileVideoStream():
    
    def __init__(self, config, name):
        queueSize=25
        self.stopped = False
        self.Q = Queue(maxsize=queueSize)
        self.config = config
        self.cvDemo = CVDemo()
        self.stream = None
        self.name = name

    def start(self):
        
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = False
        t.start()
        
        self.print("Exiting start")
        
    def update(self):
        # keep looping infinitely
        self.stream = cv2.VideoCapture(self.config['src-loc'])
        
        self.print("Video Properties")
        self.print("   video input:"+self.config['src-loc'])
        self.print("   video output:"+self.config['dest-loc'])

        self.print("   format:"+str(self.stream.get(cv2.CAP_PROP_FORMAT)))
        self.print("   fourcc:"+str(self.stream.get(cv2.CAP_PROP_FOURCC)))
        self.print("   mode:"+str(self.stream.get(cv2.CAP_PROP_MODE)))

        self.print("   buffer:"+str(self.stream.get(cv2.CAP_PROP_BUFFERSIZE)))
        self.print("   width:"+str(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.print("   height:"+str(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.print("   fps:"+str(self.stream.get(cv2.CAP_PROP_FPS)))
            
            
            
        self.print(" Starting to read frames")
        frameCnt = 0

        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                self.print(" stopped "+str(frameCnt))
                return
            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()
                self.print("     Read Frame "+str(frameCnt))

                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.print("    not grabbed")
                    self.stop()
                    break
                # add the frame to the queue
                self.Q.put((frameCnt,frame))
                frameCnt += 1
            #else:
            #    self.print("   I am full")

        
        self.print(" Finished to read frames "+str(frameCnt))
        self.stream.release()

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def more(self):
        # return True if there are still frames in the queue
        self.print("Q size="+str(self.Q.qsize()))
        return self.Q.qsize() > 0
    
    def stop(self):
        # indicate that the thread should be stopped
        self.print("stopp called")
        self.stopped = True
        
    def print(self,value):
        localTime = datetime.now()
        #localTime = datetime.now(CVDemo.newYorkTz)

        localTime = localTime.strftime(CVDemo.format)
        print(localTime +" FVS " + self.name +" " + str(value))