
from cv_store.camera_object_detector import CameraObjectDetector

class Store():
    def __init__(self, name):
        self.name = name
        self.cameraList = []
        
    def addCamera(self, config):
        camera = CameraObjectDetector(config)
        self.cameraList.append(camera)
    
    def startCameras(self):
        for camera in self.cameraList:
            camera.start()
        print("exiting Store.startCameras")
            

        
        
    