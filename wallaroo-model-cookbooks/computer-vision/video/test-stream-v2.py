import cv2
import numpy as np
from time import sleep

fps = 20
width = 640
height = 480


urlWriter = "appsrc ! videoconvert ! x264enc speed-preset=ultrafast tune=zerolatency threads=2 byte-stream=true ! flvmux ! rtmpsink location=rtmp://20.75.20.171:1935/stream"


urlWriter = "appsrc ! videoconvert ! videoscale ! video/x-raw,width="+str(width)+",height="+str(height)+" ! x264enc ! flvmux ! rtmpsink location=rtmp://20.75.20.171:1935/stream"
print(urlWriter)
out = cv2.VideoWriter(urlWriter,
    cv2.CAP_GSTREAMER, 0, fps, (width, height), True)
if not out.isOpened():
    raise Exception("can't open video writer")

while True:
    frame = np.zeros((height, width, 3), np.uint8)

    # create a red rectangle
    for y in range(0, int(frame.shape[0] / 2)):
        for x in range(0, int(frame.shape[1] / 2)):
            frame[y][x] = (0, 0, 255)

    out.write(frame)
    print("frame written to the server")

    sleep(1 / fps)
