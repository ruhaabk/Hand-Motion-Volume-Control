import cv2
import mediapipe as mp 
import time 
import HandDetectModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Get default audio device
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Get current volume
current_volume = volume.GetMasterVolumeLevelScalar()  # Value between 0.0 and 1.0
print(f"Current Volume: {current_volume * 100}%")

cap = cv2.VideoCapture(0)

detector = htm.handDetector()

maxVal = 300
minVal = 5
    #Running the webcam
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, drawBool=False)

    if (len(lmList) != 0):
        xValThumb = detector.getXVal(lmList, 4)
        yValThumb = detector.getYVal(lmList, 4)

        xValPoint = detector.getXVal(lmList, 8)
        yValPoint = detector.getYVal(lmList, 8)

        yDist = yValPoint-yValThumb;
        yDist = yDist*yDist;

        xDist = xValPoint-xValThumb;
        xDist = xDist*xDist;

        dist = yDist+xDist
        dist = math.sqrt(dist)
        normVal = (dist - minVal) / (maxVal - minVal)
        normVal = round(normVal, 2)

        if(normVal < 0.8):
            volume.SetMasterVolumeLevelScalar(normVal, None)
        
        current_volume = volume.GetMasterVolumeLevelScalar()  # Value between 0.0 and 1.0
        print(f"Current Volume: {current_volume * 100}%")

    cv2.imshow("Image", img)
    cv2.waitKey(1)