import cv2
import mediapipe as mp 
import time 

class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectConf = 0.5, trackConf = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectConf = detectConf
        self.trackConf = trackConf
        self.modelComplexity = modelComplexity

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectConf, self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils
    

    def findHands(self, img, drawBool=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB) #Will process the frame for us
        
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if drawBool:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS) #single hand

        return img
    
    def findPosition(self, img, handNum=0, drawBool = True):

        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNum]
            for id, lm in enumerate(myHand.landmark):

                height, width, depth = img.shape
                cx, cy = int(lm.x*width), int(lm.y*height)

                lmList.append([id,cx, cy])
                if drawBool:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return lmList
    
    def getXVal(self, lmList, target_id):

        for landmark in lmList:
            if landmark[0] == target_id:  # Compare id
                return landmark[1]       # Return x-value (cx)
        return None  # Return None if id is not found

    def getYVal(self, lmList, target_id):

        for landmark in lmList:
            if landmark[0] == target_id:  # Compare id
                return landmark[2]       # Return x-value (cx)
        return None  # Return None if id is not found

def main():

    cap = cv2.VideoCapture(0)

    detector = handDetector()
    #Running the webcam
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if (len(lmList) != 0):
            print(lmList[0]) #we can find the position of a particiular landmark

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()