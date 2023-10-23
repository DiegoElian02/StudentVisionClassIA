#%%
import cv2
import mediapipe as mp
import time
  

class poseDetection():
    
    def __init__(self, mode=False, upBody=False, smooth=True, detectionConf=0.5, trackConf=0.5):
        
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionConf = detectionConf
        self.trackConf = trackConf
        


        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionConf, self.trackConf)
        
        
        
        #find pose
    def findPose(self, img, draw=True):
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
            
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        
        return img
                
    def  findPostion(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                #print(id, lm)
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255,0, 150), cv2.FILLED)
        
        return lmList
    
def main():
    cap = cv2.VideoCapture('pose3.mp4')
    pTime=0
    detector = poseDetection()
    
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPostion(img, draw=False)
        #print(lmList[14])
        cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0,0, 250), cv2.FILLED)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
   
        cv2.imshow("pose detection", img)
        
        if cv2.waitKey(40) == ord("q"):
            break

    

if __name__ == '__main__':
    
    main()






cv2.destroyAllWindows()
#cap.release()
# %%
import cv2
import mediapipe as mp
import numpy as np

mPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
pose = mPose.Pose()

# Open the video file
cap = cv2.VideoCapture('6.mp4')

drawspec1 = mpDraw.DrawingSpec(thickness=8,circle_radius=8,color=(0,0,255))
drawspec2 = mpDraw.DrawingSpec(thickness=8,circle_radius=8,color=(0,255,0))
# Set the width and height of the output video
output_width = 800
output_height = 700
print('Pollo Loco')

# Loop through the frames of the input video
while True:
    print('Pollo')

    # Read a frame from the input video
    success, img = cap.read()

    h, w, c = img.shape
    imgBlank = np.zeros([h, w, c])
    imgBlank.fill(255)

    results = pose.process(img)
    mpDraw.draw_landmarks(img, results.pose_landmarks,mPose.POSE_CONNECTIONS,drawspec1,drawspec2)
    mpDraw.draw_landmarks(imgBlank, results.pose_landmarks, mPose.POSE_CONNECTIONS, drawspec1, drawspec2)

    # If the frame was not read successfully, break the loop
    if not success:
        break

    # Resize the frame
    img = cv2.resize(img, (output_width, output_height))
    imgBlank = cv2.resize(imgBlank, (output_width, output_height))

    # Show the frame
    cv2.imshow('poseDetection', img)

    cv2.imshow('ExtractedPose',imgBlank)

    # Wait for 1 millisecond
    cv2.waitKey(1)

# Release the VideoCapture object
cap.release()
