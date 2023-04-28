import cv2
import mediapipe as mp #for creating a facemesh and detect face/iris
import time
import math
import numpy as np
from playsound import playsound

cap = cv2.VideoCapture(0)
blackboard = cv2.imread('blackbox.jpeg')

# Initialize Camera for OD
fCam = cv2.VideoCapture('urbantraffic.mp4')
fCam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
fCam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

ObjLocCount = [0,0,0]
frameCount = 0
lookingFrameCount = 0
eyeFrameCount = 0 
sugg = ""
status = ""
drowsy = ""
yawnCount = 0
sleepFrame = 0
yawnFrame = 0
HRcounter = 0
HLcounter = 0
HAcounter = 0

# OpenCV DNN for OD
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale = 1/255)

#Load class lists for OD
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1, refine_landmarks = True)
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=3)
scale = 13
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

left_iris = [474,475,476,477]
right_iris = [469,470,471,472]
vertice_points_rl = [263, 362, 133, 33]

i = 0
c = 0

#function to crop the webcam to face area
def cropVid(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if np.any(face):
        y = face[0][0]
        x = face[0][1]
        w = face[0][2]
        h = face[0][3]
        height = y + h
        width = x + w
        cx = (width-x)/2
        cy = (height-y)/2
        centerY, centerX=int(y + cy), int(x + cx)
        radiusX,radiusY= int(scale*height/100),int(scale*width/100)

        minX,maxX=x-30,width+30
        minY,maxY=y-30,height+30
        cropped = image[minX:maxX, minY:maxY]
        # width = face[0][0] + face[0][2]
        # height = face[0][1] - face[0][3]
        resized_cropped = cv2.resize(cropped, (width, height))
        return resized_cropped
    return image
printFlag = "unsure"

def trackHazard(frame):
    # Object Detection
    (class_ids, scores, bboxes) = model.detect(frame)
    
    #finding the pov/roi
    hF, wF, z = frame.shape
    margin = int(wF/3)
    margin2 = margin + margin

    (class_ids, scores, bboxes) = model.detect(frame)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id]
        if(x<margin):
            #class_name += " LEFT"
            #print(class_name, " LEFT")
            ObjLocCount[0] += 1
        elif(x>margin and x<margin2):
            #class_name += " CENTER"
            #print(class_name, " CENTER")
            ObjLocCount[1] += 1
        elif(x>margin2):
            #class_name += " RIGHT"
            #print(class_name, " RIGHT")
            ObjLocCount[2] += 1
    
    return 



#main code starts here
while True:
    
    blackboard = cv2.imread('blackbox.jpeg') #for stats
    success, img = cap.read() #for face
    success2, frontFrame = fCam.read() #for Road
    frameCount += 1
    if frameCount % 4 == 0:
        trackHazard(frontFrame)
    

    # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resultCrop = cropVid(img)
    th, tw, _ = resultCrop.shape

    results = faceMesh.process(resultCrop)
    if results.multi_face_landmarks:
        count = 0
        #to draw the lines on the face
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(image = resultCrop, 
                                  landmark_list = faceLms, 
                                  connections = mpFaceMesh.FACEMESH_LEFT_EYE,
                                  landmark_drawing_spec = None,
                                  connection_drawing_spec = drawSpec)
            mpDraw.draw_landmarks(image = resultCrop, 
                                  landmark_list = faceLms, 
                                  connections = mpFaceMesh.FACEMESH_RIGHT_IRIS,
                                  landmark_drawing_spec = None,
                                  connection_drawing_spec = drawSpec)
            """ mpDraw.draw_landmarks(image = resultCrop, 
                                  landmark_list = faceLms.landmark[], 
                                  connections = mpFaceMesh.FACEMESH_RIGHT_EYE,
                                  landmark_drawing_spec = None,
                                  connection_drawing_spec = drawSpec) """
            
            print(type(faceLms))
        
        #calculate coordinates of eye landmarks
        rightEye1x = results.multi_face_landmarks[0].landmark[159].x * tw
        rightEye1y = results.multi_face_landmarks[0].landmark[159].y * th

        rightEye2x = results.multi_face_landmarks[0].landmark[145].x * tw
        rightEye2y = results.multi_face_landmarks[0].landmark[145].y * th


        leftEye1x = results.multi_face_landmarks[0].landmark[386].x * tw
        leftEye1y = results.multi_face_landmarks[0].landmark[386].y * th

        leftEye2x = results.multi_face_landmarks[0].landmark[374].x * tw
        leftEye2y = results.multi_face_landmarks[0].landmark[374].y * th

        
        leftVerLx = results.multi_face_landmarks[0].landmark[362].x * tw
        leftVerLy = results.multi_face_landmarks[0].landmark[362].y * th

        leftVerRx = results.multi_face_landmarks[0].landmark[263].x * tw
        leftVerRy = results.multi_face_landmarks[0].landmark[263].y * th
        
        rightVerLx = results.multi_face_landmarks[0].landmark[33].x * tw
        rightVerLy = results.multi_face_landmarks[0].landmark[33].y * th

        rightVerRx = results.multi_face_landmarks[0].landmark[133].x * tw
        rightVerRy = results.multi_face_landmarks[0].landmark[133].y * th

        #mouth
        LipUx = results.multi_face_landmarks[0].landmark[12].x * tw
        LipUy = results.multi_face_landmarks[0].landmark[12].y * th

        LipDx = results.multi_face_landmarks[0].landmark[15].x * tw
        LipDy = results.multi_face_landmarks[0].landmark[15].y * th

        #right eyelid
        rightEyelidUx = results.multi_face_landmarks[0].landmark[386].x * tw
        rightEyelidUy = results.multi_face_landmarks[0].landmark[386].y * th

        rightEyelidDx = results.multi_face_landmarks[0].landmark[374].x * tw
        rightEyelidDy = results.multi_face_landmarks[0].landmark[374].y * th

        #iris coordinates
        leftIrislx = results.multi_face_landmarks[0].landmark[476].x * tw
        leftIrisly = results.multi_face_landmarks[0].landmark[476].y * th
        leftIrisrx = results.multi_face_landmarks[0].landmark[474].x * tw
        leftIrisry = results.multi_face_landmarks[0].landmark[474].y * th
        
        rightIrisrx = results.multi_face_landmarks[0].landmark[469].x * tw
        rightIrisry = results.multi_face_landmarks[0].landmark[469].y * th
        rightIrislx = results.multi_face_landmarks[0].landmark[471].x * tw
        rightIrisly = results.multi_face_landmarks[0].landmark[471].y * th
        

        #print(resultCrop.shape, leftIrislx, leftIrisly)
        #cv2.circle(resultCrop, (leftIrislx, leftIrisly), 1, (0,255,0), 3)
        #cv2.circle(resultCrop, (leftVerLx,leftVerLy), 1, (255,0,0), 3)
        #cv2.circle(resultCrop, (rightEyelidUx, rightEyelidUy), 1, (0,255,0), 3)

        # Calculating eucledian distance between landmark coodinates
        eucLeftil = math.dist([leftVerLx,leftVerLy], [leftIrislx, leftIrisly])
        eucLeftir = math.dist([leftVerRx,leftVerRy], [leftIrisrx, leftIrisry])
        eucRightil = math.dist([rightVerLx,rightVerLy], [rightIrislx, rightIrisly])
        eucRightir = math.dist([rightVerRx,rightVerRy], [rightIrisrx, rightIrisry])
        
        
        eucLeft = math.dist([leftEye1x, leftEye1y],[leftEye2x, leftEye2y])
        eucRight = math.dist([rightEye1x, rightEye1y],[rightEye2x, rightEye2y])
        
        eucRightEL = math.dist([rightEyelidUx,rightEyelidUy], [rightEyelidDx, rightEyelidDy])
        eucLips = math.dist([LipUx,LipUy], [LipDx, LipDy])

        if eucRightEL < 5:
            sleepFrame += 1
        if eucLips > 45:
            yawnFrame += 1
        
        if sleepFrame > 30:
            drowsy = "Drowsy"
            print("!!ALERT DROWSINESS DETECTED!!")
            playsound('HazardAhead.mp3')
            sleepFrame = 0
        
        if yawnFrame > 30:
            yawnCount += 1
            yawnFrame = 0
        
        if yawnCount >= 3:
            drowsy = "Drowsy"
            playsound('HazardAhead.mp3')
            print("!!ALERT DROWSINESS DETECTED!!")
            yawnCount = 0
        """ print("Left: ", eucLeftil, eucLeftir)
        print("Right: ", eucRightil, eucRightir)
        print("\n") """

        ratioLeft = eucLeftil/eucLeftir
        ratioRight = eucRightil/eucRightir

        if ratioLeft > 2:
            print("looking left")
            if printFlag == "Looking Left":
                lookingFrameCount += 1
            else:
                printFlag = "Looking Left"
                lookingFrameCount = 0
            if lookingFrameCount >= 5:
                ObjLocCount[0] = 0
                sugg = ""
            
        elif ratioLeft <2 and ratioLeft>1:
            print("looking straight")
            if printFlag == "Looking Straight":
                lookingFrameCount += 1
            else:
                printFlag = "Looking Straight"
                lookingFrameCount = 0
            if lookingFrameCount >= 5:
                ObjLocCount[1] = 0
                sugg = ""
            
        elif ratioLeft <1:
            print("looking right")
            if printFlag == "Looking Right":
                lookingFrameCount += 1
            else:
                printFlag = "Looking Right"
                lookingFrameCount = 0
            if lookingFrameCount >= 5:
                ObjLocCount[2] = 0
                sugg = ""
        else:
            printFlag = "unsure"
            

        if ObjLocCount[0] > 30:
            sugg = "!!HAZARD LEFT!!"
            print("\n!!HAZARD LEFT!!")
            HLcounter += 1
            if HLcounter >20:
                #playsound('HazardLeft.mp3')
                HLcounter = 0
        if ObjLocCount[1] > 30:
            sugg = "!!HAZARD CENTER!!"
            print("\n!!HAZARD CENTER!!")
            HAcounter += 1
            if HAcounter >20:
                #playsound('HazardAhead.mp3')
                HAcounter = 0
        if ObjLocCount[2] > 30:
            sugg = "!!HAZARD RIGHT!!"
            print("\n!!HAZARD RIGHT!!")
            HRcounter += 1
            if HRcounter >20:
                #playsound('HazardRight.mp3')
                HRcounter = 0

        """ print("Left Ratio: ", ratioLeft)
        print("Right Ratio: ", ratioRight)
        print("\n") 
         """
        
        #end of main loop


    """ else:
        time.sleep(1)
        print("cont")
        continue """
        
    # print fps and looking direction on screen
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    #cv2.circle(resultCrop, (rightEyelidUx, rightEyelidUy), 1, (0,255,0), 3)
    cv2.putText(blackboard, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
    cv2.putText(blackboard, printFlag, (20, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
    cv2.putText(blackboard, sugg, (20, 130), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
    cv2.putText(blackboard, drowsy, (20, 160), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
    cv2.imshow("stats", blackboard)
    cv2.imshow("Image", resultCrop)
    cv2.imshow("ROAD", frontFrame)
    if frameCount == 999: #resetting the frame count
        frameCount = 0
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        break
cap.release()