import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
import time
import math
import pybullet as p
import pybullet_data

def calculateRoboAngle(inputAngle):

    return 0 

kuka_file = "kuka_iiwa/model.urdf"
p.connect(p.GUI)

target = p.getDebugVisualizerCamera()[11]
p.resetDebugVisualizerCamera(
    cameraDistance=1.1,
    cameraYaw=90,
    cameraPitch=-25,
    cameraTargetPosition=[target[0], target[1], 0.7])

p.setAdditionalSearchPath(pybullet_data.getDataPath())
leftKukaId = p.loadURDF(kuka_file, [0, -0.3, 0], [0, 0, 0, 1], useFixedBase=True)
rightKukaId= p.loadURDF(kuka_file, [0, 0.3, 0], [0, 0, 0, 1], useFixedBase=True)
p.loadURDF("plane.urdf", [0, 0, 0], useFixedBase=True)
p.setRealTimeSimulation(1)
p.setJointMotorControl2(leftKukaId, 0, p.POSITION_CONTROL, targetPosition=1.57)

cap = cv2.VideoCapture(0)
## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        #get right_shoulder coordinate value 
        #get right_elbow coordinate value 
        #take arctan of (y2-y1)/(x2-x1)
        #print value

        try:
            landmarks = results.pose_landmarks.landmark
        except:
            pass

        left_x3 = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x
        left_y3 = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y        
        left_x2 = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x
        left_y2 = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
        left_x1 = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
        left_y1 = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y

        right_x3 = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x
        right_y3 = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y        
        right_x2 = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x
        right_y2 = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y
        right_x1 = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
        right_y1 = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y

        left_arm_S2E_angle = math.atan(((left_y2-left_y1)/(left_x2-left_x1)))
        left_arm_E2W_angle = math.atan(((left_y3-left_y2)/(left_x3-left_x2)))
        right_arm_S2E_angle = math.atan(((right_y2-right_y1)/(right_x2-right_x1)))
        right_arm_E2W_angle = math.atan(((right_y3-right_y2)/(right_x3-right_x2)))

        p.setJointMotorControl2(leftKukaId, 0, p.POSITION_CONTROL, targetPosition=1.57)
        p.setJointMotorControl2(leftKukaId, 1, p.POSITION_CONTROL, targetPosition=0)
        p.setJointMotorControl2(leftKukaId, 2, p.POSITION_CONTROL, targetPosition=0)
        p.setJointMotorControl2(leftKukaId, 3, p.POSITION_CONTROL, targetPosition=(left_arm_S2E_angle+0.67))
        p.setJointMotorControl2(leftKukaId, 5, p.POSITION_CONTROL, targetPosition=-(left_arm_E2W_angle))

        p.setJointMotorControl2(rightKukaId, 0, p.POSITION_CONTROL, targetPosition=1.57)
        p.setJointMotorControl2(rightKukaId, 1, p.POSITION_CONTROL, targetPosition=0)
        p.setJointMotorControl2(rightKukaId, 2, p.POSITION_CONTROL, targetPosition=0)
        p.setJointMotorControl2(rightKukaId, 3, p.POSITION_CONTROL, targetPosition=(right_arm_S2E_angle-0.67))
        p.setJointMotorControl2(rightKukaId, 5, p.POSITION_CONTROL, targetPosition=(right_arm_E2W_angle))
        print(((-right_arm_S2E_angle)))
        p.stepSimulation()

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


    