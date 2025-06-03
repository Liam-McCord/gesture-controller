# basic mediapipe functionality is from example module.


import cv2
import mediapipe as mp
import numpy as np
import os
from scipy.spatial import distance
from pathlib import Path
import time
cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

dot_pos_x = np.empty(468)
dot_pos_y = np.empty(468)
dot_pos_z = np.empty(468)

dot_pos_array = np.empty((468,3))
dot_pos_array_scaled = np.empty((468,3))

dist_max = np.zeros((468,468))
current_gesture = "N/A"
loaded_gestures = []
gesture_names = []
folder = Path("saved_gestures_face")

for file in folder.glob("*.npy"):
    gesture_names.append(file.name)
    loaded_gestures.append(np.load(file))

gesture_cosine_offset = np.zeros(len(gesture_names))

def cosine_similarity(matrix1, matrix2):
    v1 = matrix1.flatten()
    v2 = matrix2.flatten()
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

with mp_holistic.Holistic(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.face_landmarks:
            for id, lm in enumerate(results.face_landmarks.landmark): 
                h, w, c = image.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                dot_pos_x[id] = cx
                dot_pos_y[id] = cy 
                dot_pos_z[id] = cz
                dot_pos_array[id,0] = cx
                dot_pos_array[id,1] = cy 
                dot_pos_array[id,2] = cz

            dist = distance.cdist(dot_pos_array, dot_pos_array, 'euclidean')
            
            dist_max[dist > dist_max] = dist[dist > dist_max]
            #dist_max = dist_max[~np.isnan(dist_max)]
            dist_scaled = dist / dist_max
            dist_scaled = dist_scaled[~np.isnan(dist_scaled)]
            gesture_similarity = [cosine_similarity(dist_scaled, gesture) for gesture in loaded_gestures]
            current_gesture = gesture_names[gesture_similarity.index(max(gesture_similarity))]

            mp_drawing.draw_landmarks(
                image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
            )

        flipped = cv2.flip(image, 1)
        cv2.putText(flipped, f"Current Gesture: {current_gesture}", (10, 450), 
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)

        cv2.imshow('MediaPipe Pose', flipped)

        if cv2.waitKey(5) & 0xFF == ord('~'):
            print("GESTURE RECORDING")
            
            gesture_name = input("New Gesture Name: ") 
            np.save(f"saved_gestures_face/gesture_{gesture_name}", dist_scaled)
            loaded_gestures = []
            gesture_names = []
            for file in folder.glob("*.npy"):
                gesture_names.append(file.name)
                loaded_gestures.append(np.load(file))

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
