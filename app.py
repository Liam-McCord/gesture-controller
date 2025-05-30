# basic mediapipe functionality from https://mediapipe.readthedocs.io/en/latest/solutions/hands.html#mediapipe-hands

import cv2
import mediapipe as mp
import time
import numpy as np

from scipy.spatial import distance
cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mpDraw = mp.solutions.drawing_utils


dot_pos_x = np.empty(20)
dot_pos_y = np.empty(20)
dot_pos_z = np.empty(20)

dot_pos_array = np.empty((20,3))

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks: # hand landmarks

        for id, lm in enumerate(hand_landmarks.landmark): # individual finger id (integer) & x,y,z coords?
            dot_pos_x[id - 1] = lm.x #updates numpy positional array, I need to make sure this is the right way to do this and that it scales properly.
            dot_pos_y[id - 1] = lm.y # the -1 is for offsetting the index as id starts at 1 and arrays index from 0.
            dot_pos_z[id - 1] = lm.z

            dot_pos_array[id - 1,0] = lm.x #updates numpy positional array, I need to make sure this is the right way to do this and that it scales properly.
            dot_pos_array[id - 1,1] = lm.y # the -1 is for offsetting the index as id starts at 1 and arrays index from 0.
            dot_pos_array[id - 1,2] = lm.z

        dist = distance.cdist(dot_pos_array,dot_pos_array,'euclidean') # adjacency matrix
        # print(np.shape(dist))
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
            


    