# basic mediapipe functionality from https://mediapipe.readthedocs.io/en/latest/solutions/hands.html#mediapipe-hands

import cv2
import mediapipe as mp
import time
import numpy as np
import pydirectinput
import os
from scipy.spatial import distance
from pathlib import Path

from collections import deque

class HandLandmark(): # enums for indexing the list
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20



class PointCoordinates():
  def __init__(self, x,y,z):
    self.x = x,
    self.y = y,
    self.z = z
  def set_coords(self, x,y,z):
     self.x = x
     self.y = y
     self.z = z
  def return_diff(self, x, y, z):
     dx = self.x - x
     dy = self.y - y
     dz = self.z - z
     return dx,dy,dz
  def move_direction(self, x, y, z, outputs, key_states):
    dx, dy, dz = self.return_diff(x, y, z)
    desired_keys = set()

    desired_keys.add(outputs[0] if dx > 0 else outputs[1])
    desired_keys.add(outputs[2] if dy > 0 else outputs[3])

    for key in outputs:
        if key in desired_keys and not key_states[key]:
            pydirectinput.keyDown(key)
            key_states[key] = True
        elif key not in desired_keys and key_states[key]:
            pydirectinput.keyUp(key)
            key_states[key] = False


interpolation_factor = 0.8
prev_x = 0
prev_y = 0

def lerp(a, b, t):
    return a + (b - a) * t # linear interpolation

def update_mouse(dot_pos_x, dot_pos_y, HandLandmark):
    global prev_x, prev_y # previous states

    wrist_x = dot_pos_x[HandLandmark.WRIST]
    wrist_y = dot_pos_y[HandLandmark.WRIST]

    # Scale coordinates (kind of arbitrary depending on screen)
    target_x = 2000 - int(wrist_x * 4)
    target_y = int(wrist_y * 4) - 500

    # Add to buffer
    x_buffer.append(target_x)
    y_buffer.append(target_y)

    # Average from buffer
    avg_x = sum(x_buffer) / len(x_buffer)
    avg_y = sum(y_buffer) / len(y_buffer)

    # Interpolate between previous position and the smoothed average
    smoothed_x = lerp(prev_x, avg_x, interpolation_factor)
    smoothed_y = lerp(prev_y, avg_y, interpolation_factor)

    # Move the mouse
    pydirectinput.moveTo(int(smoothed_x), int(smoothed_y))  # Faster updates

    # Update previous positions
    prev_x = smoothed_x
    prev_y = smoothed_y

def cosine_similarity(matrix1, matrix2):
    v1 = matrix1.flatten()
    v2 = matrix2.flatten()
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mpDraw = mp.solutions.drawing_utils

dot_pos_x = np.empty(21)
dot_pos_y = np.empty(21)
dot_pos_z = np.empty(21)

# Initialize the buffer with size 5 (for smooth movement)
buffer_size = 5
x_buffer = deque(maxlen=buffer_size)
y_buffer = deque(maxlen=buffer_size)


dot_pos_array = np.empty((21,3)) # position of points on the hand
dot_pos_array_scaled = np.empty((21,3)) # position of points on the hand scaled by the maximum value

dist_max = np.zeros((21,21))

current_gesture = "N/A"
previous_gesture = "N/A"
current_pos = PointCoordinates(0,0,0)


loaded_gestures = []
gesture_names = []
folder = Path("saved_gestures") # folder storing gesture files

for file in folder.glob("*.npy"):
    gesture_names.append(file.name)
    loaded_gestures.append(np.load(file))


gesture_cosine_offset = np.zeros(len(gesture_names))

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
    #image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks: # hand landmarks

        for id, lm in enumerate(hand_landmarks.landmark): 
            h, w, c = image.shape
            cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * w) # converts x,y,z to coordinates based on the display.
            dot_pos_x[id] = cx # scaled x coordinate of an individual point
            dot_pos_y[id] = cy 
            dot_pos_z[id] = cz
           
            dot_pos_array[id,0] = cx # scaled x coordinate of an individual point, all data for all points is put into an array to be manipulated later.
            dot_pos_array[id,1] = cy 
            dot_pos_array[id,2] = cz

        mp_drawing.draw_landmarks( # draws the points on the screen
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())


      #cv2.circle(image, (int(dot_pos_x[20]), int(dot_pos_y[20])), 10, (255, 0, 0), cv2.FILLED) 

      dist = distance.cdist(dot_pos_array,dot_pos_array,'euclidean') # adjacency matrix with distances of every point to every other point. In the future I may want to cut out the connections on the same finger that are not relevant.
      dist_max[dist > dist_max] = dist[dist > dist_max]
      dist_scaled = (dist) / (dist_max) # normalise the distance matrix with the maximum recorded distance (not perfect but should get within the ballpark of 0 to 1)
      dist_scaled = dist_scaled[~np.isnan(dist_scaled)] # replaces nan values with 1 for diagonal of adj matrix
      gesture_similarity = [cosine_similarity(dist_scaled, gesture) for gesture in loaded_gestures] # generally does cosine difference between current hand position and gestures
      current_gesture = gesture_names[gesture_similarity.index(max(gesture_similarity))] # print closest gesture similarity


      trigger_gesture = "gesture_rightclick.npy"
      trigger_gesture_2 = "gesture_click.npy"
      trigger_gesture_3 = "gesture_fist.npy"
      outputs = ['right', 'left', 'up', 'down']
      update_mouse(dot_pos_x, dot_pos_y, HandLandmark)

      if current_gesture == trigger_gesture and previous_gesture != trigger_gesture:
        pydirectinput.click()
      
      game_controller = 1
      if game_controller == 1:
        if current_gesture == trigger_gesture:
          pydirectinput.keyDown("l")
          #pydirectinput.mouseDown(_pause=False)
        else:
          pydirectinput.keyUp("l")

        if current_gesture == trigger_gesture_2:
          pydirectinput.keyDown("k")
          #pydirectinput.mouseDown(_pause=False)
        else:
          pydirectinput.keyUp("k")
      if current_gesture == trigger_gesture_2 and previous_gesture != trigger_gesture_2:
        pydirectinput.rightClick()
      if current_gesture == trigger_gesture_3 and previous_gesture != trigger_gesture_3:
        pydirectinput.scroll(1)
      

    flipped = cv2.flip(image, 1)  # Flip the image horizontally as we want it to look like a mirror

    cv2.putText(flipped, f"Current Gesture: {current_gesture}", (10, 450), 
            cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3) # display most likely gesture

    cv2.imshow('MediaPipe Hands', flipped)

    if cv2.waitKey(5) & 0xFF == ord('='): # saves a new gesture to a .npy file when ~ is pressed
      gesture_name = input("New Gesture Name: ") 
      np.save(f"saved_gestures/gesture_{gesture_name}", dist_scaled)

      # load gestures with the new gesture
      loaded_gestures = []
      gesture_names = []
      for file in folder.glob("*.npy"):
        gesture_names.append(file.name)
        loaded_gestures.append(np.load(file))

      #np.save(f"saved_gestures/gesture_{len(gesture_names)}", dist_scaled)

    if cv2.waitKey(5) & 0xFF == 27: #quit when esc is pressed
      break
    previous_gesture = current_gesture
cap.release()
          
  