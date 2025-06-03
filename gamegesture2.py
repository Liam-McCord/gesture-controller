# basic mediapipe functionality from https://mediapipe.readthedocs.io/en/latest/solutions/hands.html#mediapipe-hands

import cv2
import mediapipe as mp
import time
import numpy as np
import pydirectinput
import os
from scipy.spatial import distance
from pathlib import Path
import threading

from collections import deque

global current_gesture_output # temporary solution for communcation between threads
gesture_lock = threading.Lock()
current_gesture = "N/A"
previous_gesture = "N/A"

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


interpolation_factor = 0.5
prev_x = 0
prev_y = 0

def lerp(a, b, t):
    return a + (b - a) * t # linear interpolation

def update_mouse(dot_pos_x, dot_pos_y, HandLandmark):
    global prev_x, prev_y # previous states

    wrist_x = dot_pos_x[HandLandmark.WRIST]
    wrist_y = dot_pos_y[HandLandmark.WRIST]

    # Scale coordinates (kind of arbitrary depending on screen)
    x_offset = 1200
    target_x = x_offset - int(wrist_x * 4)
    target_y = int(wrist_y * 4) - 1000

    #target_x = 2000 - int(wrist_x * 4)
    #target_y = int(wrist_y * 4) - 500

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
buffer_size = 1
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

def hand_recog_func():
    global current_gesture
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for id, lm in enumerate(hand_landmarks.landmark):
                        h, w, c = image.shape
                        cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                        dot_pos_x[id] = cx
                        dot_pos_y[id] = cy
                        dot_pos_z[id] = cz
                        dot_pos_array[id] = [cx, cy, cz]

                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                dist = distance.cdist(dot_pos_array, dot_pos_array, 'euclidean')
                dist_max[dist > dist_max] = dist[dist > dist_max]
                dist_scaled = dist / dist_max
                dist_scaled = dist_scaled[~np.isnan(dist_scaled)]

                gesture_similarity = [cosine_similarity(dist_scaled, gesture) for gesture in loaded_gestures]
                recognized = gesture_names[gesture_similarity.index(max(gesture_similarity))]

                with gesture_lock:
                    current_gesture = recognized

            flipped = cv2.flip(image, 1)
            cv2.imshow('MediaPipe Hands', flipped)

            key = cv2.waitKey(5) & 0xFF
            if key == ord('='):
                gesture_name = input("New Gesture Name: ")
                np.save(f"saved_gestures/gesture_{gesture_name}", dist_scaled)
                gesture_names.clear()
                loaded_gestures.clear()
                for file in folder.glob("*.npy"):
                    gesture_names.append(file.name)
                    loaded_gestures.append(np.load(file))
            elif key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

def mouse_movement_func():
    global previous_gesture
    while True:
        with gesture_lock:
            gesture = current_gesture

        update_mouse(dot_pos_x, dot_pos_y, HandLandmark)

        if gesture == "gesture_rightclick.npy" and previous_gesture != "gesture_rightclick.npy":
            pydirectinput.click()

        if gesture == "gesture_click.npy" and previous_gesture != "gesture_click.npy":
            pydirectinput.rightClick()

        if gesture == "gesture_fist.npy" and previous_gesture != "gesture_fist.npy":
            pydirectinput.scroll(1)

        previous_gesture = gesture
        time.sleep(0.02)  # prevent CPU overuse

# Start threads
gesture_thread = threading.Thread(target=hand_recog_func, daemon=True)
mouse_thread = threading.Thread(target=mouse_movement_func, daemon=True)

gesture_thread.start()
mouse_thread.start()

gesture_thread.join()  # keeps the main program alive
