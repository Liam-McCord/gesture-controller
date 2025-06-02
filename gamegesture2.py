import cv2
import mediapipe as mp
import time
import numpy as np
import pydirectinput
import os
from scipy.spatial import distance
from pathlib import Path
from threading import Thread, Lock
from queue import Queue

class HandLandmark():
    WRIST, THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP, \
    INDEX_FINGER_MCP, INDEX_FINGER_PIP, INDEX_FINGER_DIP, INDEX_FINGER_TIP, \
    MIDDLE_FINGER_MCP, MIDDLE_FINGER_PIP, MIDDLE_FINGER_DIP, MIDDLE_FINGER_TIP, \
    RING_FINGER_MCP, RING_FINGER_PIP, RING_FINGER_DIP, RING_FINGER_TIP, \
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = range(21)

class PointCoordinates():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    def set_coords(self, x, y, z):
        self.x, self.y, self.z = x, y, z
    def return_diff(self, x, y, z):
        return self.x - x, self.y - y, self.z - z

# Threaded key control handler
def key_controller(key_queue, lock):
    while True:
        action, keys = key_queue.get()
        if action == "quit":
            break
        with lock:
            for key in keys:
                if action == "keydown":
                    pydirectinput.keyDown(key)
                elif action == "keyup":
                    pydirectinput.keyUp(key)
        key_queue.task_done()

# Initialize threading
key_queue = Queue()
key_lock = Lock()
thread = Thread(target=key_controller, args=(key_queue, key_lock))
thread.start()

# Main OpenCV + MediaPipe setup
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
dot_pos_array = np.empty((21, 3))
dist_max = np.zeros((21, 21))
current_gesture = previous_gesture = "N/A"
current_pos = PointCoordinates(0, 0, 0)

loaded_gestures = []
gesture_names = []
folder = Path("saved_gestures")
for file in folder.glob("*.npy"):
    gesture_names.append(file.name)
    loaded_gestures.append(np.load(file))

with mp_hands.Hands(
    model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, _ = image.shape
                    cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    dot_pos_array[id] = [cx, cy, cz]

                mp.solutions.drawing_utils.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            dist = distance.cdist(dot_pos_array, dot_pos_array, 'euclidean')
            dist_max[dist > dist_max] = dist[dist > dist_max]
            dist_scaled = dist / dist_max
            dist_scaled = dist_scaled[~np.isnan(dist_scaled)]
            gesture_similarity = [np.dot(dist_scaled.flatten(), gesture.flatten()) /
                                  (np.linalg.norm(dist_scaled.flatten()) * np.linalg.norm(gesture.flatten()))
                                  for gesture in loaded_gestures]
            current_gesture = gesture_names[np.argmax(gesture_similarity)]

            trigger_gesture = "gesture_pinch.npy"
            outputs = ['right', 'left', 'up', 'down']
            if current_gesture == trigger_gesture and previous_gesture != trigger_gesture:
                current_pos.set_coords(*dot_pos_array[HandLandmark.INDEX_FINGER_TIP])
            elif current_gesture == trigger_gesture:
                dx, dy, dz = current_pos.return_diff(*dot_pos_array[HandLandmark.INDEX_FINGER_TIP])
                # Release all keys first
                key_queue.put(("keyup", outputs))
                # Press relevant keys
                if dx > 0:
                    key_queue.put(("keydown", [outputs[0]]))
                else:
                    key_queue.put(("keydown", [outputs[1]]))
                if dy > 0:
                    key_queue.put(("keydown", [outputs[2]]))
                else:
                    key_queue.put(("keydown", [outputs[3]]))
            elif current_gesture != trigger_gesture and previous_gesture == trigger_gesture:
                key_queue.put(("keyup", outputs))
            previous_gesture = current_gesture

        flipped = cv2.flip(image, 1)
        cv2.putText(flipped, f"Gesture: {current_gesture}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.imshow('Hand Tracker', flipped)

        if cv2.waitKey(5) & 0xFF == ord('~'):
            gesture_name = input("New Gesture Name: ")
            np.save(f"saved_gestures/gesture_{gesture_name}", dist_scaled)
            # Reload gestures
            loaded_gestures.clear()
            gesture_names.clear()
            for file in folder.glob("*.npy"):
                gesture_names.append(file.name)
                loaded_gestures.append(np.load(file))
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
key_queue.put(("quit", []))
thread.join()
