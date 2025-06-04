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

import pygetwindow as gw # not necessary for linux/mac/etc
import win32gui
import win32con

from collections import deque

global current_gesture_output # temporary solution for communcation between threads



gesture_lock = threading.Lock()
current_gesture_left = "N/A"
previous_gesture_left = "N/A"

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

def pin_overlay_windows(): # pin overlay
    # Get the window handle using its title
    try:
        window = gw.getWindowsWithTitle("GameOverlay")[0]
        hwnd = window._hWnd  # Get the window handle

        # Set it to "always on top"
        win32gui.SetWindowPos(
            hwnd,
            win32con.HWND_TOPMOST,
            0, 0, 0, 0,
            win32con.SWP_NOMOVE | win32con.SWP_NOSIZE
        )
    except IndexError:
        print("Window not found.")
    print("WINDOW PINNED!")

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



def lerp(a, b, t):
    return a + (b - a) * t # linear interpolation

def update_mouse(dot_pos_x, dot_pos_y, HandLandmark, left_bound, right_bound, lower_bound, upper_bound):
    global prev_x_l, prev_y_l # previous states

    wrist_x = dot_pos_x[HandLandmark.WRIST]
    wrist_y = dot_pos_y[HandLandmark.WRIST]

    # Scale coordinates (kind of arbitrary depending on screen)
    #x_offset = 1200
    #target_x = x_offset - int(wrist_x * 4)
    #target_y = int(wrist_y * 4) - 1000

    # we have a maximum, minimum
    x_min_monitor = 1
    y_min_monitor = 1
    x_max_monitor = pydirectinput.size()[0]
    y_max_monitor = pydirectinput.size()[1]
    
    x_range = right_bound - left_bound
    y_range = upper_bound - lower_bound

    if x_range == 0 or y_range == 0:
        raise ValueError("Bounds are invalid — division by zero risk.")

    target_x = int((wrist_x - left_bound) / x_range * (x_max_monitor - x_min_monitor) + x_min_monitor)
    target_y = int((upper_bound - wrist_y) / y_range * (y_max_monitor - y_min_monitor) + y_min_monitor)
    #2000 - int(wrist_x * 4)


    #target_x = 
    #target_y = -int(wrist_y * 4) +1000 #- 500

    # Add to buffer
    x_buffer_l.append(target_x)
    y_buffer_l.append(target_y)

    # Average from buffer
    avg_x = sum(x_buffer_l) / len(x_buffer_l)
    avg_y = sum(y_buffer_l) / len(y_buffer_l)

    # Interpolate between previous position and the smoothed average
    smoothed_x = lerp(prev_x_l, avg_x, interpolation_factor)
    smoothed_y = lerp(prev_y_l, avg_y, interpolation_factor)

    # Move the mouse
    pydirectinput.moveTo(int(smoothed_x), int(smoothed_y))  # Faster updates

    # Update previous positions
    prev_x_l = smoothed_x
    prev_y_l = smoothed_y

def move_user_right_hand(dot_pos_x_r, dot_pos_y_r, HandLandmark, left_bound, right_bound, lower_bound, upper_bound, centre_rh_x, centre_rh_y):
    global prev_x_r, prev_y_r # previous states

    wrist_x = dot_pos_x_r[HandLandmark.WRIST]
    wrist_y = dot_pos_y_r[HandLandmark.WRIST]

    # Scale coordinates (kind of arbitrary depending on screen)
    #x_offset = 1200
    #target_x = x_offset - int(wrist_x * 4)
    #target_y = int(wrist_y * 4) - 1000

    # we have a maximum, minimum
    x_min_monitor = 1
    y_min_monitor = 1
    x_max_monitor = pydirectinput.size()[0]
    y_max_monitor = pydirectinput.size()[1]
    
    x_range = right_bound - left_bound
    y_range = upper_bound - lower_bound

    if x_range == 0 or y_range == 0:
        raise ValueError("Bounds are invalid — division by zero risk.")

    target_x = int((wrist_x - left_bound) / x_range * (x_max_monitor - x_min_monitor) + x_min_monitor)
    target_y = int((upper_bound - wrist_y) / y_range * (y_max_monitor - y_min_monitor) + y_min_monitor)
    #2000 - int(wrist_x * 4)


    #target_x = 
    #target_y = -int(wrist_y * 4) +1000 #- 500

    # Add to buffer
    x_buffer_r.append(target_x)
    y_buffer_r.append(target_y)

    # Average from buffer
    avg_x = sum(x_buffer_r) / len(x_buffer_r)
    avg_y = sum(y_buffer_r) / len(y_buffer_r)

    # Interpolate between previous position and the smoothed average
    smoothed_x = lerp(prev_x_r, avg_x, interpolation_factor)
    smoothed_y = lerp(prev_y_r, avg_y, interpolation_factor)

    # Move the user

    # WASD, maybe a "rest zone"
    # Use hand calibration-defined center as screen center
    screen_centre_x = centre_rh_x
    screen_centre_y = centre_rh_y

    dead_zone_x = x_max_monitor * 0.05
    dead_zone_y = y_max_monitor * 0.05

    sprint_modifier = 2

    dx = smoothed_x - screen_centre_x
    dy = smoothed_y - screen_centre_y

    # Movement logic (WASD)
    if dy > dead_zone_y:
        pydirectinput.keyDown("s", _pause=False)  # Move forward
    else:
        pydirectinput.keyUp("s", _pause=False)

    if dy < -dead_zone_y:
        pydirectinput.keyDown("w", _pause=False)  # Move backward
    else:
        pydirectinput.keyUp("w", _pause=False)

    if dx > dead_zone_x:
        pydirectinput.keyDown("d", _pause=False)  # Move right
    else:
        pydirectinput.keyUp("d", _pause=False)

    if dx < -dead_zone_x:
        pydirectinput.keyDown("a", _pause=False)  # Move left
    else:
        pydirectinput.keyUp("a", _pause=False)


    sprint_threshold_sq = (sprint_modifier * dead_zone_x) ** 2 + (sprint_modifier * dead_zone_y) ** 2
    distance_sq = dx ** 2 + dy ** 2

    if distance_sq > sprint_threshold_sq:
        pydirectinput.keyDown("shift", _pause=False)
    else:
        pydirectinput.keyUp("shift", _pause=False)
        


    # Update previous positions
    prev_x_r = smoothed_x
    prev_y_r = smoothed_y

def cosine_similarity(matrix1, matrix2):
    v1 = matrix1.flatten()
    v2 = matrix2.flatten()
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
#cv2.namedWindow("PinnedWindow", cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(0)


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mpDraw = mp.solutions.drawing_utils

dot_pos_x = np.empty(21)
dot_pos_y = np.empty(21)
dot_pos_z = np.empty(21)


dot_pos_x_r = np.empty(21)
dot_pos_y_r = np.empty(21)
dot_pos_z_r = np.empty(21)

# Initialize the buffer with size 5 (for smooth movement)
buffer_size = 30
interpolation_factor = 0.2
prev_x_l = 0
prev_y_l = 0

prev_x_r = 0
prev_y_r = 0

x_buffer_l = deque(maxlen=buffer_size)
y_buffer_l = deque(maxlen=buffer_size)

x_buffer_r = deque(maxlen=buffer_size)
y_buffer_r = deque(maxlen=buffer_size)


dot_pos_array = np.empty((21,3)) # position of points on the hand
dot_pos_array_scaled = np.empty((21,3)) # position of points on the hand scaled by the maximum value
dist_max = np.zeros((21,21))

dot_pos_array_r = np.empty((21,3)) # position of points on the hand
dot_pos_array_scaled_r = np.empty((21,3)) # position of points on the hand scaled by the maximum value
dist_max_r = np.zeros((21,21))


current_gesture_left = "N/A"
previous_gesture_left = "N/A"

current_gesture_right = "N/A"
previous_gesture_right = "N/A"


loaded_gestures = []
gesture_names = []
folder = Path("saved_gestures") # folder storing gesture files

for file in folder.glob("*.npy"):
    gesture_names.append(file.name)
    loaded_gestures.append(np.load(file))




def hand_recog_func():
    global current_gesture_left
    global current_gesture_right

    global windows_pinning
    windows_pinning = True # set to true if on windows!

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label  

                    # Convert landmarks to arrays
                    for id, lm in enumerate(hand_landmarks.landmark):
                        h, w, c = image.shape
                        cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * w) # maybe change last one to c?
                        
                        if label == "Left": # left hand
                            dot_pos_x[id] = cx
                            dot_pos_y[id] = cy
                            dot_pos_z[id] = cz
                            dot_pos_array[id] = [cx, cy, cz]
                        elif label == "Right": # right hand 
                            dot_pos_x_r[id] = cx
                            dot_pos_y_r[id] = cy
                            dot_pos_z_r[id] = cz
                            dot_pos_array_r[id] = [cx, cy, cz]
                    # Draw landmarks for both hands
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                    

                    # Only process gestures for the **left hand**
                    if label == "Left":
                        dist = distance.cdist(dot_pos_array, dot_pos_array, 'euclidean')
                        dist_max[dist > dist_max] = dist[dist > dist_max]
                        dist_scaled = dist / dist_max
                        dist_scaled = dist_scaled[~np.isnan(dist_scaled)]

                        gesture_similarity = [cosine_similarity(dist_scaled, gesture) for gesture in loaded_gestures]
                        recognized = gesture_names[gesture_similarity.index(max(gesture_similarity))]

                        with gesture_lock:
                            current_gesture_left = recognized

                    elif label == "Right":
                        dist_r = distance.cdist(dot_pos_array_r, dot_pos_array_r, 'euclidean')
                        dist_max_r[dist_r > dist_max_r] = dist_r[dist_r > dist_max_r]
                        dist_scaled_r = dist_r / dist_max_r # maximum should be the same for both hands.
                        dist_scaled_r = dist_scaled_r[~np.isnan(dist_scaled_r)]

                        gesture_similarity_r = [cosine_similarity(dist_scaled_r, gesture) for gesture in loaded_gestures]
                        recognized_r = gesture_names[gesture_similarity_r.index(max(gesture_similarity_r))]

                        with gesture_lock:
                            current_gesture_right = recognized_r
            if run_tracking == 1:
                cv2.line(image, (int(dot_pos_x_r[HandLandmark.WRIST]), int(dot_pos_y_r[HandLandmark.WRIST])), (x_centre_unscaled, y_centre_unscaled), color=(0, 255, 0), thickness=2)
            flipped = cv2.flip(image, 1)
            resized_frame = cv2.resize(flipped, (320, 240))  # Width x Height in pixels
            cv2.putText(resized_frame, f"{current_display_text}", (5, 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)
            cv2.imshow('GameOverlay', resized_frame)
            if windows_pinning == True:
                windows_pinning = False
                pin_overlay_windows()
            
            key = cv2.waitKey(5) & 0xFF


            if key == ord('='):
                gesture_name = input("New gesture_leftName: ")
                np.save(f"saved_gestures/gesture_{gesture_name}", dist_scaled)
                gesture_names.clear()
                loaded_gestures.clear()
                for file in folder.glob("*.npy"):
                    gesture_names.append(file.name)
                    loaded_gestures.append(np.load(file))
            elif key == 27: # escape key pressed
                break
            elif shutdown == 1:
                break

        cap.release()
        cv2.destroyAllWindows()

def mouse_movement_func(): 
    global previous_gesture_left
    global previous_gesture_right
    global current_display_text
    global run_tracking
    global run_tracking_right
    global x_centre_unscaled
    global y_centre_unscaled
    global shutdown
    
    
    run_tracking = False
    run_tracking_right = False
    current_display_text = "N/A"
    boundary_check = "left"
    shutdown = False
   
    left_bound = 1980
    right_bound = 1980
    lower_bound = 1980 
    upper_bound = 1980
    centre_rh_x = 500
    centre_rh_y = 500

    click_gesture = "gesture_rightclick.npy"
    right_click_gesture = "gesture_click.npy"
    while True:
        #print("H")
        with gesture_lock:
            gesture_left = current_gesture_left
            gesture_right = current_gesture_right 
        if boundary_check == "left":
            current_display_text = "Move hand to left side of screen."
            if gesture_left== "gesture_fist.npy" and previous_gesture_left != "gesture_fist.npy":  # rising edge
                left_bound = dot_pos_x[HandLandmark.WRIST]  # could also be a landmark like hand_landmarks[0].x
                boundary_check = "right"
                time.sleep(1)

        elif boundary_check == "right":
            current_display_text = "Move hand to right side of screen."
            if gesture_left== "gesture_fist.npy" and previous_gesture_left != "gesture_fist.npy":
                right_bound = dot_pos_x[HandLandmark.WRIST]
                boundary_check = "up"
                time.sleep(1)

        elif boundary_check == "up":
            current_display_text = "Move hand to top side of screen."
            if gesture_left== "gesture_fist.npy" and previous_gesture_left != "gesture_fist.npy":
                upper_bound = dot_pos_y[HandLandmark.WRIST]
                boundary_check = "down"
                time.sleep(1)

        elif boundary_check == "down":
            current_display_text = "Move hand to bottom side of screen."
            if gesture_left== "gesture_fist.npy" and previous_gesture_left != "gesture_fist.npy":
                lower_bound = dot_pos_y[HandLandmark.WRIST]
                boundary_check = "right_hand"  # or reset to "left" if looping
                time.sleep(1)
                
        elif boundary_check == "right_hand":
            current_display_text = "Move your other hand to a resting point"
            if gesture_right == "gesture_fist.npy" and previous_gesture_right != "gesture_fist.npy":

                x_min_monitor = 1
                y_min_monitor = 1
                x_max_monitor = pydirectinput.size()[0]
                y_max_monitor = pydirectinput.size()[1]
                x_range = right_bound - left_bound
                y_range = upper_bound - lower_bound

                centre_rh_x = int((dot_pos_x_r[HandLandmark.WRIST] - left_bound) / x_range * (x_max_monitor - x_min_monitor) + x_min_monitor)
                centre_rh_y = int((upper_bound - dot_pos_y_r[HandLandmark.WRIST]) / y_range * (y_max_monitor - y_min_monitor) + y_min_monitor)
                x_centre_unscaled = int(dot_pos_x_r[HandLandmark.WRIST])
                y_centre_unscaled = int(dot_pos_y_r[HandLandmark.WRIST])
                boundary_check = "finished"  # or reset to "left" if looping
                time.sleep(1)

        if boundary_check == "finished":
            
            
            if run_tracking == True: 
                update_mouse(dot_pos_x, dot_pos_y, HandLandmark, left_bound, right_bound, lower_bound, upper_bound)
                # common part
                if gesture_left == "gesture_peace.npy": # inventory
                    if previous_gesture_left != "gesture_peace.npy":
                        pydirectinput.press("e")


                # left hand part
                current_display_text = f"Tracking online, left = {gesture_left}, right = {gesture_right}"
                
            
                if gesture_left == click_gesture and previous_gesture_left != click_gesture:
                    pydirectinput.click()
                if gesture_left == right_click_gesture and previous_gesture_left != right_click_gesture:
                    pydirectinput.rightClick()

                if gesture_left == "gesture_scroll.npy" and previous_gesture_left != "gesture_scroll.npy":
                    pydirectinput.scroll(1)

                
                if gesture_left == click_gesture: # specifically for minecraft
                    pydirectinput.keyDown("l", _pause=False)
                else:
                    pydirectinput.keyUp("l", _pause=False)

                if gesture_left == right_click_gesture:
                    pydirectinput.keyDown("k", _pause=False)
                else:
                    pydirectinput.keyUp("k", _pause=False)

                if run_tracking_right == True:
                    move_user_right_hand(dot_pos_x_r, dot_pos_y_r, HandLandmark, left_bound, right_bound, lower_bound, upper_bound, centre_rh_x, centre_rh_y)
                    
                    # right hand stuff
                    trigger_gesture_r = "gesture_splay.npy"
                    trigger_gesture_2_r = "gesture_fist.npy"


                    if gesture_right == trigger_gesture_r: # crouch
                        pydirectinput.keyDown("ctrl", _pause=False)
                    else:
                        pydirectinput.keyUp("ctrl", _pause=False)

                    if gesture_right == trigger_gesture_2_r: # jump
                        pydirectinput.keyDown("space", _pause=False)
                    else:
                        pydirectinput.keyUp("space", _pause=False)



                
            if gesture_left== "gesture_toggle.npy" and previous_gesture_left != "gesture_toggle.npy":

                current_display_text = f"Tracking offline."
                run_tracking = not run_tracking
                keys_to_release = ['shift', 'ctrl', 'space', 'w', 'a', 's', 'd', 'up', 'down', 'left', 'right']

                for key in keys_to_release: # release all pressed keys
                    pydirectinput.keyUp(key)

            if gesture_right == "gesture_toggle.npy" and previous_gesture_right != "gesture_toggle.npy":

                current_display_text = f"Right Hand Tracking offline."
                run_tracking_right = not run_tracking_right
                keys_to_release = ['shift', 'ctrl', 'space', 'w', 'a', 's', 'd', 'up', 'down', 'left', 'right']

                for key in keys_to_release: # release all pressed keys
                    pydirectinput.keyUp(key)
            if gesture_left == "gesture_quit.npy" or gesture_right == "gesture_quit.npy": # emergency quit
                shutdown = True
                keys_to_release = ['shift', 'ctrl', 'space', 'w', 'a', 's', 'd', 'up', 'down', 'left', 'right']
                for key in keys_to_release: # release all pressed keys
                    pydirectinput.keyUp(key)
                break

        previous_gesture_left = gesture_left
        previous_gesture_right = gesture_right
        time.sleep(0.001)  # prevent CPU overuse



# Start threads
gesture_thread = threading.Thread(target=hand_recog_func, daemon=True)
mouse_thread = threading.Thread(target=mouse_movement_func, daemon=True)

gesture_thread.start()
mouse_thread.start()

gesture_thread.join()  # keeps the main program alive


def set_boundaries():
    boundary_check = "left"
    if boundary_check == "left":
        # display "move to left" text
        # if fist curled:
        # set left coordinates
        # switch current_boundary to 
        pass