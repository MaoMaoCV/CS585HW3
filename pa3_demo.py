# -*- coding: utf-8 -*-
import json
import cv2 as cv
import numpy as np
# from google.colab.patches import cv2_imshow

# part 1:

def load_obj_each_frame(data_file):
  with open(data_file, 'r') as file:
    frame_dict = json.load(file)
  return frame_dict

def draw_target_object_center(video_file,obj_centers):
  count = 0
  cap = cv.VideoCapture(video_file)
  frames = []
  ok, image = cap.read()
  vidwrite = cv.VideoWriter("part_1_demo.mp4", cv.VideoWriter_fourcc(*'MP4V'), 30, (700,500))
  while ok:
    pos_x,pos_y = obj_centers[count]
    count+=1
    ######!!!!#######
    image = cv.resize(image, (700, 500)) # make sure your video is resize to this size, otherwise the coords in the data file won't work !!!
    ######!!!!#######
    image = cv.circle(image, (int(pos_x),int(pos_y)), 1, (0,0,255), 2)
    vidwrite.write(image)
    ok, image = cap.read()
  vidwrite.release()
# 0.33
def alpha_beta_filter(data, alpha=0.0037, beta= 0.0032, deceleration_factor=0.25):
    x_hat = data[1][0]  # Initial x position
    y_hat = data[1][1]  # Initial y position
    vx_hat = 1    # - 横向向左
    vy_hat = -0.3  # - 竖向向上
    estimated_positions = [(x_hat, y_hat)]
    dt = 1  # Time step
    for i in range(1, len(data)):
        zx, zy = data[i]
        x_pred = x_hat + dt * vx_hat
        y_pred = y_hat +  dt * vy_hat
        x_hat = x_pred + alpha * (zx - x_pred)
        y_hat = y_pred + alpha * (zy - y_pred)
        # Apply deceleration factor to velocity directly
        vx_hat = (vx_hat + ((beta / dt) * (zx - x_pred))) * deceleration_factor
        vy_hat = (vy_hat + (beta / dt) * (zy - y_pred)) * deceleration_factor
        estimated_positions.append((x_hat, y_hat))
    return estimated_positions
def kalman_filter(tracking_points, tracking_original_points):
    # Kalman filter initialization
    dt = 1  # time step
    
    # State transition matrix (assuming constant velocity model)
    A = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    
    # Observation matrix (we can only observe positions, not velocities)
    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])
    
    # Initial state (position and velocity)
    x = np.array([[tracking_points[0][0]], [tracking_points[0][1]], [0], [0]])
    
    # Process noise covariance (Q) and Measurement noise covariance (R)
    Q = np.eye(A.shape[0]) * 0.1
    R = np.eye(H.shape[0]) * 10
    
    # Initial estimation error covariance
    P = np.eye(A.shape[0])
    
    # Updated tracking points with Kalman filter
    updated_tracking_points = []

    for i, point in enumerate(tracking_original_points):
        if point == [-1, -1]:  # If the data is missing, predict the next state
            x = np.dot(A, x)
            P = np.dot(np.dot(A, P), A.T) + Q
            predicted_position = [x[0,0], x[1,0]]
        else:
            # If the data is available, update the state
            Z = np.array([[tracking_points[i][0]], [tracking_points[i][1]]])
            
            # Kalman gain
            S = np.dot(np.dot(H, P), H.T) + R
            K = np.dot(np.dot(P, H.T), np.linalg.inv(S))
            
            # Update the state
            y = Z - np.dot(H, x)
            x = x + np.dot(K, y)
            
            # Update the error covariance
            P = P - np.dot(np.dot(K, H), P)
            
            predicted_position = [x[0,0], x[1,0]]
        
        updated_tracking_points.append(predicted_position)
    
    return updated_tracking_points

frame_dict = load_obj_each_frame("part_1_object_tracking.json")
frame_dict1 = load_obj_each_frame("refilled_tracking_points.json")
frame_dict2 = load_obj_each_frame("refilled_tracking_points2.json")
frame_dict3 = load_obj_each_frame("reversed_tracking_points3.json")
frame_dict4 = load_obj_each_frame("smoothed_tracking_points.json")
frame_dict5 = load_obj_each_frame("modified_tracking_points2.json")
tracking_original_points = load_obj_each_frame("part_1_object_tracking.json")
tracking_points = load_obj_each_frame("part_1_object_tracking_modified.json")

frame_dict10 = kalman_filter(tracking_points['obj'], tracking_original_points['obj'])

print(len(frame_dict['obj']), len(frame_dict10))
# filtered_positions = alpha_beta_filter(frame_dict['obj'])
video_file = "commonwealth.mp4"
# draw_target_object_center(video_file,filtered_positions)


draw_target_object_center(video_file,frame_dict10)

# part 2:

def draw_object(object_dict,image,color = (0, 255, 0), thickness = 2,c_color= \
                (255, 0, 0)):
  # draw box
  x = object_dict['x_min']
  y = object_dict['y_min']
  width = object_dict['width']
  height = object_dict['height']
  image = cv.rectangle(image, (x, y), (x + width, y + height), color, thickness)
  return image

def draw_objects_in_video(video_file,frame_dict):
  count = 0
  cap = cv.VideoCapture(video_file)
  frames = []
  ok, image = cap.read()
  vidwrite = cv.VideoWriter("part_2_demo.mp4", cv.VideoWriter_fourcc(*'MP4V'), 30, (700,500))
  while ok:
    ######!!!!#######
    image = cv.resize(image, (700, 500)) # make sure your video is resize to this size, otherwise the coords in the data file won't work !!!
    ######!!!!#######
    obj_list = frame_dict[str(count)]
    for obj in obj_list:
      image = draw_object(obj,image)
    vidwrite.write(image)
    count+=1
    ok, image = cap.read()
  vidwrite.release()

frame_dict = load_obj_each_frame("frame_dict.json")
video_file = "commonwealth.mp4"
draw_objects_in_video(video_file,frame_dict)

