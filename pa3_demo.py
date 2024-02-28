# -*- coding: utf-8 -*-
import json
import cv2 as cv
import numpy as np
from scipy.optimize import linear_sum_assignment
# from google.colab.patches import cv2_imshow

lost_tracks = []
lost_tracks_num = set()
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
video_file = "commonwealth.mp4"
draw_target_object_center(video_file,frame_dict10)

<<<<<<< Updated upstream
# part 2:

=======
draw_target_object_center(video_file,frame_dict5['obj'])

# draw_target_object_center(video_file,frame_dict5['obj'])
modified_data = {'obj': frame_dict5}
# print(modified_data)
# Define the path for the new JSON file
modified_file_path = 'alpha-beta.json'

def draw_kalman_trace(video_file, kalman_points):
    cap = cv.VideoCapture(video_file)
    ok, image = cap.read()
    if not ok:
        return "Error: Could not read video file."
    
    # Adjust the output path to ensure it's correctly saved in the sandbox environment
    vidwrite = cv.VideoWriter("part_1_demo.mp4", cv.VideoWriter_fourcc(*'MP4V'), 30, (700,500))
    
    trace_image = np.zeros((500, 700, 3), dtype=np.uint8)  # Predefining the trace image size based on the resized video frames
    
    for count, kalman_point in enumerate(kalman_points):
        if count > 0:  # Skip drawing line for the first point where there's no previous point
            # Draw a thick line between the previous and current Kalman filtered points for visibility
            cv.line(trace_image, (int(kalman_points[count-1][0]), int(kalman_points[count-1][1])), (int(kalman_point[0]), int(kalman_point[1])), (255, 0, 0), 5)
    
    # Process each frame in the video
    while ok:
        image = cv.resize(image, (700, 500))  # Resize to ensure consistency with coordinates
        
        # Combine the current frame with the trace image
        combined_image = cv.addWeighted(image, 1, trace_image, 0.8, 0)
        vidwrite.write(combined_image)
        ok, image = cap.read()
    
    vidwrite.release()
    return "part_1_demo.mp4"

video_file = "part_1_demo_alpha-beta.mp4" 
kalman_filtered_points = frame_dict10
# Draw the target object center and the Kalman filter trace on the video
output_video_path = draw_kalman_trace(video_file, kalman_filtered_points)
output_video_path

# part 2:

lost_tracks = []
lost_tracks_num = set()

>>>>>>> Stashed changes
def draw_object(object_dict,image,color = (0, 255, 0), thickness = 2, text_color= (255, 0, 0)):
  # draw box
  x = object_dict['x_min']
  y = object_dict['y_min']
  width = object_dict['width']
  height = object_dict['height']
  image = cv.rectangle(image, (x, y), (x + width, y + height), color, thickness)
  object_id = object_dict['id']
  label = str(object_id)
  cv.putText(image, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

  return image

def initialize_tracks(frame_data):
    tracks = []
    next_id = 0
<<<<<<< Updated upstream
    for detection in frame_data:
=======
    for detection in frame_data: 
>>>>>>> Stashed changes
        detection['id'] = next_id  # Assign an ID
        tracks.append(detection)
        next_id += 1
    return tracks, next_id

def compute_cost_matrix(current_detections, previous_tracks):
    cost_matrix = np.zeros((len(current_detections), len(previous_tracks)), dtype=np.float32)
    
    for i, current_det in enumerate(current_detections):
        for j, prev_track in enumerate(previous_tracks):
            # Calculate the centroid of each bounding box
            current_centroid = np.array([current_det['x_min'] + current_det['width'] / 2.0,
                                         current_det['y_min'] + current_det['height'] / 2.0])
            prev_centroid = np.array([prev_track['x_min'] + prev_track['width'] / 2.0,
                                      prev_track['y_min'] + prev_track['height'] / 2.0])
            
            # Compute the Euclidean distance between centroids
            distance = np.linalg.norm(current_centroid - prev_centroid)
            
            # Set the distance as the cost for the matrix
            cost_matrix[i, j] = distance
    
    return cost_matrix

def update_tracks(assignments, tracks, current_detections, next_id, cost_matrix):
    updated_tracks = []
    global lost_tracks, lost_tracks_num
    matched_detections = set()

    # Process assignments to update existing tracks
    for detection_idx, track_idx in assignments:
        # Ensure the detection index is within the range of current detections
        if cost_matrix[detection_idx, track_idx] < 70:
            detection = current_detections[detection_idx]
            track = tracks[track_idx]
            track['x_min'] = detection['x_min']
            track['y_min'] = detection['y_min']
            track['width'] = detection['width']
            track['height'] = detection['height']
            updated_tracks.append(track)
            matched_detections.add(detection_idx)
        else:
            if track['id'] not in lost_tracks_num: 
              lost_tracks.append(tracks[track_idx])
              lost_tracks_num.add(track['id'])
            else: 
              # Find and update the lost track with the same ID
              for lost_track in lost_tracks:
                  if lost_track['id'] == track['id']:
                      lost_track['x_min'] = track['x_min']
                      lost_track['y_min'] = track['y_min']
                      lost_track['width'] = track['width']
                      lost_track['height'] = track['height']
                      break

    # Add new tracks for unmatched detections
    for i, detection in enumerate(current_detections):
        if i not in matched_detections:
            detection['id'] = next_id
            updated_tracks.append(detection)
            next_id += 1

<<<<<<< Updated upstream
    print(len(lost_tracks), len(lost_tracks_num))
=======
>>>>>>> Stashed changes
    return updated_tracks, next_id


def draw_objects_in_video(video_file,frame_dict):
<<<<<<< Updated upstream
  global lost_tracks
=======
  global lost_tracks, lost_tracks_num
>>>>>>> Stashed changes
  # Initialize variables
  tracks, next_id = initialize_tracks(frame_dict["0"])  # Assuming frame_dict["0"] contains the initial frame detections
  
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
    
    # Compute cost matrix between current detections and existing tracks
    cost_matrix = compute_cost_matrix(obj_list, tracks)
  
    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

<<<<<<< Updated upstream
    
    # Update tracks with new assignments
    tracks, next_id = update_tracks(zip(row_ind, col_ind), tracks, obj_list, next_id, cost_matrix)
    
=======
    # Update tracks with new assignments
    tracks, next_id = update_tracks(zip(row_ind, col_ind), tracks, obj_list, next_id, cost_matrix)
    frame_dict[str(count)] = tracks
>>>>>>> Stashed changes

    for obj in tracks:
      image = draw_object(obj,image)
    vidwrite.write(image)
    count+=1
    ok, image = cap.read()

    if len(lost_tracks) > 0: 
       tracks = tracks + lost_tracks
<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
  vidwrite.release()
  
  return frame_dict

frame_dict = load_obj_each_frame("frame_dict.json")
video_file = "commonwealth.mp4"
<<<<<<< Updated upstream
draw_objects_in_video(video_file,frame_dict)

=======
result = draw_objects_in_video(video_file,frame_dict)

with open('tracking_data.json', 'w') as file:
    json.dump(result, file, indent=4)
>>>>>>> Stashed changes
