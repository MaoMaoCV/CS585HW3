# -*- coding: utf-8 -*-
import json
import cv2 as cv
import numpy as np
from itertools import product
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
    try:
      pos_x,pos_y = obj_centers[count]
    except:
       break
    count+=1
    ######!!!!#######
    image = cv.resize(image, (700, 500)) # make sure your video is resize to this size, otherwise the coords in the data file won't work !!!
    ######!!!!#######
    image = cv.circle(image, (int(pos_x),int(pos_y)), 1, (0,0,255), 2)
    vidwrite.write(image)
    ok, image = cap.read()
  vidwrite.release()
# 0.33
from sklearn.metrics import mean_squared_error
import numpy as np

# Define a function to calculate the MSE between estimated and true positions
def calculate_mse(estimated_positions, true_positions):
    # Convert lists to numpy arrays for easier manipulation
    estimated_positions_arr = np.array(estimated_positions)
    true_positions_arr = np.array(true_positions)
    # Calculate MSE
    return mean_squared_error(true_positions_arr, estimated_positions_arr)

# Modify the alpha-beta filter function to work with the provided data structure
def alpha_beta_filter(data, alpha, beta, deceleration_factor, vx_hat, vy_hat):
    x_hat = data[0][0]  # Assume initial x position from the first valid data point
    y_hat = data[0][1]  # Assume initial y position from the first valid data point
    estimated_positions = []
    dt = 1  # Time step
    for zx, zy in data:
        if zx == -1 or zy == -1:
            zx, zy = x_hat, y_hat
        x_pred = x_hat + dt * vx_hat
        y_pred = y_hat + dt * vy_hat
        x_hat = x_pred + alpha * (zx - x_pred)
        y_hat = y_pred + alpha * (zy - y_pred)
        vx_hat = (vx_hat + ((beta / dt) * (zx - x_pred))) * deceleration_factor
        vy_hat = (vy_hat + (beta / dt) * (zy - y_pred)) * deceleration_factor
        estimated_positions.append((x_hat, y_hat))
        dt += 1
    return estimated_positions

# # Example hyperparameters for testing
# alpha = 0.85
# beta = 0.005
# deceleration_factor = 0.95
# vx_hat = 0.5  # Initial guess for x velocity
# vy_hat = 0.5  # Initial guess for y velocity

# # Apply the alpha-beta filter with example hyperparameters
# estimated_positions = alpha_beta_filter(tracking_data['obj'], alpha, beta, deceleration_factor, vx_hat, vy_hat)

# # Calculate MSE against refilled tracking points
# mse = calculate_mse(estimated_positions, refilled_tracking_points['obj'])

# Define ranges for hyperparameters to test
# alpha_range = np.linspace(0, 1, 20)
# beta_range = np.linspace(0, 5, 20)
# deceleration_factor_range = np.linspace(0.9, 1, 20)

# # Define a function to evaluate the smoothness of estimated positions
# def evaluate_smoothness(estimated_positions):
#     velocities = [np.sqrt((x2 - x1)**2 + (y2 - y1)**2) for (x1, y1), (x2, y2) in zip(estimated_positions[:-1], estimated_positions[1:])]
#     velocity_changes = [abs(v2 - v1) for v1, v2 in zip(velocities[:-1], velocities[1:])]
#     smoothness_score = np.mean(velocity_changes)
#     return smoothness_score

# # Run hyperparameter tuning
frame_dict = load_obj_each_frame("part_1_object_tracking.json")
refilled_tracking_points = load_obj_each_frame("refilled_tracking_points.json")
# best_params = None
# best_smoothness = float('inf')
# for alpha, beta, deceleration_factor in product(alpha_range, beta_range, deceleration_factor_range):
#     estimated_positions = alpha_beta_filter(frame_dict['obj'], alpha, beta, deceleration_factor)
#     smoothness_score = evaluate_smoothness(estimated_positions)
#     if smoothness_score < best_smoothness:
#         best_smoothness = smoothness_score
#         best_params = (alpha, beta, deceleration_factor)

# print(best_params, best_smoothness)

alpha_range = np.arange(0.6, 1, 0.1)
# alpha_range = np.array([0.9])
beta_range = np.arange(0.0001, 0.001, 0.0001)
deceleration_factor_range = np.arange(0.9, 1.1, 0.1)
deceleration_factor_range = np.array([0.25])
# For simplicity, use fixed initial velocity estimates based on a reasonable assumption or average observed change
vx_hat_initial = np.arange(-1, 1, 0.2)
vy_hat_initial = np.arange(-1, 1, 0.2)
 
# Initialize variables to store the best hyperparameters and minimum MSE
best_alpha = 0
best_beta = 0
best_deceleration_factor = 0
min_mse = float('inf')

# Grid search over the range of hyperparameters
for alpha in alpha_range:
    for beta in beta_range:
        for vx in vx_hat_initial:
          for vy in vy_hat_initial:
            for deceleration_factor in deceleration_factor_range:
                # Apply the alpha-beta filter with the current set of hyperparameters
                estimated_positions = alpha_beta_filter(frame_dict['obj'], alpha, beta, deceleration_factor, vx, vy)
                # Calculate MSE against refilled tracking points
                mse = calculate_mse(estimated_positions, refilled_tracking_points['obj'])
                # Update the best hyperparameters if the current MSE is lower than the minimum found so far
                if mse < min_mse:
                    min_mse = mse
                    best_alpha = alpha
                    best_beta = beta
                    best_deceleration_factor = deceleration_factor
                    best_vx = vx
                    best_vy = vy

print(f"{best_alpha}, {best_beta}, {best_deceleration_factor}, {best_vx}, {best_vy}, {min_mse}")

# frame_dict = load_obj_each_frame("object_to_track.json")
# frame_dict = load_obj_each_frame("part_1_object_tracking.json")
filtered_positions = alpha_beta_filter(frame_dict['obj'], best_alpha, best_beta, best_deceleration_factor, best_vx, best_vy)
video_file = "commonwealth.mp4"
print(len(filtered_positions))
draw_target_object_center(video_file,filtered_positions)
print(vx_hat_initial, vy_hat_initial)
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

# frame_dict = load_obj_each_frame("frame_dict.json")
# video_file = "commonwealth.mp4"
# draw_objects_in_video(video_file,frame_dict)

