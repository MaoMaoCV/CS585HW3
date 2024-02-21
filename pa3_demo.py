# -*- coding: utf-8 -*-
import json
import cv2 as cv
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

def alpha_beta_filter(data, alpha=0.2, beta=0.1):
    x_hat = data[0][0]  # Initial x position
    y_hat = data[0][1]  # Initial y position
    vx_hat = 0  # Initial x velocity
    vy_hat = 0  # Initial y velocity
    estimated_positions = [(x_hat, y_hat)]
    dt = 1  # Time step
    for i in range(1, len(data)):
        zx, zy = data[i]
        x_pred = x_hat + dt * vx_hat
        y_pred = y_hat + dt * vy_hat
        x_hat = x_pred + alpha * (zx - x_pred)
        y_hat = y_pred + alpha * (zy - y_pred)
        vx_hat = vx_hat + (beta / dt) * (zx - x_pred)
        vy_hat = vy_hat + (beta / dt) * (zy - y_pred)
        estimated_positions.append((x_hat, y_hat))
    return estimated_positions

frame_dict = load_obj_each_frame("object_to_track.json")
filtered_positions = alpha_beta_filter(frame_dict['obj'])
video_file = "commonwealth.mp4"
draw_target_object_center(video_file,filtered_positions)

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

