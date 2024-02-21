## First, let's load the JSON file provided by the user to understand its structure
#import json
#
## Load the JSON content
#part1_file_path = 'part_1_object_tracking.json'
#part2_file_path = 'part_2_frame_dict.json'
#with open(part1_file_path, 'r') as file:
#    json_content = json.load(file)
#with open(part2_file_path, 'r') as file:
#    json_content2 = json.load(file)
#
## Since the user wants to know the "length", we'll check the structure first to understand what they might be referring to
#print(type(json_content), len(json_content["obj"]))
#
#print(type(json_content2), len(json_content2))


import json
import cv2 as cv

# Utility function to load data from JSON files
def load_data_from_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Improved function to draw target object centers
def draw_target_object_center(video_file, obj_centers, output_file="part_1_demo.mp4"):
    cap = cv.VideoCapture(video_file)
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)

    vidwrite = cv.VideoWriter(output_file, cv.VideoWriter_fourcc(*'MP4V'), fps, (frame_width, frame_height))

    for count, (pos_x, pos_y) in enumerate(obj_centers):
        ok, image = cap.read()
        if not ok:
            break  # End of video or read error
        if pos_x != -1 and pos_y != -1:  # Check for valid coordinates
            cv.circle(image, (int(pos_x), int(pos_y)), 5, (0, 0, 255), 2)
        vidwrite.write(image)

    cap.release()
    vidwrite.release()

# Function to draw bounding boxes for objects
def draw_objects_in_video(video_file, frame_dict, output_file="part_2_demo.mp4"):
    cap = cv.VideoCapture(video_file)
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)

    vidwrite = cv.VideoWriter(output_file, cv.VideoWriter_fourcc(*'MP4V'), fps, (frame_width, frame_height))

    count = 0
    ok, image = cap.read()
    while ok:
        obj_list = frame_dict.get(str(count), [])
        for obj in obj_list:
            image = draw_object(obj, image)
        vidwrite.write(image)
        count += 1
        ok, cap.read()

    cap.release()
    vidwrite.release()

# Function to draw a single object
def draw_object(object_dict, image, color=(0, 255, 0), thickness=2):
    x = object_dict['x_min']
    y = object_dict['y_min']
    width = object_dict['width']
    height = object_dict['height']
    return cv.rectangle(image, (x, y), (x + width, y + height), color, thickness)

# Load data from JSON files
object_to_track = load_data_from_json('/mnt/data/object_to_track.json')
frame_dict = load_data_from_json('/mnt/data/frame_dict.json')

# Draw target object centers and objects in video
video_file = "commonwealth.mp4"
draw_target_object_center(video_file, object_to_track['obj'], "part_1_demo.mp4")
draw_objects_in_video(video_file, frame_dict, "part_2_demo.mp4")
