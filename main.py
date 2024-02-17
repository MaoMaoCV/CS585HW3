# First, let's load the JSON file provided by the user to understand its structure
import json

# Load the JSON content
part1_file_path = 'part_1_object_tracking.json'
part2_file_path = 'part_2_frame_dict.json'
with open(part1_file_path, 'r') as file:
    json_content = json.load(file)
with open(part2_file_path, 'r') as file:
    json_content2 = json.load(file)

# Since the user wants to know the "length", we'll check the structure first to understand what they might be referring to
print(type(json_content), len(json_content["obj"]))

print(type(json_content2), len(json_content2))
