import matplotlib.pyplot as plt

import json

# Load the JSON file
file_path = 'smoothed_tracking_points.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Display the structure of the loaded data
data.keys(), {key: type(data[key]) for key in data.keys()}
# Extracting tracking points
tracking_points = data['obj']

# Separating original tracking points and filled-in positions
original_points = [point for point in tracking_points if point != [-1, -1]]
filled_in_points = [point for point in tracking_points if point == [-1, -1]]

# Unzipping the points for plotting
original_x, original_y = zip(*original_points) if original_points else ([], [])
filled_x, filled_y = zip(*filled_in_points) if filled_in_points else ([], [])

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(original_x, original_y, c='blue', label='Original Tracking Points', alpha=0.6)
plt.scatter(filled_x, filled_y, c='red', label='Filled-in Positions', alpha=0.6)
plt.title('Visualization of Tracking Points')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)
plt.show()

first_threashold = 30
second_threashold = 60
third_threashold = 150
forth_threashold = 200
modified_tracking_points = tracking_points.copy()

# Iterate through the list, starting from the second element since the first element has no previous
for i in range(1, len(modified_tracking_points)):
    # Ensure we're not trying to modify the filled-in positions marked as [-1, -1]
    if modified_tracking_points[i] != [-1, -1] and modified_tracking_points[i-1] != [-1, -1]:
        # Increase each coordinate by 1% of the previous element's respective coordinate
        if i < first_threashold and i > 0:
            modified_tracking_points[i][0] = modified_tracking_points[i-1][0] - 0.975 * pow(1.001, i)
            modified_tracking_points[i][1] = modified_tracking_points[i-1][1] - 1.001 * pow(1.001, i)
            print(modified_tracking_points[i][0], 2 * pow(1.001, i))
        if i < second_threashold and i > first_threashold -1:
            modified_tracking_points[i][0] = modified_tracking_points[i-1][0] - 0.82* pow(0.9999, i)
            modified_tracking_points[i][1] = modified_tracking_points[i-1][1] - 0.58* pow(0.9999, i)
        if i < third_threashold and i > second_threashold -1:
            modified_tracking_points[i][0] = modified_tracking_points[i-1][0] - 0.65* pow(0.9999, i)
            modified_tracking_points[i][1] = modified_tracking_points[i-1][1] - 0.56* pow(0.9999, i)
        if i < forth_threashold and i > third_threashold - 1:
            modified_tracking_points[i][0] = modified_tracking_points[i-1][0] - 0.55* pow(0.999, i)
            modified_tracking_points[i][1] = modified_tracking_points[i-1][1] - 0.475* pow(0.999, i)
        if i < 250 and i > forth_threashold - 1:
            modified_tracking_points[i][0] = modified_tracking_points[i-1][0] - 0.55 * pow(0.9999, i)
            modified_tracking_points[i][1] = modified_tracking_points[i-1][1] - 0.395 * pow(0.9999, i)

# Checking the first few elements to verify the changes
print(modified_tracking_points)
modified_data = {'obj': modified_tracking_points}

# Define the path for the new JSON file
modified_file_path = 'modified_tracking_points2.json'

# Save the modified data as a JSON file
with open(modified_file_path, 'w') as file:
    json.dump(modified_data, file)


# Load the JSON file
file_path = 'modified_tracking_points2.json'
with open(file_path, 'r') as file:
    data = json.load(file)
# Display the structure of the loaded data
data.keys(), {key: type(data[key]) for key in data.keys()}
# Extracting tracking points
tracking_points = data['obj']

# Separating original tracking points and filled-in positions
original_points = [point for point in tracking_points if point != [-1, -1]]
filled_in_points = [point for point in tracking_points if point == [-1, -1]]

# Unzipping the points for plotting
original_x, original_y = zip(*original_points) if original_points else ([], [])
filled_x, filled_y = zip(*filled_in_points) if filled_in_points else ([], [])

plt.figure(figsize=(10, 6))
plt.scatter(original_x, original_y, c='blue', label='Original Tracking Points', alpha=0.6)
plt.scatter(filled_x, filled_y, c='red', label='Filled-in Positions', alpha=0.6)
plt.title('Visualization of Tracking Points')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.grid(True)
plt.show()