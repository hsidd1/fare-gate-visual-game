import json
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

input_file = 'output.json'

# Read the JSON file
with open(input_file, 'r') as file:
    data = json.load(file)

frames = len(data)

# Create the figure and 3D subplot
fig = plt.figure(figsize=(12, 6), dpi=100)
ax = fig.add_subplot(111, projection='3d')

# Function to update the plot
def update(i):
    frame_data = data[i]

    x = frame_data['x']
    y = frame_data['y']
    z = frame_data['z']

    ax.clear()
    ax.scatter(x, y, z, c=z, cmap='viridis')
    if frame_data["TLV_Type"] == 1020: # change which tlv type to be plotted here
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Frame {i+1}')
        print(f"Frame: {frame_data['frame']}")
        print(f"Number of points in frame {i+1}: {len(x)}")
        
    print(f"maximum and minimum of each coordinate: {max(x), min(x), max(y), min(y), max(z), min(z)}")
# Create the animation
ani = animation.FuncAnimation(fig, update, frames=frames, interval=33, blit=False)

plt.show()
# fix range of xyz axes
# print frame information 
# option to change tlv type to be plotted
# save recording as video
# prnint onto plot 

# import json
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# input_file = 'output.json'

# # Read the JSON file
# with open(input_file, 'r') as file:
#     data = json.load(file)

# # Extract x, y, and z coordinates for each frame
# frames = []
# for frame_data in data:
#     frame = {
#         'x': frame_data['x'],
#         'y': frame_data['y'],
#         'z': frame_data['z']
#     }
#     frames.append(frame)

# # Plot the coordinates for each frame in 3D
# for frame in frames:
#     x = frame['x']
#     y = frame['y']
#     z = frame['z']

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(x, y, z, c=z, cmap='viridis')

#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.set_title('Frame')
#     plt.show()
