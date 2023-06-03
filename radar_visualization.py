import json
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os 

input_file = 'output.json'

# Read the JSON file
with open(input_file, 'r') as file:
    data = json.load(file)

frames = len(data)

# Create the figure and 3D subplot
fig = plt.figure(figsize=(12, 6), dpi=100)
ax = fig.add_subplot(111, projection='3d')

#frames_to_run = 500 # len data if all 
# Function to update the plot
def update(i):
 #   if i >= frames_to_run:
 #      return
    frame_data = data[i]
    x = frame_data['x']
    y = frame_data['y']
    z = frame_data['z']
    ax.clear()
    for j in range(len(x)):
        ax.scatter(x[j], y[j], z[j], c=z[j], cmap='viridis')
    ax.set_xlim3d(-7, 8)
    ax.set_ylim3d(-7, 8)
    ax.set_zlim3d(-7, 8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #ax.set_title(f'Frame {i+1}')
    ax.set_title("Radar Visualization (3D Scatter Plot)")
    text_str = f'Points in frame: {len(x)} \n Frame: {frame_data["frame"]} \n TLV Type: {frame_data["TLV_type"]}'
    ax.text(0.98, 0.02, 0.02, text_str, transform=ax.transAxes, fontsize=10, ha='right', va='bottom')
# Create the animation
anim = animation.FuncAnimation(fig, update, frames=frames, interval=33, blit=False)

plt.show()

# Save the animation
# def save():
#     output_file = 'radar3danimation.avi'
#     if os.path.exists(output_file):
#         response = input("File already exists. Do you want to overwrite it? (y/n): ")
#         if response == 'y':
#             os.remove(output_file)
#         else:
#             print("Exiting... Animation not saved.")
#             exit()
#     writervideo = animation.FFMpegWriter(fps=30)
#     anim.save(output_file, writer=writervideo)
#     print(f"Animation saved to {output_file}")
# save()


# For axis limits 
# max_x = max(max(frame['x']) for frame in data)
# max_y = max(max(frame['y']) for frame in data)
# max_z = max(max(frame['z']) for frame in data)
# min_x = min(min(frame['x']) for frame in data)
# min_y = min(min(frame['y']) for frame in data)
# min_z = min(min(frame['z']) for frame in data)

# print(f"max x: {max_x}")
# print(f"max y: {max_y}")
# print(f"max z: {max_z}")
# print(f"min x: {min_x}")
# print(f"min y: {min_y}")
# print(f"min z: {min_z}")
