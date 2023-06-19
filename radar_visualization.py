import json
import matplotlib.animation as animation
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os 
import matplotlib as mpl

input_file = 'tlv_data_log.json'
# uncomment with raw string for file path to ffmpeg.exe in case of ffmpeg issues 
#mpl.rcParams['animation.ffmpeg_path'] = r''

with open(input_file, 'r') as file:
    data = json.load(file)

total_frames = len(data)
print(f'Total frames: {total_frames}')

# modify below to change number of frames to run in animation
frames_to_run = 100 # len data if all 
marker_size = 5


# Create the figure and 3D subplot
fig = plt.figure(figsize=(9, 6), dpi=100)
ax = fig.add_subplot(111, projection='3d')

# For axis limits 
def print_max_min():
    '''
    Version 1 data max min
    max x: 14.5406686
    max y: 16.3837508
    max z: 15.0282616
    min x: -14.2102056
    min y: 0.0
    min z: -15.5174625
    '''
    max_x = max(max(frame['x']) for frame in data)
    max_y = max(max(frame['y']) for frame in data)
    max_z = max(max(frame['z']) for frame in data)
    min_x = min(min(frame['x']) for frame in data)
    min_y = min(min(frame['y']) for frame in data)
    min_z = min(min(frame['z']) for frame in data)

    print(f"max x: {max_x}")
    print(f"max y: {max_y}")
    print(f"max z: {max_z}")
    print(f"min x: {min_x}")
    print(f"min y: {min_y}")
    print(f"min z: {min_z}")
    return max_x, max_y, max_z, min_x, min_y, min_z

max_x, max_y, max_z, min_x, min_y, min_z = print_max_min()

def update(i):
    if i >= frames_to_run:
       return
    frame_data = data[i]
    #ax.view_init(elev=i, azim=i)
    x = frame_data['x']
    y = frame_data['y']
    z = frame_data['z']
    tid = frame_data.get('tid', None)
    ax.clear()
    if frame_data['TLV_type'] == 1020:
        point_colour = (0,0,0.5) 
    elif frame_data['TLV_type'] == 1010:
        point_colour = 'red'
    else:
        point_colour = 'green'
    for j in range(len(x)):
        #sc = ax.scatter(x[j], y[j], z[j], c=color, cmap='plasma', s=marker_size, alpha=0.5)
        sc = ax.scatter(x[j], y[j], z[j], color=point_colour, s=marker_size, alpha=1)
        
    # ax.set_xlim3d(-15, 15)
    # ax.set_ylim3d(0, 17)
    # ax.set_zlim3d(-16, 16)
    ax.set_xlim3d(round(min_x-1), round(max_x+1))
    ax.set_ylim3d(round(min_y-1), round(max_y+1))
    ax.set_zlim3d(round(min_z-1), round(max_z+1))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.xaxis.set_pane_color((0.992, 0.965, 0.890, 1.0))  
    ax.yaxis.set_pane_color((0.992, 0.965, 0.890, 1.0))  
    ax.zaxis.set_pane_color((0.992, 0.965, 0.890, 1.0))  

    ax.set_title("Radar Visualization (3D Scatter Plot)")
    text_str = (
    f'Points in frame: {len(x)} \n'
    f'Frame: {frame_data["frame"]} \n'
    f'TLV Type: {frame_data["TLV_type"]} \n'
    f'tid: {tid if tid else "None"}'
    )
    ax.text(0.98, 0.02, 0.02, text_str, transform=ax.transAxes, fontsize=10, ha='right', va='bottom')
    
# Create the animation
anim = animation.FuncAnimation(fig, update, frames=frames_to_run, interval=33, blit=False)
is_paused = False
def pause_resume(event):
    global is_paused
    if event.key == ' ':
        is_paused = not is_paused
        if is_paused:
            anim.event_source.stop()
        else:
            anim.event_source.start()

fig.canvas.mpl_connect('key_press_event', pause_resume)
plt.show()

# Save the animation
def save_animation(animation_object, output_file, frame_rate):
    response = input("Do you want to save the animation? (y/n): ")
    if response != 'y'.lower():
        print("Exiting... Animation not saved.")
        return
    if os.path.exists(output_file):
        response = input("File already exists. Do you want to overwrite it? (y/n): ")
        if response == 'y'.lower():
            os.remove(output_file)
            print("Overwriting file...")
        else:
            print("Exiting... Animation not saved.")
            return
    writervideo = animation.FFMpegWriter(fps=frame_rate)
    animation_object.save(output_file, writer=writervideo)
    print(f"Animation saved to {output_file}")

save_animation(anim, 'radar3danimation.avi', frame_rate=10)