import cv2
import json
import numpy as np
import yaml
from Radar_data import RadarData
# Parsing sensor specific configuration from yaml file
with open('sensor_config.yaml', 'r') as file:
    s_config = yaml.safe_load(file)

# Access the values from the loaded configuration
alpha = s_config['SensorAngles']['alpha']
beta = s_config['SensorAngles']['beta']
offsetx = s_config['Offsets']['offsetx']
offsety = s_config['Offsets']['offsety']
offsetz = s_config['Offsets']['offsetz']

print(f"alpha: {alpha}")
print(f"beta: {beta}")
print(f"offsetx: {offsetx}")
print(f"offsety: {offsety}")
print(f"offsetz: {offsetz}")

'''
# Entry Sensor Rot Matrices
s1_rotz = np.asmatrix(([math.cos(math.radians(-alpha)), -math.sin(math.radians(-alpha)),0], [math.sin(math.radians(-alpha)),math.cos(math.radians(-alpha)), 0], [0,0,1]))
s1_rotx = np.asmatrix(([1,0,0], [0,math.cos(math.radians(beta)), -math.sin(math.radians(beta))], [0, math.sin(math.radians(beta)),math.cos(math.radians(beta))]))
print(s1_rotx)
# Exit Sensor Rot Matrices
s2_rotz = np.asmatrix(([math.cos(math.radians(alpha)), -math.sin(math.radians(alpha)),0], [math.sin(math.radians(alpha)),math.cos(math.radians(alpha)), 0], [0,0,1]))
s2_rotx = np.asmatrix(([1,0,0], [0,math.cos(math.radians(-(180+beta))), -math.sin(math.radians(-(180+beta)))], [0, math.sin(math.radians(-(180+beta))),math.cos(math.radians(-(180+beta)))]))
'''

# for entry sensor
def calc_rot_matrix(alpha, beta):
    """alpha is the angle along z axis - yaw
    beta is the angle along x axis - pitch
    gamma is the angle along y axis - roll, not used here
    all angles are in degrees and counter.
    Rototation matrix is calculated in the order of z -> x -> y
    """
    rotz = np.zeros((3,3))
    rotz[0,0] = np.cos(np.radians(alpha))
    rotz[0,1] = -np.sin(np.radians(alpha))
    rotz[1,0] = np.sin(np.radians(alpha))
    rotz[1,1] = np.cos(np.radians(alpha))
    rotz[2,2] = 1
    rotx = np.zeros((3,3))
    rotx[0,0] = 1
    rotx[1,1] = np.cos(np.radians(beta))
    rotx[1,2] = -np.sin(np.radians(beta))
    rotx[2,1] = np.sin(np.radians(beta))
    rotx[2,2] = np.cos(np.radians(beta))
    return rotz, rotx

def rot_mtx_entry(alpha, beta):
    return calc_rot_matrix(alpha, beta)

def rot_mtx_exit(alpha, beta):
    return calc_rot_matrix(alpha + 180, beta)

s1_rotz_elliot, s1_rotx_elliot = rot_mtx_entry(alpha, beta)
s2_rotz_elliot, s2_rotx_elliot = rot_mtx_exit(alpha, beta)

with open('cv-config.yaml', 'r') as file:
    cv_config = yaml.safe_load(file)
# Opening JSON file
radar_data_file = cv_config['Files']['radar_data_file']
with open(radar_data_file) as json_file:
    data = json.load(json_file)

# Print the data of dictionary
# print(data.keys())
# print(data['sensors'])
# print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
# print(data['walkwayConfig'])

# process data
radar_points = []
for i in data['frames']:

    num_ob = i['sensorMessage']['metadata']['numOfDetectedObjects']

    # timestamp = ((i['sensorMessage']['metadata']['timeStamp']))
    timestamp = ((i['timestamp']))
    # print('WORLD TIME' + str(timestamp))

    for j in range(num_ob):
        s = dict()
        s['sensorId'] = (i['sensorMessage']['object']['detectedPoints'][j]['sensorId'])
        s['x'] = i['sensorMessage']['object']['detectedPoints'][j]['x']
        s['y'] = i['sensorMessage']['object']['detectedPoints'][j]['y']
        s['z'] = i['sensorMessage']['object']['detectedPoints'][j]['z']  # Flattened so it should always return be 0
        s['timestamp'] = timestamp
        # print('TIMESTAMP' + str(timestamp))
        #
        raw_coords = np.asmatrix(([s['x']], [s['y']], [s['z']])) * 10  #  cm to mm
        if s['sensorId'] == 1:
            # entry sensor
            # transformed_coords = (np.matmul(s1_rotx, np.matmul(s1_rotz, raw_coords)))
            transformed_coords = np.matmul(s1_rotz_elliot, np.matmul(s1_rotx_elliot, raw_coords))
            transformed_coords += np.array([[offsetx], [-offsety], [offsetz]])
            # scale_x = 17.25
            # scale_y = -28.75  # cm
            # scale_z = 85
        elif s['sensorId'] == 2:
            # transformed_coords = (np.matmul(s2_rotx, np.matmul(s2_rotz, raw_coords)))
            transformed_coords = np.matmul(s2_rotz_elliot, np.matmul(s2_rotx_elliot, raw_coords))
            transformed_coords += np.array([[-offsetx], [offsety], [offsetz]])
            # scale_x = -17.25
            # scale_y = 28.75
            # scale_z = 85
        else:
            print('Sensor ID not found')
            raise
        # s['x'] = float(transformed_coords[0] + scale_x)
        # s['y'] = float(transformed_coords[1] + scale_y)
        # s['z'] = float(transformed_coords[2] + scale_z)    # Flattened so it should always return be 0

        s['x'] = float(transformed_coords[0])
        s['y'] = float(transformed_coords[1])
        s['z'] = float(transformed_coords[2])
        
        radar_points.append(s)

#print(radar_points[0:6])
radar_data = RadarData(radar_points)
print(f"radar data initialized as {radar_data}")

                        # ------------------ VISUALIZATION ------------------ #
from point_cloud import StaticPoints


# Access the values from the loaded configuration
rad_cam_offset = cv_config['rad_cam_offset']
scalemm2px = cv_config['scalemm2px']
wait_ms = cv_config['wait_ms']
slider_xoffset = cv_config['TrackbarDefaults']['slider_xoffset']
slider_yoffset = cv_config['TrackbarDefaults']['slider_yoffset']
xy_trackbar_scale = cv_config['TrackbarDefaults']['xy_trackbar_scale']

print(f"rad_cam_offset: {rad_cam_offset}")
print(f"scalemm2px: {scalemm2px}")
print(f"wait_ms: {wait_ms}")
print(f"slider_xoffset: {slider_xoffset}")
print(f"slider_yoffset: {slider_yoffset}")
print(f"xy_trackbar_scale: {xy_trackbar_scale}")

# timestamps
t_rad = radar_points[0]['timestamp']   # radar's timestamp, in ms
T_RAD_BEGIN = t_rad   # radar's starting timestamp, in ms
TS_OFFSET = t_rad // 100000 * 100000   # trim off timestamp digits (first n digits) for display.

# radar camera synchronization
rad_cam_offset = rad_cam_offset - rad_cam_offset % (100)  # make sure it's multiples of video frame interval
print(f"Radar is set to be ahead of video by {rad_cam_offset}ms.")
# t_vid = cap.get(cv2.CAP_PROP_POS_MSEC)   # video's timestamp, in ms

# radar points buffer
s1_pts = []
s2_pts = []


# video frame buffer
frame_prev = None
video_file = cv_config['Files']['video_file']
cap = cv2.VideoCapture(video_file)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Number of frames: {num_frames}")

# BGR colours for drawing points on frame (OpenCV) 
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
BLUE = (255, 0, 0)

def washout(color, factor=0.2):
    # create washed out color
    return (int(color[0] * factor), int(color[1] * factor), int(color[2] * factor))

#Trackbar configuration
def x_trackbar_callback(x):
    # updates global offsets by trackbar value 
    global slider_xoffset
    slider_xoffset = cv2.getTrackbarPos('x offset', 'Radar Visualization')

def y_trackbar_callback(x):
    # updates global offsets by trackbar value 
    global  slider_yoffset
    slider_yoffset = cv2.getTrackbarPos('y offset', 'Radar Visualization')

def scale_callback(x):
    # multiplies x and y by scale value from trackbar 
    global xy_trackbar_scale
    xy_trackbar_scale = cv2.getTrackbarPos('scale %', 'Radar Visualization') / 100

cv2.namedWindow('Radar Visualization')
cv2.createTrackbar('x offset', 'Radar Visualization', slider_xoffset, 2000, x_trackbar_callback)
cv2.createTrackbar('y offset', 'Radar Visualization', slider_yoffset, 2000, y_trackbar_callback)
cv2.createTrackbar('scale %', 'Radar Visualization', int(xy_trackbar_scale*100), 200, scale_callback) # *100 and /100 to account for floating point usuability to downscale

# draw gate at top left of window, with width and height of gate. Scale to match gate location with trackbar 
def draw_gate_topleft(): 
    # initially at top left corner
    start_x = 0 
    start_y = 0 
    # modify start_x and start_y based on trackbar values
    start_x, start_y = int(start_x * xy_trackbar_scale), int(start_y * xy_trackbar_scale)
    start_x += slider_xoffset
    start_y += slider_yoffset
    rect_start = (start_x, start_y)
    # end_x and end_y are calculated based on the width and height of the gate
    end_x = offsetx * 2 * scalemm2px
    end_y = offsety * 2 * scalemm2px
    # modify end_x and end_y based on trackbar values
    end_x, end_y = int(end_x * xy_trackbar_scale), int(end_y * xy_trackbar_scale)
    end_x += slider_xoffset
    end_y += slider_yoffset
    rect_end = (end_x, end_y)
    cv2.rectangle(frame, rect_start, rect_end, BLUE, 2)

def display_video_info():
    cv2.putText(frame, "Controls - 'q': quit  'p': pause", (width-175, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 150), 1)
    cv2.putText(frame, f"end timestamp: {t_end}ms", (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    text_str = f"Curr frame ts:{(t_rad-T_RAD_BEGIN)/1000:.3f}   Replay {1:.1f}x"
    cv2.putText(frame, text_str, (10, height-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    # find timestamps of sensors
    sensor1_timestamps = list(set([coord['timestamp'] for coord in s1_pts]))
    text_str = f"s_entry ts:"
    for i, timestamp in enumerate(sensor1_timestamps):
        text_str += f" s1[{i}]: {(timestamp-TS_OFFSET)/1000:.3f}"
    cv2.putText(frame, text_str, (10, height-100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    # find timestamps of sensors
    sensor2_timestamps = list(set([coord['timestamp'] for coord in s2_pts]))
    text_str = f"s_exit ts: "
    for i, timestamp in enumerate(sensor2_timestamps):
        text_str += f" s2[{i}]: {(timestamp-TS_OFFSET)/1000:.3f}"
    cv2.putText(frame, text_str, (10, height-80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    # Draw info text
    text_str = f"nPoints:  s1:{len(s1_pts):2d}, s2:{len(s2_pts):2d}"
    cv2.putText(frame, text_str, (10, height-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

s1 = StaticPoints(cnt_thres=1)
s2 = StaticPoints(cnt_thres=1)

all_increments = 0
# main loop
while True:
    # Account for radar camera synchronization
    incr = 1000/30   # increment, in ms
    ts_start = radar_data.timestamp[0] # initial timestamp of radar points at start of program
    while round(rad_cam_offset) > 0:
        all_increments += incr
        while radar_data.timestamp[0] < ts_start + all_increments:
            print(f"Point being removed at timestamp {radar_data.timestamp[0]}")
            getattr(radar_data, 'sensorid').pop(0)
            getattr(radar_data, 'x').pop(0)
            getattr(radar_data, 'y').pop(0)
            getattr(radar_data, 'z').pop(0)
            getattr(radar_data, 'timestamp').pop(0) # updates ts_start
        rad_cam_offset -= incr
        print(f"rad_cam_offset is now: {0 if rad_cam_offset < 1 else rad_cam_offset}")
        t_rad = radar_data.timestamp[0]   # update t_rad to the timestamp of the first point in the frame

    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    frame = cv2.resize(frame, (round(width), round(height)))   # reduce frame size
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)         
    # update height and width after rotation
    height, width = frame.shape[:2]

    if round(rad_cam_offset) < 0:
            # radar to wait for video
            rad_cam_offset += 1000 / 33
            print("radar to wait for video (rad_cam_offset < 0)") 
    else:
        # take points in current RADAR frame
        radar_frame = radar_data.take_next_frame(interval=1)
        t_end = t_rad + 33    # ending timestamp, in ms
        if not radar_frame.is_empty(target_sensor_id=1):
            s1.update(radar_frame.get_points_for_display(sensor_id=1))
        if not radar_frame.is_empty(target_sensor_id=2):
            s2.update(radar_frame.get_points_for_display(sensor_id=2))

        # set static points
        radar_frame.set_static_points(s1.get_static_points())
        radar_frame.set_static_points(s2.get_static_points())

        s1_points_for_display = radar_frame.get_points_for_display(sensor_id=1)
        s2_points_for_display = radar_frame.get_points_for_display(sensor_id=2)
        print(s1_points_for_display)
        if len(s1_points_for_display) >= 1:
            for i, coord in enumerate(s1_points_for_display):
                print(coord)
                x = int((coord[0] + offsetx) * scalemm2px)  
                y = int((-coord[1] + offsety) * scalemm2px)   # y axis is flipped 
            
            # xy modifications from trackbar controls
            x = int(x * xy_trackbar_scale)
            y = int(y * xy_trackbar_scale)
            x += slider_xoffset
            y += slider_yoffset 
            if coord[2] == 1:
                cv2.circle(frame, (x,y), 4, washout(GREEN), -1)
            else:
                cv2.circle(frame, (x,y), 4, GREEN, -1)
            
        if len(s2_points_for_display) >= 1:
            for i, coord in enumerate(s2_points_for_display):
                print(coord)
                x = int((coord[0] + offsetx) * scalemm2px)   
                y = int((-coord[1] + offsety) * scalemm2px)   # y axis is flipped  
                # xy modifications from trackbar controls
                x = int(x * xy_trackbar_scale)
                y = int(y * xy_trackbar_scale)
                x += slider_xoffset
                y += slider_yoffset
                if coord[2] == 1:
                    cv2.circle(frame, (x,y), 4, washout(YELLOW), -1)
                else:
                    cv2.circle(frame, (x,y), 4, YELLOW, -1)
    draw_gate_topleft() # For 0, 0, 1 trackbar values
    display_video_info() 
    
    # after drawing points on frames, imshow the frames
    cv2.imshow('Radar Visualization', frame)

    # Key controls
    key = cv2.waitKey(wait_ms) & 0xFF
    if key == ord('q'): # quit program if 'q' is pressed 
        break
    elif key == ord('p'): # pause/unpause program if 'p' is pressed
        cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()

def yaml_update():
    while True:
        choice = input("Update final trackbar values in yaml? (y/n): ").lower()
        if choice == 'y':
            cv_config['TrackbarDefaults']['slider_xoffset'] = slider_xoffset
            cv_config['TrackbarDefaults']['slider_yoffset'] = slider_yoffset
            cv_config['TrackbarDefaults']['xy_trackbar_scale'] = xy_trackbar_scale
            with open('cv-config.yaml', 'w') as file:
                yaml.dump(cv_config, file)
            print("Trackbar values updated in yaml.")
            break
        elif choice == 'n':
            print("Trackbar values not updated in yaml.")
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

# yaml_update()