import cv2
import json
import numpy as np

# Sensor Angles
alpha = 47.5
beta = 17.5
offsetx = 325   # mm, along the width of isle
offsety = 595   # mm, along the gate isle
offsetz = 850     # mm height

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

# Opening JSON file
with open('Control_test1.json') as json_file:
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

print(radar_points[0:6])

                        # ------------------ VISUALIZATION ------------------ #
from point_cloud import StatisPoints

# timestamps
t_rad = radar_points[0]['timestamp']   # radar's timestamp, in ms
T_RAD_BEGIN = t_rad   # radar's starting timestamp, in ms
TS_OFFSET = t_rad // 100000 * 100000   # trim off timestamp digits (first n digits) for display.

# radar camera synchronization
rad_cam_offset = 2300  # ms    # positive if radar is ahead of video
rad_cam_offset = rad_cam_offset - rad_cam_offset % (100)  # make sure it's multiples of video frame interval
print(f"Radar is set to be ahead of video by {rad_cam_offset}ms.")
# t_vid = cap.get(cv2.CAP_PROP_POS_MSEC)   # video's timestamp, in ms

# radar points buffer
s1_pts = []
s2_pts = []
# static points
s1_stat = StatisPoints(cnt_thres=5)
s2_stat = StatisPoints(cnt_thres=5)
# points in previous frame
s1_pts_prev = []
s2_pts_prev = []
scale = 0.5 

# video frame buffer
frame_prev = None
cap = cv2.VideoCapture('Controlled_test.avi')
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Number of frames: {num_frames}")

wait_ms  = 1000//30   # wait time between frames, in ms

# BGR colours for drawing points on frame (OpenCV) 
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
BLUE = (255, 0, 0)
def washout(color, factor=0.2):
    # create washed out color
    return (int(color[0] * factor), int(color[1] * factor), int(color[2] * factor))
# values that modify the x and y coordinates of the radar points
slider_xoffset = 120 # mm
slider_yoffset = 115 # mm
xy_trackbar_scale = 0.5 # scale factor for x and y
# initial values of above 
initial_x_offset = slider_xoffset
initial_y_offset = slider_yoffset
initial_scale = xy_trackbar_scale
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

def draw_gate_area_default():
    # draw gate outline 
    # calculated positioning based on outer bars of gate - needs slight adjustments
    start_x = width//2 - width//6 - initial_x_offset + 10 # removing default offset from default start as its config for default offset
    start_y = height//4 - initial_y_offset  -25
    start_x += slider_xoffset
    start_y += slider_yoffset
    #start_x, start_y = int(start_x * initial_scale), int(start_y * initial_scale)
    #start_x, start_y = int(start_x * xy_trackbar_scale), int(start_y * xy_trackbar_scale)
    rect_start = (start_x, start_y)
    end_x = width//2 + width//6 - initial_x_offset + 8
    end_y = height//4 + height//2 - initial_y_offset - 30
    end_x += slider_xoffset
    end_y += slider_yoffset
    #end_x, end_y = int(end_x * initial_scale), int(end_y * initial_scale)
    #end_x, end_y = int(end_x * xy_trackbar_scale), int(end_y * xy_trackbar_scale)
    rect_end = (end_x, end_y)
    cv2.rectangle(frame, rect_start, rect_end, BLUE, 2)

# main loop
while True:
    # Account for radar camera synchronization
    incr = 1000/30   # increment, in ms
    all_increments = 0
    ts_start = radar_points[0]['timestamp'] # initial timestamp of radar points at start of program
    while round(rad_cam_offset) > 0:
        all_increments += incr
        while radar_points[0]['timestamp'] < ts_start + all_increments:
            print(f"Point being removed at timestamp {radar_points[0]['timestamp']}")
            radar_points.pop(0) # remove points that are before the current frame - should update ts_start
        rad_cam_offset -= incr
        print(f"rad_cam_offset is now: {0 if rad_cam_offset < 1 else rad_cam_offset}")
    t_rad = radar_points[0]['timestamp']   # update t_rad to the timestamp of the first point in the frame
    ret, frame = cap.read()
    if not ret:
        break
    height, width = frame.shape[:2]
    frame = cv2.resize(frame, (round(width), round(height)))   # reduce frame size
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)         
    # update height and width after rotation
    height = frame.shape[0]
    width = frame.shape[1]
   

    # take points in current RADAR frame
    t_end = t_rad + 33    # ending timestamp, in ms
    s1_pts = []
    s2_pts = []
    while len(radar_points) > 0:
        if radar_points[0]['timestamp'] > t_end:
            break
        if radar_points[0]['sensorId'] == 1:
            s1_pts.append(radar_points[0])
        elif radar_points[0]['sensorId'] == 2:
            s2_pts.append(radar_points[0])
        else:
            print("Error: sensorId not 1 or 2")
        radar_points.pop(0)   # remove the point after taking it out.
    t_rad = t_end
    # if current frame contain points, use them. Otherwise, keep points from previous frame
    if len(s1_pts) == 0:
        s1_pts = s1_pts_prev
    else:
        s1_pts_prev = s1_pts
    if len(s2_pts) == 0:
        s2_pts = s2_pts_prev
    else:
        s2_pts_prev = s2_pts

    # draw radar points, render static points as washed out color
    if len(s1_pts) >= 1:
        s1_stat.update([(coord['x'], coord['y']) for coord in s1_pts])
        for coord in s1_pts:
            x = int((coord['x'] + offsetx) * scale)  
            y = int((-coord['y'] + offsety) * scale)   # y axis is flipped 
            
            # xy modifications from trackbar controls
            x = int(x * xy_trackbar_scale)
            y = int(y * xy_trackbar_scale)
            x += slider_xoffset
            y += slider_yoffset 
            if (coord['x'], coord['y']) in s1_stat.get_static_points():
                cv2.circle(frame, (x,y), 4, washout(GREEN), -1)
            else:
                cv2.circle(frame, (x,y), 4, GREEN, -1)
    if len(s2_pts) >= 1:
        s2_stat.update([(coord['x'], coord['y']) for coord in s2_pts])
        for coord in s2_pts:
            x = int((coord['x'] + offsetx) * scale)   
            y = int((-coord['y'] + offsety) * scale)   # y axis is flipped  
            # xy modifications from trackbar controls
            x = int(x * xy_trackbar_scale)
            y = int(y * xy_trackbar_scale)
            x += slider_xoffset
            y += slider_yoffset
            if (coord['x'], coord['y']) in s2_stat.get_static_points():
                cv2.circle(frame, (x,y), 4, washout(YELLOW), -1)
            else:
                cv2.circle(frame, (x,y), 4, YELLOW, -1)
    
    # draw gate outline
    draw_gate_area_default()

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