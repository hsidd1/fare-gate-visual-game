import cv2
import yaml 
import json
#import os
from radar_points import  StaticPoints
from preprocess import load_data_sensorhost, rot_mtx_entry, rot_mtx_exit
from radar_clustering import *
import datetime

# load config
#TODO: add live config param 
with open("config.yaml", 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# load data
radar_data_file = config["Files"]["radar_data_file"]
with open(radar_data_file) as json_file:
    data = json.load(json_file)

#TODO: update for live data
radar_data = load_data_sensorhost(data)  # Original coordinates

#TODO: update for live config (likely stays unchanged)
alpha = config["SensorAngles"]["alpha"]
beta = config["SensorAngles"]["beta"]
# distance of sensor from gate centre, positive in mm
offsetx = config["SensorOffsets"]["offsetx"]
offsety = config["SensorOffsets"]["offsety"]
offsetz = config["SensorOffsets"]["offsetz"]
s1_rotz, s1_rotx = rot_mtx_entry(alpha, beta)
s2_rotz, s2_rotx = rot_mtx_exit(alpha, beta)

radar_data.transform_coord(
    s1_rotz, s1_rotx, s2_rotz, s2_rotx, offsetx, offsety, offsetz
)
print(f"Radar data transformed.\n{radar_data}\n")
rad_cam_offset = config["rad_cam_offset"]
scalemm2px = config["scalemm2px"]
wait_ms = config["wait_ms"]
slider_xoffset = config["TrackbarDefaults"]["slider_xoffset"]
slider_yoffset = config["TrackbarDefaults"]["slider_yoffset"]
xy_trackbar_scale = config["TrackbarDefaults"]["xy_trackbar_scale"]

GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
ORANGE = (0, 165, 255)

def washout(color, factor=0.2):
    # create washed out color
    return (int(color[0] * factor), int(color[1] * factor), int(color[2] * factor))

def x_trackbar_callback(*args):
    # updates global offsets by trackbar value
    global slider_xoffset
    slider_xoffset = cv2.getTrackbarPos("x offset", "Live Rad-cam synchronization")


def y_trackbar_callback(*args):
    # updates global offsets by trackbar value
    global slider_yoffset
    slider_yoffset = cv2.getTrackbarPos("y offset", "Live Rad-cam synchronization")


def scale_callback(*args):
    # multiplies x and y by scale value from trackbar
    global xy_trackbar_scale
    xy_trackbar_scale = cv2.getTrackbarPos("scale %", "Live Rad-cam synchronization") / 100


# draw gate at top left of window, with width and height of gate.
# Scale to match gate location with trackbar - returns valid display region
def draw_gate_topleft():
    # initial coords at top left corner (0,0)
    rect_start = (
        (slider_xoffset),
        (slider_yoffset)
    )
    # rect end initial coords are based on the physical width and height of the gate
    rect_end = (
        (int(offsetx * 2 * scalemm2px * xy_trackbar_scale) + slider_xoffset),
        (int(offsety * 2 * scalemm2px * xy_trackbar_scale) + slider_yoffset)
    )
    cv2.rectangle(frame, rect_start, rect_end, BLUE, 2)
    return rect_start, rect_end

def remove_points_outside_gate(points, rect_start, rect_end) -> list:
    """Remove points that are outside the gate area. 
    Returns a list of points that are inside the gate area."""
    points_in_gate = []
    for coord in points:
        x = int((coord[0] + offsetx) * scalemm2px)
        y = int((-coord[1] + offsety) * scalemm2px)
        x = int(x * xy_trackbar_scale)
        y = int(y * xy_trackbar_scale)
        x += slider_xoffset
        y += slider_yoffset
        if x < rect_start[0] or x > rect_end[0]:
            continue
        if y < rect_start[1] or y > rect_end[1]:
            continue
        points_in_gate.append(coord)
    return points_in_gate


def draw_radar_points(points, sensor_id):
    if sensor_id == 1:
        color = GREEN
    elif sensor_id == 2:
        color = YELLOW
    else:
        raise
    for coord in points:
        x = int((coord[0] + offsetx) * scalemm2px)
        y = int((-coord[1] + offsety) * scalemm2px)  # y axis is flipped
        z = int(coord[2] * scalemm2px)  # z is not used
        static = coord[3]

        # xy modifications from trackbar controls
        x = int(x * xy_trackbar_scale)
        y = int(y * xy_trackbar_scale)
        x += slider_xoffset
        y += slider_yoffset
        if static:
            cv2.circle(frame, (x, y), 4, washout(color), -1)
        else:
            cv2.circle(frame, (x, y), 4, color, -1)

def draw_clustered_points(processed_centroids, color=RED):
    for cluster in processed_centroids:
        x = int((int(cluster['x'] + offsetx) * scalemm2px))
        y = int((int(-cluster['y'] + offsety) * scalemm2px))  # y axis is flipped
        # z = int(coord[2] * scalemm2px)  # z is not used
        # static = coord[3]

        # xy modifications from trackbar controls
        x = int(x * xy_trackbar_scale)
        y = int(y * xy_trackbar_scale)
        x += slider_xoffset
        y += slider_yoffset
        cv2.circle(frame, (x, y), 10, color, -1)


def draw_bbox(centroids, cluster_point_cloud):
    for i in enumerate(centroids):
        x1, y1, x2, y2 = cluster_bbox(cluster_point_cloud, i[0])
        # convert mm to px 
        x1, y1, x2, y2 = int(x1 + offsetx) * scalemm2px, int(-y1 + offsety) * scalemm2px, int(x2 + offsetx) * scalemm2px, int(-y2 + offsety) * scalemm2px
        # modify based on trackbar
        x1, y1, x2, y2 = int(x1 * xy_trackbar_scale) + slider_xoffset, int(y1 * xy_trackbar_scale) + slider_yoffset, int(x2 * xy_trackbar_scale) + slider_xoffset, int(y2 * xy_trackbar_scale) + slider_yoffset
        object_size, object_height = obj_height(cluster_point_cloud, i[0])
        rect = cv2.rectangle(frame, (x1, y1), (x2, y2), ORANGE, 1)
        size, _ = cv2.getTextSize(f"{object_height:.1f} mm", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_width, text_height = size
        cv2.putText(rect, f"{object_height:.1f} mm", (x1, y1 - text_height - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ORANGE, 2)

cap = cv2.VideoCapture(0)

cv2.namedWindow("Live Rad-cam synchronization")
cv2.createTrackbar(
    "x offset", "Live Rad-cam synchronization", slider_xoffset, 600, x_trackbar_callback
)
cv2.createTrackbar(
    "y offset", "Live Rad-cam synchronization", slider_yoffset, 600, y_trackbar_callback
)
cv2.createTrackbar(
    "scale %", "Live Rad-cam synchronization", int(xy_trackbar_scale * 100), 200, scale_callback
)  # *100 and /100 to account for floating point usuability to downscale

# static points buffer
s1_static = StaticPoints(cnt_thres=5)
s2_static = StaticPoints(cnt_thres=5)

# previous frame buffer
s1_display_points_prev = []
s2_display_points_prev = []

# frame interval, set to the same as video
incr = 1000 / config["playback_fps"]  # frame ts increment, in ms

# radar camera synchronization
rad_cam_offset = rad_cam_offset - rad_cam_offset % (
    incr
)  # make sure it's multiples of video frame interval
print(f"Radar is set to be ahead of video by {rad_cam_offset:.1f}ms.")

# Prepare for main loop: remove radar points, if radar is ahead
all_increments = 0
ts_start = radar_data.ts[0]  # initial timestamp of radar points at start of program
if round(rad_cam_offset) > 0:
    print("rad_cam_offset is set positive, removing radar points while waiting for video.")
while round(rad_cam_offset) > 0:
    all_increments += incr
    while radar_data.ts[0] < ts_start + all_increments:
        # print(f"Point being removed at timestamp {radar_data.ts[0]}")
        radar_data.sid.pop(0)
        radar_data.x.pop(0)
        radar_data.y.pop(0)
        radar_data.z.pop(0)
        radar_data.ts.pop(0)
    rad_cam_offset -= incr
    # print(f"rad_cam_offset is now: {0 if rad_cam_offset < 1 else rad_cam_offset}")
    t_rad = radar_data.ts[0]  # timestamp of the first point in frame

# Prepare for main loop: skip video frames, if video is ahead
if round(rad_cam_offset) < 0:
    print("rad_cam_offset is set negative, waiting radar points while playing video.")
while round(rad_cam_offset) < 0:
    rad_cam_offset += incr
    ret, frame = cap.read()
    if not ret:
        break

# main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    print("Current time: ", datetime.datetime.now())
    key = cv2.waitKey(1000)
    height, width = frame.shape[:2]
    frame = cv2.resize(frame, (round(width), round(height)))  # reduce frame size
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    height, width = frame.shape[:2]

    # draw gate area and get gate area coordinates
    gate_tl, gate_br = draw_gate_topleft()

    # take points in current RADAR frame
    radar_frame = radar_data.take_next_frame(interval=incr)

    # update static points, prepare for display
    s1_display_points = []
    s2_display_points = []
    if not radar_frame.is_empty(target_sensor_id=1):
        s1_static.update(radar_frame.get_xyz_coord(sensor_id=1))
        radar_frame.set_static_points(s1_static.get_static_points())
        s1_display_points = radar_frame.get_points_for_display(sensor_id=1)

    if not radar_frame.is_empty(target_sensor_id=2):
        s2_static.update(radar_frame.get_xyz_coord(sensor_id=2))
        radar_frame.set_static_points(s2_static.get_static_points())
        s2_display_points = radar_frame.get_points_for_display(sensor_id=2)

    # remove points that are out of gate area, if configured
    if config["remove_noise"]:
        s1_display_points = remove_points_outside_gate(s1_display_points, gate_tl, gate_br)
        s2_display_points = remove_points_outside_gate(s2_display_points, gate_tl, gate_br)
        
    # retain previous frame if no new points
    if not s1_display_points:
        s1_display_points = s1_display_points_prev
    else:
        s1_display_points_prev = s1_display_points
    if not s2_display_points:
        s2_display_points = s2_display_points_prev
    else:
        s2_display_points_prev = s2_display_points

    # get all non-static points and cluster
    s1_s2_combined = [values[:-1] for values in s1_display_points + s2_display_points if values[-1] == 0]
    if len(s1_s2_combined) > 1:
        processor = ClusterProcessor(eps=250, min_samples=4)  # default: eps=400, min_samples=5 --> eps is in mm
        centroids, cluster_point_cloud = processor.cluster_points(s1_s2_combined)  # get the centroids of each
        # cluster and their associated point cloud
        draw_clustered_points(centroids)  # may not be in the abs center of bbox --> "center of mass", not area
        # centroid.
        draw_clustered_points(cluster_point_cloud, color=BLUE)  # highlight the points that belong to the detected
        # obj
        draw_bbox(centroids, cluster_point_cloud)  # draw the bounding box of each cluster

    # draw points on frame
    if s1_display_points:
        draw_radar_points(s1_display_points, sensor_id=1)
    if s2_display_points:
        draw_radar_points(s2_display_points, sensor_id=2)

    # after drawing points on frames, imshow the frames
    cv2.imshow("Live Rad-cam synchronization", frame)

    # Key controls
    key = cv2.waitKey(wait_ms) & 0xFF
    if key == ord("q"):  # quit program if 'q' is pressed
        break
    elif key == ord("p"):  # pause/unpause program if 'p' is pressed
        cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()