import os
import cv2
import json
import yaml
from radar_points import RadarData, StaticPoints
from preprocess import load_data_sensorhost, rot_mtx_entry, rot_mtx_exit
from radar_clustering import *

# ------------------ DATA PREPROCESS ------------------ #
# load configuration
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# check if video file exists
if not os.path.isfile(config["Files"]["video_file"]):
    raise FileNotFoundError(f"Video file does not exist.")

# load json data
radar_data_file = config["Files"]["radar_data_file"]
with open(radar_data_file) as json_file:
    data = json.load(json_file)

# use sensorhost format
radar_data = load_data_sensorhost(data)  # Original coordinates
print(f"Radar data loaded.\n{radar_data}\n")

# Apply transformation
alpha = config["SensorAngles"]["alpha"]
beta = config["SensorAngles"]["beta"]
# distance of sensor from gate centre, positive in mm
offsetx = config["SensorOffsets"]["offsetx"]
offsety = config["SensorOffsets"]["offsety"]
offsetz = config["SensorOffsets"]["offsetz"]
print(f"{alpha = }")
print(f"{beta = }")
print(f"{offsetx = }")
print(f"{offsety = }")
print(f"{offsetz = }")
s1_rotz, s1_rotx = rot_mtx_entry(alpha, beta)
s2_rotz, s2_rotx = rot_mtx_exit(alpha, beta)

radar_data.transform_coord(
    s1_rotz, s1_rotx, s2_rotz, s2_rotx, offsetx, offsety, offsetz
)
print(f"Radar data transformed.\n{radar_data}\n")

# ------------------ VISUALIZATION PARAMS ------------------ #
rad_cam_offset = config["rad_cam_offset"]
scalemm2px = config["scalemm2px"]
wait_ms = config["wait_ms"]
slider_xoffset = config["TrackbarDefaults"]["slider_xoffset"]
slider_yoffset = config["TrackbarDefaults"]["slider_yoffset"]
xy_trackbar_scale = config["TrackbarDefaults"]["xy_trackbar_scale"]

print(f"{rad_cam_offset = }")
print(f"{scalemm2px = }")
print(f"{wait_ms = }")
print(f"{slider_xoffset = }")
print(f"{slider_yoffset = }")
print(f"{xy_trackbar_scale = }")

# ------------------ CV2 SUPPORT FUNCTIONS ------------------ #

# BGR colours for drawing points on frame (OpenCV)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
BLUE = (255, 0, 0)
RED = (0, 0, 255)


def washout(color, factor=0.2):
    # create washed out color
    return (int(color[0] * factor), int(color[1] * factor), int(color[2] * factor))


def x_trackbar_callback(*args):
    # updates global offsets by trackbar value
    global slider_xoffset
    slider_xoffset = cv2.getTrackbarPos("x offset", "Radar Visualization")


def y_trackbar_callback(*args):
    # updates global offsets by trackbar value
    global slider_yoffset
    slider_yoffset = cv2.getTrackbarPos("y offset", "Radar Visualization")


def scale_callback(*args):
    # multiplies x and y by scale value from trackbar
    global xy_trackbar_scale
    xy_trackbar_scale = cv2.getTrackbarPos("scale %", "Radar Visualization") / 100


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


def draw_circle(rect_start, rect_end, coord, color):
    # draw coord as circle on frame, if coord is within gate
    if coord[0] < rect_start[0] or coord[0] > rect_end[0]:
        return
    if coord[1] < rect_start[1] or coord[1] > rect_end[1]:
        return
    cv2.circle(frame, (coord[0], coord[1]), 4, color, -1)


def draw_radar_points(points, sensor_id):
    remove_noise = config["remove_noise"]
    rect_start, rect_end = draw_gate_topleft()
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
            if remove_noise:
                draw_circle(rect_start, rect_end, (x, y), washout(color))
                # pass
            else:
                cv2.circle(frame, (x, y), 4, washout(color), -1)
        else:
            if remove_noise:
                draw_circle(rect_start, rect_end, (x, y), color)
                # pass
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


def display_video_info(radar_frame: RadarData, width, height):
    """Display video info on frame. width and height are the dimensions of the window."""
    # TODO: this part is broken. Need to revise.
    s1_pts = []
    s2_pts = []
    T_RAD_BEGIN = 0
    TS_OFFSET = 2.3

    cv2.putText(frame, f"end timestamp: t_end ms", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    text_str = f"Curr frame ts:{(t_rad - T_RAD_BEGIN) / 1000:.3f}   Replay {1:.1f}x"
    cv2.putText(frame, text_str, (10, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    # find timestamps of sensors
    sensor1_timestamps = list(set([coord["timestamp"] for coord in s1_pts]))
    text_str = f"s_entry ts:"
    for i, timestamp in enumerate(sensor1_timestamps):
        text_str += f" s1[{i}]: {(timestamp - TS_OFFSET) / 1000:.3f}"
    cv2.putText(frame, text_str, (10, height - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    # find timestamps of sensors
    sensor2_timestamps = list(set([coord["timestamp"] for coord in s2_pts]))
    text_str = f"s_exit ts: "
    for i, timestamp in enumerate(sensor2_timestamps):
        text_str += f" s2[{i}]: {(timestamp - TS_OFFSET) / 1000:.3f}"
    cv2.putText(frame, text_str, (10, height - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    # Draw info text
    text_str = f"nPoints:  s1:{len(s1_pts):2d}, s2:{len(s2_pts):2d}"
    cv2.putText(frame, text_str, (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)


# ------------------ VISUALIZATION ------------------ #

# video frame buffer
video_file = config["Files"]["video_file"]
cap = cv2.VideoCapture(video_file)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Number of frames: {num_frames}")

cv2.namedWindow("Radar Visualization")
cv2.createTrackbar(
    "x offset", "Radar Visualization", slider_xoffset, 600, x_trackbar_callback
)
cv2.createTrackbar(
    "y offset", "Radar Visualization", slider_yoffset, 600, y_trackbar_callback
)
cv2.createTrackbar(
    "scale %", "Radar Visualization", int(xy_trackbar_scale * 100), 200, scale_callback
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
    # Account for radar camera synchronization
    ret, frame = cap.read()
    if not ret:
        break
    height, width = frame.shape[:2]
    frame = cv2.resize(frame, (round(width), round(height)))  # reduce frame size
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    height, width = frame.shape[:2]

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

    # retain previous frame if no new points
    if len(s1_display_points) == 0:
        s1_display_points = s1_display_points_prev
    else:
        s1_display_points_prev = s1_display_points
    if len(s2_display_points) == 0:
        s2_display_points = s2_display_points_prev
    else:
        s2_display_points_prev = s2_display_points

    s1_s2_combined = [(*values[:-1],) for values in s1_display_points + s2_display_points]

    # get non-static points into a single list & cluster
    if not radar_frame.is_empty(target_sensor_id=1) or not radar_frame.is_empty(target_sensor_id=2):
        # s1_s2_combined = radar_frame.points_for_clustering() # needs updating
        if len(s1_s2_combined) > 1:
            processor = ClusterProcessor(eps=250, min_samples=4)  # default: eps=400, min_samples=5 --> eps is in mm
            centroids, cluster_point_cloud = processor.cluster_points(s1_s2_combined)  # get the centroids of each
            # cluster and their associated point cloud
            draw_clustered_points(centroids)  # may not be in the abs center of bbox --> "center of mass", not area
            # centroid btw.
            draw_clustered_points(cluster_point_cloud, color=BLUE)  # highlight the points that belong to the detected
            # obj
            for i in enumerate(centroids):
                x1, y1, x2, y2 = cluster_bbox(cluster_point_cloud, i[0])
                object_size, object_height = obj_height(cluster_point_cloud, i[0])
                # display bboxes --> convert from mm to pxl pls :)
                print(str(object_height) + " mm")

    # draw points on frame
    if len(s1_display_points) >= 1:
        draw_radar_points(s1_display_points, sensor_id=1)
    if len(s2_display_points) >= 1:
        draw_radar_points(s2_display_points, sensor_id=2)

    draw_gate_topleft()
    display_video_info(radar_frame, width, height)

    # after drawing points on frames, imshow the frames
    cv2.imshow("Radar Visualization", frame)

    # Key controls
    key = cv2.waitKey(wait_ms) & 0xFF
    if key == ord("q"):  # quit program if 'q' is pressed
        break
    elif key == ord("p"):  # pause/unpause program if 'p' is pressed
        cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()


# ------------------ SAVE CONFIG ------------------ #
def yaml_update():
    while True:
        choice = input("Update final trackbar values in yaml? (y/n): ").lower()
        if choice == "y":
            config["TrackbarDefaults"]["slider_xoffset"] = slider_xoffset
            config["TrackbarDefaults"]["slider_yoffset"] = slider_yoffset
            config["TrackbarDefaults"]["xy_trackbar_scale"] = xy_trackbar_scale
            with open("config.yaml", "w") as file:
                yaml.dump(config, file)
            print("Trackbar values updated in yaml.")
            break
        elif choice == "n":
            print("Trackbar values not updated in yaml.")
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

# yaml_update()
