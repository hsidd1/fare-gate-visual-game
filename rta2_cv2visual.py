"""Main Visualization Program for RTA2 Project. 

Flags: -v, --video_mode: process video file (default)
       -f, --frame_mode: process frames in directory
"""
import cv2
import json
import os
import time
from typing import Tuple
import yaml
import sys
from radar_points import RadarData, StaticPoints
from preprocess import *
from radar_clustering import *
# -------------- SET VISUALIZATION MODE --------------- #

# mode = "frame_mode"  # process live image frames
# mode = "video_mode"  # process video file
if len(sys.argv) > 1:
    if sys.argv[1] == "-v" or sys.argv[1] == "--video_mode":
        mode = "video_mode"
    elif sys.argv[1] == "-f" or sys.argv[1] == "--frame_mode":
        mode = "frame_mode"
    else:
        raise ValueError(f"Invalid argument: {sys.argv[1]}")
else:
    mode = "video_mode"

# ------------------ DATA PREPROCESS ------------------ #
# load configuration
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

if mode == "video_mode":
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

if mode == "frame_mode":
    if not os.path.isfile(config["Files"]["frames_radar_data"]):
        raise FileNotFoundError(f"Frames directory does not exist.")
    
    # load json data
    radar_data_file = config["Files"]["frames_radar_data"]
    with open(radar_data_file) as json_file:
        data = json.load(json_file)

    # load based on format with tlv
    # radar_data = load_data_tlv(data)
    radar_data = load_data_mqtt(data)
    print(f"Radar data loaded.\n{radar_data}\n")

TOTAL_DATA_S = (radar_data.ts[-1] - radar_data.ts[0])/1000 # total seconds of data, before removing points

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
if mode == "video_mode":
    rad_cam_offset = config["VideoModeDefaults"]["rad_cam_offset"]
    scalemm2px = config["VideoModeDefaults"]["scalemm2px"]
    wait_ms = config["VideoModeDefaults"]["wait_ms"]
    slider_xoffset = config["VideoModeDefaults"]["TrackbarDefaults"]["slider_xoffset"]
    slider_yoffset = config["VideoModeDefaults"]["TrackbarDefaults"]["slider_yoffset"]
    xy_trackbar_scale = config["VideoModeDefaults"]["TrackbarDefaults"]["xy_trackbar_scale"]
    playback_fps = config["VideoModeDefaults"]["playback_fps"]

elif mode == "frame_mode":
    rad_cam_offset = config["FrameModeDefaults"]["rad_cam_offset"]
    scalemm2px = config["FrameModeDefaults"]["scalemm2px"]
    wait_ms = config["FrameModeDefaults"]["wait_ms"]
    slider_xoffset = config["FrameModeDefaults"]["TrackbarDefaults"]["slider_xoffset"]
    slider_yoffset = config["FrameModeDefaults"]["TrackbarDefaults"]["slider_yoffset"]
    xy_trackbar_scale = config["FrameModeDefaults"]["TrackbarDefaults"]["xy_trackbar_scale"]
    playback_fps = config["FrameModeDefaults"]["playback_fps"]

print(f"{rad_cam_offset = }")
print(f"{scalemm2px = }")
print(f"{wait_ms = }")
print(f"{slider_xoffset = }")
print(f"{slider_yoffset = }")
print(f"{xy_trackbar_scale = }")
print(f"{playback_fps = }")

# ------------------ CV2 SUPPORT FUNCTIONS ------------------ #

# BGR colours for drawing points on frame (OpenCV)
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
    slider_xoffset = cv2.getTrackbarPos("x offset", "Radar Visualization")


def y_trackbar_callback(*args):
    # updates global offsets by trackbar value
    global slider_yoffset
    slider_yoffset = cv2.getTrackbarPos("y offset", "Radar Visualization")


def scale_callback(*args):
    # multiplies x and y by scale value from trackbar
    global xy_trackbar_scale
    xy_trackbar_scale = cv2.getTrackbarPos("scale %", "Radar Visualization") / 100


def draw_gate_topleft() -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """draw gate at top left of window, with width and height of gate.
    Scale to match gate location with trackbar 
    Returns valid display region start and end coordinates."""
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
        x = int(x * xy_trackbar_scale) + slider_xoffset
        y = int(y * xy_trackbar_scale) + slider_yoffset
        # skip if point is outside gate area
        if x < rect_start[0] or x > rect_end[0]:
            continue
        if y < rect_start[1] or y > rect_end[1]:
            continue
        points_in_gate.append(coord)
    return points_in_gate


def draw_radar_points(points, sensor_id) -> None:
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
        tlv_type = coord[4]
        # xy modifications from trackbar controls
        x = int(x * xy_trackbar_scale) + slider_xoffset
        y = int(y * xy_trackbar_scale) + slider_yoffset
        
        if tlv_type != 0: # if tlv type is defined
            if tlv_type == 1020:
                cv2.circle(frame, (x, y), 4, washout(color), -1)
            elif tlv_type == 1010:
                cv2.circle(frame, (x, y), 4, color, -1)
        if mode == "video_mode":
            if static:
                cv2.circle(frame, (x, y), 4, washout(color), -1)
            else:
                cv2.circle(frame, (x, y), 4, color, -1)


def draw_clustered_points(processed_centroids, color=RED) -> None:
    for cluster in processed_centroids:
        x = int((int(cluster['x'] + offsetx) * scalemm2px))
        y = int((int(-cluster['y'] + offsety) * scalemm2px))  # y axis is flipped
        # z = int(coord[2] * scalemm2px)  # z is not used
        # static = coord[3]

        # xy modifications from trackbar controls
        x = int(x * xy_trackbar_scale) + slider_xoffset
        y = int(y * xy_trackbar_scale) + slider_yoffset
        cv2.circle(frame, (x, y), 10, color, -1)


def draw_bbox(centroids, cluster_point_cloud) -> None:
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

def display_frame_info(radar_frame: RadarData, width, height) -> None:
    """Display video info on frame. width and height are the dimensions of the window."""
    # Time remaining
    cv2.putText(frame, 
                f"{0 if not radar_data.ts else (radar_data.ts[-1] - radar_data.ts[0])/1000:.2f} s remaining", 
                (10, height - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
                )
    # Number of points in frame
    cv2.putText(frame, 
                f"nPoints (frame): {len(radar_frame.x)}", 
                (10, height - 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
                )
    # Number of points in gate
    cv2.putText(
            frame, 
            f"Points in gate -- s1:{len(s1_display_points)} s2: {len(s2_display_points)}", 
            (10, height - 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
            )
    # Video config info, time elapsed, total time of data
    cv2.putText(
        frame, 
        f"Replay 1.0x, {playback_fps} fps Time Elapsed (s): {radar_data._RadarData__time_elapsed/1000:.2f} / {TOTAL_DATA_S:.2f}", 
        (10, height - 100), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
        )
    # Legend: green: s1, yellow: s2, orange: bbox, washed: static. With colour coded text, top left
    cv2.putText(
        frame, 
        "Legend: ", 
        (0, 10), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
        )
    cv2.putText(
        frame,
        "s1",
        (0, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2
        ) 
    cv2.putText(
        frame,
        "s2",
        (0, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 2
        )
    cv2.putText(
        frame,
        "Bounding box",
        (0, 70),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, ORANGE, 2
        )
    cv2.putText(
        frame,
        "Static",
        (0, 90),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, washout(GREEN), 2
        )
        
def display_control_info() -> None:
    cv2.putText(
        frame, 
        "Controls - 'q': quit  'p': pause", 
        (width-175, 20), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.35, (0, 0, 150), 1
        )
    cv2.putText(
        frame,
        "scale/offset gate region with trackbar",
        (width-217, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35, (0, 0, 150), 1
        )

# ------------------ VISUALIZATION ------------------ #

# video frame buffer
if mode == "video_mode":
    video_file = config["Files"]["video_file"]
    cap = cv2.VideoCapture(video_file)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Number of frames: {num_frames}")

# create window and trackbars
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
incr = 1000 / playback_fps  # frame ts increment, in ms

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

curr_frame = 0
frame_files = os.listdir("data/frames")

# Prepare for main loop: skip video frames, if video is ahead
if round(rad_cam_offset) < 0:
    print("rad_cam_offset is set negative, waiting radar points while playing video.")

    if mode == "video_mode":
            while round(rad_cam_offset) < 0:
                rad_cam_offset += incr
                ret, frame = cap.read()
                if not ret:
                    break

    elif mode == "frame_mode":
        frame_timestamps = [int(ts[:-4]) for ts in frame_files]
        target_timestamp = frame_timestamps[0] + rad_cam_offset
        # find the file name (timestamp) closest to the target timestamp
        # closest_frame = min(frame_timestamps, key=lambda x: abs(x - target_timestamp))
        min_difference = float("inf")
        for frame_ts in frame_timestamps:
            difference = abs(frame_ts - target_timestamp)
            if difference < min_difference:
                min_difference = difference
                closest_frame = frame_ts
            else:
                # sorted so we can break early
                break
        rad_cam_offset = 0
        curr_frame = closest_frame

# main loop
while True:
    # get frames based on mode configuration
    if mode == "video_mode":
        ret, frame = cap.read()
        if not ret:
            break
    elif mode == "frame_mode":
        if curr_frame < len(frame_files):
            frame = cv2.imread(f"data/frames/{frame_files[curr_frame]}")
            curr_frame += 1
            # account for interval between frame timestamps relative to radar data timestamp intervals
            time.sleep(incr/1000)
        else:
            break # end of frames
    height, width = frame.shape[:2]
    frame = cv2.resize(frame, (round(width), round(height)))  # reduce frame size
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    height, width = frame.shape[:2]
    
    # draw gate area and get gate area coordinates
    gate_tl, gate_br = draw_gate_topleft()

    # take points in current RADAR frame
    if mode == "video_mode":
        radar_frame = radar_data.take_next_frame(interval=incr)
    elif mode == "frame_mode":
        radar_frame = radar_data.take_next_frame(interval=incr, isTLVframe=True)
    print(f"radar_frame: {radar_frame}")

    # update static points, prepare for display
    s1_display_points = []
    s2_display_points = []
    if not radar_frame.is_empty(target_sensor_id=1):
        print("s1 non-empty")
        s1_static.update(radar_frame.get_xyz_coord(sensor_id=1))
        radar_frame.set_static_points(s1_static.get_static_points())
        s1_display_points = radar_frame.get_points_for_display(sensor_id=1)

    if not radar_frame.is_empty(target_sensor_id=2):
        print("s2 non-empty")
        s2_static.update(radar_frame.get_xyz_coord(sensor_id=2))
        radar_frame.set_static_points(s2_static.get_static_points())
        s2_display_points = radar_frame.get_points_for_display(sensor_id=2)

    print(f"{s1_display_points = }")
    print(f"{s2_display_points = }")

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

    display_frame_info(radar_frame, width, height)
    display_control_info()

    # after drawing points on frames, imshow the frames
    cv2.imshow("Radar Visualization", frame)

    # Key controls
    key = cv2.waitKey(wait_ms) & 0xFF
    if key == ord("q"):  # quit program if 'q' is pressed
        break
    elif key == ord("p"):  # pause/unpause program if 'p' is pressed
        cv2.waitKey(0)

if mode == "video_mode":
    cap.release()
cv2.destroyAllWindows()


# ------------------ SAVE CONFIG ------------------ #
def yaml_update():
    while True:
        choice = input("Update final trackbar values in yaml? (y/n): ").lower()
        if choice == "y":
            if mode == "video_mode":
                config["VideoModeDefaults"]["TrackbarDefaults"]["slider_xoffset"] = slider_xoffset
                config["VideoModeDefaults"]["TrackbarDefaults"]["slider_yoffset"] = slider_yoffset
                config["VideoModeDefaults"]["TrackbarDefaults"]["xy_trackbar_scale"] = xy_trackbar_scale
            elif mode == "frame_mode":
                config["FrameModeDefaults"]["TrackbarDefaults"]["slider_xoffset"] = slider_xoffset
                config["FrameModeDefaults"]["TrackbarDefaults"]["slider_yoffset"] = slider_yoffset
                config["FrameModeDefaults"]["TrackbarDefaults"]["xy_trackbar_scale"] = xy_trackbar_scale

            with open("config.yaml", "w") as file:
                yaml.dump(config, file)
            print("Trackbar values updated in yaml.")
            break
        elif choice == "n":
            print("Trackbar values not updated in yaml.")
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

yaml_update()
