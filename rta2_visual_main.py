# from importlib.metadata import metadata
import cv2
import json
import numpy as np
# from operator import truediv
# from scipy.ndimage.interpolation import rotate
# import math
# import pandas as pd


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

# Elliot code
# for entry sensor
# this is fundamentally wrong, think about starting from origin, then rotate and translate to corners
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
print(data.keys())
print(data['sensors'])
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print(data['walkwayConfig'])

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

import pygame
from point_cloud import StatisPoints
from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    K_SPACE,
    K_p,
    KEYDOWN,
    QUIT,
)
BLACK = (0, 0, 0)
GRAY = (50, 50, 50)
SILVER = (192, 192, 192)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)

# PyGame init
pygame.init()
# create a screen
screen_size = (1200, 720)
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption('Radar Visualizer')

# Surface for radar sub-coordinate
scale = 0.5   # for scaling the radar area.
sub_size = (int(offsetx*2*scale), int(offsety*2*scale))
sub_surface = pygame.Surface(sub_size)

# Load video
cap = cv2.VideoCapture("Controlled_test.avi")
VIDEO_SCALE = 1.0   # display size / original video size

# text formatting
FONT_SIZE = 16
font = pygame.font.SysFont('Consolas', FONT_SIZE)   # monospace fonts
# font = pygame.font.Font(pygame.font.get_default_font(), FONT_SIZE)

# framerate setting
VIDEO_FPS = 30  # frames per second
GAME_FPS = 30  # frames per second, ideally should be multiples of video FPS
# variable replay speed, TBD
# RADAR_REPLAY_INTERVAL = 50  # ms
# REPLAY_SPEED = GAME_FPS * RADAR_REPLAY_INTERVAL / 1000  # 10fps, 100ms makes 1x speed
# print('Replay speed: ', REPLAY_SPEED)

# timestamps
t_rad = radar_points[0]['timestamp']   # radar's timestamp, in ms
T_RAD_BEGIN = t_rad   # radar's starting timestamp, in ms
TS_OFFSET = t_rad // 100000 * 100000   # trim off timestamp digits (first n digits) for display.

# radar camera synchronization
rad_cam_offset = 2300  # ms    # positive if radar is ahead of video
rad_cam_offset = rad_cam_offset - rad_cam_offset % (1000/VIDEO_FPS)  # make sure it's multiples of video frame interval
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

# video frame buffer
frame_prev = None

# replay control
running = True
paused = False

# radar plot backgroud
def gridmaker():
    width = int(sub_size[0])
    height = int(sub_size[1])
    for x in range(0, width, int(20 * scale)):
        pygame.draw.line(sub_surface, GRAY, (x, 1), (x, height), 1)
    for y in range(0, height, int(20 * scale)):
        pygame.draw.line(sub_surface, GRAY, (1, y), (width, y), 1)
    # draw the center line
    pygame.draw.line(sub_surface, SILVER, (width // 2, 1), (width // 2, height), 1)
    pygame.draw.line(sub_surface, SILVER, (1, height // 2), (width, height // 2), 1)
    # draw the border
    pygame.draw.rect(sub_surface, SILVER, (0, 0, width, height), 1)

def washout(color, factor=0.2):
    # create washed out color
    return (int(color[0] * factor), int(color[1] * factor), int(color[2] * factor))

# the main loop
while running:
    if len(radar_points) == 0:
        paused = True
    if not paused:
        screen.fill((0, 0, 0))
        sub_surface.fill((0, 0, 0))

        if round(rad_cam_offset) > 0:
            # skip video frames
            rad_cam_offset -= 1000 / GAME_FPS
            ret = False
            # draw text, top right
            text_str = f"Video being paused: {round(rad_cam_offset)/1000:.3f}ms"
            text = font.render(text_str, True, SILVER, None)
            text_rect = text.get_rect()
            text_rect.topleft = (600, FONT_SIZE)
            screen.blit(text, text_rect)
        else:
            # process current video frame
            ret, frame = cap.read()
        # process video frame
        if ret:
            height, width = frame.shape[:2]
            frame = cv2.resize(frame, (round(width*VIDEO_SCALE), round(height*VIDEO_SCALE)))   # reduce frame size
            frame = cv2.flip(frame, 0)   # flip frame horizontally
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)   # rotate frame by 90 degrees
            # display on screen, by converting to pygame surface
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_prev = frame.copy()
        else:
            frame = frame_prev
            ret = True   # artificially set ret to True, so that the last video frame will be displayed
        # display on screen
        if frame is not None and ret:
            frame_surf = pygame.surfarray.make_surface(frame)
            screen.blit(frame_surf, (500, 100))

            # draw text, below the video frame
            text_str = f"Video ts {cap.get(cv2.CAP_PROP_POS_MSEC)/1000:.3f}, Frame {cap.get(cv2.CAP_PROP_POS_FRAMES):.0f}"
            text = font.render(text_str, True, SILVER, None)
            text_rect = text.get_rect()
            text_rect.topleft = (500, 100 + height*VIDEO_SCALE + FONT_SIZE)
            screen.blit(text, text_rect)

        # draw radar plot
        gridmaker()
        if round(rad_cam_offset) < 0:
            # radar to wait for video
            rad_cam_offset += 1000 / GAME_FPS
            # draw text, on top of radar plot
            text_str = f"Radar being paused: {-round(rad_cam_offset)/1000:.3f}ms"
            text = font.render(text_str, True, SILVER, None)
            text_rect = text.get_rect()
            text_rect.topleft = (100, FONT_SIZE)
            screen.blit(text, text_rect)
        else:
            # take points in current RADAR frame
            t_end = t_rad + 1000/GAME_FPS    # ending timestamp, in ms
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
                x = int((coord['x'] + offsetx) * scale)  #
                y = int((-coord['y'] + offsety) * scale)   # y axis is flipped  
                if (coord['x'], coord['y']) in s1_stat.get_static_points():
                    pygame.draw.circle(sub_surface, washout(GREEN), (x, y), 4)
                else:
                    pygame.draw.circle(sub_surface, GREEN, (x, y), 4)
        if len(s2_pts) >= 1:
            s2_stat.update([(coord['x'], coord['y']) for coord in s2_pts])
            for coord in s2_pts:
                x = int((coord['x'] + offsetx) * scale)   # 
                y = int((-coord['y'] + offsety) * scale)   # y axis is flipped  
                if (coord['x'], coord['y']) in s2_stat.get_static_points():
                    pygame.draw.circle(sub_surface, washout(YELLOW), (x, y), 4)
                else:
                    pygame.draw.circle(sub_surface, YELLOW, (x, y), 4)
        # draw radar sub surface
        screen.blit(sub_surface, (50,50))

        # draw text, bottom left
        # Game ts according to radar
        text_str = f"Curr frame ts:{(t_rad-T_RAD_BEGIN)/1000:.3f}   Replay {1:.1f}x"
        text = font.render(text_str, True, SILVER, None)
        text_rect = text.get_rect()
        text_rect.topleft = (50, screen.get_height() - FONT_SIZE * 4)
        screen.blit(text, text_rect)
        # find timestamps of sensors
        sensor1_timestamps = list(set([coord['timestamp'] for coord in s1_pts]))
        text_str = f"s_entry ts:"
        for i, timestamp in enumerate(sensor1_timestamps):
            text_str += f" s1[{i}]: {(timestamp-TS_OFFSET)/1000:.3f}"
        text = font.render(text_str, True, SILVER, None)
        text_rect = text.get_rect()
        text_rect.topleft = (50, screen.get_height() - FONT_SIZE * 3)
        screen.blit(text, text_rect)
        # find timestamps of sensors
        sensor2_timestamps = list(set([coord['timestamp'] for coord in s2_pts]))
        text_str = f"s_exit ts: "
        for i, timestamp in enumerate(sensor2_timestamps):
            text_str += f" s2[{i}]: {(timestamp-TS_OFFSET)/1000:.3f}"
        text = font.render(text_str, True, SILVER, None)
        text_rect = text.get_rect()
        text_rect.topleft = (50, screen.get_height() - FONT_SIZE * 2)
        screen.blit(text, text_rect)
        # Draw info text
        text_str = f"nPoints:  s1:{len(s1_pts):2d}, s2:{len(s2_pts):2d}"
        text = font.render(text_str, True, SILVER, None)
        text_rect = text.get_rect()
        text_rect.topleft = (50, screen.get_height() - FONT_SIZE)
        screen.blit(text, text_rect)

        # draw text, right side
        """
        # Draw coordinate text, s1_pts
        text_str = 's1_pts:\n'
        text_str += '\n'.join([f"{coord['x']:.2f}, {coord['y']:.2f}" for coord in s1_pts])
        text_lines = text_str.splitlines()
        text_surfaces = []
        for line in text_lines:
            text_surfaces.append(font.render(line, True, GREEN, None))
        y = 50
        for surface in text_surfaces:
            screen.blit(surface, (400, y))
            y += surface.get_height() + 10  # add space between lines
        # Draw coordinate text, s2_pts
        text_str = 's2_pts:\n'
        text_str += '\n'.join([f"{coord['x']:.2f}, {coord['y']:.2f}" for coord in s2_pts])
        text_lines = text_str.splitlines()
        text_surfaces = []
        for line in text_lines:
            text_surfaces.append(font.render(line, True, YELLOW, None))
        y = 50
        for surface in text_surfaces:
            screen.blit(surface, (600, y))
            y += surface.get_height() + 10
        """

    pygame.display.flip()
    tick_result = pygame.time.Clock().tick(GAME_FPS)  # framerate
    # quit
    for event in pygame.event.get():
        # Did the user hit a key?
        if event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                running = False
            if event.key == K_p:
                paused = not paused
            if event.key == K_SPACE:
                paused = not paused
        elif event.type == QUIT:
            running = False

# Done! Time to quit.
cap.release()
pygame.quit()