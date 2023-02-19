# from importlib.metadata import metadata
import json
from operator import truediv
import pygame, clock
import numpy as np
from scipy.ndimage.interpolation import rotate
import math
import pandas as pd

data_dict = []

# Sensor Angles
alpha = 47.5
beta = 17.5

# Entry Sensor Rot Matrices
s1_rotz = np.asmatrix(([math.cos(math.radians(-alpha)), -math.sin(math.radians(-alpha)),0], [math.sin(math.radians(-alpha)),math.cos(math.radians(-alpha)), 0], [0,0,1]))
s1_rotx = np.asmatrix(([1,0,0], [0,math.cos(math.radians(beta)), -math.sin(math.radians(beta))], [0, math.sin(math.radians(beta)),math.cos(math.radians(beta))]))
print(s1_rotx)
# Exit Sensor Rot Matrices
s2_rotz = np.asmatrix(([math.cos(math.radians(alpha)), -math.sin(math.radians(alpha)),0], [math.sin(math.radians(alpha)),math.cos(math.radians(alpha)), 0], [0,0,1]))
s2_rotx = np.asmatrix(([1,0,0], [0,math.cos(math.radians(-(180+beta))), -math.sin(math.radians(-(180+beta)))], [0, math.sin(math.radians(-(180+beta))),math.cos(math.radians(-(180+beta)))]))

# Elliot code
# for entry sensor
s1_rotz = np.zeros((3,3))
s1_rotz[0,0] = np.cos(np.radians(alpha))
s1_rotz[0,1] = -np.sin(np.radians(alpha))
s1_rotz[1,0] = np.sin(np.radians(alpha))
s1_rotz[1,1] = np.cos(np.radians(alpha))
s1_rotz[2,2] = 1
print(f"s1_rotz =\n{s1_rotz}")
s1_rotx = np.zeros((3,3))
s1_rotx[0,0] = 1
s1_rotx[1,1] = np.cos(np.radians(beta))
s1_rotx[1,2] = -np.sin(np.radians(beta))
s1_rotx[2,1] = np.sin(np.radians(beta))
s1_rotx[2,2] = np.cos(np.radians(beta))
print(f"s1_rotx =\n{s1_rotx}")

# for exit sensor
theta_z = alpha + 180
s2_rotz = np.zeros((3,3))
s2_rotz[0,0] = np.cos(np.radians(theta_z))
s2_rotz[0,1] = -np.sin(np.radians(theta_z))
s2_rotz[1,0] = np.sin(np.radians(theta_z))
s2_rotz[1,1] = np.cos(np.radians(theta_z))
s2_rotz[2,2] = 1
print(f"s2_rotz = \n{s2_rotz}")

s2_rotx = np.zeros((3,3))
s2_rotx[0,0] = 1
s2_rotx[1,1] = np.cos(np.radians(beta))
s2_rotx[1,2] = -np.sin(np.radians(beta))
s2_rotx[2,1] = np.sin(np.radians(beta))
s2_rotx[2,2] = np.cos(np.radians(beta))
print(f"s2_rotx = \n{s2_rotx}")

def insertionSort(data_dict):
    # Traverse through 1 to len(data_dict)
    for i in range(1, len(data_dict)):

        key = data_dict[i]

        # Move elements of data_dict[0..i-1], that are
        # greater than key, to one position ahead
        # of their current position
        j = i - 1
        while j >= 0 and key['timestamp'] < data_dict[j]['timestamp']:
            data_dict[j + 1] = data_dict[j]
            j -= 1
        data_dict[j + 1] = key

# Opening JSON file
with open('Control_test1.json') as json_file:
    data = json.load(json_file)

# Print the data of dictionary
print(data.keys())
print(data['sensors'])
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print(data['walkwayConfig'])

# process data
for i in data['frames']:

    num_ob = i['sensorMessage']['metadata']['numOfDetectedObjects']

    # timestamp = ((i['sensorMessage']['metadata']['timeStamp']))
    timestamp = ((i['timestamp']))
    print('WORLD TIME' + str(timestamp))

    for j in range(num_ob):
        s = dict()
        s['sensorId'] = (i['sensorMessage']['object']['detectedPoints'][j]['sensorId'])
        s['x'] = i['sensorMessage']['object']['detectedPoints'][j]['x']
        s['y'] = i['sensorMessage']['object']['detectedPoints'][j]['y']
        s['z'] = i['sensorMessage']['object']['detectedPoints'][j]['z']  # Flattened so it should always return be 0
        s['timestamp'] = timestamp
        # print('TIMESTAMP' + str(timestamp))

        raw_coords = np.asmatrix(([s['x']], [s['y']], [s['z']]))
        # print(raw_coords)
        if s['sensorId'] == 2:
            transformed_coords = (np.matmul(s1_rotx, np.matmul(s1_rotz, raw_coords)))
            scale_x = -17.25
            scale_y = 28.75
            scale_z = 85

        else:
            transformed_coords = (np.matmul(s2_rotx, np.matmul(s2_rotz, raw_coords)))
            scale_x = 17.25
            scale_y = -28.75
            scale_z = 85

        s['x'] = transformed_coords[0] + scale_x
        s['y'] = transformed_coords[1] + scale_y
        s['z'] = transformed_coords[2] + scale_z    # Flattened so it should always return be 0

        # print('New Values')
        # print(s['x'])
        # print(s['y'])
        # print(s['z'])

        data_dict = data_dict + [s]

insertionSort(data_dict)
offset = 0.6
for i in data_dict:
    i['x'] = int((i['x'] + 325) * offset)
    i['y'] = int((i['y'] + (595) * offset))

print(data_dict[0:6])


df = pd.DataFrame.from_dict(data_dict)
df.to_csv('Transformed Datapoints.csv', index=False, header=True)

# visual code
pygame.init()
# create a screen:
offset = 0.6
screen = pygame.display.set_mode((int(650 * offset), int(1190 * offset)))
done = False
x_whole = int(650 * offset)
ywhole = int(1190 * offset)

# (red,green,blue)
c = (50, 50, 50)

# never ending loop now:
def gridmaker():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    for x in range(0, x_whole, int(10 * offset)):
        pygame.draw.line(screen, c, (x, 1), (x, ywhole), 1)

    for y in range(0, ywhole, int(10 * offset)):
        pygame.draw.line(screen, c, (1, y), (x_whole, y), 1)


t = data_dict[0]['timestamp']
cond = True
sensor1 = []
t1 = 0
t2 = 0
sensor2 = []
while True and cond:
    screen.fill((0, 0, 0))
    gridmaker()

    for i in range(len(data_dict)):

        if t < data_dict[i]['timestamp']:
            t2 = data_dict[i]['timestamp']

            break
        elif t == data_dict[i]['timestamp']:
            if data_dict[i]['sensorId'] == 1:
                if len(sensor1) >= 1 and sensor1[-1]['timestamp'] < t:
                    sensor1 = []
                sensor1 = sensor1 + [(data_dict[i])]
                t1 = t
            else:
                if len(sensor2) >= 1 and sensor2[-1]['timestamp'] < t:
                    sensor2 = []
                sensor2 = sensor2 + [(data_dict[i])]
                t2 = t
            # pygame.draw.circle(screen,(255,0,0),(data_dict[i]['x'],data_dict[i]['y']),4)

        if i == len(data_dict) - 1:
            cond = False

        if len(sensor1) >= 1:
            for i in sensor1:
                pygame.draw.circle(screen, (255, 0, 0), (i['x'], i['y']), 4)
        if len(sensor2) >= 1:

            for i in sensor2:
                pygame.draw.circle(screen, (255, 0, 0), (i['x'], i['y']), 4)

    difference = t2 - t
    t = t2
    clock = pygame.time.Clock()

    pygame.display.update()
    print(t)
    clock.tick(difference)
