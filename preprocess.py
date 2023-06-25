import numpy as np
from radar_points import RadarData

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

def load_data_sensorhost(data: dict) -> RadarData:
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
                # transformed_coords = np.matmul(s1_rotz, np.matmul(s1_rotx, raw_coords))
                # transformed_coords += np.array([[offsetx], [-offsety], [offsetz]])
                transformed_coords = raw_coords
            elif s['sensorId'] == 2:
                # transformed_coords = np.matmul(s2_rotz, np.matmul(s2_rotx, raw_coords))
                # transformed_coords += np.array([[-offsetx], [offsety], [offsetz]])
                transformed_coords = raw_coords
            else:
                print('Sensor ID not supported')
                raise

            s['x'] = float(transformed_coords[0])
            s['y'] = float(transformed_coords[1])
            s['z'] = float(transformed_coords[2])
            
            radar_points.append(s)
    return RadarData(radar_points)
