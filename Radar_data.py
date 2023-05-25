from Radar_frame import RadarFrame
class RadarData:
    def __init__(self, sensorId1, x1, y1, z1, timestamp1, sensorId2, x2, y2, z2, timestamp2) -> None:
        self.data = [{'sensorId': sensorId1, 'x': x1, 'y': y1, 'z': z1, 'timestamp': timestamp1},
                    {'sensorId': sensorId2, 'x': x2, 'y': y2, 'z': z2, 'timestamp': timestamp2}]
        
def get_data(self):
     return self.data

def take_next_frame(self, interval):
    # return RadarFrame object with data from next frame
    timestamp = self.data[0]['timestamp'] + interval
    # check if data is empty
    if len(self.data) == 0:
        status = False
    else:
        status = True

    return RadarFrame(self.data[0]['sensorId'], self.data[0]['x'], self.data[0]['y'], self.data[0]['z'], timestamp,
                      self.data[1]['sensorId'], self.data[1]['x'], self.data[1]['y'], self.data[1]['z'], timestamp,
                      status)

