from Radar_frame import RadarFrame
class RadarData:
    # instantiate a RadarData object with data from two sensors held in lists for each key
    def __init__(self, data):
        # data is a dictionary mapping keys to  lists
        # ex {'sensorId': [1,2], 'x': [1,2], 'y': [1,2], 'z': [1,2], 'timestamp': [1,2]}
        try:
            if isinstance(data, dict) and all(isinstance(val, list) for val in data.values()) and len(data) == 5:
                self.data = data
        except:
            raise TypeError("Error in RadarData instantiation: data must be a dictionary mapping keys to lists")

def get_data(self):
     return self.data

def take_next_frame(self, interval):
    raise NotImplementedError("take_next_frame not implemented yet")
    # return RadarFrame object with data from next frame
    timestamp = self.data[0]['timestamp'] + interval
    # check if data is empty to modify status
    #status = True if len(self.data) > 0 else False
    # change status later to be checked by static points class, keep unknown status for now 
    status = -1
    # needs to be constructed differently here 
    frame_data  = 0
    # build frame data 
    return RadarFrame(frame_data, status)
    return RadarFrame(self.data[0]['sensorId'], self.data[0]['x'], self.data[0]['y'], self.data[0]['z'], timestamp,
                      self.data[1]['sensorId'], self.data[1]['x'], self.data[1]['y'], self.data[1]['z'], timestamp,
                      status)

