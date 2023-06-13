from Radar_frame import RadarFrame
class RadarData:
    # instantiate a RadarData object with data from two sensors held in lists for each key
    def __init__(self, data):
        # data is a list of dictionaries
        # ex [{'sensorId': [1,2], 'x': [1,2], 'y': [1,2], 'z': [1,2], 'timestamp': [1,2]}, ...]
        self.sensorid = []
        self.x = []
        self.y = []
        self.z = []
        self.timestamp = []
        for item in data: 
            self.sensorid.append(item['sensorId'])
            self.x.append(item['x'])
            self.y.append(item['y'])
            self.z.append(item['z'])
            self.timestamp.append(item['timestamp'])
        self.__time_elapsed = 0

    def get_num_sensors(self) -> int:
        has_sensor_1 = 1 in self.sensorid
        has_sensor_2 = 2 in self.sensorid
        
        if has_sensor_1 and not has_sensor_2:
            return 1
        elif has_sensor_2 and not has_sensor_1:
            return 1
        elif has_sensor_1 and has_sensor_2:
            return 2
        else:
            return 0

    def __str__(self):
        return f"""
        RadarData object with: {self.get_num_sensors()} sensors, \n
        {len(self.x)} points, \n
        and {self.timestamp[-1] - self.timestamp[0]} milliseconds of data \n
        """
        
    def has_data(self) -> bool:
        return len(self.x) > 0
    
    # returns radar frame object for a specified interval
    def take_next_frame(self, interval) -> RadarFrame:
        self.__time_elapsed += interval
        frame_first_ts = self.timestamp[0]
        frame_last_ts = frame_first_ts + interval
        frame_last_ts_index = self.timestamp.index(frame_last_ts)
        extracted_data = []
        for i in range(frame_last_ts_index):
            extracted_data.append({
                     'sensorId': self.sensorid[i],
                     'x': self.x[i],
                     'y': self.y[i],
                     'z': self.z[i],
                     'timestamp': self.timestamp[i]
                 })
        for i in range(frame_last_ts_index):
            del self.sensorid[0]
            del self.x[0]
            del self.y[0]
            del self.z[0]
            del self.timestamp[0]
        return RadarFrame(extracted_data)
