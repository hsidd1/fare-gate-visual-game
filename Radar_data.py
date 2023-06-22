from Radar_frame import RadarFrame
class RadarData:
    
    def __init__(self, data: 'list[dict[str, int or float]]'):
        """
        Radar data object: contains all data from radar sensors in lists for each attribute. Updated when frames are processed by take_next_frame()
        param data: a list of dictionaries 
        ex. [{
                'sensorId': 1, 
                'x': 85.43406302787685, 
                'y': 2069.789390083478, 
                'z': 1473.3243136313272, 
                'timestamp': 1663959820484
            }, ...]
        """
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
        self.__initial_timestamp = None # set in set_initial_timestamp() to avoid overwriting after deletions in take_next_frame()
    
    def __repr__(self):
        return f"""
        RadarData object with: {self.get_num_sensors()} sensors, \n
        {len(self.x)} points, \n
        and {0 if not self.timestamp else self.timestamp[-1] - self.timestamp[0]} milliseconds of data \n
        """
    
    def set_initial_timestamp(self) -> None:
        if self.__initial_timestamp is None:
            self.__initial_timestamp = self.timestamp[0]

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
        
    def has_data(self) -> bool:
        return len(self.x) > 0
    
    # returns radar frame object for a specified interval
    def take_next_frame(self, interval: int) -> RadarFrame:
        self.set_initial_timestamp() # very first timestamp in data
        frame_last_ts = self.__initial_timestamp + self.__time_elapsed + interval
        self.__time_elapsed += interval

        frame_last_ts_index = None
        for i, ts in enumerate(self.timestamp):
            if ts <= frame_last_ts:
                frame_last_ts_index = i + 1
            else:
                break
        
        if frame_last_ts_index is None:
            return RadarFrame([])
       
        extracted_data = []
        for i in range(frame_last_ts_index):
            extracted_data.append({
                     'sensorId': self.sensorid[i],
                     'x': self.x[i],
                     'y': self.y[i],
                     'z': self.z[i],
                     'timestamp': self.timestamp[i]
                 })
        del self.sensorid[:frame_last_ts_index]
        del self.x[:frame_last_ts_index]
        del self.y[:frame_last_ts_index]
        del self.z[:frame_last_ts_index]
        del self.timestamp[:frame_last_ts_index]
        return RadarFrame(extracted_data)
