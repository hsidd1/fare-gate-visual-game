class RadarFrame:
   
    def __init__(self, data):
        # data is a list of dictionaries
        self.sensorid = []
        self.x = []
        self.y = []
        self.z = []
        self.timestamp = []
        self.isStatic = [] # -1 default, 1 static, 0 not static. checked by static points class
        for item in data: 
            self.sensorid.append(item['sensorId'])
            self.x.append(item['x'])
            self.y.append(item['y'])
            self.z.append(item['z'])
            self.timestamp.append(item['timestamp'])
            self.isStatic.append(-1)
    
    def __str__(self): 
        return f"""
        RadarFrame object with: sensorid: {self.sensorid}, \n
        x: {self.x}, \n
        y: {self.y}, \n
        z: {self.z}, \n
        timestamp: {self.timestamp}, \n
        is static: {self.isStatic}\n
        """
   
    # check if a specified sensor is empty      
    def is_empty(self, sensor_id=None) -> bool:  
         # if sensor id is not passed in, check all sensors
        if sensor_id == None:
            return len(self.sensorid) == 0
        else:
        # if argument specifies sensor id, check data within that sensor id
            for id in self.sensorid: # check if passed in sensor id is in list of sensor ids
                if id == sensor_id:
                    return False 
        return True # id not found, sensor is empty

    # a getter for points list to be used for display
    def get_points_for_display(self, sensor_id) -> list:
        points_list = []
        for i, id in enumerate(self.sensorid):
            if id == sensor_id:
                points_list.append((self.x[i], self.y[i], self.isStatic[i]))
        return points_list
