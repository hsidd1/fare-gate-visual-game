class RadarFrame:
   
    def __init__(self, data: 'list[dict[str, int or float]]'):
        """
        Radar frame object contains data for a defined frame interval in lists for each attribute
        param data: a list of dictionaries
        ex. [{
                'sensorId': 2,
                'x': -280.35359191052436,
                'y': 524.516705459526,
                'z': 875.3924645059872,
                'timestamp': 1663959822542,
                'isStatic: 0
            }, ...]
        """
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
            self.isStatic.append(-1) # update in main program with static points class
    
    def __repr__(self): 
        class_str = f"RadarFrame object with {len(self.sensorid)} points."
        if len(self.sensorid) > 0:
            class_str += f" Sensor id: {set(self.sensorid)}, starting ts: {self.timestamp[0]}, ending ts: {self.timestamp[-1]}"
        return class_str
   
    # check if a specified sensor is empty      
    def is_empty(self, target_sensor_id=None) -> bool:  
         # if sensor id is not passed in, check all sensors
        if target_sensor_id is None:
            return len(self.sensorid) == 0
        else:
        # if argument specifies sensor id, check data within that sensor id only
            return not any(id == target_sensor_id for id in self.sensorid)

    # getter for points list in format to be used for display 
    def get_points_for_display(self, sensor_id) -> list:
        points_list = []
        for i, id in enumerate(self.sensorid):
            if id == sensor_id:
                points_list.append((self.x[i], self.y[i], self.z[i]))
        return points_list
    
    def set_static_points(self, points_list: list) -> None: # points list is a list of xy tuples
        # find a match of (x,y) to self.x and self.y lists and update isStatic
        for i, (x, y) in enumerate(zip(self.x, self.y)):
            if (x,y) in points_list:
                self.isStatic[i] = 1
            else:
                self.isStatic[i] = 0

