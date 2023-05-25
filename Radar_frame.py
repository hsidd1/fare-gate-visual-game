class RadarFrame:
   
    def __init__(self, data, status=-1):
            # data is a dictionary mapping keys to  lists 
            # ex. {'sensorId': [1,2], 'x': [1,2], 'y': [1,2], 'z': [1,2], 'timestamp': [1,2]}
            try:
                 if isinstance(data, dict) and all(isinstance(val, list) for val in data.values()) and len(data) == 5:
                    self.data = data
            except:
                raise TypeError("Error in RadarFrame instantiation: data must be a dictionary mapping keys to lists")
            self.isStatic = status # -1 default, 1 static, 0 not static. checked by static points class
            self.points_list = [(point['x'], point['y'], status) for point in self.data] # [[(x,y,static),(...)]

    # check if a specified sensor is empty      
    def is_empty(self, sensor_id) -> bool:  
        s_id = 0 if sensor_id == 1 else 1 # get index of sensor id in self.data
        return len(self.data[s_id]) == 0
    
    # a getter for points list to be used for display
    def get_points_for_display(self, sensor_id) -> list:
        return self.points_list
        

