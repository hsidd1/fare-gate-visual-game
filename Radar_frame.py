class RadarFrame:
    def __init__(self, sensorId1, x1, y1, z1, timestamp1, sensorId2, x2, y2, z2, timestamp2, status=-1) -> None:
        self.data = [{'sensorId': sensorId1, 'x': x1, 'y': y1, 'z': z1, 'timestamp': timestamp1},
                    {'sensorId': sensorId2, 'x': x2, 'y': y2, 'z': z2, 'timestamp': timestamp2}]
        self.isStatic = status # -1 default, 1 static, 0 not static. checked by static points class
        self.points_list = [(point['x'], point['y'], status) for point in self.data] # [[(x,y,static),(...)]

    def is_empty(self, sensor_id):
        
        s_id = 0 if sensor_id == 1 else 1 # get index of sensor id in self.data
        return len(self.data[s_id]) == 0
    
    # a getter for points list to be used for display
    def get_points_for_display(self, sensor_id) -> list:
        return self.points_list
        

