class RadarFrame:
    def __init__(self, sensorId1, x1, y1, z1, timestamp1, sensorId2, x2, y2, z2, timestamp2, status=-1) -> None:
        self.data = [{'sensorId': sensorId1, 'x': x1, 'y': y1, 'z': z1, 'timestamp': timestamp1},
                    {'sensorId': sensorId2, 'x': x2, 'y': y2, 'z': z2, 'timestamp': timestamp2}]
        self.isStatic = status # -1 default, 1 static, 0 not static. checked by static points class
        self.points_list = [(point['x'], point['y'], status) for point in self.data] # [[(x,y,static),(...)]

    def is_empty(self, sensor_id):
        return len(self.data) == 0
    
    def update_static_points(self, curr_points):
        # update self.static_points with the results from StaticPoints class.
        # if sensor id 1 is empty, set s1 pts as prev points 
        if self.is_empty(1):
            s1_pts = curr_points
        elif self.is_empty(2):
            s2_pts = curr_points
        else:
            #print("Both sensors have data")
            pass
    
    # a getter for points list to be used for display
    def get_points_for_display(self, sensor_id) -> list:
        return self.points_list
        

