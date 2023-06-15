class StaticPoints:
    def __init__(self, cnt_thres=10):
        self.static_points = []
        self.static_points_count = []
        self.cnt_max = 100 # max count for a point
        self.cnt_thres = cnt_thres # threshold for a point to be considered static

    def update(self, frame):
        """frame is a list of points in tuple, e.g. [(x1,y1),(x2,y2),...]"""
        # remove points that are not in frame
        for i in range(len(self.static_points)-1, -1, -1):
            if self.static_points[i] not in frame:
                self.static_points.pop(i)
                self.static_points_count.pop(i)

        # add new points to static_points
        for point in frame:
            if point in self.static_points:
                self.static_points_count[self.static_points.index(point)] += 1
            else:
                self.static_points.append(point)
                self.static_points_count.append(1)
        
    def get_static_points(self):
        # return a list of static points
        return [self.static_points[i] for i in range(len(self.static_points)) if self.static_points_count[i] > self.cnt_thres]

# def compute_tranformation():
