from typing import List


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class FacialLandmark5:
    def __init__(self, points: List[Point]):
        self.points = points

    @property
    def left_eye(self):
        return self.points[0]

    @property
    def right_eye(self):
        return self.points[1]

    @property
    def nose_tip(self):
        return self.points[2]

    @property
    def left_lip_corner(self):
        return self.points[3]

    @property
    def right_lip_corner(self):
        return self.points[4]
