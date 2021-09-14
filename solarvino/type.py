from typing import List


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Detection:
    def __init__(self, left, top, right, bottom, score, class_id):
        # todo: can be converted into NumpyBBox
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        self.score = score
        self.class_id = int(class_id)

    def bottom_left_point(self):
        return self.bottom, self.right

    def top_right_point(self):
        return self.top, self.right


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


class Direction:
    def __init__(self, yaw, pitch, roll):
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
