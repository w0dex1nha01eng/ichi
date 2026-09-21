from dataclasses import dataclass
from .kinematics import Pose2D

@dataclass(frozen=True, slots=True)
class Rectangle:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    def __post_init__(self):
        if self.xmin>= self.xmax or self.ymin>= self.ymax:
            raise ValueError("rectangle doesn't exist")

def circle_intersects_rectangle(center_x, center_y, radius, rectangle) -> bool:
    if center_x < rectangle.xmin:
        nearest_x = rectangle.xmin
    elif center_x >= rectangle.xmin and center_x <= rectangle.xmax:
        nearest_x = center_x
    else:
        nearest_x = rectangle.xmax

    if center_y < rectangle.ymin:
        nearest_y = rectangle.ymin
    elif center_y >= rectangle.ymin and center_y <= rectangle.ymax:
        nearest_y = center_y
    else:
        nearest_y = rectangle.ymax

    return(((nearest_x-center_x)**2+(nearest_y-center_y)**2) <= radius**2)

def robot_in_collision(pose:Pose2D, radius, obstacles, width, height) -> bool:
    if radius <=0:
        raise ValueError("radius should be positive")
    if min(height, width) <=0:
        raise ValueError("this map doesn't exist")
    if min(pose.x-radius,pose.y-radius)<=0 or pose.x + radius >= width or pose.y + radius >= height :
        return(True)

    for rectangle in obstacles:
        if circle_intersects_rectangle(pose.x,pose.y,radius, rectangle)>0:
            return True

    return False
