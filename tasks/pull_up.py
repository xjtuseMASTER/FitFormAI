import torch
import numpy as np
from math_calcu import angle


def process_angle(frame: np.array ,points: torch.Tensor) -> np.array:
    if points.size(0) == 0: return frame

    l_shoulder = tuple(points[6][:2].tolist())
    r_shoulder = tuple(points[7][:2].tolist())
    l_elbow = tuple(points[8][:2].tolist())
    r_elbow = tuple(points[9][:2].tolist())
    l_hip = tuple(points[12][:2].tolist())
    r_hip = tuple(points[13][:2].tolist())

    l_angle = angle.three_points_angle(l_elbow,l_shoulder,l_hip)
    r_angle = angle.three_points_angle(r_elbow,r_shoulder,r_hip)

