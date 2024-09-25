import math
from typing import Tuple

def three_points_angle(p1: Tuple[float,float], p2: Tuple[float,float], p3: Tuple[float,float]) -> float:
    """计算三个点之间的夹角，p2为中间折点

    Args:
        p1 (Tuple[float,float]): 起点坐标
        p2 (Tuple[float,float]): 折点坐标
        p3 (Tuple[float,float]): 终点坐标

    Returns:
        float: 角度
    """
    if p1 == (0,0) or p2 == (0,0) or p3 == (0,0):
        return -1.0
    
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    
    vector_ab = (x2 - x1, y2 - y1)
    vector_bc = (x2 - x3, y2 - y3)
    
    angle_degrees = two_vector_angle(vector_ab,vector_bc)

    return angle_degrees

def two_vector_angle(v1: Tuple[float,float], v2: Tuple[float,float]) -> float:
    """计算两个向量之间的夹角

    Args:
        v1 (Tuple[float,float]): 向量1
        v2 (Tuple[float,float]): 向量2

    Returns:
        float: 两向量夹角
    """
    if v1 == (0,0) or v2 == (0,0):
        return -1.0
    
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    
    magnitude_ab = math.sqrt(v1[0]**2 + v1[1]**2)
    magnitude_bc = math.sqrt(v2[0]**2 + v2[1]**2)
    
    cos_theta = dot_product / (magnitude_ab * magnitude_bc)
    
    angle = math.acos(cos_theta)
    
    angle_degrees = math.degrees(angle)

    return angle_degrees