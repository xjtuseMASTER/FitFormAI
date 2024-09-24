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
    
    # 计算向量 AB 和 BC
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    
    vector_ab = (x2 - x1, y2 - y1)
    vector_bc = (x2 - x3, y2 - y3)
    
    # 计算点积
    dot_product = vector_ab[0] * vector_bc[0] + vector_ab[1] * vector_bc[1]
    
    # 计算向量的模长
    magnitude_ab = math.sqrt(vector_ab[0]**2 + vector_ab[1]**2)
    magnitude_bc = math.sqrt(vector_bc[0]**2 + vector_bc[1]**2)
    
    # 计算夹角的余弦值
    cos_theta = dot_product / (magnitude_ab * magnitude_bc)
    
    # 计算夹角
    angle = math.acos(cos_theta)
    
    # 将弧度转换为度
    angle_degrees = math.degrees(angle)

    return angle_degrees