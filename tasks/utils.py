import math
import torch
import numpy as np
from typing import Tuple
from scipy.signal import find_peaks
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

__all__ = ["three_points_angle", "two_vector_angle"]

def extract_main_person(result: torch.Tensor) -> torch.Tensor:
    """抽取画面中的主体人物，一般来说拍摄主体总会是面积最大的一个，所以可以通过比较所有person的面积得到主体人物

    Args:
        result (torch.Tensor): yolo返回的results中的一帧

    Returns:
        (torch.Tensor): 主体人物的keypoints
    """
    boxes = result.boxes
    max_area = 0
    main_person_index = 0
    person_class_id = 0  #用于检测类别是否为Person

    for i, (x, y, w, h) in enumerate(boxes.xywh):
        area = w * h
        if boxes.cls[i] == person_class_id and area > max_area:
            max_area = area
            main_person_index = i

    return result.keypoints[main_person_index].data.squeeze()

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

def preprocess_wave(wave):
    """
    对波形进行预处理。

    参数:
    wave (np.ndarray): 输入的一维波形数据。

    返回:
    np.ndarray: 预处理后的波形数据。
    """
    wave = wave.to_numpy()
    wave = wave / (np.max(np.abs(wave)) - np.min(np.abs(wave)))
    return wave

def find_peaks(wave):
    """
    找到波形中的峰值。

    参数:
    wave (np.ndarray): 输入的一维波形数据。

    返回:
    Tuple[np.ndarray, dict]: 峰值的索引数组和峰值属性的字典。
    """
    wave = wave.to_numpy()
    peaks, properties = find_peaks(wave, height=0)
    return peaks, properties

def get_min_value(wave):
    """
    找到波形中的局部最小值。

    参数:
    wave (np.ndarray): 输入的一维波形数据。

    返回:
    Tuple[np.ndarray, dict]: 局部最小值的索引数组和属性的字典。
    """
    wave =  - wave.to_numpy()
    peaks, properties= np.find_peaks(wave, height=0)
    return peaks, properties


def calculate_dtw(self, wave1, wave2):
        """
        计算两个波形之间的动态时间规整距离。
        """
        dtw_distance, _ = fastdtw(wave1, wave2, dist=euclidean)
        return dtw_distance
    
def calculate_distance(self, part1, part2):
    distance = euclidean(part1, part2)
    return distance
