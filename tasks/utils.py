import math
import cv2
import torch
import numpy as np
from typing import Any, Tuple
import pandas as pd
from keypoints import Keypoints
from ultralytics import YOLO
from scipy.signal import find_peaks
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from ultralytics import YOLO

from keypoints import Keypoints

__all__ = ["extract_main_person", "show_keypoints", "video2video_base_", "three_points_angle"]

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


def show_keypoints(frame: np.array, points: torch.Tensor) -> np.array:
    """在视频帧中添加姿态节点，用于判断节点位置

    Args:
        frame (np.array): 输入视频帧
        points (torch.Tensor): 姿态骨架节点

    Returns:
        np.array: 添加了节点标注的视频帧
    """
    if points.size(0) == 0: return frame

    keypoints = Keypoints(points)

    for keypoint in keypoints.get_all_keypoints():
        cv2.putText(frame, keypoint["part"], tuple(map(int, keypoint["location"])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    return frame


def video2video_base_(process_methods: Any, input_path: str, output_path: str, model: YOLO, **keywarg: any) -> None:
    """将传入视频做处理并输出标记后视频的基本方法，将打开视频，定义编码器，逐帧分析逻辑等通用代码进行封装，以便复用，实现新的vedio2vedio方法是，只需实现相应的process_methods方法即可

    Args:
        process_methods (Any): 特定任务/视角的视频帧处理方法
        input_path (str): 视频输入地址
        output_path (str): 视频输出地址，输出格式为mp4
        model (YOLO): 使用的YOLO模型
    """
    # 打开视频文件
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # 获取视频的宽度、高度和帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 定义编解码器
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    results = model(source=input_path, stream=True, **keywarg)  
    frame_idx = 0
    for r in results:
        annotated_frame = r.plot()

        keypoints = extract_main_person(r)
        # processing
        annotated_frame = process_methods(annotated_frame, keypoints)

        out.write(annotated_frame)
            
        frame_idx += 1
    
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def euclidean_distance(point1, point2):
    """计算欧几里得距离"""
    if np.array_equal(point1, (0, 0)) or np.array_equal(point2, (0, 0)):
        return -1.0
    
    point1 = np.array(point1)
    point2 = np.array(point2)
    
    distance = np.sqrt(np.sum((point1 - point2) ** 2))
    return distance


def three_points_angle(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
    """计算三个点之间的夹角，p2为中间折点

    Args:
        p1 (Tuple[float,float]): 起点坐标
        p2 (Tuple[float,float]): 折点坐标
        p3 (Tuple[float,float]): 终点坐标

    Returns:
        float: 角度
    """
    if np.array_equal(p1, (0, 0)) or np.array_equal(p2, (0, 0)) or np.array_equal(p3, (0, 0)):
        return -1.0
    
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    vector_ab = p2 - p1
    vector_bc = p2 - p3
    
    angle_degrees = two_vector_angle(vector_ab, vector_bc)

    return angle_degrees



def two_vector_angle(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """计算两个向量之间的夹角

    Args:
        v1 (Tuple[float, float]): 向量1
        v2 (Tuple[float, float]): 向量2

    Returns:
        float: 两向量夹角（单位：度）
    """
    if np.array_equal(v1, (0, 0)) or np.array_equal(v2, (0, 0)):
        return -1.0

    v1 = np.array(v1)
    v2 = np.array(v2)

    dot_product = np.dot(v1, v2)

    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    angle = np.arccos(cos_theta)
    angle_degrees = np.degrees(angle)

    return angle_degrees



def perpendicular_point_to_line(point, line_start, line_end):
    """
    计算给定点到由两点定义的直线的垂足。
    
    Args:
        point (tuple): 给定的点 (x, y)
        line_start (tuple): 直线的起始点 (x1, y1)
        line_end (tuple): 直线的终止点 (x2, y2)
    
    Returns:
        tuple: 垂足点 (px, py)
    """
    if np.array_equal(point, (0, 0)) or np.array_equal(line_start, (0, 0)) or np.array_equal(line_end, (0, 0)):
        return -1.0
    
    P = np.array(point)
    A = np.array(line_start)
    B = np.array(line_end)
    
    AB = B - A
    AP = P - A
    
    dot_product = np.dot(AP, AB)
    AB_length_squared = np.dot(AB, AB)
    t = dot_product / AB_length_squared
    perpendicular_point = A + t * AB
    
    return tuple(perpendicular_point.astype(int))



def translate_point_by_vector(point, start_point, end_point):
    """
    将给定点沿着从起点到终点的向量平移。

    Args:
        point (tuple): 给定的点 (x, y)
        start_point (tuple): 向量的起点坐标 (x1, y1)
        end_point (tuple): 向量的终点坐标 (x2, y2)
    
    Returns:
        tuple: 平移后的点 (px, py)
    """
    if np.array_equal(point, (0, 0)) or np.array_equal(start_point, (0, 0)) or np.array_equal(end_point, (0, 0)):
        return -1
    
    P = np.array(point)
    start = np.array(start_point)
    end = np.array(end_point)
    
    vector = end - start
    
    translated_point = P + vector
    return tuple(translated_point.astype(int))