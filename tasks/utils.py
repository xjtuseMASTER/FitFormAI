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

def get_points(points: torch.Tensor) -> Tuple[float, float, float, float, float, float, float, float, float]:
    if points.size(0) == 0: return (0, 0, 0, 0)

    points = Keypoints(points)

    l_knee = points.get("l_knee")
    r_knee = points.get("r_knee")
    l_ankle = points.get("l_ankle")
    r_ankle = points.get("r_ankle")
    l_wrist = points.get("l_wrist")
    r_wrist = points.get("r_wrist")
    l_elbow = points.get("l_elbow")
    r_elbow = points.get("r_elbow")
    l_shoulder = points.get("l_shoulder")
    r_shoulder = points.get("r_shoulder")
    l_hip = points.get("l_hip")
    r_hip = points.get("r_hip")

    return 

def processor_standard(input_path: str, output_path: str, model: YOLO, **keywarg: any) -> None:
    """
    使用yolo处理**引体向上-背部视角-标准**视频,并将分析结果以csv的格式存入指定文件夹

    Args:
        input_path (str): 输入视频地址
        output_path (str): 输出视频地址
        model (YOLO): 所使用的YOLO模型
    """
    results = model(source=input_path, stream=True, **keywarg)  # generator of Results objects
    frame_idx = 0
    all_keypoints = []  # 用于存储所有帧的关键点
    for r in results:
        # processing
        keypoints = extract_main_person(r)
        all_keypoints.append(keypoints)  # 将每一帧的关键点添加到列表中    
        frame_idx += 1

    print(frame_idx)    
    df = pd.DataFrame(all_keypoints, columns=[
            'nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear',
            'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow',
            'l_wrist', 'r_wrist', 'l_hip', 'r_hip',
            'l_knee', 'r_knee', 'l_ankle', 'r_ankle'], index= list(range(1, frame_idx + 1)))
    df.to_csv(output_path, index_label='idx')

model = YOLO(r"E:\深度学习算法学习\项目管理\FitFormAI\model\yolov8m-pose.pt")
processor_standard(r"E:\深度学习算法学习\项目管理\FitFormAI\resource\引体向上\背部视角\标准\引体向上-背部-标准-03.mov",
                   r"E:\深度学习算法学习\项目管理\FitFormAI\output", model)

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
