from typing import Tuple
import cv2
import pandas as pd
from ultralytics import YOLO
import utils
import torch
import numpy as np
from keypoints import Keypoints



def extract_angles(points: torch.Tensor) -> Tuple[float, float, float, float]:
    """仰卧起坐动作必要角度信息提取,返回格式为包含四个角度的元组(l_angle_knee, r_angle_knee, l_angle_hip, r_angle_hip, back_ground_angle)"""

    if points.size(0) == 0:
        return (0, 0, 0, 0)

    keypoints = Keypoints(points)

    l_ankle = keypoints.get("l_ankle")
    r_ankle = keypoints.get("r_ankle")
    l_knee = keypoints.get("l_knee")
    r_knee = keypoints.get("r_knee")
    l_hip = keypoints.get("l_hip")
    r_hip = keypoints.get("r_hip")
    l_shoulder = keypoints.get("l_shoulder")
    r_shoulder = keypoints.get("r_shoulder")

    l_angle_knee = utils.three_points_angle(l_ankle,l_knee,l_hip)
    r_angle_knee = utils.three_points_angle(r_ankle,r_knee,r_hip)
    l_angle_hip = utils.three_points_angle(l_knee,l_hip,l_shoulder)
    r_angle_hip = utils.three_points_angle(r_knee,r_hip,r_shoulder)

    mid_ankle = (int((l_ankle[0] + r_ankle[0]) // 2), int((l_ankle[1] + r_ankle[1]) // 2))
    mid_hip = (int((l_hip[0] + r_hip[0]) // 2), int((l_hip[1] + r_hip[1]) // 2))
    mid_shoulder = (int((l_shoulder[0] + r_shoulder[0]) // 2), int((l_shoulder[1] + r_shoulder[1]) // 2))

    back_vector = ((mid_hip[0] - mid_ankle[0]), (mid_hip[1] - mid_ankle[1]))
    ground_vector = ((mid_shoulder[0] - mid_hip[0]), (mid_shoulder[1] - mid_hip[1]))
    back_ground_angle = utils.two_vector_angle(back_vector, ground_vector)

    return (l_angle_knee, r_angle_knee, l_angle_hip, r_angle_hip, back_ground_angle)



def plot_angles(frame: np.array ,points: torch.Tensor) -> np.array:
    """将提取出的角度信息绘制在每一视频帧，并返回新的视频帧"""
    keypoints = Keypoints(points)

    l_ankle = keypoints.get("l_ankle")
    r_ankle = keypoints.get("r_ankle")
    l_knee = keypoints.get("l_knee")
    r_knee = keypoints.get("r_knee")
    l_hip = keypoints.get("l_hip")
    r_hip = keypoints.get("r_hip")
    l_shoulder = keypoints.get("l_shoulder")
    r_shoulder = keypoints.get("r_shoulder")

    # extract_angles角度绘制
    l_angle_knee = utils.three_points_angle(l_ankle,l_knee,l_hip)
    r_angle_knee = utils.three_points_angle(r_ankle,r_knee,r_hip)
    l_angle_hip = utils.three_points_angle(l_knee,l_hip,l_shoulder)
    r_angle_hip = utils.three_points_angle(r_knee,r_hip,r_shoulder)


    l_angle_knee_text = f"{l_angle_knee:.2f}"
    r_angle_knee_text = f"{r_angle_knee:.2f}"
    l_angle_hip_text = f"{l_angle_hip:.2f}"
    r_angle_hip_text = f"{r_angle_hip:.2f}"

    cv2.putText(frame, l_angle_knee_text, tuple(map(int, l_knee)), cv2.FONT_HERSHEY_SIMPLEX, 1, (84, 44, 151), 2)
    cv2.putText(frame, r_angle_knee_text, tuple(map(int, r_knee)), cv2.FONT_HERSHEY_SIMPLEX, 1, (84, 44, 151), 2)
    cv2.putText(frame, l_angle_hip_text, tuple(map(int, l_hip)), cv2.FONT_HERSHEY_SIMPLEX, 1, (84, 44, 151), 2)
    cv2.putText(frame, r_angle_hip_text, tuple(map(int, r_hip)), cv2.FONT_HERSHEY_SIMPLEX, 1, (84, 44, 151), 2)

    # back_ground_angle角度绘制
    mid_ankle = (int((l_ankle[0] + r_ankle[0]) // 2), int((l_ankle[1] + r_ankle[1]) // 2))
    mid_hip = (int((l_hip[0] + r_hip[0]) // 2), int((l_hip[1] + r_hip[1]) // 2))
    mid_shoulder = (int((l_shoulder[0] + r_shoulder[0]) // 2), int((l_shoulder[1] + r_shoulder[1]) // 2))

    back_ground_angle_place = ((mid_shoulder[0] + mid_hip[0]) // 2, (mid_shoulder[1] + mid_hip[1]) // 2)

    back_vector = ((mid_hip[0] - mid_ankle[0]), (mid_hip[1] - mid_ankle[1]))
    ground_vector = ((mid_shoulder[0] - mid_hip[0]), (mid_shoulder[1] - mid_hip[1]))
    back_ground_angle = utils.two_vector_angle(back_vector, ground_vector)

    cv2.line(frame, mid_ankle, mid_hip, (255, 200, 100), thickness = 2)
    cv2.line(frame, mid_hip, mid_shoulder, (255, 200, 100), thickness = 2)
    cv2.putText(frame, f"{back_ground_angle:.2f}", back_ground_angle_place, cv2.FONT_HERSHEY_SIMPLEX, 1, (84, 44, 151), 2)

    return frame

def point_error(points: torch.Tensor) -> Tuple[float, float, float]:
    keypoints = Keypoints(points)

    l_ankle = keypoints.get("l_ankle")
    r_ankle = keypoints.get("r_ankle")
    l_knee = keypoints.get("l_knee")
    r_knee = keypoints.get("r_knee")
    l_hip = keypoints.get("l_hip")
    r_hip = keypoints.get("r_hip")

    dist_ankle = utils.euclidean_distance(l_ankle, r_ankle)
    dist_knee = utils.euclidean_distance(l_knee, r_knee)
    dist_hip = utils.euclidean_distance(l_hip, r_hip)

    return (dist_ankle, dist_knee, dist_hip)

def side_video2csv(input_path: str, output_path: str, model: YOLO, **keywarg: any) -> None:
    """
    使用YOLO处理**仰卧起坐/侧面视角**视频,并将分析结果以csv的格式存入指定文件夹

    Args:
        input_path (str): 输入视频地址
        output_path (str): 输出csv地址
        model (YOLO): 所使用的YOLO模型
    """
    results = model(source=input_path, stream=True, **keywarg)  # generator of Results objects
    frame_idx = 0
    csv_data = []
    for r in results:
        # processing
        keypoints = utils.extract_main_person(r)
        angles = extract_angles(keypoints)
        dist = point_error(keypoints)
        data = angles + dist
        csv_data.append(data)
            
        frame_idx += 1

    df = pd.DataFrame(csv_data, columns=['l_angle_knee', 'r_angle_knee', 'l_angle_hip', 'r_angle_hip', 'back_ground_angle', 'dist_ankle', 'dist_knee', 'dist_hip'], index= list(range(1, frame_idx + 1)))
    df.to_csv(output_path, index_label='idx')


def side_video2video_(frame: np.array ,points: torch.Tensor) -> np.array:
    """side_video2video的辅助方法，侧面视角的处理逻辑精简集中于此，这里定义了对每一视频帧的处理逻辑

    Args:
        frame (np.array): 原视频帧
        points (torch.Tensor): 骨架节点集合

    Returns:
        np.array: 处理后的视频帧
    """
    # frame = utils.show_keypoints(frame, points)
    frame = plot_angles(frame, points)
    return frame


def side_video2video(input_path: str, output_path: str, model: YOLO, **keywarg: any) -> None:
    """
    使用YOLO处理**仰卧起坐/侧面视角**视频,添加便于直观感受的特征展示,并将分析结果以mp4的格式存入指定文件夹

    Args:
        input_path (str): 输入视频地址
        output_path (str): 输出mp4地址
        model (YOLO): 所使用的YOLO模型
    """
    utils.video2video_base_(side_video2video_, input_path, output_path, model, **keywarg)

