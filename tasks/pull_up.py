import cv2
import torch
import numpy as np
from typing import Tuple, List, TypedDict
from math_calcu import angle
from keypoints import Keypoints

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


def process_angle(frame: np.array ,points: torch.Tensor) -> np.array:
    """引体向上动作必要角度信息提取

    Args:
        frame (np.array): 输入视频帧
        points (torch.Tensor): 姿态骨架节点

    Returns:
        np.array: 添加了角度信息的输出视频帧
    """
    if points.size(0) == 0: return frame

    points = Keypoints(points)

    l_wrist = points.get("l_wrist")
    r_wrist = points.get("r_wrist")
    l_shoulder = points.get("l_shoulder")
    r_shoulder = points.get("r_shoulder")
    l_elbow = points.get("l_elbow")
    r_elbow = points.get("r_elbow")
    l_hip = points.get("l_hip")
    r_hip = points.get("r_hip")

    l_angle_elbow = angle.three_points_angle(l_wrist,l_elbow,l_shoulder)
    r_angle_elbow = angle.three_points_angle(r_wrist,r_elbow,r_shoulder)
    l_angle_shoulder = angle.three_points_angle(l_elbow,l_shoulder,l_hip)
    r_angle_shoulder = angle.three_points_angle(r_elbow,r_shoulder,r_hip)

    l_angle_elbow_text = f"{l_angle_elbow:.2f}"
    r_angle_elbow_text = f"{r_angle_elbow:.2f}"
    l_angle_shoulder_text = f"{l_angle_shoulder:.2f}"
    r_angle_shoulder_text = f"{r_angle_shoulder:.2f}"

    cv2.putText(frame, l_angle_elbow_text, tuple(map(int, l_elbow)), cv2.FONT_HERSHEY_SIMPLEX, 1, (84, 44, 151), 2)
    cv2.putText(frame, r_angle_elbow_text, tuple(map(int, r_elbow)), cv2.FONT_HERSHEY_SIMPLEX, 1, (84, 44, 151), 2)
    cv2.putText(frame, l_angle_shoulder_text, tuple(map(int, l_shoulder)), cv2.FONT_HERSHEY_SIMPLEX, 1, (84, 44, 151), 2)
    cv2.putText(frame, r_angle_shoulder_text, tuple(map(int, r_shoulder)), cv2.FONT_HERSHEY_SIMPLEX, 1, (84, 44, 151), 2)

    return frame