import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, TypedDict
from feature_extract import angle
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


def process_angle(points: torch.Tensor) -> Tuple[float, float, float, float]:
    """引体向上动作必要角度信息提取

    Args:
        points (torch.Tensor): 姿态骨架节点

    Returns:
        angles(Tuple[float, float, float, float]):提取到的单帧角度信息,格式为包含四个角度的元组(l_angle_elbow, r_angle_elbow, l_angle_shoulder, r_angle_shoulder)
    """
    if points.size(0) == 0: return (0, 0, 0, 0)

    points = Keypoints(points)

    l_wrist = points.get("l_wrist")
    r_wrist = points.get("r_wrist")
    l_elbow = points.get("l_elbow")
    r_elbow = points.get("r_elbow")
    l_shoulder = points.get("l_shoulder")
    r_shoulder = points.get("r_shoulder")
    l_hip = points.get("l_hip")
    r_hip = points.get("r_hip")

    l_angle_elbow = angle.three_points_angle(l_wrist,l_elbow,l_shoulder)
    r_angle_elbow = angle.three_points_angle(r_wrist,r_elbow,r_shoulder)
    l_angle_shoulder = angle.three_points_angle(l_elbow,l_shoulder,l_hip)
    r_angle_shoulder = angle.three_points_angle(r_elbow,r_shoulder,r_hip)

    # l_angle_elbow_text = f"{l_angle_elbow:.2f}"
    # r_angle_elbow_text = f"{r_angle_elbow:.2f}"
    # l_angle_shoulder_text = f"{l_angle_shoulder:.2f}"
    # r_angle_shoulder_text = f"{r_angle_shoulder:.2f}"

    # cv2.putText(frame, l_angle_elbow_text, tuple(map(int, l_elbow)), cv2.FONT_HERSHEY_SIMPLEX, 1, (84, 44, 151), 2)
    # cv2.putText(frame, r_angle_elbow_text, tuple(map(int, r_elbow)), cv2.FONT_HERSHEY_SIMPLEX, 1, (84, 44, 151), 2)
    # cv2.putText(frame, l_angle_shoulder_text, tuple(map(int, l_shoulder)), cv2.FONT_HERSHEY_SIMPLEX, 1, (84, 44, 151), 2)
    # cv2.putText(frame, r_angle_shoulder_text, tuple(map(int, r_shoulder)), cv2.FONT_HERSHEY_SIMPLEX, 1, (84, 44, 151), 2)

    return (l_angle_elbow, r_angle_elbow, l_angle_shoulder, r_angle_shoulder)

def plot_angles(angle_data: list, frame_idx: int):
    """绘制角度折线图

    Args:
        angle_data (list): 包含每帧的四个角度的数据
        frame_indices (int): 帧的索引
    """
    l_elbow_angles, r_elbow_angles, l_shoulder_angles, r_shoulder_angles = zip(*angle_data)

    idx_list = list(range(1, frame_idx + 1))
    plt.figure(figsize=(10, 5))
    plt.plot(idx_list, l_elbow_angles, label='l_elbow_angles', marker='o')
    plt.plot(idx_list, r_elbow_angles, label='r_elbow_angles', marker='o')
    plt.plot(idx_list, l_shoulder_angles, label='l_shoulder_angles', marker='o')
    plt.plot(idx_list, r_shoulder_angles, label='r_shoulder_angles', marker='o')

    plt.xlabel('frame_idx')
    plt.ylabel('angles')
    plt.title('Change of the angle of pull-up action')
    plt.legend()
    plt.grid()
    plt.savefig('output/angles_plot.png')


def is_wrist_above_elbow(frame: np.array ,points: torch.Tensor) -> np.array:
    """判断手腕是否在手肘正上方，用于判断握距是否合适

    Args:
        frame (np.array): 输入视频帧
        points (torch.Tensor): 姿态骨架节点

    Returns:
        np.array: 添加了握距判断信息的输出视频帧
    """
    if points.size(0) == 0: return frame

    is_wrist_above_elbow = False
    threshold = 5

    points = Keypoints(points)

    direction_vector = (0, -1.0)

    l_wrist = points.get("l_wrist")
    r_wrist = points.get("r_wrist")
    l_elbow = points.get("l_elbow")
    r_elbow = points.get("r_elbow")

    l_vector = tuple(map(lambda x, y: x - y, l_wrist, l_elbow))
    r_vector = tuple(map(lambda x, y: x - y, r_wrist, r_elbow))

    l_angle = angle.two_vector_angle(l_vector,direction_vector)
    r_angle = angle.two_vector_angle(r_vector,direction_vector)

    l_angle_text = f"{l_angle:.2f}"
    r_angle_text = f"{r_angle:.2f}"

    l_text_location = tuple(map(lambda x, y: (x + y)/2, l_wrist, l_elbow))
    r_text_location = tuple(map(lambda x, y: (x + y)/2, r_wrist, r_elbow))

    cv2.putText(frame, l_angle_text, tuple(map(int, l_text_location)), cv2.FONT_HERSHEY_SIMPLEX, 1, (84, 44, 151), 2)
    cv2.putText(frame, r_angle_text, tuple(map(int, r_text_location)), cv2.FONT_HERSHEY_SIMPLEX, 1, (84, 44, 151), 2)

    if (l_angle + r_angle) / 2 < threshold:
           is_wrist_above_elbow = True

    cv2.putText(frame, f"Is the grip distance appropriate?:{is_wrist_above_elbow}", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    return frame

