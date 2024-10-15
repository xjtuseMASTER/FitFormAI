import cv2
import torch
import utils
import numpy as np
import pandas as pd
from typing import Tuple, List, Any
from keypoints import Keypoints
from ultralytics import YOLO

# TODO: 实现全部错误点判别，数据分流，返回值整合
def pull_up_distinguish():
    pass

# TODO: 实现握距判别
def hands_hold_distance():
    pass

# TODO: 实现手肘角度判别
def elbow():
    pass

# TODO: 实现自由落体肩胛骨松懈判别
def Loose_shoulder_blades_in_freeFall():
    pass

# TODO: 实现腿部弯曲角度过大判别
def Leg_bending_angle():
    pass

# TODO： 实现动作幅度判别
def action_amplitude():
    pass

# TODO: 实现颈部错误判别
def neck_error():
    pass

# TODO：实现腿部摇晃判别
def leg_shake():
    pass

# TODO：实现核心未收紧判别
def core_not_tighten():
    pass

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


def extract_angles(points: torch.Tensor) -> Tuple[float, float, float, float]:
    """引体向上动作必要角度信息提取

    Args:
        points (torch.Tensor): 姿态骨架节点

    Returns:
        angles(Tuple[float, float, float, float]):提取到的单帧角度信息,格式为包含四个角度的元组(l_angle_elbow, r_angle_elbow, l_angle_shoulder, r_angle_shoulder)
    """
    if points.size(0) == 0: return (0, 0, 0, 0)

    keypoints = Keypoints(points)

    l_wrist = keypoints.get("l_wrist")
    r_wrist = keypoints.get("r_wrist")
    l_elbow = keypoints.get("l_elbow")
    r_elbow = keypoints.get("r_elbow")
    l_shoulder = keypoints.get("l_shoulder")
    r_shoulder = keypoints.get("r_shoulder")
    l_hip = keypoints.get("l_hip")
    r_hip = keypoints.get("r_hip")

    l_angle_elbow = utils.three_points_angle(l_wrist,l_elbow,l_shoulder)
    r_angle_elbow = utils.three_points_angle(r_wrist,r_elbow,r_shoulder)
    l_angle_shoulder = utils.three_points_angle(l_elbow,l_shoulder,l_hip)
    r_angle_shoulder = utils.three_points_angle(r_elbow,r_shoulder,r_hip)

    return (l_angle_elbow, r_angle_elbow, l_angle_shoulder, r_angle_shoulder)



def plot_angles(frame: np.array ,points: torch.Tensor) -> np.array:
    """将提取出的角度信息绘制在每一视频帧，并返回新的视频帧

    Args:
        frame (np.array): 原视频帧
        points (torch.Tensor): 节点集合

    Returns:
        np.array: 绘制角度信息后的视频帧
    """
    angles = extract_angles(points)
    keypoints = Keypoints(points)

    l_shoulder = keypoints.get("l_shoulder")
    r_shoulder = keypoints.get("r_shoulder")
    l_elbow = keypoints.get("l_elbow")
    r_elbow = keypoints.get("r_elbow")

    l_angle_elbow_text = f"{angles[0]:.2f}"
    r_angle_elbow_text = f"{angles[1]:.2f}"
    l_angle_shoulder_text = f"{angles[2]:.2f}"
    r_angle_shoulder_text = f"{angles[3]:.2f}"

    cv2.putText(frame, l_angle_elbow_text, tuple(map(int, l_elbow)), cv2.FONT_HERSHEY_SIMPLEX, 1, (84, 44, 151), 2)
    cv2.putText(frame, r_angle_elbow_text, tuple(map(int, r_elbow)), cv2.FONT_HERSHEY_SIMPLEX, 1, (84, 44, 151), 2)
    cv2.putText(frame, l_angle_shoulder_text, tuple(map(int, l_shoulder)), cv2.FONT_HERSHEY_SIMPLEX, 1, (84, 44, 151), 2)
    cv2.putText(frame, r_angle_shoulder_text, tuple(map(int, r_shoulder)), cv2.FONT_HERSHEY_SIMPLEX, 1, (84, 44, 151), 2)

    return frame



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

    l_angle = utils.two_vector_angle(l_vector,direction_vector)
    r_angle = utils.two_vector_angle(r_vector,direction_vector)

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


def back_video2csv(input_path: str, output_path: str, model: YOLO, **keywarg: any) -> None:
    """
    使用YOLO处理**引体向上/背部视角**视频,并将分析结果以csv的格式存入指定文件夹

    Args:
        input_path (str): 输入视频地址
        output_path (str): 输出csv地址
        model (YOLO): 所使用的YOLO模型
    """
    results = model(source=input_path, stream=True, **keywarg)  # generator of Results objects
    frame_idx = 0
    angles_data = []
    for r in results:
        # processing
        keypoints = utils.extract_main_person(r)
        angles = extract_angles(keypoints)
        angles_data.append(angles)
            
        frame_idx += 1

    df = pd.DataFrame(angles_data, columns=['l_elbow_angles', 'r_elbow_angles', 'l_shoulder_angles', 'r_shoulder_angles'], index= list(range(1, frame_idx + 1)))
    df.to_csv(output_path, index_label='idx')


# TODO: 实现侧面视角的数据提取，并为合适的数据构建csv
def side_video2csv(input_path: str, output_path: str, model: YOLO, **keywarg: any) -> None:
    """
    使用YOLO处理**引体向上/侧面视角**视频,并将分析结果以csv的格式存入指定文件夹

    Args:
        input_path (str): 输入视频地址
        output_path (str): 输出csv地址
        model (YOLO): 所使用的YOLO模型
    """
    pass


def back_video2video_(frame: np.array ,points: torch.Tensor) -> np.array:
    """back_video2video的辅助方法，背部视角的处理逻辑精简集中于此，这里定义了对每一视频帧的处理逻辑

    Args:
        frame (np.array): 原视频帧
        points (torch.Tensor): 骨架节点集合

    Returns:
        np.array: 处理后的视频帧
    """
    # processing
    frame = plot_angles(frame, points)

    return frame


def back_video2video(input_path: str, output_path: str, model: YOLO, **keywarg: any) -> None:
    """
    使用YOLO处理**引体向上/背部视角**视频,添加便于直观感受的特征展示,并将分析结果以mp4的格式存入指定文件夹

    Args:
        input_path (str): 输入视频地址
        output_path (str): 输出mp4地址
        model (YOLO): 所使用的YOLO模型
    """
    utils.video2video_base_(back_video2video_, input_path, output_path, model, **keywarg)


# TODO: 侧面视角每一视频帧的处理逻辑
def side_video2video_(frame: np.array ,points: torch.Tensor) -> np.array:
    """side_video2video的辅助方法，侧面视角的处理逻辑精简集中于此，这里定义了对每一视频帧的处理逻辑

    Args:
        frame (np.array): 原视频帧
        points (torch.Tensor): 骨架节点集合

    Returns:
        np.array: 处理后的视频帧
    """
    pass


def side_video2video(input_path: str, output_path: str, model: YOLO, **keywarg: any) -> None:
    """
    使用YOLO处理**引体向上/背部视角**视频,添加便于直观感受的特征展示,并将分析结果以mp4的格式存入指定文件夹

    Args:
        input_path (str): 输入视频地址
        output_path (str): 输出mp4地址
        model (YOLO): 所使用的YOLO模型
    """
    utils.video2video_base_(side_video2video_, input_path, output_path, model, **keywarg)

