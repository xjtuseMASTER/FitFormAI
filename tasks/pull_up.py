import cv2
import torch
import utils
import numpy as np
import pandas as pd
from typing import Tuple, List, Any, TypedDict
from keypoints import Keypoints
from ultralytics import YOLO

class PullUpInfo(TypedDict):
    # TODO: 定义返回值类型
    """
    TODO
    leftElbow (Tuple[float, float]): 手肘坐标
    rightElbow (Tuple[float, float]): 手肘坐标
    leftWrist (Tuple[float, float]): 手腕坐标
    rightWrist (Tuple[float, float]): 手腕坐标
    """
    high_leftElbow : Tuple[float, float]
    high_rightElbow : Tuple[float, float]
    high_leftWrist : Tuple[float, float]
    _rightWrist : Tuple[float, float]
    wristDistance : float # 手腕距离，正值
    shoulderDistance : float # 肩膀距离，正值

    hipBoneRange : float # 髋骨极差，需要左右极差平均值, 正值

    mean_wrist_elbow_horizon_angle : Tuple[float, float] # 正值，两个-left and right，均值

    low_wrist_elbow_shoulder_angle : float # 正值, 最低点
    high_wrist_elbow_shoulder_angle : float # 正值, 最高点

    left_ankle = Tuple[float, float] # 脚踝坐标
    left_knee = Tuple[float, float] # 膝盖坐标
    right_ankle = Tuple[float, float] # 脚踝坐标
    right_knee = Tuple[float, float] # 膝盖坐标

    nose_shoulder_vertical_angle = float # 鼻子和肩膀的垂直角度，前伸正，后仰负

    kneeRange : float # 膝盖极差，需要左右极差平均值, 正值
    ankleRange : float # 脚踝极差，需要左右极差平均值, 正值

    shoulder_hipBone_y_distance : float # 肩膀和髋骨的y轴距离，正值
    wrist_shoulder_y_distance : float # 手腕和肩膀的y轴距离，正值

class PullUp:
    def __init__(self, info: PullUpInfo) -> None:
        self.info = info


    def pullUpDataProcess(self) -> PullUpInfo:
        # 应该实现到顶层函数（调用所有判别的函数），这里写上方法，避免忘记
        pass

    # TODO: 实现全部错误点判别，数据分流，返回值整合
    # 是否需要实现再议
    def pull_up_distinguish():
        pass

    def hands_hold_distance(self) -> str:
        '''判断手腕是否在手肘正上方，用于判断握距是否合适'''
        # TODO 临时阈值
        wide_threshold = 5
        narrow_threshold = 3
        wide_alpha = 1.5
        narrow_alpha = 1.3

        leftDistance = self.info['high_leftWrist'][0] - self.info['high_leftElbow'][0]
        rightDistance = self.info['high_rightWrist'][0] - self.info['high_rightElbow'][0]

        if (leftDistance < narrow_threshold and leftDistance > -wide_threshold and rightDistance < wide_threshold 
              and rightDistance > -narrow_threshold and self.info['wristDistance'] > narrow_alpha * self.info['shoulderDistance'] 
              and self.info['wristDistance'] < wide_alpha * self.info['shoulderDistance']):
            return "正确"
        elif (leftDistance > narrow_threshold and rightDistance < -narrow_threshold 
              and self.info['wristDistance'] < narrow_alpha * self.info['shoulderDistance']):
            return "握距过窄"
        elif (leftDistance < -wide_threshold and rightDistance > wide_threshold 
              and self.info['wristDistance'] > wide_alpha * self.info['shoulderDistance']):
            return "握距过宽"
        else:
            return "worng info" # TODO 高级层面收到信号后特殊处理

    def elbow(self) -> str:
        '''实现手肘角度判别'''
        # TODO 临时阈值
        tuck_threshold = 30
        back_threshold = 10

        if (self.info['mean_wrist_elbow_horizon_angle'][0] > back_threshold and self.info['mean_wrist_elbow_horizon_angle'][1] > back_threshold
              and self.info['mean_wrist_elbow_horizon_angle'][0] < tuck_threshold and self.info['mean_wrist_elbow_horizon_angle'][1] < tuck_threshold):
            return "正确"
        elif self.info['mean_wrist_elbow_horizon_angle'][0] > tuck_threshold and self.info['mean_wrist_elbow_horizon_angle'][1] > tuck_threshold:
            return "手肘内收"
        elif self.info['mean_wrist_elbow_horizon_angle'][0] < back_threshold and self.info['mean_wrist_elbow_horizon_angle'][1] < back_threshold:
            return "手肘后张"
        else:
            return "wrong info"

    def Loose_shoulder_blades_in_freeFall(self) -> str:
        '''自由落体肩胛骨松懈判别'''
        # TODO 临时阈值
        threshold = 160

        if self.info['low_wrist_elbow_shoulder_angle'] < threshold:
            return "正确"
        elif self.info['low_wrist_elbow_shoulder_angle'] > threshold:
            return "自由落体肩胛骨松懈"
        else:
            return "wrong info"

    def Leg_bending_angle(self) -> str:
        '''腿部弯曲角度过大判别'''
        if self.info['left_knee'][1] > self.info['left_ankle'][1] and self.info['right_knee'][1] > self.info['right_ankle'][1]:
            return "正确"
        elif self.info['left_knee'][1] < self.info['left_ankle'][1] or self.info['right_knee'][1] < self.info['right_ankle'][1]:
            return "膝盖弯曲角度过大"
        else :
            return "wrong info"

    def action_amplitude(self) -> str:
        '''实现动作幅度判别'''
        # TODO 临时阈值
        large_alpha = 30
        small_alpha = 10

        if (self.info['wrist_shoulder_y_distance'] < large_alpha * self.info['shoulder_hipBone_y_distance']
            and self.info['wrist_shoulder_y_distance'] > small_alpha * self.info['shoulder_hipBone_y_distance']):
            return "正确"
        elif self.info['wrist_shoulder_y_distance'] > large_alpha * self.info['shoulder_hipBone_y_distance']:
            return "动作幅度过大"
        elif self.info['wrist_shoulder_y_distance'] < small_alpha * self.info['shoulder_hipBone_y_distance']:
            return "动作幅度过小"
        else:
            return "wrong info"

    def neck_error(self) -> str:
        '''实现颈部错误判别'''
        # TODO 临时阈值
        tuck_threshold = 20
        back_threshold = -5

        if self.info['nose_shoulder_vertical_angle'] > back_threshold and self.info['nose_shoulder_vertical_angle'] < tuck_threshold:
            return "正确"
        elif self.info['nose_shoulder_vertical_angle'] < back_threshold:
            return "颈部后仰"
        elif self.info['nose_shoulder_vertical_angle'] > tuck_threshold:
            return "颈部前伸"
        else:
            return "wrong info"

    def leg_shake(self) -> str:
        '''实现腿部摇晃判别'''
        # TODO 临时阈值
        knee_threshold = 10
        ankle_threshold = 10

        if self.info['kneeRange'] < knee_threshold and self.info['ankleRange'] < ankle_threshold:
            return "正确"
        elif self.info['kneeRange'] > knee_threshold:
            return "膝盖摇晃"
        elif self.info['ankleRange'] > ankle_threshold:
            return "脚踝摇晃"
        else:
            return "wrong info"

    def core_not_tighten(self) -> str:
        '''实现核心不稳定判别'''
        # TODO: 临时阈值
        threshold = 5
        if self.info['hipBoneRange'] < threshold:
            return "正确"
        elif self.info['hipBoneRange'] > threshold:
            return "核心未收紧"
        else:
            return "wrong info"



def extract_angles(points: torch.Tensor) -> Tuple[float, float, float, float]:
    """引体向上动作必要角度信息提取,提取到的单帧角度信息,格式为包含四个角度的元组(l_angle_elbow, r_angle_elbow, l_angle_shoulder, r_angle_shoulder)"""

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
    """extract_angles将提取出的角度信息绘制在每一视频帧，并返回新的视频帧"""

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
    """判断手腕是否在手肘正上方，用于判断握距是否合适"""

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
    使用YOLO处理**引体向上/侧面视角**视频,添加便于直观感受的特征展示,并将分析结果以mp4的格式存入指定文件夹

    Args:
        input_path (str): 输入视频地址
        output_path (str): 输出mp4地址
        model (YOLO): 所使用的YOLO模型
    """
    utils.video2video_base_(side_video2video_, input_path, output_path, model, **keywarg)

