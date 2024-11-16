import cv2
import random
import pandas as pd
from ultralytics import YOLO
import utils
import torch
from advices import load_advice_by_filename
from typing import Tuple, List
import numpy as np
from keypoints import Keypoints
from task_base import TaskBase, InfoBase, ErrorDetail
from data_processor import DateProcessor

PLOT = False

class DeepsquatInfo(InfoBase):
    hip_y_trough: float
    hip_y_trough_idx: List[int]
    knee_y_mean: float
    shoulder_hip_knee_angle_trough: float
    shoulder_hip_knee_angle_trough_idx: List[int]

class Deepsquat(TaskBase):
    def __init__(self, model: YOLO) -> None:
        super().__init__()
        self.task = "deep_squats"
        self.error_list = [
            "lean_forward", "not_enough_squat_distance"
            ]
        self.model = model


    def do_analysis(self, input_path: str):
        """对外暴露的接口函数"""

        results: List[ErrorDetail] = []
        yolo_outputs = self.model(source=input_path, stream=True)
        info = self._feature_extractor(yolo_outputs)
        error_list = self._judge_error(info)
        frame_idxs = self._frame_idx_extractor(error_list, info)
        frames = self._find_out_frames(input_path, frame_idxs)
        plot_frames = self._plot_the_frames(info, error_list, frame_idxs, frames)

        advices = load_advice_by_filename(self.task + ".json")
        for i, error in enumerate(error_list):
            error_detail: ErrorDetail = {
                "error": advices[error]["error"],
                "advice": advices[error]["advice"],
                "frame": plot_frames[i]
            }
            results.append(error_detail)
        
        return results
        

    def _feature_extractor(self, yolo_outputs: list) -> DeepsquatInfo:
        """获取分析判断前所需要的所有特征信息"""
        frame_idx = 0
        data = []
        raw_keypoints = []
        for r in yolo_outputs:
            keypoints = utils.extract_main_person(r)
            # processing
            points, labels = extract_points(keypoints)
            data.append(points)
            raw_keypoints.append(keypoints)

            frame_idx += 1
        raw_data = pd.DataFrame(data, columns=labels, index=list(range(1, len(data) + 1)))
        data_processor = DateProcessor(raw_data)

        # basic data processing
        processed_knee_y = data_processor.process_wave_data("knee_y")
        processed_hip_y = data_processor.process_wave_data("hip_y")

        # more data processing
        processed_shoulder_hip_knee_angle = data_processor.process_wave_data("shoulder_hip_knee_angle")

        deepsquat_info: DeepsquatInfo = {
            'raw_keypoints': raw_keypoints,
            "raw_data": raw_data,
            "hip_y_trough": processed_hip_y["trough"],
            "hip_y_trough_idx": processed_hip_y["indices_of_troughs"],
            "knee_y_mean": processed_knee_y["mean"],
            "shoulder_hip_knee_angle_trough": processed_shoulder_hip_knee_angle["trough"],
            "shoulder_hip_knee_angle_trough_idx": processed_shoulder_hip_knee_angle["indices_of_troughs"]
        }
        return deepsquat_info

    def _lean_forward(self, info: DeepsquatInfo) -> bool:
        """判断是否前倾"""
        # TODO 临时阈值
        threshold = 80

        if info["shoulder_hip_knee_angle_trough"] < threshold:
            return True
        else:
            return False
    
    def _not_enough_squat_distance(self, info: DeepsquatInfo) -> bool:
        """判断是否下蹲不够"""
        # TODO 临时阈值
        threshold = 50

        if info["hip_y_trough"] - info["knee_y_mean"] > threshold:
            return True
        else:
            return False

    def _plot_lean_forward(self, info: DeepsquatInfo, frame_idx: int, frame: np.array) -> np.array:
        """绘制前倾的帧"""
        raw_keypoints = info["raw_keypoints"]
        keypoints = Keypoints(raw_keypoints[frame_idx])
        l_shoulder = keypoints.get_int("l_shoulder")
        r_shoulder = keypoints.get_int("r_shoulder")
        l_hip = keypoints.get_int("l_hip")
        r_hip = keypoints.get_int("r_hip")
        l_knee = keypoints.get_int("l_knee")
        r_knee = keypoints.get_int("r_knee")
        shoulder = (l_shoulder + r_shoulder) // 2
        hip = (l_hip + r_hip) // 2
        knee = (l_knee + r_knee) // 2

        angle = utils.three_points_angle(shoulder, hip, knee)
        cv2.line(frame, shoulder, hip, (0, 255, 0), 2)
        cv2.line(frame, hip, knee, (0, 255, 0), 2)
        cv2.putText(frame, str(int(angle)), hip, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        if PLOT:
            frame_filename = f"unreasonable_leg_folding_angle.jpg"
            cv2.imwrite(frame_filename, frame)
        
        return frame
    
    def _plot_not_enough_squat_distance(self, info: DeepsquatInfo, frame_idx: int, frame: np.array) -> np.array:
        """绘制下蹲不够的帧"""
        raw_keypoints = info["raw_keypoints"]
        keypoints = Keypoints(raw_keypoints[frame_idx])
        l_hip = keypoints.get_int("l_hip")
        r_hip = keypoints.get_int("r_hip")
        hip = (l_hip + r_hip) // 2

        overlay = frame.copy()
        cv2.circle(overlay, hip, 50, color = (255, 0, 0), thickness=-1)
        cv2.addWeighted(overlay, 0.5, frame, 1 - 0.5, 0, frame)

        if PLOT:
            frame_filename = f"not_enough_squat_distance.jpg"
            cv2.imwrite(frame_filename, frame)
        
        return frame

    def _frame_lean_forward(self, info: DeepsquatInfo) -> int:
        """返回前倾的帧索引"""
        return random.choice(info["shoulder_hip_knee_angle_trough_idx"])
    
    def _frame_not_enough_squat_distance(self, info: DeepsquatInfo) -> int:
        """返回下蹲不够的帧索引"""
        return random.choice(info["hip_y_trough_idx"])

def extract_points(points: torch.Tensor):
    """引体向上动作必要角度信息提取,提取到的单帧点信息,格式为包含点信息的元组"""
    if points.size(0) == 0: return (0, 0, 0, 0)
    keypoints = Keypoints(points)
    l_hip_y = keypoints.get("l_hip")[1]
    r_hip_y = keypoints.get("r_hip")[1]
    l_knee_y = keypoints.get("l_knee")[1]
    r_knee_y = keypoints.get("r_knee")[1]
    l_ankle_y = keypoints.get("l_ankle")[1]
    r_ankle_y = keypoints.get("r_ankle")[1]

    hip_y = (l_hip_y + r_hip_y) / 2
    knee_y = (l_knee_y + r_knee_y) / 2

    l_shoulder = keypoints.get("l_shoulder")
    r_shoulder = keypoints.get("r_shoulder")
    l_hip = keypoints.get("l_hip")
    r_hip = keypoints.get("r_hip")
    l_knee = keypoints.get("l_knee")
    r_knee = keypoints.get("r_knee")

    l_shoulder_hip_knee_angle = utils.three_points_angle(l_shoulder,l_hip,l_knee)
    r_shoulder_hip_knee_angle = utils.three_points_angle(r_shoulder,r_hip,r_knee)
    shoulder_hip_knee_angle = (l_shoulder_hip_knee_angle + r_shoulder_hip_knee_angle) / 2
    return (hip_y, knee_y, shoulder_hip_knee_angle), ["hip_y", "knee_y", "shoulder_hip_knee_angle"]

# # use-case
model = YOLO(r"E:\算法\项目管理\FitFormAI\model\yolov8n-pose.pt")
deepsquat = Deepsquat(model)
path = r"E:\算法\项目管理\FitFormAI\resource\引体向上\正侧面视角\手肘不合理\IMG_6086.MOV"
deepsquat.do_analysis(path)
