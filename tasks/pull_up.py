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

class PullUpInfo(InfoBase):
    wristDistance : float # 手腕距离，正值
    shoulderDistance : float # 肩膀距离，正值

    hipBoneRange : float # 髋骨极差，需要左右极差平均值, 正值

    mean_wrist_elbow_horizon_angle : Tuple[float, float] # 正值，两个-left and right，均值

    low_wrist_elbow_shoulder_angle : float # 正值, 最低点
    high_wrist_elbow_shoulder_angle : float # 正值, 最高点

    left_ankle = (float, float) # 脚踝坐标
    left_knee = (float, float) # 膝盖坐标
    right_ankle = (float, float) # 脚踝坐标
    right_knee = (float, float) # 膝盖坐标

    nose_shoulder_vertical_angle = float # 鼻子和肩膀的垂直角度，前伸正，后仰负

    kneeRange : float # 膝盖极差，需要左右极差平均值, 正值
    ankleRange : float # 脚踝极差，需要左右极差平均值, 正值

    shoulder_hipBone_y_distance : float # 肩膀和髋骨的y轴距离，正值
    wrist_shoulder_y_distance : float # 手腕和肩膀的y轴距离，正值

class PullUp(TaskBase):
    def __init__(self, model: YOLO) -> None:
        super().__init__()
        self.task = "pull_ups"
        self.error_list = [
            "hands_hold_distance", "elbow", "Loose_shoulder_blades_in_freeFall", "Leg_bending_angle", 
            'action_amplitude', "neck_error", "leg_shake", "core_not_tighten"
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
        

    def _feature_extractor(self, yolo_outputs: list) -> PullUpInfo:
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
        processed_l_ankle_x = data_processor.process_wave_data("l_ankle_x")
        processed_l_ankle_y = data_processor.process_wave_data("l_ankle_y")
        processed_l_knee_x = data_processor.process_wave_data("l_knee_x")
        processed_l_knee_y = data_processor.process_wave_data("l_knee_y")
        processed_r_ankle_x = data_processor.process_wave_data("r_ankle_x")
        processed_r_ankle_y = data_processor.process_wave_data("r_ankle_y")
        processed_r_knee_x = data_processor.process_wave_data("r_knee_x")
        processed_r_knee_y = data_processor.process_wave_data("r_knee_y")
        processed_l_shoulder_y = data_processor.process_wave_data("l_shoulder_y")
        processed_r_shoulder_y = data_processor.process_wave_data("r_shoulder_y")
        peak_body_idx = processed_r_shoulder_y["indices_of_peaks"]
        trough_body_idx = processed_r_shoulder_y["indices_of_troughs"]
        processed_shoulder_hip_y_distance = data_processor.process_unwave_data("shoulder_hip_y_distance")["mean"]
        processed_wrist_shoulder_y_distance = data_processor.process_wave_data("wrist_shoulder_y_distance")["trough"]
        hipBoneRange = data_processor.process_wave_data("hip_x_distance")
        kneeRange = data_processor.process_wave_data("knee_x_distance")
        ankleRange = data_processor.process_wave_data("ankle_x_distance")

        # more data processing
        wrist_distance = data_processor.process_unwave_data("wrist_x_distance")["mean"]
        shoulder_distance = data_processor.process_unwave_data("shoulder_x_distance")["mean"]

        processed_wrist_elbow_shoulder_angle = data_processor.process_wave_data("wrist_elbow_shoulder_angle")
        processed_nose_shoulder_vertical_angle = data_processor.process_wave_data("nose_shoulder_vertical_angle")

        pullup_info: PullUpInfo = {
            'raw_keypoints': raw_keypoints,
            "raw_data": raw_data,
            "wristDistance": wrist_distance,
            "shoulderDistance": shoulder_distance,
            "hipBoneRange": hipBoneRange["peak"] - hipBoneRange["trough"],
            "mean_wrist_elbow_shoulder_angle": processed_wrist_elbow_shoulder_angle["mean"],
            "mean_wrist_elbow_shoulder_angle_trough_idx": processed_wrist_elbow_shoulder_angle["indices_of_troughs"],
            "low_wrist_elbow_shoulder_angle": processed_wrist_elbow_shoulder_angle["peak"],
            "trough_body_idx": processed_wrist_elbow_shoulder_angle["indices_of_troughs"],
            "high_wrist_elbow_shoulder_angle": processed_wrist_elbow_shoulder_angle["trough"],
            "left_ankle": (processed_l_ankle_x["mean"], processed_l_ankle_y["mean"]),
            "left_knee": (processed_l_knee_x["mean"], processed_l_knee_y["mean"]),
            "right_ankle": (processed_r_ankle_x["mean"], processed_r_ankle_y["mean"]),
            "right_knee": (processed_r_knee_x["mean"], processed_r_knee_y["mean"]),
            "nose_shoulder_vertical_angle": processed_nose_shoulder_vertical_angle["mean"],
            "nose_shoulder_vertical_angle_peak_idx": processed_nose_shoulder_vertical_angle["indices_of_peaks"],
            "kneeRange": kneeRange["peak"] - kneeRange["trough"],
            "ankleRange": ankleRange["peak"] - ankleRange["trough"],
            "shoulder_hipBone_y_distance": processed_shoulder_hip_y_distance,
            "wrist_shoulder_y_distance": processed_wrist_shoulder_y_distance,
            "peak_body_idx": peak_body_idx
        }
        return pullup_info

    def _hands_hold_distance(self, info: PullUpInfo) -> bool:
        '''判断手腕是否在手肘正上方，用于判断握距是否合适'''
        wide_alpha = 2.40
        narrow_alpha = 1.60

        if (info['wristDistance'] > narrow_alpha * info['shoulderDistance'] 
              and info['wristDistance'] < wide_alpha * info['shoulderDistance']):
            return False
        elif (info['wristDistance'] < narrow_alpha * info['shoulderDistance']):
            return True
        elif (info['wristDistance'] > wide_alpha * info['shoulderDistance']):
            return True
        else:
            return False 

    def _elbow(self, info: PullUpInfo) -> bool:
        '''实现手肘角度判别'''
        tuck_threshold = 85

        if (info['mean_wrist_elbow_shoulder_angle'] > tuck_threshold):
            return False
        elif info['mean_wrist_elbow_shoulder_angle'] < tuck_threshold:
            return True
        else:
            return False

    def _Loose_shoulder_blades_in_freeFall(self, info: PullUpInfo) -> bool:
        '''自由落体肩胛骨松懈判别'''
        threshold = 150

        if info['low_wrist_elbow_shoulder_angle'] < threshold:
            return False
        elif info['low_wrist_elbow_shoulder_angle'] > threshold:
            return True
        else:
            return False

    def _Leg_bending_angle(self, info: PullUpInfo) -> bool:
        '''腿部弯曲角度过大判别'''
        if info['left_knee'][1] > info['left_ankle'][1] and info['right_knee'][1] > info['right_ankle'][1]:
            return False
        elif info['left_knee'][1] < info['left_ankle'][1] or info['right_knee'][1] < info['right_ankle'][1]:
            return True
        else :
            return False

    def _action_amplitude(self, info: PullUpInfo) -> bool:
        '''实现动作幅度判别'''
        large_alpha = 0.01
        small_alpha = 0.30

        if (info['wrist_shoulder_y_distance'] < large_alpha * info['shoulder_hipBone_y_distance']
            and info['wrist_shoulder_y_distance'] > small_alpha * info['shoulder_hipBone_y_distance']):
            return False
        elif info['wrist_shoulder_y_distance'] < large_alpha * info['shoulder_hipBone_y_distance']:
            return True
        elif info['wrist_shoulder_y_distance'] > small_alpha * info['shoulder_hipBone_y_distance']:
            return True
        else:
            return False

    def _neck_error(self, info: PullUpInfo) -> bool:
        '''实现颈部错误判别'''
        tuck_threshold = 35

        if info['nose_shoulder_vertical_angle'] < tuck_threshold:
            return False
        elif info['nose_shoulder_vertical_angle'] > tuck_threshold:
            return True
        else:
            return False

    def _leg_shake(self, info: PullUpInfo) -> bool:
        '''实现腿部摇晃判别'''
        knee_threshold = 150
        ankle_threshold = 180

        if info['kneeRange'] < knee_threshold and info['ankleRange'] < ankle_threshold:
            return False
        elif info['kneeRange'] > knee_threshold or info['ankleRange'] > ankle_threshold:
            return True
        else : return False

    def _core_not_tighten(self, info: PullUpInfo) -> bool:
        '''实现核心不稳定判别'''
        threshold = 120
        if info['hipBoneRange'] < threshold:
            return False
        elif info['hipBoneRange'] > threshold:
            return True
        else:
            return False
    
    def _plot_hands_hold_distance(self, info: PullUpInfo, frame_idx: int, frame: np.array) -> np.array:
        raw_keypoints = info["raw_keypoints"]
        keypoints = Keypoints(raw_keypoints[frame_idx])
        l_wrist = keypoints.get_int("l_wrist")
        r_wrist = keypoints.get_int("r_wrist")
        l_shoulder = keypoints.get_int("l_shoulder")
        r_shoulder = keypoints.get_int("r_shoulder")
        cv2.line(frame, l_wrist, r_wrist, color=(255, 0, 0), thickness=2)
        cv2.line(frame, l_shoulder, r_shoulder, color=(255, 0, 0), thickness=2)

        if PLOT:
            frame_filename = f"unreasonable_leg_folding_angle.jpg"
            cv2.imwrite(frame_filename, frame)

        return frame

    def _plot_elbow(self, info: PullUpInfo, frame_idx: int, frame: np.array) -> np.array:
        raw_keypoints = info["raw_keypoints"]
        keypoints = Keypoints(raw_keypoints[frame_idx])
        l_wrist = keypoints.get_int("l_wrist")
        r_wrist = keypoints.get_int("r_wrist")
        l_elbow = keypoints.get_int("l_elbow")
        r_elbow = keypoints.get_int("r_elbow")
        l_shoulder = keypoints.get_int("l_shoulder")
        r_shoulder = keypoints.get_int("r_shoulder")
        
        overlay = frame.copy()
        cv2.circle(overlay, l_elbow, 50, color = (255, 0, 0), thickness=-1)
        cv2.circle(overlay, r_elbow, 50, color = (255, 0, 0), thickness=-1)
        cv2.addWeighted(overlay, 0.5, frame, 1 - 0.5, 0, frame)

        cv2.line(frame, l_wrist, l_elbow, color=(255, 0, 0), thickness=2)
        cv2.line(frame, r_wrist, r_elbow, color=(255, 0, 0), thickness=2)
        cv2.line(frame, l_elbow, l_shoulder, color=(255, 0, 0), thickness=2)
        cv2.line(frame, r_elbow, r_shoulder, color=(255, 0, 0), thickness=2)
        if PLOT:
            frame_filename = f"unreasonable_leg_folding_angle.jpg"
            cv2.imwrite(frame_filename, frame)

        return frame

    def _plot_Loose_shoulder_blades_in_freeFall(self, info: PullUpInfo, frame_idx: int, frame: np.array) -> np.array:
        raw_keypoints = info["raw_keypoints"]
        keypoints = Keypoints(raw_keypoints[frame_idx])
        l_wrist = keypoints.get_int("l_wrist")
        r_wrist = keypoints.get_int("r_wrist")
        l_elbow = keypoints.get_int("l_elbow")
        r_elbow = keypoints.get_int("r_elbow")
        l_shoulder = keypoints.get_int("l_shoulder")
        r_shoulder = keypoints.get_int("r_shoulder")
        l_angle = utils.three_points_angle(l_wrist, l_elbow, l_shoulder)
        r_angle = utils.three_points_angle(r_wrist, r_elbow, r_shoulder)
        cv2.line(frame, l_wrist, l_elbow, color=(255, 0, 0), thickness=2)
        cv2.line(frame, r_wrist, r_elbow, color=(255, 0, 0), thickness=2)
        cv2.putText(frame, str(int(l_angle)), l_elbow, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(frame, l_elbow, l_shoulder, color=(255, 0, 0), thickness=2)
        cv2.line(frame, r_elbow, r_shoulder, color=(255, 0, 0), thickness=2)
        cv2.putText(frame, str(int(r_angle)), r_elbow, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        if PLOT:
            frame_filename = f"unreasonable_leg_folding_angle.jpg"
            cv2.imwrite(frame_filename, frame)

        return frame

    def _plot_Leg_bending_angle(self, info: PullUpInfo, frame_idx: int, frame: np.array) -> np.array:
        raw_keypoints = info["raw_keypoints"]
        keypoints = Keypoints(raw_keypoints[frame_idx])
        l_knee = keypoints.get_int("l_knee")
        r_knee = keypoints.get_int("r_knee")
        l_ankle = keypoints.get_int("l_ankle")
        r_ankle = keypoints.get_int("r_ankle")
        cv2.line(frame, l_knee, r_knee, color=(255, 0, 0), thickness=2)
        cv2.line(frame, l_ankle, r_ankle, color=(255, 0, 0), thickness=2)

        if PLOT:
            frame_filename = f"unreasonable_leg_folding_angle.jpg"
            cv2.imwrite(frame_filename, frame)

        return frame

    def _plot_action_amplitude(self, info: PullUpInfo, frame_idx: int, frame: np.array) -> np.array:
        raw_keypoints = info["raw_keypoints"]
        keypoints = Keypoints(raw_keypoints[frame_idx])
        l_wrist = keypoints.get_int("l_wrist")
        r_wrist = keypoints.get_int("r_wrist")
        l_shoulder = keypoints.get_int("l_shoulder")
        r_shoulder = keypoints.get_int("r_shoulder")
        nose = keypoints.get_int("nose")
        mid_shoulder = ((l_shoulder[0] + r_shoulder[0]) // 2, (l_shoulder[1] + r_shoulder[1]) // 2)
        cv2.line(frame, l_wrist, r_wrist, color=(255, 0, 0), thickness=2)
        cv2.line(frame, l_shoulder, r_shoulder, color=(255, 0, 0), thickness=2)
        cv2.circle(frame, nose, 50, color=(255, 0, 0), thickness=-1)
        cv2.line(frame, mid_shoulder, nose, color=(255, 0, 0), thickness=2)

        if PLOT:
            frame_filename = f"unreasonable_leg_folding_angle.jpg"
            cv2.imwrite(frame_filename, frame)

        return frame

    def _plot_neck_error(self, info: PullUpInfo, frame_idx: int, frame: np.array) -> np.array:
        raw_keypoints = info["raw_keypoints"]
        keypoints = Keypoints(raw_keypoints[frame_idx])
        l_shoulder = keypoints.get_int("l_shoulder")
        r_shoulder = keypoints.get_int("r_shoulder")
        nose = keypoints.get_int("nose")
        angle = utils.three_points_angle(l_shoulder, nose, (l_shoulder[0], nose[1]))
        cv2.line(frame, l_shoulder, nose, color=(255, 0, 0), thickness=2)
        cv2.line(frame, r_shoulder, nose, color=(255, 0, 0), thickness=2)
        cv2.putText(frame, str(int(angle)), l_shoulder, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        if PLOT:
            frame_filename = f"unreasonable_leg_folding_angle.jpg"
            cv2.imwrite(frame_filename, frame)

        return frame

    def _plot_leg_shake(self, info: PullUpInfo, frame_idx: int, frame: np.array) -> np.array:
        return frame  # 至少两张图片

    def _plot_core_not_tighten(self, info: PullUpInfo, frame_idx: int, frame: np.array) -> np.array:
        return frame  # 至少两张图片

    def _frame_hands_hold_distance(self, info: PullUpInfo) -> int:
        """获取能佐证存在握距不合适的视频帧"""
        len = info["raw_data"].shape[0]
        return random.randint(len // 3, 2 * len // 3)

    def _frame_elbow(self, info: PullUpInfo) -> int:
        """获取能佐证存在手肘内收的视频帧"""
        return random.choice(info["mean_wrist_elbow_shoulder_angle_trough_idx"])

    def _frame_Loose_shoulder_blades_in_freeFall(self, info: PullUpInfo) -> int:
        """获取能佐证存在自由落体肩胛骨松懈的视频帧"""
        return random.choice(info["trough_body_idx"])

    def _frame_Leg_bending_angle(self, info: PullUpInfo) -> int:
        """获取能佐证存在腿部弯曲的视频帧"""
        len = info["raw_data"].shape[0]
        return random.randint(len // 3, 2 * len // 3)

    def _frame_action_amplitude(self, info: PullUpInfo) -> int:
        """获取能佐证存在动作幅度过小的视频帧"""
        return random.choice(info["peak_body_idx"])

    def _frame_neck_error(self, info: PullUpInfo) -> int:
        """获取能佐证存在颈部前倾的视频帧"""
        return random.choices(info["nose_shoulder_vertical_angle_peak_idx"])

    def _frame_leg_shake(self, info: PullUpInfo) -> int:
        """获取能佐证存在腿部摇晃的视频帧"""
        len = info["raw_data"].shape[0]
        return random.randint(len // 3, 2 * len // 3) # 至少两张图片

    def _frame_core_not_tighten(self, info: PullUpInfo) -> int:
        """获取能佐证存在核心不稳的视频帧"""
        len = info["raw_data"].shape[0]
        return random.randint(len // 3, 2 * len // 3) # 至少两张图片

def extract_points(points: torch.Tensor):
    """引体向上动作必要角度信息提取,提取到的单帧点信息,格式为包含点信息的元组"""
    if points.size(0) == 0: return (0, 0, 0, 0)
    keypoints = Keypoints(points)
    l_wrist_x = keypoints.get("l_wrist")[0]
    l_wrist_y = keypoints.get("l_wrist")[1]
    r_wrist_x = keypoints.get("r_wrist")[0]
    r_wrist_y = keypoints.get("r_wrist")[1]
    wrist_x_distance = abs(l_wrist_x - r_wrist_x)
    l_shoulder_x = keypoints.get("l_shoulder")[0]
    l_shoulder_y = keypoints.get("l_shoulder")[1]
    r_shoulder_x = keypoints.get("r_shoulder")[0]
    r_shoulder_y = keypoints.get("r_shoulder")[1]
    shoulder_x_distance = abs(l_shoulder_x - r_shoulder_x)
    l_hip_x = keypoints.get("l_hip")[0]
    l_hip_y = keypoints.get("l_hip")[1]
    r_hip_x = keypoints.get("r_hip")[0]
    r_hip_y = keypoints.get("r_hip")[1]
    l_knee_x = keypoints.get("l_knee")[0]
    l_knee_y = keypoints.get("l_knee")[1]
    r_knee_x = keypoints.get("r_knee")[0]
    r_knee_y = keypoints.get("r_knee")[1]
    l_ankle_x = keypoints.get("l_ankle")[0]
    l_ankle_y = keypoints.get("l_ankle")[1]
    r_ankle_x = keypoints.get("r_ankle")[0]
    r_ankle_y = keypoints.get("r_ankle")[1]
    shoulder_hip_y_distance = abs((l_shoulder_y + r_shoulder_y) - (l_hip_y + r_hip_y))/2
    wrist_shoulder_y_distance = abs((l_wrist_y + r_wrist_y) - (l_shoulder_y + r_shoulder_y))/2
    hip_x_distance = (l_hip_x + r_hip_x) / 2
    knee_x_distance = (l_knee_x + r_knee_x) / 2
    ankle_x_distance = (l_ankle_x + r_ankle_x) / 2

    l_wrist = keypoints.get("l_wrist")
    r_wrist = keypoints.get("r_wrist")
    l_elbow = keypoints.get("l_elbow")
    r_elbow = keypoints.get("r_elbow")
    l_shoulder = keypoints.get("l_shoulder")
    r_shoulder = keypoints.get("r_shoulder")
    nose = keypoints.get("nose")

    l_wrist_elbow_shoulder_angle = utils.three_points_angle(l_wrist,l_elbow,l_shoulder)
    r_wrist_elbow_shoulder_angle = utils.three_points_angle(r_wrist,r_elbow,r_shoulder)
    wrist_elbow_shoulder_angle = (l_wrist_elbow_shoulder_angle + r_wrist_elbow_shoulder_angle) / 2
    nose_shoulder_vertical_angle = utils.three_points_angle(nose, l_shoulder, (l_shoulder[0], nose[1]))
    return (l_wrist_y, r_wrist_y, wrist_x_distance, l_shoulder_y, r_shoulder_y, 
            shoulder_x_distance, l_knee_x, knee_x_distance, ankle_x_distance,
            l_knee_y, r_knee_x, r_knee_y, l_ankle_x, l_ankle_y, r_ankle_x, hip_x_distance,
            r_ankle_y, shoulder_hip_y_distance, wrist_shoulder_y_distance, wrist_elbow_shoulder_angle, 
            nose_shoulder_vertical_angle), ["l_wrist_y", "r_wrist_y", "wrist_x_distance", "l_shoulder_y", "r_shoulder_y",
                "shoulder_x_distance", "l_knee_x", "knee_x_distance", "ankle_x_distance", "l_knee_y", "r_knee_x", 
                "r_knee_y", "l_ankle_x", "l_ankle_y", "r_ankle_x", "hip_x_distance", "r_ankle_y", "shoulder_hip_y_distance", 
                "wrist_shoulder_y_distance", "wrist_elbow_shoulder_angle", "nose_shoulder_vertical_angle"]

# # use-case
# model = YOLO(r"E:\算法\项目管理\FitFormAI\model\yolov8n-pose.pt")
# pullup = PullUp(model)
# path = r"E:\算法\项目管理\FitFormAI\resource\引体向上\正侧面视角\手肘不合理\IMG_6086.MOV"
# pullup.do_analysis(path)
