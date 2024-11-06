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

class SetUpInfo(InfoBase):
    peak_angle_hip: float
    trough_angle_hip: float
    peak_back_ground_angle: float
    trough_back_ground_angle: float
    mean_angle_knee: float
    angle_hip_indices_of_peaks: List[int]
    angle_hip_indices_of_troughs: List[int]
    back_ground_angle_indices_of_peaks: List[int]
    back_ground_angle_indices_of_troughs: List[int]


class SetUp(TaskBase):
    def __init__(self, model: YOLO) -> None:
        super().__init__()
        self.task = "set_ups"
        self.error_list = ["unreasonable_leg_folding_angle", "shoulder_not_touch_with_cushion", 'elbows_not_touch_thighs', "waist_bounce", "hold_head_with_hands"]
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
        

    def _feature_extractor(self, yolo_outputs: list) -> SetUpInfo:
        """获取分析判断前所需要的所有特征信息"""
        frame_idx = 0
        data = []
        raw_keypoints = []
        for r in yolo_outputs:
            keypoints = utils.extract_main_person(r)
            # processing
            angles, labels = extract_angles(keypoints)
            data.append(angles)
            raw_keypoints.append(keypoints)

            frame_idx += 1
            
        raw_data = pd.DataFrame(data, columns=labels, index=list(range(1, len(data) + 1)))
        data_processor = DateProcessor(raw_data)

        #获取peak_angle_hip,trough_angle_hip
        processed_l_angle_hip = data_processor.process_wave_data("l_angle_hip")
        processed_r_angle_hip = data_processor.process_wave_data("r_angle_hip")
        peak_angle_hip = (processed_l_angle_hip["peak"] + processed_r_angle_hip["peak"])/2
        trough_angle_hip = (processed_l_angle_hip["trough"] + processed_r_angle_hip["trough"])/2
        #获取mean_angle_knee
        processed_l_angle_knee = data_processor.process_unwave_data('l_angle_knee')
        processed_r_angle_knee = data_processor.process_unwave_data('r_angle_knee')
        mean_angle_knee = (processed_l_angle_knee["mean"] + processed_r_angle_knee["mean"])/2
        #获取peak_back_ground_angle与trough_back_ground_angle
        processed_back_ground_angle = data_processor.process_wave_data("back_ground_angle")
        peak_back_ground_angle = processed_back_ground_angle["peak"]
        trough_back_ground_angle = processed_back_ground_angle["trough"]

        setup_info: SetUpInfo = {
            'raw_keypoints': raw_keypoints,
            "raw_data": raw_data,
            "peak_angle_hip": peak_angle_hip,
            "trough_angle_hip": trough_angle_hip,
            "peak_back_ground_angle": peak_back_ground_angle,
            "trough_back_ground_angle": trough_back_ground_angle,
            "mean_angle_knee": mean_angle_knee,
            "angle_hip_indices_of_peaks": processed_l_angle_hip["indices_of_peaks"],
            "angle_hip_indices_of_troughs": processed_l_angle_hip["indices_of_troughs"],
            "back_ground_angle_indices_of_peaks": processed_back_ground_angle["indices_of_peaks"],
            "back_ground_angle_indices_of_troughs": processed_back_ground_angle["indices_of_troughs"]
        }
        return setup_info



        
    def _unreasonable_leg_folding_angle(self, info: SetUpInfo) -> bool:
        """判断折腿角度是否合理"""
        peak_threshold = 90
        trough_threshold = 60

        if info["mean_angle_knee"] >= trough_threshold and info["mean_angle_knee"] <= peak_threshold:
            return False
        else:
            return True
        
    def _frame_unreasonable_leg_folding_angle(self, info: SetUpInfo) -> int:
        """获取能佐证折腿角度不合理的视频帧"""
        return random.choice(info["angle_hip_indices_of_troughs"])
    
    def _plot_unreasonable_leg_folding_angle(self, info: SetUpInfo, frame_idx: int, frame: np.array) -> np.array:
        """对能折腿角度不合理的视频帧进行绘制"""
        raw_keypoints = info["raw_keypoints"]
        keypoints = Keypoints(raw_keypoints[frame_idx])
        ankle = keypoints.get_int("ankle")
        knee = keypoints.get_int("knee")
        hip = keypoints.get_int("hip")
        angle = utils.three_points_angle(ankle, knee, hip)

        cv2.line(frame, ankle, knee, color=(255, 0, 0), thickness=2)
        cv2.line(frame, knee, hip, color=(255, 0, 0), thickness=2)
        cv2.putText(frame, str(int(angle)), knee, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

        for point in  [ankle, knee, hip]:
            cv2.circle(frame, point, radius=5, color=(0, 255, 0), thickness=-1)

        if PLOT:
            frame_filename = f"unreasonable_leg_folding_angle.jpg"
            cv2.imwrite(frame_filename, frame)

        return frame




    def _shoulder_not_touch_with_cushion(self, info: SetUpInfo) -> bool:
        """判断肩胛骨是否触垫"""
        trough_threshold = 5
        if info["trough_back_ground_angle"] >= trough_threshold:
            return True
        else:
            return False
        
    def _frame_shoulder_not_touch_with_cushion(self, info: SetUpInfo) -> int:
        """获取能佐证肩胛骨未触垫的视频帧"""
        return random.choice(info["back_ground_angle_indices_of_troughs"])
    
    def _plot_shoulder_not_touch_with_cushion(self, info: SetUpInfo, frame_idx: int, frame: np.array) -> np.array:
        """对能佐证肩胛骨未触垫的视频帧进行绘制"""
        raw_keypoints = info["raw_keypoints"]
        keypoints = Keypoints(raw_keypoints[frame_idx])
        shoulder = keypoints.get_int("shoulder")
        hip = keypoints.get_int("hip")
        ground = (shoulder[0], hip[1])
        angle = utils.three_points_angle(shoulder,hip,ground)

        cv2.line(frame, hip, shoulder, color=(255, 0, 0), thickness=2)
        cv2.line(frame, hip, ground, color=(255, 0, 0), thickness=2)
        cv2.putText(frame, str(int(angle)), hip, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

        for point in  [hip, ground, shoulder]:
            cv2.circle(frame, point, radius=5, color=(0, 255, 0), thickness=-1)

        frame_filename = f"shoulder_not_touch_with_cushion.jpg"
        cv2.imwrite(frame_filename, frame)

        return frame

        
    def _elbows_not_touch_thighs(self, info: SetUpInfo) -> bool:
        """判断双肘是否触及大腿"""
        trough_threshold = 70
        if info["trough_angle_hip"] >= trough_threshold:
            return True
        else:
            return False
        
    
    def _frame_elbows_not_touch_thighs(self, info: SetUpInfo) -> int:
        """获取能佐证双肘未触及大腿的视频帧"""
        return random.choice(info["angle_hip_indices_of_troughs"])
    
    def _plot_elbows_not_touch_thighs(self, info: SetUpInfo, frame_idx: int, frame: np.array) -> np.array:
        """对能佐证双肘未触及大腿的视频帧进行绘制"""
        raw_keypoints = info["raw_keypoints"]
        keypoints = Keypoints(raw_keypoints[frame_idx])
        shoulder = keypoints.get_int("shoulder")
        hip = keypoints.get_int("hip")
        knee = keypoints.get_int("knee")
        elbow = keypoints.get_int("elbow")
        droop_feet = utils.perpendicular_point_to_line(elbow, hip, knee)
        elbow_line_start = utils.translate_point_by_vector(knee, droop_feet, elbow)
        elbow_line_end = utils.translate_point_by_vector(hip, droop_feet, elbow)

        cv2.line(frame, hip, knee, color=(255, 0, 0), thickness=2)
        cv2.line(frame, droop_feet, elbow, color=(0, 255, 0), thickness=2)
        cv2.line(frame, elbow_line_start, elbow_line_end, color=(255, 255, 0), thickness=2)

        for point in  [hip, knee, elbow, droop_feet, elbow_line_start, elbow_line_end]:
            cv2.circle(frame, point, radius=5, color=(0, 255, 0), thickness=-1)

        if PLOT:
            frame_filename = f"elbows_not_touch_thighs.jpg"
            cv2.imwrite(frame_filename, frame)

        return frame


    def _waist_bounce(self, info: SetUpInfo)-> bool:
        """判断是否存在腰部弹震借力的情况"""
        has_unreasonable_leg_folding_angle = self._unreasonable_leg_folding_angle(info)
        if not has_unreasonable_leg_folding_angle:
            trough_threshold = 155
        else:
            trough_threshold = 165
        
        if info["peak_angle_hip"] >= trough_threshold:
            return True
        else:
            return False
            
    def _frame_waist_bounce(self, info: SetUpInfo) -> int:
        """获取能佐证存在腰部弹震借力的视频帧"""
        return random.choice(info["angle_hip_indices_of_peaks"])
    
    def _plot_waist_bounce(self, info: SetUpInfo, frame_idx: int, frame: np.array) -> np.array:
        """对能佐证存在腰部弹震借力的视频帧进行绘制"""
        raw_keypoints = info["raw_keypoints"]
        keypoints = Keypoints(raw_keypoints[frame_idx])
        shoulder = keypoints.get_int("shoulder")
        hip = keypoints.get_int("hip")
        knee = keypoints.get_int("knee")
        angle = utils.three_points_angle(shoulder, hip, knee)

        cv2.line(frame, hip, knee, color=(255, 0, 0), thickness=2)
        cv2.line(frame, hip, shoulder, color=(255, 0, 0), thickness=2)
        cv2.putText(frame, str(int(angle)), hip, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)


        for point in  [hip, knee, shoulder]:
            cv2.circle(frame, point, radius=5, color=(0, 255, 0), thickness=-1)

        if PLOT:
            frame_filename = f"waist_bounce.jpg"
            cv2.imwrite(frame_filename, frame)

        return frame
    
    def _hold_head_with_hands(self, info: SetUpInfo)-> bool:
        """判断是否存在双手扶头的情况"""
        trough_threshold = 40

        frame_idxs = info["angle_hip_indices_of_troughs"]
        distanses = []
        raw_keypoints = info["raw_keypoints"]
        for frame_idx in frame_idxs:
            keypoints = Keypoints(raw_keypoints[frame_idx])
            eye = keypoints.get_int("eye")
            wrist = keypoints.get_int("wrist")
            distanse = utils.euclidean_distance(eye, wrist)
            if not distanse == -1.0:
                distanses.append(distanse)

        if np.min(distanses) >= trough_threshold:
            return True
        else:
            return False

    def _frame_hold_head_with_hands(self, info: SetUpInfo) -> int:
        """获取能佐证存在双手扶头的视频帧"""
        return random.choice(info["angle_hip_indices_of_troughs"])
    
    def _plot_hold_head_with_hands(self, info: SetUpInfo, frame_idx: int, frame: np.array) -> np.array:
        """对能佐证存在双手扶头的视频帧进行绘制"""
        raw_keypoints = info["raw_keypoints"]
        keypoints = Keypoints(raw_keypoints[frame_idx])
        wrist = keypoints.get_int("wrist")

        overlay = frame.copy()
        cv2.circle(overlay, wrist,radius=50, color=(0, 255, 0, 128), thickness=cv2.FILLED)
        frame_with_circle = cv2.addWeighted(frame, 1, overlay, 0.5, 0)  

        if PLOT:
            frame_filename = f"hold_head_with_hands.jpg"
            cv2.imwrite(frame_filename, frame_with_circle)

        return frame_with_circle
    

# # use-case(in main.py)
# setup = SetUp(setup_model())
# setup.do_analysis("your-vedio-path")



def extract_angles(points: torch.Tensor):
    """仰卧起坐动作必要角度信息提取,返回格式为包含四个角度的元组(l_angle_knee, r_angle_knee, l_angle_hip, r_angle_hip, back_ground_angle, mean_angle_knee, mean_angle_hip)"""

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

    mean_angle_knee = (l_angle_knee + r_angle_knee) / 2
    mean_angle_hip = (l_angle_hip + r_angle_hip) / 2

    
    mid_hip = (int((l_hip[0] + r_hip[0]) // 2), int((l_hip[1] + r_hip[1]) // 2))
    mid_shoulder = (int((l_shoulder[0] + r_shoulder[0]) // 2), int((l_shoulder[1] + r_shoulder[1]) // 2))

    back_vector = (1, 0)
    ground_vector = ((mid_shoulder[0] - mid_hip[0]), (mid_shoulder[1] - mid_hip[1]))
    back_ground_angle = utils.two_vector_angle(back_vector, ground_vector)

    return (l_angle_knee, r_angle_knee, l_angle_hip, r_angle_hip, back_ground_angle, mean_angle_knee, mean_angle_hip), ["l_angle_knee", "r_angle_knee", "l_angle_hip", "r_angle_hip", "back_ground_angle", "mean_angle_knee", "mean_angle_hip"]



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
    mid_hip = (int((l_hip[0] + r_hip[0]) // 2), int((l_hip[1] + r_hip[1]) // 2))
    mid_shoulder = (int((l_shoulder[0] + r_shoulder[0]) // 2), int((l_shoulder[1] + r_shoulder[1]) // 2))

    back_ground_angle_place = ((mid_shoulder[0] + mid_hip[0]) // 2, (mid_shoulder[1] + mid_hip[1]) // 2)

    back_vector = (1, 0)
    ground_vector = ((mid_shoulder[0] - mid_hip[0]), (mid_shoulder[1] - mid_hip[1]))
    back_ground_angle = utils.two_vector_angle(back_vector, ground_vector)

    
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
        # dist = point_error(keypoints)
        # data = angles + dist
        data = angles
        csv_data.append(data)
            
        frame_idx += 1

    df = pd.DataFrame(csv_data, columns=['l_angle_knee', 'r_angle_knee', 'l_angle_hip', 'r_angle_hip', 'back_ground_angle', 'mean_angle_knee', 'mean_angle_hip'], index= list(range(1, frame_idx + 1)))
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
