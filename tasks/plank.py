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

PLOT = False  # 是否保存绘制的帧，用于调试

class PlankInfo(InfoBase):
    """
    存储平板支撑动作分析所需的特征信息
    """
    hip_y_mean: float  # 髋部 y 坐标平均值
    shoulder_y_mean: float  # 肩部 y 坐标平均值
    ankle_y_mean: float  # 脚踝 y 坐标平均值
    back_angle_mean: float  # 背部（肩-髋-踝）角度平均值
    raw_keypoints: List[torch.Tensor]  # 原始关键点数据
    raw_data: pd.DataFrame  # 原始数据 DataFrame

class Plank(TaskBase):
    def __init__(self, model: YOLO) -> None:
        super().__init__()
        self.task = "plank"  # 任务名称为平板支撑
        self.error_list = [
            "butt_sticking_up",  # 撅屁股
            "sagging_waist"      # 塌腰
        ]
        self.model = model  # YOLO 模型实例

    def do_analysis(self, input_path: str) -> List[ErrorDetail]:
        """
        对外暴露的接口函数，分析视频中的平板支撑动作

        Args:
            input_path (str): 视频文件路径

        Returns:
            List[ErrorDetail]: 错误详情列表，包含错误类型、建议和对应的帧
        """
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

    def _feature_extractor(self, yolo_outputs: list) -> PlankInfo:
        """
        提取分析判别所需的所有特征信息

        Args:
            yolo_outputs (list): YOLO 模型的输出列表

        Returns:
            PlankInfo: 包含特征信息的 PlankInfo 实例
        """
        frame_idx = 0
        data = []
        raw_keypoints = []
        for r in yolo_outputs:
            keypoints = utils.extract_main_person(r)
            # 如果未检测到关键点，跳过该帧
            if keypoints.size(0) == 0:
                continue
            # 提取关键点数据
            points, labels = extract_points(keypoints)
            data.append(points)
            raw_keypoints.append(keypoints)
            frame_idx += 1

        raw_data = pd.DataFrame(data, columns=labels, index=list(range(1, len(data) + 1)))
        data_processor = DateProcessor(raw_data)

        # 数据处理，计算各部位 y 坐标和背部角度的平均值
        processed_hip_y = data_processor.process_unwave_data("hip_y")
        processed_shoulder_y = data_processor.process_unwave_data("shoulder_y")
        processed_ankle_y = data_processor.process_unwave_data("ankle_y")
        processed_back_angle = data_processor.process_unwave_data("back_angle")

        plank_info: PlankInfo = {
            "hip_y_mean": processed_hip_y["mean"],
            "shoulder_y_mean": processed_shoulder_y["mean"],
            "ankle_y_mean": processed_ankle_y["mean"],
            "back_angle_mean": processed_back_angle["mean"],
            "raw_keypoints": raw_keypoints,
            "raw_data": raw_data
        }
        return plank_info

    def _butt_sticking_up(self, info: PlankInfo) -> bool:
        """
        判断是否存在撅屁股的错误

        Args:
            info (PlankInfo): 包含特征信息的 PlankInfo 实例

        Returns:
            bool: 存在错误返回 True，否则返回 False
        """
        # 撅屁股时，髋部高于肩部一定阈值
        threshold = -10  # 阈值，负值表示髋部在肩部之上，可根据实际情况调整
        if info["hip_y_mean"] < info["shoulder_y_mean"] + threshold:
            return True
        else:
            return False

    def _sagging_waist(self, info: PlankInfo) -> bool:
        """
        判断是否存在塌腰的错误

        Args:
            info (PlankInfo): 包含特征信息的 PlankInfo 实例

        Returns:
            bool: 存在错误返回 True，否则返回 False
        """
        # 塌腰时，髋部低于肩部与踝部连线
        threshold = 10  # 阈值，正值表示髋部在连线之下，可根据实际情况调整
        if info["hip_y_mean"] > ((info["shoulder_y_mean"] + info["ankle_y_mean"]) / 2) + threshold:
            return True
        else:
            return False

    def _plot_butt_sticking_up(self, info: PlankInfo, frame_idx: int, frame: np.array) -> np.array:
        """
        对存在撅屁股错误的帧进行绘制

        Args:
            info (PlankInfo): 包含特征信息的 PlankInfo 实例
            frame_idx (int): 帧索引
            frame (np.array): 帧图像

        Returns:
            np.array: 绘制后的帧图像
        """
        raw_keypoints = info["raw_keypoints"]
        keypoints = Keypoints(raw_keypoints[frame_idx])
        shoulder = keypoints.get_int("shoulder")
        hip = keypoints.get_int("hip")
        ankle = keypoints.get_int("ankle")

        # 绘制肩-髋-踝的线条
        cv2.line(frame, shoulder, hip, (0, 255, 0), 2)
        cv2.line(frame, hip, ankle, (0, 255, 0), 2)

        # 计算背部角度
        angle = utils.three_points_angle(shoulder, hip, ankle)
        cv2.putText(frame, f"Angle: {int(angle)}", hip, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if PLOT:
            frame_filename = f"butt_sticking_up_{frame_idx}.jpg"
            cv2.imwrite(frame_filename, frame)

        return frame

    def _plot_sagging_waist(self, info: PlankInfo, frame_idx: int, frame: np.array) -> np.array:
        """
        对存在塌腰错误的帧进行绘制

        Args:
            info (PlankInfo): 包含特征信息的 PlankInfo 实例
            frame_idx (int): 帧索引
            frame (np.array): 帧图像

        Returns:
            np.array: 绘制后的帧图像
        """
        raw_keypoints = info["raw_keypoints"]
        keypoints = Keypoints(raw_keypoints[frame_idx])
        shoulder = keypoints.get_int("shoulder")
        hip = keypoints.get_int("hip")
        ankle = keypoints.get_int("ankle")

        # 绘制肩-髋-踝的线条
        cv2.line(frame, shoulder, hip, (0, 0, 255), 2)
        cv2.line(frame, hip, ankle, (0, 0, 255), 2)

        # 计算背部角度
        angle = utils.three_points_angle(shoulder, hip, ankle)
        cv2.putText(frame, f"Angle: {int(angle)}", hip, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if PLOT:
            frame_filename = f"sagging_waist_{frame_idx}.jpg"
            cv2.imwrite(frame_filename, frame)

        return frame

    def _frame_butt_sticking_up(self, info: PlankInfo) -> int:
        """
        获取能佐证存在撅屁股错误的帧索引

        Args:
            info (PlankInfo): 包含特征信息的 PlankInfo 实例

        Returns:
            int: 帧索引
        """
        total_frames = len(info["raw_keypoints"])
        # 选择中间位置的帧，假设错误可能出现在此处
        return total_frames // 2

    def _frame_sagging_waist(self, info: PlankInfo) -> int:
        """
        获取能佐证存在塌腰错误的帧索引

        Args:
            info (PlankInfo): 包含特征信息的 PlankInfo 实例

        Returns:
            int: 帧索引
        """
        total_frames = len(info["raw_keypoints"])
        # 选择中间位置的帧，假设错误可能出现在此处
        return total_frames // 2


def side_video2csv(input_path: str, output_path: str, model: YOLO, **keywarg: any) -> None:
    """
    使用YOLO处理平板支撑侧面视角视频，并将分析结果以CSV格式存入指定文件夹

    Args:
        input_path (str): 输入视频地址
        output_path (str): 输出CSV地址
    """
    results = model(source=input_path, stream=True, **keywarg)
    frame_idx = 0
    csv_data = []
    for r in results:
        frame_idx += 1
        keypoints = utils.extract_main_person(r)
        if keypoints.size(0) == 0:
            continue
        points, labels = extract_points(keypoints)
        csv_data.append(points)

    df = pd.DataFrame(csv_data, columns=labels, index= list(range(1, len(csv_data) + 1)))
    df.to_csv(output_path, index_label='idx')

def side_video2video_(frame: np.array, points: torch.Tensor) -> np.array:
    """
    对每一视频帧进行处理，在帧上绘制角度和关键点信息

    Args:
        frame (np.array): 原视频帧
        points (torch.Tensor): 骨架关键点集合

    Returns:
        np.array: 处理后的视频帧
    """
    # 绘制关键点
    frame = utils.show_keypoints(frame, points)
    # 绘制角度信息
    frame = plot_angles(frame, points)
    return frame

def side_video2video(input_path: str, output_path: str, model: YOLO, **keywarg: any) -> None:
    """
    使用YOLO处理平板支撑侧面视角视频，添加便于直观感受的特征展示，并将结果以视频格式存入指定文件夹

    Args:
        input_path (str): 输入视频地址
        output_path (str): 输出视频地址
    """
    utils.video2video_base_(side_video2video_, input_path, output_path, model, **keywarg)


def extract_points(points: torch.Tensor) -> Tuple[Tuple[float, float, float, float], List[str]]:
    """
    提取单帧必要的关键点信息，包括肩部、髋部、脚踝的 y 坐标和背部角度

    Args:
        points (torch.Tensor): 单帧关键点张量

    Returns:
        Tuple[Tuple[float, float, float, float], List[str]]: 关键点数据和对应的标签
    """
    keypoints = Keypoints(points)
    # 获取左右肩的 y 坐标并求平均
    l_shoulder_y = keypoints.get("l_shoulder")[1]
    r_shoulder_y = keypoints.get("r_shoulder")[1]
    shoulder_y = (l_shoulder_y + r_shoulder_y) / 2

    # 获取左右髋的 y 坐标并求平均
    l_hip_y = keypoints.get("l_hip")[1]
    r_hip_y = keypoints.get("r_hip")[1]
    hip_y = (l_hip_y + r_hip_y) / 2

    # 获取左右踝的 y 坐标并求平均
    l_ankle_y = keypoints.get("l_ankle")[1]
    r_ankle_y = keypoints.get("r_ankle")[1]
    ankle_y = (l_ankle_y + r_ankle_y) / 2

    # 获取肩、髋、踝的位置，用于计算背部角度
    shoulder = keypoints.get("shoulder")
    hip = keypoints.get("hip")
    ankle = keypoints.get("ankle")
    back_angle = utils.three_points_angle(shoulder, hip, ankle)

    return (shoulder_y, hip_y, ankle_y, back_angle), ["shoulder_y", "hip_y", "ankle_y", "back_angle"]


def plot_angles(frame: np.array ,points: torch.Tensor) -> np.array:
    """
    在视频帧上绘制角度信息

    Args:
        frame (np.array): 原视频帧
        points (torch.Tensor): 关键点张量

    Returns:
        np.array: 绘制后的帧
    """
    keypoints = Keypoints(points)

    # 获取关键点位置
    l_shoulder = keypoints.get_int("l_shoulder")
    r_shoulder = keypoints.get_int("r_shoulder")
    l_hip = keypoints.get_int("l_hip")
    r_hip = keypoints.get_int("r_hip")
    l_ankle = keypoints.get_int("l_ankle")
    r_ankle = keypoints.get_int("r_ankle")

    shoulder = ((l_shoulder[0] + r_shoulder[0]) // 2, (l_shoulder[1] + r_shoulder[1]) // 2)
    hip = ((l_hip[0] + r_hip[0]) // 2, (l_hip[1] + r_hip[1]) // 2)
    ankle = ((l_ankle[0] + r_ankle[0]) // 2, (l_ankle[1] + r_ankle[1]) // 2)

    # 绘制骨架线
    cv2.line(frame, shoulder, hip, (0, 255, 0), 2)
    cv2.line(frame, hip, ankle, (0, 255, 0), 2)

    # 绘制关键点
    for point in [shoulder, hip, ankle]:
        cv2.circle(frame, point, 5, (0, 0, 255), -1)

    # 计算背部角度
    angle = utils.three_points_angle(shoulder, hip, ankle)
    angle_text = f"Back Angle: {angle:.2f}"
    cv2.putText(frame, angle_text, hip, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return frame

if __name__ == "__main__":
    model = YOLO("/home/shyang/code/FitFormAI_Analysiser/model/yolov8x-pose.pt")
    plank = Plank(model)
    path = "/home/shyang/code/FitFormAI_Analysiser/resource/平板支撑/侧面视角/塌腰/0.mp4"
    results = plank.do_analysis(path)
    print(results)
