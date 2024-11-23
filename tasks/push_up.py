import cv2
import pandas as pd
from ultralytics import YOLO
import utils
import torch
from typing import Tuple, List
import numpy as np
from keypoints import Keypoints
from task_base import TaskBase, InfoBase, ErrorDetail
from data_processor import DateProcessor
from advices import load_advice_by_filename

PLOT = True  # 是否保存绘制的帧，用于调试

class PushUpInfo(InfoBase):
    """
    存储俯卧撑动作分析所需的特征信息
    """
    # 定义需要的特征字段
    hip_y_mean: float  # 髋部 y 坐标平均值
    shoulder_y_mean: float  # 肩部 y 坐标平均值
    elbow_angle_mean: float  # 肘关节角度平均值
    back_angle_mean: float  # 背部角度平均值
    wrist_shoulder_distance_mean: float  # 手腕与肩部的水平距离平均值
    arm_torso_angle_mean: float  # 上臂与躯干的夹角平均值
    forearm_vertical_angle_mean: float  # 小臂与垂直方向的夹角平均值
    raw_keypoints: List[torch.Tensor]  # 原始关键点数据
    raw_data: pd.DataFrame  # 原始数据 DataFrame

class PushUp(TaskBase):
    def __init__(self, model: YOLO, view: str = 'side') -> None:
        """
        初始化 PushUp 类

        Args:
            model (YOLO): YOLO 模型实例
            view (str): 视角，'side' 表示侧面视角，'front' 表示正面视角
        """
        super().__init__()
        self.task = "push_up"  # 任务名称为俯卧撑
        self.view = view  # 视角
        # 根据视角定义错误列表
        if self.view == 'side':
            self.error_list = [
                "not_in_straight_line",  # 不成直线
                "insufficient_amplitude",  # 幅度不够
                "arms_extended_forward"  # 手臂前伸
            ]
        elif self.view == 'front':
            self.error_list = [
                "insufficient_amplitude",  # 幅度不够
                "arms_abduction",  # 手臂外展
                "forearm_not_vertical"  # 小臂不垂直
            ]
        else:
            raise ValueError("视角参数错误，应为 'side' 或 'front'")
        self.model = model  # YOLO 模型实例

    def do_analysis(self, input_path: str) -> List[ErrorDetail]:
        """
        分析俯卧撑视频，返回错误详情

        Args:
            input_path (str): 视频文件路径

        Returns:
            List[ErrorDetail]: 错误详情列表
        """
        results: List[ErrorDetail] = []
        yolo_outputs = self.model(source=input_path, stream=True)
        info = self._feature_extractor(yolo_outputs)
        error_list = self._judge_error(info)
        print(error_list)
        frame_idxs = self._frame_idx_extractor(error_list, info)
        print(frame_idxs)
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

    def _feature_extractor(self, yolo_outputs: list) -> PushUpInfo:
        """
        提取分析判别所需的所有特征信息

        Args:
            yolo_outputs (list): YOLO 模型的输出列表

        Returns:
            PushUpInfo: 包含特征信息的 PushUpInfo 实例
        """
        frame_idx = 0
        data = []
        raw_keypoints = []
        for r in yolo_outputs:
            keypoints = utils.extract_main_person(r)
            if keypoints.size(0) == 0:
                continue
            points, labels = extract_points(keypoints, self.view)
            data.append(points)
            raw_keypoints.append(keypoints)
            frame_idx += 1

        raw_data = pd.DataFrame(data, columns=labels, index=list(range(len(data))))
        data_processor = DateProcessor(raw_data)

        if self.view == 'side':
            processed_hip_y = data_processor.process_unwave_data("hip_y")
            processed_shoulder_y = data_processor.process_unwave_data("shoulder_y")
            processed_elbow_angle = data_processor.process_wave_data("elbow_angle")
            processed_back_angle = data_processor.process_unwave_data("back_angle")
            processed_wrist_shoulder_distance = data_processor.process_unwave_data("wrist_shoulder_distance")

            pushup_info: PushUpInfo = {
                "hip_y_mean": processed_hip_y["mean"],
                "shoulder_y_mean": processed_shoulder_y["mean"],
                "elbow_angle_mean": processed_elbow_angle["mean"],
                "back_angle_mean": processed_back_angle["mean"],
                "wrist_shoulder_distance_mean": processed_wrist_shoulder_distance["mean"],
                "raw_keypoints": raw_keypoints,
                "raw_data": raw_data
            }
        elif self.view == 'front':
            processed_elbow_angle = data_processor.process_wave_data("elbow_angle")
            processed_arm_torso_angle = data_processor.process_unwave_data("arm_torso_angle")
            processed_forearm_vertical_angle = data_processor.process_unwave_data("forearm_vertical_angle")

            pushup_info: PushUpInfo = {
                "elbow_angle_mean": processed_elbow_angle["mean"],
                "arm_torso_angle_mean": processed_arm_torso_angle["mean"],
                "forearm_vertical_angle_mean": processed_forearm_vertical_angle["mean"],
                "raw_keypoints": raw_keypoints,
                "raw_data": raw_data
            }
        else:
            raise ValueError("视角参数错误，应为 'side' 或 'front'")

        return pushup_info

    def _not_in_straight_line(self, info: PushUpInfo) -> bool:
        """
        判断是否存在不成直线的错误

        Args:
            info (PushUpInfo): 特征信息

        Returns:
            bool: 是否存在错误
        """
        threshold = 10  # 阈值，角度偏差超过10度
        if abs(info["back_angle_mean"] - 180) > threshold:
            return True
        else:
            return False

    def _insufficient_amplitude(self, info: PushUpInfo) -> bool:
        """
        判断动作幅度是否不足

        Args:
            info (PushUpInfo): 特征信息

        Returns:
            bool: 是否存在错误
        """
        threshold = 90  # 阈值，肘关节角度大于90度
        if info["elbow_angle_mean"] > threshold:
            return True
        else:
            return False

    def _arms_extended_forward(self, info: PushUpInfo) -> bool:
        """
        判断是否存在手臂前伸的错误

        Args:
            info (PushUpInfo): 特征信息

        Returns:
            bool: 是否存在错误
        """
        threshold = 20  # 阈值
        if info["wrist_shoulder_distance_mean"] > threshold:
            return True
        else:
            return False

    def _arms_abduction(self, info: PushUpInfo) -> bool:
        """
        判断是否存在手臂外展的错误

        Args:
            info (PushUpInfo): 特征信息

        Returns:
            bool: 是否存在错误
        """
        threshold = 1.5  # 阈值，手之间的距离与肩膀之间的距离比值超过1.5视为手臂外展
        wrist_distance = info["raw_data"]["wrist_distance"]
        shoulder_distance = info["raw_data"]["shoulder_distance"]
        ratio = wrist_distance / shoulder_distance
        if ratio.max() > threshold:
            return True
        else:
            return False

    def _forearm_not_vertical(self, info: PushUpInfo) -> bool:
        """
        判断小臂是否未垂直

        Args:
            info (PushUpInfo): 特征信息

        Returns:
            bool: 是否存在错误
        """
        threshold = 15  # 阈值，偏差超过15度
        if abs(info["forearm_vertical_angle_mean"] - 0) > threshold:
            return True
        else:
            return False

    def _judge_error(self, info: PushUpInfo) -> List[str]:
        """
        判断错误类型

        Args:
            info (PushUpInfo): 特征信息

        Returns:
            List[str]: 错误类型列表
        """
        errors = []
        if self.view == 'side':
            if self._not_in_straight_line(info):
                errors.append("not_in_straight_line")
            if self._insufficient_amplitude(info):
                errors.append("insufficient_amplitude")
            if self._arms_extended_forward(info):
                errors.append("arms_extended_forward")
        elif self.view == 'front':
            if self._insufficient_amplitude(info):
                errors.append("insufficient_amplitude")
            if self._arms_abduction(info):
                errors.append("arms_abduction")
            if self._forearm_not_vertical(info):
                errors.append("forearm_not_vertical")
        return errors

    # 以下为绘制错误帧的方法
    def _plot_not_in_straight_line(self, info: PushUpInfo, frame_idx: int, frame: np.array) -> np.array:
        """
        绘制不成直线错误的帧

        Args:
            info (PushUpInfo): 特征信息
            frame_idx (int): 帧索引
            frame (np.array): 图像帧

        Returns:
            np.array: 绘制后的帧
        """
        keypoints = Keypoints(info["raw_keypoints"][frame_idx])
        # 获取关键点
        shoulder = keypoints.get_int("l_shoulder")
        hip = keypoints.get_int("l_hip")
        ankle = keypoints.get_int("l_ankle")
        # 绘制线条
        cv2.line(frame, shoulder, hip, (0, 255, 0), 2)
        cv2.line(frame, hip, ankle, (0, 255, 0), 2)
        # 计算角度
        angle = utils.three_points_angle(shoulder, hip, ankle)
        cv2.putText(frame, f"Back Angle: {int(angle)}", hip, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if PLOT:
            frame_filename = f"not_in_straight_line_{frame_idx}.jpg"
            cv2.imwrite(frame_filename, frame)
        
        return frame

    def _plot_insufficient_amplitude(self, info: PushUpInfo, frame_idx: int, frame: np.array) -> np.array:
        """
        绘制动作幅度不足的帧

        Args:
            info (PushUpInfo): 特征信息
            frame_idx (int): 帧索引
            frame (np.array): 图像帧

        Returns:
            np.array: 绘制后的帧
        """
        keypoints = Keypoints(info["raw_keypoints"][frame_idx])
        # 获取关键点
        l_shoulder = keypoints.get_int("l_shoulder")
        l_elbow = keypoints.get_int("l_elbow")
        l_wrist = keypoints.get_int("l_wrist")
        # 绘制线条
        cv2.line(frame, l_shoulder, l_elbow, (0, 255, 0), 2)
        cv2.line(frame, l_elbow, l_wrist, (0, 255, 0), 2)
        # 计算角度
        angle = utils.three_points_angle(l_shoulder, l_elbow, l_wrist)
        cv2.putText(frame, f"Elbow Angle: {int(angle)}", l_elbow, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if PLOT:
            frame_filename = f"insufficient_amplitude_{frame_idx}.jpg"
            cv2.imwrite(frame_filename, frame)
        
        return frame

    def _plot_arms_extended_forward(self, info: PushUpInfo, frame_idx: int, frame: np.array) -> np.array:
        """
        绘制手臂前伸的帧

        Args:
            info (PushUpInfo): 特征信息
            frame_idx (int): 帧索引
            frame (np.array): 图像帧

        Returns:
            np.array: 绘制后的帧
        """
        keypoints = Keypoints(info["raw_keypoints"][frame_idx])
        # 获取关键点
        l_shoulder = keypoints.get_int("l_shoulder")
        l_wrist = keypoints.get_int("l_wrist")
        # 绘制线条
        cv2.line(frame, l_shoulder, l_wrist, (0, 255, 0), 2)
        # 计算距离
        distance = abs(l_wrist[0] - l_shoulder[0])
        cv2.putText(frame, f"Distance: {int(distance)}", l_wrist, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if PLOT:
            frame_filename = f"arms_extended_forward_{frame_idx}.jpg"
            cv2.imwrite(frame_filename, frame)
        
        return frame

    def _plot_arms_abduction(self, info: PushUpInfo, frame_idx: int, frame: np.array) -> np.array:
        """
        绘制手臂外展的帧

        Args:
            info (PushUpInfo): 特征信息
            frame_idx (int): 帧索引
            frame (np.array): 图像帧

        Returns:
            np.array: 绘制后的帧
        """
        keypoints = Keypoints(info["raw_keypoints"][frame_idx])
        # 获取关键点
        l_shoulder = keypoints.get_int("l_shoulder")
        r_shoulder = keypoints.get_int("r_shoulder")
        l_wrist = keypoints.get_int("l_wrist")
        r_wrist = keypoints.get_int("r_wrist")
        # 绘制线条
        cv2.line(frame, l_shoulder, r_shoulder, (0, 255, 0), 2)
        cv2.line(frame, l_wrist, r_wrist, (0, 255, 0), 2)
        # 计算距离比值
        wrist_distance = np.linalg.norm(np.array(l_wrist) - np.array(r_wrist))
        shoulder_distance = np.linalg.norm(np.array(l_shoulder) - np.array(r_shoulder))
        ratio = wrist_distance / shoulder_distance
        cv2.putText(frame, f"Ratio: {ratio:.2f}", ((l_wrist[0] + r_wrist[0]) // 2, (l_wrist[1] + r_wrist[1]) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if PLOT:
            frame_filename = f"arms_abduction_{frame_idx}.jpg"
            cv2.imwrite(frame_filename, frame)
        
        return frame

    def _plot_forearm_not_vertical(self, info: PushUpInfo, frame_idx: int, frame: np.array) -> np.array:
        """
        绘制小臂未垂直的帧

        Args:
            info (PushUpInfo): 特征信息
            frame_idx (int): 帧索引
            frame (np.array): 图像帧

        Returns:
            np.array: 绘制后的帧
        """
        keypoints = Keypoints(info["raw_keypoints"][frame_idx])
        # 获取关键点
        l_elbow = keypoints.get_int("l_elbow")
        l_wrist = keypoints.get_int("l_wrist")
        # 绘制线条
        cv2.line(frame, l_elbow, l_wrist, (0, 255, 0), 2)
        # 计算角度
        vertical_direction = (0, -1)
        forearm_vector = (l_wrist[0] - l_elbow[0], l_wrist[1] - l_elbow[1])
        angle = utils.angle_between_vectors(forearm_vector, vertical_direction)
        cv2.putText(frame, f"Forearm Angle: {int(angle)}", l_elbow, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if PLOT:
            frame_filename = f"forearm_not_vertical_{frame_idx}.jpg"
            cv2.imwrite(frame_filename, frame)
        
        return frame

    # 以下为获取帧索引的方法
    def _frame_not_in_straight_line(self, info: PushUpInfo) -> int:
        """
        获取不成直线错误对应的帧索引

        Args:
            info (PushUpInfo): 特征信息

        Returns:
            int: 帧索引
        """
        # 从背部角度数据中找到偏离直线的帧索引
        back_angles = info["raw_data"]["back_angle"]
        threshold = 10  # 与180度的偏差超过10度视为不成直线
        deviation = abs(back_angles - 180)
        indices = deviation[deviation > threshold].index.tolist()
        if indices:
            # 返回偏差最大的帧索引
            max_deviation_idx = deviation.idxmax()
            return max_deviation_idx
        else:
            # 若未找到，则返回中间帧索引
            return len(info["raw_keypoints"]) // 2

    def _frame_insufficient_amplitude(self, info: PushUpInfo) -> int:
        """
        获取动作幅度不足错误对应的帧索引

        Args:
            info (PushUpInfo): 特征信息

        Returns:
            int: 帧索引
        """
        # 从肘关节角度数据中找到幅度不足的帧索引
        elbow_angles = info["raw_data"]["elbow_angle"]
        threshold = 90  # 肘关节角度大于90度视为幅度不足
        indices = elbow_angles[elbow_angles > threshold].index.tolist()
        if indices:
            # 返回肘关节角度最大的帧索引
            max_angle_idx = elbow_angles.idxmax()
            return max_angle_idx
        else:
            # 若未找到，则返回中间帧索引
            return len(info["raw_keypoints"]) // 2

    def _frame_arms_extended_forward(self, info: PushUpInfo) -> int:
        """
        获取手臂前伸错误对应的帧索引

        Args:
            info (PushUpInfo): 特征信息

        Returns:
            int: 帧索引
        """
        # 从手腕与肩部的水平距离中找到手臂前伸的帧索引
        distances = info["raw_data"]["wrist_shoulder_distance"]
        threshold = 20  # 水平距离超过阈值视为手臂前伸
        indices = distances[distances > threshold].index.tolist()
        if indices:
            # 返回距离最大的帧索引
            max_distance_idx = distances.idxmax()
            return max_distance_idx
        else:
            # 若未找到，则返回中间帧索引
            return len(info["raw_keypoints"]) // 2

    def _frame_arms_abduction(self, info: PushUpInfo) -> int:
        """
        获取手臂外展错误对应的帧索引

        Args:
            info (PushUpInfo): 特征信息

        Returns:
            int: 帧索引
        """
        # 从手腕和肩膀之间的距离数据中找到手臂外展的帧索引
        wrist_distances = info["raw_data"]["wrist_distance"]
        shoulder_distances = info["raw_data"]["shoulder_distance"]
        ratios = wrist_distances / shoulder_distances
        threshold = 1.5  # 比值超过1.5视为手臂外展
        # ratios > threshold and ratios < 5
        indices = ratios[(ratios > threshold) & (ratios < 5)].index.tolist()
        if indices:
            # 返回比值在 5 和 1.5 最大的帧索引
            max_ratio_idx = ratios[(ratios > threshold) & (ratios < 5)].idxmax()
            return max_ratio_idx
        else:
            # 若未找到，则返回中间帧索引
            return len(info["raw_keypoints"]) // 2

    def _frame_forearm_not_vertical(self, info: PushUpInfo) -> int:
        """
        获取小臂未垂直错误对应的帧索引

        Args:
            info (PushUpInfo): 特征信息

        Returns:
            int: 帧索引
        """
        # 从小臂与垂直方向的夹角数据中找到未垂直的帧索引
        forearm_angles = info["raw_data"]["forearm_vertical_angle"]
        threshold = 15  # 偏差超过15度视为小臂未垂直
        deviation = abs(forearm_angles - 0)
        indices = deviation[deviation > threshold].index.tolist()
        if indices:
            # 返回偏差最大的帧索引
            max_deviation_idx = deviation.idxmax()
            return max_deviation_idx
        else:
            # 若未找到，则返回中间帧索引
            return len(info["raw_keypoints"]) // 2
    

def extract_points(points: torch.Tensor, view: str) -> Tuple[List[float], List[str]]:
    """
    提取关键点信息

    Args:
        points (torch.Tensor): 关键点张量
        view (str): 视角

    Returns:
        Tuple[List[float], List[str]]: 数据和标签
    """
    keypoints = Keypoints(points)
    data = []
    labels = []

    if view == 'side':
        # 获取必要的关键点
        l_shoulder = keypoints.get("l_shoulder")
        l_elbow = keypoints.get("l_elbow")
        l_wrist = keypoints.get("l_wrist")
        l_hip = keypoints.get("l_hip")
        l_ankle = keypoints.get("l_ankle")
        # 计算肘关节角度
        elbow_angle = utils.three_points_angle(l_shoulder, l_elbow, l_wrist)
        data.append(elbow_angle)
        labels.append("elbow_angle")
        # 计算背部角度
        back_angle = utils.three_points_angle(l_shoulder, l_hip, l_ankle)
        data.append(back_angle)
        labels.append("back_angle")
        # 计算手腕与肩部的水平距离
        wrist_shoulder_distance = abs(l_wrist[0] - l_shoulder[0])
        data.append(wrist_shoulder_distance)
        labels.append("wrist_shoulder_distance")
        # 获取肩部、髋部 y 坐标
        shoulder_y = l_shoulder[1]
        hip_y = l_hip[1]
        data.extend([shoulder_y, hip_y])
        labels.extend(["shoulder_y", "hip_y"])
    elif view == 'front':
        # 获取必要的关键点
        l_shoulder = keypoints.get("l_shoulder")
        r_shoulder = keypoints.get("r_shoulder")
        l_elbow = keypoints.get("l_elbow")
        r_elbow = keypoints.get("r_elbow")
        l_wrist = keypoints.get("l_wrist")
        r_wrist = keypoints.get("r_wrist")
        l_hip = keypoints.get("l_hip")
        r_hip = keypoints.get("r_hip")
        # 计算肘关节角度
        l_elbow_angle = utils.three_points_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = utils.three_points_angle(r_shoulder, r_elbow, r_wrist)
        elbow_angle = (l_elbow_angle + r_elbow_angle) / 2
        data.append(elbow_angle)
        labels.append("elbow_angle")
        # 计算上臂与躯干的夹角
        l_arm_torso_angle = utils.three_points_angle(l_elbow, l_shoulder, l_hip)
        r_arm_torso_angle = utils.three_points_angle(r_elbow, r_shoulder, r_hip)
        arm_torso_angle = (l_arm_torso_angle + r_arm_torso_angle) / 2
        data.append(arm_torso_angle)
        labels.append("arm_torso_angle")
        # 计算小臂与垂直方向的夹角
        vertical_direction = (0, -1)
        l_forearm_vector = (l_wrist[0] - l_elbow[0], l_wrist[1] - l_elbow[1])
        r_forearm_vector = (r_wrist[0] - r_elbow[0], r_wrist[1] - r_elbow[1])
        l_forearm_angle = utils.angle_between_vectors(l_forearm_vector, vertical_direction)
        r_forearm_angle = utils.angle_between_vectors(r_forearm_vector, vertical_direction)
        forearm_vertical_angle = (l_forearm_angle + r_forearm_angle) / 2
        data.append(forearm_vertical_angle)
        labels.append("forearm_vertical_angle")
        # 计算手腕和肩膀之间的距离
        wrist_distance = np.linalg.norm(np.array(l_wrist) - np.array(r_wrist))
        shoulder_distance = np.linalg.norm(np.array(l_shoulder) - np.array(r_shoulder))
        data.extend([wrist_distance, shoulder_distance])
        labels.extend(["wrist_distance", "shoulder_distance"])
    else:
        raise ValueError("视角参数错误，应为 'side' 或 'front'")

    return data, labels

def side_video2csv(input_path: str, output_path: str, model: YOLO, **kwargs) -> None:
    """
    处理侧面视角俯卧撑视频，提取特征并保存为 CSV 文件

    Args:
        input_path (str): 输入视频路径
        output_path (str): 输出 CSV 文件路径
        model (YOLO): YOLO 模型实例
    """
    results = model(source=input_path, stream=True, **kwargs)
    data = []
    for r in results:
        keypoints = utils.extract_main_person(r)
        if keypoints.size(0) == 0:
            continue
        points, labels = extract_points(keypoints, view='side')
        data.append(points)
    df = pd.DataFrame(data, columns=labels)
    df.to_csv(output_path, index=False)

def front_video2csv(input_path: str, output_path: str, model: YOLO, **kwargs) -> None:
    """
    处理正面视角俯卧撑视频，提取特征并保存为 CSV 文件

    Args:
        input_path (str): 输入视频路径
        output_path (str): 输出 CSV 文件路径
        model (YOLO): YOLO 模型实例
    """
    results = model(source=input_path, stream=True, **kwargs)
    data = []
    for r in results:
        keypoints = utils.extract_main_person(r)
        if keypoints.size(0) == 0:
            continue
        points, labels = extract_points(keypoints, view='front')
        data.append(points)
    df = pd.DataFrame(data, columns=labels)
    df.to_csv(output_path, index=False)

def side_video2video_(frame: np.array, points: torch.Tensor) -> np.array:
    """
    处理侧面视角视频的每一帧

    Args:
        frame (np.array): 原始帧图像
        points (torch.Tensor): 关键点张量

    Returns:
        np.array: 处理后的帧图像
    """
    frame = utils.show_keypoints(frame, points)
    frame = plot_angles(frame, points, view='side')
    return frame

def front_video2video_(frame: np.array, points: torch.Tensor) -> np.array:
    """
    处理正面视角视频的每一帧

    Args:
        frame (np.array): 原始帧图像
        points (torch.Tensor): 关键点张量

    Returns:
        np.array: 处理后的帧图像
    """
    frame = utils.show_keypoints(frame, points)
    frame = plot_angles(frame, points, view='front')
    return frame

def side_video2video(input_path: str, output_path: str, model: YOLO, **kwargs) -> None:
    """
    处理侧面视角俯卧撑视频，生成带有关键点和角度信息的新视频

    Args:
        input_path (str): 输入视频路径
        output_path (str): 输出视频路径
        model (YOLO): YOLO 模型实例
    """
    utils.video2video_base_(side_video2video_, input_path, output_path, model, **kwargs)

def front_video2video(input_path: str, output_path: str, model: YOLO, **kwargs) -> None:
    """
    处理正面视角俯卧撑视频，生成带有关键点和角度信息的新视频

    Args:
        input_path (str): 输入视频路径
        output_path (str): 输出视频路径
        model (YOLO): YOLO 模型实例
    """
    utils.video2video_base_(front_video2video_, input_path, output_path, model, **kwargs)

def plot_angles(frame: np.array, points: torch.Tensor, view: str) -> np.array:
    """
    在帧图像上绘制角度信息

    Args:
        frame (np.array): 原始帧图像
        points (torch.Tensor): 关键点张量
        view (str): 视角，'side' 或 'front'

    Returns:
        np.array: 绘制后的帧图像
    """
    keypoints = Keypoints(points)

    if view == 'side':
        # 获取关键点
        l_shoulder = keypoints.get_int("l_shoulder")
        l_elbow = keypoints.get_int("l_elbow")
        l_wrist = keypoints.get_int("l_wrist")
        l_hip = keypoints.get_int("l_hip")
        l_ankle = keypoints.get_int("l_ankle")
        # 绘制骨架线
        cv2.line(frame, l_shoulder, l_elbow, (0, 255, 0), 2)
        cv2.line(frame, l_elbow, l_wrist, (0, 255, 0), 2)
        cv2.line(frame, l_shoulder, l_hip, (0, 255, 0), 2)
        cv2.line(frame, l_hip, l_ankle, (0, 255, 0), 2)
        # 绘制角度信息
        elbow_angle = utils.three_points_angle(l_shoulder, l_elbow, l_wrist)
        back_angle = utils.three_points_angle(l_shoulder, l_hip, l_ankle)
        cv2.putText(frame, f"Elbow Angle: {int(elbow_angle)}", l_elbow, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Back Angle: {int(back_angle)}", l_hip, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    elif view == 'front':
        # 获取关键点
        l_shoulder = keypoints.get_int("l_shoulder")
        r_shoulder = keypoints.get_int("r_shoulder")
        l_elbow = keypoints.get_int("l_elbow")
        r_elbow = keypoints.get_int("r_elbow")
        l_wrist = keypoints.get_int("l_wrist")
        r_wrist = keypoints.get_int("r_wrist")
        l_hip = keypoints.get_int("l_hip")
        r_hip = keypoints.get_int("r_hip")
        # 绘制骨架线
        cv2.line(frame, l_shoulder, l_elbow, (0, 255, 0), 2)
        cv2.line(frame, l_elbow, l_wrist, (0, 255, 0), 2)
        cv2.line(frame, r_shoulder, r_elbow, (0, 255, 0), 2)
        cv2.line(frame, r_elbow, r_wrist, (0, 255, 0), 2)
        # 绘制角度信息
        l_elbow_angle = utils.three_points_angle(l_shoulder, l_elbow, l_wrist)
        r_elbow_angle = utils.three_points_angle(r_shoulder, r_elbow, r_wrist)
        elbow_angle = (l_elbow_angle + r_elbow_angle) / 2
        cv2.putText(frame, f"Elbow Angle: {int(elbow_angle)}", ((l_elbow[0]+r_elbow[0])//2, (l_elbow[1]+r_elbow[1])//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        raise ValueError("视角参数错误，应为 'side' 或 'front'")

    return frame


if __name__ == "__main__":
    model = YOLO("/home/shyang/code/FitFormAI_Analysiser/model/yolov8x-pose.pt")
    pushup = PushUp(model, view='front')
    path = "/home/shyang/code/FitFormAI_Analysiser/resource/俯卧撑/正面视角/手臂外展/0.mp4"
    results = pushup.do_analysis(path)
    print(results)
    