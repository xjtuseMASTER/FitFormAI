import cv2
import numpy as np
import pandas as pd
from typing import List, TypedDict

from . import utils
from .advices import load_advice_by_filename

class InfoBase(TypedDict):
    raw_keypoints: any
    raw_data: pd.DataFrame

class ErrorDetail(TypedDict):
    error: str
    advice: str
    frame: np.array

class TaskBase:
    def __init__(self):
        self.task = "task"
    
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
    
    def _feature_extractor(self, yolo_outputs: list) -> InfoBase:
        """获取分析判断前所需要的所有特征信息"""
        raise NotImplementedError("子类必须重写 _feature_extractor 方法。")


    def _judge_error(self, info: InfoBase) -> List[str]:
        """根据特征数据实现判别逻辑"""
        error_included = []
        for error in self.error_list:
            method_name = '_' + error
            method = getattr(self, method_name, None)
            if callable(method):
                result = method(info)
                if result: 
                    error_included.append(error)
        
        return error_included


    def _frame_idx_extractor(self, error_list: List[str], info: InfoBase) -> List[int]:
        #TODO: 得到佐证视频帧索引  
        frame_idxs = []
        for error in error_list:
            method_name = '_frame_' + error
            method = getattr(self, method_name, None)
            if callable(method):
                result = method(info)
                frame_idxs.append(result)

        return frame_idxs
    
    def _find_out_frames(self,source: str, frame_idxs: List[int]) -> List[np.array]:
        """按视频帧序号从原视频中找到相应的视频帧"""
        cap = cv2.VideoCapture(source)
        frames = []
        if not cap.isOpened():
            print("无法打开视频文件")
        else:
            for frame_idx in frame_idxs:

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                else:
                    print(f"无法读取帧 {frame_idx}")
        cap.release()

        return frames

    def _plot_the_frames(self, info: InfoBase, error_list: List[str], frame_idxs: List[int], frames: List[np.array]) -> List[np.array]:
        """对找到的视频帧进行绘制"""
        frames_after_plot = []
        for i, error in enumerate(error_list):
            method_name = '_plot_' + error
            method = getattr(self, method_name, None)
            if callable(method):
                result = method(info, frame_idxs[i], frames[i])
                frames_after_plot.append(result)

        return frames_after_plot