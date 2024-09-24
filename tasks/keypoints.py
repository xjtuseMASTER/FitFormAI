import torch
from typing import Tuple, List, TypedDict

class Keypoints:
    def __init__(self, points: torch.Tensor):
        """
        初始化 Keypoints 类，存储关键点坐标。

        Args:
            points (torch.Tensor): 包含关键点坐标的张量，形状为 [17, 3]。
        """
        self.keypoints_parts = [
            'nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear',
            'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow',
            'l_wrist', 'r_wrist', 'l_hip', 'r_hip',
            'l_knee', 'r_knee', 'l_ankle', 'r_ankle'
        ]
        self.nose = self._get_point(points[0])
        self.l_eye = self._get_point(points[1])
        self.r_eye = self._get_point(points[2])
        self.l_ear = self._get_point(points[3])
        self.r_ear = self._get_point(points[4])
        self.l_shoulder = self._get_point(points[5])
        self.r_shoulder = self._get_point(points[6])
        self.l_elbow = self._get_point(points[7])
        self.r_elbow = self._get_point(points[8])
        self.l_wrist = self._get_point(points[9])
        self.r_wrist = self._get_point(points[10])
        self.l_hip = self._get_point(points[11])
        self.r_hip = self._get_point(points[12])
        self.l_knee = self._get_point(points[13])
        self.r_knee = self._get_point(points[14])
        self.l_ankle = self._get_point(points[15])
        self.r_ankle = self._get_point(points[16])

    def _get_point(self, point: torch.Tensor):
        return tuple(point[:2].tolist())
    
    class Keypoint(TypedDict):
        part: str
        location: Tuple[float, float]

    def get_all_keypoints(self) -> List[Keypoint]:
        """以列表的形式返回所有姿态节点信息,包括节点代表部位以及节点坐标

        Returns:
            List[Keypoint]: 所有姿态节点信息,Keypoint为具有"part"和"location"两个键的字典
        """
        keypoints_list = []
        
        for part in self.keypoints_parts:
            keypoints_list.append({
                'part': part,
                'location': self.get(part)
            })
        return keypoints_list
    
    def get(self, key: str) -> Tuple[float,float]:
        """获取指定姿态节点，传入的参数应为如下字符串之一：
        "nose","l_eye","r_eye","l_ear","r_ear","l_shoulder","r_shoulder","l_elbow","r_elbow","l_wrist","r_wrist","l_hip","r_hip","l_knee","r_knee","l_ankle","r_ankle"

        Args:
            key (str): 指定姿态节点

        Raises:
            ValueError: 节点未找到错误

        Returns:
            Tuple[float,float]: 获取到的姿态节点
        """
        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise ValueError(f"Keypoint '{key}' not found.")
