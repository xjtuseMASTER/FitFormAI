from typing import TypedDict, List
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

class ProcessedData(TypedDict):
    raw_data: pd.DataFrame
    mean: float
    peak: float
    trough: float
    indices_of_peaks: List[int]
    indices_of_troughs: List[int]

class DateProcessor:
    def __init__(self, raw_data: pd.DataFrame):
        self.raw_data = raw_data
        self.filter_iterations = 3

    def process_unwave_data(self, label: str) -> ProcessedData:
        """
        处理原始波形数据，通过移动平均滤波器进行平滑，并提取接近均值的点，
        同时将数据进行可视化。

        该方法执行以下步骤：
        1. 获取与给定 `label` 对应的原始数据。
        2. 多次应用移动平均滤波器（次数由 `filter_iterations` 参数指定）。
        3. 在每次滤波后提取接近原始数据均值的点。
        4. 对数据进行最终的移动平均滤波，以平滑数据。
        5. 绘制原始数据和滤波后的数据图表，并标注均值线。

        Args:
            label (str): 对应 `self.raw_data` 中数据列的标签，该列将被处理。

        Returns:
            ProcessedData: 包含处理后数据的字典，字典包括：
                - "raw_data"：处理后的数据。
                - "mean"：数据的均值。
        """

        raw_data = self.raw_data[label].values

        #滤波
        temp_data = raw_data
        for _ in range(self.filter_iterations):
            filtered_data = self._apply_moving_average(temp_data, 30)
            mean_value = np.mean(temp_data)
            temp_data = self._extract_points_near_to_mean(temp_data, filtered_data, mean_value)
        filtered_wave = self._apply_moving_average(raw_data, 10)
        mean_value = np.mean(filtered_wave)

        processed_wave_data:ProcessedData = {
            "raw_data": raw_data,
            "mean": mean_value,
        }

        return processed_wave_data

    def process_unwave_data_with_plot(self, label: str) -> ProcessedData:
        """带绘图方法的process_unwave_data函数"""

        plt.title("process_unwave_data")
        plt.xlabel("idx")
        plt.ylabel(label)

        raw_data = self.raw_data[label].values
        plt.plot(raw_data, color='#845EC2', label='raw_data', )

        #滤波
        temp_data = raw_data
        for _ in range(self.filter_iterations):
            filtered_data = self._apply_moving_average(temp_data, 30)
            mean_value = np.mean(temp_data)
            temp_data = self._extract_points_near_to_mean(temp_data, filtered_data, mean_value)
        filtered_wave = self._apply_moving_average(temp_data, 10)
        mean_value = np.mean(filtered_wave)

        plt.plot(filtered_wave, color='r', label='filtered_data', )
        plt.axhline(mean_value, color = 'y', label = 'mean')
        plt.legend()
        plt.show()

        processed_wave_data:ProcessedData = {
            "raw_data": raw_data,
            "mean": mean_value,
        }

        return processed_wave_data


    def process_wave_data(self, label: str) -> ProcessedData:
        """
        处理给定标签的波形数据并返回相关统计信息。

        此方法从原始数据中提取指定标签的波形数据，进行多次滤波以去除噪声。滤波后，计算原始数据的平均值、波形的峰值和谷值，并返回一个包含这些信息的字典。

        处理过程包括：
        1. 提取原始波形数据。
        2. 进行多次移动平均滤波以平滑波形。
        3. 根据滤波后的波形计算峰值和谷值的近似位置。
        4. 从候选峰值和谷值中选择与近似位置最接近的索引。
        5. 计算原始波形数据在这些索引处的平均值。

        Args:
            label (str): 要处理的波形数据的标签名称，对应原始数据中的列名。

        Returns:
            ProcessedWaveData: 一个字典，包含以下字段：
                - raw_data (np.array): 处理后的原始波形数据。
                - mean (float): 原始波形数据的平均值。
                - peak (float): 处理后波形数据的平均峰值。
                - trough (float): 处理后波形数据的平均谷值。
        """

        raw_wave = self.raw_data[label].values

        #滤波
        temp_wave = raw_wave
        for i in range(self.filter_iterations):
            filtered_wave = self._apply_moving_average(temp_wave, 30)
            mean_value = np.mean(filtered_wave)
            temp_wave = self._extract_points_far_from_mean(temp_wave, filtered_wave, mean_value)
        filtered_wave = self._apply_moving_average(temp_wave, 10)

        #获取波峰与波谷
        approximate_peaks, approximate_troughs = self._get_approximate_location_of_peak_and_trough(filtered_wave)

        potential_peaks = self._get_potential_peaks(filtered_wave)
        potential_troughs = self._get_potential_troughs(filtered_wave)

        peaks_indices = self._select_nearest_values(approximate_peaks, potential_peaks)
        troughs_indices = self._select_nearest_values(approximate_troughs, potential_troughs)

        peak_avg = self._calculate_average_at_indices(filtered_wave, peaks_indices)
        trough_avg = self._calculate_average_at_indices(filtered_wave, troughs_indices)

        #返回波形统计结果
        processed_wave_data:ProcessedData = {
            "raw_data": raw_wave,
            "mean": mean_value,
            "peak": peak_avg,
            "trough": trough_avg,
            "indices_of_peaks": peaks_indices,
            "indices_of_troughs": troughs_indices
        }

        return processed_wave_data


    def process_wave_data_with_plot(self, label: str) -> ProcessedData:
        """带绘图方法的process_wave_data函数"""

        plt.title("process_wave_data")
        plt.xlabel("idx")
        plt.ylabel(label)

        raw_wave = self.raw_data[label].values
        plt.plot(raw_wave, color='#845EC2', label='raw_wave', )

        #滤波
        temp_wave = raw_wave
        for i in range(self.filter_iterations):
            filtered_wave = self._apply_moving_average(temp_wave, 30)
            mean_value = np.mean(filtered_wave)
            temp_wave = self._extract_points_far_from_mean(temp_wave, filtered_wave, mean_value)
        filtered_wave = self._apply_moving_average(temp_wave, 10)

        #获取波峰与波谷
        approximate_peaks, approximate_troughs = self._get_approximate_location_of_peak_and_trough(filtered_wave)

        potential_peaks = self._get_potential_peaks(filtered_wave)
        potential_troughs = self._get_potential_troughs(filtered_wave)

        peaks_indices = self._select_nearest_values(approximate_peaks, potential_peaks)
        troughs_indices = self._select_nearest_values(approximate_troughs, potential_troughs)

        peak_avg = self._calculate_average_at_indices(filtered_wave, peaks_indices)
        trough_avg = self._calculate_average_at_indices(filtered_wave, troughs_indices)

        #绘图
        plt.axhline(y=peak_avg, color='g', linestyle='--', label='peak_avg')
        plt.axhline(y=trough_avg, color='g', linestyle='--', label='trough_avg')

        peaks_label_added = False  
        for idx in peaks_indices:
            if not peaks_label_added:
                plt.axvline(x=idx, color='r', linestyle='--', label='peaks index')
                peaks_label_added = True  
            else:
                plt.axvline(x=idx, color='r', linestyle='--')  

        troughs_label_added = False  
        for idx in troughs_indices:
            if not troughs_label_added:
                plt.axvline(x=idx, color='#00C9A7', linestyle='--', label='trough index')
                troughs_label_added = True  
            else:
                plt.axvline(x=idx, color='#00C9A7', linestyle='--')  

        plt.axhline(y=mean_value, color='b', linestyle='--', label='mean_value')
        plt.plot(filtered_wave, color='y', label='filtered_wave')
        plt.legend()
        plt.show()
        plt.savefig('tset.png')

        #返回波形统计结果
        processed_wave_data:ProcessedData = {
            "raw_data": raw_wave,
            "mean": mean_value,
            "peak": peak_avg,
            "trough": trough_avg,
            "indices_of_peaks": peaks_indices,
            "indices_of_troughs": troughs_indices
        }

        return processed_wave_data


    def _calculate_average_at_indices(self, data: np.array, indices: list) -> float:
        """
        计算指定索引位置的数据的平均值。

        Args:
            data (np.array): 一维 NumPy 数组，表示需要计算平均值的数据。
            indices (list): 指定的索引位置列表。

        Returns:
            float: 指定索引位置对应数据的平均值。
        """
        selected_values = data[indices]
        return np.mean(selected_values)


    def _get_approximate_location_of_peak_and_trough(self, wave:np.array):
        """
        计算波形中峰值和谷值的大致位置。

        该方法首先对输入波形应用移动平均，以平滑数据，从而减小噪声对峰谷检测的干扰。
        然后通过计算一阶导数的符号变化来识别峰值和谷值的近似位置：
        - 峰值的近似位置：对应一阶导数由正变负的点。
        - 谷值的近似位置：对应一阶导数由负变正的点。

        Args:
            wave (np.array): 一维 NumPy 数组，表示需要进行峰谷检测的波形数据。

        Returns:
            tuple: 包含两个元素的元组：
                - approximate_location_of_peaks (np.array): 表示波形峰值的大致位置的索引数组。
                - approximate_location_of_troughs (np.array): 表示波形谷值的大致位置的索引数组。
        """
        
        wave = self._apply_moving_average(wave, 80)
        approximate_location_of_peaks = (np.diff(np.sign(np.diff(wave))) < 0).nonzero()[0] + 1
        approximate_location_of_troughs = (np.diff(np.sign(np.diff(wave))) > 0).nonzero()[0] + 1

        return approximate_location_of_peaks, approximate_location_of_troughs
        


    def _get_potential_peaks(self, wave: np.array, window_size: int = 9, threshold: float = 0.1) -> np.array:
        """
        获取波形的峰值索引，并基于指定阈值过滤噪声。

        该方法通过计算局部窗口内的最大值来检测峰值位置。
        只有当一个点在其邻域范围内满足局部最大值条件，并且其值比前后相邻点高于指定的阈值时，才将其视为有效峰值。

        Args:
            wave (np.array): 输入的一维 NumPy 数组，表示需要分析的波形数据。
            window_size (int): 用于定义局部检测范围的窗口大小。默认为 5。
            threshold (float): 用于过滤噪声的最小高度差阈值。默认为 0.1。

        Returns:
            np.array: 包含有效峰值位置索引的 NumPy 数组。
        """
        peaks = []
        n = len(wave)

        for i in range(window_size, n - window_size):
            current_value = wave[i]
            left_values = wave[i - window_size:i]
            right_values = wave[i + 1:i + window_size + 1]

            if (current_value > np.mean(left_values)) and (current_value > np.mean(right_values)):
                if (current_value - np.mean(left_values) > threshold) and (current_value - np.mean(right_values) > threshold):
                    peaks.append(i)

        return np.array(peaks)

    
    def _get_potential_troughs(self, wave: np.array, window_size: int = 9, threshold: float = 0.2) -> np.array:
        """
        获取波形的谷值索引，并基于指定阈值过滤噪声。

        该方法通过计算局部窗口内的最小值来检测谷值位置。
        只有当一个点在其邻域范围内满足局部最小值条件，并且其值比前后相邻点低于指定的阈值时，才将其视为有效谷值。

        Args:
            wave (np.array): 输入的一维 NumPy 数组，表示需要分析的波形数据。
            window_size (int): 用于定义局部检测范围的窗口大小。默认为 5。
            threshold (float): 用于过滤噪声的最小深度差阈值。默认为 0.1。

        Returns:
            np.array: 包含有效谷值位置索引的 NumPy 数组。
        """
        troughs = []
        n = len(wave)

        for i in range(window_size, n - window_size):
            current_value = wave[i]
            left_values = wave[i - window_size:i]
            right_values = wave[i + 1:i + window_size + 1]

            if (current_value < np.mean(left_values)) and (current_value < np.mean(right_values)):
                if (np.mean(left_values) - current_value > threshold) and (np.mean(right_values) - current_value > threshold):
                    troughs.append(i)

        return np.array(troughs)


    def _extract_points_far_from_mean(self, wave1: np.array, wave2: np.array, mean_value: float) -> np.array:
        """
        从两个波形中选择距离给定 mean_value 值较远的点。

        Args:
            wave1 (np.array): 第一个波形的数据。
            wave2 (np.array): 第二个波形的数据。
            mean_value (float): 用于比较的 mean_value 值。

        Returns:
            np.array: 新波形，包含距离 mean_value 值较远的点。
        """
        distance_wave1 = np.abs(wave1 - mean_value)
        distance_wave2 = np.abs(wave2 - mean_value)

        new_wave = np.where(distance_wave1 > distance_wave2, wave1, wave2)
        return new_wave
    
    def _extract_points_near_to_mean(self, wave1: np.array, wave2: np.array, mean_value: float) -> np.array:
        """
        从两个波形中选择距离给定 mean_value 值较远的点。

        Args:
            wave1 (np.array): 第一个波形的数据。
            wave2 (np.array): 第二个波形的数据。
            mean_value (float): 用于比较的 mean_value 值。

        Returns:
            np.array: 新波形，包含距离 mean_value 值较远的点。
        """
        distance_wave1 = np.abs(wave1 - mean_value)
        distance_wave2 = np.abs(wave2 - mean_value)

        new_wave = np.where(distance_wave1 < distance_wave2, wave1, wave2)
        return new_wave


    def _select_nearest_values(self, y_approximate: np.array, y_potential: np.array) -> np.array:
        """
        从候选值中选择与大概位置最接近的值。

        Args:
            y_approximate (np.array): 一维 NumPy 数组，表示 y 的大概位置。
            y_potential (np.array): 一维 NumPy 数组，表示 y 的候选值。

        Returns:
            np.array: 包含与 y_approximate 中每个位置最接近的 y_potential 值的 NumPy 数组。
        """
        nearest_values = []

        for approx in y_approximate:
            distances = np.abs(y_potential - approx)
            
            nearest_value = y_potential[np.argmin(distances)]
            nearest_values.append(nearest_value)

        return np.array(nearest_values)
        

    def _apply_moving_average(self, data: np.array, window_size: int) -> np.array:
        """
        计算给定数据的滑动窗口平均值。

        该函数使用指定的窗口大小对输入数据进行平滑处理，
        通过在数据的两端填充边界值来处理边缘效应。


        Args:
            data (np.array): 输入的一维 NumPy 数组，表示需要进行平均处理的数据。
            window_size (int): 窗口大小，必须为正整数。


        Returns:
            np.array: 一个一维 NumPy 数组，表示经过滑动窗口平均处理后的结果,返回的数组长度与输入数组相同。
        """
        left_offset = window_size // 2
        right_offset = window_size - left_offset - 1

        left_pad_value = data[0]
        right_pad_value = data[-1]

        adjusted_data = np.pad(data, (left_offset, right_offset), 'constant', constant_values=(left_pad_value, right_pad_value))
        window = np.ones(int(window_size))/float(window_size)
        result = np.convolve(adjusted_data, window, 'same')

        return result[left_offset:-right_offset]
    

#use case
# csv_path = 'output/仰卧起坐/侧面视角/单侧发力起坐/仰卧起坐-侧面视角-单侧发力起坐(1).csv'
# label= 'l_angle_knee'
# raw_csv = pd.read_csv(csv_path)

# data_processor = DateProcessor(raw_csv)
# data_processor.process_unwave_data(label)


