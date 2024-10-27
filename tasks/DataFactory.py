import csv
from typing import TypedDict, List
import numpy as np

class WaveData(TypedDict):
    """
    包含原始csv文件，以及所有中间处理数据
    originCSV : csv file
    originData : np.ndarray 转换后的原始波形numpy
    dataNameList : list[int, str] 维度索引和name
    smoothData : np.ndarray 平滑后的波形（小窗口）
    smoothData4Period : np.ndarray 为提取周期信息平滑的波形（大窗口）
    filterData : np.ndarray 滤波后的波形
    meanPeriod_timeStampSet : np.ndarray 波形均值点集合
    highPeriod_timeStampSet : np.ndarray 波形极大值点集合
    lowPeriod_timeStampSet : np.ndarray 波形极小值点集合
    windowSize4meanPeriod : int
    windowSize4highPeriod : int
    windowSize4lowPeriod : int
    windowSize4filterDeviation : int
    TimefilterDeviation : int  filterDeviation的次数，建议赋值5
    windowSize4multiWindowsAverage : list[int] 第二次多窗口的值，建议赋值[20, 10, 5, 5, 5]
    """
    # TODO 拿np做
    originCSV : str
    originData : np.ndarray
    dataNameList : List[str]

    smoothData : np.ndarray
    smoothData4Period : np.ndarray
    filterData : np.ndarray

    meanPeriod_timeStampSet : np.ndarray
    highPeriod_timeStampSet : np.ndarray
    lowPeriod_timeStampSet : np.ndarray

    windowSize4meanPeriod : int
    windowSize4highPeriod : int
    windowSize4lowPeriod : int
    windowSize4filterDeviation : int
    TimefilterDeviation : int
    windowSize4multiWindowsAverage : List[int]

class DataFactory:
    """
    只提供processData方法，其他方法均为私有方法，不对外提供。
    可以选择处理单个波形或者全部处理。
    可以使用继承的方式，实现不同的数据处理方法。

    提供不同传入值的处理方法，可以向内传入waveData，也可以获取waveData。
    """
    def __init__(self, originCSV: str):
        """
        初始构造传入originCSV - csv file，自动创建WaveData
        """
        self._waveData = WaveData()
        self._waveData['originCSV'] = originCSV
        # arrayData处理
        self._setOriginData()
        self._setDefaultParameter()

    def getWaveData(self):
         return self._waveData

    def processSingleData(self, waveName: str):
        """
        处理单个波形
        转换 -> 滤波
        """
        pass

    def processAllData(self):
         pass
    
    def processSingleData(waveData : WaveData, waveName: str) -> WaveData:
         pass
    
    def processAllData(waveData : WaveData) -> WaveData:
         pass
    
    def _filterData(self, wave: np.array) -> None:
        """
        对波形数据进行滤波处理。
        数据准备 -> 偏差过滤 -> 多次小窗口平均（特征恢复）
        """
        self._waveData['smoothData4Period'] = self._moving_average(wave, )

    def _multiWindows_average(self, wave: np.array) -> None:
        """
        多次小窗滤波后取均值，最大可能保留特征
        建议窗口列表的最大值小于过滤偏差的窗口，并且列表中的值递减，最小值建议设置为5
        """
        waves = []
        for window_size in self._waveData['windowSize4multiWindowsAverage']:
            waves.append(self._moving_average(wave, window_size))
        result = np.zeros_like(wave)
        for wave in waves:
            result += wave
        result /= len(waves)
        return result

    def _filterDeviation(self, wave: np.array) -> None:
        # TODO
        """
        滤去偏差，返回较为平滑的曲线
        使用较小窗口滤波和两个波形每个frame取大值的方式滤去大偏差
        注意：为了保证frame的一致性，全部以子波段的形式向下传入，这里的每个参数都有其意义
        """
        mean_idx = self._waveData['meanPeriod_timeStampSet'][1] # 需要idx，改成idx
        mean_value = wave
        start_idx = 1
        end_idx = -2
        waveList = self._waveData['meanPeriod_timeStampSet'][start_idx:end_idx].tolist()
        for i in range(self._waveData['TimefilterDeviation']):
            for start_frame in waveList:
                if(wave[start_frame + 1] > mean_value):
                    self._highFilterDeviation(wave ,start_frame)# wave怎么传再好好想想

    def _highFilterDeviation(self, wave : np.array, start_frame: int, end_frame : int) -> None:
        """
        大于mean的数据滤去偏差
        注意：这里处理的是子波段
        """
        intermediate_data = self._getHighValueInTwoWaves(self._waveData['originData'], wave)
        wave = self._moving_average(intermediate_data, self._waveData['windowSize4filterDeviation'], start_frame, end_frame)

    def _lowFilterDeviation(self, wave : np.array, start_frame: int, end_frame : int) -> np.array:
        """
        小于mean的数据滤去偏差
        注意：这里处理的是子波段
        """
        intermediate_data = self._getLowValueInTwoWaves(self._waveData['originData'], wave)
        wave = self._moving_average(intermediate_data, self._waveData['windowSize4filterDeviation'], start_frame, end_frame)


    def _moving_average(data: np.array, window_size : int) -> np.array:
        """滑动窗口"""
        window = np.ones(int(window_size)) / float(window_size)
        return np.convolve(data, window, 'same')
    
    def _moving_average(data: np.array, window_size : int, start_frame : int, end_frame : int) -> np.array:
        """滑动窗口"""
        window = np.ones(int(window_size)) / float(window_size)
        full_result = np.convolve(data, window, 'same')
        return full_result[start_frame:end_frame]
    
    def _csv2np(csv_file: csv, dataName: str) -> np.array:
        l_angle_hip_values = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                l_angle_hip_values.append(float(row[dataName]))
        return np.array(l_angle_hip_values)
    
    def _getHighValueInTwoWaves(wave1: np.array, wave2: np.array) -> np.array:
            result_wave = np.copy(wave2)
            result_wave[wave2 < wave1] = wave1[wave2 < wave1]
            return result_wave
    
    def _getLowValueInTwoWaves(wave1: np.array, wave2: np.array) -> np.array:
            result_wave = np.copy(wave2)
            result_wave[wave2 > wave1] = wave1[wave2 > wave1]
            return result_wave
    
    def _removeAperiodicData(wave: np.array, startTimeStamp: int, endTimeStamp: int) -> np.array:
        """
        去除头和尾的非周期波形
        建议值：startTimeStamp 索引为 1, endTimeStamp 索引为 -1
        """
        wave[:startTimeStamp] = wave[startTimeStamp]
        wave[endTimeStamp:] = wave[endTimeStamp]
        return wave
    
    def _meanPeriod_timeStampSet(wave: np.array) -> np.array:
        """中等窗口滑动的波形用，建议值30"""
        mean_value = np.mean(wave)
        cross_points = np.where(np.diff(np.sign(wave - mean_value)))[0]
        return cross_points

    def _highPeriod_timeStampSet(wave: np.array) -> np.array:
        """
        大窗口滑动的波形用，建议值80-100
        在高平均这几个值中，一定一定要提前去除非周期波形，否则获取会有问题
        """
        peaks = (np.diff(np.sign(np.diff(wave))) < 0).nonzero()[0] + 1
        return peaks
    
    def _lowPeriod_timeStampSet(wave: np.array) -> np.array:
        """
        大窗口滑动的波形用，建议值80-100
        在低平均这几个值中，一定一定要提前去除非周期波形，否则获取会有问题
        """
        troughs = (np.diff(np.sign(np.diff(wave))) > 0).nonzero()[0] + 1
        return troughs

    def _setOriginData(self):
        with open(self._waveData['originCSV'], 'r') as f:
            reader = csv.DictReader(f)
            data = list(reader)
            self._waveData['dataNameList'] = reader.fieldnames
            self._waveData['originData'] = np.array([[float(row[name]) for name in reader.fieldnames] for row in data])

    def _getWaveNumpyIndex(self, name : str) -> int:
        """获取numpy.array中name对应的索引"""
        try:
            return self._waveData['dataNameList'].index(name)
        except ValueError:
            raise ValueError(f"{name} not found in dataNameList")
        
    def _setDefaultParameter(self) -> None:
        self._waveData['TimefilterDeviation'] = 5
        self._waveData['windowSize4meanPeriod'] = 30
        self._waveData['windowSize4highPeriod'] = 100
        self._waveData['windowSize4lowPeriod'] = 100
        self._waveData['windowSize4filterDeviation'] = 30
        self._waveData['windowSize4multiWindowsAverage'] = [20, 10, 5, 5, 5]

    def set():
         # TODO
         pass
    
    def setCSV(self, originCSV: csv):
        """更改csv文件"""
        self._waveData['originCSV'] = originCSV
        self._setOriginData()
        
    def _plot(wave : list[np.array]):
        import matplotlib.pyplot as plt
        for wave in wave:
            plt.plot(wave)
        plt.title('Waveform')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.show()
        

def test_setDataWave():
        waveData =dataFactory.getWaveData()
        print(waveData['originCSV'])
        print(waveData['dataNameList'])
        print(waveData['originData'].shape)
        print(dataFactory._getWaveNumpyIndex("r_angle_hip"))
test_csv_data = r"E:\算法\项目管理\FitFormAI\仰卧起坐-侧面视角-单侧发力起坐(1).csv"
dataFactory = DataFactory(test_csv_data)
test_setDataWave()
a = np.array([1, 2, 3, 4])
b = a[1:-2].tolist()
print(b)