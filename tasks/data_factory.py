import csv
from typing import TypedDict, List
from matplotlib import pyplot as plt
import numpy as np

class WaveData(TypedDict):
    """
    包含原始csv文件，以及所有中间处理数据
    波形数据的shape为(wave_length, wave_nums)
    originCSV : csv file
    originData : np.ndarray 转换后的原始波形numpy shape:(wave_length, wave_idx)
    dataNameList : list[str]
    smoothData : np.ndarray 平滑后的波形（小窗口）
    smoothData4Period : np.ndarray 为提取周期信息平滑的波形（大窗口）
    filterData : np.ndarray 滤波后的波形
    meanPeriod_timeStampSet : List[np.ndarray] 波形均值点集合
    highPeriod_timeStampSet : List[np.ndarray] 波形极大值点集合
    lowPeriod_timeStampSet : List[np.ndarray] 波形极小值点集合
    windowSize4meanPeriod : int
    windowSize4highPeriod : int
    windowSize4lowPeriod : int
    windowSize4filterDeviation : int
    TimefilterDeviation : int  filterDeviation的次数，建议赋值5
    windowSize4multiWindowsAverage : list[int] 第二次多窗口的值，建议赋值[20, 10, 5, 5, 5]
    mean_value : List[float] windowSize4meanPeriod的均值
    """

    originCSV : str
    originData : np.ndarray
    dataNameList : List[str]

    smoothData : np.ndarray
    smoothData4Period : np.ndarray
    smoothData4highPeriod : np.ndarray
    smoothData4lowPeriod : np.ndarray
    filterData : np.ndarray

    meanPeriod_timeStampSet : List[np.ndarray]
    highPeriod_timeStampSet : List[np.ndarray]
    lowPeriod_timeStampSet : List[np.ndarray]

    windowSize4meanPeriod : int
    windowSize4highPeriod : int
    windowSize4lowPeriod : int
    windowSize4filterDeviation : int
    TimefilterDeviation : int
    windowSize4multiWindowsAverage : List[int]

    mean_value : List[float]

class DataFactory:
    """
    默认参数均是对于周期波形的处理，对于非周期波形的处理，建议调整参数。
    只提供processData方法，其他方法均为私有方法，不对外提供。
    可以选择处理单个波形或者全部处理。
    可以使用继承的方式，实现不同的数据处理方法。

    提供不同传入值的处理方法，可以向内传入waveData，也可以获取waveData。

    注意：周期波形处理时，第一个周期默认为非周期波形，不参与处理。请不要使用索引0获取第一个周期的信息。
    """
    def __init__(self, originCSV: str):
        """
        初始构造传入originCSV - csv file，自动创建WaveData
        """
        self._waveData = WaveData()
        self._waveData['originCSV'] = originCSV
        self._setOriginData() # originData处理
        self._setDefaultParameter()
        self._setShape() # 初始化所有参数的形状

    def getWaveData(self):
         return self._waveData
    
    def processAllData(self, waveData : WaveData) -> WaveData:
        cache = self._waveData.copy()
        self._waveData = waveData
        waveData = self.processAllData()
        self._waveData = cache.copy()
        return waveData
    
    def processSingleData(self, waveData : WaveData, waveName: str) -> WaveData:
        cache = self._waveData.copy()
        self._waveData = waveData
        waveData = self.processSingleData(waveName)
        self._waveData = cache.copy()
        return waveData
    
    def processAllData(self) -> np.array:
        """
        处理所有波形
        循环取所有索引滤波
        """
        for waveName in self._waveData['dataNameList']:
            if waveName == 'idx': continue
            self.processSingleData(waveName)
        return self._waveData['filterData']

    def processSingleData(self, waveName: str) -> np.array:
        """
        处理单个波形
        取索引 -> 滤波
        可以直接调用，从waveData里拿；也可以用返回值获取
        """
        wave_idx = self._getWaveNumpyIndex(waveName)
        self._filterData(wave_idx)
        self._computeHighAndLowValueMean(wave_idx)
        return self._waveData['filterData'][:, wave_idx]
    
    def _filterData(self, wave_idx: int) -> None:
        """
        对波形数据进行滤波处理。
        数据准备 -> 偏差过滤 -> 多次小窗口平均（特征恢复）
        """
        self._data_preprocess(self._waveData['originData'], wave_idx)
        for i in range(self._waveData['TimefilterDeviation']):
            self._waveData['filterData'][:, wave_idx] = self._filterDeviation(self._waveData['filterData'][:, wave_idx], wave_idx)
        self._waveData['filterData'][:, wave_idx] = self._highAndLowFilterDeviation(self._waveData['filterData'][:, wave_idx], wave_idx)
        self._waveData['filterData'][:, wave_idx] = self._multiWindows_average(self._waveData['filterData'][:, wave_idx])


    def _data_preprocess(self, wave: np.array, wave_idx: int) -> None:
        """做波形的预处理，生成波形后续处理所需数据"""
        wave = wave[:, wave_idx]
        self._waveData['smoothData4Period'][:, wave_idx] = self._moving_average(wave, self._waveData['windowSize4meanPeriod'])
        self._waveData['smoothData4highPeriod'][:, wave_idx] = self._moving_average(wave, self._waveData['windowSize4highPeriod'])
        self._waveData['smoothData4lowPeriod'][:, wave_idx] = self._moving_average(wave, self._waveData['windowSize4lowPeriod'])

        self._waveData['meanPeriod_timeStampSet'][wave_idx] = self._meanPeriod_timeStampSet(self._waveData['smoothData4Period'][:, wave_idx], wave_idx)
        self._waveData['highPeriod_timeStampSet'][wave_idx] = self._highPeriod_timeStampSet(self._waveData['smoothData4highPeriod'][:, wave_idx], wave_idx)
        self._waveData['lowPeriod_timeStampSet'][wave_idx] = self._lowPeriod_timeStampSet(self._waveData['smoothData4lowPeriod'][:, wave_idx], wave_idx)

        self._waveData['smoothData'][:, wave_idx] = self._moving_average(wave, self._waveData['windowSize4filterDeviation'])
        self._waveData['filterData'][:, wave_idx] = self._moving_average(wave, self._waveData['windowSize4filterDeviation'])
        self._waveData['mean_value'][wave_idx] = np.mean(self._waveData['smoothData'][:, wave_idx])

    def _multiWindows_average(self, wave: np.array) -> np.array:
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
    
    def _filterDeviation(self, wave: np.array, wave_idx: int) -> np.array:
        """
        滤去较大偏差，返回较为平滑的曲线
        使用较小窗口滤波和两个波形每个frame取大值的方式滤去大偏差
        """
        self._highAndLowFilterDeviation(wave, wave_idx)
        wave = self._moving_average(wave, self._waveData['windowSize4filterDeviation'])
        return wave

    def _highAndLowFilterDeviation(self, wave: np.array, wave_idx: int) -> np.array:
        """
        整波根据高低分开处理
        注意：为了保证frame的一致性，全部以子波段的形式向下传入，这里的每个参数都有其意义，并且这里的wave已经是numpy的[:, idx]了
        """
        mean_value = self._waveData["mean_value"][wave_idx]
        start_idx = 0
        end_idx = -1 # 调成-1 或者 -2，选择是否去掉最后一个subwave
        waveList = self._waveData['meanPeriod_timeStampSet'][wave_idx][start_idx:end_idx].tolist()
        for start_frame in waveList:
            end_frame = self._waveData['meanPeriod_timeStampSet'][wave_idx][waveList.index(start_frame) + 1 + start_idx]
            middle_frame = int((start_frame + end_frame) / 2)
            if(self._waveData['smoothData4Period'][:, wave_idx][middle_frame] > mean_value):
                wave[start_frame: end_frame] = self._getHighValueInTwoWaves(self._waveData['originData'][:, wave_idx][start_frame: end_frame], wave[start_frame: end_frame])
            elif(self._waveData['smoothData4Period'][:, wave_idx][middle_frame] < mean_value):
                wave[start_frame: end_frame] = self._getLowValueInTwoWaves(self._waveData['originData'][:, wave_idx][start_frame: end_frame], wave[start_frame: end_frame])
        return wave
    
    def _moving_average(self, data: np.array, window_size: int) -> np.array:
        """滑动窗口，进行窗口起始和结束位置调整"""
        left_offset = window_size // 2
        right_offset = window_size - left_offset - 1
        left_pad_value = data[0]
        right_pad_value = data[-1]
        adjusted_data = np.pad(data, (left_offset, right_offset), 'constant', constant_values=(left_pad_value, right_pad_value))
        window = np.ones(int(window_size))/float(window_size)
        result = np.convolve(adjusted_data, window, 'same')
        return result[left_offset:-right_offset]
    
    def _getHighValueInTwoWaves(self, wave1: np.array, wave2: np.array) -> np.array:
        """
        取两个波段中较高的值
        """
        result_wave = np.copy(wave2)
        result_wave[wave2 < wave1] = wave1[wave2 < wave1]
        return result_wave
    
    def _getLowValueInTwoWaves(self, wave1: np.array, wave2: np.array) -> np.array:
        """
        取两个波段中较低的值
        """
        result_wave = np.copy(wave2)
        result_wave[wave2 > wave1] = wave1[wave2 > wave1]
        return result_wave
    
    def _removeAperiodicData(self, wave: np.array, startTimeStamp: int, endTimeStamp: int) -> np.array:
        """
        去除头和尾的非周期波形
        建议值：startTimeStamp 索引为 1, endTimeStamp 索引为 -1
        """
        wave[:startTimeStamp] = wave[startTimeStamp]
        wave[endTimeStamp:] = wave[endTimeStamp]
        return wave
    
    def _meanPeriod_timeStampSet(self, wave: np.array, wave_idx: int) -> np.array:
        """中等窗口滑动的波形用，建议值30"""
        mean_value = np.mean(wave)
        self._waveData['mean_value'][wave_idx] = mean_value
        cross_points = np.where(np.diff(np.sign(wave - mean_value)))[0]
        return cross_points

    def _highPeriod_timeStampSet(self, wave: np.array, wave_idx: int) -> np.array:
        """
        大窗口滑动的波形用，建议值80-100
        在高平均这几个值中，一定一定要提前去除非周期波形，否则获取会有问题，我保留了所有波形，在调用时去掉第一个值，第一个值可能有问题
        """
        peaks = []
        for idx in range(len(self._waveData['meanPeriod_timeStampSet'][wave_idx]) - 1):
            mean_value = self._waveData['mean_value'][wave_idx]
            start_frame = self._waveData['meanPeriod_timeStampSet'][wave_idx][idx]
            end_frame = self._waveData['meanPeriod_timeStampSet'][wave_idx][idx + 1]
            middle_frame = int((start_frame + end_frame) / 2)
            a = self._waveData['smoothData4Period'][:, wave_idx][middle_frame]
            if(self._waveData['smoothData4Period'][:, wave_idx][middle_frame] > mean_value):
                peaks.append(middle_frame)
        return np.array(peaks)

    def _lowPeriod_timeStampSet(self, wave: np.array, wave_idx: int) -> np.array:
        """
        大窗口滑动的波形用，建议值80-100
        在高平均这几个值中，一定一定要提前去除非周期波形，否则获取会有问题，我保留了所有波形，在调用时去掉第一个值，第一个值可能有问题
        """
        troughs = []
        for idx in range(len(self._waveData['meanPeriod_timeStampSet'][wave_idx]) - 1):
            mean_value = self._waveData['mean_value'][wave_idx]
            start_frame = self._waveData['meanPeriod_timeStampSet'][wave_idx][idx]
            end_frame = self._waveData['meanPeriod_timeStampSet'][wave_idx][idx + 1]
            middle_frame = int((start_frame + end_frame) / 2)
            if(self._waveData['smoothData4Period'][:, wave_idx][middle_frame] < mean_value):
                troughs.append(middle_frame)
        return np.array(troughs)

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

    def _setShape(self) -> None:
        num_samples, num_waves = self._waveData['originData'].shape
        self._waveData['smoothData'] = np.zeros((num_samples, num_waves))
        self._waveData['smoothData4Period'] = np.zeros((num_samples, num_waves))
        self._waveData['smoothData4highPeriod'] = np.zeros((num_samples, num_waves))
        self._waveData['smoothData4lowPeriod'] = np.zeros((num_samples, num_waves))
        self._waveData['filterData'] = np.zeros((num_samples, num_waves))
        self._waveData['meanPeriod_timeStampSet'] = [np.array([]) for _ in range(num_waves)]
        self._waveData['highPeriod_timeStampSet'] = [np.array([]) for _ in range(num_waves)]
        self._waveData['lowPeriod_timeStampSet'] = [np.array([]) for _ in range(num_waves)]
        self._waveData['mean_value'] = [0 for _ in range(num_waves)]

    def _computeHighAndLowValueMean(self, wave_idx: int) -> tuple[float, float]:
        """计算高平均值和低平均值"""
        self._waveData['highPeriod_timeStampSet'][wave_idx] = self._highPeriod_timeStampSet(self._waveData['filterData'][:, wave_idx], wave_idx)
        self._waveData['lowPeriod_timeStampSet'][wave_idx] = self._lowPeriod_timeStampSet(self._waveData['filterData'][:, wave_idx], wave_idx)
        highValueStamp = self._waveData['highPeriod_timeStampSet'][wave_idx]
        lowValueStamp = self._waveData['lowPeriod_timeStampSet'][wave_idx]
        return self._computeHighOrLowValueMean(wave_idx, highValueStamp), self._computeHighOrLowValueMean(wave_idx, lowValueStamp)

    def _computeHighOrLowValueMean(self, wave_idx: int, timeStampSet: np.array) -> float:
        """计算高平均值"""
        wave = self._waveData['filterData'][:, wave_idx]
        sum = 0
        len = timeStampSet.shape[0]
        for timeStamps in timeStampSet:
            sum += wave[timeStamps]
        return sum / len

    def setwindowSize4meanPeriod(self, windowSize4meanPeriod : int):
        self._waveData['windowSize4meanPeriod'] = windowSize4meanPeriod

    def setwindowSize4highPeriod(self, windowSize4highPeriod : int):
        self._waveData['windowSize4highPeriod'] = windowSize4highPeriod

    def setwindowSize4lowPeriod(self, windowSize4lowPeriod : int):
        self._waveData['windowSize4lowPeriod'] = windowSize4lowPeriod

    def setwindowSize4filterDeviation(self, windowSize4filterDeviation : int):
        self._waveData['windowSize4filterDeviation'] = windowSize4filterDeviation

    def setTimefilterDeviation(self, TimefilterDeviation : int):
        self._waveData['TimefilterDeviation'] = TimefilterDeviation

    def setwindowSize4multiWindowsAverage(self, windowSize4multiWindowsAverage : list[int]):
        self._waveData['windowSize4multiWindowsAverage'] = windowSize4multiWindowsAverage

    def setCSV(self, originCSV: csv):
        """更改csv文件"""
        self._waveData['originCSV'] = originCSV
        self._setOriginData()

    def getHighPeriod_timeStampSet(self) -> np.array:
        """获取极大值集合,第一个值慎用"""
        return self._waveData['highPeriod_timeStampSet']
    
    def getLowPeriod_timeStampSet(self) -> np.array:
        """获取极小值集合，,第一个值慎用"""
        return self._waveData['lowPeriod_timeStampSet']

    def getHighValueMean(self, waveName: str) -> float:
        """获取高平均值"""
        return self._computeHighValueMean(waveName)

    def getLowValueMean(self, waveName: str) -> float:
        """获取低平均值"""
        return self._computeLowValueMean(waveName)
        
    def plotWave(self, waveName : str):
        """绘制原始数据波和滤波后的数据"""
        import matplotlib.pyplot as plt
        wave_idx = self._getWaveNumpyIndex(waveName)
        highValueMean, lowValueMean = self._computeHighAndLowValueMean(wave_idx)
        plt.axhline(y=highValueMean, color='g', linestyle='--', label='High Value Mean')
        plt.axhline(y=lowValueMean, color='y', linestyle='--', label='Low Value Mean')
        for idx in self._waveData['highPeriod_timeStampSet'][wave_idx]:
            plt.axvline(x=idx, color='r', linestyle='--', label='High Index' if idx == self._waveData['highPeriod_timeStampSet'][wave_idx][0] else "")
        for idx in self._waveData['lowPeriod_timeStampSet'][wave_idx]:
            plt.axvline(x=idx, color='b', linestyle='--', label='Low Index' if idx == self._waveData['lowPeriod_timeStampSet'][wave_idx][0] else "")
        for idx in self._waveData['meanPeriod_timeStampSet'][wave_idx]:
            plt.axvline(x=idx, color='g', linestyle='--', label='Mean Index' if idx == self._waveData['meanPeriod_timeStampSet'][wave_idx][0] else "")
        plt.plot(self._waveData['originData'][:, wave_idx], label="originData")
        plt.plot(self._waveData['smoothData4Period'][:, wave_idx], label="smoothData4Period")
        plt.plot(self._waveData['filterData'][:, wave_idx], label="filterData")
        plt.title('Waveform')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.show()

    def csv2np(self, csv_file: csv, dataName: str) -> np.array:
        """通用方法"""
        data = []
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(float(row[dataName]))
        return np.array(data)
    
    def __plot(self, wave, label):
        import matplotlib.pyplot as plt
        plt.plot(wave, label=label)

"""
use-case
test_csv_data = r"path_to_your_csv_file"
name = "l_angle_hip"
name = "back_ground_angle"
dataFactory = DataFactory(test_csv_data)
dataFactory.processSingleData(name)
dataFactory.plotWave(name)
"""