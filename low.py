import numpy as np
import matplotlib.pyplot as plt

def csv_to_numpy_l_angle_hip(csv_file):
    import csv
    l_angle_hip_values = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            l_angle_hip_values.append(float(row['r_angle_hip']))
    return np.array(l_angle_hip_values)

def moving_average(data, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(data, window, 'same')

def multi_windows(data, windows):
    waves = []
    for window_size in windows:
        waves.append(moving_average(data, window_size))
    result = np.zeros_like(data)
    for wave in waves:
        result += wave
    result /= len(waves)
    return result

def p_1(wave1: np.array, wave2: np.array) -> np.array:
    result_wave = np.copy(wave2)
    result_wave[wave2 < wave1] = wave1[wave2 < wave1]
    return result_wave

def mean_smooth_idx(wave):
    mean_value = np.mean(wave)
    cross_points = np.where(np.diff(np.sign(wave - mean_value)))[0]
    return cross_points

def highPeriod_timeStampSet(wave) -> list:
    "大窗口滑动的波形用，建议值80-100"
    peaks = (np.diff(np.sign(np.diff(wave))) < 0).nonzero()[0] + 1
    return peaks

def removeAperiodicData(wave: np.array, startTimeStamp: int, endTimeStamp: int) -> np.array:
        """
        去除头和尾的非周期波形
        """
        wave[:startTimeStamp] = wave[startTimeStamp]
        wave[endTimeStamp:] = wave[endTimeStamp]
        return wave

# 示例用法
if __name__ == "__main__":
    csv_data = r"E:\算法\项目管理\FitFormAI\仰卧起坐-侧面视角-单侧发力起坐(1).csv"
    numpy_data = csv_to_numpy_l_angle_hip(csv_data)
    window_size = 30
    smoothed_data = moving_average(numpy_data, window_size)
    a = mean_smooth_idx(smoothed_data)
    smoothed_data = removeAperiodicData(smoothed_data, a[1], a[-1])
    print(a)
    # smoothed_data = multi_windows(smoothed_data, [30, 20])
    p2_data = smoothed_data
    for i in range(5):
        p_data = p_1(numpy_data, p2_data)
        p2_data = moving_average(p_data, window_size)

    p3 = p2_data

    p2_data = p_1(p_data, p2_data)
    p2_data = multi_windows(p2_data, [20, 10, 5, 5, 5])


    # 绘制原始数据和滤波后的数据
    for idx in a:
        plt.axvline(x=idx, color='r', linestyle='--', label='Mean Smooth Index' if idx == a[0] else "")
    plt.plot(numpy_data, label='Original Data')
    plt.plot(smoothed_data, label='Moving Average Filtered Data')
    plt.plot(p_data, label='p')
    plt.plot(p2_data, label='p2')
    plt.plot(p3, label='p3')
    plt.legend()
    plt.show()