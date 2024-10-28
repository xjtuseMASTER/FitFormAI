import os
import sys
from typing import List, Tuple

from scipy.fft import fft, ifft

from fourier_series_fit.fit import LINEAR_PENALTY_FUNCTION, best_fit, fourier_series_fct
sys.path.append("./tasks")
from matplotlib import pyplot as plt
import yaml
from pathlib import Path
from ultralytics import YOLO
from tasks.task_processor import TaskProcessor
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit


def setup_model() -> YOLO:
    """根据配置文件初始化YOLO模型

    Returns:
        YOLO:完成初始化的YOLO模型
    """
    CONFIG_FILE = "config.yaml"
    with open(CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)
    CHECKPOINTS = config["CHECKPOINTS"]
    MODELNAME = config["MODELNAME"]
    return YOLO(CHECKPOINTS + MODELNAME)



def process_by_view(src_view_dir: str) -> None:
    """按文件夹结构,将resource文件夹中所有样本都进行数据提取,并输出为cSV文件到output文件夹中方便下一步绘图"""
    dest_dir = 'output'
    task_processor = TaskProcessor(setup_model())

    for root, dirs, files in os.walk(src_view_dir):
        for file in files:
            if file == '.gitkeep': continue 
            if file.endswith(('.MOV', '.mov', '.mp4')):
                src_path = os.path.join(root, file)
                # processing
                task_processor.process_video2csv(src_path)

# process_by_view("resource/仰卧起坐/侧面视角")

src_file = "resource/仰卧起坐/侧面视角/肩胛骨未触垫/仰卧起坐-侧面视角-肩胛骨未触垫.MOV"
task_processor = TaskProcessor(setup_model())
dest_path = task_processor.process_video2csv(src_file)

def fourier_fit(x_data, y_data, num_terms):
    N = len(y_data)
    T = (x_data[-1] - x_data[0]) / N
    yf = fft(y_data)
    
    xf = np.fft.fftfreq(N, T)[:N // 2]
    
    y_fit = np.zeros_like(y_data)
    for i in range(num_terms):
        y_fit += (2.0 / N) * np.real(yf[i]) * np.cos(2.0 * np.pi * xf[i] * x_data)
    
    return y_fit


def fit_trigonometric_function(input_csv: str,x_label: str,y_labels: List[str], output_path: str) -> None:
    data = pd.read_csv(input_csv)

    y_datas = []
    x_data = data[x_label].to_numpy().astype(float)
    x_data = x_data[:len(x_data) // 2]
    for y_label in y_labels:
        y_data = data[y_label].to_numpy()
        y_data = y_data[:len(y_data) // 2]
        y_datas.append(y_data)

    interpolated_fcts = []
    for y_data in y_datas:
        fit_terms, _ , _ = best_fit(x_data, y_data, penalty_function=LINEAR_PENALTY_FUNCTION)
        print('Fit terms:', fit_terms)
        interpolated_fct = fourier_series_fct(fit_terms)
        interpolated_fcts.append(interpolated_fct)

    y_fits = []
    for interpolated_fct in interpolated_fcts:
        y_fit = [interpolated_fct(x) for x in x_data]
        y_fits.append(y_fit)

    n = len(y_labels)
    _ , axs = plt.subplots(n//2, 2, figsize=(20, 12))

    for idx, y_data in enumerate(y_datas):
        i = idx // 2
        j = idx % 2
        axs[i,j].scatter(x_data, y_data, label=y_labels[idx] + ' original', color='blue')
        axs[i,j].plot(x_data, y_fits[idx], label=y_labels[idx] + ' curve_fit', color='red')
        axs[i,j].set_title(y_labels[idx] + ' result')
        axs[i,j].set_xlabel('idx')
        axs[i,j].set_ylabel('y')
        axs[i,j].legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def process_with_sliding_window(y: List[float], window_size: int) -> Tuple[float, float]:
    local_maxima = []
    local_minima = []

    for i in range(len(y) - window_size + 1):
        window = y[i:i + window_size]
        max_index = np.argmax(window)
        min_index = np.argmin(window)
        # 检查是否是局部最大
        if (max_index > 0 and window[max_index] > window[max_index - 1]) and (max_index < window_size - 1 and window[max_index] > window[max_index + 1]):
            local_maxima.append(window[max_index])

        # 检查是否是局部最小

    # 计算局部最大值的平均
    if local_maxima:
        average_max = np.mean(local_maxima)
    else:
        average_max = None


# fit_trigonometric_function(
#     'output/仰卧起坐/侧面视角/标准/仰卧起坐-侧面视角-标准.csv',
#     "idx",
#     ["l_angle_knee", "r_angle_knee", "l_angle_hip", "r_angle_hip"],
#     'output/仰卧起坐/侧面视角/标准/仰卧起坐-侧面视角-标准.png')


