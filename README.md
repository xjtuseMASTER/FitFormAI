# FitFormAI
基于yolo的健身姿势检测与姿态矫正建议系统

## 项目结构
.  
├── README.md  
├── config.yaml  
├── requirements.txt  
├── feature_extract  
│   └── angle.py  
├── model  
│   ├── checkpoints  
│   └── main.py  
├── tasks  
│   ├── keypoints.py  
│   └── pull_up.py  
├── output  
└── vedios  

* model:保存模型参数
* tasks:任务文件夹，对于不同的健身任务，分别实现标准性判别方法
* feature_extract:对keypoints数据做特征提取的工具
* config.yaml:配置文件
* output:视频输出
* vedios:测试用原始视频

见下图：
![项目结构](/assets/项目结构.png)

## 运行
目前在`main.py`下实现了输入视频，输出特征标记后视频的功能，即`main.py`中的`process_video`方法。先通过`yolo`得到带有`keypoints`的视频输出，视频的每一帧都通过`tasks`内的`pull_up.py`所实现的方法进行特帧提取，例如：
* `process_angle`：获取`手腕-肩-跨`的角度信息
* `is_wrist_above_elbow`：判断`手腕`是否在`手肘`的正上方 

在特征提取的过程中，还对`yolo`返回的`keypoints`对象进行了类型封装，见`tasks/keypoints.py`

直接运行`main.py`即可对`vedios`中的指定视频做特征提取处理，并输出在`output`中。（`vedios`中已收集的视频可以在飞书文档中找到）

## 需要做什么
1. 观看短视频博主或阅读相关文章，调研各健身动作的标准动作以及多个错误做法
2. 搜集健身动作不同视角（如正面、正侧面等）不同错误类型的视频资源
3. 在`tasks`中实现健身动作的判别流程方法，在这个过程中可能需要实现一些数学计算方法从`keypoints`中获取特征信息（如一些角度信息或节点位置关系），这些方法可以写入`feature_extract`（当然这个文件架构不一定需要这样，如果各动作的特征提取方法差别很大，难以复用，可以直接在各个动作下实现特征提取的方法。
4. 目前我们尚不需要考虑如何获取面向用户的输出，只需要
实现各个健身动作的判别方法即可

![算法流程](/assets/算法流程.png)