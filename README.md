# FitFormAI
基于yolo的健身姿势检测与姿态矫正建议系统

## 项目结构
.
├── README.md  
├── requirements.txt  
├── config.yaml  
├── main.py   
├── model  
├── output  
│   └── 引体向上  
│       ├── 正侧面视角  
│       └── 背部视角  
│           ├── 标准  
│           ├── 握距不合适  
│           ├── 肩胛不稳定  
│           └── 脊柱侧弯  
├── resource  
│   └── 引体向上  
│       ├── 正侧面视角  
│       └── 背部视角  
│           ├── 握距不合适  
│           ├── 标准  
│           ├── 肩胛不稳定  
│           └── 脊柱侧弯  
└── tasks  
    ├── keypoints.py  
    ├── pull_up.py  
    ├── task_processor.py  
    └── utils.py   

* `model`:保存模型参数
* `config.yaml`:配置文件
* `resource`:输入文件夹，具有固定的目录结构（**动作-视角-标准/错误点**）
* `output`:输出文件夹，保持和`resource`相同的目录结构
* `main.py`:实现`resource2output`方法,将`resource`中的资源全部提取数据并输出（csv格式）到`output`的相应位置
* `tasks`:任务文件夹，对于不同的健身任务，分别实现标准性判别方法
    * `keypoint.py`:是对`yolo`模型返回的节点进行对象封装，其中的`Keypoint`对象封装了返回结果（是一个数组）中各关节位置对应数组中的位置，这样就不需要通过下标直接获取节点，而是通过例如`get("l_elbow")`的实例方法获取节点
    * `pull_up.py`:为具体健身任务实现标准性判别方法，这里是对引体向上的处理
    * `task_processor.py`由于`main.py`是在对`resource`文件夹中所有资源进行处理，不同的方法将对应不同的处理函数，`task_processor.py`中实现了`TaskProcessor`对象，封装了不同视角不同任务的处理函数，在`main.py`中就可以实现通过资源路径获取对该资源的处理函数
    * `utils.py`:通用方法，例如`extract_main_person`从`yolo`返回的所有`keypoints`中找到主体人物的`keypoint`

## 运行
`main.py`中实现了将`resource`文件夹中所有资源进行特征提取，并将结果以`csv`格式输出到`output`中的指定位置。当然，目前只实现了对于`引体向上-背部视角-xxx`的简单特征提取（在`pull_up.py`中），如两组角度信息。

## 需要做什么
1. 观看短视频博主或阅读相关文章，调研各健身动作的标准动作以及多个错误做法
2. 搜集健身动作不同视角（如正面、正侧面等）不同错误类型的视频资源
3. 在`tasks`中实现健身动作的判别流程方法
4. 目前我们尚不需要考虑如何获取面向用户的输出，只需要
实现各个健身动作的判别方法即可

![算法流程](/assets/算法流程.png)

## 数据分析
拿到足够量这样的数据后，对比分析标准做法与各不标准做法之间的差别，找到决策边界
![绘图示例](/assets/引体向上-背部视角-绘图.png)