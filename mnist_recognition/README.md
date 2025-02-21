# 手写数字识别项目

这是一个使用 PyTorch 实现的手写数字识别项目，使用 MNIST 数据集进行训练。该项目不仅实现了基本的数字识别功能，还提供了丰富的模型可视化工具，帮助理解模型的工作原理。

## 数据集说明

### MNIST 数据集概述

MNIST 是一个经典的手写数字数据集，被誉为机器学习领域的 "Hello World"：

1. 数据集规模
   - 训练集：60,000张图片
   - 测试集：10,000张图片
   - 类别：10个数字（0-9）
   - 总大小：约 11MB（压缩后）

2. 图像特征
   - 尺寸：28×28像素
   - 类型：灰度图像
   - 像素值：0-255（8位灰度）
   - 背景：黑色（0）
   - 前景：白色（255）

3. 数据分布
   - 每个数字大约6,000个训练样本
   - 每个数字大约1,000个测试样本
   - 样本经过居中和大小标准化处理

4. 数据格式
   - 图像文件：IDX格式（专门为深度学习设计）
   - 标签文件：一维数组，对应每张图片的数字标签
   - 存储方式：大端字节序（big-endian）

### 数据预处理

1. 图像标准化
   - 大小调整：保持28×28尺寸
   - 像素归一化：将像素值缩放到[0,1]区间
   - 统计归一化：应用数据集的均值(0.1307)和标准差(0.3081)

2. 数据增强
   - 轻微旋转：±15度
   - 微小平移：±2像素
   - 弹性变形：模拟手写变化
   - 高斯噪声：提高鲁棒性

3. 批处理设置
   - 训练批大小：64
   - 测试批大小：1000
   - 随机打乱：每个epoch重新打乱训练数据

### 数据质量

1. 优点
   - 高质量标注
   - 样本分布均衡
   - 预处理规范
   - 广泛使用和验证

2. 特点
   - 单个数字居中
   - 笔画清晰
   - 大小一致
   - 背景干净

3. 适用场景
   - 机器学习入门
   - 算法原型验证
   - 模型结构研究
   - 基准测试

### 使用方式

1. 自动下载
```python
from torchvision import datasets
# 训练集
train_dataset = datasets.MNIST('data', train=True, download=True)
# 测试集
test_dataset = datasets.MNIST('data', train=False, download=True)
```

2. 手动下载
   - [MNIST官方网站](http://yann.lecun.com/exdb/mnist/)
   - 下载四个文件：
     * `train-images-idx3-ubyte.gz`: 训练集图像
     * `train-labels-idx1-ubyte.gz`: 训练集标签
     * `t10k-images-idx3-ubyte.gz`: 测试集图像
     * `t10k-labels-idx1-ubyte.gz`: 测试集标签

3. 数据加载
   - 使用 PyTorch DataLoader
   - 支持多进程加载
   - 自动批处理
   - 内置数据增强

## 功能特点

1. 数字识别
   - 支持处理单个或批量手写数字图片
   - 自动进行图像预处理（大小调整、二值化等）
   - 提供预测结果和置信度
   - 支持自动重命名图片（基于预测结果）

2. 模型可视化
   - 卷积层滤波器可视化
   - 特征图可视化
   - 参数分布分析
   - 像素贡献度分析
   - 模型结构可视化

## 环境准备

### 1. 安装 Python

#### Windows:
1. 访问 [Python 官网](https://www.python.org/downloads/)
2. 下载最新的 Python 3.x 版本（推荐 3.9+）
3. 运行安装程序：
   - 勾选 "Add Python to PATH"
   - 选择 "Customize installation"
   - 确保选中 "pip" 和 "py launcher"
4. 验证安装：
```bash
python --version
pip --version
```

#### macOS:
1. 使用 Homebrew 安装（推荐）：
```bash
# 安装 Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# 安装 Python
brew install python@3.9
```

2. 或从官网下载安装包：
   - 访问 [Python 官网](https://www.python.org/downloads/)
   - 下载 macOS 安装包并运行

3. 验证安装：
```bash
python3 --version
pip3 --version
```

#### Linux (Ubuntu/Debian):
```bash
# 更新包管理器
sudo apt update
sudo apt upgrade

# 安装 Python
sudo apt install python3 python3-pip python3-venv

# 验证安装
python3 --version
pip3 --version
```

### 2. 配置开发环境

1. 下载项目：
```bash
# 使用 git（如果已安装）
git clone <项目地址>
# 或下载 zip 包并解压

# 进入项目目录
cd mnist_recognition
```

2. 创建虚拟环境：
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. 安装依赖：
```bash
# 升级 pip
pip install --upgrade pip

# 安装项目依赖
pip install -r requirements.txt
```

### 3. 验证安装

运行测试脚本确认环境配置正确：
```bash
cd src
python3 train.py --test-env
```

如果看到 "环境配置正确" 的提示，说明环境已经准备就绪。

### 常见安装问题

1. pip 安装错误：
```bash
# Windows
python -m pip install --upgrade pip
# macOS/Linux
python3 -m pip install --upgrade pip
```

2. 权限问题：
   - Windows: 以管理员身份运行命令提示符
   - macOS/Linux: 使用 `sudo` 或检查目录权限

3. PyTorch 安装失败：
   - 访问 [PyTorch 官网](https://pytorch.org/get-started/locally/)
   - 选择适合您系统的安装命令
   - 使用国内镜像源加速下载：
```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

4. 虚拟环境问题：
```bash
# 删除现有虚拟环境
rm -rf venv
# 重新创建
python3 -m venv venv
```

## 项目结构

```
mnist_recognition/
├── data/               # 数据集目录
├── models/             # 模型保存目录
├── test_images/        # 测试图片目录
├── visualizations/     # 可视化结果保存目录
├── src/               # 源代码
│   ├── model.py      # 模型定义
│   ├── train.py      # 训练脚本
│   ├── predict.py    # 预测脚本
│   └── visualize_model.py  # 模型可视化脚本
└── requirements.txt   # 项目依赖
```

## 使用说明

### 1. 训练模型

```bash
cd src
python3 train.py
```

训练过程会：
- 自动下载 MNIST 数据集
- 训练模型（默认10个epoch）
- 保存最佳模型到 `models/best_model.pth`
- 生成训练历史图表 `training_history.png`

### 2. 预测图片

将需要识别的图片放入 `test_images` 目录，然后运行：

```bash
python3 predict.py
```

预测结果会：
- 显示每张图片的预测结果和置信度
- 自动重命名置信度高于90%的图片
- 在同一窗口中显示所有图片的处理结果

### 3. 模型可视化

运行可视化脚本：

```bash
python3 visualize_model.py
```

将生成以下可视化结果：

1. 卷积层滤波器 (`conv1_filters.png`, `conv2_filters.png`)
   - 显示每个卷积层学到的特征提取器
   - 帮助理解模型在寻找什么样的特征

2. 特征图 (`feature_maps_*.png`)
   - 显示图片经过每一层处理后的特征
   - 展示模型如何逐层提取特征

3. 参数分布 (`parameters_visualization.png`)
   - 显示每一层参数的分布情况
   - 包含统计信息和直方图
   - 帮助诊断模型训练状况

4. 像素贡献度分析 (`pixel_contribution_*.png`)
   - 显示输入图像中每个像素对预测结果的贡献
   - 包含正向和负向贡献的可视化
   - 帮助理解模型的决策依据

## 模型架构

### 网络结构概述

模型采用经典的卷积神经网络（CNN）结构，包含两个卷积块和两个全连接层：

```
输入图像 (28×28) → Conv1 → MaxPool1 → Conv2 → MaxPool2 → FC1 → Dropout → FC2 → 输出 (10类)
```

### 详细层级说明

1. 输入层
   - 尺寸：28×28 灰度图像
   - 预处理：
     * 图像缩放到统一大小
     * 像素值归一化到 [0,1]
     * 应用均值(0.1307)和标准差(0.3081)归一化

2. 第一卷积块
   - 卷积层 1 (Conv1)：
     * 输入通道：1（灰度图像）
     * 输出通道：32（32个特征图）
     * 卷积核大小：3×3
     * 步长：1
     * 填充：1（保持特征图大小）
     * 激活函数：ReLU
   - 最大池化层 1 (MaxPool1)：
     * 池化窗口：2×2
     * 步长：2
     * 输出尺寸：14×14

3. 第二卷积块
   - 卷积层 2 (Conv2)：
     * 输入通道：32
     * 输出通道：64
     * 卷积核大小：3×3
     * 步长：1
     * 填充：1
     * 激活函数：ReLU
   - 最大池化层 2 (MaxPool2)：
     * 池化窗口：2×2
     * 步长：2
     * 输出尺寸：7×7

4. 全连接层
   - 展平层：
     * 将 64个 7×7 特征图展平
     * 输出维度：7×7×64 = 3136
   - 全连接层 1 (FC1)：
     * 输入维度：3136
     * 输出维度：128
     * 激活函数：ReLU
   - Dropout层：
     * 丢弃率：0.5
     * 作用：防止过拟合
   - 全连接层 2 (FC2)：
     * 输入维度：128
     * 输出维度：10（对应10个数字）
     * 激活函数：Log Softmax

### 工作原理

1. 特征提取（卷积层）
   - Conv1：提取基础特征
     * 边缘检测
     * 简单形状识别
     * 局部纹理特征
   - Conv2：组合特征
     * 复杂形状识别
     * 数字笔画组合
     * 局部结构特征

2. 特征降维（池化层）
   - 减少数据维度
   - 提取显著特征
   - 增加平移不变性
   - 控制过拟合

3. 分类决策（全连接层）
   - FC1：特征组合
     * 将局部特征组合成全局特征
     * 学习特征之间的关系
   - Dropout：正则化
     * 随机关闭一些神经元
     * 防止网络过度依赖某些特征
   - FC2：最终分类
     * 将特征映射到类别概率
     * 使用 Log Softmax 获得最终预测

### 模型特点

1. 结构优势
   - 层次化特征提取
   - 参数共享减少过拟合
   - 空间不变性强

2. 设计考虑
   - 适中的网络深度（避免过拟合）
   - 合理的通道数量增长
   - 有效的正则化策略

3. 计算效率
   - 总参数量：约 1.2M
   - 推理速度快
   - 内存占用小

### 训练策略

1. 损失函数：交叉熵损失
   - 适合多分类问题
   - 数值稳定性好

2. 优化器：Adam
   - 自适应学习率
   - 收敛速度快
   - 对超参数不敏感

3. 学习率调度
   - 初始学习率：0.001
   - 动态调整策略
   - 避免局部最小值

## 性能指标

在 MNIST 测试集上：
- 准确率：98%以上
- 训练时间：约5-10分钟（CPU）
- 预测时间：<0.1秒/张（CPU）

## 注意事项

1. 输入图片要求：
   - 格式：支持 PNG、JPG、JPEG
   - 背景：最好是白底黑字
   - 内容：单个手写数字，尽量居中

2. 环境要求：
   - Python 3.6+
   - PyTorch 1.13+
   - 至少 2GB 可用内存
   - 约 100MB 磁盘空间（包含数据集）

3. GPU 支持：
   - 如果有 CUDA 支持的 GPU，会自动使用
   - 需要安装对应的 CUDA 版本 PyTorch

## 常见问题

1. 图片预处理：
   - 如果识别效果不好，可以尝试调整图片的对比度
   - 确保数字笔画清晰，避免太多噪点

2. 可视化：
   - 如果显示窗口太大，可以调整 `visualize_model.py` 中的 `figsize` 参数
   - 可以修改颜色映射来改变可视化效果

## 未来改进

- [ ] 添加 Web 界面
- [ ] 支持实时摄像头识别
- [ ] 添加更多可视化选项
- [ ] 支持多数字识别
- [ ] 提供模型压缩版本

## 贡献指南

欢迎提交 Issue 和 Pull Request 来改进项目。提交时请：
1. 清晰描述改动的目的和实现方式
2. 确保代码风格一致
3. 添加必要的注释和文档
4. 更新相关的可视化功能 