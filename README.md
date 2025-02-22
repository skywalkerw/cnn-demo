# MNIST 手写数字识别项目

这是一个基于 PyTorch 实现的 MNIST 手写数字识别项目，包含了完整的模型训练和可视化功能。该项目不仅实现了基本的数字识别功能，还提供了丰富的神经网络内部状态可视化工具，帮助理解卷积神经网络的工作原理。

## 项目结构

```
mnist_recognition/
├── data/                   # 数据集目录
├── models/                 # 模型保存目录
├── test_images/           # 测试图片目录
├── visualizations/        # 可视化结果目录
└── src/                   # 源代码目录
    ├── model.py           # 模型定义
    ├── train.py           # 训练脚本
    ├── predict.py         # 预测脚本
    └── visualize_model.py # 可视化工具
```

## 环境要求

- Python 3.6+
- PyTorch 1.0+
- torchvision
- matplotlib
- numpy
- pillow
- tqdm
- networkx

安装依赖：
```bash
pip install torch torchvision matplotlib numpy pillow tqdm networkx
```

## 快速开始

1. **准备数据集**
```bash
cd src
python train.py  # 首次运行会自动下载MNIST数据集
```

2. **训练模型**
```bash
python train.py  # 训练完成后会保存最佳模型到 models/best_model.pth
```

3. **预测单个图片**
```bash
python predict.py ../test_images/test.png  # 预测单张图片
```

4. **运行可视化**
```bash
python visualize_model.py  # 生成所有可视化结果
```

### 预测功能说明
predict.py 提供了单张图片的预测功能：
- 支持多种图片格式（PNG、JPG、JPEG）
- 自动进行图片预处理（缩放、灰度化、标准化）
- 显示预测结果和置信度
- 可视化预处理后的图片效果

使用示例：
```bash
python predict.py ../test_images/test.png  # 基本预测
python predict.py ../test_images/test.png --show  # 显示处理过程
python predict.py ../test_images/test.png --save result.png  # 保存结果
```

## 模型架构

### 网络结构
- 输入层：28×28 灰度图像
- 卷积层1：32个3×3卷积核，ReLU激活，2×2最大池化
  * 输入尺寸：28×28×1
  * 输出尺寸：14×14×32
  * 参数量：(3×3×1+1)×32 = 320
- 卷积层2：64个3×3卷积核，ReLU激活，2×2最大池化
  * 输入尺寸：14×14×32
  * 输出尺寸：7×7×64
  * 参数量：(3×3×32+1)×64 = 18,496
- 全连接层1：7×7×64 → 128，ReLU激活，Dropout(0.5)
  * 输入维度：3,136 (7×7×64)
  * 输出维度：128
  * 参数量：3,136×128 + 128 = 401,536
- 全连接层2：128 → 10，Softmax输出
  * 输入维度：128
  * 输出维度：10
  * 参数量：128×10 + 10 = 1,290

总参数量：421,642

### 数据预处理
- 图像缩放：保持宽高比的情况下缩放到28×28
- 灰度化：转换为单通道灰度图像
- 像素归一化：值域从[0,255]映射到[0,1]
- 标准化：使用MNIST数据集的统计参数(均值=0.1307, 标准差=0.3081)

### 训练参数
- 优化器：Adam
- 损失函数：CrossEntropyLoss
- 批量大小：64
- 训练轮数：10
- 学习率：自适应（Adam）

## 可视化功能

项目提供六种可视化工具，保存在 visualizations/ 目录：

1. **卷积核可视化** (01_conv1_filters.png, 02_conv2_filters.png)
   - 展示每层卷积核的权重分布
   - 通过灰度值展示权重大小
   - 包含详细的中文说明

2. **权重分布** (03_weights_distribution.png)
   - 展示所有层的权重分布直方图
   - 用于监控网络参数的健康状态

3. **特征图** (04_feature_maps.png)
   - 展示输入图像在各层的特征提取结果
   - 直观显示网络的特征学习过程

4. **像素贡献度** (05_pixel_contribution.png)
   - 分析输入图像各像素对分类结果的贡献
   - 使用热力图展示重要区域

5. **网络结构** (06_network_structure.png)
   - 可视化完整的网络架构
   - 展示数据流动和层间连接
   - 包含激活值的动态展示

## 性能指标

- 训练集准确率：~99.5%
- 测试集准确率：~99%
- 单张图片推理时间：<0.1s (CPU)

## 注意事项

1. 字体设置
   - Windows用户使用SimHei
   - Linux用户使用DejaVu Sans
   - macOS用户使用Arial Unicode MS

2. 数据集下载
   - 使用国内镜像源加速下载
   - 首次运行时自动下载到data目录

3. 可视化运行
   - 需要先完成模型训练
   - 确保test_images目录包含测试图片
   - 建议使用较大显示器查看结果

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。提交时请：
1. 遵循现有的代码风格
2. 添加适当的注释和文档
3. 确保所有可视化功能正常工作

## 许可证

MIT License

## 联系方式

如有问题或建议，请通过Issue与我们联系。
