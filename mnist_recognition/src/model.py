import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义数字识别模型类，继承自nn.Module
class DigitRecognizer(nn.Module):
    def __init__(self):
        # 调用父类的初始化方法
        super(DigitRecognizer, self).__init__()
        
        # 第一个卷积层
        # 输入通道数：1（灰度图像）
        # 输出通道数：32（提取32个特征）
        # 卷积核大小：3x3
        # padding=1：保持特征图大小不变
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        
        # 第二个卷积层
        # 输入通道数：32（来自第一层）
        # 输出通道数：64（提取64个特征）
        # 卷积核大小：3x3
        # padding=1：保持特征图大小不变
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # 第一个全连接层
        # 输入大小：64*7*7（经过两次池化后的特征图大小）
        # 输出大小：128（隐藏层神经元数量）
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        
        # 第二个全连接层（输出层）
        # 输入大小：128（来自fc1）
        # 输出大小：10（对应0-9十个数字）
        self.fc2 = nn.Linear(128, 10)
        
        # Dropout层，随机丢弃50%的神经元，防止过拟合
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 第一个卷积块
        # 1. 卷积操作
        # 2. ReLU激活函数（增加非线性）
        # 3. 2x2最大池化（下采样，减少特征图大小）
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        # 第二个卷积块
        # 1. 卷积操作
        # 2. ReLU激活函数
        # 3. 2x2最大池化
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        # 将特征图展平成一维向量
        # -1表示自动计算batch size
        # 64*7*7是特征图的总大小
        x = x.view(-1, 64 * 7 * 7)
        
        # 全连接层处理
        # 1. 全连接层 + ReLU激活
        # 2. Dropout随机丢弃部分神经元
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # 输出层
        # 1. 全连接层得到10个输出
        # 2. log_softmax将输出转换为概率分布
        x = self.fc2(x)
        return F.log_softmax(x, dim=1) 