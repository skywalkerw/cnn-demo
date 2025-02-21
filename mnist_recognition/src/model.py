import torch
import torch.nn as nn
import torch.nn.functional as F

class DigitRecognizer(nn.Module):
    def __init__(self):
        super(DigitRecognizer, self).__init__()
        # 第一个卷积层，1个输入通道，32个输出通道，3x3卷积核
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 第二个卷积层，32个输入通道，64个输出通道，3x3卷积核
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 第一个卷积层 + ReLU + MaxPool
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        # 第二个卷积层 + ReLU + MaxPool
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        # 展平张量
        x = x.view(-1, 64 * 7 * 7)
        
        # 全连接层 + ReLU + Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # 输出层
        x = self.fc2(x)
        return F.log_softmax(x, dim=1) 