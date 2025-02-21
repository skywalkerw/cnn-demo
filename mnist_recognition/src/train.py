import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import DigitRecognizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import ssl
import torchvision.datasets.mnist as mnist

# Disable SSL certificate verification
ssl._create_default_https_context = ssl._create_unverified_context

# 修改MNIST数据集下载地址为国内镜像
mnist.URLS = [
    'https://mirrors.tuna.tsinghua.edu.cn/pytorch-datasets/MNIST/train-images-idx3-ubyte.gz',
    'https://mirrors.tuna.tsinghua.edu.cn/pytorch-datasets/MNIST/train-labels-idx1-ubyte.gz',
    'https://mirrors.tuna.tsinghua.edu.cn/pytorch-datasets/MNIST/t10k-images-idx3-ubyte.gz',
    'https://mirrors.tuna.tsinghua.edu.cn/pytorch-datasets/MNIST/t10k-labels-idx1-ubyte.gz'
]

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        progress_bar.set_postfix({
            'loss': f'{train_loss/(batch_idx+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return train_loss/len(train_loader), 100.*correct/total

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)\n')
    return test_loss, accuracy

def main():
    # 设置随机种子
    torch.manual_seed(42)
    
    # 检查是否可用CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载数据集（使用上级目录中的数据）
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    train_dataset = datasets.MNIST(data_dir, train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)
    
    # 初始化模型
    model = DigitRecognizer().to(device)
    optimizer = optim.Adam(model.parameters())
    
    # 训练参数
    epochs = 10
    best_accuracy = 0
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    # 创建模型保存目录
    os.makedirs('models', exist_ok=True)
    
    # 训练循环
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, device, train_loader, optimizer, epoch)
        test_loss, test_acc = test(model, device, test_loader)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        # 保存最佳模型
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), 'models/best_model.pth')
    
    # 绘制训练过程
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

if __name__ == '__main__':
    main() 