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

def train(model, device, train_loader, optimizer, epoch, criterion, scaler=None):
    # 将模型设置为训练模式，这会启用dropout和batch normalization等训练特有的操作
    model.train()
    # 初始化训练损失、正确预测数量和总样本数
    train_loss = 0
    correct = 0
    total = 0
    
    # 使用tqdm创建进度条，显示当前训练进度
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    # 遍历训练数据集的每个批次
    for batch_idx, (data, target) in enumerate(progress_bar):
        # 使用非阻塞传输，允许数据传输和计算重叠
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        # 清除优化器中之前累积的梯度
        optimizer.zero_grad()
        
        # 混合精度训练（仅CUDA支持，MPS支持有限）
        if scaler is not None:
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 标准精度训练（MPS或CPU）
            # 前向传播：将数据输入模型得到预测结果
            output = model(data)
            # 计算预测结果与真实标签之间的损失
            loss = criterion(output, target)
            # 反向传播：计算损失对模型参数的梯度
            loss.backward()
            # 根据梯度更新模型参数
            optimizer.step()
        
        # 累加当前批次的损失
        train_loss += loss.item()
        # 获取预测结果中概率最大的类别作为预测标签
        pred = output.argmax(dim=1, keepdim=True)
        # 统计正确预测的数量
        correct += pred.eq(target.view_as(pred)).sum().item()
        # 累加已处理的样本总数
        total += target.size(0)
        
        # 更新进度条显示当前的平均损失和准确率
        progress_bar.set_postfix({
            'loss': f'{train_loss/(batch_idx+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    # 返回整个epoch的平均损失和准确率
    return train_loss/len(train_loader), 100.*correct/total

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            # 使用非阻塞传输
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            test_loss += criterion(output, target).item()
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
    
    # 检查是否可用CUDA或MPS (Apple Silicon)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
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
    
    # 根据硬件配置优化数据加载（M4 Pro: 12核CPU, 24GB内存, 16核GPU）
    # num_workers: M4 Pro有12核心，使用8个worker（留4个给系统和MPS处理）
    # batch_size: 16核GPU需要足够大的batch size才能充分利用并行能力
    #             24GB统一内存允许更大的batch size，模型很小（22.5万参数）
    # pin_memory: MPS使用统一内存架构，作用有限但保留也无妨
    # prefetch_factor: 增加预取数量，充分利用多核CPU处理能力
    use_gpu = device.type in ['cuda', 'mps']
    
    # 根据设备类型调整参数
    if device.type == 'mps':
        # M4 Pro优化配置：16核GPU + 24GB统一内存
        # 对于16核GPU，更大的batch size可以更好地利用并行能力
        # 可以尝试: 512, 768, 1024 (如果内存允许)
        train_batch_size = 768  # 16核GPU，使用更大的batch size以充分利用并行能力
        train_num_workers = 8   # 12核CPU，使用8个worker
        train_prefetch = 4      # 更多预取以隐藏I/O延迟
        pin_memory = False      # MPS统一内存，pin_memory意义不大
    else:
        # CUDA或CPU配置
        train_batch_size = 256
        train_num_workers = 4
        train_prefetch = 2
        pin_memory = use_gpu
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=train_num_workers,
        pin_memory=pin_memory,
        prefetch_factor=train_prefetch,
        persistent_workers=True if train_num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1000,
        num_workers=min(4, train_num_workers),  # 测试时使用较少进程
        pin_memory=pin_memory,
        prefetch_factor=train_prefetch,
        persistent_workers=True if train_num_workers > 0 else False
    )
    
    print(f"DataLoader配置: batch_size={train_batch_size}, num_workers={train_num_workers}, "
          f"prefetch_factor={train_prefetch}, pin_memory={pin_memory}")
    print(f"硬件配置: 12核CPU, 16核GPU, 24GB统一内存")
    print(f"提示: 如果遇到内存不足，可以将batch_size降至512或384")
    
    # 初始化模型
    original_model = DigitRecognizer().to(device)
    
    # 使用torch.compile优化模型（PyTorch 2.0+）
    # 注意：MPS目前不支持torch.compile，只有CUDA支持
    # 这会编译模型图，显著提升推理和训练速度
    if device.type == 'cuda' and hasattr(torch, 'compile'):
        try:
            model = torch.compile(original_model, mode='reduce-overhead')
            print("Model compiled with torch.compile for better performance")
        except Exception as e:
            print(f"torch.compile failed (will use normal mode): {e}")
            model = original_model
    elif device.type == 'mps':
        print("MPS device: torch.compile not supported, using eager mode")
        model = original_model
    else:
        model = original_model
    
    # 优化器设置
    # 对于更大的batch size，Adam优化器的默认学习率仍然适用（自适应优化器）
    # 如果需要，可以按比例调整：lr = base_lr * (batch_size / base_batch_size)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 创建损失函数对象（避免每次循环都创建新对象）
    criterion = nn.CrossEntropyLoss()
    
    # 混合精度训练（MPS支持有限）
    scaler = None
    if device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        print("Using mixed precision training (AMP)")
    elif device.type == 'mps':
        # MPS的混合精度支持有限，暂时不使用
        print("MPS detected: Using full precision training (mixed precision not recommended)")
        
        # MPS特定优化：确保使用高效的设置
        # 统一内存架构下，MPS会自动管理内存，无需额外设置
    
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
        train_loss, train_acc = train(model, device, train_loader, optimizer, epoch, criterion, scaler)
        test_loss, test_acc = test(model, device, test_loader, criterion)
        
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