import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from model import DigitRecognizer
from torchvision import transforms
from PIL import Image
import os
import torch.nn.functional as F

def visualize_filters(model, layer_name, save_path=None):
    """可视化指定卷积层的滤波器"""
    # 获取指定层
    layer = dict(model.named_modules())[layer_name]
    if not isinstance(layer, nn.Conv2d):
        raise ValueError(f"{layer_name} 不是卷积层")
    
    # 获取权重
    weights = layer.weight.data.cpu()
    n_filters = weights.shape[0]
    
    # 计算子图布局
    n_cols = 8
    n_rows = (n_filters + n_cols - 1) // n_cols
    
    # 创建图形
    plt.figure(figsize=(2*n_cols, 2*n_rows))
    for i in range(n_filters):
        plt.subplot(n_rows, n_cols, i+1)
        # 对于单通道，直接显示
        if weights.shape[1] == 1:
            plt.imshow(weights[i, 0], cmap='gray')
        # 对于多通道，取平均
        else:
            plt.imshow(weights[i].mean(0), cmap='gray')
        plt.axis('off')
    plt.suptitle(f'Filters visualization - {layer_name}')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def visualize_feature_maps(model, image_path, save_dir=None):
    """可视化每一层的特征图"""
    # 图像预处理
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Pad(padding=3, fill=0),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1.0 - x),
        transforms.Lambda(lambda x: torch.clamp((x - x.min()) / (x.max() - x.min() + 1e-8) * 1.5, 0, 1)),
        transforms.Lambda(lambda x: torch.where(x > 0.085, torch.ones_like(x), torch.zeros_like(x))),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载和处理图像
    image = Image.open(image_path).convert('L')
    image_tensor = transform(image).unsqueeze(0)
    
    # 注册钩子来获取特征图
    feature_maps = {}
    def hook_fn(module, input, output):
        feature_maps[module] = output.detach().cpu()
    
    # 为所有卷积层和池化层注册钩子
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.MaxPool2d)):
            module.register_forward_hook(hook_fn)
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
    
    # 可视化每一层的特征图
    for module, feature_map in feature_maps.items():
        feature_map = feature_map.squeeze(0)
        n_features = feature_map.shape[0]
        
        # 计算子图布局
        n_cols = 8
        n_rows = (n_features + n_cols - 1) // n_cols
        
        plt.figure(figsize=(2*n_cols, 2*n_rows))
        for i in range(n_features):
            plt.subplot(n_rows, n_cols, i+1)
            plt.imshow(feature_map[i], cmap='gray')
            plt.axis('off')
        
        layer_type = module.__class__.__name__
        plt.suptitle(f'Feature maps - {layer_type}')
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'feature_maps_{layer_type}.png'))
        plt.show()

def visualize_weights_distribution(model, save_path=None):
    """可视化模型各层权重的分布"""
    n_weights = sum(1 for name, _ in model.named_parameters() if 'weight' in name)
    plt.figure(figsize=(5*n_weights, 4))
    plot_idx = 1
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            plt.subplot(1, n_weights, plot_idx)
            plt.hist(param.data.cpu().numpy().flatten(), bins=50)
            plt.title(f'Distribution - {name}')
            plt.xlabel('Weight value')
            plt.ylabel('Frequency')
            plot_idx += 1
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def visualize_parameters(model, save_dir=None):
    """可视化模型的所有参数"""
    # 创建一个大图，包含所有参数的热力图和统计信息
    params_info = []
    max_shape = (0, 0)
    
    # 收集参数信息
    for name, param in model.named_parameters():
        data = param.data.cpu().numpy()
        stats = {
            'name': name,
            'shape': data.shape,
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'data': data
        }
        params_info.append(stats)
        if len(data.shape) == 2:  # 对于2D参数，记录最大形状
            max_shape = (max(max_shape[0], data.shape[0]), 
                        max(max_shape[1], data.shape[1]))
    
    n_params = len(params_info)
    fig = plt.figure(figsize=(15, 4*n_params))
    gs = plt.GridSpec(n_params, 3, width_ratios=[2, 1, 1])
    
    for idx, param_info in enumerate(params_info):
        # 热力图
        ax1 = fig.add_subplot(gs[idx, 0])
        data = param_info['data']
        if len(data.shape) == 1:  # 1D参数（偏置）
            data = data.reshape(1, -1)
        elif len(data.shape) == 4:  # 卷积核
            # 重塑卷积核为2D显示
            n_filters, n_channels, h, w = data.shape
            data = data.reshape(n_filters * h, n_channels * w)
        im = ax1.imshow(data, cmap='coolwarm', aspect='auto')
        plt.colorbar(im, ax=ax1)
        ax1.set_title(f"Parameter: {param_info['name']}\nShape: {param_info['shape']}")
        
        # 直方图
        ax2 = fig.add_subplot(gs[idx, 1])
        ax2.hist(param_info['data'].flatten(), bins=50, density=True)
        ax2.set_title('Value Distribution')
        
        # 统计信息
        ax3 = fig.add_subplot(gs[idx, 2])
        stats_text = f"""Statistics:
Mean: {param_info['mean']:.6f}
Std: {param_info['std']:.6f}
Min: {param_info['min']:.6f}
Max: {param_info['max']:.6f}
Shape: {param_info['shape']}
Size: {np.prod(param_info['shape'])}"""
        ax3.text(0.1, 0.5, stats_text, 
                transform=ax3.transAxes,
                verticalalignment='center',
                fontfamily='monospace')
        ax3.axis('off')
    
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'parameters_visualization.png'))
    plt.show()

def visualize_pixel_contribution(model, image_path, save_dir=None):
    """可视化输入图像像素对预测结果的贡献度"""
    # 图像预处理
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Pad(padding=3, fill=0),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 1.0 - x),
        transforms.Lambda(lambda x: torch.clamp((x - x.min()) / (x.max() - x.min() + 1e-8) * 1.5, 0, 1)),
        transforms.Lambda(lambda x: torch.where(x > 0.085, torch.ones_like(x), torch.zeros_like(x))),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载和处理图像
    original_image = Image.open(image_path).convert('L')
    image_tensor = transform(original_image).unsqueeze(0)
    image_tensor.requires_grad_()  # 启用梯度计算
    
    # 获取预测结果
    model.eval()
    output = model(image_tensor)
    pred_prob = F.softmax(output, dim=1)
    pred_class = output.argmax(dim=1).item()
    confidence = pred_prob[0, pred_class].item()
    
    # 计算梯度
    model.zero_grad()
    pred_prob[0, pred_class].backward()
    
    # 获取输入图像的梯度
    gradients = image_tensor.grad[0, 0].numpy()
    processed_image = image_tensor[0, 0].detach().numpy()
    
    # 创建可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'Pixel Contribution Analysis for Digit {pred_class} (Confidence: {confidence:.2%})')
    
    # 原始图像
    axes[0, 0].imshow(original_image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # 处理后的图像
    axes[0, 1].imshow(processed_image, cmap='gray')
    axes[0, 1].set_title('Processed Image')
    axes[0, 1].axis('off')
    
    # 梯度的绝对值（像素重要性）
    importance = np.abs(gradients)
    im = axes[0, 2].imshow(importance, cmap='hot')
    axes[0, 2].set_title('Pixel Importance')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2])
    
    # 正梯度（正向贡献）
    positive_grad = np.maximum(gradients, 0)
    im = axes[1, 0].imshow(positive_grad, cmap='Greens')
    axes[1, 0].set_title('Positive Contribution')
    axes[1, 0].axis('off')
    plt.colorbar(im, ax=axes[1, 0])
    
    # 负梯度（负向贡献）
    negative_grad = np.abs(np.minimum(gradients, 0))
    im = axes[1, 1].imshow(negative_grad, cmap='Reds')
    axes[1, 1].set_title('Negative Contribution')
    axes[1, 1].axis('off')
    plt.colorbar(im, ax=axes[1, 1])
    
    # 叠加显示
    overlay = processed_image.copy()
    overlay = np.stack([overlay] * 3, axis=-1)  # 转换为RGB
    importance_normalized = importance / importance.max()
    overlay[..., 0] = np.maximum(overlay[..., 0], importance_normalized)  # 红色通道
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title('Importance Overlay')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'pixel_contribution_{pred_class}.png'))
    plt.show()

def main():
    # 创建保存目录
    os.makedirs('visualizations', exist_ok=True)
    
    # 加载模型
    model = DigitRecognizer()
    model.load_state_dict(torch.load('models/best_model.pth'))
    model.eval()
    
    # 可视化模型参数
    visualize_parameters(model, 'visualizations')
    
    # 可视化第一个卷积层的滤波器
    visualize_filters(model, 'conv1', 'visualizations/conv1_filters.png')
    
    # 可视化第二个卷积层的滤波器
    visualize_filters(model, 'conv2', 'visualizations/conv2_filters.png')
    
    # 可视化权重分布
    visualize_weights_distribution(model, 'visualizations/weights_distribution.png')
    
    # 可视化特征图和像素贡献（使用测试图像）
    test_dir = '../test_images'
    test_images = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if test_images:
        image_path = os.path.join(test_dir, test_images[0])
        visualize_feature_maps(model, image_path, 'visualizations')
        visualize_pixel_contribution(model, image_path, 'visualizations')

if __name__ == '__main__':
    main() 