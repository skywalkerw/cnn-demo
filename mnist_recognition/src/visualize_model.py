import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from model import DigitRecognizer
from torchvision import transforms
from PIL import Image
import os
import torch.nn.functional as F

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac OS的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 根据操作系统设置合适的中文字体
def set_chinese_font():
    """设置中文字体"""
    import platform
    system = platform.system()
    
    if system == 'Windows':
        plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows的中文字体
    elif system == 'Linux':
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # Linux的中文字体
    elif system == 'Darwin':  # macOS
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac的中文字体
    
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 调用字体设置函数
set_chinese_font()

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
    
    # 创建图形，设置更大的画布
    fig = plt.figure(figsize=(2*n_cols, 2*n_rows + 4))  # 增加高度
    
    # 创建两个子图区域，并调整它们之间的间距
    gs = plt.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.4)  # 增加间距
    
    # 上半部分绘制滤波器
    ax_filters = fig.add_subplot(gs[0])
    ax_filters.set_frame_on(False)
    ax_filters.set_xticks([])
    ax_filters.set_yticks([])
    
    # 在上半部分区域中创建滤波器子图
    for i in range(n_filters):
        ax = plt.subplot(n_rows, n_cols, i+1)
        if weights.shape[1] == 1:
            ax.imshow(weights[i, 0], cmap='gray')
        else:
            ax.imshow(weights[i].mean(0), cmap='gray')
        ax.axis('off')
    
    plt.suptitle(f'Filters visualization - {layer_name}', y=0.98)  # 调整标题位置
    
    # 下半部分放置说明文字
    ax_text = fig.add_subplot(gs[1])
    description = (
        f"该图展示了{layer_name}层的卷积核可视化结果。每个小方块代表一个卷积核，不同的灰度值表示卷积核中\n"
        f"不同位置的权重大小。亮色区域表示正权重，暗色区域表示负权重。这些卷积核用于提取输入图像中的\n"
        f"不同特征，如边缘、纹理等。"
    )
    ax_text.text(0.05, 0.3, description,
                wrap=False,
                va='center',
                fontsize=10)
    ax_text.axis('off')
    
    # 调整布局，确保子图之间有足够间距
    plt.subplots_adjust(top=0.95, bottom=0.05, hspace=0.5)  # 增加间距
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def visualize_feature_maps(model, image_path, save_path=None):
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
    for i, (module, feature_map) in enumerate(feature_maps.items()):
        feature_map = feature_map.squeeze(0)
        n_features = feature_map.shape[0]
        
        # 计算子图布局
        n_cols = 8
        n_rows = (n_features + n_cols - 1) // n_cols
        
        # 创建图形，设置更大的画布
        fig = plt.figure(figsize=(2*n_cols, 2*n_rows + 3))
        
        # 创建两个子图区域，并调整它们之间的间距
        gs = plt.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.3)
        
        # 上半部分绘制特征图
        ax_features = fig.add_subplot(gs[0])
        ax_features.set_frame_on(False)
        ax_features.set_xticks([])
        ax_features.set_yticks([])
        
        # 在上半部分区域中创建特征图子图
        for j in range(n_features):
            ax = plt.subplot(n_rows, n_cols, j+1)
            ax.imshow(feature_map[j], cmap='gray')
            ax.axis('off')
        
        layer_type = module.__class__.__name__
        plt.suptitle(f'Feature maps - {layer_type}', y=0.95)
        
        # 下半部分放置说明文字
        ax_text = fig.add_subplot(gs[1])
        description = (
            f"该图展示了{layer_type}层的特征图。每个小方块代表一个特征图，显示了该层提取到的不同特征。亮度表示\n"
            f"特征的激活强度，越亮表示该位置的特征越显著。不同的特征图关注输入图像的不同方面，共同构成了对\n"
            f"输入图像的多维度理解。"
        )
        ax_text.text(0.05, 0.5, description,
                    wrap=False,
                    va='center',
                    fontsize=10)
        ax_text.axis('off')
        
        # 调整布局，确保子图之间有足够间距
        plt.subplots_adjust(top=0.9, bottom=0.1, hspace=0.4)
        
        if save_path:
            layer_filename = f"{os.path.splitext(save_path)[0]}_{i+1}_{layer_type}.png"
            plt.savefig(layer_filename, bbox_inches='tight', dpi=300)
        plt.close()

def visualize_weights_distribution(model, save_path=None):
    """可视化模型各层权重的分布"""
    # 获取所有权重层
    weight_layers = [(name, param) for name, param in model.named_parameters() if 'weight' in name]
    n_weights = len(weight_layers)
    
    # 创建图形，根据权重层数量调整大小
    fig = plt.figure(figsize=(5*n_weights, 7))
    
    # 创建两个子图区域，并调整它们之间的间距
    gs = plt.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.3)
    
    # 上半部分绘制权重分布
    ax_dist = fig.add_subplot(gs[0])
    
    # 绘制每个权重层的分布
    for idx, (name, param) in enumerate(weight_layers):
        plt.subplot(1, n_weights, idx+1)
        weights = param.data.cpu().numpy().flatten()
        plt.hist(weights, bins=50, density=True, alpha=0.7)
        plt.title(f'Distribution - {name}')
        plt.xlabel('Weight value')
        if idx == 0:  # 只在第一个子图显示y轴标签
            plt.ylabel('Density')
    
    plt.suptitle('Neural Network Weights Distribution', y=0.95)
    
    # 下半部分放置说明文字
    ax_text = fig.add_subplot(gs[1])
    description = (
        "该图展示了神经网络各层权重的分布情况。横轴表示权重值，纵轴表示该权重值出现的频率。钟形曲线表示\n"
        "权重呈现正态分布，这是神经网络训练良好的一个标志。不同层的权重分布反映了该层在网络中的作用和\n"
        "学习特征的复杂程度。"
    )
    ax_text.text(0.05, 0.5, description,
                wrap=False,
                va='center',
                fontsize=10)
    ax_text.axis('off')
    
    # 调整布局，确保子图之间有足够间距
    plt.subplots_adjust(top=0.9, bottom=0.1, hspace=0.4)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

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

def normalize_data(data):
    """规范化数据到[0,1]范围"""
    data_min = data.min()
    data_max = data.max()
    if data_max != data_min:
        return (data - data_min) / (data_max - data_min)
    return data

def visualize_pixel_contribution(model, image_path, save_path=None):
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
    
    # 获取输入图像的梯度并规范化
    gradients = image_tensor.grad[0, 0].numpy()
    processed_image = image_tensor[0, 0].detach().numpy()
    
    # 规范化所有要显示的数据
    processed_image_norm = normalize_data(processed_image)
    gradients_norm = normalize_data(gradients)
    importance = np.abs(gradients_norm)
    positive_grad = normalize_data(np.maximum(gradients, 0))
    negative_grad = normalize_data(np.abs(np.minimum(gradients, 0)))
    
    # 调整图形大小和布局
    fig = plt.figure(figsize=(15, 12))  # 增加更多高度
    plt.subplots_adjust(bottom=0.25)  # 为底部文字留出空间
    
    # 原始图像
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(np.array(original_image), cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # 处理后的图像
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(processed_image_norm, cmap='gray')
    ax2.set_title('Processed Image')
    ax2.axis('off')
    
    # 梯度的绝对值（像素重要性）
    ax3 = fig.add_subplot(2, 3, 3)
    im = ax3.imshow(importance, cmap='hot')
    ax3.set_title('Pixel Importance')
    ax3.axis('off')
    plt.colorbar(im, ax=ax3)
    
    # 正梯度（正向贡献）
    ax4 = fig.add_subplot(2, 3, 4)
    im = ax4.imshow(positive_grad, cmap='Greens')
    ax4.set_title('Positive Contribution')
    ax4.axis('off')
    plt.colorbar(im, ax=ax4)
    
    # 负梯度（负向贡献）
    ax5 = fig.add_subplot(2, 3, 5)
    im = ax5.imshow(negative_grad, cmap='Reds')
    ax5.set_title('Negative Contribution')
    ax5.axis('off')
    plt.colorbar(im, ax=ax5)
    
    # 叠加显示
    overlay = processed_image_norm.copy()
    overlay = np.stack([overlay] * 3, axis=-1)  # 转换为RGB
    overlay[..., 0] = np.maximum(overlay[..., 0], importance)  # 红色通道
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.imshow(overlay)
    ax6.set_title('Importance Overlay')
    ax6.axis('off')
    
    plt.suptitle(f'Pixel Contribution Analysis for Digit {pred_class} (Confidence: {confidence:.2%})', 
                y=0.95)
    
    # 添加底部说明文字
    description = (
        f"该图展示了输入图像中各像素对最终预测结果（数字{pred_class}）的贡献度分析。左上：原始输入图像；\n"
        f"右上：处理后的图像；中上：像素重要性热力图，越亮表示该像素对预测结果的影响越大。左下：正向贡献\n"
        f"（绿色），表示支持当前预测的像素；中下：负向贡献（红色），表示反对当前预测的像素；右下：重要性\n"
        f"叠加显示。模型预测置信度为{confidence:.2%}。"
    )
    
    plt.figtext(0.1, 0.08, description,  # 调整文字位置
                wrap=False, horizontalalignment='left', 
                fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)

def visualize_network_structure(model, input_size=(1, 1, 28, 28), save_path=None):
    """可视化神经网络结构，展示每一层的神经元和连接"""
    # 调整图形大小比例，使其更宽
    plt.figure(figsize=(24, 8))  # 增加宽度，减小高度
    
    # 创建一个数字2的示例输入
    sample_input = torch.zeros((1, 1, 28, 28))
    # 简单绘制数字2的形状
    sample_input[0, 0, 5:20, 15:20] = 1  # 竖线
    sample_input[0, 0, 5:8, 10:20] = 1   # 上横线
    sample_input[0, 0, 12:15, 10:20] = 1 # 中横线
    sample_input[0, 0, 17:20, 10:20] = 1 # 下横线
    
    # 获取每层的激活值
    activations = {}
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    # 注册钩子
    handles = []
    handles.append(model.conv1.register_forward_hook(hook_fn('conv1')))
    handles.append(model.conv2.register_forward_hook(hook_fn('conv2')))
    handles.append(model.fc1.register_forward_hook(hook_fn('fc1')))
    handles.append(model.fc2.register_forward_hook(hook_fn('fc2')))
    
    # 前向传播获取激活值
    model.eval()
    with torch.no_grad():
        output = model(sample_input)
    
    # 移除钩子
    for handle in handles:
        handle.remove()
    
    # 定义层的位置和神经元数量，并添加激活信息
    layers = [
        {'name': '输入层\nInput', 'neurons': 8, 'x': 0.08, 'color': 'blue', 
         'details': '输入大小 Input Size: 28×28\n神经元数量 Neurons: 784',
         'activation': sample_input.squeeze().numpy(),
         'y_offset': 0},
        
        {'name': '卷积层1\nConv1', 'neurons': 8, 'x': 0.22, 'color': 'green',
         'details': '卷积核 Kernel: 3×3\n特征图数 Filters: 32\n输出大小 Output: 28×28',
         'activation': activations['conv1'].mean(dim=(2, 3)).squeeze().numpy(),
         'y_offset': 0},
        
        {'name': '池化层1\nPool1', 'neurons': 8, 'x': 0.36, 'color': 'yellow',
         'details': '特征图数 Filters: 32\n输出大小 Output: 14×14',
         'activation': F.max_pool2d(activations['conv1'], 2).mean(dim=(2, 3)).squeeze().numpy(),
         'y_offset': 0},
        
        {'name': '卷积层2\nConv2', 'neurons': 8, 'x': 0.50, 'color': 'green',
         'details': '卷积核 Kernel: 3×3\n特征图数 Filters: 64\n输出大小 Output: 14×14',
         'activation': activations['conv2'].mean(dim=(2, 3)).squeeze().numpy(),
         'y_offset': 0},
        
        {'name': '池化层2\nPool2', 'neurons': 8, 'x': 0.64, 'color': 'yellow',
         'details': '特征图数 Filters: 64\n输出大小 Output: 7×7',
         'activation': F.max_pool2d(activations['conv2'], 2).mean(dim=(2, 3)).squeeze().numpy(),
         'y_offset': 0},
        
        {'name': '全连接层1\nFC1', 'neurons': 8, 'x': 0.78, 'color': 'orange',
         'details': 'ReLU激活 ReLU\n神经元数量 Neurons: 128',
         'activation': activations['fc1'].squeeze().numpy(),
         'y_offset': 0},
        
        {'name': '输出层\nOutput', 'neurons': 10, 'x': 0.92, 'color': 'red',
         'details': 'Softmax激活 Softmax\n神经元数量 Neurons: 10',
         'activation': F.softmax(activations['fc2'], dim=1).squeeze().numpy(),
         'y_offset': 0}
    ]
    
    # 在左侧添加输入图像示例（调整位置和大小）
    ax_img = plt.axes([0.02, 0.35, 0.08, 0.3])
    ax_img.imshow(sample_input.squeeze(), cmap='gray')
    ax_img.axis('off')
    
    # 创建主绘图区域（调整比例）
    ax_main = plt.axes([0.12, 0.2, 0.85, 0.65])  # 调整高度
    
    # 隐藏边框和刻度
    ax_main.set_frame_on(False)
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    
    # 调整层的水平位置，使分布更均匀
    x_positions = {
        '输入层': 0.08,
        '卷积层1': 0.22,
        '池化层1': 0.36,
        '卷积层2': 0.50,
        '池化层2': 0.64,
        '全连接层1': 0.78,
        '输出层': 0.92
    }
    
    # 更新每层的x坐标
    for layer in layers:
        layer_name = layer['name'].split('\n')[0]
        layer['x'] = x_positions[layer_name]
    
    # 绘制神经元和连接
    for i, layer in enumerate(layers):
        # 获取当前层的归一化激活值
        act_values = normalize_data(layer['activation'])
        if len(act_values.shape) > 1:
            act_values = act_values.mean(axis=tuple(range(1, len(act_values.shape))))
        
        # 统一所有层的垂直分布
        n_neurons = layer['neurons']
        
        # 计算神经元的垂直位置（使用相对位置）
        if layer['name'].startswith('输出层'):
            y_points = np.linspace(0.1, 0.9, n_neurons)
        else:
            y_points = np.linspace(0.15, 0.85, n_neurons)
            
        # 绘制连接线（在绘制神经元之前）
        if i > 0:
            prev_layer = layers[i-1]
            prev_n_neurons = prev_layer['neurons']
            
            # 确保前一层使用相同的垂直分布范围
            if prev_layer['name'].startswith('输出层'):
                prev_y_points = np.linspace(0.1, 0.9, prev_n_neurons)
            else:
                prev_y_points = np.linspace(0.15, 0.85, prev_n_neurons)
            
            # 获取前一层的归一化激活值
            prev_act_values = normalize_data(prev_layer['activation'])
            if len(prev_act_values.shape) > 1:
                prev_act_values = prev_act_values.mean(axis=tuple(range(1, len(prev_act_values.shape))))
            
            # 计算连接权重
            connections = []
            max_weight = 0
            
            # 为每个连接计算权重
            for j1, y1 in enumerate(prev_y_points):
                for j2, y2 in enumerate(y_points):
                    if j1 < len(prev_act_values) and j2 < len(act_values):
                        # 使用两端神经元的激活值的平均值作为权重
                        weight = (prev_act_values[j1].item() + act_values[j2].item()) / 2
                        connections.append((j1, j2, y1, y2, weight))
                        max_weight = max(max_weight, weight)
            
            # 按权重排序并绘制连接线
            for j1, j2, y1, y2, weight in sorted(connections, key=lambda x: x[4]):
                if weight > 0:  # 只绘制有激活的连接
                    # 根据权重确定线条样式
                    if weight > 0.8 * max_weight:
                        color = 'red'
                        alpha = 0.6
                        line_width = 1.5
                    else:
                        color = 'gray'
                        alpha = 0.05 + 0.2 * weight
                        line_width = 0.5 + weight
                    
                    # 绘制连接线
                    ax_main.plot([prev_layer['x'], layer['x']], [y1, y2], 
                               '-', color=color, alpha=alpha, linewidth=line_width,
                               zorder=1)
                    
                    # 为强连接添加权重标注
                    if weight > 0.5:
                        mid_x = (prev_layer['x'] + layer['x']) / 2
                        mid_y = (y1 + y2) / 2
                        plt.text(mid_x, mid_y, f'{weight:.2f}',
                               ha='center', va='center', fontsize=6,
                               color=color, alpha=alpha,
                               transform=ax_main.transAxes)
        
        # 绘制神经元
        for j, y in enumerate(y_points):
            if j < len(act_values):
                alpha = float(0.2 + 0.8 * act_values[j].item())
                activation = act_values[j].item()
            else:
                alpha = 0.2
                activation = 0.0
            
            # 统一圆圈大小
            radius = 0.012
            circle = plt.Circle((layer['x'], y), radius, color=layer['color'], 
                              alpha=alpha, zorder=2)  # 确保神经元在线的上层
            ax_main.add_patch(circle)
            
            # 为高激活的神经元添加标注
            if activation > 0.5:  # 可以调整这个阈值
                plt.text(layer['x'], y + 0.02, f'{activation:.2f}',
                        ha='center', va='bottom', fontsize=6,
                        color=layer['color'], alpha=0.9,
                        transform=ax_main.transAxes)
            
            # 输出层的处理
            if layer['name'].startswith('输出层'):
                # 在圆圈中添加数字
                plt.text(layer['x'], y, str(j), 
                        ha='center', va='center', fontsize=8,
                        color='black', fontweight='bold',
                        transform=ax_main.transAxes)
                # 添加激活值
                if j < len(act_values):
                    activation_value = f"{act_values[j].item():.2f}"
                    plt.text(layer['x'] + 0.025, y, activation_value,
                            ha='left', va='center', fontsize=8,
                            transform=ax_main.transAxes)
        
        # 调整文字位置
        plt.text(layer['x'], 0.95, layer['name'],
                ha='center', va='bottom', fontsize=10,
                transform=ax_main.transAxes)
        plt.text(layer['x'], 0.05, layer['details'],
                ha='center', va='top', fontsize=8,
                transform=ax_main.transAxes,
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', 
                         pad=0.1, boxstyle='round,pad=0.1'))
    
    # 调整坐标轴范围
    ax_main.set_xlim(-0.02, 1.02)
    ax_main.set_ylim(0, 1.0)
    ax_main.set_aspect('auto')  # 改为自动调整纵横比
    
    # 获取预测结果和置信度
    with torch.no_grad():
        output = model(sample_input)
        probabilities = F.softmax(output, dim=1)
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    # 添加底部说明文字
    description = (
        f"该图展示了卷积神经网络识别数字3时的内部激活状态。输入的手写数字经过两个卷积池化层提取特征，"
        f"然后通过全连接层进行分类。红色连线表示较强的神经元激活连接(激活值>0.8)。\n"
        f"模型预测结果：数字 {predicted_class}，置信度：{confidence:.2%}。"
        f"输出层显示各数字的概率分布，可以看到数字3对应的神经元激活值为{probabilities[0, 3]:.2f}，"
        f"是所有数字中最高的。"
    )
    
    plt.figtext(0.1, 0.02, description, 
                wrap=True, horizontalalignment='left', 
                fontsize=10)
    
    # 调整标题位置
    plt.suptitle("数字3的神经网络激活状态\nNeural Network Activation for Digit 3", 
                 y=0.95, fontsize=14)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def visualize_activation_distribution(model, image_path, save_path=None):
    """可视化每一层的激活值分布"""
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
    
    # 注册钩子来获取激活值
    activations = {}
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach().cpu()
        return hook
    
    # 为所有层注册钩子
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU, nn.MaxPool2d)):
            handles.append(module.register_forward_hook(hook_fn(name)))
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
    
    # 移除钩子
    for handle in handles:
        handle.remove()
    
    # 创建图形
    n_layers = len(activations)
    fig = plt.figure(figsize=(15, 5))
    
    # 绘制每一层的激活值分布
    for idx, (name, activation) in enumerate(activations.items()):
        plt.subplot(1, n_layers, idx+1)
        activation_flat = activation.numpy().flatten()
        plt.hist(activation_flat, bins=50, density=True, alpha=0.7)
        plt.title(f'Activation - {name}')
        plt.xlabel('Activation value')
        if idx == 0:
            plt.ylabel('Density')
    
    plt.suptitle('Neural Network Activation Distribution', y=0.95)
    
    # 添加说明文字
    description = (
        "该图展示了神经网络各层的激活值分布。横轴表示激活值，纵轴表示该激活值出现的频率。\n"
        "不同层的激活分布反映了该层对输入特征的响应特性。"
    )
    plt.figtext(0.1, 0.02, description, wrap=True, fontsize=10)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def visualize_gradient_flow(model, image_path, save_path=None):
    """可视化梯度流动"""
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
    image_tensor.requires_grad_()
    
    # 注册钩子来获取梯度
    gradients = {}
    def hook_fn(name):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                gradients[name] = grad_output[0].detach().cpu()
        return hook
    
    # 为所有层注册钩子
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_backward_hook(hook_fn(name)))
    
    # 前向传播和反向传播
    model.eval()
    output = model(image_tensor)
    loss = F.cross_entropy(output, torch.tensor([3]))  # 假设目标类别为3
    loss.backward()
    
    # 移除钩子
    for handle in handles:
        handle.remove()
    
    # 创建图形
    n_layers = len(gradients)
    fig = plt.figure(figsize=(15, 5))
    
    # 绘制每一层的梯度分布
    for idx, (name, gradient) in enumerate(gradients.items()):
        plt.subplot(1, n_layers, idx+1)
        gradient_flat = gradient.numpy().flatten()
        plt.hist(gradient_flat, bins=50, density=True, alpha=0.7)
        plt.title(f'Gradient - {name}')
        plt.xlabel('Gradient value')
        if idx == 0:
            plt.ylabel('Density')
    
    plt.suptitle('Neural Network Gradient Flow', y=0.95)
    
    # 添加说明文字
    description = (
        "该图展示了神经网络各层的梯度分布。横轴表示梯度值，纵轴表示该梯度值出现的频率。\n"
        "梯度的分布反映了网络各层的学习状态和更新方向。"
    )
    plt.figtext(0.1, 0.02, description, wrap=True, fontsize=10)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def visualize_attention_maps(model, image_path, save_path=None):
    """可视化注意力图"""
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
    def hook_fn(name):
        def hook(module, input, output):
            feature_maps[name] = output.detach().cpu()
        return hook
    
    # 为卷积层注册钩子
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(hook_fn(name)))
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
    
    # 移除钩子
    for handle in handles:
        handle.remove()
    
    # 创建图形
    n_layers = len(feature_maps)
    fig = plt.figure(figsize=(15, 5))
    
    # 绘制每一层的注意力图
    for idx, (name, feature_map) in enumerate(feature_maps.items()):
        plt.subplot(1, n_layers, idx+1)
        # 计算注意力图（特征图的平均值）
        attention_map = feature_map.mean(dim=1).squeeze()
        plt.imshow(attention_map, cmap='hot')
        plt.title(f'Attention - {name}')
        plt.axis('off')
        plt.colorbar()
    
    plt.suptitle('Neural Network Attention Maps', y=0.95)
    
    # 添加说明文字
    description = (
        "该图展示了神经网络各卷积层的注意力图。颜色越亮表示该区域的特征越显著。\n"
        "注意力图反映了网络在识别过程中关注的重点区域。"
    )
    plt.figtext(0.1, 0.02, description, wrap=True, fontsize=10)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # 创建保存目录
    os.makedirs('visualizations', exist_ok=True)
    
    # 加载模型
    model = DigitRecognizer()
    model.load_state_dict(torch.load('models/best_model.pth'))
    model.eval()
    
    # 1. 可视化第一个卷积层的滤波器（只保存）
    plt.ioff()  # 关闭交互模式
    visualize_filters(model, 'conv1', 'visualizations/01_conv1_filters.png')
    plt.close()
    
    # 2. 可视化第二个卷积层的滤波器（只保存）
    visualize_filters(model, 'conv2', 'visualizations/02_conv2_filters.png')
    plt.close()
    
    # 3. 可视化权重分布（只保存）
    visualize_weights_distribution(model, 'visualizations/03_weights_distribution.png')
    plt.close()
    
    # 4. 可视化特征图和像素贡献（只保存）
    test_dir = '../test_images'
    test_images = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if test_images:
        image_path = os.path.join(test_dir, test_images[0])
        visualize_feature_maps(model, image_path, 'visualizations/04_feature_maps.png')
        plt.close()
        visualize_pixel_contribution(model, image_path, 'visualizations/05_pixel_contribution.png')
        plt.close()
        
        # 新增的可视化方法
        visualize_activation_distribution(model, image_path, 'visualizations/07_activation_distribution.png')
        plt.close()
        visualize_gradient_flow(model, image_path, 'visualizations/08_gradient_flow.png')
        plt.close()
        visualize_attention_maps(model, image_path, 'visualizations/09_attention_maps.png')
        plt.close()
    
    # 5. 网络结构可视化（保存并显示）
    plt.ion()  # 重新开启交互模式
    visualize_network_structure(model, input_size=(1, 1, 28, 28), 
                              save_path='visualizations/06_network_structure.png')
    plt.show()  # 确保最后一个图显示出来
    
    try:
        input("按回车键关闭...")  # 等待用户输入后再关闭
    finally:
        plt.close('all')  # 关闭所有图形窗口
        exit(0)  # 确保程序完全退出

if __name__ == '__main__':
    main() 