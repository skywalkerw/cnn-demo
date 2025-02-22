import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import DigitRecognizer
from typing import Tuple
import os
import argparse

# 设置matplotlib后端和窗口参数
plt.rcParams['figure.max_open_warning'] = 0
plt.rcParams['figure.autolayout'] = False  # 禁用自动布局
plt.rcParams['figure.figsize'] = [8, 20]  # 调整为纵向布局的大小
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 9  # 略微增加默认字体大小
plt.rcParams['axes.titlesize'] = 10  # 略微增加标题字体大小

def predict_digit(image_path: str, model_path: str, ax_row=None, show=False) -> Tuple[int, float]:
    """预测单个图片的数字
    
    Args:
        image_path: 图片路径
        model_path: 模型路径
        ax_row: matplotlib子图对象列表，用于可视化
        show: 是否显示处理过程
    
    Returns:
        (预测的数字, 置信度)
    """
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DigitRecognizer().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
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
    
    # 加载并处理图像
    image = Image.open(image_path).convert('L')
    image_tensor = transform(image).unsqueeze(0).to(device)
    processed_image = image_tensor[0, 0].cpu().numpy()
    
    # 预测
    with torch.no_grad():
        output = model(image_tensor)
        pred = output.argmax(dim=1, keepdim=True)
        prob = F.softmax(output, dim=1).max().item()
        probs = F.softmax(output, dim=1)[0].cpu().numpy()
    
    # 显示图像和结果
    if ax_row:
        ax_row[0].imshow(image, cmap='gray')
        ax_row[0].set_title('Original')
        ax_row[0].axis('off')
        
        ax_row[1].imshow(processed_image, cmap='gray')
        ax_row[1].set_title('Processed (28x28)')
        ax_row[1].axis('off')
        
        # 调整概率分布图的显示
        ax_row[2].bar(range(10), probs)
        ax_row[2].set_title('Probability Distribution')
        ax_row[2].set_xlabel('Digit', fontsize=8)
        ax_row[2].set_ylabel('Probability', fontsize=8)
        ax_row[2].tick_params(axis='both', which='major', labelsize=7)
        
        # 合并预测结果和Top 3预测的显示
        top3_values, top3_indices = torch.topk(output[0], 3)
        top3_probs = F.softmax(top3_values, dim=0)
        
        text = f"Predicted: {pred.item()}\nConfidence: {prob:.2%}\n\nTop 3 Predictions:\n"
        for i, (idx, p) in enumerate(zip(top3_indices, top3_probs)):
            text += f"{idx.item()}: {p.item():.2%}\n"
        
        ax_row[3].text(0.5, 0.5, text,
                       horizontalalignment='center',
                       verticalalignment='center',
                       fontsize=9)
        ax_row[3].axis('off')
    
    # 如果需要显示处理过程
    if show and ax_row:
        plt.show()
    
    return pred.item(), prob

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='MNIST手写数字识别')
    parser.add_argument('image_path', nargs='?', help='要预测的图片路径')
    parser.add_argument('--show', action='store_true', help='显示处理过程')
    parser.add_argument('--save', help='保存结果到指定路径')
    args = parser.parse_args()

    # 设置路径
    model_path = 'models/best_model.pth'
    
    if args.image_path:
        # 预测单张图片
        try:
            fig, axes = plt.subplots(1, 4, figsize=(12, 3)) if args.show else (None, None)
            digit, confidence = predict_digit(args.image_path, model_path, 
                                           axes if args.show else None, 
                                           show=args.show)
            print(f"预测结果: {digit}")
            print(f"置信度: {confidence:.2%}")
            
            if args.save:
                plt.savefig(args.save)
                print(f"结果已保存到: {args.save}")
                
        except Exception as e:
            print(f"预测失败: {e}")
            
    else:
        # 预测test_images目录下所有图片
        test_dir = '../test_images'
        image_files = [f for f in os.listdir(test_dir) 
                      if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print("测试目录中没有找到图片文件！")
            return
            
        # 计算图像大小
        n_images = len(image_files)
        
        # 设置画布大小参数
        max_height_per_row = 1.8  # 减小每行高度
        total_height = max_height_per_row * n_images  # 总高度根据图片数量计算
        total_width = 12  # 减小固定宽度
        
        # 创建图形
        fig = plt.figure(figsize=(total_width, total_height))
        
        # 调整图形边距
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.02, top=0.98, wspace=0.3, hspace=0.4)
        
        # 创建网格布局，单列多行
        gs = plt.GridSpec(n_images, 4, figure=fig)
        
        # 处理每张图片
        for idx, image_file in enumerate(sorted(image_files)):
            # 创建当前行的axes
            ax_row = []
            for i in range(4):
                ax = fig.add_subplot(gs[idx, i])
                ax_row.append(ax)
            
            image_path = os.path.join(test_dir, image_file)
            print(f"\n处理图片 {image_file}:")
            
            try:
                digit, confidence = predict_digit(image_path, model_path, ax_row)
                print(f"预测结果: {digit}")
                print(f"置信度: {confidence:.2%}")
                
                # 如果置信度超过90%，重命名文件
                if confidence > 0.9:
                    # 获取文件扩展名
                    file_ext = os.path.splitext(image_file)[1]
                    # 生成新文件名
                    new_filename = f"digit_{digit}{file_ext}"
                    new_path = os.path.join(test_dir, new_filename)
                    
                    # 如果新文件名已存在，添加序号
                    counter = 1
                    while os.path.exists(new_path):
                        new_filename = f"digit_{digit}_{counter}{file_ext}"
                        new_path = os.path.join(test_dir, new_filename)
                        counter += 1
                    
                    # 重命名文件
                    os.rename(image_path, new_path)
                    print(f"文件已重命名为: {new_filename}")
                    
            except Exception as e:
                print(f"预测失败: {e}")
                for ax in ax_row:
                    ax.text(0.5, 0.5, f"处理失败:\n{str(e)}",
                           horizontalalignment='center',
                           verticalalignment='center')
                    ax.axis('off')
        
        # 调整图像以适应屏幕
        mng = plt.get_current_fig_manager()
        try:
            # 尝试使用Qt后端
            screen_width = 1920  # 假设屏幕宽度
            screen_height = 1080  # 假设屏幕高度
            window_width = min(screen_width - 100, int(screen_width * 0.9))
            window_height = min(screen_height - 100, int(screen_height * 0.9))
            mng.window.setGeometry(50, 50, window_width, window_height)
            mng.window.showMaximized()
        except:
            try:
                # 尝试使用Tk后端
                mng.window.state('zoomed')
            except:
                try:
                    mng.resize(*mng.window.maxsize())
                except:
                    pass
        
        plt.show()

if __name__ == '__main__':
    main() 