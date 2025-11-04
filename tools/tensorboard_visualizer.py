"""
TensorBoard可视化工具 for YOLOv11-DN

功能:
1. 训练过程监控 (损失、精度、学习率等)
2. 领域自适应可视化 (域权重分布、域特征等)
3. 网络结构可视化
4. 特征图可视化
5. 模型参数分布
6. 梯度流分析

使用方法:
    python tools/tensorboard_visualizer.py --log_dir runs/train/yolo11-dn5
    
    # 然后在浏览器打开
    tensorboard --logdir runs/train
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from typing import Dict, List, Optional, Tuple
import cv2
from datetime import datetime

# 添加项目路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ultralytics import YOLO
from ultralytics.nn.modules import (
    DomainNormalization,
    DomainNormalizedConv,
    C3k2_DN,
    SPPF_MS,
    DAConvAtt,
    DAConvAttLite
)


class YOLOv11DNVisualizer:
    """YOLOv11-DN TensorBoard可视化器"""
    
    def __init__(
        self,
        log_dir: str = 'runs/tensorboard',
        model_path: Optional[str] = None,
        model_yaml: Optional[str] = None
    ):
        """
        初始化可视化器
        
        Args:
            log_dir: TensorBoard日志目录
            model_path: 已训练模型路径 (.pt)
            model_yaml: 模型配置文件 (.yaml)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # 加载模型
        self.model = None
        self.model_loaded = False
        if model_path:
            self.load_model(model_path)
        elif model_yaml:
            self.load_model_from_yaml(model_yaml)
        
        print(f"✓ TensorBoard日志目录: {self.log_dir}")
        print(f"✓ 启动TensorBoard: tensorboard --logdir {self.log_dir.parent}")
        
    def load_model(self, model_path: str):
        """加载已训练的模型"""
        try:
            self.model = YOLO(model_path)
            self.model_loaded = True
            print(f"✓ 模型加载成功: {model_path}")
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            self.model_loaded = False
    
    def load_model_from_yaml(self, yaml_path: str):
        """从YAML配置加载模型"""
        try:
            self.model = YOLO(yaml_path)
            self.model_loaded = True
            print(f"✓ 模型配置加载成功: {yaml_path}")
        except Exception as e:
            print(f"✗ 模型配置加载失败: {e}")
            self.model_loaded = False
    
    # ==================== 1. 训练过程监控 ====================
    
    def log_training_metrics(
        self,
        metrics: Dict[str, float],
        epoch: int,
        phase: str = 'train'
    ):
        """
        记录训练指标
        
        Args:
            metrics: 指标字典 {'loss': 0.5, 'mAP': 0.8, ...}
            epoch: 当前epoch
            phase: 'train' 或 'val'
        """
        for key, value in metrics.items():
            self.writer.add_scalar(f'{phase}/{key}', value, epoch)
    
    def log_losses(
        self,
        box_loss: float,
        cls_loss: float,
        dfl_loss: float,
        total_loss: float,
        epoch: int,
        phase: str = 'train'
    ):
        """记录各项损失"""
        losses = {
            'box_loss': box_loss,
            'cls_loss': cls_loss,
            'dfl_loss': dfl_loss,
            'total_loss': total_loss
        }
        
        for name, value in losses.items():
            self.writer.add_scalar(f'{phase}/losses/{name}', value, epoch)
        
        # 损失占比
        total = box_loss + cls_loss + dfl_loss
        if total > 0:
            ratios = {
                'box_ratio': box_loss / total,
                'cls_ratio': cls_loss / total,
                'dfl_ratio': dfl_loss / total
            }
            for name, value in ratios.items():
                self.writer.add_scalar(f'{phase}/loss_ratios/{name}', value, epoch)
    
    def log_learning_rate(self, lr: float, epoch: int):
        """记录学习率"""
        self.writer.add_scalar('train/learning_rate', lr, epoch)
    
    def log_metrics_summary(
        self,
        mAP50: float,
        mAP50_95: float,
        precision: float,
        recall: float,
        epoch: int
    ):
        """记录评估指标摘要"""
        metrics = {
            'mAP50': mAP50,
            'mAP50-95': mAP50_95,
            'precision': precision,
            'recall': recall
        }
        
        for name, value in metrics.items():
            self.writer.add_scalar(f'val/metrics/{name}', value, epoch)
        
        # F1分数
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
            self.writer.add_scalar('val/metrics/f1_score', f1, epoch)
    
    # ==================== 2. 领域自适应可视化 ====================
    
    def visualize_domain_weights(
        self,
        domain_weights: torch.Tensor,
        layer_name: str,
        step: int
    ):
        """
        可视化域权重分布
        
        Args:
            domain_weights: (B, D) 域权重张量
            layer_name: 层名称
            step: 当前步数
        """
        if domain_weights is None:
            return
        
        weights_np = domain_weights.detach().cpu().numpy()
        batch_size, num_domains = weights_np.shape
        
        # 1. 直方图 - 每个域的权重分布
        fig, axes = plt.subplots(1, num_domains, figsize=(4*num_domains, 3))
        if num_domains == 1:
            axes = [axes]
        
        for d in range(num_domains):
            axes[d].hist(weights_np[:, d], bins=20, alpha=0.7, color=f'C{d}')
            axes[d].set_xlabel(f'域{d+1}权重')
            axes[d].set_ylabel('频数')
            axes[d].set_title(f'域{d+1}权重分布')
            axes[d].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.writer.add_figure(f'domain_weights/{layer_name}/histogram', fig, step)
        plt.close(fig)
        
        # 2. 热力图 - 样本×域
        fig, ax = plt.subplots(figsize=(8, max(4, batch_size*0.3)))
        im = ax.imshow(weights_np, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        ax.set_xlabel('域索引')
        ax.set_ylabel('样本索引')
        ax.set_title(f'{layer_name} - 域权重热力图')
        plt.colorbar(im, ax=ax, label='权重值')
        
        self.writer.add_figure(f'domain_weights/{layer_name}/heatmap', fig, step)
        plt.close(fig)
        
        # 3. 标量 - 平均权重
        mean_weights = weights_np.mean(axis=0)
        for d in range(num_domains):
            self.writer.add_scalar(
                f'domain_weights/{layer_name}/domain_{d+1}_mean',
                mean_weights[d],
                step
            )
        
        # 4. 标量 - 权重熵 (衡量域选择的确定性)
        epsilon = 1e-10
        entropy = -np.sum(weights_np * np.log(weights_np + epsilon), axis=1)
        self.writer.add_scalar(
            f'domain_weights/{layer_name}/entropy_mean',
            entropy.mean(),
            step
        )
    
    def collect_domain_weights_from_model(self, input_tensor: torch.Tensor, step: int):
        """从模型中收集所有DN模块的域权重"""
        if not self.model_loaded:
            return
        
        hooks = []
        domain_weights_dict = {}
        
        def create_hook(name):
            def hook(module, input, output):
                if hasattr(module, 'domain_predictor'):
                    # 获取输入特征
                    x = input[0]
                    # 计算域权重
                    weights = module.domain_predictor(x)
                    domain_weights_dict[name] = weights
            return hook
        
        # 注册hooks
        for name, module in self.model.model.named_modules():
            if isinstance(module, (DomainNormalization, C3k2_DN, DomainNormalizedConv)):
                hook = module.register_forward_hook(create_hook(name))
                hooks.append(hook)
        
        # 前向传播
        with torch.no_grad():
            self.model.model(input_tensor)
        
        # 移除hooks
        for hook in hooks:
            hook.remove()
        
        # 可视化收集到的域权重
        for name, weights in domain_weights_dict.items():
            self.visualize_domain_weights(weights, name, step)
    
    def visualize_domain_statistics(
        self,
        domain_stats: Dict[str, np.ndarray],
        epoch: int
    ):
        """
        可视化域统计量 (μ, σ)
        
        Args:
            domain_stats: {'layer_name': {'mu': [μ₁, μ₂, ...], 'sigma': [σ₁, σ₂, ...]}}
            epoch: 当前epoch
        """
        for layer_name, stats in domain_stats.items():
            mu_values = stats.get('mu', [])
            sigma_values = stats.get('sigma', [])
            
            if len(mu_values) > 0:
                # 均值分布
                fig, ax = plt.subplots(figsize=(8, 4))
                domains = list(range(1, len(mu_values)+1))
                ax.bar(domains, mu_values, alpha=0.7, label='μ (均值)')
                ax.set_xlabel('域索引')
                ax.set_ylabel('μ值')
                ax.set_title(f'{layer_name} - 各域均值')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                self.writer.add_figure(f'domain_stats/{layer_name}/mu', fig, epoch)
                plt.close(fig)
            
            if len(sigma_values) > 0:
                # 方差分布
                fig, ax = plt.subplots(figsize=(8, 4))
                domains = list(range(1, len(sigma_values)+1))
                ax.bar(domains, sigma_values, alpha=0.7, color='orange', label='σ² (方差)')
                ax.set_xlabel('域索引')
                ax.set_ylabel('σ²值')
                ax.set_title(f'{layer_name} - 各域方差')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                self.writer.add_figure(f'domain_stats/{layer_name}/sigma', fig, epoch)
                plt.close(fig)
    
    # ==================== 3. 网络结构可视化 ====================
    
    def visualize_model_graph(self, input_size: Tuple[int, int, int, int] = (1, 3, 640, 640)):
        """可视化模型计算图"""
        if not self.model_loaded:
            print("✗ 模型未加载，无法可视化计算图")
            return
        
        try:
            dummy_input = torch.randn(input_size)
            self.writer.add_graph(self.model.model, dummy_input)
            print("✓ 模型计算图已添加到TensorBoard")
        except Exception as e:
            print(f"✗ 模型计算图可视化失败: {e}")
    
    def log_model_architecture(self):
        """记录模型架构信息"""
        if not self.model_loaded:
            return
        
        # 统计各类模块数量
        module_counts = {}
        total_params = 0
        
        for name, module in self.model.model.named_modules():
            module_type = type(module).__name__
            module_counts[module_type] = module_counts.get(module_type, 0) + 1
            
            # 计算参数量
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if params > 0:
                total_params += params
        
        # 创建文本摘要
        text = "# YOLOv11-DN 模型架构\n\n"
        text += f"## 总参数量: {total_params:,}\n\n"
        text += "## 模块统计:\n"
        
        # 按数量排序
        sorted_modules = sorted(module_counts.items(), key=lambda x: x[1], reverse=True)
        for module_type, count in sorted_modules[:20]:  # 前20个
            text += f"- {module_type}: {count}\n"
        
        # 关键模块高亮
        text += "\n## 领域自适应模块:\n"
        dn_modules = [k for k in module_counts.keys() if 'Domain' in k or 'DN' in k]
        for module_type in dn_modules:
            text += f"- **{module_type}**: {module_counts[module_type]}\n"
        
        self.writer.add_text('model/architecture', text, 0)
        print("✓ 模型架构信息已添加到TensorBoard")
    
    # ==================== 4. 特征图可视化 ====================
    
    def visualize_feature_maps(
        self,
        feature_maps: torch.Tensor,
        layer_name: str,
        step: int,
        max_channels: int = 16
    ):
        """
        可视化特征图
        
        Args:
            feature_maps: (B, C, H, W) 特征张量
            layer_name: 层名称
            step: 当前步数
            max_channels: 最多显示的通道数
        """
        if feature_maps is None or feature_maps.dim() != 4:
            return
        
        # 取第一个batch
        feat = feature_maps[0].detach().cpu()  # (C, H, W)
        num_channels = min(feat.size(0), max_channels)
        
        # 创建网格
        rows = int(np.ceil(np.sqrt(num_channels)))
        cols = int(np.ceil(num_channels / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
        axes = axes.flatten() if num_channels > 1 else [axes]
        
        for i in range(num_channels):
            channel_feat = feat[i].numpy()
            
            # 归一化到 [0, 1]
            channel_feat = (channel_feat - channel_feat.min()) / (channel_feat.max() - channel_feat.min() + 1e-8)
            
            axes[i].imshow(channel_feat, cmap='viridis')
            axes[i].set_title(f'Ch{i}')
            axes[i].axis('off')
        
        # 隐藏多余的子图
        for i in range(num_channels, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        self.writer.add_figure(f'feature_maps/{layer_name}', fig, step)
        plt.close(fig)
    
    def collect_feature_maps(self, input_tensor: torch.Tensor, step: int):
        """从模型中收集关键层的特征图"""
        if not self.model_loaded:
            return
        
        features_dict = {}
        hooks = []
        
        def create_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor) and output.dim() == 4:
                    features_dict[name] = output
            return hook
        
        # 选择关键层进行可视化
        target_layers = [
            'model.model.6',  # C3k2_DN (P4)
            'model.model.8',  # C3k2_DN (P5)
            'model.model.9',  # SPPF_MS
            'model.model.16', # P3_out
            'model.model.19', # P4_out
            'model.model.22', # P5_out
        ]
        
        # 注册hooks
        for name, module in self.model.model.named_modules():
            full_name = f'model.{name}'
            if full_name in target_layers:
                hook = module.register_forward_hook(create_hook(name))
                hooks.append(hook)
        
        # 前向传播
        with torch.no_grad():
            self.model.model(input_tensor)
        
        # 移除hooks
        for hook in hooks:
            hook.remove()
        
        # 可视化特征图
        for name, feat in features_dict.items():
            self.visualize_feature_maps(feat, name, step)
    
    # ==================== 5. 参数分布可视化 ====================
    
    def visualize_parameter_distributions(self, epoch: int):
        """可视化模型参数分布"""
        if not self.model_loaded:
            return
        
        for name, param in self.model.model.named_parameters():
            if param.requires_grad:
                # 参数值直方图
                self.writer.add_histogram(f'parameters/{name}', param.data, epoch)
                
                # 参数统计
                self.writer.add_scalar(f'param_stats/{name}/mean', param.data.mean(), epoch)
                self.writer.add_scalar(f'param_stats/{name}/std', param.data.std(), epoch)
                self.writer.add_scalar(f'param_stats/{name}/abs_max', param.data.abs().max(), epoch)
    
    def visualize_gradient_distributions(self, epoch: int):
        """可视化梯度分布"""
        if not self.model_loaded:
            return
        
        for name, param in self.model.model.named_parameters():
            if param.grad is not None:
                # 梯度直方图
                self.writer.add_histogram(f'gradients/{name}', param.grad, epoch)
                
                # 梯度统计
                grad_norm = param.grad.norm()
                self.writer.add_scalar(f'grad_norms/{name}', grad_norm, epoch)
    
    # ==================== 6. 图像可视化 ====================
    
    def log_images(
        self,
        images: torch.Tensor,
        predictions: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        step: int = 0,
        tag: str = 'images'
    ):
        """
        记录图像及预测结果
        
        Args:
            images: (B, 3, H, W) 图像张量
            predictions: 预测结果
            targets: 真实标签
            step: 当前步数
            tag: 标签
        """
        # 取前8张图像
        num_images = min(images.size(0), 8)
        img_grid = torch.clamp(images[:num_images], 0, 1)
        
        self.writer.add_images(tag, img_grid, step)
    
    def log_detection_results(
        self,
        image_paths: List[str],
        results,
        step: int,
        max_images: int = 8
    ):
        """记录检测结果可视化"""
        for i, (img_path, result) in enumerate(zip(image_paths[:max_images], results[:max_images])):
            if result.plot() is not None:
                # 获取绘制后的图像
                plotted = result.plot()
                # BGR to RGB
                plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
                # HWC to CHW
                plotted = plotted.transpose(2, 0, 1)
                
                self.writer.add_image(f'detections/image_{i}', plotted, step)
    
    # ==================== 7. 混淆矩阵 ====================
    
    def log_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        class_names: List[str],
        epoch: int,
        normalize: bool = True
    ):
        """记录混淆矩阵"""
        if normalize:
            confusion_matrix = confusion_matrix.astype('float') / (confusion_matrix.sum(axis=1)[:, np.newaxis] + 1e-6)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(confusion_matrix, cmap='Blues')
        
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
        
        plt.colorbar(im, ax=ax)
        ax.set_xlabel('预测类别')
        ax.set_ylabel('真实类别')
        ax.set_title('混淆矩阵')
        
        plt.tight_layout()
        self.writer.add_figure('confusion_matrix', fig, epoch)
        plt.close(fig)
    
    # ==================== 8. PR曲线 ====================
    
    def log_pr_curve(
        self,
        precision: np.ndarray,
        recall: np.ndarray,
        class_name: str,
        epoch: int
    ):
        """记录PR曲线"""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, 'b-', linewidth=2)
        ax.fill_between(recall, precision, alpha=0.2)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'PR Curve - {class_name}')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # 计算AP
        ap = np.trapz(precision, recall)
        ax.text(0.5, 0.5, f'AP = {ap:.3f}', transform=ax.transAxes,
                fontsize=14, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        self.writer.add_figure(f'pr_curves/{class_name}', fig, epoch)
        plt.close(fig)
    
    # ==================== 工具函数 ====================
    
    def close(self):
        """关闭writer"""
        self.writer.close()
        print("✓ TensorBoard writer已关闭")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLOv11-DN TensorBoard可视化工具')
    
    parser.add_argument('--log_dir', type=str, default='runs/tensorboard',
                       help='TensorBoard日志目录')
    parser.add_argument('--model', type=str, default=None,
                       help='模型路径 (.pt) 或配置文件 (.yaml)')
    parser.add_argument('--data', type=str, default=None,
                       help='测试数据路径')
    parser.add_argument('--visualize-graph', action='store_true',
                       help='可视化模型计算图')
    parser.add_argument('--visualize-weights', action='store_true',
                       help='可视化域权重分布')
    parser.add_argument('--visualize-features', action='store_true',
                       help='可视化特征图')
    parser.add_argument('--device', type=str, default='0',
                       help='设备 (0, 1, cpu)')
    
    return parser.parse_args()


def main():
    """主函数 - 演示使用"""
    args = parse_args()
    
    print("=" * 60)
    print("YOLOv11-DN TensorBoard可视化工具")
    print("=" * 60)
    
    # 创建可视化器
    visualizer = YOLOv11DNVisualizer(
        log_dir=args.log_dir,
        model_path=args.model if args.model and args.model.endswith('.pt') else None,
        model_yaml=args.model if args.model and args.model.endswith('.yaml') else None
    )
    
    try:
        # 可视化模型架构
        if args.visualize_graph and visualizer.model_loaded:
            print("\n[1/3] 可视化模型计算图...")
            visualizer.visualize_model_graph()
            visualizer.log_model_architecture()
        
        # 可视化域权重和特征图
        if (args.visualize_weights or args.visualize_features) and visualizer.model_loaded:
            print("\n[2/3] 生成测试数据...")
            dummy_input = torch.randn(2, 3, 640, 640)
            
            if args.visualize_weights:
                print("   - 收集域权重...")
                visualizer.collect_domain_weights_from_model(dummy_input, step=0)
            
            if args.visualize_features:
                print("   - 收集特征图...")
                visualizer.collect_feature_maps(dummy_input, step=0)
        
        # 演示训练指标记录
        print("\n[3/3] 演示训练指标记录...")
        for epoch in range(3):
            # 模拟训练指标
            visualizer.log_losses(
                box_loss=2.5 - epoch*0.5,
                cls_loss=1.5 - epoch*0.3,
                dfl_loss=1.0 - epoch*0.2,
                total_loss=5.0 - epoch*1.0,
                epoch=epoch,
                phase='train'
            )
            
            visualizer.log_metrics_summary(
                mAP50=0.6 + epoch*0.1,
                mAP50_95=0.4 + epoch*0.08,
                precision=0.7 + epoch*0.05,
                recall=0.65 + epoch*0.06,
                epoch=epoch
            )
            
            visualizer.log_learning_rate(0.01 * (0.9 ** epoch), epoch)
        
        print("\n" + "=" * 60)
        print("✓ 可视化完成！")
        print(f"✓ 日志目录: {visualizer.log_dir}")
        print(f"\n启动TensorBoard查看结果:")
        print(f"  tensorboard --logdir {visualizer.log_dir.parent}")
        print(f"  然后在浏览器打开: http://localhost:6006")
        print("=" * 60)
        
    finally:
        visualizer.close()


if __name__ == '__main__':
    main()

