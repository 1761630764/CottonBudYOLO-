"""
YOLOv11 XGrad-CAM 热力图可视化工具

使用 XGrad-CAM 方法生成模型关注区域的热力图

层位置	         特征级别	适用场景
model.model.22	高级语义	目标整体关注 ⭐
model.model.21	中高级	    目标细节关注
model.model.15	中级	   多尺度分析
model.model.9	低级	   边缘纹理分析
使用示例:
    # 单张图片
    python xgradcam_visualizer.py 
        --weights E:/Code/CottonBudYOLOv2/runs/train/base/cotton-v11-n/weights/best.pt
        --source E:/Code/CottonBudYOLOv2/images_test/test_two/target.jpg 
        --target-layer 22 
        --conf 0.45 
        --device cuda:0 
        --project runs/xgradcam 
        --name base-target
        
    # 批量处理
    python xgradcam_visualizer.py \
        --weights runs/train/yolov11n_DN_ALL_cotton/weights/best.pt \
        --source images/ \
        --target-layer 22 \
        --conf 0.25
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.utils import LOGGER, colorstr


class XGradCAM:
    """XGrad-CAM 实现"""
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: YOLO 模型
            target_layer: 目标层（字符串路径，如 'model.model.22'）
        """
        self.model = model
        self.target_layer = self._get_layer(target_layer)
        self.gradients = None
        self.activations = None
        
        # 注册钩子
        self._register_hooks()
    
    def _get_layer(self, layer_path):
        """
        根据路径获取层
        
        支持的格式:
            - "22" -> model.model[22]
            - "model.22" -> model.model[22]
            - "22.conv" -> model.model[22].conv
        """
        # 从 DetectionModel 获取内部的 Sequential model
        if hasattr(self.model, 'model'):
            layer = self.model.model
        else:
            layer = self.model
        
        parts = layer_path.split('.')
        
        # 跳过开头的 'model' (如果有)
        if parts[0] == 'model':
            parts = parts[1:]
        
        for attr in parts:
            if attr.isdigit():
                # 数字索引
                layer = layer[int(attr)]
            else:
                # 属性访问
                layer = getattr(layer, attr)
        
        return layer
    
    def _register_hooks(self):
        """注册前向和反向钩子"""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        生成 XGrad-CAM 热力图
        
        Args:
            input_tensor: 输入张量 [1, 3, H, W]
            target_class: 目标类别索引（None表示使用最高置信度类别）
        
        Returns:
            cam: 热力图 [H, W]
        """
        # 前向传播 - 直接调用模型得到原始输出
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # output 是一个 tuple，包含多个尺度的预测
        # 我们需要从中提取置信度最高的预测来进行反向传播
        
        # 如果激活图为空，说明没有通过目标层，返回空热力图
        if self.activations is None:
            h, w = input_tensor.shape[2:]
            return np.zeros((h, w), dtype=np.float32)
        
        # 从输出中找到最大的置信度分数用于反向传播
        # YOLO 输出格式: (batch, num_anchors, 4+1+num_classes)
        max_score = None
        
        if isinstance(output, (list, tuple)):
            # 遍历所有输出层找最大置信度
            for out in output:
                if isinstance(out, torch.Tensor) and out.requires_grad:
                    # 提取置信度分数 (通常在第5个位置)
                    if out.dim() >= 3:
                        # 假设格式为 [batch, anchors, features]
                        # 置信度通常在 index 4
                        if out.shape[-1] > 4:
                            scores = out[..., 4]  # objectness score
                            current_max = scores.max()
                            if max_score is None or current_max > max_score:
                                max_score = current_max
        
        # 如果没有找到有效的分数，使用激活图的和作为目标
        if max_score is None:
            max_score = self.activations.sum()
        
        # 反向传播
        if max_score.requires_grad:
            max_score.backward(retain_graph=True)
        else:
            # 如果没有梯度，返回基于激活图的简单 CAM
            h, w = input_tensor.shape[2:]
            cam = self.activations.mean(dim=1).squeeze().cpu().numpy()
            cam = cv2.resize(cam, (w, h))
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            return cam
        
        # 计算 XGrad-CAM
        if self.gradients is None or self.activations is None:
            h, w = input_tensor.shape[2:]
            return np.zeros((h, w), dtype=np.float32)
        
        # XGrad-CAM: 使用梯度加权
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        
        # 加权求和
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [1, 1, H', W']
        
        # ReLU
        cam = F.relu(cam)
        
        # 归一化
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # 调整大小到输入尺寸
        h, w = input_tensor.shape[2:]
        cam = cv2.resize(cam, (w, h))
        
        return cam


class XGradCAMVisualizer:
    """XGrad-CAM 可视化器"""
    
    def __init__(self, weights, target_layer='22', device=''):
        """
        Args:
            weights: 模型权重路径
            target_layer: 目标层路径 (如 '22' 或 'model.22')
            device: 设备
        """
        LOGGER.info(f"加载模型: {weights}")
        self.yolo = YOLO(weights)
        self.device = device if device else ('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # 获取底层模型并移到正确的设备
        self.model = self.yolo.model
        self.model.to(self.device)
        self.model.eval()
        
        # 创建 XGrad-CAM
        LOGGER.info(f"目标层: {target_layer}")
        self.xgradcam = XGradCAM(self.model, target_layer)
        
        # 颜色映射
        self.colormap = cv2.COLORMAP_JET
    
    def preprocess_image(self, image_path, imgsz=640):
        """预处理图像"""
        # 读取图像
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        orig_img = img.copy()
        
        # 调整大小
        img = cv2.resize(img, (imgsz, imgsz))
        
        # 转换为 RGB 并归一化
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        
        # 转换为张量
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        return img_tensor, orig_img
    
    def apply_colormap(self, cam, img):
        """应用颜色映射到热力图"""
        # 调整热力图到原图大小
        h, w = img.shape[:2]
        cam_resized = cv2.resize(cam, (w, h))
        
        # 转换为 uint8
        cam_uint8 = (cam_resized * 255).astype(np.uint8)
        
        # 应用颜色映射
        heatmap = cv2.applyColorMap(cam_uint8, self.colormap)
        
        # 叠加到原图
        alpha = 0.5
        overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
        
        return overlay, heatmap
    
    def visualize(self, image_path, imgsz=640, conf=0.25, save_dir=None):
        """
        可视化单张图像
        
        Args:
            image_path: 图像路径
            imgsz: 输入尺寸
            conf: 置信度阈值（保留参数以兼容性，但不使用）
            save_dir: 保存目录
        
        Returns:
            overlay: 叠加图像
            heatmap: 纯热力图
            detections: 检测结果（None）
        """
        # 预处理
        img_tensor, orig_img = self.preprocess_image(image_path, imgsz)
        
        # 生成 CAM
        cam = self.xgradcam.generate_cam(img_tensor)
        
        # 应用颜色映射
        overlay, heatmap = self.apply_colormap(cam, orig_img)
        
        # 保存结果
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            stem = Path(image_path).stem
            cv2.imwrite(str(save_dir / f"{stem}_xgradcam.jpg"), overlay)
            cv2.imwrite(str(save_dir / f"{stem}_heatmap.jpg"), heatmap)
            cv2.imwrite(str(save_dir / f"{stem}_original.jpg"), orig_img)
            
            LOGGER.info(f"已保存: {save_dir / stem}_xgradcam.jpg")
        
        return overlay, heatmap, None


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLOv11 XGrad-CAM Visualizer')
    
    parser.add_argument('--weights', type=str, required=True,
                       help='模型权重路径')
    parser.add_argument('--source', type=str, required=True,
                       help='输入图像或目录')
    parser.add_argument('--target-layer', type=str, default='22',
                       help='目标层路径 (默认: 22 - 最后一个特征层，也可用 model.22 或 21, 20 等)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='输入图像尺寸')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='置信度阈值')
    parser.add_argument('--device', type=str, default='',
                       help='设备 (cuda:0 或 cpu)')
    parser.add_argument('--project', type=str, default='runs/xgradcam',
                       help='保存项目目录')
    parser.add_argument('--name', type=str, default='exp',
                       help='实验名称')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 检查权重文件
    if not Path(args.weights).exists():
        LOGGER.error(f"权重文件不存在: {args.weights}")
        return 1
    
    # 创建保存目录
    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    LOGGER.info(colorstr('blue', 'bold', '\n' + '='*70))
    LOGGER.info(colorstr('blue', 'bold', 'YOLOv11 XGrad-CAM 可视化'))
    LOGGER.info(colorstr('blue', 'bold', '='*70))
    LOGGER.info(f"权重: {args.weights}")
    LOGGER.info(f"输入: {args.source}")
    LOGGER.info(f"目标层: {args.target_layer}")
    LOGGER.info(f"保存目录: {save_dir}")
    
    # 创建可视化器
    visualizer = XGradCAMVisualizer(
        weights=args.weights,
        target_layer=args.target_layer,
        device=args.device
    )
    
    # 获取输入文件列表
    source_path = Path(args.source)
    if source_path.is_file():
        image_files = [source_path]
    elif source_path.is_dir():
        image_files = list(source_path.glob('*.jpg')) + \
                     list(source_path.glob('*.png')) + \
                     list(source_path.glob('*.jpeg'))
    else:
        LOGGER.error(f"无效的输入路径: {args.source}")
        return 1
    
    if len(image_files) == 0:
        LOGGER.error(f"未找到图像文件: {args.source}")
        return 1
    
    LOGGER.info(f"找到 {len(image_files)} 张图像")
    
    # 处理图像
    for img_path in tqdm(image_files, desc="处理进度"):
        try:
            visualizer.visualize(
                image_path=img_path,
                imgsz=args.imgsz,
                conf=args.conf,
                save_dir=save_dir
            )
        except Exception as e:
            LOGGER.error(f"处理失败 {img_path}: {e}")
            continue
    
    LOGGER.info(colorstr('green', 'bold', f'\n✓ 完成! 结果保存在: {save_dir}'))
    return 0


if __name__ == '__main__':
    sys.exit(main())

