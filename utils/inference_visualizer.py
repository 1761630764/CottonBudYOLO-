"""
推理可视化器

提供推理和可视化的核心功能
"""

import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import Annotator, colors

from utils.heatmap_generator import HeatmapGenerator, get_colormap


class InferenceVisualizer:
    """推理可视化器"""
    
    def __init__(self, model_path, conf_thres=0.25, iou_thres=0.45, device='', 
                 use_heatmap=False, heatmap_colormap='JET', 
                 heatmap_decay=0.99, heatmap_alpha=0.5):
        """
        初始化推理可视化器
        
        Args:
            model_path: 模型权重路径
            conf_thres: 置信度阈值
            iou_thres: NMS IOU阈值
            device: 设备 (cuda:0 或 cpu)
            use_heatmap: 是否使用热力图
            heatmap_colormap: 热力图颜色映射
            heatmap_decay: 热力图衰减率
            heatmap_alpha: 热力图透明度
        """
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.use_heatmap = use_heatmap
        self.heatmap_colormap = get_colormap(heatmap_colormap)
        self.heatmap_decay = heatmap_decay
        self.heatmap_alpha = heatmap_alpha
        self.heatmap_generator = None
        
        # 统计信息
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'class_counts': defaultdict(int),
            'inference_times': [],
        }
        
        LOGGER.info(f"✅ 模型加载成功: {model_path}")
        LOGGER.info(f"   设备: {self.model.device}")
        LOGGER.info(f"   类别数: {len(self.model.names)}")
        LOGGER.info(f"   类别: {self.model.names}")
    
    def annotate_image(self, img, results, line_thickness=3, hide_labels=False, hide_conf=False):
        """
        在图像上标注检测结果
        
        Args:
            img: 输入图像
            results: YOLO推理结果
            line_thickness: 边界框线宽
            hide_labels: 是否隐藏标签
            hide_conf: 是否隐藏置信度
            
        Returns:
            标注后的图像
        """
        annotator = Annotator(img, line_width=line_thickness)
        
        boxes = results.boxes
        for box in boxes:
            # 获取坐标、置信度、类别
            xyxy = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            
            # 构建标签
            if not hide_labels:
                label = f"{self.model.names[cls]}"
                if not hide_conf:
                    label += f" {conf:.2f}"
            else:
                label = None
            
            # 绘制边界框
            annotator.box_label(xyxy, label, color=colors(cls, True))
        
        return annotator.result()
    
    def apply_heatmap(self, img, results):
        """
        应用热力图
        
        Args:
            img: 输入图像
            results: YOLO推理结果
            
        Returns:
            叠加热力图的图像
        """
        if not self.use_heatmap:
            return img
        
        # 初始化热力图生成器
        if self.heatmap_generator is None:
            self.heatmap_generator = HeatmapGenerator(
                img.shape, 
                self.heatmap_colormap,
                self.heatmap_decay, 
                self.heatmap_alpha
            )
        
        # 更新热力图
        if len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            self.heatmap_generator.update(boxes, scores)
        
        # 应用热力图
        return self.heatmap_generator.apply(img)
    
    def predict_image(self, image_path):
        """
        对单张图像进行推理
        
        Args:
            image_path: 图像路径
            
        Returns:
            results: 推理结果
            annotated_img: 标注后的图像
            inference_time: 推理时间
        """
        # 推理
        t0 = time.time()
        results = self.model.predict(
            image_path, 
            conf=self.conf_thres, 
            iou=self.iou_thres,
            device=self.device,
            verbose=False
        )[0]
        t1 = time.time()
        
        # 读取图像
        img = cv2.imread(str(image_path))
        
        # 标注
        annotated_img = self.annotate_image(img.copy(), results)
        
        # 应用热力图
        annotated_img = self.apply_heatmap(annotated_img, results)
        
        # 更新统计
        self._update_stats(results, t1 - t0)
        
        return results, annotated_img, t1 - t0
    
    def save_txt(self, results, txt_path, img_shape, save_conf=False):
        """
        保存YOLO格式的txt标注
        
        Args:
            results: 推理结果
            txt_path: 保存路径
            img_shape: 图像形状
            save_conf: 是否保存置信度
        """
        h, w = img_shape[:2]
        
        with open(txt_path, 'w') as f:
            boxes = results.boxes
            for box in boxes:
                cls = int(box.cls[0].cpu().numpy())
                xyxy = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                
                # 转换为YOLO格式 (归一化的中心坐标和宽高)
                x_center = ((xyxy[0] + xyxy[2]) / 2) / w
                y_center = ((xyxy[1] + xyxy[3]) / 2) / h
                width = (xyxy[2] - xyxy[0]) / w
                height = (xyxy[3] - xyxy[1]) / h
                
                # 写入文件
                line = f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                if save_conf:
                    line += f" {conf:.6f}"
                f.write(line + '\n')
    
    def _update_stats(self, results, inference_time):
        """更新统计信息"""
        self.stats['total_frames'] += 1
        self.stats['total_detections'] += len(results.boxes)
        self.stats['inference_times'].append(inference_time)
        
        # 统计每个类别的数量
        for box in results.boxes:
            cls = int(box.cls[0].cpu().numpy())
            cls_name = self.model.names[cls]
            self.stats['class_counts'][cls_name] += 1
    
    def print_stats(self):
        """打印统计信息"""
        if self.stats['total_frames'] == 0:
            return
        
        avg_time = np.mean(self.stats['inference_times'])
        avg_fps = 1 / avg_time if avg_time > 0 else 0
        
        print("\n" + "="*70)
        print("推理统计")
        print("="*70)
        print(f"总帧数: {self.stats['total_frames']}")
        print(f"总检测数: {self.stats['total_detections']}")
        print(f"平均检测数/帧: {self.stats['total_detections']/self.stats['total_frames']:.2f}")
        print(f"平均推理时间: {avg_time*1000:.2f}ms")
        print(f"平均FPS: {avg_fps:.2f}")
        
        if self.stats['class_counts']:
            print(f"\n各类别检测数:")
            for cls_name, count in sorted(self.stats['class_counts'].items(), 
                                          key=lambda x: x[1], reverse=True):
                print(f"  {cls_name}: {count}")
        print("="*70 + "\n")
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_frames': 0,
            'total_detections': 0,
            'class_counts': defaultdict(int),
            'inference_times': [],
        }
        
        if self.heatmap_generator is not None:
            self.heatmap_generator.reset()

