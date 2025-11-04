"""
热力图生成器

用于生成目标检测的热力图可视化
"""

import cv2
import numpy as np


class HeatmapGenerator:
    """热力图生成器"""
    
    def __init__(self, shape, colormap=cv2.COLORMAP_JET, decay=0.99, alpha=0.5):
        """
        初始化热力图生成器
        
        Args:
            shape: 图像形状 (height, width, channels)
            colormap: OpenCV颜色映射
            decay: 热力图衰减率 (0-1)，用于视频
            alpha: 热力图透明度 (0-1)
        """
        self.heatmap = np.zeros(shape[:2], dtype=np.float32)
        self.colormap = colormap
        self.decay = decay
        self.alpha = alpha
        self.shape = shape
        
    def update(self, boxes, scores=None):
        """
        更新热力图
        
        Args:
            boxes: 检测框列表 [[x1, y1, x2, y2], ...]
            scores: 置信度列表（可选）
        """
        # 应用衰减
        self.heatmap *= self.decay
        
        # 为每个检测框添加热量
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            
            # 确保坐标在图像范围内
            x1 = max(0, min(x1, self.shape[1] - 1))
            y1 = max(0, min(y1, self.shape[0] - 1))
            x2 = max(0, min(x2, self.shape[1] - 1))
            y2 = max(0, min(y2, self.shape[0] - 1))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # 计算中心点和半径
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            radius = min(x2 - x1, y2 - y1) // 2
            
            if radius <= 0:
                continue
            
            # 创建高斯热量分布
            y_coords, x_coords = np.ogrid[y1:y2, x1:x2]
            dist_squared = (x_coords - cx) ** 2 + (y_coords - cy) ** 2
            radius_squared = radius ** 2
            
            # 高斯权重
            weight = np.exp(-dist_squared / (2 * radius_squared))
            
            # 如果有置信度，使用置信度加权
            if scores is not None:
                weight *= scores[i]
            
            # 更新热力图
            self.heatmap[y1:y2, x1:x2] += weight * 10
    
    def apply(self, image):
        """
        将热力图应用到图像上
        
        Args:
            image: 原始图像
            
        Returns:
            叠加热力图的图像
        """
        # 归一化热力图
        normalized = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        normalized = normalized.astype(np.uint8)
        
        # 应用颜色映射
        colored_heatmap = cv2.applyColorMap(normalized, self.colormap)
        
        # 与原图混合
        result = cv2.addWeighted(image, 1 - self.alpha, colored_heatmap, self.alpha, 0)
        
        return result
    
    def get_heatmap_only(self):
        """
        获取纯热力图（不叠加原图）
        
        Returns:
            热力图图像
        """
        normalized = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        normalized = normalized.astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(normalized, self.colormap)
        return colored_heatmap
    
    def reset(self):
        """重置热力图"""
        self.heatmap.fill(0)


# 颜色映射字典
COLORMAP_DICT = {
    'JET': cv2.COLORMAP_JET,
    'HOT': cv2.COLORMAP_HOT,
    'COOL': cv2.COLORMAP_COOL,
    'RAINBOW': cv2.COLORMAP_RAINBOW,
    'OCEAN': cv2.COLORMAP_OCEAN,
    'WINTER': cv2.COLORMAP_WINTER,
    'SPRING': cv2.COLORMAP_SPRING,
    'SUMMER': cv2.COLORMAP_SUMMER,
    'AUTUMN': cv2.COLORMAP_AUTUMN,
    'BONE': cv2.COLORMAP_BONE,
    'PINK': cv2.COLORMAP_PINK,
    'HSV': cv2.COLORMAP_HSV,
}


def get_colormap(name):
    """
    根据名称获取颜色映射
    
    Args:
        name: 颜色映射名称
        
    Returns:
        OpenCV颜色映射常量
    """
    return COLORMAP_DICT.get(name.upper(), cv2.COLORMAP_JET)

