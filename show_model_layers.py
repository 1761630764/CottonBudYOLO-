"""
查看 YOLO 模型的层结构

使用方法:
    python show_model_layers.py --weights runs/train/yolov11n_DN_ALL_cotton/weights/best.pt

    # 基本使用
python show_model_layers.py --weights E:/Code/CottonBudYOLOv2/runs/train/yolov11n_DN_ALL_cotton/weights/best.pt

# 显示详细信息
python show_model_layers.py --weights E:/Code/CottonBudYOLOv2/runs/train/yolov11n_DN_ALL_cotton/weights/best.pt --details
"""

import argparse
from pathlib import Path
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER, colorstr


def print_model_structure(model, prefix='model', max_depth=3, current_depth=0):
    """
    递归打印模型结构
    
    Args:
        model: 模型或模块
        prefix: 当前路径前缀
        max_depth: 最大递归深度
        current_depth: 当前深度
    """
    if current_depth >= max_depth:
        return
    
    # 获取所有子模块
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}"
        module_type = module.__class__.__name__
        
        # 打印当前层信息
        indent = "  " * current_depth
        
        # 检查是否是卷积层（适合作为 CAM 目标层）
        is_conv = 'Conv' in module_type or 'C3' in module_type or 'C2' in module_type or 'SPPF' in module_type
        marker = " ⭐ [推荐用于CAM]" if is_conv else ""
        
        print(f"{indent}├─ {colorstr('cyan', full_name)}: {colorstr('yellow', module_type)}{colorstr('green', marker)}")
        
        # 如果有子模块，递归打印
        if len(list(module.children())) > 0:
            print_model_structure(module, full_name, max_depth, current_depth + 1)


def analyze_model_layers(weights_path, show_details=False):
    """
    分析模型层结构并给出 CAM 推荐
    
    Args:
        weights_path: 权重文件路径
        show_details: 是否显示详细信息
    """
    LOGGER.info(colorstr('blue', 'bold', '\n' + '='*80))
    LOGGER.info(colorstr('blue', 'bold', 'YOLOv11 模型层结构分析'))
    LOGGER.info(colorstr('blue', 'bold', '='*80))
    
    # 加载模型
    LOGGER.info(f"\n加载模型: {weights_path}")
    yolo = YOLO(weights_path)
    model = yolo.model
    
    # 打印模型摘要
    LOGGER.info(f"\n{colorstr('yellow', 'bold', '模型摘要:')}")
    LOGGER.info(f"模型类型: {model.__class__.__name__}")
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info(f"总参数: {total_params:,}")
    LOGGER.info(f"可训练参数: {trainable_params:,}")
    
    # 打印层结构
    LOGGER.info(f"\n{colorstr('yellow', 'bold', '模型层结构:')}")
    print_model_structure(model, prefix='model', max_depth=3)
    
    # 推荐的 CAM 目标层
    LOGGER.info(f"\n{colorstr('green', 'bold', '='*80)}")
    LOGGER.info(colorstr('green', 'bold', '推荐的 XGrad-CAM 目标层:'))
    LOGGER.info(colorstr('green', 'bold', '='*80))
    
    recommendations = [
        ("model.model.22", "最后一个检测头之前的特征层（最常用）"),
        ("model.model.21", "倒数第二个特征层"),
        ("model.model.20", "倒数第三个特征层"),
        ("model.model.15", "中间特征层（如果存在）"),
        ("model.model.9", "早期特征层（如果存在）"),
    ]
    
    LOGGER.info("\n推荐使用顺序（从高到低）:\n")
    for i, (layer, desc) in enumerate(recommendations, 1):
        try:
            # 尝试访问该层
            layer_obj = model
            for attr in layer.split('.'):
                layer_obj = getattr(layer_obj, attr)
            
            LOGGER.info(f"{i}. {colorstr('cyan', layer)}")
            LOGGER.info(f"   描述: {desc}")
            LOGGER.info(f"   类型: {layer_obj.__class__.__name__}")
            LOGGER.info("")
        except AttributeError:
            LOGGER.info(f"{i}. {colorstr('red', layer)} (不存在)")
            LOGGER.info(f"   描述: {desc}")
            LOGGER.info("")
    
    # 使用示例
    LOGGER.info(colorstr('yellow', 'bold', '使用示例:'))
    LOGGER.info(f"""
python xgradcam_visualizer.py \\
    --weights {weights_path} \\
    --source your_image.jpg \\
    --target-layer model.model.22 \\
    --conf 0.25
    """)
    
    # 详细信息
    if show_details:
        LOGGER.info(f"\n{colorstr('yellow', 'bold', '详细层信息:')}")
        for name, module in model.named_modules():
            if name:  # 跳过根模块
                print(f"{name}: {module.__class__.__name__}")
    
    LOGGER.info(colorstr('green', 'bold', '\n✓ 分析完成!'))


def main():
    parser = argparse.ArgumentParser(description='查看 YOLO 模型层结构')
    parser.add_argument('--weights', type=str, required=True,
                       help='模型权重路径')
    parser.add_argument('--details', action='store_true',
                       help='显示所有层的详细信息')
    
    args = parser.parse_args()
    
    # 检查文件
    if not Path(args.weights).exists():
        LOGGER.error(f"权重文件不存在: {args.weights}")
        return 1
    
    # 分析模型
    analyze_model_layers(args.weights, args.details)
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())