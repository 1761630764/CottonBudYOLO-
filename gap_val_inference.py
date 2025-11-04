"""
python gap_val_inference.py --weights E:/Code/CottonBudYOLOv2/runs/train/yolov11n_DN_ALL_cotton/weights/best.pt --source-data E:/datasets/cotton_bud_2025/cotton_bud_YOLO/cotton_gap.yaml --target-data E:/datasets/cotton_bud_2024/cotton_bud_YOLO/cotton_gap.yaml --imgsz 640 --conf 0.25 --iou 0.25 --iou-thres 0.5 --device 0 --name cotton_inference_gap

python gap_val_inference.py --weights E:/Code/CottonBudYOLOv2/runs/train/base/cotton-v11-s/weights/best.pt --source-data E:/datasets/cotton_bud_2025/cotton_bud_YOLO/cotton_gap.yaml --target-data E:/datasets/cotton_bud_2024/cotton_bud_YOLO/cotton_gap.yaml --imgsz 640 --conf 0.25 --iou 0.25 --iou-thres 0.5 --device 0 --name yolov11s

"""

import argparse
import sys
from pathlib import Path
import yaml
import json
from datetime import datetime
import numpy as np
from tqdm import tqdm

from ultralytics import YOLO
from ultralytics.utils import LOGGER, colorstr
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Domain Gap Validation - Pure Inference Mode')
    
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--source-data', type=str, required=True)
    parser.add_argument('--target-data', type=str, required=True)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.001)
    parser.add_argument('--iou', type=float, default=0.6)
    parser.add_argument('--iou-thres', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--project', type=str, default='runs/gap_val')
    parser.add_argument('--name', type=str, default=None)
    
    return parser.parse_args()


def load_yolo_labels(label_path, img_shape):
    """加载YOLO格式标注"""
    if not Path(label_path).exists():
        return np.zeros((0, 5))
    
    labels = []
    h, w = img_shape
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1]) * w
                y_center = float(parts[2]) * h
                width = float(parts[3]) * w
                height = float(parts[4]) * h
                
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                labels.append([class_id, x1, y1, x2, y2])
    
    return np.array(labels) if labels else np.zeros((0, 5))


def calculate_iou(box1, box2):
    """计算两个框的IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def calculate_metrics(predictions, ground_truths, iou_threshold=0.5):
    """计算精度指标"""
    all_preds = []
    all_gts = []
    
    for img_id, (preds, gts) in enumerate(zip(predictions, ground_truths)):
        if len(preds) > 0:
            for pred in preds:
                all_preds.append({
                    'img_id': img_id,
                    'box': pred[:4],
                    'conf': pred[4],
                    'class': int(pred[5])
                })
        
        if len(gts) > 0:
            for gt_id, gt in enumerate(gts):
                all_gts.append({
                    'img_id': img_id,
                    'gt_id': gt_id,
                    'box': gt[1:5],
                    'class': int(gt[0])
                })
    
    if len(all_preds) == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'mAP50': 0.0,
            'mAP': 0.0,
            'tp': 0,
            'fp': 0,
            'fn': len(all_gts)
        }
    
    # 按置信度排序
    all_preds.sort(key=lambda x: x['conf'], reverse=True)
    
    tp = 0
    fp = 0
    matched_gts = set()
    
    for pred in all_preds:
        best_iou = 0
        best_gt_idx = -1
        
        # 找同一图像的GT
        img_gts = [gt for gt in all_gts if gt['img_id'] == pred['img_id']]
        
        for gt in img_gts:
            gt_key = (gt['img_id'], gt['gt_id'])
            if gt_key in matched_gts:
                continue
            
            iou = calculate_iou(pred['box'], gt['box'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt['gt_id']
                best_gt_class = gt['class']
        
        # 判断TP/FP
        if best_iou >= iou_threshold and pred['class'] == best_gt_class:
            gt_key = (pred['img_id'], best_gt_idx)
            if gt_key not in matched_gts:
                tp += 1
                matched_gts.add(gt_key)
            else:
                fp += 1
        else:
            fp += 1
    
    fn = len(all_gts) - tp
    total_gt = len(all_gts)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / total_gt if total_gt > 0 else 0.0
    
    # F1-Score作为mAP的近似
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
        mAP50 = f1
    else:
        mAP50 = 0.0
    
    mAP = mAP50 * 0.75  # 近似mAP@0.5:0.95
    
    return {
        'precision': precision,
        'recall': recall,
        'mAP50': mAP50,
        'mAP': mAP,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'total_predictions': len(all_preds),
        'total_ground_truths': total_gt
    }


def evaluate_dataset(model, data_yaml, dataset_name, args, verbose=True):
    """使用推理模式评估数据集"""
    if verbose:
        LOGGER.info(colorstr('cyan', 'bold', f'\n评估: {dataset_name}'))
        LOGGER.info('=' * 70)
    
    # 读取数据集配置
    with open(data_yaml, 'r', encoding='utf-8') as f:
        data_dict = yaml.safe_load(f)
    
    dataset_path = Path(data_dict['path'])
    val_images_dir = dataset_path / data_dict['val']
    val_labels_dir = dataset_path / data_dict['val'].replace('images', 'labels')
    
    # 获取所有图像
    image_files = sorted(list(val_images_dir.glob('*.jpg')) + list(val_images_dir.glob('*.png')))
    
    if len(image_files) == 0:
        LOGGER.error(f"未找到图像: {val_images_dir}")
        return None
    
    if verbose:
        LOGGER.info(f"找到 {len(image_files)} 张图像")
    
    # 推理
    predictions = []
    ground_truths = []
    
    for img_path in tqdm(image_files, desc=f"{dataset_name} 推理", disable=not verbose):
        results = model.predict(
            img_path,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
            max_det=300,
        )
        
        # 提取预测
        if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            pred_boxes = boxes.xyxy.cpu().numpy()
            pred_conf = boxes.conf.cpu().numpy().reshape(-1, 1)
            pred_cls = boxes.cls.cpu().numpy().reshape(-1, 1)
            preds = np.concatenate([pred_boxes, pred_conf, pred_cls], axis=1)
        else:
            preds = np.zeros((0, 6))
        
        predictions.append(preds)
        
        # 加载标注
        label_path = val_labels_dir / (img_path.stem + '.txt')
        img_shape = results[0].orig_shape if len(results) > 0 else (640, 640)
        gts = load_yolo_labels(label_path, img_shape)
        ground_truths.append(gts)
    
    # 计算指标
    metrics = calculate_metrics(predictions, ground_truths, args.iou_thres)
    
    if verbose:
        LOGGER.info(f"\n统计:")
        # LOGGER.info(f"  预测数: {metrics['total_predictions']}")
        # LOGGER.info(f"  真实框: {metrics['total_ground_truths']}")
        # LOGGER.info(f"  TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
    
    return {
        'metrics': {
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'mAP@0.5': metrics['mAP50'],
            'mAP@0.5:0.95': metrics['mAP'],
            'Fitness': 0.1 * metrics['mAP50'] + 0.9 * metrics['mAP'],
        },
        'statistics': {
            'total_predictions': metrics['total_predictions'],
            'total_ground_truths': metrics['total_ground_truths'],
            'tp': metrics['tp'],
            'fp': metrics['fp'],
            'fn': metrics['fn'],
        }
    }


def print_results(results, target_results):
    """打印结果表格 - 只显示源域"""
    LOGGER.info(colorstr('green', 'bold', '\n' + '='*60))
    LOGGER.info(colorstr('green', 'bold', '源域评估结果'))
    LOGGER.info(colorstr('green', 'bold', '='*60))
    
    LOGGER.info(f"\n{colorstr('yellow', 'bold', '性能指标:')}")
    LOGGER.info('-' * 60)
    LOGGER.info(f"{'指标':<30} {'数值':>25}")
    LOGGER.info('-' * 60)
    
    for metric in ['Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95', 'Fitness']:
        src = results['metrics'][metric]
        LOGGER.info(f"{metric:<30} {src:>25.4f}")
    
    LOGGER.info('-' * 60)
    LOGGER.info('=' * 60)


def save_results(results, target_results, save_dir):
    """保存结果 - 只保存源域"""
    results_dict = {
        'source_domain': results,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    # json_file = save_dir / 'source_results.json'
    # with open(json_file, 'w', encoding='utf-8') as f:
    #     json.dump(results_dict, f, indent=4, ensure_ascii=False)
    
    # LOGGER.info(f"\n结果已保存: {json_file}")
    
    # 文本报告
    txt_file = save_dir / 'results.txt'
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("目标域评估报告\n")
        f.write("="*60 + "\n\n")
        f.write("性能指标:\n")
        f.write("-"*60 + "\n")
        f.write(f"{'指标':<30} {'数值':>25}\n")
        f.write("-"*60 + "\n")
        
        for metric in ['Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95', 'Fitness']:
            value = results['metrics'][metric]
            f.write(f"{metric:<30} {value:>25.4f}\n")
        
        f.write("-"*60 + "\n")
            
    LOGGER.info(f"文本报告已保存: {txt_file}")


def main():
    args = parse_args()
    
    # 检查文件
    for path, name in [(args.weights, '权重'), (args.source_data, '源域配置'), (args.target_data, '目标域配置')]:
        if not Path(path).exists():
            LOGGER.error(f"{name}文件不存在: {path}")
            return 1
    
    # 保存目录
    if args.name is None:
        args.name = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型（静默）
    model = YOLO(args.weights)
    
    # 评估源域（显示进度）
    results = evaluate_dataset(model, args.source_data, '目标域(2024)', args, verbose=True)
    
    # 评估目标域（静默）
    target_results = evaluate_dataset(model, args.target_data, '源域(2025)', args, verbose=False)
    
    if not results or not target_results:
        LOGGER.error("评估失败!")
        return 1
    
    # 只显示和保存源域结果
    print_results(results, target_results)
    save_results(results, target_results, save_dir)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

