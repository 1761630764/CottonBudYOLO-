"""
YOLOv11-DN 推理可视化脚本（带热力图）


使用示例：
    # 单张图片
    python infer.py --weights E:/Code/CottonBudYOLOv2/runs/train/yolov11n_DN_ALL_cotton/weights/best.pt --source E:/Code/CottonBudYOLOv2/images_test/test_one/cotton.jpg --heatmap
    
   python infer.py --weights E:/Code/4、YOLOv11_UP/runs/train/wheat-v11-s/weights/best.pt --source E:/Code/CottonBudYOLOv2/images_test/test_one/wheat.png

    # 视频
    python infer.py --weights best.pt --source video.mp4 
    
    # 文件夹
    python infer.py --weights best.pt --source images/ --heatmap --save-txt
    python infer.py --weights E:/Code/CottonBudYOLOv2/runs/train/yolov11n_DN_ALL_cotton/weights/best.pt --source E:/datasets/cotton_bud_2024/cotton_bud_YOLO/test/images/  --save-txt
"""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics.utils import LOGGER

from utils.inference_visualizer import InferenceVisualizer


def process_image(visualizer, image_path, save_dir, save_txt=False, save_conf=False, view_img=False):
    """处理单张图像"""
    LOGGER.info(f"处理图像: {image_path}")
    
    # 推理
    results, annotated_img, inf_time = visualizer.predict_image(image_path)
    
    # 保存结果
    save_path = save_dir / Path(image_path).name
    cv2.imwrite(str(save_path), annotated_img)
    
    # 保存纯热力图
    if visualizer.use_heatmap and visualizer.heatmap_generator is not None:
        heatmap_only = visualizer.heatmap_generator.get_heatmap_only()
        heatmap_path = save_dir / f"{Path(image_path).stem}_heatmap.jpg"
        cv2.imwrite(str(heatmap_path), heatmap_only)
    
    # 保存txt标注
    if save_txt:
        img = cv2.imread(str(image_path))
        txt_path = save_dir / 'labels' / f"{Path(image_path).stem}.txt"
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        visualizer.save_txt(results, txt_path, img.shape, save_conf)
    
    # 显示结果
    if view_img:
        cv2.imshow('YOLOv11-DN Inference', annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    LOGGER.info(f"✅ 保存到: {save_path} ({inf_time:.3f}s)")


def process_video(visualizer, video_path, save_dir, save_txt=False, save_conf=False, view_img=False):
    """处理视频"""
    LOGGER.info(f"处理视频: {video_path}")
    
    # 打开视频
    if str(video_path).isnumeric():
        cap = cv2.VideoCapture(int(video_path))
        save_path = save_dir / 'webcam.mp4'
    else:
        cap = cv2.VideoCapture(str(video_path))
        save_path = save_dir / Path(video_path).name
    
    if not cap.isOpened():
        LOGGER.error(f"无法打开视频: {video_path}")
        return
    
    # 获取视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    LOGGER.info(f"视频信息: {width}x{height} @ {fps}fps, {total_frames} 帧")
    
    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))
    
    frame_idx = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # 推理
            t0 = time.time()
            results = visualizer.model.predict(
                frame,
                conf=visualizer.conf_thres,
                iou=visualizer.iou_thres,
                device=visualizer.device,
                verbose=False
            )[0]
            t1 = time.time()
            
            # 可视化
            annotated_frame = visualizer.annotate_image(frame.copy(), results)
            annotated_frame = visualizer.apply_heatmap(annotated_frame, results)
            
            # 添加信息文本
            info_text = f"Frame: {frame_idx}/{total_frames if total_frames > 0 else '?'} | " \
                       f"FPS: {1/(t1-t0):.1f} | Detections: {len(results.boxes)}"
            cv2.putText(annotated_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 写入视频
            out.write(annotated_frame)
            
            # 显示结果
            if view_img:
                cv2.imshow('YOLOv11-DN Inference', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # 更新统计
            visualizer._update_stats(results, t1 - t0)
            
            # 打印进度
            if frame_idx % 30 == 0:
                elapsed = time.time() - start_time
                fps_avg = frame_idx / elapsed
                LOGGER.info(f"处理进度: {frame_idx}/{total_frames if total_frames > 0 else '?'} "
                          f"({fps_avg:.1f} fps)")
    
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    total_time = time.time() - start_time
    LOGGER.info(f"✅ 视频保存到: {save_path}")
    LOGGER.info(f"   总帧数: {frame_idx}")
    LOGGER.info(f"   总时间: {total_time:.2f}s")
    LOGGER.info(f"   平均FPS: {frame_idx/total_time:.2f}")


def process_directory(visualizer, dir_path, save_dir, save_txt=False, save_conf=False, view_img=False):
    """处理文件夹"""
    LOGGER.info(f"处理文件夹: {dir_path}")
    
    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(dir_path).glob(f'*{ext}'))
        image_files.extend(Path(dir_path).glob(f'*{ext.upper()}'))
    
    if not image_files:
        LOGGER.warning(f"文件夹中没有找到图像文件: {dir_path}")
        return
    
    LOGGER.info(f"找到 {len(image_files)} 张图像")
    
    # 处理每张图像
    for i, image_path in enumerate(image_files, 1):
        LOGGER.info(f"\n[{i}/{len(image_files)}] 处理: {image_path.name}")
        process_image(visualizer, str(image_path), save_dir, save_txt, save_conf, view_img)
    
    LOGGER.info(f"\n✅ 批量处理完成，共处理 {len(image_files)} 张图像")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLOv11-DN 推理可视化（带热力图）')
    
    # 基本参数
    parser.add_argument('--weights', type=str, default='runs/train/yolo11-dn5/weights/best.pt', help='模型权重路径')
    parser.add_argument('--source', type=str, default='E:/datasets/gwhd_2021/Oringin_YOLO/images/0001.png', help='输入源（图像/视频/文件夹/摄像头ID）')
    parser.add_argument('--output', type=str, default='runs/infer', help='输出目录')
    parser.add_argument('--name', type=str, default='exp', help='实验名称')
    
    # 推理参数
    parser.add_argument('--conf-thres', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IOU阈值')
    parser.add_argument('--device', type=str, default='0', help='设备 (cuda:0 或 cpu)')
    
    # 热力图参数
    parser.add_argument('--heatmap', action='store_true', help='启用热力图')
    parser.add_argument('--heatmap-colormap', type=str, default='JET',
                       choices=['JET', 'HOT', 'COOL', 'RAINBOW', 'OCEAN', 'WINTER', 'SPRING', 'SUMMER', 'AUTUMN'],
                       help='热力图颜色映射')
    parser.add_argument('--heatmap-decay', type=float, default=0.99, help='热力图衰减率 (0-1)')
    parser.add_argument('--heatmap-alpha', type=float, default=0.5, help='热力图透明度 (0-1)')
    
    # 可视化参数
    parser.add_argument('--view-img', action='store_true', help='显示结果')
    
    # 保存参数
    parser.add_argument('--save-txt', action='store_true', help='保存txt标注')
    parser.add_argument('--save-conf', action='store_true', help='在txt中保存置信度')
    parser.add_argument('--exist-ok', action='store_true', help='覆盖已存在的实验')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 打印标题
    print("\n" + "="*70)
    print("YOLOv11-DN 推理可视化（带热力图）")
    print("="*70 + "\n")
    
    # 检查权重文件
    if not Path(args.weights).exists():
        LOGGER.error(f"权重文件不存在: {args.weights}")
        return
    
    # 检查输入源
    if not args.source.isnumeric():
        if not Path(args.source).exists():
            LOGGER.error(f"输入源不存在: {args.source}")
            return
    
    # 创建输出目录
    save_dir = Path(args.output) / args.name
    if save_dir.exists() and not args.exist_ok:
        save_dir = Path(str(save_dir) + '_' + str(int(time.time())))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    LOGGER.info(f"输出目录: {save_dir}")
    
    # 创建推理可视化器
    visualizer = InferenceVisualizer(
        model_path=args.weights,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        device=args.device,
        use_heatmap=args.heatmap,
        heatmap_colormap=args.heatmap_colormap,
        heatmap_decay=args.heatmap_decay,
        heatmap_alpha=args.heatmap_alpha,
    )
    
    # 判断输入类型并处理
    source = args.source
    source_path = Path(source) if not source.isnumeric() else None
    
    try:
        if source.isnumeric():
            # 摄像头
            process_video(visualizer, source, save_dir, args.save_txt, args.save_conf, args.view_img)
        elif source_path.is_file():
            if source_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.flv']:
                # 视频文件
                process_video(visualizer, source, save_dir, args.save_txt, args.save_conf, args.view_img)
            else:
                # 图像文件
                process_image(visualizer, source, save_dir, args.save_txt, args.save_conf, args.view_img)
        elif source_path.is_dir():
            # 文件夹
            process_directory(visualizer, source, save_dir, args.save_txt, args.save_conf, args.view_img)
        else:
            LOGGER.error(f"不支持的输入源: {source}")
            return
        
        # 打印统计信息
        visualizer.print_stats()
        
        print("\n✅ 推理完成！")
        print(f"结果保存在: {save_dir}")
        
    except Exception as e:
        LOGGER.error(f"推理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

