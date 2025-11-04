"""
使用示例 (Examples):

python train_domain.py --model yolo11-DN-ALL.yaml --data E:/datasets/tea_bud/tea_bud_YOLO/tea.yaml --epochs 150 --name yolov11s_DN_ALL_tea --no-tensorboard

"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import torch
from datetime import datetime
from ultralytics import YOLO
from ultralytics.utils import LOGGER

# 导入TensorBoard回调
try:
    from tools.tensorboard_callback import TensorBoardCallback
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    LOGGER.warning("TensorBoard callback not found. Run without visualization.")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train YOLOv11-DN with domain adaptation')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='yolo11-dn-att.yaml',
                       help='Model configuration file (yolo11-dn.yaml or yolo11-dn-lite.yaml)')
    parser.add_argument('--weights', type=str, default=None,
                       help='Initial weights path (optional)')
    
    # Data configuration
    parser.add_argument('--data', type=str, default="E:/datasets/gwhd_2021/wheat_YOLO/wheat.yaml",
                       help='Dataset configuration file (YAML)')
    parser.add_argument('--domains', type=str, nargs='+', default=None,
                       help='List of domain names for multi-domain training')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size')
    parser.add_argument('--device', type=str, default='0',
                       help='CUDA device (e.g., 0 or 0,1,2,3 or cpu)')
    
    # Domain adaptation parameters
    parser.add_argument('--use-domain-loss', action='store_true',
                       help='Enable domain adversarial loss')
    parser.add_argument('--domain-lambda', type=float, default=0.1,
                       help='Weight for domain adversarial loss')
    parser.add_argument('--num-domains', type=int, default=3,
                       help='Number of domains for domain normalization')
    
    # Optimization
    parser.add_argument('--lr0', type=float, default=0.01,
                       help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01,
                       help='Final learning rate (lr0 * lrf)')
    parser.add_argument('--momentum', type=float, default=0.937,
                       help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                       help='Optimizer weight decay')
    parser.add_argument('--warmup-epochs', type=float, default=3.0,
                       help='Warmup epochs')
    
    # Augmentation
    parser.add_argument('--hsv-h', type=float, default=0.015,
                       help='HSV-Hue augmentation')
    parser.add_argument('--hsv-s', type=float, default=0.7,
                       help='HSV-Saturation augmentation')
    parser.add_argument('--hsv-v', type=float, default=0.4,
                       help='HSV-Value augmentation')
    parser.add_argument('--degrees', type=float, default=0.0,
                       help='Rotation augmentation (degrees)')
    parser.add_argument('--translate', type=float, default=0.1,
                       help='Translation augmentation (fraction)')
    parser.add_argument('--scale', type=float, default=0.5,
                       help='Scale augmentation (gain)')
    parser.add_argument('--shear', type=float, default=0.0,
                       help='Shear augmentation (degrees)')
    parser.add_argument('--perspective', type=float, default=0.0,
                       help='Perspective augmentation')
    parser.add_argument('--flipud', type=float, default=0.0,
                       help='Vertical flip probability')
    parser.add_argument('--fliplr', type=float, default=0.5,
                       help='Horizontal flip probability')
    parser.add_argument('--mosaic', type=float, default=1.0,
                       help='Mosaic augmentation probability')
    parser.add_argument('--mixup', type=float, default=0.0,
                       help='MixUp augmentation probability')
    
    # Output
    parser.add_argument('--project', type=str, default='runs/train',
                       help='Save results to project/name')
    parser.add_argument('--name', type=str, default='yolo11-dn',
                       help='Experiment name')
    parser.add_argument('--exist-ok', action='store_true',
                       help='Overwrite existing experiment')
    parser.add_argument('--save-period', type=int, default=-1,
                       help='Save checkpoint every x epochs (-1 to disable)')
    
    # Validation
    parser.add_argument('--val', action='store_true', default=True,
                       help='Validate during training')
    parser.add_argument('--val-domains', type=str, nargs='+', default=None,
                       help='Domains to validate on (separate from training)')
    
    # TensorBoard可视化参数
    parser.add_argument('--tensorboard', action='store_true', default=True,
                       help='Enable TensorBoard visualization (default: True)')
    parser.add_argument('--tb-dir', type=str, default='runs/tensorboard',
                       help='TensorBoard log directory')
    parser.add_argument('--tb-log-domain', action='store_true', default=True,
                       help='Log domain weight distributions (default: True)')
    parser.add_argument('--tb-log-grads', action='store_true', default=True,
                       help='Log gradient distributions (default: True)')
    parser.add_argument('--tb-log-params', action='store_true', default=True,
                       help='Log parameter distributions (default: True)')
    parser.add_argument('--tb-freq', type=int, default=1,
                       help='TensorBoard logging frequency (epochs)')
    parser.add_argument('--no-tensorboard', action='store_true',
                       help='Disable TensorBoard visualization')
    
    # Other
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')
    parser.add_argument('--workers', type=int, default=2,
                       help='Number of dataloader workers')
    parser.add_argument('--cache', type=str, default='',
                       help='Image caching: ram, disk, or empty')
    parser.add_argument('--patience', type=int, default=50,
                       help='Epochs to wait for no improvement before early stopping')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from last checkpoint')
    
    return parser.parse_args()


def validate_config(args):
    """Validate training configuration."""
    # Check model file
    model_path = Path('ultralytics/cfg/models/11') / args.model
    if not model_path.exists() and not Path(args.model).exists():
        raise FileNotFoundError(f"Model config not found: {args.model}")
    
    # Check data file
    if not Path(args.data).exists():
        raise FileNotFoundError(f"Data config not found: {args.data}")
    
    # Load data config to check domains
    with open(args.data, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
    
    LOGGER.info(f"Data config loaded: {args.data}")
    LOGGER.info(f"  Classes: {data_cfg.get('nc', 'unknown')}")
    LOGGER.info(f"  Train: {data_cfg.get('train', 'unknown')}")
    LOGGER.info(f"  Val: {data_cfg.get('val', 'unknown')}")
    
    if 'domains' in data_cfg:
        LOGGER.info(f"  Domains: {data_cfg['domains']}")
    
    return data_cfg


def setup_model(args):
    """Setup YOLOv11-DN model."""
    LOGGER.info(f"Loading model: {args.model}")
    
    # Check if model path exists
    model_cfg = args.model
    if not Path(model_cfg).exists():
        # Try in ultralytics/cfg/models/11/
        model_cfg = f"ultralytics/cfg/models/11/{args.model}"
        if not Path(model_cfg).exists():
            raise FileNotFoundError(f"Model config not found: {args.model}")
    
    # Load model
    if args.weights:
        LOGGER.info(f"Loading weights: {args.weights}")
        model = YOLO(args.weights)
    else:
        LOGGER.info("Initializing model from config")
        model = YOLO(model_cfg)
    
    # Model info
    LOGGER.info("\n" + "="*50)
    LOGGER.info("Model Configuration:")
    LOGGER.info(f"  Type: YOLOv11-DN (Domain-Normalized)")
    LOGGER.info(f"  Config: {args.model}")
    if args.use_domain_loss:
        LOGGER.info(f"  Domain Adaptation: Enabled (λ={args.domain_lambda})")
    else:
        LOGGER.info(f"  Domain Adaptation: Disabled")
    LOGGER.info("="*50 + "\n")
    
    return model


def train(args):
    """Main training function."""
    # Print header
    print("\n" + "="*70)
    print("YOLOv11-DN Domain-Adaptive Training")
    if args.tensorboard and TENSORBOARD_AVAILABLE:
        print("TensorBoard可视化: 已启用 ✓")
    print("="*70 + "\n")
    
    # 显示TensorBoard状态
    if getattr(args, 'no_tensorboard', False):
        print("TensorBoard可视化: 已禁用 (--no-tensorboard)")
    elif args.tensorboard and TENSORBOARD_AVAILABLE:
        print("TensorBoard可视化: 已启用 ✓")
        print(f"  - 域权重记录: {'✓' if args.tb_log_domain else '✗'}")
        print(f"  - 梯度记录: {'✓' if args.tb_log_grads else '✗'}")
        print(f"  - 参数记录: {'✓' if args.tb_log_params else '✗'}")
    
    # Validate configuration
    data_cfg = validate_config(args)
    
    # Setup model
    model = setup_model(args)
    
    # Setup TensorBoard callback (如果启用)
    tb_callback = None
    # 检查是否禁用TensorBoard
    tensorboard_enabled = args.tensorboard and not getattr(args, 'no_tensorboard', False)
    
    if tensorboard_enabled and TENSORBOARD_AVAILABLE:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        tb_log_dir = os.path.join(args.tb_dir, f'{args.name}_{timestamp}')
        
        LOGGER.info(f"\n{'='*50}")
        LOGGER.info("TensorBoard Configuration:")
        LOGGER.info(f"  Log Directory: {tb_log_dir}")
        LOGGER.info(f"  Domain Weights: {'Enabled' if args.tb_log_domain else 'Disabled'}")
        LOGGER.info(f"  Gradients: {'Enabled' if args.tb_log_grads else 'Disabled'}")
        LOGGER.info(f"  Parameters: {'Enabled' if args.tb_log_params else 'Disabled'}")
        LOGGER.info(f"  Log Frequency: Every {args.tb_freq} epoch(s)")
        LOGGER.info(f"\n启动TensorBoard:")
        LOGGER.info(f"  tensorboard --logdir {args.tb_dir}")
        LOGGER.info(f"  浏览器访问: http://localhost:6006")
        LOGGER.info(f"{'='*50}\n")
        
        try:
            tb_callback = TensorBoardCallback(
                log_dir=tb_log_dir,
                log_domain_weights=args.tb_log_domain,
                log_gradients=args.tb_log_grads,
                log_parameters=args.tb_log_params,
                log_freq=args.tb_freq
            )
            LOGGER.info("✓ TensorBoard callback initialized successfully\n")
        except Exception as e:
            LOGGER.warning(f"Failed to initialize TensorBoard callback: {e}")
            tb_callback = None
    elif tensorboard_enabled and not TENSORBOARD_AVAILABLE:
        LOGGER.warning("TensorBoard requested but callback not available. Continuing without visualization.\n")
    elif getattr(args, 'no_tensorboard', False):
        LOGGER.info("TensorBoard disabled by --no-tensorboard flag\n")
    
    # Prepare training arguments
    train_args = {
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'device': args.device,
        'workers': args.workers,
        'cache': args.cache if args.cache else False,
        'project': args.project,
        'name': args.name,
        'exist_ok': args.exist_ok,
        'patience': args.patience,
        'save_period': args.save_period,
        'val': args.val,
        'seed': args.seed,
        'resume': args.resume,
        
        # Optimizer
        'lr0': args.lr0,
        'lrf': args.lrf,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'warmup_epochs': args.warmup_epochs,
        
        # Augmentation
        'hsv_h': args.hsv_h,
        'hsv_s': args.hsv_s,
        'hsv_v': args.hsv_v,
        'degrees': args.degrees,
        'translate': args.translate,
        'scale': args.scale,
        'shear': args.shear,
        'perspective': args.perspective,
        'flipud': args.flipud,
        'fliplr': args.fliplr,
        'mosaic': args.mosaic,
        'mixup': args.mixup,
    }
    
    # Add domain-specific parameters
    if args.use_domain_loss:
        train_args['domain_loss'] = True
        train_args['domain_lambda'] = args.domain_lambda
    
    # Print training configuration
    LOGGER.info("Training Configuration:")
    LOGGER.info(f"  Epochs: {args.epochs}")
    LOGGER.info(f"  Batch Size: {args.batch}")
    LOGGER.info(f"  Image Size: {args.imgsz}")
    LOGGER.info(f"  Device: {args.device}")
    LOGGER.info(f"  Learning Rate: {args.lr0} -> {args.lr0 * args.lrf}")
    LOGGER.info(f"  Warmup Epochs: {args.warmup_epochs}")
    if args.domains:
        LOGGER.info(f"  Training Domains: {args.domains}")
    LOGGER.info("")
    
    # Start training with TensorBoard callbacks
    LOGGER.info("Starting training...\n")
    
    # 注册TensorBoard回调 (使用model.add_callback方法)
    if tb_callback is not None:
        LOGGER.info("Training with TensorBoard visualization enabled")
        # 在训练前注册回调到模型
        model.add_callback('on_train_start', lambda trainer: tb_callback.on_train_start(trainer))
        model.add_callback('on_train_batch_end', lambda trainer: tb_callback.on_train_batch_end(trainer))
        model.add_callback('on_train_epoch_end', lambda trainer: tb_callback.on_train_epoch_end(trainer))
        model.add_callback('on_val_end', lambda validator: tb_callback.on_val_end(validator))
        model.add_callback('on_fit_epoch_end', lambda trainer: tb_callback.on_fit_epoch_end(trainer))
        model.add_callback('on_train_end', lambda trainer: tb_callback.on_train_end(trainer))
        LOGGER.info("✓ TensorBoard callbacks registered to model\n")
    
    try:
        results = model.train(**train_args)
        
        # Print results
        print("\n" + "="*70)
        print("Training Complete!")
        print("="*70)
        LOGGER.info(f"Results saved to: {model.trainer.save_dir}")
        
        # TensorBoard信息
        if tb_callback is not None:
            print("\n" + "="*70)
            print("TensorBoard Visualization")
            print("="*70)
            LOGGER.info(f"TensorBoard logs: {tb_callback.log_dir}")
            LOGGER.info(f"\n查看训练可视化:")
            LOGGER.info(f"  1. 启动TensorBoard: tensorboard --logdir {args.tb_dir}")
            LOGGER.info(f"  2. 浏览器访问: http://localhost:6006")
            LOGGER.info(f"  3. 查看以下内容:")
            LOGGER.info(f"     - 训练/验证损失曲线")
            LOGGER.info(f"     - 评估指标 (mAP, precision, recall)")
            if args.tb_log_domain:
                LOGGER.info(f"     - 域权重分布和热力图")
            if args.tb_log_grads:
                LOGGER.info(f"     - 梯度分布")
            if args.tb_log_params:
                LOGGER.info(f"     - 参数分布")
    
    except KeyboardInterrupt:
        LOGGER.warning("\n训练被用户中断")
        if tb_callback is not None and hasattr(tb_callback, 'writer'):
            tb_callback.writer.close()
        raise
    except Exception as e:
        LOGGER.error(f"\n训练出错: {e}")
        if tb_callback is not None and hasattr(tb_callback, 'writer'):
            tb_callback.writer.close()
        raise
    
    # Validation on test domains
    if args.val_domains:
        print("\n" + "="*70)
        print("Validating on test domains...")
        print("="*70)
        
        for domain in args.val_domains:
            LOGGER.info(f"\nValidating on domain: {domain}")
            # TODO: Implement domain-specific validation
            # This would require modifying the validation data config
    
    return results


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        results = train(args)
        LOGGER.info("\n✅ Training completed successfully!")
        return 0
    except Exception as e:
        LOGGER.error(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())


"""
使用示例 (Examples):

python train_domain.py --model yolo11-DN-ALL.yaml --data E:/datasets/tea_bud/tea_bud_YOLO/tea.yaml --epochs 150 --name yolov11s_DN_ALL_tea --no-tensorboard

"""

