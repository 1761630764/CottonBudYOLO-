"""
TensorBoard训练回调 for YOLOv11-DN

在训练过程中自动记录到TensorBoard的回调类
可以直接集成到训练脚本中

使用方法:
    from tools.tensorboard_callback import TensorBoardCallback
    
    # 创建回调
    tb_callback = TensorBoardCallback(log_dir='runs/tensorboard/exp1')
    
    # 训练时使用
    model.train(
        data='wheat.yaml',
        epochs=100,
        callbacks={'on_train_epoch_end': tb_callback.on_epoch_end}
    )
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# 添加项目路径
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


class TensorBoardCallback:
    """
    YOLOv11-DN训练的TensorBoard回调
    
    自动记录:
    - 训练/验证损失
    - 评估指标 (mAP, precision, recall)
    - 学习率
    - 域权重分布 (如果有DN模块)
    - 参数和梯度分布
    """
    
    def __init__(
        self,
        log_dir: str = 'runs/tensorboard',
        log_domain_weights: bool = True,
        log_gradients: bool = False,
        log_parameters: bool = False,
        log_images: bool = True,
        log_freq: int = 1
    ):
        """
        初始化TensorBoard回调
        
        Args:
            log_dir: TensorBoard日志目录
            log_domain_weights: 是否记录域权重
            log_gradients: 是否记录梯度分布
            log_parameters: 是否记录参数分布
            log_images: 是否记录图像
            log_freq: 记录频率（每多少个epoch）
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        self.log_domain_weights = log_domain_weights
        self.log_gradients = log_gradients
        self.log_parameters = log_parameters
        self.log_images = log_images
        self.log_freq = log_freq
        
        self.global_step = 0
        
        print(f"✓ TensorBoard回调已初始化")
        print(f"  日志目录: {self.log_dir}")
        print(f"  启动命令: tensorboard --logdir {self.log_dir.parent}")
    
    def on_train_start(self, trainer):
        """训练开始时调用"""
        try:
            # 记录模型架构
            if hasattr(trainer, 'model') and trainer.model is not None:
                # 记录模型信息
                model_info = self._get_model_info(trainer.model)
                self.writer.add_text('model/info', model_info, 0)
                print(f"  ✓ 模型信息已记录")
                
                # 跳过计算图记录（YOLO模型有动态形状，trace会失败）
                # 如果需要模型结构，可以查看 model/info 文本或使用 model.info()
                print(f"  ℹ 计算图记录已跳过（YOLO模型使用动态形状）")
                
            # 强制刷新
            self.writer.flush()
        except Exception as e:
            print(f"  警告: on_train_start失败 - {e}")
            import traceback
            traceback.print_exc()
    
    def on_train_batch_end(self, trainer):
        """每个训练批次结束时调用"""
        self.global_step += 1
    
    def on_train_epoch_end(self, trainer):
        """每个训练epoch结束时调用"""
        try:
            epoch = getattr(trainer, 'epoch', 0)
            
            # 只在指定频率记录
            if epoch % self.log_freq != 0:
                return
            
            # print(f"  [TensorBoard] 记录 Epoch {epoch} 训练数据...")
            
            # 1. 记录训练损失
            if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
                loss_items = trainer.loss_items
                if hasattr(loss_items, '__len__') and len(loss_items) >= 3:
                    try:
                        box_loss = float(loss_items[0])
                        cls_loss = float(loss_items[1])
                        dfl_loss = float(loss_items[2])
                        
                        self.writer.add_scalar('train/box_loss', box_loss, epoch)
                        self.writer.add_scalar('train/cls_loss', cls_loss, epoch)
                        self.writer.add_scalar('train/dfl_loss', dfl_loss, epoch)
                        
                        total_loss = box_loss + cls_loss + dfl_loss
                        self.writer.add_scalar('train/total_loss', total_loss, epoch)
                        print(f"    ✓ 损失已记录: total={total_loss:.4f}")
                    except Exception as e:
                        print(f"    警告: 损失记录失败 - {e}")
            
            # 2. 记录学习率
            if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
                try:
                    for i, param_group in enumerate(trainer.optimizer.param_groups):
                        lr = param_group['lr']
                        self.writer.add_scalar(f'train/lr_group_{i}', lr, epoch)
                    print(f"    ✓ 学习率已记录: {lr:.6f}")
                except Exception as e:
                    print(f"    警告: 学习率记录失败 - {e}")
            
            # 3. 记录参数分布
            if self.log_parameters and hasattr(trainer, 'model') and trainer.model is not None:
                try:
                    self._log_parameter_distributions(trainer.model, epoch)
                    print(f"    ✓ 参数分布已记录")
                except Exception as e:
                    print(f"    警告: 参数分布记录失败 - {e}")
            
            # 4. 记录梯度分布
            if self.log_gradients and hasattr(trainer, 'model') and trainer.model is not None:
                try:
                    self._log_gradient_distributions(trainer.model, epoch)
                    print(f"    ✓ 梯度分布已记录")
                except Exception as e:
                    print(f"    警告: 梯度分布记录失败 - {e}")
            
            # 强制刷新
            self.writer.flush()
            
        except Exception as e:
            print(f"  警告: on_train_epoch_end失败 - {e}")
            import traceback
            traceback.print_exc()
    
    def on_val_end(self, validator):
        """验证结束时调用"""
        try:
            epoch = getattr(validator, 'epoch', 0)
            # print(f"  [TensorBoard] 记录 Epoch {epoch} 验证数据...")
            
            # 获取验证指标
            if hasattr(validator, 'metrics') and validator.metrics is not None:
                metrics = validator.metrics
                
                # mAP指标
                if hasattr(metrics, 'box') and metrics.box is not None:
                    box_metrics = metrics.box
                    try:
                        if hasattr(box_metrics, 'map') and box_metrics.map is not None:
                            self.writer.add_scalar('val/mAP50-95', float(box_metrics.map), epoch)
                        if hasattr(box_metrics, 'map50') and box_metrics.map50 is not None:
                            self.writer.add_scalar('val/mAP50', float(box_metrics.map50), epoch)
                            print(f"    ✓ mAP50: {float(box_metrics.map50):.4f}")
                        if hasattr(box_metrics, 'map75') and box_metrics.map75 is not None:
                            self.writer.add_scalar('val/mAP75', float(box_metrics.map75), epoch)
                        
                        # Precision & Recall
                        if hasattr(box_metrics, 'mp') and box_metrics.mp is not None:
                            self.writer.add_scalar('val/precision', float(box_metrics.mp), epoch)
                            print(f"    ✓ Precision: {float(box_metrics.mp):.4f}")
                        if hasattr(box_metrics, 'mr') and box_metrics.mr is not None:
                            self.writer.add_scalar('val/recall', float(box_metrics.mr), epoch)
                            print(f"    ✓ Recall: {float(box_metrics.mr):.4f}")
                        
                        # F1 Score
                        if hasattr(box_metrics, 'mp') and hasattr(box_metrics, 'mr'):
                            if box_metrics.mp is not None and box_metrics.mr is not None:
                                mp = float(box_metrics.mp)
                                mr = float(box_metrics.mr)
                                if mp + mr > 0:
                                    f1 = 2 * (mp * mr) / (mp + mr)
                                    self.writer.add_scalar('val/f1_score', f1, epoch)
                                    print(f"    ✓ F1 Score: {f1:.4f}")
                    except Exception as e:
                        print(f"    警告: 验证指标记录失败 - {e}")
            
            # 强制刷新
            self.writer.flush()
        
        except Exception as e:
            print(f"  警告: on_val_end失败 - {e}")
            import traceback
            traceback.print_exc()
    
    def on_fit_epoch_end(self, trainer):
        """训练+验证epoch结束时调用"""
        try:
            epoch = trainer.epoch
            
            # 综合记录训练和验证指标
            if hasattr(trainer, 'metrics'):
                metrics = trainer.metrics
                
                # 如果metrics是字典
                if isinstance(metrics, dict):
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            self.writer.add_scalar(f'metrics/{key}', value, epoch)
            
            # 记录域权重（如果启用）
            if self.log_domain_weights and hasattr(trainer, 'model'):
                self._log_domain_weights_from_model(trainer.model, epoch)
        
        except Exception as e:
            print(f"  警告: on_fit_epoch_end失败 - {e}")
    
    def on_train_end(self, trainer):
        """训练结束时调用"""
        try:
            # 记录最终结果
            if hasattr(trainer, 'best'):
                best_metrics = trainer.best
                if isinstance(best_metrics, dict):
                    text = "# 最佳结果\n\n"
                    for key, value in best_metrics.items():
                        text += f"- **{key}**: {value}\n"
                    self.writer.add_text('results/best_metrics', text, 0)
            
            print(f"✓ TensorBoard日志已保存到: {self.log_dir}")
            
        except Exception as e:
            print(f"  警告: on_train_end失败 - {e}")
        finally:
            self.writer.close()
    
    # ==================== 辅助函数 ====================
    
    def _get_model_info(self, model) -> str:
        """获取模型信息"""
        text = "# YOLOv11-DN 模型信息\n\n"
        
        try:
            # 统计参数量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            text += f"## 参数统计\n"
            text += f"- 总参数: {total_params:,}\n"
            text += f"- 可训练参数: {trainable_params:,}\n"
            text += f"- 冻结参数: {total_params - trainable_params:,}\n\n"
            
            # 统计模块类型
            module_counts = {}
            for name, module in model.named_modules():
                module_type = type(module).__name__
                module_counts[module_type] = module_counts.get(module_type, 0) + 1
            
            text += f"## 模块统计\n"
            sorted_modules = sorted(module_counts.items(), key=lambda x: x[1], reverse=True)
            for module_type, count in sorted_modules[:15]:
                text += f"- {module_type}: {count}\n"
            
            # 领域自适应模块
            dn_modules = {k: v for k, v in module_counts.items() if 'Domain' in k or 'DN' in k}
            if dn_modules:
                text += f"\n## 领域自适应模块\n"
                for module_type, count in dn_modules.items():
                    text += f"- **{module_type}**: {count}\n"
        
        except Exception as e:
            text += f"\n错误: {e}\n"
        
        return text
    
    def _log_parameter_distributions(self, model, epoch: int):
        """记录参数分布"""
        try:
            for name, param in model.named_parameters():
                if param.requires_grad and param.numel() > 0:
                    # 只记录关键层（避免太多数据）
                    if any(keyword in name for keyword in ['dn', 'domain', 'conv', 'weight']):
                        self.writer.add_histogram(f'parameters/{name}', param.data, epoch)
        except Exception as e:
            print(f"  警告: 记录参数分布失败 - {e}")
    
    def _log_gradient_distributions(self, model, epoch: int):
        """记录梯度分布"""
        try:
            for name, param in model.named_parameters():
                if param.grad is not None and param.requires_grad:
                    # 只记录关键层
                    if any(keyword in name for keyword in ['dn', 'domain', 'conv']):
                        self.writer.add_histogram(f'gradients/{name}', param.grad, epoch)
                        
                        # 梯度范数
                        grad_norm = param.grad.norm().item()
                        self.writer.add_scalar(f'grad_norms/{name}', grad_norm, epoch)
        except Exception as e:
            print(f"  警告: 记录梯度分布失败 - {e}")
    
    def _log_domain_weights_from_model(self, model, epoch: int):
        """从模型中提取并记录域权重"""
        try:
            from ultralytics.nn.modules import DomainNormalization, C3k2_DN
            
            domain_weights_collected = False
            
            for name, module in model.named_modules():
                # 查找DomainNormalization模块
                if isinstance(module, DomainNormalization):
                    if hasattr(module, 'domain_weights_cache'):
                        weights = module.domain_weights_cache
                        if weights is not None:
                            # 记录平均权重
                            mean_weights = weights.mean(dim=0).cpu().numpy()
                            for d, w in enumerate(mean_weights):
                                self.writer.add_scalar(
                                    f'domain_weights/{name}/domain_{d}',
                                    w,
                                    epoch
                                )
                            domain_weights_collected = True
            
            if not domain_weights_collected:
                # 如果没有缓存，尝试前向传播收集
                pass  # 需要输入数据，跳过
        
        except Exception as e:
            print(f"  警告: 记录域权重失败 - {e}")
    
    def __del__(self):
        """析构函数"""
        try:
            self.writer.close()
        except:
            pass


def integrate_tensorboard_to_trainer(
    trainer,
    log_dir: str = 'runs/tensorboard',
    **kwargs
):
    """
    将TensorBoard回调集成到trainer中
    
    Args:
        trainer: YOLO trainer对象
        log_dir: 日志目录
        **kwargs: TensorBoardCallback的其他参数
    
    Returns:
        TensorBoardCallback实例
    """
    callback = TensorBoardCallback(log_dir=log_dir, **kwargs)
    
    # 注册回调
    if hasattr(trainer, 'add_callback'):
        trainer.add_callback('on_train_start', callback.on_train_start)
        trainer.add_callback('on_train_batch_end', callback.on_train_batch_end)
        trainer.add_callback('on_train_epoch_end', callback.on_train_epoch_end)
        trainer.add_callback('on_val_end', callback.on_val_end)
        trainer.add_callback('on_fit_epoch_end', callback.on_fit_epoch_end)
        trainer.add_callback('on_train_end', callback.on_train_end)
        print("✓ TensorBoard回调已集成到trainer")
    else:
        print("✗ Trainer不支持回调，请手动调用")
    
    return callback


# 示例使用
if __name__ == '__main__':
    print("TensorBoard回调模块")
    print("请在训练脚本中导入使用:")
    print()
    print("from tools.tensorboard_callback import TensorBoardCallback")
    print()
    print("# 创建回调")
    print("tb_callback = TensorBoardCallback(log_dir='runs/tensorboard/exp1')")
    print()
    print("# 在训练中使用")
    print("model = YOLO('yolo11-dn.yaml')")
    print("model.train(")
    print("    data='wheat.yaml',")
    print("    epochs=100,")
    print("    callbacks={")
    print("        'on_train_epoch_end': tb_callback.on_train_epoch_end,")
    print("        'on_val_end': tb_callback.on_val_end,")
    print("    }")
    print(")")

