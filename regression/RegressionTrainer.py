import os
import torch
import torch.nn as nn
import numpy as np
import datetime
from paths import regression_results
from optimizer import PolyLRScheduler
from dataset import RegressionNpzDataset
from regression_logger import RegressionLogger
from torch.utils.data import DataLoader
from preprocess import process_dataset
from utils import set_seed
import yaml
import csv

class RegressionTrainer:
    def __init__(self, config_path: str, fold: int, dataset: int):
        # 加载配置文件
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # 随机种子
        seed = config.get('seed', 42)
        set_seed(seed)

        # 基本配置
        self.config = config
        self.fold = fold
        self.dataset = dataset
        self.device = config.get("device", "cuda")
        self.output_dir = self.create_output_dir()
        self.num_epochs = config.get("num_epochs", 100)

        # 模型初始化
        self.model = self.get_model()
        self.model = self.model.to(self.device)

        # 数据集
        self.train_loader, self.val_loader = self.get_data_loader()

        # 优化器
        self.optimizer = self.get_optimizer()

        # 学习率调度器
        self.lr_scheduler = self.get_scheduler()

        # 损失函数
        self.criterion = self.get_loss()

        # 日志记录器
        self.logger = RegressionLogger(log_dir=self.output_dir, config_path=config_path)

    def train_one_epoch(self):
        total_loss = 0.0
        self.model.train()
        for batch in self.train_loader:
            x = batch['image'].float().unsqueeze(1).to(self.device)
            y = batch['label'].float().to(self.device)
            pred = self.model(x)

            loss = self.criterion(pred, y.float())
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(self.train_loader.dataset)
        return avg_loss

    def validate(self):
        total_loss = 0.0
        self.model.eval()
        preds, targets = [], []
        with (torch.no_grad()):
            for batch in self.val_loader:
                x = batch['image'].float().unsqueeze(1).to(self.device)
                y = batch['label'].float().to(self.device)
                pred = self.model(x)
                loss = self.criterion(pred, y.float())
                total_loss += loss.item() * x.size(0)
                preds.append(pred.cpu().numpy())
                targets.append(y.cpu().numpy())
        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        mae = np.mean(np.abs(preds - targets))
        avg_loss = total_loss / len(self.val_loader.dataset)
        return avg_loss, mae

    def run(self):

        best_val_loss = float('inf')
        try:
            for epoch in range(self.num_epochs):
                epoch_start = datetime.datetime.now()
                train_loss = self.train_one_epoch()
                val_loss, val_mae = self.validate()
                lr = self.optimizer.param_groups[0]['lr']
                epoch_end = datetime.datetime.now()
                epoch_time = (epoch_end - epoch_start).total_seconds()
                self.logger.log_epoch(epoch, train_loss, val_loss, val_mae * 108, lr, epoch_time)
                self.logger.log_metrics(['train_loss', 'val_loss', 'MAE', "lr"],
                                        [train_loss, val_loss, val_mae * 108, lr])
                # 写入TensorBoard
                self.logger.log_metrics_tensorboard({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'MAE': val_mae * 108,
                    'lr': lr
                }, epoch)

                # self.logger.plot_progress_png

                if (epoch + 1) % 50 == 0:
                    torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'checkpoint_latest.pth'))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'checkpoint_best.pth'))
                    self.logger.info("Best model updated !")
                self.lr_scheduler.step()
        except Exception as e:
            self.logger.info(f"Training error: {e}")
            raise
        self.logger.info("Training completed.")
        # 训练结束后生成验证集预测csv
        self.save_val_scores()

    def create_output_dir(self):
        output_dir = os.path.join(regression_results, f"Dataset{self.dataset}_WB", f'fold_{self.fold}',
                                  datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def get_model(self):
        # if "resnet" in name:
        #     # norm_op字符串转为类
        #     norm_op_str = model_cfg.get("norm_op", "BatchNorm3d")
        #     norm_op = getattr(torch.nn, norm_op_str)
        #     norm_op_kwargs = model_cfg.get("norm_op_kwargs", {})
        #     # 强制eps为float类型
        #     if 'eps' in norm_op_kwargs:
        #         norm_op_kwargs['eps'] = float(norm_op_kwargs['eps'])
        #     model = ConfigurableResNet3D(
        #         in_channels=model_cfg.get("in_channels", 1),
        #         n_stages=model_cfg.get("n_stages", 4),
        #         features_per_stage=model_cfg.get("features_per_stage", [64, 128, 256, 512]),
        #         kernel_sizes=[tuple(k) for k in model_cfg.get("kernel_sizes", [[3, 3, 3]] * 4)],
        #         strides=model_cfg.get("strides", [[1, 1, 1], [1, 2, 2], [1, 2, 2], [2, 2, 2]]),
        #         conv_bias=model_cfg.get("conv_bias", False),
        #         norm_op=norm_op,
        #         norm_op_kwargs=norm_op_kwargs,
        #         layers=model_cfg.get("layers", [2, 2, 2, 2])
        #     )
        #     return model

        model_cfg = self.config.get("model", {"name": "MLP"})
        model_name = model_cfg.get("name", "MLP")
        model_class = getattr(__import__('model', fromlist=[model_name]), model_name)
        kwargs = {k: v for k, v in model_cfg.items() if k != "name"}

        # 设置默认输入尺寸
        target_shape = self.config.get("loader", {}).get("target_shape", (16, 648, 256))
        kwargs.setdefault("input_shape", target_shape)
        return model_class(**kwargs)

    def get_loss(self):
        loss_cfg = self.config.get("loss", {"name": "SmoothL1Loss"})
        loss_name = loss_cfg.get("name", "SmoothL1Loss")
        kwargs = {k: v for k, v in loss_cfg.items() if k != "name"}
        loss_class = getattr(nn, loss_name)
        return loss_class(**kwargs)

    def get_optimizer(self):
        optim_cfg = self.config.get("optimizer", {"name": "SGD"})
        optim_name = optim_cfg.get("name", "SGD")
        optim_class = getattr(torch.optim, optim_name)
        kwargs = {k: v for k, v in optim_cfg.items() if k != "name"}
        if optim_name == "SGD":
            kwargs.setdefault("momentum", 0.99)
            kwargs.setdefault("nesterov", True)
        return optim_class(self.model.parameters(), **kwargs)

    def get_scheduler(self):
        sched_cfg = self.config.get("scheduler", {"name": "PolyLRScheduler"})
        sched_name = sched_cfg.get("name", "PolyLRScheduler")
        lr = self.config.get("optimizer", {}).get("lr", 1e-3)
        if sched_name == "PolyLRScheduler":
            return PolyLRScheduler(self.optimizer, lr, self.num_epochs)

        sched_class = getattr(torch.optim.lr_scheduler, sched_name)
        kwargs = {k: v for k, v in sched_cfg.items() if k != "name"}
        return sched_class(self.optimizer, **kwargs)

    def get_data_loader(self):
        loader = self.config.get("loader", {})
        batch_size = loader.get("batch_size", 8)
        target_shape = loader.get("target_shape", (16, 648, 256))
        process_dataset(self.dataset, target_shape=target_shape, npz_key='arr', verbose=False)
        train_dataset = RegressionNpzDataset(self.dataset, target_shape, fold=self.fold, mode='train')
        val_dataset = RegressionNpzDataset(self.dataset, target_shape, fold=self.fold, mode='val')
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), \
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    def save_val_scores(self):
        """用最佳权重对验证集生成预测分数csv"""
        # 加载最佳权重
        best_ckpt = os.path.join(self.output_dir, 'checkpoint_best.pth')
        if not os.path.isfile(best_ckpt):
            self.logger.info(f"Best checkpoint not found: {best_ckpt}")
            return
        self.model.load_state_dict(torch.load(best_ckpt, map_location=self.device))
        self.model.eval()
        results = []
        with torch.no_grad():
            for batch in self.val_loader:
                x = batch['image'].float().unsqueeze(1).to(self.device)
                identifiers = batch['identifier']
                preds = self.model(x).cpu().numpy().reshape(-1)
                for id, pred in zip(identifiers, preds):
                    results.append([id, float(pred)])

        # 保存为csv
        csv_path = os.path.join(self.output_dir, 'validation.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'score'])
            writer.writerows(results)
        self.logger.info(f"Validation finished")
