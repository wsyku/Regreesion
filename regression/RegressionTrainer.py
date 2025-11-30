import os
import torch
import torch.nn as nn
import numpy as np
import datetime
from typing import Dict
from paths import regression_results
from optimizer import PolyLRScheduler
from dataset import RegressionNpzDataset
from regression_logger import RegressionLogger
from model import ConfigurableResNet3D
from utils import check_grads
from torch.utils.data import DataLoader
from preprocess import process_dataset


#TODO 1.生成预测结果2.自定义损失函数等
#TODO 随机种子设置


class RegressionTrainer:
    def __init__(self, config: Dict, fold: int, dataset: int):
        self.config = config
        self.fold = fold
        self.device = config.get("device", "cuda")
        self.model = self.get_model()
        self.dataset = dataset
        self.output_dir = self.create_output_dir()
        self.lr = config.get("lr", 1e-3)
        self.batch_size = config.get("batch_size", 8)
        self.weight_decay = config.get("weight_decay", 1e-5)
        self.num_epochs = config.get("num_epochs", 100)
        self.target_shape = config.get("target_shape", (16, 648, 256))

        self.create_preprocessed_data()

        train_dataset = RegressionNpzDataset(self.dataset, self.target_shape, fold=self.fold, mode='train')
        val_dataset = RegressionNpzDataset(self.dataset, self.target_shape, fold=self.fold, mode='val')
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.99, nesterov=True,
                                         weight_decay=self.weight_decay)
        self.scheduler = PolyLRScheduler(self.optimizer, self.lr, self.num_epochs)
        self.logger = RegressionLogger(log_dir=self.output_dir)

    def train_one_epoch(self):
        total_loss = 0.0
        self.model.train()
        for batch in self.train_loader:
            x = batch['image'].float().unsqueeze(1).to(self.device)
            y = batch['label'].float().to(self.device)
            pred = self.model(x)
            # print('pred:', pred.detach().cpu().numpy(), 'label:', y.detach().cpu().numpy())

            loss = self.criterion(pred, y.float())
            self.optimizer.zero_grad()
            loss.backward()

            # for name, param in self.model.named_parameters():
            #     if param.grad is not None:
            #         grad_mean = param.grad.mean().item()
            #         grad_max = param.grad.max().item()
            #         print(f"{name}: grad mean={grad_mean:.6e}, max={grad_max:.6e}")
            #     else:
            #         print(f"{name}: grad is None")

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
                self.logger.plot_progress_png
                if (epoch + 1) % 50 == 0:
                    torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'checkpoint_latest.pth'))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'checkpoint_best.pth'))
                    self.logger.info("Best model updated !")
                self.scheduler.step()
        except Exception as e:
            self.logger.info(f"Training error: {e}")
            raise
        self.logger.info("Training completed.")

    def create_output_dir(self):
        output_dir = os.path.join(regression_results, f"Dataset{self.dataset}_WB", f'fold_{self.fold}')
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def get_model(self):

        model_cfg = self.config.get("model", {})
        name = model_cfg.get("name", "resnet18")
        if "resnet" in name:
            # norm_op字符串转为类
            norm_op_str = model_cfg.get("norm_op", "BatchNorm3d")
            norm_op = getattr(torch.nn, norm_op_str)
            norm_op_kwargs = model_cfg.get("norm_op_kwargs", {})
            # 强制eps为float类型
            if 'eps' in norm_op_kwargs:
                norm_op_kwargs['eps'] = float(norm_op_kwargs['eps'])
            model = ConfigurableResNet3D(
                in_channels=model_cfg.get("in_channels", 1),
                n_stages=model_cfg.get("n_stages", 4),
                features_per_stage=model_cfg.get("features_per_stage", [64, 128, 256, 512]),
                kernel_sizes=[tuple(k) for k in model_cfg.get("kernel_sizes", [[3, 3, 3]] * 4)],
                strides=model_cfg.get("strides", [[1, 1, 1], [1, 2, 2], [1, 2, 2], [2, 2, 2]]),
                conv_bias=model_cfg.get("conv_bias", False),
                norm_op=norm_op,
                norm_op_kwargs=norm_op_kwargs,
                layers=model_cfg.get("layers", [2, 2, 2, 2])
            )
            return model.to(self.device)
        else:
            try:
                model_class = getattr(__import__('model', fromlist=[name]), name)
                model = model_class(**model_cfg)
                return model.to(self.device)
            except AttributeError:
                raise ValueError(f"Unsupported model name: {name}")

    def create_preprocessed_data(self):
        process_dataset(self.dataset, target_shape=self.target_shape, npz_key='arr', verbose=True)
