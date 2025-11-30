import logging
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from batchgenerators.utilities.file_and_folder_operations import join
import numpy as np
import os
import shutil
from torch.utils.tensorboard import SummaryWriter

matplotlib.use('agg')


class RegressionLogger(object):
    """
    简单的日志记录器，适用于回归任务。记录loss、lr、epoch时间等。
    支持使用标准库logging输出到文件和控制台。
    """

    def __init__(self, verbose: bool = False, log_to_file: bool = True, log_to_console: bool = True,
                 log_dir: str = None, config_path: str = None):
        self.metrics_dict = {
            'train_loss': [],
            'val_loss': [],
            'MAE': [],
            'lr': []
        }
        self.verbose = verbose
        self.log_dir = log_dir
        self.tb_writer = SummaryWriter(log_dir=log_dir)
        self.epoch = 0

        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)

        # 配置统一的日志处理器
        self._setup_logging(log_to_file, log_to_console)

        # 复制配置文件到日志目录
        if config_path :
            shutil.copy(config_path, os.path.join(self.log_dir, os.path.basename(config_path)))

    def _setup_logging(self, log_to_file: bool, log_to_console: bool):
        """配置日志处理器"""
        self.logger = logging.getLogger('RegressionLogger')
        self.logger.setLevel(logging.INFO)

        # 清除现有处理器避免重复
        self.logger.handlers.clear()

        formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        if log_to_console:
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        if log_to_file and self.log_dir:
            log_file = join(self.log_dir, f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt')
            fh = logging.FileHandler(log_file, encoding='utf-8')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def log_metrics(self, keys, values):
        """支持一次性记录多个key和value，自动顺序append，不需要epoch参数"""
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        if not isinstance(values, (list, tuple)):
            values = [values]
        if len(keys) != len(values):
            raise ValueError("keys和values长度必须一致")
        for k, v in zip(keys, values):
            if self.verbose:
                self.logger.info(f'logging {k}: {v} for epoch {len(self.metrics_dict[k])}')
            self.metrics_dict[k].append(v)

    def log_epoch(self, epoch, train_loss, val_loss, val_mae, lr, epoch_time):
        """记录epoch信息"""
        self.logger.info(f"Epoch {epoch + 1}")
        self.logger.info(f"Current learning rate: {lr:.5f}")
        self.logger.info(f"train_loss {train_loss:.4f}")
        self.logger.info(f"val_loss {val_loss:.4f}")
        self.logger.info(f"val_mae {val_mae:.4f}")
        self.logger.info(f"Epoch time: {epoch_time:.1f} s")
        self.logger.info("-----")

    @property
    def plot_progress_png(self):
        sns.set_theme(font_scale=2.5)
        # 取最短的logging长度，防止有的key没记录
        min_len = min([len(i) for i in self.metrics_dict.values()])
        if min_len == 0:
            return  # 没有数据不画图
        x_values = np.arange(min_len)
        train_losses = np.array(self.metrics_dict['train_loss'][:min_len])
        val_losses = np.array(self.metrics_dict['val_loss'][:min_len])
        maes = np.array(self.metrics_dict['MAE'][:min_len])
        lrs = np.array(self.metrics_dict['lr'][:min_len])
        # 保证目录存在
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 14), gridspec_kw={'height_ratios': [3, 1]})
        color1, color2, color3 = 'tab:blue', 'tab:red', 'tab:green'
        l1 = ax1.plot(x_values, train_losses, color=color1, label='train_loss', linewidth=3)
        l2 = ax1.plot(x_values, val_losses, color=color2, label='val_loss', linewidth=3)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        if len(train_losses) > 0 and len(val_losses) > 0:
            all_loss = np.concatenate([train_losses, val_losses])
            if all_loss.max() > all_loss.min():
                ax1.set_ylim(all_loss.min() - 0.05 * abs(all_loss.min()), all_loss.max() + 0.05 * abs(all_loss.max()))
        ax1.grid(True, which='both', axis='y', linestyle='--', alpha=0.3)
        ax1b = ax1.twinx()
        l3 = ax1b.plot(x_values, maes, color=color3, label='val_mae', linewidth=3)
        ax1b.set_ylabel('MAE', color=color3)
        ax1b.tick_params(axis='y', labelcolor=color3)
        if len(maes) > 0 and maes.max() > maes.min():
            ax1b.set_ylim(maes.min() - 0.05 * abs(maes.min()), maes.max() + 0.05 * abs(maes.max()))
        lines = l1 + l2 + l3
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='upper right')
        ax2.plot(x_values, lrs, color='tab:purple', label='learning rate', linewidth=3)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        if len(lrs) > 0 and lrs.max() > lrs.min():
            ax2.set_ylim(lrs.min() - 0.05 * abs(lrs.min()), lrs.max() + 0.05 * abs(lrs.max()))
        ax2.legend(loc='upper right')
        ax2.grid(True, which='both', axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        fig.savefig(join(self.log_dir, 'progress.png'))
        plt.close(fig)

    def info(self, msg):
        self.logger.info(msg)

    def log_metrics_tensorboard(self, metrics: dict, epoch: int):
        """将指标写入TensorBoard"""
        for k, v in metrics.items():
            self.tb_writer.add_scalar(k, v, epoch)
        self.tb_writer.flush()

if __name__ == "__main__":
    # 测试RegressionLogger
    logger = RegressionLogger(log_dir=r".\logs",log_to_file=False)
    for epoch in range(5):
        train_loss = np.random.rand()
        val_loss = np.random.rand()
        val_mae = np.random.rand()
        lr = 0.001 * (0.95 ** epoch)
        epoch_time = np.random.rand() * 10
        logger.log_epoch(epoch, train_loss, val_loss, val_mae, lr, epoch_time)
        logger.log_metrics(['train_loss', 'val_loss', 'MAE', 'lr'], [train_loss, val_loss, val_mae, lr])
    logger.plot_progress_png
