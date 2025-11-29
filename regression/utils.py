import os
import numpy as np
import random
import torch

from torchsummary import summary


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_npz_shape(npz_path):
    """
    读取npz文件并返回第一个数组的形状。
    """
    arr = np.load(npz_path)
    # 取第一个key对应的数组
    first_key = list(arr.keys())[0]
    return arr[first_key].shape

def create_onnx(model, dummy_input, onnx_path, **kwargs):
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'], **kwargs)
    print(f"Exported ONNX to {onnx_path}")


def print_model_summary(model, input_size):
    summary(model, input_size=input_size)


def check_grads(model):
    print("=== 参数梯度分析 ===")
    total_zero_grads = 0
    total_params = 0

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_mean = param.grad.mean().item()
            grad_std = param.grad.std().item()
            grad_min = param.grad.min().item()
            grad_max = param.grad.max().item()

            # 统计接近0的梯度
            zero_mask = param.grad.abs() < 1e-6
            zero_ratio = zero_mask.float().mean().item()
            zero_count = zero_mask.sum().item()

            total_zero_grads += zero_count
            total_params += param.grad.numel()

            print(f"参数: {name}")
            print(f"  形状: {list(param.grad.shape)}")
            print(f"  梯度均值: {grad_mean:.8f}")
            print(f"  梯度标准差: {grad_std:.8f}")
            print(f"  梯度范围: [{grad_min:.8f}, {grad_max:.8f}]")
            print(f"  零梯度比例: {zero_ratio:.4f} ({zero_count}/{param.grad.numel()})")
            print("---")

    overall_zero_ratio = total_zero_grads / total_params if total_params > 0 else 0
    print(f"总体零梯度比例: {overall_zero_ratio:.4f}")
    return overall_zero_ratio



if __name__ == "__main__":
    # 测试get_npz_shape函数
    test_npz_path = r"E:\machine_learning\model\nnUNet\nnUNet_results\Dataset101_WB\nnUNetTrainer__nnUNetPlans__2d\fold_0\validation\625416_2022.npz"
    shape = get_npz_shape(test_npz_path)
    print(f"The shape of the first array in the npz file is: {shape}")
