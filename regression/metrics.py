import torch


def mae(pred, target):
    return torch.mean(torch.abs(pred - target)).item()


def mse(pred, target):
    return torch.mean((pred - target) ** 2).item()
