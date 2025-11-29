import torch.nn as nn


def mae():
    return nn.L1Loss()


def mse():
    return nn.MSELoss()


def huber(delta=1.0):
    return nn.SmoothL1Loss(beta=delta)
