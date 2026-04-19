import torch
import torch.nn as nn

# Regression
MSELoss = nn.MSELoss()
MAELoss = nn.L1Loss()
SmoothL1Loss = nn.SmoothL1Loss()
HuberLoss = nn.HuberLoss()


def LogMAELoss(pred, target):
    pred = torch.log1p(torch.clamp(pred, min=0.0))
    target = torch.log1p(torch.clamp(target, min=0.0))
    return torch.mean(torch.abs(pred - target))

# Classification
BCELoss = nn.BCELoss()
BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
