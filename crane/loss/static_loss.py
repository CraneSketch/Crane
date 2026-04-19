import torch.nn as nn

# Regression
MSELoss = nn.MSELoss()
MAELoss = nn.L1Loss()
SmoothL1Loss = nn.SmoothL1Loss()
HuberLoss = nn.HuberLoss()

# Classification
BCELoss = nn.BCELoss()
BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
