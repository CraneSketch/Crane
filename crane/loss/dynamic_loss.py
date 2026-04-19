import torch
import torch.nn as nn


class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=2, weights=None, device=None):
        self.weights = weights
        learnable = True
        if weights is not None:
            learnable = False
        self.learnable = learnable
        super(AutomaticWeightedLoss, self).__init__()
        if learnable:
            params = torch.ones(num, requires_grad=learnable)
            self.params = torch.nn.Parameter(params)
        else:
            self.params = torch.tensor(weights, requires_grad=learnable, device=device)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


class Auto_Weighted_MSE_and_MAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.auto_weighted_loss = AutomaticWeightedLoss(2)
        self.mse_func = torch.nn.MSELoss()

    def forward(self, weight_pred, weight_y):
        mae = torch.mean(torch.abs((weight_pred - weight_y) / weight_y))
        mse = self.mse_func(weight_pred, weight_y)
        return self.auto_weighted_loss(mae, mse)

