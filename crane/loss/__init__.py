from .static_loss import MSELoss, MAELoss, SmoothL1Loss, HuberLoss, BCELoss, BCEWithLogitsLoss
from .dynamic_loss import Auto_Weighted_MSE_and_MAE


def build_loss_fn(loss_fn):
    if loss_fn == 'SmoothL1Loss':
        return SmoothL1Loss
    elif loss_fn == 'MSELoss':
        return MSELoss
    elif loss_fn == 'MAELoss':
        return MAELoss
    elif loss_fn == 'HuberLoss':
        return HuberLoss
    elif loss_fn == 'Auto_Weighted_MSE_and_MAE':
        return Auto_Weighted_MSE_and_MAE()
    elif loss_fn == 'BCELoss':
        return BCELoss
    elif loss_fn == 'BCEWithLogitsLoss':
        return BCEWithLogitsLoss
    else:
        raise ValueError('Unknown loss function')