"""Small utility helpers used by the current CAME evaluation scripts."""

import torch
import torch.nn as nn
from torch import optim


def create_optimizer(opt: str, model: nn.Module, lr: float, weight_decay: float):
    opt_lower = opt.lower().split("_")[-1]
    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    if opt_lower == "adam":
        return optim.Adam(parameters, **opt_args)
    if opt_lower == "adamw":
        return optim.AdamW(parameters, **opt_args)
    if opt_lower == "adadelta":
        return optim.Adadelta(parameters, **opt_args)
    if opt_lower == "radam":
        return optim.RAdam(parameters, **opt_args)
    if opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    raise ValueError(f"Invalid optimizer: {opt}")


class LogisticRegression(nn.Module):
    def __init__(self, num_dim: int, num_class: int):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double().sum().item()
    return correct / len(y_true)


__all__ = ["LogisticRegression", "accuracy", "create_optimizer"]
