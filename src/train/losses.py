import torch
import torch.nn as nn


class DynamicRangeMSELoss(nn.Module):
    """
    Weighted MSE that emphasizes accurate reconstruction near the extremes (0 or 1).
    The weight increases as the ground-truth deviates from the neutral 0.5 position,
    encouraging the model to faithfully reproduce strong expressions.
    """

    def __init__(self, boundary_weight: float = 2.0, exponent: float = 2.0):
        super().__init__()
        self.boundary_weight = boundary_weight
        self.exponent = exponent

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base = (pred - target) ** 2
        # |target - 0.5| * 2 maps 0.5 -> 0 and {0,1} -> 1
        emphasis = torch.pow(torch.abs(target - 0.5) * 2.0, self.exponent)
        weights = 1.0 + self.boundary_weight * emphasis
        weighted = base * weights
        return weighted.mean()


def build_loss_fn(loss_config: dict) -> nn.Module:
    loss_type = loss_config.get("type", "mse").lower()

    if loss_type == "dynamic_range":
        params = loss_config.get("dynamic_range", {})
        boundary_weight = params.get("boundary_weight", 2.0)
        exponent = params.get("exponent", 2.0)
        return DynamicRangeMSELoss(boundary_weight=boundary_weight, exponent=exponent)

    # Default to standard MSE
    return nn.MSELoss()
