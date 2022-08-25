import torch
import torch.nn as nn

class WassersteinLoss(nn.Module):
    def __init__(self) -> None:
        super(WassersteinLoss, self).__init__()

    def forward(
        self,
        y_pred: torch.Tensor,
        y_target: torch.Tensor,
    ) -> torch.Tensor:

        loss = - torch.mean(y_pred * y_target)
        return loss


class GradientPenalty(nn.Module):
    def __init__(self) -> None:
        super(GradientPenalty, self).__init__()

    def forward(
        self,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
    ) -> torch.Tensor:

        grad = torch.autograd.grad(
            inputs=inputs,
            outputs=outputs,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True,
        )[0]
        grad_ = torch.norm(grad.reshape(grad.size(0), -1), p=2, dim=1)
        penalty = torch.mean((1. - grad_) ** 2)
        return penalty