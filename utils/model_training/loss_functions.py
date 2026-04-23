import torch
import torch.nn as nn

class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        """
        quantiles: list like [0.05,0.1,0.5,0.9,0.95]
        """
        super().__init__()

        self.register_buffer(
            "quantiles",
            torch.tensor(quantiles).view(1, 1, -1)
        )

    def forward(self, pred, target):
        """
        pred: (B, N, Q)
        target: (B, N)
        """

        target = target.unsqueeze(-1)  # -> (B,N,1)

        error = target - pred

        loss = torch.maximum(
            self.quantiles * error,
            (self.quantiles - 1) * error
        )

        return loss.mean()