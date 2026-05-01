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
        pred: (B, N, Q) or (B, N, H, Q)
        target: (B, N) or (B, N, H)
        """
        if pred.ndim == 3:
            quantiles = self.quantiles
            target = target.unsqueeze(-1)  # -> (B,N,1)
        elif pred.ndim == 4:
            quantiles = self.quantiles.view(1, 1, 1, -1)
            target = target.unsqueeze(-1)  # -> (B,N,H,1)
        else:
            raise ValueError(f"QuantileLoss expects pred rank 3 or 4, got shape {tuple(pred.shape)}")

        if target.shape[:-1] != pred.shape[:-1]:
            raise ValueError(
                f"QuantileLoss shape mismatch: pred={tuple(pred.shape)}, target={tuple(target.shape)}"
            )

        error = target - pred

        loss = torch.maximum(
            quantiles * error,
            (quantiles - 1) * error
        )

        return loss.mean()
