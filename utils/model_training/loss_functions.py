import torch
import torch.nn as nn

class QuantileLoss(nn.Module):
    def __init__(self, quantiles, horizon_weights=None):
        """
        quantiles: list like [0.05,0.1,0.5,0.9,0.95]
        horizon_weights: optional list like [1.3,1.2,1.0] for multi-horizon loss weighting.
        """
        super().__init__()

        self.register_buffer(
            "quantiles",
            torch.tensor(quantiles).view(1, 1, -1)
        )
        if horizon_weights is None:
            self.horizon_weights = None
        else:
            self.register_buffer(
                "horizon_weights",
                torch.tensor(horizon_weights, dtype=torch.float32).view(1, 1, -1, 1)
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

        if pred.ndim == 4 and self.horizon_weights is not None:
            if self.horizon_weights.shape[2] != pred.shape[2]:
                raise ValueError(
                    "QuantileLoss horizon weight mismatch: "
                    f"pred has {pred.shape[2]} horizons, weights have {self.horizon_weights.shape[2]}"
                )
            weights = self.horizon_weights.to(device=loss.device, dtype=loss.dtype)
            weights = weights / weights.mean().clamp_min(1e-8)
            loss = loss * weights

        return loss.mean()
