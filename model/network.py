"""PyTorch network architecture for band gap regression, plus its Snake activation.

FlexNet has independent per-layer widths and activations, and is the project's only model
class. BandGapNet's cached .pt weights no longer load, though each has a .meta.json sidecar
recording the architecture behind it.
"""
import torch
import torch.nn as nn

_CORR_THRESHOLD = 0.95


class Snake(nn.Module):
    """Snake activation, f(x) = x + (1/a) * sin^2(a * x)."""
    def __init__(self, a: float = 1.0):
        super().__init__()
        self.a = a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (1.0 / self.a) * torch.sin(self.a * x) ** 2


# Activation choices that FlexNet's hidden layers pick from independently.
FLEX_ACTIVATIONS: dict[str, "type[nn.Module]"] = {
    "relu":       nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "silu":       nn.SiLU,   # a.k.a. Swish
    "gelu":       nn.GELU,
    "snake":      Snake,
}


class FlexNet(nn.Module):
    """Feedforward network with independent per-layer widths and activation choice, the
    project's only model class."""
    def __init__(
        self,
        input_dim: int,
        layer_widths: list[int],
        activations: list[str],
        log_transform: bool = False,
    ):
        """Builds the hidden stack from layer_widths, so the depth is len(layer_widths) and there
        is no separate n_layers argument. log_transform is a plain instance flag rather than a
        learned parameter, so state_dict() does not save it and a model rebuilt from a checkpoint
        needs it from its .meta.json."""
        super().__init__()
        if not layer_widths:
            raise ValueError("layer_widths must contain at least one hidden layer width")
        if len(activations) != len(layer_widths):
            raise ValueError(
                f"activations must have one entry per layer ({len(layer_widths)}), got {len(activations)}"
            )
        unknown = set(activations) - set(FLEX_ACTIVATIONS)
        if unknown:
            raise ValueError(f"activations must all be one of {sorted(FLEX_ACTIVATIONS)}, got {sorted(unknown)}")
        self.log_transform = log_transform
        layers: list[nn.Module] = []
        in_dim = input_dim
        for width, activation in zip(layer_widths, activations):
            layers.append(nn.Linear(in_dim, width))
            layers.append(FLEX_ACTIVATIONS[activation]())
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predictions in eV whatever log_transform is set to, so use this rather than
        forward() when comparing against real values. expm1 runs in float64, as a diverged model
        can push raw log1p output past float32's overflow threshold of approximately 88.7, where
        float64 allows approximately 709, and the result is left unclamped (tried and reverted,
        DECISIONS.md) since a ceiling would stop r2_score raising on non-finite input."""
        raw = self.forward(x)
        return torch.expm1(raw.double()) if self.log_transform else raw
