from typing import Optional, Literal
import torch
from torch.utils.data import DataLoader, TensorDataset

from .builders import get_model, get_loss
from ..utils import get_device

__all__ = [
    "train_model",              # GPU-native fast path
    "make_loader",              # helper for CPU-resident data
    "train_model_with_loader",  # full DataLoader pipeline
]

# ------------------------------------------------------------------ #
# Scheduler factory
# ------------------------------------------------------------------ #
def _make_scheduler(optimizer, name: str, epochs: int, kw: Optional[dict]):
    kw = {} if kw is None else kw
    name = name.lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, **kw)
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, **kw)
    elif name == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler '{name}'")

# ------------------------------------------------------------------ #
# 1. Pure-GPU training – X, y already on GPU
# ------------------------------------------------------------------ #
def train_model(
    X,
    y,
    *,
    method: Literal["mlp", "mlpclass", "linreg", "logistic"] = "mlp",
    epochs: int = 300,
    batch_size: int = 100000,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    scheduler_name: str = "cosine",
    scheduler_kw: Optional[dict] = None,
    log_interval: int = 50,
):
    """
    Fast path – assumes X, y are already GPU tensors.
    """
    device = get_device() if device is None else device
    if y.ndim == 1:                             # view only, no copy
        y = y.view(-1, 1)

    model = get_model(method, X.shape[1]).to(device)
    loss_fn, _ = get_loss(method)

    # Closed-form OLS for linear regression
    if method == "linreg":
        with torch.no_grad():
            w, *_ = torch.linalg.lstsq(X, y)
        model.w.weight.data.copy_(w.T)
        model.w.bias.data.zero_()
        return model.eval()

    optim     = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = _make_scheduler(optim, scheduler_name, epochs, scheduler_kw)

    n        = X.shape[0]
    all_idx  = torch.arange(n, device=device)
    model.train()

    for epoch in range(1, epochs + 1):
        perm = all_idx[torch.randperm(n, device=device)]
        total_loss, batches = 0.0, 0

        for i in range(0, n, batch_size):
            idx       = perm[i : i + batch_size]
            pred      = model(X[idx])
            loss      = loss_fn(pred, y[idx])

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            total_loss += loss.item()
            batches    += 1

        if scheduler is not None:
            scheduler.step()

        # if epoch == 1 or epoch % log_interval == 0 or epoch == epochs:
        #     print(f"Epoch {epoch:3d}/{epochs} | lr={optim.param_groups[0]['lr']:.2e} "
        #           f"| avg_loss={total_loss / max(batches,1):.6f}")

    return model.eval()

# ------------------------------------------------------------------ #
# 2. DataLoader helper – for CPU-resident data
# ------------------------------------------------------------------ #
def make_loader(
    X: torch.Tensor,
    y: torch.Tensor,
    idx,
    *,
    batch_size: int = 50_000,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    """
    Build a DataLoader backed by *views* of X[idx], y[idx].
    Intended for data on CPU; uses pin_memory and (optionally) workers.
    """
    ds = TensorDataset(X[idx], y[idx])
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

def train_model_with_loader(
    X,
    y,
    *,
    method: Literal["mlp", "mlpclass", "linreg", "logistic"] = "mlp",
    epochs: int = 250,
    batch_size: int = 50_000,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    scheduler_name: str = "cosine",
    scheduler_kw: Optional[dict] = None,
    log_interval: int = 50,
    num_workers: int = 0,
):
    """
    Full pipeline for data initially on CPU.
    The DataLoader streams mini-batches to GPU on the fly.
    """
    device = get_device() if device is None else device
    X      = torch.as_tensor(X, dtype=torch.float32)   # keep on CPU
    y      = torch.as_tensor(y, dtype=torch.float32).view(-1, 1)

    model     = get_model(method, X.shape[1]).to(device)
    loss_fn, _ = get_loss(method)

    if method == "linreg":
        with torch.no_grad():
            w, *_ = torch.linalg.lstsq(X.to(device), y.to(device))
        model.w.weight.data.copy_(w.T)
        model.w.bias.data.zero_()
        return model.eval()

    optim     = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = _make_scheduler(optim, scheduler_name, epochs, scheduler_kw)
    loader    = make_loader(X, y, slice(None),
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss, batches = 0.0, 0
        for xb, yb in loader:                     # xb, yb still on CPU
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            total_loss += loss.item()
            batches    += 1

        if scheduler is not None:
            scheduler.step()

        # if epoch == 1 or epoch % log_interval == 0 or epoch == epochs:
        #     print(f"Epoch {epoch:3d}/{epochs} | lr={optim.param_groups[0]['lr']:.2e} "
        #           f"| avg_loss={total_loss / max(batches,1):.6f}")

    return model.eval()