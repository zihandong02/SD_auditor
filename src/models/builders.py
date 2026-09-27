from typing import Dict, Tuple, Callable, Optional

import torch
import torch.nn as nn

# ----------------------------------------------------------------------
# 1) Individual model classes
# ----------------------------------------------------------------------

class MLPRegressor(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 32), nn.ReLU(),
            nn.Linear(32, 32),   nn.ReLU(),
            nn.Linear(32, 32),   nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):          # (batch, n_in) → (batch, 1)
        return self.net(x)

class MLPRegressor1(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 64),   nn.ReLU(),      # widened input layer
            nn.Linear(64, 128),    nn.ReLU(),      # extra hidden layer
            nn.Linear(128, 64),    nn.ReLU(),
            nn.Linear(64, 32),     nn.ReLU(),
            nn.Linear(32, 1)                      # output layer
        )

    def forward(self, x):             # (batch, n_in) → (batch, 1)
        return self.net(x)

class MLPClassifier(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 32), nn.ReLU(),
            nn.Linear(32, 32),   nn.ReLU(),
            nn.Linear(32, 32),   nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):          # logits, not sigmoid
        return self.net(x)


class LinearReg(nn.Module):
    """Classic linear regression y = X·w + b (MSE loss)."""
    def __init__(self, n_in: int):
        super().__init__()
        self.w = nn.Linear(n_in, 1, bias=True)

    def forward(self, x):
        return self.w(x)


class LogisticReg(nn.Module):
    """Binary logistic regression (returns logits)."""
    def __init__(self, n_in: int):
        super().__init__()
        self.w = nn.Linear(n_in, 1, bias=True)

    def forward(self, x):
        return self.w(x)
    

# class AlphaModel_1(nn.Module):
#     """
#     Multiclass logistic regression with optional one hidden layer.
#     If hidden_dim is None, it's just a single linear→softmax.
#     Otherwise: input → Linear(hidden_dim) → ReLU → Linear(3) → softmax.
#     """
#     def __init__(self,
#                  dim_x: int,
#                  dim_w1: int,
#                  dim_w2: int,
#                  hidden_dim: Optional[int] = None):
#         super().__init__()
#         input_dim = dim_x + dim_w1 + dim_w2

#         self.use_hidden = hidden_dim is not None
#         if self.use_hidden:
#             # hidden layer + output layer
#             self.hidden = nn.Linear(input_dim, hidden_dim, bias=True)
#             self.activation = nn.ReLU()
#             self.output  = nn.Linear(hidden_dim, 3,      bias=True)
#         else:
#             # single linear layer to 3 logits
#             self.linear = nn.Linear(input_dim, 3, bias=True)

#     def forward(self,
#                 X: torch.Tensor,   # (batch_size, dim_x)
#                 W1: torch.Tensor,  # (batch_size, dim_w1)
#                 W2: torch.Tensor   # (batch_size, dim_w2)
#                ) -> torch.Tensor:  # returns (batch_size, 3)
#         # 1) concatenate inputs
#         H = torch.cat([X, W1, W2], dim=1)  # (batch, input_dim)

#         if self.use_hidden:
#             # 2a) hidden projection + nonlinearity
#             H = self.activation(self.hidden(H))  # (batch, hidden_dim)
#             # 3a) project to 3 logits
#             logits = self.output(H)              # (batch, 3)
#         else:
#             # 2b) directly project to 3 logits
#             logits = self.linear(H)              # (batch, 3)

#         # 4) softmax → probabilities summing to 1
#         alpha = torch.softmax(logits, dim=1)     # (batch, 3)
#         return alpha


class AlphaModel(nn.Module):
    """
    Multiclass logistic regression with optional 1- or 2-layer MLP.

    If hidden_dim is None:
        input -> Linear(3) -> softmax
    Else if hidden_dim2 is None:
        input -> Linear(hidden_dim) -> ReLU -> Linear(3) -> softmax
    Else:
        input -> Linear(hidden_dim) -> ReLU -> Linear(hidden_dim2) -> ReLU -> Linear(3) -> softmax
    """
    def __init__(self,
                 dim_x: int,
                 dim_w1: int,
                 dim_w2: int,
                 hidden_dim: Optional[int] = None,
                 hidden_dim2: Optional[int] = None):
        super().__init__()
        input_dim = dim_x + dim_w1 + dim_w2

        self.use_hidden = hidden_dim is not None
        if self.use_hidden:
            # first hidden layer
            self.hidden1 = nn.Linear(input_dim, hidden_dim, bias=True)
            self.act1    = nn.ReLU()

            # second hidden layer (optional)
            out_dim2 = hidden_dim if hidden_dim2 is None else hidden_dim2
            self.has_second = hidden_dim2 is not None
            if self.has_second:
                self.hidden2 = nn.Linear(hidden_dim, out_dim2, bias=True)
                self.act2    = nn.ReLU()
                self.output  = nn.Linear(out_dim2, 3, bias=True)
            else:
                self.output  = nn.Linear(hidden_dim, 3, bias=True)
        else:
            # single linear layer to 3 logits
            self.linear = nn.Linear(input_dim, 3, bias=True)

    def forward(self,
                X: torch.Tensor,   # (batch, dim_x)
                W1: torch.Tensor,  # (batch, dim_w1)
                W2: torch.Tensor   # (batch, dim_w2)
               ) -> torch.Tensor:  # (batch, 3)
        H = torch.cat([X, W1, W2], dim=1)

        if self.use_hidden:
            H = self.act1(self.hidden1(H))
            if getattr(self, "has_second", False):
                H = self.act2(self.hidden2(H))
            logits = self.output(H)
        else:
            logits = self.linear(H)

        return torch.softmax(logits, dim=1)
# ----------------------------------------------------------------------
# 2) Registry & helpers
# ----------------------------------------------------------------------

# method → (builder_fn, loss_fn, is_classification)
_MODEL_REGISTRY: Dict[str,
    Tuple[Callable[[int], nn.Module], Callable[[], nn.Module], bool]
] = {
    "mlp":       (MLPRegressor,  nn.MSELoss,          False),
    "mlpclass":  (MLPClassifier, nn.BCEWithLogitsLoss, True),
    "linreg":    (LinearReg,     nn.MSELoss,          False),
    "logistic":  (LogisticReg,   nn.BCEWithLogitsLoss, True),
}

def get_model(name: str, n_in: int):
    """
    Build a fresh model instance according to *name*.
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {name}")
    builder, *_ = _MODEL_REGISTRY[name]
    return builder(n_in)

def get_loss(name: str):
    """
    Return the matching loss function constructor and a
    boolean flag *is_classification*.
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {name}")
    _, loss_builder, is_clf = _MODEL_REGISTRY[name]
    return loss_builder(), is_clf