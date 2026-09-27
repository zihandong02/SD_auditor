# ── standard library ───────────────────────────────────────────────────
import datetime
import json
import os
import random
from typing import Any, Dict, TYPE_CHECKING

# ── third-party libraries ──────────────────────────────────────────────
import torch

try:
    import numpy as np                              # optional dependency
    HAS_NUMPY: bool = True
except ImportError:                                # NumPy not installed
    np = None                                       # type: ignore[assignment]
    HAS_NUMPY = False

# For static type checkers: let them know 'np' exists
if TYPE_CHECKING:                                   # noqa: F401
    import numpy as np  # pragma: no cover

# ----------------------------------------------------------------------
SEED: int = 2                     # Global default seed (can be overwritten)

# ----------------------------------------------------------------------
def set_global_seed(seed: int | None = None) -> None:
    """
    Set the global random seed for Python, Torch (CPU & CUDA) and
    optionally NumPy.  Also configures cuDNN for deterministic behaviour.
    """
    global SEED
    seed = SEED if seed is None else seed
    SEED = seed

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if np is not None:
        np.random.seed(seed)

# ----------------------------------------------------------------------
def get_device(device_id: int | None = None) -> torch.device:
    """
    Return the preferred computation device.
    If `device_id` is provided and CUDA is available, it will select
    'cuda:<device_id>'. Otherwise, it defaults to 'cuda' if available,
    or 'cpu'.
    """
    if torch.cuda.is_available():
        if device_id is not None:
            return torch.device(f"cuda:{device_id}")
        return torch.device("cuda")
    return torch.device("cpu")

# ----------------------------------------------------------------------
# ------------------------------- I/O ----------------------------------
# ----------------------------------------------------------------------
def _to_json(obj: Any):
    """
    Fallback serializer for json.dump that understands Torch / NumPy
    objects and converts them to native Python types.
    """
    # Torch tensors & scalars
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (torch.int8, torch.int16, torch.int32, torch.int64,
                        torch.uint8, torch.bool)):
        return int(obj)
    if isinstance(obj, (torch.float16, torch.float32, torch.float64)):
        return float(obj)

    # NumPy arrays & scalars
    if np is not None:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int_)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float_)):
            return float(obj)

    # Fallback: stringify everything else
    return str(obj)

# ----------------------------------------------------------------------
def dump_run_simple(
    *,
    df,                           # pandas.DataFrame
    params: Dict[str, Any],
    base_dir: str = "results",
    prefix: str = "",
) -> str:
    """
    Save a single experiment run (no figures):
        results/<prefix>_<timestamp>/
            summary.csv   – machine-readable table
            summary.txt   – pretty-printed table
            params.json   – full hyper-parameter dictionary
    Returns the absolute path to the created folder.
    """
    import pandas as pd           # Lazy import to avoid hard dependency
    assert isinstance(df, pd.DataFrame), "`df` must be a pandas.DataFrame"

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir  = os.path.join(base_dir, f"{prefix}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    df.to_csv(os.path.join(out_dir, "summary.csv"), index=True)
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f_txt:
        f_txt.write(df.to_string())

    with open(os.path.join(out_dir, "params.json"), "w", encoding="utf-8") as f_json:
        json.dump(params, f_json, indent=4, default=_to_json)

    return os.path.abspath(out_dir)