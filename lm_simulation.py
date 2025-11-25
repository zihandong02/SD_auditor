"""
main.py
-------
Run the MCAR mono-debias experiment (Torch).

Usage
-----
# auto device (GPU if available), default settings
python main.py

# force CPU, 5 repetitions, tau = 2,3,5
python main.py --device cpu --reps 100 --tau_vals 2,3,5
"""

# ── stdlib ────────────────────────────────────────────────────────────
import argparse, datetime, os, sys
from pathlib import Path
from cProfile import Profile
from pstats  import Stats, SortKey
from typing  import List

# ── third‑party ───────────────────────────────────────────────────────
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
import pandas as pd
# ── internal packages ────────────────────────────────────────────────
sys.path.append(os.path.abspath(".."))  # adjust if needed

from src.utils      import set_global_seed, get_device, dump_run_simple
from src.lm_mono_debias import lm_fix_alpha, lm_change_alpha_every_iter, lm_mcar_extended  # Algorithm‑1 wrapper

# =====================================================================
# CLI
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("MCAR mono-debias experiment (Torch, single-process)")

    # ---------- device & RNG ----------
    parser.add_argument(
        "--device", default="auto",
        help="'auto' = src.utils.get_device(); otherwise pass 'cpu', 'cuda', 'cuda:1', …",
    )
    parser.add_argument("--seed", default=42, type=int)

    # ---------- distributed ----------
    parser.add_argument("--distributed", action="store_true",
                        help="Enable torch.distributed (use with torchrun or multiple GPUs)")

    # ---------- batch sizes & repetitions ----------
    parser.add_argument("--n1",   default=2000,  type=int)
    parser.add_argument("--n2",   default=20000, type=int)
    parser.add_argument("--reps", default=3,    type=int)

    # ---------- dimensions ----------
    parser.add_argument("--d_x",  default=5, type=int)
    parser.add_argument("--d_u1", default=5, type=int)
    parser.add_argument("--d_u2", default=5, type=int)

    # ---------- noise ----------
    parser.add_argument("--sigma_eps", default=2.0, type=float)

    # ---------- MCAR / CI parameters ----------
    parser.add_argument("--alpha_level", default=0.10, type=float)
    parser.add_argument("--tau_vals",    default="3",
                        help="comma‑separated list, e.g. '3,4,5'")
    parser.add_argument("--c",           default=10.0,  type=float)
    parser.add_argument("--alpha_init",  default="1.0,0.0,0.0")

    return parser.parse_args()

# =====================================================================
# experiment runner (works in single‑ or multi‑process mode)
# =====================================================================

def run_experiment(args: argparse.Namespace, rank: int = 0, world_size: int = 1):
    # ---------- device selection ----------
    if args.device == "auto":
        device = get_device(rank if torch.cuda.is_available() else None)
    else:
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device)

    if rank == 0:
        print(f"[INFO] using device: {device}; rank {rank}/{world_size}")

    set_global_seed(args.seed + rank)  # different seed per rank for diversity

    # ---------- ground‑truth parameters ----------
    theta_star = torch.arange(1, args.d_x  + 1, device=device, dtype=torch.float32) * 0.2
    beta1_star = torch.arange(1, args.d_u1 + 1, device=device, dtype=torch.float32) * 0.2
    beta2_star = torch.arange(1, args.d_u2 + 1, device=device, dtype=torch.float32)
    alpha_init = torch.tensor(
        [float(x) for x in args.alpha_init.split(",")],
        device=device,
    )
    tau_values: List[float] = [float(t) for t in args.tau_vals.split(",")]

    # partition τ values across ranks -------------------------
    tau_slice = tau_values[rank::world_size]
    print(f"[rank {rank}] gpu={device} will run tau_slice={tau_slice}", flush=True)
    rows_local = []
    for tau in tau_slice:
        res = lm_fix_alpha(
            n1=args.n1, n2=args.n2, reps=args.reps,
            d_x=args.d_x, d_u1=args.d_u1, d_u2=args.d_u2,
            theta_star=theta_star,
            beta1_star=beta1_star,
            beta2_star=beta2_star,
            sigma_eps=args.sigma_eps,
            alpha_level=args.alpha_level,
            tau=tau, c=args.c,
            alpha_init=alpha_init,
            seed=args.seed,  # offset to keep randomness disjoint
        )
        res["tau"] = tau
        rows_local.append(res)
        if rank == 0:
            print(f"[rank 0] finished τ={tau}")

    # ---------------- gather results -------------------------
    if world_size > 1:
        gathered: List[List[dict]] = [None for _ in range(world_size)]  # type: ignore
        dist.all_gather_object(gathered, rows_local)
        if rank == 0:
            rows = [item for sublist in gathered for item in sublist]
    else:
        rows = rows_local

    # ---------------- rank‑0: save & plot ---------------------
    if rank == 0:
        df = pd.DataFrame(rows).set_index("tau").round(4).sort_index()
        print(df)

        # save summary + params --------------------------------
        params = {
            "cmd": " ".join(sys.argv),
            "args": vars(args),
            "timestamp": datetime.datetime.now().isoformat(),
        }
        out_dir = dump_run_simple(df=df, params=params)
        print(f"[INFO] results saved to {out_dir}")

# =====================================================================
# entry‑point (with optional distributed / profiler)
# =====================================================================

def main():
    args = parse_args()

    # ------------- distributed initialisation -------------
    distributed = args.distributed or (
        "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    )
    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank) if torch.cuda.is_available() else None
    else:
        rank = 0
        world_size = 1

    prof = Profile(); prof.enable()
    run_experiment(args, rank=rank, world_size=world_size)
    prof.disable()

    if rank == 0:
        Stats(prof).strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(25)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()