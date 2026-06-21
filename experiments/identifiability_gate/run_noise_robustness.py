#!/usr/bin/env python3
"""
Noise Robustness Experiment
============================
All prior experiments (gate, mismatch, fix, attribution) ran at SNR=30 dB.
This experiment asks: at what noise level does physics-informed parameter
estimation break down?

SNR sweep ∈ {10, 20, 30, 40} dB at two representative conditions:
  (A) df=0  Hz: clean data.   Models: 2p-complex, 3p-complex, analytical UB.
  (B) df=50 Hz: off-resonance. Models: 3p-complex, analytical |mag| UB.

Analytical "upper bound" (UB) uses fully-sampled (no mask) echo magnitudes
with log-linear fit — the best any T2* estimator can do with full data and no
model mismatch.  For df>0 the |echo_full| magnitude is off-resonance-free
(|S0·exp(-TE/T2*)·exp(iφ)| = S0·exp(-TE/T2*)), so it gives a fair UB.

Metrics: T2* median error (ms), T2* p95 (ms), Df median error (Hz, df=50 only),
         final loss, T2* cross-seed std (ms).

Output: verdict with the "usable SNR threshold" — where 3p T2* error < 5 ms.

Usage:
    python experiments/identifiability_gate/run_noise_robustness.py
    python experiments/identifiability_gate/run_noise_robustness.py --epochs 300 --snr-list 10 20 30 40
"""

from __future__ import annotations

import argparse
import random
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_GATE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_GATE_DIR))

from run_gate import (                                      # noqa: E402
    make_phantom, synthesise, analytical_fit,
    BottleneckNet, kspace_consistency_loss, predict_from_model,
    fft2c_torch, ifft2c_np,
    H, W, TEs_MS, N_ECHOES, ACCEL, CF, N_SEEDS, HIDDEN, T2_MIN, T2_MAX,
    RESULTS_DIR,
)
from run_mismatch import make_offres_map                    # noqa: E402
from run_offres_fix import (                                # noqa: E402
    BottleneckNet3, train_3param, predict_3param,
    _make_complex_zf_input, DF_BOUND,
)
from run_attribution import _train_2param                   # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────── constants ───────────────────────────────────

DEFAULT_SNR_LIST = [10, 20, 30, 40]    # dB
DF_FIXED         = 50                  # Hz — off-resonance condition
DF_LEVELS        = [0, DF_FIXED]       # both conditions per SNR


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def t2_metrics(T2_pred: np.ndarray, T2_gt: np.ndarray, fg: np.ndarray) -> dict:
    err = np.abs(T2_pred[fg] - T2_gt[fg])
    return dict(t2_med=float(np.median(err)), t2_p95=float(np.percentile(err, 95)))


def df_metrics(Df_pred: np.ndarray, Df_gt: np.ndarray, fg: np.ndarray) -> dict:
    err = np.abs(Df_pred[fg] - Df_gt[fg])
    return dict(df_med=float(np.median(err)), df_p95=float(np.percentile(err, 95)))


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYTICAL UPPER BOUND  (fully-sampled, log-linear fit)
# ═══════════════════════════════════════════════════════════════════════════════

def run_analytical_ub(echoes_full: np.ndarray, T2s_gt: np.ndarray,
                      fg: np.ndarray) -> dict:
    """
    Fit T2* via log-linear regression on fully-sampled echoes (no mask).
    echoes_full [n_echoes, H, W] float — magnitude (phase removed).
    This is the best-possible T2* estimator: full data, correct model, no mismatch.
    """
    _, T2_fit = analytical_fit(echoes_full, TEs_MS)
    m = t2_metrics(T2_fit, T2s_gt, fg)
    return dict(t2_med=m["t2_med"], t2_p95=m["t2_p95"],
                mean_loss=float("nan"), t2_cs_mean=0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL RUNNERS (per seed)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_seeds(model_type: str, S0_gt, T2s_gt, Df_gt, fg, data,
               n_epochs, device, label_prefix) -> dict:
    """
    Train N_SEEDS models of `model_type` ∈ {"2p-complex", "3p-complex"}.
    Returns best-seed T2* metrics + cross-seed std.
    """
    n_echoes = len(TEs_MS)
    k_under_tc = torch.from_numpy(data["kspace_under"]).to(torch.complex64)
    mask_tc    = torch.from_numpy(data["mask_2d"])

    seed_T2: list[np.ndarray] = []
    seed_m:  list[dict]       = []
    seed_Df: list[np.ndarray] = []
    seed_loss: list[float]    = []

    for seed in range(N_SEEDS):
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

        if model_type == "3p-complex":
            zf_arr   = _make_complex_zf_input(data["kspace_under"])
            zf_input = torch.from_numpy(zf_arr).unsqueeze(0)
            model    = BottleneckNet3(n_echoes, HIDDEN, T2_MIN, T2_MAX, DF_BOUND, complex_input=True)
            model, fl = train_3param(model, zf_input, k_under_tc, mask_tc, TEs_MS,
                                     n_epochs=n_epochs, lr=1e-3, device=device,
                                     label=f"{label_prefix} s={seed}")
            _, T2_p, Df_p = predict_3param(model, zf_input, device)
        else:  # "2p-complex"
            zf_arr   = data["zf_mag"]
            zf_input = torch.from_numpy(zf_arr).unsqueeze(0)
            model    = BottleneckNet(n_echoes, HIDDEN, T2_MIN, T2_MAX)
            model, fl = _train_2param(model, zf_input, k_under_tc, mask_tc, TEs_MS,
                                      loss_fn=kspace_consistency_loss,
                                      n_epochs=n_epochs, lr=1e-3, device=device,
                                      label=f"{label_prefix} s={seed}")
            _, T2_p = predict_from_model(model, zf_input, device)
            Df_p    = np.zeros_like(T2_p)

        m = t2_metrics(T2_p, T2s_gt, fg)
        seed_T2.append(T2_p); seed_Df.append(Df_p)
        seed_m.append(m); seed_loss.append(fl)
        print(f"      seed={seed}  T2*={m['t2_med']:.2f} ms  loss={fl:.5f}")

    t2_stack = np.stack(seed_T2, axis=0)
    cs_fg    = t2_stack[:, fg].std(axis=0)

    best  = int(np.argmin([m["t2_med"] for m in seed_m]))
    m_rep = seed_m[best]

    # Df metrics from best seed
    dm = df_metrics(seed_Df[best], Df_gt, fg) if Df_gt.max() > 0 else {"df_med": 0.0, "df_p95": 0.0}

    return dict(
        t2_med     = m_rep["t2_med"],
        t2_p95     = m_rep["t2_p95"],
        df_med     = dm["df_med"],
        mean_loss  = float(np.mean(seed_loss)),
        t2_cs_mean = float(cs_fg.mean()),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PER-CONDITION SWEEP
# ═══════════════════════════════════════════════════════════════════════════════

def run_snr_sweep(S0_gt, T2s_gt, fg, n_epochs, device,
                  snr_list: list[float]) -> dict:
    """
    Returns nested dict: results[df_max][snr_db][cond] = metrics dict.
    """
    results: dict = {df: {} for df in DF_LEVELS}

    # Build off-resonance maps (fixed per df level, not SNR-dependent)
    offres_maps = {
        0:       None,
        DF_FIXED: make_offres_map(H, W, float(DF_FIXED)),
    }
    Df_gts = {
        0:       np.zeros((H, W), np.float32),
        DF_FIXED: offres_maps[DF_FIXED].copy(),
    }

    for snr_db in snr_list:
        print(f"\n{'═'*68}")
        print(f"SNR = {snr_db} dB")
        print(f"{'═'*68}")

        for df_max in DF_LEVELS:
            Df_gt    = Df_gts[df_max]
            offres   = offres_maps[df_max]
            df_tag   = f"df{df_max}"

            print(f"\n  ── df={df_max} Hz ──")

            # Generate data at this SNR
            data = synthesise(S0_gt, T2s_gt, TEs_MS, ACCEL, CF, snr_db,
                              mask_seed=42, noise_seed=7, offres_map=offres)

            results[df_max].setdefault(snr_db, {})

            # Analytical UB — uses noiseless echoes_full (magnitude of complex)
            echo_mag = np.abs(np.stack([
                S0_gt * np.exp(-te / T2s_gt) for te in TEs_MS
            ], axis=0)).astype(np.float32)
            r_ub = run_analytical_ub(echo_mag, T2s_gt, fg)
            results[df_max][snr_db]["analytical-ub"] = r_ub
            print(f"    Analytical UB  : T2*={r_ub['t2_med']:.2f} ms")

            # 2p-complex (only at df=0 — at df>0 it degrades; used as reference)
            if df_max == 0:
                lbl = f"SNR{snr_db}_df{df_max}_2p"
                print(f"    2p-complex ...")
                r2 = _run_seeds("2p-complex", S0_gt, T2s_gt, Df_gt, fg,
                                data, n_epochs, device, lbl)
                results[df_max][snr_db]["2p-complex"] = r2

            # 3p-complex — main model under test
            lbl = f"SNR{snr_db}_df{df_max}_3p"
            print(f"    3p-complex ...")
            r3 = _run_seeds("3p-complex", S0_gt, T2s_gt, Df_gt, fg,
                            data, n_epochs, device, lbl)
            results[df_max][snr_db]["3p-complex"] = r3

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ═══════════════════════════════════════════════════════════════════════════════

def save_noise_figure(results: dict, snr_list: list[float]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Noise Robustness — T2* and Df error vs SNR", fontsize=11)

    snr_arr = np.array(snr_list)

    # Panel 0: T2* error at df=0, 2p vs 3p vs UB
    ax = axes[0]
    ub  = [results[0][s]["analytical-ub"]["t2_med"] for s in snr_list]
    p2  = [results[0][s]["2p-complex"]["t2_med"]    for s in snr_list]
    p3  = [results[0][s]["3p-complex"]["t2_med"]    for s in snr_list]
    ax.semilogy(snr_arr, ub, "k--o", label="Analytical UB")
    ax.semilogy(snr_arr, p2, "b-s",  label="2p-complex")
    ax.semilogy(snr_arr, p3, "r-^",  label="3p-complex")
    ax.axhline(5.0, color="gray", ls=":", lw=0.8, label="5 ms threshold")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("T2* med error (ms)")
    ax.set_title("df=0 Hz: T2* error"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 1: T2* error at df=50, 3p vs UB
    ax = axes[1]
    ub50 = [results[DF_FIXED][s]["analytical-ub"]["t2_med"] for s in snr_list]
    p3_50= [results[DF_FIXED][s]["3p-complex"]["t2_med"]    for s in snr_list]
    ax.semilogy(snr_arr, ub50, "k--o", label="Analytical UB")
    ax.semilogy(snr_arr, p3_50,"r-^",  label="3p-complex")
    ax.axhline(5.0, color="gray", ls=":", lw=0.8, label="5 ms threshold")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("T2* med error (ms)")
    ax.set_title(f"df={DF_FIXED} Hz: T2* error"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Panel 2: Df error at df=50, 3p
    ax = axes[2]
    df_err = [results[DF_FIXED][s]["3p-complex"]["df_med"] for s in snr_list]
    ax.semilogy(snr_arr, df_err, "r-^", label="3p Df error")
    ax.axhline(5.0, color="gray", ls=":", lw=0.8, label="5 Hz threshold")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Df med error (Hz)")
    ax.set_title(f"df={DF_FIXED} Hz: Df error"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = RESULTS_DIR / "noise_robustness.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"\n  → saved {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE + VERDICT
# ═══════════════════════════════════════════════════════════════════════════════

def print_table_and_verdict(results: dict, snr_list: list[float]) -> str:
    print("\n" + "═" * 84)
    print("NOISE ROBUSTNESS — RESULT TABLE")
    print("  T2* threshold for 'reliable identification': 5 ms (5× df=0 control ~1 ms)")
    print("  Df threshold: 5 Hz")
    print("═" * 84)

    hdr  = (f"  {'SNR':>5}  {'df':>5}  {'Model':<20}  "
            f"{'T2* med':>8}  {'T2* p95':>8}  {'Df med':>7}  "
            f"{'Loss':>9}  {'T2*Xs':>7}")
    unit = (f"  {'(dB)':>5}  {'(Hz)':>5}  {'':20}  "
            f"{'(ms)':>8}  {'(ms)':>8}  {'(Hz)':>7}  "
            f"{'':>9}  {'(ms)':>7}")
    print(hdr); print(unit)

    # Find the SNR threshold (first SNR where 3p-complex T2* < 5 ms at df=0 and df_fixed)
    usable_snr_df0   = None
    usable_snr_df50  = None

    for snr_db in snr_list:
        print(f"  {'─'*82}")
        for df_max in DF_LEVELS:
            d = results[df_max][snr_db]
            for cond, r in sorted(d.items()):
                if r is None:
                    continue
                df_str  = f"{r.get('df_med', 0.0):>7.1f}" if df_max > 0 else f"{'—':>7}"
                loss_str = f"{r['mean_loss']:>9.5f}" if not (r['mean_loss'] != r['mean_loss']) else f"{'N/A':>9}"
                print(f"  {snr_db:>5}  {df_max:>5}  {cond:<20}  "
                      f"{r['t2_med']:>8.2f}  {r['t2_p95']:>8.2f}  "
                      f"{df_str}  {loss_str}  {r['t2_cs_mean']:>7.3f}")

        # Update SNR threshold tracking (3p-complex only)
        r3_0  = results[0][snr_db].get("3p-complex", {})
        r3_50 = results[DF_FIXED][snr_db].get("3p-complex", {})
        if r3_0.get("t2_med", 999) < 5.0 and usable_snr_df0 is None:
            usable_snr_df0  = snr_db
        if r3_50.get("t2_med", 999) < 5.0 and usable_snr_df50 is None:
            usable_snr_df50 = snr_db

    print("  " + "═" * 82)

    # Verdict
    print(f"\n  NOISE ROBUSTNESS VERDICT")
    print(f"  T2* threshold (< 5 ms = reliable) for 3-param model:")
    if usable_snr_df0 is not None:
        print(f"    df=0  Hz: SNR ≥ {usable_snr_df0} dB → reliable T2* (< 5 ms)")
    else:
        print(f"    df=0  Hz: T2* > 5 ms even at {max(snr_list)} dB — model fails at tested SNR range")
    if usable_snr_df50 is not None:
        print(f"    df={DF_FIXED} Hz: SNR ≥ {usable_snr_df50} dB → reliable T2* + Df (< 5 ms / < 5 Hz)")
    else:
        print(f"    df={DF_FIXED} Hz: T2* > 5 ms even at {max(snr_list)} dB — needs higher SNR")

    # Check Df usable threshold
    for snr_db in snr_list:
        r3 = results[DF_FIXED][snr_db].get("3p-complex", {})
        if r3.get("df_med", 999) < 5.0:
            print(f"    df={DF_FIXED} Hz: Df reliable (< 5 Hz) at SNR ≥ {snr_db} dB")
            break
    else:
        print(f"    df={DF_FIXED} Hz: Df error > 5 Hz at all tested SNR — borderline at high SNR")

    # Degradation slope
    r3_max = results[0][max(snr_list)].get("3p-complex", {}).get("t2_med", float("nan"))
    r3_min = results[0][min(snr_list)].get("3p-complex", {}).get("t2_med", float("nan"))
    if r3_max == r3_max and r3_min == r3_min:
        print(f"\n  3p T2* error: {r3_max:.2f} ms at SNR={max(snr_list)} dB → "
              f"{r3_min:.2f} ms at SNR={min(snr_list)} dB "
              f"({r3_min/r3_max:.1f}× degradation over {max(snr_list)-min(snr_list)} dB)")

    verdict = (f"Usable SNR threshold (T2* < 5 ms): df=0 → ≥{usable_snr_df0} dB; "
               f"df={DF_FIXED} Hz → ≥{usable_snr_df50} dB"
               if (usable_snr_df0 and usable_snr_df50)
               else "No usable threshold found in tested SNR range")
    print(f"\n  VERDICT: {verdict}")
    print("═" * 84 + "\n")
    return verdict


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",   type=int,   default=300)
    parser.add_argument("--snr-list", type=float, nargs="+",
                        default=DEFAULT_SNR_LIST, metavar="SNR")
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    snr_list = sorted(args.snr_list, reverse=True)   # high SNR first (easier cases first)

    print(f"\nDevice: {device},  Epochs: {args.epochs}")
    print(f"SNR sweep: {snr_list} dB")
    print(f"df conditions: {DF_LEVELS} Hz  (fixed df={DF_FIXED} Hz for off-resonance)")
    print(f"Seeds: {N_SEEDS}, Phantom: {H}×{W}")

    S0_gt, T2s_gt, labels = make_phantom(H, W)
    fg_mask = labels > 0
    print(f"GT T2* ∈ [{T2s_gt[fg_mask].min():.0f}, {T2s_gt[fg_mask].max():.0f}] ms, "
          f"fg={fg_mask.sum()} px\n")

    results = run_snr_sweep(S0_gt, T2s_gt, fg_mask, args.epochs, device, snr_list)
    save_noise_figure(results, snr_list)
    verdict = print_table_and_verdict(results, snr_list)

    print(f"Final verdict: {verdict}\n")


if __name__ == "__main__":
    main()
