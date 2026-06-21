#!/usr/bin/env python3
"""
Attribution Experiment: "COMPLETION NECESSARY" vs "IT WAS THE BUG"
====================================================================
Isolates the cause of the off-resonance BRITTLE failure reported in
run_mismatch.py (T2*~41 ms at df=20 Hz, 51× jump).

run_offres_fix.py changed TWO things simultaneously:
  a) Fixed a 1000× phase bug in synthesise() (te_ms used as if te_s)
  b) Added a third bottleneck parameter (Df)

This experiment uses IDENTICAL bug-fixed data across all conditions and varies
ONLY the model and loss to isolate which change actually matters.

CONDITIONS (same phantom, same seeds, SAME kspace data per df level):
  (1) 2p-complex  : 2-param model (S0, T2*), complex k-space DC loss    ← CONTROL
  (2) 3p-complex  : 3-param model (S0, T2*, Df), complex DC loss        ← COMPLETION FIX
  (3) 2p-mag      : 2-param model, phase-invariant magnitude DC loss     ← MECHANISM PROBE
  (R) 2p-complex on BUGGY data (offres_map×1000) — regression anchor
                   expected to reproduce ~41 ms                          ← BUG CONFIRM

VERDICT LOGIC:
  COMPLETION NECESSARY: cond(1) T2* meaningfully degraded at physical df (clearly
    worse than cond(2)) — the phase DOF is genuinely needed at physical levels.
  IT WAS THE BUG: cond(1) stays near df=0 control at physical df — off-resonance
    at physical levels was never a real failure; the 51× jump was purely the bug.
  MECHANISM: if cond(1) breaks but cond(3) recovers — the failure is the complex
    loss demanding unmodeled phase, not lost amplitude/T2* information; TWO valid
    fixes exist (model the phase via Df, OR use a phase-insensitive loss).

Usage:
    python experiments/identifiability_gate/run_attribution.py
    python experiments/identifiability_gate/run_attribution.py --epochs 400 --no-regression
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
import torch.nn as nn

_GATE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_GATE_DIR))

from run_gate import (                                          # noqa: E402
    make_phantom, synthesise,
    BottleneckNet, kspace_consistency_loss, predict_from_model,
    fft2c_torch,
    H, W, TEs_MS, N_ECHOES, ACCEL, CF, N_SEEDS, HIDDEN, T2_MIN, T2_MAX,
    RESULTS_DIR,
)
from run_mismatch import make_offres_map                        # noqa: E402
from run_offres_fix import (                                    # noqa: E402
    BottleneckNet3, train_3param, predict_3param,
    _make_complex_zf_input, DF_BOUND,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────── constants ───────────────────────────────────

SNR_DB        = 30.0
OFFRES_LEVELS = [0, 20, 50, 100]   # Hz

COND_LABELS = {
    "2p-complex"    : "(1) 2p / complex DC",
    "3p-complex"    : "(2) 3p / complex DC",
    "2p-mag"        : "(3) 2p / mag DC   ",
    "2p-complex-bug": "(R) 2p / complex DC / BUGGY data",
}


# ═══════════════════════════════════════════════════════════════════════════════
# MAGNITUDE-ONLY DC LOSS
# ═══════════════════════════════════════════════════════════════════════════════

def kspace_mag_loss(s0: torch.Tensor, t2: torch.Tensor,
                    k_under_tc: torch.Tensor, mask_tc: torch.Tensor,
                    TEs_ms: list[float]) -> torch.Tensor:
    """
    Normalised k-space DC loss comparing k-space MAGNITUDES — phase-invariant.

    If cond(1) [complex loss] fails but this variant succeeds: the mechanism is the
    complex loss penalising unmodeled phase, not a loss of T2* amplitude information.
    Both models produce real echoes; only the loss comparison differs.
    """
    s0_ = s0[0, 0]; t2_ = t2[0, 0]
    total = torch.zeros(1, device=s0.device)
    norm  = torch.zeros(1, device=s0.device)
    for e, te in enumerate(TEs_ms):
        echo_hat = s0_ * torch.exp(-te / t2_)          # [H, W] real amplitude
        k_hat    = fft2c_torch(echo_hat)                # [H, W] complex (conj-symmetric)
        k_meas   = k_under_tc[e]
        res      = mask_tc * (k_hat.abs() - k_meas.abs())   # magnitude diff only
        total    = total + (res ** 2).sum()
        norm     = norm  + (mask_tc * k_meas.abs() ** 2).sum()
    return total / (norm + 1e-8)


# ═══════════════════════════════════════════════════════════════════════════════
# GENERIC 2-PARAM TRAINING (accepts caller-supplied loss fn, avoids touching run_gate.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _train_2param(model: BottleneckNet,
                  zf_input: torch.Tensor,
                  k_under_tc: torch.Tensor,
                  mask_tc: torch.Tensor,
                  TEs_ms: list[float],
                  loss_fn,
                  n_epochs: int,
                  lr: float,
                  device: torch.device,
                  label: str = "") -> tuple[BottleneckNet, float]:
    model      = model.to(device)
    zf_input   = zf_input.to(device)
    k_under_tc = k_under_tc.to(device)
    mask_tc    = mask_tc.to(device)

    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=lr * 0.01)

    final_loss = float("nan")
    log_every  = max(1, n_epochs // 5)

    for ep in range(1, n_epochs + 1):
        model.train()
        opt.zero_grad()
        s0_hat, t2_hat = model(zf_input)
        loss = loss_fn(s0_hat, t2_hat, k_under_tc, mask_tc, TEs_ms)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if ep == n_epochs:
            final_loss = float(loss.item())
        if ep % log_every == 0:
            print(f"    [{label}] ep {ep:4d}/{n_epochs}  loss={loss.item():.5f}")

    return model, final_loss


# ═══════════════════════════════════════════════════════════════════════════════
# DATA GENERATION — single entry point with bug-fix toggle
# ═══════════════════════════════════════════════════════════════════════════════

def make_data(S0_gt: np.ndarray, T2s_gt: np.ndarray,
              df_max: int, TEs_ms: list[float] = TEs_MS,
              bugfix_phase: bool = True) -> dict:
    """
    bugfix_phase=True  → physically correct te/1000 in phase exponent.
    bugfix_phase=False → reproduces pre-fix bug: equivalent to offres_map×1000
                         so that the fixed synthesise() computes te/1000 × 1000 = te.
    Both paths call the SAME synthesise() with identical arguments except offres_map scale.
    """
    if df_max > 0:
        offres_map = make_offres_map(H, W, float(df_max))
        if not bugfix_phase:
            offres_map = offres_map * 1000.0   # undo /1000 fix → reproduce old bug
    else:
        offres_map = None
    return synthesise(S0_gt, T2s_gt, TEs_ms, ACCEL, CF, SNR_DB,
                      mask_seed=42, noise_seed=7, offres_map=offres_map)


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def t2_metrics(T2_pred: np.ndarray, T2_gt: np.ndarray, fg: np.ndarray) -> dict:
    err = np.abs(T2_pred[fg] - T2_gt[fg])
    return dict(
        t2_med=float(np.median(err)),
        t2_p95=float(np.percentile(err, 95)),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE CONDITION RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_condition(cond: str, tag: str,
                  S0_gt: np.ndarray, T2s_gt: np.ndarray,
                  fg_mask: np.ndarray, data: dict,
                  n_epochs: int, device: torch.device,
                  TEs_ms: list[float] = TEs_MS) -> dict:
    """
    Run N_SEEDS models for one condition on pre-built data.
    Uses SAME data dict (kspace_under, mask_2d, zf_mag) across conditions.
    Returns metrics keyed by best-seed T2* (consistent with prior experiments).
    """
    n_echoes = len(TEs_ms)
    k_under_tc = torch.from_numpy(data["kspace_under"]).to(torch.complex64)
    mask_tc    = torch.from_numpy(data["mask_2d"])

    seed_T2: list[np.ndarray] = []
    seed_m: list[dict]        = []
    seed_loss: list[float]    = []

    for seed in range(N_SEEDS):
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

        if cond == "3p-complex":
            zf_arr   = _make_complex_zf_input(data["kspace_under"])  # [2*n_echoes, H, W]
            zf_input = torch.from_numpy(zf_arr).unsqueeze(0)
            model    = BottleneckNet3(n_echoes, HIDDEN, T2_MIN, T2_MAX, DF_BOUND, complex_input=True)
            model, fl = train_3param(model, zf_input, k_under_tc, mask_tc, TEs_ms,
                                     n_epochs=n_epochs, lr=1e-3, device=device,
                                     label=f"{tag} s={seed}")
            _, T2_p, _ = predict_3param(model, zf_input, device)
        else:
            zf_arr   = data["zf_mag"]                                # [n_echoes, H, W]
            zf_input = torch.from_numpy(zf_arr).unsqueeze(0)
            model    = BottleneckNet(n_echoes, HIDDEN, T2_MIN, T2_MAX)
            loss_fn  = kspace_mag_loss if cond == "2p-mag" else kspace_consistency_loss
            model, fl = _train_2param(model, zf_input, k_under_tc, mask_tc, TEs_ms,
                                      loss_fn=loss_fn, n_epochs=n_epochs, lr=1e-3,
                                      device=device, label=f"{tag} s={seed}")
            _, T2_p = predict_from_model(model, zf_input, device)

        m = t2_metrics(T2_p, T2s_gt, fg_mask)
        seed_T2.append(T2_p); seed_m.append(m); seed_loss.append(fl)
        print(f"      seed={seed}  T2*_med={m['t2_med']:.2f} ms  loss={fl:.5f}")

    # Cross-seed T2* std (identifiability)
    t2_stack  = np.stack(seed_T2, axis=0)
    cs_fg     = t2_stack[:, fg_mask].std(axis=0)

    # Best-seed metric for the table (same convention as run_gate / run_offres_fix)
    best  = int(np.argmin([m["t2_med"] for m in seed_m]))
    m_rep = seed_m[best]

    return dict(
        t2_med     = m_rep["t2_med"],
        t2_p95     = m_rep["t2_p95"],
        mean_loss  = float(np.mean(seed_loss)),
        t2_cs_mean = float(cs_fg.mean()),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# VERDICT + TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def print_table_and_verdict(table: dict[tuple[int, str], dict],
                             run_regression: bool) -> str:
    """
    table keys: (df_max, cond)
    cond ∈ {"2p-complex", "3p-complex", "2p-mag", "2p-complex-bug"}
    """
    print("\n" + "═" * 82)
    print("ATTRIBUTION EXPERIMENT — COMPARISON TABLE")
    print("  Bug-fixed data: conditions (1)(2)(3).  Buggy data: regression (R).")
    print("═" * 82)

    hdr  = f"  {'df':>6}  {'Condition':<28}  {'T2* med':>8}  {'T2* p95':>8}  {'Loss':>9}  {'T2* Xseed':>10}"
    unit = f"  {'Hz':>6}  {'':28}  {'(ms)':>8}  {'(ms)':>8}  {'(norm)':>9}  {'(ms μ)':>10}"
    print(hdr); print(unit); print("  " + "─" * 78)

    def row(df_max, cond):
        r = table.get((df_max, cond))
        if r is None:
            return
        print(f"  {df_max:>6}  {COND_LABELS[cond]:<28}  "
              f"{r['t2_med']:>8.2f}  {r['t2_p95']:>8.2f}  "
              f"{r['mean_loss']:>9.5f}  {r['t2_cs_mean']:>10.3f}")

    all_conds = ["2p-complex", "3p-complex", "2p-mag"]
    for df_max in OFFRES_LEVELS:
        print(f"  {'─'*78}")
        for cond in all_conds:
            row(df_max, cond)
        if run_regression and df_max > 0:
            row(df_max, "2p-complex-bug")

    print("  " + "═" * 78)

    # ── attribution verdicts ──────────────────────────────────────────────────
    # Threshold: T2* med > 5 ms = "meaningfully degraded" for the attribution test.
    # Rationale: 2-param df=0 control is ~1 ms; 5× = 5 ms is a clear signal,
    # distinct from run-to-run noise and from the 3-param's ~1-3 ms range.
    DEGR_THRESH = 5.0   # ms

    r1_0  = table.get((0,   "2p-complex"), {})
    r1_20 = table.get((20,  "2p-complex"), {})
    r1_50 = table.get((50,  "2p-complex"), {})
    r1_100= table.get((100, "2p-complex"), {})
    r3_20 = table.get((20,  "2p-mag"),     {})
    r3_50 = table.get((50,  "2p-mag"),     {})
    r_reg = table.get((20,  "2p-complex-bug"), {})

    ctrl  = r1_0.get("t2_med", 0.0)

    cond1_ok_20  = r1_20.get("t2_med", 999) < DEGR_THRESH
    cond1_ok_50  = r1_50.get("t2_med", 999) < DEGR_THRESH
    cond3_ok_20  = r3_20.get("t2_med", 999) < DEGR_THRESH
    cond3_ok_50  = r3_50.get("t2_med", 999) < DEGR_THRESH

    print(f"\n  ATTRIBUTION ANALYSIS")
    print(f"  Degradation threshold: T2* med > {DEGR_THRESH:.0f} ms = meaningful degradation")
    print(f"  df=0 control (2p-complex): T2* med = {ctrl:.2f} ms\n")

    print(f"  df=20 Hz: 2p-complex  T2* = {r1_20.get('t2_med',0):.2f} ms "
          f"→ {'OK (< {:.0f} ms)'.format(DEGR_THRESH) if cond1_ok_20 else 'DEGRADED (≥ {:.0f} ms)'.format(DEGR_THRESH)}")
    print(f"  df=20 Hz: 2p-mag      T2* = {r3_20.get('t2_med',0):.2f} ms "
          f"→ {'OK' if cond3_ok_20 else 'DEGRADED'}")
    print(f"  df=50 Hz: 2p-complex  T2* = {r1_50.get('t2_med',0):.2f} ms "
          f"→ {'OK' if cond1_ok_50 else 'DEGRADED'}")
    print(f"  df=50 Hz: 2p-mag      T2* = {r3_50.get('t2_med',0):.2f} ms "
          f"→ {'OK' if cond3_ok_50 else 'DEGRADED'}")

    if run_regression:
        reg_val = r_reg.get("t2_med", 0.0)
        print(f"\n  REGRESSION (buggy data, df=20 Hz): T2* = {reg_val:.2f} ms "
              f"(target ≈ 41 ms — {'CONFIRMED' if reg_val > 30.0 else 'NOT reproduced'})")

    print()
    if cond1_ok_20 and cond1_ok_50:
        verdict = (
            "IT WAS THE BUG — 2-param complex-loss model (cond 1) recovers T2* "
            "correctly at physical df levels (≤50 Hz). The 51× failure in "
            "run_mismatch.py was entirely due to the 1000× phase bug, not an "
            "inherent incompatibility between the mono-exp model and off-resonance. "
            "Df DOF is a useful field-map bonus but NOT required for T2* recovery."
        )
    else:
        # cond(1) degrades — check mechanism
        if cond3_ok_20:
            verdict = (
                "COMPLETION NECESSARY + MECHANISM IDENTIFIED — 2-param complex-loss "
                "(cond 1) shows degraded T2* at physical df, but 2-param magnitude-loss "
                "(cond 3) recovers correctly. The failure is NOT lost T2* information: "
                "amplitude decay is still encoded in the k-space magnitudes. The failure "
                "is the complex DC loss demanding a phase the mono-exp model cannot produce, "
                "biasing T2* to compensate. TWO valid fixes exist: (a) add Df DOF "
                "[cond 2], or (b) use a phase-insensitive loss [cond 3]."
            )
        else:
            verdict = (
                "COMPLETION NECESSARY (information loss) — both cond(1) complex and "
                "cond(3) magnitude loss show T2* degradation. The phase modulation "
                "at these df levels distorts the k-space magnitude spectrum itself, "
                "so T2* amplitude information is corrupted regardless of loss function. "
                "Adding Df DOF [cond 2] is the only clean fix."
            )

    # Wrap at 80 chars
    import textwrap
    print("  VERDICT:")
    for line in textwrap.wrap(verdict, width=76):
        print(f"    {line}")
    print("═" * 82 + "\n")
    return verdict


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",        type=int,  default=400)
    parser.add_argument("--no-regression", action="store_true",
                        help="Skip the buggy-data regression anchor")
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"\nDevice: {device},  Epochs: {args.epochs},  SNR: {SNR_DB} dB")
    print(f"Seeds: {N_SEEDS},  Phantom: {H}×{W}")
    print(f"Conditions: {list(COND_LABELS.keys())[:3]} + {'regression' if not args.no_regression else 'NO regression'}")

    # ── phantom — shared across ALL conditions and df levels ──────────────────
    S0_gt, T2s_gt, labels = make_phantom(H, W)
    fg_mask = labels > 0
    print(f"GT T2* range: [{T2s_gt[fg_mask].min():.0f}, {T2s_gt[fg_mask].max():.0f}] ms, "
          f"fg pixels: {fg_mask.sum()}\n")

    table: dict[tuple[int, str], dict] = {}

    for df_max in OFFRES_LEVELS:
        print(f"\n{'═'*72}")
        print(f"df_max = {df_max} Hz")
        print(f"{'═'*72}")

        # ── single bug-fixed data instance — SAME for all 3 conditions ────────
        data_fixed = make_data(S0_gt, T2s_gt, df_max, TEs_MS, bugfix_phase=True)
        max_phase  = 2 * 3.14159 * df_max * TEs_MS[-1] / 1000.0
        print(f"  Max phase @ TE={TEs_MS[-1]} ms: {max_phase:.2f} rad "
              f"= {max_phase/(2*3.14159):.2f} cycles\n")

        for cond in ["2p-complex", "3p-complex", "2p-mag"]:
            print(f"  ── {COND_LABELS[cond]} ──")
            r = run_condition(cond, f"df{df_max}_{cond}",
                              S0_gt, T2s_gt, fg_mask,
                              data_fixed, args.epochs, device)
            table[(df_max, cond)] = r
            print()

        # ── regression anchor on BUGGY data (skip df=0 — identical to fixed) ──
        if not args.no_regression and df_max > 0:
            print(f"  ── {COND_LABELS['2p-complex-bug']} ──")
            data_buggy = make_data(S0_gt, T2s_gt, df_max, TEs_MS, bugfix_phase=False)
            r = run_condition("2p-complex", f"df{df_max}_bug",
                              S0_gt, T2s_gt, fg_mask,
                              data_buggy, args.epochs, device)
            table[(df_max, "2p-complex-bug")] = r
            print()

    print_table_and_verdict(table, run_regression=not args.no_regression)


if __name__ == "__main__":
    main()
