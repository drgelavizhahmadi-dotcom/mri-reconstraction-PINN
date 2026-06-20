#!/usr/bin/env python3
"""
Model-Mismatch Hardening of the Identifiability Gate
=====================================================
Tests whether identifiability degrades GRACEFULLY (bounded, biased-but-unique)
or BRITTLY (error blows up, or solution becomes non-unique) under two graded
model mismatches:

  Sweep 1 — Bi-exponential ground truth:
    S(TE) = S0 * [f*exp(-TE/T2s_short) + (1-f)*exp(-TE/T2s)]
    T2s_short = 10 ms, T2s_long = per-pixel tissue T2* from phantom
    f ∈ {0.0, 0.1, 0.2, 0.3}   (0.0 = inverse-crime control, regression check)

  Sweep 2 — B0 off-resonance complex echoes:
    echo(TE) = S0*exp(-TE/T2*) * exp(i·2π·df(x,y)·TE)
    df_max ∈ {0, 20, 50, 100} Hz   (0 Hz = control)

NETWORK AND LOSS ARE UNCHANGED (mono-exp forward model).
REFERENCE: T2*_eff = analytical mono-exp fit to clean fully-sampled signal.
  → "Method 1" always = 0 error by definition (it IS T2*_eff).
  → All errors measure what the network + undersampling ADD on top of the
    unavoidable model-class approximation limit.

VERDICT CRITERIA (per sweep):
  GRACEFUL  — T2* error grows smoothly, stays ≤ 10 ms at highest mismatch,
               AND cross-seed std < 3 ms at every level.
  BRITTLE   — error jumps > 3× from level 0 to level 1,
               OR cross-seed std exceeds 5 ms at any level.

Usage:
    python experiments/identifiability_gate/run_mismatch.py
    python experiments/identifiability_gate/run_mismatch.py --epochs 400
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

# ── import everything from the existing gate (network, training, evaluation) ──
_GATE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_GATE_DIR))

from run_gate import (                            # noqa: E402
    make_phantom, synthesise, BottleneckNet, train,
    analytical_fit, predict_from_model, compute_metrics,
    fft2c_np, ifft2c_np,
    H, W, TEs_MS, N_ECHOES, ACCEL, CF, N_SEEDS, HIDDEN, T2_MIN, T2_MAX,
    RESULTS_DIR,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────── mismatch sweeps config ──────────────────────────

BIEXP_F_LEVELS    = [0.0, 0.1, 0.2, 0.3]
BIEXP_T2S_SHORT   = 10.0   # ms
OFFRES_DF_LEVELS  = [0, 20, 50, 100]  # Hz

# ═══════════════════════════════════════════════════════════════════════════════
# OFF-RESONANCE MAP
# ═══════════════════════════════════════════════════════════════════════════════

def make_offres_map(H: int, W: int, df_max_hz: float) -> np.ndarray:
    """
    Smooth spatially-varying B0 off-resonance map scaled to ±df_max_hz.
    Uses a low-order polynomial so it resembles realistic scanner inhomogeneity.
    Returns [H, W] float32 in Hz.
    """
    yy, xx = np.mgrid[-1:1:H * 1j, -1:1:W * 1j]
    # Smooth gradient + low-frequency curvature (no high-frequency artefacts)
    field = 0.55 * xx + 0.30 * yy + 0.20 * xx * yy - 0.15 * (xx ** 2 - yy ** 2)
    field /= np.max(np.abs(field)) + 1e-8       # normalise to [-1, 1]
    return (df_max_hz * field).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE HELPERS (mismatch-specific)
# ═══════════════════════════════════════════════════════════════════════════════

def save_mismatch_figure(tag: str,
                         T2s_eff: np.ndarray,
                         T2s_pred: np.ndarray,
                         cross_seed_std: np.ndarray) -> None:
    """
    Three-panel figure at the highest mismatch level:
      (a) T2*_eff reference map
      (b) |T2*_pred − T2*_eff| error map  (best physics-only seed)
      (c) Per-pixel cross-seed std map
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(tag, fontsize=11)

    def im(ax, data, title, vmin, vmax, cmap="viridis"):
        i = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=cmap, origin="upper")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        plt.colorbar(i, ax=ax, fraction=0.046, pad=0.04)

    im(axes[0], T2s_eff, "T2*_eff (ms)", 0, 80)
    im(axes[1], np.abs(T2s_pred - T2s_eff), "│pred − T2*_eff│ (ms)", 0, 20, "hot")
    im(axes[2], cross_seed_std, "Cross-seed std (ms)", 0, 10, "hot")

    fig.tight_layout()
    safe = tag.replace(" ", "_").replace("=", "").replace("|", "").replace("/", "_")
    out = RESULTS_DIR / f"mismatch_{safe}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  → saved {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE-LEVEL RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_one_level(
    tag: str,
    S0_gt: np.ndarray,
    T2s_gt: np.ndarray,
    fg_mask: np.ndarray,
    data: dict,                  # output of synthesise() with mismatch applied
    n_epochs: int,
    snr_db: float,
    device: torch.device,
    save_figs: bool = False,
) -> dict:
    """
    Run all methods at one mismatch level vs T2*_eff.

    T2*_eff = analytical mono-exp fit to clean fully-sampled magnitude echoes.
    Method 1 always gives 0 error (it IS T2*_eff — the model-class limit).
    Methods 2–4 are evaluated against T2*_eff.
    """
    # ── T2*_eff reference ─────────────────────────────────────────────────────
    S0_eff, T2s_eff = analytical_fit(data["echoes_full"], TEs_MS)

    # Secondary: model-class bias (T2*_eff vs underlying tissue T2*)
    t2_bias_med = float(np.median(np.abs(T2s_eff[fg_mask] - T2s_gt[fg_mask])))

    # ── Method 2: analytical fit on ZF echoes vs T2*_eff ─────────────────────
    S0_ana_zf, T2s_ana_zf = analytical_fit(data["zf_mag"], TEs_MS)
    m2 = compute_metrics(T2s_ana_zf, S0_ana_zf, T2s_eff, S0_eff, fg_mask)

    # ── Build torch tensors once for all training runs ────────────────────────
    zf_input_tc = torch.from_numpy(data["zf_mag"]).unsqueeze(0)          # [1, n_echoes, H, W]
    k_under_tc  = torch.from_numpy(data["kspace_under"]).to(torch.complex64)
    mask_tc     = torch.from_numpy(data["mask_2d"])

    S0_eff_tc  = torch.from_numpy(S0_eff)
    T2s_eff_tc = torch.from_numpy(T2s_eff)

    # ── Method 3: physics-only, N_SEEDS runs ─────────────────────────────────
    seed_T2_maps:  list[np.ndarray] = []
    seed_metrics:  list[dict]       = []

    for seed in range(N_SEEDS):
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        model = BottleneckNet(N_ECHOES, HIDDEN, T2_MIN, T2_MAX)
        model = train(
            model, zf_input_tc, k_under_tc, mask_tc, TEs_MS,
            s0_gt=None, t2s_gt=None,
            n_epochs=n_epochs, lr=1e-3, device=device,
            verbose=False, label=f"{tag} phys s={seed}",
        )
        S0_pred, T2s_pred = predict_from_model(model, zf_input_tc, device)
        seed_T2_maps.append(T2s_pred)
        seed_metrics.append(compute_metrics(T2s_pred, S0_pred, T2s_eff, S0_eff, fg_mask))
        print(f"    seed={seed}  T2*_med={seed_metrics[-1]['t2_med']:.2f} ms  "
              f"S0_med={seed_metrics[-1]['s0_med_pct']:.2f}%")

    best_seed = int(np.argmin([m["t2_med"] for m in seed_metrics]))
    m3 = seed_metrics[best_seed]

    t2_stack = np.stack(seed_T2_maps, axis=0)                   # [N_SEEDS, H, W]
    cs_std_fg = t2_stack[:, fg_mask].std(axis=0)                # [fg_pixels]
    mean_cs_std = float(cs_std_fg.mean())
    max_cs_std  = float(cs_std_fg.max())
    cs_std_map  = t2_stack.std(axis=0)                          # [H, W]

    # ── Method 4: supervised with T2*_eff as target ───────────────────────────
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    model_sup = BottleneckNet(N_ECHOES, HIDDEN, T2_MIN, T2_MAX)
    model_sup = train(
        model_sup, zf_input_tc, k_under_tc, mask_tc, TEs_MS,
        s0_gt=S0_eff_tc, t2s_gt=T2s_eff_tc,
        n_epochs=n_epochs, lr=1e-3, device=device,
        verbose=False, label=f"{tag} sup",
    )
    S0_sup, T2s_sup = predict_from_model(model_sup, zf_input_tc, device)
    m4 = compute_metrics(T2s_sup, S0_sup, T2s_eff, S0_eff, fg_mask)
    print(f"    supervised  T2*_med={m4['t2_med']:.2f} ms  S0_med={m4['s0_med_pct']:.2f}%")

    # ── Optional figures at this level ────────────────────────────────────────
    if save_figs:
        best_T2_pred = seed_T2_maps[best_seed]
        save_mismatch_figure(tag, T2s_eff, best_T2_pred, cs_std_map)

    return dict(
        T2s_eff=T2s_eff,
        t2_bias_med=t2_bias_med,     # secondary: T2*_eff vs T2*_gt
        m2=m2,
        m3=m3,
        m4=m4,
        cross_seed_mean=mean_cs_std,
        cross_seed_max=max_cs_std,
        seed_metrics=seed_metrics,
        seed_T2_maps=seed_T2_maps,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SWEEP RUNNERS
# ═══════════════════════════════════════════════════════════════════════════════

def sweep_biexp(
    S0_gt: np.ndarray, T2s_gt: np.ndarray, fg_mask: np.ndarray,
    n_epochs: int, snr_db: float, device: torch.device,
) -> list[dict]:
    """Sweep bi-exponential short fraction f ∈ BIEXP_F_LEVELS."""
    print("\n" + "═" * 72)
    print("SWEEP 1 — BI-EXPONENTIAL GROUND TRUTH")
    print(f"  T2s_short={BIEXP_T2S_SHORT} ms, T2s_long=per-pixel tissue T2*")
    print("═" * 72)

    levels = []
    for f in BIEXP_F_LEVELS:
        is_highest = (f == BIEXP_F_LEVELS[-1])
        tag = f"biexp_f{f:.1f}"
        print(f"\n── f={f:.1f}  (short fraction) ──────────────────────────────────────")

        data = synthesise(
            S0_gt, T2s_gt, TEs_MS,
            accel=ACCEL, cf=CF, snr_db=snr_db,
            mask_seed=42, noise_seed=7,
            biexp_f=f, biexp_T2s_short=BIEXP_T2S_SHORT,
        )

        result = run_one_level(
            tag, S0_gt, T2s_gt, fg_mask, data,
            n_epochs=n_epochs, snr_db=snr_db, device=device,
            save_figs=is_highest,
        )
        result["level_key"] = f"f={f:.1f}"
        levels.append(result)

    return levels


def sweep_offres(
    S0_gt: np.ndarray, T2s_gt: np.ndarray, fg_mask: np.ndarray,
    n_epochs: int, snr_db: float, device: torch.device,
) -> list[dict]:
    """Sweep off-resonance max frequency df_max ∈ OFFRES_DF_LEVELS Hz."""
    print("\n" + "═" * 72)
    print("SWEEP 2 — B0 OFF-RESONANCE PHASE")
    print(f"  TEs={TEs_MS} ms, max phase at TE_last = 2π·df·{TEs_MS[-1]/1000:.3f} s")
    print("═" * 72)

    levels = []
    for df_max in OFFRES_DF_LEVELS:
        is_highest = (df_max == OFFRES_DF_LEVELS[-1])
        tag = f"offres_df{df_max}Hz"
        print(f"\n── df_max={df_max} Hz ──────────────────────────────────────────────────")
        if df_max > 0:
            max_phase = 2 * np.pi * df_max * TEs_MS[-1] / 1000.0  # rad
            print(f"   Max phase at TE={TEs_MS[-1]} ms: {max_phase:.1f} rad "
                  f"({max_phase / (2*np.pi):.2f} rotations)")

        offres_map = make_offres_map(H, W, df_max) if df_max > 0 else None

        data = synthesise(
            S0_gt, T2s_gt, TEs_MS,
            accel=ACCEL, cf=CF, snr_db=snr_db,
            mask_seed=42, noise_seed=7,
            offres_map=offres_map,
        )

        result = run_one_level(
            tag, S0_gt, T2s_gt, fg_mask, data,
            n_epochs=n_epochs, snr_db=snr_db, device=device,
            save_figs=is_highest,
        )
        result["level_key"] = f"{df_max} Hz"
        levels.append(result)

    return levels


# ═══════════════════════════════════════════════════════════════════════════════
# VERDICT TABLES + ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def print_sweep_table(sweep_name: str, levels: list[dict]) -> None:
    """Print full results table for one sweep."""
    print(f"\n{'─'*78}")
    print(f"  {sweep_name} — RESULTS TABLE")
    print(f"{'─'*78}")
    hdr = (f"  {'Level':<10}  {'Bias*':>6}  "
           f"{'ZF med':>7}  {'ZF p95':>7}  "
           f"{'Phys med':>9}  {'Phys p95':>9}  {'Phys S0%':>9}  "
           f"{'Sup med':>8}  "
           f"{'Xseed μ':>8}  {'Xseed max':>10}")
    print(hdr)
    print(f"  {'(ms/Hz)':<10}  {'(ms)':>6}  "
          f"{'(ms)':>7}  {'(ms)':>7}  "
          f"{'(ms)':>9}  {'(ms)':>9}  {'(%)':>9}  "
          f"{'(ms)':>8}  "
          f"{'(ms)':>8}  {'(ms)':>10}")
    print(f"  {'─'*74}")
    for lv in levels:
        m2, m3, m4 = lv["m2"], lv["m3"], lv["m4"]
        print(
            f"  {lv['level_key']:<10}  "
            f"{lv['t2_bias_med']:>6.2f}  "
            f"{m2['t2_med']:>7.2f}  {m2['t2_p95']:>7.2f}  "
            f"{m3['t2_med']:>9.2f}  {m3['t2_p95']:>9.2f}  {m3['s0_med_pct']:>9.2f}  "
            f"{m4['t2_med']:>8.2f}  "
            f"{lv['cross_seed_mean']:>8.3f}  {lv['cross_seed_max']:>10.3f}"
        )
    print(f"  {'─'*74}")
    print("  * Bias = median |T2*_eff − T2*_gt| (secondary, model-class limit only)")


def compute_verdict(levels: list[dict], sweep_name: str) -> str:
    """
    Classify the sweep as GRACEFUL, PARTIAL, or BRITTLE.

    GRACEFUL  : phys T2* med ≤ 10 ms at highest level
                AND cross-seed std < 3 ms at ALL levels.
    BRITTLE   : phys T2* med at level 1 > 3× level 0
                OR cross-seed std ≥ 5 ms at any level.
    PARTIAL   : everything else.
    """
    med_errs  = [lv["m3"]["t2_med"]       for lv in levels]
    cs_means  = [lv["cross_seed_mean"]    for lv in levels]

    highest_err    = med_errs[-1]
    jump_ratio     = med_errs[1] / (med_errs[0] + 1e-8) if len(med_errs) > 1 else 1.0
    max_cs_any     = max(cs_means)
    highest_cs     = cs_means[-1]

    crit_bounded   = highest_err <= 10.0
    crit_unique    = max_cs_any < 3.0
    crit_no_blowup = jump_ratio <= 3.0
    crit_no_nonuniq = max_cs_any < 5.0

    print(f"\n  Verdict criteria — {sweep_name}:")
    print(f"    Phys T2* med ≤ 10 ms at highest level    : "
          f"{'YES' if crit_bounded else 'NO '}  ({highest_err:.2f} ms)")
    print(f"    Cross-seed std < 3 ms at all levels      : "
          f"{'YES' if crit_unique else 'NO '}  (max={max_cs_any:.3f} ms)")
    print(f"    Error jump (level 1 / level 0) ≤ 3×      : "
          f"{'YES' if crit_no_blowup else 'NO '}  ({jump_ratio:.2f}×)")
    print(f"    Cross-seed std < 5 ms at all levels      : "
          f"{'YES' if crit_no_nonuniq else 'NO '}  (max={max_cs_any:.3f} ms)")

    if not crit_no_blowup or not crit_no_nonuniq:
        v = "BRITTLE"
    elif crit_bounded and crit_unique:
        v = "GRACEFUL"
    else:
        v = "PARTIAL"

    return v


def error_vs_severity_table(sweep_name: str, levels: list[dict]) -> None:
    """One-line-per-level summary of error and cross-seed std."""
    print(f"\n  Error-vs-severity ({sweep_name}):")
    print(f"    {'Level':<10}  {'Phys med (ms)':>14}  {'Xseed mean (ms)':>16}")
    for lv in levels:
        print(f"    {lv['level_key']:<10}  "
              f"{lv['m3']['t2_med']:>14.2f}  "
              f"{lv['cross_seed_mean']:>16.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Mismatch hardening of identifiability gate")
    parser.add_argument("--epochs", type=int, default=400,
                        help="Training epochs per model (default 400)")
    parser.add_argument("--snr", type=float, default=30.0,
                        help="K-space noise SNR in dB (default 30)")
    args = parser.parse_args()

    # ── device ────────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"\nDevice: {device}")
    print(f"Epochs per model: {args.epochs}, SNR: {args.snr} dB")
    print(f"Bi-exp T2s_short: {BIEXP_T2S_SHORT} ms")
    print(f"Training runs: 2 sweeps × {len(BIEXP_F_LEVELS)} levels × "
          f"{N_SEEDS + 1} models = "
          f"{2 * len(BIEXP_F_LEVELS) * (N_SEEDS + 1)} total")

    # ── Phantom (shared across all levels) ───────────────────────────────────
    S0_gt, T2s_gt, labels = make_phantom(H, W)
    fg_mask = labels > 0

    print(f"\nPhantom: {H}×{W}, fg={fg_mask.sum()} px, "
          f"T2* range {T2s_gt[fg_mask].min():.0f}–{T2s_gt[fg_mask].max():.0f} ms")

    # ── Run both sweeps ───────────────────────────────────────────────────────
    biexp_levels  = sweep_biexp(S0_gt, T2s_gt, fg_mask, args.epochs, args.snr, device)
    offres_levels = sweep_offres(S0_gt, T2s_gt, fg_mask, args.epochs, args.snr, device)

    # ═══════════════════════════════════════════════════════════════════════════
    # FINAL VERDICT OUTPUT
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n\n" + "═" * 78)
    print("MISMATCH HARDENING — FINAL VERDICT TABLES")
    print("═" * 78)

    print_sweep_table("Sweep 1 — Bi-exponential (f= short fraction)", biexp_levels)
    print_sweep_table("Sweep 2 — Off-resonance (df_max= max |df|)", offres_levels)

    print("\n" + "─" * 78)
    print("  ERROR-VS-SEVERITY SUMMARIES")
    error_vs_severity_table("Sweep 1 — Bi-exp", biexp_levels)
    error_vs_severity_table("Sweep 2 — Off-resonance", offres_levels)

    print("\n" + "─" * 78)
    print("  VERDICTS")
    v1 = compute_verdict(biexp_levels,  "Sweep 1 — Bi-exp")
    v2 = compute_verdict(offres_levels, "Sweep 2 — Off-resonance")

    print(f"\n  Sweep 1 — Bi-exponential  : {v1}")
    print(f"  Sweep 2 — Off-resonance   : {v2}")
    print("═" * 78 + "\n")

    # Regression check: f=0.0 must reproduce prior ~0.82 ms result
    ctrl_t2 = biexp_levels[0]["m3"]["t2_med"]
    ctrl_cs = biexp_levels[0]["cross_seed_mean"]
    ok = ctrl_t2 < 2.0 and ctrl_cs < 2.0
    print(f"  Regression check (f=0.0 ≈ original gate): "
          f"T2* med={ctrl_t2:.2f} ms, Xseed={ctrl_cs:.3f} ms — "
          f"{'OK' if ok else 'REGRESSION DETECTED'}")
    print()


if __name__ == "__main__":
    main()
