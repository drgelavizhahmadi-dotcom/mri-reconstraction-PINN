# FFT Normalization Audit — OpenNeuro ds006926 (commit 2f6a97a)

## Bug Description

`fft2c_torch` in `run_openneuro_repro.py` was missing `norm="ortho"`, while `fft2c_np`
(imported from `run_gate.py`) always uses `norm="ortho"`. This created a 4096× energy
mismatch (√4096=64 amplitude) between predicted k-space (torch, unscaled) and measured
k-space (numpy, ortho).

**Scope:** `fft2c_torch` is called ONLY in `kspace_loss_mag` and `kspace_loss_cpx`
(the A2 training loss, backward pass only). ALL evaluation functions (`_kres_net_mag`,
`_kres_net_cpx`, gate, NRMSE, texture, fmap correlation) use `fft2c_np` (always ortho).
The bug corrupted A2 network **training**, not metric arithmetic.

## Git Forensics

```
git log -S 'norm="ortho"' -- experiments/identifiability_gate/run_openneuro_repro.py
→ only one commit: 2f6a97a (the commit containing the results)
```

The fix and the results were committed together. The committed `openneuro_repro_v5.log`
shows A2-A initial loss = 0.174 (consistent with fixed code). Pre-fix loss was 406 → 1.0.

**Re-run verification:** Full re-run on current fixed code → bit-for-bit identical to
committed results (same seeds SEED=42, MASK_SEED=42, N_SEEDS=2, MPS deterministic).

## Cohort Manifest

| Role | Subjects | Files |
|------|----------|-------|
| Train (A3 only) | sub-a01..sub-a08 | 8 × [10 MEGRE, 3 sbref, phasediff] |
| Test | sub-a09, sub-a10 | 2 × same |

Real Siemens data confirmed (not synthetic):
- MEGRE echo-1 max: sub-a09=4095, sub-a10=3133 (12-bit Siemens range)
- sbref echo-1 max: sub-a09=36343, sub-a10=37639 (EPI scale)
- Tripwire: no 1e-6 residuals observed — CLEAR

A3 is NON-CIRCULAR (8 train subjects ≠ 2 test subjects, n≥5 ✓).

A3-B is INVALID due to train/test distribution shift (A3 trained on FS input, tested on
undersampled ZF input). A3-B metrics: NRMSE=0.69, tex=-0.13, kres=4.54 — unusable.

## Pre/Post Comparison Table

| Metric | PRE-FIX (v1 log) | POST-FIX (2f6a97a) | Status |
|--------|----------|----------|--------|
| **GATE (analytical, numpy — unaffected by bug)** | | | |
| Gate-A kres_2p (sub-a09) | 0.2207 | 0.2207 | UNCHANGED |
| Gate-A kres_2p (sub-a10) | 0.1616 | 0.1616 | UNCHANGED |
| Gate-B kres_4p (sub-a09) | 0.6114 | 0.6114 | UNCHANGED |
| Gate-B kres_4p (sub-a10) | 0.6657 | 0.6657 | UNCHANGED |
| Gate-B ratio (sub-a09) | 0.548 | 0.548 | UNCHANGED |
| Gate-B ratio (sub-a10) | 0.711 | 0.711 | UNCHANGED |
| **BRANCH A — A2 training corrupted pre-fix** | | | |
| ZF NRMSE at R=2 (mean) | 0.149 | 0.149 | UNCHANGED |
| A2-A NRMSE at R=2 (mean) | 1.000 🚨 | 0.374 | WAS-INVALID |
| A2-A texture-r at R=2 | 0.055 🚨 | 0.470 | WAS-INVALID |
| A2-A unmeas/meas ratio (R=4,8) | 1.00 🚨 | 1.58 | WAS-INVALID |
| A2-A Spearman ρ(unc,err) | -0.296 🚨 | 0.229 | WAS-INVALID |
| CHECK1 BOUNDARY Branch-A | DOES-NOT-REPRODUCE | DOES-NOT-REPRODUCE | UNCHANGED\* |
| CHECK2 BLIND SPOT Branch-A | DOES-NOT-REPRODUCE | PARTIAL (1.58) | CHANGED |
| CHECK3 ABSTENTION Branch-A | DOES-NOT-REPRODUCE | PARTIAL (0.23) | CHANGED |
| **BRANCH B — A2 training corrupted pre-fix** | | | |
| ZF NRMSE at R=2 (mean) | 0.220 | 0.220 | UNCHANGED |
| A2-B NRMSE at R=2 (mean) | 0.871 🚨 | 0.212 | WAS-INVALID |
| A2-B texture-r at R=2 | -0.032 🚨 | 0.542 | WAS-INVALID |
| A2-B unmeas/meas ratio (R=4,8) | 1.05 🚨 | 2.02 | WAS-INVALID |
| A2-B Spearman ρ(unc,err) | -0.015 🚨 | 0.248 | WAS-INVALID |
| CHECK1 BOUNDARY Branch-B | DOES-NOT-REPRODUCE | REPRODUCES (0.542) | CHANGED |
| CHECK2 BLIND SPOT Branch-B | DOES-NOT-REPRODUCE | PARTIAL (2.02) | CHANGED |
| CHECK3 ABSTENTION Branch-B | DOES-NOT-REPRODUCE | PARTIAL (0.25) | CHANGED |
| **BONUS fmap** | | | |
| A2-Df vs fmap Spearman ρ | -0.023 🚨 | 0.665 | WAS-INVALID |
| Analytical Df vs fmap ρ | 0.608 | 0.608 | UNCHANGED |
| BONUS verdict | UNCORRELATED | CORRELATES | CHANGED |

\* CHECK1 Branch-A verdict is DOES-NOT-REPRODUCE in both cases, but for different reasons:
pre-fix the A2 image is blank (S0=0); post-fix the A2 image has real content but
degrades texture vs ZF (A2 tex=0.47 < ZF tex=0.90). The post-fix finding is informative.

## Verdict Table (Paper Claims)

| Claim | PRE-FIX | POST-FIX | Verdict |
|-------|---------|---------|---------|
| Blind-spot (Branch A, unmeas/meas) | FAILS (1.00) | PARTIAL (1.58) | **WEAKENS** |
| Blind-spot (Branch B, unmeas/meas) | FAILS (1.05) | PARTIAL (2.02) | **WEAKENS** |
| Abstention (Branch A, ρ≥0.15) | FAILS (-0.30) | PARTIAL (0.23) | **WEAKENS** |
| Abstention (Branch B, ρ≥0.15) | FAILS (-0.015) | PARTIAL (0.25) | **WEAKENS** |
| Boundary softens — Branch A | INVALID (0.055) | A2 tex=0.47 < ZF=0.90 | **OVERTURNS** |
| Boundary softens — Branch B | INVALID (-0.032) | A2 tex=0.54 (>0.50 threshold) | **CONFIRMS** |
| Df-fmap correlation ρ=0.67 (A2) | FAILS (-0.023) | 0.665 | **CONFIRMS** |
| Analytical Df-fmap ρ=0.61 | 0.608 (valid) | 0.608 (valid) | **CONFIRMS** |

**Legend:** CONFIRMS = post-fix supports claim. WEAKENS = positive but sub-threshold.
OVERTURNS = claim is false (magnitude-only A2 degrades vs ZF on Branch A).

## Summary Interpretation

The post-fix results from 2f6a97a are trustworthy. The key scientific finding stands:
phase information is load-bearing for the k-space DC loss — the complex Branch B
reproduces texture recovery while the magnitude-only Branch A does not (actively hurts).
The blind-spot and abstention effects are real but weaker than claimed (PARTIAL not
REPRODUCES), consistent with real EPI data being more complex than the synthetic setting.
