# POST-HOC analysis notes — Step 12

Companion to `PREREGISTERED_step12_multiplicative.md` (committed 69edebb, before any
multiplicative data existed). **That file has not been edited.** The registered verdicts stand
exactly as computed:

| prediction | registered verdict | number |
|---|---|---|
| P1 multiplicative edges UNSTABLE under affine quotient | **FAILED** | mean affine residual 0.0324 (< 0.10 threshold) |
| P2 STABLE under power quotient | **HELD** | mean power residual 0.0042 |
| P3 exponent `p` shared across a node's edges within a pair | **HELD** | within-pair rel. \|p1−p2\| = 0.0135; across-pair std 1.1628; ratio 0.0120 |
| P4 reciprocal torus visible as `c_1*c_2` | **FAILED** | r(log\|c1\|, log\|c2\|) = −0.055; CV(c1·c2)=0.368 > CV(c1)=0.192, CV(c2)=0.253 |

This file records what the failures mean. It does not change any verdict above.

---

## P1 — why it failed, and what the data actually shows

P1 predicted multiplicative-node edges would be UNSTABLE (residual ≥ 0.10) under the affine
quotient. Measured mean was 0.0324, so P1 failed as registered.

The reason is visible in the per-pair data: **a power map with `p ≈ 1` IS an affine map.** Many
seed-pairs happened to land near `p = 1`, and those pairs are trivially affine-stable, dragging
the mean below threshold. Stratifying by how far the fitted exponent is from 1:

| \|p − 1\| | n edges | affine residual | power residual | power advantage |
|---|---|---|---|---|
| [0, 0.15) | 12 | 0.0036 | 0.0019 | 1.9× |
| [0.15, 0.5) | 18 | 0.0087 | 0.0022 | 3.9× |
| [0.5, 1.0) | 10 | 0.0433 | 0.0050 | 8.7× |
| [1.0, ∞) | 16 | 0.0738 | 0.0077 | 9.5× |

`corr(|p−1|, affine residual) = +0.649`. Affine degrades monotonically as the realized gauge
moves away from the affine sub-case, and the power quotient's advantage grows from 1.9× to 9.5×.

**Correct statement:** the affine quotient is not adequate for multiplicative nodes *in general*;
it only appears adequate on average because the exponent frequently lands near 1. P1's threshold
test was the wrong instrument — it measured an average over a mixture of near-affine and
strongly-non-affine pairs. The mechanism P1 was pointing at is present and quantified above;
the registered prediction still failed on its own terms and is reported as failed.

## P4 — why it failed

P4 predicted the reciprocal torus (`psi_1 → λ·psi_1`, `psi_2 → psi_2/λ`) would appear as
prefactors with `c_1·c_2` pinned. It did not: the two log-prefactors are essentially uncorrelated
(r = −0.055) and their product is *more* variable than either factor alone.

Two problems with the prediction, both visible only in hindsight:

1. **Nothing pins `c_1·c_2` when the outer `g` is free.** The composed map is
   `g(c_1 c_2 · T^p)`. Because `g` is itself a learned flexible function, *any* value of
   `c_1·c_2` can be absorbed by `g`. The torus is a real freedom of the node, but it is not
   observable in the prefactors — it is absorbed downstream. P4 implicitly assumed `g` was fixed
   across seeds, which it is not.
2. **`c` is slaved to `p` by the fit itself.** `corr(p, log|c_1|) = +0.953`. When fitting
   `psi_j ≈ c·psi_i^p` over a bounded positive range, `c` and `p` are strongly coupled (changing
   the exponent forces a compensating prefactor to keep the curve through the data). So the
   measured prefactors mostly report the exponent, not an independent gauge coordinate.

**Correct statement:** P4 was not a well-posed test of the torus. A well-posed test would have to
control for the outer `g` (e.g. compare `psi_1` and `psi_2` scale changes *at fixed p*, or
normalize `g` first). This was not attempted here and is left as future work rather than
retro-fitted into a passing result.

## What survives

The load-bearing multiplicative claim that *is* measured and supported:

- The gauge relating independently trained multiplicative nodes is a **power map**, not an affine
  one: power residual 0.0042 vs affine 0.0324 overall, and up to 9.5× better where the exponent
  departs from 1 (P2, HELD).
- The exponent is **shared across the two edges of a node** to 1.35% within a pair while ranging
  over roughly [−3.2, +2.4] across pairs (P3, HELD) — the exact structural analogue of the 1.1%
  shared-scale result for additive nodes at d≥2, and the strongest single piece of evidence that
  what is being measured is a node-level gauge rather than per-edge noise.
