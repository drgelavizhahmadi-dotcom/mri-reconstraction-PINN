# PRE-REGISTERED PREDICTION — Step 12, multiplicative nodes

**Status: written and committed BEFORE the multiplicative experiment was built or run.
No multiplicative-node data of any kind had been generated when this file was committed.**

This file must NOT be revised after results are seen. If the analysis turns out to need
correcting, a SECOND file (`POSTHOC_step12_corrections.md`) explains what and why, and this
file stays exactly as committed.

## Theory

For a multiplicative node `T = prod_j psi_j(x_j)`, the surviving gauge is NOT affine.
`h(prod psi_j)` stays multiplicatively separable exactly when `h(uv) = h(u)h(v)`, i.e. for
continuous `h`, the POWER MAPS `h(u) = u^c`. Plus the reciprocal torus
`psi_1 -> lambda*psi_1`, `psi_2 -> psi_2/lambda`.

So: **additive nodes -> affine gauge; multiplicative nodes -> POWER gauge.**

## Predictions (registered in this wording)

- **P1.** Multiplicative-node edges will be UNSTABLE under affine quotient.
- **P2.** They will be STABLE under POWER quotient (fit `psi_j ~ c * psi_i^p` on the positive
  part of the domain; report residual).
- **P3.** The fitted exponent `p` will be SHARED across the edges of one node within a pair
  (analogous to the 1.1% shared-scale result), and vary across pairs.
- **P4.** The reciprocal torus will show up as fitted prefactors `c_1, c_2` with
  `c_1*c_2 ~ constant` within a pair.

## Operationalization (also registered in advance)

Decisions fixed before seeing any data, so they cannot be tuned afterwards:

1. **Thresholds.** "STABLE" = mean normalized residual < 0.10; "UNSTABLE" = >= 0.10. These are
   the same thresholds used for `QUOTIENTED` throughout Steps 3-11.
2. **Affine quotient (P1)** = the existing `raw_and_quotiented` metric, unchanged, applied to
   marginal-sweep edge curves — identical code and convention to every earlier step.
3. **Power quotient (P2).** Fit `log|psi_j| = log|c| + p*log|psi_i|` by least squares over the
   sub-domain where BOTH `psi_i` and `psi_j` are strictly positive and bounded away from zero
   (`> 1e-3 * max|psi|`). Residual is reported in the ORIGINAL (not log) space, normalized by
   `||psi_j||` over that same sub-domain, so it is directly comparable to the affine residual.
   The retained-domain fraction is reported for every pair; a pair with < 30% of the grid
   retained is declared INCONCLUSIVE rather than counted either way.
4. **P3 test.** Report `|p_1 - p_2| / max(|p_1|,|p_2|)` within each pair (the exact analogue of
   the 1.1% shared-scale statistic), plus the across-pair spread of `p`, plus their ratio.
   "SHARED" = mean within-pair relative difference < 0.10 AND ratio (within/across-std) < 0.5.
5. **P4 test.** `c_1*c_2` is a single number per pair, so "constant within a pair" is tested as
   the reciprocal-torus signature it implies: across pairs, `log|c_1|` and `log|c_2|` should be
   ANTI-correlated (one branch absorbs what the other gives up), and `c_1*c_2` should have
   visibly smaller relative spread than `c_1` or `c_2` individually. Reported as: Pearson r
   between `log|c_1|` and `log|c_2|` (predicted negative), and
   `CV(c_1*c_2)` vs `CV(c_1)`, `CV(c_2)`. "HELD" = r < -0.3 AND `CV(c_1*c_2)` < both individual CVs.
6. **Convergence filter.** Same as every prior step: test `R^2 > 0.999`; all metrics reported on
   well-converged pairs with counts, and the unfiltered numbers alongside.
7. **Sign/domain handling.** The target is chosen strictly positive on the sampled domain so the
   power fit is well-posed (see below); the network's internal `psi` may still change sign, which
   is exactly why rule 3 restricts to the positive sub-domain and reports the retained fraction.

## Planned experiment

- **Target:** `f(x1, x2) = exp(-x1) * sin(pi*x2)` with `x1 ~ U(0.05, 4.0)` and `x2 ~ U(0.05, 0.95)`.
  `sin(pi*x2) > 0` strictly on that interval, so the target is strictly positive and genuinely
  multiplicatively separable. `x1` reuses Task A's domain/scale; the `sin` factor reuses Task B's
  functional form. This is the MRI magnitude-model shape (a decay factor times a modulation
  factor), separable by construction.
- **Architecture:** `MultKAN[2 -> 1 -> 1]` — a multiplicative aggregation node
  `T = psi_1(x1) * psi_2(x2)` built from the same B-spline + SiLU edge machinery as `KANLinear`
  (so the only difference from the additive experiments is `prod` instead of `sum`), followed by
  `KANLinear(1,1)` as the outer `g`. `hidden=1` mirrors Steps 8-9 exactly.
- **Protocol:** 10 seeds (same fixed list 0-9), grid=12, k=3, same optimizer/schedule family,
  forced-CPU determinism. Epoch count may need to exceed 600 as it did at d>=2; whatever is used
  is reported, along with per-seed convergence.

## Possible outcomes

A failure here is a real result and will be reported as one. In particular:
- If P1 fails (edges ARE affine-stable at a multiplicative node), the additive/multiplicative
  gauge distinction does not survive contact with trained networks.
- If P2 fails (power quotient does NOT stabilize), the gauge for multiplicative nodes is
  something other than the power maps, and the theory as stated is incomplete.
- P3/P4 can fail independently of P1/P2; each is reported separately as HELD / FAILED /
  INCONCLUSIVE with numbers.
