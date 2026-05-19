# Phase 2C — Matched-compute control (pre-committed analysis rule)

**Status:** SIGNED-OFF (**FY**, 2026-05-19 UTC); launch gated on merging this commit
before `sbatch` (launch gate checklist below unchanged).

**Design:** PubMedBERT-base × full fine-tuning × **T1 flat (T1F)**, increasing
`scientific_trainer.max_updates` from **2 048** (pre-registered factorial cell) to
**4 096**, holding corpus composition identical to T1F. This cell is compared
**seed-for-seed** to the pre-registered **T2** (staged T1→T2, 4 096 total
updates) cell to separate additional optimisation steps from oncology-projected
T2 content.

---

## Metrics to load (post hoc, no new pre-registered claims)

Per seed \(s \in \{1,\ldots,20\}\), let \(Y_{\mathrm{T1F2048}}(s)\),
\(Y_{\mathrm{T1F4096}}(s)\), and \(Y_{\mathrm{T2}}(s)\) denote **KB argmax
accuracy** — i.e. `kb_hit_A_setvalued` from `phase_b_eval.json` (Method A,
set-valued, **S_pair**), **already aggregated to the run (cell × seed) level**
exactly as in the locked Phase B pipeline.

Means are taken across the 20 seeds unless noted otherwise.

---

## Continuous compute attribution (replacing discrete 0.05 buckets)

Define seed-level component gaps (holding encoder and update regime fixed at
PubMedBERT-base × FT):

\[
d_{\mathrm{comp}}(s) = Y_{\mathrm{T1F4096}}(s) - Y_{\mathrm{T1F2048}}(s)
\]

\[
d_{\mathrm{gap}}(s) = Y_{\mathrm{T2}}(s) - Y_{\mathrm{T1F2048}}(s)
\]

**Point estimate** (ratio of *mean* gaps; stabiliser preferred to a noisy
per-seed ratio):

\[
\widehat\alpha =
\frac{\overline{d_{\mathrm{comp}}}}{\overline{d_{\mathrm{gap}}}}
\quad\text{when } \overline{d_{\mathrm{gap}}} \neq 0
\]

Interpretation: \(\widehat\alpha\) is the fraction of the **T1F→T2 KB lift**
explained by **extra gradient steps alone** if staging added no incremental
effect beyond what doubling T1F updates would produce.

**Uncertainty:** paired seed bootstrap, **B = 5 000**, RNG seed **20260518**.
For each bootstrap replicate, draw 20 seeds **with replacement**, compute
\(\overline{d_{\mathrm{comp}}}^*\) and \(\overline{d_{\mathrm{gap}}}^*\) on
the replicate, and form \(\alpha^* =
\overline{d_{\mathrm{comp}}}^*/\overline{d_{\mathrm{gap}}}^*\) (discard a
replicate if \(\overline{d_{\mathrm{gap}}}^* = 0\); if >10% of replicates are
discarded, report instability and halt for manual review). Report the **95%**
percentile interval (2.5th / 97.5th) of successful \(\alpha^*\) values.

**Verdict mapping** (uses \(\widehat\alpha\) **and** the bootstrap interval):

| Rule | Verdict |
|------|---------|
| \(\widehat\alpha < 0.20\) **and** CI upper < 0.30 | **content-dominant** |
| \(\widehat\alpha > 0.80\) **and** CI lower > 0.70 | **compute-dominant** |
| \(\widehat\alpha \in [0.20,\,0.80]\) **and** (CI width)< 0.50 | **mixed; report \(\widehat\alpha\)** |
| CI width ≥ 0.50 | **mixed; attribution uncertain** |
| \(\widehat\alpha > 1.0\) (equivalently \(\overline{Y}_{\mathrm{T1F4096}} > \overline{Y}_{\mathrm{T2}}\) with the same numerator/denominator sign pattern) | **unexpected; halt for re-framing** |

“CI width” means \((\mathrm{upper} - \mathrm{lower})\) on \(\alpha^*\).

---

## Extended variance decomposition (reporting obligation only)

- **Pre-registered headline (unchanged):** report **H7 / nine-cell** \(R_B =
  0.21\) exactly as locked in Phase B — **do not overwrite or reinterpret**
  this headline statistic.

- **Augmented diagnostic:** recompute \(R_B\) on the **ten-cell** main grid
  formed by adding the **T1F-4096** matched-compute cell alongside the original
  nine PubMedBERT-base / schedule cells. Label explicitly as
  **“augmented 10-cell \(R_B\) (includes T1F-4096)”** in tables/text so it
  cannot be confused with the pre-registered nine-cell value.

---

## Launch gate

1. Author initials / date on this `COMMITMENT.md` in Git (**after** sign-off).
2. Run `PB_PB_FT_T1F4096_s{01..20}` training + eval (Slurm array).
3. Only **after** \(Y_{\mathrm{T1F4096}}(s)\) exist for all 20 seeds: compute
   \(\widehat\alpha\), bootstrap CI, verdict, and both \(R_B\) summaries.

### Sign-off block (initialise before `sbatch`)

| Role | Initials | Date (UTC) |
|------|----------|------------|
| Attribution rule approved for launch | FY | 2026-05-19 |

After the row is complete, **commit** this `COMMITMENT.md` to the Phase~D branch,
then submit `matched_compute/sbatch/pb_ft_t1f4096_array.sbatch`.
