# Paper inserts — author-level IAA (Phase 3 carry-over)

> **Integration note:** paste during Phase 3D manually; not a LaTeX `\input` source.

## A. §4.2 — proposed addition after the existing directional-pattern sentence

To further calibrate the heuristic projection against human judgment, the first author independently labelled the same 30 stratified audit targets blinded to the heuristic and LLM outputs (label vocabulary: six family-level options matching Supplement~S3, including \code{__NEGATIVE__}). Author--heuristic agreement was Cohen's $\kappa = 0.434$ (95\% bootstrap CI $[0.214, 0.654]$); author--LLM agreement was $\kappa = 0.835$ (CI $[0.628, 1.000]$). Among the seven IAA disagreement targets (where the LLM chose \code{__NEGATIVE__} and the heuristic chose a positive relation), the author agreed with the LLM on 7 of 7, with the heuristic on 0 of 7, and took a distinct label on 0 of 7. The $\kappa$ values place author--heuristic agreement in the **moderate** band (Landis \& Koch) and author--LLM agreement in the **almost perfect** band, consistent with **high agreement between the LLM and the author** ($\kappa \approx 0.835$), while **author--heuristic agreement is lower** than LLM--heuristic ($\kappa = 0.434$ vs $\kappa = 0.561$), reflecting additional human negative calls beyond the seven LLM--heuristic disagreement rows. On those seven targets (LLM \code{__NEGATIVE__} vs heuristic positive), the **author matched the LLM in all seven cases**. Bootstrap CIs use $B=5000$, seed 20260520.

*Editorial tighten as needed.*

## B. Supplement C §S3 — new subsection (S3.x) draft

\paragraph{Author-level IAA (post-hoc validity check).}
The first author independently labelled the same 30 stratified targets (\code{random.Random(42)}; 27 \code{gene\_drug} / 3 \code{variant\_disease}) blinded to both the heuristic projection and the LLM second annotator. Cohen's $\kappa$ was 0.434 (95\% bootstrap CI $[0.214, 0.654]$) against the heuristic and 0.835 (CI $[0.628, 1.000]$) against the LLM, on the six-family vocabulary including \code{__NEGATIVE__}. Fleiss' $\kappa$ for the three raters (heuristic, LLM, author) was 0.603. On the seven targets where the LLM chose \code{__NEGATIVE__} while the heuristic selected a positive family label, the author agreed with the LLM on 7 cases, with the heuristic on 0, and chose a different label on 0. Human labels are released as \code{author\_audit\_labels.csv} alongside \code{audit\_labels.csv} under \code{analysis/inter\_annotator\_audit/author\_level\_iaa/}.

## C. Cover letter — one-sentence AI disclosure addition

In addition to the LLM second annotator (Claude Opus 4.7), the first author performed an independent blinded human labelling pass on the same 30 audit targets, with agreement statistics reported in \S4.2 and Supplement~C; no generative AI assisted this human adjudication.
