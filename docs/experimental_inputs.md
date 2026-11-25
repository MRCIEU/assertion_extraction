# Experimental inputs and learning conditions

The matched entity representation runs from E0 to E7. A missing finer class keeps the native type. DrugProt stays at its native entity representation and varies only P0–P3. The records are in `studies/schema_granularity_v1/reports/experimental_preparation/`. Frozen inputs, the historical verbalisation, and `formal_20261115` were not overwritten.

## Entity transitions

On the BioRED training candidates, the number whose displayed type changes is 21,755 from E0 to E1, 21,752 from E1 to E2, 5,115 from E2 to E3, 2,903 from E3 to E4, 2,697 from E4 to E5, 2,390 from E5 to E6, and 1,589 from E6 to E7. The figure 5,115 belongs only to E2–E3. Development, test, and BC8 have their own counts in `summary.json`. Information content is reported both for every BioRED training concept, including those that retain the native category, and for the 507 concepts with a non-empty class path.

## Matched inputs

The new BiomedBERT collection is `target_anchored_v2__biomedbert_e1354b7a_matched_e0_e7`. BioRED has 35,136 inputs and BC8 has 28,734. None exceeds 512 tokens. The longest input is exactly 512 in both corpora. Median encoded lengths are 416 and 403. Relative to the frozen windows, 1,134 BioRED inputs and 1,415 BC8 inputs use a different source span. Those identifiers are listed in `changed_windows_biored.jsonl` and `changed_windows_bc8.jsonl`. Historical E0–E2 predictions remain valid for the frozen input version. They are not the matched E0–E7 comparison where the window changed, and E3–E7 had no historical run.

Qwen3-8B and Qwen3-32B use one tokenizer. The prompted length includes the instruction, the label block, the chat template and a reserved 256 generation tokens. It is not the BiomedBERT length. No prepared prompt exceeds the 40,960-token position limit. The longest prompts are 1,249 tokens for BioRED, 1,293 for BC8 and 1,023 for DrugProt. A BioRED development E0 prompt has median 771 Qwen tokens.

## Learning conditions

The principal matrix has seven systems. BioRED crosses E0–E7 with P0–P3. BC8 is evaluation only, using the BioRED models, schema, and training demonstration pool. DrugProt uses the native entity type and P0–P3. There are 324 training runs, 108 for each of BiomedBERT, Qwen3-8B, and Qwen3-32B, and no BC8 training run. Few-shot retrieval is BM25 with five demonstrations, fixed for a candidate across entity and predicate conditions, with a negative demonstration when one exists. A separate random-demonstration file covers the 6,334 BioRED development candidates and is not part of the factorial. Supervised targets are assistant label objects. The candidate identifier is not in the loss. LoRA uses the same BF16 configuration for both Qwen sizes: rank 16, alpha 32, dropout 0.05, AdamW at 1e-4, the seven attention and MLP projections, and seeds 13, 42, and 2026. Those modules are present in both model indexes. No training was run.

The supplementary Qwen3-8B schema-conditioned adapter is outside the principal matrix. One view of each of the 21,755 BioRED training candidates is drawn per epoch, against 696,160 independent examples per epoch.

## Verbalisation and requests

Verbalisation `representation_verbalisation_v4` recovered mention text for both arguments. Of 3,119,666 prediction records, 1,558,853 have a positive verbalisation, 1,577,586 have no positive assertion, 6,241 lack an entity mention, 4,708 were rejected, 9 are missing, and 1,850 have another non-valid status. The positive texts collapse to 267,257 distinct propositions and 271,758 forward source–proposition requests. Relation-conditioned localisation requests that can be built now number 566,852. Finding-level certainty and reverse entailment are not given exact future counts. Their upper bound is the 267,257 current propositions.

## Other prepared material

The sentence comparison on 100 BioRED development documents used `en_core_sci_sm` 0.5.4, which has a parser. It shared 1,086 spans with the rule-based segmenter, with 26 spans only in the baseline and 22 only in scispaCy. That is disagreement, not boundary accuracy. `hf-hpc` has no spaCy and was not modified.

The annotation packets contain source text, mention offsets, and empty judgement fields. Fifteen of 75 BioRED test documents were drawn. Every document in that frame has an annotated relation, so the document sample has no unannotated-document stratum. The candidate sample drew 13 of 664 annotated candidates and 63 of 3,160 unannotated candidates. The inclusion probabilities are the realised fractions, 0.0196 and 0.0199.

Of the 631 lung-cancer CIViC rows, 569 are predictive, 49 prognostic, 3 diagnostic, and 10 oncogenic. Supporting and non-supporting directions are both present. No oncology extraction was run, and the curated statement is not a model input.

The sentence-level certainty checkpoint `pedropei/sentence-level-certainty`, revision `57bb19e0804a77689ca02f2b1d408d162413cdc2`, was loaded in `hf-hpc` on CPU. It is a single-output SciBERT classifier, not the aspect-level model. On the 300 SciFact development claims the raw outputs ran from 2.69 to 6.00, with median 5.01, all inside the published 1–6 interval. SciFact does not label those claims for certainty, so this is not an accuracy. The checkpoint does not include an official labelled split or a documented output transformation. The verifier `MoritzLaurer/DeBERTa-v3-xsmall-mnli-fever-anli-ling-wanli` was not run: anonymous retrieval returned HTTP 401. No GPU job was submitted. Formal execution still requires separate approval.
