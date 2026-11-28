# Human reference annotation for source findings and semantic measures

The current development sample is for method development. The current human sample is a document-disjoint diagnostic set. Neither sample estimates a corpus rate. BioRED test, BC8, and DrugProt evaluation have already been scored, so they are not untouched confirmation data.

## Judgements

The judgements are separate.

1. Which scientific finding the source expresses, without a gold label or a model prediction.
2. The exact character spans of that finding.
3. The linguistic certainty of that finding. Certainty is not polarity, statistical significance, study quality, or the probability that the claim is true.
4. Whether a displayed assertion is supported, contradicted, or left without sufficient support by a stated source context.
5. Which applicable components of the finding are preserved, validly abstracted, omitted, or distorted.

A reviewer may record several findings, several locations, or an unassessable case. No field is filled by a model. The examples prepared for mapping review leave `human_judgement` empty.

## Materials

Each item needs the original document, the target mentions, character offsets, and a stable identifier. The initial source task does not show the native label, the prediction, the requested granularity, or an automatic score. A later assertion task may show the assertion and must still hide the model identity and the automatic score.

Two reviewers and an adjudication record are required before a reference is treated as independent validation. An unfilled packet is not completed validation.

## Sample size

No design effect has been estimated, so a sample size is not claimed to achieve a stated precision. Under an independence assumption that is unlikely to hold for sentences nested in documents, a simple random sample of about 385 yields a margin near 0.05 for a proportion near one half. Candidate sizes of 60, 200, and 385 are therefore assumptions, not a justification. The representative sample, once drawn, needs a sampling unit, strata, inclusion probabilities, and analysis weights. It must include annotated positive relations, unannotated candidates, and model-generated positive assertions, and its documents must stay disjoint from method development.

Shared PubMed identifiers remain in within-corpus analyses. A cross-corpus sensitivity population excludes them.
