# Blank forms for the semantic review

These forms are for the probability sample in `reports/entity_classification/human_probability_sample.jsonl`. Development and diagnostic examples stay in the source-reference sample and are not reused here. No answer has been filled in. A model must not be used to complete them.

The sampling unit is the BioRED test document, after removal of documents already used to develop or diagnose the source procedure. Documents are stratified by whether they contain an annotated positive relation. Each document in the frame had inclusion probability 0.2. The analysis weight is 5. The seed is 20261125.

## First reading

The reader sees the document and the named entities. The reader does not see the native relation, the model, the granularity condition, or an automated score.

| Field | Entry |
| --- | --- |
| Document | |
| Entities | |
| Proposition stated by the source, in the reader’s words | |
| Character spans or sentence locations | |
| Repeated wording, and which occurrence states the proposition | |
| Linguistic certainty of that proposition | asserted / hedged / conditional / negated / uncommitted / cannot tell |
| Polarity, where the proposition has one | positive / negative / not applicable / cannot tell |

## Later assessment

This reading may use the verbalised assertion. It still does not identify the model, the granularity condition, or an automated score.

| Field | Entry |
| --- | --- |
| Assertion | |
| Source support | supported / contradicted / unsupported / cannot tell |
| Distinctions preserved | |
| Distinctions omitted, validly broadened, or distorted | |
| Adjudication, if the two readers disagree | |

A second reader completes the same form independently. Disagreements are adjudicated by a third reader who sees both forms and the document, not the model identity.
