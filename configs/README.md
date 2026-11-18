# Schemas, data contracts, and the experiment matrix

Holds the label inventories, corpus splits, model revisions, and the main-matrix specification.

Method: Predicate levels P1–P4 and, for BioRED and BC8, entity levels E0–E2. DrugProt uses its native entity types. Five renamed or randomly merged controls stay draft and are not part of the main matrix.

| Piece | Setting |
| --- | --- |
| Fine-tuning runs | 48 |
| Seeds | 13, 42, 2026 |
| BiomedBERT revision | e1354b7a3a09615f6aba48dfad4b7a613eef7062 |
| Qwen3-8B revision | b968826d9c46dd6066d109eabc6255188de91218 |
| Qwen3-32B revision | 9216db5781bf21249d130ec9da846c4624c16137 |

Selection uses the development split only. BC8 and the DrugProt official development set are evaluation, not training.
