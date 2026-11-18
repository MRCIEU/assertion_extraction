# Relation-extraction instruction

One instruction for both Qwen models. The formal runs use the V1 text.

Method: The prompt names the candidate, the dataset rule, the legal positive labels, and the marked mentions. It does not include the gold label. An empty label list is the no-relation answer. V0 is kept as the development record and is not the formal instruction.

| File | Role |
| --- | --- |
| relation_extraction_v1.txt | Formal instruction |
| relation_extraction.txt | Development record |
| output_examples/ | Legal JSON shapes |

V1 shows both a positive object and an empty list, states the cardinality of the current corpus, and says that co-occurrence or outside knowledge is not enough.
