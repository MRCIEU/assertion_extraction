# Schema-study package

Prepares candidates, trains BiomedBERT, generates Qwen answers, rolls P4 predictions up, and scores them.

Method: BioRED is one label plus No_Relation, with softmax cross-entropy. DrugProt is multi-label, with mean BCE-with-logits. Generation is greedy, bf16, batch 16, 256 new tokens, no truncation. Rollup maps saved discrete labels and drops duplicates.

| Entry | Does |
| --- | --- |
| formal_prepare | Builds the V1 request shards without calling a model |
| formal_train | Fine-tunes from the fixed checkpoint and writes best and last |
| formal_generate | Fills one shard and keeps the attempt ledger |
| formal_analyze | Scores and rolls up from the candidate tables |
| predictions | Parses an answer and rejects an illegal BioRED label set |
| granularity | Measures label entropy of the frozen entity and predicate levels |
| source_references | Builds sentence records and candidate passages without choosing findings |
| representation_readiness | Checks those records and measures MeSH coverage and alternative partitions |
| formal_protocol | Assembles the pre-training protocol without running a model |

A failed or illegal answer stays in the scored population. It is not treated as a correct empty prediction.
