# Study agreements

Records the schema, the candidate rules, the training contract, and the scoring rules that the code is written against.

Method: Five agreement files plus the research plan and the list of decisions still open. Gold labels are read from the candidate tables, not from prediction files.

| Agreement | Fixes |
| --- | --- |
| Schema | Levels, membership, and rollup |
| Data and inputs | Splits, exclusions, context budget |
| Experiments | 48 fine-tunes, prompt matrix, seeds |
| Evaluation | Micro-F1, retention, document bootstrap |
| Files and records | What is kept and what is not overwritten |

DrugProt prior use for tuning is unconfirmed. Human review and the external API are not part of this tree.
