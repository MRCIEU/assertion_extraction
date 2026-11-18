# CPU checks

Tests the schema rules, the parser, resume, and scoring without loading a model.

Method: Pytest against the package. Synthetic rows are marked as synthetic. A passing test is not a model score.

| File | Covers |
| --- | --- |
| test_study_rules.py | Schema and split rules |
| test_round5_parse.py | Legal, empty, and illegal answers |
| test_round6_resume.py | Intact batches and a second process record |
| test_round2.py through test_round4_cpu.py | Earlier CPU contracts still in force |

Gold inside a prediction file is ignored. The candidate table is the gold source.
