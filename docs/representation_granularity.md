# Information in the existing entity and predicate levels

The existing levels are ordered experimental conditions. A later level does not, by that order alone, add a fixed quantity of information. This note measures the category distinctions that the frozen levels actually introduce. It uses the BioRED training relation labels as the reference distribution for BioRED and BC8, and the DrugProt training relation labels for DrugProt. Positive gold relation labels are the observations. Candidate pairs with no positive label are counted separately and are left out of the predicate distribution.

The calculations use the study's add-one smoothing on native labels. A broader category receives the sum of its members' probabilities. Information content uses the natural logarithm already used by the retention score. Figures and the tables below state bits, obtained by dividing by ln 2. Entropy describes the label distribution supplied by a level. It does not measure how much of a scientific finding is preserved, and it does not measure what a model infers from a name or a sentence.

The detailed counts, the candidate-identifier audit, and the figures are in `/home/b5ac/freddieyu.b5ac/projects/project_1/studies/schema_granularity_v1/reports/representation_granularity/`. The prepared versions are `biored_candidates_v2`, `bc8_candidates_v2`, and `drugprot_candidates_v1`. The formal predictions are those in `formal_20261115`. The schema files are `configs/schemas/biored.yaml` and `configs/schemas/drugprot.yaml`. This measurement does not change those files, the splits, or the saved predictions.

The researcher authorised these measurements, the read-only prediction check, and this document. That authorisation does not extend to training, generation, ontology acquisition, or a source-level evaluation. Candidate passages prepared under a later authorisation are described in `docs/source_references.md`.

## Structural properties

Every native positive label has one group at each predicate level. The groups at a level are a partition of the native labels. Each finer group is a subset of its coarser group, and mapping through the intermediate levels agrees with the direct mapping. `No_Relation` is outside the positive hierarchy. These statements were checked automatically for the BioRED and DrugProt schemas. Both checks returned no structural error.

The stored direction field is uniform inside every group except `Any_Relation`. That group contains positive, negative, and nonpolar native labels, and the existing direction rule assigns it no single direction. The name therefore does not add a positive or a negative claim.

Other objections are judgements about the guidelines, not failures of the partition:

- BioRED `Other_Relation` and `Remaining_Relation` collect relations that the BioRED guideline defines separately. `Association` remains the guideline's residual relation, and `Bind` remains its physical-interaction relation.
- DrugProt `Agonism` combines `AGONIST`, `AGONIST-ACTIVATOR`, and `AGONIST-INHIBITOR`. The ChemProt guideline gives the last two opposite effects on a receptor. The schema stores all three directions as nonpolar, so the group name adds no direction field, while the guideline distinction is lost.
- `Metabolic_relation` names a family of enzymatic roles. `Other_Relation` in DrugProt collects the labels that are neither the explicit up-regulation pair nor the explicit down-regulation pair. Antagonism stays outside down-regulation, as the frozen mapping requires.

These groupings were not changed.

## Predicate information

BioRED training contributes 4,174 positive labels and 17,581 unannotated candidates, from 21,755 candidates. BC8 contributes 6,018 positive labels and 22,716 unannotated candidates. Its probabilities still come from the BioRED training counts, so its entropy equals the BioRED entropy. DrugProt training contributes 13,749 positive labels from 13,596 candidates, and 437,791 unannotated candidates. A DrugProt candidate with two labels contributes two observations.

| Corpus | Level | Positive categories | Entropy (bits) | Fraction of native entropy | Schema ceiling C |
| --- | --- | ---: | ---: | ---: | ---: |
| BioRED reference | P1 | 1 | 0 | 0 | 0 |
| BioRED reference | P2 | 3 | 1.424 | 0.853 | 0.923 |
| BioRED reference | P3 | 5 | 1.637 | 0.981 | 0.996 |
| BioRED reference | P4 | 8 | 1.668 | 1 | 1 |
| DrugProt training | P1 | 1 | 0 | 0 | 0 |
| DrugProt training | P2 | 3 | 1.482 | 0.489 | 0.550 |
| DrugProt training | P3 | 7 | 2.387 | 0.788 | 0.793 |
| DrugProt training | P4 | 13 | 3.028 | 1 | 1 |

P1 retains no distinction among positive native labels. P4 retains the native entropy exactly. The two fractions in the last columns are different quantities. The entropy fraction compares the grouped distribution with the native distribution. The schema ceiling C is the existing per-relation retention score when every fine label is mapped to its parent. Neither quantity is a reading of the source text.

Most of the BioRED entropy is already present at P2. The step from P2 to P3 adds 0.213 bits, and the step from P3 to P4 adds 0.031 bits. DrugProt continues to add information at the finer steps: 1.482 bits from P1 to P2, 0.905 from P2 to P3, and 0.641 from P3 to P4. The horizontal axis of the accompanying figures is the sequence of experimental conditions. Equal spacing on that axis is not a claim about scientific meaning.

### Where the representation actually changes

A relation changes when the set of native labels in its group becomes smaller. A repeated display name, such as `Positive_Correlation` at P4, P3, and P2, is the same representation.

| Transition | BioRED training | BC8, BioRED probabilities | DrugProt training |
| --- | ---: | ---: | ---: |
| P1 to P2, more specific | 4,174 (1) | 6,018 (1) | 13,749 (1) |
| P2 to P3, more specific | 2,323 (0.557) | 3,085 (0.513) | 5,983 (0.435) |
| P2 to P3, unchanged | 1,851 (0.443) | 2,933 (0.487) | 7,766 (0.565) |
| P3 to P4, more specific | 73 (0.017) | 197 (0.033) | 10,657 (0.775) |
| P3 to P4, unchanged | 4,101 (0.983) | 5,821 (0.967) | 3,092 (0.225) |

In BioRED, the unchanged relations from P2 to P3 are the positive and negative correlations. The 73 relations that become more specific from P3 to P4 are Comparison, Cotreatment, Drug_Interaction, and Conversion. In DrugProt, the up-regulation and down-regulation groups are unchanged from P3 to P2. The large P3-to-P4 change comes from splitting agonism, the metabolic roles, and the explicit regulation pairs. Direct regulation, antagonism, and part-of are already single native labels at P3.

On the evaluation splits, with the same reference probabilities, the BioRED test proportions are 1, 0.574, and 0.020. The DrugProt evaluation proportions are 1, 0.460, and 0.752. The training and evaluation proportions are close. The evaluation counts were not used to choose a level.

## Entity-type information

E0 supplies the mention, its context, and the target role. It is the absence of an explicit type. It is not given an entropy, and it is not treated as the root of an ontology. E2 is the six BioRED types. E1 displays gene or gene product, chemical, and sequence variant as `Molecular_Entity`. Disease or phenotypic feature, organism taxon, and cell line keep their names. The same add-one rule is applied to the six native types, and an E1 probability is the sum of its members.

| Corpus and unit | Observations | E2 entropy (bits) | E1 entropy (bits) | E1 / E2 | Share made more specific at E2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| BioRED training mentions | 13,934 | 2.151 | 1.322 | 0.615 | 0.625 |
| BioRED training concepts | 4,496 | 2.217 | 1.352 | 0.610 | 0.613 |
| BC8 mentions | 16,008 | 2.191 | 1.268 | 0.579 | 0.662 |
| BC8 concepts | 5,522 | 2.267 | 1.239 | 0.546 | 0.672 |

A normalised concept is one non-empty concept identifier in one document. BioRED training has 31 mentions without such an identifier; they are outside the concept count. No concept was attached to two native types. BC8 has four mentions, and two concepts, whose stored type is `Organism`. They are outside the six-type calculation and are not renamed. None of them is an argument of a BC8 candidate.

Every legal BioRED pair contains at least one of the three types that E1 merges. In the training candidates, 7,997 of 21,755 have one target that becomes more specific at E2 and 13,758 have two. The corresponding BC8 counts are 11,899 and 16,835, with none in either set left unchanged on both sides. Argument position follows the stored candidate order. In BioRED training, the first argument becomes more specific in 19,433 candidates and the second in 16,080.

These figures describe the type token supplied to the model. They do not describe the type information available from the mention string.

DrugProt has no E0–E2 comparison, and none was created. Its training mentions are 36,902 chemicals and 34,608 genes. The gene total combines 22,694 `GENE-Y` and 11,914 `GENE-N` annotations. The model input uses the unified gene token. The prepared file keeps the original value. DrugProt stores local mention identifiers, not normalised biological concept identifiers.

## Ontology-derived types

No UMLS or MeSH distribution is present in the project storage. The E3 table has a header and no mapping rows. A finer entity representation would require, at least, a licensed UMLS release for semantic types and concept identifiers, a MeSH tree release for descriptor hierarchy, and separate treatment of NCBI Gene, dbSNP or HGVS variant identifiers, Cellosaurus, and NCBI Taxonomy. BioRED already stores some MeSH descriptor and supplemental-concept identifiers. Those strings are not a hierarchy.

These resources do not form one ladder of granularity:

- A UMLS semantic type is a broad category, not a depth in a concept tree and not a relation label.
- A MeSH concept can have several tree numbers and therefore several depths.
- A pharmacological action is a role. The existing E3 rule forbids treating it as evidence that the mention is a drug.
- A gene-database class is not the molecular form mentioned in the text.
- The spelling of a variant identifier is not a complete variant type.
- Several of these properties can disclose the relation being predicted. A chemical class defined by inhibition, or a pharmacological action, must not be supplied as the entity type when inhibition is the label to be predicted.

Multiple valid types would have to be retained or marked as a conflict. An entity with no supported finer type would stay at its native type. Unequal tree depth would not be used as an experimental level. No such mapping was constructed.

## Reusable predictions

A condition is accepted only when every expected candidate identifier has one adopted record, no unexpected identifier appears, the document, split, and dataset agree with the candidate table, and the stored label set follows the existing status rules. A rejected, partial, or unparseable answer can remain inside an accepted condition. A file with the right number of lines is not sufficient. The generation check uses the existing adopted-answer rule and reparses the stored answer. For every evaluation candidate, the marked context and the entity mentions are present: the missing-context and missing-mention counts are zero in BioRED test, BC8, and DrugProt evaluation.

| Accepted conditions | Count | Identifier coverage | Answers that are not a single valid label set |
| --- | ---: | --- | --- |
| BiomedBERT, DrugProt, native, P1–P4, three seeds | 12 | 116,887 evaluation candidates each | None; all 1,402,644 records are valid |
| BiomedBERT, BioRED, E0, P1, seed 13 | 1 | 7,047 test and 28,734 BC8 candidates | None; all 35,781 records are valid |
| Qwen3-32B, BioRED test, E0–E2, P1–P4 | 12 | 7,047 each | 378 rejected across the 12 conditions |
| Qwen3-32B, BC8, E0–E2, P1–P4 | 12 | 28,734 each | 1,346 rejected |
| Qwen3-32B, DrugProt evaluation, P1–P4 | 4 | 116,887 each | 1,646 partial, 62 rejected, 9 unparseable |
| Qwen3-8B, BioRED test, E0–E2, P1–P4 | 12 | 7,047 each | 421 rejected |
| Qwen3-8B, BC8, E0–E2, P1–P4 | 12 | 28,734 each | 2,469 rejected |
| Qwen3-8B, DrugProt evaluation, P1 and P2 | 2 | 116,887 each | 10 rejected |

Sixty-seven conditions are accepted. Six examined outputs are not. Four are the invalid duplicate-writer DrugProt directories; the replacement runs are the accepted BiomedBERT outputs. Qwen3-8B DrugProt P3 has 14,704 evaluation candidates without an adopted answer. Qwen3-8B DrugProt P4 has 97,895 without an adopted answer. The 35 BioRED fine-tuning runs that were never started have no prediction file. They are not accepted conditions.

Seven document identifiers occur in both BioRED and DrugProt. Their splits are listed in `shared_documents.json` in the measurement directory. A future cross-corpus sensitivity population can exclude those identifiers. The stored splits were not changed. No document identifier is repeated inside one corpus.

## What this measurement does not decide

The entropy and the retention ceiling describe label categories. They do not establish that a prediction is supported by the source, and they do not measure linguistic certainty. Evidence spans are still absent from the relation records. DrugProt's previous use for tuning remains unconfirmed. The shared-document exclusion is defined and not applied. No finer entity inventory was approved.
