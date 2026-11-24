# Stage 00 assessment of the existing granularity study

This assessment records which existing data, representations, predictions, and evaluation procedures can be reused for a later study of semantic completeness and source-grounded support. The inspected implementation is commit `5e86545eece2aea3280fd9d37c59cd3d84717032` of `MRCIEU/assertion_extraction`. Prepared data and formal outputs remain in `/home/b5ac/freddieyu.b5ac/projects/project_1`. The counts below were produced by `scripts/stage00_inventory.py`; the machine-readable result is `docs/stage00_granularity_assessment/inventory.json`.

The present assessment does not establish the semantic validity of any extracted relation. It does not change candidate populations, data splits, hierarchies, or formal outputs.

The researcher authorised creation and submission of the audit code and this English document only. That authorisation does not extend to training, generation, source-level annotation, or a new experiment matrix.

## Purpose and comparison

The existing study asks how entity-type information and relation-category granularity change biomedical relation extraction when the entities are given. A coarse label can be requested directly, or a native prediction can be mapped onto that coarse label. The later question is different. It asks whether a representation preserves the meaning of the original scientific finding, and whether the extracted relation is supported by the source text. Linguistic certainty is a property of that finding. The measures already implemented compare a prediction with a native relation label.

## Dataset inventory and annotation scope

The current prepared versions are `biored_candidates_v2`, `bc8_candidates_v2`, and `drugprot_candidates_v1`. Earlier BioRED and BC8 candidate tables remain on disk and are superseded. The formal freeze hashes are `53ac88504980316c46794215a5d91f6386508496de8ebf0dc57fa6e6dd8e3e25`, `68241a5120c19cb98b534e35e17f2b7a959fde125ce836c4e25617ac62a76b36`, and `75e33dd10e46d201b932872b076a66307f2bee482605c2c8c833facff896cb25`.

BioRED is the NCBI release `BIORED_2022-04-07_sha256-c3032230bd89`, from the official PubTator train, development, and test files. BC8 is the gold-standard portion of BioREDirect, release `BC8_BIOREDIRECT_2025-01-22_sha256-29aa0b2060f9`. DrugProt is Zenodo record 5119892. Its guideline concepts are recorded under DOI `10.5281/zenodo.4957137`; that concept resource was not required for the counts below.

The annotation unit in BioRED and BC8 is a document-level relation between two normalised concepts. The prediction unit is one candidate concept pair. DrugProt annotates a relation between a chemical mention and a gene or protein mention. Its prediction unit is one within-document chemical–gene mention pair, with the chemical first. No candidate population or split was revised for this assessment.

| Dataset | Split | Documents | Entity mentions | Normalised concepts | Candidate pairs | Included positive relations | Role in the existing study |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BioRED | Train | 400 | 13,934 | 4,515 | 21,755 | 4,174 | Fine-tuning and development |
| BioRED | Dev | 100 | 3,649 | 1,237 | 6,334 | 1,161 | Model selection |
| BioRED | Test | 100 | 3,644 | 1,277 | 7,047 | 1,163 | Previously examined evaluation |
| BC8 | External | 400 | 16,012 | 5,537 | 28,734 | 6,018 | Previously examined external evaluation |
| DrugProt | Train | 2,800 | 71,510 | Unavailable | 451,387 | 13,749 | Fine-tuning |
| DrugProt | Dev | 700 | 18,019 | Unavailable | 118,778 | 3,525 | Threshold and epoch selection |
| DrugProt | Eval | 750 | 18,858 | Unavailable | 116,887 | 3,761 | Official development set, used here as evaluation |
| DrugProt | Unlabelled | 10,750 | 310,805 | Unavailable | 2,186,385 | 0 | Held out of scoring |

Included positive relations are the gold labels attached to candidates. BioRED has one positive label on each positive candidate. Its relation file contains 6,503 records: 4,178 train, 1,162 development, and 1,163 test. Five of those records are outside the candidate table. DrugProt has 21,035 relation records, all included. Because one pair may carry more than one label, the positive candidate counts are 13,596, 3,477, and 3,731. The additional labels number 153, 48, and 30.

Native BioRED and BC8 entity categories are GeneOrGeneProduct, DiseaseOrPhenotypicFeature, ChemicalEntity, SequenceVariant, OrganismTaxon, and CellLine. BC8 also contains four mentions whose stored type is Organism. Native BioRED relation categories are Positive_Correlation, Negative_Correlation, Association, Bind, Comparison, Cotreatment, Drug_Interaction, and Conversion. BC8 has no Drug_Interaction record in the included labels. DrugProt’s native relation categories are the 13 official labels. Its training entities use CHEMICAL, GENE-Y, and GENE-N; the official development entities use CHEMICAL and GENE. The prepared input type unifies the gene labels as GENE and retains the original type in `entity_type_original`.

Document identifiers are unique within each dataset, including across its splits. No document is shared by BioRED and BC8, or by BC8 and DrugProt. Seven PubMed identifiers occur in both BioRED and DrugProt:

| PubMed identifier | BioRED split | DrugProt split |
| --- | --- | --- |
| 15069170 | Dev | Dev |
| 16252083 | Test | Unlabelled |
| 16840830 | Test | Train |
| 16867246 | Dev | Dev |
| 18768591 | Train | Unlabelled |
| 25119790 | Dev | Unlabelled |
| 26102294 | Train | Unlabelled |

BioRED test and BC8 are recorded as previously examined evaluation data, so neither is an untouched first-contact set. Whether DrugProt was previously used for tuning is unresolved: `prior_tuning_use_confirmed` is false. The DrugProt evaluation split can therefore be used with the fixed split, and it cannot yet be described as an untouched confirmation set. The unlabelled DrugProt background has no relation labels and was not scored.

### Exclusions and available source fields

| Dataset | Excluded records | Reasons |
| --- | ---: | --- |
| BioRED | 68 | 63 mentions without a concept identifier; 4 self-relations; 1 unsupported gene–variant pair |
| BC8 | 38 | 20 mentions without a concept identifier; 13 relations whose concept is missing; 4 self-relations; 1 unsupported gene–organism pair |
| DrugProt | 0 | No prepared exclusion row |

These records stay outside the candidate tables. The data summary still marks BioRED and BC8 `formally_trainable: false` because those annotation problems were excluded rather than resolved. The formal runs that exist used the resulting candidate tables and did not put the excluded records back.

Original title and abstract text, mention strings, and character offsets are present for every prepared mention. BioRED and BC8 concept identifiers are present for the mentions that entered the candidate tables. DrugProt stores local mention identifiers such as `T1`, together with offsets and mention text. It does not store an external gene or chemical concept identifier in the prepared entity rows. No UMLS or MeSH distribution was found in the project data directory or the code repository.

A relation record contains the document, the two arguments, the label, and, for BioRED and BC8, a novelty value. It does not contain a source passage, a sentence identifier, or a character span for the finding. Novelty takes the values Novel (4,532) and No (1,971) in BioRED, and Novel (3,683) and No (2,353) in BC8. DrugProt novelty is null. Novelty is a separate annotation from linguistic certainty.

## Existing entity and predicate levels

Entity information has three BioRED and BC8 levels. E0 supplies the mention text, its context, and the target role, and withholds an explicit type. E1 displays GeneOrGeneProduct, ChemicalEntity, and SequenceVariant as Molecular_Entity and retains DiseaseOrPhenotypicFeature, OrganismTaxon, and CellLine. E2 displays the six original BioRED types. DrugProt uses one native input level, with the gene labels unified as GENE. Database identifiers are not given to the model. E0 is the absence of an explicit type. Molecular_Entity is the broad category, and it applies only at E1 to three of the six original types.

| Dataset | Level | Categories | What changes from the previous level |
| --- | --- | --- | --- |
| BioRED, BC8 | P1 | Any_Relation | Every positive native label shares one category |
| BioRED, BC8 | P2 | Positive_Correlation, Negative_Correlation, Other_Relation | The two correlation labels remain separate |
| BioRED, BC8 | P3 | Positive_Correlation, Negative_Correlation, Association, Bind, Remaining_Relation | Association and Bind become separate; four smaller labels share Remaining_Relation |
| BioRED, BC8 | P4 | Eight native labels | Comparison, Cotreatment, Drug_Interaction, and Conversion become separate |
| DrugProt | P1 | Any_Relation | Every positive native label shares one category |
| DrugProt | P2 | Upregulation_or_activation, Downregulation_or_inhibition, Other_Relation | Only the explicit up-regulation and down-regulation groups remain separate |
| DrugProt | P3 | Seven named groups | Agonism, antagonism, direct regulation, metabolic relation, and part-of become separate |
| DrugProt | P4 | Thirteen official labels | Members of the P3 groups become the official labels |

The P4 to P3 to P2 mapping is deterministic and is checked against the direct P4 to P2 mapping. A finer level adds a distinction only for the categories it splits. A Positive_Correlation or Negative_Correlation instance has the same category at P4, P3, and P2. Association and Bind change only between P3 and P2. The four smaller BioRED labels change between P4 and P3. DrugProt agonist labels remain outside up-regulation, and antagonist remains outside general down-regulation, by the written grouping.

The BioRED grouping follows the study’s manual grouping of the original annotation labels. It is not a newly inferred ontology. The DrugProt grouping is likewise a manual grouping of the official labels. The original annotation guidelines supply the native labels; they do not by themselves define P3 or P2.

### Availability of finer entity-type mappings

| Proposal | State | Source | Coverage | Limitation |
| --- | --- | --- | --- | --- |
| E0, E1, E2 | Implemented | `configs/schemas/entities.yaml` | All BioRED and BC8 mentions in the prepared tables | E1 collapses three molecular types and leaves the other three unchanged |
| DrugProt native type | Implemented | Prepared entities and the DrugProt input rule | All prepared DrugProt mentions | GENE-Y and GENE-N are unified for input; the original value is retained |
| E3 subtype mapping | Proposed | `src/schema_study/e3.py` and an empty template | Header only; zero mapping rows | The checker rejects three forbidden shortcuts and records conflicts, and it assigns no subtype |
| UMLS or MeSH subtype resource | Unavailable | No file in the inspected project storage | None | No coverage, conflict, or version can be calculated |

No classification finer than E2 has mapping records. This assessment does not construct one.

## Experiment completion and prediction availability

The formal run is `formal_20261115`. Its manifest lists 288 tasks: 48 fine-tuning runs and 240 generation shards. The manifest status is stale for the 16 generation jobs that were cancelled while still marked running. Completion below uses `prediction_complete.json` for fine-tuning and the adopted attempt count for generation. A shard is complete only when the adopted answers cover its frozen request count. Partial shards are incomplete conditions.

| Experiment | Dataset and version | Entity level | Predicate level | Model | Learning setting | Route | Seed | Completion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fine-tuning | BioRED v2 | E0 | P1 | BiomedBERT | Fine-tuned, three-seed design | Direct | 13 | Complete; test and BC8 predictions, 35,781 rows |
| Fine-tuning | BioRED v2 | E0–E2 | P1–P4, except the row above | BiomedBERT | Fine-tuned | Direct | 13, 42, 2026 | 35 runs planned and not executed |
| Fine-tuning | DrugProt v1 | Native | P1–P4 | BiomedBERT | Fine-tuned | Direct | 13, 42, 2026 | 12 of 12 complete |
| Fine-tuning | DrugProt v1 | Native | P2 seeds 42 and 2026; P3 seeds 13 and 2026 | BiomedBERT | Fine-tuned | Direct | As listed | Four earlier attempts are invalid duplicate-writer outputs and are excluded; the replacement runs are the completed rows above |
| Prompted generation | BioRED v2 test and BC8 v2 | E0–E2 | P1–P4 | Qwen3-32B | Frozen instruction, greedy decoding | Direct | None | Complete: 12 BioRED shards and 48 BC8 shards |
| Prompted generation | DrugProt v1 eval | Native | P1–P4 | Qwen3-32B | Frozen instruction | Direct | None | Complete: 60 shards |
| Prompted generation | BioRED v2 test and BC8 v2 | E0–E2 | P1–P4 | Qwen3-8B | Frozen instruction | Direct | None | Complete: 12 BioRED shards and 48 BC8 shards |
| Prompted generation | DrugProt v1 eval | Native | P1–P2 | Qwen3-8B | Frozen instruction | Direct | None | Complete: 30 shards |
| Prompted generation | DrugProt v1 eval | Native | P3 | Qwen3-8B | Frozen instruction | Direct | None | Incomplete: 7 of 15 shards have full adopted answers; 8 shards are partial |
| Prompted generation | DrugProt v1 eval | Native | P4 | Qwen3-8B | Frozen instruction | Direct | None | Not complete: 8 shards are partial and 7 were never started |
| Five prompt controls | BioRED | E2 | Renamed or random groupings | Qwen3-32B | Frozen instruction | Direct | Planned seed 42 for the random groupings | Draft configuration only; not executed |
| E3 and down-sampling | — | — | — | — | — | — | — | Configured or templated only; not executed |

Model revisions recorded in the freeze are BiomedBERT `e1354b7a3a09615f6aba48dfad4b7a613eef7062`, Qwen3-8B `b968826d9c46dd6066d109eabc6255188de91218`, and Qwen3-32B `9216db5781bf21249d130ec9da846c4624c16137`. The formal prompt file is `prompts/relation_extraction_v1.txt`. Completed prediction rows contain stable candidate identifiers. Direct predictions exist for every completed cell above. Native P4 predictions that can later be abstracted exist for all completed Qwen3-32B conditions, all completed Qwen3-8B BioRED and BC8 conditions, and all 12 BiomedBERT DrugProt runs. They do not exist for BiomedBERT BioRED, or for Qwen3-8B DrugProt P4.

The original marked text is available in the context files and in the frozen generation shard requests. A saved fine-tuning prediction row does not itself copy that text; it can be joined to the context by candidate identifier. The one completed BioRED prediction file joins test and BC8 and contains 35,781 rows, equal to 7,047 plus 28,734. Evaluation files exist for 39 direct scoring units and 18 Qwen3-32B roll-up units, plus the two training-count files used by the retention score. That is a subset of the completed predictions. Development, smoke, and task-suitability directories are also present. They were not accepted here as formal predictions and were not re-scored.

## Existing evaluation measures

| Measure | What it measures | Calculation | Denominator | Reuse for a source-level study |
| --- | --- | --- | --- | --- |
| Positive micro-precision, recall, and F1 | Agreement with the gold labels at the requested predicate level | Candidate-level label-set counts are summed. A wrong BioRED label contributes one false positive and one false negative. No_Relation is not a positive class | All candidates in the scored population. Precision is undefined when there is no positive prediction; recall is undefined when there is no positive gold label | Retain for label agreement. It does not measure support by the source finding |
| Per-class and macro-F1 | The same agreement for each positive category | Macro-F1 averages the fixed positive category list, including categories with no support | Class support is true positives plus false negatives for that category | Retain |
| Exact candidate accuracy and failure counts | Whether the predicted label set matches and whether the output is usable | An empty legal list is a valid no-relation. Missing, rejected, and illegal outputs remain in the population | All candidates | Retain. A format failure remains a failure |
| Label information retention | How much of the native fine label is distinguished by the predicted group | Add-one probabilities come from the training counts. Retention is the information content of the predicted group divided by that of the fine label when the fine label belongs to the group; otherwise it is zero. DrugProt uses the highest covering group | Every positive fine gold relation, including misses and failures | Retain as a label-retention measure. It compares the prediction with the native label |
| Frequency-free retention | The same distinction without training frequencies | `1 − log(|G|) / log(K)` when the fine label belongs to the predicted group; otherwise zero | The same positive fine gold relations. P1 is zero because it makes no category distinction | Retain on the same terms |
| Schema-induced loss | Information removed by using the coarser label set | `1 − C`, where `C` is the mean retention obtained by mapping every fine gold label to its parent at that level | The same positive fine gold relations | Retain. It describes the label inventory, not the scientific finding |
| Additional prediction loss | Information lost by the model beyond the schema mapping | `C − R`, where `R` is the mean retention of the actual prediction | The same positive fine gold relations | Retain for label prediction. It becomes available only where a completed prediction and the training counts exist |
| Polarity preservation | Whether an explicit positive or negative direction is kept, reversed, coarsened, missed, or placed in conflict | Gold relations without an explicit polarity are omitted. A gold pair that already contains both directions is compared as one set | Gold relations with at least one explicit polar direction | Retain for the label direction. It is not a judgement of the finding’s certainty |
| Paired document-level bootstrap interval | Document-resampling uncertainty of a difference between two saved conditions | 2,000 draws, seed 100, percentile 95 interval. Documents are drawn once and shared by the comparison; each seed is recomputed and then averaged | The documents in the scored population. The interval is conditional on the saved outputs | Retain when both sides of a comparison have complete saved predictions. It does not resample training |

Label information retention and preservation of the original finding are different quantities. The former is defined from the native label and the predicted group. The latter would require a reference that states what the source finding said. No such reference is present, so the current measures can be retained without modification for their existing purposes. A claim about meaning, source support, or linguistic certainty would require additional source-level annotation.

## Availability of records for source-level evaluation

Completed predictions can be linked to documents and candidate arguments. BioRED and BC8 candidate identifiers contain the document and the two concept identifiers. DrugProt candidate identifiers contain the document and the two local mention identifiers. Gold relation records use the same document and argument identifiers. The join is reliable for the completed files inspected here. The link does not identify which sentence asserts the relation.

| Proposed record | Availability | Consequence |
| --- | --- | --- |
| Source finding associated with a relation | The full title and abstract are available. No relation has an evidence passage or sentence identifier | A reader can be shown the document and the marked mentions. The annotation does not identify one finding when several sentences could support the pair |
| Linguistic certainty of that finding | Not annotated. BioRED and BC8 novelty is a different field; DrugProt has no novelty value | A certainty measurement requires new annotation |
| Standardised natural-language expression of gold and predicted relations | Mention strings, entity information, and predicate labels are available. No reference sentence is stored | A later template can be formed from these fields. Multiple mention strings may belong to one concept, and the template would still not be a source finding |
| Whether the extracted relation is supported by the source | Text, offsets, and mentions are available. No support judgement is stored | The materials can be assembled for a future assessment. The existing labels do not answer the support question |
| How much of the original finding is preserved | The native label and the current retention score are available | The score preserves information about the label. Preservation of the finding is not measured |

On the evaluation candidate sets, the stored model input is the full marked document for 5,567 of 7,047 BioRED test candidates, 24,923 of 28,734 BC8 candidates, and 104,909 of 116,887 DrugProt evaluation candidates. The remaining inputs are shortened windows: 1,480 BioRED single windows; 3,746 BC8 single windows and 65 dual windows; 11,612 DrugProt single windows and 366 dual windows. A window is a property of the model input. It is evidence that some target mentions do not fit together inside the 512-token budget. Sentence boundaries are not annotated, so these counts are not counts of multi-sentence findings.

Unresolved cases are therefore:

- Every included relation lacks an identifiable source passage in the existing annotation. The counts are 6,498 BioRED relations, 6,018 BC8 relations, and 21,035 DrugProt relations.
- The same entity pair can be associated with more than one textual locus because a concept may have several mentions. The annotation does not choose among them.
- DrugProt has 231 candidate pairs with more than one positive label: 153 in training, 48 in development, and 30 in evaluation. BioRED and BC8 included candidates have one positive label each.
- Relations whose support crosses more than one sentence cannot be enumerated, because the release does not mark the supporting sentence.

## Components to retain, extend, or develop

| Component | Decision | Reason |
| --- | --- | --- |
| Prepared candidate tables, splits, and exclusion lists | Retain | The populations are fixed, and the counts above were verified from them |
| E0–E2 and P1–P4 mappings | Retain | They are deterministic and already cover the native labels |
| Completed direct predictions and completed P4 outputs | Retain | They are linked to candidate identifiers and do not need to be regenerated for a later abstraction |
| Incomplete Qwen3-8B DrugProt P3 and P4 shards, and the 35 unexecuted BioRED fine-tunes | Leave incomplete | Partial or planned outputs are not completed conditions |
| Invalid duplicate-writer directories | Retain as excluded records | The replacement DrugProt runs are the completed outputs |
| Micro-F1, label retention, schema loss, prediction loss, polarity, and paired bootstrap | Retain | Each measure has a defined denominator and does not require source annotation for its current interpretation |
| Finding-level evidence, linguistic certainty, source support, and preservation of the finding | Develop only under a later authorisation | The required reference annotations are absent |
| E3 or another finer entity classification | Do not develop from the current storage | There are no mapping rows and no ontology resource in the inspected storage |
| Five prompt controls and an external-model review | Leave undecided | Their display names, reviewers, and endpoint were never frozen, and they were not run |

## Checks

The inventory script completed and wrote `inventory.json`. It reproduced the published candidate totals: BioRED 35,136, BC8 28,734, and DrugProt 2,873,437. Document identifiers were unique within each dataset. The seven cross-dataset identifiers were recovered from the document tables. All 240 generation shards were compared with their frozen request counts: 217 shards contain a full set of adopted answers, 16 cancelled DrugProt Qwen3-8B shards are partial, and 7 shards have no attempt file. Thirteen fine-tuning directories contain `prediction_complete.json`, matching the one BioRED run and the 12 DrugProt runs. The completed BioRED prediction file has 35,781 rows. The E3 template contains only its header. No UMLS or MeSH file was found under the project data directory or the code repository.

## Decisions required before a source-level stage

1. A source passage is not stored with any relation. A later study needs an explicit decision to annotate evidence spans, or to accept that the full document and its mentions are the only available context.
2. Linguistic certainty is not in the current annotations. BioRED and BC8 novelty cannot stand in for it.
3. DrugProt’s previous use for tuning remains unconfirmed. One BioRED test document, PubMed 16840830, is also in the DrugProt training split, and BioRED test document 16252083 is in the unlabelled DrugProt background. Any joint analysis needs a decision about these seven shared documents. This assessment does not change the splits.
4. The formal matrix is incomplete. Source-level work can use the completed prediction files. It cannot treat the missing BioRED fine-tunes or the incomplete Qwen3-8B DrugProt P3 and P4 conditions as available results.
5. Human review size, reviewers, and any external model endpoint remain unset. Model output is not a reference annotation.
