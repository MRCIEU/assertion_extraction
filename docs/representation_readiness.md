# Representation readiness for the biomedical relation study

The existing study asks how entity-type information and predicate granularity change relation extraction when the target entities are given. Direct prediction at a stated level is compared with prediction at the native level followed by abstraction. The later scientific question is which distinctions can be recovered reliably, and whether those distinctions help identify uncatalogued relationships, contextual limits of existing assertions, and hypotheses that still require investigation. An uncatalogued relationship is a relationship absent from a chosen catalogue. It is not, by that absence, a previously unknown biological relationship.

This note records the methodological checks made before any further model experiment. It continues from repository revision `831ad5e0de8d37df5b1c18568e729f49cd095768` on `review/stage00-granularity-assessment`. Prepared collections, splits, native annotations, frozen predicate mappings, model inputs, and saved predictions are unchanged. No relation-extraction model was trained or run. The machine records are in `studies/schema_granularity_v1/reports/representation_readiness/`. The field `repository_revision_expected` in those records names the inspected base, not a later documentation commit.

## Prepared populations

The collections remain `biored_candidates_v2`, `bc8_candidates_v2`, and `drugprot_candidates_v1`. Document identifiers are unique within each corpus. The seven PubMed identifiers shared by BioRED and DrugProt, and their stored splits, are unchanged.

| PubMed identifier | BioRED split | DrugProt split |
| --- | --- | --- |
| 15069170 | dev | dev |
| 16252083 | test | unlabeled |
| 16840830 | test | train |
| 16867246 | dev | dev |
| 18768591 | train | unlabeled |
| 25119790 | dev | unlabeled |
| 26102294 | train | unlabeled |

There is no BC8 overlap. Within-corpus analyses keep these documents. A future cross-corpus sensitivity population excludes every document identifier that occurs in more than one corpus. BioRED test, BC8, and DrugProt evaluation have already been used for scoring. They are not untouched confirmation data.

## Source-reference file integrity

Every sentence, alignment, and passage file under `studies/schema_granularity_v1/reports/source_references/` was parsed to its final record. The alignment and passage files contain the expected candidate identifiers, with no duplicate or unexpected identifiers, and they agree with each other and with the written summary. An incomplete copy previously supplied for review was a transfer failure. The stored files were not regenerated.

| Corpus | Sentences | Documents with sentences | Alignments and passages | Documents with candidates | Parse errors | Population complete |
| --- | --- | --- | --- | --- | --- | --- |
| BioRED dev and test | 2,195 | 200 | 13,381 | 198 | 0 | yes |
| BC8 | 4,587 | 400 | 28,734 | 399 | 0 | yes |
| DrugProt dev and eval | 14,659 | 1,450 | 235,665 | 1,377 | 0 | yes |

The documents present in the sentence files and absent from the alignment files have no eligible candidate. That difference is not a missing identifier. Checksums and document-split counts are in `readiness.json`.

New derived JSON, JSONL, Parquet, and TSV writers now publish by writing a partial file and replacing the destination only after the write returns. An interrupted write leaves the previous completed file in place. This applies to later outputs. It does not rewrite the files verified above.

## Entity-type resources

E0, E1, and E2 remain the entity conditions. E0 supplies no explicit type and is a control, not the root of an ontology. E1 and E2 are unchanged. No additional experimental entity level is proposed.

UMLS was not available in the research storage, and no licence or credential was used. MeSH is not treated as a substitute for the UMLS Semantic Network. Public MeSH descriptors were read from `data/raw/mesh/nlm_2026/desc2026.gz`, release `desc2026`, obtained from the NLM XML distribution. The file contains 31,110 descriptor records and has SHA-256 `ccd4d0d33bebfd4c836a59e7dd0c635b1d4e20c4f5a6abc55e2eb0ba4c8716dd`. Supplemental concept records were not acquired. The descriptor file remains in research storage and is not part of the code repository.

MeSH tree numbers, pharmacological-action records, gene and variant identifiers, and cell-line accessions are different kinds of information. A deeper tree number is not assumed to be a more appropriate type. A descriptor may have several trees. Those paths are retained as alternative classifications. They are not collapsed into one hierarchy with the native BioRED types.

### Coverage

Counts below are unique concept identifiers in the prepared entity files, and separately the mentions that carry them. They are not the training document–concept observations used for the published E1 and E2 entropy. A concept identifier that joins several MeSH descriptors with `|` is read as several descriptors. The stored identifier is not rewritten. A concept is counted as mapped only when every descriptor part has a tree in this release.

| Corpus | Unique concepts | Concepts with a descriptor tree | Mentions | Mentions with a descriptor tree |
| --- | --- | --- | --- | --- |
| BioRED | 3,869 | 1,202 (31.1%) | 21,227 | 9,031 (42.5%) |
| BC8 | 3,459 | 976 (28.2%) | 16,012 | 5,543 (34.6%) |

| Native category | BioRED concepts mapped | BioRED concepts without a descriptor tree | BC8 concepts mapped |
| --- | --- | --- | --- |
| ChemicalEntity | 520 / 652 | 131 supplemental, 1 missing | 420 / 598 |
| DiseaseOrPhenotypicFeature | 682 / 778 | 89 supplemental, 7 OMIM | 554 / 603 |
| GeneOrGeneProduct | 0 / 1,643 | 1,643 numeric | 2 / 1,201 |
| SequenceVariant | 0 / 678 | dbSNP, variant notation, or other | 0 / 974 |
| CellLine | 0 / 72 | Cellosaurus accessions | 0 / 50 |
| OrganismTaxon | 0 / 46 | numeric | 0 / 32 |

BC8 also has one numeric organism concept and four organism mentions. They are not candidate arguments. DrugProt has 419,192 mention identifiers, all of the local form used by that corpus. None is a normalised concept identifier. Linking those mentions to an external vocabulary is a separate problem from assigning a type. DrugProt remains a predicate comparison. No DrugProt entity-level condition is proposed.

### What a MeSH tree would change, and why it is not a new level

Among the 1,202 mapped BioRED concepts, 769 have more than one tree number, 48 reach more than one top-level MeSH category, and 72 have a top-level category outside the category a reader might expect from the native type alone. The corresponding BC8 figures are 637, 38, and 56 among 976 mapped concepts. Three BioRED disease concepts are composite descriptor identifiers, including learning and memory impairments, T-cell leukaemia or lymphoma, and ventricular tachycardia and fibrillation. A single displayed type would require a selection rule. No such rule is defined, because the coverage is not sufficient for a matched entity condition.

Pharmacological-action records are present for 311 mapped BioRED concepts and 237 mapped BC8 concepts. A pharmacological class defined by an action on a target can overlap a relation label. Those records are excluded from any proposed type. A gene database class is not read as the molecular form in the source text. Variant notation and dbSNP prefixes are not read as a complete variant category. Numeric identifiers are not retyped from their syntax.

The eligible population for a future comparison that required a finer type would be the subset of pairs whose arguments both received a defensible additional type. That subset was not formed. The full candidate populations of the existing experiments stay as they are. Mappings in this note are deterministic lookups. They have not had independent human review, and they are not reported as validated types.

## Predicate representations

P1–P4 remain the primary predicate conditions for BioRED and DrugProt. The native labels are the finest supervised categories. No finer relation is inferred. `No_Relation` remains outside the positive hierarchy. It records the absence of an annotated positive relation in the dataset’s scope. It does not establish that the entities have no biological relationship.

The category readings in `docs/source_references.md` are unchanged. The following interpretations remain unresolved, and they are not closed here.

- BioRED `Comparison` has no recovered prose definition of its own.
- The Conversion instruction conflicts with an Association instruction that also covers chemical modification.
- Cotreatment and Drug_Interaction are distinguished in the heading, but the recovered text does not decide an abstract that describes both.
- `Association` is a residual set of permitted annotations, not one biological mechanism.
- DrugProt direct and indirect regulation remain separate at the native level. The current up and down groups join only the activator with the indirect upregulator, and the inhibitor with the indirect downregulator.
- Agonist labels mix an increase, a decrease, and an unstated direction. They stay outside the up-regulation group. Antagonist stays outside general down-regulation.
- The metabolic group includes transporter products as well as substrates and products.
- `Other_Relation` and `Remaining_Relation` are residual groupings. A structurally valid parent is not thereby a biological generalisation.

### Alternative groupings

Two alternative partitions were defined from the native inventories. They do not replace P1–P4, and they were not chosen by extraction performance. Each is one partition, so the reported retention is native entropy against that partition. It is not a chain of equal steps.

`biored_relation_family` asks whether relation family, rather than the current correlation grouping, changes the label information. Membership is directional correlation (positive and negative together), physical binding, residual association, coadministration (cotreatment and drug interaction), chemical transformation, and unresolved comparison. Direction inside the correlation pair is omitted. Comparison remains an unresolved singleton.

`drugprot_route` asks whether the route of the interaction, rather than the current up and down groups, changes the label information. Membership is direct binding or regulation, indirect regulation, receptor modulation (the agonist labels and antagonist), metabolic role, and structural inclusion. Direction and the direct–indirect distinction inside the first group are omitted.

Information uses the existing training counts, natural logarithms, and add-one smoothing. A relation is unchanged when its alternative group contains only its native label.

| Representation | Positive categories | Training entropy (bits) | Fraction of native entropy | Training relations whose group is broader | Training relations unchanged |
| --- | --- | --- | --- | --- | --- |
| BioRED P2, existing | 3 | 1.424 | 0.853 | published previously | published previously |
| BioRED relation family | 6 | 1.226 | 0.735 | 1,893 / 4,174 | 2,281 / 4,174 |
| DrugProt P2, existing | 3 | 1.482 | 0.489 | published previously | published previously |
| DrugProt P3, existing | — | 2.387 | 0.788 | published previously | published previously |
| DrugProt route | 5 | 1.879 | 0.620 | 13,083 / 13,749 | 666 / 13,749 |

The BioRED family partition has more category names than P2 and retains less native entropy, because it merges the two directional correlations that P2 keeps apart. Those two labels account for 1,851 of the 1,893 broadened training relations. The DrugProt route partition retains more native entropy than P2 and less than P3. Only `PART-OF` is unchanged, 666 training relations. The extra retention relative to P2 is the retention of route distinctions, not evidence that the route names are the better biological generalisation.

The existing random-merge control still draws three BioRED partitions of sizes 1, 1, and 6 at seed 42. Those sizes match BioRED P2. The label frequencies are not matched. The control has no biological names. It does not apply to DrugProt, whose P2 sizes are not 1, 1, and 6.

Saved native predictions can be mapped onto either alternative. That mapping is post-prediction abstraction. It does not estimate how a model would score if it were trained or prompted to emit the alternative labels directly. Direct prediction of either alternative would be a new experiment. It is not part of the proposed protocol.

## Sentence boundaries

The recorded splitter, `abbreviation_decimal_v1`, remains the baseline. It recovers the stored sentence spans and aligns every target mention in the existing passage build. That result does not establish that every boundary is correct. The comparison method, `terminal_punctuation_v1`, splits at `.`, `!`, or `?` before an uppercase letter, a digit, or opening punctuation, and it still protects a digit–dot–digit sequence. It does not use the abbreviation list. Its sentence identifiers carry the method version, so they are not reused for the recorded spans.

On the 100 BioRED development documents, 1,112 sentence-end offsets agree. The terminal method proposes 85 additional ends. The recorded method proposes none that the terminal method lacks. Boundary precision, recall, and F1 are not reported, because there is no adjudicated reference. Agreement between the methods is not evidence that either boundary is correct.

| Example | Recorded sentences | Terminal-punctuation sentences | Human reference |
| --- | --- | --- | --- |
| `p.R229G` inside a sentence | 1 | 2 | unfilled |
| `120 min. After` | 1 | 2 | unfilled |
| `ECA3g.79544206A>T` | 2 | 2 | unfilled |
| `7. 8-Epi-PGF2` | 2 | 2 | unfilled |
| `0.5` | 1 span kept | 1 span kept | unfilled |

The first two rows are the informative disagreement. The recorded rules keep the protein-variant token together and keep `min` from ending a sentence. The terminal rule splits both. The genomic coordinate and the numbered chemical are already split by the recorded rules, including the BC8 mention that crosses two sentences. The decimal is kept by both. Which of the 85 additional ends are true sentence breaks is unresolved. The selected segmentation method is therefore unresolved. The stored sentence records stay on `abbreviation_decimal_v1`.

## Candidate passages and findings

Passage selection is unchanged in its rules: the whole document, each sentence that contains both targets, each shortest sentence interval, and that interval plus one neighbouring sentence on either side. Selection does not use the gold label, a prediction, a novelty flag, or a certainty score. A sentence that contains both entities is a potential evidence location. It does not establish that the sentence expresses the target relationship. Absence of same-sentence co-occurrence does not by itself establish that the finding requires cross-sentence reasoning.

Where a selected interval is discontinuous, the stored spans are now the separate selected regions. Unselected intervening text is not absorbed into one continuous passage. The files verified above were built when each selection was contiguous, so their single spans remain the spans of those selections. They were not rewritten.

Each alternative already records whether the original model input contained all, part, or none of its spans. The summary value `mixed` is only an aggregate across alternatives. A relation may be supported by the document and still be absent from the text a particular model received. Previous model-input files were not modified.

Finding references remain unidentified. One candidate may later have several findings, several equivalent locations, a finding that needs more than one sentence, an ambiguous location, or no finding that corresponds to the annotation. A multi-label DrugProt candidate may have a separate finding for each native label. Empty references are not filled in order to complete a table. Linguistic certainty, source-grounded support, and semantic completeness are not populated.

Deterministic selection can list co-occurrence, shortest intervals, and neighbourhoods. It cannot decide which of several assertions in one sentence is the annotated relation, which of several entity occurrences matters, which finding belongs to which label of a multi-label pair, or whether a negation or a condition applies to one proposition among several. DrugProt positive pairs in the sampled splits are same-sentence, and that fact does not make their meaning simple. A later identification procedure can keep the deterministic locations and use a constrained model only where the location is ambiguous. That model is not run here. Its reference annotations and validation criteria are not yet established.

## Review samples

The existing development sample and the human-review sample use disjoint documents. The development sample is method-development material. The human sample is enriched for difficult cases and is a diagnostic set. Its unweighted error rate would not estimate a corpus rate. A representative evaluation sample, with known selection probabilities, has not been drawn. Independent confirmation data that have not informed method selection are not available.

The human packet shows the document and the target entities. It does not show a gold label or a model prediction. Labels remain in the separate label record. No review answer has been generated by a language model, and none has been filled.

Later measurement can use separate fields for source-finding identification, assertion-level linguistic certainty, source-grounded support, semantic completeness, supported abstraction, unsupported specificity, and missing or distorted scientific information. Those fields are not assigned in this note. Linguistic certainty is not the probability that an assertion is true. Source-grounded support is not a judgement of study quality or biological truth.

## Verbalisation specification, version `representation_verbalisation_v1`

A predicate verbalisation states the proposition licensed by the category definition. It is separate from the sentence that reports the proposition, from the linguistic commitment of that sentence, and from context that the label does not encode. An unqualified triple is not an established scientific fact.

The following constraints bind every level.

- A non-directional BioRED category is not verbalised as a directional causal assertion.
- A correlation category is not verbalised as a specific molecular mechanism.
- A residual parent is verbalised as the set of native relations it contains, not as one shared mechanism.
- Direction that the grouping removed is not restored in the parent sentence.
- `No_Relation` is verbalised as the absence of an annotated positive relation, not as the absence of a biological relationship.

Illustration, not a corpus annotation. Suppose the source says that a named chemical reduced activity of a named gene product in a named disease model.

| Level | What may be said from the label alone |
| --- | --- |
| BioRED P4 `Negative_Correlation` | The annotation places the pair in the negative-correlation bin, which includes decreases and some therapeutic descriptions. |
| BioRED P2, same label | The same bin. This pair does not change at P2. |
| BioRED P1 `Any_Relation` | Some annotated positive relation is present. The direction is not available. |
| DrugProt P4 `INHIBITOR` | The annotation says the chemical inhibits the gene product, in the ChemProt sense of that label. |
| DrugProt P3 down-regulation group | The annotation is one of inhibitor or indirect downregulator. The label no longer says which. |
| DrugProt route, direct binding or regulation | The annotation is one of activator, inhibitor, or direct regulator. Direction and the specific mechanism are not available. |

The same source sentence can support the native statement and fail to support a parent sentence that adds a mechanism, or a parent sentence that drops a distinction the source actually made. That difference is the later object of source-grounded support and semantic completeness. No entailment model is selected or run here.

## Reuse of completed model outputs

The 67 prediction conditions accepted by the earlier identifier audit remain reusable for scoring and for deterministic abstraction onto P1–P4. They also remain reusable for a later deterministic mapping onto the two alternative partitions. Accepted conditions may still contain individual answers that are not valid labels. Those answers stay in the population.

The following outputs are not reusable as complete prediction sets. Four directories were produced by duplicate writers and stay excluded. Qwen3-8B DrugProt P3 and P4 are incomplete. The formal manifest still describes cancelled shards as if they were active. The formal run `formal_20261115` is not modified.

A change that only affects a downstream table, a deterministic mapping of saved labels, or a source-reference file does not require a new model call. A change to the model input, the prediction target, or the training procedure would. The alternative partitions and any finer entity type would change the prediction target if they were asked for directly. The recorded sentence version does not change the frozen model inputs. New analysis code, including a future semantic evaluator, does not by itself justify retraining.

A future semantic cache must key the source version, the assertion verbalisation, the evaluation method, and the model version. Repeated observations stay linked to their documents, candidates, and seeds. Seeds are not pooled as independent predictions. Distinct findings are not merged because they share an entity pair and a label.

## Statistical quantities reserved for a later analysis

Nominal level and information gained remain different quantities. BioRED P3 adds little entropy after P2. DrugProt P2 retains a much smaller fraction of native entropy than BioRED P2. The same level number is not the same amount of biological specificity in the two corpora. The alternative partitions show that a larger number of names can retain less label entropy.

Later paired comparisons can use document identifiers and candidate or finding identifiers for direct prediction against abstraction, entity conditions, predicate representations, relations whose label changes against relations whose label does not, model families, and corpora. The existing document-level paired bootstrap remains the relevant resampling convention. Denominators must include candidates with no positive annotation, unsupported predictions, and findings that were not recovered. Analysis restricted to successful extractions would not answer those questions. No such comparison is calculated here.

## Application to treatment response and resistance

A precision-oncology question about lung-cancer treatment response would need to separate an entity association from a specific relationship, sensitivity from resistance, a gene from a variant, a disease from a subtype, a reported finding from a proposed mechanism, and a preclinical finding from a clinical finding. It would also need to separate additional support for a known assertion from a conflicting or context-dependent finding.

BioRED can represent a directional correlation, a binding, an association, a comparison, a cotreatment, a drug interaction, or a conversion, within the limits and unresolved readings above. DrugProt can represent the thirteen ChemProt relations, including inhibition and indirect down-regulation. Neither schema has a supervised label for sensitivity versus resistance, gene versus variant, disease versus subtype, reported finding versus proposed mechanism, or preclinical versus clinical evidence. Those distinctions are application-specific. They are not created by inventing a finer benchmark label.

PubMed abstracts are already the text of these corpora. CIViC was not present in the research storage and was not acquired. Three different later outputs need different checks: a relationship reported in a paper and absent from a reference knowledge base; additional or more specific evidence for a relationship the knowledge base already records; and a hypothesis inferred by connecting findings. Absence from a knowledge base is not evidence that a relationship is false, and it is not evidence that the relationship was previously unknown to biology. No literature extraction or hypothesis generation was done.

## Readiness

| Component | Status |
| --- | --- |
| Prepared candidates, splits, and native labels | Complete for the existing experiments. Shared-document membership is recorded. |
| E0–E2 entity representations | Implemented and previously measured. Unchanged. |
| An entity level beyond E2 | Unsuitable with the resources used here. UMLS is unavailable. MeSH covers a minority of concepts, often by several trees, and does not type genes, variants, or cell lines. |
| P1–P4 predicate representations | Structurally validated. Several biological readings remain unresolved. |
| Alternative partitions | Defined and measured. They are sensitivity analyses, not a replacement protocol. Direct prediction is not authorised. |
| Information measures | Defined for nested partitions and, separately, for one complete partition. Undefined when native entropy is zero. Not a measure of source-level meaning. |
| Accepted predictions | 67 conditions reusable for scoring and for post-prediction abstraction. Incomplete DrugProt generations stay incomplete. |
| Sentence segmentation | Baseline retained. Comparison recorded. Correctness unresolved pending human boundaries. |
| Candidate passages and model-input coverage | Implemented and verified against the candidate populations. A passage is not a finding. |
| Finding references | Structure only. Identification has not been done. |
| Verbalisation | Specification `representation_verbalisation_v1` stated. Not applied to the corpus. |
| Human evaluation | Development and diagnostic samples exist and are document-disjoint. Representative and confirmation samples do not. |
| Linguistic certainty, support, and semantic completeness | Not measured. The fields are reserved. |
| Treatment-response application | The benchmark schemas cannot express the required clinical distinctions. No discovery claim follows from this note. |
| Execution integrity | Later writers publish atomically. The previous formal run is unchanged. |

## Decisions required before further training

The representation conditions proposed for a continuation of the present extraction comparison are E0, E1, and E2, and P1–P4. DrugProt stays on the predicate factor only. The two alternative partitions may be applied to saved native predictions. They should not be added as training targets while the residual and directional readings above are unresolved.

The following can wait without changing training data or prediction targets: human adjudication of sentence boundaries, finding identification, verbalisation of individual corpus relations, linguistic certainty, source-grounded support, semantic completeness, and any application corpus such as CIViC. A change to model inputs, or a decision to predict an alternative schema directly, cannot wait in that sense. It would be a new experiment.

The protocol is not frozen. The unresolved category readings and the absence of finding-level references are scientific gaps, not documentation gaps.
