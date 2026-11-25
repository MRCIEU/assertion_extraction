# Pre-training protocol for semantic granularity and source-grounded fidelity

The study asks whether predicate granularity and entity-type information change the accuracy, completeness, and source-grounded support of biomedical assertions; whether linguistic certainty modifies those effects; and whether the preserved distinctions improve assertion discovery and evidence reassessment. The hypotheses H1–H4 are to be tested. Earlier runs and the measurements below are exploratory. A confirmatory analysis has not been done, and this protocol is not frozen.

No model was trained or run for this preparation. Machine-readable records are in `studies/schema_granularity_v1/reports/formal_protocol/`. They were computed from commit `26bb312a87801e6b545507d134e6d0b156230334` with uncommitted code present; the completion file records that state rather than a substituted hash.

## Data and integrity

The prepared collections, splits, and native labels are unchanged: `biored_candidates_v2`, `bc8_candidates_v2`, and `drugprot_candidates_v1`. Source-reference files were checked again for UTF-8, JSON, required fields, identifier duplicates, the final record, checksums, and agreement with the candidate population in each target split. Every scoring-eligible candidate is present. No excluded candidate is present in those files. The structural check of a passage requires the four location kinds, integer spans, and a model-input value of all, part, or none. Identifier agreement alone is not treated as completion.

| Corpus | Sentences | Alignments and passages | Structurally complete |
| --- | --- | --- | --- |
| BioRED dev and test | 2,195 | 13,381 | yes |
| BC8 | 4,587 | 28,734 | yes |
| DrugProt dev and eval | 14,659 | 235,665 | yes |

Concept inventories no longer take the first observed entity type. BioRED has 3,868 non-empty concept identifiers, 63 mentions without an identifier, 3 composite identifiers, and 1 identifier with more than one native type. BC8 has 3,458 non-empty identifiers, 20 mentions without an identifier, and 5 type conflicts. A composite MeSH identifier is classified only when every descriptor component yields the same admissible chain.

## Entity classification

UMLS remains unavailable. The resources actually used are public and are not committed.

| Resource | Release | Role |
| --- | --- | --- |
| MeSH descriptors | desc2026 | Ancestor classes for diseases and chemicals |
| Sequence Ontology | 2026-08-07 | Variant class for a documented PubTator type code |
| NCBI Taxonomy | taxdump fetched 25 November 2026 | Ancestor ranks for organism identifiers |
| Cellosaurus | 57.0 | Cell-line category |

MONDO and ChEBI were not acquired. MeSH identifiers already supply the disease and chemical paths used here, and a second hierarchy was not merged into them. Pharmacological roles are not types. A gene identifier is not read as the molecular form in the text. An `rs` identifier is not read as a variant class. A cell line’s disease of origin is stored as an excluded association, not as the next level of the cell-line category. `part_of`, `derived_from`, and pharmacological-action links are not treated as `is_a`.

The displayed MeSH path is the lexicographically first tree whose top letter matches the native type. The descriptor itself is the concept identity and is not shown as a type. Later requested levels carry the terminal class forward and do not count that repetition as a new distinction.

| Native category | BioRED result |
| --- | --- |
| ChemicalEntity | 487 classified; 29 excluded for role wording; 131 supplemental identifiers have no descriptor tree |
| DiseaseOrPhenotypicFeature | 648 classified; 3 composite identifiers have disagreeing paths; 96 are not descriptors |
| GeneOrGeneProduct | 1,642 have no additional admissible class |
| SequenceVariant | 355 have a PubTator type code mapped to Sequence Ontology; 277 `rs` identifiers stay at the native type; 45 codes are unresolved |
| CellLine | 72 have one Cellosaurus category |
| OrganismTaxon | 44 have a taxonomic ancestor chain; 1 identifier is unresolved |

On the BioRED training split, 1,154 eligible concepts gain a new class at the first additional level. Further MeSH or taxonomic steps change 854, 704, 475, 268, 144, and 35 concepts. A pre-specified rule keeps a contiguous level only while at least 100 training concepts gain a new distinction. That rule stops after the level corresponding to E8. It does not show that eight additional levels are eight scientific distinctions. The steps are successive resource nodes. Adopting E3–E8 as primary factors before the displayed classes have been reviewed would turn tree depth into the experimental factor.

Of 21,755 scoring-eligible BioRED training candidates, 9,854 change neither argument at the first additional level, 7,754 change one, and 4,147 change both. The matched comparison is the 11,901 candidates with a change. The full-population comparison keeps all 21,755 and carries terminal classes forward. These are different populations.

DrugProt has 419,192 local mention terms and no normalised concept identifiers. No linker was applied. DrugProt stays a predicate comparison.

**Decision before any new training.** The coverage rule can supply E0–E8. The primary comparison that is ready to specify without new inputs remains E0–E2. E3–E8 require reviewed class labels, a check that longer type strings do not silently shorten the source text, and textual type descriptions rather than a new special-token vocabulary. That token-budget check has not been run.

## Predicate notation

Research levels rename the historical label spaces. They do not rename directories or justify a repeated run.

| Research level | Historical level | Positive outcome |
| --- | --- | --- |
| P0 | P1 | `RELATED_TO`, historically `Any_Relation` |
| P1 | P2 | historical P2 groups |
| P2 | P3 | historical P3 groups |
| P3 | P4 | native labels |

`NO_RELATION` stays outside the positive hierarchy. It means that no positive relation is annotated in the dataset’s scope. At P0 the entropy retained among positive native labels is zero; detecting a reported relation is a separate outcome. A prompt that says `RELATED_TO` instead of `Any_Relation` may differ in wording even though the label membership matches, so those saved generations are not reused as if the strings were the same.

The alternative partitions keep the same members. Their display names no longer say that direction or coadministration was preserved: `Correlation_direction_collapsed` and `Cotreatment_or_drug_interaction`. Retransforming the training counts gives the same information as before: BioRED 1.226 bits, retained fraction 0.735, with 1,893 of 4,174 training relations broadened; DrugProt 1.879 bits, retained fraction 0.620, with 666 relations unchanged. The random size-matched control remains a mathematical control for BioRED only.

Unresolved readings are unchanged: Comparison has no recovered definition of its own; Conversion overlaps an Association instruction; Cotreatment and Drug_Interaction are not separated by a usable test; Association and the residual parents are sets of permitted labels; agonist labels mix direction. None of these was closed by renaming a group.

Verbalisation version `representation_verbalisation_v2` uses fixed templates. A residual parent is spoken as its member set. Correlation is not spoken as a mechanism, and a missing output is not spoken as `NO_RELATION`. Proposition wording and reporting wording are separate conditions. Templates have not been applied to the corpus.

## Source text

The recorded sentence splitter remains `abbreviation_decimal_v1`. Boundary F1 is still unavailable. A neural biomedical segmenter was not run, because it needs a model asset that was not downloaded. Agreement between splitters would not establish correctness.

Candidate passages remain potential locations. Finding references are still unidentified. A later Qwen3-8B localiser is specified to return a verbatim span or to abstain. A quotation is accepted only when it occurs in the document. That model was not run. The full document and the original model input remain separate support contexts.

## Semantic measurements

Three constructs are specified and not scored.

Linguistic certainty, L, belongs to a source finding and stays fixed across extraction conditions. It is not the probability that the assertion is true. The Pei and Jurgens 2021 overall-certainty resource has not been acquired, so its scale has not been checked against a local checkpoint. An assertion-conditioned Qwen3-8B specification is prepared and not executed. One primary L measure will be selected by the pre-specified metrics: MAE and rank correlation for a validated numerical scale, macro-F1 for categories, and a scope check for sentences that contain more than one proposition. The unselected method is a sensitivity analysis, not a second scientific factor.

Source-grounded support is the forward inference distribution of entailment, contradiction, and neutral. Neutral is not contradiction. Unassessable is separate. The probability is not biological truth. SciFact and a verifier checkpoint still have to be acquired before a model is chosen. Qwen3-8B is a specified prompted verifier and was not run. Support in the whole document and support in the extraction input are separate. A passage chosen with the gold label is not the only premise for a prediction that disagrees with that label.

Completeness has two candidate proxies. Reverse entailment from the focused finding to the assertion is not a proportion of scientific information preserved. Atomic coverage is the fraction of assessable components marked preserved. A valid abstraction is counted separately and is not full preservation of the finer component. Fidelity is the pair of support and completeness. They are not averaged.

The loss decomposition into the native-annotation gap, the abstraction gap, and the extraction gap is defined only when completeness is a validated proportion. Otherwise the differences stay differences in an unvalidated proxy. A negative difference is allowed: a prediction may fit the source better than the benchmark label.

No public certainty or SciFact files were scored. No GPU semantic run was prepared as if it had been executed.

## Human reference and statistics

The annotation procedure and the unfilled mapping examples are separate from the diagnostic sample. Representative sampling and independent confirmation are not done. Sample-size figures in the annotation note are assumptions under independence, not a design effect estimated from these data.

The analysis keeps document identifiers, candidate identifiers, and seeds. Seeds are not pooled as independent predictions. The entity–predicate interaction is the paired contrast of the fine-entity effect at the fine predicate minus the same effect at the coarse predicate, on a matched population. Requested level and realised change are analysed separately. Primary contrasts are the extraction and semantic outcomes named above. No table of formal results is produced here because those experiments have not been run.

## Experiment matrix

The coverage-threshold matrix has 272 direct cells: BioRED and BC8 at E0–E8 and research P0–P3, and DrugProt at its native entity representation and the same predicate levels. BiomedBERT uses seeds 13, 42, and 2026. Qwen3-8B and Qwen3-32B are prompted. BC8 has no separate training run.

Sixty-seven accepted prediction conditions can be reused as direct predictions. Forty-eight cells also have a saved native prediction that can be abstracted without a new call. Abstraction does not estimate direct prediction. If every coverage-threshold cell were executed, 107 BiomedBERT training runs and 98 generation conditions would be new. If the primary comparison remains E0–E2 until the additional classes are reviewed, the new work is 35 BiomedBERT training runs that were not in the accepted set and 2 Qwen3-8B DrugProt generations at historical P3 and P4. Existing E0–E2 inputs stay comparable only within the frozen context files. Deeper type strings require a new matched context before they can be compared. GPU time was not remeasured, so no hour estimate is given.

Training and generation settings stay the recorded BiomedBERT and Qwen settings. They were not retuned. Model execution remains blocked.

## Application

The nightly CIViC clinical-evidence table fetched on 25 November 2026 has 5,024 rows. Rows whose disease name contains “lung” number 631, of which 448 are coded sensitivity or response and 121 resistance. This is a current snapshot. It does not reconstruct what the knowledge base contained when a paper was published. No literature was extracted and no hypothesis graph was built.

BioRED and DrugProt can represent their native relations. They cannot represent sensitivity versus resistance, gene versus variant, disease versus subtype, a reported finding versus a proposed mechanism, or preclinical versus clinical evidence as supervised labels. An application record would need the variant or molecular profile, therapy, disease, response relation, population or experimental system, direction, linguistic commitment, and source span. Those fields stay outside the benchmark tasks until each has a stated source of supervision. Absence from CIViC is neither evidence that a relationship is false nor evidence that it was previously unknown. A path through several findings is a hypothesis, not an established relation, and the relations are not treated as transitive.

## What is ready

| Component | State |
| --- | --- |
| Prepared candidates and source-reference files | Checked against the split populations |
| E0–E2 and historical predictions | Reusable where the 67 accepted conditions apply |
| E3–E8 classes | Computed, not independently reviewed, not an approved training factor |
| P0–P3 notation | Defined as a renaming of P1–P4; wording caveat for `RELATED_TO` |
| Verbalisation templates | Implemented and unapplied |
| Certainty, support, and completeness | Defined; resources and reference labels outstanding |
| Human reference | Instructions and empty review rows; annotation not done |
| Formal training and generation | Counted and not started |
| CIViC application | Snapshot counted; protocol not executed |

The decision required before formal execution is whether the entity factor remains E0–E2 or follows the unreviewed MeSH and taxonomy chain. The semantic measures still need their public checkpoints, SciFact, and human reference labels. Those gaps are scientific conditions, not a completed protocol.
