# Entity classification for the granularity study

The question is which additional entity classes can be shown to a relation extractor without changing the entity’s identity or inventing a type from a hierarchy edge. The earlier protocol counted MeSH tree nodes and reached eight additional levels. Those counts remain in `reports/formal_protocol/`. They are not the classes used here.

A class is used only when the resource states a classification relation: a ChEBI `is_a` link, a MONDO `is_a` link from a unique `equivalentTo` MeSH cross-reference, a Sequence Ontology term for a documented PubTator variant code, an NCBI Taxonomy rank above the organism itself, or a Cellosaurus category. A MeSH tree number is stored as provenance. It is not treated as a subclass. Pharmacological actions, `has_role`, and a supplemental concept’s `HeadingMappedTo` descriptor are not types. A node whose label is the concept’s own name is not a new type. Where a path branches, the shared prefix is kept and the alternative parents are retained. No path is chosen because it is the deepest.

## Resources

UMLS was not available and was not used. The files below were read from the research store. They are not part of the code checkout.

| Resource | What was used | What it did not supply |
| --- | --- | --- |
| MeSH desc2026 | 31,110 descriptors; structural trees kept, D06, D26, and D27 excluded | A formal subclass relation |
| MeSH supp2026 | 410 corpus supplemental records | A subclass; the mapped descriptor is an indexing relation |
| ChEBI ontology | `is_a` only, joined by the descriptor’s preferred CAS | `has_role`; a MeSH cross-reference inside the ontology |
| MONDO, release 2026-11-03 | 645 unique `equivalentTo` MeSH joins | A hierarchy merged with MeSH |
| NCBI `gene_info` | Eight species files, 549,208 rows | The molecular form named by a mention |
| Sequence Ontology 2026-08-07 | The PubTator code’s term | A finer class than that code; `amino_acid_substitution` is not under `substitution` |
| NCBI Taxonomy and Cellosaurus 57.0 | Ranks above the taxon; one cell-line category | Disease of origin as the next cell-line class |
| dbSNP | Not joined | A variant class for an `rs` identifier. The prefix was not used as a class |

ChEBI joined 418 descriptors to one ChEBI record by CAS. Most of those records branch at the first `is_a` step, so they have no single parent path. Twenty-one BioRED chemical concepts received a non-empty shared prefix. The other chemicals keep the native type. Their structural MeSH trees are recorded and are not displayed.

## Coverage

On BioRED, 745 concepts receive a primary class, 1,617 genes have a database gene type that is not displayed, 857 concepts have a MeSH tree that is not used as a type, 202 supplemental identifiers are not treated as subclasses, and 56 chemicals are excluded because their only trees are functional. BC8 has 795 concepts with a primary class and 1,144 genes with a recorded but unused database type. DrugProt arguments remain local mention identifiers. No entity hierarchy is proposed for DrugProt.

The additional BioRED classes are MONDO disease ancestors, Sequence Ontology variant terms, Cellosaurus categories, taxonomic ranks, and a small number of ChEBI classes. A variant code and a cell-line category stop at the first additional level. Later levels are further MONDO ancestors and taxonomic ranks. The same terminal label is carried forward; the carry-forward is not counted as a new distinction.

The coverage rule requires each successive level to change at least 100 BioRED training concepts. That rule reaches E7. The training counts of new distinctions are 507, 189, 169, 148, and 111 at E3–E7, then 73 at E8. A threshold of 250 stops at E3. A threshold of 50 reaches E8. These alternatives are properties of the class inventory, not of extraction performance.

Of 21,755 scoring-eligible BioRED training pairs, 5,115 change at least one argument between E2 and E3. The matched comparison is those 5,115 pairs. The other 16,640 are the unchanged comparison.

The proposed global conditions are E0–E7, with E0–E2 unchanged from the native representation. E0 hides the type. E1 merges gene, chemical, and variant mentions to one molecular entity. E2 is the native type. E3–E7 add the validated class and then carry the terminal class forward. A missing deeper class does not remove the native type.

## Verbalisation and measurements

Verbalisation version `representation_verbalisation_v3` states the biological proposition. A reporting sentence says that the source reports that proposition. It does not say that the dataset annotated the pair, and it does not treat discussion as support. `NO_RELATION`, a missing output, and an invalid output produce no positive assertion. A parent with one native member keeps that member’s meaning. A parent with several members is a disjunction, not one mechanism. Historical prompts still use `Any_Relation`; the reporting conversion does not authorise a new generation.

The conversion was applied to the 13 saved prediction files: 1,438,425 records, of which 51,548 positive predictions were verbalised and 1,387,095 produced no positive assertion. Gold labels use the same function.

The sentence-level certainty checkpoint is `pedropei/sentence-level-certainty`, on a 1–6 scale. The aspect-level model is not used. It was not executed: the analysis environment has no torch. The assertion-conditioned Qwen procedure is specified and was not run. The general-domain verifier checkpoint selected for later execution is `MoritzLaurer/DeBERTa-v3-xsmall-mnli-fever-anli-ling-wanli`. No SciFact model was trained. The official SciFact archive was read in place: the development split has 300 claims, with 216 supporting and 122 contradicting rationale labels. Those counts are dataset counts, not model scores.

Support probabilities must be finite and in the unit interval, and an assessable distribution must sum to one. A passage that does not fit the verifier limit is left unassessable rather than truncated. One supporting passage does not cancel a contradiction elsewhere. An unknown component status is rejected. Support and completeness are not averaged.

## Inputs, experiments, and the application sample

Additional class labels are ordinary tokenizer text. On a sample of 200 labels the longest is 9 WordPiece tokens and the median is 4. No new special tokens are added. 4,597 of 35,136 BioRED inputs already lie within 16 tokens of the 512-token budget, so an added class can make a frozen window illegal. E0–E2 inputs are unchanged and their saved predictions remain reusable when the prompt, label strings, and model revision match. E3–E7 are different inputs. Their historical predictions are not reusable, and the new input files have not been built.

The enumerated matrix over E0–E7 and P0–P3 has 49 cells whose previous predictions are reusable, 191 that require new BiomedBERT training, and 144 that require new Qwen inference. No job was launched. GPU time was not remeasured.

The CIViC nightly snapshot has 5,024 evidence rows. The lung-cancer population is the 631 rows whose disease identifier is one of DOID 1324, 3905, 3907, 3908, 3910, 4829, 5409, 6482, or 0080303. No row used the word lung without a DOID. Titles and abstracts were retrieved for all 379 linked PubMed identifiers. A curated evidence statement is not used as a model input. Absence from CIViC is not a negative label, and this snapshot is not a historical knowledge base.

A probability sample of BioRED test documents, excluding documents already used for development or diagnostic review, drew 15 documents from a frame of 75 at inclusion probability 0.2. The judgement fields are empty.

spaCy is not installed, so a biomedical sentence segmenter was not run. The recorded sentence rules remain the baseline. Independent sentence-boundary annotations are still absent.
