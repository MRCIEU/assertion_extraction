# Source references for the existing biomedical relations

The granularity measurements show that the frozen predicate levels do not add label information evenly, and that a parent category can be a valid partition without being one biological mechanism. The next question is what those categories mean in the articles from which the pairs were drawn. Before linguistic certainty, source-grounded support, or semantic completeness can be measured, two preparatory facts are needed: what each existing predicate name does and does not say, and which original passages contain the two target entities.

This note records the predicate reading, the requirements for any later entity-type extension, and a deterministic location of candidate passages. A candidate passage is text that contains the two targets, or the shortest sentence interval that does. It is not a finding, and it is not validated evidence. Entity co-occurrence identifies a place where a relationship could be expressed. The absence of a gold label does not establish that the text contains no relationship.

The prepared collections are unchanged: `biored_candidates_v2`, `bc8_candidates_v2`, and `drugprot_candidates_v1`. Predicate mappings, splits, model inputs, and saved predictions are unchanged. No ontology was acquired, and no neural model was trained or run. The machine records are in `/home/b5ac/freddieyu.b5ac/projects/project_1/studies/schema_granularity_v1/reports/source_references/`.

## Corrections to the existing implementation

Three corrections were required before the passages were built.

The entity-type mapping is read from `configs/schemas/entities.yaml`. The duplicate lists in the granularity module have been removed. Recomputing the published entropy from the stored training counts reproduces the previous figures: BioRED and BC8 retain 0, 0.853, 0.981, and 1 of the native predicate entropy at P1–P4, and DrugProt retains 0, 0.489, 0.788, and 1. The BioRED mention-level entity entropy remains 2.151 bits at E2 and 1.322 bits at E1, with retained fraction 0.615. The published granularity files were left in place.

A schema with one native label has native entropy zero after smoothing, because that label has probability 1. The retained fraction is then undefined, and so is the retention ceiling, which would divide by the same zero information. Add-one smoothing over the eight BioRED labels or the thirteen DrugProt labels does not reach this case. The earlier test used a real DrugProt label and therefore never observed the undefined value.

The CPU check had been written when every module was free of a model library. Named execution modules may now import one. Those modules are listed in `AUTHORISED_MODEL_EXECUTION_MODULES`. Every other module remains prohibited from importing PyTorch, a model class, vLLM, or an external API. Importing the command line or the models package does not load PyTorch, and the default `train` command still exits with code 3.

## Predicate categories

The BioRED guideline used here is the local file of 7 April 2022. The DrugProt definitions are those of ChemProt guideline version 6, restricted to the thirteen labels in the frozen schema. The structural checks already showed that each level is a nested partition and that `No_Relation` lies outside it. That result is unchanged. A nested partition is a statement about membership. It does not make the parent name a biological generalisation.

`No_Relation` means that the candidate has no positive annotation. It is a property of the annotation table.

### BioRED

`Positive_Correlation` collects increases and co-movements in the guideline examples: a chemical or a higher dose raising disease risk, a disease raising a measured chemical, a protein or variant contributing to disease, and a chemical raising gene expression. The reading is a directional bin. Causation, dose response, and a reported correlation remain inside it. The word correlation can suggest a statistical coefficient when the text states an experimental effect. The group is unchanged from P4 through P2.

`Negative_Correlation` collects decreases, treatments, and protective relationships, including a drug treating a disease. The reading is directional. A reader who expects only an inverse measurement will miss the therapeutic examples. The group is unchanged from P4 through P2.

`Association` is the guideline's residual bin: pairs that were not assigned a direction, an underspecified association, a drug described as safe, a variant only observed in patients, and a chemical modification such as phosphorylation. It does not name one mechanism. The name can suggest an epidemiological association. At P3 it remains this single label. At P2 it joins `Other_Relation`.

`Bind` is a physical interaction, including proteins in a complex, binding to a promoter, and a protein acting as a chemical's receptor. When the text says that the binding causes a positive or negative correlation, the guideline assigns the correlation instead. The physical reading is defensible together with that exception. The name can be read as every binding sentence in the abstract. At P2 it joins `Other_Relation`.

`Comparison` is named in the chemical–chemical inventory. The recovered prose defines the neighbouring labels and does not give Comparison its own sentence. A natural-language reading of "comparison" is available, and it can imply a ranking that the guideline does not state. The prose definition remains unresolved.

`Cotreatment` is defined as combination therapy. The reading follows that sentence. It can imply that the combination was shown to be beneficial when the text only says that the chemicals were used together.

`Drug_Interaction` is defined as a pharmacodynamic interaction when two chemicals are given together. The heading distinguishes it from combination therapy. The recovered text does not give a further test for an abstract that describes both, and the name can include pharmacokinetic interactions that the sentence does not mention.

`Conversion` has two instructions in the recovered guideline. One heading says that a chemical converts to the other. An earlier note says to annotate chemical conversion as Association. The released corpus uses Conversion. Which instruction a given abstract followed is unresolved, so the name is not yet a settled natural-language category.

`Remaining_Relation` at P3 contains Comparison, Cotreatment, Drug_Interaction, and Conversion. It keeps those four apart from correlation, association, and binding, and it drops the distinctions among them. The name is an honest residual label. It is not one chemical mechanism.

`Other_Relation` at P2 contains Association, Bind, and the four chemical–chemical labels. Directional correlation stays outside it. The members do not share a mechanism.

`Any_Relation` says that some positive annotation exists. Direction, binding, and the chemical distinctions are omitted. The schema stores the group as nonpolar because its members do not share a direction. The defensible reading is existential.

### DrugProt

`ACTIVATOR` is a chemical that binds a gene product and increases its activity. `INHIBITOR` binds and decreases activity. `DIRECT-REGULATOR` binds and changes activity when the text states no direction. `INDIRECT-UPREGULATOR` and `INDIRECT-DOWNREGULATOR` increase or decrease the gene product through other components, expression, stability, release, or modification, rather than by direct binding. These five readings follow the guideline. An indirect label can be misread as direct activation or direct inhibition.

`AGONIST` binds a receptor and alters its state when the text does not say whether the response increases or decreases. `AGONIST-ACTIVATOR` increases that response. `AGONIST-INHIBITOR` decreases it. The guideline describes those directions. The schema stores all three labels as nonpolar. `ANTAGONIST` reduces the action of another chemical, commonly at the same receptor. It is narrower than a general inhibitor, and the frozen mapping keeps it outside down-regulation.

`PART-OF` is structural inclusion, such as an amino-acid residue. `SUBSTRATE` means that the gene product acts on the chemical. `PRODUCT-OF` is an output of an enzyme or a transporter. `SUBSTRATE_PRODUCT-OF` is the combined role. The output reading includes transport, so "product" is broader than a metabolic product.

`Upregulation_or_activation` groups ACTIVATOR and INDIRECT-UPREGULATOR. An increase is preserved, and the agonist labels stay outside. Direct binding is not preserved. The word activation is the guideline's term for a direct effect, so the stored name can attach that directness to the indirect member. The same limitation applies to `Downregulation_or_inhibition`, which groups INHIBITOR and INDIRECT-DOWNREGULATOR and leaves antagonism outside. Both groups are unchanged between P3 and P2.

`Agonism` groups the three agonist labels. Receptor agonism stays apart from ordinary activation and from antagonism. The increase, the decrease, and the unstated direction are omitted. The parent name can suggest a uniform increase, while one member is defined by a decrease. The group is a structural collection of agonist labels.

`Antagonism` and `Direct_regulation_unspecified` and `Part_of` each contain one native label, so they keep that label's reading at P3. At P2 they join `Other_Relation`.

`Metabolic_relation` groups SUBSTRATE, PRODUCT-OF, and SUBSTRATE_PRODUCT-OF. It separates those roles from regulation and from structural inclusion. It does not separate input from output, and the name can imply an enzyme reaction for a transporter product.

`Other_Relation` is the complement of the explicit up and down groups. Agonism, antagonism, unspecified direct regulation, structural inclusion, and the substrate or product roles are omitted. It does not name one mechanism.

`Any_Relation` says that at least one positive label applies. A candidate may carry more than one. The defensible reading is existential.

### Representation and the source question

In the BioRED development and test documents, 2,324 positive relations were aligned. Of these, 1,064 keep the same group from P4 through P2. A further 1,222, the Association and Bind relations, first become less specific at P2. The remaining 38, all in one sentence, are the chemical–chemical labels that become less specific at P3. In BC8 the corresponding counts are 2,933, 2,888, and 197. Of those 197, 194 share a sentence and three have the targets in adjacent sentences.

No DrugProt label is unchanged from P4 through P2, because every native label shares a P2 group with another label. Of the 7,286 positive labels in the development and evaluation documents, 5,341 become less specific at P3 and 3,499 become less specific at P2. Those counts overlap: 1,554 labels change at both steps. The overlap is expected from the agonist labels and the substrate or product labels. All 7,286 labels have both mentions in one sentence.

These counts describe the annotation tables. They do not say that a passage expresses the broader category.

## Finer entity types

E0, E1, and E2 remain the entity conditions. UMLS and MeSH were not acquired, and no finer mapping was built. The passages below use the existing mention offsets and native categories.

A later mapping would have to keep four distinctions separate. A UMLS semantic type is a broad category. A MeSH tree number is a path, and one concept can have several paths of different depths. A molecular form is the gene, protein, transcript, or variant that the sentence mentions; a database class or an identifier pattern does not establish that form. A pharmacological action is a role of a chemical. It can repeat the relation being predicted, and it cannot be supplied as an entity type.

The mapping would also have to record every applicable type and path, the depth of each path, an explicit missing or ambiguous state, and the difference between a corpus category and an ontology category. A type defined as inhibitor, substrate, or therapeutic role discloses the target relation. The absence of this mapping does not block a passage for an existing pair.

## Sentences

Titles and abstracts were segmented separately with a deterministic rule. A period, question mark, or exclamation mark ends a sentence when the next significant character is an uppercase letter, a digit, or a closing mark, or when the text ends. A decimal point, a listed abbreviation, or a single alphabetical character does not end a sentence. The title–abstract separator remains the recorded space and is outside every sentence. The original characters are not rewritten. Sentence identifiers have the form `{document}:{title|abstract}:{index}`. Offsets use the joined document.

The check covered the BioRED development and test documents, all BC8 documents, and the DrugProt development and evaluation documents.

| Corpus | Documents | Sentences | Mention-text mismatches | Separator inside a sentence | Missing mention alignment |
| --- | ---: | ---: | ---: | ---: | ---: |
| BioRED dev+test | 200 | 2,195 | 0 | 0 | 0 |
| BC8 | 400 | 4,587 | 0 | 0 | 0 |
| DrugProt dev+eval | 1,450 | 14,659 | 0 | 0 | 0 |

Every stored sentence is the document slice at its recorded offsets. Gaps between sentences inside a title or an abstract contain only whitespace.

The rule also records boundaries it refuses. BioRED, BC8, and DrugProt respectively contain 583, 981, and 2,454 decimal points, 4, 8, and 43 listed abbreviations before a capital, and 65, 159, and 215 single-letter tokens before a capital. Many single-letter cases are variant notation such as `p.R229G`, which should remain inside a sentence. Some abbreviation cases are likely sentence boundaries that the rule keeps together, including `CF. We` and `min. After`.

Two crossing mentions show the opposite error. In BC8 document 36140807 the genomic string `ECA3g.79544206A>T` is split after `ECA3g`, and the annotated mention `g.79544206A>T` occupies both sentences. In DrugProt document 9313928 the mention `7. 8-Epi-PGF2` is split into the sentence `7.` and the following chemical name. Five BC8 candidates and 119 DrugProt candidates have a mention that meets more than one sentence. No BioRED candidate in the development or test documents does.

## Where the two targets occur

BioRED and BC8 arguments are normalised concepts and may have several mentions. DrugProt arguments are single mentions. The candidate identifiers and the candidate populations are the prepared tables. In these splits that is 13,381 BioRED candidates, 28,734 BC8 candidates, and 235,665 DrugProt candidates. The positive subsets are 2,324 candidates, 6,018 candidates, and 7,208 candidates. DrugProt has 7,286 positive labels because 78 candidates carry two.

| Corpus | Set | Same sentence, one location | Same sentence, several locations | Distance 1 | Distance 2 or more | Repeated mentions |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BioRED | All candidates | 4,463 | 1,185 | 2,930 | 4,803 | 8,927 |
| BioRED | Positive candidates | 1,228 | 679 | 204 | 213 | 2,212 |
| BC8 | All candidates | 9,844 | 2,994 | 6,211 | 9,685 | 19,816 |
| BC8 | Positive candidates | 2,903 | 1,844 | 698 | 573 | 5,744 |
| DrugProt | All candidates | 28,328 | 0 | 40,086 | 167,251 | 0 |
| DrugProt | Positive candidates | 7,208 | 0 | 0 | 0 | 0 |

No candidate in these splits lacks a mention alignment. Every annotated positive DrugProt relation has both mentions in exactly one sentence. That is a property of these annotations. Candidates with no positive label are often in different sentences, and those candidates are not evidence that the abstract contains no relationship.

## Candidate passages

Four passages are stored for every candidate. The complete document is one span. Each sentence that contains both targets is kept as its own alternative. The shortest sentence interval containing one mention of each target is kept, and every interval of that minimum length is kept. The same interval is also expanded by one sentence on either side, stopping at the edge of the document. Discontinuous text would be stored as separate spans. In these documents the selected sentences form one original span, including the title–abstract space when an interval crosses that boundary. The choice uses offsets only.

| Corpus | Set | No shared sentence | Several shared sentences | Several minimum intervals | Median sentences in the minimum interval | Median characters |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BioRED | All candidates | 7,733 | 1,185 | 1,989 | 2 | 373 |
| BioRED | Positive candidates | 417 | 679 | 776 | 1 | 191 |
| BC8 | All candidates | 15,896 | 2,994 | 4,526 | 2 | 369 |
| BC8 | Positive candidates | 1,271 | 1,844 | 2,156 | 1 | 201 |
| DrugProt | All candidates | 207,337 | 0 | 0 | 4 | 727 |
| DrugProt | Positive candidates | 0 | 0 | 0 | 1 | 203 |

The long DrugProt intervals belong to candidates without a positive label. A median of four sentences is the distance between two mentions, not a measured finding.

The original model input is a separate object. For each passage the records say whether that input contains all, part, or none of the passage span. Mixed means that the retained alternatives do not agree.

| Corpus | Set | Shortest interval all / part / mixed / none | Full document all / part |
| --- | --- | --- | --- |
| BioRED | All candidates | 13,296 / 7 / 78 / 0 | 11,102 / 2,279 |
| BioRED | Positive candidates | 2,273 / 0 / 51 / 0 | 1,952 / 372 |
| BC8 | All candidates | 28,247 / 119 / 357 / 11 | 24,923 / 3,811 |
| BC8 | Positive candidates | 5,770 / 7 / 238 / 3 | 5,038 / 980 |
| DrugProt | All candidates | 232,060 / 3,605 / 0 / 0 | 198,930 / 36,735 |
| DrugProt | Positive candidates | 7,208 / 0 / 0 / 0 | 6,467 / 741 |

The shortest interval usually lies inside the text the extraction model received. That overlap describes the input window. It does not say that the model identified the finding, and it does not say that the interval is the finding.

## Finding references

A finding reference is a separate object. It can hold one sentence, several sentences, several findings for one pair, several alternative locations, or an unidentified or ambiguous state. A DrugProt candidate with two native labels has two references. The prepared references are empty, with status `unidentified`. Gold labels were not copied into them as locations. No finding was generated automatically.

## Samples

The development sample uses BioRED development documents and DrugProt development documents. BC8 has no development split. The human sample uses BioRED test, BC8, and DrugProt evaluation documents, and it excludes every document identifier used in the development sample. Within each draw, identifiers are sorted and then shuffled with `random.Random`. The seeds are 20261124 plus a fixed corpus offset. Reserved positive slots ask first for a cross-sentence pair, a repeated concept, a shortened model input, and, in DrugProt, a multi-label candidate. The remaining positive slots rotate through native labels. A document is used once. Sample identifiers are assigned after a final sort.

The human documents are disjoint from this method-development sample. They are not independent of earlier scoring: BioRED test, BC8, and DrugProt evaluation were already evaluation collections. Whether DrugProt's official development collection was used for earlier tuning remains unconfirmed.

Each corpus draw contains 20 positive candidates and 5 candidates with no positive label. The development sample therefore has 40 and 10. The human sample has 60 and 15, across 75 documents. No document is reused. The human review file contains the document, the sentence identifiers, the marked mentions, and the passages. It does not contain gold labels or model predictions. Those labels are in a separate record, still without predictions and without populated findings.

The DrugProt positive pool contains no cross-sentence pair and no repeated mention, so those reserved slots are zero there. The cross-sentence DrugProt examples in the samples are candidates without a positive label. The development BioRED draw includes Association, Bind, both correlations, Comparison, and Cotreatment. Conversion and Drug_Interaction appear in the human BioRED draw. The DrugProt draws include all thirteen native labels. Two human DrugProt candidates, and three development candidates, carry two labels and therefore two empty finding references.

## What the passages can show

The following examples come from the development sample. The gold label is named so that the limit of the passage is visible. It was not an input to the selection.

PMID 17308037 has one sentence containing both `cAMP` and `phosphodiesterase`: "In kidneys, stimulation of adenylyl cyclase causes egress of cAMP, conversion of cAMP to AMP by ecto-phosphodiesterase, and metabolism of AMP to adenosine by ecto-5'-nucleotidase." The shortest passage is that sentence, and the original input contains it. The sentence also states a conversion and a further metabolic step. The passage locates the two mentions. It does not decide which of those clauses is the annotated relation.

PMID 16152606 mentions `657del5` and sporadic lymphoid malignancies together in the title and again in a later sentence about higher carrier frequencies. Both sentences satisfy the same-sentence rule, so both are retained. The procedure has no basis for discarding either location.

PMID 24036311 places `Mest` in one sentence and `Ctr9` in several. Two intervals have the minimum length of one intervening boundary: the Mest sentence with the preceding Ctr9 sentence, and the Mest sentence with the following Ctr9 sentence. The preceding sentence describes reduced expression after Ctr9 knockdown. The Mest sentence lists imprinted genes, including Mest, as aberrantly expressed. A finding could require both sentences, and the two intervals do not agree on the neighbouring sentence. The rule keeps both.

PMID 27840894 places hypophosphatemic rickets and the variant `c.1483-1G>C` three sentences apart. The shortest interval includes the enrolment sentence, the exon-sequencing sentence, and the FGF23 sentence as well as the sentence that names the variant. Those intervening sentences are part of the interval because the rule cannot identify which of them states a relation between the two targets.

Across these examples, deterministic selection can recover the original text, can show whether the targets share a sentence, and can list every minimum alternative. It cannot decide which alternative is the finding, whether a cross-sentence interval is one finding or several, or which clause inside a shared sentence carries the relation. The high rate at which the shortest interval falls inside the original model input does not change that limit. Passage coverage is not evidence-identification accuracy.

## What remains open

A model-based finder is not required to name the sentence of an annotated positive DrugProt pair in these splits: that sentence is unique. It would be a possible method for three decisions that the examples leave open: the choice among BioRED and BC8 locations, the status of the intervening sentences in a cross-sentence interval, and the choice among claims that share a sentence. The human sample is ready for a reader to see how often those decisions are obvious from the passages. This work does not make the decisions, and it does not treat them as a reason to run a model before that reading.

The unresolved predicate questions are also unchanged. Comparison still lacks a recovered prose definition. Conversion is still named both as its own label and as Association. Cotreatment and Drug_Interaction still have no further boundary test. Agonism still joins an increase, a decrease, and an unstated direction under a nonpolar schema field. Residual parents remain residual. Finer UMLS and MeSH types remain unspecified. The segmenter still joins some abbreviation boundaries and still splits some genomic and numbered chemical strings. Human judgements have not been entered.
