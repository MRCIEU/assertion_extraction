# Author IAA — disagreement structure
## A. Heuristic vs author (10 rows)
| target_id | pairing_family | heuristic | author | conf | rationale (100) |
| --- | --- | --- | --- | --- | --- |
| GL_0031 | gene_drug | DRUG_GENE_REGULATION | __NEGATIVE__ | — | Tanespimycin is not mentioned in the abstract; only crizotinib and next-generation ALK TKIs are name |
| GL_0039 | gene_drug | DRUG_GENE_REGULATION | __NEGATIVE__ | — | Same source abstract as GL_0031: tanespimycin does not appear in the text. |
| GL_0043 | gene_drug | DRUG_GENE_REGULATION | __NEGATIVE__ | — | Lapatinib is not named in the abstract; the comparators discussed are erlotinib and BIBW2992. |
| GL_0068 | gene_drug | DRUG_GENE_REGULATION | __NEGATIVE__ | — | The abstract refers only to 'EGFR tyrosine kinase inhibitors' generically; dacomitinib is not named  |
| GL_0070 | gene_drug | ASSOCIATION_GENERAL | __NEGATIVE__ | — | Gefitinib is not named in the abstract — only erlotinib, HKI-272, and cetuximab are explicitly menti |
| GL_0118 | gene_drug | DRUG_GENE_REGULATION | __NEGATIVE__ | — | Although gefitinib is named, EGFR is not mentioned anywhere in the abstract; the trial compares gefi |
| GL_0131 | gene_drug | ASSOCIATION_GENERAL | __NEGATIVE__ | — | The abstract names only 'tyrosine kinase inhibitors (TKIs)' generically; erlotinib is not mentioned  |
| GL_0132 | gene_drug | DRUG_GENE_REGULATION | __NEGATIVE__ | — | Afatinib is not named by its generic name; only its development code BIBW2992 appears once as a comp |
| GL_0138 | gene_drug | ASSOCIATION_GENERAL | __NEGATIVE__ | — | Teprotumumab is referred to only by its development code R1507, not by its generic name, so under li |
| GL_0144 | gene_drug | DRUG_GENE_REGULATION | __NEGATIVE__ | — | Dacomitinib appears only under its development code PF00299804, not its generic name, so by literal  |

## B. LLM Opus vs author (3 rows)
| target_id | pairing_family | llm_opus | author | conf | rationale (100) |
| --- | --- | --- | --- | --- | --- |
| GL_0132 | gene_drug | DRUG_GENE_REGULATION | __NEGATIVE__ | — | Afatinib is not named by its generic name; only its development code BIBW2992 appears once as a comp |
| GL_0138 | gene_drug | ASSOCIATION_GENERAL | __NEGATIVE__ | — | Teprotumumab is referred to only by its development code R1507, not by its generic name, so under li |
| GL_0144 | gene_drug | DRUG_GENE_REGULATION | __NEGATIVE__ | — | Dacomitinib appears only under its development code PF00299804, not its generic name, so by literal  |

## C. Seven known heuristic–LLM disagreement targets — author adjudication
| target_id | pairing_family | heuristic | llm_opus | author | pattern |
| --- | --- | --- | --- | --- | --- |
| GL_0031 | gene_drug | DRUG_GENE_REGULATION | __NEGATIVE__ | __NEGATIVE__ | author agrees with LLM (__NEGATIVE__ vs heuristic positive) |
| GL_0039 | gene_drug | DRUG_GENE_REGULATION | __NEGATIVE__ | __NEGATIVE__ | author agrees with LLM (__NEGATIVE__ vs heuristic positive) |
| GL_0043 | gene_drug | DRUG_GENE_REGULATION | __NEGATIVE__ | __NEGATIVE__ | author agrees with LLM (__NEGATIVE__ vs heuristic positive) |
| GL_0068 | gene_drug | DRUG_GENE_REGULATION | __NEGATIVE__ | __NEGATIVE__ | author agrees with LLM (__NEGATIVE__ vs heuristic positive) |
| GL_0070 | gene_drug | ASSOCIATION_GENERAL | __NEGATIVE__ | __NEGATIVE__ | author agrees with LLM (__NEGATIVE__ vs heuristic positive) |
| GL_0118 | gene_drug | DRUG_GENE_REGULATION | __NEGATIVE__ | __NEGATIVE__ | author agrees with LLM (__NEGATIVE__ vs heuristic positive) |
| GL_0131 | gene_drug | ASSOCIATION_GENERAL | __NEGATIVE__ | __NEGATIVE__ | author agrees with LLM (__NEGATIVE__ vs heuristic positive) |

**Counts on these seven:** agree with LLM: **7**; agree with heuristic: **0**; other: **0**.

## D. Three-way agreement (all 30)
| Pattern | Count |
| --- |:---:|
| All three agree | 20 |
| Exactly two agree: heuristic=author ≠ LLM | 0 |
| Exactly two agree: heuristic=LLM ≠ author | 3 |
| Exactly two agree: author=LLM ≠ heuristic | 7 |
| All three labels differ | 0 |
