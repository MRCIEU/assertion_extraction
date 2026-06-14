# Step 01 — Corpus alignment and CIViC relevance

Maps training corpora onto CIViC pair types, audits PMID leakage, and quantifies oncology-intersection training volume.

Method: Alignment matrix across BioRED, DrugProt, and BC5CDR; PMID overlap audit; oncology criteria on gene-drug and gene-disease training relations.

| Corpus | CIViC relevance | Admissible pair types | Leakage |
| --- | ---: | ---: | --- |
| BioRED | high | 4/4 | 0 PMIDs |
| DrugProt | partial | 1/4 | — |
| BC5CDR | low | 0/4 | — |
| Combined training-evaluation overlap | — | — | 3 PMIDs excluded |

BioRED oncology intersection (gene-disease, all three criteria): 1086 relations.
