#!/usr/bin/env python3.11
"""
Generate Phase A YAML configs: 4 encoders × 3 schemas × 10 seeds = 120 experiments.

Naming: PA_{ENCODER}_{SCHEMA}_s{SEED:02d}
  PA = Phase A
  ENCODER: RB (RoBERTa-base), PB (PubMedBERT-base), BL (BioLinkBERT-base), PL (PubMedBERT-large)
  SCHEMA: Sflat, Spair, Smech
  SEED: 01-10

All configs:
  DATA = D2 (T1→T2 staged, multi-corpus)
  UPDATE = FT (full fine-tune)
  ARCH = PP (pipeline)
  Loss = CE

Outputs:
  configs/PA_*.yaml  (120 config files)
  phase_a_experiment_ids.txt  (list of all 120 IDs for sbatch array)
"""
from __future__ import annotations
import yaml
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = SCRIPT_DIR / "configs"
CONFIG_DIR.mkdir(exist_ok=True)

PROC = "/lus/lfs1aip2/projects/b5ac/project_1/training_data_generation/data/processed"
CODE_ROOT = "/home/b5ac/freddieyu.b5ac/project_1"
FT_DATA_ROOT = "/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments"

ENCODERS = {
    "RB": {
        "name": "roberta",
        "hf_name": "roberta-base",
        "max_length": 512,
        "note": "RoBERTa-base: general domain, MLM on CommonCrawl+Books",
    },
    "PB": {
        "name": "pubmedbert",
        "hf_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "max_length": 384,
        "note": "PubMedBERT-base: biomedical domain, MLM on PubMed abstracts",
    },
    "BL": {
        "name": "biolinkbert",
        "hf_name": "michiyasunaga/BioLinkBERT-base",
        "max_length": 384,
        "note": "BioLinkBERT-base: biomedical + citation-link prediction",
    },
    "PL": {
        "name": "pubmedbert_large",
        "hf_name": "microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract",
        "max_length": 384,
        "note": "PubMedBERT-large: biomedical domain, 340M params (scale comparison)",
    },
}

SCHEMAS = {
    "Sflat": {
        "schema_id": "S_flat",
        "pair_type_filter": "sflat_legal_endpoints",
        "note": "Flat corpus-based mapping: E0xM0 baseline",
    },
    "Spair": {
        "schema_id": "S_pair",
        "pair_type_filter": "spair_legal_endpoints",
        "note": "Entity-pair-type mapping: E1xM0",
    },
    "Smech": {
        "schema_id": "S_mech",
        "pair_type_filter": "smech_legal_endpoints",
        "note": "Entity-pair + mechanism mapping: E1xM1",
    },
}


def make_config(encoder_key: str, schema_key: str, seed: int) -> dict:
    enc = ENCODERS[encoder_key]
    schema = SCHEMAS[schema_key]
    exp_id = f"PA_{encoder_key}_{schema_key}_s{seed:02d}"
    suffix = schema_key  # Sflat, Spair, Smech

    return {
        "experiment_id": exp_id,
        "phase": "phase_a_schema_selection",
        "schema_id": schema["schema_id"],
        "seed": seed,

        "encoder": {
            "name": enc["name"],
            "pretrained_model_name_or_path": enc["hf_name"],
        },

        "minimal_trainer": {"enabled": False},

        "scientific_trainer": {
            "enabled": True,
            "model_name": enc["hf_name"],
            "max_length": enc["max_length"],
            "batch_size": 4,
            "learning_rate": 2.0e-5,
            "max_pairs_per_shard": 2000,
            "eval_every_steps": 64,
            "dev_fraction": 0.12,
            "use_online_negatives": True,
            "active_t1_shards": ["biored", "drugprot", "bc5cdr"],
            "active_t2_shards": ["biored", "drugprot", "bc5cdr"],
            "active_t3_shards": [],
            "t4_max_lines": 1024,
            "t4_max_steps": 128,
            "t4_max_length": 256,
            "t4_learning_rate": 5.0e-5,
            "max_updates": 2048,
            "early_stopping_patience": 10,
            "early_stopping_min_updates": 256,
            "selection_metric": "macro_f1",
        },

        "architecture": "pipeline",
        "update_regime": "full_finetune",
        "schedule": "T1_to_T2",
        "T3_mode": "none",
        "T4_mode": "none",
        "loss_mode": "re_ce",
        "novelty_head": "off",
        "directionality_scaffold": "off",
        "source_weighting_policy": "inverse_freq_family_softmax",

        "weak_supervision": {
            "lambda_auxiliary": 0.0,
            "lambda_distill": 0.0,
            "never_treat_weak_as_gold": True,
        },

        "negative_sampling": {
            "negative_ratio": 4.0,
            "max_negatives_per_sample": 64,
            "pair_type_filter": schema["pair_type_filter"],
            "use_per_dataset_routing": True,
            "exclude_merged_jsonl_for_training": True,
        },

        "stage_weights": {"T1": 1.0, "T2": 1.0, "T3": 0.15, "T4": 0.5},
        "source_weights": {
            "biored": 1.0, "drugprot": 1.0, "bc5cdr": 1.0,
            "civic": 0.0, "civicmine": 0.0, "cancermine": 0.0,
            "oncology_lung_pubmed": 1.0,
        },

        "export": {
            "save_stage_checkpoints": True,
            "save_predictions": True,
            "save_logits": True,          # needed for KB_surface_mean (P(NEG))
            "save_per_example_provenance": True,
            "prediction_format": "jsonl_gz",
        },

        "training_data_paths": {
            "T1_shards": {
                "biored":   f"{PROC}/t1_biored_trn_{suffix}.jsonl",
                "drugprot": f"{PROC}/t1_drugprot_trn_{suffix}.jsonl",
                "bc5cdr":   f"{PROC}/t1_bc5cdr_trn_{suffix}.jsonl",
            },
            "T2_shards": {
                "biored":   f"{PROC}/t2_biored_mesh_{suffix}.jsonl",
                "drugprot": f"{PROC}/t2_drugprot_mesh_{suffix}.jsonl",
                "bc5cdr":   f"{PROC}/t2_bc5cdr_mesh_{suffix}.jsonl",
            },
            "T3": {
                "civic":      f"{PROC}/t3_civic_semantic_priors.jsonl",
                "civicmine":  f"{PROC}/t3_civicmine_weak_sentences.jsonl",
                "cancermine": f"{PROC}/t3_cancermine_priors.jsonl",
            },
            "T4": f"{PROC}/t4_unlabeled_domain_adaptation.jsonl",
        },

        "ft_data_root": FT_DATA_ROOT,

        "phase_a_metadata": {
            "encoder_key": encoder_key,
            "schema_key": schema_key,
            "encoder_note": enc["note"],
            "schema_note": schema["note"],
            "causal_role": (
                f"Schema effect via {encoder_key}: compare {encoder_key}-Sflat vs "
                f"{encoder_key}-Spair vs {encoder_key}-Smech. "
                f"Encoder effect via {schema_key}: compare RB vs PB vs BL vs PL at {schema_key}."
            ),
        },

        "purpose": (
            f"Phase A schema selection: {enc['note']} | {schema['note']} | seed={seed}"
        ),
        "analyses": "benchmark;schema_wise;kb_surface_mean;per_head_f1",
    }


def main() -> None:
    all_ids = []
    counts = {enc: {sch: 0 for sch in SCHEMAS} for enc in ENCODERS}

    for enc_key in ENCODERS:
        for schema_key in SCHEMAS:
            for seed in range(1, 11):
                config = make_config(enc_key, schema_key, seed)
                exp_id = config["experiment_id"]
                out_path = CONFIG_DIR / f"{exp_id}.yaml"
                with open(out_path, "w") as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                all_ids.append(exp_id)
                counts[enc_key][schema_key] += 1

    # Write experiment ID list
    ids_path = SCRIPT_DIR / "phase_a_experiment_ids.txt"
    ids_path.write_text("\n".join(all_ids) + "\n")

    print(f"Generated {len(all_ids)} configs in {CONFIG_DIR}")
    print(f"Experiment IDs written to {ids_path}")
    print()
    print("Summary:")
    for enc, schemas in counts.items():
        for sch, n in schemas.items():
            print(f"  PA_{enc}_{sch}_s01..s10: {n} configs")
    print()
    print("First 5 IDs:")
    for x in all_ids[:5]:
        print(f"  {x}")
    print("...")


if __name__ == "__main__":
    main()
