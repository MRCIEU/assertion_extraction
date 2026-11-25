"""CPU evaluation of the sentence-level certainty model and the small NLI verifier.

Run this module with the hf-hpc interpreter. It loads those two checkpoints and
does not train them. An entailment probability is not a probability of biological truth.
"""

from __future__ import annotations

import json
import tarfile
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from schema_study.paths import get_paths
from schema_study.records import publish_file

HF_HOME = Path("/home/b5ac/freddieyu.b5ac/projects/project_1/cache/huggingface")
CERTAINTY = "pedropei/sentence-level-certainty"
VERIFIER = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
VERIFIER_REVISION = "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"
XSMALL_VERIFIER = "MoritzLaurer/DeBERTa-v3-xsmall-mnli-fever-anli-ling-wanli"
SCIFACT_LABELS = {"SUPPORT": "entailment", "CONTRADICT": "contradiction", "NEI": "neutral"}


def _revision(repo: str) -> str:
    snapshot = HF_HOME / "hub" / f"models--{repo.replace('/', '--')}" / "refs" / "main"
    if snapshot.is_file():
        return snapshot.read_text(encoding="utf-8").strip()
    return "unrecorded"


def _load(repo: str):
    tokenizer = AutoTokenizer.from_pretrained(repo, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(repo, local_files_only=True)
    model.eval()
    config = model.config
    return tokenizer, model, {str(key): value for key, value in dict(config.id2label).items()}


def _probabilities(model, tokenizer, pairs: list[tuple[str, str]], limit: int) -> list[dict]:
    lengths = []
    for premise, hypothesis in pairs:
        token_ids = tokenizer(premise, hypothesis, truncation=False, add_special_tokens=True)["input_ids"]
        lengths.append(len(token_ids))
    results: list[dict | None] = [None] * len(pairs)
    batch: list[tuple[str, str]] = []
    batch_indexes: list[int] = []

    def flush() -> None:
        if not batch:
            return
        encoded = tokenizer(
            [premise for premise, _hypothesis in batch],
            [hypothesis for _premise, hypothesis in batch],
            truncation=False,
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            probabilities = torch.softmax(model(**encoded).logits, dim=-1)
        for index, values in zip(batch_indexes, probabilities.tolist()):
            if any(value != value or value < 0.0 or value > 1.0 for value in values):
                raise ValueError("A model probability was not finite and inside [0, 1].")
            results[index] = {"status": "assessable", "probabilities": values, "length": lengths[index]}
        batch.clear()
        batch_indexes.clear()

    for index, (pair, length) in enumerate(zip(pairs, lengths)):
        if length > limit:
            results[index] = {"status": "unassessable_context", "probabilities": None, "length": length}
            continue
        batch.append(pair)
        batch_indexes.append(index)
        if len(batch) == 8:
            flush()
    flush()
    return [row for row in results if row is not None]


def _scores(confusion: dict[str, Counter], labels: list[str]) -> dict:
    per_class = {}
    correct = 0
    total = 0
    for gold in labels:
        for predicted, count in confusion[gold].items():
            total += count
            if predicted == gold:
                correct += count
        true_positive = confusion[gold][gold]
        predicted_positive = sum(confusion[label][gold] for label in labels)
        gold_positive = sum(confusion[gold].values())
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / gold_positive if gold_positive else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        per_class[gold] = {"precision": precision, "recall": recall, "f1": f1, "support": gold_positive}
    return {"n": total, "accuracy": (correct / total) if total else None, "per_class": per_class, "confusion": {gold: dict(confusion[gold]) for gold in labels}}


def _scifact(tokenizer, model, id2label: dict[str, str]) -> dict:
    names = [id2label[str(index)] for index in range(len(id2label))]
    expected = {"entailment", "neutral", "contradiction"}
    if set(names) != expected:
        return {"status": "label_map_not_the_documented_three_way_map", "id2label": id2label, "executed": False}
    archive = get_paths().raw / "scifact" / "data.tar.gz"
    corpus = {}
    claims = []
    with tarfile.open(archive, "r:gz") as tar:
        corpus_handle = tar.extractfile("data/corpus.jsonl")
        claim_handle = tar.extractfile("data/claims_dev.jsonl")
        if corpus_handle is None or claim_handle is None:
            raise FileNotFoundError("SciFact development claims or corpus are missing from the archive.")
        for line in corpus_handle:
            row = json.loads(line)
            corpus[str(row["doc_id"])] = row
        for line in claim_handle:
            claims.append(json.loads(line))
    pairs = []
    gold = []
    rationale_counts = Counter()
    for claim in claims:
        evidence = claim.get("evidence") or {}
        if not evidence:
            for document_id in claim.get("cited_doc_ids") or []:
                document = corpus.get(str(document_id))
                if document is None:
                    continue
                premise = " ".join(document.get("abstract") or [])
                pairs.append((premise, claim["claim"]))
                gold.append("neutral")
            continue
        for document_id, items in evidence.items():
            document = corpus.get(str(document_id))
            if document is None:
                continue
            sentences = document.get("abstract") or []
            for item in items:
                rationale_counts[item.get("label")] += 1
                indexes = item.get("sentences") or []
                premise = " ".join(sentences[index] for index in indexes if 0 <= index < len(sentences))
                label = SCIFACT_LABELS.get(item.get("label"))
                if not premise or label is None:
                    continue
                pairs.append((premise, claim["claim"]))
                gold.append(label)
    scored = _probabilities(model, tokenizer, pairs, int(model.config.max_position_embeddings))
    confusion = {label: Counter() for label in ("entailment", "neutral", "contradiction")}
    assessable = unassessable = 0
    probabilities = []
    correct_flags = []
    for truth, result in zip(gold, scored):
        if result["status"] != "assessable":
            unassessable += 1
            continue
        assessable += 1
        mapping = {names[index]: value for index, value in enumerate(result["probabilities"])}
        predicted = max(mapping, key=mapping.get)
        confusion[truth][predicted] += 1
        probabilities.append(mapping[predicted])
        correct_flags.append(predicted == truth)
    bins = []
    for lower in (index / 10 for index in range(10)):
        chosen = [flag for probability, flag in zip(probabilities, correct_flags) if lower <= probability < lower + 0.1 or (lower == 0.9 and probability == 1)]
        bins.append({"lower": lower, "n": len(chosen), "accuracy": None if not chosen else sum(chosen) / len(chosen)})
    metrics = _scores(confusion, ["entailment", "neutral", "contradiction"])
    return {
        "task": "claim_verification_given_rationale_or_cited_abstract",
        "rationale_retrieval_scored": False,
        "rationale_label_counts_are_not_model_scores": dict(rationale_counts),
        "neutral_procedure": "A development claim with empty evidence is paired with each cited abstract and labelled neutral. No unrelated sentence is sampled.",
        "claims": len(claims),
        "pairs_constructed": len(pairs),
        "assessable": assessable,
        "unassessable_context": unassessable,
        "metrics": metrics,
        "calibration_bins": bins,
        "entailment_is_biological_truth": False,
    }


def _certainty(tokenizer, model, id2label: dict[str, str]) -> dict:
    labels = model.config.num_labels
    readme = HF_HOME / "hub" / "models--pedropei--sentence-level-certainty" / "snapshots"
    formula = None
    if readme.is_dir():
        for path in readme.glob("*/README.md"):
            text = path.read_text(encoding="utf-8", errors="replace")
            if "1" in text and "6" in text:
                formula = "readme_mentions_the_1_to_6_scale"
                break
    encoded = tokenizer(["The experiment demonstrates a large and consistent effect."], return_tensors="pt", truncation=False)
    with torch.no_grad():
        output = model(**encoded)
    if labels == 1:
        value = float(output.logits.reshape(-1)[0])
        return {"head": "single_output", "example_raw_output": value, "scale_transformation": formula or "not_confirmed_from_the_model_card", "id2label": id2label, "public_benchmark_executed": False}
    probabilities = torch.softmax(output.logits, dim=-1)[0].tolist()
    return {
        "head": "classification",
        "num_labels": labels,
        "id2label": id2label,
        "example_probabilities_sum": sum(probabilities),
        "scale_transformation": formula or "not_confirmed_from_the_model_card",
        "public_benchmark_executed": False,
        "reason": "No official sentence-level evaluation split was present in the local release.",
    }


def _certainty_on_scifact(tokenizer, model) -> dict:
    """Score SciFact development claims. These claims are not certainty labels."""
    archive = get_paths().raw / "scifact" / "data.tar.gz"
    claims = []
    with tarfile.open(archive, "r:gz") as tar:
        handle = tar.extractfile("data/claims_dev.jsonl")
        if handle is None:
            raise FileNotFoundError("SciFact development claims are missing.")
        for line in handle:
            claims.append(json.loads(line)["claim"])
    values = []
    for start in range(0, len(claims), 8):
        encoded = tokenizer(claims[start:start + 8], padding=True, truncation=False, return_tensors="pt")
        lengths = encoded["attention_mask"].sum(dim=1)
        if int(lengths.max()) > int(model.config.max_position_embeddings):
            raise ValueError("A SciFact claim exceeded the certainty model length and was not truncated.")
        with torch.no_grad():
            logits = model(**encoded).logits.reshape(-1).tolist()
        if any(value != value or value in {float("inf"), float("-inf")} for value in logits):
            raise ValueError("A certainty output was not finite.")
        values.extend(float(value) for value in logits)
    ordered = sorted(values)
    return {
        "texts": "scifact_development_claims",
        "n": len(ordered),
        "min": ordered[0],
        "median": ordered[len(ordered) // 2],
        "max": ordered[-1],
        "within_1_to_6": all(1.0 <= value <= 6.0 for value in ordered),
        "official_labelled_split_present": False,
        "this_is_not_a_labelled_accuracy": True,
    }


def adapt_verifier(*_args, **_kwargs) -> dict:
    """Prepare SciFact adaptation without training, and without reusing the reported development claims."""
    return {
        "started": False,
        "model": VERIFIER,
        "revision": VERIFIER_REVISION,
        "selection": "development slice carved from the SciFact training split",
        "reported_development_claims": "baseline reference only; not used to select the adapted verifier",
        "confirmation": "SciFact test claims, held out from selection",
        "reason": "formal verifier training is not authorised",
    }


def main() -> int:
    torch.set_num_threads(4)
    destination = get_paths().assert_output(get_paths().study / "reports" / "experimental_preparation")
    local_certainty = Path("/home/b5ac/freddieyu.b5ac/projects/project_1/cache/huggingface/pedropei-sentence-level-certainty")
    print("Loading the certainty model.", flush=True)
    certainty_tokenizer = AutoTokenizer.from_pretrained(local_certainty, local_files_only=True)
    certainty = AutoModelForSequenceClassification.from_pretrained(local_certainty, local_files_only=True)
    certainty.eval()
    certainty_labels = {str(key): value for key, value in dict(certainty.config.id2label).items()}
    local_verifier = Path("/home/b5ac/freddieyu.b5ac/projects/project_1/models") / "MoritzLaurer__DeBERTa-v3-base-mnli-fever-anli" / VERIFIER_REVISION
    if (local_verifier / "config.json").is_file():
        print("Loading the base verifier.", flush=True)
        verifier_tokenizer, verifier_model, verifier_labels = _load(str(local_verifier))
        verifier_status = _scifact(verifier_tokenizer, verifier_model, verifier_labels)
        verifier_status["id"] = VERIFIER
        verifier_status["revision"] = VERIFIER_REVISION
        verifier_status["executed"] = True
    else:
        verifier_status = {
            "id": VERIFIER,
            "revision": VERIFIER_REVISION,
            "executed": False,
            "reason": "The base checkpoint is not in local storage yet.",
            "xsmall_checkpoint": XSMALL_VERIFIER,
            "xsmall_anonymous_access": "HTTP 401",
        }
    payload = {
        "environment": "hf-hpc",
        "cuda_available": bool(torch.cuda.is_available()),
        "torch": torch.__version__,
        "verifier": verifier_status,
        "certainty": {
            "id": CERTAINTY,
            "revision": "57bb19e0804a77689ca02f2b1d408d162413cdc2",
            "id2label": certainty_labels,
            "num_labels": int(certainty.config.num_labels),
            "not_the_aspect_model": True,
            "architecture": certainty.config.architectures,
        },
        "scifact": verifier_status,
        "certainty_on_scifact_claims": _certainty_on_scifact(certainty_tokenizer, certainty),
        "executed_measurements": ["sentence_level_certainty_raw_output"],
        "trained": False,
    }
    publish_file(destination / "measurement_benchmark.json", lambda temporary: temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8"))
    print(json.dumps({
        "verifier_executed": payload["verifier"]["executed"],
        "certainty_revision": payload["certainty"]["revision"],
        "certainty_on_scifact_claims": payload["certainty_on_scifact_claims"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
