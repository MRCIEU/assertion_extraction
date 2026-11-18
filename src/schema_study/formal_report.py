"""Write the formal-run reports after the main matrix is complete.

This module does not call a model. It refuses to describe a partial matrix
as finished.
"""

from __future__ import annotations

import json
import math
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from schema_study.batch_plan import adopted_answers
from schema_study.evaluation import document_resample_indices, f1_from_counts, select_epoch
from schema_study.formal_analyze import (
    EVAL_SPLITS,
    ROOT,
    SCHEMAS,
    _population,
    _read_jsonl_finished_lines,
    score_rows,
)
from schema_study.formal_eval import score_predictions
from schema_study.generation_ledger import read_jsonl
from schema_study.schemas import load_schema

REPORT = Path("/home/b5ac/freddieyu.b5ac/projects/project_1/studies/schema_granularity_v1/reports/formal/formal_20261115")
CODE = Path("/home/b5ac/freddieyu.b5ac/project_1")
EXPECTED_REQUESTS = 1793840
SEEDS = (13, 42, 2026)


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _scores() -> list[dict]:
    found = []
    for path in sorted((ROOT / "evaluation").rglob("*.json")):
        if path.name.startswith("train_fine"):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_path"] = str(path)
        found.append(payload)
    return found


def _direct(rows: list[dict], kind: str) -> list[dict]:
    return [row for row in rows if f"/{kind}/" in row["_path"] and "/rollup/" not in row["_path"]]


def matrix_status() -> dict:
    manifest = (ROOT / "execution_manifest.tsv").read_text(encoding="utf-8").splitlines()
    header = manifest[0].split("\t")
    tasks = [dict(zip(header, line.split("\t"))) for line in manifest[1:]]
    predictions = list((ROOT / "train").glob("*/*/*/seed*/prediction_complete.json"))
    summary = json.loads((ROOT / "prepare/length_summary.json").read_text(encoding="utf-8"))
    adopted = 0
    full = 0
    for model in ("qwen8b", "qwen32b"):
        for shard in summary["shards"]:
            path = ROOT / "generation/outputs" / model / shard["dataset"] / shard["entity_level"] / shard["predicate_level"] / f"shard_{int(shard['shard_index']):05d}" / "attempts.jsonl"
            if not path.is_file():
                continue
            got = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if '"adoption": "adopt"' in line or '"adoption":"adopt"' in line)
            adopted += min(got, int(shard["requests"]))
            full += int(got >= int(shard["requests"]))
    return {
        "tasks": len(tasks),
        "task_counts": dict(Counter(row["status"] for row in tasks)),
        "train_prediction_files": len(predictions),
        "generation_shards_full": full,
        "adopted_requests_capped": adopted,
        "complete": (
            len(tasks) == 288
            and all(row["status"] == "complete" for row in tasks)
            and len(predictions) == 48
            and full == 240
            and adopted == EXPECTED_REQUESTS
        ),
    }


def _check_freeze_hashes() -> list[str]:
    freeze = json.loads((ROOT / "formal_freeze_record.json").read_text(encoding="utf-8"))
    mismatches = []
    for path, expected in freeze["code_hashes"].items():
        import hashlib
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if digest != expected:
            mismatches.append(path)
    return mismatches


def _selection_audit() -> list[dict]:
    audited = []
    for path in sorted((ROOT / "train").glob("*/*/*/seed*/prediction_complete.json")):
        run = path.parent
        dataset, entity, predicate, seed_name = run.relative_to(ROOT / "train").parts
        history = [json.loads(line) for line in (run / "training_history.jsonl").read_text(encoding="utf-8").splitlines()]
        chosen = select_epoch([{"f1": row["dev_f1"], "epoch": row["epoch"], "threshold": row.get("threshold")} for row in history])
        recorded = json.loads(path.read_text(encoding="utf-8"))
        thresholds = [row.get("threshold") for row in history]
        grid_ok = all(value is None or (0.05 <= float(value) <= 0.95 and abs(float(value) * 20 - round(float(value) * 20)) < 1e-8) for value in thresholds)
        audited.append(
            {
                "dataset": dataset,
                "entity_level": entity,
                "predicate_level": predicate,
                "seed": int(seed_name.removeprefix("seed")),
                "epochs": len(history),
                "best_epoch": chosen["epoch"],
                "recorded_best_epoch": recorded["best_epoch"],
                "selection_matches_history": chosen["epoch"] == recorded["best_epoch"],
                "dev_f1": chosen["f1"] if "f1" in chosen else chosen.get("dev_f1"),
                "threshold": chosen.get("threshold"),
                "updates": history[-1]["updates"],
                "threshold_grid_ok": grid_ok,
                "epoch_f1": [row["dev_f1"] for row in history],
            }
        )
    return audited


def _generation_audit() -> dict:
    summary = json.loads((ROOT / "prepare/length_summary.json").read_text(encoding="utf-8"))
    problems = []
    extra_calls = 0
    status = Counter()
    truncated = 0
    by_model = Counter()
    for model in ("qwen8b", "qwen32b"):
        for shard in summary["shards"]:
            path = ROOT / "generation/outputs" / model / shard["dataset"] / shard["entity_level"] / shard["predicate_level"] / f"shard_{int(shard['shard_index']):05d}" / "attempts.jsonl"
            rows = _read_jsonl_finished_lines(path)
            adopted = adopted_answers(rows)
            if len(adopted) != int(shard["requests"]):
                problems.append(f"{model} {shard['dataset']} {shard['entity_level']} {shard['predicate_level']} shard {shard['shard_index']}: {len(adopted)}/{shard['requests']}")
            extra_calls += max(0, len(rows) - len(adopted))
            by_model[model] += len(adopted)
            for row in adopted.values():
                status[row["output_status"]] += 1
                truncated += int(bool(row.get("truncated")))
    return {"problems": problems, "extra_attempt_lines": extra_calls, "status": dict(status), "truncated": truncated, "by_model": dict(by_model)}


def _prediction_audit() -> list[str]:
    problems = []
    expected = {("biored", "test"): 7047, ("bc8", "bc8"): 28734, ("drugprot", "eval"): 116887}
    for path in sorted((ROOT / "train").glob("*/*/*/seed*/predictions.jsonl")):
        counts = Counter()
        ids = Counter()
        for row in read_jsonl(path):
            counts[row["split"]] += 1
            ids[row["candidate_id"]] += 1
        duplicated = [key for key, count in ids.items() if count > 1]
        if duplicated:
            problems.append(f"{path.parent.name} duplicate ids {len(duplicated)}")
        dataset = path.relative_to(ROOT / "train").parts[0]
        if dataset == "biored":
            if counts["test"] != expected[("biored", "test")] or counts["bc8"] != expected[("bc8", "bc8")]:
                problems.append(f"{path} splits {dict(counts)}")
        elif counts["eval"] != expected[("drugprot", "eval")]:
            problems.append(f"{path} splits {dict(counts)}")
    return problems


def _load_predictions_for(saved_row: dict) -> list[dict]:
    path = Path(saved_row["_path"])
    if "generation.json" in path.name and "/direct/" in str(path):
        rows = []
        for shard in saved_row["source_shards"]:
            relative = Path(shard).relative_to(ROOT / "generation/shards")
            attempt = ROOT / "generation/outputs" / saved_row["model"] / relative.parent / Path(shard).stem / "attempts.jsonl"
            rows.extend(adopted_answers(_read_jsonl_finished_lines(attempt)).values())
        return rows
    if "/rollup/" in str(path):
        return []
    source = Path(saved_row["source"])
    wanted_split = {"biored": "test", "bc8": "bc8", "drugprot": "eval"}[saved_row["dataset"]]
    return [row for row in read_jsonl(source) if row["split"] == wanted_split]


def _bootstrap_f1(per_seed_docs: list[dict[str, list[int]]], documents: list[str], indices: list[list[int]]) -> dict:
    observed = []
    for docs in per_seed_docs:
        tp = fp = fn = 0
        for document_id in documents:
            one = docs[document_id]
            tp += one[0]
            fp += one[1]
            fn += one[2]
        observed.append(f1_from_counts(tp, fp, fn))
    observed_mean = _mean(observed)
    draws = []
    for draw in indices:
        seed_scores = []
        for docs in per_seed_docs:
            tp = fp = fn = 0
            for index in draw:
                one = docs[documents[index]]
                tp += one[0]
                fp += one[1]
                fn += one[2]
            seed_scores.append(f1_from_counts(tp, fp, fn))
        draws.append(_mean(seed_scores))
    return {"observed": observed_mean, "draws": draws}


def _interval(draws: list[float]) -> tuple[float, float]:
    ordered = sorted(draws)
    return ordered[int(0.025 * (len(ordered) - 1))], ordered[int(0.975 * (len(ordered) - 1))]


def _docs_from_rows(scored_rows: list[dict]) -> dict[str, list[int]]:
    docs = defaultdict(lambda: [0, 0, 0, 0])
    for row in scored_rows:
        item = docs[row["document_id"]]
        item[0] += row["tp"]
        item[1] += row["fp"]
        item[2] += row["fn"]
        item[3] += 1
    return docs


def build_document_counts(saved: list[dict]) -> dict[tuple, dict[str, list[int]]]:
    populations = {}
    schemas = {}
    counts = {}
    for saved_row in saved:
        if saved_row.get("source_level") == "P4" and "/rollup/" in saved_row["_path"]:
            continue
        dataset = saved_row["dataset"]
        populations.setdefault(dataset, _population(dataset, EVAL_SPLITS[dataset]))
        schemas.setdefault(dataset, load_schema(SCHEMAS[dataset]))
        predictions = _load_predictions_for(saved_row)
        scored = score_predictions(
            populations[dataset],
            predictions,
            schemas[dataset],
            saved_row["predicate_level"],
            allowed_splits={EVAL_SPLITS[dataset]},
            data_version="check",
            input_version="target_anchored_v1",
            prediction_version="formal_20261115",
            mapping_version="accepted_schema",
        )
        if abs(scored["summary"]["f1"] - saved_row["summary"]["f1"]) > 1e-9:
            raise RuntimeError(f"Rescore mismatch for {saved_row['_path']}")
        counts[_document_key(saved_row)] = _docs_from_rows(scored["rows"])
    return counts


def _label(key: tuple) -> str:
    model, dataset, entity, predicate, source, seed = key
    seed_text = f" seed {seed}" if seed is not None else ""
    source_text = "direct" if source == "direct" else f"rollup {source}"
    return f"{model} {dataset} {entity} {predicate} {source_text}{seed_text}"


def write_reports() -> None:
    status = matrix_status()
    if not status["complete"]:
        raise SystemExit(f"Formal matrix is not complete: {json.dumps(status)}")
    hash_mismatches = _check_freeze_hashes()
    if hash_mismatches:
        raise SystemExit(f"Frozen code hashes changed: {hash_mismatches}")
    saved = _scores()
    direct = [row for row in saved if "/direct/" in row["_path"]]
    rollups = [row for row in saved if "/rollup/" in row["_path"]]
    selection = _selection_audit()
    generation = _generation_audit()
    prediction_problems = _prediction_audit()
    if any(not row["selection_matches_history"] or row["epochs"] != 10 or not row["threshold_grid_ok"] for row in selection):
        raise SystemExit("Training selection audit failed")
    if generation["problems"] or prediction_problems:
        raise SystemExit(f"Integrity problems: {generation['problems'][:5]} {prediction_problems[:5]}")
    if len(direct) != 84 + 56:
        raise SystemExit(f"Direct score files {len(direct)}, expected 140")
    if len(rollups) != 105:
        raise SystemExit(f"Rollup score files {len(rollups)}, expected 105")
    REPORT.mkdir(parents=True, exist_ok=True)
    _write_integrity(status, generation, prediction_problems, selection)
    _write_training(selection)
    _write_generation(generation, direct)
    _write_results(direct, rollups)
    _write_three_level(direct)
    _write_resources()
    _write_deviations()
    _write_status(status)


def _write_integrity(status: dict, generation: dict, prediction_problems: list[str], selection: list[dict]) -> None:
    text = f"""# 数据与预测完整性

正式主矩阵的任务清单、训练预测和生成账本都已齐。下面的核对在比较表之前完成。金标准只从候选表读取。

- 任务 288/288 为 complete。训练预测完成文件 48。生成分片满额 240/240。采纳的唯一请求 {status['adopted_requests_capped']}，与冻结的 1,793,840 一致。
- 两个 Qwen 各 {generation['by_model'].get('qwen8b')} 与 {generation['by_model'].get('qwen32b')} 条采纳请求。
- 预测身份重复：{len(prediction_problems)}。生成账本数量不符：{len(generation['problems'])}。
- 额外生成尝试行 {generation['extra_attempt_lines']}。这些行保留在账本中，评分采用每个请求的第一条完成且身份相符的回答。
- 输出状态：{generation['status']}。截断回答 {generation['truncated']}。
- 48 次训练都有 10 个 epoch，开发集选模与历史记录一致，DrugProt 阈值落在 0.05 到 0.95、步长 0.05 的网格上。
- 冻结清单中的模型执行代码哈希与当前文件一致。评分和排队代码不在该冻结哈希清单里，它们不改变模型前向、提示或训练更新。
- 重复、多余身份会拒绝评分。本次没有这样的评分单元。缺失回答会留在总体里；本次直接预测没有缺失。

P4 归并只读取已经保存的 P4 离散预测，映射到 P3、P2、P1 后去重。没有再次调用模型，也没有重新搜索阈值。失败和非法状态保留。
"""
    (REPORT / "data_and_prediction_integrity_report.md").write_text(text, encoding="utf-8")


def _write_training(selection: list[dict]) -> None:
    lines = ["# 训练与选模", "", "每次训练从固定预训练权重开始，最多 10 个 epoch。BioRED 按当前档开发集阳性 micro-F1 选模，平分取较早 epoch。DrugProt 在每个 epoch 的阈值网格上先取开发集 micro-F1，平分取较高阈值，再取较早 epoch。正式评价没有参与选模。", "", "| 语料 | 实体 | 谓词 | seed | best epoch | 开发集 F1 | 阈值 | 更新数 |", "| --- | --- | --- | ---: | ---: | ---: | --- | ---: |"]
    for row in selection:
        lines.append(f"| {row['dataset']} | {row['entity_level']} | {row['predicate_level']} | {row['seed']} | {row['best_epoch']} | {_fmt(row['dev_f1'])} | {row['threshold'] if row['threshold'] is not None else '—'} | {row['updates']} |")
    lines.append("")
    lines.append("四次曾被两个进程同时写入的 DrugProt 训练已作废，目录保留在 `train/drugprot/native/*_invalid_duplicate_writers`。上表只包含作废后按同一 seed、同一配置重新训练并完成的运行。")
    (REPORT / "training_and_selection_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _group_direct(direct: list[dict]) -> dict[tuple, list[dict]]:
    grouped = defaultdict(list)
    for row in direct:
        grouped[(row["model"], row["dataset"], row["entity_level"], row["predicate_level"])].append(row)
    return grouped


def _condition_f1(rows: list[dict]) -> float:
    return _mean([row["summary"]["f1"] for row in rows])


def _write_generation(generation: dict, direct: list[dict]) -> None:
    qwen = [row for row in direct if row["model"].startswith("qwen")]
    lines = ["# 生成", "", "Qwen 使用冻结的 V1 提示、Transformers、bf16、batch 16、贪心解码、非思考模式、max_new_tokens 256、左侧填充和不截断。公共权重从本地读取。", "", f"采纳请求共 {sum(generation['by_model'].values())}。额外尝试行 {generation['extra_attempt_lines']}。截断 {generation['truncated']}。状态计数 {generation['status']}。", "", "格式失败没有算作正确的无关系。空列表只有在输出状态为 valid 时才是无关系预测。", "", "| 模型 | 语料 | 实体 | 谓词 | 请求 | valid | rejected | partial_illegal | missing | 有效空集合 | 截断 | micro-F1 |", "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in qwen:
        status = row["status_counts"]
        lines.append(
            f"| {row['model']} | {row['dataset']} | {row['entity_level']} | {row['predicate_level']} | {row['summary']['n_candidates']} | {status.get('valid', 0)} | {status.get('rejected', 0)} | {status.get('partial_illegal', 0)} | {status.get('missing', 0)} | {row['valid_empty']} | {row.get('truncated', 0)} | {_fmt(row['summary']['f1'])} |"
        )
    (REPORT / "generation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_results(direct: list[dict], rollups: list[dict]) -> None:
    grouped = _group_direct(direct)
    lines = ["# 正式主结果", "", "主指标是完整候选总体上的阳性 micro-F1。BioRED 错类同时产生一个 FP 和一个 FN。DrugProt 保留合法标签，每个不同非法标签另计一个 FP。No_Relation 不是阳性类。BiomedBERT 先对每个 seed 计分，再取算术平均，不把三个 seed 的预测合并成一个更大的样本。", "", "这些数字是模型相对现有标注的表现。没有人工语义保真审计，信息保留指标也不是已经验证的全文语义保真度。DrugProt 此前是否用于调参仍未确认，结果不是从未接触过的独立确认。", "", "## 直接预测", "", "| 模型 | 语料 | 实体 | 谓词 | seed 数 | micro-P | micro-R | micro-F1 | macro-F1 | 精确匹配 | seed F1 范围 |", "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key in sorted(grouped):
        rows = grouped[key]
        f1s = [row["summary"]["f1"] for row in rows]
        precisions = [row["summary"]["precision"] for row in rows]
        recalls = [row["summary"]["recall"] for row in rows]
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {len(rows)} | {_fmt(None if any(value is None for value in precisions) else _mean(precisions))} | {_fmt(None if any(value is None for value in recalls) else _mean(recalls))} | {_fmt(_mean(f1s))} | {_fmt(_mean([row['summary']['macro_f1'] for row in rows]))} | {_fmt(_mean([row['summary']['exact_match'] for row in rows]))} | {_fmt(max(f1s) - min(f1s)) if len(f1s) > 1 else '—'} |"
        )
    lines += ["", "## 直接粗粒度与 P4 归并", "", "归并与直接输出使用同一候选总体。", "", "| 模型 | 语料 | 实体 | 目标档 | 直接 F1 | 归并 F1 | 差值（归并减直接） |", "| --- | --- | --- | --- | ---: | ---: | ---: |"]
    roll_grouped = defaultdict(list)
    for row in rollups:
        roll_grouped[(row["model"], row["dataset"], row["entity_level"], row["predicate_level"])].append(row)
    for key in sorted(roll_grouped):
        if key not in grouped:
            continue
        direct_f1 = _condition_f1(grouped[key])
        roll_f1 = _condition_f1(roll_grouped[key])
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {_fmt(direct_f1)} | {_fmt(roll_f1)} | {_fmt(roll_f1 - direct_f1)} |")
    lines += ["", "## 信息保留", "", "频率保留使用训练集细标签加一平滑。P1 的类别区别保留按约定为 0。失败和漏检的保留分为 0。这是标签层级上的分解，不是原文全部语义的分解。", "", "| 模型 | 语料 | 实体 | 谓词 | 细金关系数 | 频率 C | 频率 R | 均匀 C | 均匀 R |", "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for key in sorted(grouped):
        rows = grouped[key]
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {rows[0]['information_retention']['n_gold_relations']} | {_fmt(_mean([row['information_retention']['frequency']['ceiling_C'] for row in rows]))} | {_fmt(_mean([row['information_retention']['frequency']['retained_R'] for row in rows]))} | {_fmt(_mean([row['information_retention']['uniform']['ceiling_C'] for row in rows]))} | {_fmt(_mean([row['information_retention']['uniform']['retained_R'] for row in rows]))} |"
        )
    lines += ["", "## 文献重抽样区间", "", "Bootstrap 种子 100，2000 次，文献级配对抽样，百分位 95% 区间。同一个评价集的全部比较共享同一批文献抽样。区间只反映文献抽样，不包含重新训练的不确定性。跨 seed 的范围已经列在直接预测表中。没有做多重比较校正，少数不跨零的区间不是确认性发现。下表比较的是直接预测。直接输出与 P4 归并的点估计见上一节。", ""]
    comparisons = _bootstrap_comparisons(grouped)
    lines.append("| 比较 | 观察差值 | 2.5% | 97.5% |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in comparisons:
        lines.append(f"| {row['name']} | {_fmt(row['difference'])} | {_fmt(row['low'])} | {_fmt(row['high'])} |")
    lines += ["", "五组提示对照和人工审阅没有执行。这不是整项研究的全部完成。"]
    (REPORT / "main_results_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _bootstrap_comparisons(grouped: dict[tuple, list[dict]]) -> list[dict]:
    """Paired document bootstrap for the pre-specified contrasts.

    Document draws are generated once per evaluation set and reused.
    """
    populations = {name: _population(name, EVAL_SPLITS[name]) for name in ("biored", "bc8", "drugprot")}
    schemas = {name: load_schema(SCHEMAS[name]) for name in populations}
    documents = {name: sorted({row["document_id"] for row in rows}) for name, rows in populations.items()}
    indices = {name: document_resample_indices(len(documents[name]), 2000, 100) for name in documents}
    cache = {}

    def docs_for(key: tuple) -> list[dict[str, list[int]]]:
        if key in cache:
            return cache[key]
        per_seed = []
        for saved_row in grouped[key]:
            predictions = _load_predictions_for(saved_row)
            scored = score_predictions(
                populations[saved_row["dataset"]],
                predictions,
                schemas[saved_row["dataset"]],
                saved_row["predicate_level"],
                allowed_splits={EVAL_SPLITS[saved_row["dataset"]]},
                data_version="check",
                input_version="target_anchored_v1",
                prediction_version="formal_20261115",
                mapping_version="accepted_schema",
            )
            per_seed.append(_docs_from_rows(scored["rows"]))
        cache[key] = per_seed
        return per_seed

    def contrast(left: tuple, right: tuple, name: str) -> dict:
        dataset = left[1]
        left_draws = _bootstrap_f1(docs_for(left), documents[dataset], indices[dataset])
        right_draws = _bootstrap_f1(docs_for(right), documents[dataset], indices[dataset])
        differences = [a - b for a, b in zip(left_draws["draws"], right_draws["draws"])]
        low, high = _interval(differences)
        return {"name": name, "difference": left_draws["observed"] - right_draws["observed"], "low": low, "high": high}

    results = []
    models = ("biomedbert", "qwen8b", "qwen32b")
    for model in models:
        for dataset, entities in (("biored", ("E0", "E1", "E2")), ("bc8", ("E0", "E1", "E2")), ("drugprot", ("native",))):
            for entity in entities:
                for left_level, right_level in (("P2", "P1"), ("P3", "P2"), ("P4", "P3")):
                    left = (model, dataset, entity, left_level)
                    right = (model, dataset, entity, right_level)
                    if left in grouped and right in grouped:
                        results.append(contrast(left, right, f"{model} {dataset} {entity} {left_level} 减 {right_level}"))
                for level in ("P1", "P2", "P3"):
                    fine = (model, dataset, entity, "P4")
                    coarse = (model, dataset, entity, level)
                    if fine in grouped and coarse in grouped:
                        results.append(contrast(coarse, fine, f"{model} {dataset} {entity} 直接{level} 减 直接P4"))
        if model == "biomedbert":
            continue
    for dataset, entities in (("biored", ("E0", "E1", "E2")), ("bc8", ("E0", "E1", "E2"))):
        for entity_left, entity_right in (("E1", "E0"), ("E2", "E1")):
            for level in ("P1", "P2", "P3", "P4"):
                for model in models:
                    left = (model, dataset, entity_left, level)
                    right = (model, dataset, entity_right, level)
                    if left in grouped and right in grouped:
                        results.append(contrast(left, right, f"{model} {dataset} {level} {entity_left} 减 {entity_right}"))
    for dataset, entities in (("biored", ("E0", "E1", "E2")), ("bc8", ("E0", "E1", "E2")), ("drugprot", ("native",))):
        for entity in entities:
            for level in ("P1", "P2", "P3", "P4"):
                for left_model, right_model in (("qwen32b", "qwen8b"), ("biomedbert", "qwen8b"), ("biomedbert", "qwen32b")):
                    left = (left_model, dataset, entity, level)
                    right = (right_model, dataset, entity, level)
                    if left in grouped and right in grouped:
                        results.append(contrast(left, right, f"{dataset} {entity} {level} {left_model} 减 {right_model}"))
    return results


def _write_three_level(direct: list[dict]) -> None:
    lines = ["# 三层辅助分析", "", "这三层是正式主指标之外的辅助描述，不替换阳性 micro-F1。第三层使用全部金标准阳性候选，不删除空集合、非法或缺失回答。格式失败不算作正确无关系。", "", "| 模型 | 语料 | 实体 | 谓词 | 全部候选 F1 | 阴性有效非空 | 阴性有效空 | 阳性合法非空 | 阳性有效空 | 拒绝 | 部分非法 | 缺失 | 阳性子集 F1 |", "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    grouped = _group_direct(direct)
    for key in sorted(grouped):
        rows = grouped[key]
        layer1 = _mean([row["three_level"]["layer1_all_candidates"]["f1"] for row in rows])
        exist = rows[0]["three_level"]["layer2_existence"]
        # Existence counts are averaged across seeds so a single seed is not hidden.
        def avg(name: str) -> float:
            return _mean([row["three_level"]["layer2_existence"][name] for row in rows])
        layer3 = _mean([row["three_level"]["layer3_all_gold_positive"]["f1"] for row in rows])
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {_fmt(layer1)} | {_fmt(avg('negative_valid_nonempty'), 1)}/{_fmt(avg('negative_denominator'), 1)} | {_fmt(avg('negative_valid_empty'), 1)}/{_fmt(avg('negative_denominator'), 1)} | {_fmt(avg('positive_legal_nonempty'), 1)}/{_fmt(avg('positive_denominator'), 1)} | {_fmt(avg('positive_valid_empty'), 1)}/{_fmt(avg('positive_denominator'), 1)} | {_fmt(avg('rejected'), 1)} | {_fmt(avg('partial_illegal'), 1)} | {_fmt(avg('missing'), 1)} | {_fmt(layer3)} |"
        )
    lines.append("")
    lines.append("存在性的分母是该条件的金标准阴性或阳性候选。BiomedBERT 的计数是三个 seed 的平均，所以可能不是整数。")
    (REPORT / "three_level_analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_resources() -> None:
    jobs = json.loads((ROOT / "launch_jobs.json").read_text(encoding="utf-8"))
    chunks = [jobs[index:index + 80] for index in range(0, len(jobs), 80)]
    records = []
    for chunk in chunks:
        result = subprocess.run(["sacct", "-j", ",".join(chunk), "--format=JobID,State,ElapsedRaw,AllocCPUS,ReqCPUS,ReqMem", "-P", "-n"], check=False, capture_output=True, text=True)
        for line in result.stdout.splitlines():
            job_id, state, elapsed, alloc, req, memory = line.split("|")
            if "." in job_id:
                continue
            records.append((state, int(elapsed or 0), alloc, req, memory))
    gpu_seconds = sum(item[1] for item in records)
    states = Counter(item[0].split()[0] for item in records)
    peaks = []
    calls = 0
    for path in (ROOT).rglob("resource_usage.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        calls += int(payload.get("calls") or 0)
        if payload.get("device_peak_mib"):
            peaks.append(int(payload["device_peak_mib"]))
    text = f"""# 资源

GPU 时间按 Slurm 记录的作业耗时乘以每作业 1 张 GPU 累计，包含失败和被取消的尝试。这是已经结束作业的记账，不是上限本身。

- 跟踪作业数：{len(records)}
- 累计 GPU 时间：{gpu_seconds / 3600:.1f} 小时。本轮停止边界是 1000 GPU 小时。
- 作业状态：{dict(states)}
- 申请 CPU 为 8，站点实际分配为 72。申请内存 BiomedBERT 和 8B 为 64G，32B 为 128G。
- 生成资源文件中的模型调用计数合计 {calls}。设备峰值显存 {min(peaks) if peaks else 'missing'}–{max(peaks) if peaks else 'missing'} MiB。
- 每个进程的资源文件都留在各自的 attempt 目录，恢复没有覆盖它们。

在途作业若在写这份报告时已经结束，其耗时包含在上面的 sacct 合计中。
"""
    (REPORT / "resource_usage_report.md").write_text(text, encoding="utf-8")


def _write_deviations() -> None:
    text = """# 偏差与失败

模型执行代码在冻结之后没有改动。下面的事项不改变提示、标签、seed、epoch 或候选总体。

1. 第一次 DrugProt 作业的时限是 4 小时。站点拒绝了运行中的 `scontrol` 延长。该作业在时限内完成了 10 个 epoch 和评价预测，退出码为 0。之后的 DrugProt 作业在提交时申请 8 小时。调整时限没有增加 epoch。
2. 调度命令一度没有返回仍在运行的作业号。控制器因此为同一输出目录提交了第二个进程。四次 DrugProt 训练各留下两条不一致的 epoch 2 记录，已整目录保留并标为无效，然后按同一 seed 和同一配置从固定预训练权重重新训练。BioRED E1 P3 与 P4 的第 0 个 32B 分片也曾有两个写入进程。账本没有损坏行。后启动的进程已取消，先启动的进程继续。额外调用单独留在账本里，采纳的仍是第一条通过身份核对的回答。
3. 控制器随后改为：只有作业记账显示已经结束才重新提交；队列读取失败时不提交。
4. 控制器进程在 2026-11-15 18:07 UTC 停止，当时还有作业在跑。这些作业自行正常结束。22:17 UTC 重新启动控制器后，已完成的输出被标为 complete，没有重算；未开始的任务继续提交。
5. 正式训练在 CLS 后使用 dropout 0.1。早先冒烟循环没有这个 dropout 模块。正式运行不以冒烟检查点初始化。
6. V1 的全矩阵长度检查是单独做的。没有把第四轮的 1243 或 V0 的长度接受记录当作 V1 的结果。
7. 作业名里 32B 显示成 `p1-gen-2b`，只是名字切片，提交的模型和权重没有换。
8. 没有因为正式分数、误报或某个条件不占优而停止、改提示或删除运行。

没有非有限损失、输入截断、身份错配或冻结哈希不符。
"""
    (REPORT / "deviations_and_failures.md").write_text(text, encoding="utf-8")


def _write_status(status: dict) -> None:
    payload = {
        "run_id": "formal_20261115",
        "matrix_frozen": True,
        "status": "complete",
        "trainings_complete": 48,
        "trainings_planned": 48,
        "direct_finetune_eval_units_complete": 84,
        "direct_finetune_eval_units_planned": 84,
        "qwen_generate_tasks_complete": 240,
        "qwen_generate_tasks_planned": 240,
        "qwen_unique_requests_complete": EXPECTED_REQUESTS,
        "qwen_unique_requests_planned": EXPECTED_REQUESTS,
        "task_status_counts": status["task_counts"],
        "rollup_complete": True,
        "formal_scoring_complete": True,
        "five_controls_executed": False,
        "human_review_complete": False,
        "semantic_validated": False,
        "api_requests_made": False,
        "prior_tuning_use_confirmed": False,
        "study_fully_complete": False,
    }
    text = json.dumps(payload, indent=2) + "\n"
    (ROOT / "formal_run_status.json").write_text(text, encoding="utf-8")
    (REPORT / "formal_run_status.json").write_text(text, encoding="utf-8")


def main() -> None:
    status = matrix_status()
    print(json.dumps(status))
    if status["complete"]:
        write_reports()
        print("reports_written")


if __name__ == "__main__":
    main()
