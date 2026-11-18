"""CPU checks for the round-4 job entry, credential rule, and resume state."""

import os
import subprocess
import sys
from pathlib import Path

from schema_study.generation_ledger import completed_ids, mark_generation, mark_score, read_json, remaining_steps, select_pending
from schema_study.predictions import parse_prediction_text
from schema_study.schemas import load_schema

CODE = Path("/home/b5ac/freddieyu.b5ac/project_1")
ANALYSIS = Path("/home/b5ac/freddieyu.b5ac/projects/project_1/environments/analysis/bin/python")


def _clean_env() -> dict[str, str]:
    return {"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/tmp"), "USER": os.environ.get("USER", "user")}


def test_project_token_switch_is_not_a_credential():
    allowed = _clean_env()
    allowed["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    allowed["TOKENIZERS_PARALLELISM"] = "false"
    completed = subprocess.run(
        [sys.executable, str(CODE / "scripts/slurm/inspect_env.py"), "cpu"],
        check=False,
        capture_output=True,
        text=True,
        env=allowed,
    )
    assert "credential_check=pass" in completed.stdout
    blocked = dict(allowed)
    blocked["HF_TOKEN"] = "dummy"
    failed = subprocess.run(
        [sys.executable, str(CODE / "scripts/slurm/inspect_env.py"), "cpu"],
        check=False,
        capture_output=True,
        text=True,
        env=blocked,
    )
    assert "credential_check=stop" in failed.stdout
    assert "dummy" not in failed.stdout


def test_resume_skips_completed_requests():
    requests = [{"request_id": f"r{i}"} for i in range(4)]
    done = {"r0", "r1"}
    assert [row["request_id"] for row in select_pending(requests, done, "resume", 2)] == ["r2", "r3"]
    assert select_pending(requests, done, "first", 2) == []


def test_score_failure_keeps_completed_generation(tmp_path):
    mark_generation(tmp_path, {"complete": True, "completed_nonrepeat": 48, "model_called": False})
    mark_score(tmp_path, {"complete": False, "error_type": "NameError"})
    assert remaining_steps(tmp_path) == ["score"]
    assert read_json(tmp_path / "status_generation.json")["complete"] is True


def test_saved_answers_score_through_the_job_entry(tmp_path):
    source = (CODE / "src/schema_study/smoke_qwen.py").read_text(encoding="utf-8")
    assert "_read_requests" not in source
    completed = subprocess.run(
        [str(ANALYSIS), "-m", "schema_study.smoke_qwen", "settle", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        cwd=CODE,
        env={**os.environ, "PYTHONPATH": str(CODE / "src"), "PYTHONNOUSERSITE": "1"},
    )
    assert completed.returncode == 0, completed.stderr
    generation = read_json(tmp_path / "status_generation.json")
    score = read_json(tmp_path / "status_score.json")
    assert generation["complete"] is True
    assert generation["model_called"] is False
    assert score["complete"] is True
    assert completed_ids([]) == set()


def test_job_script_stays_on_dev_and_uses_nil_batch():
    script = (CODE / "scripts/slurm/gpu_smoke.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --export=NIL" in script
    assert "\nsrun" not in script
    assert "/environments/training/bin/python" in script
    from schema_study.smoke_qwen import config

    settings = config()
    assert settings["splits"] == ["train", "dev"]
    assert "test" not in settings["splits"]
    model = Path("/home/b5ac/freddieyu.b5ac/projects/project_1/models/Qwen__Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218")
    for name in ("config.json", "tokenizer.json", "model.safetensors.index.json"):
        assert (model / name).is_file()


def test_empty_prediction_is_valid_and_distinct_from_unparsed_text():
    schema = load_schema(CODE / "configs/schemas/biored.yaml")
    empty = parse_prediction_text('{"candidate_id":"c","labels":[]}', expected_candidate_id="c", schema=schema, level="P4")
    broken = parse_prediction_text("not json", expected_candidate_id="c", schema=schema, level="P4")
    assert empty["output_status"] == "valid" and empty["predicted_labels"] == []
    assert broken["output_status"] == "missing"
