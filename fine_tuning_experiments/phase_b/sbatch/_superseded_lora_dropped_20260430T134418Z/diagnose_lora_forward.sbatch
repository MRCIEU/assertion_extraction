#!/bin/bash -l
#SBATCH -J PB_diag
#SBATCH -A brics.b5ac
#SBATCH -p workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --output=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/phase_b/diag_lora_fwd_%j.out
#SBATCH --error=/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/logs/phase_b/diag_lora_fwd_%j.err

# Diagnose why LoRA-trained models produce bit-identical 100%-NEGATIVE
# predictions despite trained adapter+classifier weights.
#
# Hypotheses tested:
#  H1: ModulesToSaveWrapper routes through ORIGINAL (frozen) classifier
#      during eval, NOT the trained modules_to_save.default copy.
#  H2: LoRA adapter is disabled / scaling=0 during forward.
#  H3: Predictions actually DO vary, but our argmax is on a wrong dim.

set -euo pipefail
export PYTHONPATH=/home/b5ac/freddieyu.b5ac/project_1
export HF_HOME=/lus/lfs1aip2/projects/b5ac/project_1/hf_hub_cache
export TRANSFORMERS_CACHE=/lus/lfs1aip2/projects/b5ac/project_1/hf_hub_cache
cd /home/b5ac/freddieyu.b5ac/project_1

/usr/bin/python3.11 - <<'PY'
import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

print("=" * 70)
print("PEFT/transformers versions")
print("=" * 70)
import peft, transformers
print(f"  peft         : {peft.__version__}")
print(f"  transformers : {transformers.__version__}")

RUN = Path("/lus/lfs1aip2/projects/b5ac/project_1/fine_tuning_experiments/runs/phase_b/PB_PB_LR_T1B_s99")
ckpt = torch.load(RUN/"checkpoints/stage_t1_end.pt", map_location="cpu", weights_only=False)
sd   = ckpt["model_state_dict"]
label2id = ckpt["label2id"]
id2label = {v: k for k, v in label2id.items()}
model_name = ckpt["model_name"]
print(f"  model_name   : {model_name}")
print(f"  label2id     : {label2id}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  device       : {device}")

print("\n" + "=" * 70)
print("(1) Fresh model + LoRA wrap, then load state_dict")
print("=" * 70)
base = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=8)
lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                  target_modules=["query","value"],
                  modules_to_save=["classifier"],
                  bias="none", task_type="SEQ_CLS")
torch.manual_seed(0)
model = get_peft_model(base, lcfg).to(device)
print(f"  active adapter: {model.active_adapter}")
print(f"  active adapters: {model.active_adapters}")

# Load
incompat = model.load_state_dict(sd, strict=False)
print(f"  missing keys (first 5): {incompat.missing_keys[:5]}")
print(f"  unexpected keys (first 5): {incompat.unexpected_keys[:5]}")

# Show the wrapped classifier
print(f"\n  classifier type: {type(model.base_model.model.classifier).__name__}")
cls = model.base_model.model.classifier
for attr in dir(cls):
    if 'adapter' in attr.lower() or 'modules_to' in attr or attr == 'original_module':
        try:
            v = getattr(cls, attr)
            print(f"    {attr}: {type(v).__name__}")
        except Exception: pass

# Print classifier weight comparison
if hasattr(cls, 'original_module'):
    om = cls.original_module
    print(f"  original classifier.weight  norm: {om.weight.norm().item():.6f}")
if hasattr(cls, 'modules_to_save'):
    mts = cls.modules_to_save
    if isinstance(mts, torch.nn.ModuleDict):
        for adapter, mod in mts.items():
            print(f"  mts[{adapter}] classifier.weight  norm: {mod.weight.norm().item():.6f}")

print("\n" + "=" * 70)
print("(2) Forward on dev sample - which classifier is used?")
print("=" * 70)
tok = AutoTokenizer.from_pretrained(model_name)
samples = [
    ("Aspirin treats headache.", "DRUG_DISEASE"),
    ("This study is unrelated.", "__NEGATIVE__"),
    ("BRCA1 mutation causes cancer.", "GENE_DISEASE"),
]
model.eval()
for txt, gold in samples:
    enc = tok(txt, return_tensors="pt", truncation=True, max_length=384).to(device)
    with torch.no_grad():
        out = model(**enc)
    logits = out.logits[0]
    pred_idx = int(logits.argmax().item())
    pred_lab = id2label[pred_idx]
    print(f"  '{txt[:40]:40}' -> {pred_lab:24} (gold={gold})")
    print(f"    logits: {[f'{l:.3f}' for l in logits.tolist()]}")

print("\n" + "=" * 70)
print("(3) Force-route through modules_to_save.default classifier")
print("=" * 70)
# Manually check if the wrapper routes correctly.
import peft.utils.other as peft_other
print(f"  ModulesToSaveWrapper class: {type(cls).__name__}")
print(f"  cls._active_adapter: {getattr(cls, '_active_adapter', 'N/A')}")

# Try calling the modules_to_save.default directly
mts_classifier = cls.modules_to_save["default"]
print(f"  Direct call to modules_to_save['default']:")
# We need the pooled output. Let's bypass classifier and run encoder manually.
backbone = model.base_model.model.bert  # the BERT model under PEFT wrapper
backbone.eval()
with torch.no_grad():
    for txt, gold in samples:
        enc = tok(txt, return_tensors="pt", truncation=True, max_length=384).to(device)
        bert_out = backbone(**enc)
        pooled = bert_out.pooler_output
        logits_via_mts = mts_classifier(pooled)[0]
        logits_via_orig = cls.original_module(pooled)[0] if hasattr(cls, 'original_module') else None
        pred_mts  = id2label[int(logits_via_mts.argmax().item())]
        pred_orig = id2label[int(logits_via_orig.argmax().item())] if logits_via_orig is not None else "n/a"
        print(f"  '{txt[:30]:30}' mts->{pred_mts:24}  orig->{pred_orig}")
        print(f"     mts logits:  {[f'{l:.3f}' for l in logits_via_mts.tolist()]}")
        if logits_via_orig is not None:
            print(f"     orig logits: {[f'{l:.3f}' for l in logits_via_orig.tolist()]}")

print("\n" + "=" * 70)
print("(4) Check the Wrapper forward behavior")
print("=" * 70)
# Hook the wrapper to see which path it takes
wrapper = cls
# ModulesToSaveWrapper has a `disable_adapters` flag
print(f"  cls.disable_adapters: {getattr(wrapper, 'disable_adapters', 'N/A')}")
print(f"  cls.active_adapter:   {getattr(wrapper, 'active_adapter', 'N/A')}")
# Print the source forward
import inspect
print("\n  ModulesToSaveWrapper.forward source:")
src = inspect.getsource(type(wrapper).forward)
for i, ln in enumerate(src.splitlines()):
    print(f"    {i+1:3d}| {ln}")
PY
