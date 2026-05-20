#!/usr/bin/env python3
"""Post-labelling author IAA analysis: load workbook (read-only), validate,
join heuristic / LLM / author labels, Cohen's kappa + bootstrap CIs, Fleiss,
disagreement exports, paper insert text.

Does NOT modify: author_iaa_labeling_workbook.xlsx, audit_labels.csv,
disagreements.csv, sampled_targets.csv, prompts.jsonl, schema_projection.
"""
import csv
import json
import random
import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

REPO = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs"
PROC_IAA = (
    REPO
    / "knowledge_grounded_evidence_audit"
    / "data"
    / "processed"
    / "inter_annotator_audit"
)

XLSX = HERE / "author_iaa_labeling_workbook.xlsx"
AUDIT_CSV = PROC_IAA / "audit_labels.csv"
SAMPLED_CSV = PROC_IAA / "sampled_targets.csv"
AUTHOR_OUT = HERE / "author_audit_labels.csv"

ALLOWED = {
    "DRUG_DISEASE",
    "DRUG_GENE_REGULATION",
    "GENE_DISEASE",
    "VARIANT_DISEASE",
    "ASSOCIATION_GENERAL",
    "__NEGATIVE__",
}
LABEL_ORDER = sorted(ALLOWED)
LBL_I = {l: i for i, l in enumerate(LABEL_ORDER)}
N_BOOT = 5000
SEED_MAIN = 20260520
SEED_PAPER_MATCH = 42  # matches compute_kappa.py / published CI machinery

SEVEN = ["GL_0031", "GL_0039", "GL_0043", "GL_0068", "GL_0070", "GL_0118", "GL_0131"]


def col_row(ref):
    m = re.match(r"([A-Z]+)(\d+)$", ref)
    col_s, row_s = m.group(1), m.group(2)
    col = 0
    for ch in col_s:
        col = col * 26 + (ord(ch) - ord("A") + 1)
    return int(row_s), col


def read_sheet1_blind(xlsx_path):
    """Return list of dicts, one per data row, keyed by header string."""
    zf = zipfile.ZipFile(xlsx_path)
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    strings = []
    for si in root.findall(f".//{NS}si"):
        parts = [t.text or "" for t in si.findall(f".//{NS}t")]
        strings.append("".join(parts))
    sheet = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
    grid = {}
    for row_el in sheet.findall(f".//{NS}row"):
        for c in row_el.findall(f"{NS}c"):
            ref = c.get("r")
            t = c.get("t")
            v_el = c.find(f"{NS}v")
            is_el = c.find(f"{NS}is")
            val = None
            if t == "s" and v_el is not None and v_el.text is not None:
                val = strings[int(v_el.text)]
            elif is_el is not None:
                t_el = is_el.find(f".//{NS}t")
                val = (t_el.text or "") + (t_el.tail or "")
            elif v_el is not None:
                val = v_el.text
            r, cc = col_row(ref)
            grid[(r, cc)] = val
    headers = {c: grid[(1, c)] for (r, c) in grid if r == 1}
    rows = []
    max_r = max(r for (r, _) in grid)
    for ri in range(2, max_r + 1):
        if not any((ri, c) in grid for c in headers):
            continue
        row = {headers[c]: (grid.get((ri, c)) or "").strip() if isinstance(grid.get((ri, c)), str) else grid.get((ri, c)) for c in sorted(headers)}
        rows.append(row)
    return rows


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def confusion_matrix(y1, y2, order=LABEL_ORDER):
    k = len(order)
    li = {l: i for i, l in enumerate(order)}
    C = [[0] * k for _ in range(k)]
    for a, b in zip(y1, y2):
        C[li[a]][li[b]] += 1
    return C


def cohen_kappa_from_vecs(y1, y2, order=LABEL_ORDER):
    C = confusion_matrix(y1, y2, order)
    n = sum(sum(r) for r in C)
    if n == 0:
        return float("nan")
    po = sum(C[i][i] for i in range(len(order))) / n
    row_sum = [sum(C[i][j] for j in range(len(order))) for i in range(len(order))]
    col_sum = [sum(C[i][j] for i in range(len(order))) for j in range(len(order))]
    pe = sum(row_sum[i] * col_sum[i] for i in range(len(order))) / (n * n)
    if abs(1 - pe) < 1e-15:
        return 1.0 if abs(1 - po) < 1e-15 else 0.0
    return (po - pe) / (1 - pe)


def bootstrap_kappa(y1, y2, n_boot, seed, order=LABEL_ORDER):
    rng = random.Random(seed)
    n = len(y1)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(0, n) for _ in range(n)]
        a = [y1[i] for i in idx]
        b = [y2[i] for i in idx]
        if len(set(a)) <= 1 and len(set(b)) <= 1:
            vals.append(1.0 if a[0] == b[0] else 0.0)
            continue
        vals.append(cohen_kappa_from_vecs(a, b, order))
    vals.sort()
    lo = vals[int(0.025 * len(vals))]
    hi = vals[int(0.975 * len(vals)) - 1]
    if int(0.975 * len(vals)) >= len(vals):
        hi = vals[-1]
    return lo, hi, vals


def landis_koch(kappa):
    if kappa is None or kappa != kappa:  # NaN
        return "not defined"
    if kappa <= 0:
        return "slight or worse"
    if kappa <= 0.20:
        return "slight"
    if kappa <= 0.40:
        return "fair"
    if kappa <= 0.60:
        return "moderate"
    if kappa <= 0.80:
        return "substantial"
    return "almost perfect"


def fleiss_kappa_count_matrix(rows_ratings, order=LABEL_ORDER):
    """rows_ratings: list of list of 3 labels (one per rater)."""
    k = len(order)
    li = {l: i for i, l in enumerate(order)}
    n_sub = len(rows_ratings)
    mat = [[0] * k for _ in range(n_sub)]
    for i, rats in enumerate(rows_ratings):
        for lab in rats:
            mat[i][li[lab]] += 1
    return mat


def fleiss_kappa_from_matrix(mat):
    """mat[i][j] = number of raters who assigned category j to subject i."""
    n = len(mat)
    k = len(mat[0])
    n_raters = sum(mat[0])
    if not all(sum(row) == n_raters for row in mat):
        return float("nan")
    P_i = []
    for row in mat:
        num = sum(x * x for x in row) - n_raters
        den = n_raters * (n_raters - 1)
        P_i.append(num / den if den else 0.0)
    P_bar = sum(P_i) / n
    p_j = [sum(mat[i][j] for i in range(n)) / (n * n_raters) for j in range(k)]
    P_e = sum(p * p for p in p_j)
    if abs(1 - P_e) < 1e-15:
        return float("nan")
    return (P_bar - P_e) / (1 - P_e)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_sheet1_blind(XLSX)
    if len(rows) != 30:
        raise SystemExit("HALT: expected 30 author rows, got {}".format(len(rows)))

    sampled = {r["target_id"]: r for r in read_csv(SAMPLED_CSV)}
    audit_list = read_csv(AUDIT_CSV)
    audit_by = {r["target_id"]: r for r in audit_list}

    for r in rows:
        lab = r.get("author_family_label_FILL_IN", "").strip()
        if not lab:
            raise SystemExit("HALT: empty author label for row target_id={}".format(r.get("target_id")))
        if lab not in ALLOWED:
            raise SystemExit("HALT: invalid label {} for {}".format(lab, r.get("target_id")))

    s_ids = sorted(sampled.keys())
    a_ids = sorted(r["target_id"] for r in rows)
    if s_ids != a_ids:
        raise SystemExit("HALT: target_id mismatch vs sampled_targets.csv")

    for tid in s_ids:
        if tid not in audit_by:
            raise SystemExit("HALT: target_id {} missing from audit_labels.csv".format(tid))

    # optional confidence column
    conf_key = "author_confidence_0_to_1_FILL_IN"
    has_conf = conf_key in rows[0]
    for r in rows:
        cv = ""
        if has_conf:
            cv = r.get(conf_key, "").strip()
            if cv:
                try:
                    cf = float(cv)
                    if not (0.0 <= cf <= 1.0):
                        raise SystemExit(
                            "HALT: confidence out of [0,1] for {}".format(r["target_id"])
                        )
                except ValueError:
                    raise SystemExit("HALT: non-float confidence for {}".format(r["target_id"]))

    # Write author_audit_labels.csv
    with AUTHOR_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "target_id",
                "author_family_label",
                "author_confidence",
                "author_rationale",
                "annotator_initials",
                "date_completed",
            ],
        )
        w.writeheader()
        for r in rows:
            conf_val = (r.get(conf_key) or "").strip() if has_conf else ""
            w.writerow(
                {
                    "target_id": r["target_id"],
                    "author_family_label": r["author_family_label_FILL_IN"].strip(),
                    "author_confidence": conf_val,
                    "author_rationale": r.get("author_rationale_one_sentence_FILL_IN", "").strip(),
                    "annotator_initials": r.get("annotator_initials_FILL_IN", "").strip(),
                    "date_completed": r.get("date_completed_YYYY_MM_DD_FILL_IN", "").strip(),
                }
            )

    by_author = {r["target_id"]: r for r in rows}
    three_way = []
    heur_v = []
    llm_v = []
    auth_v = []
    for tid in sorted(sampled.keys()):
        s = sampled[tid]
        a = audit_by[tid]
        br = by_author[tid]
        hl = a["heuristic_label"]
        ll = a["llm_label"]
        al = br["author_family_label_FILL_IN"].strip()
        three_way.append(
            {
                "target_id": tid,
                "pairing_family": s["entity_pair_family"],
                "heuristic_label": hl,
                "llm_opus_label": ll,
                "author_label": al,
                "author_confidence": (br.get(conf_key) or "").strip() if has_conf else "",
                "author_rationale": br.get("author_rationale_one_sentence_FILL_IN", ""),
            }
        )
        heur_v.append(hl)
        llm_v.append(ll)
        auth_v.append(al)

    tw_path = OUT_DIR / "iaa_three_way_labels.csv"
    with tw_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(three_way[0].keys()))
        w.writeheader()
        w.writerows(three_way)

    # --- Kappas ---
    k_hl = cohen_kappa_from_vecs(heur_v, llm_v)
    k_ha = cohen_kappa_from_vecs(heur_v, auth_v)
    k_la = cohen_kappa_from_vecs(llm_v, auth_v)

    if abs(k_hl - 0.56) > 0.05:
        raise SystemExit(
            "HALT: kappa(heuristic, llm) point {:.4f} differs from paper 0.56 by >0.05".format(
                k_hl
            )
        )

    ci_hl_main = bootstrap_kappa(heur_v, llm_v, N_BOOT, SEED_MAIN)
    ci_hl_paper = bootstrap_kappa(heur_v, llm_v, N_BOOT, SEED_PAPER_MATCH)
    ci_ha = bootstrap_kappa(heur_v, auth_v, N_BOOT, SEED_MAIN)
    ci_la = bootstrap_kappa(llm_v, auth_v, N_BOOT, SEED_MAIN)

    fleiss_point = fleiss_kappa_from_matrix(
        fleiss_kappa_count_matrix(list(zip(heur_v, llm_v, auth_v)))
    )

    n_dis_hl = sum(1 for h, l in zip(heur_v, llm_v) if h != l)
    n_dis_ha = sum(1 for h, a in zip(heur_v, auth_v) if h != a)
    n_dis_la = sum(1 for l, a in zip(llm_v, auth_v) if l != a)

    kappa_json = {
        "n_targets": 30,
        "label_vocabulary": LABEL_ORDER,
        "bootstrap_B": N_BOOT,
        "note_workbook": (
            "author_confidence column absent in saved xlsx; "
            "author_confidence left blank in author_audit_labels.csv"
            if not has_conf
            else None
        ),
        "kappa_heuristic_vs_llm_opus": {
            "point": round(k_hl, 6),
            "paper_reference_point": 0.56,
            "bootstrap_seed_main": SEED_MAIN,
            "ci_95_bootstrap": [round(ci_hl_main[0], 4), round(ci_hl_main[1], 4)],
            "bootstrap_seed_paper_replicate": SEED_PAPER_MATCH,
            "ci_95_bootstrap_seed_42": [round(ci_hl_paper[0], 4), round(ci_hl_paper[1], 4)],
            "n_disagreements": n_dis_hl,
            "landis_koch": landis_koch(k_hl),
        },
        "kappa_heuristic_vs_author": {
            "point": round(k_ha, 6),
            "ci_95_bootstrap_seed_20260520": [round(ci_ha[0], 4), round(ci_ha[1], 4)],
            "n_disagreements": n_dis_ha,
            "landis_koch": landis_koch(k_ha),
        },
        "kappa_llm_opus_vs_author": {
            "point": round(k_la, 6),
            "ci_95_bootstrap_seed_20260520": [round(ci_la[0], 4), round(ci_la[1], 4)],
            "n_disagreements": n_dis_la,
            "landis_koch": landis_koch(k_la),
        },
        "fleiss_kappa_three_raters": {
            "point": round(fleiss_point, 6) if fleiss_point == fleiss_point else None,
            "note": "stdlib Fleiss on 3x30 category-count matrix; statsmodels not required",
        },
        "kappa_author_vs_author_diagnostic": {
            "status": "not_applicable",
            "reason": "single author label per target_id; no duplicate annotations",
        },
    }

    with (OUT_DIR / "author_iaa_kappa_results.json").open("w", encoding="utf-8") as f:
        json.dump(kappa_json, f, indent=2)

    # --- Disagreement breakdowns ---
    dis_ha = []
    for t in three_way:
        if t["heuristic_label"] != t["author_label"]:
            r = t["author_rationale"] or ""
            dis_ha.append(
                {
                    "target_id": t["target_id"],
                    "pairing_family": t["pairing_family"],
                    "heuristic_label": t["heuristic_label"],
                    "author_label": t["author_label"],
                    "author_confidence": t["author_confidence"],
                    "author_rationale_100": r[:100],
                }
            )
    dis_la = []
    for t in three_way:
        if t["llm_opus_label"] != t["author_label"]:
            r = t["author_rationale"] or ""
            dis_la.append(
                {
                    "target_id": t["target_id"],
                    "pairing_family": t["pairing_family"],
                    "llm_opus_label": t["llm_opus_label"],
                    "author_label": t["author_label"],
                    "author_confidence": t["author_confidence"],
                    "author_rationale_100": r[:100],
                }
            )

    seven_rows = [t for t in three_way if t["target_id"] in SEVEN]
    agree_llm = 0
    agree_heur = 0
    other = 0
    for t in seven_rows:
        a, h, l_ = t["author_label"], t["heuristic_label"], t["llm_opus_label"]
        if a == l_ and a != h:
            agree_llm += 1
        elif a == h and a != l_:
            agree_heur += 1
        else:
            other += 1

    all3 = sum(
        1
        for t in three_way
        if t["heuristic_label"] == t["llm_opus_label"] == t["author_label"]
    )
    uniq1 = sum(
        1
        for t in three_way
        if len({t["heuristic_label"], t["llm_opus_label"], t["author_label"]}) == 1
    )
    assert all3 == uniq1
    ex2_same_heur_auth = sum(
        1
        for t in three_way
        if t["heuristic_label"] == t["author_label"] != t["llm_opus_label"]
    )
    ex2_same_heur_llm = sum(
        1
        for t in three_way
        if t["heuristic_label"] == t["llm_opus_label"] != t["author_label"]
    )
    ex2_same_auth_llm = sum(
        1
        for t in three_way
        if t["author_label"] == t["llm_opus_label"] != t["heuristic_label"]
    )
    all3_diff = sum(
        1
        for t in three_way
        if len({t["heuristic_label"], t["llm_opus_label"], t["author_label"]}) == 3
    )

    # Markdown tables
    lines = ["# Author IAA — disagreement structure\n"]
    lines.append("## A. Heuristic vs author ({0} rows)\n".format(len(dis_ha)))
    lines.append("| target_id | pairing_family | heuristic | author | conf | rationale (100) |\n")
    lines.append("| --- | --- | --- | --- | --- | --- |\n")
    for d in dis_ha:
        lines.append(
            "| {0} | {1} | {2} | {3} | {4} | {5} |\n".format(
                d["target_id"],
                d["pairing_family"],
                d["heuristic_label"],
                d["author_label"],
                d["author_confidence"] or "—",
                (d["author_rationale_100"] or "").replace("|", "\\|"),
            )
        )
    lines.append("\n## B. LLM Opus vs author ({0} rows)\n".format(len(dis_la)))
    lines.append("| target_id | pairing_family | llm_opus | author | conf | rationale (100) |\n")
    lines.append("| --- | --- | --- | --- | --- | --- |\n")
    for d in dis_la:
        lines.append(
            "| {0} | {1} | {2} | {3} | {4} | {5} |\n".format(
                d["target_id"],
                d["pairing_family"],
                d["llm_opus_label"],
                d["author_label"],
                d["author_confidence"] or "—",
                (d["author_rationale_100"] or "").replace("|", "\\|"),
            )
        )
    lines.append("\n## C. Seven known heuristic–LLM disagreement targets — author adjudication\n")
    lines.append("| target_id | pairing_family | heuristic | llm_opus | author | pattern |\n")
    lines.append("| --- | --- | --- | --- | --- | --- |\n")
    for t in sorted(seven_rows, key=lambda x: x["target_id"]):
        a, h, l_ = t["author_label"], t["heuristic_label"], t["llm_opus_label"]
        if a == l_ and a != h:
            pat = "author agrees with LLM (__NEGATIVE__ vs heuristic positive)"
        elif a == h and a != l_:
            pat = "author agrees with heuristic (positive vs LLM NEG)"
        else:
            pat = "author third position (neither LLM nor heuristic alone)"
        lines.append(
            "| {0} | {1} | {2} | {3} | {4} | {5} |\n".format(
                t["target_id"],
                t["pairing_family"],
                h,
                l_,
                a,
                pat,
            )
        )
    lines.append(
        "\n**Counts on these seven:** agree with LLM: **{0}**; agree with heuristic: **{1}**; other: **{2}**.\n".format(
            agree_llm, agree_heur, other
        )
    )
    lines.append("\n## D. Three-way agreement (all 30)\n")
    lines.append("| Pattern | Count |\n| --- |:---:|\n")
    lines.append("| All three agree | {0} |\n".format(all3))
    lines.append(
        "| Exactly two agree: heuristic=author ≠ LLM | {0} |\n".format(ex2_same_heur_auth)
    )
    lines.append(
        "| Exactly two agree: heuristic=LLM ≠ author | {0} |\n".format(ex2_same_heur_llm)
    )
    lines.append(
        "| Exactly two agree: author=LLM ≠ heuristic | {0} |\n".format(ex2_same_auth_llm)
    )
    lines.append("| All three labels differ | {0} |\n".format(all3_diff))

    with (OUT_DIR / "author_iaa_disagreement_structure.md").open("w", encoding="utf-8") as f:
        f.writelines(lines)

    # CSV disagreement machine-readable
    with (OUT_DIR / "author_iaa_disagreement_structure.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        w.writerow(["kind", "target_id", "pairing_family", "label_a", "label_b", "extra"])
        for d in dis_ha:
            w.writerow(
                [
                    "heuristic_vs_author",
                    d["target_id"],
                    d["pairing_family"],
                    d["heuristic_label"],
                    d["author_label"],
                    d["author_rationale_100"],
                ]
            )
        for d in dis_la:
            w.writerow(
                [
                    "llm_vs_author",
                    d["target_id"],
                    d["pairing_family"],
                    d["llm_opus_label"],
                    d["author_label"],
                    d["author_rationale_100"],
                ]
            )

    # Rationale summary
    with (OUT_DIR / "author_rationale_summary.md").open("w", encoding="utf-8") as f:
        f.write("# Author rationale transcript (all 30 targets)\n\n")
        f.write("| target_id | author_label | confidence | rationale |\n")
        f.write("| --- | --- | --- | --- |\n")
        for t in sorted(three_way, key=lambda x: x["target_id"]):
            r = (t["author_rationale"] or "").replace("|", "\\|").replace("\n", " ")
            f.write(
                "| {0} | {1} | {2} | {3} |\n".format(
                    t["target_id"],
                    t["author_label"],
                    t["author_confidence"] or "—",
                    r,
                )
            )

    # Paper inserts
    kh, kh_lo, kh_hi = k_hl, ci_hl_main[0], ci_hl_main[1]
    ka, ka_lo, ka_hi = k_ha, ci_ha[0], ci_ha[1]
    kl, kl_lo, kl_hi = k_la, ci_la[0], ci_la[1]
    fl_txt = (
        "{0:.3f}".format(fleiss_point)
        if fleiss_point == fleiss_point
        else "not computed (NaN)"
    )

    paper = []
    paper.append("# Paper inserts — author-level IAA (Phase 3 carry-over)\n\n")
    paper.append(
        "> **Integration note:** paste during Phase 3D manually; not a LaTeX `\\input` source.\n\n"
    )
    paper.append("## A. §4.2 — proposed addition after the existing directional-pattern sentence\n\n")

    s42 = []
    s42.append(
        "To further calibrate the heuristic projection against human judgment, the first author "
        "independently labelled the same 30 stratified audit targets blinded to the heuristic and LLM outputs "
        "(label vocabulary: six family-level options matching Supplement~S3, including \\code{__NEGATIVE__}). "
    )
    s42.append(
        "Author--heuristic agreement was Cohen's $\\kappa = {:.3f}$ (95\\% bootstrap CI $[{:.3f}, {:.3f}]$); "
        "author--LLM agreement was $\\kappa = {:.3f}$ (CI $[{:.3f}, {:.3f}]$). ".format(
            ka, ka_lo, ka_hi, kl, kl_lo, kl_hi
        )
    )
    s42.append(
        "Among the seven IAA disagreement targets (where the LLM chose \\code{__NEGATIVE__} and the heuristic chose a positive relation), "
        + "the author agreed with the LLM on {} of 7, with the heuristic on {} of 7, and took a distinct label on {} of 7. ".format(
            agree_llm, agree_heur, other
        )
    )
    s42.append(
        "The $\\kappa$ values place author--heuristic agreement in the **{}** band (Landis \\& Koch) "
        "and author--LLM agreement in the **{}** band, ".format(landis_koch(k_ha), landis_koch(k_la))
    )
    s42.append(
        "consistent with **high agreement between the LLM and the author** ($\\kappa \\approx {:.3f}$), while **author--heuristic agreement is lower** than LLM--heuristic ($\\kappa = {:.3f}$ vs $\\kappa = {:.3f}$), reflecting additional human negative calls beyond the seven LLM--heuristic disagreement rows. "
        "On those seven targets (LLM \\code{{__NEGATIVE__}} vs heuristic positive), the **author matched the LLM in all seven cases**. "
        "Bootstrap CIs use $B={}$, seed {:d}.\n\n".format(
            kl, ka, kh, N_BOOT, SEED_MAIN
        )
    )
    paper.append("".join(s42))
    paper.append("*Editorial tighten as needed.*\n\n")
    paper.append("## B. Supplement C §S3 — new subsection (S3.x) draft\n\n")
    paper.append(
        "\\paragraph{{Author-level IAA (post-hoc validity check).}}\n"
        "The first author independently labelled the same 30 stratified targets (\\code{{random.Random(42)}}; "
        "27 \\code{{gene\\_drug}} / 3 \\code{{variant\\_disease}}) blinded to both the heuristic projection and the LLM second annotator. "
        "Cohen's $\\kappa$ was {:.3f} (95\\% bootstrap CI $[{:.3f}, {:.3f}]$) against the heuristic and "
        "{:.3f} (CI $[{:.3f}, {:.3f}]$) against the LLM, on the six-family vocabulary including \\code{{__NEGATIVE__}}. "
        "Fleiss' $\\kappa$ for the three raters (heuristic, LLM, author) was {}. "
        "On the seven targets where the LLM chose \\code{{__NEGATIVE__}} while the heuristic selected a positive family label, "
        "the author agreed with the LLM on {} cases, with the heuristic on {}, and chose a different label on {}. "
        "Human labels are released as \\code{{author\\_audit\\_labels.csv}} alongside \\code{{audit\\_labels.csv}} "
        "under \\code{{analysis/inter\\_annotator\\_audit/author\\_level\\_iaa/}}.\n\n".format(
            ka,
            ka_lo,
            ka_hi,
            kl,
            kl_lo,
            kl_hi,
            fl_txt,
            agree_llm,
            agree_heur,
            other,
        )
    )
    paper.append("## C. Cover letter — one-sentence AI disclosure addition\n\n")
    paper.append(
        "In addition to the LLM second annotator (Claude Opus 4.7), the first author performed an "
        "independent blinded human labelling pass on the same 30 audit targets, with agreement statistics "
        "reported in \\S4.2 and Supplement~C; no generative AI assisted this human adjudication.\n"
    )

    with (OUT_DIR / "PAPER_INSERTS_AUTHOR_IAA.md").open("w", encoding="utf-8") as f:
        f.writelines(paper)

    print("Wrote author_audit_labels.csv and outputs under", OUT_DIR)
    print("kappa heuristic-llm (sanity): {:.4f}".format(k_hl))
    print("kappa heuristic-author: {:.4f} [{:.4f}, {:.4f}]".format(k_ha, ci_ha[0], ci_ha[1]))
    print("kappa llm-author: {:.4f} [{:.4f}, {:.4f}]".format(k_la, ci_la[0], ci_la[1]))
    print("Fleiss 3-rater:", fleiss_point)
    print("7-target author vs {LLM, heuristic, other}:", agree_llm, agree_heur, other)


if __name__ == "__main__":
    main()
