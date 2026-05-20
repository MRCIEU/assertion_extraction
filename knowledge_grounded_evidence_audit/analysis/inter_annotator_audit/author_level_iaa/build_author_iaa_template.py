#!/usr/bin/env python3
"""Build author-level IAA labeling template (TSV + minimal .xlsx) from sampled_targets.csv.

Run from repo anywhere: python3 build_author_iaa_template.py
"""
import csv
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
SAMPLED = (
    HERE.parents[2] / "data" / "processed" / "inter_annotator_audit" / "sampled_targets.csv"
)
OUT_TSV = HERE / "author_iaa_labeling_template.tsv"
OUT_REF = HERE / "REFERENCE_only_after_blind_labeling__heuristic_labels.tsv"
OUT_XLSX = HERE / "author_iaa_labeling_workbook.xlsx"

HEADERS = [
    "row_idx",
    "target_id",
    "pmid",
    "entity_pair_family",
    "entity_a_text",
    "entity_a_type",
    "entity_b_text",
    "entity_b_type",
    "abstract_text",
    "author_family_label_FILL_IN",
    "author_confidence_0_to_1_FILL_IN",
    "author_rationale_one_sentence_FILL_IN",
    "annotator_initials_FILL_IN",
    "date_completed_YYYY_MM_DD_FILL_IN",
]


def col_letter(n):
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def cell_inline(r, c, val):
    col = col_letter(c)
    xml = escape(val).replace("\n", "&#10;").replace("\r", "")
    return (
        f'<c r="{col}{r}" t="inlineStr">'
        f"<is><t xml:space=\"preserve\">{xml}</t></is></c>"
    )


def sheet_xml(rows):
    out = ['<sheetData>']
    for ri, row in enumerate(rows, start=1):
        out.append(f'<row r="{ri}">')
        for ci, val in enumerate(row, start=1):
            out.append(cell_inline(ri, ci, val))
        out.append("</row>")
    out.append("</sheetData>")
    return "".join(out)


def build_xlsx(sheet_blind, sheet_ref, path):
    s1 = sheet_xml(sheet_blind)
    s2 = sheet_xml(sheet_ref)
    workbook_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets>
  <sheet name="1_blind_labeling" sheetId="91" r:id="rId1"/>
  <sheet name="2_REF_post_blind" sheetId="92" r:id="rId2"/>
</sheets>
</workbook>"""
    rels_wb = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"
    Target="worksheets/sheet1.xml" Id="rId1"/>
  <Relationship Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"
    Target="worksheets/sheet2.xml" Id="rId2"/>
  <Relationship Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles"
    Target="styles.xml" Id="rId3"/>
</Relationships>"""
    sheet1 = f"""<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
{s1}
</worksheet>"""
    sheet2 = f"""<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
{s2}
</worksheet>"""
    styles = """<?xml version="1.0" encoding="UTF-8"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="1"><font><sz val="11"/><color theme="1"/><name val="Calibri"/></font></fonts>
  <fills count="1"><fill><patternFill patternType="none"/></fill></fills>
  <borders count="1"><border/></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>
</styleSheet>"""
    root_rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
    Target="xl/workbook.xml" Id="rId1"/>
</Relationships>"""
    ct = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml"
    ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml"
    ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/worksheets/sheet2.xml"
    ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml"
    ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", ct)
        z.writestr("_rels/.rels", root_rels)
        z.writestr("xl/workbook.xml", workbook_xml)
        z.writestr("xl/_rels/workbook.xml.rels", rels_wb)
        z.writestr("xl/styles.xml", styles)
        z.writestr("xl/worksheets/sheet1.xml", sheet1)
        z.writestr("xl/worksheets/sheet2.xml", sheet2)


def main():
    rows = []
    with SAMPLED.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 30, f"expected 30 sampled rows, got {len(rows)}"

    blind_rows: list[list[str]] = [HEADERS]
    ref_rows: list[list[str]] = [
        ["target_id", "heuristic_expected_label_REFERENCE_ONLY", "construction_confidence_REFERENCE"],
    ]

    for i, r in enumerate(rows, start=1):
        blind_rows.append(
            [
                str(i),
                r["target_id"],
                r["pmid"],
                r["entity_pair_family"],
                r["entity_a_text"],
                r["entity_a_type"],
                r["entity_b_text"],
                r["entity_b_type"],
                r["abstract_text"],
                "",
                "",
                "",
                "",
                "",
            ]
        )
        ref_rows.append(
            [
                r["target_id"],
                r["heuristic_expected_label"],
                r.get("construction_confidence", ""),
            ]
        )

    with OUT_TSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerows(blind_rows)

    with OUT_REF.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerows(ref_rows)

    build_xlsx(blind_rows, ref_rows, OUT_XLSX)
    print("Wrote:", OUT_TSV, OUT_REF, OUT_XLSX, sep="\n  ")


if __name__ == "__main__":
    main()
