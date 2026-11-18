"""Score the OCR engine outputs against the ground truth produced by make_samples.py.

usage: python -I scripts/evaluate.py <ocr_dir> <config_name>

Metrics (all computed here from the saved outputs):
  cer            character error rate, Levenshtein(gt, pred) / len(gt), on the
                 plain-text output in the engine's reading order (order-sensitive)
  char_f1        order-free character F1 (multiset of non-space chars)
  key_recall     share of ground-truth "key tokens" (amounts, dates, IDs, numbers)
                 found verbatim in the output - what matters for lending docs
  script_keep    per expected Indic script: chars of that script in output / in GT
  wrong_script   Indic chars in the output from a script that the GT does not contain
  order_acc      pairwise reading-order accuracy of the non-table GT lines
  tables         count match, shape match, cell exact-match at the same (row,col),
                 and cell recall anywhere in the predicted tables
  layout         block labels the OCR engine emitted
"""
import html
import json
import re
import sys
import unicodedata
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

ROOT = Path(sys.argv[1]).resolve()
CFG = sys.argv[2]
S = ROOT / "samples"
O = ROOT / "outputs" / CFG

BLOCKS = {
    "Devanagari": (0x0900, 0x097F),
    "Bengali": (0x0980, 0x09FF),
    "Gurmukhi": (0x0A00, 0x0A7F),
    "Gujarati": (0x0A80, 0x0AFF),
    "Oriya": (0x0B00, 0x0B7F),
}


def script_of(ch):
    o = ord(ch)
    for name, (lo, hi) in BLOCKS.items():
        if lo <= o <= hi:
            return name
    if ch.isascii() and ch.isalpha():
        return "Latin"
    return None


def norm(s):
    s = unicodedata.normalize("NFC", s)
    s = s.replace("।", "।")
    return re.sub(r"\s+", " ", s).strip()


def strip_md(md):
    md = re.sub(r"<[^>]+>", " ", md)
    md = html.unescape(md)
    md = re.sub(r"^#+\s*", "", md, flags=re.M)
    return md


class TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables, self.row, self.cell, self.in_cell = [], None, "", False

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.tables.append([])
        elif tag == "tr":
            self.row = []
        elif tag in ("td", "th"):
            self.in_cell, self.cell = True, ""
            self.span = int(dict(attrs).get("colspan", 1) or 1)

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self.row.append(norm(self.cell))
            for _ in range(self.span - 1):
                self.row.append("<span>")
            self.in_cell = False
        elif tag == "tr" and self.tables:
            self.tables[-1].append(self.row)

    def handle_data(self, data):
        if self.in_cell:
            self.cell += data


KEY_RE = re.compile(r"[A-Z]{2,}[A-Z0-9/]*\d[\w/]*|\d[\d,./\-]*\d|\d+%|\d")


def key_tokens(gt):
    toks = []
    for t in KEY_RE.findall(gt):
        if len(t) >= 2:
            toks.append(t)
    return toks


def order_accuracy(gt_lines, pred):
    pos = []
    for ln in gt_lines:
        if len(ln) < 6:
            continue
        al = fuzz.partial_ratio_alignment(ln, pred)
        if al and al.score >= 60:
            pos.append(al.dest_start)
    if len(pos) < 2:
        return None, len(pos)
    good = tot = 0
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            tot += 1
            good += pos[i] < pos[j]
    return good / tot, len(pos)


def score(doc_id, stem):
    gt_raw = (S / f"{doc_id}.gt.txt").read_text(encoding="utf-8")
    gt = norm(gt_raw)
    txt = norm((O / f"{stem}.txt").read_text(encoding="utf-8"))
    md = (O / f"{stem}.md").read_text(encoding="utf-8")
    pages = json.loads((O / f"{stem}.json").read_text(encoding="utf-8"))

    cer = Levenshtein.distance(gt, txt) / max(1, len(gt))
    cg, cp = Counter(gt.replace(" ", "")), Counter(txt.replace(" ", ""))
    tp = sum((cg & cp).values())
    prec, rec = tp / max(1, sum(cp.values())), tp / max(1, sum(cg.values()))
    f1 = 2 * prec * rec / max(1e-9, prec + rec)

    keys = key_tokens(gt)
    found = [k for k in keys if k in txt]
    missing = [k for k in keys if k not in txt]

    gs = Counter(script_of(c) for c in gt)
    ps = Counter(script_of(c) for c in txt)
    keep = {s: round(ps.get(s, 0) / gs[s], 3) for s in gs if s and s != "Latin"}
    keep["Latin"] = round(ps.get("Latin", 0) / max(1, gs.get("Latin", 0)), 3) if gs.get("Latin") else None
    wrong = {s: ps[s] for s in ps if s and s != "Latin" and s not in gs}

    # tables
    gt_tables = []
    tf = S / f"{doc_id}.gt_tables.json"
    if tf.exists():
        gt_tables = json.loads(tf.read_text(encoding="utf-8"))
    elif doc_id == "multipage_hi_gu":
        gt_tables = (json.loads((S / "hi_sanction.gt_tables.json").read_text(encoding="utf-8"))
                     + json.loads((S / "gu_kyc.gt_tables.json").read_text(encoding="utf-8")))
    tp_ = TableParser()
    tp_.feed(md)
    pt = tp_.tables
    exact = total = anywhere = shape_ok = 0
    for k, g in enumerate(gt_tables):
        p = pt[k] if k < len(pt) else []
        if p and len(p) == len(g) and all(len(pr) == len(gr) for pr, gr in zip(p, g)):
            shape_ok += 1
        flat_p = {c for r in p for c in r}
        for i, gr in enumerate(g):
            for j, gc in enumerate(gr):
                gc = norm(gc)
                if not gc:
                    continue
                total += 1
                if i < len(p) and j < len(p[i]) and p[i][j] == gc:
                    exact += 1
                if gc in flat_p:
                    anywhere += 1

    gt_lines = [norm(ln) for ln in gt_raw.splitlines() if ln.strip()]
    oacc, n_matched = order_accuracy(gt_lines, norm(strip_md(md)))

    labels = Counter(b["label"] for p in pages for b in (p.get("blocks") or []))
    return {
        "stem": stem,
        "cer": round(cer, 4),
        "char_f1": round(f1, 4),
        "key_recall": round(len(found) / max(1, len(keys)), 3),
        "key_missing": missing,
        "script_keep": keep,
        "wrong_script_chars": wrong,
        "order_acc": None if oacc is None else round(oacc, 3),
        "order_lines_matched": n_matched,
        "tables_gt": len(gt_tables), "tables_pred": len(pt), "table_shape_ok": shape_ok,
        "cell_exact": round(exact / max(1, total), 3),
        "cell_recall_anywhere": round(anywhere / max(1, total), 3),
        "layout_labels": dict(labels),
        "sources": [p["source"] for p in pages],
        "notes": [p["note"] for p in pages if p.get("note")],
    }


metrics = json.loads((O / "metrics.json").read_text(encoding="utf-8"))
rows = []
for r in metrics["results"]:
    if not r["doc_id"]:
        continue
    s = score(r["doc_id"], r["out_stem"])
    s.update({"doc_id": r["doc_id"], "file": r["file"], "mode": r["mode"],
              "median_s": r["median_s"], "s_per_page": r["s_per_page_median"],
              "peak_rss_mb": r["peak_rss_mb"]})
    rows.append(s)
(O / "eval.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

print(f"{'file':28s} {'mode':4s} {'src':5s} {'CER':>6s} {'chF1':>5s} {'keys':>5s} {'order':>5s} "
      f"{'tbl':>5s} {'shape':>5s} {'cellEx':>6s} {'cellAny':>7s} {'s/pg':>5s}  script_keep / wrong")
for s in rows:
    print(f"{s['file']:28s} {s['mode']:4s} {s['sources'][0]:5s} {s['cer']:6.3f} {s['char_f1']:5.3f} "
          f"{s['key_recall']:5.2f} {str(s['order_acc']):>5s} {s['tables_pred']}/{s['tables_gt']:<3d} "
          f"{s['table_shape_ok']:>5d} {s['cell_exact']:6.2f} {s['cell_recall_anywhere']:7.2f} "
          f"{s['s_per_page']:5.2f}  {s['script_keep']} {s['wrong_script_chars'] or ''}")
