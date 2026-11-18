"""Score the OCR engine (fused parser) on north_docs80: 80 docs x 5 languages."""
import json, re, sys, time, unicodedata
from collections import Counter
from pathlib import Path
from rapidfuzz.distance import Levenshtein
import docparse

sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).parent / "north_docs80"
LANGS = sys.argv[1:] or ["hindi", "marathi", "gujarati", "punjabi", "urdu"]


def sp(s):
    return unicodedata.normalize("NFC", re.sub(r"\s+", "", s))


def norm(md):
    md = re.sub(r"<!--.*?-->|!\[.*?\]\(.*?\)", " ", md)
    return sp(re.sub(r"[#|*]+|-{3,}", " ", md))


def noinr(s):
    return s.replace("₹", "")


res = {}
t_all = time.perf_counter()
for lang in LANGS:
    rows = []
    for k in range(80):
        stem = f"{lang}_{k:02d}"
        pdf = R / f"{stem}_{'hard' if k % 2 else 'clean'}.pdf"
        gt = (R / f"{stem}.gt.txt").read_text(encoding="utf-8")
        cells = [c for t in json.loads((R / f"{stem}.tables.json").read_text(encoding="utf-8")) for r in t for c in r]
        keys = re.findall(r"EHF/HL/\d+/\d+|\d{2}-\d{2}-\d{4}|[\d,]{6,}", gt)
        try:
            md = docparse.to_markdown(docparse.parse_document(pdf))
        except Exception as e:
            md = ""
        raw = norm(md)
        out, g = noinr(raw), noinr(sp(gt))
        d = Levenshtein.distance(g, out)
        co, cg = Counter(out), Counter(g)
        ov = sum((co & cg).values())
        P, Rc = ov / max(1, len(out)), ov / max(1, len(g))
        f1 = 2 * P * Rc / max(1e-9, P + Rc)
        cell_ok = sum(noinr(sp(c)) in out for c in cells) / len(cells)
        tabs = json.loads((R / f"{stem}.tables.json").read_text(encoding="utf-8"))
        prow = [[sp(c) for c in l.strip().strip("|").split("|")] for l in md.split(chr(10)) if l.lstrip().startswith("|")]
        ok_rows = tot_rows = 0
        for t in tabs:
            for row in t:
                tot_rows += 1
                g_ = {noinr(sp(c)) for c in row}
                ok_rows += any(len(pr) == len(row) and len(g_ & {noinr(c) for c in pr}) >= max(1, len(row) // 2 + 1 if len(row) > 2 else 1) for pr in prow)
        key_ok = sum(noinr(sp(x)) in out for x in keys) / max(1, len(keys))
        inr_n = sp(gt).count("₹")
        rows.append(dict(struct=ok_rows / max(1, tot_rows), k=k, hard=bool(k % 2), dist=d, n=len(g), f1=f1, cell=cell_ok, key=key_ok,
                         inr_gt=inr_n, inr_out=min(inr_n, raw.count("₹")), perfect=(d == 0)))
    res[lang] = rows
    print("done", lang, f"{time.perf_counter() - t_all:.0f}s", file=sys.stderr)
json.dump(res, open(R / "results.json", "w"), indent=0)


def agg(rows):
    n = sum(r["n"] for r in rows)
    acc = 100 * (1 - sum(r["dist"] for r in rows) / n)
    m = lambda f: 100 * sum(r[f] for r in rows) / len(rows)
    inr_gt = sum(r["inr_gt"] for r in rows)
    return (acc, m("f1"), m("key"), m("cell"), m("struct"), 100 * sum(r["dist"] / r["n"] < 0.05 for r in rows) / len(rows),
            100 * sum(r["inr_out"] for r in rows) / inr_gt if inr_gt else float("nan"))


print("| Language | Docs | Char accuracy | Char F1 | Key values | Table cells | Table rows with right column count | Docs >=95% | Rupee sign read |")
print("|---|---|---|---|---|---|---|---|---|")
for lang in LANGS:
    a = agg(res[lang])
    print(f"| {lang} | 80 | {a[0]:.1f}% | {a[1]:.1f}% | {a[2]:.1f}% | {a[3]:.1f}% | {a[4]:.1f}% | {a[5]:.0f}% | {a[6]:.0f}% |")
print("\nClean vs hard scan (char accuracy / char F1):")
for lang in LANGS:
    c, h = agg([r for r in res[lang] if not r["hard"]]), agg([r for r in res[lang] if r["hard"]])
    print(f"  {lang:9s} clean {c[0]:.1f}% / {c[1]:.1f}%   hard {h[0]:.1f}% / {h[1]:.1f}%")
