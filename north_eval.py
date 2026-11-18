import json, re, sys, time, unicodedata
from collections import Counter
from pathlib import Path
from rapidfuzz.distance import Levenshtein
import docparse
sys.stdout.reconfigure(encoding="utf-8")
R = Path(__file__).parent / "north_docs"
def norm(s):
    s = re.sub(r"<!--.*?-->|!\[.*?\]\(.*?\)", " ", s); s = re.sub(r"[#|*]+|-{3,}", " ", s)
    return unicodedata.normalize("NFC", re.sub(r"\s+", "", s))
def sp(s): return unicodedata.normalize("NFC", re.sub(r"\s+", "", s))
rows = []
for lang in ["hindi", "marathi", "gujarati", "punjabi", "urdu"]:
    gt = (R / f"{lang}.gt.txt").read_text(encoding="utf-8")
    cells = [c for t in json.loads((R / f"{lang}.tables.json").read_text(encoding="utf-8")) for r in t for c in r]
    keys = re.findall(r"EHF/HL/\d+/\d+|\d{2}-\d{2}-\d{4}|[\d,]{6,}|\d{6}", gt)
    for var in ["scan", "hardscan"]:
        t0 = time.perf_counter(); res = docparse.parse_document(R / f"{lang}_{var}.pdf"); dt = time.perf_counter() - t0
        out = norm(docparse.to_markdown(res)); g = sp(gt)
        cer = Levenshtein.distance(g, out) / len(g)
        cg, co = Counter(g), Counter(out); ov = sum((cg & co).values()); P, Rc = ov / max(1, len(out)), ov / len(g)
        f1 = 2 * P * Rc / max(1e-9, P + Rc)
        kr = sum(sp(k) in out for k in keys) / len(keys)
        cr = sum(sp(c) in out for c in cells) / len(cells)
        rows.append((lang, var, cer, f1, kr, cr, dt))
        (R / f"out_{lang}_{var}.md").write_text(docparse.to_markdown(res), encoding="utf-8")
print("| Language | Scan | Char accuracy (1-CER) | Char F1 | Key values found | Table cells found | s/page |\n|---|---|---|---|---|---|---|")
for l, v, c, f, k, cr, dt in rows:
    print(f"| {l} | {v} | {100*(1-c):.1f}% | {100*f:.1f}% | {100*k:.0f}% | {100*cr:.0f}% | {dt:.1f} |")
for l in ["hindi", "marathi", "gujarati", "punjabi", "urdu"]:
    rr = [r for r in rows if r[0] == l]; print(f"AVG {l}: acc {100*(1-sum(r[2] for r in rr)/2):.1f}%  F1 {100*sum(r[3] for r in rr)/2:.1f}%")
