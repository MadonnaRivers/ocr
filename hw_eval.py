"""Word-level handwritten test: IIIT-INDIC-HW-WORDS crops through the fused parser."""
import json, re, sys, unicodedata, tempfile
from pathlib import Path
from PIL import Image, ImageOps
from rapidfuzz.distance import Levenshtein
import docparse
sys.stdout.reconfigure(encoding="utf-8")
H = int(sys.argv[1]) if len(sys.argv) > 1 else 56
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
root = Path(__file__).parent / "hwtest"
tmp = Path(tempfile.mkdtemp())

def clean(md):
    md = re.sub(r"<!--.*?-->|!\[.*?\]\(.*?\)", " ", md)
    md = re.sub(r"[#|*\-]+", " ", md)
    return unicodedata.normalize("NFC", re.sub(r"\s+", "", md))

for lang in ("hindi", "tamil"):
    lab = json.load(open(root / f"{lang}.json", encoding="utf-8"))
    ex = cer_sum = n_chars = empty = 0; bad = []
    for fn, gt in list(lab.items())[:N]:
        im = Image.open(root / lang / fn).convert("L")
        h = H; w = max(1, round(im.width * h / im.height))
        im = im.resize((w, h), Image.LANCZOS)
        canvas = Image.new("L", (max(900, w + 200), 300), 255); canvas.paste(im, (100, 100)); im = canvas
        p = tmp / "w.png"; im.save(p)
        try: pred = clean(docparse.to_markdown(docparse.parse_document(p)))
        except Exception as e: pred = ""
        g = unicodedata.normalize("NFC", re.sub(r"\s+", "", gt))
        d = Levenshtein.distance(g, pred); cer_sum += d; n_chars += len(g)
        ex += pred == g; empty += pred == ""
        if pred != g and len(bad) < 8: bad.append((g, pred))
    n = min(N, len(lab))
    print(f"{lang}: {n} words | exact {100*ex/n:.1f}% | CER {100*cer_sum/n_chars:.1f}% | empty {empty}")
    for g, p in bad: print("   GT", g, "| OCR", p)
