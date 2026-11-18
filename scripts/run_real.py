"""Run the OCR engine on real documents in real_docs/, both OCR and auto modes, saving
markdown, json and layout-overlay images to real_out/.

usage: python -I scripts/run_real.py <ocr_dir>
"""
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pypdfium2 as pdfium
from PIL import ImageDraw

import indra_ocr as engine

ROOT = Path(sys.argv[1]).resolve()
D, O = ROOT / "real_docs", ROOT / "real_out"
O.mkdir(exist_ok=True)
COL = {"Section-Header": (220, 38, 38), "Text": (37, 99, 235), "Table": (22, 163, 74),
       "Page-Header": (147, 51, 234), "Page-Footer": (147, 51, 234), "Picture": (202, 138, 4),
       "List-Group": (14, 165, 233), "Form": (234, 88, 12), "Caption": (234, 88, 12)}

for mode in ("ocr", "auto"):
    conv = engine.load(threads=4, mode=mode, furniture=True)
    for pdf in sorted(D.glob("*.pdf")):
        doc = pdfium.PdfDocument(str(pdf))
        pages, md = [], []
        t0 = time.perf_counter()
        for i, p in conv.convert(pdf):
            md.append(p.markdown if p.markdown is not None else p.text)
            pages.append({"page": i + 1, "source": p.source, "note": p.note, "rotation": p.rotation,
                          "skew": p.skew, "blocks": p.blocks, "lines": [asdict(l) for l in p.lines],
                          "timings": p.timings})
            pg = doc[i]
            im = pg.render(scale=conv.page_scale(pg)).to_pil().convert("RGB")
            if p.rotation:
                im = im.rotate(p.rotation, expand=True)
            d = ImageDraw.Draw(im, "RGBA")
            for ln in p.lines:
                d.rectangle(ln.box, outline=(120, 120, 120, 200), width=1)
            for k, b in enumerate(p.blocks):
                c = COL.get(b["label"], (90, 90, 90))
                d.rectangle(b["bbox"], outline=c + (255,), fill=c + (30,), width=4)
                d.text((b["bbox"][0] + 4, b["bbox"][1] + 2), f"{k + 1}.{b['label']}", fill=c + (255,))
            im.resize((im.width * 1100 // im.width, im.height * 1100 // im.width)).save(
                O / f"{pdf.stem}.{mode}.p{i + 1}.layout.png")
            print(f"{pdf.name} {mode} p{i + 1}: {p.source} {p.timings.get('total_ms', 0):.0f}ms "
                  f"lines={len(p.lines)} blocks={[b['label'] for b in p.blocks]} note={p.note}")
        (O / f"{pdf.stem}.{mode}.md").write_text("\n\n<!-- page break -->\n\n".join(md), encoding="utf-8")
        (O / f"{pdf.stem}.{mode}.json").write_text(json.dumps(pages, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  total {time.perf_counter() - t0:.2f}s")
