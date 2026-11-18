"""OCR + cell-aware table rebuild on the real docs. Writes real_out/<doc>.fixed.md
and <doc>.fixed.p<N>.cells.png (detected cells + assigned lines).

usage: python -I scripts/run_layout_fix.py <ocr_dir> [pdf ...]
"""
import sys
import time
from pathlib import Path

import fitz
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT))
import indra_ocr as engine  # noqa: E402
from layout_fix import cells_from_image, cells_from_pdf_page, rebuild_page  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pdfs = [Path(p) for p in sys.argv[2:]] or sorted((ROOT / "real_docs").glob("*.pdf"))
O = ROOT / "real_out"
conv = engine.load(threads=4, mode="ocr", furniture=True)
SCALE = 200 / 72

for pdf in pdfs:
    doc = fitz.open(str(pdf))
    out_md = []
    t_all = time.perf_counter()
    for i, page in conv.convert(pdf):
        t0 = time.perf_counter()
        fp = doc[i]
        lines = [{"box": l.box, "text": l.text} for l in page.lines]
        cells, src = [], "none"
        if len(fp.get_text().strip()) > 50:  # born-digital page: use vector rectangles
            cells, src = cells_from_pdf_page(fp, SCALE), "pdf-vectors"
        if len(cells) < 2:
            pix = fp.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE), colorspace=fitz.csGRAY)
            gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
            cells = []
            for b in page.blocks:
                if b["label"] == "Table":
                    cells += cells_from_image(gray, b["bbox"])
            src = "image-rules"
        md_in = page.markdown if page.markdown is not None else page.text
        md, n = rebuild_page(md_in, page.blocks, lines, cells, SCALE, scanned=(src == "image-rules"))
        out_md.append(md)
        # debug overlay
        im = fp.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE))
        im = Image.frombytes("RGB", (im.width, im.height), im.samples)
        d = ImageDraw.Draw(im, "RGBA")
        for c in cells:
            d.rectangle([c.x0, c.y0, c.x1, c.y1], outline=(22, 163, 74, 255), width=3)
            d.text((c.x0 + 3, c.y0 + 2), f"r{c.r0}c{c.c0}", fill=(22, 163, 74, 255))
        for l in lines:
            d.rectangle(l["box"], outline=(37, 99, 235, 150), width=1)
        im.resize((1100, im.height * 1100 // im.width)).save(O / f"{pdf.stem}.fixed.p{i + 1}.cells.png")
        print(f"{pdf.name} p{i + 1}: cells={len(cells)} from {src}, tables rebuilt={n}, "
              f"fix {1000 * (time.perf_counter() - t0):.0f} ms")
    (O / f"{pdf.stem}.fixed.md").write_text("\n\n<!-- page break -->\n\n".join(out_md), encoding="utf-8")
    print(f"  total {time.perf_counter() - t_all:.2f}s")
