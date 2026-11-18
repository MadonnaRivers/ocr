"""Comparison engines for the real docs (run with the MAIN project venv, which has
docling + pytesseract):
  1. Tesseract 5 (tel+eng for docB, eng for docA), 300 dpi, --psm 3 (auto layout)
  2. IBM Docling (Heron layout + TableFormer; text layer for digital PDFs,
     Tesseract tel+eng OCR for scans)
usage: .venv\\Scripts\\python.exe -I ocr/scripts/compare_real.py <ocr_dir>
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
D, O = ROOT / "real_docs", ROOT / "real_out"
TESS = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = str(ROOT.parent / "tessdata")
os.environ["PATH"] = r"C:\Program Files\Tesseract-OCR;" + os.environ["PATH"]
LANGS = {"docA": "eng", "docB": "tel+eng"}

import fitz  # noqa: E402
import pytesseract  # noqa: E402
from PIL import Image  # noqa: E402

pytesseract.pytesseract.tesseract_cmd = TESS

# ---- Tesseract -----------------------------------------------------------------
for pdf in sorted(D.glob("*.pdf")):
    t0 = time.perf_counter()
    out = []
    with fitz.open(pdf) as doc:
        for pg in doc:
            pix = pg.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            out.append(pytesseract.image_to_string(img, lang=LANGS[pdf.stem], config="--psm 3"))
    (O / f"{pdf.stem}.tesseract.txt").write_text("\n\n<!-- page break -->\n\n".join(out), encoding="utf-8")
    print(f"tesseract {pdf.name}: {time.perf_counter() - t0:.1f}s")

# ---- Docling -------------------------------------------------------------------
from docling.datamodel.base_models import InputFormat  # noqa: E402
from docling.datamodel.pipeline_options import (  # noqa: E402
    EasyOcrOptions, PdfPipelineOptions, TableFormerMode, TesseractCliOcrOptions)
from docling.document_converter import DocumentConverter, PdfFormatOption  # noqa: E402

for pdf in sorted(D.glob("*.pdf")):
    langs = LANGS[pdf.stem].split("+")
    opts = PdfPipelineOptions()
    opts.do_ocr = True
    if pdf.stem == "docB":  # docling's Tesseract wrapper crashes on this page (KeyError 'text')
        opts.ocr_options = EasyOcrOptions(lang=["te", "en"], force_full_page_ocr=True)
    else:
        opts.ocr_options = TesseractCliOcrOptions(lang=langs, tesseract_cmd=TESS)
    opts.do_table_structure = True
    opts.table_structure_options.mode = TableFormerMode.ACCURATE
    opts.table_structure_options.do_cell_matching = True
    conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    t0 = time.perf_counter()
    res = conv.convert(str(pdf))
    md = res.document.export_to_markdown()
    (O / f"{pdf.stem}.docling.md").write_text(md, encoding="utf-8")
    print(f"docling {pdf.name}: {time.perf_counter() - t0:.1f}s")
