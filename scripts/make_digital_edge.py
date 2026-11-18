"""Second set of born-digital PDFs printed by headless Microsoft Edge (Chromium/Skia),
which writes a correct ToUnicode text layer for Indic scripts - unlike the MuPDF
Story PDFs (samples/*_digital.pdf), whose Devanagari/Gujarati text layer is broken.
Both are kept: the broken one mimics the many real Indian PDFs with bad font maps.

usage: python -I scripts/make_digital_edge.py <ocr_dir>
writes samples/<id>_edge.pdf
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(sys.argv[1]).resolve()
S = ROOT / "samples"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

import importlib.util  # noqa: E402

# Reuse DOCS / to_html from make_samples without re-running it.
src = (Path(__file__).parent / "make_samples.py").read_text(encoding="utf-8")
head = src.split("manifest = {}")[0]
ns = {"__name__": "make_samples_lib"}
sys.argv = [sys.argv[0], str(ROOT)]
exec(compile(head, "make_samples.py", "exec"), ns)

CSS = ns["CSS"].replace("font-family: nir;", "font-family: 'Nirmala UI';").split("body {", 1)
CSS = "@page { size: A4; margin: 18mm; }\nbody {" + CSS[1]

tmp = S / "_html"
tmp.mkdir(exist_ok=True)
for doc_id, d in ns["DOCS"].items():
    h = tmp / f"{doc_id}.html"
    h.write_text(f"<!doctype html><meta charset='utf-8'><style>{CSS}</style>{ns['to_html'](d['blocks'])}",
                 encoding="utf-8")
    out = S / f"{doc_id}_edge.pdf"
    subprocess.run([EDGE, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={out}", h.as_uri()], check=True, timeout=60,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("wrote", out.name, out.stat().st_size)
