"""OCR: fused parser (OCR + PDF-text verification + cell-aware tables + layout cleanup).

One pipeline, no mode switches. Web UI and REST API on one port:
    .venv\\Scripts\\python.exe app.py          -> http://127.0.0.1:7862
    UI: /            API docs: /docs
    API: POST /api/parse   (multipart: file=<pdf|image>)
         query: format=json|markdown (default json), include_images=true|false (default false)
    curl -F "file=@doc.pdf" http://127.0.0.1:7862/api/parse

Licence: check the model weights' licence before commercial use.
licence is settled (see REPORT.md).
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import gradio as gr
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import ImageDraw

import docparse
import indra_ocr as engine

EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
LABEL_COLORS = {
    "Section-Header": (220, 38, 38), "Title": (220, 38, 38),
    "Text": (37, 99, 235), "List-Group": (14, 165, 233),
    "Table": (22, 163, 74), "Form": (22, 163, 74),
    "Page-Header": (147, 51, 234), "Page-Footer": (147, 51, 234),
    "Caption": (234, 88, 12), "Footnote": (234, 88, 12),
    "Picture": (202, 138, 4), "Table-Of-Contents": (100, 116, 139),
}
LOCK = threading.Lock()  # one ONNX session, one document at a time

CSS = """
.gradio-container {max-width: 1500px !important}
.rendered table {border-collapse: collapse; width: 100%; margin: .6em 0}
.rendered th, .rendered td {border: 1px solid #94a3b8; padding: 4px 8px; vertical-align: top}
.rendered thead th {background: #f1f5f9}
"""


def overlay(img, page, cells=()):
    im = img.copy()
    d = ImageDraw.Draw(im, "RGBA")
    for c in cells:
        d.rectangle([c.x0, c.y0, c.x1, c.y1], outline=(6, 95, 70, 255), width=2)
    for ln in page.lines:
        d.rectangle(list(ln.box), outline=(120, 120, 120, 150), width=1)
    for k, b in enumerate(page.blocks):
        c = LABEL_COLORS.get(b["label"], (100, 100, 100))
        x0, y0, x1, y1 = b["bbox"]
        d.rectangle([x0, y0, x1, y1], outline=c + (255,), fill=c + (28,), width=3)
        tag = f"{k + 1}. {b['label']}"
        d.rectangle([x0, max(0, y0 - 22), x0 + 9 * len(tag) + 8, y0], fill=c + (230,))
        d.text((x0 + 4, max(0, y0 - 20)), tag, fill=(255, 255, 255))
    if im.width > 1400:
        im = im.resize((1400, int(im.height * 1400 / im.width)))
    return im


def strip_images(result: dict) -> dict:
    out = dict(result)
    out["pages"] = [{**p, "images": [{k: v for k, v in im.items() if k != "image_base64"} for im in p["images"]]}
                    for p in result["pages"]]
    return out


# ------------------------------------------------------------------ UI
def ui_parse(file_path):
    empty = ("", "", "{}", [], None, None)
    if not file_path:
        yield ("Upload a PDF or image first.",) + empty
        return
    path = Path(file_path)
    logs, pages, gallery = [], [], []

    def log(m):
        logs.append(f"[{time.strftime('%H:%M:%S')}] {m}")

    log(f"Received {path.name}")
    yield ("\n".join(logs),) + empty
    t0 = time.perf_counter()
    with LOCK:
        try:
            for p, d in docparse.iter_pages(path):
                pages.append(p)
                if d.get("error"):
                    log(f"Page {p['index'] + 1}: ERROR {d['error']}")
                else:
                    gallery.append((overlay(d["image"], d["ocr_page"], d["cells"]), f"Page {p['index'] + 1}"))
                    pg = d["ocr_page"]
                    conf = np.mean([ln.conf for ln in pg.lines]) if pg.lines else 0
                    log(f"Page {p['index'] + 1}: {d['ms']:.0f} ms · {len(pg.lines)} lines · conf {conf:.2f} · "
                        f"text-layer verified {100 * d['verified']:.0f}% · tables {d['tables_rebuilt']} ({d['table_source']})")
                res = docparse.document(pages, path)
                md = docparse.to_markdown(res)
                yield ("\n".join(logs), md, md, json.dumps(strip_images(res), ensure_ascii=False, indent=1),
                       gallery, None, None)
        except engine.InputError as e:
            log(f"Cannot read file: {e}")
            yield ("\n".join(logs),) + empty
            return
    res = docparse.document(pages, path)
    md = docparse.to_markdown(res)
    js = json.dumps(strip_images(res), ensure_ascii=False, indent=1)
    out = Path(tempfile.mkdtemp(prefix="ocr_"))
    (out / f"{path.stem}.md").write_text(md, encoding="utf-8")
    (out / f"{path.stem}.json").write_text(js, encoding="utf-8")
    dt = time.perf_counter() - t0
    log(f"Done — {len(pages)} page(s) in {dt:.2f}s ({dt / max(1, len(pages)):.2f} s/page)")
    yield ("\n".join(logs), md, md, js, gallery, str(out / f"{path.stem}.md"), str(out / f"{path.stem}.json"))


with gr.Blocks(title="OCR") as demo:
    gr.Markdown(
        "# OCR\n"
        "One fused pipeline: OCR · PDF text-layer verification · cell-aware tables · layout cleanup. "
        "Scripts kept as written (no translation). API: `POST /api/parse` · [docs](/docs)  \n"
        "<small>Evaluation build — weights under modified OpenRAIL-M (commercial use needs a Datalab licence). "
        "Known weak spots: ₹ sign, debit/credit columns, Gujarati inside mixed-script lines.</small>")
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            f = gr.File(label="PDF or image", file_types=sorted(EXTS), type="filepath")
            btn = gr.Button("Parse", variant="primary", size="lg")
            logbox = gr.Textbox(label="Progress", lines=10, interactive=False, autoscroll=True)
            md_dl = gr.File(label="Download Markdown")
            js_dl = gr.File(label="Download JSON")
        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.Tab("Rendered"):
                    rendered = gr.Markdown(sanitize_html=False, elem_classes="rendered")
                with gr.Tab("Layout"):
                    gal = gr.Gallery(label="Layout blocks (numbered in reading order)", columns=1,
                                     height="auto", object_fit="contain", format="jpeg")
                with gr.Tab("Markdown"):
                    raw = gr.Code(language="markdown", lines=30)
                with gr.Tab("JSON"):
                    js = gr.Code(language="json", lines=30)
    samples = Path(__file__).resolve().parent / "samples"
    ex = [str(samples / n) for n in ("mixed_bank_scan.pdf", "multipage_hi_gu_scan.pdf",
                                     "twocol_notice_scan.pdf", "pa_statement_hardscan.pdf") if (samples / n).exists()]
    if ex:
        gr.Examples([[e] for e in ex], inputs=[f], label="Sample documents")
    outs = [logbox, rendered, raw, js, gal, md_dl, js_dl]
    btn.click(ui_parse, inputs=[f], outputs=outs)
    f.upload(ui_parse, inputs=[f], outputs=outs)

# ------------------------------------------------------------------ API
api = FastAPI(title="OCR API", version="1.0")


@api.get("/api/health")
def health():
    return {"status": "ok", "model": docparse.MODEL_NAME}


@api.post("/api/parse")
def api_parse(file: UploadFile = File(...)):
    """Upload a PDF/image; Mistral-OCR-style JSON back."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in EXTS:
        raise HTTPException(415, f"Unsupported file type '{suffix}'. Allowed: {sorted(EXTS)}")
    tmp = Path(tempfile.mkdtemp(prefix="ocr_api_"))
    src = tmp / f"upload{suffix}"
    try:
        with src.open("wb") as fh:
            shutil.copyfileobj(file.file, fh)
        with LOCK:
            res = docparse.parse_document(src)
        # Mistral-OCR-style body: pages[index, markdown, images, dimensions] + usage_info
        return JSONResponse({
            "pages": res["pages"],
            "document_annotation": None,
            "usage_info": {"pages_processed": len(res["pages"]), "doc_size_bytes": res["usage_info"]["doc_size_bytes"]},
            "extractedText": docparse.to_markdown(res),
            "pageCount": len(res["pages"]),
        })
    except engine.InputError as e:
        raise HTTPException(422, f"Cannot read file: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


app = gr.mount_gradio_app(api, demo.queue(), path="/", css=CSS, theme=gr.themes.Soft(primary_hue="emerald"),
                          allowed_paths=[str(Path(__file__).resolve().parent / "samples")])

if __name__ == "__main__":
    print("[i] loading the OCR engine models...", file=sys.stderr)
    docparse.converter()
    print("[i] ready -> http://127.0.0.1:7862  (API: POST /api/parse, docs: /docs)", file=sys.stderr)
    uvicorn.run(app, host="127.0.0.1", port=7862, log_level="warning")
