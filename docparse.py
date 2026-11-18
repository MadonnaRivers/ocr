"""One-call document parser: OCR + verified text layer + cell-aware tables,
with output shaped like the Mistral OCR API response.

    from docparse import parse_document
    result = parse_document("file.pdf")      # dict, Mistral-OCR-style
    for page, debug in iter_pages("file.pdf"): ...   # streaming, page by page

CLI:
    .venv\\Scripts\\python.exe docparse.py file.pdf [-o out_dir]

Per page:
  1. OCR reads every line (always OCR, so broken Indic text layers never leak in).
  2. Born-digital PDF: each OCR line is compared with the PDF's own words in the same
     box. When they agree (fuzzy ratio >= 80, spaces ignored) the exact PDF text is
     used; when they don't (garbled Indic font encodings), the OCR text stays.
  3. Tables: cells come from the PDF's drawn rectangles, or from ruling lines found in
     the image (scans/photos). Cell text comes from the verified PDF words or the OCR
     lines. Rows, columns and spans come from the cell grid.
  4. Markdown is written block by block in the engine's reading order: headings, paragraphs,
     lists, pipe tables, and pictures as ![img-k.jpeg](img-k.jpeg) with the crop in
     "images" (base64), the same layout the Mistral OCR API returns.

Licence: check the model weights' licence before commercial use.
until a commercial licence is settled.
"""
from __future__ import annotations

import base64
import html
import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from rapidfuzz import fuzz

import indra_ocr as engine
from indra_ocr.layout_post import _ITEM, join_lines
from fixes import fix_rtl_dates, fix_rupee, merge_tables
from layout_fix import (cell_text, cells_from_image, cells_from_pdf_page, grid_from_lines, group_tables,
                        is_form, table_structures)

THREADS = 4
MODEL_NAME = "ocr+layout"
VERIFY_RATIO = 80
_conv = None


def converter():
    global _conv
    if _conv is None:
        _conv = engine.load(threads=THREADS, mode="ocr", furniture=True)
    return _conv


# ------------------------------------------------------------------ inputs
def _page_images(path: Path, conv):
    if engine.pipeline.is_pdf(path):
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(str(path))
        try:
            for i in range(len(doc)):
                pg = doc[i]
                yield pg.render(scale=conv.page_scale(pg)).to_pil().convert("RGB")
        finally:
            doc.close()
    else:
        for gray in engine.pipeline.read_images(path):
            yield Image.fromarray(gray).convert("RGB")


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s).lower()


def _pdf_words(fp, scale):
    out = []
    for x0, y0, x1, y1, w, bno, lno, _ in fp.get_text("words"):
        out.append((x0 * scale, y0 * scale, x1 * scale, y1 * scale, w, bno, lno))
    return out


def _verify_lines(lines, words):
    """Swap each OCR line's text for the PDF's words in the same box when they agree.
    Returns (texts, verified_char_fraction)."""
    texts, ok_chars, all_chars = [], 0, 0
    for ln in lines:
        x0, y0, x1, y1 = ln.box
        pad_y = 0.25 * (y1 - y0)
        inside = [w for w in words
                  if x0 - 2 <= (w[0] + w[2]) / 2 <= x1 + 2 and y0 - pad_y <= (w[1] + w[3]) / 2 <= y1 + pad_y]
        n = len(_norm(ln.text))
        all_chars += n
        if inside:
            inside.sort(key=lambda w: (w[5], w[6], w[0]))
            pdf_text = " ".join(w[4] for w in inside)
            if fuzz.ratio(_norm(ln.text), _norm(pdf_text)) >= VERIFY_RATIO:
                texts.append(pdf_text)
                ok_chars += n
                continue
        texts.append(ln.text)
    return texts, (ok_chars / all_chars if all_chars else 0.0)


# ------------------------------------------------------------------ markdown pieces
def _esc(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ").strip()


def _pipe(rows: list[list[str]]) -> str:
    rows = [r for r in rows if any(x.strip() for x in r)]
    if not rows:
        return ""
    n = max(len(r) for r in rows)
    rows = [r + [""] * (n - len(r)) for r in rows]
    keep = [j for j in range(n) if any(r[j].strip() for r in rows)]
    rows = [[r[j] for j in keep] for r in rows]
    out = ["| " + " | ".join(_esc(x) for x in rows[0]) + " |",
           "|" + "|".join(["---"] * len(keep)) + "|"]
    out += ["| " + " | ".join(_esc(x) for x in r) + " |" for r in rows[1:]]
    return "\n".join(out)


def table_to_md(table) -> str:
    if is_form(table):  # form grid: each printed row as it is, label and value cells in order
        rows = {}
        for c in table:
            rows.setdefault(c.r0, []).append(c)
        return _pipe([[cell_text(c) for c in sorted(rows[r], key=lambda c: c.c0)] for r in sorted(rows)])
    nr, nc = max(c.r1 for c in table), max(c.c1 for c in table)
    grid = [[""] * nc for _ in range(nr)]
    for c in table:
        grid[c.r0][c.c0] = cell_text(c)
    return _pipe(grid)


def html_table_to_md(h: str) -> str:
    rows = []
    for tr in re.findall(r"<tr>(.*?)</tr>", h, re.S):
        row = []
        for attrs, td in re.findall(r"<t[dh]([^>]*)>(.*?)</t[dh]>", tr, re.S):
            row.append(html.unescape(re.sub(r"<[^>]+>", " ", td)).strip())
            m = re.search(r'colspan="(\d+)"', attrs)
            row += [""] * (int(m.group(1)) - 1 if m else 0)
        rows.append(row)
    return _pipe(rows)


def _crop_b64(img: Image.Image, box) -> str:
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    crop = img.crop((max(0, x0), max(0, y0), min(img.width, x1), min(img.height, y1)))
    buf = io.BytesIO()
    crop.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ------------------------------------------------------------------ one page
def _block_lines(page, texts):
    """Map the engine's blocks to their line texts (blocks are runs of the reading order)."""
    out, k = [], 0
    for b in page.blocks:
        if b["label"] == "Picture":
            out.append([])
            continue
        n = len(b["text"].split("\n"))
        out.append(texts[k:k + n])
        k += n
    if k != len(texts):  # unexpected: fall back to the OCR block text
        return [b["text"].split("\n") if b["text"] else [] for b in page.blocks]
    return out


def _clean_blocks(page, img):
    """Layout fix: drop Picture boxes that are tiny specks, duplicates of a bigger picture,
    or sit mostly on top of text/table blocks. Picture blocks own no text lines, so
    removing them cannot shift the line-to-block mapping."""
    merge_tables(page)
    area = img.width * img.height

    def a(b):
        x0, y0, x1, y1 = b["bbox"]
        return max(0, x1 - x0) * max(0, y1 - y0)

    def inter(b, c):
        x0, y0 = max(b["bbox"][0], c["bbox"][0]), max(b["bbox"][1], c["bbox"][1])
        x1, y1 = min(b["bbox"][2], c["bbox"][2]), min(b["bbox"][3], c["bbox"][3])
        return max(0, x1 - x0) * max(0, y1 - y0)

    keep = []
    for n, b in enumerate(page.blocks):
        if b["label"] == "Picture":
            ab = a(b) or 1
            if ab < 0.004 * area:
                continue
            if any(c is not b and c["label"] != "Picture" and inter(b, c) / ab > 0.6 for c in page.blocks):
                continue
            if any(m != n and c["label"] == "Picture" and inter(b, c) / ab > 0.8 and (a(c) > ab or (a(c) == ab and m < n))
                   for m, c in enumerate(page.blocks)):
                continue
        keep.append(b)
    page.blocks = keep


_BLOCK_TYPE = {"Title": "title", "Section-Header": "title", "Text": "text", "List-Group": "list",
               "Table": "table", "Form": "table", "Picture": "image", "Page-Header": "header",
               "Page-Footer": "footer", "Caption": "text", "Footnote": "text", "Table-Of-Contents": "text"}


def _parse_page(path: Path, i: int, page, img: Image.Image, fitz_doc):
    _clean_blocks(page, img)
    debug = {"cells": [], "table_source": "none", "verified": 0.0}
    texts = [ln.text for ln in page.lines]
    words, scale = None, img.width / (fitz_doc[i].rect.width if fitz_doc else img.width)
    if fitz_doc is not None and not page.rotation and not fitz_doc[i].rotation:
        fp = fitz_doc[i]
        pw = _pdf_words(fp, scale)
        if pw:
            texts, frac = _verify_lines(page.lines, pw)
            debug["verified"] = frac
            if frac >= 0.5:
                words = pw
    # scanned page: a ruled grid under a block the OCR engine called Text (e.g. an Urdu table) is a table
    if words is None:
        try:
            for g in group_tables(cells_from_image(np.asarray(img.convert("L"))), gap=2.5 * scale):
                gx0, gy0, gx1, gy1 = (min(c.x0 for c in g), min(c.y0 for c in g),
                                      max(c.x1 for c in g), max(c.y1 for c in g))
                for b in page.blocks:
                    bx0, by0, bx1, by1 = b["bbox"]
                    ov = max(0, min(bx1, gx1) - max(bx0, gx0)) * max(0, min(by1, gy1) - max(by0, gy0))
                    if (b["label"] in ("Text", "List-Group") and len(b["text"].split("\n")) >= 4
                            and ov > 0.6 * (bx1 - bx0) * (by1 - by0)):
                        b["label"] = "Table"
        except Exception:
            pass
    # tables
    has_table = any(b["label"] == "Table" for b in page.blocks)
    structs, cells = [], []
    ocr_lines = [{"box": ln.box, "text": t} for ln, t in zip(page.lines, texts)]
    spans, k0 = [], 0  # (first line, line count) of every block, in reading order
    for b in page.blocks:
        n0 = 0 if b["label"] == "Picture" else len(b["text"].split("\n"))
        spans.append((k0, n0))
        k0 += n0
    if k0 != len(ocr_lines):
        spans = None
    if has_table:
        cells, src = [], "image-rules"
        if fitz_doc is not None and words:
            cells, src = cells_from_pdf_page(fitz_doc[i], scale), "pdf-vectors"
        if len(cells) < 2:
            gray = np.asarray(img.convert("L"))
            cells = [c for b in page.blocks if b["label"] == "Table" for c in cells_from_image(gray, b["bbox"])]
            src = "image-rules"
        ocr_lines = [{"box": ln.box, "text": t} for ln, t in zip(page.lines, texts)]
        structs = table_structures(page.blocks, ocr_lines, cells, scale,
                                   scanned=(src == "image-rules"), words=words if src == "pdf-vectors" else None)
        debug["cells"], debug["table_source"] = cells, src
    # markdown, block by block
    parts, images, t_k, h_k, n_rebuilt = [], [], 0, 0, 0
    html_tables = re.findall(r"<table>.*?</table>", page.markdown or "", re.S)
    out_blocks, prev = [], None

    def _flush():
        if prev is not None and len(parts) > prev[1]:
            content = "\n\n".join(parts[prev[1]:])
            if content.strip():
                x0, y0, x1, y1 = prev[0]["bbox"]
                e = {"top_left_x": int(x0), "top_left_y": int(y0), "bottom_right_x": int(x1), "bottom_right_y": int(y1),
                     "content": content, "confidence_scores": None,
                     "type": _BLOCK_TYPE.get(prev[0]["label"], "text")}
                if e["type"] == "table":
                    e["table_id"] = None
                out_blocks.append(e)

    for b_idx, (b, btexts) in enumerate(zip(page.blocks, _block_lines(page, texts))):
        _flush()
        prev = (b, len(parts))
        lab = b["label"]
        if lab == "Picture":
            k = len(images)
            x0, y0, x1, y1 = b["bbox"]
            images.append({"id": f"img-{k}.jpeg", "top_left_x": int(x0), "top_left_y": int(y0),
                           "bottom_right_x": int(x1), "bottom_right_y": int(y1),
                           "image_base64": _crop_b64(img, b["bbox"])})
            parts.append(f"![img-{k}.jpeg](img-{k}.jpeg)")
            continue
        if lab == "Table":
            tables = structs[t_k] if t_k < len(structs) else None
            n_html = b.get("_n", 1)
            html_t = html_tables[h_k] if (n_html == 1 and h_k < len(html_tables)) else None
            t_k += 1
            h_k += n_html
            want = len(_norm("".join(btexts)))

            def covered(md_):
                return want == 0 or len(_norm(re.sub(r"[|\-]", "", md_))) >= 0.9 * want

            md_t, tail = None, []
            cand1 = None
            if tables:  # 1. cell-aware rebuild from the drawn cells
                cand1 = "\n\n".join(t for t in (table_to_md(t) for t in tables) if t)
                ext = [(min(c.y0 for c in t), max(c.y1 for c in t)) for t in tables]
                side_by_side = any(min(a[1], z[1]) - max(a[0], z[0]) > 0.5 * min(a[1] - a[0], z[1] - z[0])
                                   for k, a in enumerate(ext) for z in ext[k + 1:])
                whole = all(sum((c.r1 - c.r0) * (c.c1 - c.c0) for c in t)
                            >= 0.9 * max(c.r1 for c in t) * max(c.c1 for c in t) for t in tables)
                # half-detected grids (missing cells) and tables split into side-by-side pieces
                # are not trusted: they go to the line-grid below
                if covered(cand1) and whole and not side_by_side:
                    md_t = cand1
                    n_rebuilt += 1
            if md_t is None and spans is not None and cells:  # 2. columns from (partial) cells, rows from OCR lines
                x0, y0, x1, y1 = b["bbox"]
                bc = [c for c in cells if x0 - 30 <= (c.x0 + c.x1) / 2 <= x1 + 30
                      and y0 - 40 <= (c.y0 + c.y1) / 2 <= y1 + 40]
                s0, n0 = spans[b_idx]
                blines = [ocr_lines[k] for k in range(s0, s0 + n0)]
                while len(blines) > 2:  # a sentence under the table that runs its full width is not a row
                    ln = blines[-1]
                    cyl = (ln["box"][1] + ln["box"][3]) / 2
                    hl = ln["box"][3] - ln["box"][1]
                    others = blines[:-1]
                    xs0, xs1 = min(q["box"][0] for q in others), max(q["box"][2] for q in others)
                    alone = not any(abs((q["box"][1] + q["box"][3]) / 2 - cyl) < 0.6 * hl for q in others)
                    if alone and (ln["box"][2] - ln["box"][0]) >= 0.5 * (xs1 - xs0) and len(ln["text"]) > 25:
                        tail.insert(0, ln["text"])
                        blines = others
                    else:
                        break
                rows = grid_from_lines(blines, bc, (y0, y1), extra_edges=(x0, x1)) if bc and blines else None
                if rows:
                    cand = _pipe(rows)
                    if cand and covered(cand + "".join(tail)):
                        md_t = cand
                        debug["table_grid"] = debug.get("table_grid", 0) + 1
                        n_rebuilt += 1
            if md_t is None and cand1 and covered(cand1):  # 3. the half-detected grid still beats the engine's table
                md_t, tail = cand1, []
                n_rebuilt += 1
            if md_t is None and html_t:  # 4. the engine's own table
                md_t = html_table_to_md(html_t) or None
                debug["table_fallback"] = debug.get("table_fallback", 0) + 1
            if md_t is None:  # 5. one cell per text line
                md_t = _pipe([[t] for t in btexts]) if len(btexts) > 1 else join_lines(btexts)
                tail = []
            parts.append(fix_rtl_dates(fix_rupee(md_t)))
            if tail:
                parts.append("\n".join(tail))
            continue
        body = join_lines(btexts)
        if not body.strip():
            continue
        body = fix_rtl_dates(fix_rupee(body))
        if lab == "Title":
            parts.append("# " + body)
        elif lab == "Section-Header":
            parts.append("## " + body)
        elif lab == "List-Group":
            items: list[list[str]] = []
            for t in btexts:
                if not items or _ITEM.match(t):
                    items.append([t])
                else:
                    items[-1].append(t)
            parts.append("\n".join(join_lines(it) for it in items))
        else:
            parts.append(body)
    _flush()
    debug["tables_rebuilt"] = n_rebuilt
    md = "\n\n".join(p for p in parts if p.strip())
    return {
        "index": i,
        "markdown": md,
        "images": images,
        "tables": [],
        "hyperlinks": [],
        "header": None,
        "footer": None,
        "dimensions": {"dpi": round(72 * scale) if fitz_doc else 0, "height": img.height, "width": img.width},
        "confidence_scores": None,
        "blocks": out_blocks,
    }, debug


# ------------------------------------------------------------------ public API
def iter_pages(path):
    """Yield (mistral_style_page, debug) per page as soon as it is parsed.
    debug: {"ocr_page", "image", "cells", "table_source", "verified", "tables_rebuilt", "ms"}"""
    path = Path(path)
    conv = converter()
    fitz_doc = None
    if engine.pipeline.is_pdf(path):
        import fitz

        fitz_doc = fitz.open(str(path))
    imgs = _page_images(path, conv)
    try:
        for i, page in conv.convert(path):
            img = next(imgs, None)
            t0 = time.perf_counter()
            if page.source == "error" or img is None:
                yield ({"index": i, "markdown": "", "images": [], "dimensions": None},
                       {"ocr_page": page, "image": img, "cells": [], "error": page.note})
                continue
            if page.rotation:
                img = img.rotate(page.rotation, expand=True)
            out, dbg = _parse_page(path, i, page, img, fitz_doc)
            dbg.update(ocr_page=page, image=img,
                       ms=page.timings.get("total_ms", 0) + 1000 * (time.perf_counter() - t0))
            yield out, dbg
    finally:
        if fitz_doc is not None:
            fitz_doc.close()


def document(pages: list, path: Path) -> dict:
    return {"pages": pages, "model": MODEL_NAME, "document_annotation": None,
            "usage_info": {"pages_processed": len(pages), "doc_size_bytes": Path(path).stat().st_size}}


def parse_document(path) -> dict:
    return document([p for p, _ in iter_pages(path)], Path(path))


def to_markdown(result: dict) -> str:
    return "\n\n".join(p["markdown"] for p in result["pages"])


if __name__ == "__main__":
    import argparse

    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("-o", "--out", default=".")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in a.files:
        t0 = time.perf_counter()
        pages = []
        for p, d in iter_pages(f):
            pages.append(p)
            print(f"{Path(f).name} p{p['index'] + 1}: {d.get('ms', 0):.0f} ms · text-layer verified "
                  f"{100 * d.get('verified', 0):.0f}% · tables {d.get('tables_rebuilt', 0)} "
                  f"({d.get('table_source', '-')}, {len(d.get('cells', []))} cells)")
        res = document(pages, Path(f))
        stem = Path(f).stem
        (out / f"{stem}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        (out / f"{stem}.md").write_text(to_markdown(res), encoding="utf-8")
        print(f"  {len(pages)} page(s) in {time.perf_counter() - t0:.2f}s -> {out / stem}.md/.json")
