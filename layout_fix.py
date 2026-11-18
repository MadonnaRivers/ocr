"""Cell-aware table reconstruction on top of OCR.

the OCR engine reads text well, but it builds tables from text-line geometry. Each wrapped
line inside a cell becomes its own row, and narrow form cells get split
("Vill / ag / e:"). This module finds the cells that are actually drawn on the
page and puts the engine's lines into them:

  1. Cells
     - born-digital PDF: filled or stroked rectangles from the PDF's vector drawings
     - scanned page:     ruling lines found with OpenCV morphology
  2. Group touching cells into tables, then derive a row/column grid from the
     cell edges (rowspan/colspan).
  3. Assign the OCR engine lines to cells. A line that crosses cell borders is split into
     words by x-position. Lines in a cell are joined in reading order; a line that
     runs to the cell's right edge is glued to the next one without a space
     (character-wrapped text such as "Ravikumar" broken as "Ra" / "vikumar").
  4. Tables without horizontal rules between data rows: rows are recovered from
     the vertical position of the lines in the first column (serial numbers).
  5. Each the OCR engine <table> is replaced in the markdown by the reconstructed table(s)
     inside its block. A form-style grid (cells that don't line up into columns)
     is written as a "label: value" list instead.

Pure geometry, no extra model; works with any OCR that gives line boxes.
"""
from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Cell:
    x0: float
    y0: float
    x1: float
    y1: float
    lines: list = field(default_factory=list)  # (y, x0, x1, text, char_w)
    r0: int = 0
    r1: int = 0
    c0: int = 0
    c1: int = 0

    @property
    def area(self):
        return (self.x1 - self.x0) * (self.y1 - self.y0)

    def contains(self, x, y, pad=1.0):
        return self.x0 - pad <= x <= self.x1 + pad and self.y0 - pad <= y <= self.y1 + pad


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u else 0


def _dedupe_and_leafs(rects, min_w, min_h, page_area):
    rects = [r for r in rects if r[2] - r[0] >= min_w and r[3] - r[1] >= min_h
             and (r[2] - r[0]) * (r[3] - r[1]) < 0.5 * page_area]
    rects.sort(key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
    keep = []
    for r in rects:
        if not any(_iou(r, k) > 0.85 for k in keep):
            keep.append(r)

    # Drop containers: a rect that fully holds 2+ smaller rects is a table/section frame.
    def inside(a, b, tol=2.5):
        return a[0] >= b[0] - tol and a[1] >= b[1] - tol and a[2] <= b[2] + tol and a[3] <= b[3] + tol

    leafs = [r for r in keep if sum(1 for o in keep if o is not r and inside(o, r)) < 2]
    return [Cell(*r) for r in leafs]


# ---------------------------------------------------------------- cell sources
def cells_from_pdf_page(fitz_page, scale: float) -> list[Cell]:
    """Rectangles drawn in a born-digital PDF page, in the engine's pixel frame (points * scale)."""
    rects = []
    for d in fitz_page.get_drawings():
        for it in d["items"]:
            if it[0] == "re":
                r = it[1]
                rects.append((r.x0 * scale, r.y0 * scale, r.x1 * scale, r.y1 * scale))
            elif it[0] == "qu":
                r = it[1].rect
                rects.append((r.x0 * scale, r.y0 * scale, r.x1 * scale, r.y1 * scale))
    W, H = fitz_page.rect.width * scale, fitz_page.rect.height * scale
    return _dedupe_and_leafs(rects, 6 * scale, 6 * scale, W * H)


def cells_from_image(gray: np.ndarray, region=None) -> list[Cell]:
    """Cells enclosed by ruling lines on a scanned page (morphological line detection)."""
    H, W = gray.shape[:2]
    ox = oy = 0
    if region is not None:
        x0, y0, x1, y1 = [int(v) for v in region]
        # OCR table boxes hug the text, not the ruling: look well beyond them for the frame.
        pad = int(max(60, 0.25 * (y1 - y0)))
        ox, oy = max(0, x0 - pad), max(0, y0 - pad)
        gray = gray[oy:min(H, y1 + pad), ox:min(W, x1 + pad)]
    h, w = gray.shape[:2]
    bw = cv2.adaptiveThreshold(255 - gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, -8)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 25), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 25)))
    hor = cv2.dilate(cv2.erode(bw, hk), hk)
    ver = cv2.dilate(cv2.erode(bw, vk), vk)
    grid = cv2.dilate(hor | ver, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(255 - grid, connectivity=4)
    rects = []
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1:
            continue  # background touching the crop border
        if area < 0.6 * cw * ch:  # not a box
            continue
        rects.append((x + ox, y + oy, x + cw + ox, y + ch + oy))
    return _dedupe_and_leafs(rects, 15, 12, H * W)


# ---------------------------------------------------------------- grid building
def _cluster(vals, tol):
    vals = sorted(vals)
    groups = [[vals[0]]]
    for v in vals[1:]:
        if v - groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [sum(g) / len(g) for g in groups]


def _nearest(edges, v):
    return int(np.argmin([abs(e - v) for e in edges]))


def group_tables(cells: list[Cell], gap: float) -> list[list[Cell]]:
    """Connected components of cells that touch (within `gap` px)."""
    n = len(cells)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        a = cells[i]
        for j in range(i + 1, n):
            b = cells[j]
            if a.x0 - gap <= b.x1 and b.x0 - gap <= a.x1 and a.y0 - gap <= b.y1 and b.y0 - gap <= a.y1:
                parent[find(i)] = find(j)
    comps = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(cells[i])
    return [c for c in comps.values() if len(c) >= 2]


def build_grid(cells: list[Cell], tol: float):
    xs = _cluster([c.x0 for c in cells] + [c.x1 for c in cells], tol)
    ys = _cluster([c.y0 for c in cells] + [c.y1 for c in cells], tol)
    for c in cells:
        c.c0, c.c1 = _nearest(xs, c.x0), _nearest(xs, c.x1)
        c.r0, c.r1 = _nearest(ys, c.y0), _nearest(ys, c.y1)
        if c.c1 <= c.c0:
            c.c1 = c.c0 + 1
        if c.r1 <= c.r0:
            c.r1 = c.r0 + 1
    return xs, ys


# ---------------------------------------------------------------- line assignment
_RTL = re.compile(r"[֐-ࣿ]")


def _cw(ch: str) -> float:
    """Rough printed width of a character, so a mixed date + Indic line is split where
    the ink really is (digits are narrow, combining marks take no room)."""
    cat = unicodedata.category(ch)
    if cat == "Mn" or ch in "‌‍":
        return 0.0
    if cat == "Mc":
        return 0.6
    if ch.isspace():
        return 0.35
    if ch.isdigit() or ch in ".,:;-/%()":
        return 0.55
    if ch.isascii():
        return 0.6
    return 1.0


def is_rtl(text: str) -> bool:
    return len(_RTL.findall(text)) > 0.4 * max(1, sum(ch.isalpha() for ch in text))


def _split_line(line, cells):
    """Yield (cell, piece) for an OCR line, splitting by words when it crosses cells."""
    x0, y0, x1, y1 = line["box"]
    text = line["text"]
    cy = (y0 + y1) / 2
    hits = [c for c in cells if c.y0 - 2 <= cy <= c.y1 + 2 and c.x0 < x1 and c.x1 > x0]
    if not hits:
        return []
    cw = (x1 - x0) / max(1, len(text))
    if len(hits) == 1:
        return [(hits[0], (cy, x0, x1, text, cw))]
    total = sum(_cw(ch) for ch in text) or 1.0
    unit = (x1 - x0) / total
    rtl = is_rtl(text)
    out = {}
    pos = 0.0
    for word in re.split(r"(\s+)", text):
        if not word:
            continue
        w = sum(_cw(ch) for ch in word)
        a, b = pos, pos + w
        pos = b
        if word.isspace():
            continue
        if rtl:
            wx0, wx1 = x1 - b * unit, x1 - a * unit
        else:
            wx0, wx1 = x0 + a * unit, x0 + b * unit
        mid = (wx0 + wx1) / 2
        c = min(hits, key=lambda h: 0 if h.x0 <= mid <= h.x1 else min(abs(mid - h.x0), abs(mid - h.x1)))
        q = out.setdefault(id(c), [c, wx0, wx1, []])
        q[1], q[2] = min(q[1], wx0), max(q[2], wx1)
        q[3].append(word)
    return [(c, (cy, a, b, " ".join(ws), cw)) for c, a, b, ws in out.values()]


def grid_from_lines(lines, cells, y_range, extra_edges=()):
    """Ruling-free table: columns come from the x edges of whatever cells were detected
    (even a partial detection shows every column), rows from the vertical position of
    the OCR lines. Lines that cross a column edge are split by word position.
    Returns list of rows (each a list of cell strings, in reading order) or None."""
    if len(cells) < 2 or not lines:
        return None
    width = max(c.x1 for c in cells) - min(c.x0 for c in cells)
    xs = _cluster([c.x0 for c in cells] + [c.x1 for c in cells] + list(extra_edges), tol=max(10.0, 0.012 * width))
    cols = [(a, b) for a, b in zip(xs, xs[1:]) if b - a > 0.04 * width]
    if len(cols) < 2:
        return None
    ya, yb = y_range
    colcells = [Cell(a, ya - 5, b, yb + 5) for a, b in cols]
    pieces = []
    for ln in lines:
        for c, pc in _split_line(ln, colcells):
            pieces.append((colcells.index(c), pc))
    if not pieces:
        return None
    hs = [ln["box"][3] - ln["box"][1] for ln in lines]
    tol = 0.6 * float(np.median(hs))
    xmid = np.mean([(pc[1] + pc[2]) / 2 for _, pc in pieces])

    def cluster(slope):
        # a tilted scan puts the right end of a row lower than the left end: undo the tilt
        ys = [pc[0] - slope * ((pc[1] + pc[2]) / 2 - xmid) for _, pc in pieces]
        order_ = sorted(range(len(pieces)), key=lambda i: ys[i])
        rows_, cur_ = [], []
        for i in order_:
            if cur_ and ys[i] - np.mean([ys[j] for j in cur_]) > tol:
                rows_.append(cur_)
                cur_ = []
            cur_.append(i)
        if cur_:
            rows_.append(cur_)
        return rows_

    best = None
    for slope in np.arange(-0.08, 0.0801, 0.005):  # about +-4.5 degrees
        r_ = cluster(slope)
        score = (len(r_), abs(slope))  # fewest rows wins: true rows are the tightest grouping
        if best is None or score < best[0]:
            best = (score, r_)
    rows = best[1]
    rtl = sum(is_rtl(pc[3]) for _, pc in pieces) > 0.5 * len(pieces)
    out = []
    for r in rows:
        row = [""] * len(cols)
        for ci in range(len(cols)):
            ps = sorted((pieces[i][1] for i in r if pieces[i][0] == ci), key=lambda q: q[1], reverse=rtl)
            row[ci] = " ".join(q[3] for q in ps)
        out.append(row[::-1] if rtl else row)
    return out


def _cell_text(c: Cell) -> str:
    """Join a cell's lines. A line that reaches the cell's text edge (within half a
    character) was wrapped mid-word ("Ra" + "vikumar"): glue it to the next line.
    A line that stops short ended at a space: join with a space."""
    if not c.lines:
        return ""
    ls = sorted(c.lines, key=lambda l: (round(l[0] / 6), l[1]))
    right = max(l[2] for l in ls)  # the cell's text edge, not its border

    def fills(l):
        return (right - l[2]) < 1.0 * l[4]

    # Only cells whose text is character-wrapped (most lines run to the edge, as in
    # web pages printed with word-break) get glued; ordinary word-wrapped cells and
    # short header labels ("Reg.Date / Exe.Date") are joined with spaces.
    char_wrapped = len(ls) >= 3 and sum(fills(l) for l in ls[:-1]) >= 0.6 * (len(ls) - 1)
    out = ls[0][3]
    for prev, cur in zip(ls, ls[1:]):
        same_row = abs(prev[0] - cur[0]) < 0.5 * max(8, prev[4] * 1.8)
        glue = (char_wrapped and not same_row and fills(prev)
                and prev[3][-1:].isalnum() and cur[3][:1].isalnum())
        glue = glue or (not same_row and prev[3][-1:] == "-" and cur[3][:1].isdigit())
        out += ("" if glue else " ") + cur[3]
    return re.sub(r"\s+", " ", out).strip()


def _implicit_rows(table: list[Cell], tol: float):
    """Split tall data cells into rows using the line positions in the first column."""
    first_col = [c for c in table if c.c0 == 0]
    extra = []
    for fc in first_col:
        anchors = sorted({round(l[0]) for l in fc.lines})
        if len(anchors) < 2 or fc.r1 - fc.r0 != 1:
            continue
        # Separate rows (serial numbers 1., 2., 3.) are spaced well apart; lines of one
        # wrapped label ("Sl" / "No.", "Date" / ":" / "08-10-") sit at line spacing.
        lh = np.median([abs(l[0] - m[0]) for l, m in zip(sorted(fc.lines), sorted(fc.lines)[1:])] or [0])
        char_h = 2.0 * np.median([l[4] for l in fc.lines])
        if lh < 2.2 * char_h:
            continue
        band = [c for c in table if c.r0 == fc.r0 and c.r1 == fc.r1]
        cuts = [(a + b) / 2 for a, b in zip(anchors, anchors[1:])]
        bounds = [fc.y0] + cuts + [fc.y1]
        for c in band:
            for k in range(len(bounds) - 1):
                sub = Cell(c.x0, bounds[k], c.x1, bounds[k + 1])
                sub.lines = [l for l in c.lines if bounds[k] <= l[0] < bounds[k + 1]]
                sub.c0, sub.c1 = c.c0, c.c1
                sub.r0 = sub.r1 = (fc.r0, k)  # placeholder, renumbered below
                extra.append((c, sub))
    if not extra:
        return table
    replaced = {id(c) for c, _ in extra}
    cells = [c for c in table if id(c) not in replaced] + [s for _, s in extra]
    # renumber rows by y
    ys2 = _cluster([c.y0 for c in cells] + [c.y1 for c in cells], tol)
    for c in cells:
        c.r0, c.r1 = _nearest(ys2, c.y0), _nearest(ys2, c.y1)
        if c.r1 <= c.r0:
            c.r1 = c.r0 + 1
    return cells


# ---------------------------------------------------------------- rendering
def _is_form(table: list[Cell], ncols: int) -> bool:
    """Form-like grid: rows have different column splits (many colspans)."""
    spans = sum(1 for c in table if c.c1 - c.c0 > 1)
    return ncols >= 8 and spans > 0.3 * len(table)


def to_html(table: list[Cell]) -> str:
    nrows = max(c.r1 for c in table)
    ncols = max(c.c1 for c in table)
    if _is_form(table, ncols):
        rows = {}
        for c in table:
            rows.setdefault(c.r0, []).append(c)
        out = []
        for r in sorted(rows):
            parts = []
            for c in sorted(rows[r], key=lambda c: c.c0):
                t = _cell_text(c)
                if t:
                    t = re.sub(r"^([^:\d]{1,40}?)\s*:\s*", r"**\1:** ", t, count=1)
                    parts.append(t)
            if parts:
                out.append("- " + " · ".join(parts))
        return "\n".join(out)
    occupied = {}
    for c in table:
        occupied[(c.r0, c.c0)] = c
    covered = set()
    rows = []
    for r in range(nrows):
        tds = []
        c_ = 0
        while c_ < ncols:
            if (r, c_) in covered:
                c_ += 1
                continue
            cell = occupied.get((r, c_))
            if cell is None:
                tds.append("<td></td>")
                c_ += 1
                continue
            rs, cs = cell.r1 - cell.r0, cell.c1 - cell.c0
            for rr in range(r, r + rs):
                for cc in range(c_, c_ + cs):
                    covered.add((rr, cc))
            attrs = (f' rowspan="{rs}"' if rs > 1 else "") + (f' colspan="{cs}"' if cs > 1 else "")
            tds.append(f"<td{attrs}>{html.escape(_cell_text(cell))}</td>")
            c_ += cs
        if any(t != "<td></td>" for t in tds):
            rows.append("<tr>" + "".join(tds) + "</tr>")
    return "<table>\n" + "\n".join(rows) + "\n</table>"


# ---------------------------------------------------------------- structured entry point
def _assign_words(words, cells):
    """Born-digital page: put the PDF's own words into cells by their centres, one
    piece per PDF text line (block_no, line_no), so no word is ever cut in two."""
    groups = {}
    for x0, y0, x1, y1, text, bno, lno in words:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        hit = [c for c in cells if c.contains(cx, cy, pad=0.5)]
        if not hit:
            continue
        c = min(hit, key=lambda h: h.area)
        g = groups.setdefault((id(c), bno, lno), [c, []])
        g[1].append((x0, y0, x1, y1, text))
    for c, ws in groups.values():
        ws.sort(key=lambda w: w[0])
        text = " ".join(w[4] for w in ws)
        x0, x1 = ws[0][0], max(w[2] for w in ws)
        cy = sum((w[1] + w[3]) / 2 for w in ws) / len(ws)
        c.lines.append((cy, x0, x1, text, (x1 - x0) / max(1, len(text))))


def table_structures(blocks: list, lines: list, cells: list[Cell], scale_px: float,
                     scanned: bool = False, words=None) -> list:
    """For each the OCR engine Table block (in order): the reconstructed tables inside it as
    lists of Cell (with r0/r1/c0/c1 and lines), or None when nothing was found.
    words: [(x0, y0, x1, y1, text, block_no, line_no)] from a verified PDF text
    layer; when given, cell text comes from it instead of the OCR lines."""
    tblocks = [b for b in blocks if b["label"] == "Table"]
    if not cells:
        return [None] * len(tblocks)
    tables = group_tables(cells, gap=2.5 * scale_px)
    tols = []
    for t in tables:
        size = max(max(c.x1 for c in t) - min(c.x0 for c in t), max(c.y1 for c in t) - min(c.y0 for c in t))
        tols.append(max(2.0 * scale_px, (0.03 if scanned else 0.012) * size))
        build_grid(t, tol=tols[-1])
    allc = [c for t in tables for c in t]
    if words:
        _assign_words(words, allc)
    else:
        for ln in lines:
            for c, piece in _split_line(ln, allc):
                c.lines.append(piece)
    tables = [_implicit_rows(t, tol) for t, tol in zip(tables, tols)]
    out = []
    for b in tblocks:
        bx0, by0, bx1, by1 = b["bbox"]
        mine = []
        for t in tables:
            tx0, ty0 = min(c.x0 for c in t), min(c.y0 for c in t)
            tx1, ty1 = max(c.x1 for c in t), max(c.y1 for c in t)
            ix = max(0, min(bx1, tx1) - max(bx0, tx0))
            iy = max(0, min(by1, ty1) - max(by0, ty0))
            if ix * iy > 0.3 * (tx1 - tx0) * (ty1 - ty0) and sum(len(c.lines) for c in t):
                mine.append((ty0, t))
        out.append([t for _, t in sorted(mine, key=lambda x: x[0])] or None)
    return out


def cell_text(c: Cell) -> str:
    return _cell_text(c)


def is_form(table: list[Cell]) -> bool:
    return _is_form(table, max(c.c1 for c in table))


# ---------------------------------------------------------------- page entry point
def rebuild_page(page_md: str, blocks: list, lines: list, cells: list[Cell], scale_px: float,
                 scanned: bool = False) -> tuple[str, int]:
    """Replace the engine's <table> HTML with cell-aware tables. Returns (markdown, n_tables_rebuilt).
    scanned: cells came from a photo/scan, so allow for tilt when lining up edges."""
    if not cells:
        return page_md, 0
    tables = group_tables(cells, gap=2.5 * scale_px)
    if not tables:
        return page_md, 0
    tols = []
    for t in tables:
        # Photographed pages are slanted: allow edges of one grid line to differ more.
        size = max(max(c.x1 for c in t) - min(c.x0 for c in t), max(c.y1 for c in t) - min(c.y0 for c in t))
        tols.append(max(2.0 * scale_px, (0.03 if scanned else 0.012) * size))
        build_grid(t, tol=tols[-1])
    allc = [c for t in tables for c in t]
    for ln in lines:
        for c, piece in _split_line(ln, allc):
            c.lines.append(piece)
    tables = [_implicit_rows(t, tol) for t, tol in zip(tables, tols)]
    tblocks = [b for b in blocks if b["label"] == "Table"]
    rendered = []
    for b in tblocks:
        bx0, by0, bx1, by1 = b["bbox"]
        mine = []
        for t in tables:
            tx0, ty0 = min(c.x0 for c in t), min(c.y0 for c in t)
            tx1, ty1 = max(c.x1 for c in t), max(c.y1 for c in t)
            ix = max(0, min(bx1, tx1) - max(bx0, tx0))
            iy = max(0, min(by1, ty1) - max(by0, ty0))
            if ix * iy > 0.3 * (tx1 - tx0) * (ty1 - ty0) and sum(len(c.lines) for c in t):
                mine.append((ty0, t))
        rendered.append("\n\n".join(to_html(t) for _, t in sorted(mine, key=lambda x: x[0])) if mine else None)
    out, k, n = [], 0, 0
    pos = 0
    for m in re.finditer(r"<table>.*?</table>", page_md, flags=re.S):
        out.append(page_md[pos:m.start()])
        rep = rendered[k] if k < len(rendered) else None
        out.append(rep if rep else m.group(0))
        n += bool(rep)
        k += 1
        pos = m.end()
    out.append(page_md[pos:])
    return "".join(out), n
