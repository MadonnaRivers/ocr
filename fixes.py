"""Post-fixes on top of the OCR engine: rupee sign repair and merging of split tables."""
import re

_PFX = r"[$€¥£*रSsFfTtZz]"
_NUM = r"(?:\d{1,3}(?:,\d{2,3})+|\d{4,})(?:\.\d{1,2})?"
_AMT = re.compile(r"^\s*(?P<p>" + _PFX + r"|8(?=\s))?\s*(?P<n>" + _NUM + r")\s*$")
_LOOSE = re.compile(r"(?<![\w,.])[$€¥£*र]\s?(?=\d{1,3}(?:,\d{2,3})+)")
_SEP = re.compile(r"^[\s|:\-]+$")


def fix_rupee(md: str) -> str:
    """the OCR engine almost never reads the rupee sign: it comes out as $, EUR, *, a Devanagari
    letter, a stray digit, or nothing. In a table column where amounts carry such a wrong
    prefix (or a real rupee sign) on at least 40% of the amounts, every amount in that
    column is written with the rupee sign. In running text a wrong currency symbol before
    an Indian-grouped number becomes the rupee sign. Columns with 'Rs.' are left alone."""
    lines, out, i = md.split("\n"), [], 0
    while i < len(lines):
        if not lines[i].lstrip().startswith("|"):
            out.append(_LOOSE.sub("₹ ", lines[i]))
            i += 1
            continue
        j = i
        while j < len(lines) and lines[j].lstrip().startswith("|"):
            j += 1
        block = lines[i:j]
        rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in block]
        data = [k for k, l in enumerate(block) if not _SEP.match(l)][1:]  # skip header row
        for c in range(max(len(r) for r in rows)):
            cells = [(k, rows[k][c]) for k in data if c < len(rows[k])]
            if any(re.search(r"\bRs\b|रु|روپ", t) for _, t in cells):
                continue
            hits = [(k, _AMT.match(t)) for k, t in cells]
            hits = [(k, m) for k, m in hits if m]
            if len(hits) < 2:
                continue
            marked = sum(1 for k, m in hits if m.group("p"))
            real = sum(1 for k, _ in hits if rows[k][c].lstrip().startswith("₹"))
            if marked + real >= 0.4 * len(hits):
                for k, m in hits:
                    rows[k][c] = "₹ " + m.group("n")
        for k, l in enumerate(block):
            out.append(l if _SEP.match(l) else "| " + " | ".join(rows[k]) + " |")
        i = j
    return "\n".join(out)


def merge_tables(page):
    """Consecutive Table blocks sitting side by side (same rows) are one table that the OCR engine
    split in two: merge them into one block."""
    out = []
    for b in page.blocks:
        p = out[-1] if out else None
        if p is not None and p["label"] == "Table" and b["label"] == "Table":
            ov = min(p["bbox"][3], b["bbox"][3]) - max(p["bbox"][1], b["bbox"][1])
            if ov > 0.5 * min(p["bbox"][3] - p["bbox"][1], b["bbox"][3] - b["bbox"][1]):
                p["bbox"] = [min(p["bbox"][0], b["bbox"][0]), min(p["bbox"][1], b["bbox"][1]),
                             max(p["bbox"][2], b["bbox"][2]), max(p["bbox"][3], b["bbox"][3])]
                p["text"] = (p["text"] or "") + "\n" + (b["text"] or "")
                p["_n"] = p.get("_n", 1) + b.get("_n", 1)
                continue
        out.append(b)
    page.blocks = out


_ISO = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
_ARABIC = re.compile(r"[؀-ۿ]")


def fix_rtl_dates(md: str) -> str:
    """In right-to-left lines the OCR engine reverses the digit groups of a date: 15-11-2026 comes
    out as 2026-11-15. Put them back as dd-mm-yyyy in lines that contain Arabic-script text
    (or in table rows where a neighbouring cell does)."""
    out = []
    for line in md.split("\n"):
        if _ARABIC.search(line):
            line = _ISO.sub(lambda m: f"{m.group(3)}-{m.group(2)}-{m.group(1)}", line)
        out.append(line)
    return "\n".join(out)
