"""Generate ground-truthed test documents for the OCR evaluation.

For each document we write:
  samples/<id>_digital.pdf   - vector PDF with a real text layer (HarfBuzz-shaped via MuPDF HTML)
  samples/<id>_scan.pdf      - image-only PDF that simulates a photocopy/scan
                               (200 dpi, grayscale, skew, noise, blur, JPEG)
  samples/<id>_scan.png      - the same scan as an image
  samples/<id>.gt.txt        - ground-truth text (reading order, one line per line)
  samples/<id>.gt_tables.json- ground-truth table cells

Fonts: Nirmala UI from Windows (covers Latin, Devanagari, Gujarati, Gurmukhi),
extracted from Nirmala.ttc into samples/_fonts/.
Usage:  python -I scripts/make_samples.py <ocr_dir>
"""
import io
import json
import random
import sys
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from fontTools.ttLib import TTCollection
from PIL import Image, ImageFilter

ROOT = Path(sys.argv[1]).resolve()
OUT = ROOT / "samples"
FONTDIR = OUT / "_fonts"
OUT.mkdir(exist_ok=True)
FONTDIR.mkdir(exist_ok=True)

coll = TTCollection(r"C:\Windows\Fonts\Nirmala.ttc")
coll.fonts[0].save(str(FONTDIR / "nirmala.ttf"))
coll.fonts[1].save(str(FONTDIR / "nirmalab.ttf"))

CSS = """
@font-face { font-family: nir; src: url(nirmala.ttf); }
@font-face { font-family: nir; font-weight: bold; src: url(nirmalab.ttf); }
body { font-family: nir; font-size: 11pt; line-height: 1.35; }
h1 { font-size: 16pt; text-align: center; margin: 0 0 6pt 0; }
h2 { font-size: 12.5pt; margin: 10pt 0 4pt 0; }
p  { margin: 0 0 5pt 0; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 8pt 0; }
td, th { border: 1px solid black; padding: 3pt 5pt; font-size: 10.5pt; }
th { font-weight: bold; background-color: #e6e6e6; }
"""

# ---------------------------------------------------------------- documents
# Each doc: list of blocks. ("h1"|"h2"|"p", text) or ("table", [header, rows...])
DOCS = {
    "en_sanction": {
        "lang": "English", "scripts": ["Latin"],
        "blocks": [
            ("h1", "LOAN SANCTION LETTER"),
            ("p", "Ref. No.: EHF/HL/2026/04518    Date: 12-11-2026"),
            ("p", "To: Mr. Rajesh Kumar Sharma, Flat 302, Shanti Apartments, Sector 21, Navi Mumbai 400706"),
            ("h2", "1. Loan Details"),
            ("p", "We are pleased to inform you that your application for a Home Loan has been sanctioned on the following terms and conditions, subject to execution of the loan agreement and creation of security."),
            ("table", [["Particulars", "Details"],
                       ["Loan Amount", "Rs. 25,00,000"],
                       ["Rate of Interest", "8.75% p.a. (floating)"],
                       ["Tenure", "240 months"],
                       ["EMI", "Rs. 22,093"],
                       ["Processing Fee", "Rs. 5,900 (incl. GST)"]]),
            ("h2", "2. Security"),
            ("p", "Equitable mortgage of the residential property situated at Survey No. 112/3, Village Ghansoli, Taluka Thane, District Thane, Maharashtra."),
            ("h2", "3. Disbursement Schedule"),
            ("table", [["Stage", "Percentage", "Amount (Rs.)"],
                       ["Agreement registration", "20%", "5,00,000"],
                       ["Plinth completion", "30%", "7,50,000"],
                       ["Slab completion", "30%", "7,50,000"],
                       ["Possession", "20%", "5,00,000"]]),
            ("p", "This sanction is valid for 90 days from the date of issue. For Easy Home Finance Ltd., Authorised Signatory."),
        ],
    },
    "hi_sanction": {
        "lang": "Hindi", "scripts": ["Devanagari"],
        "blocks": [
            ("h1", "ऋण स्वीकृति पत्र"),
            ("p", "संदर्भ संख्या: EHF/HL/2026/04519    दिनांक: 12-11-2026"),
            ("p", "प्रति: श्री सुरेश प्रसाद वर्मा, मकान संख्या 45, गांधी नगर, जयपुर, राजस्थान 302015"),
            ("h2", "1. ऋण का विवरण"),
            ("p", "हमें आपको सूचित करते हुए प्रसन्नता हो रही है कि आपका गृह ऋण आवेदन निम्नलिखित नियमों और शर्तों पर स्वीकृत किया गया है।"),
            ("table", [["विवरण", "राशि / शर्त"],
                       ["ऋण राशि", "₹ 18,50,000"],
                       ["ब्याज दर", "9.10% प्रति वर्ष"],
                       ["अवधि", "180 महीने"],
                       ["मासिक किस्त (EMI)", "₹ 18,874"],
                       ["प्रसंस्करण शुल्क", "₹ 4,720"]]),
            ("h2", "2. सुरक्षा"),
            ("p", "उधारकर्ता की आवासीय संपत्ति का बंधक, जो खसरा संख्या 218, ग्राम सांगानेर, तहसील जयपुर में स्थित है।"),
            ("p", "यह स्वीकृति जारी होने की तिथि से 90 दिनों तक मान्य है। कृपया सभी मूल दस्तावेज़ शाखा कार्यालय में जमा करें।"),
        ],
    },
    "mr_712": {
        "lang": "Marathi", "scripts": ["Devanagari"],
        "blocks": [
            ("h1", "गाव नमुना सात बारा उतारा"),
            ("p", "गाव: घणसोली    तालुका: ठाणे    जिल्हा: ठाणे"),
            ("h2", "भूधारणा पद्धती व क्षेत्र"),
            ("table", [["सर्वे क्रमांक", "उपविभाग", "क्षेत्र (हे. आर.)", "आकारणी (रु.)"],
                       ["112", "3", "0.45.20", "12.50"],
                       ["113", "1अ", "0.30.00", "8.25"],
                       ["115", "2ब", "1.02.75", "27.00"]]),
            ("h2", "भोगवटदाराचे नाव"),
            ("p", "श्री. विजय गणपत पाटील (हिस्सा 1/2) आणि श्रीमती सुनिता विजय पाटील (हिस्सा 1/2)."),
            ("p", "इतर हक्क: बँक ऑफ महाराष्ट्र, शाखा वाशी यांचा बोजा रुपये 15,00,000 चा फेरफार क्रमांक 4521 नुसार नोंदवला आहे."),
            ("p", "हा उतारा माहितीसाठी असून कोणत्याही कायदेशीर कामासाठी तलाठी यांची स्वाक्षरी आवश्यक आहे."),
        ],
    },
    "gu_kyc": {
        "lang": "Gujarati", "scripts": ["Gujarati"],
        "blocks": [
            ("h1", "ગ્રાહક ઓળખ (KYC) ફોર્મ"),
            ("p", "અરજી નંબર: EHF/KYC/2026/8812    તારીખ: 12-11-2026"),
            ("h2", "વ્યક્તિગત વિગતો"),
            ("table", [["વિગત", "માહિતી"],
                       ["અરજદારનું નામ", "હિતેશ રમેશભાઈ પટેલ"],
                       ["જન્મ તારીખ", "15-08-1985"],
                       ["સરનામું", "12, સરદાર પટેલ નગર, અમદાવાદ 380009"],
                       ["મોબાઇલ નંબર", "98250 12345"],
                       ["માસિક આવક", "₹ 85,000"]]),
            ("h2", "ઘોષણા"),
            ("p", "હું ખાતરી આપું છું કે ઉપર આપેલી તમામ માહિતી સાચી અને સંપૂર્ણ છે. ખોટી માહિતી આપવા બદલ મારી અરજી રદ થઈ શકે છે."),
            ("p", "અરજદારની સહી: ____________    સ્થળ: અમદાવાદ"),
        ],
    },
    "pa_statement": {
        "lang": "Punjabi", "scripts": ["Gurmukhi"],
        "blocks": [
            ("h1", "ਆਮਦਨ ਸਰਟੀਫਿਕੇਟ"),
            ("p", "ਸਰਟੀਫਿਕੇਟ ਨੰਬਰ: PB/LDH/2026/3307    ਮਿਤੀ: 12-11-2026"),
            ("p", "ਇਹ ਪ੍ਰਮਾਣਿਤ ਕੀਤਾ ਜਾਂਦਾ ਹੈ ਕਿ ਸ੍ਰੀ ਗੁਰਪ੍ਰੀਤ ਸਿੰਘ ਪੁੱਤਰ ਸ੍ਰੀ ਹਰਜੀਤ ਸਿੰਘ, ਵਾਸੀ ਪਿੰਡ ਸਾਹਨੇਵਾਲ, ਜ਼ਿਲ੍ਹਾ ਲੁਧਿਆਣਾ ਦੀ ਸਾਲਾਨਾ ਆਮਦਨ ਹੇਠ ਲਿਖੇ ਅਨੁਸਾਰ ਹੈ।"),
            ("table", [["ਆਮਦਨ ਦਾ ਸਰੋਤ", "ਸਾਲਾਨਾ ਰਕਮ (₹)"],
                       ["ਖੇਤੀਬਾੜੀ", "3,60,000"],
                       ["ਦੁੱਧ ਡੇਅਰੀ", "1,20,000"],
                       ["ਕਿਰਾਇਆ", "84,000"],
                       ["ਕੁੱਲ", "5,64,000"]]),
            ("p", "ਇਹ ਸਰਟੀਫਿਕੇਟ ਘਰ ਕਰਜ਼ੇ ਦੀ ਅਰਜ਼ੀ ਲਈ ਜਾਰੀ ਕੀਤਾ ਗਿਆ ਹੈ ਅਤੇ ਛੇ ਮਹੀਨਿਆਂ ਲਈ ਵੈਧ ਹੈ।"),
            ("p", "ਤਹਿਸੀਲਦਾਰ, ਲੁਧਿਆਣਾ"),
        ],
    },
    "mixed_bank": {
        "lang": "Mixed (English + Hindi + Gujarati + Punjabi)", "scripts": ["Latin", "Devanagari", "Gujarati", "Gurmukhi"],
        "blocks": [
            ("h1", "ACCOUNT STATEMENT / खाता विवरण"),
            ("p", "Account Holder / खाताधारक: Hitesh R. Patel / હિતેશ ર. પટેલ"),
            ("p", "Account No.: 0145 2210 9987    IFSC: EHFB0000145    Period: 01-10-2026 to 31-10-2026"),
            ("table", [["Date", "Narration / विवरण", "Debit (₹)", "Credit (₹)", "Balance (₹)"],
                       ["01-10-2026", "Opening Balance / प्रारंभिक शेष", "", "", "1,24,560.00"],
                       ["03-10-2026", "NEFT वेतन जमा - ABC Textiles", "", "85,000.00", "2,09,560.00"],
                       ["05-10-2026", "EMI હપ્તો - Home Loan 04518", "22,093.00", "", "1,87,467.00"],
                       ["11-10-2026", "UPI ਕਿਰਾਇਆ ਭੁਗਤਾਨ", "15,000.00", "", "1,72,467.00"],
                       ["18-10-2026", "ATM नकद निकासी", "10,000.00", "", "1,62,467.00"],
                       ["31-10-2026", "Interest / ब्याज जमा", "", "1,215.00", "1,63,682.00"]]),
            ("p", "Closing Balance / अंतिम शेष: ₹ 1,63,682.00"),
            ("p", "यह कंप्यूटर द्वारा तैयार किया गया विवरण है, हस्ताक्षर की आवश्यकता नहीं है। This is a computer generated statement."),
        ],
    },
}


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def to_html(blocks):
    parts = []
    for kind, val in blocks:
        if kind == "table":
            rows = []
            for i, r in enumerate(val):
                tag = "th" if i == 0 else "td"
                rows.append("<tr>" + "".join(f"<{tag}>{esc(c)}</{tag}>" for c in r) + "</tr>")
            parts.append("<table>" + "".join(rows) + "</table>")
        else:
            parts.append(f"<{kind}>{esc(val)}</{kind}>")
    return "<body>" + "".join(parts) + "</body>"


def gt_text(blocks):
    lines = []
    for kind, val in blocks:
        if kind == "table":
            for r in val:
                lines.append(" ".join(c for c in r if c))
        else:
            lines.append(val)
    return "\n".join(lines) + "\n"


def make_digital(html, path):
    story = fitz.Story(html=html, user_css=CSS, archive=fitz.Archive(str(FONTDIR)))
    writer = fitz.DocumentWriter(str(path))
    mediabox = fitz.paper_rect("a4")
    where = mediabox + (50, 50, -50, -50)
    more = 1
    while more:
        dev = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(dev)
        writer.end_page()
    writer.close()


def degrade(img: Image.Image, seed: int) -> Image.Image:
    rnd = random.Random(seed)
    g = img.convert("L")
    g = g.rotate(rnd.uniform(-1.2, 1.2), resample=Image.BICUBIC, expand=False, fillcolor=255)
    a = np.asarray(g).astype(np.float32)
    a = a * 0.92 + 12  # slightly grey paper, lighter ink
    a += np.random.default_rng(seed).normal(0, 9, a.shape)  # sensor noise
    a = np.clip(a, 0, 255).astype(np.uint8)
    g = Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.6))
    buf = io.BytesIO()
    g.save(buf, "JPEG", quality=55)
    return Image.open(io.BytesIO(buf.getvalue())).convert("L")


def make_scan(digital_pdf, png_path, pdf_path, seed):
    src = fitz.open(str(digital_pdf))
    out = fitz.open()
    for pno, page in enumerate(src):
        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        sc = degrade(img, seed + pno)
        if pno == 0:
            sc.save(png_path)
        buf = io.BytesIO()
        sc.save(buf, "JPEG", quality=70)
        np_ = out.new_page(width=page.rect.width, height=page.rect.height)
        np_.insert_image(np_.rect, stream=buf.getvalue())
    out.save(str(pdf_path))


manifest = {}
for i, (doc_id, d) in enumerate(DOCS.items()):
    html = to_html(d["blocks"])
    dig = OUT / f"{doc_id}_digital.pdf"
    make_digital(html, dig)
    make_scan(dig, OUT / f"{doc_id}_scan.png", OUT / f"{doc_id}_scan.pdf", seed=100 + i)
    (OUT / f"{doc_id}.gt.txt").write_text(gt_text(d["blocks"]), encoding="utf-8")
    tables = [val for kind, val in d["blocks"] if kind == "table"]
    (OUT / f"{doc_id}.gt_tables.json").write_text(json.dumps(tables, ensure_ascii=False, indent=1), encoding="utf-8")
    manifest[doc_id] = {"lang": d["lang"], "scripts": d["scripts"], "n_tables": len(tables)}
    print("wrote", doc_id)

# 2-page multi-language scanned PDF (tests multi-page handling + page order)
merged = fitz.open()
for doc_id in ("hi_sanction", "gu_kyc"):
    merged.insert_pdf(fitz.open(str(OUT / f"{doc_id}_scan.pdf")))
merged.save(str(OUT / "multipage_hi_gu_scan.pdf"))
(OUT / "multipage_hi_gu.gt.txt").write_text(
    (OUT / "hi_sanction.gt.txt").read_text(encoding="utf-8") + (OUT / "gu_kyc.gt.txt").read_text(encoding="utf-8"),
    encoding="utf-8")
manifest["multipage_hi_gu"] = {"lang": "Hindi + Gujarati (2 pages)", "scripts": ["Devanagari", "Gujarati", "Latin"], "n_tables": 2}

(OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
print("done")
