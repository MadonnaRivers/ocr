"""Harder samples: heavy scan degradation and a two-column layout (reading order).

usage: python -I scripts/make_hard.py <ocr_dir>
writes samples/<id>_hardscan.pdf (+ .png) for hi_sanction, gu_kyc, mixed_bank, pa_statement,
and samples/twocol_notice_{digital,scan}.pdf with ground truth in reading order.
"""
import io
import json
import random
import sys
from pathlib import Path

import fitz
import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(sys.argv[1]).resolve()
S = ROOT / "samples"
src = (Path(__file__).parent / "make_samples.py").read_text(encoding="utf-8")
ns = {"__name__": "lib"}
exec(compile(src.split("manifest = {}")[0], "make_samples.py", "exec"), ns)


def hard_degrade(img, seed):
    """Old photocopy: 150 dpi, 3 deg skew, uneven lighting, heavy noise, blur, low JPEG."""
    rnd = random.Random(seed)
    g = img.convert("L")
    g = g.resize((int(g.width * 0.75), int(g.height * 0.75)), Image.BILINEAR)  # 200 -> 150 dpi
    g = g.rotate(rnd.choice([-1, 1]) * rnd.uniform(2.5, 3.5), resample=Image.BICUBIC, fillcolor=235)
    a = np.asarray(g).astype(np.float32)
    h, w = a.shape
    shade = np.linspace(0.78, 1.0, w)[None, :] * np.linspace(0.88, 1.0, h)[:, None]  # lighting gradient
    a = a * shade * 0.85 + 20
    a += np.random.default_rng(seed).normal(0, 18, a.shape)
    speck = np.random.default_rng(seed + 1).random(a.shape) < 0.002  # toner specks
    a[speck] = 30
    a = np.clip(a, 0, 255).astype(np.uint8)
    g = Image.fromarray(a).filter(ImageFilter.GaussianBlur(1.0))
    buf = io.BytesIO()
    g.save(buf, "JPEG", quality=35)
    return Image.open(io.BytesIO(buf.getvalue())).convert("L")


for k, doc_id in enumerate(["hi_sanction", "gu_kyc", "mixed_bank", "pa_statement", "mr_712", "en_sanction"]):
    d = fitz.open(str(S / f"{doc_id}_digital.pdf"))
    pix = d[0].get_pixmap(dpi=200)
    img = hard_degrade(Image.frombytes("RGB", (pix.width, pix.height), pix.samples), 500 + k)
    img.save(S / f"{doc_id}_hardscan.png")
    out = fitz.open()
    p = out.new_page(width=d[0].rect.width, height=d[0].rect.height)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=60)
    p.insert_image(p.rect, stream=buf.getvalue())
    out.save(str(S / f"{doc_id}_hardscan.pdf"))
    print("wrote", doc_id, "hardscan")

# ---- two-column notice (newspaper-style property auction notice, EN + HI) ----
left = [
    ("h2", "PUBLIC NOTICE"),
    ("p", "Notice is hereby given that the property described below is mortgaged with Easy Home Finance Ltd. under loan account HL/0045182."),
    ("p", "Any person having any claim on the said property may lodge the same in writing within 15 days of this notice."),
    ("p", "Property: Flat No. 702, Building B, Sai Residency, Kharghar, Navi Mumbai 410210."),
]
right = [
    ("h2", "सार्वजनिक सूचना"),
    ("p", "सर्वसाधारण को सूचित किया जाता है कि नीचे वर्णित संपत्ति ईज़ी होम फाइनेंस लिमिटेड के पास गिरवी है।"),
    ("p", "उक्त संपत्ति पर किसी भी दावे वाला व्यक्ति इस सूचना के 15 दिनों के भीतर लिखित रूप में दावा प्रस्तुत करे।"),
    ("p", "संपत्ति: फ्लैट संख्या 702, इमारत बी, साई रेजीडेंसी, खारघर, नवी मुंबई 410210।"),
]
H = lambda blocks: "".join(f"<{k}>{ns['esc'](v)}</{k}>" for k, v in blocks)  # noqa: E731
html = ("<body><h1>EASY HOME FINANCE LTD.</h1>"
        "<table style='border:none'><tr>"
        f"<td style='border:none;vertical-align:top;width:50%;padding-right:14pt'>{H(left)}</td>"
        f"<td style='border:none;vertical-align:top;width:50%;padding-left:14pt'>{H(right)}</td>"
        "</tr></table><p>Date: 12-11-2026, Place: Navi Mumbai. Authorised Officer.</p></body>")
dig = S / "twocol_notice_digital.pdf"
ns["make_digital"](html, dig)
ns["make_scan"](dig, S / "twocol_notice_scan.png", S / "twocol_notice_scan.pdf", seed=900)
gt = ["EASY HOME FINANCE LTD."] + [v for _, v in left] + [v for _, v in right] + [
    "Date: 12-11-2026, Place: Navi Mumbai. Authorised Officer."]
(S / "twocol_notice.gt.txt").write_text("\n".join(gt) + "\n", encoding="utf-8")
(S / "twocol_notice.gt_tables.json").write_text("[]", encoding="utf-8")
man = json.loads((S / "manifest.json").read_text(encoding="utf-8"))
man["twocol_notice"] = {"lang": "English + Hindi, two columns", "scripts": ["Latin", "Devanagari"], "n_tables": 0}
(S / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote twocol_notice")
