"""Fresh North-Indian test docs (Hindi, Marathi, Gujarati, Punjabi, Urdu): PDF scan + hard scan + ground truth."""
import io, json, random, sys
from pathlib import Path
import fitz, numpy as np
from PIL import Image, ImageFilter
OUT = Path(__file__).parent / "north_docs"; FD = OUT / "_fonts"
CSS = """
@font-face { font-family: nir; src: url(nirmala.ttf); }
@font-face { font-family: nir; font-weight: bold; src: url(nirmalab.ttf); }
@font-face { font-family: ar; src: url(arial.ttf); }
@font-face { font-family: ar; font-weight: bold; src: url(arialbd.ttf); }
body { font-family: nir; font-size: 12pt; line-height: 1.4; }
body.ur { font-family: ar; direction: rtl; text-align: right; }
h1 { font-size: 17pt; text-align: center; margin: 0 0 8pt 0; }
h2 { font-size: 13pt; margin: 10pt 0 4pt 0; }
p { margin: 0 0 6pt 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 8pt 0; }
td, th { border: 1px solid black; padding: 4pt 6pt; font-size: 11.5pt; }
th { font-weight: bold; background-color: #e6e6e6; }
"""
def T(h1, ref, addr, h2, para, tbl, foot):
    return [("h1", h1), ("p", ref), ("p", addr), ("h2", h2), ("p", para), ("table", tbl), ("p", foot)]
DOCS = {
 "hindi": T("ऋण संवितरण सूचना", "संदर्भ संख्या: EHF/HL/2026/07722 दिनांक: 15-11-2026",
   "श्री अमित कुमार वर्मा, मकान नं. 18, गांधी नगर, जयपुर 302015।", "1. ऋण विवरण",
   "आपके गृह ऋण आवेदन को निम्नलिखित शर्तों पर स्वीकृत किया गया है। कृपया सभी दस्तावेज़ समय पर जमा करें।",
   [["विवरण","राशि"],["ऋण राशि","₹ 18,50,000"],["ब्याज दर","9.10% वार्षिक"],["अवधि","180 माह"],["मासिक किस्त","₹ 18,920"]],
   "यह स्वीकृति जारी होने की तिथि से 60 दिन तक मान्य है।"),
 "marathi": T("कर्ज मंजुरी पत्र", "संदर्भ क्रमांक: EHF/HL/2026/07723 दिनांक: 15-11-2026",
   "श्री. सचिन दत्तात्रय पाटील, घर क्र. 22, शिवाजी नगर, पुणे 411005.", "१. कर्जाचा तपशील",
   "आपल्या गृहकर्ज अर्जाला खालील अटींवर मंजुरी देण्यात आली आहे. कृपया सर्व कागदपत्रे वेळेत सादर करावीत.",
   [["तपशील","रक्कम"],["कर्जाची रक्कम","₹ 18,50,000"],["व्याजदर","९.१०% वार्षिक"],["कालावधी","१८० महिने"],["मासिक हप्ता","₹ 18,920"]],
   "ही मंजुरी जारी केल्याच्या तारखेपासून ६० दिवसांपर्यंत वैध आहे."),
 "gujarati": T("લોન મંજૂરી પત્ર", "સંદર્ભ ક્રમાંક: EHF/HL/2026/07724 તારીખ: 15-11-2026",
   "શ્રી કિશન મહેશભાઈ પટેલ, મકાન નં. 31, નવરંગપુરા, અમદાવાદ 380009.", "1. લોનની વિગતો",
   "તમારી હોમ લોન અરજી નીચેની શરતો પર મંજૂર કરવામાં આવી છે. કૃપા કરીને બધા દસ્તાવેજો સમયસર જમા કરાવો.",
   [["વિગત","રકમ"],["લોનની રકમ","₹ 18,50,000"],["વ્યાજ દર","9.10% વાર્ષિક"],["સમયગાળો","180 મહિના"],["માસિક હપ્તો","₹ 18,920"]],
   "આ મંજૂરી જારી થયાની તારીખથી 60 દિવસ સુધી માન્ય છે."),
 "punjabi": T("ਕਰਜ਼ਾ ਮਨਜ਼ੂਰੀ ਪੱਤਰ", "ਹਵਾਲਾ ਨੰਬਰ: EHF/HL/2026/07725 ਮਿਤੀ: 15-11-2026",
   "ਸ੍ਰੀ ਗੁਰਪ੍ਰੀਤ ਸਿੰਘ ਸੰਧੂ, ਮਕਾਨ ਨੰ. 56, ਮਾਡਲ ਟਾਊਨ, ਲੁਧਿਆਣਾ 141002।", "1. ਕਰਜ਼ੇ ਦਾ ਵੇਰਵਾ",
   "ਤੁਹਾਡੀ ਹੋਮ ਲੋਨ ਦੀ ਅਰਜ਼ੀ ਹੇਠ ਲਿਖੀਆਂ ਸ਼ਰਤਾਂ ਤੇ ਮਨਜ਼ੂਰ ਕਰ ਲਈ ਗਈ ਹੈ। ਕਿਰਪਾ ਕਰਕੇ ਸਾਰੇ ਦਸਤਾਵੇਜ਼ ਸਮੇਂ ਸਿਰ ਜਮ੍ਹਾਂ ਕਰਵਾਓ।",
   [["ਵੇਰਵਾ","ਰਕਮ"],["ਕਰਜ਼ੇ ਦੀ ਰਕਮ","₹ 18,50,000"],["ਵਿਆਜ ਦਰ","9.10% ਸਾਲਾਨਾ"],["ਮਿਆਦ","180 ਮਹੀਨੇ"],["ਮਾਸਿਕ ਕਿਸ਼ਤ","₹ 18,920"]],
   "ਇਹ ਮਨਜ਼ੂਰੀ ਜਾਰੀ ਹੋਣ ਦੀ ਮਿਤੀ ਤੋਂ 60 ਦਿਨਾਂ ਤੱਕ ਵੈਧ ਹੈ।"),
 "urdu": T("قرض منظوری کا خط", "حوالہ نمبر: EHF/HL/2026/07721 تاریخ: 15-11-2026",
   "محترم محمد عرفان احمد، مکان نمبر 45، گلشن کالونی، حیدرآباد 500018۔", "1. قرض کی تفصیلات",
   "ہمیں یہ بتاتے ہوئے خوشی ہے کہ آپ کی ہوم لون کی درخواست درج ذیل شرائط پر منظور کر لی گئی ہے۔ براہ کرم تمام دستاویزات بروقت جمع کرائیں۔",
   [["تفصیل","مقدار"],["قرض کی رقم","Rs. 18,50,000"],["شرح سود","9.10% سالانہ"],["مدت","180 ماہ"],["ماہانہ قسط","Rs. 18,920"]],
   "یہ منظوری جاری ہونے کی تاریخ سے 60 دن تک جائز ہے۔"),
}
esc = lambda s: s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
def html(blocks, cls):
    out = []
    for k, v in blocks:
        if k == "table":
            out.append("<table>" + "".join("<tr>" + "".join(f"<{'th' if i == 0 else 'td'}>{esc(c)}</{'th' if i == 0 else 'td'}>" for c in r) + "</tr>" for i, r in enumerate(v)) + "</table>")
        else: out.append(f"<{k}>{esc(v)}</{k}>")
    return f'<body class="{cls}">' + "".join(out) + "</body>"
def digital(h, path):
    story = fitz.Story(html=h, user_css=CSS, archive=fitz.Archive(str(FD)))
    w = fitz.DocumentWriter(str(path)); mb = fitz.paper_rect("a4"); more = 1
    while more:
        dev = w.begin_page(mb); more, _ = story.place(mb + (50, 50, -50, -50)); story.draw(dev); w.end_page()
    w.close()
def degrade(img, seed, hard):
    r = random.Random(seed); g = img.convert("L")
    g = g.rotate(r.uniform(-3, 3) if hard else r.uniform(-1.2, 1.2), resample=Image.BICUBIC, fillcolor=255)
    a = np.asarray(g).astype(np.float32) * (0.85 if hard else 0.92) + 12
    a += np.random.default_rng(seed).normal(0, 16 if hard else 9, a.shape)
    g = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.9 if hard else 0.6))
    b = io.BytesIO(); g.save(b, "JPEG", quality=35 if hard else 55); return Image.open(io.BytesIO(b.getvalue())).convert("L")
def scan(dpdf, out, seed, hard):
    src = fitz.open(str(dpdf)); o = fitz.open()
    for n, pg in enumerate(src):
        pix = pg.get_pixmap(dpi=150 if hard else 200)
        sc = degrade(Image.frombytes("RGB", (pix.width, pix.height), pix.samples), seed + n, hard)
        b = io.BytesIO(); sc.save(b, "JPEG", quality=70)
        p = o.new_page(width=pg.rect.width, height=pg.rect.height); p.insert_image(p.rect, stream=b.getvalue())
    o.save(str(out))
for i, (k, blocks) in enumerate(DOCS.items()):
    d = OUT / f"{k}_digital.pdf"; digital(html(blocks, "ur" if k == "urdu" else ""), d)
    scan(d, OUT / f"{k}_scan.pdf", 200 + i, False); scan(d, OUT / f"{k}_hardscan.pdf", 300 + i, True)
    lines = []
    for kind, v in blocks:
        lines += [" ".join(c for c in r if c) for r in v] if kind == "table" else [v]
    (OUT / f"{k}.gt.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / f"{k}.tables.json").write_text(json.dumps([v for kd, v in blocks if kd == "table"], ensure_ascii=False), encoding="utf-8")
    pix = fitz.open(str(d))[0].get_pixmap(dpi=60); pix.save(str(OUT / f"_preview_{k}.png")); print("wrote", k)
