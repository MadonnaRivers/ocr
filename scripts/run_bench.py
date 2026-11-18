"""Run OCR over the test samples and record outputs, timings, RAM and VRAM.

usage: python -I scripts/run_bench.py <ocr_dir> <config_name> [--gpu] [--threads N] [--repeats R]

Writes outputs/<config>/<input>.md, <input>.json and outputs/<config>/metrics.json.
Every number in metrics.json is measured in this process; nothing is copied from the repo.
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import psutil

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("config")
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--repeats", type=int, default=3)
a = ap.parse_args()

ROOT = Path(a.root).resolve()
S = ROOT / "samples"
OUT = ROOT / "outputs" / a.config
OUT.mkdir(parents=True, exist_ok=True)
proc = psutil.Process()

# ---- memory samplers ---------------------------------------------------------
nvml = None
try:
    import pynvml
    pynvml.nvmlInit()
    nvml = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    pass


def gpu_used_mb():
    if nvml is None:
        return None
    return pynvml.nvmlDeviceGetMemoryInfo(nvml).used / 2**20


class Peak:
    """Samples RSS and device-wide VRAM every 20 ms while active."""

    def __init__(self):
        self.rss = 0
        self.vram = 0.0
        self._on = False

    def __enter__(self):
        self.rss = proc.memory_info().rss
        self.vram = gpu_used_mb() or 0.0
        self._on = True
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()
        return self

    def _loop(self):
        while self._on:
            self.rss = max(self.rss, proc.memory_info().rss)
            v = gpu_used_mb()
            if v is not None:
                self.vram = max(self.vram, v)
            time.sleep(0.02)

    def __exit__(self, *exc):
        self._on = False
        self.t.join()


baseline_vram = gpu_used_mb()
rss0 = proc.memory_info().rss

# ---- load ----------------------------------------------------------------------
import indra_ocr as engine  # noqa: E402
import onnxruntime as ort  # noqa: E402

if a.gpu:
    ort.preload_dlls()  # load the pip-installed CUDA/cuDNN DLLs on Windows
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if a.gpu else None
t0 = time.perf_counter()
with Peak() as pk_load:
    conv_ocr = engine.load(threads=a.threads, mode="ocr", providers=providers)
load_s = time.perf_counter() - t0
conv_auto = engine.load(threads=a.threads, mode="auto", providers=providers)
active = conv_ocr.ocr.det.get_providers()
print(f"[{a.config}] ORT {ort.__version__} available={ort.get_available_providers()} det session providers={active}")
print(f"[{a.config}] model load {load_s:.2f}s")

# ---- inputs --------------------------------------------------------------------
manifest = json.loads((S / "manifest.json").read_text(encoding="utf-8"))
jobs = []
for doc_id in manifest:
    if doc_id == "multipage_hi_gu":
        jobs.append((f"{doc_id}_scan.pdf", doc_id, "ocr"))
        continue
    jobs.append((f"{doc_id}_scan.pdf", doc_id, "ocr"))       # scanned PDF, OCR path
    jobs.append((f"{doc_id}_scan.png", doc_id, "ocr"))       # raw image input
    jobs.append((f"{doc_id}_digital.pdf", doc_id, "auto"))   # born-digital, text layer allowed
    jobs.append((f"{doc_id}_digital.pdf", doc_id, "ocr"))    # born-digital, forced OCR
    if (S / f"{doc_id}_hardscan.pdf").exists():
        jobs.append((f"{doc_id}_hardscan.pdf", doc_id, "ocr"))  # heavily degraded photocopy
    if (S / f"{doc_id}_edge.pdf").exists():
        jobs.append((f"{doc_id}_edge.pdf", doc_id, "auto"))  # Edge-printed PDF, correct text layer
extra = ROOT.parent / "sample_hindi.png"  # earlier project sample (no ground truth)
if extra.exists():
    jobs.append((str(extra), None, "ocr"))


def run(conv, path):
    pages = []
    for i, page in conv.convert(path):
        pages.append((i, page))
    return pages


results = []
for fname, doc_id, mode in jobs:
    path = Path(fname) if os.path.isabs(fname) else S / fname
    conv = conv_ocr if mode == "ocr" else conv_auto
    times = []
    with Peak() as pk:
        for r in range(a.repeats):
            t = time.perf_counter()
            pages = run(conv, path)
            times.append(time.perf_counter() - t)
    stem = f"{path.stem}.{mode}"
    md = "\n\n".join((p.markdown if p.markdown is not None else p.text) for _, p in pages)
    txt = "\n".join(p.text for _, p in pages)
    (OUT / f"{stem}.md").write_text(md, encoding="utf-8")
    (OUT / f"{stem}.txt").write_text(txt, encoding="utf-8")
    (OUT / f"{stem}.json").write_text(
        json.dumps([{
            "page": i, "source": p.source, "note": p.note, "rotation": p.rotation, "skew": p.skew,
            "lines": [asdict(ln) for ln in p.lines], "markdown": p.markdown,
            "blocks": p.blocks, "timings": p.timings,
        } for i, p in pages], ensure_ascii=False, indent=1, default=float),
        encoding="utf-8")
    rec = {
        "file": path.name, "doc_id": doc_id, "mode": mode, "pages": len(pages),
        "sources": [p.source for _, p in pages], "notes": [p.note for _, p in pages],
        "first_run_s": round(times[0], 3),
        "median_s": round(statistics.median(times), 3),
        "s_per_page_median": round(statistics.median(times) / max(1, len(pages)), 3),
        "peak_rss_mb": round(pk.rss / 2**20, 1),
        "peak_vram_device_mb": round(pk.vram, 1) if nvml else None,
        "errors": [p.note for _, p in pages if p.source == "error"],
        "out_stem": stem,
    }
    results.append(rec)
    print(f"[{a.config}] {path.name:32s} {mode:4s} pages={rec['pages']} src={rec['sources']} "
          f"first={rec['first_run_s']}s med={rec['median_s']}s rss={rec['peak_rss_mb']}MB "
          f"vram={rec['peak_vram_device_mb']}")

meta = {
    "config": a.config, "threads": a.threads, "gpu_requested": a.gpu,
    "session_providers": active, "onnxruntime": ort.__version__, "engine": engine.__version__,
    "model_dir": str(engine.model_dir()),
    "model_load_s": round(load_s, 2),
    "rss_before_import_mb": round(rss0 / 2**20, 1),
    "rss_after_load_mb": round(proc.memory_info().rss / 2**20, 1),
    "peak_rss_overall_mb": round(max([pk_load.rss] + [0]) / 2**20, 1),
    "vram_baseline_device_mb": round(baseline_vram, 1) if baseline_vram is not None else None,
    "cpu": f"{psutil.cpu_count(logical=False)} cores / {psutil.cpu_count()} threads",
    "repeats": a.repeats,
    "results": results,
}
meta["peak_rss_overall_mb"] = max([meta["peak_rss_overall_mb"]] + [r["peak_rss_mb"] for r in results])
(OUT / "metrics.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[{a.config}] wrote {OUT / 'metrics.json'}")
