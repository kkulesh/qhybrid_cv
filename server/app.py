"""QHybrid-CV web API: classify an image with the frozen YOLO11 backbone and a hybrid (VQC) or
classical head, reporting the time spent in every stage.

Run:  uvicorn server.app:app --port 8000
Env:  QHYBRID_MODELS  directory with registry.json and <id>.pt   (default models/voc)
      QHYBRID_YOLO    YOLO11-cls weights                         (default yolo11n-cls.pt)
If web/dist exists it is served at "/", so one process hosts the whole app.
"""
import io
import json
import logging
import os
import time
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from qhybrid.encodings import make_encoding
from qhybrid.features import YoloFeatureExtractor
from qhybrid.models import HybridModel

log = logging.getLogger("uvicorn.error")  # printed in the uvicorn console
ENC_GATES = {
    "amplitude": "AmplitudeEmbedding",
    "angle": "H + RY (AngleEmbedding)",
    "zz": "H + RZ + CNOT-RZ-CNOT, 2 повтори (ZZ feature map)",
}

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("QHYBRID_MODELS", ROOT / "models" / "voc"))
YOLO_WEIGHTS = os.environ.get("QHYBRID_YOLO", "yolo11n-cls.pt")
SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
MAX_UPLOAD = 10 * 1024 * 1024

torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
registry = json.loads((MODELS_DIR / "registry.json").read_text())
classes = registry["classes"]
extractor = YoloFeatureExtractor(YOLO_WEIGHTS)


def load_model(meta):
    if meta["kind"] == "classical":
        m = torch.nn.Linear(registry["feature_dim"], len(classes))
    else:
        enc = make_encoding(meta["encoding"], meta["n"])
        m = HybridModel(registry["feature_dim"], len(classes), enc, meta["gamma"], meta["layers"])
    m.load_state_dict(torch.load(MODELS_DIR / f"{meta['id']}.pt", map_location="cpu"))
    return m.eval()


models = {m["id"]: (m, load_model(m)) for m in registry["models"]}

app = FastAPI(title="QHybrid-CV API", version="0.1")


def ms(t0):
    return round((time.perf_counter() - t0) * 1000, 3)


@torch.no_grad()
def run_head(model_id, z):
    """Run one head on features z (1 x 1280); returns probabilities and per-stage timings."""
    meta, m = models[model_id]
    t = {}
    if meta["kind"] == "classical":
        t0 = time.perf_counter(); logits = m(z); t["head"] = ms(t0)
        p = torch.softmax(logits, -1)[0]
        log.info("[%s] YOLO11 %d ознак -> лінійний класифікатор -> %s (%.3f)",
                 model_id, z.shape[1], classes[int(p.argmax())], float(p.max()))
        return p, t
    # 1. Adapter (qhybrid/models.py: Adapter): 1280 features -> k numbers prepared for this encoding
    #    (angle/ZZ: k = n angles; amplitude: k = 2^n amplitudes of a unit vector).
    t0 = time.perf_counter(); x = m.adapter(z); t["adapter"] = ms(t0)
    # 2. Quantum circuit (qhybrid/models.py: VQC): the encoding gates load x into n qubits
    #    (qhybrid/encodings.py: *Encoding.circuit), then StronglyEntanglingLayers, then <Z_i> is measured.
    #    Encoding and variational layers form one circuit, so they are timed together.
    t0 = time.perf_counter(); q = m.vqc(x); t["circuit"] = ms(t0)
    # 3. Classical head: BatchNorm + linear layer, n expectation values -> 20 class scores.
    t0 = time.perf_counter(); logits = m.head(m.norm(q)); t["head"] = ms(t0)
    p = torch.softmax(logits, -1)[0]
    xs = ", ".join(f"{v:.3f}" for v in x[0, :6].tolist())
    qs = ", ".join(f"{v:.3f}" for v in q[0, :6].tolist())
    log.info("[%s] YOLO11 %d ознак -> адаптер -> %d чисел [%s%s] -> кодування %s на %d кубітах "
             "-> VQC (%d шари) -> <Z> [%s%s] -> голова -> %s (%.3f)",
             model_id, z.shape[1], x.shape[1], xs, ", ..." if x.shape[1] > 6 else "",
             ENC_GATES[meta["encoding"]], meta["n"], meta["layers"], qs, ", ..." if q.shape[1] > 6 else "",
             classes[int(p.argmax())], float(p.max()))
    return p, t


def read_image(data: bytes) -> Image.Image:
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Файл завеликий: максимум 10 МБ.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img
    except (UnidentifiedImageError, OSError, ImportError):  # Ultralytics' Image.open patch may raise ImportError
        raise HTTPException(400, "Не вдалося прочитати зображення. Підтримуються JPEG, PNG, WebP, BMP.")


def features(img):
    t0 = time.perf_counter(); x = extractor.to_tensor(img).unsqueeze(0); t_pre = ms(t0)
    t0 = time.perf_counter(); z = extractor.extract_tensor(x); t_bb = ms(t0)
    return z, {"preprocess": t_pre, "backbone": t_bb}


def result(model_id, p, timings, top_k):
    meta = models[model_id][0]
    top = torch.topk(p, min(top_k, len(classes)))
    return {
        "model": meta,
        "label": classes[int(top.indices[0])],
        "confidence": round(float(top.values[0]), 5),
        "top": [{"label": classes[int(i)], "p": round(float(v), 5)} for v, i in zip(top.values, top.indices)],
        "timings_ms": {**timings, "total": round(sum(timings.values()), 3)},
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "models": len(models), "dataset": registry["dataset"]}


@app.get("/api/models")
def list_models():
    return {"dataset": registry["dataset"], "classes": classes, "models": [m for m, _ in models.values()]}


@app.get("/api/samples")
def list_samples():
    return [{"name": f.name, "label": f.stem.rsplit("_", 1)[0]} for f in sorted(SAMPLES_DIR.glob("*.jpg"))]


@app.get("/api/samples/{name}")
def get_sample(name: str):
    f = (SAMPLES_DIR / name).resolve()
    if f.parent != SAMPLES_DIR.resolve() or not f.is_file():
        raise HTTPException(404, "Приклад не знайдено.")
    return FileResponse(f, media_type="image/jpeg")


@app.post("/api/classify")
async def classify(file: UploadFile = File(...), model: str = Form("amplitude-8q"), top_k: int = Form(5)):
    if model not in models:
        raise HTTPException(404, f"Модель '{model}' не знайдено.")
    img = read_image(await file.read())
    z, t = features(img)
    p, th = run_head(model, z)
    return result(model, p, {**t, **th}, top_k)


@app.post("/api/classify/compare")
async def compare(file: UploadFile = File(...), top_k: int = Form(3)):
    """Run the backbone once and every head on the same features."""
    img = read_image(await file.read())
    z, t = features(img)
    out = []
    for mid in models:
        p, th = run_head(mid, z)
        out.append(result(mid, p, th, top_k))
    return {"backbone_ms": round(t["preprocess"] + t["backbone"], 3), "results": out}


dist = ROOT / "web" / "dist"
if dist.is_dir():
    app.mount("/", StaticFiles(directory=dist, html=True), name="web")
