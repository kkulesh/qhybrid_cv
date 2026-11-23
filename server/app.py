"""QHybrid-CV web API: classify an image with the frozen YOLO11 backbone and a hybrid (VQC) or
classical head, reporting the time spent in every stage; choose the encoding (KTA), train models and
add own datasets, all as background jobs with progress streamed to the browser.

Run:  uvicorn server.app:app --port 8000
Env:  QHYBRID_MODELS    VOC models: registry.json and <id>.pt      (default models/voc)
      QHYBRID_YOLO      YOLO11-cls weights                         (default yolo11n-cls.pt)
      QHYBRID_FEATURES  VOC features for encoding selection and training
                        (default features/voc.npz, made by scripts/extract_features.py)
      QHYBRID_DATASETS  where uploaded datasets are kept            (default datasets)
Other features/<name>.npz files appear as built-in datasets with models in models/<name>.
If web/dist exists it is served at "/", so one process hosts the whole app.
"""
import asyncio
import io
import json
import logging
import os
import shutil
import threading
import time
import zipfile
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from server import circuit as qcircuit, datasets, jobs

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
FEATURES = Path(os.environ.get("QHYBRID_FEATURES", ROOT / "features" / "voc.npz"))
DATA_DIR = Path(os.environ.get("QHYBRID_DATASETS", ROOT / "datasets"))
SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
MAX_UPLOAD = 10 * 1024 * 1024
MAX_DATASET_ZIP = 2 * 1024 ** 3

torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
extractor = YoloFeatureExtractor(YOLO_WEIGHTS)
extract_lock = threading.Lock()  # the extractor keeps its output in a hook, so one batch at a time


def all_datasets():
    return datasets.discover(ROOT, DATA_DIR, FEATURES, MODELS_DIR, SAMPLES_DIR)


def get_dataset(ds_id):
    ds = all_datasets().get(ds_id)
    if ds is None:
        raise HTTPException(404, f"Набір даних '{ds_id}' не знайдено.")
    return ds


class Store:
    """Trained models of one dataset, loaded once and kept in memory."""

    def __init__(self, ds):
        self.ds = ds
        reg = ds.models_dir / "registry.json"
        if reg.is_file():
            self.registry = json.loads(reg.read_text(encoding="utf-8"))
        else:
            i = datasets.info(ds)
            self.registry = {"dataset": ds.id, "classes": i.get("classes", []), "feature_dim": i.get("dim", 1280), "models": []}
        self.classes = self.registry["classes"]
        self.models = {}
        for meta in self.registry["models"]:
            f = ds.models_dir / f"{meta['id']}.pt"
            if f.is_file():
                self.models[meta["id"]] = (meta, self.load(meta, f))

    def load(self, meta, f):
        if meta["kind"] == "classical":
            m = torch.nn.Linear(self.registry["feature_dim"], len(self.classes))
        else:
            enc = make_encoding(meta["encoding"], meta["n"])
            m = HybridModel(self.registry["feature_dim"], len(self.classes), enc, meta["gamma"], meta["layers"])
        m.load_state_dict(torch.load(f, map_location="cpu"))
        return m.eval()

    def add(self, meta, model):
        self.models[meta["id"]] = (meta, model)


stores: dict[str, Store] = {}


def get_store(ds_id) -> Store:
    if ds_id not in stores:
        stores[ds_id] = Store(get_dataset(ds_id))
    return stores[ds_id]


get_store("voc")  # load the VOC models at start-up, as before

app = FastAPI(title="QHybrid-CV API", version="0.2")


def ms(t0):
    return round((time.perf_counter() - t0) * 1000, 3)


@torch.no_grad()
def run_head(st, model_id, z):
    """Run one head on features z (1 x 1280); returns probabilities and per-stage timings."""
    meta, m = st.models[model_id]
    classes = st.classes
    t = {}
    if meta["kind"] == "classical":
        t0 = time.perf_counter(); logits = m(z); t["head"] = ms(t0)
        p = torch.softmax(logits, -1)[0]
        log.info("[%s/%s] YOLO11 %d ознак -> лінійний класифікатор -> %s (%.3f)",
                 st.ds.id, model_id, z.shape[1], classes[int(p.argmax())], float(p.max()))
        return p, t
    # 1. Adapter (qhybrid/models.py: Adapter): 1280 features -> k numbers prepared for this encoding
    #    (angle/ZZ: k = n angles; amplitude: k = 2^n amplitudes of a unit vector).
    t0 = time.perf_counter(); x = m.adapter(z); t["adapter"] = ms(t0)
    # 2. Quantum circuit (qhybrid/models.py: VQC): the encoding gates load x into n qubits
    #    (qhybrid/encodings.py: *Encoding.circuit), then StronglyEntanglingLayers, then <Z_i> is measured.
    #    Encoding and variational layers form one circuit, so they are timed together.
    t0 = time.perf_counter(); q = m.vqc(x); t["circuit"] = ms(t0)
    # 3. Classical head: BatchNorm + linear layer, n expectation values -> class scores.
    t0 = time.perf_counter(); logits = m.head(m.norm(q)); t["head"] = ms(t0)
    p = torch.softmax(logits, -1)[0]
    xs = ", ".join(f"{v:.3f}" for v in x[0, :6].tolist())
    qs = ", ".join(f"{v:.3f}" for v in q[0, :6].tolist())
    log.info("[%s/%s] YOLO11 %d ознак -> адаптер -> %d чисел [%s%s] -> кодування %s на %d кубітах "
             "-> VQC (%d шари) -> <Z> [%s%s] -> голова -> %s (%.3f)",
             st.ds.id, model_id, z.shape[1], x.shape[1], xs, ", ..." if x.shape[1] > 6 else "",
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
    with extract_lock:
        t0 = time.perf_counter(); z = extractor.extract_tensor(x); t_bb = ms(t0)
    return z, {"preprocess": t_pre, "backbone": t_bb}


def result(st, model_id, p, timings, top_k):
    meta, classes = st.models[model_id][0], st.classes
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
    return {"status": "ok", "datasets": len(all_datasets()), "models": len(get_store("voc").models)}


@app.get("/api/models")
def list_models(dataset: str = "voc"):
    st = get_store(dataset)
    return {"dataset": dataset, "classes": st.classes, "models": [m for m, _ in st.models.values()]}


def samples_of(ds):
    if ds.samples_dir is None or not ds.samples_dir.is_dir():
        return []
    index = ds.samples_dir / "index.json"
    if index.is_file():
        return json.loads(index.read_text(encoding="utf-8"))
    return [{"name": f.name, "label": f.stem.rsplit("_", 1)[0]} for f in sorted(ds.samples_dir.glob("*.jpg"))]


@app.get("/api/samples")
def list_samples(dataset: str = "voc"):
    return samples_of(get_dataset(dataset))


@app.get("/api/samples/{name}")
def get_sample(name: str, dataset: str = "voc"):
    ds = get_dataset(dataset)
    if not any(s["name"] == name for s in samples_of(ds)):
        raise HTTPException(404, "Приклад не знайдено.")
    return FileResponse(ds.samples_dir / name)


def model_or_404(st, model):
    if model not in st.models:
        raise HTTPException(404, f"Модель '{model}' не знайдено в наборі '{st.ds.name}'.")


@app.post("/api/classify")
async def classify(file: UploadFile = File(...), model: str = Form("amplitude-8q"), top_k: int = Form(5),
                   dataset: str = Form("voc")):
    st = get_store(dataset)
    model_or_404(st, model)
    img = read_image(await file.read())
    z, t = features(img)
    p, th = run_head(st, model, z)
    return result(st, model, p, {**t, **th}, top_k)


@app.post("/api/classify/compare")
async def compare(file: UploadFile = File(...), top_k: int = Form(3), dataset: str = Form("voc")):
    """Run the backbone once and every head of the dataset on the same features."""
    st = get_store(dataset)
    if not st.models:
        raise HTTPException(404, "Для цього набору ще немає навчених моделей.")
    img = read_image(await file.read())
    z, t = features(img)
    out = []
    for mid in st.models:
        p, th = run_head(st, mid, z)
        out.append(result(st, mid, p, th, top_k))
    return {"backbone_ms": round(t["preprocess"] + t["backbone"], 3), "results": out}


# ---------- datasets ----------

def dataset_row(ds):
    reg = ds.models_dir / "registry.json"
    n_models = len(json.loads(reg.read_text(encoding="utf-8"))["models"]) if reg.is_file() else 0
    return {**datasets.info(ds), "models": n_models}


@app.get("/api/datasets")
def list_datasets():
    return [dataset_row(ds) for ds in all_datasets().values()]


@app.get("/api/datasets/{ds_id}")
def dataset_info(ds_id: str):
    return dataset_row(get_dataset(ds_id))


@app.post("/api/datasets")
async def upload_dataset(file: UploadFile = File(...), name: str = Form(...), test_share: float = Form(0.2)):
    """Upload a ZIP of class folders; the server unpacks it and extracts YOLO11 features as a background job."""
    name = name.strip()[:60]
    if not name:
        raise HTTPException(400, "Вкажіть назву набору даних.")
    if not 0.05 <= test_share <= 0.5:
        raise HTTPException(400, "Частка тестових зображень має бути від 5 до 50 %.")
    if jobs.running_job():
        raise HTTPException(409, "Вже виконується інше завдання. Дочекайтеся його завершення або скасуйте його.")
    ds_id = datasets.slug(name)
    up = DATA_DIR / "_uploads"
    up.mkdir(parents=True, exist_ok=True)
    zp, size = up / f"{ds_id}.zip", 0
    with open(zp, "wb") as f:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_DATASET_ZIP:
                f.close(); zp.unlink(missing_ok=True)
                raise HTTPException(413, "Архів завеликий: максимум 2 ГБ.")
            f.write(chunk)
    if not zipfile.is_zipfile(zp):
        zp.unlink(missing_ok=True)
        raise HTTPException(400, "Файл не є ZIP-архівом.")
    log.info("[набір %s] завантажено архів %s (%.1f МБ)", ds_id, file.filename, size / 2 ** 20)
    job = jobs.start("dataset", {"dataset": ds_id, "name": name, "test_share": test_share, "file": file.filename},
                     lambda j: datasets.run_import(j, zp, DATA_DIR, extractor, extract_lock, log), log)
    if job is None:
        zp.unlink(missing_ok=True)
    return {**started(job), "dataset": ds_id}


@app.delete("/api/datasets/{ds_id}")
def delete_dataset(ds_id: str):
    ds = get_dataset(ds_id)
    if ds_id == "voc":
        raise HTTPException(400, "Набір PASCAL VOC є основним і не видаляється.")
    j = jobs.running_job()
    if j and j.params.get("dataset") == ds_id:
        raise HTTPException(409, "Набір зараз використовується завданням.")
    if ds.builtin:  # features/<id>.npz made by scripts/extract_features.py, models in models/<id>
        ds.features.unlink(missing_ok=True)
        shutil.rmtree(ds.models_dir, ignore_errors=True)
    else:
        shutil.rmtree(DATA_DIR / ds_id, ignore_errors=True)
    stores.pop(ds_id, None)
    log.info("[набір %s] видалено", ds_id)
    return {"deleted": ds_id}


# ---------- encoding selection and training (background jobs, progress streamed as SSE) ----------

class SelectionParams(BaseModel):
    dataset: str = "voc"
    n: int = Field(8, ge=2, le=10)
    per_class: int = Field(20, ge=2, le=100)
    gammas: list[float] = Field([0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0], min_length=1, max_length=16)
    encodings: list[str] = Field(["angle", "zz", "amplitude"], min_length=1)
    seed: int = 0


class TrainParams(BaseModel):
    dataset: str = "voc"
    kind: str = "hybrid"  # or "classical": a linear layer on the 1280 features, for comparison
    encoding: str = "amplitude"
    n: int = Field(8, ge=2, le=10)
    gamma: float = Field(1.0, gt=0, le=10)
    layers: int = Field(2, ge=1, le=6)
    epochs: int = Field(15, ge=1, le=100)
    lr: float = Field(0.01, gt=0, le=1)
    batch_size: int = Field(64, ge=8, le=1024)
    per_class: int = Field(199, ge=2, le=100_000)
    seed: int = 0


def need_features(ds):
    if not ds.features.is_file():
        raise HTTPException(503, f"Немає файлу ознак {ds.features.name} для набору '{ds.name}'. Спершу виконайте "
                                 "python scripts/extract_features.py (див. README).")


def check_encodings(names):
    bad = [e for e in names if e not in ENC_GATES]
    if bad:
        raise HTTPException(400, f"Невідоме кодування: {', '.join(bad)}.")


def started(job):
    if job is None:
        raise HTTPException(409, "Вже виконується інше завдання. Дочекайтеся його завершення або скасуйте його.")
    return {"id": job.id}


@app.post("/api/selection")
def start_selection(p: SelectionParams):
    ds = get_dataset(p.dataset)
    need_features(ds); check_encodings(p.encodings)
    return started(jobs.start("selection", p.model_dump(), lambda j: jobs.run_selection(j, ds.features, log), log))


@app.post("/api/train")
def start_training(p: TrainParams):
    ds = get_dataset(p.dataset)
    need_features(ds)
    if p.kind not in ("hybrid", "classical"):
        raise HTTPException(400, "Тип моделі має бути hybrid або classical.")
    if p.kind == "hybrid":
        check_encodings([p.encoding])
    st = get_store(p.dataset)
    return started(jobs.start("train", p.model_dump(),
                              lambda j: jobs.run_training(j, ds.features, ds.models_dir, st.add, log), log))


@app.get("/api/jobs")
def list_jobs():
    """Jobs of this server process, newest first (the pages use it to reconnect after a reload)."""
    return [{k: v for k, v in j.snapshot().items() if k != "events"} for j in reversed(jobs.JOBS.values())]


def get_job(job_id):
    if job_id not in jobs.JOBS:
        raise HTTPException(404, "Завдання не знайдено.")
    return jobs.JOBS[job_id]


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    return get_job(job_id).snapshot()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = get_job(job_id)
    job.cancel_requested = True
    return {"id": job.id, "status": job.status}


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str):
    """Server-Sent Events: every event of the job from the start, then new ones as they appear."""
    job = get_job(job_id)

    async def stream():
        i = 0
        while True:
            while i < len(job.events):
                ev = job.events[i]; i += 1
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                if ev["type"] in jobs.FINAL:
                    return
            await asyncio.sleep(0.25)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


# ---------- circuit composer: run the drawn circuit in PennyLane ----------

class CircuitOp(BaseModel):
    kind: str
    wires: list[int]
    params: list[float] = []
    col: int = 0


class CircuitRun(BaseModel):
    nQubits: int
    ops: list[CircuitOp] = Field(max_length=2000)
    shots: int = Field(1024, ge=1, le=100000)
    seed: int = 1


@app.post("/api/circuit/run")
def run_circuit(c: CircuitRun):
    ops = [o.model_dump() for o in c.ops]
    try:
        r = qcircuit.run(c.nQubits, ops, c.shots, c.seed)
    except ValueError as e:
        raise HTTPException(400, str(e))
    gates = sum(o["kind"] not in ("MEASURE", "BARRIER") for o in ops)
    top = sorted(r.get("counts", {}).items(), key=lambda kv: -kv[1])[:4]
    log.info("[схема] PennyLane default.qubit: кубітів %d, вентилів %d, виміряно c[%s], shots %d (seed %d) -> %s (%.1f мс)",
             c.nQubits, gates, ", ".join(map(str, r["measured"])) or "-", c.shots, c.seed,
             ", ".join(f"{k}: {v}" for k, v in top) or "без вимірювань", r["time_ms"])
    return r


dist = ROOT / "web" / "dist"
if dist.is_dir():
    app.mount("/", StaticFiles(directory=dist, html=True), name="web")
