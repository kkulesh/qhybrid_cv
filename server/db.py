"""PostgreSQL storage of QCV_Hybrid metadata and results.

The database keeps what used to be scattered over JSON and CSV files: datasets, the model registry,
encoding-selection runs, experiment grids and every classification. Large binary files stay on disk and the
database stores their paths: images, YOLO11 features (.npz), model weights (.pt), yolo11n-cls.pt, logs.

The database is optional. It is used when the environment variable QHYBRID_DB holds a connection string,
for example
    postgresql://qhybrid:password@localhost:5432/qhybrid_cv
Without it (or if the server cannot be reached at start-up) the app works with the files as before.
While the database is on, the files are still written (registry.json, meta.json), so switching it off
loses nothing. Create the tables and import the existing files with  python -m scripts.init_db
"""
import logging
import os
import threading

log = logging.getLogger("uvicorn.error")

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
    id            text PRIMARY KEY,
    name          text NOT NULL,
    source        text NOT NULL,            -- 'builtin' (features/*.npz) or 'zip' (uploaded archive)
    classes       jsonb NOT NULL DEFAULT '[]',
    n_classes     integer,
    n_train       integer,
    n_test        integer,
    feature_dim   integer,
    features_path text NOT NULL,            -- YOLO11 features on disk
    models_dir    text NOT NULL,            -- model weights (.pt) on disk
    samples_dir   text,
    created       timestamptz,
    updated       timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS models (
    dataset_id    text NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    id            text NOT NULL,
    kind          text NOT NULL CHECK (kind IN ('hybrid', 'classical')),  -- classical = logistic regression
    encoding      text,                     -- amplitude / angle / zz
    n_qubits      integer,
    gamma         double precision,
    layers        integer,
    seed          integer,
    params        integer,
    val_f1        double precision,
    test_f1       double precision,
    test_acc      double precision,
    train_s       double precision,
    custom        boolean NOT NULL DEFAULT false,   -- trained on the «Тренування» page
    weights_path  text NOT NULL,            -- .pt file on disk
    extra         jsonb NOT NULL DEFAULT '{}',      -- epochs, lr, batch size, F1 of all seeds ...
    created       timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (dataset_id, id)
);

CREATE TABLE IF NOT EXISTS selection_runs (
    id            serial PRIMARY KEY,
    dataset_id    text NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    n_qubits      integer NOT NULL,
    per_class     integer NOT NULL,
    sample_size   integer,
    seed          integer,
    gammas        jsonb NOT NULL,
    encodings     jsonb NOT NULL,
    rows          jsonb NOT NULL,           -- KTA (and kernel statistics) of every encoding and gamma
    best_encoding text,
    best_gamma    double precision,
    best_kta      double precision,
    time_s        double precision,
    created       timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS experiment_runs (
    id            text PRIMARY KEY,         -- e.g. voc_round1 (folder under results/)
    name          text NOT NULL,
    source        text,                     -- the folder the CSV files were imported from
    imported      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS experiments (
    id            serial PRIMARY KEY,
    run_id        text NOT NULL REFERENCES experiment_runs(id) ON DELETE CASCADE,
    dataset       text NOT NULL,
    kind          text NOT NULL CHECK (kind IN ('hybrid', 'classical')),
    model         text NOT NULL,            -- encoding for a hybrid, baseline name for a classical model
    n             integer,                  -- qubits (hybrid) or width (baseline)
    gamma         double precision,
    seed          integer,
    params        integer,
    acc           double precision,
    f1            double precision,
    ms_per_img    double precision,
    train_s       double precision,
    extra         jsonb NOT NULL DEFAULT '{}'      -- the other CSV columns (validation history ...)
);
CREATE INDEX IF NOT EXISTS experiments_run_dataset ON experiments (run_id, dataset);

CREATE TABLE IF NOT EXISTS classifications (
    id            bigserial PRIMARY KEY,
    created       timestamptz NOT NULL DEFAULT now(),
    dataset_id    text NOT NULL,
    model_id      text NOT NULL,
    image_name    text,
    image_sha256  text,
    device        text,                     -- default.qubit / braket.local.qubit; NULL for logistic regression
    shots         integer,                  -- NULL: exact expectation values
    predicted     text NOT NULL,
    confidence    double precision,
    top           jsonb NOT NULL,           -- most probable classes with probabilities
    timings_ms    jsonb NOT NULL,
    compare       boolean NOT NULL DEFAULT false    -- part of «Порівняти всі моделі»
);
CREATE INDEX IF NOT EXISTS classifications_created ON classifications (created DESC);
"""

_state = {"enabled": None, "error": None}
_lock = threading.Lock()


def url():
    return os.environ.get("QHYBRID_DB", "").strip()


def connect():
    import psycopg  # imported here so the app runs without the driver when no database is configured
    return psycopg.connect(url(), autocommit=True, connect_timeout=5)


def enabled():
    """True when QHYBRID_DB is set and the database answered at start-up (see startup())."""
    return bool(_state["enabled"])


def status():
    return {"configured": bool(url()), "enabled": enabled(), "error": _state["error"]}


def init_schema():
    with connect() as c:
        c.execute(SCHEMA)


def startup():
    """Create the tables if needed. Any failure turns the database off for this run, with a warning."""
    with _lock:
        if not url():
            _state.update(enabled=False, error=None)
            log.info("[база] QHYBRID_DB не задано: метадані й результати зберігаються у файлах")
            return False
        try:
            init_schema()
            _state.update(enabled=True, error=None)
            log.info("[база] PostgreSQL підключено")
        except Exception as e:
            _state.update(enabled=False, error=str(e))
            log.warning("[база] не вдалося підключитися до PostgreSQL (%s); працюю з файлами", e)
        return enabled()


def _jsonb(v):
    from psycopg.types.json import Jsonb
    return Jsonb(v)


def _safe(fn):
    """Writes made while serving a request must never break it: a database error is logged and skipped."""
    def wrapper(*a, **kw):
        if not enabled():
            return None
        try:
            return fn(*a, **kw)
        except Exception as e:
            log.warning("[база] %s: %s", fn.__name__, e)
            return None
    wrapper.__name__ = fn.__name__
    return wrapper


# ---------- writes ----------

def upsert_dataset_row(c, ds, info):
    c.execute("""
        INSERT INTO datasets (id, name, source, classes, n_classes, n_train, n_test, feature_dim,
                              features_path, models_dir, samples_dir, created, updated)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
        ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name, source = EXCLUDED.source, classes = EXCLUDED.classes,
            n_classes = EXCLUDED.n_classes, n_train = EXCLUDED.n_train, n_test = EXCLUDED.n_test,
            feature_dim = EXCLUDED.feature_dim, features_path = EXCLUDED.features_path,
            models_dir = EXCLUDED.models_dir, samples_dir = EXCLUDED.samples_dir,
            created = COALESCE(datasets.created, EXCLUDED.created), updated = now()""",
              (ds.id, ds.name, "builtin" if ds.builtin else "zip", _jsonb(info.get("classes", [])),
               len(info.get("classes", [])) or None, info.get("n_train"), info.get("n_test"), info.get("dim"),
               str(ds.features), str(ds.models_dir), str(ds.samples_dir) if ds.samples_dir else None, ds.created))


MODEL_COLS = ("id", "kind", "encoding", "n", "gamma", "layers", "seed", "params", "val_f1", "test_f1", "test_acc",
              "train_s", "custom")


def upsert_model_row(c, dataset_id, meta, weights_path):
    extra = {k: v for k, v in meta.items() if k not in MODEL_COLS and k != "created"}
    c.execute("""
        INSERT INTO models (dataset_id, id, kind, encoding, n_qubits, gamma, layers, seed, params, val_f1, test_f1,
                            test_acc, train_s, custom, weights_path, extra, created)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, COALESCE(%s::timestamptz, now()))
        ON CONFLICT (dataset_id, id) DO UPDATE SET kind = EXCLUDED.kind, encoding = EXCLUDED.encoding,
            n_qubits = EXCLUDED.n_qubits, gamma = EXCLUDED.gamma, layers = EXCLUDED.layers, seed = EXCLUDED.seed,
            params = EXCLUDED.params, val_f1 = EXCLUDED.val_f1, test_f1 = EXCLUDED.test_f1,
            test_acc = EXCLUDED.test_acc, train_s = EXCLUDED.train_s, custom = EXCLUDED.custom,
            weights_path = EXCLUDED.weights_path, extra = EXCLUDED.extra""",
              (dataset_id, meta["id"], meta["kind"], meta.get("encoding"), meta.get("n"), meta.get("gamma"),
               meta.get("layers"), meta.get("seed"), meta.get("params"), meta.get("val_f1"), meta.get("test_f1"),
               meta.get("test_acc"), meta.get("train_s"), bool(meta.get("custom")), str(weights_path),
               _jsonb(extra), meta.get("created")))


@_safe
def save_dataset(ds, info):
    with connect() as c:
        upsert_dataset_row(c, ds, info)


@_safe
def delete_dataset(ds_id):
    with connect() as c:
        c.execute("DELETE FROM datasets WHERE id = %s", (ds_id,))


@_safe
def save_model(ds, info, meta, weights_path):
    with connect() as c:
        upsert_dataset_row(c, ds, info)  # the dataset row must exist for the foreign key
        upsert_model_row(c, ds.id, meta, weights_path)


@_safe
def save_selection(ds, info, params, result):
    best = result["best"]
    with connect() as c:
        upsert_dataset_row(c, ds, info)
        c.execute("""
            INSERT INTO selection_runs (dataset_id, n_qubits, per_class, sample_size, seed, gammas, encodings, rows,
                                        best_encoding, best_gamma, best_kta, time_s)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                  (ds.id, params["n"], params["per_class"], result.get("sample_size"), params.get("seed"),
                   _jsonb(params["gammas"]), _jsonb(params["encodings"]), _jsonb(result["rows"]),
                   best["encoding"], best["gamma"], best["kta"], result.get("time_s")))


@_safe
def save_classifications(dataset_id, results, image_name, image_sha256, compare=False):
    with connect() as c:
        with c.cursor() as cur:
            cur.executemany("""
                INSERT INTO classifications (dataset_id, model_id, image_name, image_sha256, device, shots, predicted,
                                             confidence, top, timings_ms, compare)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                            [(dataset_id, r["model"]["id"], image_name, image_sha256, r.get("device"), r.get("shots"),
                              r["label"], r["confidence"], _jsonb(r["top"]), _jsonb(r["timings_ms"]), compare)
                             for r in results])


def replace_experiment_run(c, run_id, name, source, rows):
    """Replace all rows of one experiment run (re-importing a folder gives the same result)."""
    c.execute("DELETE FROM experiment_runs WHERE id = %s", (run_id,))
    c.execute("INSERT INTO experiment_runs (id, name, source) VALUES (%s, %s, %s)", (run_id, name, source))
    with c.cursor() as cur:
        cur.executemany("""
            INSERT INTO experiments (run_id, dataset, kind, model, n, gamma, seed, params, acc, f1, ms_per_img,
                                     train_s, extra)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                        [(run_id, r["dataset"], r["kind"], r["model"], r.get("n"), r.get("gamma"), r.get("seed"),
                          r.get("params"), r.get("acc"), r.get("f1"), r.get("ms_per_img"), r.get("train_s"),
                          _jsonb(r.get("extra", {}))) for r in rows])


# ---------- reads ----------

def _dicts(cur):
    names = [d.name for d in cur.description]
    return [dict(zip(names, row)) for row in cur.fetchall()]


def query(sql, args=()):
    with connect() as c, c.cursor() as cur:
        cur.execute(sql, args)
        return _dicts(cur)


def models_of(dataset_id):
    """Registry entries of one dataset in the registry.json format (None if the database is off or fails)."""
    if not enabled():
        return None
    try:
        rows = query("SELECT * FROM models WHERE dataset_id = %s ORDER BY custom, created, id", (dataset_id,))
    except Exception as e:
        log.warning("[база] models_of: %s", e)
        return None
    return [model_meta(r) for r in rows]


def model_meta(r):
    return {**(r.get("extra") or {}), "id": r["id"], "kind": r["kind"], "encoding": r["encoding"], "n": r["n_qubits"],
            "gamma": r["gamma"], "layers": r["layers"], "seed": r["seed"], "params": r["params"], "val_f1": r["val_f1"],
            "test_f1": r["test_f1"], "test_acc": r["test_acc"], "train_s": r["train_s"], "custom": r["custom"],
            "created": r["created"].isoformat(timespec="seconds") if r.get("created") else None}
