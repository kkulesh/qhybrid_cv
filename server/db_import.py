"""Copy the metadata kept in files into PostgreSQL: datasets, model registries (registry.json) and experiment
grids (results/<run>/*.csv). Used by scripts/init_db.py and, for what is still missing, at server start-up."""
from server import datasets as qdatasets, db, results as qresults


def dataset_info(ds):
    """Counts from the features file; the classes come from registry.json if the features are not on disk."""
    try:
        info = qdatasets.info(ds)
    except Exception:
        info = {}
    if not info.get("classes"):
        reg = qresults.read_registry(ds) or {}
        info = {**info, "classes": reg.get("classes", []), "dim": reg.get("feature_dim")}
    return info


def import_datasets(all_datasets, only_missing_models=False):
    """Upsert every dataset and its registry.json models. only_missing_models=True adds the models of a
    dataset only when the database has none for it (start-up sync); False overwrites them (init script)."""
    counts = {"datasets": 0, "models": 0}
    with db.connect() as c:
        for ds in all_datasets.values():
            db.upsert_dataset_row(c, ds, dataset_info(ds))
            counts["datasets"] += 1
            if only_missing_models:
                cur = c.execute("SELECT count(*) FROM models WHERE dataset_id = %s", (ds.id,))
                if cur.fetchone()[0]:
                    continue
            for m in (qresults.read_registry(ds) or {}).get("models", []):
                db.upsert_model_row(c, ds.id, m, ds.models_dir / f"{m['id']}.pt")
                counts["models"] += 1
    return counts


def import_experiments(results_dir, only_new=False):
    """Every results/<run> folder with hybrid.csv / baselines.csv becomes one experiment run.
    Re-importing a run replaces its rows. only_new=True skips runs already in the database."""
    counts = {"runs": 0, "rows": 0}
    with db.connect() as c:
        known = {r[0] for r in c.execute("SELECT id FROM experiment_runs").fetchall()} if only_new else set()
        for d in qresults.run_dirs(results_dir):
            if d.name in known:
                continue
            rows = qresults.read_run_csv(d)
            db.replace_experiment_run(c, d.name, qresults.RUN_NAMES.get(d.name, d.name), str(d), rows)
            counts["runs"] += 1
            counts["rows"] += len(rows)
    return counts
