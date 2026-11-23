"""Create the PostgreSQL tables of QCV_Hybrid and import the existing files into them.

    PowerShell:  $env:QHYBRID_DB = "postgresql://qhybrid:password@localhost:5432/qhybrid_cv"
                 python -m scripts.init_db
    bash:        QHYBRID_DB=postgresql://qhybrid:password@localhost:5432/qhybrid_cv python -m scripts.init_db

Imports the datasets (built-in features/*.npz and uploaded datasets/<id>), every registry.json and the
experiment grids results/<run>/hybrid.csv and baselines.csv. Running it again updates the same rows, so it is
safe to repeat after new experiments. --results adds other folders with the same CSV layout.
Encoding-selection runs and classifications are recorded by the server from then on.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server import datasets as qdatasets, db, db_import  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", nargs="*", default=[str(ROOT / "results")], help="folders with experiment runs")
    p.add_argument("--no-experiments", action="store_true", help="import only datasets and models")
    a = p.parse_args()
    if not db.url():
        sys.exit("Задайте рядок підключення в змінній QHYBRID_DB, наприклад (PowerShell):\n"
                 '  $env:QHYBRID_DB = "postgresql://qhybrid:password@localhost:5432/qhybrid_cv"')
    db.init_schema()
    print("Таблиці створено (або вже існували).")
    all_ds = qdatasets.discover(
        ROOT, Path(os.environ.get("QHYBRID_DATASETS", ROOT / "datasets")),
        Path(os.environ.get("QHYBRID_FEATURES", ROOT / "features" / "voc.npz")),
        Path(os.environ.get("QHYBRID_MODELS", ROOT / "models" / "voc")), ROOT / "server" / "samples")
    c = db_import.import_datasets(all_ds)
    print(f"Наборів даних: {c['datasets']}, моделей: {c['models']}.")
    if not a.no_experiments:
        for r in a.results:
            c = db_import.import_experiments(r)
            print(f"{r}: запусків експериментів {c['runs']}, рядків {c['rows']}.")
    for t in ("datasets", "models", "selection_runs", "experiment_runs", "experiments", "classifications"):
        print(f"  {t}: {db.query(f'SELECT count(*) AS n FROM {t}')[0]['n']}")


if __name__ == "__main__":
    main()
