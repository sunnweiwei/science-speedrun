"""Prepare a tiny synthetic two-phase Docker verifier smoke fixture."""

import argparse
import shutil
import sys
from pathlib import Path

from fixture_data import fixture_source

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "environment" / "data"))
sys.path.insert(0, str(ROOT / "environment" / "references" / "scaffold"))
sys.path.insert(0, str(ROOT / "docs" / "maintenance" / "legacy-neural"))

from audit import run as audit_run
from materialize import run as materialize_run
from baseline import train


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    root=args.output
    source,materialized,submission=root/"source",root/"output",root/"submission"
    fixture_source(source)
    audit_run(source,materialized,ROOT/"environment"/"data"/"cms-2018-gem.zip")
    materialize_run(source,materialized,1)
    submission.mkdir(parents=True,exist_ok=True)
    starter=ROOT/"environment"/"references"/"scaffold"
    for name in ("infer.sh", "predict.py", "data.py", "README.md"):
        shutil.copy2(starter/name,submission/name)
    shutil.copy2(ROOT/"docs"/"maintenance"/"legacy-neural"/"baseline.py", submission/"model.py")
    shutil.copy2(ROOT/"docs"/"maintenance"/"legacy-neural"/"adapter.py", submission/"adapter.py")
    shutil.copy2(ROOT/"docs"/"maintenance"/"legacy-neural"/"train_baseline.py", submission/"train.py")
    (submission/"config.json").write_text('{"fixture_only": true}\n')
    train(materialized/"train",materialized/"manifest.json",submission/"model.pt",
          epochs=1,batch_size=2,max_samples=4)
    (root/"sealed").mkdir(exist_ok=True)
    (root/"results").mkdir(exist_ok=True)
    print(root)


if __name__ == "__main__":
    main()
