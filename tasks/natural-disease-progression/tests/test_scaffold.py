from pathlib import Path

import numpy as np


TASK_ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = TASK_ROOT / "environment"
DELPHI = ENVIRONMENT / "references" / "delphi"


def test_task_metadata_is_versioned() -> None:
    metadata = (TASK_ROOT / "task.toml").read_text(encoding="utf-8")
    assert 'name = "ai4sci/natural-disease-progression"' in metadata
    assert 'input_schema = "natural-disease-progression-input/v2"' in metadata


def test_task_layers_exist() -> None:
    assert (ENVIRONMENT / "data" / "README.md").is_file()
    assert (ENVIRONMENT / "references" / "scaffold" / "data.py").is_file()
    assert (ENVIRONMENT / "references" / "scaffold" / "predict.py").is_file()
    assert (ENVIRONMENT / "baseline" / "temporal-gru" / "longitudinal.py").is_file()
    assert (ENVIRONMENT / "evaluation" / "score.py").is_file()


def test_agent_surface_is_neural_architecture_development() -> None:
    instruction = (TASK_ROOT / "instruction.md").read_text(encoding="utf-8")
    contract = (ENVIRONMENT / "contract.md").read_text(encoding="utf-8")
    dockerfile = (ENVIRONMENT / "Dockerfile").read_text(encoding="utf-8")
    assert "neural architecture development track" in instruction
    assert "does not obtain its final predictions from LightGBM" in instruction
    assert "fixed population time scaling" in contract
    assert "COPY starter /workspace/starter" not in dockerfile
    assert "COPY data /workspace/data" not in dockerfile
    assert "maintenance" not in dockerfile
    assert "starter/tabular_baseline.py" not in dockerfile


def test_real_evaluator_is_present() -> None:
    source = (ENVIRONMENT / "evaluation" / "score.py").read_text()
    assert "from risk_scoring import score" in source
    assert "panel_scoring" in source


def test_verifier_separates_public_and_trusted_inputs() -> None:
    main = (TASK_ROOT / "tests" / "main_verifier.py").read_text()
    grader = (TASK_ROOT / "tests" / "grader.py").read_text()
    assert "--trusted" not in main
    assert "--trusted" in grader


def test_harbor_entry_runs_inference_not_unit_tests() -> None:
    entry = (TASK_ROOT / "tests" / "test.sh").read_text()
    assert "main_verifier.py" in entry
    assert "pytest" not in entry
    assert "/logs/verifier" in (TASK_ROOT / "tests" / "main_verifier.py").read_text()


def test_verifier_requires_review_readme_without_claiming_architecture_detection() -> None:
    main = (TASK_ROOT / "tests" / "main_verifier.py").read_text(encoding="utf-8")
    assert "submission/README.md is required for architecture review" in main
    assert "cannot establish that infer.sh actually uses" in main


def test_verifier_main_installs_official_starter_inference_dependencies() -> None:
    dockerfile = (TASK_ROOT / "tests" / "Dockerfile").read_text(encoding="utf-8")
    for dependency in ("lightgbm==4.6.0", "joblib==1.5.3"):
        assert dependency in dockerfile


def test_compose_mounts_do_not_expose_materialization_roots() -> None:
    development = (ENVIRONMENT / "docker-compose.yaml").read_text()
    verifier = (TASK_ROOT / "tests" / "docker-compose.yaml").read_text()
    # task.toml owns the network policy (TASK_STANDARD); the compose file must not override it.
    assert "network_mode" not in development
    assert 'network_mode = "no-network"' in (TASK_ROOT / "task.toml").read_text()
    assert "target: /workspace/data/train" in development
    assert "target: /workspace/data/public/validation" in development
    assert "target: /workspace/data/trusted/validation" in development
    assert "target: /data" not in development
    assert "/public/test" in verifier
    assert "target: /trusted/test" in verifier
    assert "target: /input" in verifier
    assert "target: /trusted\n" not in verifier


def test_delphi_demo_data_has_uint32_triples() -> None:
    for split in ("train", "val"):
        data = np.memmap(DELPHI / "data" / "ukb_simulated_data" / f"{split}.bin", dtype=np.uint32, mode="r")
        assert data.size > 0
        assert data.size % 3 == 0


def test_vendored_python_compiles() -> None:
    for name in ("model.py", "train.py", "utils.py", "configurator.py"):
        source = (DELPHI / name).read_text(encoding="utf-8")
        compile(source, str(DELPHI / name), "exec")
