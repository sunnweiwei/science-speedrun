"""Exercise acceptance and fail-closed behavior without GPU training."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from verify_lfmc_pair import verify


class GateTest(unittest.TestCase):
    def test_pair_and_rejections(self):
        receipt = json.loads(
            (Path(__file__).parent / "downstream-lfmc/pilot-v1-launch.json").read_text()
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt["output_root"] = str(root)
            for name, package in zip(receipt["outputs"], receipt["source_packages"]):
                output = root / name
                output.mkdir()
                package_path = root / package
                package_path.mkdir()
                trained = name == receipt["outputs"][0]
                provenance = dict(
                    encoder_sha256="encoder",
                    initialization_seed=12536,
                    variant="large",
                    parameters=307770368,
                    origin="own_from_scratch_training"
                    if trained
                    else "random_initialization",
                )
                (package_path / "provenance.json").write_text(json.dumps(provenance))
                settings = {
                    key: receipt[key]
                    for key in [
                        "seed",
                        "epochs",
                        "unfreeze_epoch",
                        "global_batch",
                        "microbatch",
                    ]
                }
                settings.update(
                    package=str(package_path),
                    pilot=True,
                    test_used=False,
                    counts=receipt["expected_counts"],
                    batch_limit_per_epoch=1,
                    source_sha256={
                        Path(k).name: v
                        for k, v in receipt["source_sha256"].items()
                        if k.startswith("environment/")
                    },
                    encoder_file_sha256="encoder",
                    encoder_state_digest="initial",
                    head_state_digest="head",
                    index_sha256="index",
                    gpu="A6000",
                    training_crop="paired",
                    validation_crop="centered",
                )
                epochs = [
                    dict(
                        epoch=i,
                        encoder_trainable=i >= 20,
                        lr=[1e-5] * (2 if i >= 20 else 1),
                        train_windows=32,
                        train_loss=1.0,
                        validation=dict(windows=32, valid_pixels=32, mse=1.0, mae=1.0),
                    )
                    for i in range(22)
                ]
                for filename, value in {
                    "settings": settings,
                    "epochs": epochs,
                    "unfreeze-boundary": dict(epoch=20, frozen_weights_unchanged=True),
                    "completion": dict(
                        epochs=22,
                        encoder_changed=True,
                        final_encoder_state_digest="final",
                        test_used=False,
                    ),
                }.items():
                    (output / (filename + ".json")).write_text(json.dumps(value))
                (output / "best.pt").touch()
                (output / "last.pt").touch()
            self.assertTrue(verify(receipt)["gate_passed"])
            path = root / receipt["outputs"][1] / "settings.json"
            original = json.loads(path.read_text())
            for key, value in [
                ("test_used", True),
                ("head_state_digest", "different"),
                ("index_sha256", "different"),
                ("source_sha256", {}),
                ("microbatch", 4),
            ]:
                changed = copy.deepcopy(original)
                changed[key] = value
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError, msg=key):
                    verify(receipt)
            path.write_text(json.dumps(original))
            path = root / receipt["outputs"][0] / "epochs.json"
            rows = json.loads(path.read_text())
            rows[20]["encoder_trainable"] = False
            path.write_text(json.dumps(rows))
            with self.assertRaises(ValueError):
                verify(receipt)


if __name__ == "__main__":
    unittest.main()
