"""Submission entry point for a FAIRChem UMA checkpoint (copy next to model.pt in /workspace/submission).

The evaluator calls make_calculator(submission_dir, device) once per worker and drives every
relaxation, displacement and strain itself. Units: total energy eV, forces eV/A, stress eV/A^3
(ASE sign convention). No network access and no checkpoint downloads at inference.
"""
from pathlib import Path


def make_calculator(submission_dir: Path, device: str):
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.calculate.pretrained_mlip import load_predict_unit

    unit = load_predict_unit(str(Path(submission_dir) / "model.pt"), device=device, inference_settings="default")
    return FAIRChemCalculator(unit, task_name="omat")
