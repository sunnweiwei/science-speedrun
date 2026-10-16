"""Explicit resource-loader corrections, never changes to CFD source assets."""
import copy

MISSING_NOZZLE_RESTART = frozenset({'Nozzle2D_fine', 'Nozzle3D_coarse'})


def bind_nozzle_restart(case_id, case_setup, restart_path):
    """Enable the provided restart that these pinned case JSONs otherwise ignore.

    Apply to the runtime case setup before constructing the native JAX-Fluids
    InputManager. Keep the source JSON/HDF5 read-only; do not interpolate a grid
    or synthesize an initial field. This alone is not an evaluator adapter.
    """
    if case_id not in MISSING_NOZZLE_RESTART:
        raise ValueError('No reviewed missing-restart correction for this case')
    if 'restart' in case_setup:
        raise ValueError('Refusing to replace an existing source restart choice')
    if not isinstance(restart_path, str) or not restart_path:
        raise ValueError('Expected the exact supplied restart path')
    result = copy.deepcopy(case_setup)
    result['restart'] = {'is_restart': True, 'file_path': restart_path}
    return result
