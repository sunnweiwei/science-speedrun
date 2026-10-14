"""tokagym: FreeGSNKE-backed magnetic-control environments for the AI4Sci tokamak task.

The trusted evaluator and the agent's training code import the same package; the
evaluator additionally owns the sealed perturbation distributions and seeds.
"""

from .machine import MachineSpec, load_machine  # noqa: F401
from .cases import CASES, Case, get_case  # noqa: F401
from .env import TokamakMagneticEnv  # noqa: F401

ENV_VERSION = "tokagym-mastu/v0.3"
