"""Shared fail-closed error for baselines without a public training recipe."""


class OfficialTrainingUnavailable(RuntimeError):
    """Raised instead of silently substituting a locally invented recipe."""
