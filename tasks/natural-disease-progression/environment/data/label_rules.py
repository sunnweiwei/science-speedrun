"""Public cohort-label semantics used by training and the trusted evaluator."""

from datetime import timedelta


def disease_label(row: dict, horizon: int) -> tuple[int, bool]:
    if not row["risk_set"] or row["ambiguous_prior"]:
        return 0, False
    cutoff = row["cutoff"].date() if hasattr(row["cutoff"], "time") else row["cutoff"]
    boundary = row["diagnosis_observation_end"]
    boundary = boundary.date() if hasattr(boundary, "time") else boundary
    death_days = row["death_days"]
    event_days = row["event_days"]
    positive = (event_days is not None and 0 <= event_days <= horizon
                and cutoff + timedelta(days=event_days) <= boundary
                and (death_days is None or event_days <= death_days))
    observed_through = cutoff + timedelta(days=horizon) <= boundary
    terminal_before_boundary = (death_days is not None and 0 <= death_days <= horizon
                                and cutoff + timedelta(days=death_days) <= boundary)
    return int(positive), bool(positive or observed_through or terminal_before_boundary)


def death_label(row: dict, horizon: int) -> tuple[int, bool]:
    cutoff = row["cutoff"].date() if hasattr(row["cutoff"], "time") else row["cutoff"]
    death_days = row["death_days"]
    positive = death_days is not None and 0 <= death_days <= horizon
    boundary = row["death_observation_end"]
    boundary = boundary.date() if hasattr(boundary, "time") else boundary
    observed_through = cutoff + timedelta(days=horizon) <= boundary
    return int(positive), bool(positive or observed_through)
