"""Primary reward: case detection for serious new-onset disease at a fixed alert budget.

For each endpoint in a prespecified clinical panel, eligible patients are complete-year patients
whose one-year label is valid (the disease is not yet recorded) and whose history carries none of
the endpoint's precursor or recoding codes. Patients are ranked by their predicted 365-day risk,
and the score is the share of next-year cases found among the top 5 % ("if 1 in 20 discharged
patients can be followed up, how many future cases are reached"). Chance is 0.05 and perfect is
1.0. The reward is the mean over the endpoints with at least MIN_POSITIVES eligible cases. That set
depends only on the frozen labels, never on predictions.

Ties are scored by their expectation under random tie-breaking, so neither row order nor a
constant prediction can earn more than chance (ceil(0.05 n)/n, i.e. 0.05 up to rounding).
"""

from __future__ import annotations

import math

import numpy as np

ALERT_RATES = (0.05, 0.01)
PRIMARY = "panel_recall_at_5pct"
MIN_POSITIVES = 20

# Prespecified before any model was scored: serious outcomes where earlier anticipation can
# change care. Obstetric, substance-use, organism-supplementary, complication-of-care and symptom
# codes are excluded because they mostly continue already recorded care. The second element lists
# history codes that make a patient ineligible: precursors or recodings of the same condition,
# beyond the task's own same-code risk set. "ICD9:" entries match unmapped ICD-9 history codes by
# prefix; other entries match three-character ICD-10 categories exactly.
PANEL = {
    "I50": ("heart failure", ("I11", "I13", "ICD9:428")),
    "I48": ("atrial fibrillation/flutter", ("ICD9:4273",)),
    "I21": ("acute myocardial infarction", ("I22", "ICD9:410", "ICD9:412")),
    "I63": ("ischaemic stroke", ("I69", "ICD9:433", "ICD9:434", "ICD9:438")),
    "I61": ("intracerebral haemorrhage", ("ICD9:431",)),
    "I26": ("pulmonary embolism", ("ICD9:4151",)),
    "I82": ("venous thrombosis", ("ICD9:453",)),
    "I71": ("aortic aneurysm/dissection", ("ICD9:441",)),
    "E11": ("type 2 diabetes", ("E08", "E09", "E10", "E13", "ICD9:250")),
    "N18": ("chronic kidney disease", ("I12", "I13", "N19", "ICD9:585", "ICD9:403", "ICD9:404")),
    "N17": ("acute kidney injury", ("ICD9:584",)),
    "A41": ("sepsis", ("ICD9:038", "ICD9:99591", "ICD9:99592")),
    "J18": ("pneumonia", ("ICD9:481", "ICD9:482", "ICD9:485", "ICD9:486")),
    "J96": ("respiratory failure", ("ICD9:51881", "ICD9:51882", "ICD9:51884")),
    "K74": ("hepatic fibrosis/cirrhosis", ("K70", "ICD9:5712", "ICD9:5715", "ICD9:5716")),
    "K72": ("hepatic failure", ("ICD9:5722",)),
    "F03": ("dementia", ("F01", "F02", "G30", "G31", "ICD9:290", "ICD9:2941", "ICD9:331")),
    "J44": ("COPD", ("J43", "ICD9:491", "ICD9:492", "ICD9:496")),
    "J84": ("interstitial lung disease", ("ICD9:515", "ICD9:516")),
    "C18": ("colon cancer", ("ICD9:153",)),
    "C22": ("liver cancer", ("ICD9:155",)),
    "C25": ("pancreatic cancer", ("ICD9:157",)),
    "C61": ("prostate cancer", ("ICD9:185",)),
    "C67": ("bladder cancer", ("ICD9:188",)),
    "C78": ("metastasis, respiratory/digestive",
            ("C77", "C79", "C80", "ICD9:196", "ICD9:197", "ICD9:198", "ICD9:199")),
    "C79": ("metastasis, other sites",
            ("C77", "C78", "C80", "ICD9:196", "ICD9:197", "ICD9:198", "ICD9:199")),
}


def history_code(code: str) -> str:
    """The history token compared with PANEL families: ICD-10 category or raw ICD-9 code."""
    return code if code.startswith("ICD9:") else code[:3]


def has_family_history(codes: set[str], family: tuple[str, ...]) -> bool:
    icd10 = {f for f in family if not f.startswith("ICD9:")}
    icd9 = tuple(f for f in family if f.startswith("ICD9:"))
    return any(c in icd10 or (icd9 and c.startswith(icd9)) for c in codes)


def recall_at(y: np.ndarray, p: np.ndarray, rate: float) -> tuple[float, float]:
    """Expected recall and PPV of flagging the top ceil(rate*n) patients, ties split evenly."""
    n, positives = len(y), int(y.sum())
    k = max(1, math.ceil(rate * n))
    threshold = np.sort(p)[::-1][k - 1]
    above = p > threshold
    tied = p == threshold
    taken_from_tie = k - int(above.sum())
    found = float(y[above].sum()) + taken_from_tie * float(y[tied].sum()) / int(tied.sum())
    return found / max(positives, 1), found / k


def calibration(y: np.ndarray, p: np.ndarray) -> dict:
    q = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    logit = np.log(q / (1 - q))
    slope = None
    if 0 < y.sum() < len(y) and np.ptp(logit) > 0:
        from sklearn.linear_model import LogisticRegression
        slope = float(LogisticRegression(C=1e6, max_iter=1000)
                      .fit(logit.reshape(-1, 1), y).coef_[0, 0])
    edges = np.quantile(q, np.linspace(0, 1, 11))
    bins = np.clip(np.searchsorted(edges, q, side="right") - 1, 0, 9)
    ece = float(sum(abs(y[bins == b].mean() - q[bins == b].mean()) * (bins == b).mean()
                    for b in range(10) if (bins == b).any()))
    return {"observed_rate": float(y.mean()), "mean_prediction": float(q.mean()),
            "prediction_to_observed": float(q.mean() / max(y.mean(), 1e-12)),
            "slope": slope, "ece_decile": ece}


def panel_score(labels365: np.ndarray, valid365: np.ndarray, prediction365: np.ndarray,
                complete_year: np.ndarray, disease_codes: list[str],
                history: list[set[str]]) -> dict:
    """labels365/valid365/prediction365: [samples, diseases] at the 365-day horizon."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    endpoints = {}
    for code, (name, family) in PANEL.items():
        if code not in disease_codes:
            continue
        j = disease_codes.index(code)
        excluded = np.fromiter((has_family_history(h, family) for h in history), bool,
                               count=len(history))
        eligible = valid365[:, j] & complete_year & ~excluded
        y = labels365[eligible, j].astype(np.int64)
        p = prediction365[eligible, j].astype(np.float64)
        entry = {"name": name, "eligible": int(eligible.sum()), "positives": int(y.sum()),
                 "family_history_excluded": int((valid365[:, j] & complete_year & excluded).sum()),
                 "in_reward": bool(y.sum() >= MIN_POSITIVES)}
        if 0 < y.sum() < len(y):
            for rate in ALERT_RATES:
                recall, ppv = recall_at(y, p, rate)
                tag = f"{round(rate * 100)}pct"
                entry[f"recall_at_{tag}"] = recall
                entry[f"ppv_at_{tag}"] = ppv
            entry.update(auroc=float(roc_auc_score(y, p)),
                         auprc=float(average_precision_score(y, p)),
                         prevalence=float(y.mean()), calibration=calibration(y, p))
        endpoints[code] = entry
    scored = [e for e in endpoints.values() if e["in_reward"]]

    def mean(key):
        return float(np.mean([e[key] for e in scored])) if scored else None

    slopes = [e["calibration"]["slope"] for e in scored if e["calibration"]["slope"] is not None]
    return {
        PRIMARY: mean("recall_at_5pct"),
        "panel_recall_at_1pct": mean("recall_at_1pct"),
        "panel_ppv_at_5pct": mean("ppv_at_5pct"),
        "panel_auroc": mean("auroc"),
        "panel_auprc": mean("auprc"),
        "panel_calibration_slope_median": float(np.median(slopes)) if slopes else None,
        "panel_prediction_to_observed_median": float(np.median(
            [e["calibration"]["prediction_to_observed"] for e in scored])) if scored else None,
        "panel_ece_decile": mean_of(scored, "ece_decile"),
        "endpoints_in_reward": [c for c, e in endpoints.items() if e["in_reward"]],
        "min_positives": MIN_POSITIVES,
        "endpoints": endpoints,
    }


def mean_of(scored: list[dict], key: str):
    return float(np.mean([e["calibration"][key] for e in scored])) if scored else None
