#!/usr/bin/env python3
"""Trusted grader for timeseries-forecast/v1 (see ../contract.md).

    python3 score.py --truth TRUTH_DIR --output OUTPUT_DIR [--report report.json] [--csv results.csv]

Per config (dataset, frequency, term) the two GIFT-Eval headline metrics are
computed as gluonts 0.15.1 does under `evaluate_model(..., axis=None,
mask_invalid_label=True)`, to float32 precision (gluonts accumulates in the
data's float32, this grader in float64, which is the whole 1.9e-7 residual in
docs/validation/golden_gluonts_full.json):

  MASE   = mean over every (instance, variate, step) with a finite label and a
           positive seasonal error of |label - q0.5| / seasonal_error(instance, variate)
           (numpy.ma masks the division by a zero seasonal error, so gluonts drops those elements)
  WQL    = (1/9) * sum_q  [ sum 2|(label - pred_q) (1{pred_q >= label} - q)| / sum |label| ]
           (sums over the same finite-label elements; the leaderboard calls this CRPS)

Each metric is divided by the task's Seasonal Naive value for the same config
(seasonal_naive_reference.json) and clipped to [REL_MIN, REL_MAX] = [1e-3, 10.0],
with single-application log-space partial-coverage blending toward REL_MAX when
a run timed out part way:

  log rel = f * log(clip(rel_covered, 1e-3, 10.0)) + (1 - f) * log(10.0)

Primary SSI V3/V4 aggregation (`7 domains -> 23 sources -> horizon`):
  1. For each configuration k, compute 0.5 * (log(rel_mase_k) + log(rel_wql_k)).
  2. Average within each (domain, source, horizon) cell across frequencies.
  3. Average across horizons within each of the 23 sources.
  4. Average across sources within each of the 7 domains.
  5. signed_score = 1 - exp(mean_domain),  score = reward = clip(signed_score, 0, 1).

The legacy flat 97-configuration geometric skill `1 - sqrt(gmean(rel MASE) * gmean(rel WQL))`
is retained in report.json as `flat_97_skill_diagnostic` (and `flat_97_reward_diagnostic`).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ts_io  # noqa: E402

REL_MIN, REL_MAX = 1e-3, 10.0
BASELINE_PATH = Path(__file__).with_name("seasonal_naive_reference.json")
QUANTILES = np.array(ts_io.QUANTILE_LEVELS)

# Frozen 97-key GIFT-Eval hierarchy and Seasonal Naive denominators (exact match to ssi_index_v3_inputs.json).
TIMESERIES_ROSTER: dict[str, tuple[str, str, str, float, float]] = {
    'bitbrains_fast_storage/5T/long': ('Web/CloudOps', 'bitbrains_fast_storage', 'long', 1.1366422318753115, 1.1773077032683998),
    'bitbrains_fast_storage/5T/medium': ('Web/CloudOps', 'bitbrains_fast_storage', 'medium', 1.2202713595882653, 1.1979128978088514),
    'bitbrains_fast_storage/5T/short': ('Web/CloudOps', 'bitbrains_fast_storage', 'short', 1.1360241737534567, 1.210386425305063),
    'bitbrains_fast_storage/H/short': ('Web/CloudOps', 'bitbrains_fast_storage', 'short', 1.2985124569638047, 1.0222395520114795),
    'bitbrains_rnd/5T/long': ('Web/CloudOps', 'bitbrains_rnd', 'long', 3.501257904408163, 1.175220506770106),
    'bitbrains_rnd/5T/medium': ('Web/CloudOps', 'bitbrains_rnd', 'medium', 4.542392314855541, 1.169291300089368),
    'bitbrains_rnd/5T/short': ('Web/CloudOps', 'bitbrains_rnd', 'short', 1.9708126376716957, 1.1015904440824371),
    'bitbrains_rnd/H/short': ('Web/CloudOps', 'bitbrains_rnd', 'short', 6.0373618687533925, 1.2432170953390527),
    'bizitobs_application/10S/long': ('Web/CloudOps', 'bizitobs_application', 'long', 3.2063450561622804, 0.04573090228155046),
    'bizitobs_application/10S/medium': ('Web/CloudOps', 'bizitobs_application', 'medium', 2.6914183385868085, 0.04268912394422428),
    'bizitobs_application/10S/short': ('Web/CloudOps', 'bizitobs_application', 'short', 2.2423302386755415, 0.03483868247005456),
    'bizitobs_l2c/5T/long': ('Web/CloudOps', 'bizitobs_l2c', 'long', 1.4542830442631411, 0.6484921267253538),
    'bizitobs_l2c/5T/medium': ('Web/CloudOps', 'bizitobs_l2c', 'medium', 1.2435997989335994, 0.5203917750796179),
    'bizitobs_l2c/5T/short': ('Web/CloudOps', 'bizitobs_l2c', 'short', 0.9860210805885202, 0.26206838623738704),
    'bizitobs_l2c/H/long': ('Web/CloudOps', 'bizitobs_l2c', 'long', 1.4260541525613812, 0.9410651242377539),
    'bizitobs_l2c/H/medium': ('Web/CloudOps', 'bizitobs_l2c', 'medium', 1.5102861226255864, 0.9042051354742671),
    'bizitobs_l2c/H/short': ('Web/CloudOps', 'bizitobs_l2c', 'short', 1.2140641267600039, 0.5211675771895117),
    'bizitobs_service/10S/long': ('Web/CloudOps', 'bizitobs_service', 'long', 1.367193469742303, 0.05345299584740217),
    'bizitobs_service/10S/medium': ('Web/CloudOps', 'bizitobs_service', 'medium', 1.3205790727650968, 0.04755831292581585),
    'bizitobs_service/10S/short': ('Web/CloudOps', 'bizitobs_service', 'short', 1.225305042620374, 0.03998233993351736),
    'car_parts/M/short': ('Sales', 'car_parts_with_missing', 'short', 1.2014638390969912, 1.7217438941460104),
    'covid_deaths/D/short': ('Healthcare', 'covid_deaths', 'short', 46.91239825526407, 0.12672327831223676),
    'electricity/15T/long': ('Energy', 'electricity', 'long', 1.1635317075319347, 0.11255332426055248),
    'electricity/15T/medium': ('Energy', 'electricity', 'medium', 1.1507882041044137, 0.11275485280084273),
    'electricity/15T/short': ('Energy', 'electricity', 'short', 1.7170601412372184, 0.16489261500071847),
    'electricity/D/short': ('Energy', 'electricity', 'short', 1.987041399842847, 0.10405987098536745),
    'electricity/H/long': ('Energy', 'electricity', 'long', 1.5240786163496851, 0.15341839756227935),
    'electricity/H/medium': ('Energy', 'electricity', 'medium', 1.3924537851702354, 0.12738701179892217),
    'electricity/H/short': ('Energy', 'electricity', 'short', 1.3577726051465433, 0.10566499518474065),
    'electricity/W/short': ('Energy', 'electricity', 'short', 2.0896995064573947, 0.0993032024444414),
    'ett1/15T/long': ('Energy', 'ett1', 'long', 1.190885705867336, 0.3401793468576481),
    'ett1/15T/medium': ('Energy', 'ett1', 'medium', 1.188305978388322, 0.32155116949786394),
    'ett1/15T/short': ('Energy', 'ett1', 'short', 0.9341549587802876, 0.24136394938567346),
    'ett1/D/short': ('Energy', 'ett1', 'short', 1.7783697877928102, 0.40843357791503654),
    'ett1/H/long': ('Energy', 'ett1', 'long', 1.4787012870437326, 0.47106248989542887),
    'ett1/H/medium': ('Energy', 'ett1', 'medium', 1.5678007461285395, 0.4349118298516912),
    'ett1/H/short': ('Energy', 'ett1', 'short', 0.9773176243434534, 0.24036844107833774),
    'ett1/W/short': ('Energy', 'ett1', 'short', 1.7689207877683926, 0.3118251468740881),
    'ett2/15T/long': ('Energy', 'ett2', 'long', 1.0126577751121104, 0.13282444765695306),
    'ett2/15T/medium': ('Energy', 'ett2', 'medium', 1.0512391501712754, 0.12412877798161984),
    'ett2/15T/short': ('Energy', 'ett2', 'short', 1.067314800011479, 0.09638438775638664),
    'ett2/D/short': ('Energy', 'ett2', 'short', 1.3901143157307532, 0.15331100213050436),
    'ett2/H/long': ('Energy', 'ett2', 'long', 1.1284016697207502, 0.20790633225768584),
    'ett2/H/medium': ('Energy', 'ett2', 'medium', 1.2384249730532626, 0.1862050814081793),
    'ett2/H/short': ('Energy', 'ett2', 'short', 0.9231678277377344, 0.08895289185095483),
    'ett2/W/short': ('Energy', 'ett2', 'short', 0.7785209533197451, 0.13367758870767996),
    'hierarchical_sales/D/short': ('Sales', 'hierarchical_sales', 'short', 1.1348033996861053, 1.7364572065613715),
    'hierarchical_sales/W/short': ('Sales', 'hierarchical_sales', 'short', 1.0250133777288337, 0.8322267526416594),
    'hospital/M/short': ('Healthcare', 'hospital', 'short', 0.9205278266364826, 0.06248817651596496),
    'jena_weather/10T/long': ('Nature', 'jena_weather', 'long', 0.7614923703210493, 0.2371949131202996),
    'jena_weather/10T/medium': ('Nature', 'jena_weather', 'medium', 0.7160100391723763, 0.21183609008512827),
    'jena_weather/10T/short': ('Nature', 'jena_weather', 'short', 0.7428874675800381, 0.15520092375681044),
    'jena_weather/D/short': ('Nature', 'jena_weather', 'short', 1.5734140297468562, 0.2105908566035542),
    'jena_weather/H/long': ('Nature', 'jena_weather', 'long', 1.2677178473920059, 0.41914340130808575),
    'jena_weather/H/medium': ('Nature', 'jena_weather', 'medium', 0.8886133410507664, 0.3428055217231077),
    'jena_weather/H/short': ('Nature', 'jena_weather', 'short', 0.7229102185557323, 0.15420788787522166),
    'kdd_cup_2018/D/short': ('Nature', 'kdd_cup_2018_with_missing', 'short', 1.4970343738270926, 0.674506129145155),
    'kdd_cup_2018/H/long': ('Nature', 'kdd_cup_2018_with_missing', 'long', 1.3355336658089667, 0.9362833347539716),
    'kdd_cup_2018/H/medium': ('Nature', 'kdd_cup_2018_with_missing', 'medium', 1.4289997072062277, 0.7587892026846513),
    'kdd_cup_2018/H/short': ('Nature', 'kdd_cup_2018_with_missing', 'short', 1.340425278558367, 0.547536580422314),
    'loop_seattle/5T/long': ('Transport', 'LOOP_SEATTLE', 'long', 1.2506857542807588, 0.12716656087996273),
    'loop_seattle/5T/medium': ('Transport', 'LOOP_SEATTLE', 'medium', 1.1532851467900647, 0.11727937998585615),
    'loop_seattle/5T/short': ('Transport', 'LOOP_SEATTLE', 'short', 0.7623397118337274, 0.08083202568463041),
    'loop_seattle/D/short': ('Transport', 'LOOP_SEATTLE', 'short', 1.7323957056549062, 0.1032547596878987),
    'loop_seattle/H/long': ('Transport', 'LOOP_SEATTLE', 'long', 1.5460268753677606, 0.18709450197084374),
    'loop_seattle/H/medium': ('Transport', 'LOOP_SEATTLE', 'medium', 1.4805126192799751, 0.16208090743299983),
    'loop_seattle/H/short': ('Transport', 'LOOP_SEATTLE', 'short', 1.2928412613913995, 0.10423508438727377),
    'm4_daily/D/short': ('Econ/Fin', 'm4', 'short', 3.278424296610481, 0.024356147648337902),
    'm4_hourly/H/short': ('Econ/Fin', 'm4', 'short', 1.1932101877276544, 0.03757255147212195),
    'm4_monthly/M/short': ('Econ/Fin', 'm4', 'short', 1.259717038974429, 0.12192063664829357),
    'm4_quarterly/Q/short': ('Econ/Fin', 'm4', 'short', 1.602247174616002, 0.09805397548466407),
    'm4_weekly/W/short': ('Econ/Fin', 'm4', 'short', 2.777295047362158, 0.06087039452311706),
    'm4_yearly/A/short': ('Econ/Fin', 'm4', 'short', 3.965954931783283, 0.13714565740983498),
    'm_dense/D/short': ('Transport', 'M_DENSE', 'short', 1.6693368812171987, 0.22686565739428607),
    'm_dense/H/long': ('Transport', 'M_DENSE', 'long', 1.4779178744488217, 0.4189507904607682),
    'm_dense/H/medium': ('Transport', 'M_DENSE', 'medium', 1.569946511898904, 0.3773835994986822),
    'm_dense/H/short': ('Transport', 'M_DENSE', 'short', 1.4875444172240992, 0.27479786933969136),
    'restaurant/D/short': ('Sales', 'restaurant', 'short', 1.0060757450046092, 0.6770164746085788),
    'saugeen/D/short': ('Nature', 'saugeenday', 'short', 3.413049305768806, 0.5850141736335629),
    'saugeen/M/short': ('Nature', 'saugeenday', 'short', 0.9763741730697157, 0.4450846286544009),
    'saugeen/W/short': ('Nature', 'saugeenday', 'short', 1.990658004828632, 0.7340352210434381),
    'solar/10T/long': ('Energy', 'solar', 'long', 0.8709317964102903, 0.6738494837655387),
    'solar/10T/medium': ('Energy', 'solar', 'medium', 0.9270005404761702, 0.6550635536405177),
    'solar/10T/short': ('Energy', 'solar', 'short', 1.1057028063509833, 0.8595386998005952),
    'solar/D/short': ('Energy', 'solar', 'short', 1.155861190610263, 0.5590651164917229),
    'solar/H/long': ('Energy', 'solar', 'long', 1.0711966477927781, 1.0779764182077312),
    'solar/H/medium': ('Energy', 'solar', 'medium', 0.9349685908749716, 0.9459874672135314),
    'solar/H/short': ('Energy', 'solar', 'short', 0.9519362458316395, 0.5916559631947232),
    'solar/W/short': ('Energy', 'solar', 'short', 1.4703506784640206, 0.20974762813186765),
    'sz_taxi/15T/long': ('Transport', 'SZ_TAXI', 'long', 0.6909431344941248, 0.4279704063043717),
    'sz_taxi/15T/medium': ('Transport', 'SZ_TAXI', 'medium', 0.7134840540400068, 0.379172158612487),
    'sz_taxi/15T/short': ('Transport', 'SZ_TAXI', 'short', 0.7644167735543712, 0.3087068505119876),
    'sz_taxi/H/short': ('Transport', 'SZ_TAXI', 'short', 0.7381672994945998, 0.21382494112013678),
    'temperature_rain/D/short': ('Nature', 'temperature_rain_with_missing', 'short', 2.0115357653222166, 1.2679455200160903),
    'us_births/D/short': ('Healthcare', 'us_births', 'short', 1.8648391912491094, 0.11952686561898423),
    'us_births/M/short': ('Healthcare', 'us_births', 'short', 0.7605310792868888, 0.016825596814792485),
    'us_births/W/short': ('Healthcare', 'us_births', 'short', 1.5634204337929731, 0.01929679613291714),
}


def _finite(x: float) -> float:
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f"Expected finite float, got {x!r}")
    return v


def _mean(xs) -> float:
    vals = list(xs)
    return sum(vals) / len(vals)


def hierarchical_time_series(ratios: dict[str, tuple[float, float]]) -> dict:
    """Exact SSI V3/V4 7-domain -> 23-source -> horizon hierarchical aggregation over all 97 leaderboard keys."""
    if set(ratios) != set(TIMESERIES_ROSTER) or len(TIMESERIES_ROSTER) != 97:
        raise ValueError("Expected the frozen 97 unique leaderboard keys")
    tree: dict[str, dict[str, dict[str, list[float]]]] = {}
    for key, (m, q) in ratios.items():
        domain, source, horizon, *_ = TIMESERIES_ROSTER[key]
        pair = [max(REL_MIN, min(REL_MAX, _finite(x))) for x in (m, q)]
        tree.setdefault(domain, {}).setdefault(source, {}).setdefault(horizon, []).append(_mean(math.log(x) for x in pair))
    if len(tree) != 7 or sum(map(len, tree.values())) != 23:
        raise ValueError("Changed GIFT-Eval hierarchy")
    domains: dict[str, float] = {}
    sources: dict[str, float] = {}
    for d, src in tree.items():
        by_source = {s: _mean(_mean(values) for values in hs.values()) for s, hs in src.items()}
        domains[d] = _mean(by_source.values())
        sources.update({f"{d} / {s}": 1.0 - math.exp(v) for s, v in by_source.items()})
    signed = 1.0 - math.exp(_mean(domains.values()))
    return {
        "score": min(1.0, max(0.0, signed)),
        "signed_score": signed,
        "domains": {d: 1.0 - math.exp(v) for d, v in domains.items()},
        "sources": sources,
    }


def score_forecast_rows(rows: list[dict]) -> dict:
    """Exact SSI V3/V4 row-table scorer (`ssi_index_v3.forecast_rows`) with single-application partial-coverage blend."""
    ratios: dict[str, tuple[float, float]] = {}
    for r in rows:
        k = r["leaderboard_key"]
        if k in ratios or k not in TIMESERIES_ROSTER:
            raise ValueError(f"Unknown or duplicate forecast configuration {k}")
        if r.get("status") == "failed":
            pair = (REL_MAX, REL_MAX)
        else:
            raw = r.get("metrics")
            has_metrics = isinstance(raw, dict) and ("mase" in raw or "task_mase" in raw) and ("wql" in raw or "task_wql" in raw)
            has_top_raw = ("mase" in r or "task_mase" in r) and ("wql" in r or "task_wql" in r)
            if has_metrics or has_top_raw:
                _, _, _, m, q = TIMESERIES_ROSTER[k]
                src = raw if has_metrics else r
                pair = (_finite(src.get("mase", src.get("task_mase"))) / m, _finite(src.get("wql", src.get("task_wql"))) / q)
                needs_partial_blend = True
            elif "rel_mase" in r:
                pair = (_finite(r["rel_mase"]), _finite(r["rel_wql"]))
                needs_partial_blend = True
            elif "relative" in r:
                pair = (_finite(r["relative"]["mase"]), _finite(r["relative"]["wql"]))
                needs_partial_blend = False
            else:
                raise ValueError(f"Missing forecast metric fields for {k}")
            if needs_partial_blend and r.get("status") == "partial":
                fraction = _finite(r["coverage"]["fraction"])
                if not 0.0 <= fraction <= 1.0:
                    raise ValueError("Invalid forecast coverage")
                pair = tuple(math.exp(fraction * math.log(max(REL_MIN, min(REL_MAX, x))) + (1.0 - fraction) * math.log(REL_MAX)) for x in pair)
        ratios[k] = pair
    return hierarchical_time_series(ratios)


def _hierarchical_configs(configs: list[dict]) -> dict:
    """Hierarchical domain -> source -> horizon aggregation for either the full 97-key roster or a subset/fixture."""
    if not configs:
        return {"score": 0.0, "signed_score": 0.0, "domains": {}, "sources": {}}
    keys = [c["leaderboard_key"] for c in configs]
    if len(configs) == 97 and len(set(keys)) == 97 and set(keys) == set(TIMESERIES_ROSTER):
        return hierarchical_time_series({c["leaderboard_key"]: (c["relative"]["mase"], c["relative"]["wql"]) for c in configs})
    tree: dict[str, dict[str, dict[str, list[float]]]] = {}
    for c in configs:
        key = c["leaderboard_key"]
        if key in TIMESERIES_ROSTER:
            domain, source, horizon, *_ = TIMESERIES_ROSTER[key]
        else:
            domain = c.get("domain") or "Unknown"
            ds_head = str(c.get("dataset", key)).split("/")[0]
            source = "m4" if ds_head.startswith("m4_") else ds_head
            horizon = c.get("term") or "short"
        pair = [max(REL_MIN, min(REL_MAX, _finite(c["relative"][k]))) for k in ("mase", "wql")]
        tree.setdefault(domain, {}).setdefault(source, {}).setdefault(horizon, []).append(_mean(math.log(x) for x in pair))
    domains: dict[str, float] = {}
    sources: dict[str, float] = {}
    for d, src in tree.items():
        by_source = {s: _mean(_mean(values) for values in hs.values()) for s, hs in src.items()}
        domains[d] = _mean(by_source.values())
        sources.update({f"{d} / {s}": 1.0 - math.exp(v) for s, v in by_source.items()})
    signed = 1.0 - math.exp(_mean(domains.values()))
    return {
        "score": min(1.0, max(0.0, signed)),
        "signed_score": signed,
        "domains": {d: 1.0 - math.exp(v) for d, v in domains.items()},
        "sources": sources,
    }


def sort_quantiles(q: np.ndarray):
    """Monotone rearrangement along the quantile axis; returns (sorted, fraction of crossing elements)."""
    s = np.sort(q, axis=-2)
    crossing = np.any(s != q, axis=-2)
    return s, float(crossing.mean()) if crossing.size else 0.0


def config_metrics(labels: np.ndarray, seasonal_error: np.ndarray, mean: np.ndarray, quantiles: np.ndarray) -> dict:
    """labels (N, D, H), seasonal_error (N, D), mean (N, D, H), quantiles (N, D, 9, H) -> MASE and WQL as gluonts computes them."""
    lab = labels.astype(np.float64)
    valid = np.isfinite(lab)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return {"mase": float("nan"), "wql": float("nan"), "n_valid_labels": 0, "n_masked_labels": int((~valid).sum())}
    q = quantiles.astype(np.float64)
    median = q[:, :, 4, :]
    se = seasonal_error.astype(np.float64)[:, :, None]
    # gluonts divides a masked array by the seasonal error; numpy.ma masks the elements where the divisor is 0
    # (and a fully-masked history yields 0), so those elements drop out of both the sum and the count.
    mase_valid = valid & np.broadcast_to(np.isfinite(se) & (se > 0), lab.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        ase = np.abs(lab - median) / se
    ase_valid = ase[mase_valid]
    mase = float(ase_valid.sum() / ase_valid.size) if ase_valid.size else float("nan")
    abs_label_sum = float(np.abs(lab[valid]).sum())
    wsql = []
    for j, level in enumerate(QUANTILES):
        pred = q[:, :, j, :]
        ql = 2.0 * np.abs((lab - pred) * ((pred >= lab).astype(np.float64) - level))
        wsql.append(float(ql[valid].sum()) / abs_label_sum if abs_label_sum > 0 else float("nan"))
    return {"mase": mase, "wql": float(np.mean(wsql)), "wsql_by_level": wsql,
            "n_valid_labels": n_valid, "n_masked_labels": int((~valid).sum()),
            "mae_median": float(np.abs(lab - median)[valid].mean()),
            "n_mase_elements": int(mase_valid.sum()),
            "n_seasonal_error_zero": int((seasonal_error == 0).sum()), "n_seasonal_error_nan": int(np.isnan(seasonal_error).sum())}


def blend(rel: float, fraction: float) -> float:
    """Single-application geometric blend toward REL_MAX for instances that were never forecast."""
    if fraction >= 1.0:
        return max(REL_MIN, min(REL_MAX, float(rel)))
    if fraction <= 0.0 or not math.isfinite(rel) or rel <= 0.0:
        return REL_MAX
    clipped = max(REL_MIN, min(REL_MAX, float(rel)))
    return max(REL_MIN, min(REL_MAX, math.exp(float(fraction) * math.log(clipped) + (1.0 - float(fraction)) * math.log(REL_MAX))))


def json_safe(obj):
    """Replace non-finite floats with null so the reports are valid JSON for strict parsers (jq, JSON.parse)."""
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def gmean(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.exp(np.mean(np.log(x)))) if x.size else float("nan")


def relative(value: float, base: float) -> float:
    if not math.isfinite(value) or not math.isfinite(base) or base <= 0.0:
        return REL_MAX
    return max(REL_MIN, min(REL_MAX, float(value) / float(base)))


def score(truth_dir: Path, output_dir: Path, baseline: dict | None = None, allow_missing_baseline: bool = False, configs=None,
          coverage: dict | None = None) -> dict:
    """Grade OUTPUT_DIR against TRUTH_DIR.

    `configs` restricts scoring to a subset of config ids (validation during training). `coverage` is
    {config_id: boolean mask over instances} and is supplied by the verifier when a run was cut short; it is never
    read from the package's own output, which the package controls.
    """
    truth_dir, output_dir = Path(truth_dir), Path(output_dir)
    index = json.loads((truth_dir / "index.json").read_text())
    ids = [c["config_id"] for c in index["configs"]]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate config_id in the truth index")
    if configs:
        wanted = set(configs)
        index["configs"] = [c for c in index["configs"] if c["config_id"] in wanted]
    if baseline is None:
        if BASELINE_PATH.exists():
            baseline = json.loads(BASELINE_PATH.read_text())["configs"]
        elif allow_missing_baseline:
            baseline = {}
        else:
            raise RuntimeError(f"Seasonal Naive reference {BASELINE_PATH} missing")
    scored, failures = [], []
    for cfg in index["configs"]:
        cid = cfg["config_id"]
        labels, se, meta = ts_io.read_truth_file(truth_dir / "cases" / f"{cid}.h5")
        entry = {**cfg, "leaderboard_key": ts_io.leaderboard_key(cfg["dataset"], cfg["term"]), "status": "ok"}
        base = baseline.get(cid)
        if base is None:
            if not allow_missing_baseline:
                raise RuntimeError(f"no Seasonal Naive reference for config {cid}")
            base = {"mase": 1.0, "wql": 1.0}
        entry["baseline"] = {"mase": base["mase"], "wql": base["wql"]}
        pred_path = output_dir / "cases" / f"{cid}.h5"
        reason, covered = None, None
        try:
            if not pred_path.exists():
                reason = "missing prediction file"
            else:
                # read the shapes first: a mis-declared dataset must not be materialized
                mean, quantiles = ts_io.read_prediction_file(pred_path, expect=labels.shape)
                covered = np.ones(labels.shape[0], bool) if not coverage or cid not in coverage else np.asarray(coverage[cid], bool)
                if covered.shape != (labels.shape[0],):
                    raise RuntimeError(f"coverage mask for {cid} has shape {covered.shape}, expected {(labels.shape[0],)}")
                if not (np.isfinite(mean[covered]).all() and np.isfinite(quantiles[covered]).all()):
                    reason = "non-finite values in prediction"
        except Exception as exc:  # noqa: BLE001 - an unreadable or mis-shaped file is a model failure, not a grader failure
            reason = f"{exc}" if isinstance(exc, ValueError) and str(exc).startswith("bad shapes") else \
                     f"unreadable prediction file: {type(exc).__name__}: {exc}"
        if reason is None:
            quantiles, entry["quantile_crossing_fraction"] = sort_quantiles(quantiles)
            fraction = float(covered.mean())
            entry["coverage"] = {"instances_scored": int(covered.sum()), "instances": int(labels.shape[0]), "fraction": fraction}
            entry["metrics"] = config_metrics(labels[covered], se[covered], mean[covered], quantiles[covered])
            entry["relative"] = {k: blend(relative(entry["metrics"][k], base[k]), fraction) for k in ("mase", "wql")}
            if not np.isfinite(entry["metrics"]["mase"]) or not np.isfinite(entry["metrics"]["wql"]):
                entry["status"] = "failed"; entry["reason"] = "non-finite metric"
                entry["relative"] = {"mase": REL_MAX, "wql": REL_MAX}
                failures.append({"config_id": cid, "reason": entry["reason"]})
            elif fraction < 1.0:
                entry["status"] = "partial"; entry["reason"] = f"{entry['coverage']['instances_scored']}/{labels.shape[0]} instances forecast"
                failures.append({"config_id": cid, "reason": entry["reason"]})
        else:
            entry["status"] = "failed"; entry["reason"] = reason
            entry["relative"] = {"mase": REL_MAX, "wql": REL_MAX}
            failures.append({"config_id": cid, "reason": reason})
        scored.append(entry)
    configs = scored
    rel_m = [c["relative"]["mase"] for c in configs]; rel_w = [c["relative"]["wql"] for c in configs]
    gm_m, gm_w = gmean(rel_m), gmean(rel_w)
    flat_skill = 1.0 - float(np.sqrt(gm_m * gm_w)) if configs else 0.0
    flat_reward = float(np.clip(flat_skill, 0.0, 1.0)) if configs else 0.0
    hier = _hierarchical_configs(configs)
    primary_score = hier["score"]
    signed_score = hier["signed_score"]
    groups = {}
    for key, fn in (("term", lambda c: c["term"]), ("domain", lambda c: c["domain"]), ("frequency", lambda c: ts_io.parse_freq(c["freq"])[1]),
                    ("variates", lambda c: "univariate" if c["n_variates"] == 1 else "multivariate")):
        groups[key] = {}
        for c in configs:
            groups[key].setdefault(fn(c), []).append(c)
        groups[key] = {g: {"n_configs": len(v), "rel_mase_gmean": gmean([c["relative"]["mase"] for c in v]),
                           "rel_wql_gmean": gmean([c["relative"]["wql"] for c in v])} for g, v in sorted(groups[key].items())}
    return {"contract": ts_io.CONTRACT, "split": index.get("split"),
            "score": primary_score, "reward": primary_score,
            "signed_score": signed_score, "skill_score": signed_score,
            "domain_scores": hier["domains"], "source_scores": hier["sources"],
            "flat_97_skill_diagnostic": flat_skill, "flat_97_reward_diagnostic": flat_reward,
            "rel_mase_gmean": gm_m, "rel_wql_gmean": gm_w,
            "n_configs": len(configs), "n_failed": sum(1 for c in configs if c["status"] == "failed"),
            "n_instances": sum(c["n_instances"] for c in configs),
            "n_instances_scored": sum(c.get("coverage", {}).get("instances_scored", c["n_instances"]) for c in configs),
            "n_partial": sum(1 for c in configs if c["status"] == "partial"), "failures": failures, "groups": groups,
            "quantile_crossing_fraction_mean": float(np.mean([c.get("quantile_crossing_fraction", 0.0) for c in configs])) if configs else 0.0,
            "configs": configs}


def write_results_csv(report: dict, path: Path, model: str):
    """Per-config values in the leaderboard's all_results.csv column names (the two scored metrics only)."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "model", "eval_metrics/MASE[0.5]", "eval_metrics/mean_weighted_sum_quantile_loss", "domain", "num_variates"])
        for c in report["configs"]:
            m = c.get("metrics", {})
            w.writerow([c["leaderboard_key"], model, m.get("mase", ""), m.get("wql", ""), c["domain"], c["n_variates"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--input", default="", help="ignored (kept for older command lines)")
    ap.add_argument("--report", default=""); ap.add_argument("--csv", default=""); ap.add_argument("--model-name", default="submission")
    ap.add_argument("--baseline", default="", help="alternative Seasonal Naive reference JSON")
    ap.add_argument("--allow-missing-baseline", action="store_true")
    args = ap.parse_args()
    baseline = json.loads(Path(args.baseline).read_text())["configs"] if args.baseline else None
    report = score(args.truth, args.output, baseline, args.allow_missing_baseline)
    if args.report:
        Path(args.report).write_text(json.dumps(json_safe(report), indent=1, allow_nan=False) + "\n")
    if args.csv:
        write_results_csv(report, Path(args.csv), args.model_name)
    print(json.dumps({k: report[k] for k in ("score", "reward", "signed_score", "skill_score", "flat_97_skill_diagnostic", "rel_mase_gmean", "rel_wql_gmean", "n_configs", "n_failed")}))


if __name__ == "__main__":
    main()
