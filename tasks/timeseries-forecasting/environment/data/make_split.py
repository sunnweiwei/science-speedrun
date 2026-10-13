#!/usr/bin/env python3
"""Write split.json (all 97 GIFT-Eval configs) and split_smoke.json from the materialized data.

    python3 make_split.py --data /data [--out split.json --smoke split_smoke.json]

Every protocol quantity (horizon, windows, seasonality, series count, variates)
is read from the Arrow files through evaluation/ts_io.py and frozen here so
the case builder can refuse data that drift from the manifest.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
import ts_io  # noqa: E402

# gift-eval notebooks: datasets with medium/long terms, and the domain table (dataset_properties.json)
SHORT = ("m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D electricity/W "
         "solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W saugeenday/D saugeenday/M "
         "saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing "
         "restaurant hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H "
         "M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H "
         "jena_weather/D bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application "
         "bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H").split()
MED_LONG = ("electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T "
            "M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T "
            "bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H").split()
DOMAIN = {"m4": "Econ/Fin", "electricity": "Energy", "ett1": "Energy", "ett2": "Energy", "solar": "Energy", "hospital": "Healthcare",
          "covid_deaths": "Healthcare", "us_births": "Healthcare", "saugeenday": "Nature", "temperature_rain_with_missing": "Nature",
          "kdd_cup_2018_with_missing": "Nature", "jena_weather": "Nature", "car_parts_with_missing": "Sales", "restaurant": "Sales",
          "hierarchical_sales": "Sales", "LOOP_SEATTLE": "Transport", "SZ_TAXI": "Transport", "M_DENSE": "Transport",
          "bitbrains_fast_storage": "Web/CloudOps", "bitbrains_rnd": "Web/CloudOps", "bizitobs_application": "Web/CloudOps",
          "bizitobs_service": "Web/CloudOps", "bizitobs_l2c": "Web/CloudOps"}
# small configs that exercise every code path (multivariate, missing labels, 10S, M4 single window) in seconds
SMOKE = ["m4_weekly__W__short", "hospital__M__short", "us_births__D__short", "car_parts_with_missing__M__short",
         "bizitobs_l2c__H__short", "solar__W__short", "ett1__W__short"]


def domain_of(dataset: str) -> str:
    base = dataset.split("/")[0]
    return DOMAIN["m4"] if base.startswith("m4_") else DOMAIN[base]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="materialized root containing gifteval/")
    ap.add_argument("--out", default=str(Path(__file__).with_name("split.json")))
    ap.add_argument("--smoke", default=str(Path(__file__).with_name("split_smoke.json")))
    args = ap.parse_args()
    root = Path(args.data) / "gifteval"
    configs = []
    for ds in SHORT:
        for term in ("short", "medium", "long"):
            if term != "short" and ds not in MED_LONG:
                continue
            c = ts_io.config_table(root, ds, term)
            c["domain"] = domain_of(ds)
            c["gifteval_subdir"] = "gifteval"
            configs.append(c)
            print(f"  {c['leaderboard_key']:44s} L={c['prediction_length']:4d} W={c['windows']:2d} m={c['seasonality']:4d} "
                  f"series={c['n_series']:6d} D={c['n_variates']:2d} test={c['n_test_instances']:7d}", file=sys.stderr)
    assert len(configs) == 97, len(configs)
    assert len({c["config_id"] for c in configs}) == 97
    man = {"schema": "timeseries-forecasting-split/v1", "contract": ts_io.CONTRACT, "context_cap": ts_io.CONTEXT_CAP,
           "protocol": "gift-eval src/gift_eval/data.py @ db743860: rolling non-overlapping windows over the last 10% of each series "
                       "(at most 20; M4 one), horizon = term multiplier x base horizon of the frequency; valid = the window before the test region",
           "n_configs": 97, "n_test_instances": sum(c["n_test_instances"] for c in configs),
           "n_valid_instances": sum(c["n_valid_instances"] for c in configs), "configs": configs}
    Path(args.out).write_text(json.dumps(man, indent=1) + "\n")
    smoke = {**man, "warning": "smoke subset for tests and dry runs; rewards from it are not the task's reward",
             "configs": [c for c in configs if c["config_id"] in SMOKE]}
    smoke["n_configs"] = len(smoke["configs"])
    smoke["n_test_instances"] = sum(c["n_test_instances"] for c in smoke["configs"])
    smoke["n_valid_instances"] = sum(c["n_valid_instances"] for c in smoke["configs"])
    Path(args.smoke).write_text(json.dumps(smoke, indent=1) + "\n")
    print(f"wrote {args.out} ({man['n_test_instances']} test instances) and {args.smoke} ({smoke['n_configs']} configs)", file=sys.stderr)


if __name__ == "__main__":
    main()
