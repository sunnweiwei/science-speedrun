"""SSI V3 task scores and scientific components. All identifiers refer to the frozen task rosters."""
import json
import math
from pathlib import Path
from statistics import mean

INPUTS = json.loads(Path(__file__).with_name("ssi_index_v3_inputs.json").read_text())
LANES = ("astra", "claude")
ATOM_ERRORS = {"phonon_avg_freq_MAE_K":419.2277237301322, "bulk_modulus_MAE_GPa":107.020216,
    "shear_modulus_MAE_GPa":54.186936, "kappa_SRME":2., "mof_bulk_modulus_MAPE_pct":100.,
    "amorphous_energy_RelRMSE_pct":100., "amorphous_force_RelRMSE_pct":100.,
    "cleavage_relMAE_pct":100., "neb_barrier_MAE_eV":.6803895877435844,
    **{k: float(v) for k, v in INPUTS.get("atom_zero_force_anchors", {}).items()}}
TRACK_GROUPS = {"Open chromatin":("atac","dnase"), "Transcription":("procap","cage","rna_seq"),
    "Protein binding and histones":("chip_tf","chip_histone"), "Splicing":("splice_sites","splice_usage","junctions"),
    "Chromatin contacts":("contact_maps",)}
SEQ_K = {"solubility":2,"subcellular_localization":10,"binary_localization":2,"fold":136,"yeast_ppi":2,"human_ppi":2}
SEQ_GROUPS = {"Function":("fluorescence","stability","beta_lactamase","solubility"),
    "Localization":("subcellular_localization","binary_localization"), "Structure":("fold","secondary_structure"),
    "Protein interactions":("yeast_ppi","human_ppi","ppi_affinity")}
FOLD_GROUPS = {"Protein monomers":("monomer_protein",),"Nucleic-acid monomers":("monomer_rna","monomer_dna"),
    "Protein interfaces":("interface_protein_protein","interface_antibody_antigen","interface_protein_peptide"),
    "Protein–nucleic acid":("interface_protein_rna","interface_protein_dna"),"Protein–ligand":("interface_protein_ligand",)}

def finite(x):
    x=float(x)
    if not math.isfinite(x):
        raise ValueError("Nonfinite V3 score input")
    return x

def clip01(x):
    return max(0.,min(1.,finite(x)))

def measurement(raw, components=None, diagnostics=None):
    return {"score":clip01(raw),"signed_score":finite(raw),"components":components or {},"diagnostics":diagnostics or {}}

def grouped(values,groups):
    return {group:mean(finite(values[k]) for k in names) for group,names in groups.items()}

def atom(rew):
    base=2*.1648/(1+.1648)
    v={k:1-finite(rew[k])/d for k,d in ATOM_ERRORS.items()}
    terms={"Discovery":(finite(rew["discovery_F1"])-base)/(1-base),"Phonons":v["phonon_avg_freq_MAE_K"],
        "Elasticity":mean((v["bulk_modulus_MAE_GPa"],v["shear_modulus_MAE_GPa"])),"Thermal conductivity":v["kappa_SRME"],
        "Metal–organic frameworks":v["mof_bulk_modulus_MAPE_pct"],
        "Amorphous materials":mean((v["amorphous_energy_RelRMSE_pct"],v["amorphous_force_RelRMSE_pct"])),
        "Cleavage":v["cleavage_relMAE_pct"],"Migration barriers":v["neb_barrier_MAE_eV"]}
    return measurement(mean(terms.values()),terms,{k:finite(rew[k]) for k in ("discovery_F1",*ATOM_ERRORS)})

def earth(rew):
    e=finite(rew["lfmc_test_mae"])
    if "awf_test_accuracy" in rew:
        a=finite(rew["awf_test_accuracy"])
        p_maj=finite(rew.get("awf_test_majority_fraction",59/205))
        diag={"AWF spatial test accuracy":a,"LFMC test MAE":e}
    else:
        a=finite(rew["awf_validation_accuracy"])
        p_maj=116/344
        diag={"AWF validation accuracy":a,"LFMC test MAE":e}
    terms={"AWF classification skill":(a-p_maj)/(1-p_maj),"LFMC regression skill":1-e/24.435180349813233}
    return measurement(mean(terms.values()),terms,diag)

def flow(rew,failed):
    groups={};gains={}
    for case,meta in INPUTS["flow_cases"].items():
        value=rew.get(case+"/"+meta["metric"]);z=meta["zero"]
        if case in failed or value is None or not math.isfinite(float(value)):
            gain=-1.
        else:
            gain=(float(value)-z)/abs(z)
            if meta["direction"]=="lower_is_better":gain=-gain
            gain=max(-1.,min(1.,gain))
        gains[case]=gain
        groups.setdefault(case.split("_")[0],[]).append(gain)
    if len(groups)!=10 or len(gains)!=26:
        raise ValueError("Unexpected fluid case roster")
    terms={k:mean(v) for k,v in groups.items()}
    return measurement(mean(terms.values()),terms,gains)

def genome(families,variants):
    modalities=grouped(families,TRACK_GROUPS)
    v={}
    for k,b in INPUTS["genome_constant"].items():
        if k!="eqtl_sign":
            v[k]=(finite(variants[k])-b)/(1-b)
    branches={"eQTL":mean((v["eqtl_causality"],v["eqtl_coefficient"])),
        "Other molecular QTL":mean(v[k] for k in ("sqtl_causality","caqtl","bqtl_spi1")),
        "CRISPRi enhancer–gene":v["enhancer_gene"],
        "Clinical and functional assays":mean((mean(v[k] for k in ("clinvar_noncoding","clinvar_splice_region","clinvar_missense")),v["mfass"])),
        "Human traits":mean((v["traitgym_mendelian"],v["traitgym_complex"]))}
    axes={"Functional genomic tracks":mean(modalities.values()),"Variant and perturbation effects":mean(branches.values())}
    return measurement(mean(axes.values()),{**axes,**modalities,**branches},
                       {**{f"Pearson / {k}":finite(n) for k,n in families.items()},**{f"Variant / {k}":finite(n) for k,n in variants.items() if k in v}})

def molecule(rew):
    groups={k:finite(rew[k+"_ma_rae"]) for k in ("qm9","compas1d","expansionrx","asap")}
    return measurement(1/(1+mean(groups.values())),{k+" MA-RAE":x for k,x in groups.items()})

def design(rates):
    rates={k:finite(rates[k]) for k in ("unconditional","sm_binder","ppi_binder","enzyme")}
    if not all(0<=v<=1 for v in rates.values()):raise ValueError("Invalid protein design pass rate")
    return measurement(rates["unconditional"]/51+50/153*(rates["sm_binder"]+rates["ppi_binder"]+rates["enzyme"]),rates)

def folding(raw):
    terms=grouped(raw,FOLD_GROUPS)
    return measurement(mean(terms.values()),terms,raw)

def sequence(tasks):
    v={};native={}
    for names in SEQ_GROUPS.values():
        for k in names:
            metric="balanced_accuracy" if k in SEQ_K else "accuracy" if k=="secondary_structure" else "rmse" if k=="ppi_affinity" else "spearman"
            x=finite(tasks[k][metric]);native[k+" / "+metric]=x
            if k in SEQ_K:v[k]=(x-1/SEQ_K[k])/(1-1/SEQ_K[k])
            elif k=="secondary_structure":v[k]=(x-.4306464618)/(1-.4306464618)
            elif k=="ppi_affinity":v[k]=1-x/3.083206134
            else:v[k]=x
    terms=grouped(v,SEQ_GROUPS)
    return measurement(mean(terms.values()),terms,native)

def time_series(ratios):
    roster=INPUTS["timeseries_roster"]
    if set(ratios)!=set(roster) or len(roster)!=97:raise ValueError("Expected the frozen 97 unique leaderboard keys")
    tree={}
    for key,(m,q) in ratios.items():
        domain,source,horizon,*_=roster[key]
        pair=[max(.001,min(10,finite(x))) for x in (m,q)]
        tree.setdefault(domain,{}).setdefault(source,{}).setdefault(horizon,[]).append(mean(math.log(x) for x in pair))
    if len(tree)!=7 or sum(map(len,tree.values()))!=23:raise ValueError("Changed GIFT-Eval hierarchy")
    domains={};sources={}
    for d,src in tree.items():
        by_source={s:mean(mean(values) for values in hs.values()) for s,hs in src.items()}
        domains[d]=mean(by_source.values());sources.update({d+" / "+s:1-math.exp(v) for s,v in by_source.items()})
    return measurement(1-math.exp(mean(domains.values())),{d:1-math.exp(v) for d,v in domains.items()},sources)

def forecast_rows(rows):
    ratios={};roster=INPUTS["timeseries_roster"]
    for r in rows:
        k=r["leaderboard_key"]
        if k in ratios or k not in roster:raise ValueError(f"Unknown or duplicate forecast configuration {k}")
        if r.get("status")=="failed":
            pair=(10.,10.)
        else:
            raw=r.get("metrics")
            has_metrics=isinstance(raw,dict) and ("mase" in raw or "task_mase" in raw) and ("wql" in raw or "task_wql" in raw)
            has_top_raw=("mase" in r or "task_mase" in r) and ("wql" in r or "task_wql" in r)
            if has_metrics or has_top_raw:
                _,_,_,m,q=roster[k];src=raw if has_metrics else r
                pair=(finite(src.get("mase",src.get("task_mase")))/m,finite(src.get("wql",src.get("task_wql")))/q)
                needs_partial_blend=True
            elif "rel_mase" in r:
                pair=(finite(r["rel_mase"]),finite(r["rel_wql"]))
                needs_partial_blend=True
            elif "relative" in r:
                pair=(finite(r["relative"]["mase"]),finite(r["relative"]["wql"]))
                needs_partial_blend=False
            else:
                raise ValueError(f"Missing forecast metric fields for {k}")
            if needs_partial_blend and r.get("status")=="partial":
                fraction=finite(r["coverage"]["fraction"])
                if not 0<=fraction<=1:raise ValueError("Invalid forecast coverage")
                pair=tuple(math.exp(fraction*math.log(max(.001,min(10.,x)))+(1.-fraction)*math.log(10.)) for x in pair)
        ratios[k]=pair
    return time_series(ratios)

BCI_CHANCE = {"tuev": 1.0 / 6.0, "ad65": 1.0 / 3.0, "openbmi": 0.5, "longitudinal": 0.5}

def bci(rew):
    terms = {}
    native = {}
    for k, c in BCI_CHANCE.items():
        val = finite(rew[k])
        native[k + " balanced accuracy"] = val
        terms[k] = (val - c) / (1.0 - c)
    return measurement(mean(terms.values()), terms, native)

def pde(groups):
    expected = ("ape2d", "pdearena", "well/active_matter", "well/turbulent_radiative_layer_2D")
    terms = {}
    for g in expected:
        val = groups[g]["reward"] if isinstance(groups[g], dict) and "reward" in groups[g] else groups[g]
        terms[g] = finite(val)
    return measurement(mean(terms.values()), terms)

def weather(rew):
    by_var = rew.get("normalized_crps_by_variable") or {}
    if by_var:
        norm_crps = mean(finite(v) for v in by_var.values())
        terms = {k: 1.0 / (1.0 + finite(v)) for k, v in by_var.items()}
        diag = {**{k + " normalized CRPS": finite(v) for k, v in by_var.items()}, "Mean normalized CRPS": norm_crps}
    else:
        norm_crps = finite(rew["normalized_crps"])
        terms = {"10-variable panel": 1.0 / (1.0 + norm_crps)}
        diag = {"Mean normalized CRPS": norm_crps}
    return measurement(1.0 / (1.0 + norm_crps), terms, diag)

def tokamak(rew, expected_episodes=105):
    if "successful_episodes" in rew and "scheduled_episodes" in rew:
        succ = int(rew["successful_episodes"])
        sched = int(rew["scheduled_episodes"])
        if sched != expected_episodes or not 0 <= succ <= sched:
            raise ValueError(f"Expected {expected_episodes} scheduled tokamak episodes, got {sched}")
        return measurement(succ / sched, {}, {"Successful episodes": succ, "Scheduled episodes": sched, "Void episodes": int(rew.get("void_episodes", 0))})
    per_case = {}
    per_seed = expected_episodes // 7
    total_succ = 0
    total_void = 0
    for k, v in rew.items():
        if k.endswith("/success_fraction"):
            case = k.split("/")[0]
            void = int(rew.get(case + "/void_episodes", 0))
            valid_ep = per_seed - void
            succ = int(round(finite(v) * valid_ep))
            per_case[case + " joint success"] = succ / per_seed
            total_succ += succ
            total_void += void
    if len(per_case) != 7:
        raise ValueError("Expected all 7 tokamak scenarios")
    return measurement(total_succ / expected_episodes, per_case,
                       {"Successful episodes": total_succ, "Scheduled episodes": expected_episodes, "Void episodes": total_void})

def massspec(rew):
    hit1 = finite(rew.get("hit_rate_at_1", rew.get("score", rew.get("reward"))))
    return measurement(hit1, {"Hit@1": hit1})

def disease(endpoints):
    if len(endpoints) != 17:
        raise ValueError(f"Expected 17 eligible clinical endpoints, got {len(endpoints)}")
    terms = {k: finite(v) for k, v in endpoints.items()}
    return measurement(mean(terms.values()), terms)

def surface_code(failure_by_distance):
    terms = {f"{k} logical success": 1.0 - finite(v) for k, v in failure_by_distance.items()}
    if len(terms) != 2:
        raise ValueError("Expected code distances 3 and 5")
    return measurement(mean(terms.values()), terms)

def drug_screening(benchmarks):
    dude = finite(benchmarks["dude"]["mean_bedroc_80.5"] if isinstance(benchmarks["dude"], dict) else benchmarks["dude"])
    lit = finite(benchmarks["lit_pcba"]["mean_bedroc_80.5"] if isinstance(benchmarks["lit_pcba"], dict) else benchmarks["lit_pcba"])
    terms = {"DUD-E BEDROC": dude, "LIT-PCBA BEDROC": lit}
    return measurement(0.5 * (dude + lit), terms)

def crystal(counts):
    yields=(counts["sun"],counts["metasun"])
    if counts["expected"]!=2500 or any(type(v)!=int or v<0 for v in yields) or sum(yields)>2500:
        raise ValueError("Invalid crystal attempt counts")
    a=counts["sun"]/2500;b=(counts["sun"]+counts["metasun"])/2500
    numbers={k:v for k,v in counts.items() if type(v) in (int,float)}
    return measurement((a+b)/2,{"Strict SUN yield":a,"Stable or metastable SUN yield":b},numbers)

def cell(contexts):
    if set(contexts)!={"near","mid","far"}:raise ValueError("Expected all three perturbation contexts")
    terms={k:finite(contexts[k]["context_score"]) for k in contexts}
    metrics={k+" / "+m:v[m] for k,v in contexts.items() for m in
             ("WMSE","amplitude_skill","NIR","repeat_NIR","identification_skill","signed_context","targets") if m in v}
    return measurement(mean(terms.values()),terms,metrics)
