"""Global low-energy yield, deduplicated once, restricted to negative formation energy.

The pinned upstream MetaSUN evaluator uses a 0.1 eV/atom energy-above-hull gate over the full
eligible pool, then performs uniqueness and novelty checks. Adding separate unique counts from the
stable and metastable bins can count equivalent structures twice, so the pool is deduplicated once.

The negative-formation-energy count is a diagnostic, not a reward: metastable phases with positive
energy relative to equilibrium are real (Sun et al., Sci. Adv. 2016), so it has no basis as a gate.
"""
import ast
import math
import re
from .upstream import final_scores

VERSION='crystal-global-unique-low-energy-yield/v2-candidate'
VERSION_NEGATIVE_FORMATION='crystal-unique-novel-low-energy-negative-formation/v2'


def _count(value, name):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0 or int(value)!=value:
        raise ValueError('Invalid '+name)
    return int(value)


def _structure_counts(document, expected):
    expected=_count(expected,'requested count')
    if expected==0:raise ValueError('Requested count must be positive')
    run_info,validity=document.get('run_info'),document.get('validity_filtering')
    if not isinstance(run_info,dict) or 'n_structures' not in run_info or not isinstance(validity,dict) or 'valid_structures' not in validity:
        raise ValueError('Missing structure counts; an absent count is not zero valid structures')
    loaded=_count(run_info['n_structures'],'loaded count')
    valid=_count(validity['valid_structures'],'valid count')
    if not 0<=valid<=loaded<=expected:raise ValueError('Inconsistent structure counts')
    return expected,loaded,valid


def _accepted_indices(document, valid):
    results=document.get('results',{})
    scores=final_scores(results.get('sun_new',results.get('sun','')))
    if 'metasun_count' not in scores or 'metasun_structure_indices' not in scores:
        raise ValueError('Missing globally deduplicated low-energy result; no legacy sum fallback')
    count=_count(scores['metasun_count'],'accepted count')
    indices=scores['metasun_structure_indices']
    if not isinstance(indices,list) or count!=len(indices):raise ValueError('Accepted count/index mismatch')
    if any(isinstance(i,bool) or not isinstance(i,int) or not 0<=i<valid for i in indices):
        raise ValueError('Invalid accepted index')
    if len(set(indices))!=len(indices):raise ValueError('Duplicate accepted indices')
    if _count(scores.get('total_structures_evaluated'),'evaluated count')!=valid:
        raise ValueError('Incomplete evaluation')
    return indices


def global_unique_low_energy_score(document, expected):
    expected,loaded,valid=_structure_counts(document,expected)
    if valid==0:
        return dict(version=VERSION,reward=0.,requested=expected,loaded=loaded,valid=0,accepted=0)
    count=len(_accepted_indices(document,valid))
    return dict(version=VERSION,reward=count/expected,requested=expected,loaded=loaded,valid=valid,accepted=count,
                energy_threshold_ev_per_atom=.1,
                scope='MLIP proxy for low-energy, globally unique, novel yield; not DFT or synthesizability confirmation.')


def _literal(source):
    """Decode a serialized upstream BenchmarkResult with AST literals only (never eval)."""
    tree=ast.parse(source.replace("defaultdict(<class 'int'>, ",'RecordedDefaultDict('),mode='eval')

    def visit(node):
        if isinstance(node,ast.Constant):return node.value
        if isinstance(node,(ast.List,ast.Tuple)):return [visit(x) for x in node.elts]
        if isinstance(node,ast.Dict):return {visit(k):visit(v) for k,v in zip(node.keys,node.values)}
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):
            value=visit(node.operand);return -value if isinstance(node.op,ast.USub) else value
        if isinstance(node,ast.Name) and node.id in ('nan','inf'):return float(node.id)
        if isinstance(node,ast.Call):
            f=node.func
            if isinstance(f,ast.Name) and f.id=='RecordedDefaultDict':return visit(node.args[0])
            if isinstance(f,ast.Attribute) and isinstance(f.value,ast.Name) and f.value.id=='np':
                return (float if 'float' in f.attr else int)(visit(node.args[0]))
            if isinstance(f,ast.Name) and (f.id in ('BenchmarkResult','MetricResult') or re.fullmatch('[A-Za-z][A-Za-z0-9_]*Config',f.id)):
                return {k.arg:visit(k.value) for k in node.keywords}
        raise ValueError('Unsupported upstream serialization')
    return visit(tree.body)


def formation_energies(document, valid):
    """Per-structure MLIP-ensemble formation energy (eV/atom), indexed like the SUN indices."""
    source=document.get('results',{}).get('stability')
    if not isinstance(source,str):raise ValueError('Missing stability result; no formation-energy fallback')
    stability=_literal(source)
    try:
        rows=stability['evaluator_results']['formation_energy']['metric_results']['formation_energy']['individual_values']
    except (KeyError,TypeError) as error:
        raise ValueError('Missing per-structure formation energies') from error
    values=[row.get('value') if isinstance(row,dict) else row for row in rows]
    if len(values)!=valid:raise ValueError('Formation energies do not cover every valid structure')
    return values


def negative_formation_low_energy_score(document, expected):
    """Unique, novel, E_hull <= 0.1 eV/atom and formation energy < 0, over all requested outputs."""
    expected,loaded,valid=_structure_counts(document,expected)
    base=dict(version=VERSION_NEGATIVE_FORMATION,requested=expected,loaded=loaded,valid=valid,
              energy_threshold_ev_per_atom=.1,formation_energy_threshold_ev_per_atom=0.)
    if valid==0:
        return dict(base,reward=0.,unique_novel_low_energy=0,accepted=0)
    indices=_accepted_indices(document,valid)
    energies=formation_energies(document,valid)
    if any(not isinstance(energies[i],(int,float)) or not math.isfinite(energies[i]) for i in indices):
        raise ValueError('Nonfinite formation energy for an accepted structure')
    accepted=sum(energies[i]<0 for i in indices)
    return dict(base,reward=accepted/expected,unique_novel_low_energy=len(indices),accepted=accepted,
                scope='MLIP proxy: globally unique, novel, within 0.1 eV/atom of the ensemble hull and below the '
                      'elemental reference; not DFT or synthesizability confirmation.')
