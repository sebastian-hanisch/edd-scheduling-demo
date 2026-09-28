"""Auswertung der EDD-Demo: EDD gegen SPT (die richtige Regel für ΣCⱼ, aber die falsche für Lmax) und gegen
zufällige Reihenfolgen, gegen die Brute-Force-Vollaufzählung (nur kleine n), und das Vehikel-B-Experiment (bleibt
EDD optimal, wenn die Maschine wegen Schichten nicht durchgehend verfügbar ist)."""

import time
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

import edd_algorithm as A
import edd_constants as C
import edd_scenario as S


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    seed: int = C.DEFAULT_SEED
    chain_seed: int = 0
    tf: float = C.DEFAULT_TF
    rdd: float = C.DEFAULT_RDD
    vehicle: str = C.DEFAULT_VEHICLE
    shift_length: int = C.DEFAULT_SHIFT_LENGTH


@lru_cache(maxsize=512)
def instance(n, seed, tf=C.DEFAULT_TF, rdd=C.DEFAULT_RDD):
    return S.generate(n, seed, tf=tf, rdd=rdd)


@dataclass
class Analysis:
    settings: Settings
    inst: object
    edd: object
    spt: object
    random_mean: float
    random_runs: int
    optimal: object

    @property
    def gap_spt(self):
        return self.spt.lmax - self.edd.lmax

    @property
    def gap_random(self):
        return self.random_mean - self.edd.lmax

    @property
    def edd_matches_optimum(self):
        return self.optimal is not None and abs(self.edd.lmax - self.optimal.lmax) < 1e-6


def analyse(settings, random_draws=20):
    """Wertet EDD auf dem gewählten Vehikel aus - Neutral (Maschine durchgehend verfügbar) oder Werkstatt/
    Logistik (Schichtgrenzen zählen mit). EDD selbst bleibt in beiden Fällen dieselbe Regel (sortiert nur nach
    Fälligkeit, kennt keine Schichten) - nur die BEWERTUNG der Reihenfolgen (und damit auch der Vollaufzählung)
    wechselt mit dem Vehikel, damit die Haupt-Kennzahlen ehrlich widerspiegeln, was auf dem gewählten Vehikel
    tatsächlich passiert (statt nur in einer Zusatzbox)."""
    inst = instance(settings.n, settings.seed, settings.tf, settings.rdd)
    p, d = inst.p, inst.d

    if settings.vehicle == "logistik":
        def ev(order):
            return A.evaluate_order_with_shifts(p, d, settings.shift_length, order)

        optimal = A.brute_force_optimal_with_shifts(p, d, settings.shift_length) if settings.n <= C.BRUTE_FORCE_MAX_N else None
    else:
        def ev(order):
            return A.evaluate_order(p, d, order)

        optimal = A.brute_force_optimal(p, d) if settings.n <= C.BRUTE_FORCE_MAX_N else None

    edd = ev(A.edd_order(d))
    spt = ev(A.spt_order(p))
    rng = np.random.default_rng(settings.chain_seed)
    random_lmaxs = [ev(A.random_order(settings.n, rng)).lmax for _ in range(random_draws)]
    return Analysis(settings, inst, edd, spt, float(np.mean(random_lmaxs)), random_draws, optimal)


# --- Sweeps und Tabellen -----------------------------------------------------------------------------------------------------------------------


def _mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def run_config(base, seeds=C.SWEEP_SEEDS, chains=C.SWEEP_CHAINS, **changes):
    s0 = replace(base, **changes)
    rows = []
    for seed in seeds:
        for ch in range(chains):
            a = analyse(replace(s0, seed=seed, chain_seed=ch))
            rows.append({"gap_spt": a.gap_spt, "gap_random": a.gap_random})
    out = {k: _mean(rows, k) for k in rows[0]}
    out["n_runs"] = len(rows)
    return out


SWEEP_VALUES = {"n": (2, 5, 10, 20, 40, 60), "tf": (0.0, 0.2, 0.4, 0.6, 0.8), "rdd": (0.2, 0.4, 0.6, 0.8, 1.0)}
SWEEP_LABELS = {"n": "Aufträge", "tf": "Fristen-Anteil (TF)", "rdd": "Fristen-Streuung (RDD)"}


def sweep(param, base=Settings(), values=None):
    values = SWEEP_VALUES[param] if values is None else values
    return [{"value": v, **run_config(base, **{param: v})} for v in values]


def optimality_check(ns=C.BRUTE_FORCE_SWEEP_N, seeds=C.SWEEP_SEEDS):
    rows = []
    for n in ns:
        matches = 0
        for seed in seeds:
            inst = instance(n, seed)
            edd_lmax = A.edd(inst.p, inst.d).lmax
            opt_lmax = A.brute_force_optimal(inst.p, inst.d).lmax
            if abs(edd_lmax - opt_lmax) < 1e-6:
                matches += 1
        rows.append({"value": n, "match_rate": matches / len(seeds)})
    return rows


def timing_sweep(ns=C.BRUTE_FORCE_SWEEP_N, seed=C.DEFAULT_SEED):
    rows = []
    for n in ns:
        inst = instance(n, seed)
        t0 = time.perf_counter()
        A.brute_force_optimal(inst.p, inst.d)
        t_bf = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(100):
            A.edd(inst.p, inst.d)
        t_edd = (time.perf_counter() - t0) / 100
        rows.append({"value": n, "brute_force_seconds": t_bf, "edd_seconds": t_edd})
    return rows


def shift_gap(n=8, seeds=C.SWEEP_SEEDS, shift_length=C.DEFAULT_SHIFT_LENGTH):
    """Vehikel-B-Härtetest: EDD (sortiert nur nach Fälligkeit, ignoriert Schichtgrenzen) gegen die echte
    Optimallösung MIT Schichten (Brute-Force, deshalb kleines n)."""
    gaps = []
    for seed in seeds:
        inst = S.generate(n, seed)
        edd_ord = A.edd_order(inst.d)
        edd_lmax = A.evaluate_order_with_shifts(inst.p, inst.d, shift_length, edd_ord).lmax
        opt_lmax = A.brute_force_optimal_with_shifts(inst.p, inst.d, shift_length).lmax
        gaps.append(edd_lmax - opt_lmax)
    return {"gap_mean": float(np.mean(gaps)), "gap_min": float(np.min(gaps)), "gap_max": float(np.max(gaps)), "n_runs": len(gaps)}


def shift_gap_sweep(shift_lengths=(480, 350, 240, 160, 120), n=8, seeds=C.SWEEP_SEEDS):
    return [{"value": s, **shift_gap(n=n, seeds=seeds, shift_length=s)} for s in shift_lengths]
