"""EDD (Earliest Due Date first) für 1||Lmax: n Aufträge auf einer Maschine, Ziel ist die maximale Verspätung
(Lateness) $L_{\\max} = \\max_j (C_j - d_j)$ zu minimieren - EDD (aufsteigend nach Fälligkeit sortieren) ist dafür
beweisbar optimal (Jackson 1955), mit demselben Beweismuster wie SPT für $\\sum C_j$: ein Vertauschungsargument.
Tauscht man zwei benachbarte Aufträge $i$ vor $j$ mit $d_i > d_j$, kann sich die maximale Verspätung der beiden
nur verringern oder gleich bleiben - jede nicht EDD-sortierte Reihenfolge lässt sich also nicht verschlechtern
und ist mindestens ebenso gut nach der Vertauschung; EDD ist ein Optimum.

Hier zusätzlich: Brute-Force-Vollaufzählung als unabhängige Gegenprobe, sowie die Schicht-Variante für Vehikel B
(Werkstatt/Logistik) - EDD kennt keine Lücken in der Maschinenverfügbarkeit, ob es trotzdem optimal bleibt, ist
Gegenstand der Messreihe, nicht vorab behauptet."""

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass
class Result:
    order: np.ndarray
    completion: np.ndarray
    lateness: np.ndarray
    lmax: float


def completion_times(p, order):
    return np.cumsum(np.asarray(p)[order].astype(np.float64))


def lateness_of(completion, d, order):
    return completion - np.asarray(d)[order].astype(np.float64)


def evaluate_order(p, d, order):
    order = np.asarray(order)
    completion = completion_times(p, order)
    lateness = lateness_of(completion, d, order)
    return Result(order, completion, lateness, float(lateness.max()))


def edd_order(d):
    """EDD: aufsteigend nach Fälligkeit; stabile Sortierung, damit Gleichstände reproduzierbar sind."""
    return np.argsort(d, kind="stable")


def edd(p, d):
    return evaluate_order(p, d, edd_order(d))


def spt_order(p):
    """SPT - die richtige Regel für ΣCⱼ, aber die FALSCHE für Lmax; als Kontrast."""
    return np.argsort(p, kind="stable")


def random_order(n, rng):
    order = np.arange(n)
    rng.shuffle(order)
    return order


def brute_force_optimal(p, d):
    n = len(p)
    best_order, best_lmax = None, np.inf
    for perm in itertools.permutations(range(n)):
        order = np.array(perm)
        lmax = evaluate_order(p, d, order).lmax
        if lmax < best_lmax:
            best_lmax, best_order = lmax, order
    return evaluate_order(p, d, best_order)


# --- Mit Schichten (Vehikel B: Werkstatt/Logistik) ------------------------------------------------------------


def completion_times_with_shifts(p, shift_length, order):
    """Wie completion_times, aber die Maschine ist nur innerhalb einer Schicht verfügbar: passt ein Auftrag
    nicht mehr in die laufende Schicht, rutscht der Start auf den Beginn der nächsten (eine Lücke)."""
    t = 0.0
    out = np.empty(len(order), dtype=np.float64)
    for idx, j in enumerate(order):
        shift_end = (np.floor(t / shift_length) + 1) * shift_length
        if t + float(p[j]) > shift_end + 1e-9:
            t = shift_end
        t += float(p[j])
        out[idx] = t
    return out


def evaluate_order_with_shifts(p, d, shift_length, order):
    order = np.asarray(order)
    completion = completion_times_with_shifts(p, shift_length, order)
    lateness = lateness_of(completion, d, order)
    return Result(order, completion, lateness, float(lateness.max()))


def brute_force_optimal_with_shifts(p, d, shift_length):
    n = len(p)
    best_order, best_lmax = None, np.inf
    for perm in itertools.permutations(range(n)):
        order = np.array(perm)
        lmax = evaluate_order_with_shifts(p, d, shift_length, order).lmax
        if lmax < best_lmax:
            best_lmax, best_order = lmax, order
    return evaluate_order_with_shifts(p, d, shift_length, best_order)
