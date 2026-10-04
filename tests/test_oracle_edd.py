"""Unabhängiges Orakel für EDD (1||Lmax): Teilmengen-DP statt Permutationsaufzählung für das Optimum, und eine
Minuten-Kalender-Simulation statt der Schichtformel für die Schicht-Variante."""

import itertools
import random

import numpy as np

import edd_algorithm as A


def _dp_lmax(p, d):
    """Optimales Lmax per Teilmengen-DP: best[S] = min_j max(best[S ohne j], sum(S) - d_j)."""
    n = len(p)
    inf = float("inf")
    best = [inf] * (1 << n)
    best[0] = -inf
    total = [0] * (1 << n)
    for m in range(1, 1 << n):
        total[m] = total[m & (m - 1)] + p[(m & -m).bit_length() - 1]
    for m in range(1, 1 << n):
        best[m] = min(max(best[m ^ (1 << j)], total[m] - d[j]) for j in range(n) if m >> j & 1)
    return best[-1]


def _calendar_lmax(p, d, shift, order):
    """Auftrag startet frühestens zur Zeit t und so, dass er vollständig in EINE Schicht [kL,(k+1)L] passt."""
    t, lmax = 0, -10**9
    for j in order:
        s = t
        while s + int(p[j]) > (s // shift + 1) * shift:
            s = (s // shift + 1) * shift
        t = s + int(p[j])
        lmax = max(lmax, t - int(d[j]))
    return lmax


def test_edd_equals_subset_dp_optimum_including_ties():
    rng = random.Random(1)
    for _ in range(250):
        n = rng.randint(1, 8)
        tie = rng.random() < 0.4
        p = [rng.randint(1, 4 if tie else 100) for _ in range(n)]
        d = [rng.randint(1, 6 if tie else 300) for _ in range(n)]
        assert A.edd(np.array(p), np.array(d)).lmax == _dp_lmax(p, d)


def test_shift_evaluation_and_optimum_match_calendar_simulation():
    rng = random.Random(2)
    for _ in range(120):
        n = rng.randint(1, 5)
        shift = rng.choice([101, 120, 200, 350, 480])
        p = [rng.randint(1, 100) for _ in range(n)]
        d = [rng.randint(1, 400) for _ in range(n)]
        order = A.edd_order(np.array(d))
        assert A.evaluate_order_with_shifts(np.array(p), np.array(d), shift, order).lmax == _calendar_lmax(p, d, shift, order)
        best = min(_calendar_lmax(p, d, shift, perm) for perm in itertools.permutations(range(n)))
        assert A.brute_force_optimal_with_shifts(np.array(p), np.array(d), shift).lmax == best
