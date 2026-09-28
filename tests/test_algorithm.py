"""edd_algorithm: EDD-Optimalität für Lmax gegen unabhängige Brute-Force-Vollaufzählung (Jackson 1955,
Vertauschungsargument empirisch geprüft), Regressionsschutz, Determinismus, Schicht-Variante."""

import itertools

import numpy as np
import pytest

import edd_algorithm as A


def _instance(seed, n):
    rng = np.random.default_rng(seed)
    p = rng.integers(1, 100, size=n).astype(np.int64)
    total = int(p.sum())
    d = np.maximum(np.round(total * (0.2 + 0.6 * rng.random(n))).astype(np.int64), p)
    return p, d


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7])
def test_edd_matches_brute_force_for_every_seed(n):
    for seed in range(10):
        p, d = _instance(seed, n)
        assert A.edd(p, d).lmax == pytest.approx(A.brute_force_optimal(p, d).lmax)


def test_edd_is_the_unique_optimum_up_to_ties():
    p, d = _instance(7, 6)
    edd_lmax = A.edd(p, d).lmax
    all_lmax = [A.evaluate_order(p, d, perm).lmax for perm in itertools.permutations(range(6))]
    assert edd_lmax == pytest.approx(min(all_lmax))


def test_completion_and_lateness_on_a_hand_picked_instance():
    """Handrechnung: drei Aufträge, EDD-Reihenfolge und Lmax von Hand nachgerechnet."""
    p = np.array([3, 1, 2])
    d = np.array([10, 2, 4])
    result = A.edd(p, d)
    assert result.order.tolist() == [1, 2, 0]                 # Fälligkeit 2, 4, 10
    assert result.completion.tolist() == [1, 3, 6]
    assert result.lateness.tolist() == [1 - 2, 3 - 4, 6 - 10]
    assert result.lmax == pytest.approx(-1.0)                 # max(-1, -1, -4) = -1


def test_edd_order_is_ascending_by_due_date():
    d = np.array([5, 2, 8, 1, 2])
    order = A.edd_order(d)
    assert d[order].tolist() == sorted(d.tolist())


def test_spt_is_a_worse_rule_for_lmax_than_edd_on_a_constructed_instance():
    """SPT optimiert ΣCⱼ, nicht Lmax - hier absichtlich ein Fall, in dem SPT für Lmax schlechter ist."""
    p = np.array([1, 1, 10])
    d = np.array([100, 100, 5])                                 # der lange Auftrag hat die knappe Frist
    edd_result = A.edd(p, d)
    spt_result = A.evaluate_order(p, d, A.spt_order(p))
    assert edd_result.lmax < spt_result.lmax


def test_random_order_is_deterministic_given_the_rng_state():
    n = 8
    a = A.random_order(n, np.random.default_rng(0))
    b = A.random_order(n, np.random.default_rng(0))
    assert a.tolist() == b.tolist()
    assert sorted(a.tolist()) == list(range(n))


# --- Mit Schichten (Vehikel B) --------------------------------------------------------------------------------


def test_shift_variant_matches_the_plain_variant_when_the_shift_never_binds():
    p, d = _instance(11, 6)
    huge_shift = 10_000_000
    plain = A.edd(p, d)
    with_shift = A.evaluate_order_with_shifts(p, d, huge_shift, plain.order)
    assert with_shift.completion.tolist() == plain.completion.tolist()
    assert with_shift.lmax == pytest.approx(plain.lmax)


def test_a_job_that_does_not_fit_the_shift_moves_to_the_next_one():
    p = np.array([6, 6])
    d = np.array([100, 100])
    shift_length = 8
    order = np.array([0, 1])
    result = A.evaluate_order_with_shifts(p, d, shift_length, order)
    # Auftrag 0: passt in [0,8), endet bei 6. Auftrag 1 (6 Minuten) passt NICHT mehr bis 8 -> rutscht auf Schicht [8,16), endet bei 14.
    assert result.completion.tolist() == [6, 14]


def test_brute_force_with_shifts_matches_independent_full_enumeration():
    p, d = _instance(13, 5)
    shift_length = 40
    best = A.brute_force_optimal_with_shifts(p, d, shift_length)
    all_lmax = [A.evaluate_order_with_shifts(p, d, shift_length, np.array(perm)).lmax for perm in itertools.permutations(range(5))]
    assert best.lmax == pytest.approx(min(all_lmax))


def test_edd_can_be_worse_than_the_true_optimum_once_shifts_bind():
    """EDD ignoriert Schichtgrenzen - das muss nicht mehr optimal bleiben, sobald sie oft genug binden. Genau
    die Frage, die Vehikel B stellt."""
    p = np.array([8, 6, 5, 3])
    d = np.array([9, 2, 3, 1])
    shift_length = 8
    edd_lmax = A.evaluate_order_with_shifts(p, d, shift_length, A.edd_order(d)).lmax
    true_opt = A.brute_force_optimal_with_shifts(p, d, shift_length).lmax
    assert edd_lmax > true_opt + 1e-6
