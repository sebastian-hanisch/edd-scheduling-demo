"""Vehikel A (Neutral) und Vehikel B (Werkstatt/Logistik): Erzeugung, Determinismus; Auswertung: Kennzahlen,
Sweeps, Optimalitäts- und Timing-Messreihe, Vehikel-B-Härtetest (Schichten)."""

from dataclasses import replace

import numpy as np
import pytest

import edd_constants as C
import edd_evaluation as ev
import edd_scenario as S
import edd_scenario_logistik as SL


# --- Vehikel A ----------------------------------------------------------------------------------------------------------------------------------


def test_instance_shape_and_bounds():
    inst = S.generate(20, 3)
    assert inst.n == 20 and inst.p.shape == (20,) and inst.d.shape == (20,) and inst.w.shape == (20,)
    assert inst.p.min() >= C.P_MIN and inst.p.max() <= C.P_MAX
    assert (inst.d >= inst.p).all()
    assert set(inst.w.tolist()) <= set(C.WEIGHT_VALUES)


def test_instance_is_deterministic_and_seed_dependent():
    a, b, c = S.generate(30, 5), S.generate(30, 5), S.generate(30, 6)
    assert np.array_equal(a.p, b.p) and np.array_equal(a.d, b.d)
    assert not np.array_equal(a.d, c.d)


def test_due_date_matches_the_potts_van_wassenhove_tf_rdd_scheme():
    """Formel per WebSearch verifiziert: dⱼ ~ U(P·(1-TF-RDD/2), P·(1-TF+RDD/2))."""
    inst = S.generate(200, 9, tf=0.4, rdd=0.6)
    total = int(inst.p.sum())
    lo, hi = total * (1 - 0.4 - 0.3), total * (1 - 0.4 + 0.3)
    unclamped = inst.d[inst.d > inst.p]                          # ohne die am unteren Rand geklemmten Werte
    assert unclamped.min() >= lo - 1 and unclamped.max() <= hi + 1


# --- Vehikel B ------------------------------------------------------------------------------------------------------------------------------


def test_logistik_instance_shares_the_same_processing_times_and_due_dates_as_neutral():
    neutral = S.generate(20, 7)
    logistik = SL.generate(20, 7)
    assert np.array_equal(neutral.p, logistik.p) and np.array_equal(neutral.d, logistik.d)


def test_logistik_instance_is_deterministic():
    a, b = SL.generate(10, 2), SL.generate(10, 2)
    assert np.array_equal(a.family, b.family) and np.array_equal(a.setup, b.setup) and a.shift_length == b.shift_length


# --- Analyse --------------------------------------------------------------------------------------------------------------------------------


def test_analysis_fields_are_consistent():
    a = ev.analyse(ev.Settings(n=20))
    assert a.edd.lmax <= a.spt.lmax + 1e-6
    assert a.gap_spt >= -1e-6 and a.gap_random >= -1e-6
    assert a.optimal is None


def test_analysis_matches_the_optimum_for_small_n():
    a = ev.analyse(ev.Settings(n=6))
    assert a.optimal is not None
    assert a.edd_matches_optimum


def test_analysis_is_deterministic_given_the_chain_seed():
    s = ev.Settings(n=20, seed=1, chain_seed=0)
    a, b, c = ev.analyse(s), ev.analyse(s), ev.analyse(replace(s, chain_seed=1))
    assert a.gap_random == pytest.approx(b.gap_random)
    assert a.gap_random != pytest.approx(c.gap_random)


# --- Sweeps und Messreihe -------------------------------------------------------------------------------------------------------------------


def test_run_config_counts_runs_and_aggregates():
    r = ev.run_config(ev.Settings(n=15))
    assert r["n_runs"] == len(C.SWEEP_SEEDS) * C.SWEEP_CHAINS
    assert r["gap_spt"] >= 0.0


def test_sweep_values_labels_and_ordering():
    assert set(ev.SWEEP_VALUES) == set(ev.SWEEP_LABELS)
    rows = ev.sweep("n", ev.Settings(), (5, 40))
    assert [r["value"] for r in rows] == [5, 40]


def test_optimality_check_always_matches():
    rows = ev.optimality_check(ns=(3, 4, 5), seeds=C.SWEEP_SEEDS)
    assert all(r["match_rate"] == 1.0 for r in rows)


def test_timing_sweep_shows_brute_force_growing_far_faster_than_edd():
    rows = ev.timing_sweep(ns=(4, 8))
    small, large = rows[0], rows[1]
    assert large["brute_force_seconds"] > small["brute_force_seconds"] * 10
    assert large["edd_seconds"] < large["brute_force_seconds"] / 100


def test_shift_gap_is_zero_when_the_shift_never_binds():
    row = ev.shift_gap(n=6, shift_length=C.NO_SHIFT_EFFECT_LENGTH)
    assert row["gap_mean"] == pytest.approx(0.0, abs=1e-6)


def test_shift_gap_can_be_positive_for_short_shifts():
    row = ev.shift_gap(n=8, shift_length=160)
    assert row["gap_mean"] > 0.0
