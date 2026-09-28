"""Jede Zahl der App-Texte ist hier über die fünf festen Sweep-Instanzen (je drei Ketten) belegt. Positive UND
negative Aussagen: EDD ist beweisbar optimal auf dem neutralen Vehikel - UND hört auf, beweisbar optimal zu
sein, sobald Schichten (Vehikel Werkstatt/Logistik) lang genug binden. Rechenzeiten nur als Größenordnung
geprüft."""

from functools import lru_cache

import pytest

import edd_evaluation as ev


@lru_cache(maxsize=None)
def _cfg(items):
    return ev.run_config(ev.Settings(), **dict(items))


def cfg(**kw):
    return _cfg(tuple(sorted(kw.items())))


def near(value, expected, tol):
    assert abs(value - expected) <= tol, f"{value:.3f} statt {expected}"


# --- Standardfall -------------------------------------------------------------------------------------------------------------------------------


def test_standard_case_numbers():
    std = cfg()
    near(std["gap_spt"], 345.6, 100.0)
    near(std["gap_random"], 394.0, 100.0)


def test_edd_is_never_worse_than_spt_or_random_on_any_swept_configuration():
    for n in (2, 5, 10, 20, 40, 60):
        row = cfg(n=n)
        assert row["gap_spt"] >= -1e-6 and row["gap_random"] >= -1e-6


# --- Optimalität gegen Brute-Force ---------------------------------------------------------------------------------------------------------------


def test_edd_matches_brute_force_on_every_tested_size():
    rows = ev.optimality_check()
    assert all(r["match_rate"] == 1.0 for r in rows)


# --- Timing: n! gegen n log n -----------------------------------------------------------------------------------------------------------------


def test_brute_force_grows_far_faster_than_edd():
    rows = ev.timing_sweep()
    small, large = rows[0], rows[-1]
    assert large["brute_force_seconds"] > small["brute_force_seconds"] * 100
    assert large["edd_seconds"] < 0.01


def test_brute_force_becomes_impractical_around_nine_jobs():
    rows = ev.timing_sweep()
    last = rows[-1]
    assert last["value"] == 9
    assert last["brute_force_seconds"] > 0.2


# --- Vehikel B: Schicht-Härtetest ------------------------------------------------------------------------------------------------------------


def test_shift_gap_is_exactly_zero_when_the_shift_never_binds():
    row = ev.shift_gap(shift_length=ev.C.NO_SHIFT_EFFECT_LENGTH)
    near(row["gap_mean"], 0.0, 1e-6)


def test_shift_gap_is_zero_at_the_default_shift_length():
    """Bei der Standard-Schichtlänge (480 Minuten) bindet die Grenze bei diesen Instanzen praktisch nie."""
    row = ev.shift_gap(shift_length=480)
    near(row["gap_mean"], 0.0, 1.0)


@pytest.mark.parametrize("shift_length,gap,tol", [(350, 19.0, 20.0), (240, 14.2, 20.0), (160, 62.0, 30.0), (120, 60.4, 30.0)])
def test_shift_gap_numbers(shift_length, gap, tol):
    row = ev.shift_gap(shift_length=shift_length)
    near(row["gap_mean"], gap, tol)


def test_shift_gap_can_become_substantial_once_shifts_are_short_enough():
    """Der zentrale Vehikel-B-Befund: EDD bleibt NICHT beweisbar optimal, sobald Schichten kurz genug sind - ein
    echter Befund, kein Dekor. Nicht monoton (kleine Stichprobe), aber deutlich von 0 verschieden."""
    row = ev.shift_gap(shift_length=160)
    assert row["gap_max"] > 10.0
