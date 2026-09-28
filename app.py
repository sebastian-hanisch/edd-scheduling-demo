"""EDD (Earliest Due Date first) - eine Warteschlange, die die schlimmste Verspätung klein hält - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zweites Stück der Konzepte-Linie "Klassische Scheduling-Theorie": n Aufträge auf einer Maschine, Ziel ist die
maximale Verspätung (Lmax) zu minimieren (1||Lmax). EDD (aufsteigend nach Fälligkeit sortieren) ist dafür
BEWEISBAR optimal (Jackson 1955) - dasselbe Vertauschungsargument-Beweismuster wie SPT für ΣCⱼ (Stück 1), nur
mit einer anderen Zielfunktion. Siehe README für die Einordnung in die Linie.

Lauffähig mit: streamlit run app.py
"""

from dataclasses import replace

import numpy as np
import streamlit as st

import edd_algorithm as A
import edd_constants as C
import edd_scenario_logistik as SL
from edd_evaluation import Settings, SWEEP_LABELS, analyse, instance, optimality_check, run_config, shift_gap, shift_gap_sweep, sweep, timing_sweep
from edd_presets import apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_chain_seed, randomize_seed, sync_query_params
from edd_visualization import build_completion_vs_due, build_schedule, build_shift_gap, build_sweep, build_timing

st.set_page_config(page_title="EDD-Scheduling – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _sweep(param, base):
    return sweep(param, base)


@st.cache_data(show_spinner=False)
def _optimality():
    return optimality_check()


@st.cache_data(show_spinner=False)
def _timing():
    return timing_sweep()


@st.cache_data(show_spinner=False)
def _shift_gap_sweep(n):
    return shift_gap_sweep(n=n)


def _fmt_int(x):
    return f"{int(round(x)):,}".replace(",", ".")


st.title("⏱️ EDD – eine Warteschlange, die die schlimmste Verspätung klein hält")
st.markdown(
    r"""
**n Aufträge auf einer Maschine, jeder mit einer Fälligkeit – gesucht ist die Reihenfolge, die die größte
Verspätung minimiert** ($1||L_{\max}$, mit $L_{\max} = \max_j (C_j - d_j)$). Die Antwort ist **EDD** (Earliest
Due Date first): einfach aufsteigend nach Fälligkeit sortieren – dasselbe Beweismuster wie SPT im ersten Stück
dieser Linie (Vertauschungsargument), nur für eine andere Zielfunktion (Jackson 1955). Tauscht man zwei
benachbarte Aufträge $i$ vor $j$ mit $d_i > d_j$, kann sich die größte Verspätung der beiden nur verringern oder
gleich bleiben – jede nicht EDD-sortierte Reihenfolge lässt sich also nicht verschlechtern; EDD ist ein Optimum.
"""
)
st.caption(
    "Zweites Stück der Konzepte-Linie „Klassische Scheduling-Theorie“ - dieselben zwei Vehikel wie im ersten "
    "Stück ([spt-scheduling-demo](https://sebastianhanisch-spt-scheduling-demo.streamlit.app/)): **Neutral** "
    "(Aufträge mit Bearbeitungszeit und Fälligkeit) und **Werkstatt/Logistik** (dieselben Aufträge, aber die "
    "Maschine ist nur innerhalb fester Schichten verfügbar) - der Umschalter ist in der Seitenleiste."
)

with st.expander("So funktioniert EDD", expanded=True):
    st.markdown(
        r"""
1. **Sortieren.** Alle Aufträge aufsteigend nach Fälligkeit $d_j$ ordnen - fertig. $O(n \log n)$, kein Suchverfahren nötig.
2. **Warum das optimal ist.** Vertauschungsargument: zwei benachbarte Aufträge mit $d_i > d_j$ tauschen verringert die größte Verspätung der beiden (oder lässt sie gleich) - jede nicht sortierte Reihenfolge lässt sich also nicht verschlechtern.
3. **Was gemessen wird.** Der Abstand einer Reihenfolge zu EDD in Minuten (Differenz der maximalen Verspätung); für kleine $n$ zusätzlich die Vollaufzählung als unabhängige Gegenprobe.
4. **Die Grenze der Annahme.** EDD setzt voraus, dass die Maschine durchgehend verfügbar ist. Das Vehikel „Werkstatt/Logistik“ prüft, was passiert, wenn Schichten Lücken hineinbringen.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(C.PRESETS.keys())
cols = st.columns(len(preset_names))
for col, name in zip(cols, preset_names):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name], key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_jobs = st.slider("Aufträge", *bounds("n_slider"), key="n_slider", step=C.N_STEP,
                        help=f"Anzahl der Aufträge. Bis {C.BRUTE_FORCE_MAX_N} läuft die Vollaufzählung aller n! Reihenfolgen live mit.")
    vehicle = st.radio("Vehikel", list(C.VEHICLE_LABELS), key="vehicle_radio", format_func=lambda k: C.VEHICLE_LABELS[k],
                        help="Neutral: durchgehend verfügbare Maschine. Werkstatt/Logistik: dieselben Aufträge, Maschine nur innerhalb fester Schichten verfügbar.")
    if vehicle == "logistik":
        shift_length = st.slider("Schichtlänge (Minuten)", *bounds("shift_length_slider"), key="shift_length_slider",
                                  help="Kurze Schichten bedeuten viele Lücken in der Maschinenverfügbarkeit.")
    else:
        shift_length = C.DEFAULT_SHIFT_LENGTH
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Seed für Bearbeitungszeiten und Fälligkeiten.")
    chain_seed = st.number_input("Zufalls-Seed der Kette", *bounds("chain_seed_input"), key="chain_seed_input", step=1,
                                  help="Steuert nur die zufällige Vergleichs-Reihenfolge - EDD selbst ist deterministisch (kein Zufall im Kern).")
    st.button("🎲 Neue Kette würfeln", width="stretch", on_click=randomize_chain_seed, help="Würfelt einen neuen Seed für die Zufalls-Vergleichsreihenfolge.")

sync_query_params({"n_slider": int(n_jobs), "seed_input": int(seed), "chain_seed_input": int(chain_seed), "vehicle_radio": vehicle,
                    "shift_length_slider": int(shift_length)})

settings = Settings(int(n_jobs), int(seed), int(chain_seed))
with st.spinner("Rechne..."):
    a = _analysis(settings)
inst = a.inst
p, d = inst.p, inst.d
data_key = (settings, vehicle, shift_length)

if vehicle == "logistik":
    edd_with_shift = A.evaluate_order_with_shifts(p, d, int(shift_length), a.edd.order)
    opt_with_shift = A.brute_force_optimal_with_shifts(p, d, int(shift_length)) if n_jobs <= C.BRUTE_FORCE_MAX_N else None

# --- EDD in Aktion ---------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 EDD in Aktion")
STEP_LABELS = {1: "1 · Aufträge", 2: "2 · Einplanen", 3: "3 · Ergebnis"}
if "edd_step" not in st.session_state or st.session_state.get("edd_step_owner") != data_key:
    st.session_state["edd_step"] = 1
    st.session_state["edd_step_owner"] = data_key
step = st.select_slider("Schritt", options=list(STEP_LABELS), key="edd_step", format_func=lambda s: STEP_LABELS[s])

if step == 2:
    it_col, itplay_col = st.columns([5, 2])
    with it_col:
        upto = st.slider("Eingeplante Aufträge", 1, int(n_jobs), value=int(n_jobs), key="edd_upto")
else:
    upto = int(n_jobs)

view_slot = st.empty()
with view_slot.container():
    if step == 1:
        st.markdown(f"**{n_jobs} Aufträge, unsortiert** (Bearbeitungszeit in Minuten)")
        st.bar_chart({"Bearbeitungszeit": p.tolist(), "Fälligkeit": d.tolist()})
    elif step == 2:
        st.markdown(f"**EDD-Reihenfolge nach {upto} von {n_jobs} Aufträgen**")
        st.plotly_chart(build_schedule(p, a.edd.order, upto=upto), width="stretch", key=f"s2_sched_{upto}")
        st.caption(f"Größte Verspätung bisher: {_fmt_int(a.edd.lateness[:upto].max())}")
    else:
        st.markdown("**Fertigstellung gegen Fälligkeit in EDD-Reihenfolge**")
        st.plotly_chart(build_completion_vs_due(a.edd.completion, d, a.edd.order), width="stretch", key="s3_curve")

if step == 1:
    st.caption(f"Bearbeitungszeiten zwischen {int(p.min())} und {int(p.max())}, Fälligkeiten zwischen {int(d.min())} und {int(d.max())} Minuten (Seed {seed}).")
elif step == 2:
    st.caption("Jeder Balken ist ein Auftrag; die Reihenfolge folgt der Fälligkeit, nicht der Bearbeitungszeit.")
else:
    st.caption(f"EDD: größte Verspätung {_fmt_int(a.edd.lmax)} Minuten. SPT (falsche Regel für dieses Ziel): {_fmt_int(a.spt.lmax)} Minuten.")

st.markdown("---")

# --- Ergebnis -------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was die Sortierung bringt")
st.caption("**Abstand:** Differenz der maximalen Verspätung (Lmax) einer Reihenfolge gegenüber EDD, in Minuten. EDD selbst ist deterministisch - nur die Zufalls-Vergleichsreihenfolge streut.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("EDD (größte Verspätung)", f"{_fmt_int(a.edd.lmax)} min", help="Die Zielgröße: größte Verspätung in EDD-Reihenfolge (negativ heißt: alle Aufträge fertig vor ihrer Frist).")
m2.metric("SPT (falsche Regel hier)", f"+{_fmt_int(a.gap_spt)} min", delta_color="off", help="SPT optimiert ΣCⱼ, nicht Lmax - hier zum Vergleich, nicht als Konkurrenz.")
m3.metric(f"Zufällige Reihenfolge (Mittel über {a.random_runs})", f"+{_fmt_int(a.gap_random)} min", delta_color="off", help="Mittel über mehrere zufällige Reihenfolgen derselben Instanz.")
if a.optimal is not None:
    m4.metric("Vollaufzählung (Gegenprobe)", "trifft EDD exakt" if a.edd_matches_optimum else "WEICHT AB", delta_color="off",
              help=f"Alle {n_jobs}! Reihenfolgen durchprobiert - unabhängige Bestätigung, dass EDD wirklich das Minimum von Lmax trifft.")
else:
    m4.metric("Vollaufzählung", f"erst ab n ≤ {C.BRUTE_FORCE_MAX_N}", delta_color="off")

if a.optimal is not None and not a.edd_matches_optimum:
    st.error("⚠️ EDD weicht von der Vollaufzählung ab - das wäre ein Fehler im Beweis oder in der Implementierung, bitte melden.")
else:
    st.success(f"✅ EDD hält die größte Verspätung {_fmt_int(a.gap_spt)} Minuten kleiner als SPT und {_fmt_int(a.gap_random)} Minuten kleiner als eine zufällige Reihenfolge - bei dieser Zielfunktion beweisbar die beste überhaupt.")

if vehicle == "logistik":
    st.markdown("**Auf dem Werkstatt/Logistik-Vehikel** (Schichtgrenzen berücksichtigt):")
    lm1, lm2 = st.columns(2)
    lm1.metric("EDD, Schichten mitgerechnet", f"{_fmt_int(edd_with_shift.lmax)} min", help="Dieselbe EDD-Reihenfolge wie oben, aber die Fertigstellungszeiten berücksichtigen jetzt Lücken an Schichtgrenzen.")
    if opt_with_shift is not None:
        gap = edd_with_shift.lmax - opt_with_shift.lmax
        lm2.metric("Echtes Optimum MIT Schichten", f"{_fmt_int(opt_with_shift.lmax)} min", delta=f"EDD ist {gap:.0f} min darüber", delta_color="off",
                   help="Vollaufzählung, die die Schichtgrenzen selbst mit optimiert - nur für kleine n möglich.")
        if gap > 0.5:
            st.warning(f"⚠️ EDD ist hier NICHT mehr optimal: {gap:.0f} Minuten über dem echten Optimum. Der Beweis oben setzt eine durchgehend verfügbare Maschine voraus - siehe 🚧 unten.")
        else:
            st.info("ℹ️ Bei dieser Instanz liegt EDD trotz Schichten exakt am Optimum - das ist nicht garantiert, siehe die Messreihe unten.")
    else:
        lm2.metric("Echtes Optimum MIT Schichten", f"erst ab n ≤ {C.BRUTE_FORCE_MAX_N}", delta_color="off")

st.markdown("---")

# --- Sweeps -----------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt der Vorsprung von der Instanz ab?")
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(SWEEP_LABELS), format_func=lambda k: SWEEP_LABELS[k], key="sweep_select")
if st.button("Sweep über 5 feste Instanzen berechnen (dauert wenige Sekunden)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {sweep_param}
if sweep_param in st.session_state.get("sweep_done", set()):
    rows_sweep = _sweep(sweep_param, Settings())
    st.plotly_chart(build_sweep(rows_sweep, SWEEP_LABELS[sweep_param]), width="stretch", key="sweep_chart")
    st.caption("Mittel über 5 feste Instanzen (Seeds 100000–100004) mit je drei Zufalls-Ketten für die Vergleichsreihenfolge.")

st.markdown("---")

# --- Experimente ------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Stimmt der Beweis wirklich? Vollaufzählung gegen EDD")
if st.button("Vollaufzählung über n = 2 bis 9 berechnen (dauert etwa 5 Sekunden)", key="opt_start"):
    st.session_state["opt_on"] = True
if st.session_state.get("opt_on"):
    rows_opt = _optimality()
    st.table({"Aufträge": [r["value"] for r in rows_opt], "Trefferquote": [f"{r['match_rate']:.0%}" for r in rows_opt]})
    st.caption("Für jede Instanzgröße 5 feste Instanzen: EDD gegen die Vollaufzählung aller n! Reihenfolgen. Jede Abweichung von 100 % wäre ein Fehler im Beweis oder in der Implementierung.")

st.markdown("---")

st.subheader("🔬 Wie teuer ist eine Vollaufzählung wirklich?")
if st.button("Rechenzeit für n = 2 bis 9 messen (dauert etwa 1 Sekunde)", key="timing_start"):
    st.session_state["timing_on"] = True
if st.session_state.get("timing_on"):
    rows_t = _timing()
    st.plotly_chart(build_timing(rows_t), width="stretch", key="timing_chart")
    last = rows_t[-1]
    st.caption(f"Bei {last['value']} Aufträgen braucht die Vollaufzählung bereits {last['brute_force_seconds']*1000:.0f} ms, EDD {last['edd_seconds']*1000:.3f} ms.")

st.markdown("---")

st.subheader("🔬 Werkstatt/Logistik: bleibt EDD optimal, wenn Schichten Lücken hineinbringen?")
if st.button("Schichtlänge von 480 bis 120 Minuten durchfahren (dauert wenige Sekunden)", key="shift_start"):
    st.session_state["shift_on"] = True
if st.session_state.get("shift_on"):
    rows_s = _shift_gap_sweep(min(int(n_jobs), C.BRUTE_FORCE_MAX_N))
    st.plotly_chart(build_shift_gap(rows_s), width="stretch", key="shift_chart")
    st.caption("EDD sortiert weiterhin nur nach Fälligkeit und ignoriert Schichtgrenzen; verglichen mit der echten Optimallösung MIT Schichten (Vollaufzählung, deshalb kleine Instanz). Bei sehr langen Schichten bindet die Grenze praktisch nie - der Abstand ist dann 0.")

st.markdown("---")

# --- Grenzen ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Maschine ist durchgehend verfügbar** | Sobald Schichten Lücken hineinbringen (Vehikel „Werkstatt/Logistik“), ist EDD nicht mehr beweisbar optimal - solange die Schichten lang genug sind, bindet die Grenze aber selten und der Abstand bleibt 0. | Kein direkter Nachfolger in dieser Linie |
| **Alle Aufträge sind gleich wichtig** | Mit unterschiedlichen Gewichten reicht "früheste Fälligkeit" nicht mehr - das Ziel wird zu gewichteter Verspätung statt der größten allein. | **Gewichtete Verspätung** (Folgestück, NP-schwer) |
| **Es gibt nur eine Verspätung, die zählt** | Zählt man ALLE verspäteten Aufträge (nicht nur den schlimmsten), ist EDD nicht mehr die richtige Regel. | **Moore-Hodgson** (Folgestück) |
| **Es gibt nur eine Maschine** | Mit mehreren Maschinen wird aus einer Sortierfrage eine Zuordnungs- UND Reihenfolgefrage. | **Johnson-Regel, LPT, Job Shop** (Folgestücke) |
"""
)
st.caption(
    "Erstes Stück dieser Linie: [spt-scheduling-demo](https://sebastianhanisch-spt-scheduling-demo.streamlit.app/) "
    "(SPT, 1||ΣCⱼ, dasselbe Beweismuster für eine andere Zielfunktion)."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Problem** ($1||L_{\max}$): $n$ Aufträge mit Bearbeitungszeit $p_j$ und Fälligkeit $d_j$ auf einer Maschine;
eine Reihenfolge $\pi$ legt die Fertigstellungszeit $C_j$ fest. Verspätung $L_j = C_j - d_j$ (negativ, wenn der
Auftrag vor seiner Frist fertig ist). Gesucht: $\pi$, das $L_{\max} = \max_j L_j$ minimiert.

**Satz (Jackson 1955).** EDD (aufsteigend nach $d_j$ sortieren) minimiert $L_{\max}$.

**Beweis (Vertauschungsargument).** Seien $i$ vor $j$ benachbart mit $d_i > d_j$. Vertauscht man sie, ändert sich
nur die Fertigstellungszeit der beiden - alle anderen Aufträge bleiben unberührt, weil die Summe der
Bearbeitungszeiten davor gleich bleibt. Nach der Vertauschung ist die Fertigstellungszeit des SPÄTER
fertigwerdenden Auftrags (jetzt $i$) unverändert gegenüber vorher (wo $j$ als zweiter fertig wurde) - $j$s neue
Verspätung ist höchstens so groß wie $i$s alte, weil $d_j < d_i$. Die maximale Verspätung der beiden kann durch
die Vertauschung also nicht steigen. Jede nicht EDD-sortierte Reihenfolge lässt sich damit ohne Verschlechterung
in Richtung EDD umsortieren; EDD ist ein Optimum.

**Kennzahl.** Abstand zu EDD $= L_{\max}(\pi) - L_{\max}(\text{EDD})$ in Minuten (keine Prozentangabe, da
$L_{\max}$ negativ werden kann). Für $n \le 9$ zusätzlich die Vollaufzählung als unabhängige Gegenprobe.

**Fälligkeiten.** $d_j \sim U(P \cdot (1 - TF - RDD/2),\, P \cdot (1 - TF + RDD/2))$ mit $P = \sum_j p_j$
(Potts & Van Wassenhove 1982/1985, Standardschema der Scheduling-Literatur).

**Grenzen.** (1) Schichtgrenzen verletzen die Voraussetzung "Maschine durchgehend verfügbar" - EDD bleibt dann
nur noch eine gute Heuristik (Vehikel B). (2) Ohne Gewichte ist "größte Verspätung" die einfachste Zielfunktion
mit Fristen - Moore-Hodgson (Anzahl verspäteter Aufträge) und gewichtete Verspätung verallgemeinern das.

Implementiert in `edd_algorithm.py` (EDD, Brute-Force-Gegenprobe, Schicht-Variante), `edd_scenario.py`/
`edd_scenario_logistik.py` (die zwei Vehikel, wortgleich aus `spt-scheduling-demo` übernommen),
`edd_evaluation.py` (Kennzahlen, Sweeps, Optimalitäts- und Timing-Messreihe, Schicht-Härtetest).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
