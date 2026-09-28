# EDD – eine Warteschlange, die die schlimmste Verspätung klein hält – Streamlit-Demo

**[→ Demo live ausprobieren](#) (Deploy offen)**

Zweites Stück der **Klassische-Scheduling-Theorie-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch
– Operations Research und Machine Learning": $n$ Aufträge mit Bearbeitungszeit $p_j$ und Fälligkeit $d_j$ auf
**einer** Maschine, Ziel ist die maximale Verspätung $L_{\max} = \max_j (C_j - d_j)$ zu minimieren ($1||L_{\max}$).

**Einordnung in die Linie:** EDD (Earliest Due Date first, Jackson 1955) ist dasselbe Beweismuster wie SPT im
ersten Stück dieser Linie ([spt-scheduling-demo](https://sebastianhanisch-spt-scheduling-demo.streamlit.app/)) –
ein Vertauschungsargument – nur für eine andere Zielfunktion: statt der Summe der Fertigstellungszeiten zu
minimieren, geht es hier um die größte Verspätung. Aufsteigend nach Fälligkeit sortieren ist beweisbar optimal.
```
SPT (1||ΣCⱼ, Vertauschungsargument)                                              [Stück 1]
 ├─ EDD (1||Lmax, dasselbe Beweismuster, andere Zielfunktion)                     [dieses Stück]
 ├─ Moore-Hodgson (1||ΣUⱼ, gierig + schlechtesten Verspäteten entfernen)          [Folgestück]
 ├─ WSPT / Smith's Rule (1||ΣwⱼCⱼ, verallgemeinert SPT mit Gewichten)             [Folgestück]
 ├─ Johnson-Regel (F2||Cmax, zweite Maschine)                                     [Folgestück]
 ├─ LPT (Pm||Cmax, parallele Maschinen)                                           [Folgestück]
 └─ Job Shop (Konvergenzpunkt: Reihenfolge UND Maschinenwahl)                     [Folgestück]
```

Ergebnis in Kürze: **EDD trifft auf jeder getesteten Instanz (n = 2 bis 9) exakt das Minimum der Vollaufzählung.**
Bei 20 Aufträgen hält EDD die größte Verspätung im Mittel **346 Minuten** kleiner als SPT (die für dieses Ziel
falsche Regel) und **394 Minuten** kleiner als eine zufällige Reihenfolge.
**Der ehrliche Bruch:** auf dem Werkstatt/Logistik-Vehikel (Schichten statt durchgehender Verfügbarkeit) bleibt
EDD bei der Standard-Schichtlänge (480 Minuten) exakt optimal – die Lücken binden dort praktisch nie. Werden die
Schichten kurz genug (≤ 350 Minuten bei dieser Instanzgröße), bindet die Grenze oft und EDD ist **nicht mehr
beweisbar optimal** – der Abstand zum echten Optimum wird zweistellig (Minuten) und ist über die getesteten
Schichtlängen nicht einmal monoton, ein echter, ungeglätteter Befund.

| Frage | Ergebnis (Mittel über 5 feste Instanzen, Seeds 100000–100004, mit je 3 Ketten-Seeds) |
|---|---|
| Standardfall (20 Aufträge) | ✅ EDD hält Lmax **346 min** kleiner als SPT und **394 min** kleiner als Zufall |
| **Beweis gegen Vollaufzählung** | ✅ **100 %** Trefferquote bei n = 2 bis 9 |
| **Rechenzeit** | ➖ Vollaufzählung bei n = 9 bereits über 1000 ms, EDD bei rund 0,005 ms |
| **Vehikel Werkstatt/Logistik** | ❌ Schichtlänge 480/350/240/160/120 Minuten: Abstand zum Optimum 0/≈19/≈14/≈62/≈60 Minuten – ab kurzen Schichten NICHT mehr beweisbar optimal, und nicht monoton |

## Was die Demo zeigt

1. **EDD in Aktion** (Schritt-Slider): **Aufträge** (Bearbeitungszeit und Fälligkeit) → **Einplanen** (Regler
   "eingeplante Aufträge") → **Ergebnis** (Fertigstellung gegen Fälligkeit in EDD-Reihenfolge).
2. **Was die Sortierung bringt:** EDD, SPT (falsche Regel hier), zufällige Reihenfolge, Vollaufzählungs-
   Gegenprobe (n ≤ 9); auf dem Werkstatt/Logistik-Vehikel zusätzlich EDD mit Schichten gegen das echte Optimum
   mit Schichten.
3. **📐 Sweep** über Aufträge, Fristen-Anteil und -Streuung.
4. **🔬 Experimente auf Abruf:** Vollaufzählung gegen EDD über n = 2 bis 9 (Beweis-Check); Rechenzeit $n!$ gegen
   $n \log n$; Schicht-Härtetest auf dem Werkstatt/Logistik-Vehikel.
5. **🚧 Grenzen:** Tabelle "Annahme – was passiert – wer setzt an".

Regler: Aufträge (2–60), **Vehikel** (Neutral/Werkstatt-Logistik – bei Werkstatt zusätzlich Schichtlänge und
Anzahl Familien), Seed der Instanz (+ 🎲), Seed der Kette (+ 🎲, steuert nur die zufällige Vergleichsreihenfolge).

## Die zwei Vehikel (aus `spt-scheduling-demo` übernommen, wortgleich für die ganze Linie)

- **Neutral** (`edd_scenario.py`): $n$ Aufträge mit Bearbeitungszeit $p_j \sim U(1, 100)$ und Fälligkeit
  $d_j \sim U(P \cdot (1-TF-RDD/2),\, P \cdot (1-TF+RDD/2))$ (Potts & Van Wassenhove 1982/1985, per WebSearch
  verifiziert) – hier die Hauptgröße.
- **Werkstatt/Logistik** (`edd_scenario_logistik.py`): dieselben Bearbeitungszeiten/Fälligkeiten, aber die
  Maschine ist nur innerhalb fester Schichten verfügbar – ein Auftrag, der nicht mehr in die laufende Schicht
  passt, rutscht auf den Beginn der nächsten. Sehr lange Schichten kollabieren exakt zum neutralen Vehikel (per
  Test belegt).

## Modell und Verfahren

- **Instanz** (`edd_scenario.py`, `edd_scenario_logistik.py`): wortgleich aus `spt-scheduling-demo` übernommen.
- **EDD** (`edd_algorithm.py`): aufsteigend nach Fälligkeit sortieren, $O(n \log n)$. Dazu die
  Brute-Force-Vollaufzählung (nur für kleine $n$) als unabhängige Gegenprobe, und die Schicht-Variante für das
  Werkstatt/Logistik-Vehikel.
- **Auswertung** (`edd_evaluation.py`): Kennzahlen, Sweeps, Optimalitäts- und Timing-Messreihe, Schicht-Härtetest.

## Was nicht funktioniert hat / Grenzen

- **Vorab-Vermutung: "EDD bleibt gut, solange Schichten nicht zu oft binden"** – **bestätigt, aber mit einer
  scharfen statt einer weichen Grenze**: bei der Standard-Schichtlänge (480 Minuten) ist der Abstand exakt 0 bei
  dieser Instanzgröße, bei 350 Minuten schon spürbar messbar (im zweistelligen Minutenbereich) und bleibt es bei
  noch kürzeren Schichten. Der Übergang ist nicht der sanfte, monoton wachsende Verlauf wie beim Rüstzeit-
  Härtetest im ersten Stück dieser Linie (dort SPT/Rüstzeiten) – hier kippt es eher plötzlich, sobald Schichten
  überhaupt binden, und schwankt danach zwischen den Instanzen (kleine Stichprobe, 5 Instanzen × kleines n).
- **Die Vollaufzählung ist die einzige echte Gegenprobe**, praktisch nur bis $n \approx 9$ nutzbar – ab dort
  vertraut die Demo dem Beweis.
- **Synthetische Instanzen:** Bearbeitungszeiten gleichverteilt, Fälligkeiten nach dem TF/RDD-Schema, keine
  Präzedenzen, ein Auftrag = eine Operation.

## Verifikation

- **Beweis gegen unabhängige Vollaufzählung:** für jede getestete Instanzgröße (n = 2 bis 9) und jede der 5
  festen Instanzen trifft EDD exakt das Minimum von $L_{\max}$ aller $n!$ Reihenfolgen – 100 % Trefferquote.
- **Schicht-Variante gegen unabhängige Vollaufzählung** und **Konsistenz-Test**: eine praktisch unendlich lange
  Schicht liefert exakt dieselbe Zielfunktion wie das neutrale Vehikel.
- **Handrechnung:** eine kleine, von Hand nachgerechnete Instanz (3 Aufträge) bestätigt EDD-Reihenfolge,
  Fertigstellungs- und Verspätungswerte exakt; eine zweite Handrechnung zeigt, dass eine Schichtgrenze eine
  Lücke einfügt, wenn ein Auftrag nicht mehr passt.
- **Alle Zahlen der App-Texte sind als Tests hinterlegt** (Standardfall, Optimalitäts-Trefferquote, Rechenzeit,
  Schicht-Härtetest, jeweils Mittel über die festen Sweep-Instanzen × Ketten; positive **und** negative
  Aussagen); alle 5 Presets geprüft; AppTest-Rauchtests (Voreinstellung, jedes Preset, jeder Schritt auf beiden
  Vehikeln, Würfel-Knöpfe, Permalink-Grenzen inkl. ungültigem Vehikel, Extremwerte, Experimente auf Abruf,
  Footer).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Schritte, Ergebnis, 📐 Sweeps, 🔬 Experimente, 🚧 Grenzen, Mathe |
| `edd_algorithm.py` | EDD, Brute-Force-Gegenprobe, Schicht-Variante |
| `edd_scenario.py` | Vehikel Neutral (aus `spt-scheduling-demo`) |
| `edd_scenario_logistik.py` | Vehikel Werkstatt/Logistik (Familien, Rüstzeit-Matrix, Schichtlänge) |
| `edd_constants.py` | Konstanten, Presets |
| `edd_evaluation.py` | Kennzahlen, Sweeps, Optimalitäts- und Timing-Messreihe, Schicht-Härtetest |
| `edd_presets.py`, `edd_visualization.py` | Permalink/Presets, Plotly-Figuren (achsengesperrt) |
| `tests/` | Beweis gegen Vollaufzählung, Szenario und Auswertung, Aussagen der App, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
