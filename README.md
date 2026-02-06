# Wine Quality ML Project

Machine Learning Projekt zur Vorhersage der Weinqualität basierend auf physikalisch-chemischen Eigenschaften.

## Übersicht

Dieses Projekt analysiert den [Wine Quality Dataset](https://archive.ics.uci.edu/dataset/186/wine+quality) vom UCI Machine Learning Repository und entwickelt Vorhersagemodelle für die Weinqualität.

**Features:**
- Jupyter Notebook mit vollständiger Datenanalyse und Modelltraining
- Interaktive Streamlit-Webanwendung
- 6 Regressionsmodelle + 5 Klassifikationsmodelle
- Hyperparameter-Optimierung

## Schnellstart

```bash
# Abhängigkeiten installieren
pip install -r requirements.txt

# Jupyter Notebook starten
jupyter notebook wine_quality_analysis.ipynb

# Streamlit App starten
streamlit run wine_quality_app.py
```

## Dataset

| Typ | Samples |
|-----|---------|
| Rotwein | 1.599 |
| Weißwein | 4.898 |
| **Gesamt** | **6.497** |

**11 Features:** Säure, Zucker, Chloride, Schwefeldioxid, Dichte, pH, Sulfate, Alkohol, Weintyp

## Ergebnisse

### Regression (Qualitätswert 3-9)
| Modell | R² |
|--------|-----|
| Random Forest (Tuned) | ~0.50 |
| Gradient Boosting | ~0.46 |

### Klassifikation (Gut/Schlecht)
| Modell | F1-Score |
|--------|----------|
| Random Forest (Tuned) | ~0.82 |
| Gradient Boosting | ~0.79 |

## Projektstruktur

```
ML-Projekt/
├── wine_quality_analysis.ipynb   # Hauptanalyse
├── wine_quality_app.py           # Streamlit App
├── wine+quality/                 # Datensätze
├── requirements.txt              # Abhängigkeiten
└── README-Ben Neunteufel.md      # Ausführliche Dokumentation
```

## Technologien

- Python 3.8+
- Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn, Plotly
- Streamlit, Jupyter

---

**Autor:** Ben Neunteufel
**Klasse:** 5AHITS
**Datum:** Dezember 2025
