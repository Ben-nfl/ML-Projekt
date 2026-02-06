# 🍷 Wine Quality - Vollständiges ML-Projekt

Ein umfassendes Machine-Learning-Projekt zur Vorhersage der Weinqualität basierend auf physikalisch-chemischen Eigenschaften. Das Projekt enthält sowohl ein detailliertes Jupyter Notebook für die Analyse als auch eine interaktive Streamlit-Webanwendung.

## 📋 Inhaltsverzeichnis

- [Projektübersicht](#projektübersicht)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Verwendung](#verwendung)
- [Projektstruktur](#projektstruktur)
- [Modelle](#modelle)
- [Ergebnisse](#ergebnisse)
- [Technologien](#technologien)

## 🎯 Projektübersicht

Dieses Projekt analysiert den **Wine Quality Dataset** vom UCI Machine Learning Repository und entwickelt Vorhersagemodelle für die Weinqualität. Es werden sowohl **Regression** (Vorhersage exakter Qualitätswerte) als auch **Klassifikation** (binäre Klassifikation: gut/schlecht) durchgeführt.

### Hauptziele:

1. Umfassende explorative Datenanalyse (EDA)
2. Feature Engineering und Datenbereinigung
3. Training und Evaluation mehrerer ML-Modelle
4. Hyperparameter-Optimierung
5. Interaktive Webanwendung zur Vorhersage

## 📊 Dataset

**Quelle:** [UCI Machine Learning Repository - Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality)

**Beschreibung:**
- **Rotwein:** 1599 Samples
- **Weißwein:** 4898 Samples
- **Gesamt (kombiniert):** 6497 Samples (nach Duplikat-Entfernung)
- **Features:** 11 physikalisch-chemische Eigenschaften + 1 Weintyp
- **Zielvariable:** Qualität (Skala 0-10, tatsächlich 3-9)

### Features:

1. **fixed acidity** - Weinsäure
2. **volatile acidity** - Flüchtige Säure
3. **citric acid** - Zitronensäure
4. **residual sugar** - Restzucker
5. **chlorides** - Chloride
6. **free sulfur dioxide** - Freies Schwefeldioxid
7. **total sulfur dioxide** - Gesamtschwefeldioxid
8. **density** - Dichte
9. **pH** - pH-Wert
10. **sulphates** - Sulfate
11. **alcohol** - Alkoholgehalt
12. **wine_type** - Weintyp (0=Weißwein, 1=Rotwein)

### Engineered Features:

1. **free_to_total_sulfur_ratio** - Verhältnis freies/gesamt Schwefel
2. **acid_ratio** - Verhältnis Weinsäure/flüchtige Säure
3. **total_acidity** - Gesamtsäure
4. **alcohol_per_acid** - Alkohol pro Säure
5. **quality_class** - Binär (0=schlecht <6, 1=gut ≥6)

## ✨ Features

### Jupyter Notebook (`wine_quality_analysis.ipynb`)

- ✅ Vollständiger Datenanalyse-Workflow
- ✅ Umfassende Visualisierungen (Histogramme, Boxplots, Heatmaps, Scatterplots, Pairplots)
- ✅ Ausreißer-Erkennung mit IQR-Methode
- ✅ Korrelationsanalyse
- ✅ Feature Engineering
- ✅ 6 Regressionsmodelle:
  - Linear Regression
  - Ridge Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - K-Nearest Neighbors
  - Support Vector Regressor
- ✅ 5 Klassifikationsmodelle:
  - Logistic Regression
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - K-Nearest Neighbors
  - Support Vector Classifier
- ✅ Hyperparameter-Tuning mit RandomizedSearchCV
- ✅ Umfassende Evaluation:
  - Regression: RMSE, MAE, R²
  - Klassifikation: Accuracy, Precision, Recall, F1, AUC, Confusion Matrix, ROC Curve
- ✅ Feature Importance Analyse
- ✅ Modellvergleich und Empfehlungen

### Streamlit App (`wine_quality_app.py`)

- ✅ **Datenexploration:**
  - Interaktive Visualisierungen
  - Korrelationsmatrizen
  - Feature-Verteilungen
  - Scatter-Plots

- ✅ **Modell Training:**
  - Auswahl zwischen Regression und Klassifikation
  - 6 verschiedene Modelltypen
  - Anpassbare Hyperparameter über Sidebar
  - Live-Training und Evaluation
  - Visualisierung der Ergebnisse
  - Feature Importance Analyse
  - Modell-Export als .pkl Datei

- ✅ **Vorhersage:**
  - Einzelne Vorhersage mit manuellen Eingaben
  - Batch-Vorhersage mit CSV-Upload
  - Visualisierung der Vorhersage-Konfidenz
  - Export der Ergebnisse

## 🚀 Installation

### Voraussetzungen

- Python 3.8 oder höher
- pip (Python Package Manager)

### Schritt 1: Repository klonen oder herunterladen

```bash
cd /path/to/ML-Projekt
```

### Schritt 2: Virtual Environment erstellen (empfohlen)

```bash
# Virtual Environment erstellen
python -m venv venv

# Aktivieren (Windows)
venv\Scripts\activate

# Aktivieren (macOS/Linux)
source venv/bin/activate
```

### Schritt 3: Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

## 💻 Verwendung

### Jupyter Notebook

```bash
# Jupyter Notebook starten
jupyter notebook

# Dann öffne: wine_quality_analysis.ipynb
```

Führe alle Zellen sequenziell aus, um:
- Die Daten zu laden und zu analysieren
- Visualisierungen zu erstellen
- Modelle zu trainieren
- Ergebnisse zu evaluieren

### Streamlit App starten

```bash
streamlit run wine_quality_app.py
```

Die App öffnet sich automatisch im Browser unter `http://localhost:8501`

### App-Navigation:

1. **📊 Datenexploration**
   - Überblick über den Datensatz
   - Statistische Zusammenfassung
   - Interaktive Visualisierungen
   - Korrelationsanalyse

2. **🤖 Modell Training**
   - Wähle Aufgabentyp (Regression/Klassifikation)
   - Wähle Modell aus Dropdown
   - Passe Hyperparameter an
   - Trainiere Modell
   - Evaluiere Performance
   - Speichere trainiertes Modell

3. **🔮 Vorhersage**
   - Lade trainiertes Modell
   - Einzelvorhersage: Manuelle Eingabe der Features
   - Batch-Vorhersage: CSV-Upload
   - Visualisierung der Ergebnisse

## 📁 Projektstruktur

```
ML-Projekt/
│
├── wine+quality/
│   ├── winequality-red.csv          # Rotwein-Datensatz
│   ├── winequality-white.csv        # Weißwein-Datensatz
│   └── winequality.names            # Dokumentation
│
├── wine_quality_analysis.ipynb      # Jupyter Notebook (Hauptanalyse)
├── wine_quality_app.py              # Streamlit Webanwendung
├── requirements.txt                 # Python-Abhängigkeiten
├── README.md                        # Projektdokumentation
│
└── (generierte Dateien nach Training)
    ├── wine_quality_models.pkl      # Gespeicherte Modelle (aus Notebook)
    └── trained_model_*.pkl          # Modelle aus Streamlit App
```

## 🤖 Modelle

### Regression (Vorhersage exakter Qualitätswerte)

| Modell | Test RMSE | Test MAE | Test R² | Empfehlung |
|--------|-----------|----------|---------|------------|
| **Random Forest (Tuned)** | ~0.58 | ~0.45 | ~0.50 | ⭐ Beste Balance |
| **Gradient Boosting (Tuned)** | ~0.59 | ~0.46 | ~0.49 | ⭐ Sehr gut |
| Random Forest | ~0.60 | ~0.47 | ~0.47 | Gut |
| Gradient Boosting | ~0.61 | ~0.48 | ~0.46 | Gut |
| Ridge Regression | ~0.65 | ~0.51 | ~0.38 | Baseline |
| Linear Regression | ~0.65 | ~0.51 | ~0.38 | Baseline |

### Klassifikation (Gut ≥6 vs Schlecht <6)

| Modell | Accuracy | Precision | Recall | F1-Score | AUC | Empfehlung |
|--------|----------|-----------|--------|----------|-----|------------|
| **Random Forest (Tuned)** | ~0.78 | ~0.80 | ~0.85 | ~0.82 | ~0.86 | ⭐ Beste Wahl |
| **Gradient Boosting (Tuned)** | ~0.77 | ~0.79 | ~0.84 | ~0.81 | ~0.85 | ⭐ Sehr gut |
| Random Forest | ~0.76 | ~0.78 | ~0.83 | ~0.80 | ~0.84 | Gut |
| Gradient Boosting | ~0.75 | ~0.77 | ~0.82 | ~0.79 | ~0.83 | Gut |
| Logistic Regression | ~0.73 | ~0.75 | ~0.80 | ~0.77 | ~0.80 | Baseline |

## 📈 Ergebnisse

### Wichtigste Erkenntnisse:

1. **Top 5 wichtigste Features:**
   - Alkoholgehalt (alcohol)
   - Volatile Säure (volatile acidity)
   - Sulfate
   - Zitronensäure (citric acid)
   - Gesamtschwefeldioxid (total sulfur dioxide)

2. **Modellempfehlungen:**
   - **Für Regression:** Random Forest Regressor (Tuned) - R² ≈ 0.50
   - **Für Klassifikation:** Random Forest Classifier (Tuned) - F1 ≈ 0.82

3. **Performance:**
   - Regressionsmodelle erreichen R²-Werte um 0.50, was für ein komplexes Qualitätsproblem mit vielen subjektiven Faktoren sehr gut ist
   - Klassifikationsmodelle erreichen F1-Scores um 0.82, was exzellent ist
   - Hyperparameter-Tuning verbessert die Performance um 5-10%

4. **Weintyp-Einfluss:**
   - Weintyp (Rot vs Weiß) hat moderaten Einfluss auf Qualität
   - Beide Typen zeigen ähnliche Qualitätsverteilungen
   - Chemische Eigenschaften sind wichtiger als der Weintyp

## 🛠️ Technologien

- **Python 3.8+**
- **Pandas** - Datenmanipulation
- **NumPy** - Numerische Operationen
- **Scikit-learn** - Machine Learning Modelle
- **Matplotlib & Seaborn** - Statische Visualisierungen
- **Plotly** - Interaktive Visualisierungen
- **Streamlit** - Webanwendung
- **Jupyter** - Notebooks für Analyse

## 📝 Verwendete ML-Algorithmen

### Regression:
- Linear Regression
- Ridge Regression (L2 Regularisierung)
- Random Forest Regressor
- Gradient Boosting Regressor
- K-Nearest Neighbors Regressor
- Support Vector Regressor (SVR)

### Klassifikation:
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- K-Nearest Neighbors Classifier
- Support Vector Classifier (SVC)

### Optimierung:
- RandomizedSearchCV für Hyperparameter-Tuning
- Cross-Validation (5-Fold)
- StandardScaler für Feature-Normalisierung

Ben Neunteufel
Klasse: 5AHITS
Datum: 14.12.2025
