# Link: https://archive.ics.uci.edu/dataset/186/wine+quality

### 1️⃣ Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

**Wichtigste Pakete:**
- pandas, numpy (Datenverarbeitung)
- scikit-learn (Machine Learning)
- matplotlib, seaborn, plotly (Visualisierung)
- streamlit (Webanwendung)
- jupyter (Notebooks)

---

### 2️⃣ Jupyter Notebook ausführen

```bash
# Jupyter starten
jupyter notebook

# Öffne: wine_quality_analysis.ipynb
# Führe alle Zellen aus: Cell → Run All
```

**Was passiert:**
- ✅ Daten werden geladen (Rot- und Weißwein)
- ✅ Explorative Datenanalyse
- ✅ Visualisierungen erstellt
- ✅ 11 ML-Modelle trainiert (6 Regression + 5 Klassifikation)
- ✅ Hyperparameter-Tuning durchgeführt
- ✅ Ergebnisse evaluiert und verglichen
- ✅ Modelle gespeichert als `wine_quality_models.pkl`

**Dauer:** ~5-10 Minuten (je nach Hardware)

---

### 3️⃣ Streamlit App starten

```bash
streamlit run wine_quality_app.py
```

**Browser öffnet automatisch:** http://localhost:8501

**Funktionen:**
- 📊 Datenexploration - Interaktive Visualisierungen
- 🤖 Modell Training - Wähle Modell, passe Hyperparameter an, trainiere
- 🔮 Vorhersage - Erstelle Einzelvorhersagen oder Batch-Vorhersagen

---

## 📱 App-Nutzung

### Seite 1: Datenexploration
1. Navigiere zu "📊 Datenexploration"
2. Erkunde Statistiken, Verteilungen, Korrelationen
3. Wähle Features für Visualisierungen aus

### Seite 2: Modell Training
1. Navigiere zu "🤖 Modell Training"
2. Wähle Aufgabentyp:
   - **Regression** → Exakte Qualitätswerte (3-9)
   - **Klassifikation** → Gut (≥6) vs Schlecht (<6)
3. Wähle Modell aus Dropdown (z.B. Random Forest)
4. Passe Hyperparameter in der Sidebar an
5. Klicke "🚀 Modell trainieren"
6. Analysiere Metriken und Visualisierungen
7. Speichere Modell mit "💾 Modell speichern"

### Seite 3: Vorhersage
1. Navigiere zu "🔮 Vorhersage"
2. Lade ein trainiertes Modell hoch (.pkl Datei)
3. **Option A - Einzelvorhersage:**
   - Gib Wein-Eigenschaften ein
   - Klicke "🔮 Vorhersage erstellen"
   - Sehe Ergebnis und Konfidenz
4. **Option B - Batch-Vorhersage:**
   - Lade CSV-Datei mit mehreren Weinen
   - Klicke "🚀 Batch Vorhersage starten"
   - Lade Ergebnisse als CSV herunter

---

## 💡 Tipps

### Beste Modelle (basierend auf Analyse):

**Für Regression:**
- Random Forest Regressor
- Gradient Boosting Regressor
- Empfohlene Hyperparameter im Notebook

**Für Klassifikation:**
- Random Forest Classifier
- Gradient Boosting Classifier
- F1-Score > 0.80 erreichbar

### Wichtigste Features für Qualität:
1. Alkoholgehalt
2. Volatile Säure
3. Sulfate
4. Zitronensäure
5. Gesamtschwefeldioxid

### Typische Weinwerte:

**Guter Wein (Quality ≥ 6):**
- Höherer Alkoholgehalt (~11-12%)
- Niedrigere volatile Säure (<0.4)
- Höhere Sulfate
- Mehr Zitronensäure

**Schlechter Wein (Quality < 6):**
- Niedrigerer Alkoholgehalt (~9-10%)
- Höhere volatile Säure (>0.5)
- Niedrigere Sulfate
- Weniger Zitronensäure

---

## 🔧 Fehlerbehebung

### Problem: Jupyter Notebook startet nicht
```bash
# Jupyter neu installieren
pip install --upgrade jupyter notebook

# Alternative: JupyterLab verwenden
pip install jupyterlab
jupyter lab
```

### Problem: Streamlit startet nicht
```bash
# Streamlit neu installieren
pip install --upgrade streamlit

# Ports prüfen
streamlit run wine_quality_app.py --server.port 8502
```

### Problem: Datensatz nicht gefunden
- Stelle sicher, dass der Ordner `wine+quality/` existiert
- Prüfe, ob die CSV-Dateien vorhanden sind:
  - `winequality-red.csv`
  - `winequality-white.csv`

### Problem: Pakete fehlen
```bash
# Alle Abhängigkeiten neu installieren
pip install -r requirements.txt --force-reinstall
```

---

## 📊 Erwartete Ergebnisse

### Regression (R²-Werte):
- Linear/Ridge Regression: ~0.38
- Random Forest: ~0.47
- Gradient Boosting: ~0.46
- **Random Forest (Tuned): ~0.50** ⭐

### Klassifikation (F1-Scores):
- Logistic Regression: ~0.77
- Random Forest: ~0.80
- Gradient Boosting: ~0.79
- **Random Forest (Tuned): ~0.82** ⭐

---

## 🎯 Nächste Schritte

1. ✅ Notebook durcharbeiten → Ergebnisse verstehen
2. ✅ Streamlit App testen → Modelle interaktiv trainieren
3. ✅ Eigene Vorhersagen erstellen → Weinqualität schätzen
4. 🔮 Erweiterungen:
   - XGBoost hinzufügen
   - SHAP Values implementieren
   - Weitere Features engineeren
   - App online deployen

---

## 📞 Hilfe

Bei Fragen oder Problemen:
1. Prüfe die ausführliche `README.md`
2. Schaue in das Jupyter Notebook (kommentiert)
3. Prüfe die Streamlit-Dokumentation: https://docs.streamlit.io

