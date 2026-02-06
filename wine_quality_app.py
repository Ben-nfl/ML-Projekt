"""
Wine Quality Prediction - Streamlit App

Diese Streamlit-Anwendung ermöglicht die interaktive Analyse und Vorhersage
der Weinqualität basierend auf physikalisch-chemischen Eigenschaften.

Features:
- Datenexploration und Visualisierung
- Modellauswahl (Regression & Klassifikation)
- Hyperparameter-Tuning
- Echtzeit-Vorhersagen
- Modell-Evaluation mit Metriken
- Feature Importance Analyse
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Visualisierung
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# Klassifikation
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Metriken
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score
)

import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    h1 {
        color: #8B0000;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Lädt und kombiniert die Wein-Datensätze"""
    try:
        red_wine = pd.read_csv('wine+quality/winequality-red.csv', sep=';')
        white_wine = pd.read_csv('wine+quality/winequality-white.csv', sep=';')

        red_wine['wine_type'] = 1
        white_wine['wine_type'] = 0

        df = pd.concat([red_wine, white_wine], axis=0, ignore_index=True)
        df = df.drop_duplicates()

        return df
    except Exception as e:
        st.error(f"Fehler beim Laden der Daten: {e}")
        return None


def create_features(df):
    """Erstellt engineered Features"""
    df_eng = df.copy()

    # Verhältnis-Features
    df_eng['free_to_total_sulfur_ratio'] = (
        df_eng['free sulfur dioxide'] / (df_eng['total sulfur dioxide'] + 1e-10)
    )
    df_eng['acid_ratio'] = (
        df_eng['fixed acidity'] / (df_eng['volatile acidity'] + 1e-10)
    )
    df_eng['total_acidity'] = (
        df_eng['fixed acidity'] + df_eng['volatile acidity'] + df_eng['citric acid']
    )
    df_eng['alcohol_per_acid'] = (
        df_eng['alcohol'] / (df_eng['total_acidity'] + 1e-10)
    )

    # Binäre Qualitätsklasse
    df_eng['quality_class'] = (df_eng['quality'] >= 6).astype(int)

    return df_eng


def get_regression_models(params):
    """Gibt Dictionary mit Regressionsmodellen zurück"""
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(
            alpha=params.get('ridge_alpha', 1.0),
            random_state=42
        ),
        'Random Forest': RandomForestRegressor(
            n_estimators=params.get('rf_n_estimators', 100),
            max_depth=params.get('rf_max_depth', None),
            min_samples_split=params.get('rf_min_samples_split', 2),
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=params.get('gb_n_estimators', 100),
            learning_rate=params.get('gb_learning_rate', 0.1),
            max_depth=params.get('gb_max_depth', 3),
            random_state=42
        ),
        'K-Nearest Neighbors': KNeighborsRegressor(
            n_neighbors=params.get('knn_n_neighbors', 5)
        ),
        'Support Vector Regressor': SVR(
            kernel=params.get('svr_kernel', 'rbf'),
            C=params.get('svr_C', 1.0)
        )
    }
    return models


def get_classification_models(params):
    """Gibt Dictionary mit Klassifikationsmodellen zurück"""
    models = {
        'Logistic Regression': LogisticRegression(
            C=params.get('lr_C', 1.0),
            max_iter=1000,
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=params.get('rf_n_estimators', 100),
            max_depth=params.get('rf_max_depth', None),
            min_samples_split=params.get('rf_min_samples_split', 2),
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=params.get('gb_n_estimators', 100),
            learning_rate=params.get('gb_learning_rate', 0.1),
            max_depth=params.get('gb_max_depth', 3),
            random_state=42
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=params.get('knn_n_neighbors', 5)
        ),
        'Support Vector Classifier': SVC(
            kernel=params.get('svc_kernel', 'rbf'),
            C=params.get('svc_C', 1.0),
            probability=True,
            random_state=42
        )
    }
    return models


def plot_feature_importance(model, feature_names, model_name):
    """Plottet Feature Importance"""
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)

        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f'Feature Importance - {model_name}',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=500)
        return fig
    return None


def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plottet Confusion Matrix"""
    cm = confusion_matrix(y_true, y_pred)

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Schlecht (<6)', 'Gut (≥6)'],
        y=['Schlecht (<6)', 'Gut (≥6)'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16}
    ))

    fig.update_layout(
        title=f'Confusion Matrix - {model_name}',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=400
    )

    return fig


def plot_roc_curve(y_true, y_pred_proba, model_name):
    """Plottet ROC Curve"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_score = auc(fpr, tpr)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'{model_name} (AUC = {auc_score:.4f})',
        line=dict(width=3)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='red')
    ))

    fig.update_layout(
        title=f'ROC Curve - {model_name}',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=500
    )

    return fig


def plot_predicted_vs_actual(y_true, y_pred, model_name, r2):
    """Plottet Predicted vs Actual für Regression"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        mode='markers',
        name='Predictions',
        marker=dict(size=5, opacity=0.6)
    ))

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(dash='dash', color='red', width=2)
    ))

    fig.update_layout(
        title=f'Predicted vs Actual - {model_name} (R² = {r2:.4f})',
        xaxis_title='Actual Quality',
        yaxis_title='Predicted Quality',
        height=500
    )

    return fig


# Main App
def main():
    st.title("🍷 Wine Quality Prediction")
    st.markdown("---")

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Wähle eine Seite:",
        ["📊 Datenexploration", "🤖 Modell Training", "🔮 Vorhersage"]
    )

    # Daten laden
    df = load_data()
    if df is None:
        st.error("Fehler beim Laden der Daten. Bitte überprüfe die Datenpfade.")
        return

    df = create_features(df)

    # Page: Datenexploration
    if page == "📊 Datenexploration":
        st.header("Datenexploration")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Anzahl Samples", len(df))
        with col2:
            st.metric("Anzahl Features", df.shape[1] - 2)  # ohne quality und quality_class
        with col3:
            st.metric("Durchschnittliche Qualität", f"{df['quality'].mean():.2f}")

        st.markdown("---")

        # Datenvorschau
        st.subheader("Datenvorschau")
        st.dataframe(df.head(20), use_container_width=True)

        # Statistische Zusammenfassung
        st.subheader("Statistische Zusammenfassung")
        st.dataframe(df.describe(), use_container_width=True)

        st.markdown("---")

        # Visualisierungen
        st.subheader("Visualisierungen")

        # Qualitätsverteilung
        col1, col2 = st.columns(2)

        with col1:
            fig1 = px.histogram(
                df, x='quality',
                title='Verteilung der Weinqualität',
                color_discrete_sequence=['steelblue']
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            quality_by_type = df.groupby(['wine_type', 'quality']).size().reset_index(name='count')
            quality_by_type['wine_type'] = quality_by_type['wine_type'].map({0: 'Weißwein', 1: 'Rotwein'})
            fig2 = px.bar(
                quality_by_type,
                x='quality',
                y='count',
                color='wine_type',
                barmode='group',
                title='Qualität: Rotwein vs Weißwein'
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Korrelations-Heatmap
        st.subheader("Korrelationsmatrix")
        corr_matrix = df.corr()

        fig3 = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect='auto',
            color_continuous_scale='RdBu_r',
            title='Korrelationsmatrix aller Features'
        )
        fig3.update_layout(height=800)
        st.plotly_chart(fig3, use_container_width=True)

        # Feature Distributions
        st.subheader("Feature Verteilungen")
        feature_to_plot = st.selectbox(
            "Wähle ein Feature:",
            [col for col in df.columns if col not in ['quality', 'quality_class']]
        )

        fig4 = px.histogram(
            df,
            x=feature_to_plot,
            marginal='box',
            title=f'Verteilung: {feature_to_plot}'
        )
        st.plotly_chart(fig4, use_container_width=True)

        # Scatter Plot
        st.subheader("Scatter Plot: Feature vs Quality")
        col1, col2 = st.columns(2)
        with col1:
            x_feature = st.selectbox(
                "X-Achse:",
                [col for col in df.columns if col not in ['quality', 'quality_class']],
                key='x_feature'
            )
        with col2:
            color_by = st.selectbox(
                "Farbe nach:",
                ['quality', 'wine_type', 'quality_class']
            )

        fig5 = px.scatter(
            df,
            x=x_feature,
            y='quality',
            color=color_by,
            title=f'Quality vs {x_feature}',
            opacity=0.6
        )
        st.plotly_chart(fig5, use_container_width=True)

    # Page: Modell Training
    elif page == "🤖 Modell Training":
        st.header("Modell Training & Evaluation")

        # Modelltyp wählen
        task = st.sidebar.selectbox(
            "Aufgabentyp:",
            ["Regression", "Klassifikation"]
        )

        # Features vorbereiten
        feature_columns = [col for col in df.columns if col not in ['quality', 'quality_class']]
        X = df[feature_columns]

        if task == "Regression":
            y = df['quality']
            st.info("**Regression:** Vorhersage des exakten Qualitätswerts (3-9)")
        else:
            y = df['quality_class']
            st.info("**Klassifikation:** Vorhersage binär - Schlecht (<6) vs Gut (≥6)")

        # Train-Test Split
        test_size = st.sidebar.slider("Test Set Größe (%)", 10, 40, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Standardisierung
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Datenaufteilung")
        st.sidebar.write(f"Training: {len(X_train)} Samples")
        st.sidebar.write(f"Test: {len(X_test)} Samples")

        # Modellauswahl
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎯 Modellauswahl")

        if task == "Regression":
            model_name = st.sidebar.selectbox(
                "Modell:",
                ['Linear Regression', 'Ridge Regression', 'Random Forest',
                 'Gradient Boosting', 'K-Nearest Neighbors', 'Support Vector Regressor']
            )
        else:
            model_name = st.sidebar.selectbox(
                "Modell:",
                ['Logistic Regression', 'Random Forest', 'Gradient Boosting',
                 'K-Nearest Neighbors', 'Support Vector Classifier']
            )

        # Hyperparameter
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚙️ Hyperparameter")

        params = {}

        if 'Random Forest' in model_name:
            params['rf_n_estimators'] = st.sidebar.slider("n_estimators", 50, 300, 100, 50)
            params['rf_max_depth'] = st.sidebar.selectbox("max_depth", [None, 10, 20, 30])
            params['rf_min_samples_split'] = st.sidebar.slider("min_samples_split", 2, 10, 2)

        elif 'Gradient Boosting' in model_name:
            params['gb_n_estimators'] = st.sidebar.slider("n_estimators", 50, 300, 100, 50)
            params['gb_learning_rate'] = st.sidebar.select_slider(
                "learning_rate", options=[0.01, 0.05, 0.1, 0.2], value=0.1
            )
            params['gb_max_depth'] = st.sidebar.slider("max_depth", 3, 10, 3)

        elif 'K-Nearest' in model_name:
            params['knn_n_neighbors'] = st.sidebar.slider("n_neighbors", 3, 15, 5)

        elif 'Ridge' in model_name:
            params['ridge_alpha'] = st.sidebar.slider("alpha", 0.1, 10.0, 1.0, 0.1)

        elif 'Logistic' in model_name:
            params['lr_C'] = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0, 0.01)

        elif 'Support Vector' in model_name:
            if task == "Regression":
                params['svr_kernel'] = st.sidebar.selectbox("kernel", ['rbf', 'linear', 'poly'])
                params['svr_C'] = st.sidebar.slider("C", 0.1, 10.0, 1.0, 0.1)
            else:
                params['svc_kernel'] = st.sidebar.selectbox("kernel", ['rbf', 'linear', 'poly'])
                params['svc_C'] = st.sidebar.slider("C", 0.1, 10.0, 1.0, 0.1)

        # Training
        if st.button("🚀 Modell trainieren", type="primary"):
            with st.spinner(f"Trainiere {model_name}..."):
                # Modell erstellen
                if task == "Regression":
                    models = get_regression_models(params)
                else:
                    models = get_classification_models(params)

                model = models[model_name]

                # Training
                model.fit(X_train_scaled, y_train)

                # Vorhersagen
                y_pred_train = model.predict(X_train_scaled)
                y_pred_test = model.predict(X_test_scaled)

                st.success("✅ Training abgeschlossen!")

                # Metriken
                st.markdown("---")
                st.subheader("📈 Modell-Performance")

                if task == "Regression":
                    # Regression Metriken
                    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                    train_mae = mean_absolute_error(y_train, y_pred_train)
                    test_mae = mean_absolute_error(y_test, y_pred_test)
                    train_r2 = r2_score(y_train, y_pred_train)
                    test_r2 = r2_score(y_test, y_pred_test)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Test RMSE", f"{test_rmse:.4f}")
                        st.metric("Train RMSE", f"{train_rmse:.4f}")
                    with col2:
                        st.metric("Test MAE", f"{test_mae:.4f}")
                        st.metric("Train MAE", f"{train_mae:.4f}")
                    with col3:
                        st.metric("Test R²", f"{test_r2:.4f}")
                        st.metric("Train R²", f"{train_r2:.4f}")

                    # Predicted vs Actual Plot
                    st.markdown("---")
                    st.subheader("Predicted vs Actual")
                    fig = plot_predicted_vs_actual(y_test, y_pred_test, model_name, test_r2)
                    st.plotly_chart(fig, use_container_width=True)

                else:
                    # Klassifikation Metriken
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

                    train_acc = accuracy_score(y_train, y_pred_train)
                    test_acc = accuracy_score(y_test, y_pred_test)
                    precision = precision_score(y_test, y_pred_test)
                    recall = recall_score(y_test, y_pred_test)
                    f1 = f1_score(y_test, y_pred_test)
                    auc_score = roc_auc_score(y_test, y_pred_proba)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Accuracy", f"{test_acc:.4f}")
                        st.metric("Precision", f"{precision:.4f}")
                    with col2:
                        st.metric("Recall", f"{recall:.4f}")
                        st.metric("F1-Score", f"{f1:.4f}")
                    with col3:
                        st.metric("AUC", f"{auc_score:.4f}")
                        st.metric("Train Accuracy", f"{train_acc:.4f}")

                    # Confusion Matrix & ROC Curve
                    st.markdown("---")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Confusion Matrix")
                        fig_cm = plot_confusion_matrix(y_test, y_pred_test, model_name)
                        st.plotly_chart(fig_cm, use_container_width=True)

                    with col2:
                        st.subheader("ROC Curve")
                        fig_roc = plot_roc_curve(y_test, y_pred_proba, model_name)
                        st.plotly_chart(fig_roc, use_container_width=True)

                # Feature Importance
                st.markdown("---")
                st.subheader("Feature Importance")
                fig_fi = plot_feature_importance(model, feature_columns, model_name)
                if fig_fi:
                    st.plotly_chart(fig_fi, use_container_width=True)
                else:
                    st.info("Dieses Modell unterstützt keine Feature Importance Analyse.")

                # Modell speichern
                st.markdown("---")
                if st.button("💾 Modell speichern"):
                    model_data = {
                        'model': model,
                        'scaler': scaler,
                        'feature_columns': feature_columns,
                        'task': task,
                        'model_name': model_name
                    }

                    filename = f'trained_model_{task.lower()}_{model_name.replace(" ", "_").lower()}.pkl'
                    with open(filename, 'wb') as f:
                        pickle.dump(model_data, f)

                    st.success(f"✅ Modell gespeichert als: {filename}")

    # Page: Vorhersage
    elif page == "🔮 Vorhersage":
        st.header("Weinqualität vorhersagen")

        # Modell-Upload
        st.sidebar.subheader("Modell laden")
        uploaded_file = st.sidebar.file_uploader(
            "Lade ein trainiertes Modell (.pkl)",
            type=['pkl']
        )

        use_trained_model = uploaded_file is not None

        if not use_trained_model:
            st.info("💡 Bitte trainiere zuerst ein Modell auf der 'Modell Training' Seite oder lade ein gespeichertes Modell hoch.")

        # Vorhersage-Modus
        prediction_mode = st.radio(
            "Vorhersage-Modus:",
            ["Einzelne Vorhersage", "Batch Vorhersage"]
        )

        if prediction_mode == "Einzelne Vorhersage":
            st.subheader("Gib die Wein-Eigenschaften ein:")

            col1, col2, col3 = st.columns(3)

            with col1:
                fixed_acidity = st.number_input("Fixed Acidity", 4.0, 16.0, 7.0, 0.1)
                volatile_acidity = st.number_input("Volatile Acidity", 0.1, 2.0, 0.5, 0.01)
                citric_acid = st.number_input("Citric Acid", 0.0, 1.0, 0.3, 0.01)
                residual_sugar = st.number_input("Residual Sugar", 0.5, 20.0, 2.0, 0.1)

            with col2:
                chlorides = st.number_input("Chlorides", 0.01, 0.5, 0.08, 0.01)
                free_sulfur = st.number_input("Free Sulfur Dioxide", 1.0, 100.0, 30.0, 1.0)
                total_sulfur = st.number_input("Total Sulfur Dioxide", 10.0, 300.0, 100.0, 1.0)
                density = st.number_input("Density", 0.99, 1.01, 0.996, 0.001)

            with col3:
                pH = st.number_input("pH", 2.5, 4.0, 3.3, 0.01)
                sulphates = st.number_input("Sulphates", 0.3, 2.0, 0.6, 0.01)
                alcohol = st.number_input("Alcohol", 8.0, 15.0, 10.0, 0.1)
                wine_type = st.selectbox("Wine Type", [0, 1], format_func=lambda x: "Weißwein" if x == 0 else "Rotwein")

            # Features berechnen
            free_to_total_ratio = free_sulfur / (total_sulfur + 1e-10)
            acid_ratio = fixed_acidity / (volatile_acidity + 1e-10)
            total_acidity = fixed_acidity + volatile_acidity + citric_acid
            alcohol_per_acid = alcohol / (total_acidity + 1e-10)

            # Input DataFrame
            input_data = pd.DataFrame({
                'fixed acidity': [fixed_acidity],
                'volatile acidity': [volatile_acidity],
                'citric acid': [citric_acid],
                'residual sugar': [residual_sugar],
                'chlorides': [chlorides],
                'free sulfur dioxide': [free_sulfur],
                'total sulfur dioxide': [total_sulfur],
                'density': [density],
                'pH': [pH],
                'sulphates': [sulphates],
                'alcohol': [alcohol],
                'wine_type': [wine_type],
                'free_to_total_sulfur_ratio': [free_to_total_ratio],
                'acid_ratio': [acid_ratio],
                'total_acidity': [total_acidity],
                'alcohol_per_acid': [alcohol_per_acid]
            })

            if st.button("🔮 Vorhersage erstellen", type="primary"):
                if use_trained_model:
                    # Lade das Modell
                    model_data = pickle.load(uploaded_file)
                    model = model_data['model']
                    scaler = model_data['scaler']
                    task = model_data['task']
                    model_name = model_data['model_name']

                    # Skalierung
                    input_scaled = scaler.transform(input_data)

                    # Vorhersage
                    prediction = model.predict(input_scaled)[0]

                    st.markdown("---")
                    st.subheader("Ergebnis:")

                    if task == "Regression":
                        st.success(f"**Vorhergesagte Qualität: {prediction:.2f}**")

                        # Gauge Chart
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=prediction,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Weinqualität"},
                            gauge={
                                'axis': {'range': [0, 10]},
                                'bar': {'color': "darkred"},
                                'steps': [
                                    {'range': [0, 4], 'color': "lightgray"},
                                    {'range': [4, 6], 'color': "gray"},
                                    {'range': [6, 8], 'color': "lightgreen"},
                                    {'range': [8, 10], 'color': "darkgreen"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 6
                                }
                            }
                        ))
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        prediction_proba = model.predict_proba(input_scaled)[0]
                        class_label = "Gut (≥6)" if prediction == 1 else "Schlecht (<6)"
                        confidence = prediction_proba[int(prediction)] * 100

                        st.success(f"**Vorhersage: {class_label}** (Konfidenz: {confidence:.1f}%)")

                        # Probability Chart
                        fig = go.Figure(data=[
                            go.Bar(
                                x=['Schlecht (<6)', 'Gut (≥6)'],
                                y=prediction_proba * 100,
                                marker_color=['red', 'green']
                            )
                        ])
                        fig.update_layout(
                            title="Klassifikations-Wahrscheinlichkeiten",
                            yaxis_title="Wahrscheinlichkeit (%)",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Bitte lade ein trainiertes Modell hoch oder trainiere ein neues Modell.")

        else:  # Batch Vorhersage
            st.subheader("Batch Vorhersage")
            st.info("Lade eine CSV-Datei mit mehreren Weinen zur Vorhersage hoch.")

            uploaded_batch = st.file_uploader("CSV-Datei hochladen", type=['csv'])

            if uploaded_batch and use_trained_model:
                batch_df = pd.read_csv(uploaded_batch)
                st.write("Geladene Daten:")
                st.dataframe(batch_df.head())

                if st.button("🚀 Batch Vorhersage starten"):
                    # Lade das Modell
                    model_data = pickle.load(uploaded_file)
                    model = model_data['model']
                    scaler = model_data['scaler']
                    task = model_data['task']

                    # Features engineering für batch
                    batch_df_eng = create_features(batch_df)
                    feature_columns = [col for col in batch_df_eng.columns
                                     if col not in ['quality', 'quality_class']]
                    X_batch = batch_df_eng[feature_columns]

                    # Skalierung
                    X_batch_scaled = scaler.transform(X_batch)

                    # Vorhersagen
                    predictions = model.predict(X_batch_scaled)

                    # Ergebnisse hinzufügen
                    if task == "Regression":
                        batch_df['Predicted Quality'] = predictions
                    else:
                        batch_df['Predicted Class'] = predictions
                        batch_df['Class Label'] = batch_df['Predicted Class'].map(
                            {0: 'Schlecht (<6)', 1: 'Gut (≥6)'}
                        )

                    st.success("✅ Vorhersagen abgeschlossen!")
                    st.dataframe(batch_df)

                    # Download
                    csv = batch_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Ergebnisse herunterladen",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )


if __name__ == "__main__":
    main()
