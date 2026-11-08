"""
src/model_training.py
Entrenamiento, optimización, validación cruzada estratificada de 10 pliegues (10-Fold Stratified CV)
y evaluación comparativa de los seis modelos del artículo base (Miah et al., 2024) frente a dos líneas base.
"""

import os
import sys

# Asegurar que el directorio raíz del proyecto esté en el PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix, ConfusionMatrixDisplay
)
import shap

from src.pipeline_builder import get_train_test_data, build_full_model_pipeline, NUMERIC_FEATURES, CATEGORICAL_FEATURES

def define_candidate_models():
    """
    Define los seis modelos avanzados adaptados del artículo base junto con dos líneas base:
      - Línea Base 1: Clasificador de Mayoría (Dummy)
      - Línea Base 2: Regresión Logística Simple ML1 (Edad, Sexo, IMC únicamente)
      - Modelo 1: Regresión Logística Regularizada (L2)
      - Modelo 2: SVM con Kernel RBF
      - Modelo 3: Árbol de Decisión (CART podado)
      - Modelo 4: Bagging / Random Forest
      - Modelo 5: XGBoost (Gradient Boosting)
      - Modelo 6: LightGBM (Histogram Gradient Boosting)
    """
    models = {
        'Línea Base: Mayoría (Dummy)': {
            'estimator': DummyClassifier(strategy='most_frequent'),
            'type': 'baseline_dummy',
            'features': 'all'
        },
        'Línea Base: Logística Simple (ML1)': {
            'estimator': LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000, random_state=42),
            'type': 'baseline_simple',
            'features': 'simple'  # solo edad, sexo, imc
        },
        'Regresión Logística (L2)': {
            'estimator': LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000, random_state=42),
            'type': 'advanced',
            'features': 'all'
        },
        'SVM (Kernel RBF)': {
            'estimator': SVC(C=1.5, kernel='rbf', probability=True, random_state=42),
            'type': 'advanced',
            'features': 'all'
        },
        'Árbol de Decisión (CART)': {
            'estimator': DecisionTreeClassifier(max_depth=5, min_samples_split=25, min_samples_leaf=15, criterion='gini', random_state=42),
            'type': 'advanced',
            'features': 'all'
        },
        'Bagging (Random Forest)': {
            'estimator': RandomForestClassifier(n_estimators=200, max_depth=9, min_samples_split=15, random_state=42, n_jobs=-1),
            'type': 'advanced',
            'features': 'all'
        },
        'XGBoost': {
            'estimator': XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.06, subsample=0.85, colsample_bytree=0.85, eval_metric='logloss', random_state=42),
            'type': 'advanced',
            'features': 'all'
        },
        'LightGBM': {
            'estimator': LGBMClassifier(n_estimators=150, num_leaves=25, max_depth=5, learning_rate=0.06, subsample=0.85, random_state=42, verbose=-1),
            'type': 'advanced',
            'features': 'all'
        }
    }
    return models

def evaluate_models_10fold_cv(X_train, y_train, models):
    """
    Ejecuta la validación cruzada estratificada de 10 pliegues (10-Fold Stratified CV)
    para cada modelo, registrando media y desviación estándar de las 5 métricas requeridas.
    """
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    cv_summary = []
    fitted_pipelines = {}
    
    print("\n" + "="*80)
    print("EJECUTANDO VALIDACIÓN CRUZADA ESTRATIFICADA DE 10 PLIEGUES (10-FOLD CV)")
    print("="*80)
    
    for name, config in models.items():
        print(f"Evaluando: {name} ...", end=" ", flush=True)
        estimator = config['estimator']
        
        # Configurar características según el tipo de modelo
        if config['features'] == 'simple':
            pipe = build_full_model_pipeline(
                estimator,
                numeric_features=['edad', 'imc'],
                categorical_features=['sexo']
            )
        else:
            pipe = build_full_model_pipeline(estimator)
            
        # Ejecutar cross_validate sin fuga de datos
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
        
        acc_mean, acc_std = np.mean(scores['test_accuracy']), np.std(scores['test_accuracy'])
        prec_mean, prec_std = np.mean(scores['test_precision']), np.std(scores['test_precision'])
        rec_mean, rec_std = np.mean(scores['test_recall']), np.std(scores['test_recall'])
        f1_mean, f1_std = np.mean(scores['test_f1']), np.std(scores['test_f1'])
        auc_mean, auc_std = np.mean(scores['test_roc_auc']), np.std(scores['test_roc_auc'])
        
        cv_summary.append({
            'Modelo': name,
            'Accuracy': f"{acc_mean*100:.2f} ± {acc_std*100:.2f}%",
            'Precisión': f"{prec_mean*100:.2f} ± {prec_std*100:.2f}%",
            'Recall (Sensibilidad)': f"{rec_mean*100:.2f} ± {rec_std*100:.2f}%",
            'F1-Score': f"{f1_mean*100:.2f} ± {f1_std*100:.2f}%",
            'ROC-AUC': f"{auc_mean*100:.2f} ± {auc_std*100:.2f}%",
            'auc_raw_mean': auc_mean,
            'rec_raw_mean': rec_mean,
            'acc_raw_mean': acc_mean
        })
        
        # Ajustar sobre el 100% de X_train para evaluación en conjunto de prueba
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe
        print("Listo.")
        
    summary_df = pd.DataFrame(cv_summary)
    summary_df.sort_values(by='auc_raw_mean', ascending=False, inplace=True)
    summary_df.to_csv("data/processed/tabla_comparativa_modelos.csv", index=False)
    
    print("\n" + "="*95)
    print("TABLA COMPARATIVA FINAL DE MODELOS (10-FOLD STRATIFIED CV SOBRE CONJUNTO DE ENTRENAMIENTO)")
    print("="*95)
    display_cols = ['Modelo', 'Accuracy', 'Precisión', 'Recall (Sensibilidad)', 'F1-Score', 'ROC-AUC']
    print(summary_df[display_cols].to_string(index=False))
    print("="*95 + "\n")
    
    return summary_df, fitted_pipelines

def evaluate_on_test_and_generate_figures(fitted_pipelines, X_test, y_test, figures_dir="assets/figures"):
    """
    Evalúa los modelos ajustados sobre el conjunto de prueba independiente (2.000 observaciones).
    Genera Curvas ROC comparativas, Matriz de Confusión y Análisis SHAP/Importancia del mejor modelo.
    """
    os.makedirs(figures_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # Figura 1: Curvas ROC Comparativas Superpuestas (Miah et al., Fig. 3 adaptada)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 7.5), dpi=300)
    
    palette = {
        'Línea Base: Mayoría (Dummy)': '#999999',
        'Línea Base: Logística Simple (ML1)': '#8c564b',
        'Regresión Logística (L2)': '#1f77b4',
        'SVM (Kernel RBF)': '#9467bd',
        'Árbol de Decisión (CART)': '#ff7f0e',
        'Bagging (Random Forest)': '#2ca02c',
        'XGBoost': '#d62728',
        'LightGBM': '#17becf'
    }
    
    test_metrics = {}
    for name, pipe in fitted_pipelines.items():
        if hasattr(pipe, "predict_proba"):
            y_pred_proba = pipe.predict_proba(X_test)[:, 1]
            auc_test = roc_auc_score(y_test, y_pred_proba)
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            test_metrics[name] = auc_test
            linestyle = '--' if 'Línea Base' in name else '-'
            linewidth = 1.5 if 'Línea Base' in name else 2.2
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_test*100:.2f}%)",
                     color=palette.get(name, '#333333'), linestyle=linestyle, linewidth=linewidth)
        else:
            y_pred = pipe.predict(X_test)
            auc_test = roc_auc_score(y_test, y_pred)
            test_metrics[name] = auc_test
            plt.plot([0, 1], [0, 1], label=f"{name} (AUC = {auc_test*100:.2f}%)",
                     color=palette.get(name, '#999999'), linestyle=':', linewidth=1.5)
            
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Clasificador Aleatorio (AUC = 50.00%)')
    plt.title('Curvas ROC Comparativas - Conjunto de Prueba Independiente (N = 2.000)\n(Adaptación y Reproducción de Miah et al., 2024 en Colombia)',
              fontsize=12.5, weight='bold', pad=12)
    plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)', fontsize=11, weight='bold')
    plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad / Recall)', fontsize=11, weight='bold')
    plt.legend(loc='lower right', frameon=True, fontsize=9.5)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    roc_fig_path = os.path.join(figures_dir, "eval_01_curvas_roc.png")
    plt.savefig(roc_fig_path, dpi=300)
    plt.close()
    print(f"Curvas ROC guardadas en: {roc_fig_path}")

    # Determinar el mejor modelo (mayor AUC en conjunto de prueba)
    best_model_name = max(test_metrics, key=test_metrics.get)
    best_pipe = fitted_pipelines[best_model_name]
    print(f"\nModelo Ganador: {best_model_name} con ROC-AUC en Test de {test_metrics[best_model_name]*100:.2f}%")
    
    # -------------------------------------------------------------------------
    # Figura 2: Matriz de Confusión Normalizada del Mejor Modelo
    # -------------------------------------------------------------------------
    y_test_pred = best_pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_test_pred)
    cm_norm = confusion_matrix(y_test, y_test_pred, normalize='true')
    
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Sin HTA (0)', 'Con HTA (1)'])
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    ax.set_title(f'Matriz de Confusión - {best_model_name}\n(Evaluación sobre Conjunto de Prueba N = 2.000)',
                 fontsize=11.5, weight='bold', pad=12)
    ax.set_xlabel('Predicción del Modelo', fontsize=11, weight='bold')
    ax.set_ylabel('Diagnóstico Real Clínico', fontsize=11, weight='bold')
    
    # Añadir porcentajes normalizados en el gráfico
    for i in range(2):
        for j in range(2):
            ax.text(j, i + 0.25, f"({cm_norm[i, j]*100:.1f}%)", ha="center", va="center", color="gray", fontsize=10)
            
    plt.tight_layout()
    cm_fig_path = os.path.join(figures_dir, "eval_02_matriz_confusion.png")
    plt.savefig(cm_fig_path, dpi=300)
    plt.close()
    print(f"Matriz de Confusión guardada en: {cm_fig_path}")

    # -------------------------------------------------------------------------
    # Figura 3: Importancia de Variables y Coeficientes Clínicos (Doble Panel)
    # -------------------------------------------------------------------------
    print("Calculando Importancia de Características y Coeficientes Clínicos...")
    prep = best_pipe.named_steps['preprocessor']
    classifier = best_pipe.named_steps['classifier']
    
    # Obtener nombres de características transformadas
    cat_encoder = prep.named_transformers_['cat'].named_steps['encoder']
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    all_feature_names = NUMERIC_FEATURES + cat_feature_names
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300)
    
    # Panel A: Coeficientes del Mejor Modelo Lineal (Regresión Logística L2)
    lr_pipe = fitted_pipelines.get('Regresión Logística (L2)', best_pipe)
    lr_clf = lr_pipe.named_steps['classifier']
    if hasattr(lr_clf, 'coef_'):
        coefs = lr_clf.coef_[0]
        coef_df = pd.DataFrame({'Característica': all_feature_names, 'Coeficiente': coefs})
        coef_df['Abs_Coef'] = coef_df['Coeficiente'].abs()
        coef_df.sort_values(by='Coeficiente', ascending=True, inplace=True)
        
        colors = ['#d9534f' if c > 0 else '#2b5c8f' for c in coef_df['Coeficiente']]
        bars1 = axes[0].barh(coef_df['Característica'], coef_df['Coeficiente'], color=colors, edgecolor='black', height=0.65)
        axes[0].axvline(0, color='black', linewidth=0.8, linestyle='--')
        axes[0].set_title('A. Coeficientes Estandarizados (Log-Odds)\nRegresión Logística (L2) - Mejor Modelo', fontsize=11.5, weight='bold', pad=12)
        axes[0].set_xlabel('Magnitud del Efecto en Log-Odds de HTA', fontsize=10.5, weight='bold')
        for b in bars1:
            w = b.get_width()
            offset = 0.02 if w >= 0 else -0.02
            ha = 'left' if w >= 0 else 'right'
            axes[0].text(w + offset, b.get_y() + b.get_height()/2, f"{w:+.2f}", va='center', ha=ha, fontsize=8.5, weight='bold')
            
    # Panel B: Importancia Relativa de Características en Ensamble (XGBoost)
    xgb_pipe = fitted_pipelines.get('XGBoost')
    if xgb_pipe and hasattr(xgb_pipe.named_steps['classifier'], 'feature_importances_'):
        xgb_clf = xgb_pipe.named_steps['classifier']
        xgb_imp = xgb_clf.feature_importances_
        imp_df = pd.DataFrame({'Característica': all_feature_names, 'Importancia': xgb_imp})
        imp_df.sort_values(by='Importancia', ascending=True, inplace=True)
        
        bars2 = axes[1].barh(imp_df['Característica'], imp_df['Importancia'], color='#2ca02c', edgecolor='black', height=0.65)
        axes[1].set_title('B. Ganancia de Información Relativa\nXGBoost - Mejor Modelo Ensamble', fontsize=11.5, weight='bold', pad=12)
        axes[1].set_xlabel('Importancia Relativa (Gain)', fontsize=10.5, weight='bold')
        for b in bars2:
            w = b.get_width()
            axes[1].text(w + 0.005, b.get_y() + b.get_height()/2, f"{w:.3f}", va='center', ha='left', fontsize=8.5, weight='bold')
        axes[1].set_xlim(0, imp_df['Importancia'].max() * 1.18)
        
    plt.tight_layout()
    imp_fig_path = os.path.join(figures_dir, "eval_03_importancia_variables_shap.png")
    plt.savefig(imp_fig_path, dpi=300)
    plt.close()
    print(f"Gráfico de Importancia guardado en: {imp_fig_path}")
        
    # Guardar el mejor modelo entrenado en disco
    os.makedirs("models", exist_ok=True)
    model_save_path = "models/mejor_modelo_hipertension.joblib"
    joblib.dump(best_pipe, model_save_path)
    print(f"Mejor modelo guardado en: {model_save_path}")

def run_full_modeling_pipeline():
    X_train, X_test, y_train, y_test = get_train_test_data()
    models = define_candidate_models()
    summary_df, fitted_pipelines = evaluate_models_10fold_cv(X_train, y_train, models)
    evaluate_on_test_and_generate_figures(fitted_pipelines, X_test, y_test)
    return summary_df

if __name__ == "__main__":
    # Las rutas del proyecto son relativas a la raíz: se ejecuta igual desde cualquier carpeta
    os.chdir(project_root)
    run_full_modeling_pipeline()
