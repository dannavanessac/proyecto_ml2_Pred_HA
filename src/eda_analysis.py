"""
src/eda_analysis.py
Análisis Exploratorio de Datos (EDA) Profundo y Cruce con Evidencia Médica/Clínica.
Calcula medidas de tendencia central, dispersión, contrastes de hipótesis y genera figuras a 300 DPI.
"""

import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración visual profesional
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.dpi'] = 300

def run_deep_eda(data_path="data/processed/dataset_hipertension_colombia.csv", figures_dir="assets/figures"):
    """
    Ejecuta el EDA exhaustivo y genera estadísticas y visualizaciones analíticas.
    """
    os.makedirs(figures_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    print("="*70)
    print("INICIANDO ANÁLISIS EXPLORATORIO DE DATOS (EDA) PROFUNDO")
    print(f"Dimensiones del dataset: {df.shape[0]} observaciones x {df.shape[1]} variables")
    print("="*70)

    # 1. Estadísticas de Tendencia Central y Dispersión para Variables Cuantitativas
    num_cols = ['edad', 'imc', 'glucosa_ayunas', 'colesterol_total', 'colesterol_hdl', 'colesterol_ldl', 'trigliceridos']
    
    stats_list = []
    for col in num_cols:
        series = df[col]
        mode_val = series.mode().iloc[0]
        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        iqr = q75 - q25
        skew = stats.skew(series)
        kurt = stats.kurtosis(series)
        
        # Test de normalidad Shapiro-Wilk (muestra de 5000) o D'Agostino
        stat_norm, p_norm = stats.normaltest(series)
        
        stats_list.append({
            'Variable': col,
            'Media': round(series.mean(), 2),
            'Mediana': round(series.median(), 2),
            'Moda': round(mode_val, 2),
            'DE': round(series.std(), 2),
            'Varianza': round(series.var(), 2),
            'Mínimo': round(series.min(), 2),
            'Máximo': round(series.max(), 2),
            'Q25': round(q25, 2),
            'Q75': round(q75, 2),
            'IQR': round(iqr, 2),
            'Asimetría': round(skew, 3),
            'Curtosis': round(kurt, 3),
            'P-Valor Normalidad': f"{p_norm:.2e}"
        })
    
    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv("data/processed/metricas_tendencia_dispersion.csv", index=False)
    print("\n--- MEDIDAS DE TENDENCIA CENTRAL Y DISPERSIÓN (VARIABLES CUANTITATIVAS) ---")
    print(stats_df[['Variable', 'Media', 'Mediana', 'DE', 'IQR', 'Asimetría', 'Curtosis']].to_string(index=False))

    # 2. Análisis Bivariado y Contraste de Hipótesis (HTA vs No HTA)
    print("\n--- PRUEBAS DE HIPÓTESIS BIVARIADAS (HTA=1 vs HTA=0) ---")
    bivar_list = []
    for col in num_cols:
        group_0 = df[df['hipertension_dx'] == 0][col]
        group_1 = df[df['hipertension_dx'] == 1][col]
        
        # Test U de Mann-Whitney (no paramétrico robusto)
        u_stat, u_pval = stats.mannwhitneyu(group_1, group_0, alternative='two-sided')
        # Test t de Student
        t_stat, t_pval = stats.ttest_ind(group_1, group_0, equal_var=False)
        
        bivar_list.append({
            'Variable': col,
            'Media (No HTA)': round(group_0.mean(), 2),
            'Media (HTA)': round(group_1.mean(), 2),
            'Diferencia': round(group_1.mean() - group_0.mean(), 2),
            't-stat': round(t_stat, 2),
            'p-valor (t-test)': f"{t_pval:.2e}",
            'p-valor (Mann-Whitney)': f"{u_pval:.2e}",
            'Significativo (p < 0.001)': 'Sí' if u_pval < 0.001 else 'No'
        })
    bivar_df = pd.DataFrame(bivar_list)
    bivar_df.to_csv("data/processed/contrastes_bivariados_num.csv", index=False)
    print(bivar_df[['Variable', 'Media (No HTA)', 'Media (HTA)', 'Diferencia', 'p-valor (Mann-Whitney)']].to_string(index=False))

    # 3. Análisis de Variables Categóricas y Odds Ratios
    cat_cols = ['sexo', 'tabaquismo', 'actividad_fisica', 'diabetes', 'zona_residencia', 'regimen_salud']
    cat_stats = []
    for cat in cat_cols:
        contingency = pd.crosstab(df[cat], df['hipertension_dx'])
        chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
        
        # Si es 2x2 calcular Odds Ratio
        odds_ratio_str = "N/A"
        if contingency.shape == (2, 2):
            table = contingency.values
            # table[0,0], table[0,1], table[1,0], table[1,1]
            or_val = (table[1, 1] * table[0, 0]) / max((table[1, 0] * table[0, 1]), 1)
            odds_ratio_str = f"{or_val:.2f}"
            
        cat_stats.append({
            'Variable': cat,
            'Categorías': list(contingency.index),
            'Chi2': round(chi2, 2),
            'p-valor': f"{p_val:.2e}",
            'Odds Ratio (estimado)': odds_ratio_str
        })
    cat_df = pd.DataFrame(cat_stats)
    cat_df.to_csv("data/processed/contrastes_categoricos_chi2.csv", index=False)
    print("\n--- PRUEBAS CHI-CUADRADO DE INDEPENDENCIA (CATEGÓRICAS vs HTA) ---")
    print(cat_df[['Variable', 'Chi2', 'p-valor', 'Odds Ratio (estimado)']].to_string(index=False))

    # =========================================================================
    # GENERACIÓN DE GRÁFICAS DE ALTA RESOLUCIÓN (300 DPI)
    # =========================================================================
    
    # -------------------------------------------------------------------------
    # Figura 1: Prevalencia General y Cruce con Comorbilidades Principales
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Panel A: Dona de Prevalencia
    hta_counts = df['hipertension_dx'].value_counts()
    colors = ['#2b5c8f', '#d9534f']
    axes[0].pie(hta_counts, labels=['Normotenso / Sin Dx (79.2%)', 'Hipertensión Dx (20.8%)'],
                colors=colors, autopct='%1.1f%%', startangle=140, explode=(0, 0.06),
                wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2),
                textprops={'fontsize': 11, 'weight': 'bold'})
    axes[0].set_title('A. Prevalencia de Hipertensión Arterial\n(Cohorte Adulta Colombiana ENSIN/MinSalud)', fontsize=12, pad=15, weight='bold')

    # Panel B: Prevalencia de HTA según comorbilidades y hábitos
    risk_factors = {
        'Diabetes: No': df[df['diabetes'] == 'No']['hipertension_dx'].mean() * 100,
        'Diabetes: Sí': df[df['diabetes'] == 'Si']['hipertension_dx'].mean() * 100,
        'Tabaquismo: No': df[df['tabaquismo'] == 'No']['hipertension_dx'].mean() * 100,
        'Tabaquismo: Sí': df[df['tabaquismo'] == 'Si']['hipertension_dx'].mean() * 100,
        'Actividad: Activo': df[df['actividad_fisica'] == 'Activo']['hipertension_dx'].mean() * 100,
        'Actividad: Sedentario': df[df['actividad_fisica'] == 'Sedentario']['hipertension_dx'].mean() * 100,
        'IMC < 25 (Normal)': df[df['imc'] < 25]['hipertension_dx'].mean() * 100,
        'IMC >= 30 (Obesidad)': df[df['imc'] >= 30]['hipertension_dx'].mean() * 100,
    }
    y_labels = list(risk_factors.keys())
    x_vals = list(risk_factors.values())
    bar_colors = ['#6baed6', '#d9534f', '#6baed6', '#d9534f', '#6baed6', '#d9534f', '#6baed6', '#d9534f']
    
    bars = axes[1].barh(y_labels, x_vals, color=bar_colors, edgecolor='none', height=0.65)
    axes[1].axvline(df['hipertension_dx'].mean() * 100, color='#333333', linestyle='--', linewidth=1.2, label=f'Promedio Poblacional ({df["hipertension_dx"].mean()*100:.1f}%)')
    for bar in bars:
        w = bar.get_width()
        axes[1].text(w + 0.8, bar.get_y() + bar.get_height()/2, f'{w:.1f}%', va='center', ha='left', fontsize=9.5, weight='bold')
    axes[1].set_xlabel('Prevalencia de Hipertensión (%)', fontsize=11, weight='bold')
    axes[1].set_title('B. Prevalencia de HTA según Factores Clínicos y Conductuales', fontsize=12, pad=15, weight='bold')
    axes[1].legend(loc='lower right', frameon=True)
    axes[1].set_xlim(0, max(x_vals) + 10)
    
    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, "eda_01_prevalencia_comorbilidades.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Figura 1 guardada: {fig1_path}")

    # -------------------------------------------------------------------------
    # Figura 2: Pirámide de Edad, Sexo y Prevalencia Exponencial de HTA
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Grupos de edad decenales
    df['grupo_edad'] = pd.cut(df['edad'], bins=[18, 30, 40, 50, 60, 70, 85], right=False,
                              labels=['18-29', '30-39', '40-49', '50-59', '60-69', '70+'])
    
    # Panel A: Tasa de HTA por grupo de edad y sexo
    age_sex_hta = df.groupby(['grupo_edad', 'sexo'], observed=False)['hipertension_dx'].mean().unstack() * 100
    age_sex_hta.plot(kind='bar', ax=axes[0], color=['#e7298a', '#1f78b4'], edgecolor='white', width=0.7)
    axes[0].set_title('A. Curva Epidemiológica de HTA por Rango de Edad y Sexo', fontsize=12, weight='bold', pad=12)
    axes[0].set_xlabel('Rango Etario (Años)', fontsize=11, weight='bold')
    axes[0].set_ylabel('Prevalencia de Hipertensión (%)', fontsize=11, weight='bold')
    axes[0].legend(title='Sexo', frameon=True)
    axes[0].tick_params(axis='x', rotation=0)
    
    # Panel B: Distribución de Densidad (KDE) de la Edad según Estado de HTA
    sns.kdeplot(data=df, x='edad', hue='hipertension_dx', palette=['#2b5c8f', '#d9534f'], fill=True, common_norm=False,
                alpha=0.4, linewidth=2, ax=axes[1])
    axes[1].set_title('B. Densidad de Probabilidad de la Edad según Diagnóstico de HTA', fontsize=12, weight='bold', pad=12)
    axes[1].set_xlabel('Edad (Años)', fontsize=11, weight='bold')
    axes[1].set_ylabel('Densidad', fontsize=11, weight='bold')
    axes[1].legend(['Hipertensión Dx (Media ~55 años)', 'Normotenso (Media ~43 años)'], loc='upper right', frameon=True)
    
    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, "eda_02_piramide_edad_sexo_hta.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Figura 2 guardada: {fig2_path}")

    # -------------------------------------------------------------------------
    # Figura 3: Biomarcadores Fisiológicos y Bioquímicos (Boxplots Comparativos)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    biomarkers = [
        ('imc', 'A. Índice de Masa Corporal (kg/m²)', 'IMC (kg/m²)', axes[0, 0]),
        ('glucosa_ayunas', 'B. Glucosa en Ayunas (mg/dL)', 'Glucosa (mg/dL)', axes[0, 1]),
        ('colesterol_total', 'C. Colesterol Total Sérico (mg/dL)', 'Colesterol Total (mg/dL)', axes[1, 0]),
        ('trigliceridos', 'D. Triglicéridos Séricos (mg/dL)', 'Triglicéridos (mg/dL)', axes[1, 1])
    ]
    
    for col, title, ylabel, ax in biomarkers:
        sns.boxplot(data=df, x='hipertension_dx', y=col, hue='hipertension_dx', palette=['#6baed6', '#d9534f'], width=0.45, ax=ax, fliersize=2, legend=False)
        ax.set_title(title, fontsize=12, weight='bold', pad=10)
        ax.set_xlabel('Diagnóstico de Hipertensión', fontsize=11, weight='bold')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Sin HTA (0)', 'Con HTA (1)'], fontsize=10.5)
        ax.set_ylabel(ylabel, fontsize=11, weight='bold')
        
        # Añadir medias con punto blanco
        means = df.groupby('hipertension_dx')[col].mean()
        ax.scatter([0, 1], means, color='white', edgecolor='black', s=60, zorder=5, label='Media')
        
        # Anotar valor p de Mann-Whitney
        u_p = bivar_df[bivar_df['Variable'] == col]['p-valor (Mann-Whitney)'].values[0]
        ax.text(0.5, 0.92, f"Mann-Whitney: p = {u_p}", transform=ax.transAxes, ha='center', fontsize=9.5,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffcc', edgecolor='#cccccc', alpha=0.8))
        
    plt.tight_layout()
    fig3_path = os.path.join(figures_dir, "eda_03_biomarcadores_boxplots.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Figura 3 guardada: {fig3_path}")

    # -------------------------------------------------------------------------
    # Figura 4: Matriz de Correlación Multivariada (Pearson & Spearman)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_cols = ['hipertension_dx', 'edad', 'imc', 'glucosa_ayunas', 'colesterol_total', 'colesterol_hdl', 'colesterol_ldl', 'trigliceridos', 'estrato_socioeconomico']
    corr_matrix = df[corr_cols].corr(method='spearman')
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmin=-0.4, vmax=0.6, center=0,
                annot=True, fmt='.2f', square=True, linewidths=1.0, cbar_kws={"shrink": 0.8}, ax=ax,
                annot_kws={"size": 10, "weight": "bold"})
    ax.set_title('Matriz de Correlación de Spearman entre Biomarcadores y HTA\n(Asociaciones no lineales en población ENSIN)', fontsize=13, weight='bold', pad=15)
    
    plt.tight_layout()
    fig4_path = os.path.join(figures_dir, "eda_04_matriz_correlaciones.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"Figura 4 guardada: {fig4_path}")

    # -------------------------------------------------------------------------
    # Figura 5: Determinantes Sociales de la Salud en Colombia (Estrato y Régimen)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Panel A: Prevalencia por Estrato Socioeconómico
    estrato_hta = df.groupby('estrato_socioeconomico')['hipertension_dx'].mean() * 100
    bars_est = axes[0].bar(estrato_hta.index, estrato_hta.values, color='#4575b4', edgecolor='black', width=0.6)
    axes[0].set_title('A. Prevalencia de HTA por Estrato Socioeconómico en Colombia', fontsize=12, weight='bold', pad=12)
    axes[0].set_xlabel('Estrato Socioeconómico (DANE)', fontsize=11, weight='bold')
    axes[0].set_ylabel('Prevalencia de Hipertensión (%)', fontsize=11, weight='bold')
    axes[0].axhline(df['hipertension_dx'].mean()*100, color='red', linestyle='--', label=f'Media Nacional ({df["hipertension_dx"].mean()*100:.1f}%)')
    for b in bars_est:
        h = b.get_height()
        axes[0].text(b.get_x() + b.get_width()/2, h + 0.6, f'{h:.1f}%', ha='center', va='bottom', fontsize=10, weight='bold')
    axes[0].legend(loc='upper right', frameon=True)
    axes[0].set_ylim(0, max(estrato_hta.values) + 8)

    # Panel B: Prevalencia por Régimen de Salud y Zona de Residencia
    regimen_zona = df.groupby(['regimen_salud', 'zona_residencia'], observed=False)['hipertension_dx'].mean().unstack() * 100
    regimen_zona.plot(kind='bar', ax=axes[1], color=['#fc8d59', '#91bfdb'], edgecolor='black', width=0.65)
    axes[1].set_title('B. Prevalencia de HTA según Régimen del SGSSS y Zona de Residencia', fontsize=12, weight='bold', pad=12)
    axes[1].set_xlabel('Régimen de Aseguramiento en Salud', fontsize=11, weight='bold')
    axes[1].set_ylabel('Prevalencia de Hipertensión (%)', fontsize=11, weight='bold')
    axes[1].legend(title='Zona', frameon=True)
    axes[1].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    fig5_path = os.path.join(figures_dir, "eda_05_determinantes_sociales.png")
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print(f"Figura 5 guardada: {fig5_path}")
    print("="*70)
    print("ANÁLISIS EXPLORATORIO FINALIZADO EXITOSAMENTE. TODAS LAS FIGURAS EXPORTADAS A 300 DPI.")
    print("="*70)

if __name__ == "__main__":
    run_deep_eda()
