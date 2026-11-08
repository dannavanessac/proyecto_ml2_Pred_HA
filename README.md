# Predicción de Hipertensión Arterial en Población Colombiana mediante Modelos Avanzados de Aprendizaje Automático

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1%2B-red.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.5%2B-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![Fase 2](https://img.shields.io/badge/Proyecto-Segunda%20Entrega%20Completada-success.svg)]()

> **Proyecto Integrador de Machine Learning II**  
> **Programa de Ciencia de Datos**  
> **Universidad Externado de Colombia**  
> **Docente:** Wilmer Darío Pineda Ríos  
> **Integrantes:**  
> - Allison Michelle Loango Rayo  
> - Danna Vanessa Caballero Urrego  
> **Fecha:** Septiembre de 2026  

---

## 📌 1. Descripción y Planteamiento del Proyecto

Este repositorio contiene el desarrollo integral de la **Segunda Entrega (Fase 2: Datos, Modelamiento y Validación)** del proyecto integrador de Machine Learning II. El trabajo consiste en la **reproducción y adaptación rigurosa al contexto colombiano** del artículo científico:

> **Miah, J., Ca, D. M., Lipu, E. R., Sayed, M. A., Mahmud, F., & Arafat, S. M. Y. (2024).** *"Improving Cardiovascular Disease Prediction Through Comparative Analysis of Machine Learning Models: A Case Study on Myocardial Infarction"*. IEEE / arXiv:2311.00517.

### Adaptación del Problema Clínico
Mientras que el artículo base se enfocó en predecir mortalidad hospitalaria por infarto en una muestra reducida de Bangladesh ($n = 600$), en este proyecto adaptamos la tarea hacia un problema de **tamizaje predictivo poblacional de alto impacto en salud pública en Colombia**: la predicción del diagnóstico de **hipertensión arterial (`HIPERTENSION_DX`)**, la principal causa subyacente de morbimortalidad cardiocerebrovascular en el país.

---

## 📂 2. Estructura del Repositorio

El proyecto sigue una arquitectura modular y profesional de ciencia de datos:

```text
SEGUNDA ENTREGA PROYECTO ML2/
├── README.md                                  # Documentación general del repositorio
├── requirements.txt                           # Dependencias reproducibles del entorno
├── ML2 - Proyecto.pdf                         # Guía oficial de la asignatura
├── Primera_Entrega_Prediccion_Cardiovascular.pdf # Documento de la primera entrega
├── data/
│   ├── raw/
│   │   └── bucaramanga_cronicas_sample.csv    # Referencia de datos abiertos Colombia (ID: 4iz7-suhz)
│   └── processed/
│       ├── dataset_hipertension_colombia.csv  # Cohorte analítica consolidada (N=10.000, 16 columnas)
│       ├── metricas_tendencia_dispersion.csv  # Estadísticas descriptivas de biomarcadores
│       ├── contrastes_bivariados_num.csv      # Contrastes Mann-Whitney U y t-Student
│       ├── contrastes_categoricos_chi2.csv    # Pruebas Chi-cuadrado y Odds Ratios
│       └── tabla_comparativa_modelos.csv      # Métricas de los 8 modelos en 10-Fold CV
├── notebooks/
│   └── Segunda_Entrega_Hipertension_ML2.ipynb # Notebook ejecutable con análisis, salidas y figuras
├── src/
│   ├── data_generation.py                     # Generación y calibración empírica (ENSIN 2015/MinSalud)
│   ├── eda_analysis.py                        # EDA profundo, contrastes de hipótesis y figuras 300 DPI
│   ├── pipeline_builder.py                    # Pipelines scikit-learn con prevención estricta de Data Leakage
│   └── model_training.py                      # 10-Fold Stratified CV, curvas ROC y exportación de modelos
├── assets/
│   └── figures/                               # Galería de figuras analíticas en alta resolución (300 DPI)
│       ├── eda_01_prevalencia_comorbilidades.png
│       ├── eda_02_piramide_edad_sexo_hta.png
│       ├── eda_03_biomarcadores_boxplots.png
│       ├── eda_04_matriz_correlaciones.png
│       ├── eda_05_determinantes_sociales.png
│       ├── eval_01_curvas_roc.png
│       ├── eval_02_matriz_confusion.png
│       └── eval_03_importancia_variables_shap.png
├── models/
│   └── mejor_modelo_hipertension.joblib       # Pipeline serializado del mejor modelo entrenado
└── reports/
    └── Informe_Segunda_Entrega_ML2.md         # Informe técnico formal (3 a 5 páginas de cuerpo)
```

---

## 🔬 3. Estrategia de Datos y Validación Clínica

Para superar la restricción de reserva estadística sobre los microdatos individuales crudos de la **ENSIN 2015** y la encuesta **SABE** (que exigen trámites institucionales formales ante el Ministerio de Salud y el ICBF), se diseñó una estrategia mixta verificable:
1. **Contraste con Datos Abiertos de Colombia:** Registro real de $110.693$ pacientes del portal `datos.gov.co` (Dataset: *39. Enfermedades crónicas en el municipio de Bucaramanga*, ID `4iz7-suhz`).
2. **Cohorte Analítica Fundamentada ($N = 10.000$):** Modelada mediante cópulas probabilísticas respetando fielmente las distribuciones marginales, desviaciones y Odds Ratios epidemiológicos de:
   - **ENSIN 2015:** Parámetros de exceso de peso ($62.4\%$), prevalencia de sedentarismo ($49\%$) y panel bioquímico de adultos (Colesterol, Glucosa, Triglicéridos).
   - **Guía de Práctica Clínica No. 18 de MinSalud:** Parámetros fisiopatológicos de interacción entre diabetes, tabaquismo y daño endotelial.
   - **Cuenta de Alto Costo (CAC 2023-2025):** Prevalencia ajustada por edad y comorbilidades.
   - **Estudio PURE Colombia:** Estimaciones de prevalencia en población de 35 a 70 años.

---

## 📊 4. Hallazgos del Análisis Exploratorio de Datos (EDA)

| Biomarcador / Factor | Media Sin HTA | Media Con HTA | Diferencia | Prueba Estadística | Significación ($p$-valor) | Hallazgo Clínico Principal |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Edad** | $42.98$ años | $54.51$ años | **$+11.53$ años** | Mann-Whitney $U$ | $p = 6.34 \times 10^{-231}$ | Rigidez vascular y aterosclerosis progresiva. |
| **IMC** | $26.26\ kg/m^2$ | $28.71\ kg/m^2$ | **$+2.45\ kg/m^2$** | Mann-Whitney $U$ | $p = 2.71 \times 10^{-90}$ | Sobrecarga hemodinámica por adiposidad visceral. |
| **Glucosa** | $96.58\ mg/dL$ | $104.46\ mg/dL$ | **$+7.88\ mg/dL$** | Mann-Whitney $U$ | $p = 4.77 \times 10^{-66}$ | Disfunción endotelial por glucotoxicidad. |
| **Colesterol Total** | $188.38\ mg/dL$ | $200.77\ mg/dL$ | **$+12.39\ mg/dL$** | Mann-Whitney $U$ | $p = 4.75 \times 10^{-44}$ | Perfil lipídico aterogénico acelerado. |
| **Colesterol HDL** | $46.94\ mg/dL$ | $43.60\ mg/dL$ | **$-3.34\ mg/dL$** | Mann-Whitney $U$ | $p = 5.82 \times 10^{-39}$ | Reducción del colesterol protector vasomotriz. |
| **Diabetes Mellitus** | $7.2\%$ prev. | $18.8\%$ prev. | **$OR = 3.39$** | Chi-cuadrado $\chi^2$ | $p = 9.09 \times 10^{-73}$ | La diabetes **triplica** el riesgo de hipertensión. |
| **Sedentarismo** | $44.1\%$ prev. | $55.3\%$ prev. | **$OR = 1.57$** | Chi-cuadrado $\chi^2$ | $p = 8.35 \times 10^{-20}$ | Inactividad física asociada a menor óxido nítrico. |

---

## ⚙️ 5. Flujo sin Fuga de Datos (Pipeline Architecture)

Para corregir la inconsistencia de validación detectada en Miah et al. (2024), el flujo implementa:
- **Partición Train/Test Estratificada (80/20):** $8.000$ observaciones en entrenamiento y $2.000$ observaciones en prueba independiente, fijando la semilla aleatoria (`random_state=42`).
- **Transformador de Columnas (`ColumnTransformer`):**
  - **Variables Numéricas:** `SimpleImputer(strategy='median')` + `RobustScaler()` (resistente a valores bioquímicos extremos).
  - **Variables Categóricas:** `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(drop='first', handle_unknown='ignore')`.
- **Garantía metodológica:** Todos los estimadores y parámetros de centrado/escalado se ajustan estrictamente dentro de cada pliegue de entrenamiento durante la validación cruzada.

---

## 🏆 6. Resultados Comparativos de Modelos

Evaluación bajo **Validación Cruzada Estratificada de 10 Pliegues (10-Fold CV)** sobre el conjunto de entrenamiento ($N = 8.000$):

| Modelo / Algoritmo | Accuracy Global | Precisión ($Clase\ 1$) | Sensibilidad / Recall ($Clase\ 1$) | F1-Score ($Clase\ 1$) | ROC-AUC (Métrica Principal) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 **Regresión Logística ($L_2$)** | **$82.04 \pm 1.16\%$** | $65.41 \pm 6.20\%$ | **$28.98 \pm 3.73\%$** | **$40.14 \pm 4.59\%$** | **$77.79 \pm 2.11\%$** |
| 🥈 **XGBoost** | $81.44 \pm 0.96\%$ | $62.04 \pm 4.33\%$ | $27.49 \pm 4.54\%$ | $37.98 \pm 5.04\%$ | $76.73 \pm 2.29\%$ |
| 🥉 **Bagging (Random Forest)** | $81.60 \pm 0.76\%$ | **$68.95 \pm 5.16\%$** | $21.06 \pm 3.35\%$ | $32.18 \pm 4.30\%$ | $76.67 \pm 2.20\%$ |
| 📌 **Línea Base: Logística Simple (ML1)** | $81.15 \pm 0.85\%$ | $61.95 \pm 4.82\%$ | $24.31 \pm 3.96\%$ | $34.81 \pm 4.54\%$ | $76.12 \pm 2.04\%$ |
| **LightGBM** | $81.41 \pm 0.77\%$ | $62.01 \pm 3.51\%$ | $27.43 \pm 3.76\%$ | $37.95 \pm 4.16\%$ | $75.98 \pm 2.37\%$ |
| **Árbol de Decisión (CART)** | $80.51 \pm 0.58\%$ | $59.01 \pm 5.12\%$ | $22.45 \pm 3.08\%$ | $32.29 \pm 3.21\%$ | $73.18 \pm 2.70\%$ |
| **SVM (Kernel RBF)** | $81.85 \pm 0.87\%$ | $70.18 \pm 6.45\%$ | $22.14 \pm 3.37\%$ | $33.60 \pm 4.54\%$ | $71.50 \pm 2.49\%$ |
| **Línea Base: Mayoría (Dummy)** | $79.17 \pm 0.06\%$ | $0.00 \pm 0.00\%$ | $0.00 \pm 0.00\%$ | $0.00 \pm 0.00\%$ | $50.00 \pm 0.00\%$ |

### Desempeño en Prueba Independiente ($N = 2.000$)
- **Mejor Modelo:** Regresión Logística ($L_2$) con **$ROC\text{-}AUC = 79.27\%$**.
- **Mejor Ensamble No Lineal:** XGBoost con **$ROC\text{-}AUC = 77.41\%$**.

---

## 📈 7. Galería de Visualizaciones Analíticas

Todas las figuras fueron generadas en alta resolución ($300\ DPI$) y están disponibles en `assets/figures/`:

1. **Prevalencia y Factores de Riesgo:** `assets/figures/eda_01_prevalencia_comorbilidades.png`
2. **Curva Epidemiológica de Edad y Sexo:** `assets/figures/eda_02_piramide_edad_sexo_hta.png`
3. **Boxplots Comparativos de Biomarcadores:** `assets/figures/eda_03_biomarcadores_boxplots.png`
4. **Matriz de Correlación de Spearman:** `assets/figures/eda_04_matriz_correlaciones.png`
5. **Determinantes Sociales y Aseguramiento:** `assets/figures/eda_05_determinantes_sociales.png`
6. **Curvas ROC Comparativas (Miah et al. Fig. 3):** `assets/figures/eval_01_curvas_roc.png`
7. **Matriz de Confusión en Prueba:** `assets/figures/eval_02_matriz_confusion.png`
8. **Importancia de Variables y Coeficientes Clínicos:** `assets/figures/eval_03_importancia_variables_shap.png`

---

## 🚀 8. Instrucciones de Reproducción y Ejecución

### Requisitos Previos
Tener instalado Python 3.10 o superior. Clonar el repositorio y posicionarse en la carpeta raíz:

```bash
git clone <URL_DEL_REPOSITORIO>
cd "SEGUNDA ENTREGA PROYECTO ML2"
```

### Instalación de Dependencias
```bash
pip install -r requirements.txt
```

### Ejecución de los Módulos Automatizados
1. **Generación y Consolidación de Datos:**
   ```bash
   python src/data_generation.py
   ```
2. **Análisis Exploratorio y Pruebas Estadísticas:**
   ```bash
   python src/eda_analysis.py
   ```
3. **Entrenamiento y Validación Cruzada de Modelos:**
   ```bash
   python src/model_training.py
   ```

### Ejecución del Notebook Interactivo
Para explorar de manera interactiva todo el flujo, abrir Jupyter:
```bash
jupyter notebook notebooks/Segunda_Entrega_Hipertension_ML2.ipynb
```

---

## 📚 9. Referencias Oficiales
1. **Miah, J. et al. (2024).** *Improving Cardiovascular Disease Prediction Through Comparative Analysis of Machine Learning Models: A Case Study on Myocardial Infarction*. IEEE / arXiv:2311.00517.
2. **Ministerio de Salud y Protección Social, ICBF, DANE. (2015).** *Encuesta Nacional de la Situación Nutricional — ENSIN 2015*.
3. **Ministerio de Salud y Protección Social. (2018).** *Guía de Práctica Clínica para el diagnóstico y tratamiento de la Hipertensión Arterial en Colombia (GPC No. 18)*.
4. **Cuenta de Alto Costo (CAC). (2025).** *Situación de la ERC, la Hipertensión Arterial y la Diabetes Mellitus en Colombia*.
5. **Estudio PURE Colombia. (2020).** *Prevalence, awareness, treatment, and control of hypertension in Colombia: results from the PURE study*. The Lancet / Rev. Col. Cardiología.
