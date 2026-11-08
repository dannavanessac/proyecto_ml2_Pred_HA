# Informe de Entrega Final – Predicción de Hipertensión Arterial

**Asignatura:** Machine Learning II  
**Curso:** Universidad Externado de Colombia  
**Equipo:**
- Allison Michelle Loango Rayo
- Danna Vanessa Caballero Urrego

---

## 1. Introducción

Este documento sintetiza el trabajo desarrollado en la **Entrega Final** del proyecto integrador de Machine Learning II. Se retoma el flujo completo implementado en la segunda entrega y se amplía con la selección del modelo definitivo, su interpretación profunda, un despliegue básico mediante Streamlit y una reflexión crítica sobre la calidad, limitaciones y proyecciones a futuro.

---

## 2. Metodología General

1. **Obtención de datos** – Se combinaron datos abiertos de la base `bucaramanga_cronicas_sample.csv` (40⁷ registros) con un proceso de simulación fundamentada que genera una cohorte analítica de **10 000** pacientes (`dataset_hipertension_colombia.csv`).
2. **Pre‑procesamiento** – Imputación median/most_frequent, escalado robusto para variables numéricas y codificación one‑hot para categóricas, todo encapsulado en un `ColumnTransformer` dentro de un `Pipeline` de scikit‑learn.
3. **Entrenamiento y validación** – Estrategia **Stratified 80/20 Train‑Test** con **10‑Fold CV** anidado. Se entrenaron ocho algoritmos (Regresión Logística, XGBoost, Random Forest, LightGBM, CART, SVM, Dummy y otros) y se almacenaron métricas de Accuracy, Precisión, Recall, F1 y ROC‑AUC.
4. **Selección del modelo final** – Se priorizó ROC‑AUC como métrica principal, estabilidad de la curva y capacidad interpretativa. El modelo con mejor desempeño en prueba independiente (N=2000) fue **Regresión Logística (penalización L2)** con ROC‑AUC = **79.27 %**.
5. **Interpretación** – Se aplicaron técnicas SHAP (para modelos de árbol) y se mostraron los coeficientes de la regresión logística. Los factores más influyentes fueron edad, IMC, glucosa y antecedentes de diabetes.
6. **Despliegue** – Se construyó una aplicación Streamlit (**src/app_streamlit.py**) que permite al usuario introducir variables y obtener la probabilidad de hipertensión, además de una visualización SHAP (cuando el modelo es de árbol). El pipeline completo se carga desde `models/mejor_modelo_hipertension.joblib`.

---

## 3. Modelo Final y Reproducción del Método

- **Modelo:** `LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000)`
- **Pipeline:** `ColumnTransformer` → `LogisticRegression`
- **Serialización:** `joblib.dump(pipeline, 'models/mejor_modelo_hipertension.joblib')`
- **Rendimiento en Test:**
  - **ROC‑AUC:** 79.27 %
  - **Accuracy:** 81.00 %
  - **Recall (Clase 1):** 31.00 %
  - **Precision (Clase 1):** 68.00 %

El proceso de entrenamiento se reproduce ejecutando:
```bash
python src/model_training.py
```
El script ya incluye la selección automática del mejor modelo y la exportación a `models/mejor_modelo_hipertension.joblib`.

---

## 4. Interpretación de Resultados

### 4.1 Coeficientes de la Regresión Logística
| Variable | Coeficiente | Odds Ratio |
|----------|------------|------------|
| **Edad** | 0.075 | 1.08 |
| **IMC**   | 0.112 | 1.12 |
| **Glucosa** | 0.038 | 1.04 |
| **Diabetes (Sí/No)** | 0.92 | 2.51 |
| **Sedentarismo (Sí/No)** | 0.45 | 1.57 |
> Los coeficientes positivos indican que mayores valores aumentan la probabilidad de hipertensión. El factor de riesgo más fuerte es la presencia de diabetes (OR ≈ 2.5).

### 4.2 SHAP (para XGBoost – referencia) 
Se incluyeron los gráficos `assets/figures/eval_03_importancia_variables_shap.png` que muestran la distribución de impacto de cada predictor sobre la salida del modelo.

---

## 5. Despliegue Básico

El archivo **src/app_streamlit.py** contiene la aplicación web. Para ejecutarla:
```bash
pip install -r requirements.txt   # si aún no está instalado
streamlit run src/app_streamlit.py
```
La aplicación:
1. Carga automáticamente el pipeline serializado.
2. Genera formularios dinámicos basados en los tipos de variables del dataset.
3. Calcula probabilidad y clasificación.
4. (Opcional) Muestra un gráfico SHAP cuando el modelo subyacente lo permite.

---

## 6. Calidad, Limitaciones y Proyección

| Aspecto | Comentario |
|---------|------------|
| **Calidad de datos** | Los datos reales de Bucaramanga presentan sesgos de muestreo y posibles valores atípicos; la simulación se basa en distribuciones marginales y odds ratios publicados, lo que puede no capturar interdependencias complejas. |
| **Supuestos de modelo** | La regresión logística asume linealidad en el log‑odds y dependencia independiente de variables; aunque funciona bien, podría mejorar con interacciones no lineales. |
| **Riesgo de sobreajuste** | La validación cruzada anidada y la prueba independiente mitigaron el sobreajuste, pero la baja sensibilidad (≈31 %) indica que el modelo tiende a sub‑predicar casos positivos. |
| **Interpretabilidad** | Los coeficientes son fáciles de explicar; sin embargo, la falta de interacciones limita la captura de efectos combinados. |
| **Despliegue** | Streamlit brinda una UI rápida, pero no está preparada para producción (sin gestión de versiones, autenticación, logging). |
| **Proyecciones** | Futuras mejoras incluyen: (i) uso de modelos de ensamblado no lineal (e.g., Stacking) para mejorar Recall, (ii) incorporación de datos longitudinales, (iii) despliegue en Docker con CI/CD y monitorización de drift, (iv) extensión a MLOps con MLflow. |

---

## 7. Conclusiones

- Se reprodujo y adaptó con éxito el flujo de modelado del artículo base al contexto colombiano.
- El modelo final elegido (Logística L2) cumple con la métrica principal y ofrece interpretabilidad clínica.
- El despliegue Streamlit permite una evaluación interactiva y sirve como base para futuros entornos productivos.
- Las limitaciones identificadas abren caminos claros para profundizar en la robustez y escalabilidad del proyecto.

---

## 8. Referencias
1. Miah, J. *et al.* (2024). *Improving Cardiovascular Disease Prediction Through Comparative Analysis of Machine Learning Models*. IEEE / arXiv:2311.00517.
2. Ministerio de Salud y Protección Social (2018). *Guía de Práctica Clínica No. 18 – Hipertensión Arterial en Colombia*.
3. ENSIN 2015 – Encuesta Nacional de la Situación Nutricional.
4. Dataset abierto: *39. Enfermedades crónicas en el municipio de Bucaramanga* (ID `4iz7-suhz`).
5. Pure Colombia Study (2020).

---

*Este informe se entrega en formato Markdown para facilitar su revisión y conversión a PDF.*
