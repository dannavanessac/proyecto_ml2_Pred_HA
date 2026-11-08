# Informe de Entrega Final – Predicción de Hipertensión Arterial

**Asignatura:** Machine Learning II  
**Curso:** Universidad Externado de Colombia  
**Programa:** Ciencia de Datos  
**Docente:** Wilmer Darío Pineda Ríos  
**Equipo:**
- Allison Michelle Loango Rayo
- Danna Vanessa Caballero Urrego

**Repositorio:** https://github.com/dannavanessac/proyecto_ml2_Pred_HA

---

## 1. Introducción

Este documento sintetiza el trabajo desarrollado en la **Entrega Final** del proyecto integrador de Machine Learning II. Se retoma el flujo completo implementado en la segunda entrega y se amplía con la selección del modelo definitivo, su interpretación, un despliegue básico mediante Streamlit y una reflexión crítica sobre la calidad, las limitaciones y las proyecciones del trabajo.

El proyecto reproduce y adapta al contexto colombiano el artículo de Miah et al. (2024), cambiando el problema clínico: en lugar de la mortalidad por infarto en una muestra reducida de Bangladesh, se predice el **diagnóstico de hipertensión arterial** (`hipertension_dx`) como herramienta de tamizaje poblacional.

---

## 2. Metodología General

1. **Obtención de datos** – Se descargó una muestra de **5.000 registros** de datos abiertos de Colombia (`bucaramanga_cronicas_sample.csv`, dataset *39. Enfermedades crónicas en el municipio de Bucaramanga*, ID `4iz7-suhz`), usada como referencia empírica. Con base en distribuciones y odds ratios publicados (ENSIN 2015, GPC No. 18 de MinSalud, Cuenta de Alto Costo y estudio PURE Colombia) se generó una **cohorte analítica de 10.000 pacientes** (`dataset_hipertension_colombia.csv`, 16 columnas) con una prevalencia de hipertensión de 20,8 %.
2. **Pre-procesamiento** – Imputación por mediana (numéricas) y por moda (categóricas), escalado robusto (`RobustScaler`) para las numéricas y codificación one-hot (`drop='first'`) para las categóricas, todo encapsulado en un `ColumnTransformer` dentro de un `Pipeline` de scikit-learn. Se excluye el identificador `id_paciente`, y las 14 variables predictoras son: edad, sexo, estrato socioeconómico, zona de residencia, régimen de salud, IMC, actividad física, tabaquismo, diabetes, glucosa en ayunas, colesterol total, HDL, LDL y triglicéridos.
3. **Entrenamiento y validación** – Partición **estratificada 80/20** (8.000 registros de entrenamiento y 2.000 de prueba independiente, `random_state=42`) y **validación cruzada estratificada de 10 pliegues (10-Fold CV)** sobre el conjunto de entrenamiento. Se evaluaron **ocho configuraciones**: seis modelos (Regresión Logística L2, SVM con kernel RBF, Árbol de Decisión CART, Bagging/Random Forest, XGBoost y LightGBM) y dos líneas base (clasificador de mayoría y regresión logística simple con edad, sexo e IMC). Los hiperparámetros están fijados en `src/model_training.py`; no se realizó búsqueda de hiperparámetros.
4. **Selección del modelo final** – Se priorizó el ROC-AUC como métrica principal, la estabilidad entre pliegues y la capacidad interpretativa. El modelo con mejor desempeño, tanto en validación cruzada (ROC-AUC = 77,79 ± 2,11 %) como en la prueba independiente (N = 2.000), fue la **Regresión Logística con penalización L2**, con un **ROC-AUC en prueba de 79,27 %**.
5. **Interpretación** – Se analizaron los coeficientes de la regresión logística y la importancia de variables de XGBoost como referencia (`assets/figures/eval_03_importancia_variables_shap.png`). En la aplicación, cada predicción individual se explica con SHAP. Los factores más influyentes fueron **edad, diabetes, colesterol LDL, IMC y triglicéridos**.
6. **Despliegue** – Se construyó una aplicación Streamlit (`src/app_streamlit.py`) que permite ingresar las 14 variables y obtener la probabilidad de hipertensión, el nivel de riesgo y una explicación SHAP de la predicción. El pipeline completo se carga desde `models/mejor_modelo_hipertension.joblib`.

---

## 3. Modelo Final y Reproducción del Método

- **Modelo:** `LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, random_state=42)`
- **Pipeline:** `ColumnTransformer` → `LogisticRegression`
- **Serialización:** `joblib.dump(pipeline, 'models/mejor_modelo_hipertension.joblib')`

**Rendimiento en validación cruzada (10 pliegues, entrenamiento):**

| ROC-AUC | Accuracy | Precisión (clase 1) | Recall (clase 1) | F1 (clase 1) |
| :---: | :---: | :---: | :---: | :---: |
| 77,79 ± 2,11 % | 82,04 ± 1,16 % | 65,41 ± 6,20 % | 28,98 ± 3,73 % | 40,14 ± 4,59 % |

**Rendimiento en la prueba independiente (N = 2.000, umbral de decisión 0,5):**

| ROC-AUC | Accuracy | Precisión (clase 1) | Recall (clase 1) | F1 (clase 1) |
| :---: | :---: | :---: | :---: | :---: |
| **79,27 %** | 81,85 % | 63,64 % | 30,22 % | 40,98 % |

Matriz de confusión en prueba: 1.511 verdaderos negativos, 72 falsos positivos, 291 falsos negativos y 126 verdaderos positivos (`assets/figures/eval_02_matriz_confusion.png`). Es decir, de los 417 casos positivos del conjunto de prueba, el modelo detecta 126.

El proceso completo se reproduce ejecutando, desde el repositorio:
```bash
python src/data_generation.py   # genera la cohorte analítica
python src/eda_analysis.py      # análisis exploratorio y figuras
python src/model_training.py    # validación cruzada, selección y exportación del modelo
```
El script de entrenamiento selecciona automáticamente el mejor modelo por ROC-AUC en prueba y lo exporta a `models/mejor_modelo_hipertension.joblib`.

---

## 4. Interpretación de Resultados

### 4.1 Coeficientes de la Regresión Logística

Las variables numéricas se escalaron con `RobustScaler`, por lo que el odds ratio (OR) de una variable numérica corresponde a un aumento equivalente a su rango intercuartílico (IQR). En las categóricas, el OR compara contra la categoría de referencia.

| Variable | Coeficiente (log-odds) | Odds Ratio |
|----------|:---------------------:|:----------:|
| **Edad** | +1,10 | 3,00 |
| **Diabetes (Sí)** | +0,73 | 2,07 |
| **Colesterol LDL** | +0,65 | 1,91 |
| **IMC** | +0,47 | 1,60 |
| **Triglicéridos** | +0,39 | 1,48 |
| **Tabaquismo (Sí)** | +0,31 | 1,37 |
| **Actividad física (Sedentario)** | +0,26 | 1,30 |
| **Régimen de salud (Especial)** | +0,23 | 1,26 |
| **Sexo (Masculino)** | +0,15 | 1,17 |
| Colesterol total | −0,37 | 0,69 |
| Colesterol HDL | −0,09 | 0,91 |
| Glucosa en ayunas | −0,01 | 0,99 |

> Los coeficientes positivos indican que mayores valores aumentan la probabilidad de hipertensión. Los factores más influyentes son la **edad** y la **diabetes** (esta última, el factor categórico más fuerte, con OR ≈ 2,1). El coeficiente negativo del colesterol total y el coeficiente casi nulo de la glucosa no implican un efecto protector: estas variables están muy relacionadas con LDL, HDL, triglicéridos y diabetes, y el modelo reparte su efecto entre ellas, por lo que deben interpretarse en conjunto.

En el análisis exploratorio (pruebas Chi-cuadrado) se observaron odds ratios marginales de 3,39 para diabetes y 1,57 para sedentarismo; difieren de los del modelo porque estos últimos están ajustados por el resto de variables.

### 4.2 Importancia de variables (figura) y SHAP (aplicación)

La figura `assets/figures/eval_03_importancia_variables_shap.png` muestra en el panel A los coeficientes de la regresión logística y en el panel B la importancia relativa (*gain*) de XGBoost, como referencia de un ensamble no lineal. En la aplicación Streamlit, cada predicción se explica con SHAP (`LinearExplainer`, adecuado para un modelo lineal) mediante un gráfico de cascada que muestra qué variables aumentan o disminuyen la probabilidad estimada respecto a una persona promedio.

---

## 5. Despliegue Básico

El archivo `src/app_streamlit.py` contiene la aplicación web. Para ejecutarla, desde la raíz del repositorio:
```bash
pip install -r requirements.txt   # si aún no está instalado
streamlit run src/app_streamlit.py
```
La aplicación:
1. Carga automáticamente el pipeline serializado y el dataset procesado (rangos y valores por defecto), mediante rutas relativas al repositorio.
2. Presenta un formulario organizado en tres secciones (datos personales y sociales, medidas corporales y hábitos, exámenes de laboratorio), con una explicación, las unidades y los valores de referencia de cada variable, además de una pestaña **Guía de variables** y una calculadora de IMC.
3. Calcula la probabilidad de hipertensión y la muestra como porcentaje con un nivel de riesgo (bajo < 20 %, moderado < 50 %, alto < 80 %, muy alto ≥ 80 %), junto con la clasificación binaria del modelo.
4. Muestra una explicación SHAP de los factores que más influyeron en la predicción.
5. Incluye un aviso de que es una herramienta académica de apoyo que no reemplaza el diagnóstico médico.

---

## 6. Calidad, Limitaciones y Proyección

| Aspecto | Comentario |
|---------|------------|
| **Calidad de datos** | La cohorte de 10.000 registros es analítica: se generó a partir de distribuciones marginales y odds ratios publicados, por lo que no corresponde a microdatos clínicos individuales reales y puede no capturar interdependencias complejas. La muestra real de Bucaramanga (5.000 registros) se usa solo como referencia empírica y puede tener sesgos de muestreo y valores atípicos. |
| **Supuestos del modelo** | La regresión logística asume linealidad en el log-odds y no incluye interacciones entre variables; aunque funciona bien, podría mejorar con efectos no lineales. |
| **Riesgo de sobreajuste** | La validación cruzada estratificada de 10 pliegues y la prueba independiente mitigan el sobreajuste: el ROC-AUC en prueba (79,27 %) es consistente con el de validación (77,79 ± 2,11 %). Sin embargo, no se realizó búsqueda de hiperparámetros. |
| **Sensibilidad baja** | Con el umbral de 0,5, el recall es de ≈ 30 %: el modelo deja sin detectar cerca de 7 de cada 10 casos positivos. Una probabilidad baja no descarta la enfermedad. |
| **Interpretabilidad** | Los coeficientes son fáciles de explicar; sin embargo, la falta de interacciones limita la captura de efectos combinados, y la colinealidad entre variables de laboratorio dificulta interpretar cada una por separado. |
| **Despliegue** | Streamlit brinda una interfaz rápida, pero no está preparada para producción (sin gestión de versiones, autenticación ni registro de eventos). |
| **Proyecciones** | Futuras mejoras: (i) modelos de ensamble no lineal (por ejemplo, stacking) y ajuste del umbral de decisión para mejorar el recall, (ii) incorporación de datos reales y longitudinales, (iii) despliegue en Docker con CI/CD y monitoreo de deriva, (iv) extensión a MLOps con MLflow. |

---

## 7. Conclusiones

- Se reprodujo y adaptó el flujo de modelado del artículo base al contexto colombiano, con un pipeline libre de fuga de datos.
- El modelo final elegido (Regresión Logística L2) obtuvo el mejor ROC-AUC (79,27 % en prueba) y ofrece interpretabilidad clínica; edad, diabetes, LDL e IMC son los factores más influyentes.
- La sensibilidad (≈ 30 %) es la principal debilidad del modelo y orienta las mejoras futuras.
- El despliegue en Streamlit permite una evaluación interactiva y explicable, y sirve como base para futuros entornos productivos.
- Las limitaciones identificadas, en particular el origen analítico de los datos, abren caminos claros para profundizar en la robustez y escalabilidad del proyecto.

---

## 8. Referencias
1. Miah, J. *et al.* (2024). *Improving Cardiovascular Disease Prediction Through Comparative Analysis of Machine Learning Models: A Case Study on Myocardial Infarction*. IEEE / arXiv:2311.00517.
2. Ministerio de Salud y Protección Social (2018). *Guía de Práctica Clínica No. 18 – Hipertensión Arterial en Colombia*.
3. ENSIN 2015 – Encuesta Nacional de la Situación Nutricional.
4. Dataset abierto: *39. Enfermedades crónicas en el municipio de Bucaramanga* (ID `4iz7-suhz`).
5. Estudio PURE Colombia (2020).

---

*Este informe se entrega en formato Markdown para facilitar su revisión y conversión a PDF.*
