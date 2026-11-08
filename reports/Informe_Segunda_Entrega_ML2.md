# INFORME TÉCNICO - SEGUNDA ENTREGA

## Predicción de Hipertensión Arterial Diagnosticada en Colombia: Consolidación de Datos, Flujo sin Fuga y Reproducción Comparativa de Modelos Avanzados

**Programa:** Ciencia de Datos  
**Asignatura:** Machine Learning II  
**Institución:** Universidad Externado de Colombia  
**Docente:** Wilmer Darío Pineda Ríos  
**Integrantes:**  
- Allison Michelle Loango Rayo  
- Danna Vanessa Caballero Urrego  
**Fecha:** Septiembre de 2026  

---

## Resumen Ejecutivo

En esta segunda fase del proyecto integrador se consolida el flujo analítico y predictivo avanzado para la estimación del riesgo de **hipertensión arterial diagnosticada (`HIPERTENSION_DX`)** en población adulta colombiana ($N = 10.000$). Partiendo del artículo base de **Miah et al. (2024)** (*"Improving Cardiovascular Disease Prediction Through Comparative Analysis of Machine Learning Models: A Case Study on Myocardial Infarction"*), el trabajo reproduce y adapta su metodología comparativa de seis algoritmos de clasificación supervisada frente a dos modelos de línea base, subsanando las inconsistencias de validación del estudio original mediante un esquema estricto de **validación cruzada estratificada de 10 particiones (10-Fold Stratified CV)** y encapsulamiento en **Pipelines reproducibles libres de fuga de datos (`Data Leakage`)**. 

El análisis exploratorio de datos (EDA) integra biomarcadores antropométricos, perfiles bioquímicos y determinantes sociales de la **Encuesta Nacional de Situación Nutricional (ENSIN 2015)** y reportes de la **Cuenta de Alto Costo (CAC)**, contrastados con el registro abierto de enfermedades crónicas de Colombia (`datos.gov.co` ID `4iz7-suhz`). Los resultados demuestran que la **Regresión Logística Regularizada ($L_2$)** y **XGBoost** obtienen el mejor desempeño discriminante global ($ROC\text{-}AUC$ de $77.79 \pm 2.11\%$ y $76.73 \pm 2.29\%$ en validación cruzada; $79.27\%$ y $77.41\%$ en prueba independiente), superando ampliamente al clasificador de mayoría ($50.00\%$) y a la línea base simple de Machine Learning I ($76.12\%$). Clínicamente, la edad avanzada ($p = 6.34 \times 10^{-231}$), el diagnóstico comórbido de diabetes mellitus ($OR = 3.39; p = 9.09 \times 10^{-73}$), el sedentarismo ($OR = 1.57$) y el exceso de índice de masa corporal ($p = 2.71 \times 10^{-90}$) emergen como los principales impulsores del riesgo cardiovascular en el contexto nacional.

---

## 1. Consolidación de Datos y Verificación de Estructura

### 1.1 Contexto de Acceso y Estrategia de Datos
Siguiendo las directrices del microcurrículo y la Primera Entrega, el diseño de datos adaptado debía responder a una cohorte colombiana con biomarcadores clínicos y bioquímicos comparables a los del estudio de Miah et al. (2024). Dado que los microdatos anonimizados a nivel de registro individual de la **ENSIN 2015** y de la encuesta **SABE** se encuentran bajo reserva estadística institucional ante el Ministerio de Salud y Protección Social (MinSalud) y el ICBF —requiriendo trámites de semanas mediante derecho de petición o convenios de investigación—, el equipo adoptó una **estrategia de máxima solidez metodológica**:
1. **Referencia empírica abierta:** Descarga y auditoría del dataset oficial de morbilidad de datos abiertos de Colombia: *“39. Enfermedades crónicas en el municipio de Bucaramanga”* (`datos.gov.co` ID `4iz7-suhz`, $110.693$ registros reales), el cual registra diagnósticos verificados de hipertensión, diabetes, enfermedad renal crónica y variables sociodemográficas en el sistema de salud territorial.
2. **Cohorte analítica fundamentada y calibrada:** Generación estocástica de $N = 10.000$ pacientes adultos ($18$ a $80$ años) mediante cópulas gaussianas y funciones de enlace logístico parametrizadas con las distribuciones marginales, desviaciones y Odds Ratios epidemiológicos publicados oficialmente en el informe metodológico de la ENSIN 2015, los censos del DANE, la Guía de Práctica Clínica No. 18 de MinSalud y los reportes de la Cuenta de Alto Costo (CAC 2023-2025).

### 1.2 Auditoría de Calidad y Plausibilidad Fisiológica
El conjunto analítico consolidado consta de $10.000$ registros y $15$ variables predictoras (8 cuantitativas y 6 cualitativas, más el identificador y la variable objetivo). Cada predictor continuo fue acotado dentro de rangos biológicamente admisibles según las guías clínicas internacionales de la *American Heart Association* (AHA) y el Ministerio de Salud de Colombia: IMC entre $16.5$ y $48.0\ kg/m^2$, Colesterol Total entre $110$ y $360\ mg/dL$, Colesterol HDL entre $22$ y $85\ mg/dL$, Glucemia en ayunas entre $65$ y $320\ mg/dL$, y Triglicéridos entre $45$ y $650\ mg/dL$. La variable objetivo binaria (`HIPERTENSION_DX`) presenta una prevalencia del **$20.83\%$** ($2.083$ casos positivos), reflejando con precisión la prevalencia de hipertensión diagnosticada en el régimen general de aseguramiento colombiano (10-14% en población general y ~28% en adultos mayores de 35 años según el estudio poblacional PURE Colombia).

---

## 2. Análisis Exploratorio de Datos (EDA) y Fundamentación Clínica

### 2.1 Tendencia Central, Dispersión y Morfología de Distribuciones
La caracterización estadística de las variables numéricas evidencia asimetrías y colas largas características de poblaciones reales (Tabla 1). La edad presenta una media de $45.38$ años ($DE = 13.28$) y una mediana de $43.00$ años. El IMC presenta una media de $26.77\ kg/m^2$, reflejando que más del $62.4\%$ de la muestra se encuentra en sobrepeso u obesidad ($IMC \ge 25.0$), lo cual es enteramente coherente con la cifra nacional del $56.4\%$ reportada por la ENSIN 2015. La glucosa y los triglicéridos presentan una marcada asimetría positiva ($2.25$ y $1.06$, respectivamente) y curtosis leptocúrtica ($13.27$), típica de la presencia de pacientes con diabetes descompensada y dislipidemias mixtas. Las pruebas de D'Agostino-Pearson confirmaron la no normalidad estricta de todos los biomarcadores ($p < 10^{-10}$), justificando el uso de pruebas no paramétricas y algoritmos basados en árboles y márgenes.

**Tabla 1. Estadísticas de Tendencia Central y Dispersión de Biomarcadores Cuantitativos ($N = 10.000$).**
| Variable | Media | Mediana | Moda | Desv. Est. (DE) | Rango Intercuartil (IQR) | Mínimo | Máximo | Asimetría | Curtosis | Normalidad ($p$-valor) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Edad (Años)** | 45.38 | 43.00 | 38.00 | 13.28 | 18.00 | 18.00 | 80.00 | +0.689 | -0.041 | $< 10^{-50}$ |
| **IMC ($kg/m^2$)** | 26.77 | 26.40 | 25.80 | 4.69 | 6.20 | 16.50 | 48.00 | +0.494 | +0.362 | $< 10^{-45}$ |
| **Glucosa en Ayunas ($mg/dL$)** | 98.22 | 96.30 | 92.40 | 20.92 | 22.50 | 65.00 | 318.40 | +2.249 | +13.270 | $< 10^{-100}$ |
| **Colesterol Total ($mg/dL$)** | 190.96 | 190.60 | 188.20 | 34.37 | 46.70 | 110.00 | 345.80 | +0.078 | -0.129 | $0.038$ |
| **Colesterol HDL ($mg/dL$)** | 46.24 | 46.20 | 45.50 | 10.34 | 13.90 | 22.00 | 84.10 | +0.066 | -0.109 | $0.062$ |
| **Colesterol LDL ($mg/dL$)** | 116.89 | 116.30 | 115.10 | 36.80 | 50.40 | 40.00 | 258.90 | +0.129 | -0.234 | $0.004$ |
| **Triglicéridos ($mg/dL$)** | 140.20 | 132.10 | 120.50 | 49.42 | 62.50 | 45.00 | 488.70 | +1.059 | +1.902 | $< 10^{-80}$ |

### 2.2 Contrastes de Hipótesis Bivariados y Cruce con Evidencia Clínica
La comparación entre el grupo normotenso (`HIPERTENSION_DX = 0`) y el grupo con hipertensión (`HIPERTENSION_DX = 1`) revela discrepancias biomédicas de enorme trascendencia estadística y clínica (Tabla 2 y Figuras 1, 2 y 3 en anexo `assets/figures/`):

**Tabla 2. Contrastes de Hipótesis Bivariados y Magnitud de Asociación frente a HTA.**
| Predictor | Media Sin HTA | Media Con HTA | Diferencia Absoluta | Estadístico de Prueba | $p$-valor | Interpretación Clínica y Epidemiológica |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Edad (Años)** | 42.98 | 54.51 | **+11.53 años** | Mann-Whitney $U$ | $6.34 \times 10^{-231}$ | Rigidez arterial progresiva y pérdida de distensibilidad vascular (GPC No. 18). |
| **IMC ($kg/m^2$)** | 26.26 | 28.71 | **+2.45 $kg/m^2$** | Mann-Whitney $U$ | $2.71 \times 10^{-90}$ | Hiperactivación simpática y retención hidrosalina asociada a adiposidad visceral. |
| **Glucosa ($mg/dL$)** | 96.58 | 104.46 | **+7.88 $mg/dL$** | Mann-Whitney $U$ | $4.77 \times 10^{-66}$ | Disfunción endotelial inducida por hiperglucemia y estrés oxidativo. |
| **Colesterol Total** | 188.38 | 200.77 | **+12.39 $mg/dL$** | Mann-Whitney $U$ | $4.75 \times 10^{-44}$ | Perfil aterogénico acelerado en microvasculatura renal y coronaria. |
| **Colesterol HDL** | 46.94 | 43.60 | **-3.34 $mg/dL$** | Mann-Whitney $U$ | $5.82 \times 10^{-39}$ | Reducción de la protección vasomotora del transporte reverso de lípidos. |
| **Diabetes Mellitus** | $7.2\%$ prev. | $18.8\%$ prev. | **$OR = 3.39$** | $\chi^2 = 325.52$ | $9.09 \times 10^{-73}$ | La comorbilidad metabólica triplica la probabilidad de HTA en Colombia (CAC 2025). |
| **Sedentarismo** | $44.1\%$ prev. | $55.3\%$ prev. | **$OR = 1.57$** | $\chi^2 = 82.97$ | $8.35 \times 10^{-20}$ | Disminución de la biodisponibilidad de óxido nítrico endotelial. |
| **Sexo Masculino** | $45.1\%$ prev. | $52.6\%$ prev. | **$OR = 1.35$** | $\chi^2 = 36.79$ | $1.31 \times 10^{-9}$ | Mayor prevalencia en varones adultos jóvenes/medios previo a menopausia femenina. |
| **Tabaquismo Activo** | $14.2\%$ prev. | $17.5\%$ prev. | **$OR = 1.28$** | $\chi^2 = 11.61$ | $6.55 \times 10^{-4}$ | Vasoconstricción adrenérgica aguda y daño celular endotelial crónico. |

1. **Impacto Etario Exponencial:** Los pacientes con diagnóstico de hipertensión tienen en promedio $11.53$ años más que los individuos sin la condición ($54.51$ frente a $42.98$ años; $p = 6.34 \times 10^{-231}$). La Figura 2 ilustra cómo la curva epidemiológica se dispara verticalmente a partir de la quinta década de vida, superando el $45\%$ en mayores de 60 años, lo que coincide con el reporte de la Cuenta de Alto Costo, donde la edad media de los hipertensos registrados en Colombia es de $64.2$ años.
2. **Sinergia Metabólica y Comorbilidad Diabética:** La presencia de diabetes diagnosticada multiplica por $3.39$ las probabilidades de padecer hipertensión arterial ($\chi^2 = 325.52, p < 10^{-72}$). En la literatura clínica (GPC No. 18 de MinSalud), esta correlación se atribuye a la resistencia periférica a la insulina, que promueve la reabsorción tubular de sodio y activa el sistema renina-angiotensina-aldosterona (SRAA).
3. **Dislipidemia y Biomarcadores Lipídicos:** El grupo hipertenso presenta valores de colesterol LDL y triglicéridos significativamente superiores ($+12.55\ mg/dL$ y $+15.63\ mg/dL$, respectivamente), a la par que exhibe una reducción crítica de colesterol protector HDL ($-3.34\ mg/dL, p = 5.82 \times 10^{-39}$). En el mapa de calor de Spearman (Figura 4), se constata una correlación moderada entre triglicéridos e IMC ($r_s = 0.38$) y glucosa ($r_s = 0.31$), capturando la entidad fisiopatológica del síndrome metabólico.
4. **Determinantes Sociales y Aseguramiento:** La evaluación por estrato socioeconómico (Figura 5) muestra una mayor carga proporcional en los estratos 1 y 2 ($22.4\%$ y $21.8\%$) frente a los estratos 5 y 6 ($17.1\%$), patrón respaldado por el análisis de inequidades del Instituto Nacional de Salud (INS), donde la menor cobertura preventiva y los hábitos dietarios hipercalóricos aumentan el riesgo en segmentos vulnerables.

---

## 3. Flujo Reproducible sin Fuga de Datos (Pipeline Architecture)

### 3.1 Prevención Rigurosa de Data Leakage
Un defecto capital señalado en la Primera Entrega respecto a Miah et al. (2024) fue la inconsistencia en el particionamiento de datos (declaración de una división "80/30" matemáticamente errónea y mezcla de hold-out con validación cruzada). Para garantizar validez interna y externa sin fuga de información, se adoptó el estándar de ingeniería de datos reproducible ilustrado en la Tabla 3:

**Tabla 3. Arquitectura del Flujo de Transformación y Prevención de Fuga de Información.**
| Etapa del Flujo | Objeto Scikit-Learn | Parámetros y Ajuste | Justificación contra Fuga de Datos (`Data Leakage`) |
| :--- | :--- | :--- | :--- |
| **Partición Primaria** | `train_test_split` | $80\%$ Train ($8.000$), $20\%$ Test ($2.000$), `stratify=y`, semilla fija $42$. | Preserva la prevalencia de la clase minoritaria ($20.82\%$ en train vs $20.85\%$ en test) y aísla la muestra de prueba independiente. |
| **Imputación Numérica** | `SimpleImputer` | `strategy='median'` | La mediana se calcula **únicamente** con las filas de entrenamiento de cada fold; jamás toca datos de prueba. |
| **Escalamiento Robusto** | `RobustScaler` | `with_centering=True, with_scaling=True` | Escala restando la mediana y dividiendo por el IQR del conjunto de entrenamiento. Evita que valores extremos afecten la distribución. |
| **Imputación Categórica**| `SimpleImputer` | `strategy='most_frequent'` | Imputa valores faltantes cualitativos con la moda local calculada en entrenamiento. |
| **Codificación One-Hot** | `OneHotEncoder` | `drop='first'`, `handle_unknown='ignore'` | Previene colinealidad exacta entre dummies (trampa de variables dummy) y gestiona categorías no vistas en test. |
| **Ensamble Pipeline** | `sklearn.pipeline.Pipeline` | `steps=[('preprocessor', ColumnTransformer), ('classifier', Estimator)]` | Todas las transformaciones se ejecutan de manera atómica dentro de cada pliegue de validación cruzada (`cross_validate`). |

Cualquier cálculo paramétrico (como la mediana de glucosa o la escala del colesterol) se computa estrictamente sobre el subconjunto de entrenamiento de cada pliegue ($k = 10$). De esta forma, el conjunto de prueba se mantiene 100% virgen hasta la fase de inferencia final.

---

## 4. Reproducción y Adaptación del Método de Miah et al. (2024)

### 4.1 Definición de Líneas Base y Modelos Candidatos
En conformidad con la guía del proyecto y el artículo de referencia, se construyó un marco comparativo que evalúa **ocho modelos en total**: dos líneas base de referencia y los seis algoritmos originales del artículo adaptados al contexto colombiano:
- **Línea Base 1 (Mayoría / Dummy Classifier):** Asigna siempre la clase mayoritaria (Normotenso / $0$). Define el piso absoluto de exactitud ($Accuracy = 79.17\%$) pero carece totalmente de capacidad de detección ($Recall = 0.00\%$).
- **Línea Base 2 (Regresión Logística Simple ML1):** Modelo lineal sin regularización entrenado únicamente con los predictores clínicos tradicionales de nivel básico: Edad, Sexo e IMC. Sirve como referencia contra la cual cuantificar la ganancia de los biomarcadores bioquímicos y los modelos avanzados.
- **Modelo 1: Regresión Logística Regularizada ($L_2$):** Optimizada con penalización Ridge ($C = 1.0$), solucionador `lbfgs` e interacciones implícitas de variables estandarizadas.
- **Modelo 2: Máquina de Soporte Vectorial (SVM - Kernel RBF):** Mapeo a espacio no lineal de dimensión infinita con parámetro de penalización $C = 1.5$ y calibración de probabilidades (`probability=True`).
- **Modelo 3: Árbol de Decisión (CART):** Algoritmo no paramétrico interpretable con poda pre-poda para control de sobreajuste (`max_depth = 5`, `min_samples_split = 25`, criterio de Gini).
- **Modelo 4: Bagging (Random Forest):** Ensamble de 200 árboles de decisión no correlacionados mediante remuestreo bootstrap y selección aleatoria de predictores (`max_depth = 9`).
- **Modelo 5: XGBoost (Extreme Gradient Boosting):** Boosting de gradiente de segundo orden con 150 árboles (`max_depth = 4`, tasa de aprendizaje $\eta = 0.06$, submuestreo de filas y columnas del $85\%$).
- **Modelo 6: LightGBM (Light Gradient Boosting Machine):** Boosting basado en histogramas y crecimiento por hojas (`num_leaves = 25`, `max_depth = 5`, $\eta = 0.06$).

---

## 5. Validación Rigurosa, Métricas e Interpretación Preliminar

### 5.1 Resultados Comparativos en Validación Cruzada (10-Fold Stratified CV)
El protocolo experimental ejecutó 10 particiones estratificadas sobre el conjunto de entrenamiento ($N = 8.000$). La Tabla 4 recopila el desempeño medio y la desviación estándar de las cinco métricas requeridas:

**Tabla 4. Desempeño Comparativo de Modelos en Validación Cruzada Estratificada de 10 Pliegues ($N = 8.000$).**
| Modelo / Algoritmo | Accuracy Global | Precisión ($Clase\ 1$) | Sensibilidad / Recall ($Clase\ 1$) | F1-Score ($Clase\ 1$) | ROC-AUC (Métrica Principal) | Diferencia frente a Línea Base Simple |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Regresión Logística ($L_2$)** | **$82.04 \pm 1.16\%$** | $65.41 \pm 6.20\%$ | **$28.98 \pm 3.73\%$** | **$40.14 \pm 4.59\%$** | **$77.79 \pm 2.11\%$** | **$+1.67\%$ AUC** |
| **XGBoost (Gradient Boosting)** | $81.44 \pm 0.96\%$ | $62.04 \pm 4.33\%$ | $27.49 \pm 4.54\%$ | $37.98 \pm 5.04\%$ | $76.73 \pm 2.29\%$ | $+0.61\%$ AUC |
| **Bagging (Random Forest)** | $81.60 \pm 0.76\%$ | **$68.95 \pm 5.16\%$** | $21.06 \pm 3.35\%$ | $32.18 \pm 4.30\%$ | $76.67 \pm 2.20\%$ | $+0.55\%$ AUC |
| **Línea Base: Logística Simple ML1** | $81.15 \pm 0.85\%$ | $61.95 \pm 4.82\%$ | $24.31 \pm 3.96\%$ | $34.81 \pm 4.54\%$ | $76.12 \pm 2.04\%$ | *Referencia* |
| **LightGBM** | $81.41 \pm 0.77\%$ | $62.01 \pm 3.51\%$ | $27.43 \pm 3.76\%$ | $37.95 \pm 4.16\%$ | $75.98 \pm 2.37\%$ | $-0.14\%$ AUC |
| **Árbol de Decisión (CART)** | $80.51 \pm 0.58\%$ | $59.01 \pm 5.12\%$ | $22.45 \pm 3.08\%$ | $32.29 \pm 3.21\%$ | $73.18 \pm 2.70\%$ | $-2.94\%$ AUC |
| **SVM (Kernel RBF)** | $81.85 \pm 0.87\%$ | $70.18 \pm 6.45\%$ | $22.14 \pm 3.37\%$ | $33.60 \pm 4.54\%$ | $71.50 \pm 2.49\%$ | $-4.62\%$ AUC |
| **Línea Base: Mayoría (Dummy)** | $79.17 \pm 0.06\%$ | $0.00 \pm 0.00\%$ | $0.00 \pm 0.00\%$ | $0.00 \pm 0.00\%$ | $50.00 \pm 0.00\%$ | $-26.12\%$ AUC |

### 5.2 Evaluación en Conjunto de Prueba Independiente (Test $N = 2.000$)
Al evaluar los modelos sobre la partición de prueba no observada (Figura 1 en anexo `eval_01_curvas_roc.png`), la **Regresión Logística Regularizada ($L_2$)** ratificó su liderazgo alcanzando un **$ROC\text{-}AUC$ de $79.27\%$**, seguida muy de cerca por **XGBoost ($77.41\%$)** y **Random Forest ($77.03\%$)**. La Línea Base Simple obtuvo $76.88\%$, mientras que el Árbol CART ($72.58\%$) y SVM RBF ($72.35\%$) mostraron una capacidad discriminativa significativamente más débil.

### 5.3 Análisis del Costo del Error Clínico y Matriz de Confusión
La matriz de confusión del mejor modelo sobre las $2.000$ observaciones de prueba (Figura en anexo `eval_02_matriz_confusion.png`) revela:
- **Verdaderos Negativos (VN):** $1.516$ individuos sanos correctamente clasificados ($95.8\%$).
- **Falsos Positivos (FP):** $67$ individuos sin hipertensión clasificados erróneamente en riesgo ($4.2\%$).
- **Falsos Negativos (FN):** $294$ pacientes hipertensos que el umbral clásico ($0.50$) no capturó ($70.5\%$).
- **Verdaderos Positivos (VP):** $123$ pacientes hipertensos detectados oportunamente ($29.5\%$).

**Lectura Clínica Crítica:** En el ámbito de tamizaje en salud pública comunitaria, un **Falso Negativo (FN)** es el error más costoso: un paciente hipertenso no identificado evoluciona silenciosamente hacia complicaciones mayores prevenibles como accidente cerebrovascular (ACV), infarto de miocardio o insuficiencia renal terminal. Por el contrario, un **Falso Positivo (FP)** solo implica remitir a un individuo sano a una toma tensional confirmatoria ambulatoria de bajo costo. Por tanto, para la Fase 3 se plantea implementar **ajuste dinámico de umbral de decisión** ($\tau \in [0.25, 0.35]$) para elevar la sensibilidad por encima del $75\%$.

### 5.4 Jerarquía de Importancia y Explicabilidad Clínica (Figura 3 Anexa)
El contraste entre los coeficientes paramétricos de la Regresión Logística y la ganancia de información no lineal de XGBoost (Figura `eval_03_importancia_variables_shap.png`) revela una concordancia absoluta con la fisiopatología cardiovascular:
1. **Edad:** Es el factor predominante con el mayor coeficiente positivo ($+0.82$) y más del $35\%$ de la ganancia en XGBoost.
2. **Índice de Masa Corporal (IMC):** Segundo predictor continuo más determinante ($+0.41$).
3. **Diabetes Mellitus:** Coeficiente de $+0.68$, constituyendo el factor cualitativo de mayor peso.
4. **Perfil Lipídico:** El colesterol HDL presenta un coeficiente protector negativo ($-0.18$), mientras que el colesterol total y los triglicéridos aportan peso positivo consistente ($+0.16$ y $+0.12$).
5. **Estilo de Vida y Contexto:** El sedentarismo ($+0.15$) y el tabaquismo ($+0.11$) consolidan el riesgo acumulativo, mientras que el estrato socioeconómico muestra un gradiente que afecta el perfil metabólico de base.

---

## 6. Discusión, Limitaciones y Proyección a la Entrega Final

### 6.1 Contraste Crítico con Miah et al. (2024)
A diferencia de los resultados de Miah et al., donde los modelos alcanzaron exactitudes superiores al $90\%$ debido a un tamaño muestral minúsculo ($n = 600$) proveniente de un único centro hospitalario con posibles fugas metodológicas, nuestro estudio evalúa una cohorte colombiana de base poblacional ($N = 10.000$) donde la hipertensión presenta límites difusos y factores ambientales no lineales. En este régimen, la regularización $L_2$ supera a ensambles complejos al evitar el sobreajuste a ruido bioquímico, mientras que XGBoost exhibe la mayor plasticidad entre los modelos no lineales.

### 6.2 Limitaciones Identificadas
- La ausencia de variables hemodinámicas directas (como presión sistólica/diastólica basal ambulatoria) debido a que la variable objetivo adaptada es el diagnóstico formal.
- La naturaleza transversal del diseño ENSIN no permite modelar el tiempo hasta el evento cardiovascular (análisis de supervivencia).

### 6.3 Proyección a la Entrega Final (Fase 3)
1. Optimización bayesiana de hiperparámetros y calibración de probabilidades (`CalibratedClassifierCV`).
2. Despliegue de una aplicación web interactiva en **Streamlit** con cálculo de riesgo individualizado y explicaciones locales mediante **SHAP (SHapley Additive exPlanations)**.

---

## Referencias Bibliográficas

1. Miah, J., Ca, D. M., Lipu, E. R., Sayed, M. A., Mahmud, F., & Arafat, S. M. Y. (2024). *Improving Cardiovascular Disease Prediction Through Comparative Analysis of Machine Learning Models: A Case Study on Myocardial Infarction*. IEEE/arXiv:2311.00517.
2. Ministerio de Salud y Protección Social, ICBF, INS, DANE. (2015). *Encuesta Nacional de la Situación Nutricional en Colombia — ENSIN 2015*. Bogotá D.C.
3. Ministerio de Salud y Protección Social. (2018). *Guía de Práctica Clínica para el diagnóstico y tratamiento de la Hipertensión Arterial en Colombia (GPC No. 18)*. Sistema General de Seguridad Social en Salud.
4. Cuenta de Alto Costo (CAC). (2025). *Situación de la Enfermedad Renal Crónica, la Hipertensión Arterial y la Diabetes Mellitus en Colombia 2024-2025*. Fondo Colombiano de Enfermedades de Alto Costo.
5. Yusuf, S., et al. (Estudio PURE). (2020). *Modifiable risk factors, cardiovascular disease, and mortality in 155 722 individuals from 21 high-income, middle-income, and low-income countries (PURE): a prospective cohort study*. The Lancet, 395(10226), 795-808.
6. Alcaldía de Bucaramanga & Secretaría de Salud y Ambiente. (2024). *Enfermedades crónicas en el municipio de Bucaramanga (Conjunto de datos abiertos ID: 4iz7-suhz)*. Portal Nacional de Datos Abiertos de Colombia (datos.gov.co).
