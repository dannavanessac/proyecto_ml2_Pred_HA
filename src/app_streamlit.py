import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Configuración de la página (estilo dinámico, colores médicos)
# ---------------------------------------------------------------
st.set_page_config(page_title="Predicción de Hipertensión Arterial", page_icon="🩺", layout="centered")

# Añadir CSS para fondo dinámico (aplicado a .stApp)
st.markdown(
    """
    <style>
    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    .stApp {
        background: linear-gradient(270deg, #e0f7fa, #b3e5fc, #e0f2f1);
        background-size: 600% 600%;
        animation: gradientBG 20s ease infinite;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #004080;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        font-weight: bold;
    }
    .stSelectbox > div > div > div {
        background-color: #e6f2ff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



st.title('🩺 Predicción de Hipertensión Arterial')
st.markdown('Ingrese los valores de los biomarcadores y variables sociodemográficas para obtener la probabilidad de diagnóstico de hipertensión.')

# ---------------------------------------------------------------
# Carga del pipeline entrenado
# ---------------------------------------------------------------
pipeline_path = r"C:/Users/ANDRES CABALLERO/Documents/Entrega Final Proyecto ML2/proyecto_repo/proyectoml2-prediccion-HA-main/models/mejor_modelo_hipertension.joblib"
pipeline = joblib.load(pipeline_path)

# ---------------------------------------------------------------
# Información descriptiva de cada variable (se muestra como tooltip/help)
# ---------------------------------------------------------------
variable_info = {
    "EDAD": "Edad del individuo en años.",
    "SEXO": "Sexo del individuo (Masculino / Femenino).",
    "IMC": "Índice de masa corporal (kg/m²).",
    "GLUCOSA": "Nivel de glucosa en sangre (mg/dL).",
    "COLESTEROL_TOTAL": "Colesterol total en sangre (mg/dL).",
    "COLESTEROL_HDL": "Colesterol HDL (colesterol \"bueno\") en sangre (mg/dL).",
    "COLESTEROL_LDL": "Colesterol LDL (colesterol \"malo\") en sangre (mg/dL).",
    "TRIGLICERIDOS": "Nivel de triglicéridos en sangre (mg/dL).",
    "DIABETES": "Presencia de diabetes (Sí / No).",
    "SEDENTARISMO": "Nivel de actividad física (Sí / No).",
    "HIPERTENSION_DX": "Variable objetivo: diagnóstico de hipertensión (1 = Sí, 0 = No).",
    # Completar con más variables si aparecen en el dataset
}

# ---------------------------------------------------------------
# Determinar las columnas del dataset procesado (excluyendo la variable objetivo)
# ---------------------------------------------------------------
sample_df = pd.read_csv(r"C:/Users/ANDRES CABALLERO/Documents/Entrega Final Proyecto ML2/proyecto_repo/proyectoml2-prediccion-HA-main/data/processed/dataset_hipertension_colombia.csv")
feature_names = [col for col in sample_df.columns if col != "HIPERTENSION_DX"]

# ---------------------------------------------------------------
# Formulario de ingreso con ayuda contextual (help)
# ---------------------------------------------------------------
st.subheader('📊 Ingrese los valores del paciente')
input_data = {}
for feat in feature_names:
    description = variable_info.get(feat, "")
    if pd.api.types.is_numeric_dtype(sample_df[feat]):
        default_val = float(sample_df[feat].median())
        val = st.number_input(feat, value=default_val, help=description)
    else:
        options = sorted(sample_df[feat].dropna().unique().tolist())
        val = st.selectbox(feat, options, help=description)
    input_data[feat] = val

# ---------------------------------------------------------------
# Botón de predicción
# ---------------------------------------------------------------
if st.button('🔎 Predecir'):
    df_input = pd.DataFrame([input_data])
    # Predicción de probabilidad y clase
    prob = pipeline.predict_proba(df_input)[0, 1]
    pred = pipeline.predict(df_input)[0]

    # Mensaje de probabilidad amigable
    prob_pct = prob * 100
    if prob_pct < 20:
        risk_level = "bajo"
    elif prob_pct < 50:
        risk_level = "moderado"
    elif prob_pct < 80:
        risk_level = "alto"
    else:
        risk_level = "muy alto"

    st.subheader('📈 Resultado de la predicción')
    st.write(
        f"Según el modelo, la probabilidad de que el paciente tenga **hipertensión** es aproximadamente **{prob_pct:.1f}%**. "
        f"Esto corresponde a un **riesgo {risk_level}**."
    )
    st.write(f"Clasificación binaria (0 = No hipertensión, 1 = Sí hipertensión): **{int(pred)}**")

    # -----------------------------------------------------------
    # Explicación SHAP (compatible con modelo dentro del pipeline)
    # -----------------------------------------------------------
    try:
        # Obtener modelo y preprocesador del pipeline
        model = pipeline.named_steps.get('classifier') or pipeline.named_steps.get('model')
        preprocessor = pipeline.named_steps.get('preprocessor')
        if model is None:
            raise ValueError('Modelo no encontrado en el pipeline.')

        # Transformar los datos con el preprocesador para que SHAP reciba la misma forma que el modelo espera
        X_transformed = preprocessor.transform(df_input) if preprocessor is not None else df_input

        # Usar un explainer genérico (funciona tanto para modelos lineales como de árbol)
        explainer = shap.Explainer(model, X_transformed)
        shap_values = explainer(X_transformed)
        st.subheader('🔎 Importancia de variables (SHAP)')
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(bbox_inches='tight')
    except Exception as e:
        st.warning(f'No se pudo generar explicación SHAP: {e}')
