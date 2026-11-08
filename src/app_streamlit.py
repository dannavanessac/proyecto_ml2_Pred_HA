"""
Predictor de Hipertensión Arterial · Proyecto ML2 · Universidad Externado de Colombia
Ejecutar desde la raíz del repositorio:  streamlit run src/app_streamlit.py
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

# ---------------------------------------------------------------
# Rutas relativas al repositorio (funcionan en cualquier computador)
# ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "mejor_modelo_hipertension.joblib"
DATA_PATH = ROOT / "data" / "processed" / "dataset_hipertension_colombia.csv"

st.set_page_config(
    page_title="Predicción de Hipertensión Arterial",
    page_icon="🩺",
    layout="wide",
)

# ---------------------------------------------------------------
# Estilos: fondo dinámico azul hospitalario, tarjetas y tipografías
# ---------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Nunito:wght@400;600;700&display=swap');

@keyframes gradientBG { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
@keyframes driftCross { from{background-position:0 0} to{background-position:240px 240px} }
@keyframes floatBlobs { 0%{transform:translate(0,0) scale(1)} 50%{transform:translate(3%,-4%) scale(1.08)} 100%{transform:translate(0,0) scale(1)} }
@keyframes ecg { from{stroke-dashoffset:600} to{stroke-dashoffset:0} }

.stApp {
    isolation: isolate;
    font-family: 'Nunito', sans-serif;
    background: linear-gradient(135deg, #e6f3fc, #cfe6f8, #d6eef0, #dbe8fb);
    background-size: 400% 400%;
    animation: gradientBG 22s ease infinite;
}
/* Cruces médicas que se desplazan lentamente */
.stApp::before {
    content: ""; position: fixed; inset: 0; z-index: -1; pointer-events: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E%3Cpath d='M56 40h8v16h16v8H64v16h-8V64H40v-8h16z' fill='%230a6cc2' fill-opacity='0.07'/%3E%3C/svg%3E");
    animation: driftCross 50s linear infinite;
}
/* Halos de luz flotantes */
.stApp::after {
    content: ""; position: fixed; inset: -10%; z-index: -1; pointer-events: none;
    background:
        radial-gradient(circle at 15% 20%, rgba(64,156,255,.22), transparent 40%),
        radial-gradient(circle at 85% 30%, rgba(0,184,169,.16), transparent 38%),
        radial-gradient(circle at 60% 85%, rgba(30,100,200,.18), transparent 42%);
    animation: floatBlobs 28s ease-in-out infinite;
}
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1180px; padding-top: 2rem; }

h1, h2, h3, h4 { font-family: 'Poppins', sans-serif !important; color: #0b3d75; }

/* Encabezado tipo página web */
.hero {
    position: relative; overflow: hidden;
    background: linear-gradient(135deg, rgba(255,255,255,.92), rgba(232,244,255,.88));
    border: 1px solid rgba(10,108,194,.18); border-left: 8px solid #0a6cc2;
    border-radius: 22px; padding: 30px 36px 26px; margin-bottom: 22px;
    box-shadow: 0 12px 32px rgba(11,61,117,.14); backdrop-filter: blur(6px);
}
.hero-kicker { font-family:'Poppins',sans-serif; font-size:.78rem; letter-spacing:.16em; text-transform:uppercase; color:#0a6cc2; font-weight:600; }
.hero-title { font-family:'Poppins',sans-serif; font-size:2.3rem; line-height:1.15; font-weight:700; color:#0b3d75; margin:6px 0 10px; }
.hero-title span { color:#0a8fd6; }
.hero-text { font-size:1.05rem; color:#2c4a6b; max-width:760px; margin:0 0 14px; }
.chips { display:flex; flex-wrap:wrap; gap:8px; }
.chip { background:#e3f1fd; color:#0b3d75; border:1px solid #b9dcf7; border-radius:999px; padding:4px 13px; font-size:.82rem; font-weight:700; }
.ecg { position:absolute; right:24px; top:20px; width:300px; opacity:.55; }
.ecg path { fill:none; stroke:#0a8fd6; stroke-width:3; stroke-linecap:round; stroke-linejoin:round; stroke-dasharray:600; animation: ecg 3.2s linear infinite; }
@media (max-width: 900px) { .ecg { display:none; } .hero-title { font-size:1.7rem; } }

/* Tarjetas de sección (st.container con borde) */
[class*="st-key-card_"], div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,.86); border: 1px solid rgba(10,108,194,.16) !important;
    border-radius: 18px !important; box-shadow: 0 6px 20px rgba(11,61,117,.09);
}
.sec-title { font-family:'Poppins',sans-serif; font-weight:600; color:#0b3d75; font-size:1.08rem; margin:0 0 2px; }
.sec-sub { color:#5a7392; font-size:.9rem; margin:0 0 8px; }
[data-testid="stCaptionContainer"] { color:#5a7392; }

/* Botón principal */
[data-testid="stFormSubmitButton"] button, .stButton button {
    background: linear-gradient(135deg, #0a6cc2, #0a8fd6); color:#fff; border:none; border-radius:12px;
    font-family:'Poppins',sans-serif; font-weight:600; padding:.6rem 1.2rem; width:100%;
    box-shadow: 0 6px 16px rgba(10,108,194,.35); transition: transform .15s ease, box-shadow .15s ease;
}
[data-testid="stFormSubmitButton"] button:hover, .stButton button:hover { transform: translateY(-2px); box-shadow: 0 10px 22px rgba(10,108,194,.45); color:#fff; }
[data-testid="stForm"] { border:none !important; padding:0 !important; background:transparent !important; box-shadow:none !important; }

/* Pestañas */
.stTabs [data-baseweb="tab-list"] { gap:6px; }
.stTabs [data-baseweb="tab"] { font-family:'Poppins',sans-serif; font-weight:600; color:#0b3d75; background:rgba(255,255,255,.6); border-radius:12px 12px 0 0; padding:8px 18px; }
.stTabs [aria-selected="true"] { background:#fff; color:#0a6cc2; }

/* Resultado */
.res-card { text-align:center; padding:8px 4px 4px; }
.res-label { font-size:.85rem; color:#5a7392; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
.res-pct { font-family:'Poppins',sans-serif; font-size:3.4rem; font-weight:700; line-height:1.1; }
.res-bar { height:14px; background:#e1ecf7; border-radius:999px; overflow:hidden; margin:10px 0 12px; }
.res-fill { height:100%; border-radius:999px; }
.badge { display:inline-block; color:#fff; font-weight:700; border-radius:999px; padding:5px 16px; font-family:'Poppins',sans-serif; font-size:.92rem; }
.placeholder { text-align:center; color:#5a7392; padding:26px 8px; }
.placeholder .big { font-size:2.6rem; }
.disclaimer { background:#fff8e1; border:1px solid #f3d98b; color:#6b5200; border-radius:12px; padding:10px 14px; font-size:.86rem; margin-top:10px; }

/* Guía de variables */
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(330px,1fr)); gap:14px; }
.vcard { background:rgba(255,255,255,.9); border:1px solid rgba(10,108,194,.16); border-radius:16px; padding:16px 18px; box-shadow:0 4px 14px rgba(11,61,117,.08); }
.vcard h4 { margin:0 0 4px; font-size:1.02rem; }
.vtag { display:inline-block; font-size:.72rem; font-weight:700; color:#0a6cc2; background:#e3f1fd; border-radius:999px; padding:2px 10px; margin-bottom:6px; }
.vcard p { margin:4px 0; font-size:.92rem; color:#2c4a6b; }
.vcard .ref { color:#0b3d75; font-weight:700; }
"""
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def html(block: str):
    """Renderiza HTML sin que Markdown lo interprete como bloque de código."""
    clean = "\n".join(line.strip() for line in block.splitlines() if line.strip())
    st.markdown(clean, unsafe_allow_html=True)


def card(key: str):
    try:
        return st.container(border=True, key=f"card_{key}")
    except TypeError:  # versiones antiguas de Streamlit sin parámetro key
        return st.container(border=True)


# ---------------------------------------------------------------
# Carga de datos y modelo
# ---------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


pipeline = load_pipeline()
df = load_data()

# ---------------------------------------------------------------
# Descripción de cada variable (nombre de columna real del dataset)
# ---------------------------------------------------------------
VARS = {
    "edad": dict(
        label="Edad", group="Datos personales y sociales", unit="años",
        what="Años cumplidos de la persona.",
        enter="Escribe la edad en años (el modelo se entrenó con personas de 19 a 80 años).",
        ref="Rango del modelo: 19 – 80 años",
    ),
    "sexo": dict(
        label="Sexo", group="Datos personales y sociales", unit="",
        what="Sexo registrado de la persona.",
        enter="Selecciona Femenino o Masculino.",
        ref="",
    ),
    "estrato_socioeconomico": dict(
        label="Estrato socioeconómico", group="Datos personales y sociales", unit="1 a 6",
        what="Clasificación colombiana de las viviendas según sus servicios públicos; se usa como aproximación al nivel socioeconómico.",
        enter="Elige el estrato que aparece en la factura de servicios públicos (1 es el más bajo y 6 el más alto).",
        ref="Valores posibles: 1, 2, 3, 4, 5, 6",
    ),
    "zona_residencia": dict(
        label="Zona de residencia", group="Datos personales y sociales", unit="",
        what="Lugar donde vive habitualmente la persona.",
        enter="Urbana (ciudad o cabecera municipal) o Rural (campo o vereda).",
        ref="",
    ),
    "regimen_salud": dict(
        label="Régimen de salud", group="Datos personales y sociales", unit="",
        what="Tipo de afiliación al sistema de salud en Colombia.",
        enter="Contributivo: afiliado que cotiza (empleado o independiente). Subsidiado: afiliado con apoyo del Estado. Especial: regímenes de excepción (por ejemplo, fuerzas militares o magisterio).",
        ref="",
    ),
    "imc": dict(
        label="Índice de masa corporal (IMC)", group="Medidas corporales y hábitos", unit="kg/m²",
        what="Relación entre el peso y la estatura: peso (kg) ÷ estatura² (m).",
        enter="Escribe el IMC. Si no lo conoces, usa la calculadora que está debajo del campo.",
        ref="Normal: 18,5 – 24,9 · Sobrepeso: 25 – 29,9 · Obesidad: ≥ 30",
    ),
    "actividad_fisica": dict(
        label="Actividad física", group="Medidas corporales y hábitos", unit="",
        what="Nivel de movimiento físico en la vida diaria.",
        enter="Activo si hace ejercicio o se mueve con regularidad; Sedentario si casi no lo hace (la OMS recomienda al menos 150 minutos semanales de actividad moderada).",
        ref="",
    ),
    "tabaquismo": dict(
        label="Tabaquismo", group="Medidas corporales y hábitos", unit="",
        what="Si la persona fuma cigarrillo o tabaco.",
        enter="Sí si fuma; No si no fuma.",
        ref="",
    ),
    "diabetes": dict(
        label="Diabetes", group="Medidas corporales y hábitos", unit="",
        what="Diagnóstico médico previo de diabetes.",
        enter="Sí si un médico la ha diagnosticado; No en caso contrario.",
        ref="",
    ),
    "glucosa_ayunas": dict(
        label="Glucosa en ayunas", group="Exámenes de laboratorio", unit="mg/dL",
        what="Azúcar en la sangre medida después de al menos 8 horas sin comer.",
        enter="Valor del examen de laboratorio, en mg/dL.",
        ref="Normal: 70 – 99 · Prediabetes: 100 – 125 · Diabetes: ≥ 126",
    ),
    "colesterol_total": dict(
        label="Colesterol total", group="Exámenes de laboratorio", unit="mg/dL",
        what="Suma de todo el colesterol que circula en la sangre.",
        enter="Valor del perfil lipídico, en mg/dL.",
        ref="Deseable: < 200 · Límite alto: 200 – 239 · Alto: ≥ 240",
    ),
    "colesterol_hdl": dict(
        label="Colesterol HDL («bueno»)", group="Exámenes de laboratorio", unit="mg/dL",
        what="Lipoproteína que ayuda a retirar el colesterol de las arterias; entre más alto, mejor.",
        enter="Valor del perfil lipídico, en mg/dL.",
        ref="Bajo: < 40 · Protector: ≥ 60",
    ),
    "colesterol_ldl": dict(
        label="Colesterol LDL («malo»)", group="Exámenes de laboratorio", unit="mg/dL",
        what="Lipoproteína que puede acumularse en las paredes de las arterias.",
        enter="Valor del perfil lipídico, en mg/dL.",
        ref="Óptimo: < 100 · Límite alto: 130 – 159 · Alto: ≥ 160",
    ),
    "trigliceridos": dict(
        label="Triglicéridos", group="Exámenes de laboratorio", unit="mg/dL",
        what="Tipo de grasa presente en la sangre, relacionado con la alimentación y el exceso de peso.",
        enter="Valor del perfil lipídico, en mg/dL.",
        ref="Normal: < 150 · Límite alto: 150 – 199 · Alto: ≥ 200",
    ),
}

DISPLAY = {"Si": "Sí", "No": "No"}  # el dataset guarda «Si» sin tilde


def help_text(key: str) -> str:
    v = VARS[key]
    txt = f"{v['what']} {v['enter']}"
    return f"{txt} {v['ref']}".strip()


# ---------------------------------------------------------------
# Widgets de entrada
# ---------------------------------------------------------------
def num_input(key: str, step: float, integer: bool = False):
    v = VARS[key]
    col = df[key]
    lo, hi, med = col.min(), col.max(), col.median()
    if integer:
        args = dict(min_value=int(lo), max_value=int(hi), value=int(med), step=int(step))
    else:
        args = dict(min_value=float(lo), max_value=float(hi), value=round(float(med), 1), step=float(step), format="%.1f")
    unit = f" ({v['unit']})" if v["unit"] else ""
    val = st.number_input(f"{v['label']}{unit}", help=help_text(key), key=key, **args)
    st.caption(v["ref"] if v["ref"] else v["what"])
    return val


def cat_input(key: str, horizontal: bool = True):
    v = VARS[key]
    options = sorted(df[key].dropna().unique().tolist())
    default = options.index(df[key].mode().iloc[0])
    fmt = lambda x: DISPLAY.get(x, x)
    if horizontal and len(options) <= 2:
        val = st.radio(v["label"], options, index=default, format_func=fmt, horizontal=True, help=help_text(key), key=key)
    else:
        val = st.selectbox(v["label"], options, index=default, format_func=fmt, help=help_text(key), key=key)
    st.caption(v["what"])
    return val


# ---------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------
html("""
<div class="hero">
<svg class="ecg" viewBox="0 0 400 60"><path d="M0 30 L70 30 L82 30 L92 8 L104 52 L116 30 L190 30 L202 30 L212 14 L222 44 L232 30 L400 30"/></svg>
<div class="hero-kicker">Proyecto Machine Learning II · Universidad Externado de Colombia</div>
<div class="hero-title">Predictor de <span>Hipertensión Arterial</span></div>
<p class="hero-text">Herramienta de apoyo que estima la probabilidad de que una persona tenga hipertensión a partir de sus datos personales, hábitos y exámenes de laboratorio. Completa los campos (cada uno tiene una explicación) y pulsa <b>Calcular probabilidad</b>.</p>
<div class="chips">
<span class="chip">🧪 Modelo: regresión logística</span>
<span class="chip">📊 14 variables de entrada</span>
<span class="chip">🇨🇴 Contexto colombiano</span>
</div>
</div>
""")

tab_eval, tab_guia = st.tabs(["🩺 Evaluación del paciente", "📖 Guía de variables"])

# ---------------------------------------------------------------
# Pestaña 1: formulario + resultado
# ---------------------------------------------------------------
with tab_eval:
    col_form, col_res = st.columns([1.75, 1], gap="large")

    with col_form:
        with st.form("form_paciente", border=False):
            with card("personal"):
                html('<div class="sec-title">👤 Datos personales y sociales</div><div class="sec-sub">Información general de la persona.</div>')
                c1, c2 = st.columns(2)
                with c1:
                    edad = num_input("edad", 1, integer=True)
                    zona = cat_input("zona_residencia")
                    estrato = st.select_slider(
                        "Estrato socioeconómico", options=[1, 2, 3, 4, 5, 6],
                        value=int(df["estrato_socioeconomico"].median()),
                        help=help_text("estrato_socioeconomico"), key="estrato_socioeconomico",
                    )
                    st.caption("1 = más bajo · 6 = más alto (aparece en la factura de servicios)")
                with c2:
                    sexo = cat_input("sexo")
                    regimen = cat_input("regimen_salud", horizontal=False)

            with card("habitos"):
                html('<div class="sec-title">🏃 Medidas corporales y hábitos</div><div class="sec-sub">Peso, estilo de vida y antecedentes.</div>')
                c1, c2 = st.columns(2)
                with c1:
                    imc = num_input("imc", 0.1)
                    actividad = cat_input("actividad_fisica")
                with c2:
                    tabaco = cat_input("tabaquismo")
                    diabetes = cat_input("diabetes")

            with card("lab"):
                html('<div class="sec-title">🧪 Exámenes de laboratorio</div><div class="sec-sub">Valores de sangre en ayunas, en mg/dL.</div>')
                c1, c2 = st.columns(2)
                with c1:
                    glucosa = num_input("glucosa_ayunas", 0.1)
                    col_total = num_input("colesterol_total", 0.1)
                    trigl = num_input("trigliceridos", 0.1)
                with c2:
                    hdl = num_input("colesterol_hdl", 0.1)
                    ldl = num_input("colesterol_ldl", 0.1)

            enviado = st.form_submit_button("🔎 Calcular probabilidad")

        with st.expander("⚖️ ¿No conoces tu IMC? Calcúlalo aquí"):
            k1, k2 = st.columns(2)
            peso = k1.number_input("Peso (kg)", 30.0, 250.0, 70.0, 0.5)
            talla = k2.number_input("Estatura (m)", 1.0, 2.3, 1.70, 0.01)
            st.metric("Tu IMC es", f"{peso / talla ** 2:.1f} kg/m²")
            st.caption("Copia este valor en el campo «Índice de masa corporal (IMC)».")

    # --------- Resultado ---------
    with col_res:
        with card("resultado"):
            html('<div class="sec-title">📈 Resultado</div>')
            if not enviado:
                html("""
                <div class="placeholder"><div class="big">🩺</div>
                Completa los datos y pulsa <b>Calcular probabilidad</b> para ver el resultado aquí.</div>
                """)
            else:
                valores = {
                    "edad": edad, "sexo": sexo, "estrato_socioeconomico": estrato,
                    "zona_residencia": zona, "regimen_salud": regimen, "imc": imc,
                    "actividad_fisica": actividad, "tabaquismo": tabaco, "diabetes": diabetes,
                    "glucosa_ayunas": glucosa, "colesterol_total": col_total,
                    "colesterol_hdl": hdl, "colesterol_ldl": ldl, "trigliceridos": trigl,
                }
                df_input = pd.DataFrame([valores])[list(pipeline.feature_names_in_)]
                prob = float(pipeline.predict_proba(df_input)[0, 1])
                pred = int(pipeline.predict(df_input)[0])
                pct = prob * 100

                if pct < 20:
                    nivel, color = "Riesgo bajo", "#2e9e6b"
                elif pct < 50:
                    nivel, color = "Riesgo moderado", "#d99a00"
                elif pct < 80:
                    nivel, color = "Riesgo alto", "#e8590c"
                else:
                    nivel, color = "Riesgo muy alto", "#c92a2a"

                html(f"""
                <div class="res-card">
                <div class="res-label">Probabilidad de hipertensión</div>
                <div class="res-pct" style="color:{color}">{pct:.1f}%</div>
                <div class="res-bar"><div class="res-fill" style="width:{min(pct, 100):.1f}%;background:{color}"></div></div>
                <span class="badge" style="background:{color}">{nivel}</span>
                </div>
                """)
                st.write("")
                st.write(
                    "Clasificación del modelo (umbral 50 %): "
                    + ("**posible hipertensión (1)**" if pred == 1 else "**sin hipertensión (0)**")
                )
                html("""
                <div class="disclaimer">⚠️ Herramienta académica de apoyo. No reemplaza el diagnóstico médico:
                un resultado bajo no descarta la enfermedad y un resultado alto no la confirma. La presión arterial
                debe medirla un profesional de la salud.</div>
                """)

                # ---- Explicación SHAP (modelo lineal) ----
                with st.expander("🔍 ¿Qué factores influyeron en este resultado?"):
                    try:
                        prep = pipeline.named_steps["preprocessor"]
                        clf = pipeline.named_steps["classifier"]
                        fondo = prep.transform(df.drop(columns=["id_paciente", "hipertension_dx"]).sample(100, random_state=42))
                        x_t = prep.transform(df_input)

                        nombres = []
                        for n in prep.get_feature_names_out():
                            n = n.split("__", 1)[1]
                            etiqueta = next((VARS[k]["label"] + (": " + n[len(k) + 1:] if n != k else "")
                                             for k in VARS if n == k or n.startswith(k + "_")), n)
                            nombres.append(etiqueta)

                        explainer = shap.LinearExplainer(clf, fondo)
                        sv = explainer(x_t)
                        exp = shap.Explanation(values=sv.values[0], base_values=sv.base_values[0], feature_names=nombres)

                        plt.close("all")
                        shap.plots.waterfall(exp, max_display=8, show=False)
                        st.pyplot(plt.gcf(), clear_figure=True)
                        st.caption("Las barras rojas aumentan la probabilidad de hipertensión y las azules la disminuyen, comparado con una persona promedio.")
                    except Exception as e:
                        st.warning(f"No se pudo generar la explicación: {e}")

# ---------------------------------------------------------------
# Pestaña 2: guía de variables
# ---------------------------------------------------------------
with tab_guia:
    st.markdown("### ¿Qué significa cada variable?")
    st.caption("Consulta aquí qué mide cada campo y cómo diligenciarlo. Los valores de referencia son orientativos.")
    for grupo in ["Datos personales y sociales", "Medidas corporales y hábitos", "Exámenes de laboratorio"]:
        st.markdown(f"#### {grupo}")
        tarjetas = ""
        for k, v in VARS.items():
            if v["group"] != grupo:
                continue
            unidad = f"Unidad: {v['unit']}" if v["unit"] else "Categórica"
            ref = f'<p class="ref">📌 {v["ref"]}</p>' if v["ref"] else ""
            tarjetas += (
                f'<div class="vcard"><span class="vtag">{unidad}</span><h4>{v["label"]}</h4>'
                f'<p><b>¿Qué es?</b> {v["what"]}</p><p><b>¿Qué colocar?</b> {v["enter"]}</p>{ref}</div>'
            )
        html(f'<div class="grid">{tarjetas}</div>')
        st.write("")

st.caption(
    "Modelo entrenado con una cohorte analítica de 10.000 registros calibrada con parámetros de la ENSIN 2015 · "
    "Proyecto Integrador de Machine Learning II · Universidad Externado de Colombia"
)
