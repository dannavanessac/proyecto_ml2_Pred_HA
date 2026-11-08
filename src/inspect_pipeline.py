"""
src/inspect_pipeline.py
Utilidad para inspeccionar el pipeline guardado en models/mejor_modelo_hipertension.joblib:
pasos, modelo final y variables de entrada que espera.
Uso (desde cualquier carpeta):  python src/inspect_pipeline.py
"""
from pathlib import Path
 
import joblib
 
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "mejor_modelo_hipertension.joblib"
 
pipeline = joblib.load(MODEL_PATH)
print("Pasos del pipeline:", list(pipeline.named_steps.keys()))
print("Modelo final:", pipeline.named_steps["classifier"])
print("Variables de entrada:", list(pipeline.feature_names_in_))
