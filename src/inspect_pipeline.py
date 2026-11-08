import joblib, pprint, json
path = r'C:/Users/ANDRES CABALLERO/Documents/Entrega Final Proyecto ML2/proyecto_repo/proyectoml2-prediccion-HA-main/models/mejor_modelo_hipertension.joblib'
pipeline = joblib.load(path)
print('Pipeline steps:', list(pipeline.named_steps.keys()))
# Show model type
model = pipeline.named_steps.get('model')
print('Model object:', model)
