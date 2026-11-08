"""
src/pipeline_builder.py
Módulo de construcción de Pipelines reproducibles y libres de fuga de datos (Data Leakage).
Utiliza ColumnTransformer, Imputación, Escalado Robusto y OneHotEncoding dentro del estándar scikit-learn.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder

# Definición de variables predictoras según la estructura del proyecto adaptado
NUMERIC_FEATURES = [
    'edad',
    'imc',
    'glucosa_ayunas',
    'colesterol_total',
    'colesterol_hdl',
    'colesterol_ldl',
    'trigliceridos',
    'estrato_socioeconomico'
]

CATEGORICAL_FEATURES = [
    'sexo',
    'zona_residencia',
    'regimen_salud',
    'actividad_fisica',
    'tabaquismo',
    'diabetes'
]

TARGET_COLUMN = 'hipertension_dx'

def get_train_test_data(data_path="data/processed/dataset_hipertension_colombia.csv", test_size=0.20, random_state=42):
    """
    Carga el dataset y ejecuta la partición estratificada Train/Test.
    Garantiza que la proporción de la clase positiva (HTA) se conserve idéntica en ambos subconjuntos.
    Retorna X_train, X_test, y_train, y_test.
    """
    df = pd.read_csv(data_path)
    
    # Excluir identificador administrativo no predictivo
    drop_cols = ['id_paciente', TARGET_COLUMN]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols]
    y = df[TARGET_COLUMN]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    print("\n" + "="*65)
    print("PARTICIÓN DE DATOS ESTRATIFICADA (PREVENCIÓN DE FUGA)")
    print("="*65)
    print(f"Total observaciones: {len(df)}")
    print(f"Conjunto de Entrenamiento (80%): {X_train.shape[0]} registros | Prevalencia HTA: {y_train.mean()*100:.2f}%")
    print(f"Conjunto de Prueba Independiente (20%): {X_test.shape[0]} registros | Prevalencia HTA: {y_test.mean()*100:.2f}%")
    print("="*65 + "\n")
    
    return X_train, X_test, y_train, y_test

def build_preprocessing_pipeline(numeric_features=None, categorical_features=None):
    """
    Construye el transformador de columnas modular sin fuga de datos.
    - Numéricas: Imputación por mediana + RobustScaler (resistente a extremos bioquímicos).
    - Categóricas: Imputación por moda + OneHotEncoder(drop='first', sparse_output=False).
    """
    if numeric_features is None:
        numeric_features = NUMERIC_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES
        
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor

def build_full_model_pipeline(classifier, numeric_features=None, categorical_features=None):
    """
    Ensambla el Pipeline completo combinando el preprocesador con cualquier estimador clasificador.
    Esto garantiza reproducibilidad estricta dentro de validación cruzada y GridSearchCV.
    """
    preprocessor = build_preprocessing_pipeline(numeric_features, categorical_features)
    
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    return full_pipeline

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_train_test_data()
    prep = build_preprocessing_pipeline()
    X_trans = prep.fit_transform(X_train)
    print("Pipeline ajustado en X_train exitosamente.")
    print(f"Dimensiones de entrada: {X_train.shape} -> Dimensiones procesadas: {X_trans.shape}")
