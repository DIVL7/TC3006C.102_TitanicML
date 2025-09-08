import numpy as np
import pandas as pd

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas (trim)."""
    out = df.copy()
    out.columns = [c.strip() for c in df.columns]
    return out


def engineer_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ingeniería de variables base para Titanic:
      - FamilySize = SibSp + Parch + 1
      - IsAlone = 1 si FamilySize <= 1
      - Title (extraído de Name, agrupando títulos raros)
      - Age_bin (bines de edad)
      - FarePerPerson = Fare / FamilySize (FamilySize==0 -> 1)
      - HasCabin = 1 si Cabin no es nulo
    """
    out = df.copy()

    # Mapeo defensivo
    col = {c.lower(): c for c in out.columns}
    def g(name: str): return col.get(name.lower())

    sibsp = g("SibSp"); parch = g("Parch")
    age = g("Age"); fare = g("Fare")
    name = g("Name"); cabin = g("Cabin")

    # FamilySize & IsAlone
    if sibsp and parch:
        out["FamilySize"] = out[sibsp].fillna(0) + out[parch].fillna(0) + 1
        out["IsAlone"] = (out["FamilySize"] <= 1).astype(int)
    else:
        out["FamilySize"] = np.nan
        out["IsAlone"] = np.nan

    # Title (agrupando títulos poco frecuentes)
    if name:
        titles = out[name].astype(str).str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
        rare_map = {
            "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
            "Lady": "Royalty", "Countess": "Royalty", "the Countess": "Royalty",
            "Sir": "Royalty", "Jonkheer": "Royalty", "Don": "Royalty", "Dona": "Royalty",
            "Capt": "Officer", "Col": "Officer", "Major": "Officer", "Dr": "Officer", "Rev": "Officer"
        }
        out["Title"] = titles.replace(rare_map)
    else:
        out["Title"] = np.nan

    # Bines de edad
    if age:
        out["Age_bin"] = pd.cut(
            out[age],
            bins=[0, 12, 18, 30, 45, 60, 80],
            labels=["0-11", "12-17", "18-29", "30-44", "45-59", "60-79"]
        )
    else:
        out["Age_bin"] = np.nan

    # Fare por persona
    if fare and "FamilySize" in out:
        denom = out["FamilySize"].replace(0, 1)
        out["FarePerPerson"] = out[fare] / denom
    else:
        out["FarePerPerson"] = np.nan

    # HasCabin
    if cabin:
        out["HasCabin"] = (~out[cabin].isna()).astype(int)
    else:
        out["HasCabin"] = np.nan

    return out

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

class EngineerBasicFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Aplica engineer_basic_features a un DataFrame y devuelve DataFrame.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return engineer_basic_features(X)

FARE = "Fare"  # ajusta si tu constante tiene otro nombre

class AddFareLogTransformer(BaseEstimator, TransformerMixin):
    """
    Agrega la columna Fare_log = log1p(Fare).
    - Soporta entrada como DataFrame o ndarray y devuelve del mismo tipo.
    - Acepta 'fare_col' (nombre de columna) y/o 'fare_idx' (índice entero) para compatibilidad.
    """
    def __init__(self, fare_col: str = FARE, fare_idx: int | None = None):
        self.fare_col = fare_col
        self.fare_idx = fare_idx
        self.is_dataframe_ = False

    def fit(self, X, y=None):
        # Detecta tipo de entrada que verá en transform
        self.is_dataframe_ = hasattr(X, "iloc")
        # No necesitamos aprender nada; solo validar si aplica
        return self

    def transform(self, X):
        is_df = hasattr(X, "iloc")

        if is_df:
            X_out = X.copy()
            # Si existe la columna por nombre, la usamos
            if self.fare_col in X_out.columns:
                vals = pd.to_numeric(X_out[self.fare_col], errors="coerce").fillna(0.0).to_numpy()
                X_out[self.fare_col + "_log"] = np.log1p(vals)
            # Si no existe, no hacemos nada (ya pudo haber sido transformada antes)
            return X_out

        # ndarray
        if isinstance(self.fare_idx, int) and 0 <= self.fare_idx < X.shape[1]:
            vals = X[:, self.fare_idx]
            fare_log = np.log1p(vals)
            return np.hstack([X, fare_log.reshape(-1, 1)])

        # Si no tenemos ni columna (en DF) ni índice (en ndarray), devolvemos tal cual
        return X

def get_feature_definitions() -> dict:
    return {
        "Title": {
            "description": "Título extraído del nombre",
            "type": "categorical",
            "creation_method": "regex extraction (agrupado en {Mr, Mrs, Miss, Master, Officer, Royalty})",
            "missing_handling": "none",
            "values": ["Mr", "Mrs", "Miss", "Master", "Officer", "Royalty"]
        },
        "FamilySize": {
            "description": "Tamaño de familia = SibSp + Parch + 1",
            "type": "numeric",
            "creation_method": "sum",
            "missing_handling": "fillna(0) en SibSp/Parch antes de sumar"
        },
        "IsAlone": {
            "description": "Indicador de viajar solo (FamilySize <= 1)",
            "type": "binary",
            "creation_method": "threshold on FamilySize",
            "missing_handling": "derivado de FamilySize"
        },
        "FarePerPerson": {
            "description": "Tarifa por persona ~ Fare / FamilySize",
            "type": "numeric",
            "creation_method": "division",
            "missing_handling": "FamilySize==0 -> 1"
        },
        "Fare_log": {
            "description": "Transformación logarítmica de Fare",
            "type": "numeric",
            "creation_method": "log1p (natural) aplicado después de imputación",
            "missing_handling": "median imputation previa"
        },
        "HasCabin": {
            "description": "Indicador de tener número de cabina conocido",
            "type": "binary",
            "creation_method": "notnull(Cabin)",
            "missing_handling": "none"
        },
        "Age_bin": {
            "description": "Bines de edad",
            "type": "categorical_ordered",
            "creation_method": "pd.cut en [0,12,18,30,45,60,80] con etiquetas ['0-11','12-17','18-29','30-44','45-59','60-79']",
            "missing_handling": "depende de imputación de Age previa si se aplica en modelado"
        }
    }

def build_preprocessing_pipeline(df_sample: pd.DataFrame) -> Pipeline:
    """
    Pipeline común:
      1) EngineerBasicFeaturesTransformer (DataFrame -> DataFrame)
      2) ColumnTransformer:
         - num: imputer(median) -> AddFareLogTransformer -> scaler
         - cat: imputer(most_frequent) -> OneHotEncoder(ignore)
    """
    # Nombres defensivos
    col = {c.lower(): c for c in df_sample.columns}
    def g(name: str): return col.get(name.lower())

    SURV = g("Survived")
    PCLASS = g("Pclass")
    SEX = g("Sex")
    AGE = g("Age")
    FARE = g("Fare")
    EMB = g("Embarked")

    # Columnas (incluye features que generaremos)
    engineered_num = ["FamilySize", "FarePerPerson"]
    engineered_cat = ["Title", "HasCabin", "IsAlone", "Age_bin"]

    numeric_features = [x for x in [AGE, FARE, g("SibSp"), g("Parch")] if x] + engineered_num
    categorical_features = [x for x in [SEX, PCLASS, EMB] if x] + engineered_cat

    # Índice de FARE dentro del bloque numérico (tras engineer_basic_features + selección)
    fare_idx = None
    if FARE in numeric_features:
        fare_idx = numeric_features.index(FARE)

    # Pipelines
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("add_fare_log", AddFareLogTransformer(fare_idx=fare_idx)),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ])

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", ohe),
    ])

    prep = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True
    )

    pipe = Pipeline(steps=[
        ("feature_eng", EngineerBasicFeaturesTransformer()),
        ("prep", prep),
    ])

    return pipe
