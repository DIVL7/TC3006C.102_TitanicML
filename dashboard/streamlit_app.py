# =========================
# Imports y configuración
# =========================
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import joblib

from sklearn.pipeline import Pipeline

try:
    # para convertir decision_function en probas si el modelo no tiene predict_proba
    from scipy.special import expit
except Exception:
    expit = lambda z: 1 / (1 + np.exp(-z))

# --- Ubica la raíz del proyecto y asegúrala en sys.path para deserializar pickles con `src.*`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Importa símbolos usados dentro del pipeline para que el unpickler los resuelva
# (ajusta estos imports según tus clases/funciones reales en src.preprocessing)
try:
    from src.preprocessing import (
        AddFareLogTransformer,
        FeatureNamePassthrough,
    )
except Exception:
    # Si no existen (o no se usan), no pasa nada: el unpickler solo los necesita si están en el pipeline.
    pass

# =========================
# Constantes de rutas
# =========================
PIPE_PATH = ROOT / "data" / "processed" / "preprocessing_pipeline.pkl"
MODEL_PATH = ROOT / "models" / "best_model.pkl"  # cambia si tu archivo tiene otro nombre

TITLE_OPTIONS = ["Mr", "Mrs", "Miss", "Master", "Officer", "Royalty"]
EMBARKED_OPTIONS = {"Cherbourg (C)": "C", "Queenstown (Q)": "Q", "Southampton (S)": "S"}
PCLASS_LABELS = {1: "1ra", 2: "2da", 3: "3ra"}

# Paleta colorblind-friendly
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442"]

# Mapeo de sexo (UI -> dataset)
SEX_OPTIONS = {"Hombre": "male", "Mujer": "female"}


# =========================
# Utilidades de modelo
# =========================
@st.cache_resource(show_spinner=False)
def load_artifacts(pipe_path: Path, model_path: Path):
    preproc = joblib.load(pipe_path)
    raw_model = joblib.load(model_path)
    model = _unwrap_estimator(raw_model)
    return preproc, model


def _unwrap_estimator(obj):
    """
    Si 'obj' es un dict (p.ej. {'model': clf, ...} o {'best_estimator_': ...}),
    intenta recuperar un estimador sklearn con predict_proba/decision_function/predict.
    """
    # Caso 1: ya es estimador/pipeline con API sklearn
    if hasattr(obj, "predict_proba") or hasattr(obj, "decision_function") or hasattr(obj, "predict"):
        return obj

    # Caso 2: es Pipeline
    if isinstance(obj, Pipeline):
        return obj

    # Caso 3: es dict -> busca claves comunes
    if isinstance(obj, dict):
        for key in (
            "best_estimator_",
            "model",
            "estimator",
            "estimator_",
            "clf",
            "pipeline",
            "final_model",
            "sk_model",
        ):
            if key in obj:
                return _unwrap_estimator(obj[key])

    # Caso 4: lista/tupla con un estimador dentro
    if isinstance(obj, (list, tuple)) and len(obj) > 0:
        for x in obj:
            m = _unwrap_estimator(x)
            if m is not None and (hasattr(m, "predict") or hasattr(m, "predict_proba") or hasattr(m, "decision_function")):
                return m

    # Si no se pudo, se devuelve tal cual (fallará luego con mensaje claro)
    return obj


def predict_proba(preproc, model, df_raw: pd.DataFrame) -> float:
    """
    Aplica preproc (si el modelo NO es pipeline) y calcula probabilidad de Survived=1.
    Soporta modelos sin predict_proba usando decision_function -> sigmoide.
    """
    # ¿El modelo ya es un Pipeline completo? Entonces le pasamos df_raw directamente.
    if isinstance(model, Pipeline):
        X_in = df_raw
    else:
        X_in = preproc.transform(df_raw)

    # 1) Si tiene predict_proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_in)
        if proba.ndim == 1:
            proba = np.vstack([1.0 - proba, proba]).T
        return float(proba[:, 1][0])

    # 2) Si no tiene predict_proba pero sí decision_function
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_in)
        if isinstance(scores, (list, tuple)):
            scores = np.asarray(scores)
        if np.ndim(scores) == 1:
            p1 = expit(scores)
        else:
            # one-vs-rest binario: usa la segunda columna si existe; si no, usa la última
            if scores.shape[1] == 2:
                p1 = expit(scores[:, 1] - scores[:, 0])
            else:
                p1 = expit(scores[:, -1])
        return float(p1[0])

    # 3) Último recurso: predict 0/1 y devolvemos 0.0/1.0
    if hasattr(model, "predict"):
        yhat = model.predict(X_in)
        return float(np.clip(yhat[0], 0, 1))

    raise AttributeError(
        "El objeto cargado no expone predict_proba, decision_function ni predict. "
        "Revisa el contenido de tu pickle (guarda solo el estimador final) o cómo guardas el modelo."
    )


# =========================
# Utilidades de UI y datos
# =========================
def synth_name_with_title(title: str) -> str:
    # Formato parecido al dataset: "Surname, Title. GivenName"
    return f"Doe, {title}. John"


def base_input_row(
    title="Mr",
    sex="male",
    age=30,
    pclass=3,
    sibsp=0,
    parch=0,
    fare=32.2,
    embarked="S",
    cabin=None,
):
    """Construye un DataFrame de una fila con columnas 'raw' esperadas por el preproc."""
    name = synth_name_with_title(title)
    ticket = "A/5 21171"  # dummy estable
    row = {
        "PassengerId": 99999,
        "Survived": np.nan,  # ignorado
        "Pclass": int(pclass),
        "Name": name,
        "Sex": sex,
        "Age": float(age),
        "SibSp": int(sibsp),
        "Parch": int(parch),
        "Ticket": ticket,
        "Fare": float(fare),
        "Cabin": "" if cabin in [None, ""] else str(cabin),
        "Embarked": embarked,
    }
    return pd.DataFrame([row])


def section_header(title: str, subtitle: str = ""):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def prob_badge(p: float):
    pct = f"{100*p:.1f}%"
    if p >= 0.75:
        icon = "🌟"
    elif p >= 0.5:
        icon = "✅"
    elif p >= 0.25:
        icon = "⚠️"
    else:
        icon = "❌"
    st.markdown(f"### {icon} Probabilidad de sobrevivir: **{pct}**")


# =========================
# Configuración de página
# =========================
st.set_page_config(
    page_title="Simulador Titanic — Contexto histórico",
    page_icon="🚢",
    layout="wide",
)

st.title("🚢 Simulador de Supervivencia — Titanic (1912)")
st.markdown(
    """
**Objetivo:** explorar cómo **título social**, **tamaño de familia** y **tarifa** alteran
la probabilidad de supervivencia, y permitir una **simulación general**.  
**Importante:** Este simulador **no** es una herramienta de decisión actual.
Refleja patrones históricos (p. ej., *“mujeres y niños primero”*) presentes en los datos.
"""
)
with st.expander("⚖️ Disclaimer ético e histórico", expanded=True):
    st.write(
        "Este demo busca **ilustrar disparidades contextuales de la época**. "
        "Los resultados muestran cómo género y clase influyeron en 1912. "
        "No deben usarse para decisiones reales; el modelo puede amplificar "
        "sesgos del dataset original."
    )

# =========================
# Carga de artifacts
# =========================
try:
    preproc, model = load_artifacts(PIPE_PATH, MODEL_PATH)
except Exception as e:
    st.error(
        f"No fue posible cargar el pipeline o el modelo.\n\n"
        f"- Pipeline: `{PIPE_PATH}`\n- Modelo: `{MODEL_PATH}`\n\n{e}"
    )
    st.stop()

# =========================
# Sidebar — contexto base
# =========================
st.sidebar.header("Parámetros base")
base_sex_label = st.sidebar.selectbox(
    "Sexo", list(SEX_OPTIONS.keys()), index=0, help="Usado como base para las simulaciones."
)
base_sex = SEX_OPTIONS[base_sex_label]
base_age = st.sidebar.slider("Edad", 0, 80, 30, help="Edad base (años).")
base_pclass = st.sidebar.select_slider("Clase", options=[1, 2, 3], value=3, help="1=1ra, 2=2da, 3=3ra")
base_embarked = st.sidebar.selectbox("Puerto", list(EMBARKED_OPTIONS.keys()), index=2)
base_embarked_code = EMBARKED_OPTIONS[base_embarked]
st.sidebar.caption("Estos valores son el **contexto base** para las simulaciones de cada pestaña.")

tabs = st.tabs(
    ["🎖️ Título social", "👨‍👩‍👧‍👦 Tamaño de familia", "💷 Tarifa (Fare)", "🧪 Simulación general"]
)

# =========================
# Tab 1: Título social
# =========================
with tabs[0]:
    section_header("Título social", "Explora la influencia del *título* extraído del nombre (p. ej., Mr, Mrs, Miss…).")

    colA, colB = st.columns([1, 1])
    with colA:
        title_sel = st.selectbox("Selecciona un título", TITLE_OPTIONS, index=0)

    with colB:
        # Etiquetas claras en español
        acomp_hp = st.number_input("Acompañantes (hermanos/pareja)", min_value=0, max_value=8, value=0)
        padres_hijos = st.number_input("Familiares directos (padres/hijos)", min_value=0, max_value=8, value=0)
        fare_base = st.number_input("Tarifa (Fare)", min_value=0.0, max_value=600.0, value=32.2, step=0.1)

    df_row = base_input_row(
        title=title_sel,
        sex=base_sex,
        age=base_age,
        pclass=base_pclass,
        sibsp=acomp_hp,          # mapeo a SibSp del dataset
        parch=padres_hijos,      # mapeo a Parch del dataset
        fare=fare_base,
        embarked=base_embarked_code,
        cabin=None,
    )
    p_curr = predict_proba(preproc, model, df_row)
    prob_badge(p_curr)

    # Comparativa de todos los títulos
    st.markdown("**Comparación de probabilidades por título (parámetros base de la barra lateral):**")
    rows = []
    for t in TITLE_OPTIONS:
        df_t = df_row.copy()
        df_t.loc[:, "Name"] = synth_name_with_title(t)
        rows.append({"Título": t, "Probabilidad": predict_proba(preproc, model, df_t)})
    df_titles = pd.DataFrame(rows)

    chart = (
        alt.Chart(df_titles)
        .mark_bar()
        .encode(
            x=alt.X("Título:N", sort=TITLE_OPTIONS, title="Título"),
            y=alt.Y("Probabilidad:Q", scale=alt.Scale(domain=(0, 1)), title="P(sobrevivir)"),
            color=alt.Color("Título:N", scale=alt.Scale(range=PALETTE[:len(TITLE_OPTIONS)]), legend=None),
            tooltip=["Título", alt.Tooltip("Probabilidad:Q", format=".1%")],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, use_container_width=True)

# =========================
# Tab 2: Tamaño de familia
# =========================
with tabs[1]:
    section_header("Tamaño de familia", "Cambia los **acompañantes** y **familiares directos**; el pipeline creará FamilySize/IsAlone.")

    col1, col2 = st.columns(2)
    with col1:
        fam_size = st.slider("Tamaño de familia total (contándote a ti)", 1, 10, 3)
    with col2:
        st.caption("Se distribuye automáticamente: `Tamaño = Acompañantes (hermanos/pareja) + Familiares directos (padres/hijos) + 1`")

    # Genera pares (acomp_hp, padres_hijos) que sumen fam_size-1 y evalúa
    combos = []
    total = fam_size - 1
    for acomp in range(0, total + 1):
        padres = total - acomp
        df_tmp = base_input_row(
            title="Mr",
            sex=base_sex,
            age=base_age,
            pclass=base_pclass,
            sibsp=acomp,         # SibSp
            parch=padres,        # Parch
            fare=32.2,
            embarked=base_embarked_code,
        )
        combos.append({
            "Acompañantes (H/P)": acomp,
            "Familiares (P/H)": padres,
            "Probabilidad": predict_proba(preproc, model, df_tmp)
        })
    df_fs = pd.DataFrame(combos)

    st.markdown(f"**Probabilidades para Tamaño de familia = {fam_size}:**")
    chart_fs = (
        alt.Chart(df_fs)
        .mark_circle(size=120)
        .encode(
            x=alt.X("Acompañantes (H/P):Q", axis=alt.Axis(format="d"), title="Acompañantes (hermanos/pareja)"),
            y=alt.Y("Familiares (P/H):Q", axis=alt.Axis(format="d"), title="Familiares directos (padres/hijos)"),
            color=alt.Color("Probabilidad:Q", scale=alt.Scale(scheme="teals"), title="P(sobrevivir)"),
            tooltip=[
                alt.Tooltip("Acompañantes (H/P):Q", title="Acompañantes (H/P)", format="d"),
                alt.Tooltip("Familiares (P/H):Q", title="Familiares (P/H)", format="d"),
                alt.Tooltip("Probabilidad:Q", format=".1%", title="P(sobrevivir)"),
            ],
        )
        .properties(height=360)
    )
    st.altair_chart(chart_fs, use_container_width=True)

# =========================
# Tab 3: Tarifa (Fare)
# =========================
with tabs[2]:
    section_header("Tarifa (Fare)", "Explora cómo **Tarifa (Fare)** cambia la probabilidad, manteniendo el resto fijo.")

    fare_curr = st.slider("Tarifa (Fare)", 0.0, 600.0, 32.2, step=0.5)
    df_row = base_input_row(
        title="Mr",
        sex=base_sex,
        age=base_age,
        pclass=base_pclass,
        sibsp=0,
        parch=0,
        fare=fare_curr,
        embarked=base_embarked_code,
    )
    p_curr = predict_proba(preproc, model, df_row)
    prob_badge(p_curr)

    # Curva Fare → prob
    grid = np.linspace(0, 200, 60)
    series = []
    for f in grid:
        df_f = df_row.copy()
        df_f.loc[:, "Fare"] = f
        series.append({"Tarifa": f, "Probabilidad": predict_proba(preproc, model, df_f)})
    df_curve = pd.DataFrame(series)

    chart_f = (
        alt.Chart(df_curve)
        .mark_line(point=True)
        .encode(
            x=alt.X("Tarifa:Q", title="Tarifa (Fare)"),
            y=alt.Y("Probabilidad:Q", scale=alt.Scale(domain=(0, 1)), title="P(sobrevivir)"),
            color=alt.value(PALETTE[0]),
            tooltip=[alt.Tooltip("Tarifa:Q", format=".2f"), alt.Tooltip("Probabilidad:Q", format=".1%")],
        )
        .properties(height=320)
    )
    st.altair_chart(chart_f, use_container_width=True)

# =========================
# Tab 4: Simulación general
# =========================
with tabs[3]:
    section_header("Simulación general", "Introduce tus datos y estima la probabilidad (contexto 1912).")

    with st.form("sim_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            title_sel = st.selectbox("Título social", TITLE_OPTIONS, index=0)
            sex_label = st.selectbox("Sexo", list(SEX_OPTIONS.keys()), index=0)
            sex_sel = SEX_OPTIONS[sex_label]
            age_sel = st.number_input("Edad", min_value=0, max_value=80, value=30)
        with c2:
            pclass_sel = st.select_slider("Clase", options=[1, 2, 3], value=3)
            acomp_hp = st.number_input("Acompañantes (hermanos/pareja)", min_value=0, max_value=8, value=0)
            padres_hijos = st.number_input("Familiares directos (padres/hijos)", min_value=0, max_value=8, value=0)
        with c3:
            fare_sel = st.number_input("Tarifa (Fare)", min_value=0.0, max_value=600.0, value=32.2, step=0.1)
            embarked_sel = st.selectbox("Puerto", list(EMBARKED_OPTIONS.keys()), index=2)
            cabin_text = st.text_input("Cabina (opcional)", value="")  # vacío = sin cabina

        submitted = st.form_submit_button("Calcular probabilidad")
        if submitted:
            df_user = base_input_row(
                title=title_sel,
                sex=sex_sel,
                age=age_sel,
                pclass=pclass_sel,
                sibsp=acomp_hp,         # mapeo a SibSp
                parch=padres_hijos,     # mapeo a Parch
                fare=fare_sel,
                embarked=EMBARKED_OPTIONS[embarked_sel],
                cabin=cabin_text if cabin_text.strip() else None,
            )
            p_user = predict_proba(preproc, model, df_user)
            prob_badge(p_user)

            st.caption(
                "Nota: la predicción refleja patrones del dataset histórico. "
                "No implica causalidad ni es adecuada para decisiones reales."
            )

# Footer
st.markdown("---")
st.caption(
    "© Proyecto titanic-ml-project — Este demo reproduce patrones del Titanic (1912). "
    "Visualizaciones colorblind-friendly (Altair) y controles para explorar variables clave."
)
