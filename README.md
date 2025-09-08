# 🚢 Titanic ML Project

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1.1-9cf.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Proyecto end-to-end de *Machine Learning* aplicado al Titanic. Analizamos factores de supervivencia, construimos un pipeline reproducible, comparamos modelos, interpretamos resultados (SHAP) y evaluamos *fairness* con foco en las disparidades históricas de 1912. Incluye un **dashboard interactivo** en Streamlit.

---

## 📂 Estructura

```plaintext
titanic-ml-project/
├── README.md
├── requirements.txt
├── LICENSE
├── data/
│   ├── raw/               # Datos originales
│   ├── processed/         # Pipelines, features y datos procesados
│   └── README.md          # Diccionario de datos (dataset original)
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Modeling.ipynb
│   ├── 04_Interpretability.ipynb
│   └── 05_Fairness.ipynb
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── utils.py
├── models/
│   └── best_model.pkl
├── results/
│   ├── figures/
│   ├── tables/
│   └── metrics.json
└── dashboard/
    └── streamlit_app.py
````

---

## 🚀 Instalación

```bash
git clone https://github.com/usuario/titanic-ml-project.git
cd titanic-ml-project
pip install -r requirements.txt
```

Recomendado: usar un entorno virtual (e.g., `python -m venv .venv` y luego `source .venv/bin/activate` o `.\.venv\Scripts\activate` en Windows).

---

## ▶️ Uso

1. Ejecutar notebooks en orden (`notebooks/01_...` → `05_...`).
2. Lanzar el dashboard:

```bash
streamlit run dashboard/streamlit_app.py
```

> Si cargas pickles (`preprocessing_pipeline.pkl`, `best_model.pkl`) en el dashboard, asegúrate de ejecutar la app **desde la raíz del repo** para que `src/` sea importable.

---

## 🧭 Flujo del proyecto

1. **EDA**: estadísticas, distribución y relaciones; figuras con estilo consistente (300 DPI, captions claros y colorblind-friendly).
2. **Preprocesamiento**: imputaciones, one-hot, *feature engineering* (Title, FamilySize, IsAlone, HasCabin, Fare\_log, FarePerPerson).

   * Pipeline picklable guardado en `data/processed/preprocessing_pipeline.pkl`.
   * Definiciones en `feature_definitions.json`.
3. **Modelado**: LogReg, Random Forest, XGBoost, SVM.

   * Tabla comparativa, curvas ROC, matrices de confusión.
   * Búsqueda de hiperparámetros (Grid/Random Search).
4. **Umbrales y balance**: optimización de `recall`/`precision` por estrategia (Youden, F1-max, recall\@precision).
5. **Interpretabilidad**: SHAP summary, waterfall y top importances (comparación entre modelos).
6. **Fairness**: métricas por grupo (TPR, FPR, Precisión, Paridad Demográfica), visualización de disparidades, análisis interseccional y trade-offs.
7. **Dashboard (Streamlit)**: simulaciones por título social, tamaño de familia, tarifa y formulario general (sexo/edad/clase).

   * Interfaz **en español** y *disclaimer* ético/histórico.

---

## 📊 Resultados clave

* Variables más influyentes: **Sexo**, **Clase**, **Tarifa**, **Título social** y **Tamaño de familia**.
* La **Regresión Logística** ofreció un equilibrio sólido entre rendimiento e interpretabilidad; ajustamos umbrales para mejorar *recall* con control de *precision*.
* Se detectaron **disparidades** marcadas (p. ej., mujeres y 1.ª clase con mayor probabilidad de supervivencia), coherentes con el contexto histórico (“mujeres y niños primero”).

---

## ⚖️ Consideraciones éticas

* Este proyecto refleja **patrones históricos de 1912** y puede amplificar sesgos del dataset.
* Las simulaciones del dashboard son **educativas**, no prescriptivas, ni aplicables a contextos actuales.

---

## 📜 Licencia

Este proyecto está bajo licencia MIT. Ver [LICENSE](LICENSE).

````