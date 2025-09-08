from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from statsmodels.stats.contingency_tables import mcnemar

from .utils import p, ensure_dirs

def binary_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }

def cv_scores_auc(model, X, y, cv=5, random_state=42) -> np.ndarray:
    cvk = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=cvk, scoring="roc_auc", n_jobs=None)
    return scores

def ttest_paired(a: np.ndarray, b: np.ndarray) -> float:
    """
    t-test pareado (AUC en CV). Devuelve p-valor de dos colas.
    """
    from scipy.stats import ttest_rel
    stat, p = ttest_rel(a, b)
    return float(p)

def mcnemar_pvalue(y_true, y_pred_a, y_pred_b) -> float:
    """
    McNemar en test set (binario).
    """
    tab = confusion_matrix(y_pred_a, y_pred_b, labels=[0,1])
    # Construimos tabla de desacuerdos con respecto a ground truth:
    # b = A correcto / B incorrecto; c = A incorrecto / B correcto
    # Implementación con conteos directos:
    a_correct = (y_pred_a == y_true)
    b_correct = (y_pred_b == y_true)
    b = int(np.sum(a_correct & ~b_correct))
    c = int(np.sum(~a_correct & b_correct))
    # Statsmodels:
    result = mcnemar([[0, b],[c, 0]], exact=False, correction=True)
    return float(result.pvalue)

def save_metrics_table(df: pd.DataFrame, name: str, description: str):
    ensure_dirs()
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out = p("results", "tables", f"{name}_{ts}.csv")
    meta = p("results", "tables", f"{name}_{ts}.meta.json")
    df.to_csv(out, index=False)
    meta.write_text(json.dumps({"name": name, "description": description}, ensure_ascii=False, indent=2), encoding="utf-8")
    return out

def save_metrics_json(obj: dict, rel_path="results/metrics.json"):
    """
    Guarda JSON convirtiendo recursivamente tipos de numpy/pandas/Path a nativos de Python.
    Evita TypeError: Object of type int64/float32/... is not JSON serializable.
    """
    from pathlib import Path as _Path
    import numpy as _np
    import pandas as _pd

    def _to_python(o):
        # básicos
        if isinstance(o, (str, int, float, bool)) or o is None:
            return o
        # numpy escalares / arrays
        if isinstance(o, (_np.integer,)):
            return int(o)
        if isinstance(o, (_np.floating,)):
            return float(o)
        if isinstance(o, (_np.ndarray,)):
            return [_to_python(x) for x in o.tolist()]
        # pandas
        if isinstance(o, _pd.Timestamp):
            return o.isoformat()
        if isinstance(o, _pd.Series):
            return {k: _to_python(v) for k, v in o.to_dict().items()}
        if isinstance(o, _pd.DataFrame):
            return {k: [_to_python(v) for v in o.tolist()] for k, o in o.items()}
        # Path
        if isinstance(o, _Path):
            return str(o)
        # dict / list / tuple
        if isinstance(o, dict):
            return {str(k): _to_python(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_to_python(v) for v in o]
        # fallback
        try:
            return str(o)
        except Exception:
            return repr(o)

    ensure_dirs()
    out = p(rel_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    obj_py = _to_python(obj)
    out.write_text(json.dumps(obj_py, ensure_ascii=False, indent=2), encoding="utf-8")
    return out

from sklearn.metrics import precision_recall_curve, average_precision_score

def pr_auc(y_true, y_proba) -> float:
    """Área bajo la curva Precision–Recall (AP)."""
    return float(average_precision_score(y_true, y_proba))

def metrics_from_threshold(y_true, y_proba, thr: float) -> dict:
    """Métricas con un umbral específico."""
    y_pred = (y_proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0  # recall de la clase 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    roc = roc_auc_score(y_true, y_proba)
    pr = pr_auc(y_true, y_proba)
    return {
        "threshold": float(thr),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)
    }

def sweep_thresholds(y_true, y_proba, n=201) -> pd.DataFrame:
    """Barrido uniforme de umbrales en [0,1] y métricas resultantes."""
    thrs = np.linspace(0, 1, n)
    rows = [metrics_from_threshold(y_true, y_proba, t) for t in thrs]
    return pd.DataFrame(rows)

def choose_thresholds(y_true, y_proba, min_precision=0.75) -> dict:
    """
    Devuelve varias elecciones útiles de umbral:
    - f1_max: maximiza F1
    - youden: maximiza Youden J = recall + specificity - 1 (balancea ambas clases)
    - recall_at_precision: maximiza recall con precisión >= min_precision
    """
    df = sweep_thresholds(y_true, y_proba)
    # F1 máximo
    f1_idx = df["f1"].idxmax()
    # Youden J
    j_idx = (df["recall"] + df["specificity"] - 1).idxmax()
    # Recall sujeto a precisión mínima
    feas = df[df["precision"] >= min_precision]
    rp_idx = feas["recall"].idxmax() if not feas.empty else df["recall"].idxmax()
    picks = {
        "f1_max": df.loc[f1_idx].to_dict(),
        "youden": df.loc[j_idx].to_dict(),
        "recall_at_precision": df.loc[rp_idx].to_dict(),
    }
    return 

# Interpretabilidad / Importancias
from sklearn.inspection import permutation_importance

def get_feature_names_from_preprocessor(preproc, sample_df: pd.DataFrame | None = None) -> list[str]:
    """
    Devuelve nombres de salida del preprocesador.
    1) Intenta get_feature_names_out (si los steps lo soportan).
    2) Intenta en el paso 'prep' (ColumnTransformer).
    3) Fallback: usa salida pandas **solo en copia configurada** del preproc.
    """
    # 1) Directo
    try:
        return list(preproc.get_feature_names_out())
    except Exception:
        pass

    # 2) Paso 'prep'
    try:
        if hasattr(preproc, "named_steps") and "prep" in preproc.named_steps:
            return list(preproc.named_steps["prep"].get_feature_names_out())
    except Exception:
        pass

    # 3) Fallback con salida pandas sin tocar configuración global
    if sample_df is None:
        raise ValueError("get_feature_names_from_preprocessor necesita sample_df para el fallback.")

    try:
        # Algunos estimadores soportan .set_output(transform="pandas")
        preproc_pd = preproc.set_output(transform="pandas")
        Xt_sample = preproc_pd.transform(sample_df.iloc[:1])
        if hasattr(Xt_sample, "columns"):
            return [str(c) for c in Xt_sample.columns]
    except Exception:
        pass

    # Último recurso: dimensiones
    Xt_shape = preproc.transform(sample_df.iloc[:1]).shape
    return [f"f_{i}" for i in range(Xt_shape[1])]

def importances_logreg(model, feature_names: list[str]) -> pd.DataFrame:
    """|coef| normalizados como importancia."""
    coefs = np.ravel(model.coef_)
    imp = np.abs(coefs)
    imp = imp / (imp.sum() + 1e-12)
    return pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values("importance", ascending=False)

def importances_tree(model, feature_names: list[str]) -> pd.DataFrame:
    """feature_importances_ normalizadas."""
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return pd.DataFrame({"feature": feature_names, "importance": np.zeros(len(feature_names))})
    imp = imp / (imp.sum() + 1e-12)
    return pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values("importance", ascending=False)

def importances_permutation(model, X, y, feature_names: list[str], n_repeats=15, random_state=42) -> pd.DataFrame:
    """Permutation importance (útil para SVM RBF)."""
    pi = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state, scoring="roc_auc")
    imp = pi.importances_mean.clip(min=0)
    if imp.sum() > 0:
        imp = imp / imp.sum()
    return pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values("importance", ascending=False)

def feature_names_from_fitted_preproc(preproc, df_sample: pd.DataFrame, fare_col: str = "Fare") -> list[str]:
    """
    Reconstruye los nombres de salida del ColumnTransformer 'prep' ya *fiteado*,
    sin depender de get_feature_names_out(), preservando el orden real:
    primero numéricas (incluyendo Fare_log si se añade), luego categóricas OHE.
    """
    if not hasattr(preproc, "named_steps") or "prep" not in preproc.named_steps:
        raise ValueError("El preprocesador no contiene el paso 'prep' (ColumnTransformer).")

    ct = preproc.named_steps["prep"]  # ColumnTransformer ya ajustado
    if not hasattr(ct, "transformers_"):
        raise ValueError("El ColumnTransformer no está ajustado (llama .fit primero).")

    num_names, cat_names = [], []

    # --- NUMÉRICAS ---
    # Localiza el bloque numérico y sus columnas originales
    num_pipeline = None
    num_cols_spec = None
    for name, trans, cols in ct.transformers_:
        if name == "num":
            num_pipeline = trans  # suele ser Pipeline
            num_cols_spec = cols
            break

    if num_pipeline is not None:
        # 'cols' puede ser lista de nombres o índices; normalizamos a nombres
        if isinstance(num_cols_spec, (list, tuple, np.ndarray)):
            if len(num_cols_spec) > 0 and isinstance(num_cols_spec[0], (int, np.integer)):
                # índices -> nombres desde df_sample
                num_names = [str(df_sample.columns[i]) for i in num_cols_spec]
            else:
                num_names = [str(c) for c in num_cols_spec]
        else:
            # selector callable/boolean mask no esperado aquí; hacemos un fallback
            num_names = [c for c in df_sample.columns if c not in []]

        # Si el pipeline numérico contiene el paso 'add_fare_log', añadimos la nueva columna
        try:
            if hasattr(num_pipeline, "named_steps") and "add_fare_log" in num_pipeline.named_steps:
                if fare_col in num_names:
                    num_names = num_names + [f"{fare_col}_log"]
        except Exception:
            pass

    # --- CATEGÓRICAS ---
    cat_pipeline = None
    cat_cols_spec = None
    for name, trans, cols in ct.transformers_:
        if name == "cat":
            cat_pipeline = trans
            cat_cols_spec = cols
            break

    if cat_pipeline is not None:
        # Normalizamos columnas categóricas a nombres
        if isinstance(cat_cols_spec, (list, tuple, np.ndarray)):
            if len(cat_cols_spec) > 0 and isinstance(cat_cols_spec[0], (int, np.integer)):
                cat_input_names = [str(df_sample.columns[i]) for i in cat_cols_spec]
            else:
                cat_input_names = [str(c) for c in cat_cols_spec]
        else:
            cat_input_names = []

        # Hallamos el OneHotEncoder dentro del pipeline categórico
        from sklearn.preprocessing import OneHotEncoder
        ohe = None
        if hasattr(cat_pipeline, "named_steps"):
            for step in cat_pipeline.named_steps.values():
                if isinstance(step, OneHotEncoder):
                    ohe = step
                    break
        else:
            if isinstance(cat_pipeline, OneHotEncoder):
                ohe = cat_pipeline

        if ohe is None or not hasattr(ohe, "categories_"):
            # Si no encontramos OHE, devolvemos los nombres de entrada categóricos tal cual
            cat_names = cat_input_names
        else:
            # Construimos <col>_<cat> preservando el orden real de OHE
            cat_names = []
            for col_name, cats in zip(cat_input_names, ohe.categories_):
                for cat in cats:
                    cat_names.append(f"{col_name}_{cat}")

    # El orden de salida del ColumnTransformer es por defecto el orden declarado: num luego cat
    return num_names + cat_names

# Fairness helpers
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def _rates_from_confusion(cm: np.ndarray) -> dict:
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn + 1e-12)          # True Positive Rate (recall de la clase positiva)
    fpr = fp / (fp + tn + 1e-12)          # False Positive Rate
    precision = tp / (tp + fp + 1e-12)    # Precisión
    return {"TPR": float(tpr), "FPR": float(fpr), "Precision": float(precision)}

def group_metrics_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    group: pd.Series,
    group_name: str,
    positive_label: int = 1,
) -> pd.DataFrame:
    """
    Métricas por grupo:
    - TPR (recall positivo), FPR, Precision
    - Dem.Par. = P(ŷ=1 | grupo)  (paridad demográfica)
    """
    rows = []
    gser = pd.Series(group)
    for gval in gser.dropna().unique():
        idx = (gser == gval).values
        yt, yp = y_true[idx], y_pred[idx]
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        rates = _rates_from_confusion(cm)
        dem_par = float(np.mean(yp == positive_label))
        rows.append({
            "Atributo": group_name,
            "Grupo": str(gval),
            "TPR": rates["TPR"],
            "FPR": rates["FPR"],
            "Precision": rates["Precision"],
            "Dem.Par.": dem_par,
            "N": int(idx.sum()),
        })
    return pd.DataFrame(rows)

def group_metrics_intersection_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    group_a: pd.Series,
    name_a: str,
    group_b: pd.Series,
    name_b: str,
) -> pd.DataFrame:
    """Métricas por intersección A×B (p.ej., Sex×Pclass)."""
    rows = []
    ga, gb = pd.Series(group_a), pd.Series(group_b)
    for a in ga.dropna().unique():
        for b in gb.dropna().unique():
            mask = (ga == a).values & (gb == b).values
            if mask.sum() == 0:
                continue
            yt, yp = y_true[mask], y_pred[mask]
            cm = confusion_matrix(yt, yp, labels=[0, 1])
            rates = _rates_from_confusion(cm)
            dem_par = float(np.mean(yp == 1))
            rows.append({
                "Atributo": f"{name_a}×{name_b}",
                "Grupo": f"{name_a}={a} & {name_b}={b}",
                "TPR": rates["TPR"],
                "FPR": rates["FPR"],
                "Precision": rates["Precision"],
                "Dem.Par.": dem_par,
                "N": int(mask.sum()),
            })
    return pd.DataFrame(rows)

def sweep_group_thresholds(y_true: np.ndarray, y_prob: np.ndarray, group: pd.Series, n: int = 101) -> pd.DataFrame:
    """
    Barrido de umbrales por grupo: devuelve TPR/FPR/Precision/Dem.Par. por threshold.
    """
    thrs = np.linspace(0, 1, n)
    out = []
    gser = pd.Series(group)
    for g in gser.dropna().unique():
        mask = (gser == g).values
        yt_g, ys_g = y_true[mask], y_prob[mask]
        for t in thrs:
            yp_g = (ys_g >= t).astype(int)
            cm = confusion_matrix(yt_g, yp_g, labels=[0, 1])
            rates = _rates_from_confusion(cm)
            dem_par = float(np.mean(yp_g == 1))
            out.append({
                "Grupo": str(g), "threshold": float(t),
                "TPR": rates["TPR"], "FPR": rates["FPR"], "Precision": rates["Precision"],
                "Dem.Par.": dem_par, "N": int(mask.sum()),
            })
    return pd.DataFrame(out)
