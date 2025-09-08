from pathlib import Path
import json
import pandas as pd
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def p(*parts: str) -> Path:
    """Ruta segura relativa al root del proyecto."""
    return PROJECT_ROOT.joinpath(*parts)

def ensure_dirs():
    """Crea carpetas de resultados si no existen."""
    for sub in ["results/figures", "results/tables", "data/processed", "models"]:
        p(sub).mkdir(parents=True, exist_ok=True)

def save_table(df: pd.DataFrame, name: str, description: str = "") -> Path:
    """
    Guarda una tabla CSV y un .meta.json con descripción.
    """
    ensure_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{name}_{ts}"
    csv_path = p("results", "tables", f"{base}.csv")
    meta_path = p("results", "tables", f"{base}.meta.json")
    df.to_csv(csv_path, index=False)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"name": name, "description": description}, f, ensure_ascii=False, indent=2)
    return csv_path

def save_json(obj: dict, rel_path: str | Path) -> Path:
    """
    Guarda un dict como JSON relativo al root del proyecto.
    """
    ensure_dirs()
    out = p(rel_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return out

def save_caption(path: Path, caption: str):
    """
    Guarda un archivo .txt con el mismo nombre de la figura para el caption.
    """
    cap_path = path.with_suffix(".txt")
    cap_path.write_text(caption, encoding="utf-8")

def read_csv(path_like: str | Path) -> pd.DataFrame:
    """
    Lee CSV con dtype flexible (silencioso) y trims de columnas.
    """
    df = pd.read_csv(path_like)
    df.columns = [c.strip() for c in df.columns]
    return df
