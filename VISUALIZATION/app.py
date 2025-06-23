# Streamlit ML Playground
# -----------------------
# Autor: Dickin
# Última modificación: 23-06-2025 (radar chart + perfiles cluster)
#
# ▸ Novedades
#   • El gráfico de barras del centroide se reemplaza por un **Radar chart**
#     (Plotly polar) que muestra la forma del cluster.
#   • Cada número de cluster se mapea ahora a un perfil de jugadora con un
#     color asociado, que se muestra junto al resultado.
# -----------------------------------------------------------------------------

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ── Rutas --------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "trained_models"
if not MODEL_DIR.exists():
    MODEL_DIR = BASE_DIR / "models"
CONFIG_PATH = BASE_DIR / "config.json"

# ── Perfiles de cluster ------------------------------------------------------

CLUSTER_PROFILE = {
    0: {"name": "INTERIORES DEFENSIVAS",     "color": "blue"},
    1: {"name": "ANOTADORAS BASE",            "color": "red"},
    2: {"name": "EXTERIORES DE ROTACION",     "color": "green"},
    3: {"name": "INTERIORES DOMINANTES",      "color": "purple"},
}

# ── Helpers cacheados --------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    return joblib.load(MODEL_DIR / f"{model_name}.pkl")

# ── Side-bar inputs ----------------------------------------------------------

def _cast_numeric(v: Optional[Any], *, to_float: bool):
    if v is None:
        return None
    return float(v) if to_float else int(v)

def build_sidebar_inputs(metric_defs: Sequence[dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for m in metric_defs:
        fid = m["id"]
        label = m.get("label", fid)
        ftype = m.get("type", "float").lower()
        default = m.get("default", 0)

        if ftype == "str":
            out[fid] = st.sidebar.text_input(label, value=str(default))
            continue

        is_float = ftype != "int"
        fmt = "%.2f" if is_float else "%d"
        step_d = 1.0 if is_float else 1

        value = _cast_numeric(default, to_float=is_float)
        min_v = _cast_numeric(m.get("min"), to_float=is_float)
        max_v = _cast_numeric(m.get("max"), to_float=is_float)
        step_v = _cast_numeric(m.get("step", step_d), to_float=is_float)

        out[fid] = st.sidebar.number_input(
            label,
            min_value=min_v,
            max_value=max_v,
            value=value,
            step=step_v,
            format=fmt,
        )
    return out

# ── Funciones de casteo/dtype ------------------------------------------------

def ensure_float32(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns
    if not num_cols.empty:
        df[num_cols] = df[num_cols].astype(np.float32)
    return df

def align_kmeans_centers_dtype(model):
    """Convierte cluster_centers_ a float32 si es necesario."""
    if hasattr(model, "steps") and model.steps:
        km = model.steps[-1][-1]
    else:
        km = model
    if getattr(km, "__class__", None).__name__ == "KMeans" and hasattr(km, "cluster_centers_"):
        km.cluster_centers_ = km.cluster_centers_.astype(np.float32)

# ── Inferencia ---------------------------------------------------------------

def run_inference(model, df: pd.DataFrame) -> Dict[str, Any]:
    align_kmeans_centers_dtype(model)

    if hasattr(model, "cluster_centers_"):
        return {"tipo": "clustering", "cluster": int(model.predict(df)[0])}

    if hasattr(model, "predict_proba"):
        y_pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0]
        return {
            "tipo": "clasificacion",
            "prediccion": y_pred,
            "probabilidades": proba,
            "clases": list(getattr(model, "classes_", [])),
        }

    return {"tipo": "regresion", "prediccion": float(model.predict(df)[0])}

# ── Visualización ------------------------------------------------------------

def radar_from_centroid(centroid: pd.Series, color: str, name: str) -> go.Figure:
    """Devuelve un radar chart Plotly a partir de un centroide."""
    r = centroid.values.tolist() + [centroid.values[0]]  # cerrar polígono
    theta = centroid.index.tolist() + [centroid.index[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=r,
            theta=theta,
            fill="toself",
            name=name,
            line=dict(color=color),
        )
    )
    fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=False)
    return fig


def show_results(result: Dict[str, Any], df: pd.DataFrame, model):
    st.subheader("Resultados")
    tipo = result["tipo"]

    if tipo == "clustering":
        cid = result["cluster"]
        prof = CLUSTER_PROFILE.get(cid, {"name": "Perfil desconocido", "color": "gray"})
        st.markdown(
            f"**Cluster asignado:** {cid} — "
            f"<span style='color:{prof['color']}; font-weight:bold'>{prof['name']}</span>",
            unsafe_allow_html=True,
        )

        if hasattr(model, "cluster_centers_"):
            centers = pd.DataFrame(model.steps[-1][-1].cluster_centers_, columns=df.columns)
            centroid = centers.loc[cid]
            fig = radar_from_centroid(centroid, prof["color"], prof["name"])
            st.plotly_chart(fig, use_container_width=True)

    elif tipo == "clasificacion":
        st.markdown(f"**Clase predicha:** {result['prediccion']}")
        clases = result["clases"]
        if clases:
            fig = px.bar(
                pd.DataFrame({"Clase": clases, "Probabilidad": result["probabilidades"]}),
                x="Clase",
                y="Probabilidad",
                title="Distribución de probabilidades",
            )
            st.plotly_chart(fig, use_container_width=True)

    elif tipo == "regresion":
        st.markdown(f"**Valor predicho:** {result['prediccion']:.3f}")

# ── Main ---------------------------------------------------------------------

def main():
    st.set_page_config(page_title="ML Playground", layout="wide")
    st.title("🧪 ML Playground – Tus modelos PyCaret al instante")

    if not CONFIG_PATH.exists():
        st.error("No se encontró config.json. Revisa la ruta.")
        st.stop()

    try:
        cfg = load_config(CONFIG_PATH)
    except json.JSONDecodeError as e:
        st.error(f"Error de sintaxis en config.json: {e}")
        st.stop()

    model_names = sorted(cfg.keys())
    if not model_names:
        st.warning("No hay modelos definidos en config.json")
        st.stop()

    # Sidebar --------------------------------------------------------------
    st.sidebar.header("⚙️ Configuración")
    model_choice = st.sidebar.selectbox("Selecciona un modelo", model_names, index=0)

    metric_defs = cfg[model_choice]
    st.sidebar.subheader("🔢 Métricas de entrada")
    input_vals = build_sidebar_inputs(metric_defs)

    run_btn = st.sidebar.button("🚀 Ejecutar modelo", use_container_width=True)
    st.sidebar.markdown("---")
    st.sidebar.caption("Define 'model_col' en config.json si el nombre real difiere del mostrado.")

    st.markdown("---")

    if run_btn:
        row: Dict[str, Any] = {}
        for m in metric_defs:
            row[m.get("model_col") or m["id"].replace("_", " ")] = input_vals[m["id"]]

        df = pd.DataFrame([row])
        model_path = MODEL_DIR / f"{model_choice}.pkl"
        if not model_path.exists():
            st.error(f"No se encontró {model_path.relative_to(BASE_DIR)}")
            st.stop()
        model = load_model(model_choice)

        if hasattr(model, "feature_names_in_"):
            missing = [c for c in model.feature_names_in_ if c not in df.columns]
            if missing:
                st.error(f"Faltan columnas requeridas: {missing}.")
                st.stop()
            df = df[model.feature_names_in_]

        df = ensure_float32(df)

        res = run_inference(model, df)
        show_results(res, df, model)
    else:
        st.info("Rellena las métricas y pulsa **Ejecutar modelo** para ver resultados.")

if __name__ == "__main__":
    main()
