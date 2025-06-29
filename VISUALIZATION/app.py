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
DATA_PATH = BASE_DIR / "data" / "LFENDESA_UnifiedData.csv"


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

@st.cache_data(show_spinner="Cargando datos…")
def load_players() -> pd.DataFrame:
    df = pd.read_csv(
        DATA_PATH, sep=";", decimal=".",  # separador «;»  :contentReference[oaicite:3]{index=3}
        dtype={"Temporada": int, "Equipo": str, "Nombre": str}
    )
    # Convertir a numéricas todas las columnas salvo las 3 primeras
    num_cols = df.columns[3:]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    return df

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


def show_results(name: str, team:str, result: Dict[str, Any], df: pd.DataFrame, model, *, top_k: int = 5) -> None:
    if name or team:
        st.subheader(f"Resultados 🤖 {name} &nbsp;&nbsp;|&nbsp;&nbsp;**Equipo:** {team}")
    else:
        st.subheader("Resultados")

    tipo = result["tipo"]


    # ── CLUSTERING ──────────────────────────────────────────────────────────
    if tipo == "clustering":
        cid = result["cluster"]
        prof = CLUSTER_PROFILE.get(cid, {"name": "Perfil desconocido", "color": "gray"})
        st.markdown(
            (
                f"**Cluster asignado:** {cid} — "
                f"<span style='color:{prof['color']}; font-weight:bold'>{prof['name']}</span>"
            ),
            unsafe_allow_html=True,
        )

        # ▸ Radar del centroide ------------------------------------------------
        if hasattr(model, "cluster_centers_"):
            centers = pd.DataFrame(model.steps[-1][-1].cluster_centers_, columns=df.columns)
            centroid = centers.loc[cid]
            fig_rad = radar_from_centroid(centroid, prof["color"], prof["name"])
            st.plotly_chart(fig_rad, use_container_width=True)
        else:
            centroid = None  # fallback por si no hay KMeans estándar

        # ▸ Top‑k métricas de la jugadora --------------------------------------
        numeric_metrics = df.select_dtypes(include=[np.number]).iloc[0]
        if numeric_metrics.empty:
            st.info("No se encontraron métricas numéricas para la jugadora.")
            return

        top_metrics = numeric_metrics.sort_values(ascending=False).head(top_k)
        fig_bar = px.bar(
            x=top_metrics.index,
            y=top_metrics.values,
            title=f"Top {top_k} métricas de la jugadora",
            labels={"x": "Métrica", "y": "Valor"},
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ▸ Comparativa Jugadora vs Centroide ----------------------------------
        if centroid is not None:
            common_cols = [c for c in top_metrics.index if c in centroid.index]
            if common_cols:
                comp_df = pd.DataFrame(
                    {
                        "Métrica": common_cols,
                        "Jugadora": numeric_metrics[common_cols].values,
                        "Centroide": centroid[common_cols].values,
                    }
                )
                fig_comp = px.bar(
                    comp_df.melt(id_vars="Métrica", var_name="Entidad", value_name="Valor"),
                    x="Métrica",
                    y="Valor",
                    color="Entidad",
                    barmode="group",
                    title="Comparación Jugadora vs Centroide (métricas TOP)",
                )
                st.plotly_chart(fig_comp, use_container_width=True)

    # ── CLASIFICACIÓN ────────────────────────────────────────────────────────
    elif tipo == "clasificacion":
        # Mapeo de clase a etiqueta
        class_labels = {0: "Bajo", 1: "Medio", 2: "Alto"}
        pred = result['prediccion']
        clases = result.get("clases", [])
        probabilidades = result.get("probabilidades", [])

        # Si las clases son ints, mapea a etiquetas
        clase_predicha = class_labels.get(pred, str(pred))

        st.markdown(f"**Clase predicha:** {clase_predicha}")

        if clases:
            # Mapea las clases a etiquetas para el gráfico
            etiquetas = [class_labels.get(c, str(c)) for c in clases]
            fig = px.pie(
                pd.DataFrame({
                    "Clase": etiquetas,
                    "Probabilidad": probabilidades,
                }),
                names="Clase",
                values="Probabilidad",
                title=f"Distribución de probabilidades (Clase predicha: {clase_predicha})",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig.update_traces(textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

    # ── REGRESIÓN ────────────────────────────────────────────────────────────
    elif tipo == "regresion":
        st.markdown(f"**Valor predicho:** {result['prediccion']:.3f}")

# ── Métricas dataset ---------------------------------------------------------

def show_dataset_metrics(df: pd.DataFrame, cfg: dict | None = None) -> None:
    """Renderiza métricas descriptivas enriquecidas del *dataset*.

    ▸ **Resumen estadístico** (tabla interactiva).
    ▸ **Distribuciones**: histograma o boxplot para cualquier métrica numérica.
    ▸ **Serie temporal**: media de una métrica por temporada.
    ▸ **Correlaciones**: mapa de calor de Pearson entre métricas.
    """

    st.subheader("📊 Métricas generales del dataset")

    # ── Selección de columnas numéricas ──────────────────────────────────────
    num_cols = df.select_dtypes(include=[np.number]).columns
    if num_cols.empty:
        st.warning("No se encontraron columnas numéricas en el CSV.")
        return

    # ── Pestañas ----------------------------------------------------------------
    tab_resumen, tab_dist, tab_serie, tab_corr = st.tabs(
        ["Resumen", "Distribuciones", "Serie temporal", "Correlación"]
    )

    # ▸ Resumen estadístico ---------------------------------------------------
    with tab_resumen:
        summary = df[num_cols].describe().T  # transposed for readability
        summary.rename(
            columns={
                "mean": "media",
                "std": "std",
                "min": "mín",
                "25%": "q1",
                "50%": "mediana",
                "75%": "q3",
                "max": "máx",
            },
            inplace=True,
        )
        st.dataframe(summary, use_container_width=True)

    # ▸ Distribuciones --------------------------------------------------------
    with tab_dist:
        col1, col2 = st.columns(2)
        with col1:
            metric = st.selectbox("Variable", num_cols, index=0)
        with col2:
            chart_kind = st.radio("Tipo de gráfico", ["Histograma", "Boxplot"], horizontal=True)

        if chart_kind == "Histograma":
            fig = px.histogram(df, x=metric, nbins=30, title=f"Distribución de {metric}")
        else:
            fig = px.box(df, y=metric, points="all", title=f"Boxplot de {metric}")
        st.plotly_chart(fig, use_container_width=True)

    # ▸ Serie temporal --------------------------------------------------------
    with tab_serie:
        if "Temporada" in df.columns:
            default_metric = "Puntos" if "Puntos" in num_cols else num_cols[0]
            metric_ts = st.selectbox("Métrica", num_cols, index=num_cols.get_loc(default_metric))
            df_ts = (
                df.groupby("Temporada")[metric_ts].mean()
                .reset_index()
                .sort_values("Temporada")
            )
            fig = px.line(
                df_ts,
                x="Temporada",
                y=metric_ts,
                markers=True,
                title=f"Evolución de {metric_ts} por temporada",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("La columna 'Temporada' no está disponible para serie temporal.")

    # ▸ Correlación -----------------------------------------------------------
    with tab_corr:
        corr = df[num_cols].corr().round(2)
        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu",
            title="Matriz de correlación (Pearson)",
        )
        st.plotly_chart(fig, use_container_width=True)

# ── Main ---------------------------------------------------------------------

def main():
    st.set_page_config(page_title="🏀", layout="wide")
    st.title("🏀 Predición de Rendimiento y Resultados en Baloncesto")

    if not CONFIG_PATH.exists():
        st.error("No se encontró config.json. Revisa la ruta.")
        st.stop()

    try:
        cfg = load_config(CONFIG_PATH)
    except json.JSONDecodeError as e:
        st.error(f"Error de sintaxis en config.json: {e}")
        st.stop()
        
    # Cargar dataset completo una sola vez (para la pestaña de métricas)
    df_dataset = load_players()
    metrics_label = "📊 Métricas dataset"
    model_display_map = {
        v["Model Name"]: k
        for k, v in cfg.items()
        if "Model Name" in v
    } 
    
    model_names = [metrics_label] + list(model_display_map.keys())

    if not model_names:
        st.warning("No hay modelos definidos en config.json")
        st.stop()

    # Sidebar --------------------------------------------------------------
    st.sidebar.header("⚙️ Configuración")
    
    options = model_names
    
    selection = st.sidebar.selectbox("Selecciona un modelo", options, index=0)
    if selection == metrics_label:
        # Ocultar elementos de entrada innecesarios en la sidebar
        st.sidebar.markdown("---")
        st.sidebar.caption("Las métricas del dataset no requieren entradas.")
        show_dataset_metrics(df_dataset, cfg)
    else:
        model_key = model_display_map[selection]
        metric_defs = cfg[model_key]["Metrics"]
        st.sidebar.subheader("🔢 Métricas de entrada")
        input_vals = build_sidebar_inputs(metric_defs)
        nombre = input_vals['Nombre']
        equipo = input_vals['Equipo']
        
        run_btn = st.sidebar.button("🚀 Ejecutar modelo", use_container_width=True)
        st.sidebar.markdown("---")
        st.sidebar.caption("Define 'model_col' en config.json si el nombre real difiere del mostrado.")

        st.markdown("---")

        if run_btn:
            row: Dict[str, Any] = {}
            for m in metric_defs:
                row[m.get("model_col") or m["id"].replace("_", " ")] = input_vals[m["id"]]

            df = pd.DataFrame([row])
            model_path = MODEL_DIR / f"{model_key}.pkl"
            if not model_path.exists():
                st.error(f"No se encontró {model_path.relative_to(BASE_DIR)}")
                st.stop()
            model = load_model(model_key)

            if hasattr(model, "feature_names_in_"):
                missing = [c for c in model.feature_names_in_ if c not in df.columns]
                if missing:
                    st.error(f"Faltan columnas requeridas: {missing}.")
                    st.stop()
                df = df[model.feature_names_in_]

            df = ensure_float32(df)

            res = run_inference(model, df)
            show_results(nombre, equipo, res, df, model)
        else:
            st.info("Rellena las métricas y pulsa **Ejecutar modelo** para ver resultados.")

if __name__ == "__main__":
    main()
