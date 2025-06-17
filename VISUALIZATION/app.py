import streamlit as st
import logging
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from typing import Dict
from dotenv import load_dotenv
import os

load_dotenv()

# ---------------------------
# Configuración de logging
# ---------------------------
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    filename=LOG_DIR / "app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)
logger.info("Streamlit module imported")

# ---------------------------
# Carpeta de modelos y utilidades
# ---------------------------
MODEL_DIR = Path(os.getenv("MODEL_DIR"))

@st.cache_data
def list_model_paths(pattern: str = "*.pkl") -> list[Path]:
    """Devuelve una lista ordenada de rutas de modelos (.pkl) en MODEL_DIR."""
    
    paths = sorted(MODEL_DIR.glob(pattern))
    logger.info("Model list refreshed: %s", [p.name for p in paths])
    return paths

@st.cache_resource
def load_model(model_path: Path):
    """Carga un modelo pickle y lo cachea por ruta."""
    with model_path.open("rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded from %s", model_path)
    return model

# ---------------------------
# Predicción a partir de métricas
# ---------------------------

def predict(model, metrics: Dict[str, float]) -> float:
    """Genera una predicción (probabilidad) a partir de un diccionario de métricas."""
    X = np.array([list(metrics.values())])
    if hasattr(model, "predict_proba"):
        pred = model.predict_proba(X)[0, 1]
    else:
        pred = model.predict(X)[0]
    logger.info("Prediction made with metrics %s => %s", metrics, pred)
    return float(pred)

# ---------------------------
# Función principal de la app
# ---------------------------

def main():
    st.set_page_config(page_title="Demo de Predicción", layout="centered")
    st.title("Demo de Predicción con Múltiples Modelos")

    # Sidebar: selección de modelo y entrada de métricas
    with st.sidebar:
        st.header("Configuración del modelo")

        model_paths = list_model_paths()
        if not model_paths:
            st.error("No se encontraron modelos en la carpeta 'models/'. Añade tus ficheros .pkl y recarga la página.")
            st.stop()

        model_names = [p.stem for p in model_paths]
        selected_name = st.selectbox("Selecciona el modelo", model_names)
        selected_path = model_paths[model_names.index(selected_name)]
        st.markdown(f"**Archivo:** `{selected_path.name}`")

        st.header("Introduce las métricas")
        with st.form("input_form", clear_on_submit=False):
            metric1 = st.number_input("Métrica 1", value=0.0, format="%.2f")
            metric2 = st.number_input("Métrica 2", value=0.0, format="%.2f")
            metric3 = st.number_input("Métrica 3", value=0.0, format="%.2f")
            submitted = st.form_submit_button("Predecir")

    # Carga (cacheada) del modelo elegido
    model = load_model(selected_path)

    # Estado para conservar la última predicción entre reruns
    if "prediction" not in st.session_state:
        st.session_state["prediction"] = None

    if submitted:
        metrics_dict = {
            "metric1": metric1,
            "metric2": metric2,
            "metric3": metric3,
        }
        st.session_state["prediction"] = predict(model, metrics_dict)
        st.session_state["last_metrics"] = metrics_dict
        st.session_state["last_model"] = selected_name

    if st.session_state["prediction"] is not None:
        st.subheader(f"Resultado de la predicción • Modelo: {st.session_state.get('last_model')}")
        st.metric("Probabilidad de clase positiva", f"{st.session_state['prediction']:.2%}")

        chart_data = pd.DataFrame({
            "Clase": ["Negativa", "Positiva"],
            "Probabilidad": [1 - st.session_state["prediction"], st.session_state["prediction"]],
        }).set_index("Clase")
        st.bar_chart(chart_data)

        st.write("Métricas utilizadas para esta predicción:")
        st.json(st.session_state.get("last_metrics", {}))

        if LOG_DIR.joinpath("app.log").exists():
            with open(LOG_DIR / "app.log", "rb") as log_file:
                st.download_button("Descargar log", log_file, file_name="app.log")

    st.caption("Versión con selección dinámica de modelos. Coloca tus ficheros .pkl en la carpeta 'models/'.")

# ---------------------------
# Entry point
# ---------------------------

if __name__ == "__main__":
    main()