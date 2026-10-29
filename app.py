import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Siniestros de Quito",
    page_icon="🚗",
    layout="wide"
)

BASE = Path(__file__).resolve().parent

DATOS = BASE / "datos" / "siniestros_quito_2022_2024_limpio.csv"
MODELO = BASE / "modelos" / "random_forest_final_2022_2023.joblib"

VARIABLES = ["MES", "DIA", "HORA", "ZONA", "CLASE", "CAUSA"]

@st.cache_data
def cargar_datos():
    return pd.read_csv(DATOS)

@st.cache_resource
def cargar_modelo():
    return joblib.load(MODELO)

df = cargar_datos()

st.sidebar.title("Siniestros Quito")

seccion = st.sidebar.radio(
    "Navegación",
    ["Análisis de siniestros", "Predicción de severidad"]
)

st.sidebar.caption(
    "Trabajo de titulación · Datos 2022–2024"
)

if seccion == "Análisis de siniestros":

    st.title("Análisis de siniestros de tránsito")
    st.write("Distrito Metropolitano de Quito · 2022–2024")

    anio = st.selectbox(
        "Selecciona el año",
        ["Todos", 2022, 2023, 2024]
    )

    zona = st.selectbox(
        "Selecciona la zona",
        ["Todas", "URBANA", "RURAL"]
    )

    filtrados = df.copy()

    if anio != "Todos":
        filtrados = filtrados[filtrados["AÑO"] == anio]

    if zona != "Todas":
        filtrados = filtrados[
            filtrados["ZONA_NOMBRE"] == zona
        ]

    total = len(filtrados)
    altas = (filtrados["SEVERIDAD"] == "ALTA").sum()
    porcentaje = altas / total * 100 if total else 0

    c1, c2, c3 = st.columns(3)

    c1.metric("Siniestros", f"{total:,}")
    c2.metric("Severidad ALTA", int(altas))
    c3.metric("Porcentaje ALTA", f"{porcentaje:.2f}%")

    st.subheader("Distribución de severidad")

    conteos = (
        filtrados["SEVERIDAD"]
        .value_counts()
        .reindex(["BAJA", "MEDIA", "ALTA"], fill_value=0)
    )

    st.bar_chart(conteos)

    st.caption(
        "Resultados descriptivos de los registros históricos."
    )

else:

    st.title("Predicción de severidad")

    st.write(
        "Ingresa las características registradas "
        "de un siniestro para estimar su severidad."
    )

    clases = {
        1: "Atropellos",
        2: "Caída de pasajeros",
        3: "Choques",
        4: "Estrellamientos",
        5: "Rozamientos",
        6: "Volcamientos",
        7: "Pérdida de pista",
        8: "Otros"
    }

    causas = {
        1: "Embriaguez o droga",
        2: "MAL REBASAMIENTO INVADIR CARRIL",
        3: "EXCESO VELOCIDAD",
        4: "IMPERICIA E IMPRUDENCIA DEL CONDUCTOR",
        5: "Imprudencia del peatón",
        6: "Daños mecánicos",
        7: "NO RESPETA LAS SEÑALES DE TRÁNSITO",
        8: "Factores climáticos",
        9: "Mal estado de la vía",
        10: "Otras causas"
    }

    dias = [
        "Lunes", "Martes", "Miércoles", "Jueves",
        "Viernes", "Sábado", "Domingo"
    ]

    with st.form("formulario_siniestro"):

        c1, c2 = st.columns(2)

        with c1:
            mes = st.selectbox("Mes", range(1, 13))
            dia = st.selectbox(
                "Día", range(1, 8),
                format_func=lambda x: dias[x - 1]
            )
            hora = st.selectbox("Hora", range(24))

        with c2:
            zona = st.selectbox(
                "Zona", [1, 2],
                format_func=lambda x:
                "Urbana" if x == 1 else "Rural"
            )
            clase = st.selectbox(
                "Clase de siniestro", list(clases),
                format_func=lambda x: clases[x]
            )
            causa = st.selectbox(
                "Causa", list(causas),
                format_func=lambda x: causas[x]
            )

        enviar = st.form_submit_button(
            "Estimar severidad"
        )

    if enviar:

        entrada = pd.DataFrame([{
            "MES": mes,
            "DIA": dia,
            "HORA": hora,
            "ZONA": zona,
            "CLASE": clase,
            "CAUSA": causa
        }], columns=VARIABLES)

        # Mantener los tipos del dataset utilizado
        for variable in VARIABLES:
            entrada[variable] = entrada[variable].astype(
                df[variable].dtype
            )

        prediccion = cargar_modelo().predict(entrada)[0]

        st.success(f"Severidad estimada: {prediccion}")

        st.caption(
            "Estimación del modelo basada en características "
            "registradas; no equivale a una predicción "
            "de la ocurrencia de accidentes."
        )

    st.info(
        "El modelo tiene limitaciones de precisión, "
        "especialmente para severidad ALTA. "
        "No debe utilizarse como sistema autónomo "
        "para tomar decisiones de seguridad vial."
    )
