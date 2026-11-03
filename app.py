"""Aplicación académica: siniestros de tránsito en Quito (2022–2024).

Versión de presentación y visualización, 3 de noviembre de 2026.
Conserva el pipeline Random Forest y sus seis variables de entrada.
"""
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Siniestros de tránsito en Quito | Tesis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ajustes de presentación. Sin alterar los resultados ni el modelo.
st.markdown(
    """
    <style>
    .block-container {max-width: 1400px; padding-top: 2rem; padding-bottom: 2.5rem;}
    h1 {letter-spacing: -0.025em;}
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, .24);
        border-radius: 13px;
        padding: 16px 18px;
        background: rgba(128, 128, 128, .06);
    }
    div[data-testid="stMetricValue"] {font-weight: 700;}
    </style>
    """,
    unsafe_allow_html=True,
)

BASE = Path(__file__).resolve().parent
DATOS = BASE / "datos" / "siniestros_quito_2022_2024_limpio.csv"
MODELO = BASE / "modelos" / "random_forest_final_2022_2023.joblib"
VARIABLES = ["MES", "DIA", "HORA", "ZONA", "CLASE", "CAUSA"]
SEVERIDADES = ["BAJA", "MEDIA", "ALTA"]
COLORES_SEVERIDAD = ["#49A894", "#E7AD52", "#DB6B65"]
DIAS_SEMANA = [
    "LUNES", "MARTES", "MIÉRCOLES", "JUEVES",
    "VIERNES", "SÁBADO", "DOMINGO",
]


@st.cache_data
def cargar_datos():
    return pd.read_csv(DATOS)


@st.cache_resource
def cargar_modelo():
    return joblib.load(MODELO)


def mapa_etiquetas(datos, codigo, nombre):
    """Leer directamente los códigos y las etiquetas del CSV limpio."""
    pares = datos[[codigo, nombre]].drop_duplicates().sort_values(codigo)
    return dict(zip(pares[codigo].astype(int), pares[nombre]))


def grafico_barras(tabla, categoria, *, alto, etiqueta_eje, limite_etiqueta=275, orden="-x"):
    return (
        alt.Chart(tabla)
        .mark_bar(size=22, cornerRadiusEnd=4, color="#5CA6D8")
        .encode(
            x=alt.X("Total:Q", title="Número de siniestros", axis=alt.Axis(format=",d")),
            y=alt.Y(
                f"{categoria}:N", title=None, sort=orden,
                axis=alt.Axis(labelLimit=limite_etiqueta, labelFontSize=12),
            ),
            tooltip=[
                alt.Tooltip(f"{categoria}:N", title=etiqueta_eje),
                alt.Tooltip("Total:Q", title="Siniestros", format=",d"),
            ],
        )
        .properties(height=alto)
    )


df = cargar_datos()

st.sidebar.title("Siniestros Quito")
seccion = st.sidebar.radio(
    "Navegación", ["Análisis de siniestros", "Predicción de severidad"]
)
st.sidebar.divider()
st.sidebar.caption("Trabajo de titulación · Quito · 2022–2024")
st.sidebar.caption("Análisis descriptivo y estimación condicionada; no predice ocurrencia de accidentes.")

if seccion == "Análisis de siniestros":
    st.title("Análisis histórico de siniestros de tránsito")
    st.caption("Distrito Metropolitano de Quito · Registros históricos 2022–2024")
    st.write("Selecciona los filtros para explorar la distribución de los siniestros registrados.")

    filtro_a, filtro_b = st.columns(2)
    with filtro_a:
        anio = st.selectbox("Año", ["Todos", 2022, 2023, 2024])
    with filtro_b:
        zona = st.selectbox("Zona", ["Todas", "URBANA", "RURAL"])

    filtrados = df.copy()
    if anio != "Todos":
        filtrados = filtrados.loc[filtrados["AÑO"] == anio]
    if zona != "Todas":
        filtrados = filtrados.loc[filtrados["ZONA_NOMBRE"] == zona]

    total = len(filtrados)
    altas = int((filtrados["SEVERIDAD"] == "ALTA").sum())
    pct_alta = (altas * 100 / total) if total else 0
    m1, m2, m3 = st.columns(3)
    m1.metric("Siniestros registrados", f"{total:,}")
    m2.metric("Severidad ALTA", f"{altas:,}")
    m3.metric("Porcentaje ALTA", f"{pct_alta:.2f}%")

    if not total:
        st.info("No hay registros para los filtros seleccionados.")
    else:
        conteos = (
            filtrados["SEVERIDAD"]
            .value_counts()
            .reindex(SEVERIDADES, fill_value=0)
            .rename_axis("Severidad")
            .reset_index(name="Total")
        )
        conteos["Porcentaje"] = 100 * conteos["Total"] / total
        clases = filtrados["CLASE_NOMBRE"].value_counts().rename_axis("Clase").reset_index(name="Total")
        causas = (
            filtrados["CAUSA_NOMBRE"].value_counts().head(8)
            .rename_axis("Causa").reset_index(name="Total")
        )
        dias = (
            filtrados["DIA_NOMBRE"].value_counts()
            .reindex(DIAS_SEMANA, fill_value=0)
            .rename_axis("Día").reset_index(name="Total")
        )

        st.divider()
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.subheader("Distribución de severidad")
            donut = (
                alt.Chart(conteos)
                .mark_arc(innerRadius=76, outerRadius=138)
                .encode(
                    theta=alt.Theta("Total:Q", stack=True),
                    color=alt.Color(
                        "Severidad:N",
                        scale=alt.Scale(domain=SEVERIDADES, range=COLORES_SEVERIDAD),
                        sort=SEVERIDADES,
                        legend=alt.Legend(title="Severidad", orient="bottom"),
                    ),
                    tooltip=[
                        alt.Tooltip("Severidad:N"),
                        alt.Tooltip("Total:Q", title="Siniestros", format=",d"),
                        alt.Tooltip("Porcentaje:Q", title="Porcentaje", format=".1f"),
                    ],
                )
                .properties(height=315)
            )
            st.altair_chart(donut, width="stretch")
            resumen = " · ".join(
                f"{row.Severidad}: {row.Total:,} ({row.Porcentaje:.1f}%)"
                for row in conteos.itertuples(index=False)
            )
            st.caption(resumen)
        with c2:
            st.subheader("Siniestros por clase")
            st.altair_chart(
                grafico_barras(clases, "Clase", alto=315, etiqueta_eje="Clase", limite_etiqueta=240),
                width="stretch",
            )

        st.divider()
        c3, c4 = st.columns(2, gap="large")
        with c3:
            st.subheader("Principales causas registradas")
            st.altair_chart(
                grafico_barras(causas, "Causa", alto=350, etiqueta_eje="Causa", limite_etiqueta=380),
                width="stretch",
            )
            st.caption("Se muestran las ocho causas más frecuentes en el filtro seleccionado. Pasa el cursor por cada barra para leer el nombre completo.")
        with c4:
            st.subheader("Siniestros por día de la semana")
            st.altair_chart(
                grafico_barras(
                    dias, "Día", alto=350, etiqueta_eje="Día",
                    limite_etiqueta=170, orden=DIAS_SEMANA,
                ),
                width="stretch",
            )

        st.info(
            "Estos gráficos resumen los registros históricos del dataset. "
            "No representan la ubicación exacta de los siniestros ni predicen dónde ocurrirán. "
            "Los datos disponibles distinguen zona urbana y rural, no sectores o coordenadas."
        )

else:
    st.title("Estimación condicionada de severidad")
    st.caption("Modelo seleccionado: Random Forest · Categorías: BAJA, MEDIA y ALTA")
    st.write(
        "Ingresa características de un siniestro ya registrado o de un escenario hipotético. "
        "El modelo estima una categoría de severidad **condicionada a esas características**; "
        "no estima la probabilidad de que ocurra un accidente."
    )

    dias = mapa_etiquetas(df, "DIA", "DIA_NOMBRE")
    zonas = mapa_etiquetas(df, "ZONA", "ZONA_NOMBRE")
    clases = mapa_etiquetas(df, "CLASE", "CLASE_NOMBRE")
    causas = mapa_etiquetas(df, "CAUSA", "CAUSA_NOMBRE")

    with st.form("formulario_siniestro"):
        izquierda, derecha = st.columns(2, gap="large")
        with izquierda:
            mes = st.selectbox("Mes", range(1, 13))
            dia = st.selectbox("Día de la semana", list(dias), format_func=lambda x: dias[x].capitalize())
            hora = st.selectbox("Hora (0–23)", range(24))
        with derecha:
            zona = st.selectbox("Zona", list(zonas), format_func=lambda x: zonas[x].capitalize())
            clase = st.selectbox("Clase de siniestro", list(clases), format_func=lambda x: clases[x].capitalize())
            causa = st.selectbox("Causa registrada", list(causas), format_func=lambda x: causas[x].capitalize())
        enviar = st.form_submit_button("Estimar severidad", type="primary")

    if enviar:
        entrada = pd.DataFrame([{
            "MES": mes, "DIA": dia, "HORA": hora,
            "ZONA": zona, "CLASE": clase, "CAUSA": causa,
        }], columns=VARIABLES)
        for variable in VARIABLES:
            entrada[variable] = entrada[variable].astype(df[variable].dtype)
        prediccion = cargar_modelo().predict(entrada)[0]
        st.success(f"Severidad estimada: {prediccion}")
        st.caption("Resultado de clasificación del modelo para las características introducidas; no es un pronóstico de ocurrencia.")

    st.warning(
        "**Limitaciones del modelo:** su desempeño es desigual entre categorías, "
        "especialmente para severidad ALTA. El resultado puede ser incorrecto; "
        "no debe utilizarse para decisiones operativas ni como sistema autónomo "
        "de seguridad vial. Las métricas completas están documentadas en los notebooks de evaluación."
    )
