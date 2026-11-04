import streamlit as st
import pandas as pd
import joblib
from html import escape

from figures import barras, donut
from pathlib import Path

st.set_page_config(
    page_title="Siniestros de Quito",
    page_icon="◈",
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

st.html(f"<style>{(BASE / 'estilos.css').read_text(encoding='utf-8')}</style>")


def numero(valor):
    return f"{valor:,}".replace(",", ".")


def decimal(valor):
    return f"{valor:.2f}".replace(".", ",")


def encabezado(numero_modulo, titulo, descripcion):
    st.html(
        f'<p class="eyebrow">QUITO · INVESTIGACIÓN ACADÉMICA · {numero_modulo}</p>'
        f'<h1 class="page-heading">{titulo}</h1>'
        f'<p class="page-intro">{descripcion}</p>'
    )


AVISO = (
    "La estimación de severidad está condicionada a las características de un siniestro "
    "ya registrado o hipotético. No predice la ocurrencia de accidentes ni constituye consejo operativo."
)


with st.sidebar:
    st.html('<div class="brand"><strong>Siniestros Quito</strong><span>Panel de investigación</span></div>')

seccion = st.sidebar.radio(
    "Navegación",
    ["Análisis de siniestros", "Predicción de severidad"]
)

st.sidebar.divider()
st.sidebar.markdown("**Distrito Metropolitano de Quito**")
st.sidebar.caption("Registros históricos · 2022–2024")
st.sidebar.caption("Trabajo de titulación · Aprendizaje automático")
with st.sidebar.expander("Acerca del estudio"):
    st.write(
        "Análisis y predicción de la severidad de los siniestros de tránsito "
        "en el Distrito Metropolitano de Quito mediante aprendizaje automático."
    )
    st.caption("Modelo Random Forest entrenado con datos de 2022–2023.")

if seccion == "Análisis de siniestros":

    encabezado(
        "01", "Análisis de siniestros",
        "Una lectura descriptiva de los registros de tránsito del Distrito Metropolitano "
        "de Quito. Explora su severidad, clase, causa y distribución semanal."
    )

    with st.container(border=True):
        st.markdown("**Delimita el análisis**")
        filtro_anio, filtro_zona = st.columns(2)
        anio = filtro_anio.selectbox("Selecciona el año", ["Todos", 2022, 2023, 2024])
        zona = filtro_zona.selectbox("Selecciona la zona", ["Todas", "URBANA", "RURAL"])

    filtrados = df.copy()

    if anio != "Todos":
        filtrados = filtrados[filtrados["AÑO"] == anio]

    if zona != "Todas":
        filtrados = filtrados[filtrados["ZONA_NOMBRE"] == zona]

    total = len(filtrados)
    altas = (filtrados["SEVERIDAD"] == "ALTA").sum()
    porcentaje = altas / total * 100 if total else 0

    periodo = "2022–2024" if anio == "Todos" else str(anio)
    zona_visible = "Todas las zonas" if zona == "Todas" else f"Zona {zona.lower()}"
    st.caption(f"VISTA ACTUAL · {periodo} · {zona_visible}")
    st.html(
        '<div class="kpi-grid">'
        f'<div class="kpi"><div class="kpi-label">Siniestros registrados</div>'
        f'<div class="kpi-value">{numero(total)}</div><div class="kpi-detail">Registros en la selección</div></div>'
        f'<div class="kpi kpi-high"><div class="kpi-label">Severidad ALTA</div>'
        f'<div class="kpi-value">{numero(int(altas))}</div><div class="kpi-detail">Siniestros de severidad alta</div></div>'
        f'<div class="kpi"><div class="kpi-label">Proporción de ALTA</div>'
        f'<div class="kpi-value">{decimal(porcentaje)} %</div><div class="kpi-detail">Del total seleccionado</div></div>'
        '</div>'
    )

    # =========================
    # PREPARACIÓN DE TABLAS
    # =========================

    conteos = (
        filtrados["SEVERIDAD"]
        .value_counts()
        .reindex(["BAJA", "MEDIA", "ALTA"], fill_value=0)
        .reset_index()
    )
    conteos.columns = ["Severidad", "Total"]

    clases_df = (
        filtrados["CLASE_NOMBRE"]
        .value_counts()
        .reset_index()
    )
    clases_df.columns = ["Clase", "Total"]

    causas_df = (
        filtrados["CAUSA_NOMBRE"]
        .value_counts()
        .head(8)
        .reset_index()
    )
    causas_df.columns = ["Causa", "Total"]

    orden_dias = [
        "LUNES", "MARTES", "MIÉRCOLES",
        "JUEVES", "VIERNES", "SÁBADO", "DOMINGO"
    ]

    dias_df = (
        filtrados["DIA_NOMBRE"]
        .value_counts()
        .reindex(orden_dias, fill_value=0)
        .reset_index()
    )
    dias_df.columns = ["Día", "Total"]

    # Porcentajes descriptivos calculados sobre la selección actual.
    conteos["Porcentaje"] = conteos["Total"] / total * 100 if total else 0
    conteos["Orden"] = range(len(conteos))

    col_g1, col_g2 = st.columns(2, gap="large")
    with col_g1, st.container(border=True):
        st.subheader("Distribución de severidad")
        st.caption("Participación de cada categoría en la selección.")
        st.altair_chart(donut(conteos), width="stretch", theme=None)
        filas = []
        for fila in conteos.itertuples(index=False):
            filas.append(
                f'<div class="severity-row"><span class="severity-label">'
                f'<i class="dot dot-{fila.Severidad.lower()}" aria-hidden="true"></i>{fila.Severidad}</span>'
                f'<span>{numero(fila.Total)}</span><strong>{decimal(fila.Porcentaje)} %</strong></div>'
            )
        st.html('<div class="severity-list">' + ''.join(filas) + '</div>')

    with col_g2, st.container(border=True):
        st.subheader("Siniestros por clase")
        st.caption("Clases registradas, ordenadas por frecuencia.")
        st.altair_chart(barras(clases_df, "Clase", alto=390), width="stretch", theme=None)

    col_g3, col_g4 = st.columns(2, gap="large")
    with col_g3, st.container(border=True):
        st.subheader("Principales causas")
        st.caption("Las ocho causas más frecuentes de la selección.")
        st.altair_chart(barras(causas_df, "Causa", alto=460), width="stretch", theme=None)

    with col_g4, st.container(border=True):
        st.subheader("Distribución semanal")
        st.caption("Siniestros por día, de lunes a domingo.")
        st.altair_chart(barras(dias_df, "Día", orden=orden_dias, alto=460), width="stretch", theme=None)

    st.html(f'<div class="scope-note">{AVISO}</div>')
    st.caption("Resultados descriptivos de los registros históricos. Los filtros se aplican a todos los indicadores y gráficos.")

else:

    encabezado(
        "02", "Predicción de severidad",
        "Ingresa las características de un siniestro registrado o hipotético "
        "para obtener la categoría de severidad estimada por el modelo Random Forest."
    )
    st.html(f'<div class="scope-note">{AVISO}</div>')

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

    meses = [
        "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
        "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
    ]

    with st.form("formulario_siniestro"):
        st.subheader("Características del siniestro")
        st.caption("Completa los seis campos y selecciona Estimar severidad.")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**01 · Momento del siniestro**")
            mes = st.selectbox("Mes", range(1, 13), format_func=lambda x: meses[x - 1])
            dia = st.selectbox(
                "Día", range(1, 8),
                format_func=lambda x: dias[x - 1]
            )
            hora = st.selectbox("Hora", range(24), format_func=lambda x: f"{x:02d}:00")

        with c2:
            st.markdown("**02 · Zona y características**")
            zona = st.selectbox(
                "Zona", [1, 2],
                format_func=lambda x: "Urbana" if x == 1 else "Rural"
            )
            clase = st.selectbox(
                "Clase de siniestro", list(clases),
                format_func=lambda x: clases[x]
            )
            causa = st.selectbox(
                "Causa", list(causas),
                format_func=lambda x: causas[x]
            )

        enviar = st.form_submit_button("Estimar severidad", type="primary", width="stretch")

    if enviar:

        entrada = pd.DataFrame([{
            "MES": mes,
            "DIA": dia,
            "HORA": hora,
            "ZONA": zona,
            "CLASE": clase,
            "CAUSA": causa
        }], columns=VARIABLES)

        for variable in VARIABLES:
            entrada[variable] = entrada[variable].astype(df[variable].dtype)

        prediccion = cargar_modelo().predict(entrada)[0]

        categoria_resultado = str(prediccion)
        estilo_resultado = categoria_resultado.lower() if categoria_resultado in ["BAJA", "MEDIA", "ALTA"] else ""
        st.html(
            f'<div class="result {estilo_resultado}" role="status">'
            '<p class="eyebrow">RESULTADO DEL MODELO</p>'
            f'<h2>Severidad estimada · {escape(categoria_resultado)}</h2>'
            '<p>Estimación para las características enviadas en el formulario.</p></div>'
        )
        st.caption(AVISO)

    st.warning(
        "El modelo presenta un desempeño limitado, especialmente para severidad ALTA. "
        "La estimación debe interpretarse dentro del alcance académico del estudio "
        "y no debe utilizarse como sistema autónomo para tomar decisiones de seguridad vial."
    )

st.html(
    '<p class="footer-note">Trabajo de titulación · Distrito Metropolitano de Quito · '
    'Datos 2022–2024</p>'
)
