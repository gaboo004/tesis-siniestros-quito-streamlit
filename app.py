"""Tablero de tesis para los siniestros de tránsito registrados en Quito."""
from base64 import b64encode
from html import escape
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from figures import (
    COLORES_SEVERIDAD,
    SEVERIDADES,
    conteo_local,
    grafico_barras,
    grafico_evolucion,
    grafico_severidad,
)

BASE = Path(__file__).resolve().parent
DATOS = BASE / "datos" / "siniestros_quito_2022_2024_limpio.csv"
MODELO = BASE / "modelos" / "random_forest_final_2022_2023.joblib"
VARIABLES = ["MES", "DIA", "HORA", "ZONA", "CLASE", "CAUSA"]
ANIOS = [2022, 2023, 2024]
MESES = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
]
DIAS = ["LUNES", "MARTES", "MIÉRCOLES", "JUEVES", "VIERNES", "SÁBADO", "DOMINGO"]
HISTORICO = "Análisis histórico"
ESTIMACION = "Estimación de severidad"
LOGGER = logging.getLogger(__name__)

PORTADA_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 550 290" fill="none">
        <defs>
          <pattern id="tesis-reticula" width="26" height="26" patternUnits="userSpaceOnUse">
            <path d="M26 0H0V26" stroke="#C8F7EE" stroke-opacity=".13"/>
          </pattern>
          <linearGradient id="tesis-linea" x1="104" y1="281" x2="522" y2="15"
                          gradientUnits="userSpaceOnUse">
            <stop stop-color="#76DDC1" stop-opacity=".08"/>
            <stop offset="1" stop-color="#8FEACE" stop-opacity=".85"/>
          </linearGradient>
        </defs>
        <path fill="url(#tesis-reticula)" d="M80 0H550V290H80z"/>
        <g stroke="url(#tesis-linea)">
          <path stroke-width="1.5" d="M15 277c90 0 144 0 198-39 73-53 70-145 146-193 40-25 95-33 202-32"/>
          <path stroke-width="2" d="M37 287c93 0 154-5 209-45 72-53 72-143 141-183 41-24 91-28 174-28"/>
          <path stroke-width="1.5" stroke-dasharray="6 8" d="M64 298c93 0 151-5 205-47 71-55 68-136 136-175 37-21 78-25 157-26"/>
          <path stroke-width="2" d="M91 308c89-1 151-8 202-49 70-56 66-128 131-164 35-19 72-24 138-24"/>
          <path stroke-width="1.5" d="M122 318c83-2 143-12 193-51 68-54 66-121 126-153 34-18 66-21 121-22"/>
          <path stroke-opacity=".45" d="M127 19c86 53 94 116 165 155 66 36 137 19 270 18"/>
          <path stroke-opacity=".35" d="M114 42c88 51 96 113 169 154 64 36 142 19 279 19"/>
        </g>
        <circle cx="328" cy="221" r="18" stroke="#93ECCE" stroke-opacity=".45"/>
        <circle cx="328" cy="221" r="5" fill="#A6F4D8"/>
        <circle cx="434" cy="53" r="4" fill="#9CE9D2"/>
        <circle cx="246" cy="242" r="4" fill="#9CE9D2" fill-opacity=".75"/>
      </svg>
"""
MARCA_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" fill="none">
              <path d="M5 25c7 0 8-5 11-10s5-7 11-7M5 20c4 0 5-3 8-8s6-8 14-8"
                    stroke="#9EE9D0" stroke-width="2" stroke-linecap="round"/>
              <path d="M5 29c9 0 12-6 15-11s4-6 7-6"
                    stroke="#9EE9D0" stroke-width="2" stroke-linecap="round" opacity=".6"/>
            </svg>
"""

st.set_page_config(
    page_title="Severidad vial en Quito · Tablero de tesis",
    page_icon=":material/traffic:",
    layout="wide",
    initial_sidebar_state="auto",
)


@st.cache_data(show_spinner=False)
def cargar_datos():
    datos = pd.read_csv(DATOS)
    columnas = {
        *VARIABLES, "AÑO", "SEVERIDAD", "DIA_NOMBRE",
        "ZONA_NOMBRE", "CLASE_NOMBRE", "CAUSA_NOMBRE",
    }
    if faltantes := columnas.difference(datos.columns):
        raise ValueError(f"Columnas ausentes en el CSV: {', '.join(sorted(faltantes))}")
    return datos


@st.cache_resource(show_spinner=False)
def cargar_modelo():
    import joblib

    return joblib.load(MODELO)


def mapa_etiquetas(datos, codigo, nombre):
    """Mantiene los códigos y etiquetas del CSV empleado por el formulario original."""
    pares = datos[[codigo, nombre]].drop_duplicates().sort_values(codigo)
    return dict(zip(pares[codigo].astype(int), pares[nombre]))


def porcentaje(valor, decimales=1):
    return f"{valor:.{decimales}f}".replace(".", ",") + " %"


def imagen_decorativa(svg, clase):
    contenido = b64encode(svg.strip().encode()).decode("ascii")
    return f'<img class="{clase}" src="data:image/svg+xml;base64,{contenido}" alt="" aria-hidden="true">'


def portada(seccion):
    es_historico = seccion == HISTORICO
    titulo = "Siniestros de tránsito<br>en Quito" if es_historico else "Estimación condicionada<br>de severidad"
    texto = (
        "Explora cómo se distribuyen los siniestros registrados en el Distrito "
        "Metropolitano de Quito entre 2022 y 2024."
        if es_historico else
        "Describe un siniestro registrado o hipotético para obtener su categoría "
        "de severidad estimada por el Random Forest de la tesis."
    )
    numero, etiqueta, ficha = (
        ("01", "ANÁLISIS HISTÓRICO", "2022–2024") if es_historico else
        ("02", "MODELO DE CLASIFICACIÓN", "Random Forest")
    )
    st.html(f"""
    <section class="tesis-hero" aria-labelledby="tesis-titulo">
      {imagen_decorativa(PORTADA_SVG, "tesis-trazado")}
      <div class="tesis-hero-copy">
        <div class="tesis-kicker"><span>{numero}</span> {etiqueta}</div>
        <h1 id="tesis-titulo">{titulo}</h1>
        <p>{texto}</p>
        <span class="tesis-hero-tag">{ficha}</span>
      </div>
    </section>
    """)


def encabezado(etiqueta, titulo, descripcion):
    st.html(f"""
    <div class="tesis-seccion">
      <p class="tesis-eyebrow">{etiqueta}</p>
      <h2>{titulo}</h2>
      <p class="tesis-subtexto">{descripcion}</p>
    </div>
    """)


def nota(titulo, cuerpo):
    st.html(f'<aside class="tesis-nota"><strong>{titulo}</strong><p>{cuerpo}</p></aside>')


def grafico(chart):
    st.altair_chart(chart, width="stretch", theme=None)


def distribucion(datos, columna, nombre):
    return datos[columna].value_counts().rename_axis(nombre).reset_index(name="Total")


def leyenda_severidad(conteos):
    filas = []
    for color, fila in zip(COLORES_SEVERIDAD, conteos.itertuples(index=False)):
        filas.append(f"""
        <li style="--nivel:{color}">
          <span class="tesis-punto" aria-hidden="true"></span>
          <span>{fila.Severidad.capitalize()}</span>
          <strong>{porcentaje(fila.Porcentaje)}</strong>
          <small>{conteo_local(fila.Total)} siniestros</small>
        </li>
        """)
    st.html(f'<ul class="tesis-leyenda" aria-label="Distribución por severidad">{"".join(filas)}</ul>')


def mostrar_historico(datos):
    with st.container(key="tesis-filtros", border=True):
        encabezado("EXPLORACIÓN", "Filtra los registros", "Los indicadores y gráficos responden a la selección de año y zona.")
        filtro_anio, filtro_zona = st.columns(2, gap="medium")
        anio = filtro_anio.selectbox("Año", ["Todos", *ANIOS], key="anio_historico")
        zona = filtro_zona.selectbox(
            "Zona", ["Todas", "URBANA", "RURAL"], format_func=str.capitalize, key="zona_historica",
        )

    filtrados = datos
    if anio != "Todos":
        filtrados = filtrados.loc[filtrados["AÑO"].eq(anio)]
    if zona != "Todas":
        filtrados = filtrados.loc[filtrados["ZONA_NOMBRE"].eq(zona)]

    total = len(filtrados)
    altas = int(filtrados["SEVERIDAD"].eq("ALTA").sum())
    proporcion = porcentaje(100 * altas / total, 2) if total else "—"
    registro, severidad, participacion = st.columns(3)
    registro.metric("Siniestros registrados", conteo_local(total), border=True)
    severidad.metric("Severidad alta", conteo_local(altas), border=True)
    participacion.metric("Proporción de severidad alta", proporcion, border=True)
    if not total:
        st.info("No hay registros para la combinación seleccionada.")
        return

    conteos = (
        filtrados["SEVERIDAD"].value_counts()
        .reindex(SEVERIDADES, fill_value=0)
        .rename_axis("Severidad")
        .reset_index(name="Total")
    )
    conteos["Porcentaje"] = 100 * conteos["Total"] / total

    with st.container(key="tesis-panorama"):
        tendencia, niveles = st.columns([1.65, 1], gap="medium")
        with tendencia.container(key="tesis-evolucion", border=True):
            encabezado("01 / TIEMPO", "Evolución mensual", "Número de siniestros registrados por mes.")
            grafico(grafico_evolucion(filtrados, anio))
        with niveles.container(key="tesis-niveles", border=True):
            encabezado("02 / SEVERIDAD", "Distribución de severidad", "Participación de cada nivel en la selección.")
            grafico(grafico_severidad(conteos))
            leyenda_severidad(conteos)

    with st.container(key="tesis-causas", border=True):
        encabezado("03 / CAUSA REGISTRADA", "Principales causas", "Las ocho causas más frecuentes entre los registros seleccionados.")
        grafico(
            grafico_barras(distribucion(filtrados, "CAUSA_NOMBRE", "Causa").head(8), "Causa"),
        )

    clases, dias = st.columns(2, gap="medium")
    with clases.container(key="tesis-clases", border=True):
        encabezado("04 / CLASE", "Siniestros por clase", "Frecuencia por tipo de siniestro registrado.")
        grafico(
            grafico_barras(distribucion(filtrados, "CLASE_NOMBRE", "Clase"), "Clase"),
        )
    with dias.container(key="tesis-dias", border=True):
        encabezado("05 / DÍA DE LA SEMANA", "Siniestros por día", "Frecuencia agregada de lunes a domingo.")
        grafico(
            grafico_barras(distribucion(filtrados, "DIA_NOMBRE", "Día"), "Día", orden=DIAS, color="#087A70"),
        )
    nota(
        "Alcance de los registros",
        "Los conteos describen los siniestros del archivo histórico. La cobertura geográfica "
        "distingue únicamente entre zona urbana y rural.",
    )


def mostrar_resultado(nivel, contexto):
    nivel_html = escape(str(nivel))
    valores = "".join(
        f"<li><span>{etiqueta}</span><strong>{escape(str(valor))}</strong></li>"
        for etiqueta, valor in contexto
    )
    st.html(f"""
    <section class="tesis-resultado nivel-{nivel_html.lower()}" aria-live="polite"
             aria-labelledby="tesis-resultado-titulo">
      <p class="tesis-eyebrow">RESULTADO DEL MODELO</p>
      <h2 id="tesis-resultado-titulo">Severidad estimada</h2>
      <p class="tesis-nivel">{nivel_html}</p>
      <p>Clasificación para el escenario ingresado.</p>
      <ul aria-label="Valores utilizados para la estimación">{valores}</ul>
    </section>
    """)


def mostrar_estimacion(datos):
    dias = mapa_etiquetas(datos, "DIA", "DIA_NOMBRE")
    zonas = mapa_etiquetas(datos, "ZONA", "ZONA_NOMBRE")
    clases = mapa_etiquetas(datos, "CLASE", "CLASE_NOMBRE")
    causas = mapa_etiquetas(datos, "CAUSA", "CAUSA_NOMBRE")

    with st.container(key="tesis-estimacion"):
        entradas, salida = st.columns([1.3, 1], gap="medium")
        with entradas:
            with st.form("formulario_siniestro", border=True):
                encabezado("01 / ENTRADAS", "Configura el escenario", "Completa los seis campos para ejecutar la estimación.")
                campo_mes, campo_dia, campo_hora = st.columns(3, vertical_alignment="bottom")
                mes = campo_mes.selectbox("Mes", range(1, 13), format_func=lambda valor: MESES[valor - 1])
                dia = campo_dia.selectbox("Día de la semana", list(dias), format_func=lambda valor: dias[valor].capitalize())
                hora = campo_hora.selectbox("Hora", range(24), format_func=lambda valor: f"{valor:02d}:00")
                campo_zona, campo_clase = st.columns(2)
                zona = campo_zona.selectbox("Zona", list(zonas), format_func=lambda valor: zonas[valor].capitalize())
                clase = campo_clase.selectbox("Clase de siniestro", list(clases), format_func=lambda valor: clases[valor].capitalize())
                causa = st.selectbox("Causa registrada", list(causas), format_func=lambda valor: causas[valor].capitalize())
                enviar = st.form_submit_button("Estimar severidad", type="primary", width="stretch")
        with salida:
            if enviar:
                st.session_state.pop("tesis_resultado", None)
                entrada = pd.DataFrame(
                    [[mes, dia, hora, zona, clase, causa]], columns=VARIABLES,
                ).astype(datos[VARIABLES].dtypes.to_dict())
                try:
                    with st.spinner("Ejecutando el modelo…"):
                        nivel = cargar_modelo().predict(entrada)[0]
                except Exception:
                    LOGGER.exception("Error durante la estimación de severidad")
                    st.error("La estimación no está disponible. Revisa el registro del servidor y el archivo del modelo.")
                else:
                    contexto = (
                        ("Mes", MESES[mes - 1]), ("Día", dias[dia].capitalize()),
                        ("Hora", f"{hora:02d}:00"), ("Zona", zonas[zona].capitalize()),
                        ("Clase", clases[clase].capitalize()), ("Causa", causas[causa].capitalize()),
                    )
                    st.session_state.tesis_resultado = (nivel, contexto)

            if resultado := st.session_state.get("tesis_resultado"):
                mostrar_resultado(*resultado)
            elif not enviar:
                st.html("""
                <section class="tesis-resultado tesis-espera" aria-labelledby="tesis-espera-titulo">
                  <p class="tesis-eyebrow">02 / RESULTADO</p>
                  <h2 id="tesis-espera-titulo">Tu estimación</h2>
                  <p>Al enviar el formulario, aquí aparecerá la categoría estimada y un resumen de los valores utilizados.</p>
                  <div class="tesis-niveles" aria-label="Categorías posibles">
                    <span>Baja</span><span>Media</span><span>Alta</span>
                  </div>
                </section>
                """)

    nota(
        "Cómo interpretar el resultado",
        "La categoría es una estimación de severidad condicionada a los seis valores ingresados. "
        "Su alcance es académico y excluye tanto el pronóstico de ocurrencia de siniestros como "
        "la toma de decisiones operativas.",
    )
    with st.expander("Limitaciones del modelo", expanded=True):
        st.markdown(
            "- El desempeño difiere entre categorías. Interpreta con especial cautela la clase **ALTA**.\n"
            "- Las seis variables son una descripción parcial del siniestro. La clasificación puede "
            "ser incorrecta y puede perder validez en otros lugares o periodos.\n"
            "- El modelo entregado corresponde al entrenamiento de 2022–2023. "
            "Los datos de 2024 se reservan para la evaluación temporal.\n"
            "- Consulta la evaluación de la tesis para conocer las métricas por clase y la matriz de confusión."
        )


def main():
    st.html(BASE / "estilos.css")
    with st.sidebar:
        st.html(f"""
        <div class="tesis-marca">
          <div class="tesis-isotipo" aria-hidden="true">
            {imagen_decorativa(MARCA_SVG, "tesis-marca-svg")}
          </div>
          <p>Severidad vial <span>Quito · 2022–2024</span></p>
        </div>
        """)
        seccion = st.radio("Navegación", [HISTORICO, ESTIMACION], key="modulo")
        st.html("""
        <div class="tesis-sidebar-nota">
          <p class="tesis-eyebrow">TRABAJO DE TITULACIÓN</p>
          <p>Distrito Metropolitano<br>de Quito · Ecuador</p>
        </div>
        """)

    portada(seccion)
    try:
        datos = cargar_datos()
    except Exception:
        LOGGER.exception("Error al cargar los datos históricos")
        st.error("No se pudieron cargar los datos. Verifica el CSV en la carpeta datos/ y revisa el registro del servidor.")
        st.stop()

    if seccion == HISTORICO:
        st.session_state.pop("tesis_resultado", None)
        mostrar_historico(datos)
    else:
        mostrar_estimacion(datos)

    st.html("""
    <footer class="tesis-footer">
      <strong>Severidad vial en Quito</strong>
      <span>Tablero de investigación · Trabajo de titulación</span>
    </footer>
    """)


if __name__ == "__main__":
    main()
