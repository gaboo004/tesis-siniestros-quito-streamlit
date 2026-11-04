"""Gráficos descriptivos de los registros históricos."""

from textwrap import fill

import altair as alt
import pandas as pd


SEVERIDADES = ["BAJA", "MEDIA", "ALTA"]
COLORES_SEVERIDAD = ["#087A70", "#B56900", "#AF3655"]
TINTA = "#102D40"
TEXTO_SECUNDARIO = "#496477"
CUADRICULA = "#E3EDF2"
AZUL = "#174F79"
VERDE = "#087A70"
MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


def conteo_local(valor):
    return f"{int(valor):,}".replace(",", ".")


def _estilo(grafico):
    return (
        grafico.configure(background="transparent", font="Inter, sans-serif")
        .configure_view(stroke=None)
        .configure_axis(
            domain=False,
            gridColor=CUADRICULA,
            labelColor=TEXTO_SECUNDARIO,
            labelFontSize=11,
            labelPadding=8,
            tickSize=0,
        )
    )


def grafico_evolucion(filtrados, anio):
    """Contar registros en cada mes de los años incluidos por el filtro."""
    anios = [2022, 2023, 2024] if anio == "Todos" else [anio]
    meses = pd.MultiIndex.from_product([anios, range(1, 13)], names=["AÑO", "MES"])
    tabla = (
        filtrados.groupby(["AÑO", "MES"]).size()
        .reindex(meses, fill_value=0)
        .reset_index(name="Total")
    )
    tabla["Periodo"] = [f"{MESES[mes - 1]} {a}" for a, mes in meses]
    tabla["Etiqueta"] = [
        f"{MESES[mes - 1]} {a % 100:02d}" if len(anios) > 1 else MESES[mes - 1]
        for a, mes in meses
    ]
    tabla["Registros"] = tabla["Total"].map(conteo_local)
    etiquetas = tabla["Etiqueta"].tolist()
    marcas_eje = etiquetas[::6] + etiquetas[-1:] if len(anios) > 1 else etiquetas
    base = alt.Chart(tabla).encode(
        x=alt.X(
            "Etiqueta:O",
            title=None,
            sort=etiquetas,
            scale=alt.Scale(padding=0.25),
            axis=alt.Axis(values=marcas_eje, labelAngle=0, labelOverlap="greedy", grid=False),
        ),
        y=alt.Y("Total:Q", title=None, axis=alt.Axis(format="d", tickMinStep=1, tickCount=5)),
    )
    area = base.mark_area(color="#DDEEF5", opacity=0.8, aria=False)
    linea = base.mark_line(color=AZUL, strokeWidth=2.7, aria=False)
    puntos = base.mark_circle(size=46, color=AZUL, stroke="white", strokeWidth=1.25, opacity=1).encode(
        tooltip=[
            alt.Tooltip("Periodo:N", title="Mes"),
            alt.Tooltip("Registros:N", title="Siniestros"),
        ]
    )
    return _estilo(
        (area + linea + puntos).properties(
            height=345,
            description="Siniestros registrados por mes en el período y la zona seleccionados.",
        )
    )


def grafico_severidad(conteos):
    """Mostrar el reparto de severidades y el total en el centro del anillo."""
    tabla = conteos.assign(
        Orden=conteos["Severidad"].map({nombre: i for i, nombre in enumerate(SEVERIDADES)}),
        Registros=conteos["Total"].map(conteo_local),
        Participacion=conteos["Porcentaje"].map(lambda valor: f"{valor:.1f} %".replace(".", ",")),
    )
    anillo = (
        alt.Chart(tabla)
        .transform_filter(alt.datum.Total > 0)
        .mark_arc(
            innerRadius=alt.ExprRef(expr="min(74, width * 0.34)"),
            outerRadius=alt.ExprRef(expr="min(104, width * 0.48)"),
            cornerRadius=3,
            padAngle=0.025,
        )
        .encode(
            theta=alt.Theta("Total:Q", stack=True),
            order=alt.Order("Orden:Q"),
            color=alt.Color(
                "Severidad:N",
                scale=alt.Scale(domain=SEVERIDADES, range=COLORES_SEVERIDAD),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Severidad:N"),
                alt.Tooltip("Registros:N", title="Siniestros"),
                alt.Tooltip("Participacion:N", title="Participación"),
            ],
        )
    )
    centro = alt.Chart(pd.DataFrame({"Total": [conteo_local(tabla["Total"].sum())]})).encode(
        x=alt.value(alt.ExprRef(expr="width / 2")),
        y=alt.value(alt.ExprRef(expr="height / 2")),
    )
    cifra = centro.mark_text(
        fontSize=alt.ExprRef(expr="min(32, width / 5.2)"),
        fontWeight=700,
        color=TINTA,
        dy=-9,
        aria=False,
    ).encode(text="Total:N")
    unidad = centro.mark_text(
        text="siniestros", fontSize=12, fontWeight=500, color=TEXTO_SECUNDARIO, dy=19, aria=False,
    )
    return _estilo(
        (anillo + cifra + unidad).properties(
            height=227,
            description="Distribución de los siniestros registrados entre las tres categorías de severidad.",
        )
    )


def grafico_barras(tabla, categoria, *, orden=None, color=VERDE):
    """Dibujar barras horizontales con etiquetas de lectura directa en cada fila."""
    nombres = tabla[categoria].str.capitalize()
    datos = tabla.assign(
        Nombre=nombres,
        Etiqueta=nombres.map(lambda valor: fill(valor, width=26, break_long_words=False)),
        Registros=tabla["Total"].map(conteo_local),
    )
    if orden is None:
        orden = datos.sort_values(["Total", categoria], ascending=[False, True])[categoria].tolist()
    maximo = max(1, int(datos["Total"].max()))
    base = alt.Chart(datos).encode(y=alt.Y(f"{categoria}:N", sort=orden, axis=None))
    pista = base.mark_bar(
        orient="horizontal", size=8, cornerRadius=4, color=CUADRICULA, yOffset=11, aria=False,
    ).encode(x=alt.value(0), x2=alt.value("width"))
    barra = base.mark_bar(orient="horizontal", size=8, cornerRadius=4, color=color, yOffset=11).encode(
        x=alt.X("Total:Q", scale=alt.Scale(domain=[0, maximo], nice=False), axis=None),
        x2=alt.value(0),
        tooltip=[
            alt.Tooltip(f"{categoria}:N"),
            alt.Tooltip("Registros:N", title="Siniestros"),
        ],
    )
    etiqueta = base.mark_text(
        align="left", baseline="bottom", lineBreak="\n", lineHeight=13,
        text=alt.ExprRef(expr="width < 300 ? datum.Etiqueta : datum.Nombre"),
        dy=alt.ExprRef(expr="width < 300 && indexof(datum.Etiqueta, '\\n') >= 0 ? -10 : 3"),
        fontSize=12, fontWeight=500, color=TINTA, aria=False,
    ).encode(x=alt.value(0))
    cifra = base.mark_text(
        align="right", baseline="bottom", dy=3, fontSize=12, fontWeight=700, color=TINTA, aria=False,
    ).encode(x=alt.value("width"), text="Registros:N")
    return _estilo(
        (pista + barra + etiqueta + cifra).properties(
            height=len(datos) * 46,
            description=f"Cantidad de siniestros registrados por {categoria.lower()}.",
        )
    )
