"""Presentación Altair de los conteos calculados por la aplicación."""

import textwrap

import altair as alt


SEVERIDADES = ["BAJA", "MEDIA", "ALTA"]
COLORES = ["#167d8d", "#c28b26", "#b74b42"]


def acabado(chart):
    return (
        chart.configure_view(stroke=None)
        .configure_axis(
            labelFont="Arial", titleFont="Arial", labelFontSize=12,
            titleFontSize=12, titleFontWeight="normal", labelColor="#405a64",
            titleColor="#405a64", gridColor="#e6edef", domain=False,
            tickColor="#d8e3e6", titlePadding=14,
        )
        .configure_legend(labelFontSize=12, labelColor="#405a64", title=None)
        .configure(background="transparent")
    )


def donut(conteos):
    chart = alt.Chart(conteos).mark_arc(
        innerRadius=65, outerRadius=100, stroke="#ffffff", strokeWidth=3
    ).encode(
        theta=alt.Theta("Total:Q", stack=True),
        color=alt.Color(
            "Severidad:N", scale=alt.Scale(domain=SEVERIDADES, range=COLORES),
            legend=alt.Legend(orient="bottom", direction="horizontal"),
        ),
        order=alt.Order("Orden:Q"),
        tooltip=[
            alt.Tooltip("Severidad:N"), alt.Tooltip("Total:Q", title="Siniestros", format=",d"),
            alt.Tooltip("Porcentaje:Q", title="Porcentaje", format=".2f"),
        ],
    ).properties(height=240)
    return acabado(chart)


def barras(tabla, categoria, *, orden=None, alto=350, ancho_etiqueta=18):
    # Saltos de línea solo visuales. El tooltip conserva la categoría completa.
    datos = tabla.copy()
    datos["Etiqueta"] = datos[categoria].map(
        lambda valor: "\n".join(textwrap.wrap(str(valor).capitalize(), ancho_etiqueta))
    )
    etiquetas = None
    if orden is not None:
        etiquetas = ["\n".join(textwrap.wrap(valor.capitalize(), ancho_etiqueta)) for valor in orden]
    chart = alt.Chart(datos).mark_bar(
        color="#167d8d", cornerRadiusEnd=4, size=18
    ).encode(
        x=alt.X("Total:Q", title="Número de siniestros", axis=alt.Axis(tickCount=4, format="d")),
        y=alt.Y(
            "Etiqueta:N", sort=etiquetas if etiquetas is not None else "-x", title=None,
            axis=alt.Axis(
                labelExpr="split(datum.label, '\\n')", labelLimit=0,
                labelLineHeight=15, labelPadding=10, ticks=False,
            ),
        ),
        tooltip=[alt.Tooltip(f"{categoria}:N"), alt.Tooltip("Total:Q", title="Siniestros", format=",d")],
    ).properties(height=alto)
    return acabado(chart)
