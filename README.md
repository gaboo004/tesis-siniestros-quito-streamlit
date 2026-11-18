# Siniestros de tránsito en Quito

Aplicación web del trabajo de titulación **Análisis y predicción de la severidad de los siniestros de tránsito en el Distrito Metropolitano de Quito mediante aprendizaje automático**, desarrollado por Gabriel Naranjo en la Pontificia Universidad Católica del Ecuador.

El proyecto reúne la exploración de registros de 2022–2024 y la clasificación retrospectiva de su severidad mediante un modelo Random Forest.

- [Consultar la aplicación en Streamlit](https://tesis-siniestros-quito.streamlit.app/)
- [Consultar los cuadernos, datos y materiales de la investigación](https://drive.google.com/drive/folders/1JYhUFYFqxz2TYDEn5wGrCkPbXT8957T2)

## Funciones de la aplicación

**Análisis de siniestros.** Permite filtrar por año y zona, consultar el número de registros y la proporción de severidad alta, y examinar las distribuciones por severidad, clase, causa registrada y día de la semana.

**Estimación de severidad.** Recibe mes, día de la semana, hora, zona, clase del siniestro y causa registrada. El modelo devuelve una de las categorías baja, media o alta.

## Datos y definición de severidad

La base procesada contiene **11.099 siniestros registrados en Quito**. Los archivos originales proceden del **Instituto Nacional de Estadística y Censos del Ecuador (INEC), Estadísticas de Transporte de 2022, 2023 y 2024**. La investigación integró los archivos anuales, homologó sus códigos y seleccionó los registros del cantón Quito. El archivo de este repositorio corresponde a esa versión procesada.

| Categoría | Criterio aplicado al registro | Siniestros |
| --- | --- | ---: |
| Baja | Sin personas fallecidas ni lesionadas | 5.653 |
| Media | Con personas lesionadas y sin fallecidos | 4.580 |
| Alta | Con al menos una persona fallecida | 866 |

La categoría alta se basa en los fallecimientos consignados en el registro del siniestro.

## Modelo y evaluación

Se seleccionó un Random Forest balanceado con seis atributos categóricos y preprocesamiento integrado mediante codificación one-hot. El entrenamiento final utilizó **7.210 registros de 2022 y 2023**. La evaluación temporal comprendió **3.889 registros de 2024**.

| Resultado en 2024 | Valor |
| --- | ---: |
| Exactitud global | 0,5647 |
| Exactitud balanceada | 0,5017 |
| F1 macro | 0,4796 |
| Sensibilidad para severidad alta | 36,45 % |
| Precisión para severidad alta | 20,07 % |

El modelo identificó 109 de los 299 siniestros con fallecidos del año de prueba. La clase del siniestro y la causa registrada se conocen después de ocurrido el hecho, por lo que la estimación tiene alcance retrospectivo. Su uso académico debe considerar la frecuencia de omisiones y falsas alarmas documentada en la tesis.

## Organización del repositorio

```text
app.py                 Interfaz y navegación de Streamlit
figures.py             Gráficos del módulo descriptivo
estilos.css            Presentación visual
requirements.txt       Dependencias de ejecución
datos/                 Base histórica procesada
modelos/               Modelo entrenado con su preprocesamiento
.devcontainer/         Configuración del entorno de desarrollo
```

El archivo `modelos/random_forest_final_2022_2023.joblib` contiene el modelo que utiliza el formulario. La aplicación carga ese archivo para realizar las estimaciones.

## Ejecución local

Se utiliza Python 3.12 y las versiones fijadas en `requirements.txt`. Desde la carpeta del proyecto se ejecutan los siguientes comandos.

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

La aplicación necesita las carpetas `datos/` y `modelos/` junto al programa principal.

## Despliegue

La versión publicada se ejecuta en Streamlit Community Cloud desde la rama `main`, con `app.py` como archivo de entrada. Los cuadernos de preparación, exploración, comparación de modelos y evaluación se conservan en la carpeta de investigación enlazada al inicio.
