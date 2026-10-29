# Siniestros de tránsito en el Distrito Metropolitano de Quito

Aplicación académica de Streamlit que presenta un análisis histórico de siniestros de tránsito registrados en Quito (2022-2024) y una estimación condicionada de severidad (BAJA, MEDIA o ALTA) mediante un modelo Random Forest entrenado previamente.

## Alcance y limitaciones

El modelo estima severidad a partir de las características de un siniestro ya registrado; **no predice si ocurrirá un siniestro**. Presenta limitaciones de desempeño, en particular en la categoría ALTA, y no debe utilizarse como sistema autónomo de decisión de seguridad vial.

## Organización

- `app.py`: interfaz Streamlit con los dos módulos.
- `requirements.txt`: bibliotecas necesarias (scikit-learn 1.6.1).
- `datos/siniestros_quito_2022_2024_limpio.csv`: datos históricos procesados.
- `modelos/random_forest_final_2022_2023.joblib`: Pipeline preentrenado (sin reentrenamiento).

## Ejecución

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Despliegue

En Streamlit Community Cloud: seleccionar este repositorio, la rama principal y el archivo `app.py`.

**Nota sobre los datos:** verificar atribución y permisos de redistribución de la fuente antes de publicar el repositorio como público.
