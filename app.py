
import base64
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================

st.set_page_config(
    page_title="Inseguridad Alimentaria",
    page_icon="🍽️",
    layout="wide"
)


# =========================================================
# ESTILOS
# =========================================================

def cargar_css():
    ruta_css = Path("assets/style.css")
    if ruta_css.exists():
        st.markdown(
            f"<style>{ruta_css.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True
        )

def svg_a_base64(ruta):
    archivo = Path(ruta)
    if archivo.exists():
        contenido = archivo.read_text(encoding="utf-8")
        return base64.b64encode(contenido.encode("utf-8")).decode("utf-8")
    return ""

cargar_css()

logo_base64 = svg_a_base64("assets/logo.svg")

if logo_base64:
    logo_html = f'<img class="logo" src="data:image/svg+xml;base64,{logo_base64}" width="95">'
else:
    logo_html = "🍽️"

st.markdown(f"""
<div class="hero">
  {logo_html}
  <h1> Predicción de Inseguridad Alimentaria</h1>
  <p>Modelo predictivo y clasificación de riesgo por distrito en Lima Metropolitana</p>
</div>
""", unsafe_allow_html=True)


# =========================================================
# CARGA DE DATOS
# =========================================================

ARCHIVO_DATASET = "Entregable_3_Dataset_Transformado_Inseguridad_Alimentaria.xlsx"

@st.cache_data
def cargar_datos():
    ruta = Path(ARCHIVO_DATASET)
    if not ruta.exists():
        st.error(
            f"No se encontró el archivo {ARCHIVO_DATASET}. "
            "Verifica que esté en la misma carpeta que app.py."
        )
        st.stop()

    df = pd.read_excel(ruta)

    columnas_necesarias = [
        "Distrito",
        "Año",
        "Ingreso_Laboral",
        "Gasto_Alimentos",
        "Inflacion_Alimentaria",
        "Integrantes_Hogar",
        "Porcentaje_Gasto_Alimentos",
        "Indice_Vulnerabilidad",
        "Probabilidad_Enfermedad_Alimentaria",
        "Nivel_Riesgo"
    ]

    faltantes = [col for col in columnas_necesarias if col not in df.columns]
    if faltantes:
        st.error("Faltan columnas necesarias en el dataset: " + ", ".join(faltantes))
        st.stop()

    return df

df = cargar_datos()


# =========================================================
# ENTRENAMIENTO DE MODELOS
# =========================================================

@st.cache_resource
def entrenar_modelos(df):
    data = df.copy()

    columnas_categoricas = [
        "Distrito",
        "Zona",
        "Tipo_Empleo",
        "Nivel_Educativo",
        "Programa_Social",
        "Acceso_Agua",
        "Acceso_Desague",
        "Estado_Nutricional",
        "Nivel_Riesgo"
    ]

    encoders = {}

    for col in columnas_categoricas:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            encoders[col] = le

    variables_modelo = [
        "Año",
        "Distrito",
        "Ingreso_Laboral",
        "Gasto_Alimentos",
        "Inflacion_Alimentaria",
        "Integrantes_Hogar",
        "Porcentaje_Gasto_Alimentos",
        "Indice_Vulnerabilidad"
    ]

    X = data[variables_modelo]
    y_regresion = data["Probabilidad_Enfermedad_Alimentaria"]
    y_clasificacion = data["Nivel_Riesgo"]

    X_train, X_test, y_train_reg, y_test_reg = train_test_split(
        X,
        y_regresion,
        test_size=0.20,
        random_state=42
    )

    _, _, y_train_clf, y_test_clf = train_test_split(
        X,
        y_clasificacion,
        test_size=0.20,
        random_state=42
    )

    modelo_regresion = RandomForestRegressor(
        n_estimators=250,
        random_state=42,
        max_depth=8
    )

    modelo_clasificacion = RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        max_depth=8
    )

    modelo_regresion.fit(X_train, y_train_reg)
    modelo_clasificacion.fit(X_train, y_train_clf)

    pred_reg = modelo_regresion.predict(X_test)
    pred_clf = modelo_clasificacion.predict(X_test)

    metricas = {
        "MAE": mean_absolute_error(y_test_reg, pred_reg),
        "R2": r2_score(y_test_reg, pred_reg),
        "Accuracy": accuracy_score(y_test_clf, pred_clf)
    }

    return modelo_regresion, modelo_clasificacion, encoders, variables_modelo, metricas


def generar_prediccion_futura(anio, df, modelo_regresion, modelo_clasificacion, encoders, variables_modelo):
    base = (
        df.groupby("Distrito")
        .agg({
            "Ingreso_Laboral": "mean",
            "Gasto_Alimentos": "mean",
            "Inflacion_Alimentaria": "mean",
            "Integrantes_Hogar": "mean",
            "Porcentaje_Gasto_Alimentos": "mean",
            "Indice_Vulnerabilidad": "mean"
        })
        .reset_index()
    )

    base["Año"] = anio

    datos_modelo = base.copy()
    datos_modelo["Distrito"] = encoders["Distrito"].transform(datos_modelo["Distrito"].astype(str))

    X_futuro = datos_modelo[variables_modelo]

    base["Probabilidad"] = modelo_regresion.predict(X_futuro)
    base["Probabilidad"] = base["Probabilidad"].clip(0, 100)

    codigos_riesgo = modelo_clasificacion.predict(X_futuro).astype(int)
    base["Clasificación"] = encoders["Nivel_Riesgo"].inverse_transform(codigos_riesgo)

    base = base.sort_values("Probabilidad", ascending=False).reset_index(drop=True)

    return base


def exportar_excel(tabla):
    salida = BytesIO()
    with pd.ExcelWriter(salida, engine="openpyxl") as writer:
        tabla.to_excel(writer, index=False, sheet_name="Resultados")
    return salida.getvalue()


def exportar_pdf(anio, tabla, top, explicacion):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet

    salida = BytesIO()
    doc = SimpleDocTemplate(salida, pagesize=letter)
    estilos = getSampleStyleSheet()
    elementos = []

    elementos.append(Paragraph("Reporte de Predicción de Inseguridad Alimentaria", estilos["Title"]))
    elementos.append(Spacer(1, 12))
    elementos.append(Paragraph(f"Año seleccionado: {anio}", estilos["Heading2"]))
    elementos.append(Paragraph(f"Distrito con mayor probabilidad: {top['Distrito']}", estilos["Normal"]))
    elementos.append(Paragraph(f"Probabilidad estimada: {top['Probabilidad']:.2f}%", estilos["Normal"]))
    elementos.append(Paragraph(f"Clasificación: {top['Clasificación']}", estilos["Normal"]))
    elementos.append(Spacer(1, 12))
    elementos.append(Paragraph("Explicación automática", estilos["Heading2"]))
    elementos.append(Paragraph(explicacion, estilos["Normal"]))
    elementos.append(Spacer(1, 12))

    resumen = tabla[["Distrito", "Probabilidad", "Clasificación"]].head(10).copy()
    resumen["Probabilidad"] = resumen["Probabilidad"].round(2)

    datos_tabla = [resumen.columns.tolist()] + resumen.values.tolist()
    tabla_pdf = Table(datos_tabla)
    tabla_pdf.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E7D32")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))

    elementos.append(tabla_pdf)
    doc.build(elementos)

    return salida.getvalue()


# =========================================================
# PANEL DE CONTROL
# =========================================================

if "entrenado" not in st.session_state:
    st.session_state.entrenado = False

st.sidebar.header("⚙️ Panel de control")
st.sidebar.markdown("Selecciona los parámetros de análisis.")

anio_seleccionado = st.sidebar.selectbox(
    "Selector de año",
    list(range(2026, 2036)),
    index=4
)

distrito_seleccionado = st.sidebar.selectbox(
    "Selector de distrito",
    sorted(df["Distrito"].unique())
)

busqueda = st.sidebar.text_input(
    "Barra de búsqueda de distrito",
    ""
)


# =========================================================
# ACCIONES
# =========================================================

st.markdown('<div class="section-title"> Acciones del aplicativo</div>', unsafe_allow_html=True)

b1, b2, b3, b4, b5 = st.columns(5)

boton_entrenar = b1.button("Entrenar modelo")
boton_predecir = b2.button("Realizar predicción")
boton_clasificar = b3.button("Clasificar distritos")
boton_pdf = b4.button("Exportar PDF")
boton_excel = b5.button("Exportar Excel")

if boton_entrenar or not st.session_state.entrenado:
    with st.spinner("Entrenando modelo predictivo y modelo de clasificación..."):
        modelo_regresion, modelo_clasificacion, encoders, variables_modelo, metricas = entrenar_modelos(df)

        st.session_state.modelo_regresion = modelo_regresion
        st.session_state.modelo_clasificacion = modelo_clasificacion
        st.session_state.encoders = encoders
        st.session_state.variables_modelo = variables_modelo
        st.session_state.metricas = metricas
        st.session_state.entrenado = True

    st.success("Modelos entrenados correctamente.")

modelo_regresion = st.session_state.modelo_regresion
modelo_clasificacion = st.session_state.modelo_clasificacion
encoders = st.session_state.encoders
variables_modelo = st.session_state.variables_modelo
metricas = st.session_state.metricas

tabla_resultados = generar_prediccion_futura(
    anio_seleccionado,
    df,
    modelo_regresion,
    modelo_clasificacion,
    encoders,
    variables_modelo
)

if busqueda.strip():
    tabla_filtrada = tabla_resultados[
        tabla_resultados["Distrito"].str.contains(busqueda, case=False, na=False)
    ].copy()
else:
    tabla_filtrada = tabla_resultados.copy()

top = tabla_resultados.iloc[0]
fila_distrito = tabla_resultados[tabla_resultados["Distrito"] == distrito_seleccionado].iloc[0]


# =========================================================
# DASHBOARD
# =========================================================

st.markdown('<div class="section-title"> Dashboard con indicadores</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)

m1.metric("Año analizado", anio_seleccionado)
m2.metric("Distrito más vulnerable", top["Distrito"])
m3.metric("Probabilidad máxima", f"{top['Probabilidad']:.2f}%")
m4.metric("Clasificación", top["Clasificación"])

m5, m6, m7 = st.columns(3)
m5.metric("Error MAE", f"{metricas['MAE']:.2f}")
m6.metric("R²", f"{metricas['R2']:.2f}")
m7.metric("Accuracy", f"{metricas['Accuracy'] * 100:.1f}%")


# =========================================================
# PREDICCIÓN
# =========================================================

st.markdown('<div class="section-title">🔮 PREDICCIÓN</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="pred-card">
  <h2>Año: {anio_seleccionado}</h2>
  <p>Distrito con mayor probabilidad</p>
  <h3>{top["Distrito"]}</h3>
  <h2>Probabilidad: {top["Probabilidad"]:.2f}%</h2>
</div>
""", unsafe_allow_html=True)


# =========================================================
# CLASIFICACIÓN
# =========================================================

st.markdown('<div class="section-title"> CLASIFICACIÓN</div>', unsafe_allow_html=True)

clasificacion = tabla_resultados[["Distrito", "Clasificación", "Probabilidad"]].head(12).copy()
clasificacion["Probabilidad"] = clasificacion["Probabilidad"].round(2)

st.dataframe(clasificacion, use_container_width=True)


# =========================================================
# CONSULTA INDIVIDUAL
# =========================================================

st.markdown('<div class="section-title"> Consulta por distrito seleccionado</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

c1.metric("Distrito", distrito_seleccionado)
c2.metric("Probabilidad", f"{fila_distrito['Probabilidad']:.2f}%")
c3.metric("Nivel de riesgo", fila_distrito["Clasificación"])


# =========================================================
# TABLA COMPLETA
# =========================================================

st.markdown('<div class="section-title"> Tabla de resultados</div>', unsafe_allow_html=True)

tabla_final = tabla_filtrada[
    [
        "Distrito",
        "Año",
        "Ingreso_Laboral",
        "Gasto_Alimentos",
        "Porcentaje_Gasto_Alimentos",
        "Indice_Vulnerabilidad",
        "Probabilidad",
        "Clasificación"
    ]
].copy()

for col in [
    "Ingreso_Laboral",
    "Gasto_Alimentos",
    "Porcentaje_Gasto_Alimentos",
    "Indice_Vulnerabilidad",
    "Probabilidad"
]:
    tabla_final[col] = tabla_final[col].round(2)

st.dataframe(tabla_final, use_container_width=True)


# =========================================================
# GRÁFICOS
# =========================================================

st.markdown('<div class="section-title"> Gráficos dinámicos</div>', unsafe_allow_html=True)

g1, g2 = st.columns(2)

with g1:
    fig1 = px.bar(
        tabla_resultados.head(10),
        x="Distrito",
        y="Probabilidad",
        color="Clasificación",
        title=f"Top 10 distritos con mayor probabilidad - {anio_seleccionado}"
    )
    st.plotly_chart(fig1, use_container_width=True)

with g2:
    fig2 = px.scatter(
        tabla_resultados,
        x="Ingreso_Laboral",
        y="Probabilidad",
        color="Clasificación",
        hover_name="Distrito",
        size="Gasto_Alimentos",
        title="Ingreso laboral vs probabilidad"
    )
    st.plotly_chart(fig2, use_container_width=True)


# =========================================================
# EXPLICACIÓN AUTOMÁTICA
# =========================================================

explicacion = (
    f"El modelo asigna mayor probabilidad a {top['Distrito']} porque presenta un ingreso laboral "
    f"promedio aproximado de S/ {top['Ingreso_Laboral']:.2f}, un porcentaje del ingreso destinado a alimentos "
    f"de {top['Porcentaje_Gasto_Alimentos']:.2f} y un índice de vulnerabilidad de {top['Indice_Vulnerabilidad']:.2f}. "
    f"Estas condiciones aumentan el riesgo de padecer enfermedades alimentarias asociadas a la inseguridad alimentaria."
)

st.markdown('<div class="section-title"> Explicación automática</div>', unsafe_allow_html=True)
st.markdown(f'<div class="info-box">{explicacion}</div>', unsafe_allow_html=True)


# =========================================================
# EXPORTACIONES
# =========================================================

st.markdown('<div class="section-title"> Exportar resultados</div>', unsafe_allow_html=True)

e1, e2 = st.columns(2)

with e1:
    st.download_button(
        "Exportar Excel",
        data=exportar_excel(tabla_final),
        file_name=f"resultados_inseguridad_alimentaria_{anio_seleccionado}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

with e2:
    st.download_button(
        "Exportar PDF",
        data=exportar_pdf(anio_seleccionado, tabla_resultados, top, explicacion),
        file_name=f"reporte_inseguridad_alimentaria_{anio_seleccionado}.pdf",
        mime="application/pdf"
    )
