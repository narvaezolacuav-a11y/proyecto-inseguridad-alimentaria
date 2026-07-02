
import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
from pathlib import Path

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

st.set_page_config(page_title="Inseguridad Alimentaria", page_icon="🍽️", layout="wide")

css = Path("assets/style.css")
if css.exists():
    st.markdown(f"<style>{css.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <img class="logo" src="assets/logo.svg" width="95">
  <h1>🍽️ Predicción de Inseguridad Alimentaria</h1>
  <p>Modelo predictivo y clasificación de riesgo por distrito en Lima Metropolitana</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def cargar_datos():
    return pd.read_excel("Entregable_3_Dataset_Transformado_Inseguridad_Alimentaria.xlsx")

df = cargar_datos()

@st.cache_resource
def entrenar(df):
    data = df.copy()
    encoders = {}
    cat_cols = ["Distrito","Zona","Tipo_Empleo","Nivel_Educativo","Programa_Social","Acceso_Agua","Acceso_Desague","Estado_Nutricional","Nivel_Riesgo"]
    for col in cat_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le

    features = ["Año","Distrito","Ingreso_Laboral","Gasto_Alimentos","Inflacion_Alimentaria","Integrantes_Hogar","Porcentaje_Gasto_Alimentos","Indice_Vulnerabilidad"]
    X = data[features]
    y_reg = data["Probabilidad_Enfermedad_Alimentaria"]
    y_clf = data["Nivel_Riesgo"]

    X_train, X_test, yr_train, yr_test = train_test_split(X, y_reg, test_size=.2, random_state=42)
    _, _, yc_train, yc_test = train_test_split(X, y_clf, test_size=.2, random_state=42)

    reg = RandomForestRegressor(n_estimators=220, random_state=42)
    clf = RandomForestClassifier(n_estimators=220, random_state=42)
    reg.fit(X_train, yr_train)
    clf.fit(X_train, yc_train)

    metrics = {
        "MAE": mean_absolute_error(yr_test, reg.predict(X_test)),
        "R2": r2_score(yr_test, reg.predict(X_test)),
        "Accuracy": accuracy_score(yc_test, clf.predict(X_test))
    }
    return reg, clf, encoders, features, metrics

def preparar_futuro(anio, df, reg, clf, encoders, features):
    base = df.groupby("Distrito").agg({
        "Ingreso_Laboral":"mean",
        "Gasto_Alimentos":"mean",
        "Inflacion_Alimentaria":"mean",
        "Integrantes_Hogar":"mean",
        "Porcentaje_Gasto_Alimentos":"mean",
        "Indice_Vulnerabilidad":"mean"
    }).reset_index()
    base["Año"] = anio
    model = base.copy()
    model["Distrito"] = encoders["Distrito"].transform(model["Distrito"].astype(str))
    X_future = model[features]
    base["Probabilidad"] = reg.predict(X_future)
    base["Nivel_Riesgo_Code"] = clf.predict(X_future).astype(int)
    base["Clasificación"] = encoders["Nivel_Riesgo"].inverse_transform(base["Nivel_Riesgo_Code"])
    return base.sort_values("Probabilidad", ascending=False)

def excel_bytes(tabla):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        tabla.to_excel(writer, index=False, sheet_name="Resultados")
    return output.getvalue()

def pdf_bytes(anio, tabla, top, explicacion):
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    output = BytesIO()
    doc = SimpleDocTemplate(output, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Reporte de Predicción de Inseguridad Alimentaria", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"Año seleccionado: {anio}", styles["Heading2"]),
        Paragraph(f"Distrito con mayor probabilidad: {top['Distrito']}", styles["Normal"]),
        Paragraph(f"Probabilidad estimada: {top['Probabilidad']:.2f}%", styles["Normal"]),
        Paragraph(f"Clasificación: {top['Clasificación']}", styles["Normal"]),
        Spacer(1, 12),
        Paragraph("Explicación automática", styles["Heading2"]),
        Paragraph(explicacion, styles["Normal"]),
        Spacer(1, 12)
    ]
    resumen = tabla[["Distrito","Probabilidad","Clasificación"]].head(10).copy()
    resumen["Probabilidad"] = resumen["Probabilidad"].round(2)
    t = Table([resumen.columns.tolist()] + resumen.values.tolist())
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#2E7D32")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), .5, colors.grey)
    ]))
    story.append(t)
    doc.build(story)
    return output.getvalue()

if "modelo_entrenado" not in st.session_state:
    st.session_state.modelo_entrenado = False
if "resultados" not in st.session_state:
    st.session_state.resultados = None

st.sidebar.image("assets/logo.svg", width=115)
st.sidebar.header("Panel de control")
anio = st.sidebar.selectbox("Selector de año", list(range(2026, 2036)), index=4)
distrito_sel = st.sidebar.selectbox("Selector de distrito", sorted(df["Distrito"].unique()))
busqueda = st.sidebar.text_input("Barra de búsqueda de distrito", "")

st.markdown('<div class="section-title"> Acciones del aplicativo</div>', unsafe_allow_html=True)
a1, a2, a3, a4, a5 = st.columns(5)
entrenar_btn = a1.button("Entrenar modelo")
predecir_btn = a2.button("Realizar predicción")
clasificar_btn = a3.button("Clasificar distritos")
pdf_btn = a4.button("Preparar PDF")
excel_btn = a5.button("Preparar Excel")

if entrenar_btn or not st.session_state.modelo_entrenado:
    with st.spinner("Entrenando modelos..."):
        reg, clf, encoders, features, metrics = entrenar(df)
        st.session_state.reg = reg
        st.session_state.clf = clf
        st.session_state.encoders = encoders
        st.session_state.features = features
        st.session_state.metrics = metrics
        st.session_state.modelo_entrenado = True
    st.success("Modelo entrenado correctamente.")

reg = st.session_state.reg
clf = st.session_state.clf
encoders = st.session_state.encoders
features = st.session_state.features
metrics = st.session_state.metrics

if predecir_btn or clasificar_btn or st.session_state.resultados is None:
    st.session_state.resultados = preparar_futuro(anio, df, reg, clf, encoders, features)

tabla = preparar_futuro(anio, df, reg, clf, encoders, features)
top = tabla.iloc[0]
fila_distrito = tabla[tabla["Distrito"] == distrito_sel].iloc[0]

tabla_mostrar = tabla[tabla["Distrito"].str.contains(busqueda, case=False, na=False)].copy() if busqueda.strip() else tabla.copy()

st.markdown('<div class="section-title"> Dashboard con indicadores</div>', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Año", anio)
m2.metric("Distrito más vulnerable", top["Distrito"])
m3.metric("Probabilidad máxima", f"{top['Probabilidad']:.2f}%")
m4.metric("Clasificación principal", top["Clasificación"])

st.markdown('<div class="section-title"> PREDICCIÓN</div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="pred-card">
  <h2>Año: {anio}</h2>
  <p>Distrito con mayor probabilidad</p>
  <h3>{top["Distrito"]}</h3>
  <h2>Probabilidad: {top["Probabilidad"]:.2f}%</h2>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title"> CLASIFICACIÓN</div>', unsafe_allow_html=True)
clasificacion = tabla[["Distrito","Clasificación","Probabilidad"]].head(12).copy()
clasificacion["Probabilidad"] = clasificacion["Probabilidad"].round(2)
st.dataframe(clasificacion, use_container_width=True)

st.markdown('<div class="section-title"> Consulta por distrito seleccionado</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
c1.metric("Distrito", distrito_sel)
c2.metric("Probabilidad", f"{fila_distrito['Probabilidad']:.2f}%")
c3.metric("Riesgo", fila_distrito["Clasificación"])

st.markdown('<div class="section-title"> Tabla de resultados</div>', unsafe_allow_html=True)
tabla_final = tabla_mostrar[["Distrito","Año","Ingreso_Laboral","Gasto_Alimentos","Porcentaje_Gasto_Alimentos","Indice_Vulnerabilidad","Probabilidad","Clasificación"]].copy()
for col in ["Ingreso_Laboral","Gasto_Alimentos","Porcentaje_Gasto_Alimentos","Indice_Vulnerabilidad","Probabilidad"]:
    tabla_final[col] = tabla_final[col].round(2)
st.dataframe(tabla_final, use_container_width=True)

st.markdown('<div class="section-title"> Gráficos dinámicos</div>', unsafe_allow_html=True)
g1, g2 = st.columns(2)
with g1:
    fig = px.bar(tabla.head(10), x="Distrito", y="Probabilidad", color="Clasificación", title=f"Top 10 distritos con mayor probabilidad - {anio}")
    st.plotly_chart(fig, use_container_width=True)
with g2:
    fig2 = px.scatter(tabla, x="Ingreso_Laboral", y="Probabilidad", color="Clasificación", hover_name="Distrito", size="Gasto_Alimentos", title="Ingreso laboral vs probabilidad")
    st.plotly_chart(fig2, use_container_width=True)

explicacion = (
    f"El modelo asigna mayor probabilidad a {top['Distrito']} porque presenta un ingreso laboral "
    f"promedio aproximado de S/ {top['Ingreso_Laboral']:.2f}, un porcentaje del ingreso destinado a alimentos "
    f"de {top['Porcentaje_Gasto_Alimentos']:.2f} y un índice de vulnerabilidad de {top['Indice_Vulnerabilidad']:.2f}. "
    f"Estas condiciones aumentan el riesgo de padecer enfermedades alimentarias asociadas a la inseguridad alimentaria."
)

st.markdown('<div class="section-title"> Explicación automática</div>', unsafe_allow_html=True)
st.markdown(f'<div class="info-box">{explicacion}</div>', unsafe_allow_html=True)

st.markdown('<div class="section-title"> Exportar resultados</div>', unsafe_allow_html=True)
e1, e2 = st.columns(2)
with e1:
    st.download_button("Exportar Excel", data=excel_bytes(tabla_final), file_name=f"resultados_inseguridad_alimentaria_{anio}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
with e2:
    st.download_button("Exportar PDF", data=pdf_bytes(anio, tabla, top, explicacion), file_name=f"reporte_inseguridad_alimentaria_{anio}.pdf", mime="application/pdf")

