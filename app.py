
from io import BytesIO
from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

# Configuración inicial de página
st.set_page_config(page_title="Inseguridad Alimentaria", layout="wide", initial_sidebar_state="expanded")
DATASET = "Entregable_3_Dataset_Transformado_Inseguridad_Alimentaria.xlsx"

def load_css():
    css_path = Path("assets/style.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
load_css()

@st.cache_data
def load_data():
    path = Path(DATASET)
    if not path.exists():
        st.error(f"No se encontró el archivo {DATASET} en el directorio actual.")
        st.stop()
    df = pd.read_excel(path)
    return df

df = load_data()

# Modelo predictivo optimizado
@st.cache_resource
def train_model(df_input):
    df_model = df_input.copy()
    le = LabelEncoder()
    df_model["Distrito_C"] = le.fit_transform(df_model["Distrito"])
    
    features = ["Año", "Distrito_C", "Ingreso_Laboral", "Gasto_Alimentos", "Inflacion_Alimentaria", "Integrantes_Hogar"]
    X = df_model[features]
    y = df_model["Probabilidad_Enfermedad_Alimentaria"]
    
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X, y)
    return model, le, features

model, encoder, features = train_model(df)

# --- PANEL DE CONTROL SIDEBAR ---
st.sidebar.markdown('<div class="sidebar-title">PLATAFORMA IA</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-subtitle">INSEGURIDAD ALIMENTARIA</div>', unsafe_allow_html=True)

year = st.sidebar.selectbox("Año de Proyección", list(range(2026, 2036)), index=0)
search_district = st.sidebar.text_input("Filtrar por Distrito", "")

# --- CÁLCULO DE PROYECCIONES ---
base_districts = df.groupby("Distrito").agg({
    "Ingreso_Laboral": "mean",
    "Gasto_Alimentos": "mean",
    "Inflacion_Alimentaria": "mean",
    "Integrantes_Hogar": "mean"
}).reset_index()

base_districts["Año"] = year
base_districts["Distrito_C"] = encoder.transform(base_districts["Distrito"])

X_pred = base_districts[features]
base_districts["Probabilidad"] = model.predict(X_pred)
base_districts["Probabilidad"] = base_districts["Probabilidad"].clip(0, 100)

def asignar_riesgo(p):
    if p >= 50: return "ALTO", "#EF4444"
    if p >= 25: return "MEDIO", "#F59E0B"
    return "BAJO", "#10B981"

res_risk = [asignar_riesgo(p) for p in base_districts["Probabilidad"]]
base_districts["Nivel de Riesgo"] = [r[0] for r in res_risk]
base_districts["Color"] = [r[1] for r in res_risk]

def interpretar(r):
    if r == "ALTO": return "Prioridad Crítica: Intervención alimentaria inmediata recomendada."
    if r == "MEDIO": return "Vulnerabilidad Moderada: Monitoreo preventivo de canasta básica."
    return "Situación Estable: Índices óptimos de seguridad alimentaria."

base_districts["Interpretación"] = base_districts["Nivel de Riesgo"].apply(interpretar)
sorted_results = base_districts.sort_values(by="Probabilidad", ascending=False).reset_index(drop=True)
sorted_results["#"] = sorted_results.index + 1

# Filtrado dinámico
if search_district.strip():
    filtered = sorted_results[sorted_results["Distrito"].str.contains(search_district, case=False, na=False)].copy()
else:
    filtered = sorted_results.copy()

top = sorted_results.iloc[0]

# --- CUERPO PRINCIPAL DEL APLICATIVO ---
st.markdown('<div class="main-title">Seguridad Alimentaria en Lima Metropolitana</div>', unsafe_allow_html=True)
st.markdown(f'<div class="main-subtitle">Reporte analítico y proyecciones predictivas mediante Machine Learning para el año {year}</div>', unsafe_allow_html=True)

# Bloque Principal: Predicción Crítica
st.markdown(f"""
<div class="pred-card">
    <h2>DISTRITO CON MAYOR VULNERABILIDAD IDENTIFICADO ({year})</h2>
    <h3>{top["Distrito"]}</h3>
    <div class="pred-value">{top["Probabilidad"]:.1f}%</div>
    <p style="margin-top:10px; opacity:0.9;">Probabilidad estimada de presentar brechas en seguridad alimentaria</p>
</div>
""", unsafe_allow_html=True)

# Grid Interactivo: Tabla y Leyendas
st.markdown('<div class="table-card">', unsafe_allow_html=True)
st.markdown(f'<div class="table-title">Ranking General de Riesgo por Distrito ({year})</div>', unsafe_allow_html=True)

def styled_badge(row):
    return f'<span class="badge" style="background-color:{row["Color"]};">{row["Nivel de Riesgo"]}</span>'

table_show = filtered[["#", "Distrito", "Probabilidad", "Nivel de Riesgo", "Interpretación"]].copy()
table_show["Probabilidad"] = table_show["Probabilidad"].apply(lambda val: f"{val:.1f}%")
table_show["Nivel de Riesgo"] = filtered.apply(styled_badge, axis=1)

st.markdown(table_show.to_html(escape=False, index=False), unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Cuadro explicativo automático inferior
st.markdown(f"""
<div class="explanation">
    <b>Interpretación del Algoritmo:</b><br>
    Para el periodo fiscal proyectado <b>{year}</b>, el distrito de <b>{top["Distrito"]}</b> se sitúa en la parte superior del ranking de riesgo. El algoritmo RandomForest detecta este comportamiento debido a un ingreso laboral promedio histórico indexado de S/. {top["Ingreso_Laboral"]:.2f} acoplado a un volumen de carga familiar promedio de {top["Integrantes_Hogar"]:.1f} miembros.
</div>
""", unsafe_allow_html=True)

# Zona de Descargas Ejecutivas
st.markdown("<br>", unsafe_allow_html=True)
d1, d2 = st.columns(2)

with d1:
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        sorted_results[["#", "Distrito", "Año", "Ingreso_Laboral", "Gasto_Alimentos", "Probabilidad", "Nivel de Riesgo"]].to_excel(writer, index=False, sheet_name="Proyección")
    st.download_button(
        label="📥 Descargar Base de Datos (Excel)",
        data=excel_buffer.getvalue(),
        file_name=f"Reporte_Seguridad_Alimentaria_{year}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

with d2:
    if REPORTLAB_OK:
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = [
            Paragraph(f"Reporte Analitico de Vulnerabilidad - Año {year}", styles["Title"]),
            Spacer(1, 15),
            Paragraph(f"Distrito critico detectado: {top['Distrito']} con {top['Probabilidad']:.1f}% de riesgo.", styles["Heading2"]),
            Spacer(1, 15)
        ]
        doc.build(story)
        st.download_button(
            label="📄 Descargar Ficha Ejecutiva (PDF)",
            data=pdf_buffer.getvalue(),
            file_name=f"Ficha_Ejecutiva_{year}.pdf",
            mime="application/pdf"
        )
    else:
        st.button("📄 Ficha Ejecutiva PDF no disponible", disabled=True)
