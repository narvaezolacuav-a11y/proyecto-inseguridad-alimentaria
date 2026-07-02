
import base64
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
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

# Configuración de página e interfaz moderna
st.set_page_config(page_title="Plataforma IA - Seguridad Alimentaria", layout="wide", initial_sidebar_state="expanded")
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
        st.error(f"No se encontró el archivo '{DATASET}'. Asegúrate de que esté en la misma raíz que app.py.")
        st.stop()
    df = pd.read_excel(path)
    return df

df = load_data()

# =========================================================
# ENTRENAMIENTO ROBUSTO DEL MODELO
# =========================================================
@st.cache_resource
def train_model(df_input):
    df_model = df_input.copy()
    le = LabelEncoder()
    df_model["Distrito_C"] = le.fit_transform(df_model["Distrito"])
    
    # Variables predictoras clave incluyendo la dimensión temporal
    features = ["Año", "Distrito_C", "Ingreso_Laboral", "Gasto_Alimentos", "Inflacion_Alimentaria", "Integrantes_Hogar", "Porcentaje_Gasto_Alimentos", "Indice_Vulnerabilidad"]
    X = df_model[features]
    y = df_model["Probabilidad_Enfermedad_Alimentaria"]
    
    # Ajustamos hiperparámetros para mejorar la sensibilidad a los cambios del año
    model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
    model.fit(X, y)
    return model, le, features

model, encoder, features = train_model(df)

# =========================================================
# PANEL DE CONTROL (BARRA LATERAL - AZUL PROFUNDO)
# =========================================================
st.sidebar.markdown('<div class="sidebar-title">PLATAFORMA IA</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-subtitle">SEGURIDAD ALIMENTARIA LIMA</div>', unsafe_allow_html=True)

year = st.sidebar.selectbox("Año de Proyección Futura", list(range(2026, 2036)), index=0)
search_district = st.sidebar.text_input("🔍 Buscar o filtrar distrito", "")

# =========================================================
# MOTOR DE INFERENCIA DINÁMICA (CORRECCIÓN DE CÁLCULO)
# =========================================================
# Obtenemos el último año histórico del dataset para calcular las proyecciones relativas
ultimo_anio_historico = int(df["Año"].max())
anios_de_proyeccion = year - ultimo_anio_historico

# Agrupamos la base histórica por distrito
base_districts = df.groupby("Distrito").agg({
    "Ingreso_Laboral": "mean",
    "Gasto_Alimentos": "mean",
    "Inflacion_Alimentaria": "mean",
    "Integrantes_Hogar": "mean",
    "Porcentaje_Gasto_Alimentos": "mean",
    "Indice_Vulnerabilidad": "mean"
}).reset_index()

# Simulamos variaciones realistas del entorno socioeconómico según pasan los años
# El paso del tiempo reduce el poder adquisitivo levemente e incrementa la inflación acumulada
base_districts["Año"] = year
base_districts["Ingreso_Laboral"] = base_districts["Ingreso_Laboral"] * (1 - (0.015 * anios_de_proyeccion))
base_districts["Inflacion_Alimentaria"] = base_districts["Inflacion_Alimentaria"] * (1 + (0.03 * anios_de_proyeccion))
base_districts["Porcentaje_Gasto_Alimentos"] = (base_districts["Gasto_Alimentos"] / base_districts["Ingreso_Laboral"]) * 100

# Transformación de categorías
base_districts["Distrito_C"] = encoder.transform(base_districts["Distrito"])

# Ejecución de la predicción en el Regresor
X_pred = base_districts[features]
base_districts["Probabilidad"] = model.predict(X_pred)
base_districts["Probabilidad"] = base_districts["Probabilidad"].clip(0, 100)

# Distribución correcta de Umbrales Semafóricos (Bajo, Medio, Alto)
def clasificar_umbrales(p):
    if p >= 60.0:
        return "ALTO", "#EF4444" # Rojo Pastel Corporativo
    elif p >= 40.0:
        return "MEDIO", "#F59E0B" # Ámbar
    else:
        return "BAJO", "#10B981" # Verde

umbrales = [clasificar_umbrales(p) for p in base_districts["Probabilidad"]]
base_districts["Nivel de Riesgo"] = [u[0] for u in umbrales]
base_districts["Color"] = [u[1] for u in umbrales]

sorted_results = base_districts.sort_values(by="Probabilidad", ascending=False).reset_index(drop=True)
sorted_results["#"] = sorted_results.index + 1

# Filtrado dinámico por la barra de búsqueda
if search_district.strip():
    filtered = sorted_results[sorted_results["Distrito"].str.contains(search_district, case=False, na=False)].copy()
else:
    filtered = sorted_results.copy()

top = sorted_results.iloc[0]

# =========================================================
# DISEÑO GRÁFICO DE LA PÁGINA CENTRAL
# =========================================================
st.markdown('<div class="main-title">Seguridad Alimentaria en Lima Metropolitana</div>', unsafe_allow_html=True)
st.markdown(f'<div class="main-subtitle">Monitoreo predictivo mediante algoritmos de Inteligencia Artificial para el año fiscal {year}</div>', unsafe_allow_html=True)

# Sección: Tarjetas de Resumen Ejecutivo Superior
col_m1, col_m2, col_m3 = st.columns(3)

with col_m1:
    st.markdown(f"""
    <div class="metric-card">
        <div style="color: #64748B; font-size: 13px; font-weight:600; text-transform: uppercase;">Distrito de Mayor Riesgo</div>
        <div style="color: #0F172A; font-size: 26px; font-weight: 800; margin-top: 5px;">{top["Distrito"]}</div>
    </div>
    """, unsafe_allow_html=True)

with col_m2:
    st.markdown(f"""
    <div class="metric-card">
        <div style="color: #64748B; font-size: 13px; font-weight:600; text-transform: uppercase;">Probabilidad Máxima Calculada</div>
        <div style="color: #EF4444; font-size: 26px; font-weight: 800; margin-top: 5px;">{top["Probabilidad"]:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col_m3:
    st.markdown(f"""
    <div class="metric-card">
        <div style="color: #64748B; font-size: 13px; font-weight:600; text-transform: uppercase;">Distribución de Alertas</div>
        <div style="color: #1E293B; font-size: 15px; font-weight: 700; margin-top: 8px;">
            <span style="color:#EF4444;">● Alto: {(base_districts["Nivel de Riesgo"]=="ALTO").sum()}</span> | 
            <span style="color:#F59E0B;">● Medio: {(base_districts["Nivel de Riesgo"]=="MEDIO").sum()}</span> | 
            <span style="color:#10B981;">● Bajo: {(base_districts["Nivel de Riesgo"]=="BAJO").sum()}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Bloque Central: Predicción Crítica Destacada
st.markdown(f"""
<div class="pred-card">
    <h2>REPORTE CRÍTICO DE PROYECCIÓN VULNERABLE ({year})</h2>
    <h3>{top["Distrito"]}</h3>
    <div class="pred-value">{top["Probabilidad"]:.1f}%</div>
    <p style="margin-top:12px; font-size:14px; opacity:0.85;">Este distrito requiere evaluación prioritaria en políticas de asistencia social alimentaria.</p>
</div>
""", unsafe_allow_html=True)

# Bloque: Tabla Principal de Resultados Estilizada con Badges HTML
st.markdown('<div class="table-card">', unsafe_allow_html=True)
st.markdown(f'<div class="table-title">Ranking de Probabilidad y Clasificación de Riesgo ({year})</div>', unsafe_allow_html=True)

def render_styled_table(row):
    return f'<span class="badge" style="background-color:{row["Color"]};">{row["Nivel de Riesgo"]}</span>'

table_data = filtered[["#", "Distrito", "Ingreso_Laboral", "Inflacion_Alimentaria", "Probabilidad"]].copy()
table_data["Ingreso_Laboral"] = table_data["Ingreso_Laboral"].apply(lambda v: f"S/. {v:.2f}")
table_data["Inflacion_Alimentaria"] = table_data["Inflacion_Alimentaria"].apply(lambda v: f"{v:.2f}%")
table_data["Probabilidad"] = table_data["Probabilidad"].apply(lambda v: f"<b>{v:.1f}%</b>")
table_data["Nivel de Riesgo"] = filtered.apply(render_styled_table, axis=1)

st.markdown(table_data.to_html(escape=False, index=False), unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Bloque: Caja Informativa de Interpretación
st.markdown(f"""
<div class="explanation">
    <b>Análisis de la simulación temporal:</b> Al proyectar el año <b>{year}</b>, el modelo evalúa las tendencias macroeconómicas acumuladas. Los distritos clasificados con riesgo <b>ALTO</b> reflejan una contracción en el poder de compra real combinada con una alta densidad de integrantes por familia, mientras que los distritos en riesgo <b>MEDIO</b> o <b>BAJO</b> presentan mayor resiliencia ante la inflación de la canasta básica.
</div>
""", unsafe_allow_html=True)

# Sección Inferior: Botones Profesionales de Exportación
st.markdown("<br>", unsafe_allow_html=True)
col_e1, col_e2 = st.columns(2)

with col_e1:
    excel_io = BytesIO()
    with pd.ExcelWriter(excel_io, engine="openpyxl") as writer:
        sorted_results[["#", "Distrito", "Año", "Ingreso_Laboral", "Inflacion_Alimentaria", "Probabilidad", "Nivel de Riesgo"]].to_excel(writer, index=False, sheet_name="Inferencia")
    st.download_button(
        label="📥 Exportar Data Consolidada (Excel)",
        data=excel_io.getvalue(),
        file_name=f"Reporte_Analitico_{year}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

with col_e2:
    if REPORTLAB_OK:
        pdf_io = BytesIO()
        doc = SimpleDocTemplate(pdf_io, pagesize=letter)
        styles = getSampleStyleSheet()
        story = [
            Paragraph(f"Reporte Institucional de Vulnerabilidad Alimentaria - {year}", styles["Title"]),
            Spacer(1, 20),
            Paragraph(f"Distrito Critico Identificado: {top['Distrito']} con un indice de riesgo de {top['Probabilidad']:.1f}%.", styles["BodyText"])
        ]
        doc.build(story)
        st.download_button(
            label="📄 Descargar Ficha Ejecutiva (PDF)",
            data=pdf_io.getvalue(),
            file_name=f"Ficha_Tecnica_{year}.pdf",
            mime="application/pdf"
        )
    else:
        st.button("📄 Reporte PDF (Librería ReportLab no detectada)", disabled=True)
