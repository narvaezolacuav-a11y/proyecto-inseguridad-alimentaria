
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
        st.error(f"No se encontró el archivo {DATASET}. Debe estar en la misma carpeta que app.py.")
        st.stop()
    df = pd.read_excel(path)
    required = ["Distrito","Año","Ingreso_Laboral","Gasto_Alimentos","Inflacion_Alimentaria","Integrantes_Hogar","Porcentaje_Gasto_Alimentos","Indice_Vulnerabilidad","Probabilidad_Enfermedad_Alimentaria"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error("Faltan columnas obligatorias en el dataset: " + ", ".join(missing))
        st.stop()
    return df

def nivel_riesgo(prob):
    if prob >= 75: return "Muy Alto"
    if prob >= 50: return "Alto"
    if prob >= 25: return "Medio"
    return "Bajo"

def interpretacion(nivel):
    if nivel == "Muy Alto": return "Probabilidad muy alta de presentar inseguridad alimentaria."
    if nivel == "Alto": return "Probabilidad alta de presentar inseguridad alimentaria."
    if nivel == "Medio": return "Probabilidad media de presentar inseguridad alimentaria."
    return "Probabilidad baja de presentar inseguridad alimentaria."

def color_riesgo(nivel):
    return {"Muy Alto":"#B00000","Alto":"#F97316","Medio":"#EAB308","Bajo":"#16A34A"}.get(nivel,"#16A34A")

@st.cache_resource
def train_model(df):
    work = df.copy()
    le = LabelEncoder()
    work["Distrito_Cod"] = le.fit_transform(work["Distrito"].astype(str))
    features = ["Año","Distrito_Cod","Ingreso_Laboral","Gasto_Alimentos","Inflacion_Alimentaria","Integrantes_Hogar","Porcentaje_Gasto_Alimentos","Indice_Vulnerabilidad"]
    model = RandomForestRegressor(n_estimators=300, max_depth=9, random_state=42)
    model.fit(work[features], work["Probabilidad_Enfermedad_Alimentaria"])
    return model, le, features

def project_by_year(df, year, model, le, features):
    base = df.groupby("Distrito").agg({
        "Ingreso_Laboral":"mean",
        "Gasto_Alimentos":"mean",
        "Inflacion_Alimentaria":"mean",
        "Integrantes_Hogar":"mean",
        "Porcentaje_Gasto_Alimentos":"mean",
        "Indice_Vulnerabilidad":"mean",
    }).reset_index()
    years_forward = year - 2025
    district_code = pd.factorize(base["Distrito"])[0] + 1
    pressure = 1 + ((district_code % 7) - 3) * 0.012
    base["Año"] = year
    base["Inflacion_Alimentaria"] = (base["Inflacion_Alimentaria"] * (1 + 0.020 * years_forward) * pressure).clip(1,18)
    base["Ingreso_Laboral"] = base["Ingreso_Laboral"] * (1 + 0.018 * years_forward) * (1 + ((district_code % 5) - 2) * 0.006)
    base["Gasto_Alimentos"] = base["Gasto_Alimentos"] * (1 + 0.030 * years_forward) * pressure
    base["Porcentaje_Gasto_Alimentos"] = (base["Gasto_Alimentos"] / base["Ingreso_Laboral"]).clip(0.08,0.95)
    base["Indice_Vulnerabilidad"] = (base["Indice_Vulnerabilidad"] * (1 + 0.013 * years_forward) + base["Inflacion_Alimentaria"] * 0.45 + base["Porcentaje_Gasto_Alimentos"] * 5).clip(0,100)
    model_data = base.copy()
    model_data["Distrito_Cod"] = le.transform(model_data["Distrito"].astype(str))
    pred = model.predict(model_data[features])
    temporal_adjustment = years_forward * 1.15 + (base["Inflacion_Alimentaria"] - df["Inflacion_Alimentaria"].mean()) * 0.70 + (base["Porcentaje_Gasto_Alimentos"] - df["Porcentaje_Gasto_Alimentos"].mean()) * 13
    base["Probabilidad"] = (pred + temporal_adjustment).clip(0,100)
    base["Nivel de Riesgo"] = base["Probabilidad"].apply(nivel_riesgo)
    base["Interpretación"] = base["Nivel de Riesgo"].apply(interpretacion)
    base = base.sort_values("Probabilidad", ascending=False).reset_index(drop=True)
    base.insert(0, "#", range(1, len(base)+1))
    return base

def styled_badge(nivel):
    return f'<span class="badge" style="background:{color_riesgo(nivel)};">{nivel}</span>'

def make_excel(df):
    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Ranking")
    return out.getvalue()

def make_pdf(year, top, table):
    if not REPORTLAB_OK: return None
    out = BytesIO()
    doc = SimpleDocTemplate(out, pagesize=letter)
    styles = getSampleStyleSheet()
    elems = [
        Paragraph("Predicción y Clasificación de Inseguridad Alimentaria", styles["Title"]),
        Spacer(1,12),
        Paragraph(f"Año de predicción: {year}", styles["Heading2"]),
        Paragraph(f"Distrito con mayor probabilidad: {top['Distrito']}", styles["Normal"]),
        Paragraph(f"Probabilidad estimada: {top['Probabilidad']:.2f}%", styles["Normal"]),
        Paragraph(f"Nivel de riesgo: {top['Nivel de Riesgo']}", styles["Normal"]),
        Spacer(1,12)
    ]
    ranking = table[["#","Distrito","Probabilidad","Nivel de Riesgo","Interpretación"]].head(10).copy()
    ranking["Probabilidad"] = ranking["Probabilidad"].round(2)
    t = Table([ranking.columns.tolist()] + ranking.values.tolist())
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#00492F")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),0.5,colors.grey),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold")]))
    elems.append(t)
    doc.build(elems)
    return out.getvalue()

df = load_data()
model, district_encoder, features = train_model(df)

st.sidebar.markdown("""
<div class="sidebar-title">INSEGURIDAD<br>ALIMENTARIA</div>
<div class="sidebar-subtitle">LIMA METROPOLITANA</div>
<div class="side-item-active">Inicio</div>
<div class="side-item">Predicción</div>
<div class="side-item">Clasificación</div>
<div class="side-item">Resultados</div>
<div class="side-item">Exportar Ranking (Excel)</div>
<div class="side-item">Exportar PDF</div>
<div class="side-footer">Proyecto de Ciencia de Datos<br>Inseguridad Alimentaria en<br>Lima Metropolitana<br><br>© 2025</div>
""", unsafe_allow_html=True)

head_left, head_right = st.columns([2.1,1])
with head_left:
    st.markdown('<div class="main-title-small">PREDICCIÓN Y CLASIFICACIÓN DE</div><div class="main-title-big">INSEGURIDAD ALIMENTARIA</div><div class="location">Lima Metropolitana</div>', unsafe_allow_html=True)

st.markdown('<div class="control-panel">', unsafe_allow_html=True)
c1,c2,c3,c4 = st.columns([1.1,1.25,1.15,1.35])
with c1:
    st.markdown('<div class="label">AÑO PARA LA PREDICCIÓN</div>', unsafe_allow_html=True)
    year = st.selectbox("", list(range(2026,2036)), index=4, label_visibility="collapsed")
results = project_by_year(df, year, model, district_encoder, features)
with c2:
    st.markdown('<div class="label">BUSCAR DISTRITO</div>', unsafe_allow_html=True)
    search = st.text_input("", placeholder="Buscar distrito...", label_visibility="collapsed")
with c3:
    st.markdown('<div class="label">MOSTRAR SOLO RIESGO</div>', unsafe_allow_html=True)
    risk_filter = st.selectbox("", ["Todos los niveles","Muy Alto","Alto","Medio","Bajo"], label_visibility="collapsed")
with c4:
    st.markdown('<div class="label">&nbsp;</div>', unsafe_allow_html=True)
    st.button("CLASIFICAR DISTRITOS")
st.markdown('</div>', unsafe_allow_html=True)

filtered = results.copy()
if search.strip():
    filtered = filtered[filtered["Distrito"].str.contains(search, case=False, na=False)]
if risk_filter != "Todos los niveles":
    filtered = filtered[filtered["Nivel de Riesgo"] == risk_filter]
top = results.iloc[0]

with head_right:
    st.download_button("EXPORTAR RANKING (EXCEL)", data=make_excel(results), file_name=f"ranking_inseguridad_alimentaria_{year}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    pdf_data = make_pdf(year, top, results)
    if pdf_data:
        st.download_button("EXPORTAR PDF", data=pdf_data, file_name=f"reporte_inseguridad_alimentaria_{year}.pdf", mime="application/pdf")

r1,r2 = st.columns([1,1])
with r1:
    st.markdown(f"""
    <div class="result-card">
      <div class="card-title">DISTRITO CON MAYOR PROBABILIDAD</div>
      <div>Año seleccionado: <span class="year-badge">{year}</span></div><br>
      <div class="prediction-box">
        <div class="pred-district">{top["Distrito"]}</div>
        <div class="pred-label">Probabilidad estimada de inseguridad alimentaria</div>
        <div class="pred-value">{top["Probabilidad"]:.1f} %</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
with r2:
    st.markdown(f"""
    <div class="result-card">
      <div class="card-title">CLASIFICACIÓN DE NIVEL DE RIESGO ({year})</div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px;">
        <div class="risk-box" style="border:1px solid #F6A5A5;background:#FFF5F5;"><div class="risk-title" style="color:#B00000;">MUY ALTO</div><div class="risk-range">≥ 75%</div></div>
        <div class="risk-box" style="border:1px solid #FDBA74;background:#FFF7ED;"><div class="risk-title" style="color:#F97316;">ALTO</div><div class="risk-range">50% - 74%</div></div>
        <div class="risk-box" style="border:1px solid #FDE68A;background:#FFFBEB;"><div class="risk-title" style="color:#EAB308;">MEDIO</div><div class="risk-range">25% - 49%</div></div>
        <div class="risk-box" style="border:1px solid #86EFAC;background:#F0FDF4;"><div class="risk-title" style="color:#16A34A;">BAJO</div><div class="risk-range">&lt; 25%</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="table-card">', unsafe_allow_html=True)
st.markdown(f'<div class="table-title">RANKING DE DISTRITOS - NIVEL DE RIESGO ({year})</div>', unsafe_allow_html=True)
table_show = filtered[["#","Distrito","Probabilidad","Nivel de Riesgo","Interpretación"]].copy()
table_show["Probabilidad"] = table_show["Probabilidad"].round(1)
table_show["Nivel de Riesgo"] = table_show["Nivel de Riesgo"].apply(styled_badge)
st.markdown(table_show.to_html(escape=False, index=False), unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="explanation">
<b>Interpretación automática:</b><br>
Para el año <b>{year}</b>, el distrito con mayor probabilidad de presentar inseguridad alimentaria es
<b>{top["Distrito"]}</b>, con una probabilidad estimada de <b>{top["Probabilidad"]:.1f}%</b>.
El resultado cambia por año porque el aplicativo proyecta ingreso laboral, gasto en alimentos,
inflación alimentaria e índice de vulnerabilidad antes de calcular la probabilidad.
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="footer">Proyecto de Ciencia de Datos - Inseguridad Alimentaria en Lima Metropolitana © 2025</div>', unsafe_allow_html=True)
