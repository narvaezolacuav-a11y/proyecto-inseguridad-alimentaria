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
    required = [
        "Distrito", "Año", "Ingreso_Laboral", "Gasto_Alimentos", "Inflacion_Alimentaria",
        "Integrantes_Hogar", "Porcentaje_Gasto_Alimentos", "Indice_Vulnerabilidad",
        "Probabilidad_Enfermedad_Alimentaria"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error("Faltan columnas obligatorias en el dataset: " + ", ".join(missing))
        st.stop()
    return df


def normalize_series(series, low=0, high=100):
    min_value = float(series.min())
    max_value = float(series.max())
    if max_value == min_value:
        return pd.Series([50] * len(series), index=series.index)
    return low + (series - min_value) * (high - low) / (max_value - min_value)


def clasificar_riesgo_iria(iria):
    if iria >= 75:
        return "Muy Alto"
    elif iria >= 55:
        return "Alto"
    elif iria >= 35:
        return "Medio"
    else:
        return "Bajo"


def interpretacion(nivel):
    if nivel == "Muy Alto":
        return "Probabilidad muy alta de presentar inseguridad alimentaria."
    elif nivel == "Alto":
        return "Probabilidad alta de presentar inseguridad alimentaria."
    elif nivel == "Medio":
        return "Probabilidad media de presentar inseguridad alimentaria."
    return "Probabilidad baja de presentar inseguridad alimentaria."


def color_riesgo(nivel):
    return {"Muy Alto": "#8B0000", "Alto": "#F97316", "Medio": "#EAB308", "Bajo": "#16A34A"}.get(nivel, "#16A34A")


def styled_badge(nivel):
    return f'<span class="badge" style="background:{color_riesgo(nivel)};">{nivel}</span>'


@st.cache_resource
def train_model(df):
    work = df.copy()
    encoder = LabelEncoder()
    work["Distrito_Cod"] = encoder.fit_transform(work["Distrito"].astype(str))
    features = [
        "Año", "Distrito_Cod", "Ingreso_Laboral", "Gasto_Alimentos", "Inflacion_Alimentaria",
        "Integrantes_Hogar", "Porcentaje_Gasto_Alimentos", "Indice_Vulnerabilidad"
    ]
    model = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
    model.fit(work[features], work["Probabilidad_Enfermedad_Alimentaria"])
    return model, encoder, features


def calcular_iria(base):
    base["IRIA"] = (
        0.30 * normalize_series(base["Porcentaje_Gasto_Alimentos"], 0, 100)
        + 0.25 * normalize_series(base["Indice_Vulnerabilidad"], 0, 100)
        + 0.20 * (100 - normalize_series(base["Ingreso_Laboral"], 0, 100))
        + 0.15 * normalize_series(base["Inflacion_Alimentaria"], 0, 100)
        + 0.10 * normalize_series(base["Gasto_Alimentos"], 0, 100)
    ).clip(0, 100).round(2)
    return base


def project_by_year(df, year, model, encoder, features):
    base = df.groupby("Distrito").agg({
        "Ingreso_Laboral": "mean", "Gasto_Alimentos": "mean", "Inflacion_Alimentaria": "mean",
        "Integrantes_Hogar": "mean", "Porcentaje_Gasto_Alimentos": "mean", "Indice_Vulnerabilidad": "mean",
    }).reset_index()

    years_forward = year - 2025
    district_code = pd.factorize(base["Distrito"])[0] + 1
    pressure = 1 + ((district_code % 7) - 3) * 0.010
    base["Año"] = year

    base["Inflacion_Alimentaria"] = (base["Inflacion_Alimentaria"] * (1 + 0.012 * years_forward) * pressure).clip(1, 15)
    base["Ingreso_Laboral"] = base["Ingreso_Laboral"] * (1 + 0.020 * years_forward) * (1 + ((district_code % 5) - 2) * 0.004)
    base["Gasto_Alimentos"] = base["Gasto_Alimentos"] * (1 + 0.024 * years_forward) * pressure
    base["Porcentaje_Gasto_Alimentos"] = (base["Gasto_Alimentos"] / base["Ingreso_Laboral"]).clip(0.08, 0.85)
    base["Indice_Vulnerabilidad"] = (
        base["Indice_Vulnerabilidad"] * (1 + 0.006 * years_forward)
        + base["Inflacion_Alimentaria"] * 0.25
        + base["Porcentaje_Gasto_Alimentos"] * 2.5
    ).clip(0, 100)

    model_data = base.copy()
    model_data["Distrito_Cod"] = encoder.transform(model_data["Distrito"].astype(str))
    ml_pred = model.predict(model_data[features])

    risk_score = (
        0.32 * normalize_series(base["Porcentaje_Gasto_Alimentos"], 0, 100)
        + 0.27 * normalize_series(base["Indice_Vulnerabilidad"], 0, 100)
        + 0.18 * normalize_series(base["Inflacion_Alimentaria"], 0, 100)
        + 0.13 * normalize_series(base["Gasto_Alimentos"], 0, 100)
        + 0.10 * (100 - normalize_series(base["Ingreso_Laboral"], 0, 100))
    )
    model_score = normalize_series(pd.Series(ml_pred, index=base.index), 0, 100)
    raw_probability = 0.55 * risk_score + 0.45 * model_score
    calibrated = normalize_series(raw_probability, 15, 85)
    year_shift = (year - 2024) * 0.35
    base["Probabilidad"] = (calibrated + year_shift).clip(5, 95).round(2)

    base = calcular_iria(base)
    base["Nivel de Riesgo"] = base["IRIA"].apply(clasificar_riesgo_iria)
    base["Interpretación"] = base["Nivel de Riesgo"].apply(interpretacion)
    base = base.sort_values("Probabilidad", ascending=False).reset_index(drop=True)
    base.insert(0, "#", range(1, len(base) + 1))
    return base


def make_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Ranking")
    return output.getvalue()


def make_pdf(year, top_high, top_low, table):
    if not REPORTLAB_OK:
        return None
    output = BytesIO()
    doc = SimpleDocTemplate(output, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = [
        Paragraph("Predicción y Clasificación de Inseguridad Alimentaria", styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"Año de predicción: {year}", styles["Heading2"]),
        Paragraph(f"Distrito con mayor riesgo: {top_high['Distrito']} - {top_high['Probabilidad']:.2f}%", styles["Normal"]),
        Paragraph(f"Distrito con menor riesgo: {top_low['Distrito']} - {top_low['Probabilidad']:.2f}%", styles["Normal"]),
        Spacer(1, 12)
    ]
    ranking = table[["#", "Distrito", "Probabilidad", "IRIA", "Nivel de Riesgo", "Interpretación"]].copy()
    ranking["Probabilidad"] = ranking["Probabilidad"].round(2)
    ranking["IRIA"] = ranking["IRIA"].round(2)
    pdf_table = Table([ranking.columns.tolist()] + ranking.values.tolist())
    pdf_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#00492F")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    elements.append(pdf_table)
    doc.build(elements)
    return output.getvalue()


df = load_data()
model, district_encoder, features = train_model(df)

st.sidebar.markdown("""
<div class="sidebar-title">INSEGURIDAD<br>ALIMENTARIA</div>
<div class="sidebar-subtitle">LIMA METROPOLITANA</div>
<div class="side-item-active">Inicio</div>
<div class="side-item">Predicción</div>
<div class="side-item">Clasificación</div>
<div class="side-item">Resultados</div>
<div class="side-divider"></div>
<div class="side-item">Exportar Ranking (Excel)</div>
<div class="side-item">Exportar PDF</div>
<div class="side-footer">Proyecto de Ciencia de Datos<br>Inseguridad Alimentaria en<br>Lima Metropolitana<br><br>© 2025</div>
""", unsafe_allow_html=True)

header_left, header_right = st.columns([2.15, 1])
with header_left:
    st.markdown("""
    <div class="main-title-small">PREDICCIÓN Y CLASIFICACIÓN DE</div>
    <div class="main-title-big">INSEGURIDAD ALIMENTARIA</div>
    <div class="location">Lima Metropolitana</div>
    """, unsafe_allow_html=True)

st.markdown('<div class="control-panel">', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns([1.05, 1.25, 1.15, 1.35])
with c1:
    st.markdown('<div class="label">AÑO PARA LA PREDICCIÓN</div>', unsafe_allow_html=True)
    year = st.selectbox("", list(range(2024, 2036)), index=6, label_visibility="collapsed")

results = project_by_year(df, year, model, district_encoder, features)

with c2:
    st.markdown('<div class="label">SELECCIONAR DISTRITO</div>', unsafe_allow_html=True)
    district_selected = st.selectbox("", ["Todos los distritos"] + sorted(results["Distrito"].unique()), label_visibility="collapsed")
with c3:
    st.markdown('<div class="label">MOSTRAR SOLO RIESGO</div>', unsafe_allow_html=True)
    risk_filter = st.selectbox("", ["Todos los niveles", "Muy Alto", "Alto", "Medio", "Bajo"], label_visibility="collapsed")
with c4:
    st.markdown('<div class="label">BUSCAR DISTRITO</div>', unsafe_allow_html=True)
    search = st.text_input("", placeholder="Buscar distrito...", label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

filtered = results.copy()
if district_selected != "Todos los distritos":
    filtered = filtered[filtered["Distrito"] == district_selected]
if search.strip():
    filtered = filtered[filtered["Distrito"].str.contains(search, case=False, na=False)]
if risk_filter != "Todos los niveles":
    filtered = filtered[filtered["Nivel de Riesgo"] == risk_filter]

top_high = results.iloc[0]
top_low = results.iloc[-1]
counts = results["Nivel de Riesgo"].value_counts()

with header_right:
    export_cols = ["#", "Distrito", "Probabilidad", "IRIA", "Nivel de Riesgo", "Interpretación"]
    st.download_button(
        "EXPORTAR RANKING (EXCEL)",
        data=make_excel(results[export_cols]),
        file_name=f"ranking_inseguridad_alimentaria_{year}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    pdf_data = make_pdf(year, top_high, top_low, results)
    if pdf_data:
        st.download_button(
            "EXPORTAR PDF",
            data=pdf_data,
            file_name=f"reporte_inseguridad_alimentaria_{year}.pdf",
            mime="application/pdf"
        )

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f'<div class="kpi-card"><div class="kpi-title">Distritos en riesgo muy alto</div><div class="kpi-number">{int(counts.get("Muy Alto", 0))}</div><div class="kpi-sub">Clasificación anual {year}</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="kpi-card"><div class="kpi-title">Distritos en riesgo alto</div><div class="kpi-number">{int(counts.get("Alto", 0))}</div><div class="kpi-sub">Clasificación anual {year}</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="kpi-card"><div class="kpi-title">Distritos en riesgo medio</div><div class="kpi-number">{int(counts.get("Medio", 0))}</div><div class="kpi-sub">Clasificación anual {year}</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="kpi-card"><div class="kpi-title">Distritos en riesgo bajo</div><div class="kpi-number">{int(counts.get("Bajo", 0))}</div><div class="kpi-sub">Clasificación anual {year}</div></div>', unsafe_allow_html=True)

r1, r2, r3 = st.columns([1, .9, 1.2])
with r1:
    st.markdown(f"""
    <div class="result-card">
        <div class="card-title">DISTRITO CON MAYOR PROBABILIDAD</div>
        <div>Año seleccionado: <span class="year-badge">{year}</span></div><br>
        <div class="prediction-box">
            <div class="pred-district">{top_high["Distrito"]}</div>
            <div class="pred-label">Probabilidad estimada de inseguridad alimentaria</div>
            <div class="pred-value">{top_high["Probabilidad"]:.1f} %</div>
            <div class="pred-label">IRIA: {top_high["IRIA"]:.1f} | Riesgo: {top_high["Nivel de Riesgo"]}</div>
            <div class="progress"><div class="progress-fill" style="width:{top_high["Probabilidad"]:.0f}%;"></div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
with r2:
    st.markdown(f"""
    <div class="result-card">
        <div class="card-title" style="color:#8B0000;">DISTRITO CON MENOR PROBABILIDAD</div>
        <div>Año seleccionado: <span class="year-badge-red">{year}</span></div><br>
        <div class="low-box">
            <div class="low-district">{top_low["Distrito"]}</div>
            <div class="pred-label">Probabilidad estimada de inseguridad alimentaria</div>
            <div class="low-value">{top_low["Probabilidad"]:.1f} %</div>
            <div class="pred-label">IRIA: {top_low["IRIA"]:.1f} | Riesgo: {top_low["Nivel de Riesgo"]}</div>
            <div class="progress"><div class="progress-fill-red" style="width:{top_low["Probabilidad"]:.0f}%;"></div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
with r3:
    st.markdown(f"""
    <div class="result-card">
        <div class="card-title">CLASIFICACIÓN DE NIVEL DE RIESGO ({year})</div>
        <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:14px;">
            <div class="risk-box" style="border:1px solid #F6A5A5; background:#FFF5F5;"><div class="risk-title" style="color:#8B0000;">MUY ALTO</div><div class="risk-range">75 - 100</div></div>
            <div class="risk-box" style="border:1px solid #FDBA74; background:#FFF7ED;"><div class="risk-title" style="color:#F97316;">ALTO</div><div class="risk-range">55 - 74</div></div>
            <div class="risk-box" style="border:1px solid #FDE68A; background:#FFFBEB;"><div class="risk-title" style="color:#EAB308;">MEDIO</div><div class="risk-range">35 - 54</div></div>
            <div class="risk-box" style="border:1px solid #86EFAC; background:#F0FDF4;"><div class="risk-title" style="color:#16A34A;">BAJO</div><div class="risk-range">0 - 34</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="table-card">', unsafe_allow_html=True)
st.markdown(f'<div class="table-title">TABLA DE TODOS LOS DISTRITOS - PROBABILIDAD, IRIA Y CLASIFICACIÓN ({year})</div>', unsafe_allow_html=True)
table_show = filtered[["#", "Distrito", "Probabilidad", "IRIA", "Nivel de Riesgo", "Interpretación"]].copy()
table_show["Probabilidad"] = table_show["Probabilidad"].round(1)
table_show["IRIA"] = table_show["IRIA"].round(1)
table_show["Nivel de Riesgo"] = table_show["Nivel de Riesgo"].apply(styled_badge)
st.markdown(table_show.to_html(escape=False, index=False), unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="explanation">
<b>Interpretación automática:</b><br>
Para el año <b>{year}</b>, el distrito con mayor probabilidad estimada de presentar inseguridad alimentaria es
<b>{top_high["Distrito"]}</b>, con <b>{top_high["Probabilidad"]:.1f}%</b> y un IRIA de <b>{top_high["IRIA"]:.1f}</b>.
El distrito con menor probabilidad es <b>{top_low["Distrito"]}</b>, con <b>{top_low["Probabilidad"]:.1f}%</b> y un IRIA de <b>{top_low["IRIA"]:.1f}</b>.
La clasificación anual muestra <b>{int(counts.get("Muy Alto", 0))}</b> distritos en riesgo muy alto,
<b>{int(counts.get("Alto", 0))}</b> en riesgo alto,
<b>{int(counts.get("Medio", 0))}</b> en riesgo medio y
<b>{int(counts.get("Bajo", 0))}</b> en riesgo bajo.
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="footer">Proyecto de Ciencia de Datos - Inseguridad Alimentaria en Lima Metropolitana © 2025</div>', unsafe_allow_html=True)

