from io import BytesIO
from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import seaborn as sns

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

st.set_page_config(
    page_title="Inseguridad Alimentaria",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATASET = "Entregable_3_Dataset_Transformado_Inseguridad_Alimentaria.xlsx"

def load_css():
    """Cargar estilos CSS personalizados"""
    css_content = """
    <style>
    :root {
        --primary-color: #00492F;
        --secondary-color: #F97316;
        --success-color: #16A34A;
        --warning-color: #EAB308;
        --danger-color: #8B0000;
        --light-bg: #F5F5F5;
    }
    
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(245,245,245,0.95) 100%), 
                    url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 800"><defs><pattern id="pattern" patternUnits="userSpaceOnUse" width="100" height="100"><circle cx="50" cy="50" r="1" fill="%23E5E7EB"/></pattern></defs><rect width="1200" height="800" fill="url(%23pattern)"/></svg>');
        background-attachment: fixed;
        color: #333333;
    }
    
    .main-title-big {
        font-size: 2.8rem;
        font-weight: 800;
        color: var(--primary-color);
        margin: 10px 0;
        letter-spacing: -0.5px;
    }
    
    .main-title-small {
        font-size: 0.9rem;
        color: #666666;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }
    
    .location {
        font-size: 1rem;
        color: #999999;
        margin-top: 5px;
    }
    
    .control-panel {
        background: linear-gradient(135deg, #F9FAFB 0%, #F3F4F6 100%);
        padding: 25px;
        border-radius: 12px;
        margin: 25px 0;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    .label {
        font-size: 0.75rem;
        font-weight: 700;
        color: var(--primary-color);
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 8px;
        display: block;
    }
    
    .kpi-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F9FAFB 100%);
        border: 2px solid #E5E7EB;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        border-color: var(--primary-color);
    }
    
    .kpi-title {
        font-size: 0.85rem;
        color: #666666;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
    }
    
    .kpi-number {
        font-size: 2.5rem;
        font-weight: 800;
        color: var(--primary-color);
        line-height: 1;
    }
    
    .kpi-sub {
        font-size: 0.7rem;
        color: #999999;
        margin-top: 8px;
        font-weight: 500;
    }
    
    .result-card {
        background: #FFFFFF;
        border: 2px solid #E5E7EB;
        border-radius: 12px;
        padding: 24px;
        margin: 15px 0;
    }
    
    .card-title {
        font-size: 0.8rem;
        font-weight: 700;
        color: var(--primary-color);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 15px;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #F0F9FF 0%, #F5FAFB 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid var(--primary-color);
    }
    
    .low-box {
        background: linear-gradient(135deg, #FEF2F2 0%, #FAF5F5 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid var(--danger-color);
    }
    
    .pred-district {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--primary-color);
        margin: 10px 0;
    }
    
    .low-district {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--danger-color);
        margin: 10px 0;
    }
    
    .pred-label {
        font-size: 0.8rem;
        color: #666666;
        margin: 10px 0 5px 0;
        font-weight: 500;
    }
    
    .pred-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--primary-color);
        margin: 8px 0;
    }
    
    .low-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--danger-color);
        margin: 8px 0;
    }
    
    .year-badge {
        background-color: var(--primary-color);
        color: white;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .year-badge-red {
        background-color: var(--danger-color);
        color: white;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .progress {
        width: 100%;
        height: 8px;
        background-color: #E5E7EB;
        border-radius: 10px;
        overflow: hidden;
        margin-top: 15px;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--primary-color) 0%, #0D7A54 100%);
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    
    .progress-fill-red {
        height: 100%;
        background: linear-gradient(90deg, var(--danger-color) 0%, #B91C1C 100%);
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    
    .badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 600;
        color: white;
        text-align: center;
    }
    
    .table-card {
        background: #FFFFFF;
        border: 2px solid #E5E7EB;
        border-radius: 12px;
        padding: 20px;
        margin: 30px 0;
        overflow-x: auto;
        animation: slideInLeft 0.5s ease-in-out;
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .table-title {
        font-size: 1rem;
        font-weight: 700;
        color: var(--primary-color);
        margin-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .table-card table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
    }
    
    .table-card th {
        background-color: var(--primary-color);
        color: white;
        padding: 12px 10px;
        text-align: left;
        font-weight: 600;
        border: none;
        font-size: 0.8rem;
        letter-spacing: 0.3px;
    }
    
    .table-card td {
        padding: 10px 10px;
        border-bottom: 1px solid #E5E7EB;
        color: #333333;
        font-size: 0.85rem;
    }
    
    .table-card tr {
        transition: all 0.2s ease;
    }
    
    .table-card tr:hover {
        background-color: #F0F9FF;
        transform: scale(1.01);
    }
    
    .table-card tr:last-child td {
        border-bottom: none;
    }
    
    .explanation {
        background: linear-gradient(135deg, #F0F9FF 0%, #F5FAFB 100%);
        border-left: 4px solid var(--primary-color);
        padding: 20px;
        border-radius: 8px;
        margin: 30px 0;
        font-size: 0.95rem;
        line-height: 1.6;
        color: #333333;
    }
    
    .explanation b {
        color: var(--primary-color);
    }
    
    .footer {
        text-align: center;
        color: #999999;
        font-size: 0.85rem;
        padding: 30px 0;
        border-top: 1px solid #E5E7EB;
        margin-top: 50px;
    }
    
    .sidebar-title {
        font-size: 1.4rem;
        font-weight: 800;
        color: var(--primary-color);
        text-align: center;
        line-height: 1.3;
        margin-bottom: 5px;
    }
    
    .sidebar-subtitle {
        text-align: center;
        color: #666666;
        font-size: 0.8rem;
        margin-bottom: 20px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .side-item-active {
        background-color: var(--primary-color);
        color: white;
        padding: 10px;
        border-radius: 6px;
        margin: 8px 0;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .side-item {
        padding: 10px;
        color: #666666;
        margin: 8px 0;
        font-size: 0.9rem;
        cursor: pointer;
        border-radius: 6px;
        transition: all 0.3s ease;
    }
    
    .side-item:hover {
        background-color: #F3F4F6;
        color: var(--primary-color);
    }
    
    .side-divider {
        height: 1px;
        background-color: #E5E7EB;
        margin: 15px 0;
    }
    
    .side-footer {
        text-align: center;
        color: #999999;
        font-size: 0.75rem;
        margin-top: 30px;
        padding-top: 20px;
        border-top: 1px solid #E5E7EB;
        line-height: 1.5;
    }
    
    .stSelectbox, .stTextInput {
        margin-bottom: 0;
    }
    
    .stSelectbox [data-baseweb="select"] {
        background-color: #FFFFFF;
        border: 1px solid #D1D5DB;
        border-radius: 8px;
    }
    
    .stTextInput input {
        background-color: #FFFFFF;
        border: 1px solid #D1D5DB;
        border-radius: 8px;
        padding: 10px;
    }
    
    .stDownloadButton > button {
        background-color: var(--primary-color);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        background-color: #0D7A54;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 73, 47, 0.3);
    }
    
    .chart-card {
        background: #FFFFFF;
        border: 2px solid #E5E7EB;
        border-radius: 12px;
        padding: 20px;
        margin: 30px 0;
        animation: slideInRight 0.5s ease-in-out;
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .chart-title {
        font-size: 1rem;
        font-weight: 700;
        color: var(--primary-color);
        margin-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    @media (max-width: 768px) {
        .main-title-big {
            font-size: 2rem;
        }
        
        .kpi-number {
            font-size: 2rem;
        }
        
        .pred-district {
            font-size: 1.4rem;
        }
        
        .table-card {
            padding: 15px;
        }
        
        .table-card th,
        .table-card td {
            padding: 8px;
            font-size: 0.75rem;
        }
    }
    </style>
    """
    st.markdown(css_content, unsafe_allow_html=True)

load_css()

@st.cache_data
def load_data():
    path = Path(DATASET)
    if not path.exists():
        st.error(f"No se encontró el archivo {DATASET}. Debe estar en la misma carpeta que app.py.")
        st.stop()
    df = pd.read_excel(path)
    required = [
        "Distrito", "Año", "Ingreso_Laboral", "Gasto_Alimentos",
        "Inflacion_Alimentaria", "Integrantes_Hogar",
        "Porcentaje_Gasto_Alimentos", "Indice_Vulnerabilidad",
        "Probabilidad_Enfermedad_Alimentaria"
    ]
    missing = [col for col in required if col not in df.columns]
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
    """Clasifica el riesgo basado en IRIA"""
    if iria >= 75:
        return "Muy Alto"
    if iria >= 55:
        return "Alto"
    if iria >= 35:
        return "Medio"
    return "Bajo"

def interpretacion(nivel):
    interpretaciones = {
        "Muy Alto": "Probabilidad muy alta de presentar inseguridad alimentaria.",
        "Alto": "Probabilidad alta de presentar inseguridad alimentaria.",
        "Medio": "Probabilidad media de presentar inseguridad alimentaria.",
        "Bajo": "Probabilidad baja de presentar inseguridad alimentaria."
    }
    return interpretaciones.get(nivel, "")

def color_riesgo(nivel):
    colores = {
        "Muy Alto": "#8B0000",
        "Alto": "#F97316",
        "Medio": "#EAB308",
        "Bajo": "#16A34A"
    }
    return colores.get(nivel, "#16A34A")

def styled_badge(nivel):
    return f'<span class="badge" style="background-color:{color_riesgo(nivel)};">{nivel}</span>'

def calcular_iria(base):
    """Calcula IRIA y clasifica riesgo basado en IRIA"""
    base["IRIA"] = (
        0.30 * normalize_series(base["Porcentaje_Gasto_Alimentos"], 0, 100)
        + 0.25 * normalize_series(base["Indice_Vulnerabilidad"], 0, 100)
        + 0.20 * (100 - normalize_series(base["Ingreso_Laboral"], 0, 100))
        + 0.15 * normalize_series(base["Inflacion_Alimentaria"], 0, 100)
        + 0.10 * normalize_series(base["Gasto_Alimentos"], 0, 100)
    ).clip(0, 100).round(2)
    
    # Clasificar riesgo basado en IRIA
    base["Nivel de Riesgo"] = base["IRIA"].apply(clasificar_riesgo_iria)
    
    return base

@st.cache_resource
def train_models(df):
    work = df.copy()
    district_encoder = LabelEncoder()
    work["Distrito_Cod"] = district_encoder.fit_transform(work["Distrito"].astype(str))

    features = [
        "Año", "Distrito_Cod", "Ingreso_Laboral", "Gasto_Alimentos",
        "Inflacion_Alimentaria", "Integrantes_Hogar",
        "Porcentaje_Gasto_Alimentos", "Indice_Vulnerabilidad"
    ]

    y_reg = work["Probabilidad_Enfermedad_Alimentaria"]

    # Calcular IRIA y clasificación
    work = calcular_iria(work.copy())
    work["Nivel_Riesgo_Modelo"] = work["Nivel de Riesgo"].astype(str)

    class_encoder = LabelEncoder()
    y_clf = class_encoder.fit_transform(work["Nivel_Riesgo_Modelo"].astype(str))
    X = work[features]

    Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(X, y_reg, test_size=0.20, random_state=42)

    stratify_arg = y_clf if len(set(y_clf)) > 1 and pd.Series(y_clf).value_counts().min() >= 2 else None
    Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(X, y_clf, test_size=0.20, random_state=42, stratify=stratify_arg)

    regressor = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42)
    classifier = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42, class_weight="balanced")

    regressor.fit(Xtr_r, ytr_r)
    classifier.fit(Xtr_c, ytr_c)

    metrics = {
        "MAE": mean_absolute_error(yte_r, regressor.predict(Xte_r)),
        "R2": r2_score(yte_r, regressor.predict(Xte_r)),
        "Accuracy": accuracy_score(yte_c, classifier.predict(Xte_c))
    }
    return regressor, classifier, district_encoder, class_encoder, features, metrics

def project_by_year(df, year, regressor, classifier, district_encoder, class_encoder, features):
    base = df.groupby("Distrito").agg({
        "Ingreso_Laboral": "mean",
        "Gasto_Alimentos": "mean",
        "Inflacion_Alimentaria": "mean",
        "Integrantes_Hogar": "mean",
        "Porcentaje_Gasto_Alimentos": "mean",
        "Indice_Vulnerabilidad": "mean",
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
        base["Indice_Vulnerabilidad"] * (1 + 0.006 * years_forward) +
        base["Inflacion_Alimentaria"] * 0.25 +
        base["Porcentaje_Gasto_Alimentos"] * 2.5
    ).clip(0, 100)

    model_data = base.copy()
    model_data["Distrito_Cod"] = district_encoder.transform(model_data["Distrito"].astype(str))
    X_future = model_data[features]

    ml_pred = regressor.predict(X_future)

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

    # Calcular IRIA y clasificación basada en IRIA
    base = calcular_iria(base)

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
        Paragraph(
            f"Distrito con mayor riesgo: {top_high['Distrito']} - {top_high['Probabilidad']:.2f}%",
            styles["Normal"]
        ),
        Paragraph(
            f"Distrito con menor riesgo: {top_low['Distrito']} - {top_low['Probabilidad']:.2f}%",
            styles["Normal"]
        ),
        Spacer(1, 12)
    ]
    ranking = table[["#", "Distrito", "Probabilidad", "Nivel de Riesgo", "Interpretación"]].copy()
    ranking["Probabilidad"] = ranking["Probabilidad"].round(2)
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

def create_top10_chart(results):
    """Crea gráfica de Top 10 distritos con mayor riesgo"""
    top10 = results.nlargest(10, "IRIA")[["Distrito", "IRIA"]].reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    colors_bar = ["#8B0000" if x >= 75 else "#F97316" if x >= 55 else "#EAB308" if x >= 35 else "#16A34A" 
                  for x in top10["IRIA"]]
    
    bars = ax.barh(top10["Distrito"], top10["IRIA"], color=colors_bar, edgecolor="black", linewidth=1.2)
    
    # Añadir valores en las barras
    for i, (bar, val) in enumerate(zip(bars, top10["IRIA"])):
        ax.text(val + 1, i, f"{val:.1f}", va="center", fontweight="bold", fontsize=7)
    
    ax.set_xlabel("Índice IRIA", fontsize=9, fontweight="bold", color="#00492F")
    ax.set_ylabel("Distrito", fontsize=9, fontweight="bold", color="#00492F")
    ax.set_title("Top 10 Distritos - Mayor Riesgo", 
                 fontsize=10, fontweight="bold", color="#00492F", pad=12)
    ax.set_xlim(0, 105)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_facecolor("#F9FAFB")
    fig.patch.set_facecolor("white")
    
    plt.tight_layout()
    return fig

def create_risk_distribution_chart(results):
    """Crea gráfica de distribución de riesgos"""
    risk_counts = results["Nivel de Riesgo"].value_counts()
    colors_pie = {"Muy Alto": "#8B0000", "Alto": "#F97316", "Medio": "#EAB308", "Bajo": "#16A34A"}
    
    colors_list = [colors_pie.get(level, "#16A34A") for level in risk_counts.index]
    
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    wedges, texts, autotexts = ax.pie(
        risk_counts.values,
        labels=risk_counts.index,
        autopct="%1.1f%%",
        colors=colors_list,
        startangle=90,
        textprops={"fontsize": 9, "fontweight": "bold", "color": "white"},
        explode=[0.05 if x == "Muy Alto" else 0 for x in risk_counts.index]
    )
    
    # Mejorar textos
    for text in texts:
        text.set_color("#00492F")
        text.set_fontsize(9)
        text.set_fontweight("bold")
    
    ax.set_title("Distribución de Riesgos", 
                 fontsize=10, fontweight="bold", color="#00492F", pad=12)
    
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    return fig

# Cargar datos y entrenar modelos
df = load_data()
regressor, classifier, district_encoder, class_encoder, features, metrics = train_models(df)

# Sidebar
st.sidebar.markdown("""
<div class="sidebar-title">INSEGURIDAD<br>ALIMENTARIA</div>
<div class="sidebar-subtitle">LIMA METROPOLITANA</div>
<div class="side-item-active">Inicio</div>
<div class="side-divider"></div>
<div class="side-footer">Proyecto de Ciencia de Datos<br>Inseguridad Alimentaria en<br>Lima Metropolitana<br><br>© 2025</div>
""", unsafe_allow_html=True)

# Header
header_left, header_right = st.columns([2.15, 1])
with header_left:
    st.markdown("""
    <div class="main-title-small">PREDICCIÓN Y CLASIFICACIÓN DE</div>
    <div class="main-title-big">INSEGURIDAD ALIMENTARIA</div>
    <div class="location">Lima Metropolitana</div>
    """, unsafe_allow_html=True)

# Control Panel
st.markdown('<div class="control-panel">', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns([1.05, 1.25, 1.15, 1.35])

with c1:
    st.markdown('<div class="label">AÑO PARA LA PREDICCIÓN</div>', unsafe_allow_html=True)
    year = st.selectbox("", list(range(2024, 2036)), index=6, label_visibility="collapsed", key="year_select")

results = project_by_year(df, year, regressor, classifier, district_encoder, class_encoder, features)

with c2:
    st.markdown('<div class="label">SELECCIONAR DISTRITO</div>', unsafe_allow_html=True)
    district_selected = st.selectbox(
        "",
        ["Todos los distritos"] + sorted(results["Distrito"].unique()),
        label_visibility="collapsed",
        key="district_select"
    )

with c3:
    st.markdown('<div class="label">MOSTRAR SOLO RIESGO</div>', unsafe_allow_html=True)
    risk_filter = st.selectbox(
        "",
        ["Todos los niveles", "Muy Alto", "Alto", "Medio", "Bajo"],
        label_visibility="collapsed",
        key="risk_select"
    )

with c4:
    st.markdown('<div class="label">BUSCAR DISTRITO</div>', unsafe_allow_html=True)
    search = st.text_input("", placeholder="Buscar distrito...", label_visibility="collapsed", key="search_input")

st.markdown('</div>', unsafe_allow_html=True)

# Filtrar datos
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

# Export buttons
with header_right:
    export_cols = ["#", "Distrito", "Probabilidad", "Nivel de Riesgo", "Interpretación"]
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

# KPI Cards
st.markdown("<br>", unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-title">Riesgo muy alto</div>'
        f'<div class="kpi-number">{int(counts.get("Muy Alto", 0))}</div>'
        f'<div class="kpi-sub">Distritos</div></div>',
        unsafe_allow_html=True
    )
with k2:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-title">Riesgo alto</div>'
        f'<div class="kpi-number">{int(counts.get("Alto", 0))}</div>'
        f'<div class="kpi-sub">Distritos</div></div>',
        unsafe_allow_html=True
    )
with k3:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-title">Riesgo medio</div>'
        f'<div class="kpi-number">{int(counts.get("Medio", 0))}</div>'
        f'<div class="kpi-sub">Distritos</div></div>',
        unsafe_allow_html=True
    )
with k4:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-title">Riesgo bajo</div>'
        f'<div class="kpi-number">{int(counts.get("Bajo", 0))}</div>'
        f'<div class="kpi-sub">Distritos</div></div>',
        unsafe_allow_html=True
    )

# Result Cards
st.markdown("<br>", unsafe_allow_html=True)
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
            <div class="pred-label">Nivel de Riesgo: {top_high["Nivel de Riesgo"]}</div>
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
            <div class="pred-label">Nivel de Riesgo: {top_low["Nivel de Riesgo"]}</div>
            <div class="progress"><div class="progress-fill-red" style="width:{top_low["Probabilidad"]:.0f}%;"></div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with r3:
    st.markdown("""
    <div class="result-card">
        <div class="card-title">MODELOS UTILIZADOS</div>
        <div class="prediction-box">
            <div class="pred-label">Predicción de Probabilidad</div>
            <div class="pred-district" style="font-size: 1.3rem;">Random Forest Regressor</div>
            <br>
            <div class="pred-label">Clasificación de Riesgo</div>
            <div class="pred-district" style="font-size: 1.3rem;">Random Forest Classifier</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 📊 TABLE Y GRÁFICAS LADO A LADO
st.markdown("<br>", unsafe_allow_html=True)

table_col, charts_col = st.columns([1.4, 1])

# TABLA DINÁMICA
with table_col:
    st.markdown('<div class="table-card">', unsafe_allow_html=True)
    st.markdown(
        f'<div class="table-title">📋 RANKING DE DISTRITOS ({year})</div>',
        unsafe_allow_html=True
    )

    table_show = filtered[["#", "Distrito", "Probabilidad", "Nivel de Riesgo", "Interpretación"]].copy()
    table_show["Probabilidad"] = table_show["Probabilidad"].apply(lambda x: f"{x:.1f}%")
    table_show["Nivel de Riesgo"] = table_show["Nivel de Riesgo"].apply(styled_badge)
    
    # Usar HTML para mayor control
    html_table = table_show.to_html(escape=False, index=False, classes="ranking-table")
    st.markdown(html_table, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# GRÁFICAS DINÁMICAS
with charts_col:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="chart-title">📊 ANÁLISIS VISUAL</div>', unsafe_allow_html=True)
    
    # Las gráficas se actualizan automáticamente cuando cambia el año
    st.pyplot(create_top10_chart(results), use_container_width=True)
    st.pyplot(create_risk_distribution_chart(results), use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Explanation
st.markdown(f"""
<div class="explanation">
<b>Interpretación de resultados:</b><br>
Para el año <b>{year}</b>, el distrito con mayor probabilidad estimada de presentar inseguridad alimentaria es
<b>{top_high["Distrito"]}</b>, con <b>{top_high["Probabilidad"]:.1f}%</b> de probabilidad. 
El modelo clasifica <b>{int(counts.get("Muy Alto", 0))}</b> distritos en riesgo muy alto,
<b>{int(counts.get("Alto", 0))}</b> en riesgo alto,
<b>{int(counts.get("Medio", 0))}</b> en riesgo medio y
<b>{int(counts.get("Bajo", 0))}</b> en riesgo bajo.
El distrito con menor probabilidad es <b>{top_low["Distrito"]}</b>, con <b>{top_low["Probabilidad"]:.1f}%</b>.
<br><br>
<b>Nota:</b> El nivel de riesgo está determinado por el Índice IRIA (Índice de Riesgo de Inseguridad Alimentaria) que pondera factores económicos, de vulnerabilidad y alimentarios.
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown(
    '<div class="footer">Proyecto de Ciencia de Datos - Inseguridad Alimentaria en Lima Metropolitana © 2025</div>',
    unsafe_allow_html=True
)
