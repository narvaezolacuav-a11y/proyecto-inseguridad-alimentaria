from io import BytesIO
from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

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
        "Porcentaje_Gasto_Alimentos"
