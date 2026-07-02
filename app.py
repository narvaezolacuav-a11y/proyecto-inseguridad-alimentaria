
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

st.set_page_config(
    page_title="Inseguridad Alimentaria",
    page_icon="🍽️",
    layout="wide"
)

# =========================
# ESTILOS Y ANIMACIONES
# =========================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #F8FFF4 0%, #E9F7EF 40%, #FFF8E7 100%);
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #1B5E20;
    text-align: center;
    padding: 18px;
    animation: fadeInDown 1s ease-in-out;
}

.sub-title {
    text-align: center;
    color: #4E5D4E;
    font-size: 18px;
    margin-bottom: 25px;
    animation: fadeIn 1.5s ease-in-out;
}

.card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    border-left: 7px solid #43A047;
    animation: fadeInUp 0.8s ease-in-out;
}

.result-card {
    background: linear-gradient(135deg, #2E7D32, #66BB6A);
    color: white;
    padding: 26px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 8px 24px rgba(0,0,0,0.18);
    animation: pulseSoft 2s infinite;
}

.result-card h2 {
    color: white;
    font-size: 28px;
}

.result-card h3 {
    color: white;
    font-size: 38px;
}

.info-box {
    background: #FFFDE7;
    border-left: 6px solid #FBC02D;
    padding: 18px;
    border-radius: 12px;
    color: #4E342E;
    animation: fadeIn 1.2s ease-in-out;
}

.section-title {
    color: #1B5E20;
    font-weight: 800;
    font-size: 26px;
    margin-top: 22px;
    margin-bottom: 12px;
}

[data-testid="stMetricValue"] {
    color: #1B5E20;
    font-weight: 800;
}

div.stButton > button {
    background: linear-gradient(90deg, #43A047, #FBC02D);
    color: white;
    border: none;
    border-radius: 15px;
    padding: 12px 25px;
    font-weight: 700;
    transition: 0.3s;
}

div.stButton > button:hover {
    transform: scale(1.04);
    box-shadow: 0 6px 18px rgba(0,0,0,0.25);
}

@keyframes fadeInDown {
    from {opacity: 0; transform: translateY(-25px);}
    to {opacity: 1; transform: translateY(0);}
}

@keyframes fadeInUp {
    from {opacity: 0; transform: translateY(25px);}
    to {opacity: 1; transform: translateY(0);}
}

@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}

@keyframes pulseSoft {
    0% {transform: scale(1);}
    50% {transform: scale(1.015);}
    100% {transform: scale(1);}
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🍽️ Sistema Predictivo de Inseguridad Alimentaria</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predicción y clasificación de riesgo alimentario en distritos de Lima Metropolitana</div>', unsafe_allow_html=True)


# =========================
# CARGA DE DATOS
# =========================

@st.cache_data
def cargar_datos():
    return pd.read_excel("Entregable_3_Dataset_Transformado_Inseguridad_Alimentaria.xlsx")

df = cargar_datos()


# =========================
# ENTRENAMIENTO
# =========================

@st.cache_resource
def entrenar_modelos(df):
    df_model = df.copy()

    encoders = {}
    columnas_cat = [
        "Distrito", "Zona", "Tipo_Empleo", "Nivel_Educativo",
        "Programa_Social", "Acceso_Agua", "Acceso_Desague",
        "Estado_Nutricional", "Nivel_Riesgo"
    ]

    for col in columnas_cat:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        encoders[col] = le

    variables = [
        "Año",
        "Distrito",
        "Ingreso_Laboral",
        "Gasto_Alimentos",
        "Inflacion_Alimentaria",
        "Integrantes_Hogar",
        "Porcentaje_Gasto_Alimentos",
        "Indice_Vulnerabilidad"
    ]

    X = df_model[variables]
    y_pred = df_model["Probabilidad_Enfermedad_Alimentaria"]
    y_class = df_model["Nivel_Riesgo"]

    X_train, X_test, y_train_pred, y_test_pred = train_test_split(
        X, y_pred, test_size=0.2, random_state=42
    )

    _, _, y_train_class, y_test_class = train_test_split(
        X, y_class, test_size=0.2, random_state=42
    )

    modelo_pred = RandomForestRegressor(n_estimators=200, random_state=42)
    modelo_class = RandomForestClassifier(n_estimators=200, random_state=42)

    modelo_pred.fit(X_train, y_train_pred)
    modelo_class.fit(X_train, y_train_class)

    pred_test = modelo_pred.predict(X_test)
    class_test = modelo_class.predict(X_test)

    mae = mean_absolute_error(y_test_pred, pred_test)
    r2 = r2_score(y_test_pred, pred_test)
    acc = accuracy_score(y_test_class, class_test)

    return modelo_pred, modelo_class, encoders, variables, mae, r2, acc


modelo_pred, modelo_class, encoders, variables, mae, r2, acc = entrenar_modelos(df)


# =========================
# PANEL LATERAL
# =========================

st.sidebar.title("⚙️ Panel de control")
anio = st.sidebar.selectbox("Selecciona el año", list(range(2026, 2036)))
st.sidebar.info("El sistema predice la probabilidad por distrito y clasifica el nivel de riesgo.")


# =========================
# MÉTRICAS
# =========================

st.markdown('<div class="section-title">📌 Resultados del entrenamiento</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
c1.metric("Error MAE", f"{mae:.2f}")
c2.metric("R²", f"{r2:.2f}")
c3.metric("Accuracy", f"{acc*100:.1f}%")


# =========================
# PREDICCIÓN FUTURA
# =========================

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
base_model = base.copy()
base_model["Distrito_Texto"] = base_model["Distrito"]
base_model["Distrito"] = encoders["Distrito"].transform(base_model["Distrito"].astype(str))

X_futuro = base_model[variables]

base["Probabilidad_Predicha"] = modelo_pred.predict(X_futuro)
base["Nivel_Riesgo_Codificado"] = modelo_class.predict(X_futuro)
base["Nivel_Riesgo"] = encoders["Nivel_Riesgo"].inverse_transform(base["Nivel_Riesgo_Codificado"].astype(int))

base = base.sort_values("Probabilidad_Predicha", ascending=False)
top = base.iloc[0]

st.markdown('<div class="section-title">🔮 Predicción principal</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="result-card">
    <h2>Año seleccionado: {anio}</h2>
    <p>Distrito con mayor probabilidad de padecer enfermedades alimentarias</p>
    <h3>{top["Distrito"]}</h3>
    <h2>{top["Probabilidad_Predicha"]:.2f}%</h2>
    <p>Nivel de riesgo: <b>{top["Nivel_Riesgo"]}</b></p>
</div>
""", unsafe_allow_html=True)


# =========================
# TABLA
# =========================

st.markdown('<div class="section-title">📊 Clasificación de distritos</div>', unsafe_allow_html=True)

tabla = base[[
    "Distrito",
    "Año",
    "Ingreso_Laboral",
    "Gasto_Alimentos",
    "Probabilidad_Predicha",
    "Nivel_Riesgo"
]].copy()

tabla["Ingreso_Laboral"] = tabla["Ingreso_Laboral"].round(2)
tabla["Gasto_Alimentos"] = tabla["Gasto_Alimentos"].round(2)
tabla["Probabilidad_Predicha"] = tabla["Probabilidad_Predicha"].round(2)

st.dataframe(tabla, use_container_width=True)


# =========================
# GRÁFICO
# =========================

st.markdown('<div class="section-title">📈 Ranking de probabilidad</div>', unsafe_allow_html=True)

top10 = tabla.head(10)

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(top10["Distrito"], top10["Probabilidad_Predicha"])
ax.set_title(f"Top 10 distritos con mayor probabilidad - {anio}")
ax.set_xlabel("Distrito")
ax.set_ylabel("Probabilidad (%)")
ax.tick_params(axis="x", rotation=75)
ax.grid(axis="y")

st.pyplot(fig)


# =========================
# CONSULTA INDIVIDUAL
# =========================

st.markdown('<div class="section-title">🔎 Consulta individual</div>', unsafe_allow_html=True)

distrito = st.selectbox("Selecciona un distrito", sorted(tabla["Distrito"].unique()))
fila = tabla[tabla["Distrito"] == distrito].iloc[0]

a, b, c = st.columns(3)
a.metric("Distrito", distrito)
b.metric("Probabilidad", f"{fila['Probabilidad_Predicha']:.2f}%")
c.metric("Clasificación", fila["Nivel_Riesgo"])


# =========================
# DESCARGA
# =========================

st.markdown('<div class="section-title">⬇️ Descargar resultados</div>', unsafe_allow_html=True)

csv = tabla.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Descargar resultados CSV",
    data=csv,
    file_name=f"resultados_prediccion_{anio}.csv",
    mime="text/csv"
)


# =========================
# INTERPRETACIÓN
# =========================

st.markdown(f"""
<div class="info-box">
<b>Interpretación:</b><br>
Para el año <b>{anio}</b>, el modelo identifica a <b>{top["Distrito"]}</b> como el distrito con mayor probabilidad estimada.
Este resultado se basa en variables como ingreso laboral, gasto en alimentos, inflación alimentaria e índice de vulnerabilidad.
</div>
""", unsafe_allow_html=True)
