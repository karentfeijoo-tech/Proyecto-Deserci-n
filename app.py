import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.set_page_config(page_title="Predicción de Deserción", layout="centered")
st.title("🎓 Predicción de Deserción Estudiantil")

# =====================
# CARGA Y PREPROCESO
# =====================
@st.cache_data
def cargar_datos():
    ruta = "REPORTE_RECORD_ESTUDIANTIL_ANONIMIZADO.xlsx"
    df = pd.read_excel(ruta)

    df['PROMEDIO'] = df['PROMEDIO'].astype(str).str.replace(',', '.').astype(float)

    df_agg = df.groupby(['ESTUDIANTE', 'PERIODO']).agg(
        prom_global=('PROMEDIO', 'mean'),
        repeticiones=('NO. VEZ', lambda x: (x > 1).sum()),
        carga_academica=('COD_MATERIA', 'count')
    ).reset_index()

    df_agg['PERIODO_ORD'] = df_agg['PERIODO'].str.replace(' ', '')
    df_agg = df_agg.sort_values(['ESTUDIANTE', 'PERIODO_ORD'])

    df_agg['DESERTA'] = (
        df_agg.groupby('ESTUDIANTE')['PERIODO_ORD']
        .shift(-1)
        .isna()
        .astype(int)
    )

    ultimo_periodo = df_agg['PERIODO_ORD'].max()
    df_agg = df_agg[df_agg['PERIODO_ORD'] != ultimo_periodo]

    return df_agg


# =====================
# EJECUCIÓN
# =====================
try:
    df = cargar_datos()

    if df.empty:
        st.warning("⚠️ El procesamiento dejó los datos vacíos.")
        st.stop()

    # =====================
    # MODELO
    # =====================
    X = df[['prom_global', 'repeticiones', 'carga_academica']]
    y = df['DESERTA']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )

    modelo = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    modelo.fit(X_train, y_train)

    # =====================
    # MÉTRICAS
    # =====================
    st.subheader("📊 Calidad del Modelo")

    y_pred = modelo.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{report['accuracy']:.2f}")
    c2.metric("Recall (Deserción)", f"{report['1']['recall']:.2f}")
    c3.metric("F1-score", f"{report['1']['f1-score']:.2f}")

    # =====================
    # FORMULARIO DE PREDICCIÓN
    # =====================
    st.markdown("---")
    st.subheader("🧑‍🎓 Evaluar nuevo estudiante")

    promedio_in = st.number_input("Promedio académico", 0.0, 10.0, 7.5)
    repeticiones_in = st.number_input("Número de materias repetidas", 0, 10, 0)
    carga_in = st.number_input("Número de materias inscritas", 1, 15, 5)

    if st.button("🚀 Calcular Riesgo"):
        nuevo = np.array([[promedio_in, repeticiones_in, carga_in]])
        nuevo_scaled = scaler.transform(nuevo)
        prob = modelo.predict_proba(nuevo_scaled)[0][1]

        st.markdown(f"### 🔮 Probabilidad de deserción: **{prob:.2%}**")

        if prob >= 0.7:
            st.error("🔴 **ALTO RIESGO:** Intervención académica urgente.")
        elif prob >= 0.4:
            st.warning("🟡 **RIESGO MODERADO:** Requiere seguimiento.")
        else:
            st.success("🟢 **BAJO RIESGO:** Perfil de permanencia.")

    # =====================
    # IMPORTANCIA DE VARIABLES
    # =====================
    with st.expander("📈 Ver importancia de variables"):
        coef_df = pd.DataFrame({
            'Variable': X.columns,
            'Coeficiente': modelo.coef_[0]
        }).sort_values(by='Coeficiente')
        st.bar_chart(coef_df.set_index('Variable'))

except Exception as e:
    st.error(f"❌ Error inesperado: {e}")
