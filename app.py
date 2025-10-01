import streamlit as st
import joblib
import numpy as np
from pathlib import Path

# Configuración de la página
st.set_page_config(page_title="Modelo IA para Predicción Cardiaca", layout="centered")

# Título y subtítulo
st.title("Modelo IA para Predicción Cardiaca")
st.subheader("Elaborado por Jose Alejandro ®  UNAB 2025")

st.write("---")

# Rutas a los archivos (deben estar en el mismo directorio que app.py)
MODEL_PATH = Path("svc_model.jb")
SCALER_PATH = Path("scaler.jb")

# Cargar modelos con cache para evitar recargas innecesarias
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        return None

@st.cache_resource
def load_scaler(path):
    try:
        return joblib.load(path)
    except Exception as e:
        return None

model = load_model(MODEL_PATH)
scaler = load_scaler(SCALER_PATH)

if model is None:
    st.error("No se pudo cargar el archivo 'svc_model.jb'. Asegúrate de que esté en el mismo directorio que app.py.")

if scaler is None:
    st.error("No se pudo cargar el archivo 'scaler.jb'. Asegúrate de que esté en el mismo directorio que app.py.")

st.markdown("### Ingrese los datos del paciente")

# Sliders según la especificación
edad = st.slider("Edad (años)", min_value=25, max_value=80, value=55, step=1)
colesterol = st.slider("Colesterol (mg/dL)", min_value=120, max_value=600, value=242, step=2)

st.markdown("---")

# Botón para predecir
if st.button("Predecir"):
    if model is None or scaler is None:
        st.error("El modelo o el scaler no están disponibles. Revisa los archivos en el directorio.")
    else:
        # Preparar los datos en el mismo orden que se entrenó: ['edad', 'colesterol']
        X_raw = np.array([[edad, colesterol]], dtype=float)
        try:
            X_scaled = scaler.transform(X_raw)
        except Exception as e:
            st.error(f"Error al aplicar el scaler: {e}")
            st.stop()

        # Predecir
        try:
            pred = model.predict(X_scaled)
            pred_val = int(pred[0])
        except Exception as e:
            st.error(f"Error al predecir con el modelo: {e}")
            st.stop()

        # Mostrar resultado
        if pred_val == 0:
            st.success("Resultado: 0 — El paciente NO sufrirá del corazón según el modelo.")
            st.image("https://img.freepik.com/vector-gratis/ilustracion-personas-corazon-sano_53876-37150.jpg", caption="Corazón sano")
        else:
            st.error("Resultado: 1 — El paciente SUFRIRÁ del corazón según el modelo.")
            st.image("https://as01.epimg.net/deporteyvida/imagenes/2017/10/28/portada/1509177885_209365_1509178036_noticia_normal.jpg", caption="Problema cardiaco")

        # Si el modelo tiene predict_proba mostrar probabilidad
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_scaled)[0]
                st.write(f"Probabilidad no sufrir (0): {proba[0]:.3f}")
                st.write(f"Probabilidad sufrir (1): {proba[1]:.3f}")
            except Exception:
                # Algunos SVC no tienen predict_proba si no se entrenaron con probability=True
                pass

        # Mostrar valores escalados opcionalmente
        st.markdown("**Valores originales**")
        st.write({"edad": edad, "colesterol": colesterol})
        st.markdown("**Valores normalizados (entrada al modelo)**")
        st.write(X_scaled.tolist())

st.markdown("---")
st.caption("Notas: \n- El modelo y el scaler deben estar en el mismo directorio que este archivo (svc_model.jb y scaler.jb).\n- Los features esperados en el orden: ['edad', 'colesterol'].\n- El comportamiento del modelo depende de cómo fue entrenado. Este app sólo despliega la predicción del archivo SVC provisto.")

# Instrucciones rápidas para ejecutar
st.info("Para ejecutar esta aplicación: en la terminal, ejecutar `streamlit run app.py` en el directorio donde estén los archivos.")
