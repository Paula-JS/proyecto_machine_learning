import streamlit as st
import numpy as np
import pickle

st.set_page_config(
    page_title="Precios Casas Prefabricadas APP",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Establece el precio de tus viviendas prefabricadas aquí  🏠💶🔎")
st.subheader("_For EcoHabitat_")
st.write("Bienvenido a tu aplicación online para establecer los precios de viviendas prefabricadas de EcoHabitat, esta herramienta te ayudará a establecer los precios de forma sencilla y eficaz basándose en un modelo de Machine Learning de predicción de precios de viviendas. ¡Vamos a ello!")

# Crear las columnas
col1, col2 = st.columns(2)

with col1:
    # Path del modelo preentrenado
    MODEL_PATH = "../modelo/modelo_prediccion_precios_casas_prefab.pkl"
    SCALER_PATH = "../modelo/scaler.pkl"

    # Función que devuelve la predicción
    def model_prediction(x_in, model, scaler):
        x = np.asarray(x_in).reshape(1, -1)
        x_scaled = scaler.transform(x)
        preds = model.predict(x_scaled)
        return preds

    # Cargar el modelo y el scaler solo una vez
    @st.cache_resource
    def load_model():
        with open(MODEL_PATH, "rb") as model_file, open(SCALER_PATH, "rb") as scaler_file:
            model = pickle.load(model_file)
            scaler = pickle.load(scaler_file)
        return model, scaler

    model, scaler = load_model()

    # Entrada de datos
    metros_cuadrados = st.text_input("Metros cuadrados: ")
    baños = st.text_input("Número de baños: ")
    habitaciones = st.text_input("Número de habitaciones: ")
    hormigon = st.text_input("¿Contiene hormigón? Indique (0) si es no, (1) si es sí: ")
    modelo_vivienda = st.text_input("Indique el modelo de vivienda mediante números (mirar tabla de contenidos):")

    # Botón para realizar la predicción
    if st.button("Haz tu predicción aquí ⬅️"):
        # Validar si las entradas son válidas
        try:
            x_in = [np.float_(metros_cuadrados),
                    np.float_(baños),
                    np.float_(habitaciones),
                    np.float_(hormigon),
                    np.float_(modelo_vivienda)]
            predicts = model_prediction(x_in, model, scaler)
            
            
            #Respuesta
            st.markdown(f"<div style='background-color: #32CD32; padding: 10px; border-radius: 5px;'><h3 style='color: white; text-align: center;'><strong>El precio estimado es de: {predicts[0]:.2f} € 🏠</strong></h3></div>", unsafe_allow_html=True)

            # Efecto de confeti (solo para un toque divertido)
            st.balloons()

        except Exception as e:
            st.error(f"Error al realizar la predicción: {e}")
            
with col2:
    # Mostrar la imagen
    st.image("../imagenes/Casa-prefabricada-105-Precio-52800.jpg", caption=" ", use_container_width=True)
