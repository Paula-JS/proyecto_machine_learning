#importamos las librerías necesarias
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template, url_for
import sklearn
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Precios Casas Prefabricadas APP",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Establece el precio de tus viviendas prefabricadas aquí  🏠💶🔎")
st.subheader("_For EcoHabitat_")
st.write("Bienvenido a tu aplicación online para establecer los precios de viviendas prefabricadas de EcoHabitat, esta herramienta te ayudará a establecer los precios de forma sencilla y eficaz basándose en un modelo de Machine Learning de predicción de precios de viviendas. ¡Vamos a ello!")
st.image("../imagenes/Casa-prefabricada-105-Precio-52800.jpg", caption=" ", width=700)

 # Path del modelo preentrenado
MODEL_PATH = "../modelo/modelo_prediccion_precios_casas_prefab.pkl"
SCALER_PATH = "../modelo/scaler.pkl"

# Función que devuelve la predicción

def model_prediction (x_in, model, scaler):
    x = np.asarray(x_in).reshape(1, -1)
    x_scaled = scaler.transform(x)
    preds = model.predict(x_scaled)
    
    return preds

def main():
    
    model = " "
    
    # Se carga el modelo
    if model == " ":
        with open(MODEL_PATH, "rb") as model_file, open(SCALER_PATH, "rb") as scaler_file:
            model = pickle.load(model_file)
            scaler = pickle.load(scaler_file)
        print(type(scaler))

    #Lectura de datos

    metros_cuadrados = st.text_input("Metros cuadrados: ")
    baños = st.text_input("Número de baños: ")
    habitaciones = st.text_input("Número de habitaciones: ")
    hormigon = st.text_input ("¿Contiene hormigón? indique (0) si es no, (1) si es sí: ")
    modelo_vivienda  = st.text_input("Indique el modelo de vivienda mediante números (mirar tabla de contenidos: )")

    # El botón predicción se usa para iniciar el procesamiento

    if st.button("Predicción: "):
        x_in = [np.float_(metros_cuadrados.title()),
                np.float_(baños.title()),
                np.float_(habitaciones.title()),
                np.float_(hormigon.title()),
                np.float_(modelo_vivienda.title())]
        predicts = model_prediction (x_in, model, scaler)
        st.success("El precio estimado de la vivienda es de: {} €".format(predicts[0]))

if __name__ == "__main__":
    main() 
     
    