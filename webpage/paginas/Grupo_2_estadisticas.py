import streamlit as st
import numpy as np
import face_recognition
import pickle
from pathlib import Path
import streamlit_authenticator as stauth
import matplotlib.pyplot as plt
import cv2
from st_pages import Page, Section, add_page_title, show_pages

# st.set_page_config(layout="wide")
# Datos ficticios

alumnos = ['Juan', 'Pedro', 'María', 'Perla', 'Diego']
asistencias = np.random.randint(1, 10, len(alumnos))
participaciones = np.random.randint(1, 20, len(alumnos))
dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes']
asistencia_por_dia = np.random.randint(5, 20, len(dias))

def grafica_asistencias():
    fig, ax = plt.subplots()
    ax.barh(alumnos, asistencias)
    ax.set_xlabel('Asistencias')
    ax.set_title('Recuento de Asistencia')
    return fig

def grafica_asistencia_por_dia():
    fig, ax = plt.subplots()
    ax.scatter(dias, asistencia_por_dia, color='blue')
    ax.plot(dias, asistencia_por_dia, color='red', linestyle='-', linewidth=1)
    ax.set_ylabel('Asistencia')
    ax.set_title('Asistencia por día')
    return fig

def grafica_participaciones():
    fig, ax = plt.subplots()
    ax.barh(alumnos, participaciones, color='green')
    ax.set_xlabel('Participaciones')
    ax.set_title('Participación por alumno')
    return fig

col1, col2 = st.columns([2, 1])

# Gráfico de asistencias en la columna 2
with col2:
    
    # Gráfico de participaciones en la columna 2
    st.pyplot(grafica_participaciones())
    
    # Gráfico de asistencia por día en la columna 1
    st.pyplot(grafica_asistencia_por_dia())
    
# Cámara en tiempo real en la columna 1
with col1:
    st.pyplot(grafica_asistencias())