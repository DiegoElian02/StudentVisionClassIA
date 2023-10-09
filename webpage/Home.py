import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2

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

# Streamlit layout
st.title("Clase Inteligencia Artificial II")

# Dividir la pantalla en dos columnas de igual tamaño
col1, col2, col3 = st.columns([1, 2, 1.5])

# Gráfico de asistencias en la columna 2
with col1:
    st.pyplot(grafica_asistencias())
    
    # Gráfico de participaciones en la columna 2
    st.pyplot(grafica_participaciones())
    
# Cámara en tiempo real en la columna 1
with col2:
    # Gráfico de asistencia por día en la columna 1
    st.pyplot(grafica_asistencia_por_dia())
    
    
with col3:
    st.write("Cámara en tiempo real:")
    
    run = st.checkbox('Run Webcam')
    FRAME_WINDOW = st.empty()
    camera = cv2.VideoCapture(0)

    while run:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
    else:
        st.write('Stopped')

    


