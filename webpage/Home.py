import streamlit as st
import numpy as np
import face_recognition
import pickle
from pathlib import Path
import streamlit_authenticator as stauth
import matplotlib.pyplot as plt
import cv2

st.set_page_config(layout="wide")
#--- User auth
names = ["Diego Elian", "Jose Romo", "Ana Cardenas", "Elias Garza"]
usernames = ["diegoelian02", "joseromo", "anacardenas", "doctorsexo"]

file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)
    
authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
                                    "assistance_dashboard", "abcdef", cookie_expiry_days = 0)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
    st.error("Username/password is incorrect")

if authentication_status == None:
    st.warning("Please enter your username and password")
    
if authentication_status == True:
    

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

    authenticator.logout("Logout", "main")
    
    # Dividir la pantalla en dos columnas de igual tamaño
    col1, col2, col3 = st.columns([2, 1, 1.5])

    # Gráfico de asistencias en la columna 2
    with col2:
        
        # Gráfico de participaciones en la columna 2
        st.pyplot(grafica_participaciones())
        
        # Gráfico de asistencia por día en la columna 1
        st.pyplot(grafica_asistencia_por_dia())
        
    # Cámara en tiempo real en la columna 1
    with col1:
        st.pyplot(grafica_asistencias())
        
        
    # with col3:
    #     st.write("Cámara en tiempo real:")
        
    #     run = st.checkbox('Run Webcam')
    #     FRAME_WINDOW = st.empty()
    #     camera = cv2.VideoCapture(0)

    #     while run:
    #         _, frame = camera.read()
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         FRAME_WINDOW.image(frame)
    #     else:
    #         st.write('Stopped')


    import os

    def list_files_in_folder(folder_path):
        file_names = []
        for file in os.listdir(folder_path):
            if os.path.isfile(os.path.join(folder_path, file)):
                file_names.append(file)
        return file_names

    known_face_encodings = []
    known_face_names = []

    for person in list_files_in_folder('known_faces'):
        face = face_recognition.load_image_file(f"known_faces/{person}")
        face_face_encoding = face_recognition.face_encodings(face)[0]
        
        known_face_encodings.append(face_face_encoding)
        known_face_names.append(person[:-4])

    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    iter = 0
        
    with col3:
        st.write("Cámara en tiempo real:")
        run = st.checkbox('Run Webcam')
        FRAME_WINDOW = st.empty()
        camera = cv2.VideoCapture(0)

        if run:
            ret, frame = camera.read()
            if process_this_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                face_names = []
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                    face_names.append(name)
                
                process_this_frame = iter == 0
                iter = (iter + 1) % 4
                
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame)
