import streamlit as st
import numpy as np
import face_recognition
import pickle
# import threading
from pathlib import Path
import streamlit_authenticator as stauth
import matplotlib.pyplot as plt
import cv2
from typing import List
import queue
from datetime import datetime
from deta import Deta
from st_pages import Page, Section, add_page_title, show_pages
from google.cloud import storage


storage_client = storage.Client.from_service_account_json('webpage/magnetic-clone-404500-14b2b165bd29.json')
bucket = storage_client.get_bucket('clases_equipo4')

# st.set_page_config(layout="wide")
#--- User auth

DETAKEY = "b0nuscm7yka_CWJRCsPHdCAkTspwGnHoM7jcg2HPu3Zs"
deta = Deta(DETAKEY)
db = deta.Base("grupo1_alumnos")
db2 = deta.Base("grupo1_asistencia")

def fetch_alumnos():
    res = db.fetch()
    return res.items

col1, col2 = st.columns([1,1])


import os

def list_files_in_folder(folder_path):
    file_names = []
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)):
            file_names.append(file)
    return file_names

known_face_encodings = []
known_face_names = []

for person in list_files_in_folder('webpage/known_faces'):
    face = face_recognition.load_image_file(f"webpage/known_faces/{person}")
    face_face_encoding = face_recognition.face_encodings(face)[0]
        
    known_face_encodings.append(face_face_encoding)
    known_face_names.append(person[:-4])

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
iter = 0
reconocidos = set()

st.write("Estos son los alumnos del grupo 1 de la clase de inteligencia artificial avanzada:")
f = fetch_alumnos()
alumnos = [alumno['nombre'] for alumno in f]
asistencia = {alumno: {"asistio": False, "fecha": None, "participaciones": 0} for alumno in alumnos}

for alumno in reconocidos:
    print(alumno)
    print(asistencia)
    if alumno in asistencia:
        st.session_state[alumno] = True

# if 'video_writer' not in st.session_state:
#     st.session_state['video_writer'] = None

result_queue = queue.Queue()


# def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
#     image = frame.to_ndarray(format="bgr24")
#     small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
#     face_locations = face_recognition.face_locations(small_frame)
#     face_encodings = face_recognition.face_encodings(face_locations, face_locations)
#     face_names = []
#     for face_encoding in face_encodings:
#             matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
#             name = "Unknown"
#             face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
#             best_match_index = np.argmin(face_distances)
#             if matches[best_match_index]:
#                 name = known_face_names[best_match_index]
#             face_names.append(name) 
#     result_queue.put(name)
#     return av.VideoFrame.from_ndarray(image, format="bgr24")

caras=[]

def registo_dia(key: str, estado, fecha, participaciones):
    db2.put({"alumno" : key, "asistio" : estado, "fecha" : fecha, "participaciones": participaciones})

with col1:
    st.header("Registro de Asistencia")
    for alumno in alumnos:
        if alumno in st.session_state:
            asistio = st.checkbox(f"Asistió {alumno}", value=st.session_state[alumno])
        else:
            asistio = st.checkbox(f"Asistió {alumno}", value=False)
        st.session_state[alumno] = asistio
        if asistio:
            asistencia[alumno]["fecha"] = datetime.now()
with col2:
    st.header("Número de Participaciones")
    for alumno in alumnos:
        participaciones = st.number_input(f"Participaciones {alumno}", min_value=0, value=asistencia[alumno]["participaciones"])
        asistencia[alumno]["participaciones"] = participaciones

if st.button("Guardar Asistencia"):
    for alumno, datos in asistencia.items():
        # estado = True if datos["asistio"] else False
        estado = True if st.session_state[alumno] else False
        fecha = datos["fecha"].strftime("%Y-%m-%d %H:%M:%S") if datos["fecha"] else "N/A"
        
        blob = bucket.blob(f'uploads/{fecha}.avi')
        participaciones = datos["participaciones"]
        if estado:
            registo_dia(alumno,estado,fecha,participaciones)
            blob.upload_from_filename('webpage/output.avi')
            st.success("Asistencia guardada con éxito")
