import streamlit as st
import numpy as np
import face_recognition
import pickle
import threading
from pathlib import Path
import streamlit_authenticator as stauth
import matplotlib.pyplot as plt
import cv2
import av
from datetime import datetime
from deta import Deta
from st_pages import Page, Section, add_page_title, show_pages
from google.cloud import storage
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
storage_client = storage.Client.from_service_account_json('webpage/magnetic-clone-404500-14b2b165bd29.json')
bucket = storage_client.get_bucket('clases_equipo4')


lock = threading.Lock()
img_container = {"img": None}

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# st.set_page_config(layout="wide")
#--- User auth

DETAKEY = "b0nuscm7yka_CWJRCsPHdCAkTspwGnHoM7jcg2HPu3Zs"
deta = Deta(DETAKEY)
db = deta.Base("grupo1_alumnos")
db2 = deta.Base("grupo1_asistencia")

def fetch_alumnos():
    res = db.fetch()
    return res.items

col1, col2, col3 = st.columns([3,1,1])


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

if 'video_writer' not in st.session_state:
    st.session_state['video_writer'] = None

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    with lock:
        img_container["img"] = img
    return frame
    
    # if True:
    #     small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    #     rgb_small_frame = small_frame
    #     face_locations = face_recognition.face_locations(rgb_small_frame)
    #     face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    #     face_names = []
    #     for face_encoding in face_encodings:
    #         matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    #         name = "Unknown"
    #         face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    #         best_match_index = np.argmin(face_distances)
    #         if matches[best_match_index]:
    #             name = known_face_names[best_match_index]
    #         face_names.append(name) 
    #     for (top, right, bottom, left), name in zip(face_locations, face_names):
    #         top *= 4
    #         right *= 4
    #         bottom *= 4
    #         left *= 4
    #         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    #         cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    #         font = cv2.FONT_HERSHEY_DUPLEX
    #         cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    #         for name in face_names:
    #             if name!= "Unknown" and name in st.session_state:
    #                 st.session_state[name] = True

    # return av.VideoFrame.from_ndarray(frame, format="bgr24")



with col1:
    ctx = webrtc_streamer(key="example", video_frame_callback=video_frame_callback)
    fig_place = st.empty()
    # fig, ax = plt.subplots(1, 1)
    # flip = st.checkbox("Flip")
    
    while ctx.state.playing:
        with lock:
            frame = img_container["img"]
        if frame is None:
            continue
        print("polloloco")
        
        # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # rgb_small_frame = small_frame
        # face_locations = face_recognition.face_locations(rgb_small_frame)
        # face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        # face_names = []
        # for face_encoding in face_encodings:
        #     matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        #     name = "Unknown"
        #     face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        #     best_match_index = np.argmin(face_distances)
        #     if matches[best_match_index]:
        #         name = known_face_names[best_match_index]
        #     face_names.append(name)
        # for name in face_names:
        #     if name!= "Unknown" and name in st.session_state:
        #         st.session_state[name] = True
        
        
    # st.write("Cámara en tiempo real con deteccion:")
    # # run = st.checkbox('Run Webcam')
    # FRAME_WINDOW = st.empty()
    # # camera = cv2.VideoCapture(-1)
    # webrtc_ctx = webrtc_streamer(key="example", rtc_configuration=RTC_CONFIGURATION)

    # ### Streamlit WebRTC API to capture video stream from webcam
    # # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # while webrtc_ctx:
    #     frame = webrtc_ctx.video_receiver.get_frame()
    #     # _, frame = camera.read()
    #     print(frame)
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     # Inicializar VideoWriter una vez que la cámara comienza a funcionar correctamente
    #     # if st.session_state['video_writer'] is None:
    #     #     frame_height, frame_width = frame.shape[:2]
    #     #     st.session_state['video_writer'] = cv2.VideoWriter('webpage/output.avi', fourcc, 20.0, (frame_width, frame_height))
        
        # if process_this_frame:
        #     small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        #     rgb_small_frame = small_frame
        #     face_locations = face_recognition.face_locations(rgb_small_frame)
        #     face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        #     face_names = []
        #     for face_encoding in face_encodings:
        #         matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        #         name = "Unknown"
        #         face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        #         best_match_index = np.argmin(face_distances)
        #         if matches[best_match_index]:
        #             name = known_face_names[best_match_index]
        #         face_names.append(name) 
        #     for (top, right, bottom, left), name in zip(face_locations, face_names):
        #         top *= 4
        #         right *= 4
        #         bottom *= 4
        #         left *= 4
        #         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        #         cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        #         font = cv2.FONT_HERSHEY_DUPLEX
        #         cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        #         for name in face_names:
        #             if name!= "Unknown" and name in st.session_state:
        #                 st.session_state[name] = True
                        
    #     st.session_state['video_writer'].write(frame)

    #     process_this_frame = True
    #     # process_this_frame = iter == 0
    #     # iter = (iter + 1) % 2
    #     process_this_frame = True
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     webrtc_ctx.video_frame_transmitter.transmit(frame)
    #     # FRAME_WINDOW.image(frame)
    
    # if not run and st.session_state['video_writer'] is not None:
    #     st.session_state['video_writer'].release()
    #     st.session_state['video_writer'] = None
    # # camera.release() 

def registo_dia(key: str, estado, fecha, participaciones):
    db2.put({"alumno" : key, "asistio" : estado, "fecha" : fecha, "participaciones": participaciones})

with col2:
    st.header("Registro de Asistencia")
    for alumno in alumnos:
        if alumno in st.session_state:
            asistio = st.checkbox(f"Asistió {alumno}", value=st.session_state[alumno])
        else:
            asistio = st.checkbox(f"Asistió {alumno}", value=False)
        st.session_state[alumno] = asistio
        if asistio:
            asistencia[alumno]["fecha"] = datetime.now()
with col3:
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
