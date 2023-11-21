import streamlit as st
import numpy as np
import face_recognition
import pickle
from pathlib import Path
import streamlit_authenticator as stauth
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
from deta import Deta
from st_pages import Page, Section, add_page_title, show_pages
from ultralytics import YOLO
from pydantic import BaseModel

# st.set_page_config(layout="wide")
#--- User auth

DETAKEY = "b0nuscm7yka_CWJRCsPHdCAkTspwGnHoM7jcg2HPu3Zs"
deta = Deta(DETAKEY)
db = deta.Base("grupo2_alumnos")
db2 = deta.Base("grupo2_asistencia")

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

def _is_raising_hand(keypoints):
    nose = keypoints[0]
    left_hand = keypoints[9]
    right_hand = keypoints[10]
    
    if(left_hand[1] < nose[1] or right_hand[1] < nose[1]):
        return True
    return False

def _look4face(keypoints):
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    nose = keypoints[0]
    
    down = int(max(left_shoulder[1], right_shoulder[1]))
    left = int(left_shoulder[0])
    right = int(right_shoulder[0])
    up = int(down + 3*(nose[1] - down))
    
    return (left, up), (right, down)

def _list_files_in_folder(folder_path):
    file_names = []
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)):
            file_names.append(file)
    return file_names

def _recognize_face(frame, know_faces_dir = r'..\Face Recognition\known_faces'):
    known_face_encodings = []
    known_face_names = []

    for person in _list_files_in_folder(know_faces_dir):
        face = face_recognition.load_image_file(know_faces_dir + f"/{person}")
        face_face_encoding = face_recognition.face_encodings(face)[0]
        
        known_face_encodings.append(face_face_encoding)
        known_face_names.append(person[:-4])
        
    face_locations = []
    face_encodings = []
    face_names = []
    
        # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # # If a match was found in known_face_encodings, just use the first one.
        
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)
    
    if face_names:
        return face_names[0]
    else:
        return 'Unknown'

def count_participations(video_path:str, known_faces_dir:str, fps:float= 30/4, delta_time_part:float = 5):
    
    participation_dict = {name[:-4]:0 for name in _list_files_in_folder(known_faces_dir)}
    
    model = YOLO('yolov8n-pose.pt')
    video = video_path
    vid = cv2.VideoCapture(video) 
    
    time = 0
    last_recog_time = -delta_time_part

    while vid.isOpened():
        ret , frame = vid.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.65, show=False)
        no_show_frame = frame.copy()
        # frame = results[0].plot(boxes=True)   
        
        if results[0]: # Verificar si hay detecciones
            for person in results[0]: # Recorrer la lista de objetos (Personas) detectados

                keypoints = person.keypoints.xy.cpu().numpy()[0]

                if(_is_raising_hand(keypoints) and time - last_recog_time > delta_time_part):
                    last_recog_time = time

                    left_up, right_down = _look4face(keypoints)
                    # cv2.rectangle(frame, left_up, right_down, (255, 0, 0), 3)
                    face = no_show_frame[left_up[1]:right_down[1], right_down[0]:left_up[0]]
                    
                    # Convert the cropped face to RGB
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)    
                    # cv2.imwrite('Face.jpg', face)
                    
                    face_name = _recognize_face(face, known_faces_dir)
                    
                    if face_name != 'Unknown':
                        last_recog_time = time
                        participation_dict[face_name] += 1
                        
        # cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
        
        time += 1/fps

    vid.release()
    cv2.destroyAllWindows()
    
    return participation_dict



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
reconocidos = set()


st.write("Estos son los alumnos del grupo 2 de la clase de inteligencia artificial avanzada:")
f = fetch_alumnos()
alumnos = [alumno['nombre'] for alumno in f]
asistencia = {alumno: {"asistio": False, "fecha": None, "participaciones": 0} for alumno in alumnos}

for alumno in reconocidos:
    print(alumno)
    print(asistencia)
    print("lol?")
    if alumno in asistencia:
        st.session_state[alumno] = True

with col1:
    # st.write("Cámara en tiempo real:")
            
    # run = st.checkbox('Run Webcam')
    # FRAME_WINDOW = st.empty()
    # camera = cv2.VideoCapture(0)
    # while run:
    #     _, frame = camera.read()
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     FRAME_WINDOW.image(frame)
    # else:
    #     st.write('Stopped')

    st.write("Cámara en tiempo real con deteccion:")
    run = st.checkbox('Run Webcam')
    FRAME_WINDOW = st.empty()
    camera = cv2.VideoCapture(0)
    
    while run:
        _, frame = camera.read()
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
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
                for name in face_names:
                    if name!= "Unknown" and name in st.session_state:
                        st.session_state[name] = True
        # process_this_frame = iter == 0
        # iter = (iter + 1) % 2
        process_this_frame = True
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

def registo_dia(key: str, estado, fecha, participaciones):
    return db2.put({"alumno" : key, "asistio" : estado, "fecha" : fecha, "participaciones": participaciones})


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
        estado = True if datos["asistio"] else False
        fecha = datos["fecha"].strftime("%Y-%m-%d %H:%M:%S") if datos["fecha"] else "N/A"
        participaciones = datos["participaciones"]
        if estado:
            registo_dia(alumno,estado,fecha,participaciones)
    st.success("Asistencia guardada con éxito")
