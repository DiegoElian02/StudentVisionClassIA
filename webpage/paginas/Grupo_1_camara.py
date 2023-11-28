
import streamlit as st
# st.legacy_caching.clear_cache()
import numpy as np
import pandas as pd
import face_recognition
from pathlib import Path
import streamlit_authenticator as stauth
import matplotlib.pyplot as plt
import cv2
from typing import List

from datetime import datetime
from deta import Deta
from st_pages import Page, Section, add_page_title, show_pages
from google.cloud import storage

from ultralytics import YOLO
from pydantic import BaseModel

storage_client = storage.Client.from_service_account_json('magnetic-clone-404500-14b2b165bd29.json')
bucket = storage_client.get_bucket('clases_equipo4')
blob = bucket.blob(f'uploads/{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.mp4')      

DETAKEY = "b0nuscm7yka_CWJRCsPHdCAkTspwGnHoM7jcg2HPu3Zs"
deta = Deta(DETAKEY)
db = deta.Base("grupo1_alumnos")
db2 = deta.Base("grupo1_asistencia")
db3 = deta.Base("grupo1_participacion")

def fetch_alumnos():
    res = db.fetch()
    return res.items

col1, col2 = st.columns([3,1])

import os
def _list_files_in_folder(folder_path):
    file_names = []
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)):
            file_names.append(file)
    return file_names

def _is_raising_hand(keypoints):
    nose = keypoints[0]
    left_hand = keypoints[9]
    right_hand = keypoints[10]
    right_elbow = keypoints[8]
    left_elbow = keypoints[7]
    right_should = keypoints[6]
    left_should = keypoints[5]
    
    
    if(
        (left_hand[1] < nose[1] and left_hand[1] > 0 and left_elbow[1] < left_should[1]) or
        (right_hand[1] < nose[1] and right_hand[1] > 0 and right_elbow[1] < right_should[1])
        ):
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

def _recognize_face(frame, know_faces_dir):
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
    counter = 0
    time = 0
    last_recog_time = -delta_time_part / 2

    while vid.isOpened():
        ret , frame = vid.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.65, show=False)
        no_show_frame = frame.copy()
        
        if results[0]: # Verificar si hay detecciones
            for person in results[0]: # Recorrer la lista de objetos (Personas) detectados

                keypoints = person.keypoints.xy.cpu().numpy()[0]

                if(_is_raising_hand(keypoints) and time - last_recog_time > delta_time_part):
                    # last_recog_time = time

                    left_up, right_down = _look4face(keypoints)
                    face = no_show_frame[left_up[1]:right_down[1], right_down[0]:left_up[0]]
                    
                    # Convert the cropped face to RGB
                    try:
                        face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)  
                        cv2.imwrite(f'Face{counter}.jpg', face)
                        box_frame = results[0].plot(boxes=True)
                        cv2.imwrite(f'Frame{counter}.jpg', box_frame)
                        counter += 1
                          
                        face_name = _recognize_face(face, known_faces_dir)
                        
                        print(face_name)
                        
                        if face_name != 'Unknown':
                            last_recog_time = time
                            participation_dict[face_name] += 1
                    except Exception as e:
                        print(f"Paso alguien. Error: {e}")
                        
                        
        # cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
        
        time += 1/fps

    vid.release()
    cv2.destroyAllWindows()
    
    return participation_dict

def registo_dia(key: str, estado, fecha, participaciones):
    db2.put({"alumno" : key, "asistio" : estado, "fecha" : fecha, "participaciones": participaciones})

def registo_part(name: str, fecha, participaciones):
    db3.put({"alumno" : name,  "fecha" : fecha, "participaciones": participaciones})

#Creando face encodigns de repositorio de caras
known_face_encodings = []
known_face_names = []
for person in _list_files_in_folder('known_faces'):
    face = face_recognition.load_image_file(f"known_faces/{person}")
    face_face_encoding = face_recognition.face_encodings(face)[0]
        
    known_face_encodings.append(face_face_encoding)
    known_face_names.append(person[:-4])

#Añadiendo los alumnos disponibles
st.write("Estos son los alumnos del grupo 1 de la clase de inteligencia artificial avanzada:")
f = fetch_alumnos()
# st.write(f)
alumnos = set([alumno['nombre'] for alumno in f])

#añadiendo set de reconocidos
st.session_state['reconocidos'] = []

if 'video_writer' not in st.session_state:
    st.session_state['video_writer'] = None

with col1:
    st.write("Cámara en tiempo real con deteccion:")
    run = st.checkbox('Run Webcam')
    FRAME_WINDOW = st.empty()
    camera = cv2.VideoCapture(0)

    # Definir el codec e inicializar el objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    #Definir iteradores de frames
    iter = 0
    process_this_frame = True
    
    while run:
        _, frame = camera.read()

        # Inicializar VideoWriter una vez que la cámara comienza a funcionar correctamente
        if st.session_state['video_writer'] is None:
            frame_height, frame_width = frame.shape[:2]
            st.session_state['video_writer'] = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))
        frame_to_save = frame.copy()
        
        if process_this_frame and iter < 20:
            small_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
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
            
            #adding boxes
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 1
                right *= 1
                bottom *= 1
                left *= 1
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            
            
            print(face_names)
            for name in face_names:
                if(name!= "Unknown" 
                   and name in alumnos 
                   and name not in st.session_state):
                        st.session_state[name] = True
                        registo_dia(name, True, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 0)
                        # st.session_state[f'check_{name}'] = True

        iter = (iter + 1)
        if(iter%4 == 0):
            st.session_state['video_writer'].write(frame_to_save)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

    if not run and st.session_state['video_writer'] is not None:
        st.session_state['video_writer'].release()
        st.session_state['video_writer'] = None
    camera.release() 

with col2:
    st.header("Registro de Asistencia")
    for alumno in alumnos:
        if alumno in st.session_state and st.session_state[alumno]:
            asistio = st.checkbox(f"Asistió {alumno}", value= True)
        else:
            asistio = st.checkbox(f"Asistió {alumno}", value=False)
        st.session_state[alumno] = asistio

if st.button("Terminar Clase"):
    part_dict = count_participations(video_path = r'output.mp4',
                     known_faces_dir=r'known_faces',
                     fps=30/4,
                     delta_time_part=6)
    part_df = pd.DataFrame(list(part_dict.items()), columns=['Nombre', 'Participaciones'])
    st.table(part_df[part_df['Participaciones'] > 0])
    
    for i in part_dict.items():
        if(i[1] > 0):
            registo_part(i[0], datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i[1])
    
    blob.upload_from_filename('output.mp4')
    
    # for alumno, datos in asistencia.items():
    #     # estado = True if datos["asistio"] else False
    #     estado = True if st.session_state[alumno] else False
    #     fecha = datos["fecha"].strftime("%Y-%m-%d %H:%M:%S") if datos["fecha"] else "N/A"
        

    #     participaciones = datos["participaciones"]
    #     if estado:
    #         registo_dia(alumno,estado,fecha,participaciones)
    #         st.success("Asistencia guardada con éxito")
