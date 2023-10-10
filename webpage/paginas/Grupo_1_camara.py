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
#--- User auth


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

# st.write("Cámara en tiempo real:")
# run = st.checkbox('Run Webcam')
# FRAME_WINDOW = st.empty()
# camera = cv2.VideoCapture(0)
# if run:
#     ret, frame = camera.read()
#     if process_this_frame:
#         small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
#         rgb_small_frame = small_frame
#         face_locations = face_recognition.face_locations(rgb_small_frame)
#         face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

#         face_names = []
#         for face_encoding in face_encodings:
#             matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
#             name = "Unknown"
#             face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
#             best_match_index = np.argmin(face_distances)
#             if matches[best_match_index]:
#                 name = known_face_names[best_match_index]
#             face_names.append(name)

#         process_this_frame = iter == 0
#         iter = (iter + 1) % 4

#         for (top, right, bottom, left), name in zip(face_locations, face_names):
#             top *= 4
#             right *= 4
#             bottom *= 4
#             left *= 4
#             cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#             cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
#             font = cv2.FONT_HERSHEY_DUPLEX
#             cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         FRAME_WINDOW.image(frame)
