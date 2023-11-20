#%%
from ultralytics import YOLO
import cv2
import numpy as np
from pydantic import BaseModel
import os
import face_recognition

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
    video = r'C:\Users\elias\Dropbox\Carrera\7mo Semestre\Bloque 2\StudentVisionClassIA\PoseDetection\media\VideoRapido.mp4'
    vid = cv2.VideoCapture(video) 
    
    time = 0
    last_recog_time = -delta_time_part

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
                    last_recog_time = time

                    left_up, right_down = _look4face(keypoints)
                    face = no_show_frame[left_up[1]:right_down[1], right_down[0]:left_up[0]]
                    
                    # Convert the cropped face to RGB
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)    

                    face_name = _recognize_face(face)
                    
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

    
#%%

count_participations(video_path = r'C:\Users\elias\Dropbox\Carrera\7mo Semestre\Bloque 2\StudentVisionClassIA\PoseDetection\media\VideoRapido.mp4',
                     known_faces_dir=r'..\Face Recognition\known_faces',
                     fps=30/4,
                     delta_time_part=4)

