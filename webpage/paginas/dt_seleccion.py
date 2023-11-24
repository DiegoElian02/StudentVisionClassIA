"""Object detection demo with MobileNet SSD.
This model and code are based on
https://github.com/robmarkcole/object-detection-app
"""

import logging
import queue
from pathlib import Path
from typing import List, NamedTuple
import os

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import face_recognition

import urllib.request

def download_file(url, download_to: Path, expected_size=None):
    # Don't download the file twice.
    # (If possible, verify the download using the file length.)
    if download_to.exists():
        if expected_size:
            if download_to.stat().st_size == expected_size:
                return
        else:
            st.info(f"{url} is already downloaded.")
            if not st.button("Download again?"):
                return

    download_to.parent.mkdir(parents=True, exist_ok=True)

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % url)
        progress_bar = st.progress(0)
        with open(download_to, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)"
                        % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()

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

#Guardando las caras conocidas
known_face_encodings = []
known_face_names = []

for person in _list_files_in_folder(r'webpage/known_faces'):
    face = face_recognition.load_image_file(r'webpage/known_faces' + f"/{person}")
    face_face_encoding = face_recognition.face_encodings(face)[0]
    
    known_face_encodings.append(face_face_encoding)
    known_face_names.append(person[:-4])
    
HERE = Path(__file__).parent
ROOT = HERE

logger = logging.getLogger(__name__)


MODEL_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.caffemodel"  # noqa: E501
MODEL_LOCAL_PATH = ROOT / "./models/MobileNetSSD_deploy.caffemodel"
PROTOTXT_URL = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.prototxt.txt"  # noqa: E501
PROTOTXT_LOCAL_PATH = ROOT / "./models/MobileNetSSD_deploy.prototxt.txt"

CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

@st.cache_resource  # type: ignore
def generate_label_colors():
    return np.random.uniform(0, 255, size=(len(CLASSES), 3))


COLORS = generate_label_colors()

download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=23147564)
download_file(PROTOTXT_URL, PROTOTXT_LOCAL_PATH, expected_size=29353)


# Session-specific caching
cache_key = "object_detection_dnn"
if cache_key in st.session_state:
    net = st.session_state[cache_key]
else:
    net = cv2.dnn.readNetFromCaffe(str(PROTOTXT_LOCAL_PATH), str(MODEL_LOCAL_PATH))
    st.session_state[cache_key] = net

score_threshold = st.slider("Score threshold", 0.0, 1.0, 0.5, 0.05)

# NOTE: The callback will be called in another thread,
#       so use a queue here for thread-safety to pass the data
#       from inside to outside the callback.
# TODO: A general-purpose shared state object may be more useful.
result_queue: "queue.Queue[List[str]]" = queue.Queue()


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")

    small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
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

    # Render bounding boxes and captions
    # for detection in detections:
    #     caption = f"{detection.label}: {round(detection.score * 100, 2)}%"
    #     color = COLORS[detection.class_id]
    #     xmin, ymin, xmax, ymax = detection.box.astype("int")

    #     cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
    #     cv2.putText(
    #         image,
    #         caption,
    #         (xmin, ymin - 15 if ymin - 15 > 15 else ymin + 15),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         0.5,
    #         color,
    #         2,
    #     )

    result_queue.put(face_names)

    return av.VideoFrame.from_ndarray(image, format="bgr24")


webrtc_ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if st.checkbox("Show the detected labels", value=True):
    if webrtc_ctx.state.playing:
        labels_placeholder = st.empty()
        # NOTE: The video transformation with object detection and
        # this loop displaying the result labels are running
        # in different threads asynchronously.
        # Then the rendered video frames and the labels displayed here
        # are not strictly synchronized.
        while True:
            result = result_queue.get()
            labels_placeholder.table(result)

st.markdown(
    "This demo uses a model and code from "
    "https://github.com/robmarkcole/object-detection-app. "
    "Many thanks to the project."
)