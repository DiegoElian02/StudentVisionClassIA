import streamlit as st
import numpy as np
import pandas as pd
import face_recognition
import pickle
from pathlib import Path
import streamlit_authenticator as stauth
import matplotlib.pyplot as plt
import cv2
from st_pages import Page, Section, add_page_title, show_pages
from deta import Deta
from datetime import datetime
import calendar
# st.set_page_config(layout="wide")

DETAKEY = "b0nuscm7yka_CWJRCsPHdCAkTspwGnHoM7jcg2HPu3Zs"
deta = Deta(DETAKEY)
db = deta.Base("grupo1_alumnos")
db2 = deta.Base("grupo1_asistencia")
db3 = deta.Base("grupo1_participacion")

def fetch_base():
    res = db.fetch()
    return res.items

def get_alumno(key):
    return db.get(key)

def update_alumno(key, updates):
    return db.update(key, updates)

def delete_alumno(key):
    return db.delete(key)

data = list(db2.fetch({}).items)

data = [{**entry, 'day_of_week': calendar.day_name[datetime.strptime(entry['fecha'], "%Y-%m-%d %H:%M:%S").weekday()]} for entry in data]

df_asistencias = pd.DataFrame(data)
df_asistencias["fecha"] = df_asistencias["fecha"].apply(lambda x: pd.to_datetime(x).date())
alumnos_hoy = df_asistencias[df_asistencias["fecha"] == df_asistencias["fecha"].max()]["alumno"].unique()
st.write(alumnos_hoy)
alumnos = list(set([item['alumno'] for item in data]))

asistencias = []

data_participaciones = list(db3.fetch({}).items)

data_participaciones = [{**entry, 'day_of_week': calendar.day_name[datetime.strptime(entry['fecha'], "%Y-%m-%d %H:%M:%S").weekday()]} for entry in data_participaciones]

for alumno in alumnos:
    asistencias.append(sum([1 for item in data if item['alumno'] == alumno and item['asistio']]))

participaciones = []
for alumno in alumnos:
    participaciones.append(sum([item['participaciones'] for item in data if item['alumno'] == alumno]))

dias = ['monday', 'tuesday', 'wednesday', 'thrusday', 'friday']
asistencia_por_dia = []
for dia in dias:
    asistencia_por_dia.append(sum([1 for item in data if item['day_of_week'].lower() == dia and item['asistio']]))


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

alumnos_participacion = list(set([item['alumno'] for item in data_participaciones]))
participaciones = []
alumnos_participacion = list(set([item['alumno'] for item in data_participaciones]))
for alumno in alumnos_participacion:
    participaciones.append(sum([item['participaciones'] for item in data_participaciones if item['alumno'] == alumno]))

def grafica_participaciones():
    fig, ax = plt.subplots()
    ax.barh(alumnos_participacion, participaciones, color='green')
    ax.set_xlabel('Participaciones')
    ax.set_title('Participación por alumno')
    return fig

col1, col2 = st.columns([2, 1])

with col2:
    
    st.pyplot(grafica_participaciones())
    
    st.pyplot(grafica_asistencia_por_dia())
    
with col1:
    st.pyplot(grafica_asistencias())
    
st.write(list(alumnos_hoy))
st.table(df_asistencias)
# st.table(pd.DataFrame({"Asistencia de hoy" : alumnos_hoy}))