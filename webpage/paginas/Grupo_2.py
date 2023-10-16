import streamlit as st
from deta import Deta

st.title("Inteligencia artificial avanzada I")
# st.set_page_config(layout="wide")
# Verificar si la lista de alumnos ya existe en el estado de la sesión
# if "lista_alumnos" not in st.session_state:
#     st.session_state.lista_alumnos = ["Diego Rodriguez (21 años)", "Ana Cardenas (21 años)", "Jose Romo (22 años)"]

DETAKEY = "b0nuscm7yka_pJWqFJMpuzDYxCpuD83GFPQe98mdJjXj"
deta = Deta(DETAKEY)

db = deta.Base("grupo2_alumnos")

def nuevo_alumno(key: str, nombre: str, apellido: str, edad: int, asistencias: int, participaciones : int):
    return db.put({"key" : key, "nombre" : nombre, "apellido": apellido, "edad": edad, "asistencias": asistencias, "participaciones": participaciones})

def fetch_alumnos():
    res = db.fetch()
    return res.items

def get_alumno(key):
    return db.get(key)

def update_alumno(key, updates):
    return db.update(key, updates)

def delete_alumno(key):
    return db.delete(key)

# Diseño de dos columnas
col1, col2 = st.columns([2, 1])

with col1:
    st.write("Estos son los alumnos del grupo 2 de la clase de inteligencia artificial avanzada:")
    # Mostrar lista de alumnos
    f = fetch_alumnos()
    alumnos = [alumno['nombre'] for alumno in f] 
    for alumno in alumnos:
        st.markdown(f"- {alumno}")

with col2:
    st.subheader("Agregar alumno")
    
    # Campos de entrada
    matricula_nueva = st.text_input("Matricula del alumno")
    nombre_nuevo = st.text_input("Nombre del alumno")
    apellido_nuevo = st.text_input("Apellido del alumno")
    edad_nuevo = st.number_input("Edad del alumno", min_value=1, max_value=100, step=1)
    
    # Botón para agregar alumno
    add_button = st.button('Registrar')
    if add_button and nombre_nuevo:
        nuevo_alumno(matricula_nueva, nombre_nuevo, apellido_nuevo, edad_nuevo, 0, 0)
        # Limpiar campos después de agregar
        st.experimental_rerun()
