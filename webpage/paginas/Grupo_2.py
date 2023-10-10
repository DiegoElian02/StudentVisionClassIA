import streamlit as st
# st.set_page_config(layout="wide")
st.title("Inteligencia artificial avanzada I")

# Verificar si la lista de alumnos ya existe en el estado de la sesión
if "lista_alumnos2" not in st.session_state:
    st.session_state.lista_alumnos2 = ["Fede Medina (21 años)", "Michelle Morales (21 años)", "Fernanda Alcubilla (22 años)"]

# Diseño de dos columnas
col1, col2 = st.columns([2, 1])

with col1:
    st.write("Estos son los alumnos del grupo 1 de la clase de inteligencia artificial avanzada:")
    # Mostrar lista de alumnos
    for alumno in st.session_state.lista_alumnos2:
        st.markdown(f"- {alumno}")

with col2:
    st.subheader("Agregar alumno")
    
    # Campos de entrada
    nombre_nuevo = st.text_input("Nombre del alumno")
    apellido_nuevo = st.text_input("Apellido del alumno")
    edad_nuevo = st.number_input("Edad del alumno", min_value=1, max_value=100, step=1)
    
    # Botón para agregar alumno
    add_button = st.button('Registrar')
    if add_button and nombre_nuevo:
        alumno_completo = f"{nombre_nuevo} {apellido_nuevo} ({edad_nuevo} años)"
        st.session_state.lista_alumnos2.append(alumno_completo)
        # Limpiar campos después de agregar
        st.experimental_rerun()