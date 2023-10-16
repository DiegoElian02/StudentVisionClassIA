from deta import Deta

DETAKEY = "b0nuscm7yka_pJWqFJMpuzDYxCpuD83GFPQe98mdJjXj"
deta = Deta(DETAKEY)

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