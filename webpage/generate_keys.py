import pickle
from pathlib import Path

import streamlit_authenticator as stauth

names = ["Diego Elian", "Jose Romo", "Ana Cardenas", "Elias Garza"]
usernames = ["diegoelian02", "joseromo", "anacardenas", "doctorsexo"]

passwords = ["mollyamber", "applewatch", "iphone15", "estanzuela"]

hashed_passwords = stauth.Hasher(passwords).generate()

file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("wb") as file:
    pickle.dump(hashed_passwords, file)
 



