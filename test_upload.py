
import streamlit as st

st.title("Test upload PDF")

file = st.file_uploader(
    "Choisir un CV",
    type=["pdf"]
)

if file is not None:
    st.success("Upload réussi !")
    st.write("Nom :", file.name)
    st.write("Taille :", file.size, "octets")
