import streamlit as st
from src.ai_misko_asistentas.gemini_wrapper import identify_mushroom

st.set_page_config(page_title="AI Miško Asistentas", page_icon="🍄")

st.title("🍄 Grybų atpažinimas iš nuotraukos")
st.write("Įkelkite grybo nuotrauką ir Gemini nustatys tikėtiną lotynišką pavadinimą.")

uploaded_file = st.file_uploader(
    "Įkelkite grybo nuotrauką",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file:
    st.image(uploaded_file, caption="Įkelta nuotrauka", use_container_width=True)

    if st.button("Atpažinti lotynišką pavadinimą"):
        with st.spinner("Gemini analizuoja nuotrauką..."):
            try:
                latin_name = identify_mushroom(uploaded_file)
                st.success("Nustatytas lotyniškas pavadinimas:")
                st.markdown(f"### *{latin_name}*")
            except Exception as e:
                st.error(f"Klaida: {e}")
