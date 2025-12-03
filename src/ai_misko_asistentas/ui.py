import streamlit as st
from src.ai_misko_asistentas.gemini_wrapper import identify_mushroom

st.set_page_config(page_title="AI Miško Asistentas", page_icon="🍄")

st.title("🍄 Grybų atpažinimas iš nuotraukos")

model_choice = st.radio(
    "Pasirinkite atpažinimo metodą:",
    ["Vietinis modelis", "Gemini AI"],
    index=1
)

st.write("Įkelkite grybo nuotrauką ir sistema nustatys tikėtiną lotynišką pavadinimą.")

uploaded_file = st.file_uploader(
    "Įkelkite grybo nuotrauką",
    type=["jpg", "jpeg", "png"],
)

MODEL_PATH = "src/ai_misko_asistentas/models/mushroom_model.keras"
CLASSES_PATH = "src/ai_misko_asistentas/mushroom_classes.txt"


@st.cache_resource
def load_local_model_and_classes():
    import tensorflow as tf

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASSES_PATH, "r", encoding="utf-8") as f:
        class_names = f.read().splitlines()

    return model, class_names


def predict_local_mushroom(image_file):
    import tensorflow as tf
    import numpy as np

    model, class_names = load_local_model_and_classes()

    img = tf.keras.utils.load_img(image_file, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_batch = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_batch)
    pred_index = int(np.argmax(predictions[0]))
    pred_class = class_names[pred_index]
    confidence = float(np.max(predictions[0])) * 100.0

    return pred_class, confidence


if uploaded_file:
    st.image(uploaded_file, caption="Įkelta nuotrauka", use_container_width=True)

    if st.button("Atpažinti lotynišką pavadinimą"):
        # --- Gemini kelias ---
        if model_choice == "Gemini AI":
            with st.spinner("Gemini analizuoja nuotrauką..."):
                try:
                    latin_name = identify_mushroom(uploaded_file)
                    st.success("Nustatytas lotyniškas pavadinimas:")
                    st.markdown(f"### *{latin_name}*")
                except Exception as e:
                    st.error(f"Klaida: {e}")

        else:
            with st.spinner("Vietinis modelis analizuoja nuotrauką..."):
                try:
                    pred_class, confidence = predict_local_mushroom(uploaded_file)
                    st.success("Nustatytas lotyniškas pavadinimas:")
                    st.markdown(f"### *{pred_class}*")
                    st.write(f"Tikimybė: **{confidence:.2f}%**")
                except Exception as e:
                    st.error(f"Klaida: {e}")
