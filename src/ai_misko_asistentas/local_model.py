import tensorflow as tf
import numpy as np

MODEL_PATH = "../.. /src/ai_misko_asistentas/models/mushroom_model.keras".replace(" ", "")
CLASSES_PATH = "../.. /src/ai_misko_asistentas/mushroom_classes.txt".replace(" ", "")

IMG_SIZE = (224, 224)

_model = None
_class_names = None


def load_local_model():
    global _model, _class_names
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
    if _class_names is None:
        with open(CLASSES_PATH, "r", encoding="utf-8") as f:
            _class_names = f.read().splitlines()


def predict_local_mushroom(image_file):
    load_local_model()

    img = tf.keras.utils.load_img(image_file, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_batch = tf.expand_dims(img_array, 0)

    predictions = _model.predict(img_batch)
    pred_index = int(np.argmax(predictions[0]))
    pred_class = _class_names[pred_index]
    confidence = float(np.max(predictions[0])) * 100.0

    return pred_class, confidence
