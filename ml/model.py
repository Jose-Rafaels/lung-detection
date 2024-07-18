from keras.preprocessing import image
from keras.models import load_model
from keras.applications.vgg19 import preprocess_input
import numpy as np

CLASS_LABELS = ['adenocarcinoma', 'large cell carcinoma', 'normal', 'squamous cell carcinoma']
# Fungsi untuk memuat model dan memprediksi gambar
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def resnet152(img_path):
    # Memuat model
    model_path = "ml/ResNet152_model_best.h5" 
    model = load_model(model_path)
    img_array = prepare_image(img_path)
    img_array = preprocess_input(img_array)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_label = CLASS_LABELS[predicted_class[0]]
    return predicted_label


def vgg19(img_path):
    # Memuat model
    model_path = "ml/VGG19_model_best.h5"
    model = load_model(model_path) 
    img_array = prepare_image(img_path)
    img_array /= 255.0
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=-1)
    predicted_label = CLASS_LABELS[predicted_class[0]]
    return predicted_label
