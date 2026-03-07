import tensorflow as tf
import numpy as np
from PIL import Image


model = tf.keras.models.load_model('model/plant_model.h5')

class_names = [
'Potato___Early_blight',
'Potato___Late_blight',
'Potato___healthy',
'Tomato___Early_blight',
'Tomato___Late_blight',
'Tomato___healthy'
]

def predict_image(image):

    image = image.resize((224,224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)

    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    return class_names[class_index], confidence
