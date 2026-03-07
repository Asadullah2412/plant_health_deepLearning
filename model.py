import tensorflow as tf
import numpy as np
from PIL import Image


model = tf.keras.models.load_model('')

class_names = ['healthy','disease']

def predict_image(image):
    image = image.resize((224,224))
    image = np.array(image)/255.0
    image =np.expand_dims(image,axis=0)

    prediction = model.predict(image)
    class_index = np.argmax(prediction)

    return class_names[class_index]
