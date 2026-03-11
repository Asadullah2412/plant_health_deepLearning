import tensorflow as tf
import numpy as np
from PIL import Image
from keras.layers import TFSMLayer


disease_model = TFSMLayer(
    "model/plant_model_serving",
    call_endpoint="serving_default"
)

leaf_model = TFSMLayer(
    'model/leaf_model_serving',
    call_endpoint="serving_default"
)
def is_leaf(image):

    img = image.resize((224,224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # pred = leaf_model.predict(img)[0][0]
    outputs = leaf_model(img)
    prediction = list(outputs.values())[0]
    prediction = prediction.numpy()

    return prediction < 0.5

class_names = [
'Potato Early blight',
'Potato Late blight',
'Potato healthy',
'Tomato Early blight',
'Tomato Late blight',
'Tomato healthy'
]

def predict_image(image):

    image = image.resize((224,224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0).astype(np.float32)

    # Call the model directly (no predict())
    outputs = disease_model(image)

    # TFSMLayer returns a dict → extract tensor
    prediction = list(outputs.values())[0]

    prediction = prediction.numpy()

    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    return class_names[class_index], confidence