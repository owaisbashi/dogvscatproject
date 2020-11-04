import base64
import numpy as np
import io
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask

app= Flask(__name__)

def get_model():
    global model
    model=load_model('cats_and_dogs_small_2.h5')
    print('modelloaded')
def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image=image.convert( "RGB")
    image=image.resize(target_size)
    image=img_to_array(image)
    image=np.expand_dims(image,axis=0)
    return image

print('keras model loaded')
get_model()

@app.route("/predict",methods=["POST"])
def predict():
    message = request.get_json(force=True)
    print(message)
    encoded= message['image']
    decoded= base64.b64decode(encoded)
    image=Image.open(io.BytesIO(decoded))
    preprocessed_image=preprocess_image(image,target_size=(150,150))
    prediction=model.predict(preprocessed_image).tolist()
    print(prediction)
    response={
        'prediction': {
            'dog' : prediction[0][0],
            'cat' : prediction[0][1]
        }
    }
    return (jsonify(response))

if __name__ == '__main__':
    app.run(debug=True)