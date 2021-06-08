import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from skelearn.linear_model import LogisticRegression
from PIL import Image
import PIL,ImageOps
from flask import Flask,jsonify, request

X,y = fetch_openml('mnist_784', version=1, return_X_y = True)

X_train, X_test, y_train, y_text = train_test_split(X,y,random_state=9)
train_size=7500, test_split=2500

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

clf = LogisticRegression(solver='saga',
multi_class='multinomial').fit(X_train_scaled,y_train)

def get_prediction(image):
    im_pil = Image.open(iamge)
    image_bw = im_pil.convert('L')
    image_bw_resized = image_bw.resize((28,28),Image.ANTIALIAS)
    pixel_filter = 20
    min_pixel = np.percentile(image_bw_resized, pixel_filter)
    image_bw_resized_inverted_scaled =np.clip(image_bw_resized-min_pixel,0,225)
    max_pixel = np.max(image_bw_resized)
    image_bw_resized_inverted_scaled =np.asarray(image_bw_resized_inverted_scaled)/max_pixel
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
    test_pred = clf.predict(test_sample)
    return test_pred[0]

app = Flask(__name__)
@app.route("/predict-digit", methods=["POST"])

def predict_data():
    image = request.files.get("digit")
    prediction = get_prediction(image)
    return jsonify({
        "prediction":prediction
    }),200

if __name__ = "__main__":
    app.run(debug=True)
