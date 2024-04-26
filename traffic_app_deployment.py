from flask import Flask, request, render_template, url_for
from flask_cors import cross_origin
import os
from werkzeug.utils import secure_filename
import pickle
import numpy as np
import pandas as pd
import cv2

app = Flask(__name__)

df=pd.read_csv("C:/Users/hp/Downloads/traffic_Data/traffic_sign_labels.csv")
dict1=df.set_index('ClassId').to_dict()
classes=dict1['Name']

def image_processing(img):
    model = pickle.load(open("./model/model.pkl", "rb"))    
    data=[]
    image = cv2.imread(img)
    image_resized = cv2.resize(image, (128, 128), interpolation = cv2.INTER_AREA)
    data.append(np.array(image_resized))
    data = np.array(data)
    data = data.reshape((data.shape[0], 128*128*3))
    data = data.astype(float)/255    
    Y_pred = model.predict(data)
    return Y_pred

@app.route('/')
@cross_origin()
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
@cross_origin()
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        file_path = secure_filename(f.filename)
        f.save(file_path)
        # Make prediction
        result = image_processing(file_path)
        s = [str(i) for i in result]
        a = int("".join(s))
        result = "Predicted Traffic🚦Sign is: " +classes[a]
        os.remove(file_path)
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)