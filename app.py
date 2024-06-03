import os

import numpy as np

# Keras
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
import librosa
import sqlite3

app = Flask(__name__)

UPLOAD_FOLDER = 'files'


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

UPLOAD_FOLDER1 = 'static/uploads/'

def specificity_m(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    specificity = true_negatives / (possible_negatives + K.epsilon())
    return specificity

def sensitivity_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    sensitivity = true_positives / (possible_positives + K.epsilon())
    return sensitivity

def mae(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))

def mse(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))



model_path2 = 'models/modelV2.h5' # load .h5 Model

custom_objects = {
    'f1_score': f1_score,
    'recall_m': recall_score,
    'precision_m': precision_score,
    'specificity_m': specificity_m,
    'sensitivity_m': sensitivity_m,
    'mae' : mae,
    'mse' : mse
}


model = load_model(model_path2, custom_objects=custom_objects)

model_name = open("tk.pkl","rb")
scaler = pickle.load(model_name)


def extract_features(data):
    result = np.array([])
    
    mfccs = librosa.feature.mfcc(y=data, sr=22050, n_mfcc=58)
    mfccs_processed = np.mean(mfccs.T,axis=0)
    result = np.array(mfccs_processed)
    
    return result

      
    
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/abstract')
def abstract():
    return render_template('abstract.html')

@app.route('/notebook')
def notebook():
    return render_template('notebook.html')

@app.route('/try_again')
def try_again():
    return render_template('tryagain.html') 



@app.route('/afterprediction',methods=['GET','POST'])
def afterprediction():
    print("Entered")
    
    print("Entered here")
    file = request.files['files'] # fet input
    filename = file.filename        
    print("@@ Input posted = ", filename)
        
    file_path = os.path.join(UPLOAD_FOLDER1, filename)
    file.save(file_path)

    duration = 3 
    test_data, _ = librosa.load(file_path, duration=duration, res_type='kaiser_fast')
    test_features = extract_features(test_data)
    test_features = scaler.transform(test_features.reshape(1, -1))  # Scale the features
    test_features = np.expand_dims(test_features, axis=2)  # Add a dimension for CNN input

    # Make predictions using the trained model
    predictions = model.predict(test_features)
    predicted_class = np.argmax(predictions)

    classes = {0:'Angry', 1:'Calm', 2:'Disgust', 3:'Fear', 4:'Happy', 5:'Neutral', 6:'Sad', 7:'Surprise'}
      
    pred = classes[predicted_class]
    print(pred)
    
    return render_template('afterprediction.html', pred_output=pred)  


if __name__ == '__main__':
    app.run(debug=True)