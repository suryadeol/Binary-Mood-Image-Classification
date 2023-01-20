from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

import tensorflow as tf
import os
import matplotlib.pyplot as plt
from numpy import *
from tensorflow.keras.models import load_model

app = Flask('__name__')

model=load_model('image classification.h5')
@app.route('/', methods = ['GET', 'POST'])
def index():
    return render_template("mood.html") 



@app.route('/mood_predict', methods = ['GET', 'POST'])
def mood_predict():
    if(request.method == 'POST'):
        f=request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        print(file_path)
        # Make prediction
       
        img=plt.imread(file_path)
        

        #resize according to layers
        resize=tf.image.resize(img,(256,256))
        resize=resize/255
        #expand your image array
        img=expand_dims(resize,0)
        pred=model.predict(img)
        p=pred[0]

        if p[0]>0.5:
            res="Sad"
        else:
            res="Happy"
    return render_template("open.html",n=res)


    
    
if __name__ == "__main__":
    app.run(debug = True)

