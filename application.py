import pickle
from flask import Flask,redirect,url_for,render_template,request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application =Flask(__name__)
app = application

## import ridge regressor and standard scaler pickle
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler_model = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods= ['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        temperature = float(request.form['temperature'])
        RH = float(request.form['RH'])
        Ws = float(request.form['Ws'])
        Rain = float(request.form['Rain'])
        FFMC = float(request.form['FFMC'])
        DMC = float(request.form['DMC'])
        ISI = float(request.form['ISI'])
        Classes = int(request.form['Classes'])
        Region = int(request.form['Region'])

        new_data_scaled = standard_scaler_model.transform([[temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = ridge_model.predict(new_data_scaled)
        return render_template('home.html', results=result[0])
    else:
        return render_template('home.html')


if __name__ == '__main__':
    
    app.run(host = '0.0.0.0', debug=True)