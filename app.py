from dis import dis
import pickle
from flask import Flask,request,app,jsonify,url_for,render_template

import numpy as np
import pandas as pd

app = Flask(__name__)
## load model
rfcmodel = pickle.load(open('rfcmodel.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():
    TT4 = request.form['TT4']
    FTI = request.form['FTI']
    T3 = request.form['T3']
    T4U = request.form['T4U']
    age = request.form['age']
    sex = int(request.form['sex'])
    X=[TT4,FTI,T3,T4U,age,sex]
    data=[float(x) for x in X]
    final_input = np.array(data).reshape(1,-1)
    print(final_input)
    output = rfcmodel.predict(final_input)[0]
    if output == 0:
        disease = "Compensated Hypothyroid ! Take Care"
    elif output == 1:
        disease = "Negative"
    elif output == 2:
        disease = "Primary Hypothyroid ! Take Care"
    elif output == 3:
        disease = "Secondary Hypothyroid ! Take Care"
    return render_template("home.html",prediction_text = "You may have a{}".format(disease))

if __name__=="__main__":
    app.run(debug=True)

