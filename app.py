from flask import Flask , request, render_template
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('breastcancer.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    data1 = float(request.form['a'])
    data2 = float(request.form['b'])
    data3 = float(request.form['c'])
    data4 = float(request.form['d'])
    data5 = float(request.form['e'])
    data6 = float(request.form['f'])
    data7 = float(request.form['g'])
    data8 = float(request.form['h'])
    data9 = float(request.form['i'])
    data10 = float(request.form['j'])
    data11 = float(request.form['k'])
    data12 = float(request.form['l'])
    data13 = float(request.form['m'])
    data14 = float(request.form['n'])
    data15 = float(request.form['o'])
    data16 = float(request.form['p'])
    data17 = float(request.form['q'])
    data18 = float(request.form['r'])
    data19 = float(request.form['s'])
    data20 = float(request.form['t'])
    data21 = float(request.form['u'])
    data22 = float(request.form['v'])
    data30 = float(request.form['w'])
    data23 = float(request.form['x'])
    data24 = float(request.form['y'])
    data25 = float(request.form['z'])
    data26 = float(request.form['aa'])
    data27 = float(request.form['ab'])
    data28 = float(request.form['ac'])
    data29 = float(request.form['ad'])
    
    df = np.array([[data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13,data14,data15,data16,data17,data18,data19,data20,data21,data22,data30,data23,data24,data25,data26,data27,data28,data29]])
    pred = model.predict(df)
    return render_template('predict.html', data=pred)


if __name__ == '__main__':
    app.run(debug=True)