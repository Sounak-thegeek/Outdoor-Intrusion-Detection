import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for
import pickle
import pandas as pd
from mcculw import ul
from mcculw.enums import ULRange
from mcculw.ul import ULError
 
app = Flask(__name__)
model_path = r"C:\Users\Sounak Banerjee\PycharmProjects\FlaskDemo\venv\model.pkl"
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/start_prediction', methods=['POST'])
def start_prediction():
    return redirect(url_for('predict'))

app = Flask(__name__)
@app.route('/predict')
def predict():
    def extract_features(data):
        # Here the complete function to extract features is implemented
        
    boardnum = 0
    airange = ULRange.BIP10VOLTS
    channel = 1
    l = []
    while True:
        try:
            value = ul.a_in(boardnum, channel, airange)
            eng = ul.to_eng_units(boardnum, airange, value)
            l.append(float('{:.6f}'.format(eng)))
            if len(l) == 4000:
                features = extract_features(l)
                df = pd.DataFrame([features], columns=['mean', 'variance', 'skewness', 'kurtosis', 'rms', 'peak_to_peak' ,'zero_crossing_rate', 'psd_mean')
                prediction = model.predict(df)
                result = str(prediction[0])
                if result == '0':
                    source="1-person walk"
                elif result == '1':
                    source="2-person walk"
                elif result == '2':
                    source="No disturbance"
                elif result == '3':
                    source="3-person walk"
                elif result == '4':
                    source="scooty-person"
                elif result == '5':
                    source="bike"
                return render_template('disturbance.html', prediction_text=source)
                l = []
        except ULError as e:
            return render_template('disturbance.html', prediction_text=f"Error: {e}")
 
if __name__ == "__main__":
    app.run(debug=True)
