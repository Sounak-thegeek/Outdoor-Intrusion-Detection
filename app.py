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
 
@app.route('/predict')
def predict():
    def extract_features(data):
        data_chunk = np.array(data)
        std = np.std(data_chunk)
        var = np.var(data_chunk)
        mx = np.max(data_chunk)
        mn = np.min(data_chunk)
        ptp = np.ptp(data_chunk)
        fq = np.percentile(data_chunk, 25)
        sq = np.percentile(data_chunk, 50)
        tq = np.percentile(data_chunk, 75)
 
        sqrt = np.sqrt(np.mean(data_chunk ** 2))
        se = np.sum(data_chunk ** 2)
        entr = -np.sum(np.log2(data_chunk[data_chunk > 0]) * data_chunk[data_chunk > 0])
 
        features = [std, var, mx, mn, ptp, fq, sq, tq, sqrt, se, entr]
        return features
 
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
                df = pd.DataFrame([features], columns=['std_dev', 'variance', 'max', 'min', 'range', '25th_percentile', '50th_percentile', '75th_percentile', 'rms', 'signal_energy', 'entropy'])
                prediction = model.predict(df)
                result = str(prediction[0])
                return render_template('disturbance.html', prediction_text=result)
                l = []
        except ULError as e:
            return render_template('disturbance.html', prediction_text=f"Error: {e}")
 
if __name__ == "__main__":
    app.run(debug=True)
 