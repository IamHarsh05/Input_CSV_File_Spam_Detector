import spacy
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from flask import Flask, render_template, request, send_file
vect = pickle.load(open('vec.pkl', 'rb'))
model = pickle.load(open('mdel.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        
        if not file:
            return render_template('index.html', error_message='Please upload a file')
        
        try:
            df = pd.read_csv(file, encoding="ISO-8859-1")
        except:
            return render_template('index.html', error_message='Invalid file format. Please upload a CSV file')
        
        en = spacy.load('en_core_web_sm')
        sw_spacy = en.Defaults.stop_words
        df['clean'] = df['v2'].apply(lambda x: ' '.join(
            [word for word in x.split() if word.lower() not in (sw_spacy)]))
        
        count = vect.transform(df['clean'])
        model = pickle.load(open('mdel.pkl', 'rb'))
        predictions = model.predict(count)
        
        df['Prediction'] = predictions
        df.to_csv('predictions.csv', index=False)
        
        return send_file('predictions.csv', as_attachment=True)
    
    return render_template('index.html')


@app.route("/forward", methods=['POST'])
def move_forward():
    if request.method == 'POST':
        input_string = request.form['text']
        data = [input_string]

        input_data_features = vect.transform(data).toarray()
        preds = model.predict(input_data_features)
    
        if preds != 'ham':
            return render_template('index.html', value="Spam")
        else:
            return render_template('index.html', value="Not Spam")

if __name__ == "__main__":
    app.run(debug=True)
