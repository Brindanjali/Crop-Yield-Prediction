from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle
import pandas as pd

# Load model and preprocessor
dtr = pickle.load(open('dtr.pkl', 'rb'))
preprocesser = pickle.load(open('preprocesser.pkl', 'rb'))

# Load the CSV dataset
yield_df = pd.read_csv('yield_df.csv')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fetch_features', methods=['POST'])
def fetch_features():
    data = request.get_json()
    year = int(data['Year'])
    area = data['Area'].strip()
    item = data['Item'].strip()

    filtered = yield_df[
        (yield_df['Year'] == year) &
        (yield_df['Area'].str.strip().str.lower() == area.lower()) &
        (yield_df['Item'].str.strip().str.lower() == item.lower())
    ]

    if not filtered.empty:
        rainfall = filtered.iloc[0]['average_rain_fall_mm_per_year']
        pesticides = filtered.iloc[0]['pesticides_tonnes']
        temps = sorted(filtered['avg_temp'].dropna().unique())

        return jsonify({
            'average_rain_fall_mm_per_year': rainfall,
            'pesticides_tonnes': pesticides,
            'avg_temp_options': temps
        })
    else:
        return jsonify({'error': 'No data found'}), 404

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        Year = int(request.form['Year'])
        average_rain_fall_mm_per_year = float(request.form['average_rain_fall_mm_per_year'])
        pesticides_tonnes = float(request.form['pesticides_tonnes'])
        avg_temp = float(request.form['avg_temp'])
        Area = request.form['Area']
        Item = request.form['Item']

        features = pd.DataFrame([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]],
                                columns=['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item'])

        transformed_features = preprocesser.transform(features)
        prediction = dtr.predict(transformed_features).reshape(-1, 1)

        return render_template('index.html', prediction=prediction[0][0])

if __name__ == '__main__':
    app.run(debug=True)
