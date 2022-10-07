from flask import Flask, render_template, jsonify, flash, request
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

@app.route("/")
def home_page():
    return render_template("index.html"), 200

@app.route("/predict", methods=["GET"])
def predict():
    year = request.args.get('year')
    month = request.args.get('month')
    daily_rainfall = request.args.get('daily_rainfall')
    rainfall_30m = request.args.get('30m_rainfall')
    rainfall_60m = request.args.get('60m_rainfall')
    rainfall_120m = request.args.get('120m_rainfall')
    mean_temperature = request.args.get('mean_temperature')
    max_temperature = request.args.get('max_temperature')
    min_temperature = request.args.get('min_temperature')
    mean_wind_speed = request.args.get('mean_wind_speed')
    max_wind_speed = request.args.get('max_wind_speed')
    dwelling_type = request.args.get('dwelling_type')
    estate = request.args.get('estate')
    region = request.args.get('region')
    model = request.args.get('model')

    model_dict = {
        "Ensemble": "ensemble_gbr.pkl",
        "SkLearn Linear Regression": "sklearn_linear_regression_(ohe).pkl",
    }
    dwelling_type_dict = {
        '1-room / 2-room': 11,
        '3-room': 12,
        '4-room': 13, 
        '5-room / Executive': 14,
        'Landed Property': 15, 
        'Private Apartment / Condominium': 16,
        'Private Housing': 17,
        'Public Housing': 18
    }
    month_dict ={
        'January': 19,
        'February': 20,
        'March': 21,
        'April': 22,
        'May': 23,
        'June': 24,
        'July': 25,
        'August': 26,
        'September': 27,
        'October': 28,
        'November': 29,
        'December': 30
    }
    estate_dict = {
        'Ang Mo Kio': 31,
        'Bedok': 32,
        'Bishan': 33,
        'Bukit Batok': 34,
        'Bukit Merah': 35,
        'Bukit Panjang': 36,
        'Bukit Timah': 37,
        'Central Region': 38,
        'Changi': 39,
        'Choa Chu Kang': 40,
        'Clementi': 41,
        'Downtown': 42,
        'East Region': 43,
        'Geylang': 44,
        'Hougang': 45,
        'Jurong East': 46,
        'Jurong West': 47,
        'Kallang': 48,
        'Mandai': 49,
        'Marine Parade': 50,
        'Museum': 51,
        'Newton': 52,
        'North East Region': 53,
        'North Region': 54,
        'Novena': 55,
        'Orchard': 56,
        'Outram': 57,
        'Pasir Ris': 58,
        'Paya Lebar': 59,
        'Pioneer': 60,
        'Punggol': 61,
        'Queenstown': 62,
        'River Valley': 63,
        'Rochor': 64,
        'Seletar': 65,
        'Sembawang': 66,
        'Sengkang': 67,
        'Serangoon': 68,
        'Singapore River': 69,
        'Southern Islands': 70,
        'Sungei Kadut': 71,
        'Tampines': 72,
        'Tanglin': 73,
        'Toa Payoh': 74,
        'West Region': 75,
        'Woodlands': 76,
        'Yishun': 77
    }
    region_dict = {
        'Central': 78,
        'East': 79,
        'North East': 80,
        'North': 81,
        'West': 82,
    }

    model = joblib.load("models/" + model_dict[model])
    electricity_predict = [
        year, daily_rainfall, rainfall_30m, rainfall_60m, rainfall_120m, mean_temperature, max_temperature, min_temperature, mean_wind_speed, max_wind_speed, 
        0, 0, 0, 0, 0, 0, 0, 0, # dwelling_types
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # months
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # estates
        0, 0, 0, 0, 0 # regions
    ]
    print(electricity_predict)

    electricity_predict[dwelling_type_dict[dwelling_type]] = 1
    electricity_predict[month_dict[month]] = 1
    electricity_predict[estate_dict[estate]] = 1
    electricity_predict[region_dict[region]] = 1

    electricity_predict = [electricity_predict]
    predicted_electricity_values = model.predict(electricity_predict)

    return jsonify({
        "predicted_electricity_values": predicted_electricity_values[0]
    }), 200

if __name__ == '__main__':
    app.run(port=3000, debug=True)