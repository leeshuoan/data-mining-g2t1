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
    rainfall_120m = request.args.get('120m_rainfall')
    mean_temperature = request.args.get('mean_temperature')
    max_temperature = request.args.get('max_temperature')
    min_temperature = request.args.get('min_temperature')
    mean_wind_speed = request.args.get('mean_wind_speed')
    max_wind_speed = request.args.get('max_wind_speed')
    dwelling_type = request.args.get('dwelling_type')
    estate = request.args.get('estate')
    region = request.args.get('region')
    model_name = request.args.get('model')

    model_dict = {
        "SkLearn Linear Regression": "sklearn_linear_regression_(ohe).pkl",
        "Ridge Regression": "ridge.pkl",
        "Lasso Regression": "lasso.pkl",
        "Elastic Net": "elasticnet.pkl",
        "Stochastic Gradient Descent Regressor": "sgdRegressor.pkl",
        "Support Vector Regression": "svr.pkl",
        "K-Nearest Neighbors Regressor": "knn.pkl",
        "Extra Tree Regressor": "extraTree.pkl",
        "Random Forest Regressor": "randomForestReg.pkl",
        "Stacking Regressor": "stacking2.pkl",
        "Voting Regressor": "votingModel2.pkl",
        "Gradient Boosting Regressor": "ensemble_gbr.pkl",
    }
    dwelling_type_dict = {
        '1-room / 2-room': 8,
        '3-room': 9,
        '4-room': 10, 
        '5-room / Executive': 11,
        'Landed Property': 12, 
        'Private Apartment / Condominium': 13,
        'Private Housing': 14,
        'Public Housing': 15
    }
    month_dict ={
        'January': 16,
        'February': 17,
        'March': 18,
        'April': 19,
        'May': 20,
        'June': 21,
        'July': 22,
        'August': 23,
        'September': 24,
        'October': 25,
        'November': 26,
        'December': 27
    }
    estate_dict = {
        'Ang Mo Kio': 28,
        'Bedok': 29,
        'Bishan': 30,
        'Bukit Batok': 31,
        'Bukit Merah': 32,
        'Bukit Panjang': 33,
        'Bukit Timah': 34,
        'Central Region': 35,
        'Changi': 36,
        'Choa Chu Kang': 37,
        'Clementi': 38,
        'Downtown': 39,
        'East Region': 40,
        'Geylang': 41,
        'Hougang': 42,
        'Jurong East': 43,
        'Jurong West': 44,
        'Kallang': 45,
        'Mandai': 46,
        'Marine Parade': 47,
        'Museum': 48,
        'Newton': 49,
        'North East Region': 50,
        'North Region': 51,
        'Novena': 52,
        'Orchard': 53,
        'Outram': 54,
        'Pasir Ris': 55,
        'Paya Lebar': 56,
        'Pioneer': 57,
        'Punggol': 58,
        'Queenstown': 59,
        'River Valley': 60,
        'Rochor': 61,
        'Seletar': 62,
        'Sembawang': 63,
        'Sengkang': 64,
        'Serangoon': 65,
        'Singapore River': 66,
        'Southern Islands': 67,
        'Sungei Kadut': 68,
        'Tampines': 69,
        'Tanglin': 70,
        'Toa Payoh': 71,
        'West Region': 72,
        'Woodlands': 73,
        'Yishun': 74
    }
    region_dict = {
        'Central': 75,
        'East': 76,
        'North East': 77,
        'North': 78,
        'West': 79,
    }

    model = joblib.load("models/" + model_dict[model_name])
    electricity_predict = [
        float(year), float(daily_rainfall), float(rainfall_120m), float(mean_temperature), float(max_temperature), float(min_temperature), float(mean_wind_speed), float(max_wind_speed), 
        0, 0, 0, 0, 0, 0, 0, 0, # dwelling_types
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # months
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # estates
        0, 0, 0, 0, 0 # regions
    ]

    print(dwelling_type_dict[dwelling_type])
    electricity_predict[dwelling_type_dict[dwelling_type]] = 1
    print(month_dict[month])
    electricity_predict[month_dict[month]] = 1
    print(estate_dict[estate])
    electricity_predict[estate_dict[estate]] = 1
    print(region_dict[region])
    electricity_predict[region_dict[region]] = 1
    
    print(electricity_predict)
    electricity_predict = [electricity_predict]
    predicted_electricity_values = model.predict(electricity_predict)

    return jsonify({
        "classifier": model_name,
        "predicted_electricity_values": predicted_electricity_values[0]
    }), 200

if __name__ == '__main__':
    app.run(port=3000, debug=True)