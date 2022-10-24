from flask import Flask, render_template, jsonify, flash, request
from flask_cors import CORS
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)
CORS(app)

df = pd.read_excel('combined.xlsx', index_col=0)
df_one_hot_encoded = df.copy()
del df_one_hot_encoded['Avg kWh']
del df_one_hot_encoded['Highest 30 min Rainfall (mm)']
del df_one_hot_encoded['Highest 60 min Rainfall (mm)']

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
        "SkLearn Linear Regression": "lr_ohe.pkl",
        "Ridge Regression": "ridge.pkl",
        "Lasso Regression": "lasso.pkl",
        "Elastic Net": "elasticnet.pkl",
        "Stochastic Gradient Descent Regressor": "sgdRegressor.pkl",
        "Support Vector Regression": "svr.pkl",
        "K-Nearest Neighbors Regressor": "knn.pkl",
        "Extra Tree Regressor": "ensemble_etr.pkl",
        "Random Forest Regressor": "ensemble_rfr.pkl",
        "Stacking Regressor": "second_sr.pkl",
        "Voting Regressor": "voting_reg_v2.pkl",
        "Gradient Boosting Regressor": "ensemble_gbr.pkl",
        "DNN (eLU)": "DNN (eLU)",
        "DNN (GeLU)": "DNN (GeLU)",
        "DNN (Leaky ReLU)": "DNN (Leaky ReLU)",
        "DNN (ReLU)": "DNN (ReLU)",
        "DNN (SeLU)": "DNN (SeLU)",
        "DNN (Sigmoid)": "DNN (Sigmoid)",
        "DNN (Swish)": "DNN (Swish)",
        "DNN (Tanh)": "DNN (Tanh)"
    }
    month_dict = {
        'January': 1, 'February': 2, 'March': 3,'April': 4, 
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    region_dict = {
        "Central": "Central Region", "East": "East Region", "North East": "North East Region", "North": "North Region", "West": "West Region"
    }
    if "DNN" in model_name:
        model = tf.keras.models.load_model("models/" + model_dict[model_name])
    else:
        model = pickle.load(open("models/" + model_dict[model_name], 'rb'))
    df_one_hot_encoded.loc[len(df_one_hot_encoded.index)] = [dwelling_type, int(year), int(month_dict[month]), region_dict[region], estate, float(daily_rainfall), float(rainfall_120m), float(mean_temperature), float(max_temperature), float(min_temperature), float(mean_wind_speed), float(max_wind_speed)] 
    features_df = pd.get_dummies(df_one_hot_encoded, columns=['Dwelling Type', 'Month', 'Towns', 'Region'])
    if "DNN" not in model_name:
        for column in features_df.columns:
                features_df[column] = (features_df[column] - features_df[column].min()) / (features_df[column].max() - features_df[column].min())
    unseen = features_df.values[-1].tolist()
    electricity_predict = [unseen]
    if "DNN" in model_name:
        predicted_value = model.predict(electricity_predict).flatten()
    else:
        predicted_value = model.predict(electricity_predict)
    #print(predicted_value)
    return jsonify({
        "classifier": model_name,
        "predicted_value": predicted_value[0]
    }), 200

if __name__ == '__main__':
    app.run(port=3000, debug=True)