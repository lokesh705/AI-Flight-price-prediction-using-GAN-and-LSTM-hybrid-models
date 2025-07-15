from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Load LSTM model and scalers
lstm_model = tf.keras.models.load_model("lstm_model.h5", compile=False)
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Load GAN models (optional)
gan_generator = tf.keras.models.load_model("gan_generator.h5", compile=False)
gan_discriminator = tf.keras.models.load_model("gan_discriminator.h5", compile=False)

@app.route('/')
def home():
    return render_template('index.html')  # Load from /templates/index.html

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_vector = {col: 0 for col in feature_columns}

    input_vector['duration'] = float(data['duration'])
    input_vector['days_left'] = float(data['days_left'])
    input_vector['stops'] = int(data['stops'])

    for feature in feature_columns:
        if feature in data:
            input_vector[feature] = 1

    input_df = pd.DataFrame([input_vector])
    X_scaled = scaler_X.transform(input_df)
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    y_pred_scaled = lstm_model.predict(X_lstm, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)[0][0]

    return jsonify({"predicted_price": round(float(y_pred), 2)})

@app.route('/generate', methods=['GET'])
def generate_synthetic_data():
    noise = tf.random.normal([1, 100])
    synthetic_sample = gan_generator.predict(noise, verbose=0)
    return jsonify({"synthetic_sample": synthetic_sample.tolist()})

if __name__ == "__main__":
    app.run(debug=True)

