````markdown
# ✈️ Flight Ticket Price Prediction Using LSTM, GAN, and Regression

## 📌 Project Overview

This project focuses on predicting flight ticket prices using deep learning models such as **Long Short-Term Memory (LSTM)** and **Generative Adversarial Networks (GAN)**, compared against a traditional **regression model**. The goal is to provide accurate flight fare estimations based on various features such as airline, source city, destination, stops, duration, and days left for departure.

## 📁 Dataset

The dataset used (`Clean_Dataset.csv`) consists of 6000+ rows and includes:
- Airline
- Source City
- Destination City
- Departure and Arrival Times
- Stops
- Duration
- Days Left
- Price (Target variable)

The data has been cleaned and encoded using one-hot encoding techniques.

## 🧠 Models Used

### 🔹 1. **LSTM Model**
- Sequence-based model that handles time-dependent features.
- Inputs are reshaped into 3D for LSTM layers.
- Trained to minimize MSE and optimize prediction accuracy.

### 🔹 2. **GAN + LSTM**
- Synthetic data generation using a GAN Generator.
- LSTM model is trained on both real and synthetic data.
- Improved generalization and reduced overfitting.

### 🔹 3. **Regression**
- A baseline linear regression model.
- Provides quick and interpretable predictions.

## 🧰 Tech Stack

- Python
- TensorFlow / Keras
- Pandas, NumPy
- Matplotlib, Seaborn
- Flask (Backend)
- HTML, CSS, JavaScript (Frontend)
- VS Code / Jupyter

## 📦 Installation

1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/flight-price-prediction
   cd flight-price-prediction
````

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask app:

   ```bash
   python app.py
   ```

4. Open `index.html` in your browser.

## 🖼️ Visualizations

* Bar Graphs of actual vs. predicted prices
* Distribution plot of prices
* Comparison plots for MAE, MSE, RMSE, R² across models

## 📈 Performance Evaluation

| Model      | MAE     | MSE   | RMSE    | R² Score |
| ---------- | ------- | ----- | ------- | -------- |
| Regression | 1753.92 | 5.09M | 2256.12 | 0.6723   |
| LSTM       | 1521.28 | 4.34M | 2082.25 | 0.7136   |
| GAN + LSTM | 1303.17 | 3.68M | 1918.63 | 0.7913   |

* GAN + LSTM model outperformed the other methods in all metrics.
* R² value close to 0.79 indicates strong predictive accuracy.

## 📊 Training Insights

### LSTM

* Trained for 100 epochs with batch size of 32.
* Showed gradual convergence of loss.
* Required reshaping data to (samples, time\_steps, features).

### GAN

* Generator trained to produce realistic feature vectors.
* Discriminator trained to distinguish real from fake samples.
* Enhanced data diversity leading to improved LSTM learning.

## 🚀 Features

* Predict flight prices in real-time
* Modern and responsive frontend
* Categorical feature selection
* Synthetic data generation using GAN
* API endpoint for prediction and generation

## 📌 File Structure

```
├── app.py                   # Flask backend
├── lstm_model.h5            # Trained LSTM model
├── gan_generator.h5         # Trained GAN Generator
├── gan_discriminator.h5     # GAN Discriminator
├── scaler_X.pkl             # Scaler for input features
├── scaler_y.pkl             # Scaler for target
├── feature_columns.pkl      # Feature columns used in training
├── index.html               # Frontend UI
├── Clean_Dataset.csv        # Cleaned flight dataset
└── README.md
```

## 🧪 Experimental Results & Discussion

* GAN-enhanced LSTM achieved the best balance between error reduction and generalization.
* Regression gave a good baseline but struggled with complex patterns.
* Visualizations clearly show that LSTM and GAN models better captured the variance in flight prices.

## 💡 Future Enhancements

* Add NLP-based features from flight reviews or descriptions
* Deploy as a cloud-based web service
* Introduce dynamic time-based pricing trends

## 👨‍💻 Author

* **Name**: \[G.Lokesh]]
* **Institution**: \[MGR EDUCATIONAL AND RESEARCH INSTITUTE]
* **Email**: [lokeshgeriki@gmail.com]

## 📜 License

This project is for academic and educational use. Free to use with credit.


