import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.optimizers import Adam

def preprocess_data(csv_path):
    df = pd.read_csv(csv_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    if 'flight' in df.columns:
        df = df.drop(columns=['flight'])

    # Encode stops
    df['stops'] = df['stops'].map({'zero': 0, 'one': 1, 'two_or_more': 2})

    # Convert to float
    df['duration'] = df['duration'].astype(float)
    df['days_left'] = df['days_left'].astype(float)
    df['price'] = df['price'].astype(float)

    # One-hot encode
    categorical_cols = ['airline', 'source_city', 'departure_time', 'arrival_time', 'destination_city', 'class']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    feature_columns = [col for col in df_encoded.columns if col != 'price']
    X = df_encoded[feature_columns].values
    y = df_encoded['price'].values.reshape(-1, 1)

    # Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    with open('scaler_X.pkl', 'wb') as f: pickle.dump(scaler_X, f)
    with open('scaler_y.pkl', 'wb') as f: pickle.dump(scaler_y, f)
    with open('feature_columns.pkl', 'wb') as f: pickle.dump(feature_columns, f)

    return X_scaled, y_scaled, feature_columns

def build_gan(input_dim, noise_dim=100):
    generator = Sequential([
        Dense(64, activation='relu', input_dim=noise_dim),
        Dense(128, activation='relu'),
        Dense(input_dim, activation='linear')
    ], name="generator")

    discriminator = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ], name="discriminator")
    discriminator.compile(optimizer=Adam(0.0002), loss='binary_crossentropy', metrics=['accuracy'])

    discriminator.trainable = False
    gan_input = Input(shape=(noise_dim,))
    gan_output = discriminator(generator(gan_input))
    gan_model = Model(gan_input, gan_output)
    gan_model.compile(optimizer=Adam(0.0002), loss='binary_crossentropy')

    return generator, discriminator, gan_model

def train_gan(generator, discriminator, gan_model, Xy_real, epochs=500, batch_size=128):
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        idx = np.random.randint(0, Xy_real.shape[0], batch_size)
        real_data = Xy_real[idx]

        noise = np.random.normal(0, 1, (batch_size, generator.input_shape[1]))
        fake_data = generator.predict(noise, verbose=0)

        d_loss_real = discriminator.train_on_batch(real_data, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_data, fake_labels)

        noise = np.random.normal(0, 1, (batch_size, generator.input_shape[1]))
        g_loss = gan_model.train_on_batch(noise, real_labels)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs} [D real: {d_loss_real[0]:.4f}, fake: {d_loss_fake[0]:.4f}] [G loss: {g_loss:.4f}]")

    return generator

def generate_synthetic_data(generator, n_samples, noise_dim=100):
    noise = np.random.normal(0, 1, (int(n_samples), noise_dim))
    synthetic_data = generator.predict(noise, verbose=0)
    X_syn = synthetic_data[:, :-1]
    y_syn = synthetic_data[:, -1].reshape(-1, 1)
    return X_syn, y_syn

def train_lstm(X_train, y_train):
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(X_train, y_train, epochs=50, batch_size=128, verbose=1)
    return model

def main():
    X_scaled, y_scaled, _ = preprocess_data("Clean_Dataset.csv")
    Xy_real = np.hstack((X_scaled, y_scaled))
    input_dim = Xy_real.shape[1]
    noise_dim = 100

    generator, discriminator, gan_model = build_gan(input_dim, noise_dim)
    generator = train_gan(generator, discriminator, gan_model, Xy_real, epochs=500, batch_size=256)

    generator.save('gan_generator.h5')
    discriminator.save('gan_discriminator.h5')

    X_synth, y_synth = generate_synthetic_data(generator, n_samples=X_scaled.shape[0], noise_dim=noise_dim)
    X_combined = np.vstack([X_scaled, X_synth])
    y_combined = np.vstack([y_scaled, y_synth])

    model = train_lstm(X_combined, y_combined)
    model.save("lstm_model.h5")
    print("All models trained and saved.")

if __name__ == "__main__":
    main()
