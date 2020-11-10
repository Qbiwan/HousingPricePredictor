import os

import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data,
                                                              housing.target)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full,
                                                      y_train_full)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

model = keras.models.Sequential([
                                keras.layers.Dense(30, activation="relu",
                                                   input_shape=X_train.shape[1:]
                                                   ),
                                keras.layers.Dense(1)
                                ])

model.compile(loss="mean_squared_error", optimizer="sgd")

if __name__ == "__main__":

    print("Training model...")
    print("Model Summary:")
    print(model.summary())
    print("Start training...")
    history = model.fit(X_train_scaled, y_train,
                        epochs=20,
                        validation_data=(X_valid_scaled, y_valid)
                        )
    print("Training ends...")
    print("MSE on Testset:")
    mse_test = model.evaluate(X_test_scaled, y_test)

    model_path = "model/"
    print(f"Saving model at {model_path}")
    os.makedirs(model_path, exist_ok=True)
    model.save(os.path.join(model_path, "model.h5"))
    joblib.dump(scaler, os.path.join(model_path, "scaler.pkl"))
    print(f"Saving scaler at {model_path}")
