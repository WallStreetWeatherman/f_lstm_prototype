import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, median_absolute_error
from sklearn.model_selection import train_test_split
from keras import mixed_precision
from keras import regularizers
import os
import random
import csv
import keyboard

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
else:
    print('No GPU detected')

# Set a global policy
mixed_precision.set_global_policy('mixed_float16')

def load_and_preprocess_data(filepath):
    # Load your data
    data = pd.read_csv(filepath)

    # Choose 'Close', 'Open', 'High' and 'Low' prices
    selected_features = data[['Close', 'Open', 'High', 'Low']]
    selected_features = np.array(selected_features)

    # Scale the data
    scaler = MinMaxScaler()
    selected_features = scaler.fit_transform(selected_features)

    # Create a separate scaler for 'Close' price
    scaler_close = MinMaxScaler()
    close_price = np.array(data['Close']).reshape(-1, 1)
    scaler_close.fit(close_price)

    return selected_features, scaler, scaler_close

def save_results_to_file(filename, random_seed, mae, mse, rmse, medae, r2, pred_accuracy):
    # check if file exists
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Seed", "MAE", "MSE", "RMSE", "MedAE", "R2", "Prediction Accuracy"])
        writer.writerow([random_seed, mae, mse, rmse, medae, r2, pred_accuracy])

def split_data(closing_price):
    train_size = int(len(closing_price) * 0.8)  # Let's use 80% of the data for training
    train_data = closing_price[:train_size]
    val_size = int(len(train_data) * 0.2)  # 20% of the training data for validation
    val_data = train_data[-val_size:]
    train_data = train_data[:-val_size]
    test_data = closing_price[train_size:]
    return train_data, val_data, test_data

def create_sequences(data):
    # Create sequences of 60 days and target
    X = []
    y = []
    for i in range(hyperparameters["sequence_length"], len(data)):
        X.append(data[i-hyperparameters["sequence_length"]:i])
        y.append(data[i, 0]) # Here we are still predicting 'Close' price, so the target is the first column
    X, y = np.array(X), np.array(y)

    return X, y

hyperparameters = {
    "lstm_units": 50,
    "dropout_rate": 0.2,
    "batch_size": 32,
    "optimizer": 'adam',
    "patience": 32,
    "sequence_length": 60,  # Number of days for sequences
    "epochs": 100, 
}

# Then, in your create_model function, use these hyperparameters:
def create_model(X_train, y_train, X_val, y_val):
    model = Sequential()
    model.add(LSTM(units=hyperparameters["lstm_units"], return_sequences=True, input_shape=(hyperparameters["sequence_length"], X_train.shape[2]), 
                   kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001))) 
    model.add(Dropout(hyperparameters["dropout_rate"]))
    model.add(LSTM(units=hyperparameters["lstm_units"], return_sequences=True, 
                   kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)))
    model.add(Dropout(hyperparameters["dropout_rate"]))
    model.add(LSTM(units=hyperparameters["lstm_units"], return_sequences=False, 
                   kernel_regularizer=regularizers.l1_l2(l1=0.001, l2=0.001)))
    model.add(Dropout(hyperparameters["dropout_rate"]))
    model.add(Dense(units=25))  # Adding an additional Dense layer
    model.add(Dense(units=1))

    model.compile(optimizer=hyperparameters["optimizer"], loss='mean_squared_error')

    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=hyperparameters["patience"], 
                                                   mode='min', restore_best_weights=True)

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=hyperparameters["epochs"], 
              batch_size=hyperparameters["batch_size"], callbacks=[es_callback])
    
    return model

def predict_price(model, closing_price, scaler, scaler_close):
    # Get the last 'sequence_length' days' closing prices and scale them
    last_sequence_days = closing_price[-hyperparameters["sequence_length"]:]  # This should already be a 2D array
    last_sequence_days_scaled = scaler.transform(last_sequence_days)

    # Create a sequence and reshape
    X_test = []
    X_test.append(last_sequence_days_scaled)
    X_test = np.array(X_test)

    # Predict the closing price
    pred_price = model.predict(X_test)

    # Undo scaling for the closing price only
    pred_price = scaler_close.inverse_transform(pred_price)  # We only need the inverse transform for the 'Close' price
    
    return pred_price

def evaluate_model(model, X_test, y_test, scaler_close):
    # Predict prices
    predicted_prices = model.predict(X_test)
    
    # Undo scaling
    predicted_prices = scaler_close.inverse_transform(predicted_prices)
    y_test = y_test.reshape(-1, 1) # reshaping from 1D to 2D array
    y_test_unscaled = scaler_close.inverse_transform(y_test)

     # Compute MAE, MSE, RMSE, Median Absolute Error and R2
    mae = mean_absolute_error(y_test_unscaled, predicted_prices)
    mse = mean_squared_error(y_test_unscaled, predicted_prices)
    rmse = np.sqrt(mse)
    medae = median_absolute_error(y_test_unscaled, predicted_prices)
    r2 = r2_score(y_test_unscaled, predicted_prices)

    return mae, mse, rmse, medae, r2

def main(seed):
    # Generate a random seed
    random_seed = seed
    print("\033[92m" + "Random seed used: " + str(random_seed) + "\033[0m")
    
    os.environ['PYTHONHASHSEED']=str(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)
    
    filepath = 'C:\\Users\\Nick\\Desktop\\Programming\\python\\historical data csvs\\stocks\\RTX.csv'
    
    # Update here to get both scalers
    closing_price, scaler, scaler_close = load_and_preprocess_data(filepath)

    # Exclude the last day for model training and testing
    closing_price_model = closing_price[:-1]
    print("Closing price for model (last day excluded):", closing_price_model[-1][0])

    # Split your data into train, validation, and test datasets
    train_data, val_data, test_data = split_data(closing_price_model)

    # Create sequences for train, validation, and test datasets
    X_train, y_train = create_sequences(train_data)
    X_val, y_val = create_sequences(val_data)
    X_test, y_test = create_sequences(test_data)

    # Train the model on the train data
    model = create_model(X_train, y_train, X_val, y_val)

    # Evaluate the model on the test data
    mae, mse, rmse, medae, r2 = evaluate_model(model, X_test, y_test, scaler_close)
    
    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    print("Median Absolute Error:", medae)
    print("R-squared:", r2)

    # Predict the next day's price
    pred_price = predict_price(model, closing_price, scaler, scaler_close)
    print("Predicted next day's price:", pred_price[0][0])

    # Get the actual last day's price and scale it
    actual_last_day_price_scaled = closing_price[-1]  # Get the scaled value from the numpy array
    actual_last_day_price = scaler.inverse_transform([actual_last_day_price_scaled])  # Inverse transform to get the original price back
    print("Actual last day's price:", actual_last_day_price[0][0])

    # Calculate the prediction accuracy
    pred_accuracy = 100 - (abs(pred_price[0][0] - actual_last_day_price[0][0]) / actual_last_day_price[0][0] * 100)
    print("Prediction accuracy:", pred_accuracy, "%")

    # call the function to save the results to the file
    save_results_to_file("F_LSTM_V3.12_results.csv", random_seed, mae, mse, rmse, medae, r2, pred_accuracy)

if __name__ == "__main__":
    for random_seed in range(101):  # Loop over numbers 0-100
        try:
            if keyboard.is_pressed('Z'):  # If the 'Z' key is pressed, break the loop
                print('You pressed Z, ending the script...')
                break
            else:
                main()  # Otherwise, run the main function
        except:
            break  # If user does ctrl+c or an error happened then it will exit the loop