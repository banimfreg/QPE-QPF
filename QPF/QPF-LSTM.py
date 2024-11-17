#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.optimizers import Adam
import tensorflow as tf
import random as python_random

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
python_random.seed(42)


FILE_PATH = 'AggregatedDailyRainfall.xlsx'
ROLLING_WINDOW_SIZE = 60
N_STEPS = 30
N_SPLITS = 5
SIGNIFICANCE_THRESHOLD = 0.1  # Define a threshold for filtering

# Hyperparameter Grid
epoch_counts = [50]
hidden_layer_sizes = [50]
batch_sizes = [32]
learning_rates = [0.001]


#hidden_layer_sizes = [100, 200]
#batch_sizes = [32, 64, 128, 256]
#learning_rates = [0.0001, 0.001, 0.01]
#epoch_counts = [50, 100]
                  
def load_data(file_path):
    """Load data from Excel file."""
    return pd.read_excel(file_path, index_col=[0])

def preprocess_data(data, location):
    """Preprocess data by calculating rolling mean and log transformation."""
    rolling_mean = data[location].rolling(window=ROLLING_WINDOW_SIZE, min_periods=1).mean()
    log_transformed = np.log1p(rolling_mean)
    return log_transformed

def inverse_transform(data):
    """Inverse the log1p transformation."""
    return np.expm1(data)

def create_sequences(data, n_steps):
    """Create sequences for LSTM model."""
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data.iloc[i - n_steps:i].values)
        y.append(data.iloc[i])
    X_array, y_array = np.array(X), np.array(y)
    #print(f"Created sequences: X shape = {X_array}, y shape = {y_array}")
    return X_array, y_array

def build_model(n_features, hidden_layer_size, learning_rate):
    """Build LSTM model with dynamic hyperparameters and variable input dimensions."""
    model = Sequential([
        Input(shape=(N_STEPS, n_features)),  # Dynamic input shape based on number of steps and features
        LSTM(hidden_layer_size, return_sequences=True),
        Dropout(0.2),
        LSTM(hidden_layer_size, return_sequences=False),
        Dropout(0.2),
        Dense(n_features)  # Output layer size matches the number of features
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, X_test, y_test, batch_size, n_epochs):
    """Train the model with specific batch size and number of epochs."""
    history = model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size, validation_data=(X_test, y_test),  verbose=0)
    return history

def evaluate_model(model, X_train, y_train, X_test, y_test, transform_func=None):
    """Evaluate the model on both transformed and original scales."""
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    if transform_func:
        y_train_pred_transformed = transform_func(y_train_pred)
        y_test_pred_transformed = transform_func(y_test_pred)
        y_train_transformed = transform_func(y_train)
        y_test_transformed = transform_func(y_test)
    else:
        y_train_pred_transformed = y_train_pred
        y_test_pred_transformed = y_test_pred
        y_train_transformed = y_train
        y_test_transformed = y_test

    # Avoid evaluating trivial cases with all-zero values
    if np.all(y_train_transformed == 0) or np.all(y_train_pred_transformed == 0) or np.std(y_train_transformed) == 0 or np.std(y_train_pred_transformed) == 0:
        correlation_train = np.nan
    else:
        correlation_train = np.corrcoef(y_train_transformed.flatten(), y_train_pred_transformed.flatten())[0, 1]

    if np.all(y_test_transformed == 0) or np.all(y_test_pred_transformed == 0) or np.std(y_test_transformed) == 0 or np.std(y_test_pred_transformed) == 0:
        correlation_test = np.nan
    else:
        correlation_test = np.corrcoef(y_test_transformed.flatten(), y_test_pred_transformed.flatten())[0, 1]

    metrics = {
        'mae_train': mean_absolute_error(y_train_transformed, y_train_pred_transformed),
        'mse_train': mean_squared_error(y_train_transformed, y_train_pred_transformed),
        'rmse_train': np.sqrt(mean_squared_error(y_train_transformed, y_train_pred_transformed)),
        'r2_train': r2_score(y_train_transformed, y_train_pred_transformed),
        'correlation_train': correlation_train,
        'mae_test': mean_absolute_error(y_test_transformed, y_test_pred_transformed),
        'mse_test': mean_squared_error(y_test_transformed, y_test_pred_transformed),
        'rmse_test': np.sqrt(mean_squared_error(y_test_transformed, y_test_pred_transformed)),
        'r2_test': r2_score(y_test_transformed, y_test_pred_transformed),
        'correlation_test': correlation_test
    }

    return metrics

def main():
    data = load_data(FILE_PATH)
    
    locations = data.columns

    significant_locations = []
    all_results = []

    for location in locations:
        print(f"Processing location: {location}")
        preprocessed_data = preprocess_data(data, location)
        X, y = create_sequences(preprocessed_data, N_STEPS)
        
        tscv = TimeSeriesSplit(n_splits=N_SPLITS)

        results = []
        average_results = []
        best_metrics = None
        best_configuration = None

        # Grid search over hyperparameters
        for hidden_layer_size in hidden_layer_sizes:
            for batch_size in batch_sizes:
                for learning_rate in learning_rates:
                    for n_epochs in epoch_counts:
                        all_fold_metrics = []
                        print(f"\nTesting model for {location} with {hidden_layer_size} hidden units, batch size {batch_size}, learning rate {learning_rate}, epochs {n_epochs}")
                        for fold_index, (train_index, test_index) in enumerate(tscv.split(X), 1):
                            X_train, X_test = X[train_index], X[test_index]
                            y_train, y_test = y[train_index], y[test_index]
                            
                            # Check for all-zero sequences
                            if np.all(X_train == 0) or np.all(y_train == 0):
                                print(f"Fold {fold_index} for {location} has all-zero training data, skipping this fold.")
                                continue

                            if np.all(X_test == 0) or np.all(y_test == 0):
                                print(f"Fold {fold_index} for {location} has all-zero test data, skipping this fold.")
                                continue
                            
                            #print(f"Fold {fold_index}: X_train shape = {X_train.shape}, y_train shape = {y_train.shape}")
                            #print(f"Fold {fold_index}: X_test shape = {X_test.shape}, y_test shape = {y_test.shape}")

                            model = build_model(1, hidden_layer_size, learning_rate)
                            history = train_model(model, X_train, y_train, X_test, y_test, batch_size, n_epochs)
                            metrics = evaluate_model(model, X_train, y_train, X_test, y_test, transform_func=inverse_transform)
                            all_fold_metrics.append(metrics)
                            results.append({
                                'Location': location,
                                'Hidden Layers': hidden_layer_size,
                                'Batch Size': batch_size,
                                'Learning Rate': learning_rate,
                                'Epochs': n_epochs,
                                'Fold': fold_index,
                                **metrics
                            })
                            #print(f"Fold {fold_index} Metrics: {metrics}")

                        # Calculate average metrics across folds and store separately
                        if all_fold_metrics:
                            avg_metrics = {key: np.mean([m[key] for m in all_fold_metrics]) for key in all_fold_metrics[0]}
                            average_results.append({
                                'Location': location,
                                'Hidden Layers': hidden_layer_size,
                                'Batch Size': batch_size,
                                'Learning Rate': learning_rate,
                                'Epochs': n_epochs,
                                **avg_metrics,
                                'Configuration': 'Average'
                            })
                            print(f"Average Metrics Across All Folds for {location}: {avg_metrics}")

                            # Update best configuration if it has the lowest MSE test
                            if best_metrics is None or avg_metrics['mse_test'] < best_metrics['mse_test']:
                                best_metrics = avg_metrics
                                best_configuration = (hidden_layer_size, batch_size, learning_rate, n_epochs)

        # Filter significant locations based on a significance threshold for correlation
        if best_metrics and best_metrics['correlation_test'] >= SIGNIFICANCE_THRESHOLD:
            significant_locations.append(location)
            all_results.extend(results + average_results)

    # Create DataFrame from results and save to Excel for all locations
        # Create DataFrame from results and save to Excel for all locations
    results_df = pd.DataFrame(all_results)
    results_df.to_excel('lstm_model_results_all_locations.xlsx', index=False)

    print("\nSignificant Locations Based on Correlation Threshold:")
    for location in significant_locations:
        print(location)

    print("\nBest Configuration for Each Significant Location:")
    for location in significant_locations:
        location_results = results_df[results_df['Location'] == location]
        best_result = location_results[location_results['Configuration'] == 'Average'].sort_values(by='mse_test').iloc[0]
        print(f"\nLocation: {location}")
        print(f"Hidden Layers: {best_result['Hidden Layers']}, Batch Size: {best_result['Batch Size']}, Learning Rate: {best_result['Learning Rate']}, Epochs: {best_result['Epochs']}")
        print(f"Best Metrics: {best_result[['mae_train', 'mse_train', 'rmse_train', 'r2_train', 'correlation_train', 'mae_test', 'mse_test', 'rmse_test', 'r2_test', 'correlation_test']].to_dict()}")

if __name__ == "__main__":
    main()

