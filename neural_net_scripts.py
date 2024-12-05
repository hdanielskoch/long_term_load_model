import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
# from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
# from IPython.display import display
import keras_tuner as kt
from tensorflow.keras import Sequential, layers
from sklearn.model_selection import RandomizedSearchCV

# from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator




def prepare_sequences(X_scaled, y_scaled, sequence_length=24, forecast_horizon=1, scaler_features=None, is_prediction=False):
    """
    Prepare sequences for LSTM training or prediction.

    Args:
        df: pandas DataFrame with all features
        sequence_length: number of time steps to look back
        forecast_horizon: number of time steps to predict ahead
        scaler_features: Scaler for features (used for prediction data)
        scaler_target: Scaler for target (used for prediction data)
        is_prediction: If True, use existing scalers for scaling

    Returns:
        X: Array of sequences (features)
        y: Target values (only for training)
        scaler_features: Scaler used for feature scaling (if fitted in this function)
        scaler_target: Scaler used for target scaling (if fitted in this function)
        feature_columns: List of feature column names
    """
    
    X, y = [], []

    # Create sequences
    for i in range(len(X_scaled) - sequence_length - forecast_horizon + 1):
        feature_seq = X_scaled[i:(i + sequence_length)]
        X.append(feature_seq)
        
        if not is_prediction:
            target = y_scaled[i + sequence_length]
            y.append(target)

    if is_prediction:
        return np.array(X)
    else:
        return np.array(X), np.array(y)
    
    
def prepare_lstm_training_data(
    load_features_timeboxed,
    train_start,
    train_end,
    val_start,
    val_end,
    test_start=None,
    test_end=None,
    target_col='kwh_per_customer',
    datetime_col='datetime',
    sequence_length=24,
    forecast_horizon=1,
    verbose=True
):
    """
    Prepare data for LSTM training including scaling and sequence creation.
    
    Parameters:
    -----------
    load_features_timeboxed : pd.DataFrame
        Preprocessed features DataFrame
    train_start, train_end : pd.Timestamp
        Start and end dates for training period
    val_start, val_end : pd.Timestamp
        Start and end dates for validation period
    test_start, test_end : pd.Timestamp, optional
        Start and end dates for test period
    target_col : str
        Name of target column
    datetime_col : str
        Name of datetime column
    sequence_length : int
        Length of input sequences
    forecast_horizon : int
        Number of steps ahead to predict
    verbose : bool
        Whether to print shapes of output arrays
        
    Returns:
    --------
    dict containing:
        - X_train_sequences, y_train_sequences
        - X_val_sequences, y_val_sequences
        - X_test_sequences, y_test_sequences (if test dates provided)
        - train_data, val_data, test_data (original data for these periods)
        - scaler_X, scaler_y (fitted scalers)
    """
    from hourly_load_norm import create_scalers, prepare_and_scale_data
    from neural_net_scripts import prepare_sequences
    
    # Create scalers using training data only
    scaler_X, scaler_y = create_scalers(
        df=load_features_timeboxed,
        target_col=target_col,
        datetime_col=datetime_col,
        train_start=train_start,
        train_end=train_end
    )
    
    # Prepare training data
    X_train_scaled, y_train_scaled, train_idx = prepare_and_scale_data(
        df=load_features_timeboxed,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        target_col=target_col,
        datetime_col=datetime_col,
        start_date=train_start,
        end_date=train_end
    )
    train_valid_idx = train_idx[sequence_length:]
    train_data = load_features_timeboxed.loc[train_valid_idx]
    
    # Prepare validation data
    X_val_scaled, y_val_scaled, val_idx = prepare_and_scale_data(
        df=load_features_timeboxed,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        target_col=target_col,
        datetime_col=datetime_col,
        start_date=val_start,
        end_date=val_end
    )
    val_valid_idx = val_idx[sequence_length:]
    val_data = load_features_timeboxed.loc[val_valid_idx]
    
    # Create sequences
    X_train_sequences, y_train_sequences = prepare_sequences(
        X_train_scaled, y_train_scaled, 
        sequence_length=sequence_length, 
        forecast_horizon=forecast_horizon
    )
    X_val_sequences, y_val_sequences = prepare_sequences(
        X_val_scaled, y_val_scaled, 
        sequence_length=sequence_length, 
        forecast_horizon=forecast_horizon
    )
    
    result = {
        'X_train_sequences': X_train_sequences,
        'y_train_sequences': y_train_sequences,
        'X_val_sequences': X_val_sequences,
        'y_val_sequences': y_val_sequences,
        'train_data': train_data,
        'val_data': val_data,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }
    
    # Prepare test data if dates provided
    if test_start is not None and test_end is not None:
        X_test_scaled, y_test_scaled, test_idx = prepare_and_scale_data(
            df=load_features_timeboxed,
            scaler_X=scaler_X,
            scaler_y=scaler_y,
            target_col=target_col,
            datetime_col=datetime_col,
            start_date=test_start,
            end_date=test_end
        )
        test_valid_idx = test_idx[sequence_length:]
        test_data = load_features_timeboxed.loc[test_valid_idx]
        
        X_test_sequences, y_test_sequences = prepare_sequences(
            X_test_scaled, y_test_scaled, 
            sequence_length=sequence_length, 
            forecast_horizon=forecast_horizon
        )
        
        result.update({
            'X_test_sequences': X_test_sequences,
            'y_test_sequences': y_test_sequences,
            'test_data': test_data
        })
    
    if verbose:
        print('Training Data:', X_train_sequences.shape, y_train_sequences.shape, train_data.shape)
        print('Validation Data:', X_val_sequences.shape, y_val_sequences.shape, val_data.shape)
        if test_start is not None:
            print('Test Data:', X_test_sequences.shape, y_test_sequences.shape, test_data.shape)
    
    return result

#Print Scaled MAPE every epoch
class RealTimeMAPECallback(Callback):
    def __init__(self, X_train, y_test, X_val, y_val, y_scaler):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_test
        self.X_val = X_val
        self.y_val = y_val
        self.y_scaler = y_scaler

    def on_epoch_end(self, epoch, logs=None):
        # Get predictions on training data
        y_pred_train_scaled = self.model.predict(self.X_train)
        
        # Inverse transform predictions and actuals
        y_pred_train = self.y_scaler.inverse_transform(y_pred_train_scaled)
        y_train_true = self.y_scaler.inverse_transform(self.y_train)
        
        # Calculate MAPE
        mape_train = mean_absolute_percentage_error(y_train_true, y_pred_train) * 100
        print(f"Epoch {epoch + 1} - Real-time MAPE on training data: {mape_train:.3f}%")
        
        # Get predictions on validation data
        y_pred_val_scaled = self.model.predict(self.X_val)
        
        # Inverse transform predictions and actuals
        y_pred_val = self.y_scaler.inverse_transform(y_pred_val_scaled)
        y_val_true = self.y_scaler.inverse_transform(self.y_val)
        
        # Calculate MAPE
        mape_val = mean_absolute_percentage_error(y_val_true, y_pred_val) * 100
        print(f"Epoch {epoch + 1} - Real-time MAPE on validation data: {mape_val:.3f}%")

def create_lstm_model(sequence_length, n_features, params=None):
    """
    Create LSTM model for load forecasting
    
    Args:
        sequence_length: Length of input sequences
        n_features: Number of features
        params: Dictionary of model parameters. If None, uses default values.
    """
    if params is None:
        # Default parameters for hyperparameter tuning
        params = {
            'lstm1_units': 16,
            'learning_rate': 0.01,
            'dropout1_rate': 0.2,
            'use_second_layer': False,
            'use_batchnorm': True,
            'l2_1': None,
            'l2_2': None
        }
    
    layers = [
        Input(shape=(sequence_length, n_features)),
        LSTM(params['lstm1_units'], 
             activation='relu',
             return_sequences=params.get('use_second_layer', False),
             recurrent_dropout=params.get('recurrent_dropout', 0.0),
             kernel_regularizer=tf.keras.regularizers.L2(params['l2_1']) if params.get('l2_1') else None),
        BatchNormalization() if params.get('use_batchnorm', True) else Dropout(0)
        # Dropout(params['dropout1_rate'])
    ]
    
    # Optional second LSTM layer
    if params.get('use_second_layer', False):
        layers.extend([
            LSTM(params.get('lstm2_units', params['lstm1_units']), 
                 activation='relu',
                 kernel_regularizer=tf.keras.regularizers.L2(params['l2_2']) if params.get('l2_2') else None),
            BatchNormalization() if params.get('use_batchnorm', True) else Dropout(0),
            # Dropout(params.get('dropout2_rate', params['dropout1_rate']))
        ])
    
    layers.append(Dense(1))
    
    model = Sequential(layers)
    model.compile(
        optimizer=Adam(learning_rate=params.get('learning_rate', 0.01)),
        loss='mse',
        metrics=['mae', 'mape']
    )
    
    return model

class CustomCallback(Callback):
    """Callback for temporal weighting and top-N metrics calculation"""
    def __init__(self, train_X, val_X, val_y, decay_factor, scaler_y):
        super().__init__()
        self.train_X = train_X
        self.val_X = val_X
        self.val_y = val_y
        self.decay_factor = decay_factor
        self.scaler_y = scaler_y
        self.metrics_log = []
        
    # def on_train_begin(self, logs=None):
        # Set up temporal weights using train_X
        # n_samples = len(self.train_X)
        # weights = np.exp(np.linspace(-self.decay_factor, 0, n_samples))
        # self.model.sample_weight = weights / np.mean(weights)
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predictions = self.model.predict(self.val_X, verbose=0)
        
        # Inverse transform predictions and actual values
        val_predictions_unscaled = self.scaler_y.inverse_transform(val_predictions.reshape(-1, 1)).flatten()
        val_y_unscaled = self.scaler_y.inverse_transform(self.val_y.reshape(-1, 1)).flatten()

        # Calculate APE once using unscaled values
        ape = np.abs((val_y_unscaled - val_predictions_unscaled) / val_y_unscaled) * 100

        # Get indices of top load hours using unscaled values
        top_indices = np.argsort(val_y_unscaled)[::-1]

        # Store APE values for top hours
        top_apes = ape[top_indices]

        # Calculate metrics using the stored values
        metrics = {
            'top_10_mape': np.mean(top_apes[:10]),
            'top_50_mape': np.mean(top_apes[:50]),
            'top_100_mape': np.mean(top_apes[:100]),
            'top_500_mape': np.mean(top_apes[:500]),
            'top_1000_mape': np.mean(top_apes[:1000]),
            'overall_mape': np.mean(ape)
        }

        # Add metrics to logs and store
        logs.update(metrics)
        if hasattr(self.model, '_tuner_trial'):
            trial = self.model._tuner_trial
            self.metrics_log.append({
                'trial_id': trial.trial_id,
                'epoch': epoch,
                **trial.hyperparameters.values,
                **metrics
            })
            pd.DataFrame(self.metrics_log).to_csv('hyperband_results.csv', index=False)
            print(f"\nEpoch {epoch} Results:")
            for name, value in metrics.items():
                print(f"{name}: {value:.2f}%")
                
class TopNMetricsCallback(Callback):
    def __init__(self, val_X, val_y, scaler_y, n_hours):
        super().__init__()
        self.val_X = val_X
        self.val_y = val_y
        self.scaler_y = scaler_y
        self.n_hours = n_hours
        self.best_top_n_mape = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Get predictions
        y_pred = self.model.predict(self.val_X, verbose=0)

        # Inverse transform if scaler provided
        if self.scaler_y is not None:
            y_pred = self.scaler_y.inverse_transform(y_pred)
            y_true = self.scaler_y.inverse_transform(self.val_y)
        else:
            y_true = self.val_y

        # Calculate errors
        errors = np.abs((y_true - y_pred) / y_true) * 100
        sorted_indices = np.argsort(y_true.flatten())

        # Calculate MAPE for top N hours
        top_indices = sorted_indices[-self.n_hours:]
        top_n_mape = np.mean(errors.flatten()[top_indices])
        logs[f'top_{self.n_hours}_mape'] = top_n_mape

        # Update best MAPE
        if top_n_mape < self.best_top_n_mape:
            self.best_top_n_mape = top_n_mape
                
class MultiMetricCallback(Callback):
    def __init__(self, val_X, val_y, scaler_y, n_hours_list):
        super().__init__()
        self.val_X = val_X
        self.val_y = val_y
        self.scaler_y = scaler_y
        self.n_hours_list = n_hours_list
        self.current_metrics = {}  # Add this to store current metrics

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Get predictions
        y_pred = self.model.predict(self.val_X, verbose=0)

        # Inverse transform if scaler provided
        if self.scaler_y is not None:
            y_pred = self.scaler_y.inverse_transform(y_pred)
            y_true = self.scaler_y.inverse_transform(self.val_y)
        else:
            y_true = self.val_y

        # Calculate overall MAPE
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        logs['val_mape'] = mape
        self.current_metrics['val_mape'] = mape  # Store in instance variable

        # Calculate errors once
        errors = np.abs((y_true - y_pred) / y_true) * 100
        sorted_indices = np.argsort(y_true.flatten())

        # Calculate MAPE for each top N hours
        for n in self.n_hours_list:
            top_indices = sorted_indices[-n:]
            top_n_mape = np.mean(errors.flatten()[top_indices])
            logs[f'val_top_{n}_mape'] = top_n_mape
            self.current_metrics[f'top_{n}_mape'] = top_n_mape  # Store in instance variable
            
class TopNMetricsObjective(kt.Objective):
    def __init__(self, n_hours=500, scaler_y=None):
        super().__init__(name=f'top_{n_hours}_mape', direction='min')
        self.n_hours = n_hours
        self.scaler_y = scaler_y
        self.all_metrics = []
    
    def get_value(self, trial, model, val_X, val_y):
        val_predictions = model.predict(val_X)
        
        # Inverse transform both predictions and actual values
        val_predictions_unscaled = self.scaler_y.inverse_transform(val_predictions.reshape(-1, 1)).flatten()
        val_y_unscaled = self.scaler_y.inverse_transform(val_y.reshape(-1, 1)).flatten()
        
        # Calculate absolute percentage errors on unscaled values
        ape = np.abs((val_y_unscaled - val_predictions_unscaled) / val_y_unscaled) * 100
        
        # Sort by actual unscaled load values
        sorted_indices = np.argsort(val_y_unscaled)[::-1]
        
        # Calculate MAPEs for different top N hours
        metrics = {
            'top_10_mape': np.mean(ape[sorted_indices[:10]]),
            'top_50_mape': np.mean(ape[sorted_indices[:50]]),
            'top_100_mape': np.mean(ape[sorted_indices[:100]]),
            'top_500_mape': np.mean(ape[sorted_indices[:500]]),
            'top_1000_mape': np.mean(ape[sorted_indices[:1000]]),
            'overall_mape': np.mean(ape)
        }
        
        print(metrics)
        
        # Store metrics along with hyperparameters
        self.all_metrics.append({
            'trial_id': trial.trial_id,
            **trial.hyperparameters.values,
            **metrics
        })
        
        
        # Print current trial results
        print(f"\nTrial {trial.trial_id} Results:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.2f}%")
            
        return metrics[f'top_{self.n_hours}_mape']


def perform_hyperparameter_tuning(
    train_X, train_y, val_X, val_y, 
    n_hours=500,  # List of top N hours to track
    hp_ranges=None,
    scaler_y=None
):
    # Default hyperparameter ranges
    default_ranges = {
        'lstm_units': [16, 24, 32, 64],
        'learning_rate': {'min': 1e-6, 'max': 1e-1},
        'recurrent_dropout': {'min': 0.0, 'max': 0.3, 'step': 0.1},
    }
    
    # objective = TopNMetricsObjective(n_hours=n_hours)
    # objective.scaler_y = scaler_y  # Pass the scaler to the objective
    hp_ranges = hp_ranges or default_ranges
    
    def build_model(hp):
        params = {'use_batchnorm': True}
        
        # Handle lstm_units
        if isinstance(hp_ranges['lstm_units'], list):
            params['lstm1_units'] = hp.Choice('lstm_units', values=hp_ranges['lstm_units'])
        else:
            params['lstm1_units'] = hp_ranges['lstm_units']
        
        # Handle learning_rate
        lr_config = hp_ranges['learning_rate']
        params['learning_rate'] = hp.Float(
            'learning_rate',
            min_value=lr_config['min'],
            max_value=lr_config['max'],
            sampling=lr_config.get('sampling', 'linear')
        )
        
        # Handle dropout_rate
        if isinstance(hp_ranges['recurrent_dropout'], (int, float)):
            params['recurrent_dropout'] = hp_ranges['recurrent_dropout']
        else:
            dr_config = hp_ranges['recurrent_dropout']
            params['recurrent_dropout'] = hp.Float(
                'recurrent_dropout',
                min_value=dr_config['min'],
                max_value=dr_config['max'],
                step=dr_config.get('step')
            )
        
        return create_lstm_model(train_X.shape[1], train_X.shape[2], params)
    
    tuner = kt.Hyperband(
        build_model,
        objective=kt.Objective(f'top_{n_hours}_mape', direction='min'),
        max_epochs=50,
        factor=3,
        directory='lstm_hyperband',
        project_name='lstm_multi_metric',
        overwrite=True
    )
    
    callbacks = [
        TopNMetricsCallback(val_X, val_y, scaler_y, n_hours),
        EarlyStopping(monitor=f'top_{n_hours}_mape', patience=5, restore_best_weights=True)
    ]
    
    tuner.search(
        train_X, train_y,
        validation_data=(val_X, val_y),
        epochs=50,
        callbacks=callbacks,
        verbose=1
    )
    
    # Get best trial results - Fixed metrics access
    best_trial = tuner.oracle.get_best_trials(1)[0]
    # Get best trial results
    # best_metrics = objective.all_metrics[-1]  # Get metrics from the last trial
    
    return tuner, tuner.get_best_hyperparameters(num_trials=1)[0]


def display_all_trials(tuner):
    # Extract all trials
    trials = tuner.oracle.trials.values()

    # Prepare a list to store the results
    trial_data = []

    # Loop through each trial and extract relevant information
    for trial in trials:
        # Get the history and extract the last step (epoch) number
        history = trial.metrics.get_history('val_loss')
        epochs_run = history[-1].step if history else 'N/A'

        # Get all hyperparameters for this trial
        trial_info = {
           'Trial ID': trial.trial_id,
           'Num epochs': epochs_run,
           'Validation Loss': trial.score
        }

        # Add all hyperparameters dynamically
        for param_name, param_value in trial.hyperparameters.values.items():
            trial_info[param_name] = param_value

        trial_data.append(trial_info)

    # Convert the results to a pandas DataFrame for better visualization
    df = pd.DataFrame(trial_data)

    # Sort by validation loss (lower is better)
    df.sort_values(by='Validation Loss', ascending=True, inplace=True)
    return df
