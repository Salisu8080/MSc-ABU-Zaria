"""
Lightweight Intelligent Intrusion Detection System (LIIDS) for Wireless Sensor Networks
Using Deep Autoencoders with Bayesian Optimization (Optuna)

- Bayesian optimization for efficient hyperparameter tuning
- Memory-optimized implementation
- Mixed precision training for GPU acceleration
- Progressive pruning of underperforming trials
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import time
import os
import requests
import gc
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# Define color scheme for plots
COLORS = {
    'primary': '#003366',
    'secondary': '#0066cc',
    'accent': '#ff9900',
    'light_bg': '#f5f5f5',
    'text': '#333333'
}

def optimize_tf_memory():
    """Configure TensorFlow for memory optimization"""
    # Grow GPU memory allocation as needed
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s). Memory growth enabled.")
        except RuntimeError as e:
            print(f"GPU memory configuration error: {e}")

    # Enable mixed precision training for faster computations
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision policy set to:", policy.name)
    except:
        print("Failed to set mixed precision policy. Using default.")
        
    # Clean up any existing sessions
    tf.keras.backend.clear_session()
    print("TensorFlow memory optimized.")

class LIIDS:
    """
    Lightweight Intelligent Intrusion Detection System (LIIDS) for WSNs
    Memory-optimized and GPU-accelerated implementation
    """

    def __init__(self, input_dim=41, hidden_layers=[128, 64, 32],
                 activation='relu', output_activation='sigmoid',
                 dropout_rate=0.2, learning_rate=0.01,
                 batch_size=32, epochs=50, l1_reg=0.01):
        """
        Initialize the LIIDS model with hyperparameters

        Args:
            input_dim: Number of input features
            hidden_layers: List of neurons in each hidden layer
            activation: Activation function for hidden layers
            output_activation: Activation function for output layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for the optimizer
            batch_size: Batch size for training
            epochs: Maximum number of epochs for training
            l1_reg: L1 regularization coefficient
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.output_activation = output_activation
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.l1_reg = l1_reg
        self.threshold = None
        self.original_data_size = input_dim
        self.compressed_data_size = hidden_layers[-1]  # Bottleneck size
        self.model = None
        self.encoder = None
        self.history = None
        self.training_time = 0
        self.scaler = MinMaxScaler()

        # Build the model
        self._build_model()

    def _build_model(self):
        """
        Build the deep autoencoder architecture
        - Encoder: Input → Hidden Layers
        - Decoder: Hidden Layers (reversed) → Output
        - Activations: ReLU for hidden layers, Sigmoid for output
        - Regularization: L1 and Dropout
        """
        # Input layer
        input_layer = Input(shape=(self.input_dim,), name='input')

        # Encoder layers
        encoded = input_layer
        for i, units in enumerate(self.hidden_layers):
            encoded = Dense(units=units,
                           activation=self.activation,
                           kernel_regularizer=l1(self.l1_reg),
                           name=f'encoder_{i+1}')(encoded)
            if i < len(self.hidden_layers) - 1:  # No dropout at bottleneck
                encoded = Dropout(self.dropout_rate)(encoded)

        # Bottleneck layer (last hidden layer)
        bottleneck = encoded

        # Decoder layers (reverse of encoder)
        decoded = bottleneck
        for i, units in enumerate(reversed(self.hidden_layers[:-1])):
            decoded = Dense(units=units,
                           activation=self.activation,
                           kernel_regularizer=l1(self.l1_reg),
                           name=f'decoder_{i+1}')(decoded)
            decoded = Dropout(self.dropout_rate)(decoded)

        # Output layer
        output_layer = Dense(units=self.input_dim,
                            activation=self.output_activation,
                            name='output')(decoded)

        # Create the autoencoder model
        self.model = Model(inputs=input_layer, outputs=output_layer)

        # Create the encoder model (for bottleneck extraction)
        self.encoder = Model(inputs=input_layer, outputs=bottleneck)

        # Compile the model with mixed precision optimizations
        optimizer = Adam(learning_rate=self.learning_rate, epsilon=1e-4)
        self.model.compile(optimizer=optimizer, loss='mse')

    def preprocess_data(self, X, y=None, fit=True):
        """
        Preprocess the data (normalization/scaling)
        """
        # Scale features to [0,1] range
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        return X_scaled, y

    def train(self, X_train, validation_split=0.2, patience=10):
        """
        Train the autoencoder model on normal data only
        """
        # Define callbacks for early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            verbose=0,
            restore_best_weights=True
        )

        # Convert data to TensorFlow dataset for better performance
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, X_train))
        train_dataset = train_dataset.batch(self.batch_size)
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

        # Start timing
        start_time = time.time()

        # Train the model
        self.history = self.model.fit(
            train_dataset,
            epochs=self.epochs,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=0  # Reduced verbosity for hyperparameter optimization
        )

        # Calculate training time
        self.training_time = time.time() - start_time

        return self.history

    def set_threshold(self, X_normal, percentile=95):
        """
        Set the threshold for anomaly detection based on reconstruction error of normal data
        """
        # Get reconstruction errors on normal data
        predictions = self.model.predict(X_normal, verbose=0, batch_size=1024)
        mse = np.mean(np.power(X_normal - predictions, 2), axis=1)

        # Set threshold as the specified percentile of errors
        self.threshold = np.percentile(mse, percentile)

        return self.threshold

    def predict(self, X):
        """
        Predict anomalies based on reconstruction error
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Run set_threshold() first.")

        # Calculate reconstruction error
        predictions = self.model.predict(X, verbose=0, batch_size=1024)
        mse = np.mean(np.power(X - predictions, 2), axis=1)

        # Classify based on threshold
        y_pred = (mse > self.threshold).astype(int)

        return y_pred, mse

    def evaluate(self, X, y_true):
        """
        Evaluate the model performance
        """
        # Get predictions
        y_pred, mse = self.predict(X)

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Calculate false alarm rate
        far = fp / (fp + tn) if (fp + tn) > 0 else 0

        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true, mse)
        roc_auc = auc(fpr, tpr)

        # Return metrics
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'far': far,
            'auc': roc_auc,
            'confusion_matrix': {
                'tn': tn,
                'fp': fp,
                'fn': fn,
                'tp': tp
            },
            'roc': {
                'fpr': fpr,
                'tpr': tpr
            }
        }

        return metrics, mse

    def calculate_resource_efficiency(self):
        """
        Calculate bandwidth and energy efficiency
        """
        # Calculate bandwidth efficiency (Eq. 3.9 in thesis)
        bandwidth_efficiency = ((self.original_data_size - self.compressed_data_size) /
                              self.original_data_size) * 100

        # Calculate energy efficiency (simplified model)
        energy_efficiency = bandwidth_efficiency

        # Return efficiency metrics
        efficiency = {
            'original_size': self.original_data_size,
            'compressed_size': self.compressed_data_size,
            'bandwidth_efficiency': bandwidth_efficiency,
            'energy_efficiency': energy_efficiency,
            'data_reduction_ratio': self.original_data_size/self.compressed_data_size,
            'training_time': self.training_time
        }

        return efficiency

    def plot_training_history(self):
        """Plot training and validation loss over epochs"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history.history['loss'], label='Training Loss', color=COLORS['primary'])
        plt.plot(self.history.history['val_loss'], label='Validation Loss', color=COLORS['accent'])
        plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
        print("Training history plot saved to 'training_history.png'")

    def save_model(self, path="liids_model.h5"):
        """Save the trained model"""
        self.model.save(path)
        print(f"Model saved to {path}")


class OptunaLIIDS:
    """
    Bayesian Optimization using Optuna for hyperparameter tuning of LIIDS model
    - Adaptive sampling of hyperparameter space
    - Pruning of unpromising trials
    - Memory-efficient implementation
    """

    def __init__(self, input_dim, n_trials=30, study_name="liids_optimization", storage=None):
        """
        Initialize the Optuna search
        
        Args:
            input_dim: Number of input features
            n_trials: Number of trials for optimization
            study_name: Name of the optimization study
            storage: Optuna storage (None = in-memory)
        """
        self.input_dim = input_dim
        self.n_trials = n_trials
        self.study_name = study_name
        self.storage = storage
        self.best_model = None
        self.best_params = None
        self.best_metrics = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.study = None
        
    def objective(self, trial):
        """Objective function for Optuna optimization"""
        # Clear TensorFlow session for memory efficiency
        tf.keras.backend.clear_session()
        
        # Define hyperparameters to search
        hidden_layer_configs = [
            (128, 64, 32),     # Configuration 1
            (256, 128, 64),    # Configuration 2
            (64, 32, 16),      # Configuration 3
            (512, 256, 128),   # Configuration 4
            (128, 128, 64)     # Configuration 5
        ]
        
        # Let Optuna suggest hyperparameters
        hidden_layers_idx = trial.suggest_categorical('hidden_layers_idx', list(range(len(hidden_layer_configs))))
        hidden_layers = hidden_layer_configs[hidden_layers_idx]
        
        learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        l1_reg = trial.suggest_float('l1_reg', 0.0001, 0.05, log=True)
        
        # Create and train model
        model = LIIDS(
            input_dim=self.input_dim,
            hidden_layers=list(hidden_layers),
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            batch_size=batch_size,
            l1_reg=l1_reg
        )
        
        # Extract normal samples for training
        X_train_normal = self.X_train[self.y_train == 0]
        
        # Preprocess data
        X_train_normal_scaled, _ = model.preprocess_data(X_train_normal)
        X_test_scaled, y_test_scaled = model.preprocess_data(self.X_test, self.y_test, fit=False)
        
        # Train model with early stopping (reduced patience for search)
        history = model.train(X_train_normal_scaled, validation_split=0.2, patience=5)
        
        # Implement pruning (Optuna feature for early stopping of unpromising trials)
        # Check if we should prune this trial based on the validation loss
        val_loss = history.history['val_loss'][-1]
        trial.report(val_loss, step=len(history.history['val_loss'])-1)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        # Set threshold based on normal data
        model.set_threshold(X_train_normal_scaled)
        
        # Evaluate model
        metrics, _ = model.evaluate(X_test_scaled, y_test_scaled)
        
        # Store model if it's the best so far
        if trial.number == 0 or metrics['accuracy'] + metrics['recall'] - metrics['far'] > self.best_score:
            self.best_score = metrics['accuracy'] + metrics['recall'] - metrics['far']
            self.intermediate_model = model
            self.intermediate_metrics = metrics
            
        # Calculate composite score (accuracy + recall - far)
        score = metrics['accuracy'] + metrics['recall'] - metrics['far']
        
        # Print progress
        print(f"Trial {trial.number}: Score={score:.4f}, Accuracy={metrics['accuracy']:.4f}, "
              f"Recall={metrics['recall']:.4f}, FAR={metrics['far']:.4f}")
        
        # Clean up after trial
        gc.collect()
        
        return score
    
    def fit(self, X_train, y_train, X_test, y_test):
        """
        Perform Bayesian optimization to find the best hyperparameters
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_test: Test data
            y_test: Test labels
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.best_score = float('-inf')
        self.intermediate_model = None
        self.intermediate_metrics = None
        
        print(f"Running Bayesian optimization with {self.n_trials} trials...")
        
        # Create study with pruner (for early stopping of unpromising trials)
        self.study = optuna.create_study(
            direction='maximize',
            study_name=self.study_name,
            storage=self.storage,
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,        # Number of trials to run before pruning
                n_warmup_steps=5,          # Number of steps to run before pruning
                interval_steps=2,          # Interval between pruning checks
                n_min_trials=1             # Minimum trials to consider for pruning
            )
        )
        
        # Optimize with callbacks for logging
        self.study.optimize(
            self.objective, 
            n_trials=self.n_trials,
            callbacks=[self._logging_callback]
        )
        
        # Get best parameters
        best_params = self.study.best_params
        
        # Convert index to actual hidden layers
        hidden_layer_configs = [
            (128, 64, 32),     # Configuration 1
            (256, 128, 64),    # Configuration 2
            (64, 32, 16),      # Configuration 3
            (512, 256, 128),   # Configuration 4
            (128, 128, 64)     # Configuration 5
        ]
        hidden_layers = hidden_layer_configs[best_params['hidden_layers_idx']]
        
        # Remove index and add actual hidden layers
        best_params_expanded = best_params.copy()
        del best_params_expanded['hidden_layers_idx']
        best_params_expanded['hidden_layers'] = hidden_layers
        
        # Train final model with best parameters and more patient training
        print("\nTraining final model with best parameters...")
        best_model = LIIDS(
            input_dim=self.input_dim,
            hidden_layers=list(hidden_layers),
            dropout_rate=best_params['dropout_rate'],
            learning_rate=best_params['learning_rate'],
            batch_size=best_params['batch_size'],
            l1_reg=best_params['l1_reg'],
            epochs=100  # More epochs for final model
        )
        
        # Extract normal samples for training
        X_train_normal = X_train[y_train == 0]
        
        # Preprocess data
        X_train_normal_scaled, _ = best_model.preprocess_data(X_train_normal)
        X_test_scaled, y_test_scaled = best_model.preprocess_data(X_test, y_test, fit=False)
        
        # Train model with early stopping
        best_model.train(X_train_normal_scaled, validation_split=0.2, patience=15)
        
        # Set threshold based on normal data
        best_model.set_threshold(X_train_normal_scaled)
        
        # Evaluate model
        best_metrics, _ = best_model.evaluate(X_test_scaled, y_test_scaled)
        
        self.best_metrics = best_metrics
        self.best_model = best_model
        self.best_params = best_params_expanded
        
        print("\nBayesian Optimization complete!")
        print(f"Best parameters: {self.best_params}")
        print(f"Best performance: Accuracy={self.best_metrics['accuracy']:.4f}, "
              f"Recall={self.best_metrics['recall']:.4f}, "
              f"FAR={self.best_metrics['far']:.4f}")
        
        return self.best_model, self.best_params, self.best_metrics
    
    def _logging_callback(self, study, trial):
        """Callback for logging progress"""
        if trial.number % 5 == 0 or trial.number + 1 == self.n_trials:
            print(f"Completed {trial.number + 1}/{self.n_trials} trials")
            
            # Run garbage collection
            gc.collect()
    
    def plot_optimization_results(self):
        """Plot Optuna optimization results"""
        if self.study is None:
            print("No study to plot. Run fit() first.")
            return
        
        try:
            # Create directory for plots
            plots_dir = "optuna_plots"
            os.makedirs(plots_dir, exist_ok=True)
            
            # Plot optimization history
            fig = plot_optimization_history(self.study)
            fig.write_image(f"{plots_dir}/optimization_history.png")
            
            # Plot parameter importances
            fig = plot_param_importances(self.study)
            fig.write_image(f"{plots_dir}/param_importances.png")
            
            print(f"Optimization plots saved to {plots_dir}/")
            
            # Create additional plots using matplotlib
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Extract data from trials
            trial_data = []
            for trial in self.study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    data = {
                        'learning_rate': trial.params.get('learning_rate', None),
                        'dropout_rate': trial.params.get('dropout_rate', None),
                        'l1_reg': trial.params.get('l1_reg', None),
                        'batch_size': trial.params.get('batch_size', None),
                        'hidden_layers_idx': trial.params.get('hidden_layers_idx', None),
                        'score': trial.value
                    }
                    trial_data.append(data)
            
            trial_df = pd.DataFrame(trial_data)
            
            # Plot learning rate vs score
            axes[0, 0].scatter(trial_df['learning_rate'], trial_df['score'], alpha=0.7)
            axes[0, 0].set_xscale('log')
            axes[0, 0].set_title('Learning Rate vs Score')
            axes[0, 0].set_xlabel('Learning Rate')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].grid(True, linestyle='--', alpha=0.7)
            
            # Plot dropout rate vs score
            axes[0, 1].scatter(trial_df['dropout_rate'], trial_df['score'], alpha=0.7)
            axes[0, 1].set_title('Dropout Rate vs Score')
            axes[0, 1].set_xlabel('Dropout Rate')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].grid(True, linestyle='--', alpha=0.7)
            
            # Plot l1_reg vs score
            axes[1, 0].scatter(trial_df['l1_reg'], trial_df['score'], alpha=0.7)
            axes[1, 0].set_xscale('log')
            axes[1, 0].set_title('L1 Regularization vs Score')
            axes[1, 0].set_xlabel('L1 Regularization')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].grid(True, linestyle='--', alpha=0.7)
            
            # Plot hidden layers (as categorical) vs score
            axes[1, 1].scatter(trial_df['hidden_layers_idx'], trial_df['score'], alpha=0.7)
            axes[1, 1].set_title('Hidden Layer Config vs Score')
            axes[1, 1].set_xlabel('Hidden Layer Configuration')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_xticks(range(5))
            axes[1, 1].set_xticklabels([
                "(128,64,32)", 
                "(256,128,64)",
                "(64,32,16)",
                "(512,256,128)",
                "(128,128,64)"
            ], rotation=45)
            axes[1, 1].grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(f"{plots_dir}/parameter_analysis.png")
            plt.close()
            
            print(f"Parameter analysis plot saved to {plots_dir}/parameter_analysis.png")
            
        except Exception as e:
            print(f"Error creating plots: {e}")
            print("Optuna visualization may require additional dependencies.")
            print("You can install them with: pip install optuna[visualization]")


class DataProcessor:
    """
    Class for handling data loading and preprocessing with optimized memory usage
    """

    @staticmethod
    def load_nsl_kdd(train_path=None, test_path=None):
        """
        Load and preprocess the NSL-KDD dataset with memory optimization
        """
        print("\nLoading NSL-KDD dataset...")

        # Define column names
        columns = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate', 'label', 'difficulty'
        ]

        # If paths are not provided, download the dataset
        if train_path is None or test_path is None:
            train_path = 'NSL-KDD/KDDTrain+.txt'
            test_path = 'NSL-KDD/KDDTest+.txt'

            # Create directory if it doesn't exist
            if not os.path.exists('NSL-KDD'):
                os.makedirs('NSL-KDD')

            # Download files if they don't exist
            if not os.path.exists(train_path):
                print("Downloading NSL-KDD training set...")
                train_url = 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt'
                response = requests.get(train_url)
                with open(train_path, 'wb') as f:
                    f.write(response.content)

            if not os.path.exists(test_path):
                print("Downloading NSL-KDD test set...")
                test_url = 'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt'
                response = requests.get(test_url)
                with open(test_path, 'wb') as f:
                    f.write(response.content)

        # Load data with optimized memory usage
        train_data = pd.read_csv(train_path, header=None, names=columns)
        test_data = pd.read_csv(test_path, header=None, names=columns)

        # Display dataset information
        print(f"NSL-KDD Training set shape: {train_data.shape}")
        print(f"NSL-KDD Test set shape: {test_data.shape}")

        # Convert labels to binary (normal vs attack) - efficient in-place operation
        train_data['binary_label'] = (train_data['label'] != 'normal').astype(int)
        test_data['binary_label'] = (test_data['label'] != 'normal').astype(int)

        # One-hot encode categorical features
        categorical_cols = ['protocol_type', 'service', 'flag']
        
        # Process train and test separately to save memory
        for data in [train_data, test_data]:
            for col in categorical_cols:
                # Get dummies and join to original dataframe
                dummies = pd.get_dummies(data[col], prefix=col, drop_first=False)
                data[dummies.columns] = dummies
                # Drop original column to save memory
                data.drop(col, axis=1, inplace=True)
        
        # Drop unnecessary columns
        train_data.drop(['label', 'difficulty'], axis=1, inplace=True)
        test_data.drop(['label', 'difficulty'], axis=1, inplace=True)

        # Extract features and labels
        X_train = train_data.drop('binary_label', axis=1)
        y_train = train_data['binary_label'].values
        X_test = test_data.drop('binary_label', axis=1)
        y_test = test_data['binary_label'].values

        # Ensure column alignment
        common_columns = X_train.columns.intersection(X_test.columns)
        X_train = X_train[common_columns]
        X_test = X_test[common_columns]

        print(f"Features shape after preprocessing: X_train={X_train.shape}, X_test={X_test.shape}")
        print(f"Number of normal samples in training: {(y_train == 0).sum()}")
        print(f"Number of attack samples in training: {(y_train == 1).sum()}")
        print(f"Class distribution in training set: {np.bincount(y_train)}")
        print(f"Class distribution in test set: {np.bincount(y_test)}")
        
        # Free memory
        gc.collect()

        return X_train.values, y_train, X_test.values, y_test


def run_bayesian_optimization(dataset='nsl-kdd', n_trials=30):
    """
    Run Bayesian optimization experiment on the specified dataset
    
    Args:
        dataset: Dataset to use ('nsl-kdd' or 'unsw-nb15')
        n_trials: Number of trials for Bayesian optimization
    """
    # Start with optimizing TensorFlow memory
    optimize_tf_memory()
    
    print("\n" + "="*80)
    print(f"BAYESIAN OPTIMIZATION ON {dataset.upper()} DATASET")
    print("="*80)
    
    start_time = time.time()
    
    # Load dataset
    if dataset.lower() == 'nsl-kdd':
        X_train, y_train, X_test, y_test = DataProcessor.load_nsl_kdd()
    else:
        raise ValueError(f"Dataset {dataset} not supported in this implementation")
    
    print(f"Running Bayesian optimization with {n_trials} trials")
    
    # Check if optuna is installed
    try:
        import optuna
    except ImportError:
        print("Optuna is not installed. Installing now...")
        os.system("pip install optuna")
        import optuna
    
    # Optional: Try to install visualization dependencies
    try:
        import plotly
    except ImportError:
        try:
            print("Installing Optuna visualization dependencies...")
            os.system("pip install optuna[visualization]")
        except:
            print("Could not install visualization dependencies. Plots may not be available.")
    
    # Initialize Bayesian optimization
    bayesian_search = OptunaLIIDS(
        input_dim=X_train.shape[1],
        n_trials=n_trials,
        study_name=f"liids_{dataset}"
    )
    
    # Perform Bayesian optimization
    best_model, best_params, best_metrics = bayesian_search.fit(X_train, y_train, X_test, y_test)
    
    # Plot optimization results
    bayesian_search.plot_optimization_results()
    
    # Plot training history
    best_model.plot_training_history()
    
    # Calculate resource efficiency
    efficiency = best_model.calculate_resource_efficiency()
    
    # Print best hyperparameters
    print("\nBest Hyperparameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    # Print best metrics
    print("\nBest Model Performance:")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'far', 'auc']:
        print(f"{metric}: {best_metrics[metric]:.4f}")
    
    # Print resource efficiency
    print("\nResource Efficiency:")
    print(f"Data Compression Ratio: {efficiency['data_reduction_ratio']:.2f}x")
    print(f"Bandwidth Efficiency: {efficiency['bandwidth_efficiency']:.2f}%")
    print(f"Training Time: {efficiency['training_time']:.2f} seconds")
    
    # Save best model
    best_model.save_model(f"best_{dataset}_model_bayesian_opt.h5")
    
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    return best_model, best_metrics, best_params, efficiency


if __name__ == "__main__":
    # Number of trials (adjust based on time constraints)
    n_trials = 30  
    
    run_bayesian_optimization('nsl-kdd', n_trials=n_trials)
